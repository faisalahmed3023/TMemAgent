from __future__ import annotations
import pandas as pd
from copy import deepcopy
from typing import List, Tuple, Dict, Any, Set, Optional
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import json, math, time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from Amazon_reviews.memory_model import DomainSpecificMemoryBank
from Amazon_reviews.memory_model import MemoryManagement
from Amazon_reviews.memory_model import TransferableMemoryBank

from openai import OpenAI


deployment_name = "gpt-4o-mini"
client = OpenAI(api_key="")

class Data_Structure:
    """
    General-purpose data handler for Amazon-like datasets.

    Expected:
      Users file columns: user_id, (optional) helpful_vote, ...
      Items file columns: parent_asin, title, categories, details, average_rating, rating_number, price, bought_together, ...
      Behaviors file columns: user_id, parent_asin, title, text, rating, timestamp, ...

    All IDs are treated as strings (safe for Amazon).
    """

    def __init__(
        self,
        users_path: str,
        items_path: str,
        behaviors_path: str,
        like_threshold: float = 4.0,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        rating_col: str = "rating",
        time_col: str = "timestamp",
        dedup_user_item_keep_last: bool = True,
    ):
        self.like_threshold = like_threshold
        self.user_col = user_col.lower()
        self.item_col = item_col.lower()
        self.rating_col = rating_col.lower()
        self.time_col = time_col.lower()

        # Load
        self.users = pd.read_csv(users_path)
        self.items = pd.read_csv(items_path)
        self.behaviors = pd.read_csv(behaviors_path)

        # Normalize column names
        self.users.columns = [c.strip().lower() for c in self.users.columns]
        self.items.columns = [c.strip().lower() for c in self.items.columns]
        self.behaviors.columns = [c.strip().lower() for c in self.behaviors.columns]

        # Basic column checks
        self._require_cols(self.users, [self.user_col], "users")
        self._require_cols(self.items, [self.item_col], "items")
        self._require_cols(self.behaviors, [self.user_col, self.item_col], "behaviors")
        if self.rating_col in self.behaviors.columns:
            # ensure numeric
            self.behaviors[self.rating_col] = pd.to_numeric(self.behaviors[self.rating_col], errors="coerce")
        if self.time_col in self.behaviors.columns:
            self.behaviors[self.time_col] = pd.to_numeric(self.behaviors[self.time_col], errors="coerce")

        # Enforce string IDs (Amazon IDs are strings)
        self.users[self.user_col] = self.users[self.user_col].astype(str)
        self.items[self.item_col] = self.items[self.item_col].astype(str)
        self.behaviors[self.user_col] = self.behaviors[self.user_col].astype(str)
        self.behaviors[self.item_col] = self.behaviors[self.item_col].astype(str)

        # Optional: deduplicate (user,item) keep last by timestamp (or last row)
        if dedup_user_item_keep_last:
            if self.time_col in self.behaviors.columns:
                self.behaviors = self.behaviors.sort_values(self.time_col)
                self.behaviors = self.behaviors.drop_duplicates(
                    subset=[self.user_col, self.item_col], keep="last"
                )
            else:
                self.behaviors = self.behaviors.drop_duplicates(
                    subset=[self.user_col, self.item_col], keep="last"
                )

        # Fast lookups
        self.user_by_id: Dict[str, pd.Series] = {
            str(r[self.user_col]): r for _, r in self.users.iterrows()
        }
        self.item_by_id: Dict[str, pd.Series] = {
            str(r[self.item_col]): r for _, r in self.items.iterrows()
        }

        # Split-related state
        self.train_behaviors: Optional[pd.DataFrame] = None
        self.val_behaviors: Optional[pd.DataFrame] = None
        self.test_behaviors: Optional[pd.DataFrame] = None
        self.mode: str = "full"  # full/train/val/test

    # --------------------------
    # Internal helpers
    # --------------------------
    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: List[str], name: str):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"[{name}] missing required columns: {missing}. Found: {list(df.columns)}")

    def _get_active_behaviors(self) -> pd.DataFrame:
        if self.mode == "full" or self.train_behaviors is None:
            return self.behaviors
        if self.mode == "train":
            if self.train_behaviors is None:
                raise RuntimeError("Train split not initialized. Call dataset_split_*() first.")
            return self.train_behaviors
        if self.mode == "val":
            if self.val_behaviors is None:
                raise RuntimeError("Val split not initialized. Call dataset_split_*() first.")
            return self.val_behaviors
        if self.mode == "test":
            if self.test_behaviors is None:
                raise RuntimeError("Test split not initialized. Call dataset_split_*() first.")
            return self.test_behaviors
        raise ValueError(f"Unknown mode: {self.mode}")

    # --------------------------
    # Public API
    # --------------------------
    def set_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in {"full", "train", "val", "test"}:
            raise ValueError("mode must be one of: full, train, val, test")
        if mode in {"train", "val", "test"} and self.train_behaviors is None:
            raise RuntimeError("You must call dataset_split_*() before using train/val/test mode.")
        self.mode = mode

    # Convenience accessors
    def all_user_ids(self) -> List[str]:
        return self.users[self.user_col].astype(str).tolist()

    def all_item_ids(self) -> List[str]:
        return self.items[self.item_col].astype(str).tolist()

    def get_user(self, uid: str) -> Optional[pd.Series]:
        return self.user_by_id.get(str(uid))

    def get_item(self, iid: str) -> Optional[pd.Series]:
        return self.item_by_id.get(str(iid))

    # --------------------------
    # Positives / preferences (mode-aware)
    # --------------------------
    def user_positive_items(self, uid: str, thr: Optional[float] = None) -> List[str]:
        thr = self.like_threshold if thr is None else thr
        df = self._get_active_behaviors()
        if self.rating_col not in df.columns:
            raise RuntimeError("No rating column in behaviors; cannot compute positives by threshold.")
        pos = df[(df[self.user_col] == str(uid)) & (df[self.rating_col] >= thr)]
        return pos[self.item_col].astype(str).tolist()

    def user_pos_df(self, uid: str, thr: Optional[float] = None) -> pd.DataFrame:
        thr = self.like_threshold if thr is None else thr
        df = self._get_active_behaviors()
        if self.rating_col not in df.columns:
            raise RuntimeError("No rating column in behaviors; cannot compute positives by threshold.")
        return df[(df[self.user_col] == str(uid)) & (df[self.rating_col] >= thr)].copy()

    def get_user_interactions(self, uid: str) -> pd.DataFrame:
        """All interactions for uid (mode-aware), sorted by timestamp if available."""
        df = self._get_active_behaviors()
        out = df[df[self.user_col] == str(uid)].copy()
        if self.time_col in out.columns:
            out = out.sort_values(self.time_col)
        return out.reset_index(drop=True)

    def get_user_rated_items(self, uid: str, min_rating: Optional[float] = None) -> pd.DataFrame:
        """
        Return all items interacted/rated by user (mode-aware) merged with item metadata when available.
        """
        df = self._get_active_behaviors()
        u = str(uid)
        sub = df[df[self.user_col] == u].copy()
        if min_rating is not None and self.rating_col in sub.columns:
            sub = sub[sub[self.rating_col] >= float(min_rating)]

        merged = sub.merge(self.items, on=self.item_col, how="left", suffixes=("", "_item"))
        return merged.reset_index(drop=True)

    # --------------------------
    # Dataset splitting
    # --------------------------
    def dataset_split_sequential(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        min_train: int = 1,
        min_val: int = 1,
        min_test: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Per-user chronological split (recommended for sequential models).
        Each user's interactions are sorted by timestamp (if present),
        then split by ratios.

        Stores: self.train_behaviors, self.val_behaviors, self.test_behaviors
        """
        df = self.behaviors.copy()
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        if self.time_col in df.columns:
            df = df.sort_values(self.time_col)

        trains, vals, tests = [], [], []
        for uid, g in df.groupby(self.user_col):
            if self.time_col in g.columns:
                g = g.sort_values(self.time_col)
            n = len(g)
            if n < (min_train + min_val + min_test):
                continue

            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = n - n_train - n_val

            # enforce mins
            if n_train < min_train:
                n_train = min_train
            if n_val < min_val:
                n_val = min_val
            n_test = n - n_train - n_val
            if n_test < min_test:
                # back off val then train if needed
                deficit = min_test - n_test
                take_from_val = min(deficit, max(0, n_val - min_val))
                n_val -= take_from_val
                deficit -= take_from_val

                take_from_train = min(deficit, max(0, n_train - min_train))
                n_train -= take_from_train
                deficit -= take_from_train

                n_test = n - n_train - n_val
                if n_test < min_test:
                    continue  # cannot satisfy mins

            trains.append(g.iloc[:n_train])
            vals.append(g.iloc[n_train:n_train + n_val])
            tests.append(g.iloc[n_train + n_val:])

        train_df = pd.concat(trains).reset_index(drop=True) if trains else df.iloc[0:0].copy()
        val_df = pd.concat(vals).reset_index(drop=True) if vals else df.iloc[0:0].copy()
        test_df = pd.concat(tests).reset_index(drop=True) if tests else df.iloc[0:0].copy()

        self.train_behaviors = train_df
        self.val_behaviors = val_df
        self.test_behaviors = test_df
        return train_df, val_df, test_df

    def dataset_split_random(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Random split by rows (not time-aware).
        """
        df = self.behaviors.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df.iloc[n_train + n_val:].reset_index(drop=True)

        self.train_behaviors = train_df
        self.val_behaviors = val_df
        self.test_behaviors = test_df
        return train_df, val_df, test_df

    # --------------------------
    # Counts
    # --------------------------
    def get_user_num(self) -> int:
        return int(self.users[self.user_col].nunique())

    def get_item_num(self) -> int:
        return int(self.items[self.item_col].nunique())


def safe_json_parse(txt: str) -> Dict[str, Any]:
    """
    LLM responses are not always valid JSON. This extracts the first
    {...} block from any string without crashing.
    """
    try:
        s, e = txt.find("{"), txt.rfind("}")
        return {} if s == -1 or e == -1 else json.loads(txt[s:e+1])
    except Exception:
        return {}

# ======================================================
# Latent token container (per-user parameters)
# Each token k has mean mu_k and logvar lv_k (diagonal Gaussian)
# ======================================================
@dataclass
class LatentToken:
    """
    Each user needs K continuous latent vectors that represent their learned preference "direction" in genre space.
    We model each as a diagonal Gaussian (mu, logvar) so we can sample from it during RL exploration.

    Stores mu (mean) and logvar (log-variance) as numpy arrays.
    sample() implements the reparameterization trick:
            z = mu + exp(0.5 * logvar) * epsilon,  epsilon ~ N(0,I)
    """
    mu: np.ndarray        # (d,)
    logvar: np.ndarray    # (d,)

    def sample(self, rng: np.random.RandomState) -> np.ndarray:
        sigma = np.exp(0.5 * self.logvar)
        eps = rng.normal(size=self.mu.shape)
        return self.mu + sigma * eps

# ======================================================
# Encoder to map genres -> latent target direction (warm-up)
# ======================================================
class GenreEncoder:
    """
    We need to convert discrete category strings (like "Electronics") into continuous vectors so we can compute dot-product similarity with user latent vectors.

    Assigns each genre/category a random fixed unit vector on first encounter, then stores it permanently.
    encode_genre_mixture() returns the weighted average of genre vectors for a given genre distribution dict.
    """
    def __init__(self, latent_dim: int, seed: int = 42):
        self.latent_dim = latent_dim
        self.rng = np.random.RandomState(seed)
        self.emb: Dict[str, np.ndarray] = {}

    def _get_vec(self, g: str) -> np.ndarray:
        if g not in self.emb:
            v = self.rng.normal(scale=0.3, size=(self.latent_dim,))
            v /= (np.linalg.norm(v) + 1e-8)
            self.emb[g] = v
        return self.emb[g]

    def encode_genre_mixture(self, genre_weights: Dict[str, float]) -> np.ndarray:
        if not genre_weights:
            return np.zeros(self.latent_dim, dtype=np.float32)
        v = np.zeros(self.latent_dim, dtype=np.float32)
        total = 0.0
        for g, w in genre_weights.items():
            v += float(w) * self._get_vec(g)
            total += float(w)
        if total > 0:
            v /= total
        return v

# ======================================================
# LLMPolicy with LatentR3-style latent reasoning
# ======================================================
class LLMPolicy:
    """
    Latent reasoning policy that:
      - Maintains K compact latent tokens per user (continuous vectors).
      - Supports warm-up (SFT-like) to initialize latents from genre prefs.
      - Runs RL steps with perplexity-based rewards and batch-baseline advantage.
      - Inference uses latent tokens to re-rank candidates; no verbose CoT.

    This updates *only* user-specific latent parameters (mu/logvar), leaving the LLM frozen.
    """

    def __init__(
        self,
        data,                           # your Data_Structure instance
        client,                         # OpenAI / Azure client
        model:        str   = "gpt-4o-mini",
        temperature:  float = 0.3,
        max_retries:  int   = 3,
        lr:           float = 1e-2,
        latent_dim:   int   = 32,
        K:            int   = 5,        # number of latent tokens per user
        seed:         int   = 123,
        sigma_noise:  float = 0.25,     # extra noise floor for RL exploration
        use_logprobs: bool  = False,
    ):
        # ── Store raw Data_Structure ──────────────────────────────────────
        self.ds          = data         # kept for mode-aware calls (train/val/test)
        self.client      = client
        self.model       = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.lr          = lr
        self.latent_dim  = latent_dim
        self.K           = K
        self.sigma_noise = sigma_noise
        self.use_logprobs= use_logprobs
        self.gamma       = 0.9
        self.trait_decay = 0.95

        self.rng         = np.random.RandomState(seed)
        self.encoder     = GenreEncoder(latent_dim, seed=seed)

        # Per-user profile storage — auto-initialised on first access
        self.user_profiles = defaultdict(self._init_profile)

        # Genre preference cache — computed once per user, reused
        self._genre_pref_cache: Dict[str, Dict[str, float]] = {}

        # ── Build the shaped tables MetaCoTTrainer reads ──────────────────
        # After this call, self.movies and self._item_lookup are ready
        self._build_movies_table()
        self._build_users_table()

        print(
            f"[LLMPolicy] Ready — "
            f"{len(self.ds.all_user_ids())} users | "
            f"{len(self.movies)} items"
        )


    # =========================================================================
    # FUNCTION — _build_movies_table
    # -------------------------------------------------------------------------
    # WHY    : MetaCoTTrainer reads self.data.movies with columns:
    #              item_id, title, genre (pipe-separated), genre_set (set)
    #          Your items.csv has:
    #              parent_asin, title, categories, average_rating, price, ...
    #          This function builds the correctly shaped table from your data.
    #
    # WHAT   : Step 1 — copies items DataFrame
    #          Step 2 — renames parent_asin → item_id
    #          Step 3 — converts categories column to pipe-separated genre string
    #          Step 4 — builds genre_set (a Python set) for fast intersection
    #          Step 5 — builds a fast dict lookup: item_id → row
    #          Stores result as self.movies and self._item_lookup
    #
    # CHANGE : Completely new. Original assumed movies table was pre-built.
    # =========================================================================
    def _build_movies_table(self):
        items = self.ds.items.copy()

        # Step 1: rename parent_asin → item_id
        items["item_id"] = items[self.ds.item_col].astype(str)

        # Step 2: convert categories → pipe-separated genre string
        if "categories" in items.columns:
            items["genre"] = items["categories"].apply(self._parse_categories)
        else:
            items["genre"] = "Unknown"

        # Step 3: build genre_set from the pipe-separated genre string
        # MetaCoTTrainer uses genre_set for Jaccard overlap computation
        items["genre_set"] = items["genre"].apply(
            lambda x: set(str(x).split("|")) - {"", "Unknown"}
            if pd.notna(x) else set()
        )

        # Step 4: select and store final columns
        keep = ["item_id", "title", "genre", "genre_set"]
        for col in ["average_rating", "rating_number", "price"]:
            if col in items.columns:
                keep.append(col)
        self.movies = items[keep].reset_index(drop=True)

        # Step 5: fast lookup dict (used by candidate sampler and scoring)
        self._item_lookup: Dict[str, Any] = {
            row["item_id"]: row
            for _, row in self.movies.iterrows()
        }


    # =========================================================================
    # FUNCTION — _build_users_table
    # -------------------------------------------------------------------------
    # WHY    : persona_description() reads age and status from user row.
    #          Your users.csv only has user_id and helpful_vote.
    #          We add synthetic age/status so the persona system doesn't crash.
    #          If you later add real demographics, just add those columns to
    #          users.csv and they'll be picked up automatically.
    #
    # WHAT   : Copies users DataFrame, renames user_col → user_id,
    #          adds synthetic age (20-65) and status (job role) if missing,
    #          builds a fast dict lookup: user_id → dict.
    #
    # CHANGE : New. Original assumed these columns existed in the data object.
    # =========================================================================
    def _build_users_table(self):
        users = self.ds.users.copy()
        users["user_id"] = users[self.ds.user_col].astype(str)

        n   = len(users)
        rng = np.random.RandomState(42)

        if "age" not in users.columns:
            users["age"] = rng.randint(20, 65, size=n)

        if "status" not in users.columns:
            users["status"] = rng.choice(
                ["engineer", "writer", "scientist", "artist", "lawyer"],
                size=n
            )

        self.users = users

        # Fast lookup: user_id (str) → dict
        self._user_lookup: Dict[str, dict] = {
            str(r["user_id"]): r.to_dict()
            for _, r in users.iterrows()
        }


    # =========================================================================
    # FUNCTION — _parse_categories  [STATIC]
    # -------------------------------------------------------------------------
    # WHY    : Amazon's categories column has inconsistent formats across
    #          different product domains. We need one clean pipe-separated
    #          string that GenreEncoder and genre_set can both consume.
    #
    # WHAT   : Handles three formats:
    #            Format 1 — Python list string: "['Electronics', 'Computers']"
    #                       → "Electronics|Computers"
    #            Format 2 — Hierarchy string:  "Electronics > Laptops > Gaming"
    #                       → "Electronics|Laptops|Gaming"
    #            Format 3 — Comma separated:   "Books, Fiction"
    #                       → "Books|Fiction"
    #            Fallback  — plain string or NaN → "Unknown"
    #
    # CHANGE : Completely new. Bridges Amazon category formats to the pipe-
    #          separated genre format the rest of the class expects.
    # =========================================================================
    @staticmethod
    def _parse_categories(raw) -> str:
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            return "Unknown"
        s = str(raw).strip()
        if not s:
            return "Unknown"

        # Format 1: Python list string  ['A', 'B', 'C']
        if s.startswith("["):
            try:
                parsed = json.loads(s.replace("'", '"'))
                if isinstance(parsed, list):
                    tokens = [str(x).split(">")[-1].strip() for x in parsed]
                    result = "|".join(t for t in tokens if t)
                    return result if result else "Unknown"
            except Exception:
                # Strip brackets and fall through to comma handling
                s = re.sub(r"[\[\]'\"]", "", s).strip()

        # Format 2: Hierarchy  A > B > C
        if ">" in s:
            tokens = [t.strip() for t in s.split(">") if t.strip()]
            return "|".join(tokens)

        # Format 3: Comma-separated  A, B
        if "," in s:
            tokens = [t.strip() for t in s.split(",") if t.strip()]
            return "|".join(tokens)

        # Fallback: plain string
        return s if s else "Unknown"


    # =========================================================================
    # FUNCTION — _quick_sentiment  [STATIC]
    # -------------------------------------------------------------------------
    # WHY    : Your behaviors.csv has a 'text' column (review text).
    #          Review sentiment is a free signal that tells us whether a user
    #          truly liked an item beyond just the star rating.
    #          A 4-star rating with "amazing quality, highly recommend" is
    #          stronger preference signal than a 4-star rating with no review.
    #
    # WHAT   : Tokenizes text, counts positive vs negative keyword matches,
    #          returns (pos - neg) / total in the range [-1, 1].
    #          Returns 0.0 if no sentiment keywords found.
    #
    # CHANGE : Completely new. Exploits the 'text' column in behaviors.csv
    #          that the original movie-domain code never had access to.
    # =========================================================================
    @staticmethod
    def _quick_sentiment(text: str) -> float:
        pos_words = {
            "great", "love", "excellent", "amazing", "perfect", "good",
            "best", "awesome", "fantastic", "wonderful", "recommend",
            "happy", "satisfied", "quality", "easy", "superb", "outstanding",
        }
        neg_words = {
            "bad", "terrible", "awful", "worst", "poor", "broken",
            "disappointed", "useless", "waste", "horrible", "return",
            "cheap", "defective", "slow", "hard", "annoying", "misleading",
        }
        tokens = set(re.findall(r"\b\w+\b", text.lower()))
        pos    = len(tokens & pos_words)
        neg    = len(tokens & neg_words)
        total  = pos + neg
        return 0.0 if total == 0 else float((pos - neg) / total)


    # =========================================================================
    # FUNCTION — _init_profile
    # -------------------------------------------------------------------------
    # WHY    : Every new user needs a profile before any learning happens.
    #          This is the default factory for self.user_profiles[uid].
    #
    # WHAT   : Creates:
    #            preferences — genre → float score dict (starts empty)
    #            traits      — OCEAN personality, all initialised at 0.5
    #            memory      — list of past (descriptor, reward, traits) entries
    #            latents     — K LatentTokens with small random mu and logvar=-2
    #                          (logvar=-2 means sigma≈0.14 — tight initial dist)
    #
    # CHANGE : Unchanged from original.
    # =========================================================================
    def _init_profile(self) -> Dict[str, Any]:
        latents = [
            LatentToken(
                mu     = self.rng.normal(scale=0.1, size=(self.latent_dim,)).astype(np.float32),
                logvar = np.full((self.latent_dim,), -2.0, dtype=np.float32),
            )
            for _ in range(self.K)
        ]
        return {
            "preferences": defaultdict(float),
            "traits": {
                "openness":          0.5,
                "conscientiousness": 0.5,
                "extraversion":      0.5,
                "agreeableness":     0.5,
                "neuroticism":       0.5,
            },
            "memory":  [],
            "latents": latents,
        }
    
    def attach_memory_stack(
        self,
        uid: int,
        stm: DomainSpecificMemoryBank,
        ltm: TransferableMemoryBank,
        MLM: MemoryManagement,
    ):
        """
        Attach memory modules to one user profile.
        """
        uid = str(uid)
        prof = self.user_profiles[uid]
        prof["STM"] = stm
        prof["LTM"] = ltm
        prof["MEM_MLM"] = MLM

    def memory_stack(
        self,
        uid: int,
    ) -> Tuple[Optional[DomainSpecificMemoryBank], Optional[TransferableMemoryBank], Optional[MemoryManagement]]:
        """
        Return (STM, LTM, MEM_MLM) for a user.
        """
        uid = str(uid)
        prof = self.user_profiles.get(uid, {})
        return prof.get("STM"), prof.get("LTM"), prof.get("MEM_MLM")


    # =========================================================================
    # FUNCTION — all_user_ids
    # -------------------------------------------------------------------------
    def all_user_ids(self) -> List[str]:
        return self.ds.all_user_ids()


    # =========================================================================
    # FUNCTION — get_user
    # -------------------------------------------------------------------------
    def get_user(self, uid) -> dict:
        return self._user_lookup.get(str(uid), {
            "user_id":      str(uid),
            "age":          35,
            "status":       "engineer",
            "helpful_vote": 0,
        })


    # =========================================================================
    # FUNCTION — get_user_rated_items
    # -------------------------------------------------------------------------
    def get_user_rated_items(self, uid, min_rating: float = 4.0) -> pd.DataFrame:
        df = self.ds.get_user_rated_items(str(uid), min_rating=min_rating)

        if df.empty:
            return df

        df = df.copy()

        # Add genre column (required by extract_ground_truth_profile)
        if "genre" not in df.columns:
            if "categories" in df.columns:
                df["genre"] = df["categories"].apply(self._parse_categories)
            else:
                df["genre"] = "Unknown"

        # Add item_id column (required by candidate sampler)
        if "item_id" not in df.columns:
            df["item_id"] = df[self.ds.item_col].astype(str)

        return df.reset_index(drop=True)


    # =========================================================================
    # FUNCTION — infer_user_genre_prefs
    # -------------------------------------------------------------------------
    def infer_user_genre_prefs(self, uid) -> Dict[str, float]:
        uid = str(uid)

        # Return cached result if available
        if uid in self._genre_pref_cache:
            return self._genre_pref_cache[uid]

        genre_scores: Dict[str, float] = defaultdict(float)
        total = 0.0

        # Credibility boost: users with more helpful votes signal more reliably
        user_row    = self.get_user(uid)
        helpful     = float(user_row.get("helpful_vote") or 0)
        credibility = 1.0 + min(helpful / 100.0, 0.5)   # range: 1.0 → 1.5

        # All interactions for this user (mode-aware via Data_Structure)
        interactions = self.ds.get_user_interactions(uid)

        if interactions.empty:
            self._genre_pref_cache[uid] = {}
            return {}

        for _, row in interactions.iterrows():
            # Get item genre from our pre-built lookup
            iid  = str(row[self.ds.item_col])
            item = self._item_lookup.get(iid)
            if item is None:
                continue

            genres = str(item["genre"]).split("|")

            # Signal 1: rating weight
            rating_val = row.get(self.ds.rating_col)
            if rating_val is not None and not (isinstance(rating_val, float) and math.isnan(rating_val)):
                rating_w = float(rating_val) / 5.0
            else:
                rating_w = 0.6    # neutral weight when rating is missing

            # Signal 2: review text sentiment
            text_val  = row.get("text", "")
            sentiment = self._quick_sentiment(str(text_val)) \
                        if isinstance(text_val, str) and text_val.strip() \
                        else 0.0

            # Combined weight fusing all three signals
            # sentiment in [-1,1] → rescaled to [0,1] as 0.5 + 0.5*sentiment
            combined_w = credibility * (0.7 * rating_w + 0.3 * (0.5 + 0.5 * sentiment))

            for g in genres:
                g = g.strip()
                if g and g != "Unknown":
                    genre_scores[g] += combined_w
                    total           += combined_w

        # Normalise to probability distribution
        if total > 0:
            genre_scores = {g: v / total for g, v in genre_scores.items()}

        result = dict(genre_scores)
        self._genre_pref_cache[uid] = result
        return result


    # =========================================================================
    # FUNCTION — top_genres
    # -------------------------------------------------------------------------
    def top_genres(self, uid, n: int = 3) -> List[str]:
        prefs = self.infer_user_genre_prefs(uid)
        return sorted(prefs, key=prefs.get, reverse=True)[:n]


    # =========================================================================
    # FUNCTION — extract_ground_truth_profile
    # -------------------------------------------------------------------------
    def extract_ground_truth_profile(self, uid):
        uid     = str(uid)
        df      = self.get_user_rated_items(uid, min_rating=self.ds.like_threshold)
        profile = self.user_profiles[uid]

        if df.empty:
            return profile

        for _, row in df.iterrows():
            genre_str = str(row.get("genre", "Unknown"))
            genres    = [g.strip() for g in genre_str.split("|")
                         if g.strip() and g.strip() != "Unknown"]

            # Rating weight: normalise to [0, 1]
            try:
                rating_w = float(row.get(self.ds.rating_col, 3.0)) / 5.0
            except (TypeError, ValueError):
                rating_w = 0.6

            # Sentiment adjustment: ±30% based on review text
            text_val  = row.get("text", "")
            sentiment = self._quick_sentiment(str(text_val)) \
                        if isinstance(text_val, str) and text_val.strip() \
                        else 0.0
            weight = rating_w * (1.0 + 0.3 * sentiment)

            for g in genres:
                profile["preferences"][g] += float(weight)

        # Normalise preferences to sum = 1
        total = sum(profile["preferences"].values())
        if total > 0:
            for g in profile["preferences"]:
                profile["preferences"][g] /= total

        return profile


    # =========================================================================
    # FUNCTION — sft_warmup_step
    # -------------------------------------------------------------------------
    def sft_warmup_step(self, uid, alpha: float = 0.5):
        uid     = str(uid)
        profile = self.user_profiles[uid]
        prefs   = dict(profile["preferences"])

        if not prefs:
            # No preference signal available — keep random initialisation
            return

        # Genre encoder converts {genre: weight} → single latent_dim vector
        target = self.encoder.encode_genre_mixture(prefs).astype(np.float32)

        for tok in profile["latents"]:
            tok.mu = ((1.0 - alpha) * tok.mu + alpha * target).astype(np.float32)


    # =========================================================================
    # FUNCTION — _sample_latent_pack
    # -------------------------------------------------------------------------
    def _sample_latent_pack(self, uid, K_samples: int = 4) -> List[List[np.ndarray]]:
        uid      = str(uid)
        profile  = self.user_profiles[uid]
        base_seq = [tok.mu.copy() for tok in profile["latents"]]   # deterministic
        packs    = [base_seq]

        for _ in range(K_samples - 1):
            seq = []
            for tok in profile["latents"]:
                sigma = np.exp(0.5 * tok.logvar)
                eps   = self.rng.normal(size=sigma.shape)
                seq.append(tok.mu + (sigma + self.sigma_noise) * eps)
            packs.append(seq)

        return packs   # length = K_samples


    # =========================================================================
    # FUNCTION — _latent_vector
    # -------------------------------------------------------------------------
    def _latent_vector(self, uid) -> np.ndarray:
        uid     = str(uid)
        profile = self.user_profiles[uid]
        vecs    = [t.mu for t in profile["latents"]]
        v       = np.mean(vecs, axis=0)
        return v / (np.linalg.norm(v) + 1e-8)


    # =========================================================================
    # FUNCTION — _genre_vector
    # -------------------------------------------------------------------------
    def _genre_vector(self, genres: List[str]) -> np.ndarray:
        weights = {g: 1.0 for g in genres}
        v       = self.encoder.encode_genre_mixture(weights)
        return v / (np.linalg.norm(v) + 1e-8)


    # =========================================================================
    # FUNCTION — gamma_prompt_tag
    # -------------------------------------------------------------------------
    def gamma_prompt_tag(self, gamma: float) -> str:
        if gamma <= 0.6:
            return "Preference: concise but reason explicitly on relevant factors."
        elif gamma <= 0.8:
            return "Preference: balanced reasoning; combine multiple viewpoints."
        else:
            return "Preference: deliberate reasoning; explore alternative perspectives."


    # =========================================================================
    # FUNCTION — persona_description
    # -------------------------------------------------------------------------
    def persona_description(self, user_row) -> str:
        if isinstance(user_row, pd.Series):
            user_row = user_row.to_dict()

        age    = int(user_row.get("age", 35))
        status = str(user_row.get("status", "engineer")).lower()

        if   age < 25: age_trait = "youthful and adaptive"
        elif age < 40: age_trait = "balanced and analytical"
        elif age < 60: age_trait = "experienced and thoughtful"
        else:          age_trait = "wise and reflective"

        discipline_traits = {
            "writer":    ("creative imagination",  "values narrative depth"),
            "engineer":  ("structured logic",      "prefers systematic reasoning"),
            "doctor":    ("empathy and ethics",    "values moral depth"),
            "farmer":    ("patience and realism",  "appreciates perseverance"),
            "scientist": ("curiosity",             "seeks cause-effect explanations"),
            "lawyer":    ("analytical debate",     "enjoys critical reasoning"),
            "artist":    ("aesthetic sensitivity", "values beauty and emotion"),
        }
        cog_trait, pref_trait = discipline_traits.get(
            status, ("open-minded curiosity", "adapts across categories")
        )

        # Add credibility note from helpful_vote column (unique to Amazon data)
        helpful = int(user_row.get("helpful_vote", 0) or 0)
        cred_note = (
            f" This user has written {helpful} helpful reviews, "
            f"indicating deliberate and reliable preferences."
            if helpful > 10 else ""
        )
        return (
            f"The user is a {age_trait} {status}. "
            f"They exhibit {cog_trait} and {pref_trait}.{cred_note}"
        )


    # =========================================================================
    # FUNCTION — build_prompt
    # -------------------------------------------------------------------------
    def build_prompt(self, uid) -> str:
        uid        = str(uid)
        user       = self.get_user(uid)
        persona    = self.persona_description(user)
        prefs      = self.infer_user_genre_prefs(uid)
        top_g      = self.top_genres(uid, n=3)
        traits     = self.user_profiles[uid]["traits"]
        traits_str = ", ".join(f"{k}={v:.2f}" for k, v in traits.items())
        budget_msg = self.gamma_prompt_tag(self.gamma)

        # Sample 30 items for the catalog context
        sample_n   = min(30, len(self.movies))
        catalog_df = self.movies.sample(n=sample_n, random_state=0)

        catalog = []
        for _, r in catalog_df.iterrows():
            entry = {
                "item_id": str(r["item_id"]),
                "title":   str(r.get("title", "")),
                "genres":  sorted(list(r.get("genre_set", {"Unknown"}))),
            }
            # Add average_rating if available (Amazon-specific quality signal)
            avg = r.get("average_rating")
            if avg is not None and not (isinstance(avg, float) and math.isnan(avg)):
                entry["avg_rating"] = round(float(avg), 2)
            catalog.append(entry)

        return (
            f"You are a product recommender using compact latent thinking.\n"
            f"{budget_msg}\n\n"
            f"PERSONA: {persona}\n"
            f"OCEAN: {traits_str}\n"
            f"Top categories: {top_g if top_g else 'Unknown'}\n"
            f"Category distribution: {json.dumps(prefs, ensure_ascii=False)}\n"
            f"Catalog sample: {json.dumps(catalog, ensure_ascii=False)}\n\n"
            f"IMPORTANT:\n"
            f"- Choose ONE product ONLY from the provided CANDIDATE IDs.\n"
            f"- In reasoning_steps[0].item_id, return that ID (must be in candidates)."
        )


    # =========================================================================
    # FUNCTION — call_llm
    # -------------------------------------------------------------------------
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        for _ in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model       = self.model,
                    temperature = self.temperature,
                    max_tokens  = 200,
                    messages    = [
                        {"role": "system", "content": "Output compact JSON only."},
                        {"role": "user",   "content": (
                            prompt
                            + "\nReturn JSON with 'latent_thoughts' "
                              "(intent/novelty/mood) and 'reasoning_steps' "
                              "(item_id, title, reasoning)."
                        )},
                    ],
                )
                content = resp.choices[0].message.content or ""
                parsed  = safe_json_parse(content)
                if parsed:
                    # Normalise to list form
                    lt = parsed.get("latent_thoughts")
                    if isinstance(lt, dict):
                        parsed["latent_thoughts"] = [lt]
                    rs = parsed.get("reasoning_steps")
                    if isinstance(rs, dict):
                        parsed["reasoning_steps"] = [rs]
                    return parsed
            except Exception:
                time.sleep(1.5)

        # Final fallback if all retries fail
        return {
            "latent_thoughts": [{"name": "intent", "content": "general"}],
            "reasoning_steps": [],
        }


    # =========================================================================
    # FUNCTION — generate_trace
    # -------------------------------------------------------------------------
    def generate_trace(self, uid) -> Dict[str, Any]:
        return self.call_llm(self.build_prompt(str(uid)))


    # =========================================================================
    # FUNCTION — _build_persona_update_prompt
    # -------------------------------------------------------------------------
    def _build_persona_update_prompt(self, uid, adv_val: float) -> str:
        uid     = str(uid)
        user    = self.get_user(uid)
        persona = self.persona_description(user)
        rubric  = (
            "Openness:\n"
            "[Positive] Receptive to new products; Curious about new categories\n"
            "[Negative] Prefers familiar brands; Resistant to new categories\n\n"
            "Conscientiousness:\n"
            "[Positive] Goal-oriented purchases; Organised, provides useful reviews\n"
            "[Negative] Unfocused; Little review effort\n\n"
            "Extraversion:\n"
            "[Positive] Active reviewer; Enjoys sharing opinions\n"
            "[Negative] Hesitant to review; Passive consumer\n\n"
            "Agreeableness:\n"
            "[Positive] Empathetic reviews; Cooperative; Polite language\n"
            "[Negative] Harsh reviews; Uncooperative; Rude\n\n"
            "Neuroticism:\n"
            "[Positive] Emotional response; Easily disappointed\n"
            "[Negative] Stable; Confident in product choices\n"
        )
        return (
            f"You rate a user's behavioural signal using OCEAN.\n"
            f"Persona: {persona}\n"
            f"Signal: advantage={adv_val:.3f} "
            f"(positive=desirable behaviour, negative=undesirable).\n\n"
            f"Rubric:\n{rubric}\n"
            "Output STRICT JSON only: 'descriptor' (≤8 words, lowercase) "
            "and 'judgments' (array of 5 objects, one per trait: "
            "openness, conscientiousness, extraversion, agreeableness, neuroticism). "
            "Each: {\"trait\": ..., \"polarity\": \"positive\"|\"negative\", "
            "\"evidence\": [1-3 tokens]}. No extra keys."
        )


    # =========================================================================
    # FUNCTION — gen_descriptor_via_llm
    # -------------------------------------------------------------------------
    def gen_descriptor_via_llm(self, uid, adv_val: float) -> Tuple[str, Dict[str, int]]:
        wanted = ["openness", "conscientiousness", "extraversion",
                  "agreeableness", "neuroticism"]

        def norm_trait(x: str) -> str:
            x = (x or "").strip().lower()
            return {
                "open": "openness",
                "conscientious": "conscientiousness",
                "extroversion": "extraversion",
                "agreeable": "agreeableness",
                "neurotic": "neuroticism",
            }.get(x, x)

        def norm_polarity(x: str) -> int:
            s = (x or "").strip().lower()
            if s in {"positive","pos","+","increase","high","more","active"}: return +1
            if s in {"negative","neg","-","decrease","low","less","stable"}: return -1
            return 0

        def parse_payload(txt: str) -> Tuple[str, Dict[str, int], bool]:
            parsed  = safe_json_parse(txt)
            desc    = str(parsed.get("descriptor","")).strip().lower() if parsed else ""
            jlist   = parsed.get("judgments", []) if parsed else []
            pol_map = {t: 0 for t in wanted}
            if isinstance(jlist, list):
                for j in jlist:
                    if not isinstance(j, dict): continue
                    t = norm_trait(j.get("trait",""))
                    p = norm_polarity(j.get("polarity",""))
                    if t in pol_map: pol_map[t] = p
            ok = bool(desc) and any(v != 0 for v in pol_map.values())
            return desc, pol_map, ok

        prompt = self._build_persona_update_prompt(uid, adv_val)

        # Try 1: json_object mode
        try:
            resp = self.client.chat.completions.create(
                model           = self.model,
                temperature     = 0.0,
                max_tokens      = 180,
                response_format = {"type": "json_object"},
                messages        = [
                    {"role": "system", "content": "Return STRICT JSON only."},
                    {"role": "user",   "content": prompt},
                ],
            )
            desc, pol_map, ok = parse_payload(resp.choices[0].message.content or "")
            if ok:
                return desc, pol_map
        except Exception:
            pass

        # Try 2: plain prompt with example
        try:
            mini = (
                'Output exactly: {"descriptor":"short phrase",'
                '"judgments":['
                '{"trait":"openness","polarity":"positive","evidence":["curious"]},'
                '{"trait":"conscientiousness","polarity":"positive","evidence":["goal"]},'
                '{"trait":"extraversion","polarity":"positive","evidence":["engage"]},'
                '{"trait":"agreeableness","polarity":"positive","evidence":["polite"]},'
                '{"trait":"neuroticism","polarity":"negative","evidence":["confident"]}]}'
            )
            resp = self.client.chat.completions.create(
                model       = self.model,
                temperature = 0.0,
                max_tokens  = 160,
                messages    = [
                    {"role": "system", "content": "Return STRICT JSON only."},
                    {"role": "user",   "content": mini},
                ],
            )
            desc, pol_map, ok = parse_payload(resp.choices[0].message.content or "")
            if ok:
                return desc, pol_map
        except Exception:
            pass

        # Fallback: derive from advantage value sign
        desc = self._fallback_descriptor_from_adv(adv_val)
        pol_map = {
            "openness":          +1 if adv_val >= 0 else -1,
            "conscientiousness": +1 if adv_val >= 0 else -1,
            "extraversion":      +1 if adv_val >= 0 else -1,
            "agreeableness":     +1 if adv_val >= 0 else -1,
            "neuroticism":       +1 if adv_val <  0 else -1,
        }
        return desc, pol_map


    # =========================================================================
    # FUNCTION — _fallback_descriptor_from_adv
    # -------------------------------------------------------------------------
    def _fallback_descriptor_from_adv(self, adv_val: float) -> str:
        if   adv_val >=  0.25: return "curious engaged cooperative confident"
        elif adv_val >=  0.0:  return "curious balanced polite stable"
        elif adv_val >= -0.25: return "familiar hesitant indifferent distracted"
        else:                  return "resistant avoid uncooperative discouraged"


    # =========================================================================
    # FUNCTION — update_user_profile_llm
    # -------------------------------------------------------------------------
    def update_user_profile_llm(
        self,
        uid,
        reward:     float,
        descriptor: str,
        judgments:  Dict[str, int],
    ):
        uid     = str(uid)
        profile = self.user_profiles[uid]
        reward  = float(np.clip(reward, -1.0, 1.0))
        α       = self.lr

        for trait in ["openness", "conscientiousness", "extraversion",
                      "agreeableness", "neuroticism"]:
            sign = int(judgments.get(trait, 0))
            if sign == 0:
                continue
            if trait == "neuroticism":
                # Neuroticism: high = bad; LLM "positive" means more neurotic
                delta = α * (abs(reward) if sign > 0 else -abs(reward))
            else:
                delta = α * (reward if sign > 0 else -abs(reward))

            profile["traits"][trait] = float(
                np.clip(profile["traits"][trait] + delta, 0.0, 1.0)
            )

        # Mean-revert toward 0.5 to prevent extreme values accumulating
        for k in profile["traits"]:
            profile["traits"][k] = float(
                np.clip(0.5 + self.trait_decay * (profile["traits"][k] - 0.5), 0.0, 1.0)
            )

        profile["memory"].append({
            "latent": descriptor,
            "reward": reward,
            "traits": dict(profile["traits"]),
        })


    # =========================================================================
    # FUNCTION — summarize_user
    # -------------------------------------------------------------------------
    def summarize_user(self, uid) -> str:
        uid     = str(uid)
        profile = self.user_profiles[uid]
        return json.dumps({
            "preferences": dict(sorted(
                profile["preferences"].items(), key=lambda x: -x[1]
            )[:5]),
            "traits":       profile["traits"],
            "memory_len":   len(profile["memory"]),
            "latent_norms": [float(np.linalg.norm(t.mu)) for t in profile["latents"]],
        }, indent=2, ensure_ascii=False)


def softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    z = (x - np.max(x)) / max(tau, 1e-8)
    e = np.exp(z)
    return e / np.sum(e)


def kl_diag_gaussians(
    mu:     np.ndarray,
    logvar: np.ndarray,
    mu0:    np.ndarray,
    logvar0:np.ndarray,
) -> float:
    """KL( N(mu,σ²) || N(mu0,σ0²) ) for diagonal Gaussians."""
    mu   = np.asarray(mu,     dtype=np.float64)
    lv   = np.asarray(logvar, dtype=np.float64)
    mu0  = np.asarray(mu0,    dtype=np.float64)
    lv0  = np.asarray(logvar0,dtype=np.float64)
    var  = np.exp(lv)
    var0 = np.exp(lv0)
    term1 = lv0 - lv
    term2 = (var + (mu - mu0) ** 2) / (var0 + 1e-12)
    return 0.5 * float(np.sum(term1 + term2 - 1.0))

class TinyDecoder:

    def __init__(self, alpha_init: float = 0.7, tau_init: float = 0.5, lr: float = 1e-2):
        self.raw_alpha = float(np.log(alpha_init / (1 - alpha_init + 1e-8) + 1e-8))
        self.raw_tau   = float(np.log(np.exp(tau_init) - 1.0 + 1e-8))
        self.lr        = float(lr)

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _softplus(x: float) -> float:
        return math.log(1.0 + math.exp(x))

    def get_alpha_tau(self) -> Tuple[float, float]:
        return float(self._sigmoid(self.raw_alpha)), float(max(self._softplus(self.raw_tau), 1e-3))

    def score_mix(self, genre_match: float, latent_sim: float) -> float:
        a, _ = self.get_alpha_tau()
        return a * genre_match + (1.0 - a) * latent_sim

    def mle_step(
        self,
        probs:            np.ndarray,
        gt_index:         int,
        d_scores_d_alpha: np.ndarray,
        d_scores_d_tau:   Optional[np.ndarray] = None,
    ):
        alpha   = self._sigmoid(self.raw_alpha)
        tau     = self._softplus(self.raw_tau)
        inv_tau = 1.0 / max(tau, 1e-8)

        grad_logp_wrt_scores                = np.zeros_like(probs)
        grad_logp_wrt_scores[gt_index]     += inv_tau
        grad_logp_wrt_scores               -= probs * inv_tau

        dlogp_dalpha     = float(np.sum(grad_logp_wrt_scores * d_scores_d_alpha))
        dlogp_draw_alpha = dlogp_dalpha * alpha * (1.0 - alpha)
        self.raw_alpha  += self.lr * dlogp_draw_alpha

        if d_scores_d_tau is not None:
            self.raw_tau += 0.1 * self.lr * float(d_scores_d_tau)
        else:
            self.raw_tau += 0.01 * self.lr * (0.5 - float(probs[gt_index]))


# =============================================================================
# CLASS — MetaCoTTrainer
# =============================================================================
class MetaCoTTrainer:

    def __init__(
        self,
        policy,
        beta:             float = 0.02,
        clip:             float = 0.5,
        ema_tau:          float = 0.0,
        use_decoder_head: bool  = True,
        dec_lr:           float = 5e-3,
        enable_traces:    bool  = False,
        trace_every:      int   = 10,
        prm_mix:          float = 0.5,
    ):
        self.pi          = policy
        self.beta        = float(beta)
        self.clip        = float(clip)
        self.ema_tau     = float(ema_tau)
        self.decoder     = TinyDecoder(lr=dec_lr) if use_decoder_head else None

        self.ref_profiles = self._snapshot_ref_profiles()

        self.history      = {"qstar": [], "e_rl2": [], "traces": []}
        self.print_every  = 0
        self._global_step = 0

        self.enable_traces = bool(enable_traces)
        self.trace_every   = max(1, int(trace_every))
        self.prm_mix       = float(np.clip(prm_mix, 0.0, 1.0))
        self.beta_min      = 1e-3
        self.beta_max      = 0.5
        self.target_kl     = 1.0


    # =========================================================================
    # FUNCTION — set_logging
    # -------------------------------------------------------------------------
    def set_logging(self, print_every: int = 0):
        self.print_every = int(print_every)


    # =========================================================================
    # FUNCTION — set_traces
    # -------------------------------------------------------------------------
    def set_traces(self, enable: bool = True, every: int = 10, prm_mix: float = 0.5):
        self.enable_traces = bool(enable)
        self.trace_every   = max(1, int(every))
        self.prm_mix       = float(np.clip(prm_mix, 0.0, 1.0))


    # =========================================================================
    # FUNCTION — _adapt_beta
    # -------------------------------------------------------------------------
    def _adapt_beta(self, kl_mean: float):
        if kl_mean > 1.5 * self.target_kl:
            self.beta = min(self.beta * 1.5, self.beta_max)
        elif kl_mean < 0.5 * self.target_kl:
            self.beta = max(self.beta * 0.9, self.beta_min)


    # =========================================================================
    # FUNCTION — recent_stats
    # -------------------------------------------------------------------------
    def recent_stats(self, mode: str = "qstar", last: int = 5) -> List[Dict]:
        return self.history.get(mode, [])[-last:]


    # =========================================================================
    # FUNCTION — user_snapshot
    # -------------------------------------------------------------------------
    def user_snapshot(self, uid) -> Dict[str, Any]:
        uid  = str(uid)                              # CHANGE: str cast
        prof = self.pi.user_profiles[uid]
        prefs = dict(sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5])
        return {
            "traits":       dict(prof["traits"]),
            "top_prefs":    prefs,
            "latent_norms": [float(np.linalg.norm(t.mu)) for t in prof["latents"]],
        }


    # =========================================================================
    # FUNCTION — _snapshot_ref_profiles
    # -------------------------------------------------------------------------
    def _snapshot_ref_profiles(self) -> Dict:
        ref = {}
        for uid, prof in self.pi.user_profiles.items():
            ref[uid] = [
                {"mu": t.mu.copy(), "logvar": t.logvar.copy()}
                for t in prof["latents"]
            ]
        return ref


    # =========================================================================
    # FUNCTION — _ensure_user_in_ref
    # -------------------------------------------------------------------------
    def _ensure_user_in_ref(self, uid) -> None:
        uid = str(uid)                               # CHANGE: str cast
        if uid not in self.ref_profiles:
            prof = self.pi.user_profiles[uid]
            self.ref_profiles[uid] = [
                {"mu": t.mu.copy(), "logvar": t.logvar.copy()}
                for t in prof["latents"]
            ]


    # =========================================================================
    # FUNCTION — kl_to_ref_for_user
    # -------------------------------------------------------------------------
    def kl_to_ref_for_user(self, uid) -> float:
        uid = str(uid)                               # CHANGE: str cast
        self._ensure_user_in_ref(uid)
        prof     = self.pi.user_profiles[uid]
        total_kl = 0.0
        for k, t in enumerate(prof["latents"]):
            mu0  = self.ref_profiles[uid][k]["mu"]
            lv0  = self.ref_profiles[uid][k]["logvar"]
            total_kl += kl_diag_gaussians(t.mu, t.logvar, mu0, lv0)
        return float(total_kl)


    # =========================================================================
    # FUNCTION — kl_to_ref_for_pack  [NEW — WAS MISSING]
    # -------------------------------------------------------------------------
    def kl_to_ref_for_pack(self, uid, seq: List[np.ndarray]) -> float:
        uid = str(uid)                               # CHANGE: str cast
        self._ensure_user_in_ref(uid)
        prof     = self.pi.user_profiles[uid]
        total_kl = 0.0
        for k, sampled_mu in enumerate(seq):
            # Current logvar (variance is not sampled, only mu is)
            lv_current = prof["latents"][k].logvar
            mu0        = self.ref_profiles[uid][k]["mu"]
            lv0        = self.ref_profiles[uid][k]["logvar"]
            total_kl  += kl_diag_gaussians(sampled_mu, lv_current, mu0, lv0)
        return float(total_kl)


    # =========================================================================
    # FUNCTION — ema_update_ref
    # -------------------------------------------------------------------------
    def ema_update_ref(self):
        if self.ema_tau <= 0.0:
            return
        tau = self.ema_tau
        for uid, prof in self.pi.user_profiles.items():
            self._ensure_user_in_ref(uid)
            for k, t in enumerate(prof["latents"]):
                self.ref_profiles[uid][k]["mu"]     = (1.0 - tau) * self.ref_profiles[uid][k]["mu"]     + tau * t.mu
                self.ref_profiles[uid][k]["logvar"] = (1.0 - tau) * self.ref_profiles[uid][k]["logvar"] + tau * t.logvar


    # =========================================================================
    # FUNCTION — _genre_match
    # -------------------------------------------------------------------------
    def _genre_match(self, user_top: set, item_genres: List[str]) -> float:
        gset  = set(item_genres)
        denom = len(user_top | gset)
        if denom <= 0:
            return 0.0
        return float(len(user_top & gset)) / float(denom)


    # =========================================================================
    # FUNCTION — _latent_vector
    # -------------------------------------------------------------------------
    def _latent_vector(self, uid) -> np.ndarray:
        return self.pi._latent_vector(str(uid))      # CHANGE: str cast


    # =========================================================================
    # FUNCTION — _item_vec
    # -------------------------------------------------------------------------
    def _item_vec(self, genres: List[str]) -> np.ndarray:
        return self.pi._genre_vector(genres)


    # =========================================================================
    # FUNCTION — item_logprob_given_Z
    # -------------------------------------------------------------------------
    def item_logprob_given_Z(
        self,
        uid:        Any,
        candidates: List[Any],
        gt_item:    Any,
    ) -> Tuple[float, np.ndarray, np.ndarray, int]:

        uid  = str(uid)                              # CHANGE: str cast
        prof = self.pi.user_profiles[uid]
        user_top = set(
            sorted(prof["preferences"], key=prof["preferences"].get, reverse=True)[:5]
        )

        R           = self._latent_vector(uid)
        genre_terms = []
        sim_terms   = []

        for mid in candidates:
            mid_str = str(mid)                       # CHANGE: str cast for item lookup

            # Look up item row in policy.data.movies
            row = self.pi.movies[
                self.pi.movies["item_id"] == mid_str
            ]

            if row.empty:
                genre_terms.append(0.0)
                sim_terms.append(0.0)
                continue

            # Parse pipe-separated genre string (standardised by _parse_categories)
            genres = str(row.iloc[0]["genre"]).split("|")
            genres = [g.strip() for g in genres if g.strip() and g.strip() != "Unknown"]

            gm  = self._genre_match(user_top, genres)
            I   = self._item_vec(genres)
            sim = float(np.dot(R, I))

            genre_terms.append(gm)
            sim_terms.append(sim)

        # Mix genre_match and latent_sim
        if self.decoder is None:
            raw_scores = 0.7 * np.array(genre_terms) + 0.3 * np.array(sim_terms)
            tau        = 0.5
        else:
            a, tau     = self.decoder.get_alpha_tau()
            raw_scores = a * np.array(genre_terms) + (1.0 - a) * np.array(sim_terms)

        probs = softmax_np(raw_scores, tau=tau)

        # Find ground-truth index
        candidates_str = [str(c) for c in candidates]  # CHANGE: str cast for index()
        gt_str         = str(gt_item)                   # CHANGE: str cast
        try:
            gt_index = candidates_str.index(gt_str)
        except ValueError:
            return -20.0, probs, raw_scores, -1

        log_p = float(np.log(probs[gt_index] + 1e-12))
        return log_p, probs, raw_scores, gt_index


    # =========================================================================
    # FUNCTION — _prm_reward
    # -------------------------------------------------------------------------
    def _prm_reward(self, trace: Dict[str, Any]) -> float:
        try:
            if not isinstance(trace, dict):
                return 0.0

            lt = trace.get("latent_thoughts", [])
            if isinstance(lt, dict):  lt = [lt]
            if lt is None:            lt = []

            rs = trace.get("reasoning_steps", [])
            if isinstance(rs, dict):  rs = [rs]
            if rs is None:            rs = []

            score = 0.0

            # Score latent thoughts
            for obj in lt:
                if not isinstance(obj, dict):
                    continue
                name    = str(obj.get("name",    "")).lower()
                content = str(obj.get("content", "")).lower()

                if "intent" in name:
                    for w in ["clear", "specific", "goal", "match", "relevant"]:
                        if w in content:
                            score += 0.15
                if "novelty" in name:
                    if any(x in content for x in ["medium", "balanced"]): score += 0.05
                    if "low" in content:                                   score -= 0.05
                if "mood" in name:
                    if any(w in content for w in ["thoughtful","focused","engaged","empathetic"]):
                        score += 0.1

            # Score reasoning step
            if rs and isinstance(rs[0], dict):
                step0  = rs[0]
                # CHANGE: check for "item_id" not "movie_id"
                if "item_id" in step0:
                    score += 0.25
                reason = str(step0.get("reasoning", ""))
                if 0 < len(reason) <= 200: score += 0.15
                elif len(reason) > 500:    score -= 0.05

            return float(np.clip(score, -1.0, 1.0))

        except Exception:
            return 0.0


    # =========================================================================
    # FUNCTION — _maybe_trace_and_mix
    # -------------------------------------------------------------------------
    def _maybe_trace_and_mix(self, uid, base_reward: float, mode: str) -> float:
        use_trace = self.enable_traces and (self._global_step % self.trace_every == 0)
        if not use_trace:
            return base_reward

        try:
            uid   = str(uid)                         # CHANGE: str cast
            trace = self.pi.generate_trace(uid)

            if isinstance(trace, dict):
                lt = trace.get("latent_thoughts", [])
                if isinstance(lt, dict): trace["latent_thoughts"] = [lt]
                rs = trace.get("reasoning_steps", [])
                if isinstance(rs, dict): trace["reasoning_steps"] = [rs]

            prm         = self._prm_reward(trace)
            base_scaled = float(np.tanh(base_reward))
            mixed       = self.prm_mix * base_scaled + (1.0 - self.prm_mix) * prm

            short = {
                "step":        self._global_step,
                "mode":        mode,
                "uid":         uid,               # CHANGE: str not int
                "base_reward": float(base_reward),
                "base_scaled": base_scaled,
                "prm":         float(prm),
                "mixed":       float(mixed),
            }
            try:
                short["latent_thoughts"] = (trace.get("latent_thoughts") or [])[:3]
                short["reasoning_head"]  = (trace.get("reasoning_steps") or [])[:1]
            except Exception:
                pass
            self.history.setdefault("traces", []).append(short)

            return float(mixed)

        except Exception as e:
            print("LLM failed in _maybe_trace_and_mix:", repr(e))
            return base_reward


    # =========================================================================
    # FUNCTION — _batch_prm_for_uids
    # -------------------------------------------------------------------------
    def _batch_prm_for_uids(
        self,
        uids:        List[Any],
        mode:        str = "qstar",
        max_workers: int = 4,
    ) -> Dict[str, float]:

        # CHANGE: str(u) instead of int(u) — Amazon IDs are not integers
        unique_uids: List[str] = list(dict.fromkeys(str(u) for u in uids))
        results: Dict[str, float] = {}

        def worker(uid: str):                        # CHANGE: str not int
            trace = self.pi.generate_trace(uid)
            if isinstance(trace, dict):
                lt = trace.get("latent_thoughts", [])
                if isinstance(lt, dict): trace["latent_thoughts"] = [lt]
                rs = trace.get("reasoning_steps", [])
                if isinstance(rs, dict): trace["reasoning_steps"] = [rs]

            prm   = self._prm_reward(trace)
            short = {
                "step": self._global_step,
                "mode": mode,
                "uid":  uid,                         # CHANGE: str
                "prm":  float(prm),
            }
            try:
                short["latent_thoughts"] = (trace.get("latent_thoughts") or [])[:3]
                short["reasoning_head"]  = (trace.get("reasoning_steps") or [])[:1]
            except Exception:
                pass
            return uid, prm, short

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, uid) for uid in unique_uids]
            for fut in as_completed(futures):
                try:
                    uid, prm, short = fut.result()
                    results[uid] = prm
                    self.history.setdefault("traces", []).append(short)
                except Exception as e:
                    print("LLM failed in _batch_prm_for_uids:", repr(e))

        return results


    # =========================================================================
    # FUNCTION — MCoTqstar_step
    # -------------------------------------------------------------------------
    def MCoTqstar_step(
        self,
        batch_pairs: List[Tuple[Any, List[Any], Any]],
        K_samples:   int            = 4,
        beta:        Optional[float] = None,
    ):
        if beta is None:
            beta = self.beta

        use_trace = self.enable_traces and (self._global_step % self.trace_every == 0)

        # ==================================================================
        # A.1 — Sample K latent packs per user, score each one
        # ==================================================================
        # For each user:
        #   - sample K versions of their latent (1 deterministic + K-1 noisy)
        #   - for each version: temporarily override latent mu, compute
        #     log p(gt|Z,q), compute KL penalty, get base_reward
        #   - restore original latent mu after all K evaluations
        # ==================================================================
        print("A.1")
        groups = []

        for uid, candidates, gt_item in batch_pairs:
            uid     = str(uid)                       # CHANGE: str cast
            gt_item = str(gt_item)                   # CHANGE: str cast
            candidates = [str(c) for c in candidates] # CHANGE: str cast

            packs            = self.pi._sample_latent_pack(uid, K_samples=K_samples)
            base_rewards     = []
            original_latents = [t.mu.copy() for t in self.pi.user_profiles[uid]["latents"]]

            for seq in packs:
                # Temporarily set latents to this sampled configuration
                for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                    t.mu = seq[k].astype(np.float32)

                logp, _, _, _ = self.item_logprob_given_Z(uid, candidates, gt_item)

                # CHANGE: kl_to_ref_for_pack now exists (was missing in original)
                kl_pen = self.kl_to_ref_for_pack(uid, seq)
                kl_pen = min(kl_pen, 5.0)            # trust-region cap

                base = logp - beta * kl_pen
                base_rewards.append(base)

            # Restore original latent mu
            for k, t in enumerate(self.pi.user_profiles[uid]["latents"]):
                t.mu = original_latents[k]

            groups.append((uid, candidates, gt_item, packs, base_rewards))

        # ==================================================================
        # A.1.5 — Optional PRM mixing
        # ==================================================================
        # If traces are enabled and this is a trace step:
        #   - run LLM trace for every user in batch (parallel)
        #   - mix: reward = prm_mix*tanh(base) + (1-prm_mix)*prm_score
        # Otherwise: keep base_rewards as-is
        # ==================================================================
        print("A.1.5")
        groups_with_rewards = []
        first_rewards       = []

        if use_trace:
            uid_list = [uid for uid, _, _, _, _ in groups]
            # CHANGE: _batch_prm_for_uids now returns str keys
            uid_prm  = self._batch_prm_for_uids(uid_list, mode="qstar")
        else:
            uid_prm = {}

        for uid, candidates, gt_item, packs, base_rewards in groups:
            if use_trace and uid in uid_prm:
                prm_val = uid_prm[uid]
                rewards = [
                    float(self.prm_mix * np.tanh(b) + (1.0 - self.prm_mix) * prm_val)
                    for b in base_rewards
                ]
            else:
                rewards = [float(b) for b in base_rewards]

            groups_with_rewards.append((uid, candidates, gt_item, packs, rewards))
            first_rewards.append(rewards[0])

        groups = groups_with_rewards

        # ==================================================================
        # A.2 — Batch baseline and normalised advantages
        # ==================================================================
        # baseline = mean of first-pack rewards across all users in batch
        # advantage = (reward - baseline) / std(rewards)
        # Subtracting baseline reduces variance in the gradient estimate.
        # ==================================================================
        print("A.2")
        sbar  = float(np.mean(first_rewards))
        denom = float(np.linalg.norm(np.array(first_rewards) - sbar)) + 1e-8

        # ==================================================================
        # A.3 — Latent update + OCEAN personality update
        # ==================================================================
        # For each user:
        #   - compute advantages for all K packs
        #   - find best pack (argmax advantage)
        #   - update mu: tok.mu += lr * clip(adv_best * (best_seq - tok.mu), ±clip)
        #   - update logvar: expand if any pack was good, contract if all bad
        #   - call LLM for OCEAN descriptor and update personality traits
        # ==================================================================
        print("A.3")
        step_stats = {"mode": "qstar", "n_users": len(batch_pairs), "baseline": sbar}
        all_best = []
        all_adv  = []
        all_kl   = []

        for uid, candidates, gt_item, packs, rewards in groups:
            advs     = [(sk - sbar) / denom for sk in rewards]
            best_idx = int(np.argmax(advs))
            best_seq = packs[best_idx]
            prof     = self.pi.user_profiles[uid]

            all_best.append(rewards[best_idx])
            all_adv.append(advs[best_idx])
            all_kl.append(self.kl_to_ref_for_user(uid))

            # Update each latent token's mu
            for k, tok in enumerate(prof["latents"]):
                delta = best_seq[k] - tok.mu
                step  = np.clip(advs[best_idx] * delta, -self.clip, self.clip)
                tok.mu = (tok.mu + self.pi.lr * step).astype(np.float32)

                # Exploration radius: contract if all packs scored below average,
                # expand slightly if at least one pack scored above average
                if max(advs) < 0:
                    tok.logvar = np.clip(tok.logvar - 0.03, -6.0, -0.2)
                else:
                    tok.logvar = np.clip(tok.logvar + 0.02, -6.0, -0.2)

            # OCEAN personality update via LLM
            rew_norm = float(np.tanh(rewards[best_idx]))
            adv_star = float(advs[best_idx])
            desc, pol_map = self.pi.gen_descriptor_via_llm(uid, adv_star)
            self.pi.update_user_profile_llm(
                uid, reward=rew_norm, descriptor=desc, judgments=pol_map
            )

        # ==================================================================
        # B — TinyDecoder MLE step
        # ==================================================================
        # For each user, using the best latent pack found in A.3:
        #   - recompute genre and sim scores with best_seq temporarily set
        #   - compute ∂scores/∂alpha = genre_terms - sim_terms
        #   - call decoder.mle_step() to update raw_alpha via gradient ascent
        # ==================================================================
        print("B")
        if self.decoder is not None:
            for uid, candidates, gt_item, packs, rewards in groups:
                advs     = [(sk - sbar) / denom for sk in rewards]
                best_idx = int(np.argmax(advs))
                best_seq = packs[best_idx]
                prof     = self.pi.user_profiles[uid]

                original_latents = [t.mu.copy() for t in prof["latents"]]
                for k, t in enumerate(prof["latents"]):
                    t.mu = best_seq[k].astype(np.float32)

                user_top = set(
                    sorted(prof["preferences"], key=prof["preferences"].get, reverse=True)[:5]
                )
                R = self.pi._latent_vector(uid)

                genre_terms, sim_terms = [], []
                for mid in candidates:
                    mid_str = str(mid)               # CHANGE: str cast
                    row = self.pi.movies[
                        self.pi.movies["item_id"] == mid_str
                    ]
                    if row.empty:
                        genre_terms.append(0.0)
                        sim_terms.append(0.0)
                        continue
                    genres = str(row.iloc[0]["genre"]).split("|")
                    genres = [g.strip() for g in genres if g.strip() and g.strip() != "Unknown"]
                    gm     = self._genre_match(user_top, genres)
                    I      = self._item_vec(genres)
                    sim    = float(np.dot(R, I))
                    genre_terms.append(gm)
                    sim_terms.append(sim)

                a, tau     = self.decoder.get_alpha_tau()
                raw_scores = a * np.array(genre_terms) + (1.0 - a) * np.array(sim_terms)
                probs      = softmax_np(raw_scores, tau=tau)

                gt_str = str(gt_item)
                candidates_str = [str(c) for c in candidates]  # CHANGE: str cast
                try:
                    gt_index = candidates_str.index(gt_str)
                except ValueError:
                    gt_index = -1

                if gt_index >= 0:
                    d_scores_d_alpha = np.array(genre_terms) - np.array(sim_terms)
                    self.decoder.mle_step(probs, gt_index, d_scores_d_alpha)

                # Restore latents
                for k, t in enumerate(prof["latents"]):
                    t.mu = original_latents[k]

        # ==================================================================
        # Logging and beta adaptation
        # ==================================================================
        print("B.1")
        step_stats.update({
            "reward_mean": float(np.mean(all_best)),
            "reward_std":  float(np.std(all_best)),
            "adv_mean":    float(np.mean(all_adv)),
            "kl_mean":     float(np.mean(all_kl)),
            "kl_std":      float(np.std(all_kl)),
        })
        self._adapt_beta(step_stats["kl_mean"])
        self.history["qstar"].append(step_stats)

        if self.print_every and (len(self.history["qstar"]) % self.print_every == 0):
            i = len(self.history["qstar"])
            print(
                f"[q-STaR step {i}] "
                f"baseline={sbar:.4f} "
                f"reward={step_stats['reward_mean']:.4f} "
                f"adv={step_stats['adv_mean']:.4f} "
                f"kl={step_stats['kl_mean']:.4f}",
                flush=True,
            )

        self._global_step += 1


    # =========================================================================
    # FUNCTION — MCoTtrain_qstar
    # -------------------------------------------------------------------------
    def MCoTtrain_qstar(
        self,
        batches_of_pairs: List[List[Tuple[Any, List[Any], Any]]],
        K_samples:        int            = 4,
        beta:             Optional[float] = None,
        ema_every:        int            = 0,
    ):
        for it, pairs in enumerate(batches_of_pairs, 1):
            self.MCoTqstar_step(pairs, K_samples=K_samples, beta=beta)
            if ema_every and (it % ema_every == 0):
                self.ema_update_ref()

