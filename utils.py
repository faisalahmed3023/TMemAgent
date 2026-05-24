import math
from collections import Counter
from itertools import combinations
from typing import Dict, List, Optional, Any
from datetime import datetime
from copy import deepcopy
import json
import re
from Amazon_reviews.memory_model import TransferableMemoryBank
from Amazon_reviews.ModelClassifier import EmbeddingRatingRegressorHead

# ===============================
# Central User Storage Classes
# ===============================

class UserState:
    """
    Pure storage class for one user's information.

    This class only stores user-related data and does not perform
    any reasoning, inference, aggregation, or scoring.

    Stored sections:
        - profile              : static/basic user info
        - preferences          : inferred or explicit user preferences
        - short_term_memory    : lightweight recent interaction snapshots
        - long_term_memory     : transferred STM payloads / summaries
        - traits               : user characteristics / behavioural tendencies
        - interaction_stats    : counters / simple statistics
        - notes                : arbitrary extra user information
        - created_at           : creation time
        - updated_at           : last updated time
    """

    def __init__(
        self,
        user_id: str,
        profile: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None,
        short_term_memory: Optional[List[Dict[str, Any]]] = None,
        long_term_memory: Optional[List[Dict[str, Any]]] = None,
        traits: Optional[Dict[str, Any]] = None,
        interaction_stats: Optional[Dict[str, Any]] = None,
        notes: Optional[Dict[str, Any]] = None,
    ):
        self.user_id = str(user_id)

        self.profile = profile or {}
        self.preferences = preferences or {}
        self.short_term_memory = short_term_memory or []
        self.long_term_memory = long_term_memory or []
        self.traits = traits or {}
        self.interaction_stats = interaction_stats or {}
        self.notes = notes or {}

        self.created_at = datetime.utcnow().isoformat()
        self.updated_at = self.created_at

    def touch(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "profile": deepcopy(self.profile),
            "preferences": deepcopy(self.preferences),
            "short_term_memory": deepcopy(self.short_term_memory),
            "long_term_memory": deepcopy(self.long_term_memory),
            "traits": deepcopy(self.traits),
            "interaction_stats": deepcopy(self.interaction_stats),
            "notes": deepcopy(self.notes),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def __repr__(self):
        return (
            f"UserState(user_id={self.user_id}, "
            f"prefs={len(self.preferences)}, "
            f"stm={len(self.short_term_memory)}, "
            f"ltm={len(self.long_term_memory)})"
        )


class UserStateStore:
    """
    Shared repository for all users.

    This store can be passed into STM, MemoryController, LTM, recommender,
    user profiler, or any other class that needs to read/update user data.

    It only stores data.
    """

    def __init__(self):
        self._users: Dict[str, UserState] = {}

    # ---------------------------------------------------------
    # Core access
    # ---------------------------------------------------------

    def create_user(
        self,
        user_id: str,
        profile: Optional[Dict[str, Any]] = None,
    ) -> UserState:
        user_id = str(user_id)
        if user_id not in self._users:
            self._users[user_id] = UserState(user_id=user_id, profile=profile)
        return self._users[user_id]

    def get_user(self, user_id: str) -> Optional[UserState]:
        return self._users.get(str(user_id))

    def get_or_create_user(
        self,
        user_id: str,
        profile: Optional[Dict[str, Any]] = None,
    ) -> UserState:
        user_id = str(user_id)
        if user_id not in self._users:
            self._users[user_id] = UserState(user_id=user_id, profile=profile)
        return self._users[user_id]

    def exists(self, user_id: str) -> bool:
        return str(user_id) in self._users

    # ---------------------------------------------------------
    # Update dict sections
    # ---------------------------------------------------------

    def update_profile(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.profile.update(data)
        user.touch()

    def update_preferences(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.preferences.update(data)
        user.touch()

    def update_traits(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.traits.update(data)
        user.touch()

    def update_interaction_stats(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.interaction_stats.update(data)
        user.touch()

    def update_notes(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.notes.update(data)
        user.touch()

    # ---------------------------------------------------------
    # Replace sections
    # ---------------------------------------------------------

    def set_preferences(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.preferences = dict(data)
        user.touch()

    def set_traits(self, user_id: str, data: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.traits = dict(data)
        user.touch()

    def set_short_term_memory(self, user_id: str, entries: List[Dict[str, Any]]):
        user = self.get_or_create_user(user_id)
        user.short_term_memory = list(entries)
        user.touch()

    def set_long_term_memory(self, user_id: str, entries: List[Dict[str, Any]]):
        user = self.get_or_create_user(user_id)
        user.long_term_memory = list(entries)
        user.touch()

    # ---------------------------------------------------------
    # Append sections
    # ---------------------------------------------------------

    def append_short_term_memory(self, user_id: str, entry: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.short_term_memory.append(entry)
        user.touch()

    def append_long_term_memory(self, user_id: str, entry: Dict[str, Any]):
        user = self.get_or_create_user(user_id)
        user.long_term_memory.append(entry)
        user.touch()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def clear_short_term_memory(self, user_id: str):
        user = self.get_or_create_user(user_id)
        user.short_term_memory = []
        user.touch()

    def all_user_ids(self) -> List[str]:
        return list(self._users.keys())

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {uid: user.to_dict() for uid, user in self._users.items()}

    def __len__(self):
        return len(self._users)

    def __repr__(self):
        return f"UserStateStore(total_users={len(self._users)})"


def category_entropy(picked_categories, base=2):
    """
    Shannon entropy of a single user's picked item categories.

    Parameters
    ----------
    picked_categories : list[str]
        Categories of items the user picked/clicked/watched (e.g., ["Sports","Sports","Tech"]).
    base : int/float
        Log base (2 = bits, math.e = nats).

    Returns
    -------
    float
        Entropy. 0 means all picks are the same category.
    """
    if not picked_categories:
        return 0.0

    counts = Counter(picked_categories)
    total = sum(counts.values())
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * (math.log(p) / math.log(base))
    return ent


def snd_from_user_category_lists(user_to_categories, distance="js", smoothing=1e-12, base=2):
    """
    System Neural Diversity (SND) computed from users' category-pick histories.

    We model each user's behavior as a probability distribution over categories.
    Then:
        d(i,j) = distance between distributions of user i and j
        SND = mean of d(i,j) over all unique pairs   (same aggregation as Eq.(4) in the paper)

    Parameters
    ----------
    user_to_categories : dict[user_id, list[str]]
        Example:
            {
              "u1": ["Sports","Sports","Tech"],
              "u2": ["Politics","Politics"],
              "u3": ["Sports","Politics"]
            }
    distance : str
        "js"  -> Jensen-Shannon *distance* (metric). Returned value in [0, 1] if base=2.
        "tv"  -> Total Variation distance (metric). Returned value in [0, 1].
    smoothing : float
        Small epsilon added to probabilities to avoid log(0) for JS.
    base : int/float
        Log base for JS (2 gives bounded JS distance).

    Returns
    -------
    float
        SND value (mean pairwise distance).
    dict
        Pairwise distance matrix as a nested dict: D[u][v] = d(u,v)
    """
    users = list(user_to_categories.keys())
    n = len(users)
    if n < 2:
        return 0.0, {u: {u: 0.0} for u in users}

    # Build global category vocabulary
    vocab = sorted({cat for cats in user_to_categories.values() for cat in cats})
    idx = {c: i for i, c in enumerate(vocab)}
    m = len(vocab)

    def to_dist(cats):
        # Convert list of categories -> probability vector over vocab
        if not cats:
            # If a user has no picks, treat as uniform (or could be all-zeros; uniform is safer)
            return [1.0 / m] * m if m > 0 else []
        cnt = Counter(cats)
        total = sum(cnt.values())
        p = [0.0] * m
        for c, k in cnt.items():
            if c in idx:
                p[idx[c]] = k / total
        return p

    dists = {u: to_dist(user_to_categories[u]) for u in users}

    def tv_dist(p, q):
        # Total Variation: 0.5 * L1
        return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q))

    def js_distance(p, q):
        # Jensen-Shannon distance = sqrt(JS divergence)
        # JS(p,q) = 0.5*KL(p||m)+0.5*KL(q||m), where m=0.5*(p+q)
        # We add smoothing to avoid log(0).
        def _kl(a, b):
            s = 0.0
            for ai, bi in zip(a, b):
                ai2 = ai + smoothing
                bi2 = bi + smoothing
                s += ai2 * (math.log(ai2 / bi2) / math.log(base))
            return s

        mvec = [(pi + qi) * 0.5 for pi, qi in zip(p, q)]
        js = 0.5 * _kl(p, mvec) + 0.5 * _kl(q, mvec)
        return math.sqrt(max(js, 0.0))

    if distance not in {"js", "tv"}:
        raise ValueError("distance must be 'js' or 'tv'.")

    dist_fn = js_distance if distance == "js" else tv_dist

    # Compute pairwise distances + matrix
    D = {u: {v: 0.0 for v in users} for u in users}
    pair_vals = []
    for u, v in combinations(users, 2):
        d = dist_fn(dists[u], dists[v])
        D[u][v] = d
        D[v][u] = d
        pair_vals.append(d)

    snd = sum(pair_vals) / len(pair_vals) if pair_vals else 0.0
    return snd, D

# user_to_cats = {
#     "u1": ["Sports","Sports","Tech"],
#     "u2": ["Politics","Politics","Politics"],
#     "u3": ["Sports","Politics"]
# }

# print("Entropy u1:", category_entropy(user_to_cats["u1"]))
# snd, D = snd_from_user_category_lists(user_to_cats, distance="js")
# print("SND:", snd)
# print("d(u1,u2):", D["u1"]["u2"])

def safe_json_parse(text: str) -> Dict[str, Any]:
    """
    Parse JSON safely from raw LLM output.
    Tries:
      1. direct json.loads
      2. extract first {...} block
    """
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    raise ValueError(f"Could not parse JSON from LLM output: {text}")


def generate_rating_from_ltm(
    llm,
    user_id: str,
    test_item_meta: dict,
    tf_memory_bank: TransferableMemoryBank,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
) -> dict:
    """
    Generate a predicted rating for a test item using user's retrieved LTM context.

    Args:
        llm              : Mandatory OpenAI/Azure-compatible client
        user_id          : Target user id
        test_item_meta   : Metadata of the test item
        tf_memory_bank   : TransferableMemoryBank instance
        top_k            : Number of relevant memories to retrieve
        model            : Model/deployment name

    Returns:
        {
            "user_id": ...,
            "item_id": ...,
            "predicted_rating": int,
            "reasoning": str,
            "used_memory_count": int
        }
    """
    if llm is None:
        raise ValueError("llm must be provided.")
    if tf_memory_bank is None:
        raise ValueError("tf_memory_bank must be provided.")

    memory_context = tf_memory_bank.build_memory_context_for_item(
        user_id=user_id,
        item_meta=test_item_meta,
        top_k=top_k,
    )

    retrieved_memories = tf_memory_bank.retrieve_for_item(
        user_id=user_id,
        item_meta=test_item_meta,
        top_k=top_k,
    )

    prompt = (
        f"User ID: {user_id}\n\n"
        f"Relevant long-term memory:\n{memory_context}\n\n"
        f"Test item metadata:\n{json.dumps(test_item_meta, ensure_ascii=False)}\n\n"
        "Task:\n"
        "Predict the most likely user rating for this item on a 1-5 integer scale.\n"
        "Use the user's long-term preferences and the item metadata.\n\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{'
        '"predicted_rating": 4, '
        '"reasoning": "short explanation in one sentence"'
        '}'
    )

    response = llm.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=120,
        messages=[
            {
                "role": "system",
                "content": (
                    "You predict a user's likely rating from long-term preference memory. "
                    "Return valid JSON only."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw = (response.choices[0].message.content or "").strip()
    parsed = safe_json_parse(raw)

    predicted_rating = int(round(float(parsed["predicted_rating"])))
    predicted_rating = max(1, min(5, predicted_rating))

    return {
        "user_id": str(user_id),
        "item_id": test_item_meta.get("parent_asin"),
        "predicted_rating": predicted_rating,
        "reasoning": parsed.get("reasoning", ""),
        "used_memory_count": len(retrieved_memories),
    }

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Tuple


def format_item_text_amazon(item_meta: Dict[str, Any]) -> str:
    """
    Convert Amazon item metadata into one text string for the embedding head.
    """
    title = str(item_meta.get("title", "") or "")
    genres = item_meta.get("genres", [])
    if not isinstance(genres, list):
        genres = [genres] if genres else []
    genres_text = ", ".join(str(g) for g in genres if str(g).strip())

    description = str(item_meta.get("description", "") or "")
    return f"title: {title}\ngenres: {genres_text}\ndescription: {description}"


def get_item_meta_for_amazon(
    policy,
    mdata,
    mid: str,
) -> Dict[str, Any]:
    """
    Build normalized item_meta for one Amazon item_id / parent_asin.
    Uses policy.movies + raw item table.
    """
    mid = str(mid)

    movie_match = policy.movies[policy.movies["item_id"].astype(str) == mid]
    if movie_match.empty:
        return {}

    movie = movie_match.iloc[0].to_dict()

    raw_item = mdata.get_item(mid)
    raw_item_dict = raw_item.to_dict() if raw_item is not None else {}

    genres = []
    if "genre_set" in movie and isinstance(movie["genre_set"], set):
        genres = sorted(list(movie["genre_set"]))
    elif "genre" in movie and pd.notna(movie["genre"]):
        genres = [g.strip() for g in str(movie["genre"]).split("|") if g.strip()]

    item_meta = {
        "item_id": mid,
        "title": movie.get("title", ""),
        "genres": genres,
        "description": raw_item_dict.get("details", ""),
    }
    return item_meta


def build_embedding_pairs_for_users(
    attach_uids: List[str],
    df: pd.DataFrame,
    policy,
    mdata,
    memory_mlm,
    include_persona: bool = False,
) -> Tuple[List[Tuple[str, str]], List[int], List[str], List[str]]:
    """
    Build (memory_text, item_text) pairs and aligned labels for a set of users.

    Returns:
        pairs     : list of (memory_text, item_text)
        ratings   : list of int ratings
        uid_list  : list of user_ids
        mid_list  : list of item_ids
    """
    attach_uids = [str(uid) for uid in attach_uids]

    work_df = df.copy()
    work_df[mdata.user_col] = work_df[mdata.user_col].astype(str)
    work_df[mdata.item_col] = work_df[mdata.item_col].astype(str)

    sub_df = work_df[work_df[mdata.user_col].isin(attach_uids)].copy()
    print("Subset size:", len(sub_df))

    pairs = []
    ratings = []
    uid_list = []
    mid_list = []

    for _, row in sub_df.iterrows():
        uid = str(row[mdata.user_col])
        mid = str(row[mdata.item_col])

        try:
            rating = int(float(row[mdata.rating_col]))
        except Exception:
            continue

        if uid not in policy.user_profiles:
            continue

        item_meta = get_item_meta_for_amazon(policy, mdata, mid)
        if not item_meta:
            continue

        persona = ""
        try:
            persona = policy.persona_description(policy.get_user(uid))
        except Exception:
            persona = ""

        prof = policy.user_profiles.get(uid, {})
        top_prefs = {}
        if isinstance(prof.get("preferences"), dict):
            top_prefs = dict(
                sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]
            )

        task_ctx = memory_mlm.build_task_context(
            uid=uid,
            task="rating_prediction",
            item_meta=item_meta,
            persona=persona,
            top_prefs=top_prefs,
        )

        memory_text = "\n".join(task_ctx.get("memories", []))
        if include_persona:
            memory_text = (task_ctx.get("persona", "") + "\n" + memory_text).strip()

        item_text = format_item_text_amazon(item_meta)

        pairs.append((memory_text, item_text))
        ratings.append(rating)
        uid_list.append(uid)
        mid_list.append(mid)

    print(f"Built {len(pairs)} (memory_text, item_text) pairs for {len(attach_uids)} users.")
    return pairs, ratings, uid_list, mid_list


def fit_embedding_rating_head(
    pairs: List[Tuple[str, str]],
    ratings: List[int],
    emb_adapter,
):
    """
    Fit the ordinal/regression head on memory-text + item-text pairs.
    """
    clf_head = EmbeddingRatingRegressorHead(emb_adapter)
    clf_head.fit(pairs, ratings)
    print("Classifier fitted.")
    return clf_head


def evaluate_embedding_rating_head(
    clf_head,
    attach_uids: List[str],
    df: pd.DataFrame,
    policy,
    mdata,
    memory_mlm,
    include_persona: bool = False,
    verbose: bool = False,
):
    """
    Evaluate on a dataframe and return overall + per-user metrics.
    """
    attach_uids = [str(uid) for uid in attach_uids]

    work_df = df.copy()
    work_df[mdata.user_col] = work_df[mdata.user_col].astype(str)
    work_df[mdata.item_col] = work_df[mdata.item_col].astype(str)

    sub_df = work_df[work_df[mdata.user_col].isin(attach_uids)].copy()
    print("Eval subset size:", len(sub_df))

    user_true = defaultdict(list)
    user_pred = defaultdict(list)

    y_true = []
    y_pred = []

    for _, row in sub_df.iterrows():
        uid = str(row[mdata.user_col])
        mid = str(row[mdata.item_col])

        try:
            true_rating = int(float(row[mdata.rating_col]))
        except Exception:
            continue

        if uid not in policy.user_profiles:
            continue

        item_meta = get_item_meta_for_amazon(policy, mdata, mid)
        if not item_meta:
            continue

        persona = ""
        try:
            persona = policy.persona_description(policy.get_user(uid))
        except Exception:
            persona = ""

        prof = policy.user_profiles.get(uid, {})
        top_prefs = {}
        if isinstance(prof.get("preferences"), dict):
            top_prefs = dict(
                sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]
            )

        task_ctx = memory_mlm.build_task_context(
            uid=uid,
            task="rating_prediction",
            item_meta=item_meta,
            persona=persona,
            top_prefs=top_prefs,
        )

        memory_text = "\n".join(task_ctx.get("memories", []))
        if include_persona:
            memory_text = (task_ctx.get("persona", "") + "\n" + memory_text).strip()

        item_text = format_item_text_amazon(item_meta)

        pred_rating = clf_head.predict_rating(memory_text, item_text)

        y_true.append(true_rating)
        y_pred.append(pred_rating)

        user_true[uid].append(true_rating)
        user_pred[uid].append(pred_rating)

        if verbose:
            loss = abs(pred_rating - true_rating)
            print(f"uid={uid} mid={mid} | loss={loss} pred={pred_rating} true={true_rating}")

    if not y_true:
        print("No evaluation samples found.")
        return {
            "overall": {"MAE": None, "RMSE": None, "n": 0},
            "per_user": {},
        }

    y_true_arr = np.array(y_true, dtype=float)
    y_pred_arr = np.array(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_pred_arr - y_true_arr)))
    rmse = float(np.sqrt(np.mean((y_pred_arr - y_true_arr) ** 2)))

    per_user = {}
    for uid in sorted(user_true.keys()):
        y_t = np.array(user_true[uid], dtype=float)
        y_p = np.array(user_pred[uid], dtype=float)

        mae_u = float(np.mean(np.abs(y_p - y_t)))
        rmse_u = float(np.sqrt(np.mean((y_p - y_t) ** 2)))

        per_user[uid] = {
            "MAE": mae_u,
            "RMSE": rmse_u,
            "n": int(len(y_t)),
        }

    results = {
        "overall": {
            "MAE": mae,
            "RMSE": rmse,
            "n": int(len(y_true_arr)),
        },
        "per_user": per_user,
    }

    print(f"Overall MAE  = {mae:.3f}")
    print(f"Overall RMSE = {rmse:.3f}")

    for uid in sorted(per_user.keys()):
        row = per_user[uid]
        print(f"uid={uid} | MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} n={row['n']}")

    return results

def predict_fixed_item_ratings(
    clf_head,
    selected_uids: list,
    fixed_items_df: pd.DataFrame,
    policy,
    memory_mlm,
    include_persona: bool = False,
    verbose: bool = False,
):
    """
    Predict ratings for every user on the same fixed set of items.

    Inputs:
        selected_uids   : list of users
        fixed_items_df  : dataframe with columns like
                          [item_id, title, genres, description]
        policy          : LLMPolicy
        memory_controller
        include_persona : whether to prepend persona to memory text
        verbose         : print each prediction

    Returns:
        pred_df : one row per (user, item) with predicted rating
    """
    selected_uids = [str(uid) for uid in selected_uids]

    all_rows = []

    for uid in selected_uids:
        prof = policy.user_profiles[uid]

        persona = ""
        try:
            persona = policy.persona_description(policy.get_user(uid))
        except Exception:
            persona = ""

        top_prefs = {}
        if isinstance(prof.get("preferences"), dict):
            top_prefs = dict(
                sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]
            )

        for _, item_row in fixed_items_df.iterrows():
            item_meta = {
                "item_id": str(item_row["item_id"]),
                "title": item_row.get("title", ""),
                "genres": item_row.get("genres", []),
                "description": item_row.get("description", ""),
            }

            task_ctx = memory_mlm.build_task_context(
                uid=uid,
                task="rating_prediction",
                item_meta=item_meta,
                persona=persona,
                top_prefs=top_prefs,
            )

            memory_text = "\n".join(task_ctx.get("memories", []))
            if include_persona:
                memory_text = (task_ctx.get("persona", "") + "\n" + memory_text).strip()

            item_text = format_item_text_amazon(item_meta)

            pred_rating = clf_head.predict_rating(memory_text, item_text)

            row_out = {
                "user_id": uid,
                "item_id": item_meta["item_id"],
                "title": item_meta["title"],
                "genres": item_meta["genres"],
                "description": item_meta["description"],
                "pred_rating": pred_rating,
                "memory_count": len(task_ctx.get("memories", [])),
            }

            all_rows.append(row_out)

            if verbose:
                print(
                    f"user={uid} | item={item_meta['item_id']} | "
                    f"pred_rating={pred_rating} | title={item_meta['title']}"
                )

    pred_df = pd.DataFrame(all_rows)
    print(f"Built predictions for {len(selected_uids)} users x {len(fixed_items_df)} items = {len(pred_df)} rows")
    return pred_df

def predict_item_ratings(
    clf_head,
    selected_uids: list,
    df: pd.DataFrame,
    policy,
    memory_mlm,
    include_persona: bool = False,
    verbose: bool = False,
    mode: str = "fixed",   # "fixed" or "ground_truth"
):
    """
    Predict ratings in two modes:

    mode="fixed":
        Every selected user is scored on every unique item in df.

    mode="ground_truth":
        Only existing user-item rows in df are scored.

    Expected dataframe formats:
      - raw Amazon interaction df with parent_asin
      - or normalized df with item_id

    Returns:
        pred_df : dataframe with predicted ratings
    """
    selected_uids = [str(uid) for uid in selected_uids]
    work_df = df.copy()

    # detect item id column
    if "item_id" in work_df.columns:
        item_id_col = "item_id"
    elif "parent_asin" in work_df.columns:
        item_id_col = "parent_asin"
    else:
        raise KeyError(
            f"df must contain 'item_id' or 'parent_asin'. "
            f"Found columns: {work_df.columns.tolist()}"
        )

    # detect user column
    if "user_id" not in work_df.columns:
        raise KeyError(
            f"df must contain 'user_id'. Found columns: {work_df.columns.tolist()}"
        )

    work_df["user_id"] = work_df["user_id"].astype(str)
    work_df[item_id_col] = work_df[item_id_col].astype(str)

    all_rows = []

    if mode == "fixed":
        # use all unique items for every selected user
        item_df = work_df.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)

        for uid in selected_uids:
            prof = policy.user_profiles[uid]

            persona = ""
            try:
                persona = policy.persona_description(policy.get_user(uid))
            except Exception:
                persona = ""

            top_prefs = {}
            if isinstance(prof.get("preferences"), dict):
                top_prefs = dict(
                    sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]
                )

            for _, item_row in item_df.iterrows():
                mid = str(item_row[item_id_col])

                item_meta = get_item_meta_for_amazon(policy, policy.ds, mid)
                if not item_meta:
                    item_meta = {
                        "item_id": mid,
                        "title": item_row.get("title", ""),
                        "genres": item_row.get("genres", []),
                        "description": item_row.get("description", ""),
                    }

                task_ctx = memory_mlm.build_task_context(
                    uid=uid,
                    task="rating_prediction",
                    item_meta=item_meta,
                    persona=persona,
                    top_prefs=top_prefs,
                )

                memory_text = "\n".join(task_ctx.get("memories", []))
                if include_persona:
                    memory_text = (task_ctx.get("persona", "") + "\n" + memory_text).strip()

                item_text = format_item_text_amazon(item_meta)
                pred_rating = clf_head.predict_rating(memory_text, item_text)

                row_out = {
                    "user_id": uid,
                    "item_id": item_meta["item_id"],
                    "title": item_meta.get("title", ""),
                    "genres": item_meta.get("genres", []),
                    "description": item_meta.get("description", ""),
                    "pred_rating": pred_rating,
                    "memory_count": len(task_ctx.get("memories", [])),
                }
                all_rows.append(row_out)

                if verbose:
                    print(
                        f"[fixed] user={uid} | item={item_meta['item_id']} | "
                        f"pred_rating={pred_rating} | title={item_meta.get('title', '')}"
                    )

    elif mode == "ground_truth":
        # only score the actual user-item rows in the dataframe
        gt_df = work_df[work_df["user_id"].isin(selected_uids)].copy().reset_index(drop=True)

        for _, row in gt_df.iterrows():
            uid = str(row["user_id"])
            mid = str(row[item_id_col])

            prof = policy.user_profiles[uid]

            persona = ""
            try:
                persona = policy.persona_description(policy.get_user(uid))
            except Exception:
                persona = ""

            top_prefs = {}
            if isinstance(prof.get("preferences"), dict):
                top_prefs = dict(
                    sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]
                )

            item_meta = get_item_meta_for_amazon(policy, policy.ds, mid)
            if not item_meta:
                item_meta = {
                    "item_id": mid,
                    "title": row.get("title", ""),
                    "genres": row.get("genres", []),
                    "description": row.get("description", ""),
                }

            task_ctx = memory_mlm.build_task_context(
                uid=uid,
                task="rating_prediction",
                item_meta=item_meta,
                persona=persona,
                top_prefs=top_prefs,
            )

            memory_text = "\n".join(task_ctx.get("memories", []))
            if include_persona:
                memory_text = (task_ctx.get("persona", "") + "\n" + memory_text).strip()

            item_text = format_item_text_amazon(item_meta)
            pred_rating = clf_head.predict_rating(memory_text, item_text)

            row_out = {
                "user_id": uid,
                "item_id": item_meta["item_id"],
                "title": item_meta.get("title", ""),
                "genres": item_meta.get("genres", []),
                "description": item_meta.get("description", ""),
                "pred_rating": pred_rating,
                "memory_count": len(task_ctx.get("memories", [])),
            }

            # keep true rating if present
            if policy.ds.rating_col in row.index:
                row_out["true_rating"] = row[policy.ds.rating_col]

            all_rows.append(row_out)

            if verbose:
                print(
                    f"[ground_truth] user={uid} | item={item_meta['item_id']} | "
                    f"pred_rating={pred_rating} | title={item_meta.get('title', '')}"
                )

    else:
        raise ValueError("mode must be either 'fixed' or 'ground_truth'")

    pred_df = pd.DataFrame(all_rows)

    if mode == "fixed":
        print(
            f"Built fixed predictions for {len(selected_uids)} users x "
            f"{len(work_df.drop_duplicates(subset=[item_id_col]))} unique items "
            f"= {len(pred_df)} rows"
        )
    else:
        print(f"Built ground-truth predictions for {len(pred_df)} user-item rows")

    return pred_df


# def predict_fixed_item_ratings(
#     clf_head,
#     selected_uids: list,
#     fixed_items_df: pd.DataFrame,
#     policy,
#     memory_controller,
#     include_persona: bool = False,
#     verbose: bool = False,
# ):
#     """
#     Predict ratings for every user on the provided item dataframe.

#     The input dataframe may be either:
#       1. a normalized fixed-items dataframe with columns like:
#          item_id, title, genres, description
#       2. a raw Amazon interaction dataframe containing parent_asin

#     Returns:
#         pred_df : one row per (user, item) with predicted rating
#     """
#     selected_uids = [str(uid) for uid in selected_uids]
#     work_df = fixed_items_df.copy()

#     # detect item id column
#     if "item_id" in work_df.columns:
#         item_id_col = "item_id"
#     elif "parent_asin" in work_df.columns:
#         item_id_col = "parent_asin"
#     else:
#         raise KeyError(
#             f"fixed_items_df must contain 'item_id' or 'parent_asin'. "
#             f"Found columns: {work_df.columns.tolist()}"
#         )

#     work_df[item_id_col] = work_df[item_id_col].astype(str)

#     # keep only unique items
#     work_df = work_df.drop_duplicates(subset=[item_id_col]).reset_index(drop=True)

#     all_rows = []

#     for uid in selected_uids:
#         prof = policy.user_profiles[uid]

#         persona = ""
#         try:
#             persona = policy.persona_description(policy.get_user(uid))
#         except Exception:
#             persona = ""

#         top_prefs = {}
#         if isinstance(prof.get("preferences"), dict):
#             top_prefs = dict(
#                 sorted(prof["preferences"].items(), key=lambda x: -x[1])[:5]
#             )

#         for _, item_row in work_df.iterrows():
#             mid = str(item_row[item_id_col])

#             # build normalized item_meta from policy + raw data
#             item_meta = get_item_meta_for_amazon(policy, policy.ds, mid)
#             if not item_meta:
#                 # fallback from current row if lookup fails
#                 item_meta = {
#                     "item_id": mid,
#                     "title": item_row.get("title", ""),
#                     "genres": item_row.get("genres", []),
#                     "description": item_row.get("description", ""),
#                 }

#             task_ctx = memory_mlm.build_task_context(
#                 uid=uid,
#                 task="rating_prediction",
#                 item_meta=item_meta,
#                 persona=persona,
#                 top_prefs=top_prefs,
#             )

#             memory_text = "\n".join(task_ctx.get("memories", []))
#             if include_persona:
#                 memory_text = (task_ctx.get("persona", "") + "\n" + memory_text).strip()

#             item_text = format_item_text_amazon(item_meta)
#             pred_rating = clf_head.predict_rating(memory_text, item_text)

#             row_out = {
#                 "user_id": uid,
#                 "item_id": item_meta["item_id"],
#                 "title": item_meta.get("title", ""),
#                 "genres": item_meta.get("genres", []),
#                 "description": item_meta.get("description", ""),
#                 "pred_rating": pred_rating,
#                 "memory_count": len(task_ctx.get("memories", [])),
#             }

#             all_rows.append(row_out)

#             if verbose:
#                 print(
#                     f"user={uid} | item={item_meta['item_id']} | "
#                     f"pred_rating={pred_rating} | title={item_meta.get('title', '')}"
#                 )

#     pred_df = pd.DataFrame(all_rows)
#     print(
#         f"Built predictions for {len(selected_uids)} users x "
#         f"{len(work_df)} unique items = {len(pred_df)} rows"
#     )
#     return pred_df

def build_fixed_items_df_from_test(
    test_df: pd.DataFrame,
    mdata,
    policy,
    n_items: int = 100,
    seed: int = 42,
    item_ids: list = None,
):
    """
    Build one shared fixed item dataframe from test_df item pool.

    Returns:
        fixed_items_df with columns:
            item_id, title, genres, description
    """
    item_col = mdata.item_col

    work_df = test_df.copy()
    work_df[item_col] = work_df[item_col].astype(str)

    if item_ids is not None:
        fixed_item_ids = [str(x) for x in item_ids]
    else:
        unique_items = work_df[item_col].dropna().astype(str).drop_duplicates().tolist()
        rng = np.random.RandomState(seed)
        if len(unique_items) <= n_items:
            fixed_item_ids = unique_items
        else:
            fixed_item_ids = rng.choice(unique_items, size=n_items, replace=False).tolist()

    rows = []
    for iid in fixed_item_ids:
        movie_match = policy.movies[policy.movies["item_id"].astype(str) == str(iid)]
        if movie_match.empty:
            continue

        movie = movie_match.iloc[0].to_dict()
        raw_item = mdata.get_item(iid)
        raw_item_dict = raw_item.to_dict() if raw_item is not None else {}

        genres = []
        if "genre_set" in movie and isinstance(movie["genre_set"], set):
            genres = sorted(list(movie["genre_set"]))
        elif "genre" in movie and pd.notna(movie["genre"]):
            genres = [g.strip() for g in str(movie["genre"]).split("|") if g.strip()]

        rows.append({
            "item_id": str(iid),
            "title": movie.get("title", ""),
            "genres": genres,
            "description": raw_item_dict.get("details", ""),
        })

    fixed_items_df = pd.DataFrame(rows).drop_duplicates(subset=["item_id"]).reset_index(drop=True)

    print(f"Fixed shared items: {len(fixed_items_df)}")
    return fixed_items_df, fixed_item_ids

def build_user_category_lists_from_predictions(
    pred_df: pd.DataFrame,
    top_k: int = 10,
    min_rating: float = None,
    user_col: str = "user_id",
    rating_col: str = "pred_rating",
    genre_col: str = "genres",
):
    """
    Convert predicted user-item ratings into per-user picked category lists.

    Rules:
      - if min_rating is given: keep items with pred_rating >= min_rating
      - otherwise: keep top_k items per user

    Returns:
        dict[user_id] -> list of category strings
    """
    if pred_df.empty:
        return {}

    work = pred_df.copy()
    work[user_col] = work[user_col].astype(str)

    user_to_categories = {}

    for uid, g in work.groupby(user_col):
        g = g.sort_values(rating_col, ascending=False).copy()

        if min_rating is not None:
            picked = g[g[rating_col] >= float(min_rating)].copy()
        else:
            picked = g.head(top_k).copy()

        cats = []
        for _, row in picked.iterrows():
            genres = row.get(genre_col, [])
            if genres is None:
                continue
            if not isinstance(genres, list):
                genres = [genres]
            cats.extend([str(x) for x in genres if str(x).strip()])

        user_to_categories[str(uid)] = cats

    return user_to_categories


def compute_entropy_and_snd_from_predictions(
    pred_df: pd.DataFrame,
    top_k: int = 10,
    min_rating: float = None,
    distance: str = "js",
    user_col: str = "user_id",
    rating_col: str = "pred_rating",
    genre_col: str = "genres",
):
    """
    Compute:
      1) per-user category entropy
      2) overall SND across users

    Returns:
        entropy_df : dataframe with entropy per user
        snd_value  : overall SND
        snd_matrix : pairwise user distance dict
        user_to_categories : dict[user_id] -> category list
    """
    user_to_categories = build_user_category_lists_from_predictions(
        pred_df=pred_df,
        top_k=top_k,
        min_rating=min_rating,
        user_col=user_col,
        rating_col=rating_col,
        genre_col=genre_col,
    )

    entropy_rows = []
    for uid, cats in user_to_categories.items():
        ent = category_entropy(cats, base=2)
        entropy_rows.append({
            "user_id": uid,
            "n_categories_used": len(cats),
            "unique_categories": len(set(cats)),
            "category_entropy": ent,
        })

    entropy_df = pd.DataFrame(entropy_rows).sort_values("user_id").reset_index(drop=True)

    snd_value, snd_matrix = snd_from_user_category_lists(
        user_to_categories=user_to_categories,
        distance=distance,
        base=2,
    )

    return entropy_df, snd_value, snd_matrix, user_to_categories