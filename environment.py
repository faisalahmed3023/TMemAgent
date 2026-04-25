from recommendation_model import LightGCN
from recommendation_model import MF
from recommendation_model import SASRec

import numpy as np
import math
from typing import List, Optional, Tuple

def sample_candidates_including_gt(data, uid: int, k: int, rng: np.random.RandomState) -> Tuple[List[int], int]:
    """
    Pick a ground-truth positive for uid (rating >= 4) and fill the rest with random items.
    Returns (candidates, gt_item). If no positive exists, falls back to random gt.
    """
    df_pos = data.get_user_rated_items(uid, min_rating=4)
    if not df_pos.empty:
        gt_item = int(df_pos.sample(n=1, random_state=rng.randint(1, 1_000_000))["item_id"].iloc[0])
    else:
        # fallback to random
        gt_item = int(data.movies.sample(n=1, random_state=rng.randint(1, 1_000_000))["item_id"].iloc[0])

    candidates = {gt_item}
    # fill
    while len(candidates) < k:
        mid = int(data.movies.sample(n=1, random_state=rng.randint(1, 1_000_000))["item_id"].iloc[0])
        candidates.add(mid)
    return list(candidates), gt_item

# =============================================================================
# Candidate sampler — Amazon-aware (replaces sample_candidates_including_gt)
# =============================================================================

def sample_candidates_amazon(
    policy,                                      # LLMPolicy instance
    uid,
    k:              int   = 10,
    min_rating:     float = 4.0,
    hard_neg_ratio: float = 0.4,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Sample k candidates (1 ground-truth + k-1 negatives) for one user.

    Ground-truth selection
    ----------------------
    Prefers the item where  (user_rating - item_avg_rating)  is highest.
    This "surprise" score identifies the user's most genuine strong preference,
    as opposed to simply rating a universally well-received item.
    Falls back to highest-rated item when average_rating is unavailable.

    Negative types
    --------------
    hard  (hard_neg_ratio fraction of k-1):
        Same top-level category as GT, not in user's positives.
        Forces the latent to distinguish fine-grained preferences.
    easy  (remaining slots):
        Random items with no category overlap.

    Returns
    -------
    (candidates, gt_item_id)  — candidates is a shuffled list of k str item IDs
                                 always containing gt_item_id.
    ([], None)                — if the user has no rated items at all.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    uid = str(uid)

    # ── Step 1: Get positive items for this user ──────────────────────────
    # FIX: call policy.get_user_rated_items() directly — NOT policy.data.get_user_rated_items()
    pos_df = policy.get_user_rated_items(uid, min_rating=min_rating)

    # Fall back to any rated item if no strong positives exist
    if pos_df.empty:
        pos_df = policy.get_user_rated_items(uid, min_rating=1.0)
    if pos_df.empty:
        return [], None

    # ── Step 2: Ensure item_id column exists ──────────────────────────────
    # get_user_rated_items() in LLMPolicy guarantees this column, but guard anyway
    if "item_id" not in pos_df.columns:
        pos_df = pos_df.copy()
        pos_df["item_id"] = pos_df[policy.ds.item_col].astype(str)

    # ── Step 3: Select ground-truth item via surprise score ───────────────
    # FIX: use policy._item_lookup — NOT policy.data._items
    def surprise_score(row) -> float:
        iid  = str(row["item_id"])
        item = policy._item_lookup.get(iid)      # FIX: _item_lookup not data._items
        if item is None:
            return 0.0
        avg = item.get("average_rating")
        if avg is None or (isinstance(avg, float) and math.isnan(avg)):
            return 0.0
        try:
            return float(row.get(policy.ds.rating_col, 3.0)) - float(avg)
        except (TypeError, ValueError):
            return 0.0

    pos_df       = pos_df.copy()
    pos_df["_s"] = pos_df.apply(surprise_score, axis=1)
    pos_df       = pos_df.sort_values("_s", ascending=False)
    gt_item      = str(pos_df.iloc[0]["item_id"])

    # ── Step 4: Get GT category set ──────────────────────────────────────
    # FIX: _item_lookup values may be pd.Series — access by key, not .get()
    gt_row = policy._item_lookup.get(gt_item)
    if gt_row is None:
        gt_cats = set()
    else:
        try:
            gs = gt_row["genre_set"]
            gt_cats = gs if isinstance(gs, set) else \
                      set(str(gt_row["genre"]).split("|")) - {"", "Unknown"}
        except (KeyError, TypeError):
            gt_cats = set()

    # ── Step 5: Build exclusion set (user's positives) ───────────────────
    positive_set = set(pos_df["item_id"].astype(str).values)
    positive_set.add(gt_item)

    # FIX: use policy.movies directly — NOT policy.data.movies
    all_iids = policy.movies["item_id"].tolist()

    # ── Step 6: Sample hard negatives ────────────────────────────────────
    # Same category as GT, not in user's positives.
    #
    # FIX: _item_lookup stores pd.Series rows from iterrows().
    # Calling  (series or {}).get("genre_set")  forces pandas to evaluate
    # the Series as a boolean → raises "truth value of a Series is ambiguous".
    #
    # Solution: helper function that safely extracts genre_set as a Python set,
    # handling both pd.Series rows and plain dict rows.
    def _safe_genre_set(iid: str) -> set:
        item = policy._item_lookup.get(iid)
        if item is None:
            return set()
        # pd.Series: use item["genre_set"] directly (no `or {}` needed)
        try:
            gs = item["genre_set"]
        except (KeyError, TypeError):
            return set()
        # Ensure it is actually a Python set, not a pd.Series or NaN
        if isinstance(gs, set):
            return gs
        if isinstance(gs, float):   # NaN from missing data
            return set()
        # Fallback: rebuild from genre string
        try:
            genre_str = item["genre"] if not isinstance(item["genre"], float) else "Unknown"
            return set(str(genre_str).split("|")) - {"", "Unknown"}
        except Exception:
            return set()

    hard_pool = [
        iid for iid in all_iids
        if iid not in positive_set
        and len(_safe_genre_set(iid) & gt_cats) > 0
    ]
    n_hard    = max(1, int((k - 1) * hard_neg_ratio))
    hard_negs = list(
        rng.choice(hard_pool, size=min(n_hard, len(hard_pool)), replace=False)
    ) if hard_pool else []

    # ── Step 7: Sample easy negatives ────────────────────────────────────
    # Random items with no overlap — fills remaining slots
    n_easy    = (k - 1) - len(hard_negs)
    easy_pool = [
        iid for iid in all_iids
        if iid not in positive_set and iid not in hard_negs
    ]
    easy_negs = list(
        rng.choice(easy_pool, size=min(n_easy, len(easy_pool)), replace=False)
    ) if easy_pool else []

    # ── Step 8: Assemble and shuffle ─────────────────────────────────────
    candidates = [gt_item] + hard_negs + easy_negs
    rng.shuffle(candidates)
    return candidates[:k], gt_item

mf1 = MF(
        behavior_csv_path="Amazon_reviews/Behavior_Books_filtered2.csv",
        latent_dim=128,
        lr=5e-4,
        weight_decay=1e-5,
        batch_size=2048
    )

mf1.load_and_prepare(dedup_last=True)
mf1.split_sequential(train_ratio=0.7)

mf1.fit(
        epochs=100,
        eval_every=10,
        early_stop=False,
        patience=5,
        eval_k=10,
        eval_n_neg=999
    )

for K in [10, 50, 100]:
    r1, nd1 = mf1.evaluate_sampled(k=K, n_neg=999)
    print(f"[Sampled] Recall@{K}: {r1:.4f} | NDCG@{K}: {nd1:.4f}")