# ===============================
# Data Schema Reference
# ===============================
# User Behavior fields:
#   user_id, parent_asin, title, text, rating, timestamp
#
# Item Meta fields:
#   parent_asin, title, categories, details, average_rating,
#   rating_number, price, bought_together

import json
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from collections import Counter
from datetime import datetime
from Amazon_reviews.utils import UserStateStore
from copy import deepcopy


# ===============================
# Memory Entry
# ===============================

class MemoryEntry:
    """
    A single memory unit capturing one user interaction fused with item metadata.

    Attributes:
        user_behavior   : Raw user behavior fields (user_id, parent_asin, title,
                          text, rating, timestamp)
        item_meta       : Raw item metadata fields (parent_asin, title, categories,
                          details, average_rating, rating_number, price, bought_together)
        importance      : Float [0,1] reflecting interaction significance
        characteristics : Derived descriptive traits from both behavior + item meta
    """

    def __init__(
        self,
        user_behavior: dict,
        item_meta: dict,
        importance: float,
        characteristics: dict,
    ):
        self.user_behavior = user_behavior
        self.item_meta = item_meta
        self.importance = importance
        self.characteristics = characteristics

    def to_dict(self) -> dict:
        return {
            "user_behavior":   self.user_behavior,
            "item_meta":       self.item_meta,
            "importance":      self.importance,
            "characteristics": self.characteristics,
        }

    def __repr__(self):
        return (
            f"MemoryEntry("
            f"asin={self.user_behavior.get('parent_asin')}, "
            f"rating={self.user_behavior.get('rating')}, "
            f"importance={self.importance:.2f})"
        )


# ===============================
# Short-Term Memory (STM)
# ===============================

class DomainSpecificMemoryBank:
    """
    Per-user Domain Specific Memory Bank that:
      1. Accepts user behavior + item metadata and stores them as MemoryEntry objects.
      2. Each entry carries: user_behavior, item_meta, importance, characteristics.
      3. Once `capacity` entries accumulate, generates a high-level user preference
         insight via LLM and transfers the full batch to the MemoryController.
      4. Resets after every transfer, ready for the next interaction batch.

    New:
      - Writes recent snapshots into UserStateStore.short_term_memory
      - Writes/update preference & trait information into UserStateStore
    """

    def __init__(
        self,
        user_id: str,
        llm,
        memory_controller,
        user_state_store: UserStateStore,
        capacity: int = 5,
        user_profile: Optional[Dict[str, Any]] = None,
    ):
        self.user_id = str(user_id)
        self.llm = llm
        self.memory_controller = memory_controller
        self.user_state_store = user_state_store
        self.capacity = capacity

        # Ensure shared storage has this user
        self.user_state_store.get_or_create_user(self.user_id, profile=user_profile)

        # Core per-user STM buffer
        self._entries: List[MemoryEntry] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        user_behavior: dict,
        item_meta: dict,
        importance: Optional[float] = None,
    ) -> Optional[dict]:
        """
        Add one interaction to this user's STM.

        Returns:
            Transfer payload dict if capacity was reached, otherwise None.
        """
        if str(user_behavior.get("user_id")) != self.user_id:
            raise ValueError(
                f"user_behavior user_id '{user_behavior.get('user_id')}' "
                f"does not match STM user_id '{self.user_id}'"
            )

        if importance is None:
            importance = self._compute_importance(user_behavior, item_meta)

        characteristics = self._extract_characteristics(user_behavior, item_meta)

        entry = MemoryEntry(
            user_behavior=user_behavior,
            item_meta=item_meta,
            importance=importance,
            characteristics=characteristics,
        )
        self._entries.append(entry)

        # Store lightweight STM snapshot into shared user store
        self.user_state_store.append_short_term_memory(
            self.user_id,
            {
                "parent_asin": user_behavior.get("parent_asin"),
                "title": user_behavior.get("title") or item_meta.get("title"),
                "rating": user_behavior.get("rating"),
                "timestamp": user_behavior.get("timestamp"),
                "importance": importance,
                "characteristics": deepcopy(characteristics),
            },
        )

        # Update some simple interaction stats in shared store
        user = self.user_state_store.get_or_create_user(self.user_id)
        total_seen = int(user.interaction_stats.get("total_interactions_seen", 0)) + 1
        self.user_state_store.update_interaction_stats(
            self.user_id,
            {
                "total_interactions_seen": total_seen,
                "last_interaction_asin": user_behavior.get("parent_asin"),
                "last_interaction_time": user_behavior.get("timestamp"),
            },
        )

        # Trigger transfer once capacity is reached
        if len(self._entries) >= self.capacity:
            return self._transfer()

        return None

    def peek(self) -> List[dict]:
        """Return current STM entries as a list of dicts (non-destructive)."""
        return [e.to_dict() for e in self._entries]

    def reset(self):
        """Manually clear local STM entries only."""
        self._entries = []

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        return (
            f"ShortTermMemory(user={self.user_id}, "
            f"entries={len(self._entries)}/{self.capacity})"
        )

    # ------------------------------------------------------------------
    # Importance Computation
    # ------------------------------------------------------------------

    def _compute_importance(self, user_behavior: dict, item_meta: dict) -> float:
        """
        Auto-compute importance [0,1] from available signals:
          - User rating vs item average_rating  → deviation signal
          - Review text length                  → engagement signal
          - Item popularity (rating_number)     → context weight
        """
        score = 0.5

        try:
            user_rating = float(user_behavior.get("rating", 3.0))
            avg_rating = float(item_meta.get("average_rating", 3.0))
            deviation = abs(user_rating - avg_rating)
            score += min(deviation * 0.1, 0.2)
        except (TypeError, ValueError):
            pass

        text = user_behavior.get("text") or ""
        word_count = len(text.split())
        if word_count > 100:
            score += 0.2
        elif word_count > 30:
            score += 0.1

        try:
            user_rating = float(user_behavior.get("rating", 3.0))
            if user_rating >= 4.5 or user_rating <= 1.5:
                score += 0.1
        except (TypeError, ValueError):
            pass

        return round(min(max(score, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # Characteristics Extraction
    # ------------------------------------------------------------------

    def _extract_characteristics(self, user_behavior: dict, item_meta: dict) -> dict:
        """
        Derive structured characteristics from user behavior + item metadata.
        """
        chars = {}

        # Rating level
        try:
            r = float(user_behavior.get("rating", 3.0))
            chars["rating_level"] = (
                "high" if r >= 4.0 else
                "medium" if r >= 2.5 else
                "low"
            )
            chars["raw_rating"] = r
        except (TypeError, ValueError):
            pass

        # Rating deviation
        try:
            user_r = float(user_behavior.get("rating", 3.0))
            avg_r = float(item_meta.get("average_rating", 3.0))
            chars["rating_deviation"] = round(user_r - avg_r, 2)
        except (TypeError, ValueError):
            pass

        # Sentiment hint
        text = (user_behavior.get("text") or "").lower()
        if text:
            positive_kw = {"love", "great", "excellent", "amazing", "perfect", "best"}
            negative_kw = {"hate", "terrible", "awful", "worst", "poor", "bad", "broken"}
            pos = sum(1 for w in positive_kw if w in text)
            neg = sum(1 for w in negative_kw if w in text)
            chars["sentiment_hint"] = (
                "positive" if pos > neg else
                "negative" if neg > pos else
                "neutral"
            )
            chars["review_length"] = len(text.split())

        # Timestamp recency
        try:
            ts = user_behavior.get("timestamp")
            if ts:
                dt = (
                    datetime.fromtimestamp(int(ts) / 1000)
                    if int(ts) > 1e10
                    else datetime.fromtimestamp(int(ts))
                )
                age_days = (datetime.now() - dt).days
                chars["recency"] = (
                    "recent" if age_days <= 30 else
                    "moderate" if age_days <= 180 else
                    "old"
                )
        except (TypeError, ValueError, OSError):
            pass

        # Categories
        categories = item_meta.get("categories")
        if categories:
            chars["categories"] = (
                categories if isinstance(categories, list)
                else [str(categories)]
            )

        # Price tier
        try:
            price = float(
                str(item_meta.get("price", "0"))
                .replace("$", "")
                .replace(",", "")
            )
            chars["price_tier"] = (
                "budget" if price < 20 else
                "mid" if price < 100 else
                "premium"
            )
            chars["raw_price"] = price
        except (TypeError, ValueError):
            pass

        # Popularity
        try:
            rating_num = int(item_meta.get("rating_number", 0))
            chars["is_popular"] = rating_num >= 1000
            chars["rating_number"] = rating_num
        except (TypeError, ValueError):
            pass

        # Bought together
        bt = item_meta.get("bought_together")
        chars["has_bundle"] = bool(bt and len(bt) > 0)

        return chars

    # ------------------------------------------------------------------
    # LLM Insight Generation
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        """Build a structured LLM prompt from all current entries."""
        lines = []
        for i, entry in enumerate(self._entries, 1):
            ub = entry.user_behavior
            meta = entry.item_meta
            ch = entry.characteristics
            lines.append(
                f"Interaction {i}:\n"
                f"  Item         : {ub.get('title') or meta.get('title', 'Unknown')}\n"
                f"  Categories   : {meta.get('categories', 'N/A')}\n"
                f"  User Rating  : {ub.get('rating')} "
                f"(avg={meta.get('average_rating')}, n={meta.get('rating_number')})\n"
                f"  Price        : {meta.get('price', 'N/A')}\n"
                f"  Review       : {str(ub.get('text', ''))[:200]}\n"
                f"  Traits       : rating_level={ch.get('rating_level')}, "
                f"sentiment={ch.get('sentiment_hint')}, "
                f"price_tier={ch.get('price_tier')}, "
                f"popular={ch.get('is_popular')}"
            )
        return "\n\n".join(lines)

    def _generate_insight(self) -> str:
        """
        Call the LLM to produce a one-sentence high-level user preference insight.
        """
        prompt = (
            f"The following are {len(self._entries)} recent item interactions "
            f"for a single user:\n\n"
            f"{self._build_prompt()}\n\n"
            "Based on these interactions, infer ONE concise sentence describing "
            "this user's general preference pattern — covering what item types, "
            "price range, quality expectations, or behavioural tendencies they show. "
            "Do NOT list individual items. Return one sentence only."
        )
        try:
            response = self.llm.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=80,
                messages=[
                    {"role": "system", "content": "Return one sentence only."},
                    {"role": "user", "content": prompt},
                ],
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            return (
                "User tends to engage with moderately priced, "
                "well-reviewed items across consistent categories."
            )

    # ------------------------------------------------------------------
    # Transfer & Aggregation
    # ------------------------------------------------------------------

    def _merge_characteristics(self) -> dict:
        """
        Aggregate characteristics across current entries.
        """
        merged = {}
        all_keys = set(k for e in self._entries for k in e.characteristics)

        numeric_keys = {
            "raw_rating", "raw_price", "rating_number",
            "rating_deviation", "review_length"
        }

        for key in all_keys:
            values = [
                e.characteristics[key]
                for e in self._entries
                if key in e.characteristics
            ]
            if not values:
                continue

            if key in numeric_keys:
                try:
                    nums = [float(v) for v in values]
                    merged[key] = {
                        "mean": round(sum(nums) / len(nums), 4),
                        "min": min(nums),
                        "max": max(nums),
                    }
                except (TypeError, ValueError):
                    pass

            elif key == "categories":
                flat = [
                    c for sublist in values
                    for c in (sublist if isinstance(sublist, list) else [sublist])
                ]
                top = Counter(flat).most_common(3)
                merged["top_categories"] = [cat for cat, _ in top]

            else:
                dominant, count = Counter(values).most_common(1)[0]
                merged[key] = {
                    "dominant": dominant,
                    "count": count,
                    "total": len(values),
                    "distribution": dict(Counter(values)),
                }

        return merged

    def _transfer(self) -> dict:
        """
        1. Generate LLM insight from current batch.
        2. Package transfer payload.
        3. Send to MemoryController.
        4. Update shared UserStateStore.
        5. Reset local STM buffer.
        """
        insight = self._generate_insight()
        entries_snap = [e.to_dict() for e in self._entries]
        avg_imp = sum(e.importance for e in self._entries) / len(self._entries)
        merged_chars = self._merge_characteristics()

        payload = {
            "user_id": self.user_id,
            "insight": insight,
            "entries": entries_snap,
            "avg_importance": round(avg_imp, 4),
            "merged_characteristics": merged_chars,
            "batch_size": len(self._entries),
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Update shared store before reset
        self.user_state_store.append_long_term_memory(self.user_id, payload)

        # Keep latest preference insight
        self.user_state_store.update_preferences(
            self.user_id,
            {
                "latest_batch_insight": insight,
                "last_top_categories": merged_chars.get("top_categories", []),
                "last_avg_importance": round(avg_imp, 4),
            },
        )

        # Keep latest traits summary
        trait_update = {}
        for key in ["rating_level", "sentiment_hint", "price_tier", "recency"]:
            if key in merged_chars and isinstance(merged_chars[key], dict):
                trait_update[f"dominant_{key}"] = merged_chars[key].get("dominant")
        if trait_update:
            self.user_state_store.update_traits(self.user_id, trait_update)

        # Update stats
        user = self.user_state_store.get_or_create_user(self.user_id)
        total_batches = int(user.interaction_stats.get("total_batches_transferred", 0)) + 1
        self.user_state_store.update_interaction_stats(
            self.user_id,
            {
                "total_batches_transferred": total_batches,
                "last_transfer_time": payload["timestamp"],
                "last_batch_size": payload["batch_size"],
            },
        )

        # Route through controller
        self.memory_controller.receive(payload)

        # Clear local STM + shared STM snapshots after transfer
        self.reset()
        self.user_state_store.clear_short_term_memory(self.user_id)

        return payload



class LTMMemoryUnit:
    """
    One compact long-term memory unit for a user.
    """

    def __init__(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        source_batch_ts: str,
        batch_size: int,
        avg_importance: float,
        categories: Optional[List[str]] = None,
        traits: Optional[Dict[str, Any]] = None,
    ):
        self.memory_id = memory_id
        self.user_id = str(user_id)
        self.content = content
        self.source_batch_ts = source_batch_ts
        self.batch_size = batch_size
        self.avg_importance = float(avg_importance)
        self.categories = categories or []
        self.traits = traits or {}
        self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "user_id": self.user_id,
            "content": self.content,
            "source_batch_ts": self.source_batch_ts,
            "batch_size": self.batch_size,
            "avg_importance": self.avg_importance,
            "categories": self.categories,
            "traits": self.traits,
            "created_at": self.created_at,
        }

    def __repr__(self):
        return (
            f"LTMMemoryUnit(user_id={self.user_id}, "
            f"memory_id={self.memory_id}, "
            f"categories={self.categories})"
        )


# ===============================
# Transferable Memory Bank
# ===============================

class TransferableMemoryBank:
    """
    Stores compact long-term memory contents for each user.
    """

    def __init__(self):
        self._store: Dict[str, List[LTMMemoryUnit]] = {}
        self._counter = 0

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------

    def _new_memory_id(self) -> str:
        self._counter += 1
        return f"ltm_{self._counter:06d}"

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _tokenize(self, text: str) -> List[str]:
        text = self._normalize_text(text)
        return re.findall(r"[a-z0-9]+", text)

    def _normalize_categories(self, cats) -> List[str]:
        if cats is None:
            return []

        if isinstance(cats, list):
            vals = cats
        else:
            s = str(cats).strip()
            if not s:
                return []
            vals = re.split(r"[|,;/]+", s)

        out = []
        for x in vals:
            x = self._normalize_text(x)
            if x:
                out.append(x)

        return list(dict.fromkeys(out))

    def _jaccard_tokens(self, a: str, b: str) -> float:
        ta = set(self._tokenize(a))
        tb = set(self._tokenize(b))
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / max(len(ta | tb), 1)

    def _jaccard_lists(self, a: List[str], b: List[str]) -> float:
        sa = set(self._normalize_categories(a))
        sb = set(self._normalize_categories(b))
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / max(len(sa | sb), 1)

    def _is_duplicate(
        self,
        user_id: str,
        content: str,
        categories: Optional[List[str]] = None,
    ) -> bool:
        uid = str(user_id)
        if uid not in self._store:
            return False

        norm_content = self._normalize_text(content)
        new_cats = set(self._normalize_categories(categories or []))

        for mem in self._store[uid]:
            old_norm = self._normalize_text(mem.content)

            if old_norm == norm_content:
                return True

            if old_norm[:120] == norm_content[:120]:
                old_cats = set(self._normalize_categories(mem.categories))
                if new_cats.intersection(old_cats):
                    return True

        return False

    def _iter_memories(self, uid: Optional[str] = None) -> List[LTMMemoryUnit]:
        if uid is not None:
            return list(self._store.get(str(uid), []))

        all_memories = []
        for mems in self._store.values():
            all_memories.extend(mems)
        return all_memories

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def add_memory(
        self,
        user_id: str,
        content: str,
        source_batch_ts: str,
        batch_size: int,
        avg_importance: float,
        categories: Optional[List[str]] = None,
        traits: Optional[Dict[str, Any]] = None,
        allow_duplicate: bool = False,
    ) -> Optional[dict]:
        uid = str(user_id)

        if not allow_duplicate and self._is_duplicate(uid, content, categories):
            return None

        unit = LTMMemoryUnit(
            memory_id=self._new_memory_id(),
            user_id=uid,
            content=content,
            source_batch_ts=source_batch_ts,
            batch_size=batch_size,
            avg_importance=avg_importance,
            categories=self._normalize_categories(categories),
            traits=traits,
        )

        self._store.setdefault(uid, []).append(unit)
        return unit.to_dict()

    def get_user_memories(self, user_id: str) -> List[dict]:
        return [m.to_dict() for m in self._store.get(str(user_id), [])]

    def get_user_memory_units(self, user_id: str) -> List[LTMMemoryUnit]:
        return list(self._store.get(str(user_id), []))

    def all_users(self) -> List[str]:
        return list(self._store.keys())

    def __len__(self):
        return sum(len(v) for v in self._store.values())

    def __repr__(self):
        return f"LongTermMemory(total_memories={len(self)}, users={len(self._store)})"

    # ---------------------------------------------------------
    # Retrieval for test item
    # ---------------------------------------------------------

    def retrieve_for_item(
        self,
        movie,
        min_score: float = 0.20,
        top_k: Optional[int] = None,
        uid: Optional[str] = None,
    ) -> List[LTMMemoryUnit]:
        candidate_memories = self._iter_memories(uid=uid)
        if not candidate_memories:
            return []

        title = (
            movie.get("title", "")
            or movie.get("movie_title", "")
            or movie.get("name", "")
        )

        item_categories = self._normalize_categories(
            movie.get("genre")
            or movie.get("genres")
            or movie.get("categories")
        )

        desc = (
            movie.get("description", "")
            or movie.get("plot", "")
            or movie.get("details", "")
        )

        movie_text = f"title={title}; categories={' | '.join(item_categories)}; desc={desc}"

        scored = []

        for m in candidate_memories:
            text_sim = self._jaccard_tokens(movie_text, m.content)

            imp = float(np.clip(m.avg_importance, 0.0, 1.0))
            importance_weight = 0.5 + 0.5 * imp

            category_sim = self._jaccard_lists(item_categories, m.categories)

            mem_text = self._normalize_text(m.content)
            polarity_boost = 0.0
            if any(w in mem_text for w in ["love", "enjoy", "like", "prefer", "favorite", "favour"]):
                polarity_boost += 0.05
            if any(w in mem_text for w in ["hate", "dislike", "boring", "avoid"]):
                polarity_boost -= 0.05

            score = (
                0.55 * text_sim +
                0.25 * category_sim +
                0.20 * importance_weight +
                polarity_boost
            )

            if score >= min_score:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        memories = [m for _, m in scored]

        if top_k is not None:
            memories = memories[:top_k]

        return memories

    def build_memory_context_for_item(
        self,
        user_id: str,
        item_meta: dict,
        top_k: int = 5,
    ) -> str:
        memories = self.retrieve_for_item(
            movie=item_meta,
            top_k=top_k,
            uid=user_id,
        )

        if not memories:
            return "No long-term memory available for this user."

        lines = []
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"Memory {i}:\n"
                f"  Content    : {mem.content}\n"
                f"  Categories : {mem.categories}\n"
                f"  Traits     : {mem.traits}\n"
                f"  Importance : {mem.avg_importance}"
            )
        return "\n\n".join(lines)


# ===============================
# Memory Management
# ===============================

class MemoryManagement:
    """
    Receives STM transfer payloads and converts them into compact,
    user-specific long-term memory units.

    Flow:
      STM payload -> generate high-level LTM content via LLM -> store in LTM

    Notes:
      - llm is mandatory
      - transferable_memory_bank is mandatory
      - user_state_store is mandatory

    This module can also run memory training and rebuild user LTMs
    from an external memory_store.
    """

    def __init__(
        self,
        user_state_store,
        long_term_memory,
        llm,
        policy=None,
    ):
        if llm is None:
            raise ValueError("llm must be provided to MemoryController.")
        if user_state_store is None:
            raise ValueError("user_state_store must be provided to MemoryController.")
        if long_term_memory is None:
            raise ValueError("long_term_memory must be provided to MemoryController.")

        self._store: Dict[str, List[dict]] = {}
        self.user_state_store = user_state_store
        self.long_term_memory = long_term_memory
        self.llm = llm
        self.policy = policy

    def attach_to_user(self, uid: str, stm: 'DomainSpecificMemoryBank', ltm: 'TransferableMemoryBank'):
        prof = self.policy.user_profiles[uid]
        prof["STM"] = stm
        prof["LTM"] = ltm

    # ============================================================
    # Core receive / store logic
    # ============================================================

    def receive(self, payload: dict):
        uid = str(payload["user_id"])
        self._store.setdefault(uid, []).append(payload)

        self.user_state_store.get_or_create_user(uid)

        ltm_content = self._generate_ltm_content(payload)
        ltm_traits = self._extract_ltm_traits(payload)
        ltm_categories = payload.get("merged_characteristics", {}).get("top_categories", [])

        stored = self.long_term_memory.add_memory(
            user_id=uid,
            content=ltm_content,
            source_batch_ts=payload.get("timestamp", datetime.utcnow().isoformat()),
            batch_size=payload.get("batch_size", 0),
            avg_importance=payload.get("avg_importance", 0.0),
            categories=ltm_categories,
            traits=ltm_traits,
            allow_duplicate=False,
        )

        if stored is not None:
            self.user_state_store.append_long_term_memory(uid, stored)

        print(
            f"[MemoryController] user='{uid}' | "
            f"batch={payload.get('batch_size', 0)} | "
            f"ltm_added={'yes' if stored is not None else 'no'} | "
            f"content: {ltm_content}"
        )

    def _extract_ltm_traits(self, payload: dict) -> dict:
        merged = payload.get("merged_characteristics", {})
        traits = {}

        for key in ["rating_level", "sentiment_hint", "price_tier", "recency", "is_popular"]:
            value = merged.get(key)
            if isinstance(value, dict) and "dominant" in value:
                traits[f"dominant_{key}"] = value["dominant"]
            elif value is not None and not isinstance(value, dict):
                traits[f"dominant_{key}"] = value

        return traits

    # ============================================================
    # Task context
    # ============================================================

    def build_task_context(
        self,
        uid: int,
        task: str,
        item_meta: Optional[Dict[str, Any]] = None,
        persona: Optional[str] = None,
        top_prefs: Optional[Dict[str, Any]] = None,
        min_score: float = 0.20,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        uid = str(uid)
        memories: List[str] = []

        if item_meta is not None:
            movie = {
                "title": item_meta.get("title", ""),
                "genre": (
                    item_meta.get("genre")
                    or item_meta.get("genres")
                    or item_meta.get("categories")
                ),
                "description": (
                    item_meta.get("description", "")
                    or item_meta.get("plot", "")
                    or item_meta.get("details", "")
                ),
            }

            matched = self.long_term_memory.retrieve_for_item(
                movie=movie,
                min_score=min_score,
                top_k=top_k,
                uid=uid,
            )
            memories = [m.content for m in matched]

        return {
            "task": task,
            "persona": persona or "",
            "top_prefs": top_prefs or {},
            "memories": memories,
        }

    # ============================================================
    # LLM helpers
    # ============================================================

    def _generate_ltm_content(self, payload: dict) -> str:
        prompt = self._build_ltm_prompt(payload)

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate one concise high-level long-term user memory. "
                        "It must be generic, reusable, unique, and must not mention exact item names. "
                        "Return one sentence only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise ValueError("LLM returned empty long-term memory content.")
        return text

    def _build_ltm_prompt(self, payload: dict) -> str:
        merged = payload.get("merged_characteristics", {})
        insight = payload.get("insight", "")
        top_categories = merged.get("top_categories", [])

        return (
            f"STM batch insight: {insight}\n"
            f"Top categories: {top_categories}\n"
            f"Merged characteristics: {json.dumps(merged, ensure_ascii=False)}\n\n"
            "Write one unique long-term user preference memory that summarizes "
            "stable preference patterns, price sensitivity, quality expectation, "
            "and behavioral tendency. Avoid exact item names."
        )

    # ============================================================
    # Optional helper: movie dict normalization
    # ============================================================

    def get_movie_dict(self, mdata, mid: int) -> Dict[str, Any]:
        """
        Convert item_id into a rich movie dict with normalized 'genres'.
        Assumes mdata.movies has item_id and some genre-like field.
        """
        m = mdata.movies[mdata.movies["item_id"] == mid].iloc[0].to_dict()

        if "genres" in m and isinstance(m["genres"], str):
            g = [x.strip() for x in m["genres"].split("|") if x.strip()]
            m["genres"] = g
        elif "genre" in m and isinstance(m["genre"], str):
            g = [x.strip() for x in m["genre"].split("|") if x.strip()]
            m["genres"] = g
        elif "genre_set" in m:
            m["genres"] = list(m["genre_set"])
        elif "categories" in m and isinstance(m["categories"], list):
            m["genres"] = m["categories"]
        else:
            m["genres"] = []

        return m

    # ============================================================
    # Memory-store training support
    # ============================================================

    def update_memory_success(
        self,
        memory_store: Dict[Any, Dict[str, Any]],
        task_ctx: Dict[str, Any],
        true_rating: int,
        pred_rating: int,
        uid: Optional[str] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Update per-memory success statistics from task context.
        memory_store format:
            {
                mem_key: {
                    "memory": str,
                    "success": int,
                    "fail": int,
                    "importance": float,
                    ...
                }
            }
        """
        uid = str(uid) if uid is not None else None
        used_memories = task_ctx.get("memories", [])
        is_correct = int(pred_rating == true_rating)

        for mem_text in used_memories:
            key = mem_text.strip()
            if key not in memory_store:
                memory_store[key] = {
                    "memory": mem_text,
                    "success": 0,
                    "fail": 0,
                    "importance": 0.5,
                    "user_id": uid,
                }

            if is_correct:
                memory_store[key]["success"] += 1
                memory_store[key]["importance"] = min(
                    1.0, memory_store[key].get("importance", 0.5) + 0.05
                )
            else:
                memory_store[key]["fail"] += 1
                memory_store[key]["importance"] = max(
                    0.1, memory_store[key].get("importance", 0.5) - 0.03
                )

        return memory_store

    def llm_predict_rating(
        self,
        memory_text: str,
        title: str,
        genres: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Predict rating with the LLM.

        Expected JSON:
        {
          "rating": 1-5,
          "explanation": "..."
        }
        """
        prompt = (
            f"User memory:\n{memory_text}\n\n"
            f"Item title: {title}\n"
            f"Genres: {genres}\n"
            f"Description: {description}\n\n"
            "Predict the user's likely rating from 1 to 5. "
            "Return JSON with keys: rating, explanation."
        )

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": "You are a recommender-system reasoning assistant. Return valid JSON only."
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = (response.choices[0].message.content or "").strip()

        try:
            obj = json.loads(raw)
        except Exception:
            # light fallback
            obj = {"rating": 3, "explanation": raw}

        rating = obj.get("rating", 3)
        try:
            rating = int(rating)
        except Exception:
            rating = 3

        rating = max(1, min(5, rating))
        obj["rating"] = rating
        obj["explanation"] = obj.get("explanation", "")
        return obj

    def _llm_refine_memory(
        self,
        memories: List[str],
        item_meta: Dict[str, Any],
        pred_rating: int,
        true_rating: int,
    ) -> List[str]:
        """
        Refine existing memories when prediction is wrong.
        Returns a small list of revised memory sentences.
        """
        prompt = (
            f"Existing memories:\n{json.dumps(memories, ensure_ascii=False)}\n\n"
            f"Item meta: {json.dumps(item_meta, ensure_ascii=False)}\n"
            f"Predicted rating: {pred_rating}\n"
            f"True rating: {true_rating}\n\n"
            "Revise the user memories so they better reflect stable user preference. "
            "Do not mention exact item names. Return JSON list of concise memory sentences."
        )

        response = self.llm.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON list only."
                },
                {"role": "user", "content": prompt},
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        try:
            out = json.loads(raw)
            if isinstance(out, list):
                return [str(x).strip() for x in out if str(x).strip()]
        except Exception:
            pass

        return []

    def _build_ltm_payload_from_refined_text(
        self,
        user_id: str,
        text: str,
        item_meta: Dict[str, Any],
        avg_importance: float = 0.8,
    ) -> dict:
        """
        Convert refined text into a payload suitable for receive().
        """
        genres = (
            item_meta.get("genres")
            or item_meta.get("genre")
            or item_meta.get("categories")
            or []
        )
        if not isinstance(genres, list):
            genres = [genres]

        merged = {
            "top_categories": genres,
            "sentiment_hint": {"dominant": "refined"},
        }

        return {
            "user_id": str(user_id),
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": 1,
            "avg_importance": avg_importance,
            "insight": text,
            "merged_characteristics": merged,
        }

    # ============================================================
    # Main training loop for one user
    # ============================================================

    def run_memory_training_for_user(
        self,
        policy,
        user_id: str,
        mdata,
        train_df,
        EPOCHS: int = 3,
        memory_store: Optional[dict] = None,
    ):
        """
        Amazon-adapted memory training loop for one user.

        Uses:
          - current LTM retrieval
          - LLM rating prediction
          - memory refinement when prediction is wrong
          - memory success/failure statistics

        Returns:
            memory_store, history
        """
        if memory_store is None:
            memory_store = {}

        user_id = str(user_id)
        history = []

        df = train_df.copy()
        df[mdata.user_col] = df[mdata.user_col].astype(str)
        df[mdata.item_col] = df[mdata.item_col].astype(str)

        user_train_df = df[df[mdata.user_col] == user_id].copy()
        if user_train_df.empty:
            print(f"[memory-train] User {user_id} has no train rows, skipping.")
            return memory_store, history

        prof = policy.user_profiles[user_id]
        ltm = prof.get("LTM", None)

        for ep in range(EPOCHS):
            user_train_df = user_train_df.sample(
                frac=1.0,
                random_state=42 + ep
            ).reset_index(drop=True)

            total_loss = 0.0
            total_n = 0

            for _, row in user_train_df.iterrows():
                mid = str(row[mdata.item_col])

                try:
                    true_rating = int(float(row[mdata.rating_col]))
                except Exception:
                    true_rating = 3

                # user row
                try:
                    user_row = policy.get_user(user_id)
                except Exception:
                    user_row = {"user_id": user_id}

                # item row from policy.movies
                movie_match = policy.movies[
                    policy.movies["item_id"].astype(str) == mid
                ]
                if movie_match.empty:
                    continue

                movie = movie_match.iloc[0].to_dict()

                # raw item row from Data_Structure.items
                raw_item = mdata.get_item(mid)
                raw_item_dict = raw_item.to_dict() if raw_item is not None else {}

                # normalize genres
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

                # persona
                persona = ""
                try:
                    persona = policy.persona_description(user_row)
                except Exception:
                    persona = ""

                # top preferences
                top_prefs = {}
                if isinstance(prof.get("preferences"), dict):
                    top_prefs = dict(
                        sorted(
                            prof["preferences"].items(),
                            key=lambda x: -x[1]
                        )[:5]
                    )

                task_ctx = self.build_task_context(
                    uid=user_id,
                    task="rating_prediction",
                    item_meta=item_meta,
                    persona=persona,
                    top_prefs=top_prefs,
                    min_score=0.20,
                    top_k=5,
                )

                memory_text = task_ctx["persona"] + "\n" + "\n".join(task_ctx["memories"])

                pred_obj = self.llm_predict_rating(
                    memory_text=memory_text,
                    title=item_meta["title"],
                    genres=",".join(item_meta["genres"]),
                    description=item_meta["description"],
                )

                try:
                    pred_rating = int(pred_obj["rating"])
                except Exception:
                    pred_rating = 3

                trace_text = pred_obj.get("explanation", "no-explanation")

                # refine memories when prediction is wrong
                if task_ctx["memories"] and pred_rating != true_rating:
                    refined_list = self._llm_refine_memory(
                        task_ctx["memories"],
                        item_meta,
                        pred_rating,
                        true_rating,
                    )
                    for txt in refined_list:
                        payload = self._build_ltm_payload_from_refined_text(
                            user_id=user_id,
                            text=txt,
                            item_meta=item_meta,
                            avg_importance=0.8,
                        )
                        self.receive(payload)

                loss = abs(pred_rating - true_rating)
                total_loss += loss
                total_n += 1

                print(f"[user {user_id}] loss={loss} pred={pred_rating} true={true_rating}")

                memory_store = self.update_memory_success(
                    memory_store=memory_store,
                    task_ctx=task_ctx,
                    true_rating=true_rating,
                    pred_rating=pred_rating,
                    uid=user_id,
                )

                # optional episode record
                episode = {
                    "user_profile": persona,
                    "movie": {
                        "item_id": mid,
                        "title": item_meta["title"],
                        "genres": item_meta["genres"],
                    },
                    "trace": trace_text,
                    "predicted_rating": pred_rating,
                    "true_rating": true_rating,
                }

                # optional STM hook
                stm = prof.get("STM", None)
                if stm is not None and hasattr(stm, "add_task_episode"):
                    try:
                        stm.add_task_episode(episode, importance=0.5)
                    except Exception:
                        pass

            avg_loss = total_loss / max(1, total_n)
            print(
                f"[memory-train] User {user_id} Epoch {ep+1}/{EPOCHS} "
                f"| train L1={avg_loss:.3f}"
            )

            history.append(
                {
                    "epoch": ep + 1,
                    "train_loss": float(avg_loss),
                    "ltm_size": len(ltm.get_user_memory_units(user_id)) if ltm is not None else None,
                }
            )

        return memory_store, history
    def dedupe_memory_store_for_user(
        self,
        user_store: Dict[Any, Dict[str, Any]],
        sim_threshold: float = 0.80,
    ):
        """
        Deduplicate one user's memory_store by semantic similarity and merge stats.

        Expected user_store format:
            {
                mem_key: {
                    "memory": str,
                    "success": int,
                    "fail": int,
                    "importance": float,
                    "categories": list,   # optional
                    "traits": dict,       # optional
                    ...
                }
            }

        Returns:
            deduped_store: dict with merged records
        """
        if not user_store:
            return {}

        try:
            from sentence_transformers import SentenceTransformer
            from numpy.linalg import norm
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for dedupe_memory_store_for_user(). "
                "Install it first."
            ) from e
    
        emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    
        def get_embedding(text: str) -> np.ndarray:
            emb = emb_model.encode(
                str(text),
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return emb.astype("float32")
    
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(a @ b / (norm(a) * norm(b) + 1e-8))
    
        mem_ids = list(user_store.keys())
        texts = [str(user_store[mid].get("memory", "")) for mid in mem_ids]
    
        if not texts:
            return {}
    
        embeddings = np.stack([get_embedding(t) for t in texts], axis=0)
    
        # -----------------------------------------
        # Greedy clustering
        # -----------------------------------------
        clusters = []  # [{"centroid": np.ndarray, "indices": [int]}]
    
        for idx, emb in enumerate(embeddings):
            if not clusters:
                clusters.append({"centroid": emb.copy(), "indices": [idx]})
                continue
    
            best_sim = -1.0
            best_cluster_idx = None
    
            for c_idx, c in enumerate(clusters):
                sim = cosine_sim(emb, c["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_cluster_idx = c_idx
    
            if best_sim >= sim_threshold:
                clusters[best_cluster_idx]["indices"].append(idx)
                member_embs = embeddings[clusters[best_cluster_idx]["indices"]]
                clusters[best_cluster_idx]["centroid"] = member_embs.mean(axis=0)
            else:
                clusters.append({"centroid": emb.copy(), "indices": [idx]})
    
        # -----------------------------------------
        # Merge each cluster into one record
        # -----------------------------------------
        deduped_store = {}
    
        for c_idx, cluster in enumerate(clusters, 1):
            idxs = cluster["indices"]
            cluster_ids = [mem_ids[i] for i in idxs]
            cluster_recs = [user_store[mem_ids[i]] for i in idxs]
    
            # representative = highest (success, importance, -fail)
            best_idx = None
            best_score = None
            for i in idxs:
                rec = user_store[mem_ids[i]]
                score = (
                    float(rec.get("success", 0)),
                    float(rec.get("importance", 0.0)),
                    -float(rec.get("fail", 0)),
                )
                if best_idx is None or score > best_score:
                    best_idx = i
                    best_score = score
    
            rep_mid = mem_ids[best_idx]
            rep_rec = user_store[rep_mid].copy()
    
            merged_success = sum(float(r.get("success", 0)) for r in cluster_recs)
            merged_fail = sum(float(r.get("fail", 0)) for r in cluster_recs)
            merged_importance = max(float(r.get("importance", 0.5)) for r in cluster_recs)
    
            merged_categories = []
            for r in cluster_recs:
                cats = r.get("categories", [])
                if cats is None:
                    continue
                if not isinstance(cats, list):
                    cats = [cats]
                merged_categories.extend([str(x).strip() for x in cats if str(x).strip()])
            merged_categories = list(dict.fromkeys(merged_categories))
    
            merged_traits = {}
            for r in cluster_recs:
                traits = r.get("traits", {})
                if isinstance(traits, dict):
                    merged_traits.update(traits)
    
            merged_rec = rep_rec.copy()
            merged_rec["memory"] = rep_rec.get("memory", "")
            merged_rec["success"] = int(merged_success)
            merged_rec["fail"] = int(merged_fail)
            merged_rec["importance"] = float(merged_importance)
            merged_rec["categories"] = merged_categories
            merged_rec["traits"] = merged_traits
            merged_rec["cluster_members"] = cluster_ids
            merged_rec["cluster_size"] = len(cluster_ids)
            merged_rec["representative_id"] = rep_mid
    
            new_key = f"dedup_{c_idx:04d}_{rep_mid}"
            deduped_store[new_key] = merged_rec
    
        print(
            f"[dedupe_memory_store_for_user] "
            f"before={len(user_store)} | after={len(deduped_store)} | "
            f"merged={len(user_store) - len(deduped_store)}"
        )

        return deduped_store

    # ============================================================
    # Rebuild LTM from memory store
    # ============================================================
    def rebuild_ltm_from_memory_store(
        self,
        ltm,
        user_id,
        memory_store,
        success_thr: Optional[int] = None,
        top_k: int = 40,
    ):
        """
        Rebuild one user's LTM from memory_store after inspecting success stats.

        memory_store format:
            {
                key: {
                    "memory": str,
                    "success": int,
                    "importance": float,
                    "categories": list,
                    "traits": dict,
                    ...
                }
            }

        If success_thr is None, a threshold is chosen automatically
        from the success-value distribution.
        """
        uid = str(user_id)

        if not memory_store:
            print("[rebuild_ltm_from_memory_store] memory_store is empty")
            return
    
        # -----------------------------------------
        # Step 1: inspect success statistics
        # -----------------------------------------
        success_values = [
            float(rec.get("success", 0))
            for rec in memory_store.values()
        ]
    
        if not success_values:
            print("[rebuild_ltm_from_memory_store] no success values found")
            return
    
        min_success = min(success_values)
        max_success = max(success_values)
        mean_success = sum(success_values) / len(success_values)
    
        print(
            f"[rebuild_ltm_from_memory_store] user={uid} | "
            f"total_memories={len(memory_store)} | "
            f"min_success={min_success:.2f} | "
            f"max_success={max_success:.2f} | "
            f"mean_success={mean_success:.2f}"
        )
    
        # -----------------------------------------
        # Step 2: choose threshold if not provided
        # -----------------------------------------
        if success_thr is None:
            # simple automatic rule:
            # keep memories at or above mean success
            success_thr = mean_success
            print(
                f"[rebuild_ltm_from_memory_store] auto success_thr={success_thr:.2f}"
            )
    
        # -----------------------------------------
        # Step 3: filter successful memories
        # -----------------------------------------
        filtered = {
            mid: rec
            for mid, rec in memory_store.items()
            if float(rec.get("success", 0)) >= float(success_thr)
        }
    
        if not filtered:
            print(
                "[rebuild_ltm_from_memory_store] "
                "no memories passed the threshold; LTM left unchanged"
            )
            return

        # -----------------------------------------
        # Step 4: sort by success and keep top_k
        # -----------------------------------------
        items = sorted(
            filtered.items(),
            key=lambda x: float(x[1].get("success", 0)),
            reverse=True,
        )[:top_k]

        max_success_filtered = max(float(rec.get("success", 0)) for _, rec in items)

        print(
            f"[rebuild_ltm_from_memory_store] kept={len(items)} "
            f"after threshold={float(success_thr):.2f}"
        )
    
        # -----------------------------------------
        # Step 5: clear only this user's LTM
        # -----------------------------------------
        if hasattr(ltm, "_store"):
            ltm._store[uid] = []
    
        # -----------------------------------------
        # Step 6: rebuild LTM with selected memories
        # -----------------------------------------
        for _, rec in items:
            text = rec.get("memory", "")
            base_imp = float(rec.get("importance", 0.5))
            succ = float(rec.get("success", 0.0))
            cats = rec.get("categories", [])
            traits = rec.get("traits", {})
    
            success_factor = (succ / max_success_filtered) if max_success_filtered > 0 else 1.0
            importance = float(0.3 + 0.7 * base_imp * success_factor)
    
            ltm.add_memory(
                user_id=uid,
                content=text,
                source_batch_ts=datetime.utcnow().isoformat(),
                batch_size=1,
                avg_importance=importance,
                categories=cats,
                traits=traits,
                allow_duplicate=False,
            )
    
        print(
            f"[rebuild_ltm_from_memory_store] user={uid} rebuilt with "
            f"{len(ltm.get_user_memory_units(uid))} memories"
        )

    def rebuild_all_users_ltm_from_memory_store(
        self,
        policy,
        memory_store_by_user,
        success_thr: int = 50,
        top_k: int = 40,
    ):
        """
        Rebuild each user's LTM in policy.user_profiles[uid]['LTM']
        from memory_store_by_user.
        """
        for uid, user_store in memory_store_by_user.items():
            try:
                int_uid = int(uid)
            except Exception:
                int_uid = uid

            prof = policy.user_profiles.get(int_uid)
            if prof is None:
                print(f"[rebuild_all_users_ltm] uid={int_uid} not in policy.user_profiles, skipping")
                continue

            ltm = prof.get("LTM")
            if ltm is None:
                print(f"[rebuild_all_users_ltm] uid={int_uid} has no LTM, skipping")
                continue

            print(f"[rebuild_all_users_ltm] rebuilding LTM for uid={int_uid}")
            self.rebuild_ltm_from_memory_store(
                ltm=ltm,
                user_id=int_uid,
                memory_store=user_store,
                success_thr=success_thr,
                top_k=top_k,
            )

    # ============================================================
    # History / misc
    # ============================================================

    def get_user_history(self, user_id: str) -> List[dict]:
        return self._store.get(str(user_id), [])

    def all_users(self) -> List[str]:
        return list(self._store.keys())

    def __repr__(self):
        return f"MemoryController(total_users={len(self._store)})"
    