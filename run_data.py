import json
import csv
import os, sys
from collections import defaultdict
import pandas as pd

csv.field_size_limit(sys.maxsize)

# -----------------------------
# 1) Helpers
# -----------------------------
FIELDS = [
    "rating", "title", "text", "images", "asin", "parent_asin",
    "user_id", "timestamp", "verified_purchase", "helpful_vote"
]

def extract_unique_users(jsonl_path):
    users = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            uid = rec.get("user_id")
            if uid:
                users.add(uid)
    return users

def update_intersection(jsonl_path, common_users=None):
    current_users = extract_unique_users(jsonl_path)
    if common_users is None:
        return current_users
    return common_users & current_users


def write_common_rows(jsonl_path, common_users, out_csv_path):
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)

    with open(out_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDS)
        writer.writeheader()

        with open(jsonl_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                uid = rec.get("user_id")
                if uid in common_users:
                    row = {k: rec.get(k) for k in FIELDS}
                    if row["images"] is not None:
                        row["images"] = json.dumps(row["images"], ensure_ascii=False)
                    writer.writerow(row)

# File paths
books_path = "Books.jsonl"
movies_path = "Movies_and_TV.jsonl"
electronics_path = "Electronics.jsonl"

# Extract users
common_users_all = None
for path in [books_path, movies_path, electronics_path]:
    common_users_all = update_intersection(path, common_users_all)

# -----------------------------
# 4) Export common-user rows into 3 CSVs
# -----------------------------
print("Common users across all 3 datasets:", len(common_users_all))

write_common_rows(books_path, common_users_all, "Amazon_reviews/Books_Reviews2.csv")
write_common_rows(movies_path, common_users_all, "Amazon_reviews/Movies_and_TV_Reviews2.csv")
write_common_rows(electronics_path, common_users_all, "Amazon_reviews/Electronics_Reviews2.csv")

print("Saved:")


# -----------------------------
# Metadata columns to keep
# -----------------------------
META_FIELDS = [
    "main_category", "title", "average_rating", "rating_number",
    "features", "description", "price", "images", "videos", "store",
    "categories", "details", "parent_asin", "bought_together"
]

LIST_OR_DICT_FIELDS = [
    "features", "description", "images", "videos",
    "categories", "details", "bought_together"
]

def clean_lines(file_obj):
    for line in file_obj:
        yield line.replace("\x00", "")

# # -----------------------------
# # Function to generate meta CSV
# # -----------------------------
def generate_meta_csv(review_csv_path, meta_jsonl_path, output_csv_path):
    parent_asins = set()

    with open(review_csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(clean_lines(f))
        for row in reader:
            pa = row.get("parent_asin")
            if pa:
                parent_asins.add(pa)

    print(f"\n{review_csv_path}")
    print("Unique parent_asin in reviews:", len(parent_asins))

    kept = 0
    seen = set()

    with open(output_csv_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=META_FIELDS)
        writer.writeheader()

        with open(meta_jsonl_path, "r", encoding="utf-8", errors="replace") as in_f:
            for line in in_f:
                if not line.strip():
                    continue
                line = line.replace("\x00", "")
                rec = json.loads(line)
                pa = rec.get("parent_asin")

                if pa in parent_asins and pa not in seen:
                    row = {k: rec.get(k) for k in META_FIELDS}

                    for k in LIST_OR_DICT_FIELDS:
                        if row.get(k) is not None:
                            row[k] = json.dumps(row[k], ensure_ascii=False)

                    writer.writerow(row)
                    kept += 1
                    seen.add(pa)

    print("Matched meta rows written:", kept)
    print("Saved:", output_csv_path)

# ==================================================
# Run for all three domains
# ==================================================
generate_meta_csv(
    review_csv_path="Amazon_reviews/Books_Reviews2.csv",
    meta_jsonl_path="meta_Books.jsonl",
    output_csv_path="Amazon_reviews/meta_Books2.csv"
)

generate_meta_csv(
    review_csv_path="Amazon_reviews/Movies_and_TV_Reviews2.csv",
    meta_jsonl_path="meta_Movies_and_TV.jsonl",
    output_csv_path="Amazon_reviews/meta_Movies_and_TV2.csv"
)

generate_meta_csv(
    review_csv_path="Amazon_reviews/Electronics_Reviews2.csv",
    meta_jsonl_path="meta_Electronics.jsonl",
    output_csv_path="Amazon_reviews/meta_Electronics2.csv"
)


def clean_lines(file_obj):
    for line in file_obj:
        yield line.replace("\x00", "")

def clean_text(value):
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value

# ======================================================
# USER + BEHAVIOR FILE FROM REVIEW CSV
# ======================================================
def generate_user_and_behavior(review_csv, user_out, behavior_out):

    user_helpful = defaultdict(int)

    with open(review_csv, "r", encoding="utf-8", errors="replace", newline="") as f, \
         open(behavior_out, "w", newline="", encoding="utf-8") as beh_f:

        reader = csv.DictReader(clean_lines(f))

        beh_writer = csv.DictWriter(beh_f, fieldnames=[
            "user_id", "parent_asin", "title",
            "text", "rating", "timestamp"
        ])
        beh_writer.writeheader()

        for row in reader:
            uid = clean_text(row.get("user_id"))
            pa = clean_text(row.get("parent_asin"))

            if not uid or not pa:
                continue

            hv = row.get("helpful_vote")
            if hv not in (None, ""):
                try:
                    user_helpful[uid] += int(hv)
                except ValueError:
                    pass

            beh_writer.writerow({
                "user_id": uid,
                "parent_asin": pa,
                "title": clean_text(row.get("title")),
                "text": clean_text(row.get("text")),
                "rating": clean_text(row.get("rating")),
                "timestamp": clean_text(row.get("timestamp"))
            })

    with open(user_out, "w", newline="", encoding="utf-8") as user_f:
        user_writer = csv.DictWriter(user_f, fieldnames=["user_id", "helpful_vote"])
        user_writer.writeheader()

        for uid, total_vote in user_helpful.items():
            user_writer.writerow({
                "user_id": uid,
                "helpful_vote": total_vote
            })

    print(f"Generated: {user_out}, {behavior_out}")


# ======================================================
# ITEM FILE FROM META CSV
# ======================================================
def generate_item_file(meta_csv, item_out):

    with open(meta_csv, "r", encoding="utf-8", errors="replace", newline="") as f, \
         open(item_out, "w", newline="", encoding="utf-8") as item_f:

        reader = csv.DictReader(clean_lines(f))

        item_writer = csv.DictWriter(item_f, fieldnames=[
            "parent_asin", "title", "categories",
            "details", "average_rating",
            "rating_number", "price", "bought_together"
        ])
        item_writer.writeheader()

        for row in reader:
            item_writer.writerow({
                "parent_asin": clean_text(row.get("parent_asin")),
                "title": clean_text(row.get("title")),
                "categories": clean_text(row.get("categories")),
                "details": clean_text(row.get("details")),
                "average_rating": clean_text(row.get("average_rating")),
                "rating_number": clean_text(row.get("rating_number")),
                "price": clean_text(row.get("price")),
                "bought_together": clean_text(row.get("bought_together"))
            })

    print(f"Generated: {item_out}")

# ======================================================
# RUN FOR BOOKS
# ======================================================
generate_user_and_behavior(
    review_csv="Amazon_reviews/Books_Reviews2.csv",
    user_out="Amazon_reviews/User_Books2.csv",
    behavior_out="Amazon_reviews/Behavior_Books2.csv"
)

generate_item_file(
    meta_csv="Amazon_reviews/meta_Books2.csv",
    item_out="Amazon_reviews/Item_Books2.csv"
)


# ======================================================
# RUN FOR MOVIES
# ======================================================
generate_user_and_behavior(
    review_csv="Amazon_reviews/Movies_and_TV_Reviews2.csv",
    user_out="Amazon_reviews/User_Movies2.csv",
    behavior_out="Amazon_reviews/Behavior_Movies2.csv"
)

generate_item_file(
    meta_csv="Amazon_reviews/meta_Movies_and_TV2.csv",
    item_out="Amazon_reviews/Item_Movies2.csv"
)


# ======================================================
# RUN FOR Electronics
# ======================================================
generate_user_and_behavior(
    review_csv="Amazon_reviews/Electronics_Reviews2.csv",
    user_out="Amazon_reviews/User_Electronics2.csv",
    behavior_out="Amazon_reviews/Behavior_Electronics2.csv"
)

generate_item_file(
    meta_csv="Amazon_reviews/meta_Electronics2.csv",
    item_out="Amazon_reviews/Item_Electronics2.csv"
)

def analyze_behavior(file_path, domain_name):
    print("\n" + "="*60)
    print(f"📊 Domain: {domain_name}")
    print("="*60)

    df = pd.read_csv(file_path)

    total_interactions = len(df)
    total_users = df["user_id"].nunique()
    total_items = df["parent_asin"].nunique()

    print("Total interactions:", total_interactions)
    print("Total users:", total_users)
    print("Total items:", total_items)

    # -------------------------------------------------
    # User interaction statistics
    # -------------------------------------------------
    user_counts = df.groupby("user_id").size()

    print("\n--- User Interaction Stats ---")
    print("Min interactions per user:", user_counts.min())
    print("Avg interactions per user:", round(user_counts.mean(), 2))
    print("Max interactions per user:", user_counts.max())

    # 🔹 Users with > 10 interactions
    users_gt_10 = (user_counts > 10).sum()
    percentage_gt_10 = (users_gt_10 / total_users) * 100

    print("\nUsers with >10 interactions:", users_gt_10)
    print("Percentage:", round(percentage_gt_10, 2), "%")

    # -------------------------------------------------
    # Item interaction statistics
    # -------------------------------------------------
    item_counts = df.groupby("parent_asin").size()

    print("\n--- Item Interaction Stats ---")
    print("Min interactions per item:", item_counts.min())
    print("Avg interactions per item:", round(item_counts.mean(), 2))
    print("Max interactions per item:", item_counts.max())

    # -------------------------------------------------
    # Rating distribution
    # -------------------------------------------------
    print("\n--- Rating Distribution ---")
    rating_dist = df["rating"].value_counts().sort_index()
    for rating, count in rating_dist.items():
        percentage = (count / total_interactions) * 100
        print(f"Rating {rating}: {count} ({percentage:.2f}%)")


# =====================================================
# Run for all three datasets
# =====================================================

analyze_behavior("Amazon_reviews/Behavior_Books2.csv", "Books")
analyze_behavior("Amazon_reviews/Behavior_Movies2.csv", "Movies & TV")
analyze_behavior("Amazon_reviews/Behavior_Electronics2.csv", "Electronics")