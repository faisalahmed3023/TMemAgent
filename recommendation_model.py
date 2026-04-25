import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Set, Tuple, List
import scipy.sparse as sp

class MF:
    """
    BPR-MF pipeline:
      - read Behavior_*.csv (user_id, parent_asin, rating, timestamp, etc.)
      - sort by timestamp, dedup (user_id, parent_asin) keeping last
      - label-encode user/item
      - sequential split per user (80/20)
      - train BPR-MF
      - evaluate with sampled negatives (Recall/NDCG@K)
    """

    # -------------------------
    # Model
    # -------------------------
    class BPRMF(nn.Module):
        def __init__(self, num_users, num_items, latent_dim=64):
            super().__init__()
            self.user_emb = nn.Embedding(num_users, latent_dim)
            self.item_emb = nn.Embedding(num_items, latent_dim)

        def score(self, u, i):
            return (self.user_emb(u) * self.item_emb(i)).sum(dim=1)

        def forward(self, u, pos_i, neg_i):
            pos = self.score(u, pos_i)
            neg = self.score(u, neg_i)
            return pos, neg

    # -------------------------
    # Dataset
    # -------------------------
    class BPRDataset(Dataset):
        def __init__(self, users, pos_items, user_pos, num_items, seed=42):
            self.users = users.astype(int)
            self.pos_items = pos_items.astype(int)
            self.user_pos = user_pos
            self.num_items = int(num_items)
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):
            u = int(self.users[idx])
            pos = int(self.pos_items[idx])

            # sample negative not in user's positives
            while True:
                neg = int(self.rng.integers(0, self.num_items))
                if neg not in self.user_pos[u]:
                    break

            return (
                torch.tensor(u, dtype=torch.long),
                torch.tensor(pos, dtype=torch.long),
                torch.tensor(neg, dtype=torch.long),
            )

    # -------------------------
    # Init
    # -------------------------
    def __init__(
        self,
        behavior_csv_path: str,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        time_col: str = "timestamp",
        latent_dim: int = 128,
        lr: float = 5e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 2048,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.behavior_csv_path = behavior_csv_path
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.latent_dim = latent_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # will be set later
        self.df = None
        self.train_df = None
        self.test_df = None
        self.train_hist = None
        self.test_hist = None
        self.eval_users = None

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.num_users = None
        self.num_items = None

        self.user_pos = None
        self.model = None
        self.optimizer = None

        # reproducibility
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    # -------------------------
    # Data processing
    # -------------------------
    def load_and_prepare(self, dedup_last=True):
        df = pd.read_csv(self.behavior_csv_path)
        df = df.sort_values(self.time_col)

        if dedup_last:
            df = df.drop_duplicates(subset=[self.user_col, self.item_col], keep="last")

        df["user_idx"] = self.user_encoder.fit_transform(df[self.user_col].astype(str))
        df["item_idx"] = self.item_encoder.fit_transform(df[self.item_col].astype(str))

        self.df = df
        self.num_users = df["user_idx"].nunique()
        self.num_items = df["item_idx"].nunique()

        print(f"After dedup -> interactions: {len(df)} users: {self.num_users} items: {self.num_items}")
        return df

    def split_sequential(self, train_ratio=0.8, min_train=1, min_test=1):
        assert self.df is not None, "Call load_and_prepare() first."

        train_parts, test_parts = [], []
        for u, g in self.df.groupby("user_idx"):
            g = g.sort_values(self.time_col)
            cut = int(len(g) * train_ratio)
            if cut < min_train or (len(g) - cut) < min_test:
                continue
            train_parts.append(g.iloc[:cut])
            test_parts.append(g.iloc[cut:])

        self.train_df = pd.concat(train_parts).reset_index(drop=True)
        self.test_df = pd.concat(test_parts).reset_index(drop=True)

        print("Users train:", self.train_df["user_idx"].nunique(),
              "Users test:", self.test_df["user_idx"].nunique())
        print("Train size:", len(self.train_df), "Test size:", len(self.test_df))

        # histories for eval
        self.train_hist = self.train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.test_hist = self.test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.eval_users = sorted(set(self.train_hist) & set(self.test_hist))

        # user_pos for negative sampling (IMPORTANT: build AFTER split)
        self.user_pos = defaultdict(set)
        for u, i in zip(self.train_df["user_idx"].values, self.train_df["item_idx"].values):
            self.user_pos[int(u)].add(int(i))

        return self.train_df, self.test_df

    # -------------------------
    # Training
    # -------------------------
    @staticmethod
    def bpr_loss(pos, neg):
        return -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()

    def fit(
        self,
        epochs=100,
        eval_every=10,
        early_stop=True,
        patience=5,
        eval_k=10,
        eval_n_neg=999,
    ):
        assert self.train_df is not None, "Call split_sequential() first."

        self.model = MF.BPRMF(self.num_users, self.num_items, latent_dim=self.latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        ds = MF.BPRDataset(
            users=self.train_df["user_idx"].values,
            pos_items=self.train_df["item_idx"].values,
            user_pos=self.user_pos,
            num_items=self.num_items,
            seed=self.seed,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

        best_ndcg = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            n = 0

            for u, pos_i, neg_i in loader:
                u = u.to(self.device)
                pos_i = pos_i.to(self.device)
                neg_i = neg_i.to(self.device)

                pos, neg = self.model(u, pos_i, neg_i)
                loss = MF.bpr_loss(pos, neg)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                bs = u.size(0)
                total += loss.item() * bs
                n += bs

            print(f"Epoch {epoch}, Avg BPR loss: {total/n:.4f}")

            if eval_every and (epoch % eval_every == 0):
                r, nd = self.evaluate_sampled(k=eval_k, n_neg=eval_n_neg)
                print(f"  Eval @Epoch {epoch}: Recall@{eval_k}={r:.4f}, NDCG@{eval_k}={nd:.4f}")

                if early_stop:
                    if nd > best_ndcg:
                        best_ndcg = nd
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                        if bad >= patience:
                            print(f"Early stopping. Best NDCG@{eval_k}: {best_ndcg:.4f}")
                            break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    # -------------------------
    # Evaluation (sampled)
    # -------------------------
    @torch.no_grad()
    def rank_with_sampled_negatives(self, u, pos_items, n_neg=999, k=10):
        self.model.eval()

        seen = self.train_hist.get(u, set())
        pos_items = list(pos_items)

        # sample negatives not in seen/positives
        forbidden = seen | set(pos_items)
        rng = np.random.default_rng(self.seed + u)  # stable per-user
        negs = []
        while len(negs) < n_neg:
            cand = int(rng.integers(0, self.num_items))
            if cand not in forbidden:
                negs.append(cand)

        candidates = np.array(pos_items + negs, dtype=np.int64)
        u_tensor = torch.tensor([u], dtype=torch.long, device=self.device).repeat(len(candidates))
        i_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)

        scores = self.model.score(u_tensor, i_tensor).detach().cpu().numpy()

        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        topk_items = candidates[topk_idx]
        return topk_items, set(pos_items)

    @staticmethod
    def recall_at_k(recs, positives):
        return len(set(recs) & positives) / len(positives) if positives else 0.0

    @staticmethod
    def ndcg_at_k(recs, positives):
        dcg = 0.0
        for r, it in enumerate(recs, start=1):
            if it in positives:
                dcg += 1.0 / np.log2(r + 1)
        ideal = min(len(positives), len(recs))
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_sampled(self, k=10, n_neg=999):
        assert self.model is not None, "Call fit() first."

        rs, ns = [], []
        for u in self.eval_users:
            pos = self.test_hist.get(u, set())
            if not pos:
                continue
            recs, pos_set = self.rank_with_sampled_negatives(u, pos_items=pos, n_neg=n_neg, k=k)
            rs.append(MF.recall_at_k(recs, pos_set))
            ns.append(MF.ndcg_at_k(recs, pos_set))

        return float(np.mean(rs)), float(np.mean(ns))

    # -------------------------
    # Recommend for a raw user_id
    # -------------------------
    @torch.no_grad()
    def recommend(self, user_id: str, k=10, n_neg=5000):
        """
        Returns top-k recommended parent_asin IDs from a sampled candidate pool.
        For full ranking over all items, we can add a method, but it's expensive for 100k items.
        """
        assert self.model is not None, "Call fit() first."

        u = int(self.user_encoder.transform([str(user_id)])[0])
        seen = self.train_hist.get(u, set())

        rng = np.random.default_rng(self.seed + 999)
        candidates = []
        while len(candidates) < n_neg:
            cand = int(rng.integers(0, self.num_items))
            if cand not in seen:
                candidates.append(cand)

        candidates = np.array(candidates, dtype=np.int64)
        u_tensor = torch.tensor([u], dtype=torch.long, device=self.device).repeat(len(candidates))
        i_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)
        scores = self.model.score(u_tensor, i_tensor).detach().cpu().numpy()

        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        top_items = candidates[topk_idx]

        # map back to original parent_asin
        return list(self.item_encoder.inverse_transform(top_items))
    

class LightGCN:
    """
    LightGCN for implicit ranking using BPR loss.
    Works with Behavior_*.csv having columns:
      user_id, parent_asin, timestamp (and others)
    """

    class _Model(nn.Module):
        def __init__(self, num_users: int, num_items: int, latent_dim: int, n_layers: int):
            super().__init__()
            self.num_users = num_users
            self.num_items = num_items
            self.latent_dim = latent_dim
            self.n_layers = n_layers

            self.user_emb = nn.Embedding(num_users, latent_dim)
            self.item_emb = nn.Embedding(num_items, latent_dim)

            # init embeddings
            nn.init.normal_(self.user_emb.weight, std=0.01)
            nn.init.normal_(self.item_emb.weight, std=0.01)

        def get_ego_embeddings(self):
            return torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)

        def forward(self, norm_adj: torch.Tensor):
            """
            Returns final user/item embeddings after LightGCN propagation.
            norm_adj is sparse COO tensor of shape (N, N), N = users+items
            """
            ego = self.get_ego_embeddings()
            all_layers = [ego]

            x = ego
            for _ in range(self.n_layers):
                x = torch.sparse.mm(norm_adj, x)
                all_layers.append(x)

            out = torch.stack(all_layers, dim=0).mean(dim=0)
            users, items = torch.split(out, [self.num_users, self.num_items], dim=0)
            return users, items

        @staticmethod
        def bpr_loss(u_emb, pos_emb, neg_emb):
            pos = (u_emb * pos_emb).sum(dim=1)
            neg = (u_emb * neg_emb).sum(dim=1)
            return -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()

    def __init__(
        self,
        behavior_csv_path: str,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        time_col: str = "timestamp",
        latent_dim: int = 64,
        n_layers: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 2048,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.behavior_csv_path = behavior_csv_path
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        self.df = None
        self.train_df = None
        self.test_df = None
        self.num_users = None
        self.num_items = None

        self.train_hist = None
        self.test_hist = None
        self.user_pos = None
        self.eval_users = None

        self.norm_adj = None
        self.model = None
        self.opt = None

    # ----------------------------
    # Data preparation
    # ----------------------------
    def load_and_prepare(self, dedup_last: bool = True):
        df = pd.read_csv(self.behavior_csv_path)
        df = df.sort_values(self.time_col)

        if dedup_last:
            df = df.drop_duplicates(subset=[self.user_col, self.item_col], keep="last")

        df["user_idx"] = self.user_encoder.fit_transform(df[self.user_col].astype(str))
        df["item_idx"] = self.item_encoder.fit_transform(df[self.item_col].astype(str))

        self.df = df
        self.num_users = int(df["user_idx"].nunique())
        self.num_items = int(df["item_idx"].nunique())

        print(f"After dedup -> interactions: {len(df)} users: {self.num_users} items: {self.num_items}")
        return df

    def split_sequential(self, train_ratio: float = 0.8, min_train: int = 1, min_test: int = 1):
        assert self.df is not None, "Call load_and_prepare() first."

        train_parts, test_parts = [], []
        for u, g in self.df.groupby("user_idx"):
            g = g.sort_values(self.time_col)
            cut = int(len(g) * train_ratio)
            if cut < min_train or (len(g) - cut) < min_test:
                continue
            train_parts.append(g.iloc[:cut])
            test_parts.append(g.iloc[cut:])

        self.train_df = pd.concat(train_parts).reset_index(drop=True)
        self.test_df = pd.concat(test_parts).reset_index(drop=True)

        print("Users train:", self.train_df["user_idx"].nunique(),
              "Users test:", self.test_df["user_idx"].nunique())
        print("Train size:", len(self.train_df), "Test size:", len(self.test_df))

        self.train_hist = self.train_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.test_hist = self.test_df.groupby("user_idx")["item_idx"].apply(set).to_dict()
        self.eval_users = sorted(set(self.train_hist) & set(self.test_hist))

        self.user_pos = defaultdict(set)
        for u, i in zip(self.train_df["user_idx"].values, self.train_df["item_idx"].values):
            self.user_pos[int(u)].add(int(i))

        return self.train_df, self.test_df

    # ----------------------------
    # Graph construction
    # ----------------------------
    def _build_norm_adj(self):
        """
        Build normalized adjacency for user-item bipartite graph:
          A = [[0, R],
               [R^T, 0]]
        norm = D^{-1/2} A D^{-1/2}
        """
        assert self.train_df is not None, "Call split_sequential() first."

        u = self.train_df["user_idx"].values.astype(np.int64)
        i = self.train_df["item_idx"].values.astype(np.int64)

        # Build R (users x items) with 1s
        R = sp.coo_matrix(
            (np.ones_like(u, dtype=np.float32), (u, i)),
            shape=(self.num_users, self.num_items),
            dtype=np.float32
        )

        # Bipartite adjacency
        upper = sp.hstack([sp.csr_matrix((self.num_users, self.num_users), dtype=np.float32), R.tocsr()])
        lower = sp.hstack([R.T.tocsr(), sp.csr_matrix((self.num_items, self.num_items), dtype=np.float32)])
        A = sp.vstack([upper, lower]).tocsr()

        # Normalize: D^{-1/2} A D^{-1/2}
        deg = np.array(A.sum(axis=1)).flatten()
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg_inv_sqrt == np.inf] = 0.0
        D_inv_sqrt = sp.diags(deg_inv_sqrt)

        norm = D_inv_sqrt @ A @ D_inv_sqrt
        norm = norm.tocoo()

        # Convert to torch sparse tensor
        indices = torch.from_numpy(np.vstack([norm.row, norm.col]).astype(np.int64))
        values = torch.from_numpy(norm.data.astype(np.float32))
        shape = torch.Size(norm.shape)

        self.norm_adj = torch.sparse_coo_tensor(indices, values, shape).coalesce().to(self.device)
        return self.norm_adj

    # ----------------------------
    # Training helpers
    # ----------------------------
    def _sample_batch(self):
        """
        Uniformly sample users from train interactions, then sample:
          - one positive from user's positives
          - one negative not in user's positives
        """
        rng = np.random.default_rng(self.seed)
        users = rng.choice(self.train_df["user_idx"].values, size=self.batch_size, replace=True)

        pos_items = np.empty(self.batch_size, dtype=np.int64)
        neg_items = np.empty(self.batch_size, dtype=np.int64)

        for idx, u in enumerate(users):
            u = int(u)
            pos = rng.choice(list(self.user_pos[u]))
            while True:
                neg = int(rng.integers(0, self.num_items))
                if neg not in self.user_pos[u]:
                    break
            pos_items[idx] = int(pos)
            neg_items[idx] = neg

        return (
            torch.tensor(users, dtype=torch.long, device=self.device),
            torch.tensor(pos_items, dtype=torch.long, device=self.device),
            torch.tensor(neg_items, dtype=torch.long, device=self.device),
        )

    # ----------------------------
    # Fit
    # ----------------------------
    def fit(self, epochs: int = 200, eval_every: int = 10, k_eval: int = 10, n_neg_eval: int = 999,
            early_stop: bool = True, patience: int = 5):
        assert self.train_df is not None, "Call split_sequential() first."

        self._build_norm_adj()
        self.model = LightGCN._Model(self.num_users, self.num_items, self.latent_dim, self.n_layers).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_ndcg = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0
            steps = max(1, len(self.train_df) // self.batch_size)

            for _ in range(steps):
                users, pos_i, neg_i = self._sample_batch()

                user_all, item_all = self.model(self.norm_adj)
                u_emb = user_all[users]
                pos_emb = item_all[pos_i]
                neg_emb = item_all[neg_i]

                loss = LightGCN._Model.bpr_loss(u_emb, pos_emb, neg_emb)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += float(loss.item())

            print(f"Epoch {epoch}, Avg loss: {total_loss/steps:.4f}")

            if eval_every and (epoch % eval_every == 0):
                r, nd = self.evaluate_sampled(k=k_eval, n_neg=n_neg_eval)
                print(f"  Eval @Epoch {epoch}: Recall@{k_eval}={r:.4f}, NDCG@{k_eval}={nd:.4f}")

                if early_stop:
                    if nd > best_ndcg:
                        best_ndcg = nd
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                        if bad >= patience:
                            print(f"Early stopping. Best NDCG@{k_eval}: {best_ndcg:.4f}")
                            break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    # ----------------------------
    # Evaluation (sampled)
    # ----------------------------
    @torch.no_grad()
    def _rank_with_sampled_negatives(self, u: int, pos_items: Set[int], n_neg: int = 999, k: int = 10):
        self.model.eval()
        user_all, item_all = self.model(self.norm_adj)

        seen = self.train_hist.get(u, set())
        pos_list = list(pos_items)

        forbidden = seen | set(pos_list)
        rng = np.random.default_rng(self.seed + u)
        negs = []
        while len(negs) < n_neg:
            cand = int(rng.integers(0, self.num_items))
            if cand not in forbidden:
                negs.append(cand)

        candidates = np.array(pos_list + negs, dtype=np.int64)
        u_emb = user_all[u].unsqueeze(0)  # [1, d]
        cand_emb = item_all[candidates]   # [C, d]
        scores = (u_emb * cand_emb).sum(dim=1).detach().cpu().numpy()

        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        top_items = candidates[topk_idx]
        return top_items, set(pos_list)

    @staticmethod
    def _recall_at_k(recs, positives):
        return len(set(recs) & positives) / len(positives) if positives else 0.0

    @staticmethod
    def _ndcg_at_k(recs, positives):
        dcg = 0.0
        for r, it in enumerate(recs, start=1):
            if it in positives:
                dcg += 1.0 / np.log2(r + 1)
        ideal = min(len(positives), len(recs))
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_sampled(self, k: int = 10, n_neg: int = 999):
        assert self.model is not None and self.norm_adj is not None, "Call fit() first."

        rs, ns = [], []
        for u in self.eval_users:
            pos = self.test_hist.get(u, set())
            if not pos:
                continue
            recs, pos_set = self._rank_with_sampled_negatives(u, pos_items=pos, n_neg=n_neg, k=k)
            rs.append(LightGCN._recall_at_k(recs, pos_set))
            ns.append(LightGCN._ndcg_at_k(recs, pos_set))

        return float(np.mean(rs)), float(np.mean(ns))

    # ----------------------------
    # Recommend for a raw user_id
    # ----------------------------
    @torch.no_grad()
    def recommend(self, user_id: str, k: int = 10, n_candidates: int = 5000):
        """
        Returns top-k recommended parent_asin IDs from a sampled candidate pool.
        (Full ranking over all items is expensive at 100k+ scale.)
        """
        u = int(self.user_encoder.transform([str(user_id)])[0])

        self.model.eval()
        user_all, item_all = self.model(self.norm_adj)

        seen = self.train_hist.get(u, set())
        rng = np.random.default_rng(self.seed + 999)

        cands = []
        while len(cands) < n_candidates:
            cand = int(rng.integers(0, self.num_items))
            if cand not in seen:
                cands.append(cand)

        cands = np.array(cands, dtype=np.int64)
        scores = (user_all[u].unsqueeze(0) * item_all[cands]).sum(dim=1).detach().cpu().numpy()

        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        top_items = cands[topk_idx]

        return list(self.item_encoder.inverse_transform(top_items))
    

class SASRec:
    """
    SASRec (self-attention sequential recommender) for next-item ranking.

    Input: Behavior CSV with columns (at least):
      user_id, parent_asin, timestamp
    """

    # ----------------------------
    # Model
    # ----------------------------
    class _Model(nn.Module):
        def __init__(
            self,
            num_items: int,
            max_len: int = 50,
            d_model: int = 64,
            n_heads: int = 2,
            n_layers: int = 2,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.num_items = num_items
            self.max_len = max_len
            self.d_model = d_model

            # item indices: 0 = PAD, 1..num_items are real items
            self.item_emb = nn.Embedding(num_items + 1, d_model, padding_idx=0)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.dropout = nn.Dropout(dropout)
            self.layernorm = nn.LayerNorm(d_model)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

            nn.init.normal_(self.item_emb.weight, std=0.01)
            nn.init.normal_(self.pos_emb.weight, std=0.01)

        def forward(self, seq_items: torch.LongTensor):
            """
            seq_items: [B, L] with 0 as PAD
            returns: hidden states [B, L, d_model]
            """
            B, L = seq_items.shape
            pos = torch.arange(L, device=seq_items.device).unsqueeze(0).repeat(B, 1)

            x = self.item_emb(seq_items) + self.pos_emb(pos)
            x = self.layernorm(self.dropout(x))

            # Causal attention mask: prevent attending to future positions
            # mask shape: [L, L], True means "blocked"
            causal_mask = torch.triu(torch.ones(L, L, device=seq_items.device), diagonal=1).bool()

            # Key padding mask: True for PAD positions
            pad_mask = (seq_items == 0)

            h = self.encoder(x, mask=causal_mask, src_key_padding_mask=pad_mask)
            return h

        def predict_logits(self, h_last: torch.Tensor, items: torch.LongTensor):
            """
            h_last: [B, d]
            items: [B] item ids in 1..num_items
            returns logits [B]
            """
            v = self.item_emb(items)  # [B, d]
            return (h_last * v).sum(dim=1)

    # ----------------------------
    # Dataset for training
    # ----------------------------
    class _TrainDataset(Dataset):
        """
        For each user sequence, sample one position t and predict item at t+1
        Input sequence is items[ max(0, t-max_len+1) : t+1 ] (includes item at t)
        Target is items[t+1]
        """
        def __init__(
            self,
            user_seqs: Dict[int, List[int]],
            num_items: int,
            max_len: int,
            n_neg: int = 1,
            seed: int = 42,
        ):
            self.user_ids = list(user_seqs.keys())
            self.user_seqs = user_seqs
            self.num_items = num_items
            self.max_len = max_len
            self.n_neg = n_neg
            self.rng = np.random.default_rng(seed)

            # build user->set for fast negative sampling
            self.user_item_set = {u: set(seq) for u, seq in user_seqs.items()}

        def __len__(self):
            return len(self.user_ids)

        def __getitem__(self, idx):
            u = self.user_ids[idx]
            seq = self.user_seqs[u]

            if len(seq) < 2:
               # should not happen if split_sequential filtered correctly
               # but just in case, return a valid dummy sample
               # (better: raise, but dummy prevents crashing)
               prefix = [0] * (self.max_len - 1) + [seq[0]]
               target = seq[0]
               negs = [1] * self.n_neg
               return (
                    torch.tensor(prefix, dtype=torch.long),
                    torch.tensor(target, dtype=torch.long),
                    torch.tensor(negs, dtype=torch.long),
                )

            # need at least 2 items to have a next-item target
            # caller should filter users accordingly
            t = int(self.rng.integers(0, len(seq) - 1))  # predict t+1
            prefix = seq[max(0, t - self.max_len + 1): t + 1]
            target = seq[t + 1]

            # pad to max_len (left pad with 0)
            pad_len = self.max_len - len(prefix)
            seq_pad = [0] * pad_len + prefix

            # negatives
            negs = []
            forb = self.user_item_set[u]
            while len(negs) < self.n_neg:
                cand = int(self.rng.integers(1, self.num_items + 1))
                if cand not in forb:
                    negs.append(cand)

            return (
                torch.tensor(seq_pad, dtype=torch.long),
                torch.tensor(target, dtype=torch.long),
                torch.tensor(negs, dtype=torch.long),
            )

    # ----------------------------
    # Init
    # ----------------------------
    def __init__(
        self,
        behavior_csv_path: str,
        user_col: str = "user_id",
        item_col: str = "parent_asin",
        time_col: str = "timestamp",
        max_len: int = 50,
        d_model: int = 64,
        n_heads: int = 2,
        n_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 256,
        n_neg_train: int = 1,
        device: Optional[str] = None,
        seed: int = 42,
    ):
        self.behavior_csv_path = behavior_csv_path
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_neg_train = n_neg_train
        self.seed = seed

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        np.random.seed(seed)
        torch.manual_seed(seed)

        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()

        self.df = None
        self.num_users = None
        self.num_items = None

        # sequences in item ids 1..num_items
        self.user_seqs_all = None
        self.user_seqs_train = None
        self.user_seqs_test = None  # test positives (future part)

        self.eval_users = None

        self.model = None
        self.opt = None

    # ----------------------------
    # Data
    # ----------------------------
    def load_and_prepare(self, dedup_last: bool = True):
        df = pd.read_csv(self.behavior_csv_path)
        df = df.sort_values(self.time_col)

        if dedup_last:
            df = df.drop_duplicates(subset=[self.user_col, self.item_col], keep="last")

        df["user_idx"] = self.user_encoder.fit_transform(df[self.user_col].astype(str))
        df["item_idx0"] = self.item_encoder.fit_transform(df[self.item_col].astype(str))

        # shift items by +1 so 0 is PAD
        df["item_idx"] = df["item_idx0"].astype(int) + 1

        self.df = df
        self.num_users = int(df["user_idx"].nunique())
        self.num_items = int(df["item_idx"].max())  # already shifted

        print(f"After dedup -> interactions: {len(df)} users: {self.num_users} items: {self.num_items}")
        return df

    def build_sequences(self, min_len: int = 2):
        assert self.df is not None, "Call load_and_prepare() first."

        user_seqs = {}
        for u, g in self.df.groupby("user_idx"):
            seq = g.sort_values(self.time_col)["item_idx"].astype(int).tolist()
            if len(seq) >= min_len:
                user_seqs[int(u)] = seq

        self.user_seqs_all = user_seqs
        print("Users with seq_len>=2:", len(user_seqs))
        return user_seqs

    def split_sequential(self, train_ratio: float = 0.8):
        """
        Per-user sequential split:
          train_seq = first 80%
          test_seq  = last 20% (future positives)
        """
        assert self.user_seqs_all is not None, "Call build_sequences() first."

        train_seqs = {}
        test_seqs = {}

        for u, seq in self.user_seqs_all.items():
            cut = int(len(seq) * train_ratio)
            
            # REQUIRE: train len >= 2 for SASRec training, test len >= 1 for eval
            if cut < 2:
               continue
            if (len(seq) - cut) < 1:
               continue
            
            train_seqs[u] = seq[:cut]
            test_seqs[u] = seq[cut:]

        self.user_seqs_train = train_seqs
        self.user_seqs_test = test_seqs
        self.eval_users = sorted(set(train_seqs) & set(test_seqs))

        print("Users train/test:", len(self.eval_users))
        return train_seqs, test_seqs

    # ----------------------------
    # Train
    # ----------------------------
    @staticmethod
    def _bpr_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor):
        """
        pos_logits: [B]
        neg_logits: [B, n_neg]
        """
        # maximize pos > neg
        diff = pos_logits.unsqueeze(1) - neg_logits
        return -torch.log(torch.sigmoid(diff) + 1e-10).mean()

    def fit(self, epochs: int = 50, eval_every: int = 5, k_eval: int = 10, n_neg_eval: int = 999,
            early_stop: bool = True, patience: int = 5):
        assert self.user_seqs_train is not None, "Call split_sequential() first."

        self.model = SASRec._Model(
            num_items=self.num_items,
            max_len=self.max_len,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        ds = SASRec._TrainDataset(
            user_seqs=self.user_seqs_train,
            num_items=self.num_items,
            max_len=self.max_len,
            n_neg=self.n_neg_train,
            seed=self.seed,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=True)

        best_ndcg = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            total = 0.0
            n = 0

            for seq_pad, target, negs in loader:
                seq_pad = seq_pad.to(self.device)     # [B, L]
                target = target.to(self.device)       # [B]
                negs = negs.to(self.device)           # [B, n_neg]

                h = self.model(seq_pad)               # [B, L, d]
                h_last = h[:, -1, :]                  # predict next from last position

                pos_logits = self.model.predict_logits(h_last, target)  # [B]
                # neg logits: compute per negative
                B, nneg = negs.shape
                neg_flat = negs.view(-1)              # [B*nneg]
                h_rep = h_last.unsqueeze(1).repeat(1, nneg, 1).view(-1, self.d_model)
                neg_logits = self.model.predict_logits(h_rep, neg_flat).view(B, nneg)

                loss = SASRec._bpr_loss(pos_logits, neg_logits)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total += float(loss.item()) * seq_pad.size(0)
                n += seq_pad.size(0)

            print(f"Epoch {epoch}, Avg loss: {total/n:.4f}")

            if eval_every and (epoch % eval_every == 0):
                r, nd = self.evaluate_sampled(k=k_eval, n_neg=n_neg_eval)
                print(f"  Eval @Epoch {epoch}: Recall@{k_eval}={r:.4f}, NDCG@{k_eval}={nd:.4f}")

                if early_stop:
                    if nd > best_ndcg:
                        best_ndcg = nd
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                        bad = 0
                    else:
                        bad += 1
                        if bad >= patience:
                            print(f"Early stopping. Best NDCG@{k_eval}: {best_ndcg:.4f}")
                            break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    # ----------------------------
    # Eval (sampled)
    # ----------------------------
    @torch.no_grad()
    def _score_candidates(self, u: int, candidates: np.ndarray):
        """
        Score candidate items for user u using last max_len items from train seq.
        """
        self.model.eval()
        seq = self.user_seqs_train[u]
        seq = seq[-self.max_len:]  # last L
        seq_pad = [0] * (self.max_len - len(seq)) + seq

        seq_pad = torch.tensor(seq_pad, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, L]
        h = self.model(seq_pad)
        h_last = h[:, -1, :]  # [1, d]

        items = torch.tensor(candidates, dtype=torch.long, device=self.device)
        h_rep = h_last.repeat(items.size(0), 1)
        scores = self.model.predict_logits(h_rep, items).detach().cpu().numpy()
        return scores

    @staticmethod
    def _recall_at_k(recs, positives):
        return len(set(recs) & positives) / len(positives) if positives else 0.0

    @staticmethod
    def _ndcg_at_k(recs, positives):
        dcg = 0.0
        for r, it in enumerate(recs, start=1):
            if it in positives:
                dcg += 1.0 / np.log2(r + 1)
        ideal = min(len(positives), len(recs))
        idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate_sampled(self, k: int = 10, n_neg: int = 999):
        assert self.model is not None, "Call fit() first."
        rs, ns = [], []

        for u in self.eval_users:
            positives = set(self.user_seqs_test[u])  # future items (could be multiple)
            if not positives:
                continue

            # sample negatives not in user's full history (train+test) to be strict
            forb = set(self.user_seqs_all[u])
            rng = np.random.default_rng(self.seed + u)

            negs = []
            while len(negs) < n_neg:
                cand = int(rng.integers(1, self.num_items + 1))
                if cand not in forb:
                    negs.append(cand)

            candidates = np.array(list(positives) + negs, dtype=np.int64)
            scores = self._score_candidates(u, candidates)

            topk_idx = np.argpartition(scores, -k)[-k:]
            topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
            topk_items = candidates[topk_idx]

            rs.append(SASRec._recall_at_k(topk_items, positives))
            ns.append(SASRec._ndcg_at_k(topk_items, positives))

        return float(np.mean(rs)), float(np.mean(ns))

    # ----------------------------
    # Recommend for raw user_id
    # ----------------------------
    @torch.no_grad()
    def recommend(self, user_id: str, k: int = 10, n_candidates: int = 5000):
        u = int(self.user_encoder.transform([str(user_id)])[0])

        # sample candidate pool excluding full history
        forb = set(self.user_seqs_all.get(u, []))
        rng = np.random.default_rng(self.seed + 999)

        cands = []
        while len(cands) < n_candidates:
            cand = int(rng.integers(1, self.num_items + 1))
            if cand not in forb:
                cands.append(cand)

        cands = np.array(cands, dtype=np.int64)
        scores = self._score_candidates(u, cands)

        topk_idx = np.argpartition(scores, -k)[-k:]
        topk_idx = topk_idx[np.argsort(scores[topk_idx])[::-1]]
        top_items = cands[topk_idx]

        # map back to original parent_asin
        # remember: internal items are +1 shifted, so subtract 1 before inverse_transform
        top_zero = (top_items - 1).astype(int)
        return list(self.item_encoder.inverse_transform(top_zero))