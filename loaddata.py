import os
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import kneighbors_graph

class YelpRatingDataset(Dataset):
    def __init__(self, data_df):
        self.users = torch.tensor(data_df['user_id'].values, dtype=torch.long)
        self.items = torch.tensor(data_df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(data_df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


def load_graph_from_npz(file_path):
    adj_matrix = sp.load_npz(file_path).tocoo()
    row = torch.from_numpy(adj_matrix.row).to(torch.long)
    col = torch.from_numpy(adj_matrix.col).to(torch.long)
    return torch.stack([row, col], dim=0)


def generate_implicit_features(train_df, num_users, num_items, latent_dim=20):
    """
    Generate implicit features via SVD on the binarized training rating matrix.
    Matches the methodology described in the paper.
    """
    print(f"Generating SVD implicit features (dim={latent_dim})...")

    # 1. Binarize continuous ratings: any interaction > 0 becomes 1
    row = train_df['user_id'].values
    col = train_df['item_id'].values
    data = np.ones(len(train_df))  # Binarized feedback

    # Create sparse rating matrix
    X_bina = sp.coo_matrix((data, (row, col)), shape=(num_users, num_items)).asfptype()

    # 2. Apply SVD
    k = min(latent_dim, min(num_users, num_items) - 1)
    u, s, vt = svds(X_bina, k=k)

    # 3. Compute implicit embeddings: U * \Sigma^{1/2} and V^T * \Sigma^{1/2}
    s_sqrt = np.diag(np.sqrt(s))
    S_u_imp = np.dot(u, s_sqrt)
    S_b_imp = np.dot(vt.T, s_sqrt)

    return torch.tensor(S_u_imp, dtype=torch.float32), torch.tensor(S_b_imp, dtype=torch.float32)


def load_mggat_data(data_dir="dataset/PA/", num_item_graphs=4):
    print("Loading explicit data...")

    df_data = pd.read_csv(os.path.join(data_dir, "data.csv"))

    train_df = df_data[df_data['is_train'] == True]
    tune_df = df_data[df_data['is_tune'] == True]
    test_df = df_data[df_data['is_test'] == True]

    train_dataset = YelpRatingDataset(train_df)
    tune_dataset = YelpRatingDataset(tune_df)
    test_dataset = YelpRatingDataset(test_df)

    # Load Explicit Features
    df_user_feat = pd.read_csv(os.path.join(data_dir, "user_features.csv"))
    df_item_feat = pd.read_csv(os.path.join(data_dir, "item_features.csv"))

    S_u_exp = torch.tensor(df_user_feat.values, dtype=torch.float32)
    S_b_exp = torch.tensor(df_item_feat.values, dtype=torch.float32)

    print("Loading network graphs...")
    graph_files = [
        "item_graph.npz",
        "item_graph_geo.npz",
        "item_graph_covisit.npz",
        "item_graph_segment.npz"
    ][:num_item_graphs]

    edge_indices_b = []
    for f in graph_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            edge_indices_b.append(load_graph_from_npz(path))

    print(f"Business graphs loaded: {len(edge_indices_b)}")

    # User Graph: Only ONE friendship graph (No 2-hop noise)
    edge_indices_u = [
        load_graph_from_npz(os.path.join(data_dir, "user_graph.npz"))
    ]

    # edge_indices_b = [load_graph_from_npz(os.path.join(data_dir, "item_graph.npz"))]

    return S_u_exp, S_b_exp, edge_indices_u, edge_indices_b, train_dataset, tune_dataset, test_dataset, train_df


def build_all_item_graphs(data_dir="dataset/PA/", k=10):
    item_graph = sp.load_npz(f"{data_dir}/item_graph.npz")

    print("Building geo graph...")
    coords = pd.read_csv(f"{data_dir}/item_coordinates.csv").fillna(0)
    item_graph_geo = kneighbors_graph(coords.values, k)
    sp.save_npz(f"{data_dir}/item_graph_geo.npz", sp.coo_matrix(item_graph_geo))

    print("Building co-visit graph...")
    data = pd.read_csv(f"{data_dir}/data.csv")
    train = data[data['is_train'] == True]
    num_items = pd.read_csv(f"{data_dir}/item_features.csv").shape[0]
    num_users = train['user_id'].max() + 1  # 确保矩阵维度包容所有 user_id

    row_idx = train['user_id'].values
    col_idx = train['item_id'].values
    vals = np.ones(len(train))
    user_item_mat = sp.csr_matrix((vals, (row_idx, col_idx)), shape=(num_users, num_items))

    item_covisit_mat = user_item_mat.T.dot(user_item_mat)
    item_covisit_mat.setdiag(0)
    item_covisit_mat.eliminate_zeros()

    top_k_rows, top_k_cols, top_k_data = [], [], []
    for i in range(num_items):
        row = item_covisit_mat.getrow(i)
        if row.nnz > 0:
            indices = row.indices
            counts = row.data
            if len(counts) > k:
                # argpartition 是找出最大 k 个值的索引，速度比全排序快
                best_idx = np.argpartition(counts, -k)[-k:]
                best_indices = indices[best_idx]
            else:
                best_indices = indices

            top_k_rows.extend([i] * len(best_indices))
            top_k_cols.extend(best_indices)
            top_k_data.extend([1.0] * len(best_indices))  # 保持图的无权二值化，或者你也可以保留频次作为权重

    item_graph_covisit = sp.csr_matrix(
        (top_k_data, (top_k_rows, top_k_cols)),
        shape=(num_items, num_items)
    )
    sp.save_npz(f"{data_dir}/item_graph_covisit.npz",
                sp.coo_matrix(item_graph_covisit))

    print("Building segment graph...")
    segments = pd.read_csv(f"{data_dir}/item_segments.csv")
    item_graph_segment = kneighbors_graph(segments.values, k)
    sp.save_npz(f"{data_dir}/item_graph_segment.npz", sp.coo_matrix(item_graph_segment))

    print("All graphs saved.")
    return [item_graph, item_graph_geo, item_graph_covisit, item_graph_segment]

# if __name__ == '__main__':
    # try:
    #     S_u, S_b, eu, eb, train_ds, tune_ds, test_ds,train_df = load_mggat_data()
    #
    #     print("\n=== Dataset Statistics ===")
    #     print(f"Users: {S_u.shape[0]} | Final User Feature Dim (Exp + Imp): {S_u.shape[1]}")
    #     print(f"Items: {S_b.shape[0]} | Final Item Feature Dim (Exp + Imp): {S_b.shape[1]}")
    #     print("-" * 30)
    #     print(f"User graphs loaded: {len(eu)} | Edges in G_u: {eu[0].shape[1]}")
    #     print(f"Business graphs loaded: {len(eb)}")
    #     for i, g in enumerate(eb):
    #         print(f"  Item Graph {i + 1} edges: {g.shape[1]}")
    #     print("-" * 30)
    #     print(f"Train samples: {len(train_ds)}")
    #     print(f"Tune (Val) samples: {len(tune_ds)}")
    #     print(f"Test samples: {len(test_ds)}")
    #
    #     train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    #     batch_u, batch_i, batch_r = next(iter(train_loader))
    #     print("\nOne batch output:")
    #     print(f"Users shape: {batch_u.shape}, Items shape: {batch_i.shape}, Ratings shape: {batch_r.shape}")
    #
    # except Exception as e:
    #     print(f"Error info: {e}")
    # build_all_item_graphs()