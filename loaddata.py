import os
import torch
import pandas as pd
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader


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



def load_mggat_data(data_dir="dataset/PA/"):
    print("Loading data...")

    df_data = pd.read_csv(os.path.join(data_dir, "data.csv"))

    train_df = df_data[df_data['is_train'] == True]
    tune_df = df_data[df_data['is_tune'] == True]
    test_df = df_data[df_data['is_test'] == True]

    train_dataset = YelpRatingDataset(train_df)
    tune_dataset = YelpRatingDataset(tune_df)
    test_dataset = YelpRatingDataset(test_df)

    df_user_feat = pd.read_csv(os.path.join(data_dir, "user_features.csv"))
    df_item_feat = pd.read_csv(os.path.join(data_dir, "item_features.csv"))

    S_u = torch.tensor(df_user_feat.values, dtype=torch.float32)
    S_b = torch.tensor(df_item_feat.values, dtype=torch.float32)

    edge_indices_u = [
        load_graph_from_npz(os.path.join(data_dir, "user_graph.npz")),
        load_graph_from_npz(os.path.join(data_dir, "user_2hop_graph.npz"))
    ]

    edge_indices_b = [
        load_graph_from_npz(os.path.join(data_dir, "item_graph.npz")),
        load_graph_from_npz(os.path.join(data_dir, "item_2hop_graph.npz"))
    ]

    return S_u, S_b, edge_indices_u, edge_indices_b, train_dataset, tune_dataset, test_dataset



if __name__ == '__main__':
    try:
        S_u, S_b, eu, eb, train_ds, tune_ds, test_ds = load_mggat_data()

        print("\n=== Dataset Statistics ===")
        print(f"Users: {S_u.shape[0]} | User Feature Dim: {S_u.shape[1]}")
        print(f"Items: {S_b.shape[0]} | Item Feature Dim: {S_b.shape[1]}")
        print("-" * 30)
        print(f"User 1-hop edges: {eu[0].shape[1]}")
        print(f"User 2-hop edges: {eu[1].shape[1]}")
        print(f"Item 1-hop edges: {eb[0].shape[1]}")
        print(f"Item 2-hop edges: {eb[1].shape[1]}")
        print("-" * 30)
        print(f"Train samples: {len(train_ds)}")
        print(f"Tune (Val) samples: {len(tune_ds)}")
        print(f"Test samples: {len(test_ds)}")
        # test DataLoader
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        batch_u, batch_i, batch_r = next(iter(train_loader))
        print("\nOne batch output:")
        print(f"Users shape: {batch_u.shape}, Items shape: {batch_i.shape}, Ratings shape: {batch_r.shape}")

    except FileNotFoundError as e:
        print(
            f"Error info: {e}")