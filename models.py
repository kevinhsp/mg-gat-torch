import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

from loaddata import load_mggat_data


# ============================================================================
# 1. Multi-Graph Graph Attention Layer (Corresponds to Eq. 2, Eq. 3, Eq. 4)
# ============================================================================
class MultiGraphGATLayer(MessagePassing):
    """
    Multi-Graph Attention Network Layer.
    Computes neighbor importance (NIG) and aggregates features across multiple graphs.
    """

    def __init__(self, in_dim, out_dim, num_graphs):
        # aggr='add' corresponds to the summation in Eq. 4
        super(MultiGraphGATLayer, self).__init__(aggr='add', node_dim=0)
        self.num_graphs = num_graphs
        self.out_dim = out_dim

        # Eq. 2: Linear transformation for auxiliary information (W^(1))
        self.W_1 = nn.Linear(in_dim, out_dim, bias=False)

        # Eq. 3: Learnable coefficients for Feature Relevance (FR)
        # Split into focal node (self) and neighbor node (nb) for efficient computation
        self.a_self = nn.Parameter(torch.empty(1, out_dim))
        self.a_nb = nn.Parameter(torch.empty(1, out_dim))

        # Eq. 3: Learnable weights for different graph types (\omega_g)
        self.omega = nn.Parameter(torch.empty(num_graphs))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.a_self)
        nn.init.xavier_uniform_(self.a_nb)
        nn.init.ones_(self.omega) / self.num_graphs

    def forward(self, S, edge_indices):
        """
        Args:
            S (Tensor): Node auxiliary information [num_nodes, in_dim]
            edge_indices (List[Tensor]): List of edge_index for multiple graphs
        Returns:
            H_2 (Tensor): Aggregated embeddings [num_nodes, out_dim]
        """
        # Eq. 2: First linear transformation (H^(1) = W^(1) * S)
        H_1 = self.W_1(S)

        # Pre-compute attention components: a^T * H^(1)
        alpha_self = (H_1 * self.a_self).sum(dim=-1)
        alpha_nb = (H_1 * self.a_nb).sum(dim=-1)

        # Eq. 4: Aggregate embeddings over multiple graphs
        H_2 = torch.zeros_like(H_1)
        for g in range(self.num_graphs):
            edge_index = edge_indices[g]
            # PyG propagate calls message() and aggregates results
            H_2_g = self.propagate(edge_index, x=H_1, alpha_self=alpha_self, alpha_nb=alpha_nb)
            # Add with learnable graph weight \omega_g
            H_2 = H_2 + self.omega[g] * H_2_g

        return H_2

    def message(self, x_j, alpha_self_i, alpha_nb_j, index, ptr, size_i):
        """
        Computes attention scores \alpha^{l \rightarrow j} (Eq. 3)
        """
        # attention_score = a^T [H_i || H_j]
        attention_score = alpha_self_i + alpha_nb_j
        attention_score = self.leaky_relu(attention_score)

        # Softmax normalization over the neighborhood
        alpha = softmax(attention_score, index, ptr, size_i)

        # Return weighted features
        return x_j * alpha.unsqueeze(-1)


# ============================================================================
# 2. Main MG-GAT Recommender Model (Corresponds to Eq. 5, Eq. 6, Eq. 7)
# ============================================================================
class MGGATRecommender(nn.Module):
    """
    Complete MG-GAT model integrating auxiliary features, multi-graph structures,
    and rating prediction.
    """

    def __init__(self, num_users, num_items, u_in_dim, i_in_dim, latent_dim, final_dim, num_u_graphs, num_i_graphs):
        super(MGGATRecommender, self).__init__()

        # --- Stage 1: Graph Attention Layers (Eq. 2, 3, 4) ---
        self.user_gat = MultiGraphGATLayer(u_in_dim, latent_dim, num_u_graphs)
        self.item_gat = MultiGraphGATLayer(i_in_dim, latent_dim, num_i_graphs)

        # --- Stage 2: Nonlinear Dense Layer (Eq. 5) ---
        # User nonlinear dense layer parameters
        self.W_u_2 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_us_2 = nn.Linear(u_in_dim, latent_dim, bias=True)  # bias acts as b_u^(1)

        # Item (Business) nonlinear dense layer parameters
        self.W_b_2 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_bs_2 = nn.Linear(i_in_dim, latent_dim, bias=True)

        # --- Stage 3: Final Aggregation Layer (Eq. 6) ---
        self.W_u_3 = nn.Linear(latent_dim, final_dim, bias=False)
        self.H_u_4 = nn.Embedding(num_users, final_dim)  # Latent representation from rating matrix

        self.W_b_3 = nn.Linear(latent_dim, final_dim, bias=False)
        self.H_b_4 = nn.Embedding(num_items, final_dim)

        # --- Stage 4: Rating Prediction Biases (Eq. 7) ---
        self.b_u_x = nn.Embedding(num_users, 1)
        self.b_b_x = nn.Embedding(num_items, 1)
        self.b_x = nn.Parameter(torch.zeros(1))

        # Rating normalization bounds
        self.r_min = 1.0
        self.r_max = 5.0

        self._init_weights()

    def _init_weights(self):
        # Initialize embeddings with small random values
        nn.init.normal_(self.H_u_4.weight, std=0.01)
        nn.init.normal_(self.H_b_4.weight, std=0.01)
        nn.init.zeros_(self.b_u_x.weight)
        nn.init.zeros_(self.b_b_x.weight)

    def forward(self, user_indices, item_indices, S_u, S_b, edge_indices_u, edge_indices_b):
        """
        Forward pass for predicting user-item ratings.
        """
        # Eq. 4: Compute neighbor-aggregated embeddings
        H_u_2 = self.user_gat(S_u, edge_indices_u)
        H_b_2 = self.item_gat(S_b, edge_indices_b)

        # Eq. 5: Nonlinear dense layer
        # H^(3) = actv_1(W^(2)*H^(2) + W_s^(2)*S + b^(1))
        H_u_3 = F.elu(self.W_u_2(H_u_2) + self.W_us_2(S_u))
        H_b_3 = F.elu(self.W_b_2(H_b_2) + self.W_bs_2(S_b))

        # Eq. 6: Final embedding aggregation
        # U = actv_2(W^(3)*H^(3)) + H^(4)
        U_all = F.elu(self.W_u_3(H_u_3)) + self.H_u_4.weight
        B_all = F.elu(self.W_b_3(H_b_3)) + self.H_b_4.weight

        # Extract embeddings for the current batch
        U_batch = U_all[user_indices]
        B_batch = B_all[item_indices]

        # Eq. 7: Rating prediction
        # U_i * B_j^T + biases
        dot_product = (U_batch * B_batch).sum(dim=1)
        raw_prediction = dot_product + \
                         self.b_u_x(user_indices).squeeze() + \
                         self.b_b_x(item_indices).squeeze() + \
                         self.b_x

        # norm() = (max - min) * sigmoid(x) + min
        pred_ratings = (self.r_max - self.r_min) * torch.sigmoid(raw_prediction) + self.r_min

        return pred_ratings


if __name__ == '__main__':
    print("=== Step 1: Loading Real Data ===")
    # Change "./" to your actual directory containing the files
    S_u, S_b, edge_indices_u, edge_indices_b, train_ds, tune_ds, test_ds = load_mggat_data()

    # Extract dimensions directly from the loaded real data
    num_users, u_in_dim = S_u.shape
    num_items, i_in_dim = S_b.shape
    num_u_graphs = len(edge_indices_u)
    num_i_graphs = len(edge_indices_b)

    print("\n=== Step 2: Initializing Model ===")
    latent_dim = 32  # Hyperparameter: Intermediate hidden dimension
    final_dim = 32  # Hyperparameter: Final embedding dimension

    # We highly recommend pushing to GPU if available due to the 11M edges
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = MGGATRecommender(
        num_users, num_items, u_in_dim, i_in_dim,
        latent_dim, final_dim, num_u_graphs, num_i_graphs
    ).to(device)

    # Move all graphs and features to the selected device
    S_u = S_u.to(device)
    S_b = S_b.to(device)
    edge_indices_u = [e.to(device) for e in edge_indices_u]
    edge_indices_b = [e.to(device) for e in edge_indices_b]

    print("\n=== Step 3: Running Forward & Backward Pass on a Real Batch ===")
    train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
    batch_u, batch_i, true_ratings = next(iter(train_loader))

    # Move batch to device
    batch_u = batch_u.to(device)
    batch_i = batch_i.to(device)
    true_ratings = true_ratings.to(device)

    # Forward Pass
    model.train()
    print("Executing forward pass (this might take a moment due to 11M edges)...")
    predictions = model(batch_u, batch_i, S_u, S_b, edge_indices_u, edge_indices_b)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5].detach().cpu().numpy()}")

    # Backward Pass
    criterion = nn.MSELoss()
    loss = criterion(predictions, true_ratings)
    print("Executing backward pass...")
    loss.backward()

    print(f"Real Batch Initial Loss: {loss.item():.4f}")
    if model.W_u_2.weight.grad is not None:
        print("Success! Gradients successfully propagated through the real data.")
    else:
        print("Warning: Gradient calculation failed.")