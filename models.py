import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, coalesce


def get_activation(name):
    activations = {
        'elu': F.elu,
        'exponential': torch.exp,
        'hard_sigmoid': F.hardsigmoid,
        'linear': lambda x: x,  # 线性即不改变
        'relu': F.relu,
        'selu': F.selu,
        'sigmoid': torch.sigmoid,
        'softplus': F.softplus,
        'softsign': F.softsign,
        'tanh': torch.tanh
    }
    return activations.get(name, F.elu)
# ============================================================================
# 1. User Attention Layer ( Eq. 3)
# ============================================================================
class UserGATLayer(MessagePassing):

    def __init__(self, in_dim, out_dim):
        super(UserGATLayer, self).__init__(aggr='add', node_dim=0)
        self.W_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.a_self = nn.Parameter(torch.empty(1, out_dim))
        self.a_nb = nn.Parameter(torch.empty(1, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.a_self)
        nn.init.xavier_uniform_(self.a_nb)

    def forward(self, S, edge_index):
        H_1 = self.W_1(S)
        alpha_self = (H_1 * self.a_self).sum(dim=-1)
        alpha_nb = (H_1 * self.a_nb).sum(dim=-1)

        return self.propagate(edge_index, x=H_1, alpha_self=alpha_self, alpha_nb=alpha_nb)

    def message(self, x_j, alpha_self_i, alpha_nb_j, index, ptr, size_i):
        attention_score = self.leaky_relu(alpha_self_i + alpha_nb_j)
        alpha = softmax(attention_score, index, ptr, size_i)
        return x_j * alpha.unsqueeze(-1)


# ============================================================================
# 2. Business  Attention Layer (Eq. 3 )
# ============================================================================
class BusinessMultiGraphGATLayer(MessagePassing):

    def __init__(self, in_dim, out_dim, num_graphs):
        super(BusinessMultiGraphGATLayer, self).__init__(aggr='add', node_dim=0)
        self.num_graphs = num_graphs
        self.W_1 = nn.Linear(in_dim, out_dim, bias=False)
        self.a_self = nn.Parameter(torch.empty(1, out_dim))
        self.a_nb = nn.Parameter(torch.empty(1, out_dim))
        self.omega = nn.Parameter(torch.empty(num_graphs))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_1.weight)
        nn.init.xavier_uniform_(self.a_self)
        nn.init.xavier_uniform_(self.a_nb)
        nn.init.constant_(self.omega, 1.0 / self.num_graphs)

    def forward(self, S, edge_indices):
        num_nodes = S.shape[0]
        H_1 = self.W_1(S)
        alpha_self = (H_1 * self.a_self).sum(dim=-1)
        alpha_nb = (H_1 * self.a_nb).sum(dim=-1)

        all_edges = []
        all_weights = []
        for g, edge_index in enumerate(edge_indices):
            all_edges.append(edge_index)
            all_weights.append(self.omega[g].expand(edge_index.shape[1]))

        combined_edge_index = torch.cat(all_edges, dim=1)
        combined_weights = torch.cat(all_weights, dim=0)

        unified_edge_index, unified_weights = coalesce(
            combined_edge_index, combined_weights, num_nodes=num_nodes, reduce='add'
        )

        H_2 = self.propagate(unified_edge_index, x=H_1,
                             alpha_self=alpha_self, alpha_nb=alpha_nb,
                             omega_weight=unified_weights)
        return H_2

    def message(self, x_j, alpha_self_i, alpha_nb_j, omega_weight, index, ptr, size_i):
        # Equation 3
        attention_score = omega_weight * (alpha_self_i + alpha_nb_j)
        attention_score = self.leaky_relu(attention_score)
        alpha = softmax(attention_score, index, ptr, size_i)
        return x_j * alpha.unsqueeze(-1)


# ============================================================================
# 3. Main MG-GAT Recommender Model
# ============================================================================
class MGGATRecommender(nn.Module):
    def __init__(self, num_users, num_items, u_in_dim, i_in_dim, latent_dim, final_dim, num_u_graphs, num_i_graphs, actv_in_name='elu', actv_out_name='elu'):
        super(MGGATRecommender, self).__init__()

        self.user_gat = UserGATLayer(u_in_dim, latent_dim)
        self.item_gat = BusinessMultiGraphGATLayer(i_in_dim, latent_dim, num_i_graphs)

        self.W_u_2 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_us_2 = nn.Linear(u_in_dim, latent_dim, bias=True)

        self.W_b_2 = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_bs_2 = nn.Linear(i_in_dim, latent_dim, bias=True)

        self.W_u_3 = nn.Linear(latent_dim, final_dim, bias=False)
        self.H_u_4 = nn.Embedding(num_users, final_dim)

        self.W_b_3 = nn.Linear(latent_dim, final_dim, bias=False)
        self.H_b_4 = nn.Embedding(num_items, final_dim)

        self.b_u_x = nn.Embedding(num_users, 1)
        self.b_b_x = nn.Embedding(num_items, 1)
        self.b_x = nn.Parameter(torch.zeros(1))

        self.actv_1 = get_activation(actv_in_name)
        self.actv_2 = get_activation(actv_out_name)

        self.r_min = 1.0
        self.r_max = 5.0
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.H_u_4.weight, std=0.01)
        nn.init.normal_(self.H_b_4.weight, std=0.01)
        nn.init.zeros_(self.b_u_x.weight)
        nn.init.zeros_(self.b_b_x.weight)

    def forward(self, user_indices, item_indices, S_u, S_b, edge_indices_u, edge_indices_b):

        H_u_2 = self.user_gat(S_u, edge_indices_u[0])
        H_b_2 = self.item_gat(S_b, edge_indices_b)

        H_u_3 = self.actv_1(self.W_u_2(H_u_2) + self.W_us_2(S_u))
        H_b_3 = self.actv_1(self.W_b_2(H_b_2) + self.W_bs_2(S_b))

        U_all = self.actv_2(self.W_u_3(H_u_3)) + self.H_u_4.weight
        B_all = self.actv_2(self.W_b_3(H_b_3)) + self.H_b_4.weight

        U_batch = U_all[user_indices]
        B_batch = B_all[item_indices]

        dot_product = (U_batch * B_batch).sum(dim=1)
        raw_prediction = dot_product + \
                         self.b_u_x(user_indices).squeeze() + \
                         self.b_b_x(item_indices).squeeze() + \
                         self.b_x

        pred_ratings = (self.r_max - self.r_min) * torch.sigmoid(raw_prediction) + self.r_min
        return pred_ratings