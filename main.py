import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax
from utils import load_data, load_graph_np, visualize_cluster
import process

# Configuration
parser = argparse.ArgumentParser(description='Graph Contrastive Learning')
parser.add_argument('--name', type=str, default='citeseer', help='Dataset name')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--n_clusters', type=int, default=3, help='Number of clusters')
parser.add_argument('--n_z', type=int, default=200, help='Latent space dimension')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
args = parser.parse_args()

args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.pretrain_path = f"data/{args.name}_pretrain.pkl"

# Data Loading & Feature Propagation
dataset = load_data(args.name)
adj_np, features, labels, idx_train, idx_val, idx_test = process.load_data(args.name)

def multi_scale_feature_propagation(features, adj, scales=[1, 2, 3]):
    """ Generate multi-scale propagated features. """
    propagated = [features]
    for _ in scales:
        propagated.append(adj @ propagated[-1])
    return np.mean(propagated, axis=0)  # Averaging multi-scale features

features = multi_scale_feature_propagation(features, adj_np)
data = torch.tensor(features, dtype=torch.float32).to(device)

# Model Definition
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_clusters):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dim, output_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.prototypes = nn.Parameter(torch.randn(n_clusters, output_dim))

    def forward(self, x):
        return self.projection_head(self.encoder(x))

# Loss Functions
def structural_contrastive_loss(embeddings, temperature=0.5):
    """ Compute contrastive loss for structural similarity. """
    u1, u2 = embeddings, embeddings
    pos_sim = torch.exp(torch.sum(u1 * u2, dim=1) / temperature)
    neg_sim = torch.exp(torch.mm(u1, u2.T) / temperature).sum(dim=1)
    return -torch.mean(torch.log(pos_sim / (neg_sim + 1e-8)))

def semantic_contrastive_loss(features, prototypes, labels, temperature=0.5):
    """ Compute contrastive loss for semantic similarity. """
    pos_sim = torch.exp(torch.sum(features * prototypes[labels], dim=1) / temperature)
    neg_sim = torch.exp(torch.mm(features, prototypes.T) / temperature).sum(dim=1)
    return -torch.mean(torch.log(pos_sim / (neg_sim + 1e-8)))

# DPMM-Based Prototype Inference
class DPMMPrototypeInference(nn.Module):
    def __init__(self, feature_dim, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.prototypes = None  # Initialized dynamically

    def update(self, new_prototypes):
        if self.prototypes is None:
            self.prototypes = new_prototypes.clone()
        else:
            self.prototypes.data = self.momentum * self.prototypes.data + (1 - self.momentum) * new_prototypes

    def inference(self, features):
        """ Perform clustering and update prototypes. """
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
        cluster_labels = kmeans.fit_predict(features.cpu().numpy())
        cluster_centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
        return cluster_centroids, torch.tensor(cluster_labels, dtype=torch.long, device=device)

# Training Setup
model = ContrastiveModel(
    input_dim=features.shape[1],
    hidden_dim=args.hidden_dim,
    output_dim=args.n_z,
    n_clusters=args.n_clusters
).to(device)

optimizer = Adam(model.parameters(), lr=args.lr)
prototype_manager = DPMMPrototypeInference(feature_dim=args.n_z)

# Training Loop
for epoch in range(args.epochs):
    optimizer.zero_grad()
    embeddings = model(data)

    # E-step: Cluster inference
    with torch.no_grad():
        cluster_centroids, cluster_labels = prototype_manager.inference(embeddings)
    
    # Structural contrastive loss
    struct_loss = structural_contrastive_loss(embeddings)

    # Semantic contrastive loss
    sem_loss = semantic_contrastive_loss(embeddings, cluster_centroids, cluster_labels)

    # Total loss
    loss = struct_loss + sem_loss
    loss.backward()
    optimizer.step()

    # M-step: Update prototypes
    prototype_manager.update(cluster_centroids)

    # Logging
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d} | Loss: {loss.item():.4f}')

        # Visualize clustering results every 10 epochs
        visualize_cluster(embeddings.detach().cpu().numpy(), cluster_labels.cpu().numpy())



take a look at the code below, is there anything you can improve 


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):

        h_1 = F.relu(self.fc_1(x))

        h_2 = self.fc_2(h_1)

        return h_2

class simple_encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(simple_encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features):
        output = torch.mm(features, self.weight)
        output = F.relu(output)
        return output

class S3CL_Model(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim):
        super(S3CL_Model, self).__init__()
        self.encoder = simple_encoder(in_dim, out_dim)
        self.encoder_momt = simple_encoder(in_dim, out_dim)
        self.projector = MLP(in_dim, hidden_dim, out_dim)
        self.projector_momt = MLP(in_dim, hidden_dim, out_dim)
    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update 
        """
        for param_ori, param_momt in zip(self.encoder.parameters(), self.encoder_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
        for param_ori, param_momt in zip(self.projector.parameters(), self.projector_momt.parameters()):
            param_momt.data = param_momt.data * self.m + param_ori.data * (1. - self.m)
    def forward(self, x):
        h = self.encoder(x)
        h_p = self.projector(h)
        with torch.no_grad():  
            self._momentum_update()  
            h_momt = self.encoder_momt(x)
            h_p_momt = self.projector_momt(h_momt)
        return h, h_p, h_p_momt
