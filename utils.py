import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def evaluation(y, adj, data, model, idx_train, idx_test):
    clf = LogisticRegression(random_state=0, max_iter=2000)
    with torch.no_grad():
        out = model.encoder(data, adj)
        train_embs = out[idx_train, :] 
        test_embs = out[idx_test, :]
        train_labels = torch.Tensor(y[idx_train])
        test_labels = torch.Tensor(y[idx_test])
    clf.fit(train_embs, train_labels)
    pred_test_labels = clf.predict(test_embs)
    return accuracy_score(test_labels, pred_test_labels)

def propagate(adj, alpha, label, class_num, iter_num):
    if label.shape[1] == 1:
        dense_label = np.zeros([label.shape[0], class_num])
        for i in range(label.shape[0]):
            dense_label[i, label[i, 0]] = 1
    else:
        dense_label = label
    
    H = dense_label
    Z = dense_label
    for i in range(iter_num):
        Z = (1 - alpha) * adj * Z + alpha * H
    Z = softmax(Z, axis=1)
    
    return Z
    
def visualize_cluster(encoder_out, y_pred):
    pca_before = PCA(n_components=2)
    pca_before.fit(encoder_out)
    pca_before_transform = pca_before.fit_transform(encoder_out)
    plt.scatter(pca_before_transform[:, 0], pca_before_transform[:, 1], c=y_pred, s=10)

def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj_dense = torch.tensor(adj.todense(), dtype=torch.float32)
    adj_sparse = sparse_mx_to_torch_sparse_tensor(adj)

    return adj_dense, adj_sparse

def normalize_aug_adj(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj_dense = torch.tensor(adj.todense(), dtype=torch.float32)
    adj_sparse = sparse_mx_to_torch_sparse_tensor(adj)
    return adj_sparse

def load_graph_np(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)


    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj_dense = adj.todense()
    adj_sparse = adj

    return adj_dense, adj_sparse


def load_graph_ori(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # adj = adj + sp.eye(adj.shape[0])


    return adj


class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


def load_data_from_mat(dataset):
    data = sio.loadmat("data/{}.mat".format(dataset))
    features = data["Attributes"]
    adj = data["Network"]
    labels = data['Label']

    labels = [label[0] for label in labels]
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(np.array(features.todense()))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return features, adj, labels


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def _rank2_trace(x):
    return torch.einsum('ii', x)


def _rank2_diag(x):
    eye = torch.eye(x.size(0)).type_as(x)
    out = eye * x.unsqueeze(1).expand(*x.size(), x.size(0))
    return out