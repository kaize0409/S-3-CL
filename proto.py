from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import  load_graph, load_data, load_graph_np, propagate, evaluation
from evaluation import eva
import scipy.sparse as sp
from numpy import linalg as LA
import process
from models import S3CL_Model
from dpmm import DPMM

def get_proto_loss(feature, centroid, label_momt, proto_norm):
    
    feature_norm = torch.norm(feature, dim=-1)
    feature = torch.div(feature, feature_norm.unsqueeze(1))
    
    centroid_norm = torch.norm(centroid, dim=-1)
    centroid = torch.div(centroid, centroid_norm.unsqueeze(1))
    
    sim_zc = torch.matmul(feature, centroid.t())
    
    sim_zc_normalized = torch.div(sim_zc, proto_norm)
    sim_zc_normalized = torch.exp(sim_zc_normalized)
    sim_2centroid = torch.gather(sim_zc_normalized, -1, label_momt) 
    sim_sum = torch.sum(sim_zc_normalized, -1, keepdim=True)
    sim_2centroid = torch.div(sim_2centroid, sim_sum)
    loss = torch.mean(sim_2centroid.log())
    loss = -1 * loss
    return loss

def get_proto_norm(feature, centroid, labels):
    num_data = feature.shape[0]
    each_cluster_num = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        each_cluster_num[i] = np.sum(labels==i)
    proto_norm_term = np.zeros([args.n_clusters])
    for i in range(args.n_clusters):
        norm_sum = 0
        for j in range(num_data):
            if labels[j] == i:
                norm_sum = norm_sum + LA.norm(feature[j] - centroid[i], 2)
        proto_norm_term[i] = norm_sum / (each_cluster_num[i] * np.log2(each_cluster_num[i] + 10))

    proto_norm_momt = torch.Tensor(proto_norm_term)
    return proto_norm_momt

def train(dataset):
    model = S3CL_Model(args.n_input, 256, 512).to(device)

    model.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

    optimizer = Adam(model.parameters(), lr=args.lr)

    adj, features, labels, idx_train, idx_val, idx_test = process.load_data(dataset)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    _, adj_np = load_graph_np(args.name, args.k)
    diff = np.load('data/diff_{}_{}.npy'.format(dataset, 0.05), allow_pickle=True)


    features, _ = process.preprocess_features(features)
    features = torch.FloatTensor(features[np.newaxis])
    labels = torch.FloatTensor(labels[np.newaxis])

    norm_adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
    norm_diff = sp.csr_matrix(diff)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, z_momt = model.gae(data, adj)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z_momt.data.cpu().numpy())
    eva(y, y_pred, 'pae')

    init_labels = kmeans.labels_
    label_momt = torch.Tensor(init_labels).unsqueeze(1)
    label_momt = label_momt.to(torch.int64)
    ori_center = kmeans.cluster_centers_
    centroid_momt = torch.Tensor(ori_center)

    label_kmeans_ori = kmeans.labels_[:, np.newaxis]
    
    with torch.no_grad():
        h, out, out_momt = model(data, adj)

    DP_model = DPMM(out_momt)
    estimated_K, ps_labels = DP_model.fit(out)
    args.n_clusters = estimated_K
    label_momt = torch.Tensor(ps_labels).unsqueeze(1) 
    centroid_momt = np.dot(ps_labels.T, out_momt) / np.sum(ps_labels.T, axis = 1)[:, np.newaxis]

    label_propagated = propagate(adj_np, 0.1, label_kmeans_ori, args.n_clusters, 10)

    centers_propagated = np.dot(label_propagated.T, z_momt) / np.sum(label_propagated.T, axis = 1)[:, np.newaxis]

    label_propagated_hard = np.argmax(label_propagated, axis=1)
    label_propagated_hard = label_propagated_hard[:, np.newaxis]

    label_momt = torch.Tensor(label_propagated_hard)
    label_momt = label_momt.to(torch.int64)

    proto_norm_momt = get_proto_norm(z_momt, ori_center, label_kmeans_ori)

    _, _, _, idx_train, _, idx_test = process.load_data('citeseer')

    best_acc_clf = 0

    for epoch in range(40):    
        h, out, out_momt = model(data, adj)

        proto_loss = get_proto_loss(out, centroid_momt, label_momt, proto_norm_momt)

        loss =  proto_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        with torch.no_grad():
            h, out, out_momt = model(data, adj)

            classification_acc = evaluation(y, adj, data, model, idx_train, idx_test)

            print('gnn classification accuracy:' + str(classification_acc))

            if classification_acc > best_acc_clf:
                best_acc_clf = classification_acc
            
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
            y_pred = kmeans.fit_predict(out_momt.data.cpu().numpy())
            # ps_labels = kmeans.labels_[:, np.newaxis]
            DP_model = DPMM(out_momt)
            estimated_K, ps_labels = DP_model.fit(out_momt)
            label_propagated = propagate(adj_np, 0.1, ps_labels, args.n_clusters, 10)
            centers_propagated = np.dot(label_propagated.T, out_momt) / np.sum(label_propagated.T, axis = 1)[:, np.newaxis]
            label_propagated_hard = np.argmax(label_propagated, axis=1)
            label_propagated_hard = label_propagated_hard[:, np.newaxis]
            label_momt = torch.Tensor(label_propagated_hard)
            label_momt = label_momt.to(torch.int64)
            centroid_momt = torch.Tensor(centers_propagated)
            proto_norm_momt = get_proto_norm(out_momt, ori_center, label_kmeans_ori)

    print('Best gnn classification accuracy: ' + str(best_acc_clf))

