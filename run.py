import argparse
import statistics
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import register_data_args

from model import *  
from utils import *  
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score


def compute_reconstruction_loss(graph, model, features):
    """整图结构重建损失（MSE）。"""
    adj_matrix = graph.adjacency_matrix().to_dense()
    reconstructed_adj, _ = model(features)
    return F.mse_loss(reconstructed_adj, adj_matrix)


def compute_node_reconstruction_errors(graph, model, features):
    """逐节点结构残差（L1 行和），用于打分。"""
    adj_matrix = graph.adjacency_matrix().to_dense()
    reconstructed_adj, _ = model(features)
    node_errors = torch.sum(torch.abs(reconstructed_adj - adj_matrix), dim=1)
    return node_errors.detach().cpu().numpy()




def train_joint_and_score(graph, feats_ori, feats_hat, args):
    # ---- 超参 ----
    alpha = args.alpha           
    w_struct = args.w_struct    
    w_attr = args.w_attr         
    epochs = args.epochs

    in_dim = feats_ori.shape[1]


    struct_net = GlobalAutoencoder(graph, in_dim, args.out_dim, nn.PReLU())
   
    ad_net = AttributeDegreeAutoencoder(graph, in_dim, args.out_dim)

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        graph = graph.to(device)
        struct_net = struct_net.to(device)
        ad_net = ad_net.to(device)
        feats_ori = feats_ori.cuda()
        feats_hat = feats_hat.cuda()

    
    optim = torch.optim.Adam(list(struct_net.parameters()) + list(ad_net.parameters()), lr=args.lr)

    best = float('inf')
    for ep in range(epochs):
        struct_net.train(); ad_net.train()
        optim.zero_grad()

        # --- 结构 ---
        loss_struct_ori = compute_reconstruction_loss(graph, struct_net, feats_ori)
        loss_struct_hat = compute_reconstruction_loss(graph, struct_net, feats_hat)

        # --- 属性 / 度
        out_ori = ad_net(feats_ori)
        out_hat = ad_net(feats_hat)
        loss_attr_ori = out_ori['feat_loss']
        loss_attr_hat = out_hat['feat_loss']
        loss_deg_ori = out_ori['deg_loss']
        loss_deg_hat = out_hat['deg_loss']

        # ---- 组合损失（无跨视图约束）----
        view_hat =(1-w_attr) *(w_struct * loss_struct_hat + (1 - w_struct) * loss_deg_hat) + w_attr * loss_attr_hat
        view_ori = (1-w_attr)*(w_struct * loss_struct_ori + (1 - w_struct) * loss_deg_ori) + w_attr * loss_attr_ori
        loss = alpha * view_hat + (1 - alpha) * view_ori

        loss.backward()
        optim.step()

        if loss.item() < best:
            best = loss.item()

        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0:
            print(f"[JOINT] Epoch {ep+1}/{epochs} | total={loss.item():.6f} | "
                  f"S(ori)={loss_struct_ori.item():.4f}/S(hat)={loss_struct_hat.item():.4f} | "
                  f"D(ori)={loss_deg_ori.item():.4f}/D(hat)={loss_deg_hat.item():.4f} | "
                  f"A(ori)={loss_attr_ori.item():.4f}/A(hat)={loss_attr_hat.item():.4f}")

    # ===== 训练完计算逐节点打分（同一公式） =====
    struct_net.eval(); ad_net.eval()
    with torch.no_grad():
        # 结构：逐节点误差
        s_ori = compute_node_reconstruction_errors(graph, struct_net, feats_ori)
        s_hat = compute_node_reconstruction_errors(graph, struct_net, feats_hat)
        # 属性/度：逐节点误差
        o_ori = ad_net(feats_ori)
        o_hat = ad_net(feats_hat)
        a_ori = o_ori['feat_err_node'].squeeze(1).cpu().numpy()
        a_hat = o_hat['feat_err_node'].squeeze(1).cpu().numpy()
        d_ori = o_ori['deg_err_node'].squeeze(1).cpu().numpy()
        d_hat = o_hat['deg_err_node'].squeeze(1).cpu().numpy()
    # 归一化到可比尺度
    S_ori = normalize_array(s_ori); S_hat = normalize_array(s_hat)
    D_ori = normalize_array(d_ori); D_hat = normalize_array(d_hat)
    A_ori = normalize_array(a_ori); A_hat = normalize_array(a_hat)


    # 同式组合得到最终分数
    score = alpha * ((1-w_attr) *(w_struct * S_hat + (1 - w_struct) * D_hat) + w_attr * A_hat) \
    + (1 - alpha) * ((1-w_attr) *(w_struct * S_ori + (1 - w_struct) * D_ori) + w_attr * A_ori)
    # score = alpha * (w_struct * S_hat  + w_attr * A_hat) \
    # + (1 - alpha) * (w_struct * S_ori  + w_attr * A_ori)

    parts = {
    'S_ori': S_ori, 'S_hat': S_hat,
    'D_ori': D_ori, 'D_hat': D_hat,
    'A_ori': A_ori, 'A_hat': A_hat,
    }
    return score, parts




# =============================
# 主流程
# =============================


def main(args):
    seed_everything(args.seed)


    # 数据
    if args.data in ["Cora", "Citeseer", "ACM"]:
        graph = my_load_data(args.data)
    else:
        graph = my_load_data_bond(args.data)


    feats = graph.ndata['feat']
    labels = graph.ndata['label']


    # 标准化 + 热核扩散
    standardized_feats = feats if args.data in ["reddit", "disney"] else (feats - feats.mean()) / (feats.std() + 1e-12)
    diffused_features = heat_diffusion(graph, standardized_feats, args.lamd, args.t)


    if args.gpu >= 0:
        graph = graph.to(args.gpu)


    # 联合训练 & 打分
    scores, parts = train_joint_and_score(graph, standardized_feats, diffused_features, args)


    # 评估
    labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    auc = roc_auc_score(labels_np, scores)


    sorted_idx = np.argsort(scores)
    k = int(sum(labels_np))
    topk_idx = sorted_idx[-k:]
    pred_labels = np.zeros_like(labels_np)
    pred_labels[topk_idx] = 1


    rec_k = recall_score(np.ones(k), labels_np[topk_idx])
    ap = average_precision_score(labels_np, scores)


    print("AUC={:.4f} | Recall@k={:.4f} | AP={:.4f}".format(auc, rec_k, ap))
    return auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint training (structure + degree + attribute) with two views')
    register_data_args(parser)


    # 数据/扩散
    parser.add_argument('--data', type=str, default='books')
    parser.add_argument('--lamd', type=float, default=0.7)
    parser.add_argument('--t', type=int, default=4)


    # 联合损失权重
    parser.add_argument('--alpha', type=float, default=0.3, help='增强视图权重 α')
    parser.add_argument('--w-struct', dest='w_struct', type=float, default=0.6, help='结构权重 w_s')
    parser.add_argument('--w-attr', dest='w_attr', type=float, default=0.2, help='属性权重 w_attr')


    # 模型/训练
    parser.add_argument('--out-dim', dest='out_dim', type=int, default=128)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)


    # 随机种子
    parser.add_argument('--seed', type=int, default=2072)


    args = parser.parse_args()
    print(args)


    args.seed =246085
    result = main(args)

    print("final auc:{:.4f}".format(result))
    
