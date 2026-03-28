import argparse
import os
import os.path as osp
import sys
from time import perf_counter as t

import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.utils import to_scipy_sparse_matrix, dropout_adj
from torch_geometric.datasets import Planetoid, CitationFull, Amazon, Coauthor
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, InMemoryDataset

from model import Encoder, Model, drop_feature
from eval import evaluation


# ---------------- Training ----------------
def train(model: Model, x, edge_index, optimizer, drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2):
    model.train()
    optimizer.zero_grad()

    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]

    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)

    pre_z1 = model(x_1, edge_index_1, 1, None, None)
    pre_z2 = model(x_2, edge_index_2, 2, None, None)

    z1 = model(x_1, edge_index_1, 1, pre_z1, pre_z2)
    z2 = model(x_2, edge_index_2, 2, pre_z1, pre_z2)

    loss = model.loss(z1, z2, edge_index_1, edge_index_2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


# ---------------- Testing ----------------
def test(model: Model, x, edge_index, y, name, device, data, learning_rate2, weight_decay2, final=False):
    model.eval()
    z = model(x, edge_index, 1, None, None)
    return evaluation(z, y, name, device, data, learning_rate2, weight_decay2)


# ---------------- Dataset Loader ----------------
def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP', 'computers', 'photo', 'CS', 'Physics']
    name = 'dblp' if name == 'DBLP' else name

    if name in ['CS', 'Physics']:
        return Coauthor(path, name, transform=T.NormalizeFeatures())
    if name in ['computers', 'photo']:
        return Amazon(path, name, transform=T.NormalizeFeatures())
    return (CitationFull if name == 'dblp' else Planetoid)(path, name, transform=T.NormalizeFeatures())


# ---------------- Main ----------------
# 1_nanoPOTS&N2
# 2_SCoPE2_Leduc&plexDIA
# 3_pSCoPE_Huffman_plexDIA
# 4_pSCoPE_Leduc&plexDIA
# 5_SCoPE2_Leduc&pSCoPE_Leduc
# SCoPE2_Specht

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='3_pSCoPE_Huffman_plexDIA',
                        help='dataset name OR custom graph filename (without .pt/.pth)')
    parser.add_argument('--dataset_dir', type=str, default='dataset_bag',
                        help='relative to script dir; where custom .pt/.pth graph files live')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--mode', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr2', type=float, default=1e-3)
    parser.add_argument('--tau', type=float, default=0.5)
    # =======================================================
    parser.add_argument('--dfr1', type=float, default=0.4)
    parser.add_argument('--dfr2', type=float, default=0.4)
    parser.add_argument('--der1', type=float, default=0.1)
    parser.add_argument('--der2', type=float, default=0.1)

    parser.add_argument('--lv', type=int, default=1)
    parser.add_argument('--cutway', type=int, default=2)
    parser.add_argument('--cutrate', type=float, default=1.0)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--wd2', type=float, default=1e-4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--num_epochs', type=int, default=500)
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    learning_rate = args.lr
    learning_rate2 = args.lr2   # ✅ 修复
    drop_edge_rate_1, drop_edge_rate_2 = args.der1, args.der2
    drop_feature_rate_1, drop_feature_rate_2 = args.dfr1, args.dfr2
    tau, mode, nei_lv = args.tau, args.mode, args.lv
    cutway, cutrate = args.cutway, args.cutrate
    num_hidden, num_proj_hidden, num_layers = args.num_hidden, args.num_proj_hidden, args.num_layers
    num_epochs = args.num_epochs
    weight_decay, weight_decay2 = args.wd, args.wd2
    activation, base_model = F.relu, GCNConv

    # ---------- paths ----------
    cur_path = osp.abspath(__file__)
    cur_dir = osp.dirname(cur_path)
    path_default = osp.join(osp.expanduser('~'), 'datasets', args.dataset)

    # ---------- dataset loading ----------
    custom_file = None
    for ext in ['.pt', '.pth']:
        cand = osp.join(cur_dir, args.dataset_dir, args.dataset + ext)
        if osp.exists(cand):
            custom_file = cand
            break

    dataset, data = None, None
    if custom_file:
        print(f'发现图数据: {custom_file}  — 尝试加载...')
        loaded = torch.load(custom_file, weights_only=False)
        if isinstance(loaded, Data):
            data = loaded
        elif isinstance(loaded, dict) and 'data' in loaded:
            data = loaded['data']
        elif isinstance(loaded, (list, tuple)) and len(loaded) > 0 and hasattr(loaded[0], 'x'):
            data = loaded[0]
        elif isinstance(loaded, InMemoryDataset):
            dataset = loaded
            try:
                data = dataset[0]
            except Exception:
                data = None
        elif hasattr(loaded, 'x') and hasattr(loaded, 'edge_index'):
            data = loaded
        else:
            raise RuntimeError(f'无法识别 {custom_file} 的内容类型: {type(loaded)}')
        print('Custom data loaded. data type:', type(data))
    else:
        print('未找到自定义文件，使用内置数据集加载:', args.dataset)
        dataset = get_dataset(path_default, args.dataset)
        data = dataset[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if data is None:
        raise RuntimeError('未能加载到图数据 (data is None)')
    data = data.to(device)

    # ---------- feature dim ----------
    if dataset is not None and hasattr(dataset, 'num_features'):
        in_dim = dataset.num_features
    elif hasattr(data, 'x') and data.x is not None:
        in_dim = int(data.x.shape[1])
    else:
        raise RuntimeError('无法确定输入特征维度')

    # ---------- model ----------
    encoder = Encoder(in_dim, num_hidden, activation, mode,
                      base_model=base_model, k=num_layers, cutway=cutway, cutrate=cutrate, tau=tau).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, mode, tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # ---------- higher-order neighbors ----------
    edge_idx_for_scipy = data.edge_index.cpu() if data.edge_index.is_cuda else data.edge_index
    num_nodes = int(data.x.shape[0])
    coo1 = to_scipy_sparse_matrix(edge_idx_for_scipy, num_nodes=num_nodes).tocsr()
    coo_lv, tmp = coo1.copy(), coo1.copy()
    for _ in range(2, nei_lv + 1):
        tmp = tmp @ coo1
        coo_lv += tmp
    coo1 = (coo_lv > 0).astype(int).tocoo()
    indices = np.vstack((coo1.row.astype(np.int64), coo1.col.astype(np.int64)))
    edge_index = torch.LongTensor(indices).to(device)

    # ---------- save path ----------
    model_save_path = osp.join(cur_dir, 'model', args.dataset + '.pth')
    embedding_save_path = osp.join(cur_dir, 'embedding', f"{args.dataset}_embedding_final.npy")
    os.makedirs(osp.dirname(model_save_path), exist_ok=True)
    os.makedirs(osp.dirname(embedding_save_path), exist_ok=True)

    # ---------- training ----------
    start, prev = t(), t()
    if not args.test:
        for epoch in range(1, num_epochs + 1):
            loss = train(model, data.x, edge_index, optimizer,
                         drop_edge_rate_1, drop_edge_rate_2,
                         drop_feature_rate_1, drop_feature_rate_2)
            now = t()
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
            prev = now
            if epoch % 10 == 0:
                print("=== Test ===")
                _ = test(model, data.x, data.edge_index, data.y, args.dataset,
                         device, data, learning_rate2, weight_decay2, final=True)

        # ---------- save model ----------
        torch.save({
            'epoch': num_epochs,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, model_save_path)
        print('模型保存至:', model_save_path)

        # ---------- save embedding ----------
        model.eval()
        with torch.no_grad():
            z = model(data.x, data.edge_index, 1, None, None).cpu().numpy()
        np.save(embedding_save_path, z)
        print('embedding 保存至:', embedding_save_path)

    else:
        if osp.exists(model_save_path):
            checkpoint = torch.load(model_save_path)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f'已加载模型 (epoch={checkpoint["epoch"]})')
        else:
            print('未找到模型文件:', model_save_path)
            sys.exit(0)

    # ---------- final eval ----------
    print("=== Final Evaluation ===")
    accs = torch.tensor([test(model, data.x, data.edge_index, data.y,
                              args.dataset, device, data,
                              learning_rate2, weight_decay2, final=True)
                         for _ in range(10)])
    fin_acc, fin_std = accs.mean().item(), accs.std().item()
    print(f'Final Accuracy: {fin_acc:.4f} ± {fin_std:.4f}')
