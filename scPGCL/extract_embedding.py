# extract_embedding.py
import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from model import Encoder, Model          # 与训练时一致
from data_process import load_data        # 读取自定义 .pt 数据


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SCoPE2_Specht',
                        help='数据集名（不带扩展名），会读取 dataset_bag/<name>.pt')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--dataset_dir', type=str, default='dataset_bag',
                        help='自定义 .pt 所在目录（相对项目根目录）')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='训练好的权重所在目录（相对项目根目录）')
    parser.add_argument('--output_dir', type=str, default='embedding',
                        help='embedding 输出目录（相对项目根目录）')
    # ===== 下面这些超参必须与训练一致 =====
    parser.add_argument('--mode', type=int, default=4)
    parser.add_argument('--num_hidden', type=int, default=512)
    parser.add_argument('--num_proj_hidden', type=int, default=512)
    parser.add_argument('--base_model', type=str, default='GCNConv',
                        help='GCNConv / SAGEConv / GATConv（需与训练一致）')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--cutway', type=int, default=2)
    parser.add_argument('--cutrate', type=float, default=1.0)
    args = parser.parse_args()

    # ===== 设备 =====
    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===== 路径 =====
    base_dir = osp.dirname(osp.abspath(__file__))
    dataset_path = osp.join(base_dir, args.dataset_dir, f'{args.dataset}.pt')
    model_path   = osp.join(base_dir, args.model_dir,   f'{args.dataset}.pkl')
    out_dir      = osp.join(base_dir, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_npy_path = osp.join(out_dir, f'{args.dataset}_embedding.npy')

    print(f'加载数据: {dataset_path}')
    if not osp.exists(dataset_path):
        raise FileNotFoundError(f'未找到数据文件: {dataset_path}')

    # ===== 读取图数据 =====
    data = load_data(dataset_path)
    data = data.to(device)
    print(f'Data: x={tuple(data.x.shape)}, edge_index={tuple(data.edge_index.shape)}')

    # ===== 重建模型结构（需与训练一致）=====
    BASE_MODELS = {
        'GCNConv': GCNConv,
        'SAGEConv': SAGEConv,
        'GATConv': GATConv,
    }
    if args.base_model not in BASE_MODELS:
        raise ValueError(f'不支持的 base_model {args.base_model}, 请选择 {list(BASE_MODELS.keys())}')
    base_model_cls = BASE_MODELS[args.base_model]

    num_features = data.x.shape[1]
    activation = F.relu
    encoder = Encoder(
        num_features,
        args.num_hidden,
        activation,
        args.mode,
        base_model=base_model_cls,
        k=args.num_layers,
        cutway=args.cutway,
        cutrate=args.cutrate,
        tau=args.tau
    ).to(device)

    model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.mode, args.tau).to(device)

    # ===== 加载权重 =====
    print(f'加载模型参数: {model_path}')
    if not osp.exists(model_path):
        raise FileNotFoundError(f'未找到模型权重: {model_path}')
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    # ===== 抽取 embedding =====
    with torch.no_grad():
        z = model(data.x, data.edge_index, 1, None, None)  # [N, num_hidden]
    print(f'Embedding 形状: {tuple(z.shape)}')

    # ===== 只保存 .npy =====
    np.save(out_npy_path, z.cpu().numpy())
    print(f'Embedding (.npy) 保存到: {out_npy_path}')


if __name__ == '__main__':
    main()
