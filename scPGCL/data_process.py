import torch
import numpy as np

def load_data(path, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    加载 .pt 图数据，并生成 train/val/test mask
    """
    # PyTorch 2.6+ 安全加载
    data = torch.load(path, weights_only=False)

    # 没有标签的情况
    if getattr(data, "y", None) is None:
        print("⚠️ data.y 为空，本次不生成 mask")
        return data

    # 处理标签
    if isinstance(data.y, list):
        if isinstance(data.y[0], str):  # 字符串类别
            labels = sorted(set(data.y))  # ✅ 保证顺序一致
            label_map = {k: i for i, k in enumerate(labels)}
            data.y = torch.tensor([label_map[k] for k in data.y], dtype=torch.long)
        else:  # 数字 list
            data.y = torch.tensor(data.y, dtype=torch.long)

    elif isinstance(data.y, np.ndarray):  # 🔥 新增 numpy.ndarray 处理
        data.y = torch.from_numpy(data.y).long()

    elif isinstance(data.y, torch.Tensor):
        data.y = data.y.long()

    else:
        raise ValueError(f"❌ data.y 类型无法识别: {type(data.y)}")

    # 生成 mask
    num_nodes = data.y.shape[0]
    np.random.seed(seed)
    torch.manual_seed(seed)
    idx = np.random.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train+n_val]] = True
    test_mask[idx[n_train+n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    print(f"已生成 train/val/test mask")
    print(f"train: {train_mask.sum().item()}, val: {val_mask.sum().item()}, test: {test_mask.sum().item()}")

    # 覆盖保存
    torch.save(data, path)
    print(f"✅ 已清洗并保存到 {path}")

    return data


if __name__ == '__main__':
    dataset_path = "dataset_bag/4_pSCoPE_Leduc&plexDIA_0.11.pt"
    data = load_data(dataset_path)
    print(data)
    print("y:", data.y[:10])  # 打印前10个标签
