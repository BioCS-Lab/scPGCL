import os
import torch
import warnings
from scprotein import *
from utils import *

warnings.filterwarnings('ignore')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# =====================
# 参数设置（可直接修改，无需 argparse）
# =====================
seed = 39788
threshold = 0.2
feature_preprocess = True
save_path = "./dataset_bag/SCoPE2_Specht.pt"

# =====================
# 固定随机种子
# =====================
setup_seed(seed)

# =====================
# 读取数据
# =====================
proteins_all, cell_list, features = load_sc_proteomic_features(stage1)
labels = load_cell_type_labels()

print("Proteins:", proteins_all)
print("Cells:", cell_list)
print("Features shape:", features.shape)

# =====================
# 生成图数据
# =====================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = graph_generations(features, threshold, feature_preprocess, labels).to(device)

# =====================
# 保存 .pt 文件
# =====================
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(data, save_path)
print(f"Graph data saved to {save_path}")