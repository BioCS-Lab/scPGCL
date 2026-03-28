import matplotlib
matplotlib.use("TkAgg")
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
from scprotein import *
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import warnings


warnings.filterwarnings("ignore")
seed = 1

def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 


def dimension_reduce(embedding):
    X_trans_PCA = PCA(n_components=50, random_state=seed).fit_transform(embedding)  
    X_trans = TSNE(n_components=2,random_state=seed).fit_transform(X_trans_PCA)
    return X_trans



# load ground truth cell label
Y_cell_type_label = load_cell_type_labels()
label_dict = {'sc_m0':1, 'sc_u':0}
target_names = ['Macrophage','Monocyte']
Y_label = np.array(itemgetter(*list(Y_cell_type_label))(label_dict))



# load learned cell embedding
X_fea = np.load(r'C:\Users\Administrator\PycharmProjects\pythonProject\scPGCL-main\参数分析\dfr=0.1\der=0.25\SCoPE2_Specht_embedding_final.npy')
print(X_fea.shape)


k_means = KMeans(n_clusters=len(target_names))
y_predict = k_means.fit_predict(X_fea)
df_result = pd.DataFrame()
df_result['ARI'] = [np.round(adjusted_rand_score(Y_label,y_predict),3)]
df_result['ASW'] = [np.round(silhouette_score(X_fea,y_predict),3)]
df_result['NMI'] = [np.round(normalized_mutual_info_score(Y_label,y_predict),3)]
df_result['PS'] = [np.round(purity_score(Y_label,y_predict),3)]
print(df_result)


X_trans_learned = dimension_reduce(X_fea)



# # plot —— 仅调整显示风格，不改变原逻辑
# fig = plt.figure(figsize=(6, 6))
#
# # —— 色彩保持简单但与前图风格一致（Set3 / tab20 体系） ——
# colors = [plt.cm.Set3(4), plt.cm.Set3(10)]
#
# # —— 点形状根据 cell type 统一 ——
# markers = ["o", "^"]   # 两类：第1类 = o，第2类 = ^
#
# for i in range(len(target_names)):
#     plt.scatter(
#         X_trans_learned[Y_label == i, 0],
#         X_trans_learned[Y_label == i, 1],
#         s=10,
#         color=colors[i],
#         marker=markers[i],
#         label=target_names[i]
#     )
#
# # —— 坐标格式统一（与第一个图一致） ——
# plt.xlabel('tsne 1')
# plt.ylabel('tsne 2')
# plt.xticks([])
# plt.yticks([])
#
# # —— 图例放在图内部右上角，不遮挡主要区域 ——
# plt.legend(
#     loc='upper right',
#     frameon=True,
#     facecolor='white',
#     fontsize=10,
#     borderpad=0.4,
#     bbox_to_anchor=(0.98, 0.98)   # 图内右上角偏左一点
# )
#
# plt.tight_layout()
#
# save_dir = r"C:\Users\Administrator\Desktop"
#
#
# plt.savefig(fr"{save_dir}\SCoPE2_Specht.pdf", bbox_inches='tight')
#
# plt.show()