import argparse
import random
import sys
from scprotein import *
import numpy     as np
import scipy.sparse as sp
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score
from utils import *


# 数据集构建
def integrate_sc_proteomic_features(dataset1, dataset2):
    # load individual scp data
    adata1 = sc.read_h5ad('./raw_data/{}.h5ad'.format(dataset1))
    adata2 = sc.read_h5ad('./raw_data/{}.h5ad'.format(dataset2))
    protein_data1, protein_data2 = adata1.X, adata2.X
    protein_data1 = np.nan_to_num(protein_data1)
    protein_data2 = np.nan_to_num(protein_data2)
    cell_num1, cell_num2 = protein_data1.shape[0], protein_data2.shape[0]
    proteins1, proteins2 = list(adata1.var_names), list(adata2.var_names)

    # define batch label and cell type labels for both two datasets
    batch_label = np.concatenate((np.zeros(cell_num1), np.ones(cell_num2))).astype(int)
    cell_type1, cell_type2 = list(adata1.obs['cell_type']), list(adata2.obs['cell_type'])
    cell_type_with_dataname = cell_type1 + cell_type2

    cell_type1 = [i.split('(')[0] for i in cell_type1]
    cell_type2 = [i.split('(')[0] for i in cell_type2]
    overlap_cell_type = list(set(cell_type1) & set(cell_type2))
    print('overlap celltype:', overlap_cell_type)

    cell_type_all = cell_type1 + cell_type2
    cell_type_dic = dict(zip(set(cell_type_all), range(len(set(cell_type_all)))))
    cell_type_label = np.array(itemgetter(*list(cell_type_all))(cell_type_dic))
    overlap_cell_type_label = [cell_type_dic[i] for i in overlap_cell_type]

    # search overlap protein from both two datasets
    proteins1_pd, proteins2_pd = pd.DataFrame(proteins1, columns=['protein_name']), pd.DataFrame(proteins2, columns=[
        'protein_name'])
    overlap_protein = pd.merge(proteins1_pd, proteins2_pd, on=['protein_name'])
    overlap_protein = list(overlap_protein['protein_name'])
    print('overlap protein nums:', len(overlap_protein))

    # construct overlap protein features
    features_concat = np.zeros((cell_num1 + cell_num2, len(overlap_protein)))
    for i, protein in enumerate(overlap_protein):
        index1 = proteins1.index(protein)
        protein_data1_slice = protein_data1[:, index1]
        index2 = proteins2.index(protein)
        protein_data2_slice = protein_data2[:, index2]
        protein_data_slice_concat = np.concatenate([protein_data1_slice, protein_data2_slice])
        features_concat[:, i] = protein_data_slice_concat

    return batch_label, cell_type_with_dataname, cell_type_label, overlap_cell_type_label, features_concat


batch_label,cell_type_with_dataname,cell_type_label,overlap_cell_type_label, features = integrate_sc_proteomic_features('nanoPOTS','N2')