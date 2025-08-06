from SRW_KNN_greedy import *
import torch
def get_anchors(h0,h1,map_pairs,num_unique_labels):
    #又一个点，提前预训练，统一样本的潜在空间
    # print(h0.shape[0],num_unique_labels,'ghjhggjf')
    #初始化随机游走参数
    num_walks0, walk_length0, similarity_threshold0, num_anchors0, coverage_radius0 = para(h0,h0.shape[0],num_unique_labels)
    num_walks1, walk_length1, similarity_threshold1, num_anchors1, coverage_radius1 = para(h1, h1.shape[0],num_unique_labels)
    transition_matrix0,transition_matrix1 = visit(h0),visit(h1)#转移概率矩阵
    #访问矩阵
    visit_matrix0,visit_matrix1 = random_walk_batch_paths(transition_matrix0, num_walks0, walk_length0), random_walk_batch_paths(transition_matrix1, num_walks1, walk_length1)
    #
    node_importance0, node_importance1 = visit_matrix0.sum(dim=0),visit_matrix1.sum(dim=0)
    # # 使用贪心覆盖算法选择锚点
    anchor_indices0 = greedy_cover_with_importance(h0, node_importance0, coverage_radius0, num_anchors0)
    anchor_indices1 = greedy_cover_with_importance(h1, node_importance1, coverage_radius1, num_anchors1)
    combined_indices = torch.cat((anchor_indices0, anchor_indices1))
    unique_indices = torch.unique(combined_indices)#合并索引去重
    len_indices=len(unique_indices)
    mapdata0,mapdata1=torch.tensor(map_pairs[0]),torch.tensor(map_pairs[1])
    anchors0,anchors1 = mapdata0[unique_indices].float(),mapdata1[unique_indices].float()# 提取锚点(降维前）
    return anchors0,anchors1,len_indices