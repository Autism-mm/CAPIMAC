import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from run import *
# 固定随机数种子，生成100个二维数据点
# torch.manual_seed(99)
def para(data,num_nodes,num_class):
    similarity_threshold = 0.4  # 相似度阈值
    num_anchors =  num_class*2# 锚点数量
    # num_anchors =26
    distances = cosineSimilartydis(data, data)
    # 排除对角线上的自身距离（0）的平均值
    mean_distance = distances[~torch.eye(distances.size(0), dtype=torch.bool)].mean()
    coverage_radius=mean_distance*0.3 # 贪心覆盖算法中的覆盖半径
    #到时候写一个对齐数据少于锚点数量error的提示
    if num_nodes < 100:  # 小图
        num_walks,walk_length = 20,3
    elif num_nodes < 1000:  # 中型图
        num_walks,walk_length = 10,5
    elif num_nodes < 10000:  # 大型图
        num_walks,walk_length = 5,10
    else:  # 超大图
        num_walks,walk_length = 3,20
    return num_walks,walk_length,similarity_threshold,num_anchors,coverage_radius




def cosineSimilarty(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    # A2 = A / (torch.norm(A, dim=0, p=2, keepdim=True) + 0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)
    # B2 = B / (torch.norm(B, dim=0, p=2, keepdim=True) + 0.000001)


    W=torch.mm(A,B.t())
    max_values,_ = torch.max(W, axis=0)
    min_values,_ = torch.min(W, axis=0)
    normalized_matrix = (W - min_values) / (max_values - min_values)
    normalized_matrix = torch.nan_to_num(normalized_matrix, nan=0.0001)
    return normalized_matrix

def cosineSimilartydis(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)

    W=torch.mm(A,B.t())
    max_values, _ = torch.max(W, axis=0)
    min_values, _ = torch.min(W, axis=0)
    normalized_matrix = (W - min_values) / (max_values - min_values)
    normalized_matrix = torch.nan_to_num(normalized_matrix, nan=0.0001)
    return 1-normalized_matrix
# # 随机游走参数
#
#
# Step 1: 初始化完全图的转移概率矩阵
# distances = torch.cdist(data, data, p=2)  # 计算所有点之间的欧几里得距离
# adj_matrix = torch.exp(-distances)  # 高斯权重：距离越小权重越高
# def visit(data):
#     adj_matrix=cosineSimilarty(data,data)
#     print(np.shape(adj_matrix))
#     adj_matrix.fill_diagonal_(0)  # 去掉自身连接
#     transition_matrix = adj_matrix / adj_matrix.sum(dim=1, keepdim=True)  # 归一化为转移概率
#     return transition_matrix


def visit(data, alpha=0.5):
    """
    根据给定的节点特征矩阵data和参数alpha计算转移矩阵。
    使用余弦相似度矩阵作为转移的相似度度量。
    计算公式：r_mu(x_i) = (x_i / mu_i) ^ -alpha
    """
    num_nodes = data.size(0)

    # 计算节点间的余弦相似度矩阵
    adj_matrix = cosineSimilarty(data, data)

    # 归一化每一行，确保每行相似度和为1
    adj_matrix.fill_diagonal_(0)  # 去掉自身连接
    adj_matrix = torch.nan_to_num(adj_matrix, nan=0.0001)  # 防止NaN值

    # 归一化为转移概率，确保每行的和为1
    row_sums = adj_matrix.sum(dim=1, keepdim=True) + 0.000001  # 防止除以零
    adj_matrix = adj_matrix / row_sums  # 归一化为转移概率

    # 防止出现概率为零的行（所有相似度为零时）
    adj_matrix = torch.nan_to_num(adj_matrix, nan=0.0001)  # 替换NaN为小值
    adj_matrix = torch.clamp(adj_matrix, min=0.0001)  # 防止小于0的概率值

    # 根据 alpha 修改相似度矩阵
    transition_matrix = adj_matrix ** (-alpha)  # 应用公式 r_mu(x_i) = (x_i / mu_i) ^ -alpha

    # 再次归一化转移矩阵，使得每行的和为1
    transition_matrix = transition_matrix / (transition_matrix.sum(dim=1, keepdim=True) + 0.000001)

    # 检查是否有行的和仍然为0，若有则设置为均匀分布
    zero_rows = (transition_matrix.sum(dim=1) == 0)
    if zero_rows.any():
        transition_matrix[zero_rows] = 1.0 / num_nodes  # 对于零行，设置均匀分布

    return transition_matrix


# 优化方法
def random_walk_batch_paths(transition_matrix, num_walks, walk_length):
    """
    批量化生成随机游走路径，并统计访问频次。
    """
    num_nodes = transition_matrix.size(0)
    visit_matrix = torch.zeros_like(transition_matrix,device='cuda')  # 初始化访问频率矩阵
    for start_node in range(num_nodes):  # 遍历每个起始节点
        # 初始化起点
        paths = torch.full((num_walks, walk_length + 1), start_node, dtype=torch.long,device='cuda')  # 每行一条路径
        for step in range(walk_length):  # 生成完整路径

            probs = transition_matrix[paths[:, step]]  # 当前步节点的转移概率
            next_nodes = torch.multinomial(probs, 1).squeeze()  # 采样下一个节点
            paths[:, step + 1] = next_nodes

        # 累计所有路径的访问频率
        for path in paths:
            visit_matrix[start_node].index_add_(0, path, torch.ones_like(path, dtype=torch.float,device='cuda'))
    visit_matrix -= torch.diag(torch.full((num_nodes,), num_walks, dtype=visit_matrix.dtype,device='cuda'))
    return visit_matrix


# visit_matrix = random_walk_batch_paths(transition_matrix, num_walks, walk_length)
#
# # visit_matrix = random_walk_parallel(transition_matrix, num_walks, walk_length)
#
# Step 3: 归一化访问频率为相似度,构建基于阈值的 kNN 图
def thresholded_knn(visit_matrix,similarity_threshold):
    similarity_matrix = visit_matrix / visit_matrix.max()
    thresholded_adj = (similarity_matrix > similarity_threshold).float()  # 保留相似度大于阈值的边
    return thresholded_adj
# # Step 5: 贪心覆盖算法选择锚点
def greedy_cover_with_importance(data, importance_scores, r, num_anchors):
    """
    贪心覆盖算法用于选择锚点
    :param data: 数据点，形状为 (n_samples, n_features)
    :param importance_scores: 每个点的重要性分数 (随机游走访问频率)
    :param r: 覆盖半径
    :param num_anchors: 需要选择的锚点数量
    :return: 锚点索引
    """
    distances = cosineSimilartydis(data,data)  # 计算点对点距离
    selected = []  # 选择的锚点索引
    covered = torch.zeros(data.size(0), dtype=torch.bool,device='cuda')  # 覆盖标志位
    sorted_indices = torch.argsort(importance_scores, descending=True)  # 按重要性排序
    cluster_selected = torch.zeros(data.size(0), dtype=torch.bool, device='cuda')  # 集群是否被选中锚点标记

    while len(selected) < num_anchors:
        # prev_covered_sum = covered.sum().item()  # 上一次覆盖点的数量

        for idx in sorted_indices:
            if len(selected) >= num_anchors:
                break
            if not covered[idx] and not cluster_selected[idx]:  # 如果当前点未被覆盖，且所属集群未选过锚点
                selected.append(idx)  # 选择锚点

                cluster_selected[idx] = 1  # 标记所属集群已选锚点
                # 将当前锚点覆盖范围内的点标记为已覆盖
                covered |= distances[idx] <= r
                covered[idx] = 1#调了半天，锚点自己没有被覆盖
        selected_anchors = set(selected)  # 当前已选择的锚点集合
        selected_anchors_tensor = torch.tensor(list(selected_anchors), device='cuda')
        # 检查是否所有集群都已被选过锚点
        if covered.sum().item() == data.size(0):
            print("所有点已被覆盖，重置覆盖状态")
            # 记录已选的锚点，重置覆盖标志
            covered[:] = 0
            covered[selected_anchors_tensor] = 1  # 恢复已选锚点的覆盖状态
            print(len(selected))
        # elif covered.sum().item() == prev_covered_sum:
        #     print("没有新的点被覆盖，终止选择锚点")
        #     break  # 如果没有新点被覆盖，跳出循环
    return torch.tensor(selected,device='cuda')

# 计算节点的重要性（访问频率的总和）
# node_importance = visit_matrix.sum(dim=1)
#
# # 使用贪心覆盖算法选择锚点
# anchor_indices = greedy_cover_with_importance(data, node_importance, coverage_radius, num_anchors)
# anchors = data[anchor_indices]  # 提取锚点

# # Step 6: 可视化结果
# # from sklearn.decomposition import PCA
# # import matplotlib.pyplot as plt
# #
# # # 假设 data 和 anchors 是 5维张量
# # pca = PCA(n_components=2, random_state=42)
# #
# # # 降维到 2D
# # data_2d = pca.fit_transform(data.detach().cpu().numpy())
# # anchors_2d = pca.transform(anchors.detach().cpu().numpy())
# #
# # # 绘制统一显示的散点图
# # plt.figure(figsize=(8, 8))
# # plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', label="Data Points", alpha=0.5, s=30)
# # plt.scatter(anchors_2d[:, 0], anchors_2d[:, 1], color="red", label="Anchor Points", s=100, edgecolor='black')
# # plt.title("Unified Visualization with PCA")
# # plt.legend()
# # plt.show()
#
#
#
# # 使用 UMAP 降维
# reducer = umap.UMAP(n_components=2)
# data_2d = reducer.fit_transform(data.detach().cpu().numpy())
# anchors_2d = reducer.transform(anchors.detach().cpu().numpy())
#
# # 绘制统一显示图
# plt.figure(figsize=(8, 8))
# plt.scatter(data_2d[:, 0], data_2d[:, 1], c='blue', label="Data Points", alpha=0.5, s=30)
# plt.scatter(anchors_2d[:, 0], anchors_2d[:, 1], color="red", label="Anchor Points", s=20, edgecolor='black')
# plt.title("Unified Visualization with UMAP")
# plt.legend()
# plt.show()
