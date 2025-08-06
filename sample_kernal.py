import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 高斯核函数
def gaussian_kernel(x, x_i, bandwidth):
    return np.exp(-0.5 * ((x - x_i) / bandwidth) ** 2)


# 核回归插值函数（支持多维）
def kernel_regression_multi_dim(x_known, y_known, x_targets, bandwidth):
    """
    x_known: 已知点的 x 坐标 (1D array)
    y_known: 已知点的 y 值，多维数组 (2D array, shape: [n_samples, n_features])
    x_target: 需要插值的 x 坐标 (scalar)
    bandwidth: 核函数的带宽参数
    """
    # 计算核权重
    y_targets = []  # 存储每个目标点的插值结果

    for x_target in x_targets:
        # 计算核权重
        weights = np.array([gaussian_kernel(x_target, x_i, bandwidth) for x_i in x_known])
        weights /= weights.sum()  # 权重归一化

        # 对每个维度分别插值
        y_target = np.sum(weights[:, np.newaxis] * y_known, axis=0)
        y_targets.append(y_target)

    return np.array(y_targets)


def insert_and_sort(x_known, y_known, x_targets, y_targets):
    # 合并数据
    # print(np.shape(y_known))
    # print(np.shape(y_targets))
    x_combined = np.concatenate((x_known, x_targets))
    y_combined = np.vstack((y_known, y_targets))

    # 按 x_combined 排序
    sorted_indices = np.argsort(x_combined)
    x_known_sorted = x_combined[sorted_indices]
    y_known_sorted = y_combined[sorted_indices]

    return x_known_sorted, y_known_sorted