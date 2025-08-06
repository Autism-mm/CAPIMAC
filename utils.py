import numpy as np
import random
from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint
import math
import torch
def TT_split(n_all, test_prop, seed):
    '''
    split data into training, testing dataset
    '''
    random.seed(seed)
    random_idx = random.sample(range(n_all), n_all)
    train_num = np.ceil((1-test_prop) * n_all).astype(int)
    train_idx = random_idx[0:train_num]
    test_num = np.floor(test_prop * n_all).astype(int)
    test_idx = random_idx[-test_num:]
    return train_idx, test_idx

def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.3 of the paper
    :return:Sn
    """
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    max_iterations = 200  # 设置最大循环次数
    iterations = 0  # 初始化循环次数

    while error >= 0.005 and iterations < max_iterations:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()#生成一个len^view的矩阵，矩阵每一行只有一个1
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)#0.25
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))

        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
        iterations=iterations+1
    return matrix

def cosineSimilartydis(A,B):
    A=A/(torch.norm(A,dim=1,p=2,keepdim=True)+0.000001)
    B=B/(torch.norm(B,dim=1,p=2,keepdim=True)+0.000001)

    W=torch.mm(A,B.t())
    max_values, _ = torch.max(W, axis=0)
    min_values, _ = torch.min(W, axis=0)
    denominator = max_values - min_values
    denominator = torch.clamp(denominator, min=1e-6)
    normalized_matrix = (W - min_values) / denominator
    return 1-normalized_matrix

def find_nanchor(A,B):
    print(A.device)
    W=cosineSimilartydis(A, B)#表示距离
    n = math.ceil(W.shape[0]/19)
    # print(n)
    # 复制矩阵A以避免修改原始矩阵
    modified_matrix_A = W.clone()
    print(modified_matrix_A.device,'de')
    for col in range(modified_matrix_A.shape[1]):
        min_indices = np.argpartition(modified_matrix_A[:, col], n)[:n]
        modified_matrix_A[min_indices, col] = 0

    return modified_matrix_A