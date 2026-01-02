
import math
import scipy
import numpy as np
import torch
from collections import defaultdict


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def recall(ranked_list, ground_list):
    hits = 0
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id in ground_list:
            hits += 1
    rec = hits / (1.0 * len(ground_list))
    return rec

def precision(ranked_list,ground_truth):
    pred = list(map(lambda x: x in ground_truth, ranked_list))
    pred = np.array(pred).astype("float")
    right_pred = pred.sum() # shape=[100,] 表示100个用户的hit
    precis_n = len(ranked_list)
    precis = right_pred/precis_n
    return precis

def ndcg(ranked_list,ground_truth):
    pred = list(map(lambda x: x in ground_truth, ranked_list))
    pred = np.array(pred).astype("float")

    test_matrix = np.zeros(len(ranked_list)) # len = 10/20/30
    # test_matrix = np.zeros((len(pred_data), k))
    length = len(ranked_list) if len(ranked_list) <= len(ground_truth) else len(ground_truth)
    test_matrix[:length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, len(ranked_list) + 2))) # 这个ndcg更大的原因是 这里计算idcg所用的长度是k，而不是groundtruth的长度
    dcg = pred*(1./np.log2(np.arange(2, len(ranked_list) + 2)))
    dcg = np.sum(dcg)
    # idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    # print(ndcg)
    # ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


'''
def ndcg(ranked_list, ground_truth):
    dcg = 0
    # idcg = IDCG(len(ground_truth))
    idcg = IDCG(len(ranked_list))
    for i in range(len(ranked_list)):
        id = ranked_list[i]
        if id not in ground_truth:
        # if id not in ground_truth[:len(ranked_list)]:
            continue
        rank = i + 1
        dcg += 1 / math.log(rank + 1, 2)
    return dcg / idcg
'''

def IDCG(n):
    idcg = 0
    for i in range(n):
        idcg += 1 / math.log(i + 2, 2)
    return idcg


def js_topk(topk_items, sens, test_u2i, n_users, n_items, topk):
    rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    truth_rank_topk_items = np.zeros((n_users, n_items), dtype=np.int32)
    test_topk_items = topk_items.tolist()
    # for uid in range(n_users): #原始的
    for uid in list(test_u2i.keys()):
        rank_topk_items[uid][test_topk_items[uid][:topk]] = 1
        truth_rank_topk_items[uid][test_u2i[uid]] = 1

    truth_rank_topk_items = truth_rank_topk_items & rank_topk_items

    index1 = (sens == 1)
    index2 = ~index1

    rank_dis1 = np.sum(rank_topk_items[index1], axis=0)
    rank_dis2 = np.sum(rank_topk_items[index2], axis=0)
    truth_rank_dis1 = np.sum(truth_rank_topk_items[index1], axis=0)
    truth_rank_dis2 = np.sum(truth_rank_topk_items[index2], axis=0)

    rank_js_distance = scipy.spatial.distance.jensenshannon(rank_dis1, rank_dis2)
    truth_rank_js_distance = scipy.spatial.distance.jensenshannon(truth_rank_dis1, truth_rank_dis2)

    return rank_js_distance, truth_rank_js_distance

