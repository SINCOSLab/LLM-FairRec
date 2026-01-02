# -*- coding: utf-8 -*-
"""
@author: LMC_ZC

"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from sklearn.metrics import roc_auc_score

from utils import *
from models import *
from tqdm import tqdm

import scipy.stats as stats

import pdb
import sys

set_seed(2024)


def train_gcn_baseline(model, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, valid_u2i, train_set,
                       all_augment_data, args):
    optimizer_G = optim.Adam(model.parameters(), lr=args.lr)
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)

    # # 1、把增广数据集进行划分，添加到原始的数据中，构建增广数据
    # np.random.shuffle(all_augment_data)
    # k, m = divmod(len(all_augment_data), 3)
    # splited_data_part = [all_augment_data[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(3)]
    # train_u2i_augSplit = {}
    # for idx, cur_part_data in enumerate(splited_data_part):
    #     train_u2i_augSplit[idx] = {k: v.copy() for k, v in train_u2i.items()}
    #     for u, v in cur_part_data:
    #         train_u2i_augSplit[idx][u].append(v)
    #     for u in train_u2i_augSplit[idx]:  # 去重
    #         train_u2i_augSplit[idx][u] = list(set(train_u2i_augSplit[idx][u]))
    # # 2、根据划分数据构建对应的dataset，构建DataLoader
    # train_set_aug = train_set.copy()
    # train_set_aug['userid'] = train_set_aug['userid'].tolist()
    # train_set_aug['itemid'] = train_set_aug['itemid'].tolist()
    # train_set_augSplit = {i: {} for i in range(len(splited_data_part))}
    # for idx, cur_part_data in enumerate(splited_data_part):
    #     train_set_augSplit[idx]['userid'] = train_set_aug['userid']
    #     train_set_augSplit[idx]['itemid'] = train_set_aug['itemid']
    #     for d in cur_part_data:
    #         train_set_augSplit[idx]['userid'].append(d[0])
    #         train_set_augSplit[idx]['itemid'].append(d[1])
    # for idx in train_set_augSplit:
    #     train_set_augSplit[idx]['userid'] = np.array(train_set_augSplit[idx]['userid'])
    #     train_set_augSplit[idx]['itemid'] = np.array(train_set_augSplit[idx]['itemid'])
    # dataset_augSet = {}
    # for idx in train_set_augSplit:
    #     dataset_augSet[idx] = BPRTrainLoader(train_set_augSplit[idx], train_u2i_augSplit[idx], n_items)

    best_perf = 0.0
    for epoch in range(args.num_epochs):
        train_res = {
            'bpr_loss': 0.0,
            'emb_loss': 0.0,
        }
        # if epoch > 180:  # 先用原始数据训练一段时间
        #     # 2、根据当前epoch，选取对应的dataset，构建DataLoader
        #     augdata_idx = ((epoch - 181) // 10) % 3
        #     train_loader = DataLoader(dataset_augSet[augdata_idx], shuffle=True, batch_size=args.batch_size,
        #                               num_workers=args.num_workers)
        #     # 3、根据当前epoch，构建graph，设置模型的邻接矩阵
        #     graph = Graph(n_users, n_items, train_u2i_augSplit[augdata_idx])
        #     norm_adj = graph.generate_ori_norm_adj().to(args.device)  # 因为是要赋给在GPU上的模型，所以要to(device)
        #     model.norm_adj = norm_adj

        for uij in train_loader:
            u = uij[0].type(torch.long).to(args.device)
            i = uij[1].type(torch.long).to(args.device)
            j = uij[2].type(torch.long).to(args.device)

            main_user_emb, main_item_emb = model.forward()
            bpr_loss, emb_loss = calc_bpr_loss(main_user_emb, main_item_emb, u, i, j)
            emb_loss = emb_loss * args.l2_reg
            loss = bpr_loss + emb_loss

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            train_res['bpr_loss'] += bpr_loss.item()
            train_res['emb_loss'] += emb_loss.item()

        main_user_emb, main_item_emb = model.forward()
        scores = torch.matmul(main_user_emb, main_item_emb.T)
        male_mask = torch.tensor(u_sens, dtype=torch.float32) == 0
        female_mask = torch.tensor(u_sens, dtype=torch.float32) == 1
        male_scores = scores[male_mask]
        female_scores = scores[female_mask]
        male_mean_ratings = male_scores.mean(dim=0)
        female_mean_ratings = female_scores.mean(dim=0)
        item_gender_diff = 0.5 * torch.abs((male_mean_ratings - female_mean_ratings).mean(dim=0))
        optimizer_G.zero_grad()
        item_gender_diff.backward()
        optimizer_G.step()

        train_res['bpr_loss'] = train_res['bpr_loss'] / len(train_loader)
        train_res['emb_loss'] = train_res['emb_loss'] / len(train_loader)

        print()
        training_logs = 'epoch: %d, ' % epoch
        for name, value in train_res.items():
            training_logs += name + ':' + '%.6f' % value + ' '
        print(training_logs)
        if epoch > 30:  # 前面都不用测试
            with torch.no_grad():
                t_user_emb, t_item_emb = model.forward()
                test_res = ranking_evaluate(
                    user_emb=t_user_emb.detach().cpu().numpy(),
                    item_emb=t_item_emb.detach().cpu().numpy(),
                    n_users=n_users,
                    n_items=n_items,
                    train_u2i=train_u2i,
                    test_u2i=test_u2i,
                    # test_u2i=valid_u2i,
                    sens=u_sens,
                    num_workers=args.num_workers)

                print_results(test_res)

                if best_perf < test_res['ndcg@10']:
                    best_perf = test_res['ndcg@10']
                    torch.save(model, args.param_path)
                    print('save successful')
    print()
    with torch.no_grad():
        best_model = torch.load(args.param_path)
        t_user_emb, t_item_emb = best_model.forward()
        test_res = ranking_evaluate(
            user_emb=t_user_emb.detach().cpu().numpy(),
            item_emb=t_item_emb.detach().cpu().numpy(),
            n_users=n_users,
            n_items=n_items,
            train_u2i=train_u2i,
            test_u2i=test_u2i,
            sens=u_sens,
            num_workers=args.num_workers)

        print_results(test_res)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='ml_gcn_baseline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bakcbone', type=str, default='gcn')
    parser.add_argument('--dataset', type=str, default='./data/ml-1m/process/process_ours_age.pkl')  # 原始的是process.pkl
    parser.add_argument('--emb_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--l2_reg', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--log_path', type=str, default='logs/gcn_base_withvalid.txt')
    parser.add_argument('--param_path', type=str,default='param/gcn_base_noparity.pth')  # gcn_base_aug_30his
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    sys.stdout = Logger(args.log_path)
    print(args)

    with open(args.dataset, 'rb') as f:
        train_u2i = pickle.load(f)
        train_i2u = pickle.load(f)
        test_u2i = pickle.load(f)
        test_i2u = pickle.load(f)
        train_set = pickle.load(f)
        test_set = pickle.load(f)
        valid_u2i = pickle.load(f)
        valid_set = pickle.load(f)
        user_side_features = pickle.load(f)
        n_users, n_items = pickle.load(f)

    graph = Graph(n_users, n_items, train_u2i)
    norm_adj = graph.generate_ori_norm_adj()

    gcn = LightGCN(n_users, n_items, norm_adj, args.emb_size, args.n_layers, args.device)

    # u_sens = user_side_features['gender'].astype(np.int32)
    u_sens = user_side_features['age'].astype(np.int32)

    dataset = BPRTrainLoader(train_set, train_u2i, n_items)

    # with torch.no_grad():
    #     best_model = torch.load(args.param_path)
    #     t_user_emb, t_item_emb = best_model.forward()
    #     test_res = ranking_evaluate(
    #         user_emb=t_user_emb.detach().cpu().numpy(),
    #         item_emb=t_item_emb.detach().cpu().numpy(),
    #         n_users=n_users,
    #         n_items=n_items,
    #         train_u2i=train_u2i,
    #         test_u2i=test_u2i,
    #         sens=u_sens,
    #         num_workers=args.num_workers)
    #
    #     print_results(test_res)

    train_gcn_baseline(gcn, dataset, u_sens, n_users, n_items, train_u2i, test_u2i, valid_u2i, train_set, None, args)  # 这里倒数第三个参数train_u2i只是用来去除测试集的，因此用原始的
    sys.stdout = None
