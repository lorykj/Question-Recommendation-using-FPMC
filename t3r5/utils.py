import csv
import math
import numpy as np


def sigmoid(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))


def load_data_from_dir(dirname):
    fname_user_idxseq = dirname + '/' + 'idxseq.txt'
    fname_user_list = dirname + '/' + 'user_idx_list.txt'
    fname_item_list = dirname + '/' + 'item_idx_list.txt'
    user_set = load_idx_list_file(fname_user_list)
    item_set = load_idx_list_file(fname_item_list)

    data_list = []
    with open(fname_user_idxseq, 'r') as f:
        for line in f:
            line = [int(s) for s in line.strip().split()]
            user = line[0]
            b_tm1 = list(set(line[1:]))  # 获取除了最后一个之外的所有物品
            label = line[-1]

            data_list.append((user, label, b_tm1))

    return data_list, user_set, item_set


def load_idx_list_file(fname, delimiter=','):
    idx_set = set()
    with open(fname, 'r') as f:
        # discard header
        f.readline()

        for line in csv.reader(f, delimiter=delimiter, quotechar='"'):
            idx = int(line[0])
            idx_set.add(idx)
    return idx_set


# def data_to_3_list(data_list):
#     u_list = []
#     i_list = []
#     b_tm1_list = []
#     basket_list = []
#     b_tm0 = []
#     k = 3
#     max_l = 0
#     #  data_list = [(uid,lastid,[id1,id2,id3...]),(uid,lastid,[]),]
#     for d in data_list:
#         b_tm0 = d[2]
#         if len(b_tm0) < k + 1:
#             continue
#         if len(b_tm0) > max_l:
#             max_l = len(b_tm0)
#         u_list.append(d[0])
#         i_list.append(d[1])
#         basket = list(b_tm0[-k:])  # 获取最后k个物品作为basket
#         # b_tm1 = d[2][:-k]  # 获取除最后5个物品外的物品作为b_tm1
#         basket_list.append(basket)
#         b_tm1_list.append(b_tm0[:-1])
#     basket_list = np.array(basket_list)
#     for b_tm1 in b_tm1_list:
#         b_tm1.extend([-1 for i in range(max_l - 1 - len(b_tm1))])
#     b_tm1_list = np.array(b_tm1_list)
#
#     return u_list, i_list, b_tm1_list, basket_list

def data_to_3_list(data_list):
    u_list = []
    i_list = []
    b_tm1_list = []
    basket_list = []
    b_tm0 = []
    k = 3
    max_l = 0
    #  data_list = [(uid,lastid,[id1,id2,id3...]),(uid,lastid,[]),]
    for d in data_list:
        b_tm1 = d[2]
        b_tm0 = d[2]
        b_tm0.append(int(d[1]))
        if len(b_tm0) < k:
            continue
        if len(b_tm1) > max_l:
            max_l = len(b_tm0)
        u_list.append(d[0])
        i_list.append(d[1])
        basket = list(b_tm0[-k:])  # 获取最后k个物品作为basket
        basket_list.append(basket)
        b_tm1_list.append(b_tm1)
    basket_list = np.array(basket_list)
    for b_tm1 in b_tm1_list:
        b_tm1.extend([-1 for i in range(max_l - len(b_tm1))])
    b_tm1_list = np.array(b_tm1_list)

    return u_list, i_list, b_tm1_list, basket_list
