import numpy as np
from numba import jit
from utils import *

import FPMC as FPMC_basic


class FPMC(FPMC_basic.FPMC):
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        super(FPMC, self).__init__(n_user, n_item, n_factor, learn_rate, regular)

    def evaluation(self, data_3_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        precision, recall, hr, mrr, auc = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], self.VUI_m_VIU,
                                                         self.VIL_m_VLI)

        return precision, recall, hr, mrr, auc

    def learn_epoch(self, data_3_list, neg_batch_size):
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_3_list[0], data_3_list[1], data_3_list[2], neg_batch_size,
                                             np.array(list(self.item_set)), self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=5, neg_batch_size=5, eval_per_epoch=True,
                       ret_in_score=False):
        global te_3_list, precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr
        tr_3_list = data_to_3_list(tr_data)
        if te_data is not None:
            te_3_list = data_to_3_list(te_data)

        precision_list = []
        recall_list = []
        hr_list = []
        mrr_list = []
        auc_list = []  # 添加 auc_list

        for epoch in range(n_epoch):
            self.learn_epoch(tr_3_list, neg_batch_size)

            if eval_per_epoch:
                precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr = self.evaluation(tr_3_list)  # 修改这一行
                print('Epoch %d: Train Precision: %.4f, Recall: %.4f, HR: %.4f, MRR: %.4f, AUC: %.4f' % (  # 修改这一行
                    epoch, precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr))  # 修改这一行

                if te_data is not None:
                    precision_te, recall_te, hr_te, mrr_te, auc_te = self.evaluation(te_3_list)  # 修改这一行
                    print('Epoch %d: Test Precision: %.4f, Recall: %.4f, HR: %.4f, MRR: %.4f, AUC: %.4f' % (  # 修改这一行
                        epoch, precision_te, recall_te, hr_te, mrr_te, auc_te))  # 修改这一行
                    precision_list.append(precision_te)
                    recall_list.append(recall_te)
                    hr_list.append(hr_te)
                    mrr_list.append(mrr_te)
                    auc_list.append(auc_te)  # 添加 auc_list
                else:
                    precision_list.append(precision_tr)
                    recall_list.append(recall_tr)
                    hr_list.append(hr_tr)
                    mrr_list.append(mrr_tr)
                    auc_list.append(auc_tr)  # 添加 auc_list
            else:
                print('Epoch %d done' % epoch)

        if not eval_per_epoch:
            precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr = self.evaluation(tr_3_list)  # 修改这一行
            print('Train Precision: %.4f, Recall: %.4f, HR: %.4f, MRR: %.4f, AUC: %.4f' % (
                precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr))  # 修改这一行

            if te_data is not None:
                precision_te, recall_te, hr_te, mrr_te, auc_te = self.evaluation(te_3_list)  # 修改这一行
                print('Test Precision: %.4f, Recall: %.4f, HR: %.4f, MRR: %.4f, AUC: %.4f' % (
                    precision_te, recall_te, hr_te, mrr_te, auc_te))  # 修改这一行
                precision_list.append(precision_te)
                recall_list.append(recall_te)
                hr_list.append(hr_te)
                mrr_list.append(mrr_te)
                auc_list.append(auc_te)  # 添加 auc_list
            else:
                precision_list.append(precision_tr)
                recall_list.append(recall_tr)
                hr_list.append(hr_tr)
                mrr_list.append(mrr_tr)
                auc_list.append(auc_tr)  # 添加 auc_list

        if te_data is not None:
            if ret_in_score:
                return np.mean(precision_list), np.mean(recall_list), np.mean(hr_list), np.mean(mrr_list), (
                    precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr)
            else:
                return np.mean(precision_list), np.mean(recall_list), np.mean(hr_list), np.mean(mrr_list), np.mean(
                    auc_list)  # 修改这一行
        else:
            return None


@jit(nopython=True)
def compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL):
    acc_val = 0.0
    for l in b_tm1:
        acc_val += np.dot(VIL[i], VLI[l])
    return (np.dot(VUI[u], VIU[i]) + (acc_val / len(b_tm1)))


@jit(nopython=True)
def learn_epoch_jit(u_list, i_list, b_tm1_list, neg_batch_size, item_set, VUI, VIU, VLI, VIL, learn_rate, regular):
    for iter_idx in range(len(u_list)):
        d_idx = np.random.randint(0, len(u_list))
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]

        j_list = np.random.choice(item_set, size=neg_batch_size, replace=False)

        z1 = compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL)
        for j in j_list:
            z2 = compute_x_jit(u, j, b_tm1, VUI, VIU, VLI, VIL)
            delta = 1 - sigmoid_jit(z1 - z2)

            VUI_update = learn_rate * (delta * (VIU[i] - VIU[j]) - regular * VUI[u])
            VIUi_update = learn_rate * (delta * VUI[u] - regular * VIU[i])
            VIUj_update = learn_rate * (-delta * VUI[u] - regular * VIU[j])

            VUI[u] += VUI_update
            VIU[i] += VIUi_update
            VIU[j] += VIUj_update

            eta = np.zeros(VLI.shape[1])
            for l in b_tm1:
                eta += VLI[l]
            eta = eta / len(b_tm1)

            VILi_update = learn_rate * (delta * eta - regular * VIL[i])
            VILj_update = learn_rate * (-delta * eta - regular * VIL[j])
            VLI_updates = np.zeros((len(b_tm1), VLI.shape[1]))
            for idx, l in enumerate(b_tm1):
                VLI_updates[idx] = learn_rate * ((delta * (VIL[i] - VIL[j]) / len(b_tm1)) - regular * VLI[l])

            VIL[i] += VILi_update
            VIL[j] += VILj_update
            for idx, l in enumerate(b_tm1):
                VLI[l] += VLI_updates[idx]

    return VUI, VIU, VLI, VIL


@jit(nopython=True)
def sigmoid_jit(x):
    if x >= 0:
        return math.exp(-np.logaddexp(0, -x))
    else:
        return math.exp(x - np.logaddexp(x, 0))


@jit(nopython=True)
def compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI):
    former = VUI_m_VIU[u]
    latter = np.zeros(VIL_m_VLI.shape[0])
    for idx in range(VIL_m_VLI.shape[0]):
        for l in b_tm1:
            latter[idx] += VIL_m_VLI[idx, l]
    latter = latter / len(b_tm1)

    return former + latter


@jit(nopython=True)
def evaluation_jit(u_list, i_list, b_tm1_list, VUI_m_VIU, VIL_m_VLI):
    precision_total = 0.0
    recall_total = 0.0
    hr_total = 0.0
    mrr_total = 0.0
    auc_total = 0.0
    ndcg_total = 0.0
    novelty_total = 0.0
    diversity_total = 0.0

    len_u = len(u_list)
    len_r = len_u
    k = 5

    for d_idx in range(len_u):
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]
        scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)
        # scores长度是item文件中最大的数字+1
        # print(len(scores))

        # 计算精确度：如果序列长度小于2，则不计算，因为只能推荐那个出现的那个物品
        if len(b_tm1) < 2:
            len_r -= 1
            continue
        r = min(k, len(b_tm1))
        r_indices = np.argsort(-scores)[:r]
        if i in r_indices:
            precision_total += 1 / r

        # 计算召回率
        if i in r_indices:
            recall_total += 1

        # 计算命中率
        hit_rate = 1 if i == scores.argmax() else 0
        hr_total += hit_rate

        # 计算MRR
        rank = len(np.where(scores > scores[i])[0]) + 1
        mrr_total += 1.0 / rank

        # 计算auc
        r_indices = np.argsort(-scores)[:r + 1]
        min_r = r_indices[-1]
        auc = 0.0
        negetive = np.arange(0, len(scores) - 1, 1)

        for j in negetive:
            if scores[i] > scores[j]:
                auc += 1
            elif scores[i] == scores[j]:
                auc += 0.5

        auc_total += auc / len(negetive)

        # 计算ndcg
        position_indices = np.where(r_indices == i)[0]
        if position_indices.size > 0:
            ndcg_total += 1.0 / np.log2(position_indices[0] + 2)

        # 计算新颖度
        novelty_score = 0.0
        for j in r_indices:
            position_index = np.where(b_tm1[::-1] == j)[0]
            if position_index.size > 0:
                position = position_index[0] + 1  # 倒着索引的位置需要加1
                # novelty_score += 1 / position_index
                novelty_score += position / len(b_tm1)
            else:
                novelty_score += 1  # 如果找不到，则默认新颖度为1
        novelty_total += novelty_score / k

        # 计算多样性
        diversity_total += novelty_score / (0.5 * k * (k - 1))
    # 计算平均值
    precision_avg = precision_total / len_r
    recall_avg = recall_total / len_r
    hit_rate_avg = hr_total / len_u
    mrr_avg = mrr_total / len_u
    auc_avg = auc_total / len_u
    ndcg_avg = ndcg_total / len_r
    novelty_avg = novelty_total / (len_r - 1)
    diversity_avg = diversity_total / (len_r - 1)

    # return precision_avg, recall_avg, hit_rate_avg, mrr_avg, auc_avg
    return precision_avg, auc_avg, ndcg_avg, novelty_avg, diversity_avg
