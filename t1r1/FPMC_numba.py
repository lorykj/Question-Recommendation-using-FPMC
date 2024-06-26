from numba import jit
from sklearn.metrics import precision_score, recall_score, roc_auc_score

import FPMC as FPMC_basic
from utils import *


class FPMC(FPMC_basic.FPMC):
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        super(FPMC, self).__init__(n_user, n_item, n_factor, learn_rate, regular)
        self.VIL = None
        self.VLI = None
        self.VIU = None
        self.VUI = None

    def evaluation(self, data_3_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        precision, recall, auc, mrr = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], self.VUI_m_VIU,
                                                     self.VIL_m_VLI)

        return precision, recall, auc, mrr

    def learn_epoch(self, data_3_list, neg_batch_size):
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_3_list[0], data_3_list[1], data_3_list[2], neg_batch_size,
                                             np.array(list(self.item_set)), self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=5, neg_batch_size=5, eval_per_epoch=False,
                       ret_in_score=False):
        global te_3_list, precision_in, recall_in, auc_in, mrr_in, precision_out, recall_out, auc_out, mrr_out
        tr_3_list = data_to_3_list(tr_data)
        if te_data is not None:
            te_3_list = data_to_3_list(te_data)

        for epoch in range(n_epoch):

            self.learn_epoch(tr_3_list, neg_batch_size)

            if eval_per_epoch:
                precision_in, recall_in, auc_in, mrr_in = self.evaluation(tr_3_list)
                if te_data is not None:
                    precision_out, recall_out, auc_out, mrr_out = self.evaluation(te_3_list)
                    print(
                        'In sample - Precision: %.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f \t Out sample - Precision: '
                        '%.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f' % (
                            precision_in, recall_in, auc_in, mrr_in, precision_out, recall_out, auc_out, mrr_out))
                else:
                    print('In sample - Precision: %.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f' % (
                        precision_in, recall_in, auc_in, mrr_in))
            else:
                print('epoch %d done' % epoch)

        if not eval_per_epoch:
            precision_in, recall_in, auc_in, mrr_in = self.evaluation(tr_3_list)
            if te_data is not None:
                precision_out, recall_out, auc_out, mrr_out = self.evaluation(te_3_list)
                print(
                    'In sample - Precision: %.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f \t Out sample - Precision: %.4f, '
                    'Recall: %.4f, AUC: %.4f, MRR: %.4f' % (
                        precision_in, recall_in, auc_in, mrr_in, precision_out, recall_out, auc_out, mrr_out))
            else:
                print('In sample - Precision: %.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f' % (
                    precision_in, recall_in, auc_in, mrr_in))

        if te_data is not None:
            if ret_in_score:
                return precision_in, recall_in, auc_in, mrr_in, precision_out, recall_out, auc_out, mrr_out
            else:
                return precision_out, recall_out, auc_out, mrr_out
        else:
            return None


@jit(nopython=True)
def compute_x_jit(u, i, b_tm1, VUI, VIU, VLI, VIL):
    acc_val = 0.0
    for l in b_tm1:
        acc_val += np.dot(VIL[i], VLI[l])
    return np.dot(VUI[u], VIU[i]) + (acc_val / len(b_tm1))


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


# @jit(nopython=False)
# def evaluation_jit(u_list, i_list, b_tm1_list, VUI_m_VIU, VIL_m_VLI):
#     y_true = []
#     y_scores = []
#     rr_list = []
#
#     for d_idx in range(len(u_list)):
#         u = u_list[d_idx]
#         i = i_list[d_idx]
#         b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]
#         scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)
#
#         y_true.append(1)
#         y_scores.append(scores[i])
#
#         for j in range(len(scores)):
#             if j != i:
#                 y_true.append(0)
#                 y_scores.append(scores[j])
#
#         rank = len(np.where(scores > scores[i])[0]) + 1
#         rr = 1.0 / rank
#         rr_list.append(rr)
#
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)
#
#     precision = precision_score(y_true, y_scores.round(), average='weighted', zero_division=1)
#     recall = recall_score(y_true, y_scores.round(), average='weighted', zero_division=1)
#     auc = roc_auc_score(y_true, y_scores, average='weighted', multi_class='ovo')
#     mrr = np.mean(rr_list)
#
#     return precision, recall, auc, mrr


@jit(nopython=False, nogil=False)
def evaluation_jit_impl(y_true, y_scores, rr_list):
    precision = precision_score(y_true, y_scores.round(), average='weighted', zero_division=1)
    recall = recall_score(y_true, y_scores.round(), average='weighted', zero_division=1)
    auc = roc_auc_score(y_true, y_scores, average='weighted', multi_class='ovo')
    mrr = np.mean(rr_list)

    return precision, recall, auc, mrr


@jit(nopython=False)
def evaluation_jit(u_list, i_list, b_tm1_list, VUI_m_VIU, VIL_m_VLI):
    y_true = []
    y_scores = []
    rr_list = []
    hr_total = 0
    len_r = len(u_list)
    k = 3
    for d_idx in range(len(u_list)):
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]
        scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)

        if len(b_tm1) < 2:
            len_r -= 1
            continue
        r = min(k, len(b_tm1))
        r_indices = np.argsort(-scores)[:r]
        for ri in r_indices:
            y_true.append(1)
            y_scores.append(scores[i])

        for j in range(len(scores)):
            if j not in r_indices:
                y_true.append(0)
                y_scores.append(scores[j])
        # y_true.append(1)
        # y_scores.append(scores[i])
        #
        # for j in range(len(scores)):
        #     if j != i:
        #         y_true.append(0)
        #         y_scores.append(scores[j])
        correct_count = 0
        hit_rate = 1 if i == scores.argmax() else 0
        hr_total += hit_rate
        rank = len(np.where(scores > scores[i])[0]) + 1
        rr = 1.0 / rank
        rr_list.append(rr)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    precision, recall, auc, mrr = evaluation_jit_impl(y_true, y_scores, rr_list)
    hr = hr_total / len(u_list)
    return precision, recall, hr, mrr
