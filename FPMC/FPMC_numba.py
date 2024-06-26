from numba import jit
import FPMC as FPMC_basic
from utils import *


class FPMC(FPMC_basic.FPMC):
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        super(FPMC, self).__init__(n_user, n_item, n_factor, learn_rate, regular)

    def evaluation(self, data_3_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
        (precision, recall, hr, mrr,
         auc, ndcg, novelty, diversity) = evaluation_jit(data_3_list[0], data_3_list[1], data_3_list[2], data_3_list[3],
                                                         self.item_set, self.VUI_m_VIU, self.VIL_m_VLI)
        return precision, recall, hr, mrr, auc, ndcg, novelty, diversity

    def learn_epoch(self, data_3_list, neg_batch_size):
        VUI, VIU, VLI, VIL = learn_epoch_jit(data_3_list[0], data_3_list[1], data_3_list[2], neg_batch_size,
                                             np.array(list(self.item_set)), self.VUI, self.VIU, self.VLI, self.VIL,
                                             self.learn_rate, self.regular)
        self.VUI = VUI
        self.VIU = VIU
        self.VLI = VLI
        self.VIL = VIL

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=10, neg_batch_size=10, eval_per_epoch=True,
                       ret_in_score=False):
        tr_3_list = data_to_3_list(tr_data)
        te_3_list = data_to_3_list(te_data)

        precision_list = []
        recall_list = []
        hr_list = []
        mrr_list = []
        auc_list = []
        ndcg_list = []
        novelty_list = []
        diversity_list = []

        for epoch in range(n_epoch):
            self.learn_epoch(tr_3_list, neg_batch_size)

            if eval_per_epoch:
                (precision_tr, recall_tr, hr_tr, mrr_tr,
                 auc_tr, ndcg_tr, novelty_tr, diversity_tr) = self.evaluation(tr_3_list)
                print('Epoch %d: Train Precision: %.4f, Recall: %.4f, HR: %.4f, MRR: %.4f, '
                      'AUC: %.4f, NDCG: %.4f, Novelty: %.4f, Diversity: %.4f' %
                      (epoch, precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr, ndcg_tr, novelty_tr, diversity_tr))

                (precision_te, recall_te, hr_te, mrr_te,
                 auc_te, ndcg_te, novelty_te, diversity_te) = self.evaluation(te_3_list)
                print('Epoch %d: Test Precision: %.4f, Recall: %.4f, HR: %.4f, MRR: %.4f, '
                      'AUC: %.4f, NDCG: %.4f, Novelty: %.4f, Diversity: %.4f' %
                      (epoch, precision_te, recall_te, hr_te, mrr_te, auc_te, ndcg_te, novelty_te, diversity_te))
            else:
                print('Epoch %d done' % epoch)

        if not eval_per_epoch:
            precision_tr, recall_tr, hr_tr, mrr_tr, auc_tr, ndcg_tr, novelty_tr, diversity_tr = self.evaluation(
                tr_3_list)
            precision_te, recall_te, hr_te, mrr_te, auc_te, ndcg_te, novelty_te, diversity_te = self.evaluation(
                te_3_list)
        precision_list.append(precision_te)
        recall_list.append(recall_te)
        hr_list.append(hr_te)
        mrr_list.append(mrr_te)
        auc_list.append(auc_te)
        ndcg_list.append(ndcg_te)
        novelty_list.append(novelty_te)
        diversity_list.append(diversity_te)

        return (np.mean(precision_list), np.mean(recall_list), np.mean(hr_list), np.mean(mrr_list),
                np.mean(auc_list), np.mean(ndcg_list), np.mean(novelty_list), np.mean(diversity_list))


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


@jit(nopython=True)
def evaluation_jit(u_list, i_list, b_tm1_list, basket_list, item_set, VUI_m_VIU, VIL_m_VLI):
    precision_sum = 0.0
    recall_sum = 0.0
    hr_sum = 0.0
    mrr_sum = 0.0
    auc_sum = 0.0
    ndcg_sum = 0.0
    novelty_sum = 0.0
    diversity_sum = 0.0
    k = 5  # 推荐列表容量

    for d_idx in range(len(u_list)):
        u = u_list[d_idx]
        i = i_list[d_idx]
        b_tm1 = b_tm1_list[d_idx][b_tm1_list[d_idx] != -1]
        scores = compute_x_batch_jit(u, b_tm1, VUI_m_VIU, VIL_m_VLI)
        basket = basket_list[d_idx][basket_list[d_idx] != -1]
        km_indices = np.argsort(-scores)[:k]
        # 求真实列表和推荐列表交集
        intersection = np.intersect1d(km_indices, basket)

        # 计算 Precision
        intersection_count = len(intersection)
        precision = intersection_count / k
        precision_sum += precision

        # 计算 Recall
        recall = intersection_count / len(basket)
        recall_sum += recall

        # 计算 HR (Hit Rate)
        hr = 1 if i in km_indices else 0
        hr_sum += hr

        # 计算 MRR
        rank = len(np.where(scores > scores[i])[0]) + 1
        rr = 1.0 / rank
        mrr_sum += rr

        # 计算 AUC
        positive = basket
        auc = 0.0
        sample = np.array(list(item_set))
        result = []
        for val in sample:
            # 如果元素不在positive数组中，则添加到负样本数组中
            if val not in positive:
                result.append(val)
        negative = np.array(result)
        for ps in positive:
            for ns in negative:
                if scores[ps] > scores[ns]:
                    auc += 1
                elif scores[ps] == scores[ns]:
                    auc += 0.5
        auc_sum += auc / (len(positive) * len(negative))

        # 计算 NDCG
        ndcg_idx = 0.0
        for j in basket_list[d_idx]:
            position = np.where(km_indices == j)[0]
            if position.size > 0:
                ndcg_idx += 1.0 / np.log2(position[0] + 2)
        ndcg_sum += ndcg_idx / k

        # 计算新颖度
        novelty_idx = 0.0
        for j in km_indices:
            position_k = np.where(b_tm1[::-1] == j)[0]
            if position_k.size > 0:
                position_k = position_k[0] + 1  # 倒着的位置是索引加1
                novelty_idx += position_k / len(b_tm1)
            else:
                novelty_idx += 1  # 如果找不到，则新颖度为1
        factor = novelty_idx / k
        novelty_sum += factor
        # 计算多样性
        diversity_sum += factor / (0.5 * k)

    len_u = len(u_list)
    precision = precision_sum / len_u
    recall = recall_sum / len_u
    hr = hr_sum / len_u
    mrr = mrr_sum / len_u
    auc = auc_sum / len_u
    ndcg = ndcg_sum / len_u
    novelty = novelty_sum / len_u
    diversity = diversity_sum / len_u

    return precision, recall, hr, mrr, auc, ndcg, novelty, diversity
