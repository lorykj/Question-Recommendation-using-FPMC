import pickle
import random
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from utils import *


class FPMC:
    def __init__(self, n_user, n_item, n_factor, learn_rate, regular):
        self.VIL_m_VLI = None
        self.VUI_m_VIU = None
        self.VLI = None
        self.VIL = None
        self.VIU = None
        self.VUI = None
        self.user_set = set()
        self.item_set = set()

        self.n_user = n_user
        self.n_item = n_item

        self.n_factor = n_factor
        self.learn_rate = learn_rate
        self.regular = regular

    @staticmethod
    def dump(fpmcObj, fname):
        pickle.dump(fpmcObj, open(fname, 'wb'))

    @staticmethod
    def load(fname):
        return pickle.load(open(fname, 'rb'))

    def init_model(self, std=0.01):
        self.VUI = np.random.normal(0, std, size=(self.n_user, self.n_factor))
        self.VIU = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VIL = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VLI = np.random.normal(0, std, size=(self.n_item, self.n_factor))
        self.VUI_m_VIU = np.dot(self.VUI, self.VIU.T)
        self.VIL_m_VLI = np.dot(self.VIL, self.VLI.T)

    def compute_x(self, u, i, b_tm1):
        acc_val = 0.0
        for l in b_tm1:
            acc_val += np.dot(self.VIL[i], self.VLI[l])
        return np.dot(self.VUI[u], self.VIU[i]) + (acc_val / len(b_tm1))

    def compute_x_batch(self, u, b_tm1):
        former = self.VUI_m_VIU[u]
        latter = np.mean(self.VIL_m_VLI[:, b_tm1], axis=1).T
        return former + latter

    # def evaluation(self, data_list):
    #     np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
    #     np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)
    #
    #     y_true = []
    #     y_scores = []
    #     rr_list = []
    #     for (u, i, b_tm1) in data_list:
    #         scores = self.compute_x_batch(u, b_tm1)
    #
    #         y_true.append(1)
    #         y_scores.append(scores[i])
    #
    #         for j in range(self.n_item):
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
    def evaluation(self, data_list):
        np.dot(self.VUI, self.VIU.T, out=self.VUI_m_VIU)
        np.dot(self.VIL, self.VLI.T, out=self.VIL_m_VLI)

        total_samples = len(data_list)
        total_precision = 0
        total_recall = 0
        total_auc = 0
        total_mrr = 0

        for (u, i, b_tm1) in data_list:
            scores = self.compute_x_batch(u, b_tm1)

            y_true = np.zeros(self.n_item)
            y_true[i] = 1

            y_scores = np.zeros(self.n_item)
            y_scores[i] = scores[i]
            for j in range(self.n_item):
                if j != i:
                    y_scores[j] = scores[j]

            precision = precision_score(y_true, y_scores.round(), average='weighted', zero_division=1)
            recall = recall_score(y_true, y_scores.round(), average='weighted', zero_division=1)
            auc = roc_auc_score(y_true, y_scores, average='weighted', multi_class='ovo')

            rank = len(np.where(scores > scores[i])[0]) + 1
            rr = 1.0 / rank

            total_precision += precision
            total_recall += recall
            total_auc += auc
            total_mrr += rr

        precision_avg = total_precision / total_samples
        recall_avg = total_recall / total_samples
        auc_avg = total_auc / total_samples
        mrr_avg = total_mrr / total_samples

        return precision_avg, recall_avg, auc_avg, mrr_avg

    def learn_epoch(self, tr_data, neg_batch_size):
        for iter_idx in range(len(tr_data)):
            (u, i, b_tm1) = random.choice(tr_data)

            exclu_set = self.item_set - {i}
            j_list = random.sample(exclu_set, neg_batch_size)

            z1 = self.compute_x(u, i, b_tm1)
            for j in j_list:
                z2 = self.compute_x(u, j, b_tm1)
                delta = 1 - sigmoid(z1 - z2)

                VUI_update = self.learn_rate * (delta * (self.VIU[i] - self.VIU[j]) - self.regular * self.VUI[u])
                VIUi_update = self.learn_rate * (delta * self.VUI[u] - self.regular * self.VIU[i])
                VIUj_update = self.learn_rate * (-delta * self.VUI[u] - self.regular * self.VIU[j])

                self.VUI[u] += VUI_update
                self.VIU[i] += VIUi_update
                self.VIU[j] += VIUj_update

                eta = np.mean(self.VLI[b_tm1], axis=0)
                VILi_update = self.learn_rate * (delta * eta - self.regular * self.VIL[i])
                VILj_update = self.learn_rate * (-delta * eta - self.regular * self.VIL[j])
                VLI_update = self.learn_rate * (
                        (delta * (self.VIL[i] - self.VIL[j]) / len(b_tm1)) - self.regular * self.VLI[b_tm1])

                self.VIL[i] += VILi_update
                self.VIL[j] += VILj_update
                self.VLI[b_tm1] += VLI_update

    def learnSBPR_FPMC(self, tr_data, te_data=None, n_epoch=10, neg_batch_size=5, eval_per_epoch=False):
        global precision_out, recall_out, auc_out, mrr_out
        for epoch in range(n_epoch):
            self.learn_epoch(tr_data, neg_batch_size=neg_batch_size)

            if eval_per_epoch:
                precision_in, recall_in, auc_in, mrr_in = self.evaluation(tr_data)
                if te_data is not None:
                    precision_out, recall_out, auc_out, mrr_out = self.evaluation(te_data)
                    print(
                        'Epoch %d: Precision (In/Out): %.4f/%.4f, Recall (In/Out): %.4f/%.4f, AUC (In/Out): '
                        '%.4f/%.4f, MRR (In/Out): %.4f/%.4f' %
                        (epoch, precision_in, precision_out, recall_in, recall_out, auc_in, auc_out, mrr_in, mrr_out))
                else:
                    print('Epoch %d: Precision: %.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f' %
                          (epoch, precision_in, recall_in, auc_in, mrr_in))
            else:
                print('epoch %d done' % epoch)

        if not eval_per_epoch:
            precision_in, recall_in, auc_in, mrr_in = self.evaluation(tr_data)
            if te_data is not None:
                precision_out, recall_out, auc_out, mrr_out = self.evaluation(te_data)
                print(
                    'Precision (In/Out): %.4f/%.4f, Recall (In/Out): %.4f/%.4f, AUC (In/Out): %.4f/%.4f, '
                    'MRR (In/Out): %.4f/%.4f' %
                    (precision_in, precision_out, recall_in, recall_out, auc_in, auc_out, mrr_in, mrr_out))
            else:
                print('Precision: %.4f, Recall: %.4f, AUC: %.4f, MRR: %.4f' % (precision_in, recall_in, auc_in, mrr_in))

        if te_data is not None:
            return precision_out, recall_out, auc_out, mrr_out
        else:
            return None
