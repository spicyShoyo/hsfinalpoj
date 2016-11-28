from egonetwork import EgoNetwork
from evaluation import Evaluation
import numpy as np

NUM_CIRCLE = 11
ALPHA = 0.5
LAMBDA = 1


class CesnaPlus:
    def __init__(self, ego_id):
        self.ego_id = str(ego_id)
        self.network = EgoNetwork(ego_id)
        self.idx2id_dic = {}
        self.id2idx_dic = {}
        self.neighbor_dic = {}
        self.num_k = 0
        self.num_u = 0
        self.num_c = NUM_CIRCLE
        self.x_mat = None
        self.preprocessing()
        self.delta = (-np.log(1 - 1 / self.num_u)) ** 0.5
        self.f_mat = np.zeros((self.num_u, self.num_c))
        self.w_mat = np.zeros((self.num_k, self.num_c))
        self.q_mat = None
        self.f_ft_mat = None

        self.f_mat[:, -1] = 1
        self.w_mat += 1 / self.num_k
        self.init_f_mat()

        self.adj_mat = np.zeros((self.num_u, self.num_u))
        self.neg_adj_mat = np.zeros((self.num_u, self.num_u))
        for i in range(self.num_u):
            for j in range(self.num_u):
                if j in self.neighbor_dic[i]:
                    self.adj_mat[i][j] = 1
                    self.adj_mat[j][i] = 1
                else:
                    self.neg_adj_mat[i][j] = 1
                    self.neg_adj_mat[j][i] = 1
            self.neg_adj_mat[i][i] = 0

    def preprocessing(self):
        node_feat = {}
        feat_file_name = "facebook/" + self.ego_id + ".feat"
        feat_file = open(feat_file_name, 'r')
        for cur_line in feat_file:
            cur_list = [int(x) for x in cur_line.strip('\n').split(' ')]
            node_id = cur_list[0]
            node_feat[node_id] = cur_list[1:]
            node_feat[node_id] = cur_list[1:]
        self.num_u = len(node_feat)
        self.num_k = len(self.network.featname_list)
        self.x_mat = np.zeros((self.num_u, self.num_k))
        node_idx = 0
        for node_id in node_feat:
            if node_id != int(self.ego_id):
                self.id2idx_dic[node_id] = node_idx
                self.idx2id_dic[node_idx] = node_id
                self.x_mat[node_idx] += np.array(node_feat[node_id])
                node_idx += 1

        for node_id in self.id2idx_dic:
            self.neighbor_dic[self.id2idx_dic[node_id]] = {}
            if node_id not in self.network.adj_mat:
                continue
            else:
                for neighbor_node_id in self.network.adj_mat[node_id]:
                    self.neighbor_dic[self.id2idx_dic[node_id]][self.id2idx_dic[neighbor_node_id]] = 1

    def init_f_mat(self):
        count = 0
        for i in range(self.num_u):
            if len(self.neighbor_dic[i]) == 0:
                self.f_mat[i][count] = self.delta
                count += 1
            if count == self.num_c - 1:
                break

    def update_q(self):
        self.q_mat = (self.w_mat @ self.f_mat.T).T #to align Q_uk

    def update_f_ft(self):
        self.f_ft_mat = self.f_mat @ self.f_mat.T

    def get_new_f_mat(self):
        exp_f_ft = np.exp(-self.f_ft_mat)
        exp_f_ft = exp_f_ft / (1 - exp_f_ft)
        f_t_mat = self.f_mat.T
        d_lg_fu_mat = f_t_mat @ (self.adj_mat * exp_f_ft)- f_t_mat @ self.neg_adj_mat
        d_lx_fu_mat = (self.x_mat - self.q_mat) @ self.w_mat
        res = f_t_mat + ALPHA * (d_lg_fu_mat + d_lx_fu_mat.T)
        res[res<0] = 0
        return res.T

    def get_new_w_mat(self):
        return self.w_mat + ALPHA * ((self.x_mat - self.q_mat).T @ self.f_mat - LAMBDA * np.sign(self.w_mat))

    def update_f(self):
        self.f_mat = self.get_new_f_mat()
        self.f_mat[:, -1] = 1

    def update_w(self):
        self.w_mat = self.get_new_w_mat()
        #normalize?
        self.w_mat /= np.linalg.norm(self.w_mat, axis=0)

    def update(self):
        self.update_q()
        self.update_f_ft()
        self.update_f()
        self.update_q() #since f is updated
        self.update_w()
        # self.get_eval()

    def get_eval(self): #jaccard
        circle_list_detected = [list(self.id2idx_dic.keys())]
        for c in range(self.num_c-1):
            cur_circle_list = []
            for u in range(self.num_u):
                if self.f_mat[u][c] >= self.delta:
                    cur_circle_list.append(self.idx2id_dic[u])
            if len(cur_circle_list) > 0:
                circle_list_detected.append(cur_circle_list)
        eval_obj = Evaluation(circle_list_detected, self.network)
        res = eval_obj.get_score()
        print(res)
        return res

# NODE_ID_LIST = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
#
# for i in NODE_ID_LIST:
#     print(i)
#     a = CesnaPlus(i)
#     res = 0
#     for j in range(10):
#         a.update()
#         res = max(res, a.get_eval())
#     print("\t", res)

def test(n):
    a = CesnaPlus(n)
    res = a.get_eval()
    for i in range(30):
        a.update()
        res = max(res, a.get_eval())
    print(n, "best: ", res)


test(1912)
