from egonetwork import EgoNetwork
from evaluation import Evaluation
from eval import evaluation
from zliu import run_all_except
import numpy as np

NUM_CIRCLE = 6
ALPHA = 0.5
LAMBDA = 1
'''
This is our final implementation of CESNA framework.
There are also 3 ways of initilization.
conductance, no neighbor node and apriori
'''


class CesnaNew:
    def __init__(self, ego_id):
        '''
        Initialize
        '''
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
        self.init_f_mat_conductance()

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
        '''
        preprocessing
        '''
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

    def get_conductance(self, node_idx):
        '''
        helper function for initialize by conductance
        '''
            #http://courses.cms.caltech.edu/cs139/notes/lecture10.pdf
        if node_idx not in self.neighbor_dic:
            return 0
        deltas = 0
        for i in self.neighbor_dic[node_idx]:
            if i in self.neighbor_dic:
                for j in self.neighbor_dic:
                    if not (j in self.neighbor_dic[node_idx] or j == node_idx):
                        deltas += 1
        ds = 0
        dvs = 0
        for i in range(self.num_u):
            if i in self.neighbor_dic:
                if i in self.neighbor_dic[node_idx] or i == node_idx:
                    ds += len(self.neighbor_dic[i])
                else:
                    dvs += len(self.neighbor_dic[i])
        return deltas / ((ds + dvs) / 2)

    def init_f_mat_conductance(self):
        '''
        initialize by conductance
        '''
        conductance_dic = {}
        for i in range(self.num_u):
            conductance_dic[i] = self.get_conductance(i)
        circle_count = 0
        for i in range(self.num_u):
            minimal = True
            if i in self.neighbor_dic:
                for j in self.neighbor_dic[i]:
                    if conductance_dic[j] < i:
                        minimal = False
            if minimal:
                self.f_mat[i][circle_count] = self.delta
                if i in self.neighbor_dic:
                    for j in self.neighbor_dic[i]:
                        self.f_mat[j][circle_count] = self.delta
                circle_count += 1
            if circle_count == NUM_CIRCLE - 1:
                break

    def init_f_mat(self):
        '''
        other initialize methods
        '''
        '''
        no neighbor node below
        '''
        # count = 0
        # for i in range(self.num_u):
        #     if len(self.neighbor_dic[i]) == 0:
        #         self.f_mat[i][count] = self.delta
        #         count += 1
        #     if count == self.num_c - 1:
        #         break
        '''
        apriori below
        '''
        # res = run_all_except(self.ego_id)
        # obj = evaluation(res, self.ego_id)
        # obj.eval()
        # l = sorted(obj.circle_list_detected, key=lambda x:len(x), reverse=True)[1:]
        # for i in range(NUM_CIRCLE-1):
        #     for node_id in l[i]:
        #         if node_id not in self.id2idx_dic:
        #             continue
        #         else:
        #             self.f_mat[self.id2idx_dic[node_id]][i] = self.delta
        '''
        apriori and feature
        '''
        res = run_all_except(self.ego_id)
        obj = evaluation(res, self.ego_id)
        obj.eval()
        temp = []
        for i in range(len(obj.circle_list_feature)):
            temp.append((obj.circle_list_feature[i], obj.circle_list_detected[i]))
        l = sorted(temp, key=lambda x:len(x[1]), reverse=True)[1:]
        for i in range(NUM_CIRCLE-1):
            for node_id in l[i][1]:
                if node_id not in self.id2idx_dic:
                    continue
                else:
                    self.f_mat[self.id2idx_dic[node_id]][i] = self.delta
            feat_count = 0
            for feat_name in l[i][0]:
                if feat_name in self.network.featname_list_back:
                    feat_count += 1
            for feat_name in l[i][0]:
                if feat_name in self.network.featname_list_back:
                    self.w_mat[self.network.featname_list_back[feat_name]][i] = 1 / feat_count

    def update_q(self):
        '''
        update Q formula
        optimize by matrix multiplication
        '''
        self.q_mat = (self.w_mat @ self.f_mat.T).T #to align Q_uk

        self.q_mat = self.q_mat.clip(-20, 20)
        self.q_mat = 1 / (1 + np.exp(self.q_mat))

    def update_f_ft(self):
        '''
        update f dot f.T
        '''
        self.f_ft_mat = self.f_mat @ self.f_mat.T

    def get_new_w_mat(self):
        '''
        update w matrix
        optmized by matrix multiplication
        '''
        return self.w_mat + ALPHA * ((self.x_mat - self.q_mat).T @ self.f_mat - LAMBDA * np.sign(self.w_mat))

    def lgfu(self,u, c):
        '''
        get the lgfu part for update f matrix
        '''
        res = 0
        for v in range(self.num_u):
            if v == u:
                continue
            if self.adj_mat[u][v] == 1:
                temp = np.exp(-self.f_ft_mat[u][v])
                res += self.f_mat[v][c] * (temp / (1-temp))
            else:
                res -= self.f_mat[v][c]
        return res

    def lxfu(self, u, c):
        '''
        get the lxfu part for update f matrix
        '''
        return (self.x_mat[u] - self.q_mat[u]) @ self.w_mat[:, c]

    def update_f_u_c(self, u, c):
        '''
        update single cell of f matrix
        '''
        self.f_mat[u][c] = max(0, self.f_mat[u][c] + ALPHA * (self.lgfu(u, c) + self.lxfu(u, c)))

    def update_f(self):
        '''
        update whole f matrix
        '''
        for u in range(self.num_u):
            for c in range(self.num_c):
                self.update_f_u_c(u, c)
        self.f_mat[:, -1] = 1

    def update_w(self):
        '''
        update w matrix
        and normalize
        '''
        self.w_mat = self.get_new_w_mat()
        self.w_mat[self.w_mat<0] = 0
        norm = np.linalg.norm(self.w_mat, axis=0)
        for i in range(len(norm)):
            if norm[i] == 0:
                norm[i] = 1
        self.w_mat /= norm

    def prop(self):
        '''
        get score of the current state
        '''
        exp_f_ft = np.exp(-self.f_ft_mat)
        exp_f_ft = exp_f_ft / (1 - exp_f_ft)
        f_t_mat = self.f_mat.T
        d_lg_fu_mat = f_t_mat @ (self.adj_mat * exp_f_ft)- f_t_mat @ self.neg_adj_mat
        lg = np.sum(d_lg_fu_mat) / 2
        lg = 0
        for i in range(self.num_u):
            for j in range(i+1, self.num_u):
                if self.adj_mat[i][j] == 1:
                    lg += (1-np.exp(-self.f_ft_mat[i][j]))
                else:
                    lg -= self.f_ft_mat[i][j]
        lx = 0
        for u in range(self.num_u):
            for k in range(self.num_k):
                lx += self.x_mat[u][k] * np.log(self.q_mat[u][k]) + (1 - self.x_mat[u][k]) * np.log(1 - self.q_mat[u][k])
        # print(lg + lx)
        return np.log(-(lg + lx))

    def update(self):
        self.update_q()
        self.update_f_ft()
        self.update_f()
        self.update_q() #since f is updated
        self.update_w()

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
        # print(res, self.prop())
        # print(res)
        return res

NODE_ID_LIST = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
# NODE_ID_LIST = [348, 3980, 414, 686, 698]
def test(n):
    print(n, "----------------------")
    a = CesnaNew(n)
    res = 0
    for i in range(31):
        a.update()
        res = a.get_eval()
        if i % 10 == 0:
            print("\t", "iter", i, ":", res)
    print(n, "result: ", res)
    print("----------------------")

# test(3980)

for n in NODE_ID_LIST:
    test(n)
