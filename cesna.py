from egonetwork import EgoNetwork
from evaluation import Evaluation
import numpy as np

CIRCLE_NUM = 11
ALPHA = 0.5
LAMBDA = 1

'''
Please refer to cesnanew.py for final implementation
'''

class Cesana:
    def __init__(self, ego_id):
        self.ego_id = str(ego_id)
        self.network = EgoNetwork(ego_id)
        self.idx_dic = {}
        self.idx_dic_back = {}
        self.neighbor_dic = {}
        self.x_mat = self.get_x_mat() #exclude ego, numpy array, node index starting from 0
        self.f_mat = np.zeros((len(self.network.node_feat)-1, CIRCLE_NUM)) #exclude ego, numpy array
        self.w_mat = np.zeros((len(self.network.featname_list), CIRCLE_NUM))
        self.k_num = (len(self.network.featname_list))
        self.w_mat += 1 / self.k_num    #all init to 1/k **I guess**
        self.u_num = len(self.idx_dic)
        self.new_f_mat = np.zeros((len(self.network.node_feat)-1, CIRCLE_NUM))
        self.new_w_mat = np.zeros((len(self.network.featname_list), CIRCLE_NUM))
        self.delta = (-np.log(1 - 1 / self.u_num)) ** 0.5
        self.f_mat[:, -1] = 1   #add one for not having zero
        self.w_f_t = None
        self.f_t_t = None

    def get_x_mat(self):
        node_feat = {}
        feat_file_name = "facebook/" + self.ego_id + ".feat"
        feat_file = open(feat_file_name, 'r')
        for cur_line in feat_file:
            cur_list = [int(x) for x in cur_line.strip('\n').split(' ')]
            node_id = cur_list[0]
            node_feat[node_id] = cur_list[1:]
        res = np.zeros((len(node_feat), len(self.network.featname_list)))
        count = 0
        for key in node_feat:
            if key != int(self.ego_id):
                self.idx_dic[key] = count
                self.idx_dic_back[count] = key
                res[count] += np.array(node_feat[key])
                count += 1

        for key in self.idx_dic:
            self.neighbor_dic[self.idx_dic[key]] = {}
            if key not in self.network.adj_mat:
                continue
            for kkey in self.network.adj_mat[key]:
                self.neighbor_dic[self.idx_dic[key]][self.idx_dic[kkey]] = 1
        return res

    '''
    above for init and file io
    =========================
    below for init f_mat
    '''
    def init_f_mat(self):
        '''
        since the smallest conductance is the ones with no neighbors
        '''
        count = 0
        for i in range(self.u_num):
            if len(self.neighbor_dic[i]) == 0:
                self.f_mat[i][count] = self.delta
                count += 1
            if count == CIRCLE_NUM - 2:
                break

    def get_conductance(self, node_idx):
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
        for i in range(self.u_num):
            if i in self.neighbor_dic:
                if i in self.neighbor_dic[node_idx] or i == node_idx:
                    ds += len(self.neighbor_dic[i])
                else:
                    dvs += len(self.neighbor_dic[i])
        # print(ds, dvs)
        # return deltas / (min(ds, dvs))  #need to deal with node with no neighbor
        return deltas / ((ds + dvs) / 2)

    def init_f_mat_conductance(self):
        conductance_dic = {}
        for i in range(self.u_num):
            conductance_dic[i] = self.get_conductance(i)
        circle_count = 0
        for i in range(self.u_num):
            minimal = True
            if i in self.neighbor_dic:
                for j in self.neighbor_dic[i]:
                    if conductance_dic[j] < i:
                        minimal = False
            if minimal:
                self.f_mat[i][circle_count] = 1
                if i in self.neighbor_dic:
                    for j in self.neighbor_dic[i]:
                        self.f_mat[j][circle_count] = 1
                circle_count += 1
            if circle_count == CIRCLE_NUM - 2:
                break



    '''
    above for init f_mat
    =========================
    below for setup for next update
    '''
    def update_w_f_t(self):
        self.w_f_t = self.w_mat @ self.f_mat.T
        # self.w_f_t = self.w_f_t.clip(-20, 20)
        self.w_f_t = 1 / (1 + np.exp(-self.w_f_t))

    def update_f_f_t(self):
        self.f_t_t = self.f_mat @ self.f_mat.T

    def get_q_u(self, u):
        return self.w_f_t[:, u]

    def get_q_k(self, k):
        return self.w_f_t[k]

    '''
    above for setup for next update
    =========================
    below for update f
    '''

    def d_lg_fu(self, u, c):
        sum_first = 0
        sum_second = 0
        for v in range(self.u_num): #all nodes
            if v == u:
                continue
            if v in self.neighbor_dic[u]:
                temp = np.exp(-self.f_t_t[u][v])
                sum_first += self.f_mat[v][c] * temp / (1 - temp)
            else:
                sum_second += self.f_mat[v][c]
        return sum_first - sum_second

    def d_lx_fu(self, u, c):
        res = (self.x_mat[u] - self.get_q_u(u)) @ self.w_mat[:, c]
        return res

    def f_new_uc(self, u, c):
        temp = self.f_mat[u][c] + ALPHA * (self.d_lg_fu(u, c) + self.d_lx_fu(u, c))
        return max(0, temp)

    '''
    above for update f
    =========================
    below for update w
    '''

    def sum_d_log_wkc(self, k, c):
        res = self.x_mat[:, k] - self.get_q_k(k) @ self.f_mat[:, c]
        return np.sum(res)

    def w_new_kc(self, k, c):
        s = self.sum_d_log_wkc(k, c)
        temp = s - LAMBDA * np.sign(self.w_mat[k][c])
        return self.w_mat[k][c] + temp

    '''
    above for update w
    =========================
    below for update functions
    '''
    def update_f(self):
        for u in range(len(self.new_f_mat)):
            for c in range(len(self.new_f_mat[u])):
                self.new_f_mat[u][c] = self.f_new_uc(u, c)
        self.f_mat = self.new_f_mat.copy()
        self.f_mat[:, -1] = 1   #add one for not having zero

    def update_w(self):
        for k in range(len(self.new_w_mat)):
            for c in range(len(self.new_w_mat[k])):
                self.new_w_mat[k][c] = self.w_new_kc(k, c)
        #normalize?
        for c in range(len(self.new_w_mat[0])):
            s = np.sum(self.new_w_mat[:, c]) ** 2
            self.new_w_mat[:, c] /= s

        self.w_mat = self.new_w_mat.copy()

    def update(self):
        self.update_w_f_t()
        self.update_f_f_t()
        self.update_f()
        self.update_w()
        self.get_eval()

    '''
    above for update functions
    =========================
    below for evaluation
    '''

    def get_circle(self):
        print(self.f_mat)
        print(self.w_mat)
        for c in range(CIRCLE_NUM-1): #leave last one
            print("Cirlce", c, end="")
            for u in range(self.u_num):
                if self.f_mat[u][c] >= self.delta:
                    print(", ", self.idx_dic_back[u],  end="")
            print()

    def get_eval(self): #jaccard
        circle_list_detected = [list(self.idx_dic.keys())]
        for c in range(CIRCLE_NUM-1):
            cur_circle_list = []
            for u in range(self.u_num):
                if self.f_mat[u][c] >= self.delta:   #hack
                    cur_circle_list.append(self.idx_dic_back[u])
            if len(cur_circle_list) > 0:
                circle_list_detected.append(cur_circle_list)
        eval_obj = Evaluation(circle_list_detected, self.network)
        res = eval_obj.get_score()
        print(res)

a = Cesana(1912)
a.init_f_mat()
a.get_eval()
while True:
    a.update()
