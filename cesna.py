from egonetwork import EgoNetwork
from evaluation import Evaluation
import numpy as np

CIRCLE_NUM = 11
ALPHA = 0.5
LAMBDA = 1

class Cesana:
    def __init__(self, ego_id):
        self.ego_id = str(ego_id)
        self.network = EgoNetwork(ego_id)
        self.idx_dic = {}
        self.idx_dic_back = {}
        self.neighbor_dic = {}
        self.x_mat = self.get_x_mat() #exclude ego, numpy array, node index starting from 0
        self.f_mat = np.zeros((len(self.network.node_feat)-1, CIRCLE_NUM)) #exclude ego, numpy array
        # self.w_mat = np.random.rand((len(self.network.featname_list) * CIRCLE_NUM)).reshape((len(self.network.featname_list), CIRCLE_NUM))
        self.w_mat = np.zeros((len(self.network.featname_list), CIRCLE_NUM))
        self.k_num = (len(self.network.featname_list))
        self.w_mat += 1 / self.k_num    #all init to 1/k **I guess**
        self.u_num = len(self.idx_dic)
        self.new_f_mat = np.zeros((len(self.network.node_feat)-1, CIRCLE_NUM))
        self.new_w_mat = np.zeros((len(self.network.featname_list), CIRCLE_NUM))
        self.delta = (-np.log(1 - 1 / self.u_num)) ** 0.5
        self.f_mat[:, -1] = 1   #add one for not having zero

    def get_conductance(self, node_idx):
            #http://courses.cms.caltech.edu/cs139/notes/lecture10.pdf
        if node_idx not in self.neighbor_dic:
            return 0
        neighbor_hood = {}
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
                if (i in self.neighbor_dic[node_idx] or i == node_idx):
                    ds += len(self.neighbor_dic[i])
                else:
                    dvs += len(self.neighbor_dic[i])
        # print(ds, dvs)
        # return deltas / (min(ds, dvs))  #need to deal with node with no neighbor
        return deltas / ((ds + dvs) / 2)

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


    def get_x_mat(self):
        node_feat = {}
        feat_file_name = "facebook/" + self.ego_id + ".feat"
        feat_file = open(feat_file_name, 'r')
        for cur_line in feat_file:
            cur_list = [int(x) for x in cur_line.strip('\n').split(' ')]
            node_id = cur_list[0]
            node_feat[node_id] = cur_list[1:]

        feat_dic = {}
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
        below for updating f
        =========================
        '''
    def q_uk(self, u, k):
        s = self.w_mat[k] @ self.f_mat[u]   #should .T in theory, but numpy ignores that
        if s > 20:
            s = 20
        if s < -20:
            s = -20
        return 1 / (1 + np.exp(-s))

    def q_u(self, u):
        s = self.w_mat @ self.f_mat[u].T
        s = np.clip(s, a_min=-20, a_max=20)
        return 1 / (1 + np.exp(-s))

    def d_lg_fu(self, u, c):
        sum_first = 0
        sum_second = 0
        for v in range(self.u_num): #all nodes
            if v == u:
                continue
            if v in self.neighbor_dic[u]:
                temp = np.exp(-(self.f_mat[u] @ self.f_mat[v].T))
                sum_first += self.f_mat[v][c] * temp / (1 - temp)
            else:
                sum_second += self.f_mat[v][c]
        return sum_first - sum_second

    def d_lx_fu(self, u, c):
        res = np.sum((self.x_mat[u] - self.q_u(u)) @ self.w_mat[:, c])
        return res

    def f_new_uc(self, u, c):
        temp = self.f_mat[u][c] + ALPHA * (self.d_lg_fu(u, c) + self.d_lx_fu(u, c))
        return max(0, temp)
        '''
        =========================
        above for updating f
        '''

        '''
        below for updating w
        =========================
        '''

    def d_log_wkc(self, u, k, c):
        return (self.x_mat[u][k] - self.q_uk(u, k)) * self.f_mat[u][c]

    def w_new_kc(self, k, c):
        s = 0
        for u in range(self.u_num):
            s += self.d_log_wkc(u, k, c)
        temp = s - LAMBDA * np.sign(self.w_mat[k][c])
        return self.w_mat[k][c] + temp
        '''
        =========================
        above for updating w
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
        self.w_mat = self.new_w_mat.copy()

    def update(self):
        self.update_f()
        self.update_w()
        self.get_eval()

    def get_circle(self):
        print(self.f_mat)
        print(self.w_mat)
        for c in range(CIRCLE_NUM-1): #leave last one
            print("Cirlce", c, end="")
            for u in range(self.u_num):
                if self.f_mat[u][c] >= self.delta:
                    print(", ", self.idx_dic_back[u],  end="")
            print()

    def get_eval(self):
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

a = Cesana(0)
a.init_f_mat()
a.get_eval()
# for i in range(10): #try updating two times first
while True:
    a.update()
