from egonetwork import EgoNetwork
import numpy as np

CIRCLE_NUM = 11
ALPHA = 0.5
LAMBDA = 0.5

class Cesana:
    def __init__(self, ego_id):
        self.ego_id = str(ego_id)
        self.network = EgoNetwork(ego_id)
        self.idx_dic = {}
        self.idx_dic_back = {}
        self.neighbor_dic = {}
        self.x_mat = self.get_x_mat() #exclude ego, numpy array, node index starting from 0
        self.f_mat = np.zeros((len(self.network.node_feat)-1, CIRCLE_NUM)) #exclude ego, numpy array
        self.f_mat[:, -1] = 1   #add one for not having zero
        # self.w_mat = np.random.rand((len(self.network.featname_list) * CIRCLE_NUM)).reshape((len(self.network.featname_list), CIRCLE_NUM))
        self.w_mat = np.zeros((len(self.network.featname_list), CIRCLE_NUM))
        self.k_num = (len(self.network.featname_list))
        self.w_mat[:, -1] += 1 / self.k_num
        self.u_num = len(self.idx_dic)
        self.new_f_mat = np.zeros((len(self.network.node_feat)-1, CIRCLE_NUM))
        self.new_w_mat = np.zeros((len(self.network.featname_list), CIRCLE_NUM))
        self.delta = (-np.log(1 - 1 / self.u_num)) ** 0.5

    def get_x_mat(self):
        feat_dic = {}
        res = np.zeros((len(self.network.node_feat)-1, len(self.network.featname_list)))
        count = 0
        for key in self.network.node_feat:
            if key != int(self.ego_id):
                self.idx_dic[key] = count
                self.idx_dic_back[count] = key
                res[count] += np.array(self.network.node_feat[key])
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
        s = 0
        for k in range(self.k_num):
            s += (self.x_mat[u][k] - self.q_uk(u, k)) * self.w_mat[k][c]
        return s

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

    def update_w(self):
        for k in range(len(self.new_w_mat)):
            for c in range(len(self.new_w_mat[k])):
                self.new_w_mat[k][c] = self.w_new_kc(k, c)
        self.w_mat = self.new_w_mat.copy()
        self.get_circle

    def update(self):
        self.update_f()
        self.update_w()
        self.get_circle()

    def get_circle(self):
        print(self.f_mat)
        print(self.w_mat)
        for c in range(CIRCLE_NUM):
            print("Cirlce", c, end="")
            for u in range(self.u_num):
                if self.f_mat[u][c] > self.delta:
                    print(", ", self.idx_dic_back[u],  end="")
            print()


a = Cesana(0)
for i in range(2):
    a.update()
