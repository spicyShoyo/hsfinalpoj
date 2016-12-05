import numpy as np
from freqpattern import FreqPattern
from egonetwork import EgoNetwork
from girvannewman import GirvanNewman

NODE_ID_LIST = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
MIN_SUPPORT_RATE = 0.7

'''
Mine circles based on membership
'''
class Membership:
    def __init__(self, node_id):
        self.node_id = node_id
        self.edge_file_name = "facebook/" + str(node_id) + ".edges"
        self.cur_ego_network = EgoNetwork(node_id)
        self.num_node = len(self.cur_ego_network.node_feat)
        self.adj_dic = {}
        self.membership = np.zeros(self.num_node).astype(np.int32)
        self.node_id_dic = {}
        self.node_id_dic_back = {}
        self.fix()
        self.small_scc = []
        self.largest_scc = None
        self.local_node_id_dic = {}
        self.local_node_id_dic_back = {}
        self.group_list = None
        self.featname_list = self.cur_ego_network.featname_list

    def fix(self):
        cur_file = open(self.edge_file_name, 'r')
        cur_node_id = 0
        for cur_line in cur_file:
            i, j = [int(k) for k in cur_line.strip('\n').split(' ')]
            if i not in self.node_id_dic:
                self.node_id_dic[i] = cur_node_id
                self.node_id_dic_back[cur_node_id] = i
                cur_node_id += 1
            if j not in self.node_id_dic:
                self.node_id_dic[j] = cur_node_id
                self.node_id_dic_back[cur_node_id] = j
                cur_node_id += 1

            i_id = self.node_id_dic[i]
            j_id = self.node_id_dic[j]
            self.add_edge(i_id, j_id)

    def check_edge(self, i, j):
        if i in self.adj_dic:
            if j in self.adj_dic[i]:
                return True
        return False

    def add_edge(self, i, j):
            if i not in self.adj_dic:
                self.adj_dic[i] = {}
            self.adj_dic[i][j] = 1
            if j not in self.adj_dic:
                self.adj_dic[j] = {}
            self.adj_dic[j][i] = 1

    def check_num_group(self):
        new_membership = np.zeros(self.num_node).astype(np.int32)
        new_num_group = 0
        visited = np.zeros(self.num_node)
        while 0 in visited:
            q = []
            for i in range(len(visited)):
                if visited[i] == 0:
                    visited[i] = 1
                    q.append(i)
                    new_num_group += 1
                    new_membership[i] = new_num_group - 1
                    break

            while len(q) != 0:
                cur = q.pop(0)
                if cur not in self.adj_dic:
                    continue
                for key in self.adj_dic[cur]:
                    if visited[key] == 1:
                        continue
                    visited[key] = 1
                    new_membership[key] = new_membership[cur]
                    q.append(key)
        self.membership = new_membership


    def get_group_list(self):
        self.check_num_group()
        group_dic = {}
        for node_id, cur_membershp in enumerate(self.membership):
            if cur_membershp not in group_dic:
                group_dic[cur_membershp] = []
            group_dic[cur_membershp].append(node_id)
        group_list_sorted = (sorted(group_dic.values(), key=lambda x: len(x)))
        self.small_scc = []
        self.largest_scc = group_list_sorted[-1]
        group_list_sorted = group_list_sorted[0:-1]

        for cur_group in group_list_sorted:
            if (len(cur_group) >= 3):
                self.small_scc.append(cur_group)

    def get_scc_edge_encode(self, scc):
        '''
        get edge of the scc, node encoded start from 0
        for conversion, first use local dic then global dic
        '''
        cur_node_id = 0
        self.local_node_id_dic = {}
        self.local_node_id_dic_back = {}
        res = []
        for node_id in scc:
            if node_id not in self.local_node_id_dic:
                self.local_node_id_dic[node_id] = cur_node_id
                self.local_node_id_dic_back[cur_node_id] = node_id
                cur_node_id += 1
        for i in range(len(scc)-1):
            for j in range(i+1, len(scc)):
                if self.check_edge(scc[i], scc[j]):
                    res.append((self.local_node_id_dic[scc[i]], self.local_node_id_dic[scc[j]]))    #res is encoded
        return res, cur_node_id

    def mine_largest_scc(self):
        local_edge_list, local_num_node= self.get_scc_edge_encode(self.largest_scc)
        gn_obj = GirvanNewman(local_edge_list, local_num_node)
        local_membership = gn_obj.run_girvan_newman()
        local_group_dic = {}
        for key in self.local_node_id_dic_back:
            cur_group =  local_membership[key]
            if cur_group not in local_group_dic:
                local_group_dic[cur_group] = []
            local_group_dic[cur_group].append(self.local_node_id_dic_back[key])
        res = list(local_group_dic.values()) + self.small_scc
        return res  #global encoded

    def mine_group_list(self):
        self.check_num_group()
        self.get_group_list()
        self.group_list = self.mine_largest_scc()

    def get_feat_list(self, group):
        cur_fp_obj = FreqPattern(self.cur_ego_network, group, MIN_SUPPORT_RATE)
        res = cur_fp_obj.get_freq_pattern_list()
        return res

    def out_res(self, decoded_group_list):
        fn = "./out/" + str(self.node_id) + ".groups"
        cur_file = open(fn, 'w')
        for cur_group in decoded_group_list:
            cur_str = ""
            first = True
            for node in cur_group:
                if first:
                    cur_str += str(node)
                    first = False
                else:
                    cur_str += ("," + str(node))
            cur_str += "\n"
            cur_file.write(cur_str)
        cur_file.close()

    def mine_all_circle(self):
        self.mine_group_list()
        decoded_group_list = []
        for group in self.group_list:
            decoded_group_list.append([])
            for node in group:
                decoded_group_list[-1].append(self.node_id_dic_back[node])
        self.out_res(decoded_group_list)
        res = []
        for group in decoded_group_list:
            if len(group) <= 2:
                continue
            print(len(group))
            cur_res = self.get_feat_list(group)
            max_pattern_len = sorted(cur_res.keys())[-1]
            max_pattern_list = [x[0] for x in cur_res[max_pattern_len]]
            cur_circle_feature = []
            for cur_feat_list in max_pattern_list:
                cur_circle_feature += cur_feat_list
            cur_circle_featname = [self.featname_list[x] for x in cur_circle_feature]
            res.append(cur_circle_featname)
            print(cur_circle_featname)
        return res


a = Membership(0)
a.mine_all_circle()


# for i in NODE_ID_LIST:
#     print(i)
#     a = Membership(i)
#     a.mine_all_circle()
