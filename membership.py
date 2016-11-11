import numpy as np
from egonetwork import EgoNetwork

NODE_ID_LIST = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]

class Membership:
    def __init__(self, node_id):
        self.edge_file_name = "facebook/" + str(node_id) + ".edges"
        self.cur_ego_network = EgoNetwork(node_id)
        self.num_node = len(self.cur_ego_network.node_feat)
        self.adj_mat = np.zeros((self.num_node, self.num_node)).astype(np.int32)
        self.adj_dic = {}
        self.betweeness = {}
        self.membership = np.zeros(self.num_node).astype(np.int32)
        self.num_group = 1
        self.m = 0
        self.num_edge_removed = 0
        self.degree = np.zeros(self.num_node).astype(np.int32)
        self.node_id_dic = {}
        self.node_id_dic_back = {}
        self.fix()

    def fix(self):
        print(self.num_node)
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

    def add_edge(self, i, j):
            self.adj_mat[i][j] = 1
            self.adj_mat[j][i] = 1
            if i not in self.adj_dic:
                self.adj_dic[i] = {}
            self.adj_dic[i][j] = 1
            if j not in self.adj_dic:
                self.adj_dic[j] = {}
            self.adj_dic[j][i] = 1
            self.m += 1
            self.degree[i] += 1
            self.degree[j] += 1
            self.betweeness[frozenset((i, j))] = 0

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
        if new_num_group != self.num_group:
            self.num_group = new_num_group
            self.membership = new_membership
            return False
        else:
            return True
    '''
    todo:
        1. create potential communities
        2. mine community common features
        3. use ego features
        4. refractor
    '''
    def mine_all_circle(self):
        self.check_num_group()
        group_dic = {}
        for node_id, cur_membershp in enumerate(self.membership):
            if cur_membershp not in group_dic:
                group_dic[cur_membershp] = []
            group_dic[cur_membershp].append(node_id)
        group_count = 0
        largest_scc = (sorted(group_dic.values(), key=lambda x: len(x))[-1])
        group_dic_new = {}
        for group_id in group_dic:
            if len(group_dic[group_id]) >= 2:    #magic number
                print("l:", len(group_dic[group_id]))
                group_dic_new[group_count] = group_dic[group_id]
                group_count += 1
        group_dic = group_dic_new

for i in NODE_ID_LIST:
    print(i)
    a = Membership(i)
    a.check_num_group()
    a.mine_all_circle()
