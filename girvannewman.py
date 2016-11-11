import numpy as np
from betweenness import BetweenessExact, BetweenessAppro
FILES = ["Barabasi.txt", "ErdosRenyi.txt", "WattsStrogatz.txt"]

EXPECTED_GROUP_NUM = 5 #magic number

class GirvanNewman:
    def __init__(self, edge_list, num_node):
        self.num_node = num_node
        self.edge_list = edge_list
        self.adj_mat = np.zeros((self.num_node, self.num_node)).astype(np.int32)
        self.adj_dic = {}
        self.betweeness = {}
        self.membership = np.zeros(self.num_node).astype(np.int32)
        self.num_group = 1
        self.m = 0
        self.num_edge_removed = 0
        self.degree = np.zeros(self.num_node).astype(np.int32)
        for cur_edge in self.edge_list:
            i, j = cur_edge
            self.add_edge(i, j)

    def get_modularity(self):
        sum_s = 0
        for s in range(self.num_group):
            sum_i = 0
            for i in range(self.num_node):
                if self.membership[i] != s:
                    continue
                sum_j = 0
                for j in range(self.num_node):
                    if self.membership[j] != s:
                        continue
                    sum_j += self.adj_mat[i][j] - ((self.degree[i] * self.degree[j]) / (2.0 * self.m))
                sum_i += sum_j
            sum_s += sum_i
        return sum_s / (2.0 * self.m)

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

    def remove_edge(self, i, j):
        self.adj_mat[i][j] = 0
        self.adj_mat[j][i] = 0
        del self.adj_dic[i][j]
        del self.adj_dic[j][i]
        self.m -= 1
        self.degree[i] -= 1
        self.degree[j] -= 1
        del self.betweeness[frozenset((i, j))]

    def get_betweeness_exact(self):
        b = BetweenessExact(self, self.num_node)
        res = b.calc_bet()
        return res

    def get_betweeness_appro(self):
        b = BetweenessAppro(self, self.num_node)
        res = b.calc_bet()
        return res

    def remove_highest_edge(self, method):
        if method == 0:
            i, j = self.get_betweeness_exact()
            self.remove_edge(i, j)
            return
        else:
            i, j = self.get_betweeness_appro()
            self.remove_edge(i, j)
            return


    def run_girvan_newman(self, method=1):
        for i in range(EXPECTED_GROUP_NUM):
            while self.check_num_group():
                self.remove_highest_edge(method)
                self.num_edge_removed += 1
                if self.num_edge_removed % 5 == 0:
                    print(self.num_edge_removed, "edge removed")
            print("gourp num: ", self.num_group, )
        return self.membership
