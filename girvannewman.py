import numpy as np
import random
from egonetwork import EgoNetwork

'''
Not done yet
'''
class BetweenessAppro:
    def __init__(self, network, num_node):
        self.network = network
        self.num_node = num_node
        self.new_bet_dic = {}
        self.took = {}
        self.done = {}

    def bfs(self, cur_node):
        q = [cur_node]
        degree = {}
        degree[cur_node] = 1
        parent = {}
        child = {}
        level = {}
        child[cur_node] = []
        level[cur_node] = 0
        parent[cur_node] = [-1]
        level_list = {}
        level_list[0] = [cur_node]
        score = {}
        score[0] = 1
        cur_level = 0
        bet_dic = {}
        while len(q) != 0:
            i = q.pop(0)
            if level[i] == cur_level:
                cur_level += 1
            if cur_level not in level_list:
                level_list[cur_level] = []

            if i not in self.network.adj_dic:
                print("No edge: ", i)
                continue

            for j in self.network.adj_dic[i]:
                if j not in level:
                    level_list[cur_level].append(j)
                    level[j] = cur_level
                    parent[j] = [i]
                    child[j] = []
                    child[i].append(j)
                    q.append(j)
                    degree[j] = degree[i]
                    score[j] = 1
                elif level[j] == cur_level:
                    parent[j].append(i)
                    child[i].append(j)
                    degree[j] += degree[i]
                else:
                    continue
        if len(level_list[cur_level]) == 0:
            del level_list[cur_level]
            cur_level -= 1

        for l in range(cur_level, 0, -1):
            cur_nodes = level_list[l]
            for cur_node in cur_nodes:
                for child_node in child[cur_node]:
                    cur_edge = frozenset((cur_node, child_node))
                    score[cur_node] += bet_dic[cur_edge]
                sum_degree = 0
                for parent_node in parent[cur_node]:
                    sum_degree += degree[parent_node]
                for parent_node in parent[cur_node]:
                    cur_edge = frozenset((parent_node, cur_node))
                    if cur_edge not in bet_dic:
                        bet_dic[cur_edge] = 0
                    bet_dic[cur_edge] += score[cur_node] * degree[parent_node] / sum_degree
        for e in bet_dic:
            if e not in self.new_bet_dic:
                self.new_bet_dic[e] = bet_dic[e]
                if self.new_bet_dic[e] < 5 * self.num_node:
                    self.done[e] = 1
            else:
                self.new_bet_dic[e] +=bet_dic[e]
                if e in self.done:
                    if self.new_bet_dic[e] >= 5 * self.num_node:
                        del self.done[e]

    def sample(self):
        res = random.randint(0, self.num_node - 1)
        while res in self.took:
            res = random.randint(0, self.num_node - 1)
        self.took[res] = 1
        return res

    def calc_bet(self):
        for k in range(self.num_node // 10):   #k = |V|/10
            self.bfs(self.sample())
            if len(self.done) == 0:
                break   #delta(e) >=c|V|

        cur_best = 0
        best_key = None
        for key in self.new_bet_dic:
            if self.new_bet_dic[key] > cur_best:
                cur_best = self.new_bet_dic[key]
                best_key = key
        return list(best_key)


class GirvanNewman:
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
        self.fix()

    def fix(self):
        cur_file = open(self.edge_file_name, 'r')
        for cur_line in cur_file:
            i, j = [int(k) for k in cur_line.strip('\n').split(' ')]
            self.add_edge(i - 1, j - 1)

    def out_cur_edges(self, method):
        fn = "out/" + str(method) + " " + str(self.num_group) + " " + self.fn
        out_file = open(fn, 'w')
        for key in self.betweeness:
            i, j = list(key)
            s = str(i) + " " + str(j) + "\n"
            out_file.write(s)
        out_file.close()
        return

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

    def get_circle_feat(self):
        # self.check_num_group()
        # cirlce_dic = {}
        # for node_id, cur_membershp in enumerate(self.membership):
        #     if cur_membershp not in cirlce_dic:
        #         cirlce_dic[cur_membershp] = []
        #     cirlce_dic[cur_membershp].append(node_id - 1)   #because no 0 node
        pass

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

    def out_res(self, method, modularity_scores, past):
        out_file = open("res/" + "res" + str(method) + " " + self.fn, 'w')
        for i in modularity_scores:
            s = "rm: " + str(i[0]) + " mod: " + str(i[1]) + "\n"
            out_file.write(s)
        out_file.write(str(past) + " secs")
        out_file.close()
        return

    def run_girvan_newman(self, method=1):
        modularity_scores = []
        modularity_scores.append((self.num_edge_removed, self.get_modularity()))
        for i in range(4):
            while self.check_num_group():
                self.remove_highest_edge(method)
                self.num_edge_removed += 1
                if self.num_edge_removed % 25 == 0:
                    print(self.num_edge_removed, "edge removed")
            modularity_scores.append((self.num_edge_removed, self.get_modularity()))
            print("gourp num: ", self.num_group, )
        return modularity_scores

a = GirvanNewman(0)
a.run_girvan_newman()
