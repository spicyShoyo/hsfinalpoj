import numpy as np
import random

'''
This class calculate the betweeness of a node
'''


class BetweenessExact:
    def __init__(self, network, num_nodes):
        self.network = network
        self.num_nodes = num_nodes
        self.new_bet_dic = {}

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
            else:
                self.new_bet_dic[e] +=bet_dic[e]

    def calc_bet(self):
        for i in range(self.num_nodes):
            # if i % 500 == 0:
            #     print("betweeness in progress: ", i)
            self.bfs(i)
        cur_best = 0
        best_key = None
        for key in self.new_bet_dic:
            if self.new_bet_dic[key] > cur_best:
                cur_best = self.new_bet_dic[key]
                best_key = key
        print(best_key, cur_best)
        return list(best_key)


class BetweenessAppro:
    def __init__(self, network, num_nodes):
        self.network = network
        self.num_nodes = num_nodes
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
                if self.new_bet_dic[e] < 5 * self.num_nodes:
                    self.done[e] = 1
            else:
                self.new_bet_dic[e] +=bet_dic[e]
                if e in self.done:
                    if self.new_bet_dic[e] >= 5 * self.num_nodes:
                        del self.done[e]

    def sample(self):
        res = random.randint(0, self.num_nodes - 1)
        while res in self.took:
            res = random.randint(0, self.num_nodes - 1)
        self.took[res] = 1
        return res

    def calc_bet(self):
        for k in range(self.num_nodes // 10):   #k = |V|/10
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
