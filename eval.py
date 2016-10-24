from zliu import run_all, run_all_except
from egonetwork import EgoNetwork

NODE_ID_LIST = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]

class evaluation:
    def __init__(self, res, node_id):
        self.res = res
        self.network = EgoNetwork(node_id)
        self.feat_dic = {self.network.featname_list[key]: key for key in self.network.featname_list}
        self.feat_set = {key: frozenset([i for i, x in enumerate(self.network.node_feat[key]) if x != 0])for key in self.network.node_feat}
        self.circle_list_detected = []
        self.dic = {}

    def eval(self):
        for item_set in self.res:
            flag = True
            dic = {}
            cur_circle = []
            for item in item_set:
                if item not in self.feat_dic:
                    flag = False
                    break
                item_idx = self.feat_dic[item]
                for node in self.feat_set:
                    if node in dic:
                        continue
                    if item_idx not in self.feat_set[node]:
                        dic[node] = False
            if not flag:
                continue
            for node in self.feat_set:
                if node not in dic:
                    cur_circle.append(node)
            self.circle_list_detected.append(cur_circle)
        return

    def get_jaccard(self, a, b):
        if a in self.dic:
            if b in self.dic[a]:
                return self.dic[a][b]
        anb = a.intersection(b)
        res = len(anb) / (len(a) + len(b) - len(anb))
        if a not in self.dic:
            self.dic[a] = {}
        if b not in self.dic:
            self.dic[b] = {}
        self.dic[a][b] = res
        self.dic[b][a] = res
        return res

    def get_score(self):
        detected = [frozenset([j for j in x]) for x in self.circle_list_detected]
        truth = [frozenset([j for j in x]) for x in self.network.circle_list]

        first = 0
        for d in detected:
            cur = -1
            for t in truth:
                cur = max(cur, self.get_jaccard(d, t))
            first += cur
        first = 1.0 / (2.0 * len(detected)) * first

        second = 0
        for t in truth:
            cur = -1
            for d in detected:
                cur = max(cur, self.get_jaccard(t, d))
            second += cur
        second = 1.0 / (2.0 * len(truth)) * second

        res = first + second
        return res

s = 0
for i in NODE_ID_LIST:
    res = run_all_except(i)
    a = evaluation(res, i)
    a.eval()
    res = a.get_score()
    print("node", i, ":", res)
    s += res

print("avg: ", s / 10.0)
