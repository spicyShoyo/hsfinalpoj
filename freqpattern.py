from egonetwork import EgoNetwork
from apriori import Apriori

class FreqPattern:
    def __init__(self, ego_network, cur_circle, min_support_rate):
        self.min_support_rate = min_support_rate
        self.item_list = ego_network.featname_list
        self.transaction_vec = {}
        self.transaction_list = {}
        self.transactions = []
        for transaction_id in cur_circle:
            self.transaction_vec[transaction_id] = ego_network.node_feat[transaction_id]
            self.transaction_list[transaction_id] =  [x for x in range(len(self.transaction_vec[transaction_id])) if self.transaction_vec[transaction_id][x] == 1]
            self.transactions.append(self.transaction_list[transaction_id])

    def get_freq_pattern_list(self):
        cur_apriori = Apriori(self.transactions)
        cur_res = cur_apriori.apriori(self.min_support_rate * len(self.transactions))
        fixed_res = self.fix_result(cur_res)
        return fixed_res

    def fix_result(self, cur_res):
        #should have used set
        res = {}
        for key in cur_res:
            cur_list = []
            cur_dic = {}
            for item_set in cur_res[key]:
                if frozenset(item_set[0]) in cur_dic:
                    cur_dic[frozenset(item_set[0])] += item_set[1]
                else:
                    cur_dic[frozenset(item_set[0])] = item_set[1]
            for kkey in cur_dic:
                cur_list.append((list(kkey), cur_dic[kkey]))
            res[key] = cur_list
        return res


# a = FreqPattern(EgoNetwork(0), 0, 0.5)
# a.get_freq_pattern_list()
