from egonetwork import EgoNetwork
from freqpattern import FreqPattern

MIN_CIRCLE_LEN = 6  #magic number

class CircleMiner:
    def __init__(self, node_id, min_support_rate):
        self.cur_ego_network = EgoNetwork(node_id)
        self.min_support_rate = min_support_rate
        self.featname_list = {} #remove "anonymized feature i"
        for key in self.cur_ego_network.featname_list:
            self.featname_list[key] = '-'.join(self.cur_ego_network.featname_list[key].split(';')[:-1])


    def mine_all_circle(self):
        for cur_circle_id in range(len(self.cur_ego_network.circle_list)):
            if len(self.cur_ego_network.circle_list[cur_circle_id]) > MIN_CIRCLE_LEN:
                cur_res = self.mine_circle(cur_circle_id)
                max_pattern_len = sorted(cur_res.keys())[-1]
                max_pattern_list = [x[0] for x in cur_res[max_pattern_len]]
                cur_circle_feature = []
                for cur_feat_list in max_pattern_list:
                    cur_circle_feature += cur_feat_list
                cur_circle_featname = [self.featname_list[x] for x in cur_circle_feature]
                print(cur_circle_featname)

    def mine_circle(self, circle_id):
        cur_fp_obj = FreqPattern(self.cur_ego_network, circle_id, self.min_support_rate)
        res = cur_fp_obj.get_freq_pattern_list()
        return res


# a = CircleMiner(0, 0.5)
# a.mine_all_circle()
