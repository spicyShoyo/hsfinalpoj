from circleminer import CircleMiner


'''
evaluate apriori
'''

MIN_SUPPORT_RATE = 0.7 #magic number
NODE_ID_LIST = [0, 107, 1684, 1912, 3437, 348, 3980, 414, 686, 698]
def mine_node(node_id):
    cur_obj = CircleMiner(node_id, MIN_SUPPORT_RATE)
    cur_res = cur_obj.mine_all_circle()
    res = []
    for cur_feat_list in cur_res:
        dic = {}
        for cur_feat in cur_feat_list:
            if cur_feat in dic:
                continue
            dic[cur_feat] = 1
        if len(dic) > 1:
            res.append(frozenset([x for x in dic]))
    return res

def run_all():
    dic = {}
    for node_id in NODE_ID_LIST:
        res = mine_node(node_id)
        for i in res:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
    res = [list(key) for key in dic if dic[key] > 1]
    return res

def run_all_except(n):
    dic = {}
    for node_id in NODE_ID_LIST:
        if node_id == n:
            continue
        res = mine_node(node_id)
        for i in res:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
    res = [list(key) for key in dic if dic[key] > 1]
    return res

def main():
    cur_res = run_all()
    for i in range(len(cur_res)):
        print("Circle-"+str(i), cur_res[i])

#main()
