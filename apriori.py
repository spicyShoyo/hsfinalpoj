'''
perform Apriori algorithm
'''


class Apriori:
    def __init__(self, transactions):
        self.transactions = transactions
        self.count = {}
        self.items = []
        self.init()


    def init(self):
        for transaction in self.transactions:
            for c in transaction:
                if c in self.count:
                    self.count[c] += 1
                else:
                    self.count[c] = 1
                    self.items.append(c)


    def get_count(self, item_set):
        count = 0
        for transaction in self.transactions:
            flag = True
            for c in item_set:
                if c not in transaction:
                    flag = False
                    break
            if not flag:
                continue
            count += 1
        return count

    def apriori(self, min_support):
        res = {}
        cur_res = []
        new_items = []
        cur_item_sets = [[x] for x in self.items]
        for item_set in cur_item_sets:
            cur_count = self.get_count(item_set)
            if cur_count >= min_support:
                new_items.append(item_set[0])
                cur_res.append((item_set, cur_count))
        self.items = new_items
        res[1] = cur_res

        cur_item_sets = self.get_item_sets(cur_res)
        idx = 2
        while True:
            cur_res = []
            for item_set in cur_item_sets:
                cur_count = self.get_count(item_set)
                if cur_count >= min_support:
                    cur_res.append((item_set, cur_count))
            if len(cur_res) == 0:
                break
            res[idx] = cur_res
            idx += 1
            cur_item_sets = self.get_item_sets(cur_res)
        return res

    def get_item_sets(self, cur_res):
        res = []
        dup = []
        for item_set in cur_res:
            for c in self.items:
                if c not in item_set[0]:
                    cur = (item_set[0] + [])
                    cur.append(c)
                    if cur not in dup:
                        res.append(cur)
                        dup.append(set(cur))
        return res
