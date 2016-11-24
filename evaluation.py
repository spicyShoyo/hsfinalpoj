class Evaluation:
    def __init__(self, circle_list_detected, network):
        self.circle_list_detected = circle_list_detected
        self.network = network
        self.dic = {}

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
