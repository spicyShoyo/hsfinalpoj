from copy import deepcopy
from load_data import get_adj_list
from eval import evaluation
from functools import reduce

def prune_graph(adj_list, min_size):
    removed_nodes = [n for n in adj_list.keys() if len(adj_list[n]) < min_size]
    while len(removed_nodes) > 0:
        node = removed_nodes.pop()
        if node in adj_list:
            neighbors = adj_list.pop(node)
            for n in neighbors:
                adj_list[n].remove(node)
                if len(adj_list[n]) < min_size:
                    removed_nodes.append(n)
    return adj_list

def find_cliques_at_node(node_id, adj_list):
    max_clique, local_max_clique, local_size = (), (), 0
    queue = [(node_id,)]
    # DFS.
    while len(queue) > 0:
        curr_clique = queue.pop()
        if (not set(curr_clique).issubset(set(local_max_clique)) and
            not set(curr_clique).issubset(set(max_clique)) and len(list(filter(
            lambda n: len(adj_list[n]) <= len(max_clique), curr_clique))) == 0):
            expanded_nodes = reduce(set.intersection,
                                    [set(adj_list[n]) for n in curr_clique])
            # Avoid duplication and prune dfs search tree.
            expanded_nodes = list(filter(lambda i: (i > max(curr_clique) and
                  i not in curr_clique and len(adj_list[i]) > len(max_clique)),
                  sorted(expanded_nodes, reverse=True)))
            for n in expanded_nodes:
                queue.append(curr_clique + (n,))
            if len(expanded_nodes) == 0:
                if len(curr_clique) > local_size:
                    local_max_clique, local_size = curr_clique, len(curr_clique)
                else:
                    local_max_clique += curr_clique
                if len(curr_clique) > len(max_clique):
                    max_clique = curr_clique
    return tuple(sorted(max_clique))

def find_cliques(adj_lists, min_size=4):
    """
        adj_lists: a dictionary with node id as key and
                   a list of neighbor node ids as value
        rtype: a list of tuples, where each tuple is a clique
    """
    adj_lists_pruned = prune_graph(deepcopy(adj_lists), min_size)
    cliques = []
    for nid in adj_lists_pruned:
        clique = find_cliques_at_node(nid, adj_lists_pruned)
        m = []
        for i in cliques:
            m.append(set(i))
        if (len(clique) >= min_size and
            len(list(filter(set(clique).issubset, m)))==0):
            cliques.append(clique)
    return cliques

def run():
    node_ids = [107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
    jaccard_similarity = []
    for nid in node_ids:
        evl = evaluation([], nid)
        m = []
        l = find_cliques(get_adj_list(nid))
        for i in l:
            m.append(list(i))
        evl.circle_list_detected = m
        print(find_cliques(get_adj_list(nid)))
        jaccard_similarity.append(evl.get_score())
        print(nid, jaccard_similarity[-1])
    print("result:", jaccard_similarity)
    print("Average:", sum(jaccard_similarity) / len(jaccard_similarity))

run()
