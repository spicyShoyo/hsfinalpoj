import numpy as np
data_dir_ = "./facebook/"

def get_adj_list(node_id):
    node_ids = np.loadtxt(data_dir_ + str(node_id) + ".feat", dtype=int)[:, 0]
    adj_list = {i : [] for i in node_ids}
    f = open(data_dir_ + str(node_id) + ".edges")
    for line in f:
        vertices = map(int, line.rstrip("\n").split(" "))
        adj_list[vertices[0]].append(vertices[1])
    return adj_list        
   
def get_all_features(node_ids):
    return list(reduce(set.union, ([set([l[l.index(' ') + 1 : l.rindex(';')]
                        for l in open(data_dir_ + str(node_id) + ".featnames")])
                        for node_id in node_ids])))
    
def project_features(node_id, feature_schema, examples):
    examples = np.c_[examples, np.zeros(len(examples), dtype=int)]
    features = [line[line.index(' ') + 1 : line.rindex(';')]
                for line in open(data_dir_ + str(node_id) + ".featnames")]
    ego_fmat = np.loadtxt(data_dir_ + str(node_id) + ".egofeat", dtype=int)
    on_features = {features[i] : i for i in range(len(ego_fmat)) if ego_fmat[i]}
    indices = {i : (-1 if feature_schema[i] not in on_features
                       else on_features[feature_schema[i]])
               for i in range(len(feature_schema))}
    return np.hstack([examples[:,indices[i]].reshape(len(examples), 1)
                     for i in range(len(feature_schema))])

def generate_training_data(node_id, feature_schema):
    fmat = np.loadtxt(data_dir_ + str(node_id) + ".feat", dtype=int)
    ego_fmat = np.loadtxt(data_dir_ + str(node_id) + ".egofeat", dtype=int)
    examples = project_features(node_id, feature_schema,
                                np.bitwise_and(fmat[:, 1:], ego_fmat))
    circles = [map(int, line.rstrip('\n').split("\t")[1:])
               for line in open(data_dir_ + str(node_id) + ".circles")]
    labels = np.c_[np.zeros((len(examples), len(circles)), dtype=int),
                   np.ones(len(examples), dtype=int)]
    id_to_index = {fmat[i][0] : i for i in range(len(fmat))}
    for circle_id in range(len(circles)):
        for node_id in circles[circle_id]:
            labels[id_to_index[node_id]][circle_id] = 1
            labels[id_to_index[node_id]][-1] = 0
    return zip(examples, labels)

def generate_testing_data(node_id, feature_schema):
    examples = generate_training_data(node_id, feature_schema)
    fmat = np.loadtxt(data_dir_ + str(node_id) + ".feat", dtype=int)
    return zip(fmat[:, 0], examples)
