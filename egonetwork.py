class EgoNetwork:
    def __init__(self, ego_id):
        self.ego_id = str(ego_id)
        self.circle_list = []
        self.adj_mat = {}
        self.featname_list = {}
        self.node_feat = {}
        self.read_files()

    def read_files(self):
        self.read_circle_list()
        self.read_edge_list()
        self.read_featname_list()
        self.read_feat_list()

    def read_circle_list(self):
        circle_file_name = "facebook/" + self.ego_id + ".circles"
        circle_file = open(circle_file_name, 'r')
        for cur_line in circle_file:
            cur_circle = [int(x) for x in cur_line.strip('\n').split('\t')[1:]]
            self.circle_list.append(cur_circle)

    def read_edge_list(self):
        edge_file_name = "facebook/" + self.ego_id + ".edges"
        edge_file = open(edge_file_name, 'r')
        for cur_line in edge_file:
            node_x, node_y = [int(k) for k in cur_line.strip('\n').split(' ')]
            if node_x in self.adj_mat:
                self.adj_mat[node_x][node_y] = 1
            else:
                self.adj_mat[node_x] = {}
                self.adj_mat[node_x][node_y] = 1
            if node_y in self.adj_mat:
                self.adj_mat[node_y][node_x] = 1
            else:
                self.adj_mat[node_y] = {}
                self.adj_mat[node_y][node_x] = 1

    def read_featname_list(self):
        featname_file_name = "facebook/" + self.ego_id + ".featnames"
        featname_file = open(featname_file_name, 'r')
        for cur_line in featname_file:
            cur_list = cur_line.strip('\n').split(' ')
            featname_id = int(cur_list[0])
            featname = ' '.join(cur_list[1:])
            self.featname_list[featname_id] = featname

    def read_feat_list(self):
        feat_file_name = "facebook/" + self.ego_id + ".feat"
        feat_file = open(feat_file_name, 'r')
        for cur_line in feat_file:
            cur_list = [int(x) for x in cur_line.strip('\n').split(' ')]
            node_id = cur_list[0]
            self.node_feat[node_id] = cur_list[1:]
        #read ego feat
        feat_file_name = "facebook/" + self.ego_id + ".egofeat"
        feat_file = open(feat_file_name, 'r')
        for cur_line in feat_file:
            cur_list = [int(x) for x in cur_line.strip('\n').split(' ')]
            node_id = cur_list[0]
            self.node_feat[node_id] = cur_list[1:]
            break   #should be just one line

#a = EgoNetwork(0)
