import numpy as np
import sys

from load_data import get_all_features
from load_data import generate_training_data, generate_testing_data

def lms(example, weight_and_theta, step_size):
    return weight_and_theta + step_size * (example[0].reshape(-1, 1).dot(
                              example[1].reshape(1, -1) -
                              example[0].dot(weight_and_theta).reshape(1, -1)))

def hinge_loss(example, weight_and_theta, learning_rate, margin = 1):
    for i in range(len(example[1])):
        if example[1][i] * (weight_and_theta[:, i].dot(example[0])) < margin:
            weight_and_theta[:, i] += learning_rate * example[1][i] * example[0]
    return weight_and_theta

def winnow(example, weight_and_theta, promotion_rate):
    for i in range(len(example[1])):
        weight_and_theta[:, i] *= promotion_rate ** (example[1][i] * example[0])
    return weight_and_theta

def stochastic_gradient_descent(training_examples, update_rule, step_size,
                                max_iteration = 100):
    weight_and_theta = np.random.rand(len(training_examples[0][0]) + 1,
                                      len(training_examples[0][1]))
    for i in range(max_iteration):
        random_order = range(len(training_examples))
        np.random.shuffle(random_order) # randomize the order of example
        for d in random_order:
             weight_and_theta = update_rule(
                (np.r_[training_examples[d][0], -1], training_examples[d][1]),
                weight_and_theta, step_size)
    return weight_and_theta

def learn_circles(training_examples, testing_data, update_rule,
                  margin=0, step_size=1e-3):
    learned_weights = stochastic_gradient_descent(
                                    training_examples, update_rule, step_size)
    max_num_circles = learned_weights.shape[1]
    circles = {i : [] for i in range(max_num_circles)}
    for node_id, example in testing_data:
        circles_in = filter(lambda i: np.r_[example[0], -1].dot(
                        learned_weights)[i] >= margin, range(max_num_circles))
        for c in circles_in:
            circles[c].append(node_id)
    return circles

node_ids = [107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
training_node_ids = [107]
testing_node_ids = [1912]
feature_schema = get_all_features(node_ids)
training_examples = reduce(lambda x, y: x + y,
         [generate_training_data(i, feature_schema) for i in training_node_ids])
testing_data = reduce(lambda x, y: x + y,
         [generate_testing_data(i, feature_schema) for i in testing_node_ids])
#circles = filter(lambda l: len(l) > 0,
#               learn_circles(training_examples, testing_data, lms, 0).values())
circles = filter(lambda l: len(l) > 0,
       learn_circles(training_examples, testing_data, hinge_loss, 1).values())
#circles = filter(lambda l: len(l) > 0,
#    learn_circles(training_examples, testing_data, winnow, 0, 1.0001).values())

from eval import evaluation
evl = evaluation([], testing_node_ids[0])
evl.circle_list_detected = circles
print "Jaccard Simularity:", evl.get_score()
