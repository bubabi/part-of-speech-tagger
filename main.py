import operator

import numpy as np

from hmm_builder import HMMBuilder
from parser import InputParser
from viterbi import Viterbi
from test_handler import TestHandler

path = "./metu.txt"

def split_training_data_to_test_data():
    data_set = list()
    for line in open(path):
        data_set.append(line.strip())

    return data_set[:3960], data_set[3960:]


if __name__ == '__main__':

    train_set, test_set = split_training_data_to_test_data()

    parser = InputParser(train_set)
    transition_counts = parser.get_transition_counts()
    emission_counts, corpus_size = parser.get_emission_counts()

    hmm_builder = HMMBuilder(transition_counts, emission_counts)
    transition_probability = hmm_builder.build_transition_probability()
    emission_probability = hmm_builder.build_emission_probability()

    state_size = len(transition_probability.keys())
    tag_labels = list(transition_probability.keys())
    viterbi = Viterbi(state_size, transition_probability, transition_counts, emission_probability, emission_counts, tag_labels, corpus_size)

    # viterbi.run(["bu", "şekilde", "burak", "geldi"])
    # l = viterbi.backtracking()
    # print(l)
    test_handler = TestHandler(test_set, viterbi)
    test_handler.parse()
