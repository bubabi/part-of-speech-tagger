import operator
import numpy as np
from hmm_builder import HMMBuilder
from parser import InputParser
from viterbi import Viterbi
from test_handler import TestHandler
import random

path = "./metu.txt"

def split_training_data_to_test_data():
    data_set = list()
    for line in open(path):
        data_set.append(line.strip())

    # random.shuffle(data_set)
    return data_set[:3960], data_set[3960:]


if __name__ == '__main__':

    train_set, test_set = split_training_data_to_test_data()

    parser = InputParser(train_set)
    transition_counts = parser.get_transition_counts()
    emission_counts, corpus = parser.get_emission_counts()

    hmm_builder = HMMBuilder(transition_counts, emission_counts)
    transition_probability = hmm_builder.build_transition_probability()
    # emission probabilities were calculated by smoothing manually in the Viterbi class.

    # frequency of words with frequency 1 is only observed once
    once_words = hmm_builder.get_only_once_words()
    # total number of tags
    state_size = len(transition_probability.keys())
    # only tag labels for backtracking
    tag_labels = list(transition_probability.keys())
    # called k or alpha which will be used in add-k smoothing
    alpha = 0.5

    viterbi = Viterbi(state_size, transition_probability, transition_counts,
                        emission_counts, tag_labels, corpus, once_words, alpha)

    test_handler = TestHandler(test_set, viterbi)
    test_handler.parse()
