import operator

from hmm_builder import HMMBuilder
from parser import InputParser

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
    emission_counts = parser.get_emission_counts()

    hmm_builder = HMMBuilder(transition_counts, emission_counts)
    transition_probability = hmm_builder.build_transition_probability()
    emission_probability = hmm_builder.build_emission_probability()

