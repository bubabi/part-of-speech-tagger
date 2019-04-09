from collections import defaultdict


class HMMBuilder(object):

    def __init__(self, transition_counts, emission_counts):
        self.transition_counts = transition_counts
        self.emission_counts = emission_counts

    def get_tag_count(self, tag):
        return sum(self.transition_counts[tag].values())

    def build_transition_probability(self):
        transition_probability = defaultdict(dict)
        for pre_tag, post_tag_counts in self.transition_counts.items():
            pre_tag_count = self.get_tag_count(pre_tag)
            for post_tag, count in post_tag_counts.items():
                #print(pre_tag, post_tag, count)
                #print(self.transition_counts[pre_tag][post_tag])
                #if pre_tag == '<s>': pre_tag_count = 1
                transition_probability[pre_tag][post_tag] = self.transition_counts[pre_tag][post_tag] / pre_tag_count
        return transition_probability

    def build_emission_probability(self):
        emission_probability = defaultdict(dict)
        for tag, word_counts in self.emission_counts.items():
            for word in word_counts.keys():
                emission_probability[tag][word] = self.emission_counts[tag][word] / sum(self.emission_counts[tag].values())
        return emission_probability
