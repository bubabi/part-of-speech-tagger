from collections import defaultdict


class HMMBuilder(object):

    def __init__(self, transition_counts, emission_counts):
        self.transition_counts = transition_counts
        self.emission_counts = emission_counts

    def get_tag_count(self, tag):
        return sum(self.transition_counts[tag].values())

    def get_only_once_words(self):
        once_words = dict()
        for k, v in self.emission_counts.items():
            for word, count in v.items():
                if count == 1:
                    try:
                        once_words[k] += 1
                    except:
                        once_words[k] = 1
        once_words['Punc'] = 0
        return once_words

    def build_transition_probability(self):
        transition_probability = defaultdict(dict)
        for pre_tag, post_tag_counts in self.transition_counts.items():
            pre_tag_count = self.get_tag_count(pre_tag)
            for post_tag, count in post_tag_counts.items():
                transition_probability[pre_tag][post_tag] = self.transition_counts[pre_tag][post_tag] / pre_tag_count
        return transition_probability

    def build_emission_probability(self):
        emission_probability = defaultdict(dict)
        for tag, word_counts in self.emission_counts.items():
            for word in word_counts.keys():
                emission_probability[tag][word] = self.emission_counts[tag][word] / sum(self.emission_counts[tag].values())
        return emission_probability
