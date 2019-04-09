from collections import defaultdict
from typing import Optional


class InputParser(object):

    def __init__(self, data):
        self.data = data # type: Optional[list]

    def get_transition_counts(self):
        transition_pairs = defaultdict(dict)
        start_token = "start/<s>"
        for line in self.data:
            words_with_tag = line.split()
            words_with_tag.insert(0, start_token)
            for i in range(len(words_with_tag)-1):
                pre_tag = words_with_tag[i].split("/")[1]
                post_tag = words_with_tag[i+1].split("/")[1]
                transition_pairs[pre_tag][post_tag] = transition_pairs.get(pre_tag, {}).get(post_tag, 0) + 1

        return transition_pairs

    def get_emission_counts(self):
        corpus = set()
        tags = list()
        emission_pairs = defaultdict(dict)
        for line in self.data:
            words_with_tag = line.split()
            for element in words_with_tag:
                pairs = element.split("/")
                word = pairs[0].lower()
                corpus.add(word)
                tag = pairs[1]
                emission_pairs[tag][word] = emission_pairs.get(tag, {}).get(word, 0) + 1
        return emission_pairs, len(corpus)
