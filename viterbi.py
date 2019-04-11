import numpy as np


class Viterbi(object):
    def __init__(self, state_size, transition_probs, transition_counts, emission_probs, emission_counts, tag_labels, corpus, once_words):
        self.state_size = state_size
        self.transition_probs = transition_probs
        self.transition_counts = transition_counts
        self.emission_probs = emission_probs
        self.emission_counts = emission_counts
        self.tag_labels = tag_labels
        self.backpointers = None
        self.last_tag = None
        self.sentence_size = 0
        self.corpus = corpus
        self.once_words = once_words

    def get_emission_tag_count(self, tag):
        return sum(self.emission_counts[tag].values())

    def get_transition_tag_count(self, tag):
        return sum(self.transition_counts[tag].values())

    def transition_start_with(self, tag_a):
        bigram_starts_with_tag = len([tag_b for tag_b, count in self.transition_counts[tag_a].items() if count > 0])
        return bigram_starts_with_tag

    def transition_end_with(self, tag_b):
        bigram_ends_with_tag = len([count for tag_one, count in self.transition_counts.items() if count.get(tag_b, 0) > 0])
        return bigram_ends_with_tag

    def num_of_transition_combinations(self):
        return len(self.transition_counts) ** 2

    def get_smoothed_transition(self, pre_tag, post_tag):
        times_tag_occurs = self.get_transition_tag_count(pre_tag)
        term_a = max((self.transition_counts[pre_tag].get(post_tag, 0) - 0.75), 0) / times_tag_occurs

        lambda_parameter = 0.75 / times_tag_occurs * self.transition_start_with(pre_tag)
        p_continuation = self.transition_end_with(post_tag) / self.num_of_transition_combinations()

        return term_a + (lambda_parameter * p_continuation)

    def backtracking(self):
        pre_tag = self.last_tag
        tag_list = [pre_tag]

        for m in range(self.sentence_size - 1, -1, -1):
            pre_tag = self.backpointers[pre_tag, m]
            tag_list.append(pre_tag)

        tag_list.reverse()

        return [self.tag_labels[tag] for tag in tag_list[1:]]

    def run(self, sentence):
        tag_indexes = list(range(self.state_size))
        self.sentence_size = len(sentence)
        path = np.zeros(shape=(len(tag_indexes), self.sentence_size + 1))
        backpointers = np.zeros(shape=(len(tag_indexes), self.sentence_size + 1), dtype=np.int)
        tags = self.transition_probs.keys()
        alpha = 0.5

        for count, q in enumerate(tags):
            if q == "<s>": continue
            #print("<s> to", q, self.transition_probs['<s>'].get(q, 0), ">>>>>>>>>>>>> ", sentence[0], "from ", q, self.emission_probs[q].get(sentence[0]))

            if sentence[0] in self.corpus:
                emission = (self.emission_counts.get(q).get(sentence[0], 0)) / \
                        (self.get_emission_tag_count(q))
            else:
                emission = (self.once_words[q]) / (7096*self.get_emission_tag_count(q))

            #transition = self.get_smoothed_transition('<s>', q)
            transition = self.transition_probs['<s>'].get(q, 0)
            path[count, 0] = transition * emission
            backpointers[count, 0] = 0

        for t in range(1, self.sentence_size):

            for count, q in enumerate(tags):
                if q == "<s>": continue

                if sentence[t] in self.corpus:
                    emission = (self.emission_counts.get(q).get(sentence[t], 0)) / \
                            (self.get_emission_tag_count(q))
                else:
                    emission = (self.once_words[q]) / (7096*self.get_emission_tag_count(q))

                if sentence[t] in ["\'", "\"", ".", ",", "?", ":", "..."] and q == "Punc":
                    emission = 1

                if "\'" in sentence[t] and len(sentence[0]) > 1 and q == "Noun":
                    emission = 1

                # transition = self.get_smoothed_transition(qp, q)
                # transition = self.transition_probs[qp].get(q, 0)

                path[count, t] = np.max(
                    [path[countp, t - 1] * self.transition_probs[qp].get(q, 0) * emission for countp, qp in enumerate(tags)]
                )

                backpointers[count, t] = np.argmax(
                    [path[countp, t - 1] * self.transition_probs[qp].get(q, 0) for countp, qp in enumerate(tags)]
                )

        last_tag = np.argmax([path[q, self.sentence_size - 1] for q in range(len(tags))])

        self.last_tag = last_tag
        self.backpointers = backpointers
