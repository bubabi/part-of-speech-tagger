import numpy as np


class Viterbi(object):
    def __init__(self, state_size, transition_probs, emission_probs, tag_labels):
        self.state_size = state_size
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
        self.tag_labels = tag_labels
        self.backpointers = None
        self.last_tag = None
        self.sentence_size = 0

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

        for count, q in enumerate(tags):
            if q == "<s>": continue
            path[count, 0] = self.transition_probs['<s>'].get(q, 0) * self.emission_probs[q].get(sentence[0], 0)
            backpointers[count, 0] = 0

        for t in range(1, self.sentence_size):
            for count, q in enumerate(tags):
                path[count, t] = np.max(
                    [path[countp, t - 1] * self.transition_probs[qp].get(q, 0) * self.emission_probs[q].get(sentence[t], 0)
                     for countp, qp in enumerate(tags)]
                )

                # print("tag: ", q, path[count, t])

                backpointers[count, t] = np.argmax(
                    [path[countp, t - 1] * self.transition_probs[qp].get(q, 0) for countp, qp in enumerate(tags)]
                )
            # print()

        last_tag = np.argmax([path[q, self.sentence_size - 1] for q in range(len(tags))])

        self.last_tag = last_tag
        self.backpointers = backpointers
