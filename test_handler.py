class TestHandler(object):

    def __init__(self, test_set, viterbi):
        self.test_set = test_set
        self.viterbi = viterbi

    def parse(self):
        num_of_true = 0
        num_of_words = 0
        for line in self.test_set:
            sentence = list()
            only_tag_sequence = list()
            words_with_tag = line.split()
            for element in words_with_tag:
                pairs = element.split("/")
                word = pairs[0].lower()
                tag = pairs[1]
                sentence.append(word)
                only_tag_sequence.append(tag)

            self.viterbi.run(sentence)
            l = self.viterbi.backtracking()
            for i in range(len(sentence)):
                if only_tag_sequence[i] == l[i]:
                    num_of_true += 1
                    num_of_words += 1
                else:
                    num_of_words += 1

        print(num_of_true, num_of_words)
        print((100*num_of_true) / num_of_words)

        # print(sentence)
        # print(l)
        # print(only_tag_sequence)
        # print()
