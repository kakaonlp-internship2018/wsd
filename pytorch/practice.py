"""
pytorch practice
__author__ = 'jeff.yu (jeff.yu@kakaocorp.com)'
__copyright__ = 'No copyright, just copyleft!'
"""

###########
# imports #
###########
import re
import sys
import pickle
import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim


if __name__ == '__main__':
    # FILE PATH #
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    VECTORS = os.path.join(THIS_FOLDER, 'vectors.bin')

    # define regex pattern
    TKN_PTN = re.compile(r'.*__[\d][\d].*')

    # define LOGGER
    PROGRAM = os.path.basename(sys.argv[0])
    LOGGER = logging.getLogger(PROGRAM)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)

    # define argparser
    PARSER = argparse.ArgumentParser(description='This is resource builder.')
    PARSER.add_argument('TRAIN_SET', type=str, metavar='Training_set',
                        help='Transformed training set file')
    PARSER.add_argument('--dim', type=int, default=200,
                        help='Word embedding dimension, default=200')
    PARSER.add_argument('--target', type=str, default='관/NNG',
                        help='Specify target word for disambiguation, default=관/NNG')
    ARGS = PARSER.parse_args()

    # set global variables
    TRAIN_SET = ARGS.TRAIN_SET
    EMBEDDING_DIM = ARGS.dim
    TARGET_WORD = ARGS.target
    HIDDEN_DIM = 100

#############
#   class   #
#############


class MyLSTM(nn.Module):
    """
    My Bi-LSTM class.
    """

    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(MyLSTM, self).__init__()
        with open(VECTORS, 'rb') as fr_vectors:
            self.word_embeddings = pickle.load(fr_vectors)

        self.unknown_emb = self.word_embeddings["<unk>"]

        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(hidden_dim * 2, output_size)

    def init_hidden(self):
        return (torch.zeros(2, 1, self.hidden_dim), torch.zeros(2, 1, self.hidden_dim))

    # input sentence should be formed like "BOS My/POS name/POS is/POS jeff/POS EOS"
    def sentence_to_embeds(self, sentence):
        result = []
        for word in sentence:
            result.append(self.word_embeddings.get(word, self.unknown_emb))

        return torch.FloatTensor(result)

    def forward(self, sentence):
        embeds = self.sentence_to_embeds(sentence)
        bi_lstm_out, self.hidden = self.bi_lstm(
            embeds.view(len(embeds), 1, -1), self.hidden)
        sense_space = self.fc(bi_lstm_out.view(len(embeds), -1))
        sense_scores = F.log_softmax(sense_space, dim=1)
        return sense_scores


#############
# functions #
#############

def get_sense_number(word):
    """
    Arg:
        word: word containing sense number (WORD__NN/POS)
    Return:
        sense number of the word (NN)
    """
    return word[word.index("/")-2:word.index("/")]


def make_one_hot_vectors(input_list):
    """
    Args:
        input_list: some list
    Returns:
        list of one hot vector of each elements
    """

    input_set = list(set(input_list))
    input_set.sort()
    print(input_set)
    result = []
    for elt in input_list:
        temp = [0]*len(input_set)
        temp[input_set.index(elt)] = 1
        result.append(temp)

    return result


def make_single_word_trn_set(target_word):
    """
    Args:
        target_word: target word for make training set and answer set
    Returns:
        tuple of training set and answer set.
        training set is formed like [["BOS", 가/NNG, 나/NNG, "EOS"],["BOS", ...],].
        It is word list of a sentences and doesn't contain target word

        answer set is formed like [[0,0,0,1], [0,0,1,0], [0,0,1,0], ...]
        one hot vectors of answer sense.
    """
    sentences = []
    answers = []
    with open(TRAIN_SET, 'r') as fr_trn_set:
        for line in fr_trn_set:
            line = line.rstrip()
            line_without_sense = re.sub(r'__[\d][\d]', '', line).split()
            if target_word not in line_without_sense:  # there is no target word in the sentence
                continue
            tokens = line.split()
            for index, token in enumerate(tokens):
                if TKN_PTN.match(token) and re.sub(r'__[\d][\d]', '', token) == target_word:
                    temp = line_without_sense
                    temp.pop(index)
                    sense = get_sense_number(token)
                    sentences.append(['BOS']+temp+['EOS'])
                    answers.append(sense)

    return sentences, make_one_hot_vectors(answers)


def main():
    """
    this is main function
    """
    single_word_trn_set, answer_set = make_single_word_trn_set(
        TARGET_WORD)  # result : ([word list], [one hot answer list])
    print(single_word_trn_set[:5], answer_set[:5])
    model = MyLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(answer_set[0]))
    # loss_function =
    # optimizer =
    with torch.no_grad():
        inputs = single_word_trn_set[0]
        score = model(inputs)
        print(score)


if __name__ == '__main__':
    main()
