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

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


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
    PARSER.add_argument('TEST_SET', type=str, metavar='Test_set',
                        help='Transformed test set file')
    PARSER.add_argument('--dim', type=int, default=200,
                        help='Word embedding dimension, default=200')
    PARSER.add_argument('--target', type=str, default='관/NNG',
                        help='Specify target word for disambiguation, default=관/NNG')
    ARGS = PARSER.parse_args()

    # set global variables
    TRAIN_SET = ARGS.TRAIN_SET
    TEST_SET = ARGS.TEST_SET
    EMBEDDING_DIM = ARGS.dim
    TARGET_WORD = ARGS.target
    HIDDEN_DIM = 100
    EPOCHS = 20
    BATCH_SIZE = 50

#############
#   class   #
#############


class MyLSTM(nn.Module):
    """
    My Bi-LSTM class.
    """

    def __init__(self, embedding_dim, hidden_dim, output_size, batch_size):
        super(MyLSTM, self).__init__()

        # init embedding layer
        with open(VECTORS, 'rb') as fr_vectors:
            glove = pickle.load(fr_vectors)

            # index of <unk> = 0, index of <pad> = 1
            key_list = ["<unk>"]+["<pad>"]+list(glove.keys())
            padding_idx = 1

            # key = word, value = index
            self.index_dic = make_index_dic(key_list)

            matrix_len = len(self.index_dic)
            w_matrix = np.zeros((matrix_len, embedding_dim))
            for key, value in self.index_dic.items():
                if value == padding_idx:
                    continue
                w_matrix[value] = glove[key]

            self.emb_layer = nn.Embedding(matrix_len, embedding_dim, padding_idx=padding_idx)
            # initialize weight as pretrained glove model
            self.emb_layer.load_state_dict({'weight': torch.FloatTensor(w_matrix)})
            # glove emb_layer must not be trained.
            self.emb_layer.weight.requires_grad = False

        self.batch_size = batch_size

        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden_dim = hidden_dim
        self.hidden = self.init_hidden()
        self.fcl = nn.Linear(hidden_dim * 2, output_size)

    def init_hidden(self):
        """
        initialize hidden states
        """
        return (torch.zeros(2, 1, self.hidden_dim), torch.zeros(2, 1, self.hidden_dim))
        # initial value should be zeros? what about randn?

    # input sentence should be formed like "BOS My/POS name/POS is/POS jeff/POS EOS"
    def sentence_to_index(self, sentence):
        """
        Args:
            sentence: word list, containing target word,
                formed like "BOS My/POS name/POS is/POS jeff/POS EOS"
        Returns:
            index list
        """

        result = []
        for word in sentence:
            result.append(self.index_dic.get(word, 0))
            # if there no word in dictionary, its index is 0 (unk)

        return torch.LongTensor(result)

    def forward(self, sentence, target_index):
        """
        forward input sequence to get a vector for classification
        Args:
            sentence: word list, containing target word,
                formed like "BOS My/POS name/POS is/POS jeff/POS EOS"
            target_index: index of target word in sentence
        Returns:
            top level output vector of the model, dimension = the number of sense(class)
        """
        index_list = self.sentence_to_index(sentence)
        embeds = self.emb_layer(index_list)
        bi_lstm_out, self.hidden = self.bi_lstm(
            embeds.view(len(embeds), 1, -1), self.hidden)
        sense_space = self.fcl(bi_lstm_out.view(len(embeds), -1))
        return sense_space[target_index].view(1, -1)


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


# maybe not used
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

# maybe not used
def make_index_list(input_list):
    """
    Args:
        input_list: some list
    Returns:
        list of index of each elements
    """

    input_set = list(set(input_list))
    input_set.sort()
    print(input_set)
    result = []
    for elt in input_list:
        result.append(input_set.index(elt))

    return result


def to_sentence_answer_index_set(target_word, path):
    """
    Args:
        target_word: target word for make training set and answer set
        path: file path of txt file
    Returns:
        tuple of training sentences, answers and target_index list.
        training sentences is formed like [["BOS", 가/NNG, 나/NNG, "EOS"],["BOS", ...],].
        It is word list of a sentences and contains target word

        answer set is formed like ['01', '03', '09', '13', ...]
        sense nubmers of answer senses.

        index_list is list of target word index.
        [9, 2, 3, 15, ....]
    """
    sentences = []
    answers = []
    target_index = []
    count = 0
    with open(path, 'r') as fr_txt:
        print("make sentence set and answer set from ", path)
        for line in fr_txt:
            line = line.rstrip()
            line_without_sense = re.sub(r'__[\d][\d]', '', line).split()
            if target_word not in line_without_sense:  # there is no target word in the sentence
                continue
            tokens = line.split()
            for index, token in enumerate(tokens):
                if TKN_PTN.match(token) and re.sub(r'__[\d][\d]', '', token) == target_word:
                    # temp.pop(index)
                    #temp[index] = "<target>"
                    sense = get_sense_number(token)
                    sentences.append(['BOS']+line_without_sense+['EOS'])
                    answers.append(sense)
                    target_index.append(index+1)
                    count = count + 1
                    if count % 100 == 0:
                        print(count, " instances were made")

    print("working done, total ", count, " instances were made")

    return sentences, answers, target_index


def train(model, epochs, loss_function, optimizer, training_set, print_count=1):
    """
    Arg:
        model: wsd model
        epochs: training epochs
        loss_function: loss_function of model
        optimizer: optimizer of model
        training_set: tuple list of (sentence, answer, target_index).
            result of to_sentence_answer_index_set() function
        print_count: every print_count epochs, current loss will be printed
    Returns:
        model trained
    """

    current_loss = 0
    for epoch in range(epochs):
        for sentence, answer, target_index in training_set:
            model.zero_grad()
            model.hidden = model.init_hidden()

            score = model(sentence, target_index)

            loss = loss_function(score, torch.tensor(answer).view(1))
            current_loss = current_loss + loss.item()

            loss.backward()
            optimizer.step()

        if epoch % print_count == 0:
            print("epoch: ", epoch, "current_loss: ", current_loss)
            current_loss = 0


def make_index_dic(input_list):
    """
    Arg:
        input_list: some list
    Return:
        dictionary, key = elements of input list, value = index of each element.
        {'01': 0, '03': 1, '13': 2, ...}
    """
    result = {}
    index = 0
    for elt in input_list:
        if result.get(elt, -1) == -1:
            result[elt] = index
            index = index + 1
    return result


def list_to_index_list(input_list, index_dic):
    """
    Args:
        input_list: some list
        index_dic: index dictionary for input list, it can be made using make_index_dic() function
    Returns:
        list of index
    """
    return map(lambda y: index_dic.get(y, -1), input_list)


def evaluate(model, test_set):
    """
    Args:
        model: wsd model
        test_set: tuple list of (test_sentence, answer, target_index).
            result of to_sentence_answer_index_set() function
    Returns:
        prints result of evaluation
    """
    with torch.no_grad():
        total = len(test_set)
        correct = 0
        for sentence, answer, target_index in test_set:
            score = model(sentence, target_index)
            _, prediction = torch.max(score, 1)
            if answer == prediction.item():
                correct = correct + 1

    print(total, "instances were tested,", correct, "answers were correct")
    try:
        print("Accuracy: ", (correct/total) * 100, "%")
    except ZeroDivisionError:
        print("Accuracy: 0.0 %")



def main():
    """
    this is main function
    """

    #### training part ####
    # prepare resources for training
    single_word_trn_set, answer_set, target_index_list = to_sentence_answer_index_set(
        TARGET_WORD, TRAIN_SET)
    answer_index_dic = make_index_dic(answer_set)  # It is reused in evaluation
    answer_set = list_to_index_list(answer_set, answer_index_dic)
    # it is equal to the number of sense class of the word
    how_many_senses = len(answer_index_dic.keys())
    training_set = list(
        zip(single_word_trn_set, answer_set, target_index_list))

    # define model
    model = MyLSTM(EMBEDDING_DIM, HIDDEN_DIM, how_many_senses, BATCH_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())


    # train model
    train(model, EPOCHS, loss_function, optimizer, training_set, print_count=2)


    #### test part ####
    # prepare resources for test
    single_word_test_set, test_answer_set, test_target_index_list = to_sentence_answer_index_set(
        TARGET_WORD, TEST_SET)
    test_answer_set = list_to_index_list(test_answer_set, answer_index_dic)
    test_set = list(zip(single_word_test_set, test_answer_set, test_target_index_list))

    # test model
    evaluate(model, test_set)


if __name__ == '__main__':
    main()
