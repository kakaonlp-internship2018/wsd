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
    EPOCHS = 100
    BATCH_SIZE = -1

#############
#   class   #
#############


class MyLSTM(nn.Module):
    """
    My Bi-LSTM class.
    """

    def __init__(self, embedding_dim, hidden_dim, output_size, batch_size):
        super(MyLSTM, self).__init__()

        ###### init embedding layer
        with open(VECTORS, 'rb') as fr_vectors:
            glove = pickle.load(fr_vectors)

            # index of <unk> = 0, index of <pad> = 1
            key_list = ["<unk>"]+["<pad>"]+list(glove.keys())
            padding_idx = 1

            # key = word, value = index
            self.glove_index_dic = make_index_dic(key_list)

            matrix_len = len(self.glove_index_dic)
            w_matrix = np.zeros((matrix_len, embedding_dim))
            for key, value in self.glove_index_dic.items():
                if value == padding_idx:
                    continue
                w_matrix[value] = glove[key]

            self.emb_layer = nn.Embedding(matrix_len, embedding_dim, padding_idx=padding_idx)
            # initialize weight as pretrained glove model
            self.emb_layer.load_state_dict({'weight': torch.FloatTensor(w_matrix)})
            # glove emb_layer must not be trained.
            self.emb_layer.weight.requires_grad = False

        # self.batch_size = batch_size

        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden_dim = hidden_dim
        # self.hidden = self.init_hidden(batch_size)
        self.hidden = self.init_hidden(1)
        self.fcl = nn.Linear(hidden_dim * 2, output_size)

    def init_hidden(self, batch_size):
        """
        initialize hidden states
        every iterations, it is called. (to let model know that new sequence starts)
        """
        return (torch.zeros(2, batch_size, self.hidden_dim),
                torch.zeros(2, batch_size, self.hidden_dim))
        # initial value should be zeros? what about randn?

    def forward(self, batch, batch_size, length_list, target_index_list):
        """
        forward input sequence to get vectors for classification
        Args:
            batch: (full) batch of input sequences
            batch_size: batch size
            length_list: length of each sequence
            target_index: index of each target word in sentence
        Returns:
            top level output vector of the model,
            dimension = batch_size * the number of sense(class)
        """
        # length_list is sorted in decreasing order.
        max_seq_len = length_list[0]

        # every iterations, hidden state of lstm should be initialized.
        # (to let model know that new sequence starts)
        self.hidden = self.init_hidden(batch_size)

        # (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        embeds = self.emb_layer(batch)

        # pack input to make padding hidden in lstm
        packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, length_list,
                                                                batch_first=True)

        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim*2)
        packed_bi_lstm_out, self.hidden = self.bi_lstm(packed_embeds, self.hidden)

        # undo the packing operation
        bi_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_bi_lstm_out,
                                                                batch_first=True)
        # for memory arrangement
        bi_lstm_out = bi_lstm_out.contiguous()

        # (batch_size, seq_len, hidden_dim*2) -> (batch_size * seq_len, hidden_dim*2)
        bi_lstm_out = bi_lstm_out.view(-1, bi_lstm_out.shape[2])

        # (batch_size * seq_len, hidden_dim*2) -> (batch_size * seq_len, output_dim)
        sense_space = self.fcl(bi_lstm_out)

        # (batch_size * seq_len, output_dim) -> (batch_size, seq_len, output_dim)
        sense_space = sense_space.view(batch_size, max_seq_len, -1)

        result = []
        # popping output vectors of target words
        for idx, target_index in enumerate(target_index_list):
            result.append(sense_space[idx][target_index])

        # result: (batch_size, output_dim)
        result = torch.stack(result)
        return torch.FloatTensor(result)


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

    sentence_list, answer_list, target_index_list = zip(*training_set)

    # make batch from training instances
    batch, length_list = instances_to_batch(sentence_list, model.glove_index_dic)
    batch_size = len(sentence_list)

    current_loss = 0
    for epoch in range(epochs):
        model.zero_grad()

        # scores: (batch_size, sense_dim)
        scores = model(batch, batch_size, length_list, target_index_list)

        loss = loss_function(scores, torch.tensor(answer_list))
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

        sentence_list, answer_list, target_index_list = zip(*test_set)

        # make batch from test instances
        batch, length_list = instances_to_batch(sentence_list, model.glove_index_dic)
        batch_size = len(length_list)

        # scores: (batch_size, sense_dim)
        scores = model(batch, batch_size, length_list, target_index_list)
        # predictions: list of index of maximum element
        _, predictions = torch.max(scores, 1)

        for idx, answer in enumerate(answer_list):
            if answer == predictions[idx].item():
                correct = correct + 1

    print(total, "instances were tested,", correct, "answers were correct")
    try:
        print("Accuracy: ", (correct/total) * 100, "%")
    except ZeroDivisionError:
        print("Accuracy: 0.0 %")


def instances_to_batch(sentence_list, glove_index_dic):
    """
    sentence_list, glove_index_dic -> padded_index_batch, length_list
    Args:
        sentence_list: list of word list [["BOS", "my/POS", "name/POS", "is/POS", ...], [...],...]
        glove_index_dic: key=word, value=index

    Returns:
        padded_index_batch:
            [[3, 5, 3, 2, 3, 4, 2, 3]
             [2, 7, 8, 8, 1, 1, 1, 1]
             [3, 2, 1, 1, 1, 1, 1, 1]
             [8, 1, 1, 1, 1, 1, 1, 1]], 1 is index of <pad>
        length_list: length of each sentence(except padding)
    """

    index_list = []
    length_list = []
    #batch size : full batch, all instances for the word
    batch_size = len(sentence_list)
    for sentence in sentence_list:
        indices = list(map(lambda word: glove_index_dic.get(word, 0), sentence))
        # if there is no word in dictionary, its index is 0 (unk)
        index_list.append(indices)
        length_list.append(len(indices))
    longest_length = max(length_list)

    # 1 = index of <pad>
    padded_matrix = np.ones((batch_size, longest_length))

    # fill the matrix with index
    for idx, sentence_len in enumerate(length_list):
        indices = index_list[idx]
        padded_matrix[idx, 0:sentence_len] = indices[:sentence_len]

    # return full batch and length_list
    return torch.LongTensor(padded_matrix), length_list

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
    # sort for pack_padded_sequence()
    training_set.sort(key=lambda y: len(y[0]), reverse=True)

    # define model
    model = MyLSTM(EMBEDDING_DIM, HIDDEN_DIM, how_many_senses, BATCH_SIZE)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # train model
    train(model, EPOCHS, loss_function, optimizer, training_set, print_count=20)

    #### test part ####
    # prepare resources for test
    single_word_test_set, test_answer_set, test_target_index_list = to_sentence_answer_index_set(
        TARGET_WORD, TEST_SET)
    test_answer_set = list_to_index_list(test_answer_set, answer_index_dic)
    test_set = list(zip(single_word_test_set, test_answer_set, test_target_index_list))

    # sort for pack_padded_sequence()
    test_set.sort(key=lambda y: len(y[0]), reverse=True)
    # test model
    evaluate(model, test_set)

if __name__ == '__main__':
    main()
