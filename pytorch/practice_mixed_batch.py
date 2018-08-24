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
from logger import Logger
import argparse
import math
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim



if __name__ == '__main__':
    # define regex pattern
    TKN_PTN = re.compile(r'.*__[\d][\d].*')

    # define argparser
    PARSER = argparse.ArgumentParser(description='This is resource builder.')
    PARSER.add_argument('TRAIN_SET', type=str, metavar='Training_set',
                        help='Transformed training set file')
    PARSER.add_argument('TEST_SET', type=str, metavar='Test_set',
                        help='Transformed test set file')
    PARSER.add_argument('--dim', type=int, default=200,
                        help='Word embedding dimension, default=200')
    PARSER.add_argument('--batch', type=int, default=256,
                        help='Set batch size, default=256')
    PARSER.add_argument('--epoch', type=int, default=300,
                        help='Set epoch size, default=300')
    PARSER.add_argument('--patience', type=int, default=10,
                        help='Set patience, default=10')
    PARSER.add_argument('--target', type=str, default=None,
                        help='Specify target word for disambiguation, default=None')
    PARSER.add_argument('--gpu', type=int, default=0,
                        help='select GPU device, default=0')
    PARSER.add_argument('--relu', dest='relu', action='store_true', default=False,
                        help='apply relu, default=False')
    PARSER.add_argument('--early', dest='early', action='store_false', default=True,
                        help='apply early stopping, default=True')
    PARSER.add_argument('--dropout', dest='dropout', action='store_true', default=False,
                        help='apply dropout, default=False')
    PARSER.add_argument('--verbose', dest='verbose', action='store_false', default=True,
                        help='Print every target word accuracy, default=True')
    PARSER.add_argument('--data', dest='build_data_set', action='store_true', default=False,
                        help='Build data set, default=False')
    PARSER.add_argument('--nword', type=int, default=-1,
                        help='the number of target word, default=-1')
    PARSER.add_argument('--win', type=int, default=-1,
                        help='window_size, default=-1')
    PARSER.add_argument('--exp', type=str, default='temp_experiment',
                        help='name of experiment, default=temp_experiment')
    PARSER.add_argument('--check', dest='check', action='store_true', default=False,
                        help='Check result, default=False')
    PARSER.add_argument('--best', dest='best', action='store_true', default=False,
                        help='User model that has best validation acc to evaluate, default=False')
    PARSER.add_argument('--weak', type=int, default=-1,
                        help='set weakness, default=-1')    
    ARGS = PARSER.parse_args()

    # set global variables
    TRAIN_SET = ARGS.TRAIN_SET
    TEST_SET = ARGS.TEST_SET
    EMBEDDING_DIM = ARGS.dim
    SINGLE_TARGET_WORD = ARGS.target
    RELU = ARGS.relu
    DROPOUT = ARGS.dropout
    GPU = ARGS.gpu
    BATCH_SIZE = ARGS.batch
    BEST = ARGS.best
    VERBOSE = ARGS.verbose
    EPOCHS = ARGS.epoch
    PATIENCE = ARGS.patience
    EARLY = ARGS.early
    BUILD_DATA_SET = ARGS.build_data_set
    NWORD = ARGS.nword
    WINDOW_SIZE = ARGS.win
    EXP_NAME = ARGS.exp
    CHECK = ARGS.check
    WEAK = ARGS.weak
    HIDDEN_DIM = 128
    HIDDEN2_DIM = 64
    ENTROPY_THRESHOLD = 0.1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU)

# define LOGGER
    LOGGER = Logger('./logs/'+EXP_NAME)

    # FILE PATH #
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    VECTORS = os.path.join(THIS_FOLDER, 'vectors.bin')
    ENTROPY_DIC = os.path.join(THIS_FOLDER, 'entropy_dic.bin')
    DATA_SET = os.path.join(THIS_FOLDER, 'data_set.bin')
    VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
    RESULT_FILE = os.path.join(THIS_FOLDER, 'results/'+EXP_NAME+'.bin')


#############
#   class   #
#############


class MyLSTM(nn.Module):
    """
    My Bi-LSTM class.
    """

    def __init__(self, embedding_dim, hidden_dim, hidden2_dim, output_size_dic):
        super(MyLSTM, self).__init__()

        # initialize layers
        self.embedding_dim = embedding_dim
        self.init_emb()
        self.bi_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim
        self.hidden2_dim = hidden2_dim
        self.hidden = self.init_hidden(1)
        self.hidden2 = set_device(nn.Linear(self.hidden_dim * 2, self.hidden2_dim))
        self.output_size_dic = output_size_dic
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        # init multi output layers
        # super important part for multitask learning
        self.layer_list = nn.ModuleList()
        self.init_fully_connected_layers()

    def init_fully_connected_layers(self):
        """
        initialize multi output layers and
        save as dictionary, key=WORD/POS, value=(fc_layer1, fc_layer2)
        named fc_dic
        """
        self.fc_dic = {}
        for key, value in self.output_size_dic.items():
            new_layer = set_device(nn.Linear(self.hidden2_dim, value))
            self.fc_dic[key] = new_layer

    def init_emb(self):
        """
        initialize embedding layer
        named emb_layer
        in addition, glove_index_dic is also initialized
        """
        with open(VECTORS, 'rb') as fr_vectors:
            glove = pickle.load(fr_vectors)

            # index of <unk> = 0, index of <pad> = 1
            key_list = ["<unk>"]+["<pad>"]+list(glove.keys())
            padding_idx = 1

            # key = word, value = index
            self.glove_index_dic = make_index_dic(key_list)

            # initialize weight matrix of embedding layer
            matrix_len = len(self.glove_index_dic)
            w_matrix = np.zeros((matrix_len, self.embedding_dim))
            for key, value in self.glove_index_dic.items():
                if value == padding_idx:
                    continue
                # because embedding layer works as look up table,
                # a row of matrix is eqaul to embedding vector of row index
                w_matrix[value] = glove[key]

            self.emb_layer = nn.Embedding(matrix_len, self.embedding_dim, padding_idx=padding_idx)
            # initialize weight as pretrained glove model
            self.emb_layer.load_state_dict({'weight': set_device(torch.FloatTensor(w_matrix))})
            # glove emb_layer must not be trained.
            self.emb_layer.weight.requires_grad = False

    def init_hidden(self, batch_size):
        """
        initialize hidden states
        every iterations, it is called. (to let model know that new sequence starts)
        """
        return (set_device(torch.zeros(4, batch_size, self.hidden_dim)),
                set_device(torch.zeros(4, batch_size, self.hidden_dim)))
        # initial value should be zeros? what about randn?

    def reset_layer_list(self):
        """
        reset layer_list
        """
        self.layer_list = nn.ModuleList()

    def forward(self, batch, batch_size, length_list, target_index_list, target_word_list):
        """
        forward input sequence to get vectors for classification
        Args:
            batch: (full) batch of input sequences
            batch_size: batch size
            length_list: length of each sequence
            target_index_list: index of each target word in sentence
            target_word_list: target_word "WORD/POS" list
        Returns:
            list of top level output vector of the model
        """
        # reset parameters to be seen by optimizer
        self.reset_layer_list()

        # length_list is sorted in decreasing order.
        # max_seq_len = length_list[0]

        # every iterations, hidden state of lstm should be initialized.
        # (to let model know that new sequence starts)
        self.hidden = self.init_hidden(batch_size)

        # (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        embeds = self.emb_layer(batch)

        # pack input to make padding hidden in lstm
        if WINDOW_SIZE == -1:
            packed_embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, length_list,
                                                                    batch_first=True)

            # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim*2)
            packed_bi_lstm_out, self.hidden = self.bi_lstm(packed_embeds, self.hidden)

            # undo the packing operation
            bi_lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_bi_lstm_out,
                                                                    batch_first=True)
        else:
            bi_lstm_out, _ = self.bi_lstm(embeds, self.hidden)

        # for memory arrangement
        bi_lstm_out = bi_lstm_out.contiguous()

        # (batch_size, seq_len, hidden_dim*2) -> (batch_size * seq_len, hidden_dim*2)
        bi_lstm_out = bi_lstm_out.view(-1, bi_lstm_out.shape[2])

        # (batch_size * seq_len, hidden_dim*2) -> (batch_size * seq_len, hidden2_dim)
        hidden_out2 = self.hidden2(bi_lstm_out)
        # apply ReLU
        if RELU:
            hidden_out2 = self.relu(hidden_out2)
        # apply dropout
        if DROPOUT:
            hidden_out2 = self.dropout(hidden_out2)

        # (batch_size * seq_len, hidden2_dim) -> (batch_size, seq_len, hidden2_dim)
        hidden_out2 = hidden_out2.view(batch_size, -1, self.hidden2_dim)

        # (batch_size, seq_len, hidden2_dim) -> list of variable dimension output vectors
        # output layer is chosen by target_word
        result = []
        for batch_index in range(batch_size):
            unit_input = hidden_out2[batch_index]
            current_output_layer = self.fc_dic[target_word_list[batch_index]]
            # super important part for multitask learning
            # all layers must be in special data structure ModuleList.
            # so that optimizer can see the layer as parameter.
            # optimizer can not see native python list or dict as parameter.
            self.layer_list.append(current_output_layer)

            # (seq_len, hidden2_dim) -> (seq_len, output_dim)
            unit_output = current_output_layer(unit_input)
            # (seq_len, output_dim) -> (output_dim)
            # popping output vector of target word
            target_result = unit_output[target_index_list[batch_index]]
            result.append(target_result)

        # result: list of variable dimension ouput vectors
        return result


#############
# functions #
#############

def set_device(tensor):
    """
    wrapper function of cuda()
    if cuda is available, set cuda.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
        #return tensor.cuda(GPU)
    return tensor

def get_sense_number(word):
    """
    Arg:
        word: word containing sense number (WORD__NN/POS)
    Return:
        sense number of the word (NN)
    """
    return word[word.index("/")-2:word.index("/")]

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

# maybe not used
def list_to_index_list(input_list, index_dic):
    """
    Args:
        input_list: some list
        index_dic: index dictionary for input list, it can be made using make_index_dic() function
    Returns:
        list of index
    """
    return map(lambda y: index_dic.get(y, -1), input_list) # -1 means unknown in evaluation

def build_data_set(target_word_list, path, answer_index_dic={}):
    """
    Args:
        target_word_list: list of target word for make training set and answer set
        path: input file path of txt file
    Returns:
        tuple list,
            tuple of training sentences, answers, target_index and target_word list.
            training sentences is formed like [["BOS", 가/NNG, 나/NNG, "EOS"],["BOS", ...],].
            It is word list of a sentences and contains target word

            answer set is formed like ['01', '03', '09', '13', ...]
            sense nubmers of answer senses.

            index_list is list of target word index in sentence.
            [9, 2, 3, 15, ....]
    """

    result = []
    answer_set_dic = {} # key = target_word, value = str_answer_list
    count = 0
    with open(path, 'r') as fr_txt:
        print("build data set from ", path)
        for line in fr_txt:
            line = line.rstrip()

            line_without_sense = re.sub(r'__[\d][\d]', '', line).split()
            # if len(set(target_word_list) & set(line_without_sense)) == 0:
            if not set(target_word_list) & set(line_without_sense):
                # there is no target word in the sentence
                continue
            tokens = line.split()
            for index, token in enumerate(tokens):
                # if token is homograph and included in target_word_list
                if TKN_PTN.match(token) and target_word_list.count(re.sub(r'__[\d][\d]',
                                                                          '', token)) != 0:
                    # prepare tuple data
                    target_word = re.sub(r'__[\d][\d]', '', token)
                    sentence = ['BOS']+line_without_sense+['EOS']
                    answer = get_sense_number(token)
                    target_index = index+1 # +1: BOS index
                    data = (sentence, answer, target_index, target_word)

                    if WINDOW_SIZE > -1:
                        sentence = (['BOS']*WINDOW_SIZE)+line_without_sense+(['EOS']*WINDOW_SIZE)
                        new_index = index + WINDOW_SIZE
                        sentence = sentence[new_index-WINDOW_SIZE:new_index+WINDOW_SIZE+1]
                        target_index = new_index
                        data = (sentence, answer, target_index, target_word)

                    # collect answers for each target_word
                    str_answer_list = answer_set_dic.get(target_word, [])
                    str_answer_list.append(answer)
                    answer_set_dic[target_word] = str_answer_list
                    # append to target word dataset
                    result.append(data)

                    count = count + 1
                    if count % 1000 == 0:
                        print(count, " instances were made")

    print("working done, total ", count, " instances were made")
    result, answer_index_dic = make_answer_index(result, answer_set_dic, answer_index_dic)
    return result, answer_index_dic

def instances_to_batch(batch_sentences, glove_index_dic):
    """
    batch_sentences, glove_index_dic -> padded_index_batch, length_list
    Args:
        batch_sentences: sorted, sentence list in batch
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
    batch_size = len(batch_sentences)
    for sentence in batch_sentences:
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
    return set_device(torch.LongTensor(padded_matrix)), length_list


def train(model, epochs, loss_function, training_data, validating_data, glove_index_dic, print_count=1):
    """
    Arg:
        model: wsd model
        epochs: training epochs
        loss_function: loss_function of model
        training_data:
                tuple list of (sentence, answer, target_index, target_word).
                result of build_data_set() function
        validation_data: same structure
        print_count: every print_count epochs, current loss will be printed
    Returns:
        model trained
    """
    best_model = None
    best_epoch = 0
    bad_epoch_count = 0
    best_validation_loss = math.inf
    best_validation_acc = 0
    train_loss, validation_loss, train_acc, validation_acc = 0, 0, 0, 0
    for epoch in range(1, epochs+1):
        if bad_epoch_count >= PATIENCE and EARLY:
            print("Training stopped at epoch", epoch)
            print("epoch: ", epoch, "train_loss: ", train_loss)
            print("epoch: ", epoch, "validate_loss: ", validation_loss)
            print("epoch: ", epoch, "train_acc: ", train_acc)
            print("epoch: ", epoch, "validate_acc: ", validation_acc)
            return best_epoch, best_model

        ### training part
        model.train()
        train_current_loss = 0
        train_correct = 0
        train_total = len(training_data)
        batch_index = 0
        for batch_index in range(0, math.ceil(len(training_data)/BATCH_SIZE)):
            start = batch_index*BATCH_SIZE
            end = min(start+BATCH_SIZE, len(training_data))
            data_for_batch = training_data[start:end]
            # sort data for padding
            # y[0] : sentence
            data_for_batch.sort(key=lambda y: len(y[0]), reverse=True)

            # unzip to use data
            batch_sentences, batch_answers, batch_target_indices, batch_target_words\
                = zip(*data_for_batch)

            # make batch from training instances
            batch, length_list = instances_to_batch(batch_sentences, glove_index_dic)
            batch_size = len(data_for_batch)

            # scores: list of outputs(variable length)
            model.zero_grad()
            scores = model(batch, batch_size, length_list, batch_target_indices, batch_target_words)

            # find loss and sum
            unit_loss_list = []
            for batch_index_inner in range(batch_size):
                score = scores[batch_index_inner]
                score = score.view(1, -1)
                _, prediction = torch.max(score, 1)
                answer = batch_answers[batch_index_inner]
                if prediction.item() == answer:
                    train_correct = train_correct+1
                answer = set_device(torch.tensor(answer).view(1))
                unit_loss_list.append(loss_function(score, answer))

            loss = sum(unit_loss_list)
            train_current_loss = train_current_loss + loss.item()

            # super important part
            # model.parameters change every batch.
            # optimizer should know changed model.parameters. (output layer)
            optimizer = optim.Adam(model.parameters())

            loss.backward()
            optimizer.step()

        train_loss = train_current_loss/(batch_index+1)
        train_acc = (train_correct/train_total)*100

        ### validation part
        model.eval()
        validation_current_loss = 0
        validation_correct = 0
        validation_total = len(validating_data)
        batch_index = 0
        for batch_index in range(0, math.ceil(len(validating_data)/BATCH_SIZE)):
            start = batch_index*BATCH_SIZE
            end = min(start+BATCH_SIZE, len(training_data))
            data_for_batch = validating_data[start:end]

            with torch.no_grad():
                # sort data for padding
                # y[0] : sentence
                data_for_batch.sort(key=lambda y: len(y[0]), reverse=True)

                # unzip to use data
                batch_sentences, batch_answers, batch_target_index_list, batch_target_word_list\
                        = zip(*data_for_batch)

                # make batch from test instances
                batch, length_list = instances_to_batch(batch_sentences, glove_index_dic)
                batch_size = len(length_list)

                # scores: list of output vectors
                scores = model(batch, batch_size, length_list, batch_target_index_list,
                               batch_target_word_list)

                # find loss and sum
                unit_loss_list = []
                for batch_index_inner in range(batch_size):
                    score = scores[batch_index_inner]
                    score = score.view(1, -1)
                    _, prediction = torch.max(score, 1)
                    answer = batch_answers[batch_index_inner]
                    if prediction.item() == answer:
                        validation_correct = validation_correct+1
                    answer = set_device(torch.tensor(answer).view(1))
                    unit_loss_list.append(loss_function(score, answer))

                validation_current_loss = validation_current_loss + sum(unit_loss_list).item()

        validation_loss = validation_current_loss/(batch_index+1)
        validation_acc = (validation_correct/validation_total)*100

        if epoch % print_count == 0:
            print("epoch: ", epoch, "train_loss: ", train_loss)
            print("epoch: ", epoch, "validate_loss: ", validation_loss)
            print("epoch: ", epoch, "train_acc: ", train_acc)
            print("epoch: ", epoch, "validate_acc: ", validation_acc)

        LOGGER.scalar_summary('train_loss', train_loss, epoch)
        LOGGER.scalar_summary('train_acc', train_acc, epoch)
        LOGGER.scalar_summary('validate_loss', validation_loss, epoch)
        LOGGER.scalar_summary('validate_acc', validation_acc, epoch)

        if best_validation_acc > validation_acc or (best_validation_acc == validation_acc and best_validation_loss < validation_loss):
            bad_epoch_count = bad_epoch_count + 1
        else:
            best_validation_acc = validation_acc
            best_validation_loss = validation_loss
            bad_epoch_count = 0
            best_epoch = epoch
            best_model = copy.deepcopy(model)
    
    return best_epoch, set_device(best_model)

def evaluate(model, test_data, glove_index_dic, print_result=True):
    """
    Args:
        model: wsd model
        test_data:
            tuple list of (test_sentence, answer, target_index, target_word).
            result of build_data_set() function
    Returns:
        return accuracy
        prints result of evaluation
    """
    model.eval()
    with torch.no_grad():
        # test data can be made full batch? 
        # !!!! no, gpu will be gone if batch size is tooooo big
        # but so far, it is okay. :)
        target_total_dic = {}
        target_correct_dic = {}

        correct = 0
        total = len(test_data)

        # sort data for padding
        # y[0] : sentence
        test_data.sort(key=lambda y: len(y[0]), reverse=True)

        # unzip to use data
        batch_sentences, batch_answers, batch_target_index_list, batch_target_word_list\
                = zip(*test_data)

        # make batch from test instances
        batch, length_list = instances_to_batch(batch_sentences, glove_index_dic)
        batch_size = len(length_list)

        # scores: list of output vectors
        scores = model(batch, batch_size, length_list, batch_target_index_list,
                       batch_target_word_list)

        for batch_index in range(batch_size):
            target_word = batch_target_word_list[batch_index]
            score = scores[batch_index]
            score = score.view(1, -1)
            answer = batch_answers[batch_index]
            _, prediction = torch.max(score, 1)
            target_total_dic[target_word] = target_total_dic.get(target_word, 0)+1

            if answer == prediction.item():
                correct = correct+1
                target_correct_dic[target_word] = target_correct_dic.get(target_word, 0)+1
        if VERBOSE and print_result:
            for key, target_total in target_total_dic.items():
                print(key+": ", target_total, "instances were tested,",
                      target_correct_dic.get(key, 0), "answers were correct")
                try:
                    print("Accuracy: ", (target_correct_dic.get(key, 0)/target_total) * 100, "%")
                except ZeroDivisionError:
                    print("Accuracy: 0.0 %")
        if print_result:
            print("Total ", total, "instances were tested,", correct, "answers were correct")
            try:
                print("Accuracy: ", (correct/total) * 100, "%")
            except ZeroDivisionError:
                print("Accuracy: 0.0 %")
            with open(RESULT_FILE, 'wb') as fw_result:
                acc_result = {}
                for target_word in batch_target_word_list:
                    target_correct = target_correct_dic.get(target_word, 0)
                    target_total = target_total_dic.get(target_word, -1)
                    acc_result[target_word] = (target_correct,
                                               target_total,
                                               (target_correct/target_total)*100)
                acc_result['TOTAL'] = (len(acc_result.keys()),
                                       correct,
                                       total,
                                       (correct/total)*100)
                acc_result['META'] = ARGS
                pickle.dump(acc_result, fw_result)

    return (correct/total) * 100

def split_validation_set(training_data, division):
    # TODO: make generalized popping
    """
    Args:
        training_data: result of build_data_set(target_word_list, TRAIN_SET)
        division: validation set size is training_set_size/division
    Returns:
        (training_data, validation_data)
    """
    validation_data = []
    idx = 0
    count = 0
    for _ in range(len(training_data)):
        if count % division == 0:
            new = training_data.pop(idx)
            validation_data.append(new)
            idx = idx-1
        count = count+1
        idx = idx+1

    return training_data, validation_data

def make_answer_index(data_set, answer_set_dic, answer_index_dic={}):
    """
    Args:
        data_set: tuple list, result of build_data_set(target_word_list, DATA_SET)
        answer_set_dic: key=WORD/POS, value=str_answer_list
    Return:
        answer_index_dic: key=WORD/POS, value=dictionary that has (key=sense, value=index)
        data_set: transformed data_set from input data_set.
                  sense number answer is transformed to index answer ex)'02' -> 1 ...
    """
    if not answer_index_dic:
        # build answer_index_dic
        for key, str_answer_list in answer_set_dic.items():
            index_dic = make_index_dic(str_answer_list) # key=sense number, value=index
            answer_index_dic[key] = index_dic

    # change answer(sense number) to answer(index)
    for idx, (sentence, answer, target_index, target_word) in enumerate(data_set):
        # 0 means unk
        index_answer = answer_index_dic[target_word].get(answer, 0)
        data_set[idx] = (sentence, index_answer, target_index, target_word)

    return data_set, answer_index_dic

def build_all_data_set():
    """
    build data_set as file
    return all data set
    """
    with open(ENTROPY_DIC, 'rb') as fr_ent, open(DATA_SET, 'wb') as fw_result:
        ent_dic = pickle.load(fr_ent)
        #target_word_list = ["관/NNG", "원/NNG", "감수/NNG", "정수/NNG", "거사/NNG",
        #                   "유/NNP", "국/NNG", "표시/NNG", "배출되/VV"]
        target_word_list = [k for k, v in ent_dic.items() if v >= ENTROPY_THRESHOLD]

        # prepare resources for training
        training_data, answer_index_dic = build_data_set(target_word_list, TRAIN_SET)

        # split validation_set from training_set, 9:1
        training_data, validation_data = split_validation_set(training_data, 10)

        # get sense_len by iterating answer_index_dic
        sense_len_dic = {} # key=WORD/POS, value=sense_size
        for key, value in answer_index_dic.items():
            # it is equal to the number of sense class of the word
            sense_len = len(value.keys())
            sense_len_dic[key] = sense_len
         # prepare resources for test

        test_data, _ = build_data_set(target_word_list, TEST_SET, answer_index_dic)

        data_set = {}
        data_set['trn'] = training_data
        data_set['val'] = validation_data
        data_set['test'] = test_data
        data_set['sense'] = sense_len_dic

        pickle.dump(data_set, fw_result)
        print("building data set done!")

    return training_data, validation_data, test_data, sense_len_dic

def filter_top_nword(training_data, validation_data, test_data):
    """
    Args:
        data sets
    Returns:
        filtered data sets (only contains entropy top nword)
    """
    with open(ENTROPY_DIC, 'rb') as fr_ent, open(VOCA, 'rb') as fr_voca:
        if SINGLE_TARGET_WORD:
            ent_rank = [SINGLE_TARGET_WORD]
        else:
            ent_dic = pickle.load(fr_ent)
            item_list = list(ent_dic.items())
            item_list.sort(key=lambda y: y[1], reverse=True)
            # y[1] = entropy of word
            ent_rank, _ = zip(*item_list)
            # ent_rank: WORD/POS list
            ent_rank = ent_rank[0:min(len(ent_rank), NWORD)]
        
        if WEAK != -1:
            voca = pickle.load(fr_voca)
            ent_rank = list(filter(lambda y: sum(list(voca[y].values())) <= WEAK, ent_rank))

        print("target weak words:", ent_rank, len(ent_rank))

        # y[3] = target_word
        training_data = list(filter(lambda x: x[3] in ent_rank, training_data))
        validation_data = list(filter(lambda y: y[3] in ent_rank, validation_data))
        test_data = list(filter(lambda y: y[3] in ent_rank, test_data))
    
        return training_data, validation_data, test_data

def print_result():
    """
    print result from specific result file
    """
    with open(RESULT_FILE, 'rb') as fr_result, open(ENTROPY_DIC, 'rb') as fr_ent:
        ent_dic = pickle.load(fr_ent)
        result_dic = pickle.load(fr_result)
        meta = result_dic['META']
        total = result_dic['TOTAL']
        del result_dic['META']
        del result_dic['TOTAL']
        print(meta)
        if SINGLE_TARGET_WORD:
            print(result_dic.get(SINGLE_TARGET_WORD, "No such word in result"), ent_dic[SINGLE_TARGET_WORD])
        else:
            item_list = list(map(lambda y: (ent_dic[y[0]], y), list(result_dic.items())))
            item_list.sort(key=lambda y: y[0], reverse=True)
            for item in item_list:
                print(item)
        print(total)
 

def main():
    """
    this is main function
    """
    if CHECK:
        print_result()
        return

    if BUILD_DATA_SET:
        training_data, validation_data, test_data, sense_len_dic = build_all_data_set()
    else:
        with open(DATA_SET, 'rb') as fr_data_set:
            data_set = pickle.load(fr_data_set)
            training_data = data_set['trn']
            validation_data = data_set['val']
            test_data = data_set['test']
            sense_len_dic = data_set['sense']

    if NWORD != -1 or SINGLE_TARGET_WORD:
        training_data, validation_data, test_data = filter_top_nword(training_data,
                                                                     validation_data, test_data)

    #### training part ####
    # define model
    model = MyLSTM(EMBEDDING_DIM, HIDDEN_DIM, HIDDEN2_DIM, sense_len_dic)
    glove_index_dic = model.glove_index_dic
    set_device(model)
    loss_function = set_device(nn.CrossEntropyLoss())

    # train model
    best_epoch, best_model = train(model, EPOCHS, loss_function, training_data,
                                   validation_data, glove_index_dic, print_count=10)

    print("best_model at", best_epoch)
    #### test part ####
    # test model
    if BEST:
        model = best_model
    evaluate(model, test_data, glove_index_dic)

if __name__ == '__main__':
    main()
