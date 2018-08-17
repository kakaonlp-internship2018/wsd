"""
resource builder
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
from sklearn import svm


if __name__ == '__main__':
    # FILE PATH #
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
    MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
    ENTROPY_DIC = os.path.join(THIS_FOLDER, 'entropy_dic.bin')
    SVM_DIC = os.path.join(THIS_FOLDER, 'svm_dic.bin')
    GLOVE = os.path.join(THIS_FOLDER, 'vectors.bin')
    GLOVE_TXT = os.path.join(THIS_FOLDER, 'glove/vectors.txt')
    TRN_DIC = os.path.join(THIS_FOLDER, 'trn_dic.bin')

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
    PARSER.add_argument('--ent', type=float, default=0.1,
                        help='Entropy threshold, default=0.1')
    PARSER.add_argument('--min_max', dest='min_max', action='store_true', default=False,
                        help='Add min_max vector to feature vector')
    #PARSER.add_argument('--weight', help='Apply weight to feature vector')
    PARSER.add_argument('--win', type=int, default=2,
                        help='Set window size, default=2, 0: whole sentence (sum case)')
    PARSER.add_argument('--dim', type=int, default=100,
                        help='Set embedding dimension, default=100')
    PARSER.add_argument('--merge', type=str, default='concat', choices=['concat', 'sum'],
                        help='How to merge feature vector, default=concat')
    # PARSER.add_argument('--model', type=str, default='ssdfad', choice
    ARGS = PARSER.parse_args()

    # set global variables
    TRAIN_SET = ARGS.TRAIN_SET
    ENTROPY_THRESHOLD = ARGS.ent
    VECTOR_DIMENSION = ARGS.dim
    HALF_WINDOW_SIZE = ARGS.win
    MIN_MAX = ARGS.min_max
    #WEIGHT = ARGS.weight
    MERGE = ARGS.merge


#############
# functions #
#############

def calculate_entropy(count_list):
    """
    Arg:
        count_list : list of the word appearing count
    Return:
        shannon entropy of the word

    Returns shannon entropy of input count_list
    """

    return sum(map(lambda y: -y/sum(count_list)*np.log2(y/sum(count_list)), count_list))


def make_feature_vector(glove_model, sentence, target_word_index, \
        vector_dimension, merge, min_max, win_size):
    """
    this is wrapper of "make_feature_vector_sum" and "make_feature_vector_concat"
    """
    if merge == 'concat':
        return make_feature_vector_concat(glove_model, \
                                sentence, target_word_index, vector_dimension, min_max, win_size)
    elif merge == 'sum':
        return make_feature_vector_sum(glove_model, \
                                sentence, target_word_index, vector_dimension, min_max, win_size)


def make_feature_vector_sum(glove_model, sentence, target_word_index, \
        vector_dimension, min_max, win_size):
    """
    Arg:
        glove_model : word embedding model
        sentence : word(WORD/POS) list
        target_word_index : index of target word in list
        vector_dimension : word embedding dimension
        min_max : if true, concat min_max vector to feature_vector
        win_size : window size to sum, 0 : sum all words in sentence
    return:
        sum vector of features
    make simple sum vector of all features
    """

    # this can be simplified by applying <unk> embedding for express unknown word.
    # the code below exclude all unknown word in making feature vector.

    embedded_words = []
    index_list = []
    for index, token in enumerate(sentence):
        if index == target_word_index or glove_model.get(token) is not None:
            embedded_words.append(token)
            index_list.append(index)

    bos_list = ["BOS"] * win_size
    eos_list = ["EOS"] * win_size
    new_index = index_list.index(target_word_index) + win_size
    embedded_words = bos_list + embedded_words + eos_list

    sum_vector = np.zeros([vector_dimension, ])
    if min_max:
        max_vector = np.zeros([vector_dimension, ])
        min_vector = np.full([vector_dimension, ], np.Inf)

    for index, token in enumerate(embedded_words):
        if (index < new_index - win_size or index > new_index + win_size) and win_size != 0:
            continue
        if index == new_index:
            continue
        try:
            sum_vector = sum_vector + glove_model[token]
            if min_max:
                max_vector = np.fmax(max_vector, glove_model[token])
                min_vector = np.fmin(min_vector, glove_model[token])
        except KeyError:
            pass

    # if all tokens in sentence did not hit
    if (sum_vector == np.zeros([vector_dimension, ])).all():
        return None
    if min_max:
        return np.concatenate((sum_vector, max_vector, min_vector))
    return sum_vector


def make_feature_vector_concat(glove_model, sentence, target_word_index, \
        vector_dimension, min_max, win_size):
    """
    Arg:
        glove_model : word embedding model
        sentence : word(WORD/POS) list
        target_word_index : index of target word in list
        vector_dimension : word embedding dimension
        min_max : if true, concat min_max vector to feature_vector
        win_size : window size to concatenate

    return:
        concat vector of features

    make simple concat vector of all features
    """

    # this can be simplified by applying <unk> embedding for express unknown word.
    # the code below exclude all unknown word in making feature vector.

    embedded_words = []
    index_list = []
    for index, token in enumerate(sentence):
        if index == target_word_index or glove_model.get(token) is not None:
            embedded_words.append(token)
            index_list.append(index)

    bos_list = ["BOS"] * win_size
    eos_list = ["EOS"] * win_size
    new_index = index_list.index(target_word_index) + win_size
    embedded_words = bos_list + embedded_words + eos_list

    concat_vector = []
    if min_max:
        max_vector = np.zeros([vector_dimension, ])
        min_vector = np.full([vector_dimension, ], np.Inf)

    for index, token in enumerate(embedded_words):
        if index < new_index - win_size or index > new_index + win_size:
            continue
        if index == new_index:
            continue
        try:
            concat_vector = concat_vector + glove_model[token]
            if min_max:
                max_vector = np.fmax(max_vector, glove_model[token])
                min_vector = np.fmin(min_vector, glove_model[token])
        except KeyError:
            pass

    if min_max:
        return np.concatenate((np.array(concat_vector), max_vector, min_vector))
    return np.array(concat_vector)


def build_voca():
    """
    voca : key = word_without_sense, value = freq_dic
    freq_dic : key = sense_number, value = appearing count

    Function that build homograph vocabulary from training set
    vocabulary is an dictionary, key = "WORD/POS", value = freq_dic
    freq_dic is also an dictionary, key = "NN" (word sense), value = appearing count

    the function saves vocabulary as file "voca.bin" in current directory
    """
    vocabulary = {}
    with open(TRAIN_SET, 'r') as fr_train, open(VOCA, 'wb') as fw_voca:
        for line in fr_train:
            line = line.replace("\n", "")
            new_tokens = re.split('[ ]', line)
            for token in new_tokens:
                if TKN_PTN.match(token):
                    key = re.sub(r'__[\d][\d]', '', token)
                    sense_number = token[token.index("/")-2:token.index("/")]
                    freq_dic = vocabulary.get(key, {})
                    freq_dic[sense_number] = freq_dic.get(sense_number, 0) + 1
                    vocabulary[key] = freq_dic
        pickle.dump(vocabulary, fw_voca)


def make_glove_bin():
    """
    transform "glove/vectors.txt" to "vectors.bin"
    """
    with open(GLOVE_TXT, 'r') as fr_txt, open(GLOVE, 'wb') as fw_bin:
        glove_model = {}

        count = 0
        for line in fr_txt:
            vals = line.rstrip().split(' ')
            glove_model[vals[0]] = [float(x) for x in vals[1:]]
            count = count + 1
            if count % 100000 == 0:
                LOGGER.info("%s word vectors loaded", str(count))
        LOGGER.info("Loading embedding vectors done, %s", str(count))

        pickle.dump(glove_model, fw_bin)


def build_training_data_for_svm():
    """
    build training data dictionary that has key = WORD/POS, value = [(sense, feature vector)]
    It is used to train each svm model
    It needs pre-trained glove model as file "vectors.bin"

    """
    with open(TRAIN_SET, 'r') as fr_train, open(ENTROPY_DIC, 'rb') as fr_ent_dic,\
            open(GLOVE, 'rb') as fr_glove:
        ent_dic = pickle.load(fr_ent_dic)
        glove_model = pickle.load(fr_glove)

        count = 0
        # build training data for each difficult word
        train_dic = {}  # key = WORD/POS, value = [(sense, feature vector)]
        for line in fr_train:
            line = line.replace("\n", "")
            new_tokens = re.split('[ ]', line)

            for index, token in enumerate(new_tokens):
                if TKN_PTN.match(token):
                    key = re.sub(r'__[\d][\d]', '', token)
                    if ent_dic[key] >= ENTROPY_THRESHOLD:
                        tokens_for_emb = re.split(
                            '[ ]', re.sub(r'__[\d][\d]', '', line))
                        feature_vector = make_feature_vector(glove_model, tokens_for_emb, \
                                        index, VECTOR_DIMENSION, MERGE, MIN_MAX, HALF_WINDOW_SIZE)
                        if feature_vector is None:
                            continue
                        value = train_dic.get(key, [])
                        sense = token[token.index("/")-2:token.index("/")]

                        value.append((sense, feature_vector))
                        train_dic[key] = value
                        count = count + 1
                        if count % 100000 == 0:
                            LOGGER.info(
                                "%s tokens finished building training data", str(count))

        LOGGER.info("Building training data done, %s", str(count))

        # with open(TRN_DIC, 'wb') as fw_trn_dic:
        #    pickle.dump(train_dic, fw_trn_dic)
        return train_dic


def build_svm_for_difficult_word():
    """
    build svm model for each difficult word
    all svm models are stored in svm_dic (key: "WORD/POS", value: sklearn_svm_model)
    and svm_dic is saved as file "svm_dic.bin"

    """
    with open(SVM_DIC, 'wb') as fw_svm_dic: # open(TRN_DIC, 'rb') as fr_trn_dic,
        #train_dic = pickle.load(fr_trn_dic)
        train_dic = build_training_data_for_svm()
        count = 0
        svm_dic = {}
        # build svm model for each difficult word
        for key, training_data in train_dic.items():
            svm_model = svm.LinearSVC()  # todo: try various svm model.
            sense_list, vector_list = zip(*training_data)
            try:
                svm_model.fit(vector_list, sense_list)
            except ValueError:  # if there is only one class in training data
                continue
            svm_dic[key] = svm_model
            count = count + 1
            if count % 1000 == 0:
                LOGGER.info("%s words finished training svm model", str(count))

        LOGGER.info("Training svm model done, %s", str(count))

        # for convenience, save meta data in svm_dic file (win size, dim, ...)
        svm_dic["META"] = ARGS

        pickle.dump(svm_dic, fw_svm_dic)


def build_max_freq_dic_and_ent_dic():
    """
    Function that build max_freq_dic and entropy_dic from vocabulary
    max_freq_dic is an dictionary,
    key = "WORD/POS", value = the most frequent sense of the word (formed "WORD__NN/POS")
    the function saves max_freq_dic as file "max_freq_dic.bin" in current directory

    entropy_dic is also an dictionary,
    key = "WORD/POS", value = shannon entropy of the word
    It also will be saved as file "entropy_dic.bin" in current directory

    """
    with open(VOCA, 'rb') as fr_voca, open(MAX_FREQ_DIC, 'wb') as fw_max_freq_dic,\
            open(ENTROPY_DIC, 'wb') as fw_entropy_dic:
        vocabulary = pickle.load(fr_voca)
        max_freq_dic = {}
        entropy_dic = {}
        for key, freq_dic in vocabulary.items():
            max_freq_dic[key] = key[:key.index("/")] + \
                "__" + max(freq_dic, key=freq_dic.get) + key[key.index("/"):]
            entropy_dic[key] = calculate_entropy(freq_dic.values())

        # sorted_ent_list = sorted(entropy_dic.items(), \
        #                key=operator.itemgetter(1), reverse=True)

        pickle.dump(entropy_dic, fw_entropy_dic)
        pickle.dump(max_freq_dic, fw_max_freq_dic)


def main():
    """
    this is main function
    """

    build_voca()
    print("building voca done!")

    build_max_freq_dic_and_ent_dic()
    print("building max_freq_dic and entropy_dic done!")

    make_glove_bin()

    # build_training_data_for_svm()
    build_svm_for_difficult_word()
    print("building svm models done!")


if __name__ == '__main__':
    main()
