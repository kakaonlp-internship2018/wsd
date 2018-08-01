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
import numpy as np
from sklearn import svm

# FILE PATH #

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
ENTROPY_DIC = os.path.join(THIS_FOLDER, 'entropy_dic.bin')
SVM_DIC = os.path.join(THIS_FOLDER, 'svm_dic.bin')
GLOVE = os.path.join(THIS_FOLDER, 'vectors.bin')
GLOVE_TXT = os.path.join(THIS_FOLDER, 'glove/vectors.txt')

TKN_PTN = re.compile(r'.*__[\d][\d].*')

try:
    TRAIN_SET = os.path.join(THIS_FOLDER, sys.argv[1])
    TEST_SET = os.path.join(THIS_FOLDER, sys.argv[2])
    ENTROPY_THRESHOLD = float(sys.argv[3])
except BaseException:
    print(
        'usage: python3 baseline.py [transformed_train_file] [transfromed_test_file] ent_threshold')
    sys.exit(1)


# dimension of embedding vector
VECTOR_DIMENSION = 100

# define LOGGER

PROGRAM = os.path.basename(sys.argv[0])
LOGGER = logging.getLogger(PROGRAM)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


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


def make_feature_vector(model, sentence, target_word_index, vector_dimension):
    """
    Arg:
        model : gensim w2v model
        sentence : word(WORD__NN/POS) list
        target_word_index : index of target word in list

    return:
        sum vector of features

    make simple sum vector of all features
    """

    sum_vector = np.zeros([vector_dimension, ])
    for index, token in enumerate(sentence):
        if index == target_word_index:
            continue
        try:
            sum_vector = sum_vector + model[token]
        except KeyError:
            pass

    # if all tokens in sentence did not hit
    if (sum_vector == np.zeros([vector_dimension, ])).all():
        return None
    return sum_vector


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
    transform "vectors.txt" to "vectors.bin"
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


def build_svm_for_difficult_word():
    """
    build svm model for each difficult word
    all svm models are store in svm_dic (key: "WORD/POS", value: sklearn_svm_model)
    and svm_dic is saved as file "svm_dic.bin"

    It needs pre-trained glove model as file "glove/vectors.txt"
    """
    with open(TRAIN_SET, 'r') as fr_train, open(ENTROPY_DIC, 'rb') as fr_ent_dic,\
            open(GLOVE, 'rb') as fr_glove:
        # open(SVM_DIC, 'rb') as fr_svm_dic:
        ent_dic = pickle.load(fr_ent_dic)
        glove_model = pickle.load(fr_glove)
        """
        try:
            svm_dic = pickle.load(fr_svm_dic)
        except EOFError:
            svm_dic = {}
        """
        svm_dic = {}

        count = 0
        # build training data for each difficult word
        train_dic = {}  # key = WORD/POS, value = [(sense, feature sum vector)]
        for line in fr_train:
            line = line.replace("\n", "")
            new_tokens = re.split('[ ]', line)

            for index, token in enumerate(new_tokens):
                if TKN_PTN.match(token):
                    key = re.sub(r'__[\d][\d]', '', token)
                    # and svm_dic.get(key) is None:
                    if ent_dic[key] >= ENTROPY_THRESHOLD:
                        tokens_for_emb = re.split(
                            '[ ]', re.sub(r'__[\d][\d]', '', line))
                        feature_vector = make_feature_vector(
                            glove_model, tokens_for_emb, index, VECTOR_DIMENSION)
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

        count = 0
        # build svm model for each difficult word
        for key, training_data in train_dic.items():
            svm_model = svm.LinearSVC()
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

        with open(SVM_DIC, 'wb') as fw_svm_dic:
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

    # build_voca()
    #print("building voca done!")

    # build_max_freq_dic_and_ent_dic()
    #print("building max_freq_dic and entropy_dic done!")

    make_glove_bin()

    build_svm_for_difficult_word()
    print("building svm models done!")


if __name__ == '__main__':
    main()
