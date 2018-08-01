"""
multi model applier with entropy threshold
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
import numpy as np
#import matplotlib.pyplot as plt

# FILE PATH #

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
ANSWER = os.path.join(THIS_FOLDER, 'answer.txt')
ENTROPY_DIC = os.path.join(THIS_FOLDER, 'entropy_dic.bin')
SVM_DIC = os.path.join(THIS_FOLDER, 'svm_dic.bin')
GLOVE = os.path.join(THIS_FOLDER, 'vectors.bin')

TKN_PTN = re.compile(r'.*__[\d][\d].*')

try:
    TRAIN_SET = os.path.join(THIS_FOLDER, sys.argv[1])
    TEST_SET = os.path.join(THIS_FOLDER, sys.argv[2])
    ENTROPY_THRESHOLD = float(sys.argv[3])
except BaseException:
    print(
        'usage: python3 baseline.py [transformed_train_file] [transfromed_test_file] ent_threshold')
    sys.exit(1)

# set specific strategy for each part
# 0: MFS(Most Frequent Sense), 1: SVM
STRATEGY_FOR_HIGH = 1
STRATEGY_FOR_LOW = 0

# dimension of embedding vector
VECTOR_DIMENSION = 100

#############
# functions #
#############


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

    if (sum_vector == np.zeros([vector_dimension, ])).all(): # if all tokens in sentence did not hit
        return None
    return sum_vector


def make_answer():
    """
    # Function that makes answer from test set #
    # for each homograph word, this function chooses most frequent sense of the word #
    # if the homograph word is not trained before, the function chooses "UNKNOWN" as answer word.
    and it will cause wrong answering in evaluation #
    # answer sentences are saved as file, "answer.txt" in current directory #
    """

    with open(TEST_SET, 'r') as fr_test, \
            open(ANSWER, 'w') as fw_answer, \
            open(MAX_FREQ_DIC, 'rb') as fr_max_freq_dic, \
            open(ENTROPY_DIC, 'rb') as fr_ent_dic, \
            open(GLOVE, 'rb') as fr_vectors, \
            open(SVM_DIC, 'rb') as fr_svm_dic:

        svm_dic = pickle.load(fr_svm_dic)
        max_freq_dic = pickle.load(fr_max_freq_dic)
        ent_dic = pickle.load(fr_ent_dic)
        glove_model = pickle.load(fr_vectors)

        for line in fr_test:
            line = line.replace("\n", "")
            tokens = re.split('[ ]', line)

            for index, token in enumerate(tokens):
                if TKN_PTN.match(token):
                    query_word = re.sub(r'__[\d][\d]', '', token)

                    # high
                    if ent_dic.get(query_word, -1) >= ENTROPY_THRESHOLD:
                        if STRATEGY_FOR_HIGH == 0:
                            answer_word = max_freq_dic.get(
                                query_word, "UNKNOWN")
                        else:
                            feature_vector = make_feature_vector(glove_model, re.split(
                                '[ ]', re.sub(r'__[\d][\d]', '', line)), index, VECTOR_DIMENSION)
                            if svm_dic.get(query_word) is not None and feature_vector is not None:
                                answer_sense = svm_dic[query_word].predict(
                                    np.array([feature_vector]))[0]
                                answer_word = query_word[:query_word.index(
                                    "/")] + "__" + answer_sense + query_word[query_word.index("/"):]
                            else:
                                answer_word = "UNKNOWN"
                    # low
                    else:
                        answer_word = max_freq_dic.get(query_word, "UNKNOWN")

                # not homograph case
                else:
                    answer_word = token
                fw_answer.write(answer_word + " ")

            fw_answer.write('\n')


def evaluate():
    """
    # evaluate answer by comparing "test_set.txt" with "answer.txt" #
    # only homograph words are counted for scoring #
    """

    with open(TEST_SET, 'r') as fr_test, open(ANSWER, 'r') as fr_answer:
        #, open(ENTROPY_DIC, 'rb') as fr_ent_dic:

        #ent_dic = pickle.load(fr_ent_dic)
        count = 0
        correct = 0

        #plt.hist(list(ent_dic.values()), bins=20)
        # plt.savefig('histogram')
        #plt.hist([v for v in ent_dic.values() if v != 0], bins=20)
        # plt.savefig('histogram2')

        checked_words = set([])
        while True:
            test_line = fr_test.readline().replace("\n", "")
            answer_line = fr_answer.readline().replace("\n", "")

            if not test_line or not answer_line:
                break

            test_tokens = re.split('[ ]', test_line)
            answer_tokens = re.split('[ ]', answer_line)

            merged_tokens = zip(test_tokens, answer_tokens)

            for token1, token2 in merged_tokens:
                key = re.sub(r'__[\d][\d]', '', token1)
                if TKN_PTN.match(token1):# and ent_dic.get(key, -1) >= ENTROPY_THRESHOLD:
                    count = count + 1
                    checked_words.add(key)
                    if token1 == token2:
                        correct = correct + 1

        print("The number of target words : ", len(checked_words))
        print("The number of homograph eo-jeul : ", count)
        print("The number of correct answer : ", correct)
        print("Accuracy : ", (correct / count) * 100)


def main():
    """
    this is main function
    """

    make_answer()
    print("answering done!")

    evaluate()


if __name__ == '__main__':
    main()
