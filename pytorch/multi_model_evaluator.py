"""
multi model applier with entropy threshold
__author__ = 'jeff.yu (jeff.yu@kakaocorp.com)'
__copyright__ = 'No copyright, just copyleft!'
"""

###########
# imports #
###########
import re
import pickle
import os
import argparse
import numpy as np
import resource_builder as rb
#import matplotlib.pyplot as plt

# FILE PATH #

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
ANSWER = os.path.join(THIS_FOLDER, 'answer.txt')
ENTROPY_DIC = os.path.join(THIS_FOLDER, 'entropy_dic.bin')
GLOVE = os.path.join(THIS_FOLDER, 'vectors.bin')

# define pattern
TKN_PTN = re.compile(r'.*__[\d][\d].*')

# define argparser
PARSER = argparse.ArgumentParser(description='This is multi_model_evaluator.')
PARSER.add_argument('TEST_SET', type=str, metavar='Test_set',
                    help='Transformed test set file')
PARSER.add_argument('--ent', type=float, default=0.1,
                    help='Entropy threshold, default=0.1')
PARSER.add_argument('--svm', type=str, default='svm_dic.bin',
                    help='Specify svm_dic file, default="svm_dic.bin"')
PARSER.add_argument('--mfs', dest='mfs', action='store_true', default=False,
                    help='set this if you want to apply MFS to low level words')
PARSER.add_argument('--nword', type=int, default=-1,
                    help='the number of target word, default=-1')
PARSER.add_argument('--weak', type=int, default=-1,
                    help='weakness, default=-1')
PARSER.add_argument('--target', type=str, default=None,
                    help='Specify target word for disambiguation, default=None')
PARSER.add_argument('--result', type=str, default='svm_result.bin',
                    help='Specify result file name, default=svm_result.bin')



ARGS = PARSER.parse_args()
TEST_SET = ARGS.TEST_SET
ENTROPY_THRESHOLD = ARGS.ent
WEAK = ARGS.weak
SVM_DIC = os.path.join(THIS_FOLDER, ARGS.svm)
MFS = ARGS.mfs
NWORD = ARGS.nword
SINGLE_TARGET_WORD = ARGS.target
RESULT= ARGS.result

# set global variables
with open(SVM_DIC, 'rb') as fr_svm:
    META = pickle.load(fr_svm)["META"]
    VECTOR_DIMENSION = META.dim
    HALF_WINDOW_SIZE = META.win
    MIN_MAX = META.min_max
    MERGE = META.merge
    RESULT_FILE = os.path.join(THIS_FOLDER, RESULT)

# set specific strategy for each part
# 0: MFS(Most Frequent Sense), 1: SVM
STRATEGY_FOR_HIGH = 1
STRATEGY_FOR_LOW = 0


#############
# functions #
#############

def predict(sentence, index, level, max_freq_dic, svm_dic, glove_model):
    """
    Args:
        sentence : sentence that include target word. all sense numbers should be removed
        index : index of target word
        level : difficulty level of target word (0: low, 1: high)
        max_freq_dic : max_freq_dic
        svm_dic : svm_dic
        glove_model : glove_model

    Returns:
        predicted word sense ("WORD__NN/POS" form)

    # if the homograph word is not trained before, the function chooses "UNKNOWN" as answer word.
    and it will cause wrong answering in evaluation #
    """
    query_word = sentence[index]

    if level == STRATEGY_FOR_HIGH:
        feature_vector = rb.make_feature_vector(
            glove_model, sentence, index, VECTOR_DIMENSION, MERGE, MIN_MAX, HALF_WINDOW_SIZE)
        if svm_dic.get(query_word) is not None and feature_vector is not None:
            answer_sense = svm_dic[query_word].predict(
                np.array([feature_vector]))[0]
            answer_word = query_word[:query_word.index(
                "/")] + "__" + answer_sense + query_word[query_word.index("/"):]
        else:
            # svm have not been trained or feature_vector could not be formed
            answer_word = "UNKNOWN"

    else:
        answer_word = max_freq_dic.get(query_word, "UNKNOWN")

    return answer_word

def top_nword(ent_dic, n):
    """
    Args:
        ent_dic: entropy dictionary
        n: cut off rank 
    Returns:
        top n words
    """
    with open(VOCA, 'rb') as fr_voca:
        if SINGLE_TARGET_WORD:
            ent_rank = [SINGLE_TARGET_WORD]
        else:
            item_list = list(ent_dic.items())
            item_list.sort(key=lambda y: y[1], reverse=True)
            # y[1] = entropy of word
            ent_rank, _ = zip(*item_list)
            # ent_rank: WORD/POS list
            ent_rank = ent_rank[0:min(len(ent_rank), n)]
        if WEAK != -1:
            voca = pickle.load(fr_voca)
            ent_rank = list(filter(lambda y: sum(list(voca[y].values())) <= WEAK, ent_rank))

    return ent_rank


def evaluate():
    """
    # evaluate answer by calling predict function for every homograph words #
    # only homograph words are counted for scoring #
    """

    with open(TEST_SET, 'r') as fr_test, open(ENTROPY_DIC, 'rb') as fr_ent_dic, \
            open(MAX_FREQ_DIC, 'rb') as fr_max_freq_dic, \
            open(GLOVE, 'rb') as fr_vectors, \
            open(SVM_DIC, 'rb') as fr_svm_dic:

        svm_dic = pickle.load(fr_svm_dic)
        max_freq_dic = pickle.load(fr_max_freq_dic)
        ent_dic = pickle.load(fr_ent_dic)
        glove_model = pickle.load(fr_vectors)

        count = 0
        correct = 0

        #plt.hist(list(ent_dic.values()), bins=20)
        # plt.savefig('histogram')
        #plt.hist([v for v in ent_dic.values() if v != 0], bins=20)
        # plt.savefig('histogram2')

        checked_words = set([])
        result_dic = {}
        total_dic = {}
        correct_dic = {}
        nword = top_nword(ent_dic, NWORD)
        for test_line in fr_test:
            test_line = test_line.replace("\n", "")
            test_tokens = re.split('[ ]', test_line)

            for index, token in enumerate(test_tokens):
                key = re.sub(r'__[\d][\d]', '', token)

                if not key in nword:
                    continue

                if TKN_PTN.match(token):
                    # if ent_dic.get(key, -1) >= ENTROPY_THRESHOLD:
                    if ent_dic.get(key, -1) < ENTROPY_THRESHOLD:
                        continue
                        total_dic[key] = total_dic.get(key, 0)+1
                        count = count + 1
                        checked_words.add(key)
                        answer_word = predict(re.split(
                            '[ ]', re.sub(r'__[\d][\d]', '', test_line)), \
                            index, STRATEGY_FOR_HIGH, max_freq_dic, svm_dic, glove_model)
                        if token == answer_word:
                            correct = correct + 1
                            correct_dic[key] = correct_dic.get(key, 0)+1

                    else:
                        if MFS:  # if MFS option activated
                            count = count + 1
                            checked_words.add(key)
                            answer_word = max_freq_dic.get(key, "UNKNOWN")
                            if token == answer_word:
                                correct = correct + 1

        with open(RESULT_FILE, 'wb') as fw_result:
            acc_result = {}
            for target_word, target_total in total_dic.items():
                target_correct = correct_dic.get(target_word, 0)
                result_dic[target_word] = (target_correct,
                                           target_total,
                                           (target_correct/target_total)*100)

            result_dic['TOTAL'] = (len(checked_words),
                                   correct,
                                   count,
                                   (correct/count)*100)
            result_dic['META'] = META
            pickle.dump(result_dic, fw_result)

        print("The number of target words : ", len(checked_words))
        print("The number of homograph eo-jeul : ", count)
        print("The number of correct answer : ", correct)
        print("Accuracy : ", (correct / count) * 100)


def main():
    """
    this is main function
    """

    evaluate()


if __name__ == '__main__':
    main()
