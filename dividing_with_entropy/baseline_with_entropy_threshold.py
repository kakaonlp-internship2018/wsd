"""
baseline evalutator with entropy threshold
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
#import operator
import numpy as np



# FILE PATH #
# train_set.txt and test_set.txt should be in current directory #

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
ANSWER = os.path.join(THIS_FOLDER, 'answer.txt')
ENTROPY_DIC = os.path.join(THIS_FOLDER, 'entropy_dic.bin')

TKN_PTN = re.compile(r'.*__[\d][\d].*')

try:
    TRAIN_SET = os.path.join(THIS_FOLDER, sys.argv[1])
    TEST_SET = os.path.join(THIS_FOLDER, sys.argv[2])
    ENTROPY_THRESHOLD = float(sys.argv[3])
except BaseException:
    print('usage: python3 baseline.py [transformed_train_file] [transfromed_test_file] entropy_threshold')
    sys.exit(1)


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
            max_freq_dic[key] \
            = key[:key.index("/")] + "__" + max(freq_dic, key=freq_dic.get) + key[key.index("/"):]
            entropy_dic[key] = calculate_entropy(freq_dic.values())

        #sorted_ent_list = sorted(entropy_dic.items(), \
        #                key=operator.itemgetter(1), reverse=True)

        pickle.dump(entropy_dic, fw_entropy_dic)
        pickle.dump(max_freq_dic, fw_max_freq_dic)



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
        open(MAX_FREQ_DIC, 'rb') as fr_max_freq_dic:

        max_freq_dic = pickle.load(fr_max_freq_dic)
        for line in fr_test:
            line = line.replace("\n", "")
            tokens = re.split('[ ]', line)

            for token in tokens:
                if TKN_PTN.match(token):
                    query_word = re.sub(r'__[\d][\d]', '', token)
                    answer_word = max_freq_dic.get(query_word, "UNKNOWN")
                    # TODO: answer_word = answer_with_entropy(query_word, low_threshold, high_threshold)
                else:
                    answer_word = token
                fw_answer.write(answer_word + " ")

            fw_answer.write('\n')

# TODO
def answer_with_entropy(query_word, low_th, high_th):
    """
    Args:
        query_word: query_word to be disambiguated
        low_th : lower entropy threshold
        high_th : higher entropy threshold

    Return:
        answer_word (formed "WORD__NN/POS)
    """
    # 입력 엔트로피에 기준에 따라 낮은 파트 중간 파트 높은 파트 모델 다른거 써서 answer_word 찾아냄.

def evaluate():
    """
    # evaluate answer by comparing "test_set.txt" with "answer.txt" #
    # only homograph words are counted for scoring #
    """

    with open(TEST_SET, 'r') as fr_test, open(ANSWER, 'r') as fr_answer, open(ENTROPY_DIC, 'rb') as fr_ent_dic:
        
        ent_dic = pickle.load(fr_ent_dic)
        count = 0
        correct = 0

        while True:
            test_line = fr_test.readline().replace("\n", "")
            answer_line = fr_answer.readline().replace("\n", "")

            if not test_line or not answer_line:
                break

            test_tokens = re.split('[ ]', test_line)
            answer_tokens = re.split('[ ]', answer_line)

            merged_tokens = zip(test_tokens, answer_tokens)

            for token1, token2 in merged_tokens:
                if TKN_PTN.match(token1) and ent_dic.get(re.sub(r'__[\d][\d]', '', token1), 0) >= ENTROPY_THRESHOLD:
                    count = count + 1
                    if token1 == token2:
                        correct = correct + 1

        print("The number of homograph eo-jeul : ", count)
        print("The number of correct answer : ", correct)
        print("Accuracy : ", (correct / count) * 100)


def main():
    """
    this is main function
    """

    #build_voca()
    #print("building voca done!")

    build_max_freq_dic_and_ent_dic()
    print("building max_freq_dic and entropy_dic done!")

    make_answer()
    print("answering done!")

    evaluate()

if __name__ == '__main__':
    main()
