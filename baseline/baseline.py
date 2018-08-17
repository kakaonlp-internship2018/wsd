"""
baseline evalutator
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
import nltk

# You need to download nltk package 'punkt' first
# nltk.download('punkt')


# FILE PATH #
# train_set.txt and test_set.txt should be in current directory #

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
ANSWER = os.path.join(THIS_FOLDER, 'answer.txt')

TKN_PTN = re.compile(r'.*__[\d][\d].*')

try:
    TRAIN_SET = os.path.join(THIS_FOLDER, sys.argv[1])
    TEST_SET = os.path.join(THIS_FOLDER, sys.argv[2])
except BaseException:
    print('usage: python3 baseline.py transformed_train_file_name transfromed_test_file_name')
    sys.exit(1)


#############
# functions #
#############

def build_voca_with_freq_dic():
    """
    voca : key = word_without_sense, value = freq_dic
    freq_dic : key = sense_number, value = appearing count
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

def build_voca():
    """
    Function that build homograph vocabulary from training set
    vocabulary is an dictionary, key = "WORD__NN/POS" value = appearing count
    the function saves vocabulary as file "voca.bin" in current directory
    """

    with open(TRAIN_SET, 'r') as fr_train, open(VOCA, 'wb') as fw_voca:
        token_list = []
        for line in fr_train:
            line = line.replace("\n", "")
            new_tokens = re.split('[ ]', line)
            for token in new_tokens:
                if TKN_PTN.match(token):
                    token_list.append(token)

        text = nltk.Text(token_list)
        vocabulary = text.vocab()

        pickle.dump(vocabulary, fw_voca)

def build_max_freq_dic2():
    """
    another version of build_max_freq_dic function
    It uses build_voca_with_freq_dic()
    """
    with open(VOCA, 'rb') as fr_voca, open(MAX_FREQ_DIC, 'wb') as fw_max_freq_dic:
        vocabulary = pickle.load(fr_voca)
        print(len(vocabulary))
        max_freq_dic = {}
        for key, freq_dic in vocabulary.items():
            max_freq_dic[key] \
            = key[:key.index("/")] + "__" + max(freq_dic, key=freq_dic.get) + key[key.index("/"):]

        pickle.dump(max_freq_dic, fw_max_freq_dic)



def build_max_freq_dic():
    """
    Function that build MaxFreqDic from vocabulary
    MaxFreqDic is an dictionary,
    key = "WORD/POS", value = the most frequent sense of the word (formed "WORD__NN/POS")
    the function saves MaxFreqDic as file "max_freq_dic.bin" in current directory
    """

    with open(VOCA, 'rb') as fr_voca, open(MAX_FREQ_DIC, 'wb') as fw_max_freq_dic:
        vocabulary = pickle.load(fr_voca)
        keys = vocabulary.keys()
        keys_without_sense = set(map(lambda y: re.sub(r'__[\d][\d]', '', y), keys))
        max_freq_dic = {}


        for key in keys_without_sense:
            split_key = re.split('/', key)
            word = split_key[0]
            pos = split_key[1]

            max_freq = -1
            max_sense = "UNKNOWN"
            for num in range(1, 99):
                target_word = '{}__{:02d}/{}'.format(word, num, pos)
                freq = vocabulary.get(target_word, -1)
                if freq > max_freq:
                    max_freq = freq
                    max_sense = target_word

            max_freq_dic[key] = max_sense

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
                if TKN_PTN.match(token1):
                    count = count + 1
                    if token1 == token2:
                        correct = correct + 1

        print("The number of homograph : ", count)
        print("The number of correct answer : ", correct)
        print("Accuracy : ", (correct / count) * 100)


def main():
    """
    this is main function
    """

    #build_voca()
    #build_voca_with_freq_dic()
    #print("building voca done!")

    #build_max_freq_dic()
    build_max_freq_dic2()
    print("building max_freq_dic done!")

    make_answer()
    print("answering done!")

    evaluate()

if __name__ == '__main__':
    main()
