import re
from nltk.tokenize import word_tokenize
import nltk
import pickle
import os
import sys

''' You need to download nltk package 'punkt' first'''
#nltk.download('punkt') 


''' FILE PATH '''
''' train_set.txt and test_set.txt should be in current directory '''

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
VOCA = os.path.join(THIS_FOLDER, 'voca.bin')
MAX_FREQ_DIC = os.path.join(THIS_FOLDER, 'max_freq_dic.bin')
ANSWER = os.path.join(THIS_FOLDER, 'answer.txt')

try:
    TRAIN_SET = os.path.join(THIS_FOLDER, sys.argv[1])
    TEST_SET = os.path.join(THIS_FOLDER, sys.argv[2])
except:
    print('usage: python3 baseline.py transformed_train_file_name transfromed_test_file_name')
    sys.exit(1)




''' Function that build homograph vocabulary from training set '''
''' vocabulary is an dictionary, key = "WORD__NN/POS" value = appearing count '''
''' the function saves vocabulary as file "voca.bin" in current directory '''

def buildVoca():
    fr = open(TRAIN_SET, 'r')
    fw_voca = open(VOCA, 'wb')
    tokenList = []
    while True:
        line = fr.readline()
        if not line: break
        line = line.replace("\n","")
        newTokens = re.split('[ ]',line)
        for token in newTokens:
            if re.compile('.*__[\d][\d].*').match(token):
                tokenList.append(token)
    
    text = nltk.Text(tokenList)
    vocabulary = text.vocab()

    pickle.dump(vocabulary, fw_voca)
    fw_voca.close()
    fr.close()

    return


''' Function that build MaxFreqDic from vocabulary '''
''' MaxFreqDic is an dictionary, key = "WORD/POS", value = the most frequent sense of the word (formed "WORD__NN/POS") '''
''' the function saves MaxFreqDic as file "max_freq_dic.bin" in current directory '''

def buildMaxFreqDic():
    fr_voca = open(VOCA, 'rb')
    fw_maxFreqDic = open(MAX_FREQ_DIC, 'wb')
    vocabulary = pickle.load(fr_voca)
    

    keys = vocabulary.keys()
    keysWithoutSense = set(map(lambda y: re.sub('__[\d][\d]', '', y), keys))
    maxFreqDic = {}
    for key in keysWithoutSense:
        splitKey = re.split('/',key)
        word = splitKey[0]
        pos = splitKey[1]

        max = -1
        maxSense = "UNKNOWN"
        for n in range(1,99):
            if n < 10 :
                targetWord = word+"__0"+str(n)+"/"+pos
            else :
                targetWord = word+"__"+str(n)+"/"+pos
            freq = vocabulary.get(targetWord, -1)
            if freq > max:
                max = freq
                maxSense = targetWord

        #keyWithSenseList = list(filter(lambda y: re.match(word+"__[\d][\d]/"+pos, y) != None, keys))
        #if keyWithSenseList != []:
        #    freqList = list(map(lambda y: vocabulary[y], keyWithSenseList))
        #    maxSense = keyWithSenseList[freqList.index(max(freqList))]
        #    maxFreqDic[key] = maxSense
        
        maxFreqDic[key] = maxSense
    
    pickle.dump(maxFreqDic, fw_maxFreqDic)

    fw_maxFreqDic.close()
    fr_voca.close()

''' Function that makes answer from test set '''
''' for each homograph word, this function chooses most frequent sense of the word '''
''' if the homograph word is not trained before, the function chooses "UNKNOWN" as answer word. and it will cause wrong answering in evaluation '''
''' answer sentences are saved as file, "answer.txt" in current directory '''

def makeAnswer():
    fr_test = open(TEST_SET, 'r')
    fw_answer = open(ANSWER, 'w')
    fr_maxFreqDic = open(MAX_FREQ_DIC, 'rb')
    maxFreqDic = pickle.load(fr_maxFreqDic)

    while True:
        line = fr_test.readline()
        if not line: break
        line = line.replace("\n","")
        tokens = re.split('[ ]',line)

        for token in tokens:
            if re.compile('.*__[\d][\d].*').match(token):
                queryWord = re.sub('__[\d][\d]', '', token)
                answerWord = maxFreqDic.get(queryWord, "UNKNOWN")
            else:
                answerWord = token
            fw_answer.write(answerWord+" ")

        fw_answer.write('\n')

    fr_maxFreqDic.close()
    fr_test.close()
    fw_answer.close()
    return


''' evaluate answer by comparing "test_set.txt" with "answer.txt" '''
''' only homograph words are counted for scoring '''
def evaluate():
    fr_test = open(TEST_SET, 'r')
    fr_answer = open(ANSWER, 'r')

    count = 0
    correct = 0

    while True:
        testLine = fr_test.readline().replace("\n","")
        answerLine = fr_answer.readline().replace("\n","")

        if not testLine or not answerLine : break

        testTokens = re.split('[ ]', testLine)
        answerTokens = re.split('[ ]', answerLine)

        mergedTokens = zip(testTokens, answerTokens)

        for token1, token2 in mergedTokens:
            if re.compile('.*__[\d][\d].*').match(token1):
                count = count+1
                if token1 == token2:
                    correct = correct+1

    print("The number of homograph : ",count)
    print("The number of correct answer : ",correct)
    print("Precision : ",(correct/count)*100)
        



buildVoca()
print("building voca done!")

buildMaxFreqDic()
print("building maxFreqDic done!")

makeAnswer()
print("answering done!")

evaluate()


