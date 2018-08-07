"""
statistic.py
__author__ = 'jeff.yu (jeff.yu@kakaocorp.com)'
__copyright__ = 'No copyright, just copyleft!'
"""
import pickle

PRINT_ENT_INFO = False
PRINT_WORD_COUNT = True

def main():
    """
    this is main function
    """
    with open("/Users/kakao/WSD/wsd/entropy_multi_model/entropy_dic.bin", 'rb') as fr_ent, \
    open("/Users/kakao/WSD/wsd/entropy_multi_model/voca.bin", 'rb') as fr_voca:
        ent_dic = pickle.load(fr_ent)
        voca = pickle.load(fr_voca)

        ent_list = {}
        ent_list['0'] = [(key, ent) for key, ent in ent_dic.items() if ent == 0]
        ent_list['0.1'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 0.09 and ent <= 0.1]
        ent_list['0.3'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 0.29 and ent <= 0.3]
        ent_list['0.5'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 0.49 and ent <= 0.5]
        ent_list['0.7'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 0.69 and ent <= 0.7]
        ent_list['1'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 0.99 and ent <= 1]
        ent_list['1.5'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 1.49 and ent <= 1.5]
        ent_list['2'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 1.99 and ent <= 2]
        ent_list['2.5'] = [(key, ent) for key, ent in ent_dic.items() if ent >= 2.39 and ent <= 2.5]

        if PRINT_ENT_INFO:
            for ent, key_ent_list in ent_list.items():
                for index, (key, _) in enumerate(key_ent_list):
                    if index > 2:
                        break
                    print("word: {}, entropy: {}, distribution: {}".format(key, \
                        ent, [v for k, v in voca[key].items()]))
                print()

        if PRINT_WORD_COUNT:
            print("total homographs: {}\nentropy 0 homographs: {}\nentropy over 0.1 homographs: {}".
                  format(len(ent_dic.items()), len([k for k, v in ent_dic.items() if v == 0]), \
                        len([k for k, v in ent_dic.items() if v >= 0.1])))

if __name__ == '__main__':
    main()
