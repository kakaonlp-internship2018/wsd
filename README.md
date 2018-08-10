# wsd
Word Sense Disambiguation

## 코퍼스 전처리
1. /entropy_multi_model 디렉토리의 transform.py 를 이용해 트레이닝 코퍼스와 테스트 코퍼스를 전처리 합니다.
    - transform.py에 커맨드라인 인자로 [처리전 울산 코퍼스 파일이름] 과 [처리후 파일이름] 를 주어 실행하면 됩니다. 
    - 전처리 과정에서 의미번호가 80 이상인 단어들의 의미태그를 제거하며, 모든 숫자단어들을 "NUM" 으로 치환합니다.
    ~~~ 
    $ python3 transform.py ulsan_corpus_train.txt train_set.txt  
    $ python3 transform.py ulsan_corpus_test.txt test_set.txt 
    ~~~
    
## Glove word embedding model 준비
1. 워드 임베딩 학습을 위한 코퍼스를 준비합니다. (ex. KOMORAN 형태소 분석기를 거친 한국어 위키덤프 코퍼스)  
2. [glove 사용법](https://github.com/kakaonlp-internship2018/wsd/blob/feature/3/entropy_multi_model/glove/README.md)에 따라 워드임베딩 모델을 학습합니다. vectors.txt 파일로 결과가 저장됩니다.  
3. vectors.txt 파일을 glove/vectors.txt 경로에 준비합니다.  

## Baseline 측정법
1. /baseline 디렉토리의 baseline.py 를 이용해 MFS 사전을 구축하고 테스트 코퍼스에 대해 중의성 해소 후 정확도를 측정합니다.
    - baseline.py에 커맨드라인 인자로 [전처리된 트레이닝셋 파일이름] 과 [전처리된 테스트셋 파일이름] 을 주어 실행하면 됩니다.
    ~~~
    $ python3 baseline.py train_set.txt test_set.txt
    ~~~
    - 실행시 전체 동형이의어 사전 구축, 최빈의미 사전 구축, 단어의미 중의성 해소, 정확도 측정 순으로 작업이 진행됩니다.
    - 전체 동형이의어 사전 구축 및 최빈의미 사전, 중의성해소 결과는 모두 파일 형태로 현재 디렉토리에 저장됩니다.

## MFS + SVM 모델 측정법
1. /entropy_multi_model 디렉토리의 resource_builder.py 를 실행하여 모델 학습 및 평가에 필요한 각종 파일들을 준비합니다.  
    - 다음과 같이 commandline arguments 를 확인할 수 있습니다.
    ~~~
    $ python3 resource_builder.py -h
    usage: resource_builder.py [-h] [--ent ENT] [--min_max] [--win WIN]
                           [--dim DIM] [--merge {concat,sum}]
                           Training_set

    This is resource builder.

    positional arguments:
        Training_set          Transformed training set file

    optional arguments:
      -h, --help            show this help message and exit
      --ent ENT             Entropy threshold, default=0.1
      --min_max             Add min_max vector to feature vector
      --win WIN             Set window size, default=2
      --dim DIM             Set embedding dimension, default=100
      --merge {concat,sum}  How to merge feature vector, default=concat
    ~~~
    - 필수 인자로 전처리된 training_set 파일을 받으며 나머지 인자에 대한 설명은 위 내용과 같습니다.
    - 최종적으로 svm_dic.bin 파일이 생성되어 이를 evaluation에 사용하게됩니다.
    - 실행예시 (entropy 0.1 이상인 단어에 대해서만 svm 모델을 생성하며, 피쳐벡터를 앞뒤 단어 2개씩 concatenate 하여 구성, 이때 워드임베딩 차원은 100차원, 피쳐벡터에 MIN, MAX 벡터도 추가)
    ~~~
    $ python3 resource_builder.py train_set.txt --ent 0.1 --min_max --win 2 --dim 100 --merge concat
    ~~~
    
2. /entropy_multi_model 디렉토리의 multi_model_evaluator.py 를 실행하여 테스트셋의 중의성 해소를 진행하고 평가합니다.
    - 다음과 같이 commandline arguments 를 확인할 수 있습니다.
    ~~~
    $ python3 multi_model_evaluator.py -h
    usage: multi_model_evaluator.py [-h] [--ent ENT] [--svm SVM] [--mfs] Test_set

    This is multi_model_evaluator.

    positional arguments:
      Test_set    Transformed test set file

    optional arguments:
      -h, --help  show this help message and exit
      --ent ENT   Entropy threshold, default=0.1
      --svm SVM   Specify svm_dic file, default="svm_dic.bin"
      --mfs       set this if you want to apply MFS to low level words
    ~~~
    - 필수 인자로 전처리된 test_set 파일을 받으며 나머지 인자에 대한 설명은 위 내용과 같습니다.
    - 중의성 해소를 위해 1번에서 만들어진 svm_dic.bin 파일을 사용합니다.
    - 실행예시 (entropy 0.3 이상인 단어에 대해서 svm 모델을 적용, svm_dic 파일을 직접 지정, 엔트로피 0.3 미만인 단어에 대해서 mfs를 적용한 정확도로 출력
    ~~~
    $ python3 multi_model_evaluator.py test_set.txt --ent 0.3 --svm my_svm_dic.bin --mfs
    ~~~
    
