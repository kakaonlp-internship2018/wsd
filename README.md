# WSD
Word Sense Disambiguation
  
실험에 대한 전반적인 설명은 [발표자료](https://github.com/kakaonlp-internship2018/wsd/blob/ene5135-for-presentation/jeff_WSD_0829.pdf)를 참고하세요

## 코퍼스 전처리
1. /entropy_multi_model 디렉토리의 transform.py 를 이용해 트레이닝 코퍼스와 테스트 코퍼스를 전처리 합니다.
    - transform.py에 커맨드라인 인자로 [처리전 울산 코퍼스 파일이름] 과 [처리후 파일이름] 를 주어 실행하면 됩니다. 
    - 전처리 과정에서 의미번호가 80 이상인 단어들의 의미태그를 제거하며, 모든 숫자단어들을 "NUM" 으로 치환합니다.
    ~~~ 
    $ python3 transform.py ulsan_corpus_train.txt train_set.txt  
    $ python3 transform.py ulsan_corpus_test.txt test_set.txt 
    ~~~  
2. 위와같이 전처리를 거치지 않더라도 코퍼스가 다음과같이 
    - 형태소 분석 및 의미번호 태깅이 되어있는 문장이  
    - 한 Line당 하나씩 있는 형태면 됩니다.
    ~~~
    ...
    13 검사__03/NNG 비용__03/NNG 은/JX NUM/SN 만__06/NR 원__01/NNB 정도__11/NNG ./SF
    14 감마__02/NNG 나이프/NNG 로/JKB 간질__04/NNG 치료/NNG
    15 ..
    ...
    ~~~

## Glove word embedding model 준비
1. 워드 임베딩 학습을 위한 코퍼스를 준비합니다. (ex. KOMORAN 형태소 분석기를 거친 한국어 위키덤프 코퍼스)  
2. [glove 사용법](https://github.com/stanfordnlp/GloVe)에 따라 워드임베딩 모델을 학습합니다. vectors.txt 파일로 결과가 저장됩니다. 
    ~~~
    $ git clone http://github.com/stanfordnlp/glove
    $ cd glove && make
    $ ./demo.sh 
    ~~~ 
    이때 demo.sh의 CORPUS 부분을 수정하여 준비한 코퍼스 경로를 입력해줍니다.  
3. 생성된 vectors.txt 파일을 glove/vectors.txt 경로에 준비합니다.  

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
      
      
## 딥러닝 모델 측정법
1. /pytorch 디렉토리의 practice_mixed_batch.py 를 실행하여 모델 학습 및 평가에 필요한 각종 파일들을 준비하고 학습 및 평가 해볼 수 있습니다.
    - 다음과 같이 commandline arguments 를 확인할 수 있습니다.
    ~~~
    $ python3 practice_mixed_batch.py -h
    positional arguments:
      Training_set         Transformed training set file
      Test_set             Transformed test set file

    optional arguments:
      -h, --help           show this help message and exit
      --dim DIM            Word embedding dimension, default=200
      --batch BATCH        Set batch size, default=256
      --epoch EPOCH        Set epoch size, default=300
      --patience PATIENCE  Set patience, default=10
      --target TARGET      Specify target word for disambiguation, default=None
      --gpu GPU            select GPU device, default=0
      --relu               apply relu, default=False
      --early              apply early stopping, default=True
      --dropout            apply dropout, default=False
      --verbose            Print every target word accuracy, default=True
      --data               Build data set, default=False
      --nword NWORD        the number of target word, default=-1
      --win WIN            window_size, default=-1
      --exp EXP            name of experiment, default=temp_experiment
      --check              Check result, default=False
      --best               User model that has best validation acc to evaluate,
                           default=False
      --weak WEAK          set weakness, default=-1
    ~~~  
    
2. 최초실행시 다음과 같이 코퍼스를 통해 학습 및 테스트 인스턴스를 생성합니다. 같은 디렉토리에 vectors.bin 파일과 ent_dic.bin 파일이 필요합니다.
    ~~~
    $ python3 practice_mixed_batch.py training_set.txt test_set.txt --data
    ~~~
  
3. 커맨드라인 인자를 통해 모델의 여러 하이퍼파라미터를 지정해줄수 있습니다.  
    - ex) early stopping patience 10, 세번째 GPU device 사용, best validation model 사용, 타겟 단어 지정, 실험 이름 지정
    ~~~
    $ python3 practice_mixed_batch.py training_set.txt test_set.txt --patience 10 --gpu 2 --best --target 관/NNG --exp 관_실험
    ~~~
4. 평가 결과는 results/exp_name.bin 으로 저장되며 다음과 같이 결과를 확인할 수 있습니다.
    ~~~
    $ python3 practice_mixed_batch.py training_set.txt test_set.txt --check --exp exp_name
    ~~~
