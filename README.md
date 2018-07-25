# wsd
Word Sense Disambiguation

## Baseline 측정법

1. /baseline 디렉토리의 transform.py 를 이용해 트레이닝 코퍼스와 테스트 코퍼스를 전처리 합니다.
    - transform.py에 커맨드라인 인자로 [처리전 울산 코퍼스 파일이름] 과 [처리후 파일이름] 를 주어 실행하면 됩니다.
    - ex) python3 transform.py ulsan_corpus_train.txt train_set.txt
	  python3 transform.py ulsan_corpus_test.txt test_set.txt

2. /baseline 디렉토리의 baseline.py 를 이용해 MFS 사전을 구축하고 테스트 코퍼스에 대해 중의성 해소 후 정확도를 측정합니다.
    - baseline.py에 커맨드라인 인자로 [전처리된 트레이닝셋 파일이름] 과 [전처리된 테스트셋 파일이름] 을 주어 실행하면 됩니다.
    - ex) python3 baseline.py train_set.txt test_set.txt
    - 실행시 전체 동형이의어 사전 구축, 최빈의미 사전 구축, 단어의미 중의성 해소, 정확도 측정 순으로 작업이 진행됩니다.
    - 전체 동형이의어 사전 구축 및 최빈의미 사전, 중의성해소 결과는 모두 파일 형태로 현재 디렉토리에 저장됩니다.
