# Data Preparation

학습을 수행하기 위해, 데이터셋에 대하여 다음과 같은 전처리 과정이 필요합니다.

- **Corpora Shuffle (코포라 셔플)**
- **ko/en division of Corpora(코포라 한영 말뭉치 분리)**
- **Tokenize**
- **Subword Segmentation**



## Sample Dataset

[/src/data/](https://github.com/iml1111/machine-translation/tree/main/src/data) 경로에는 실제로 학습 데이터가 전처리되어 가는 과정을 10개의 샘플에 대하여 준비해두었습니다.

[corpus.sample.tsv](https://github.com/iml1111/machine-translation/blob/main/src/data/corpus.sample.tsv)에서 시작하여 위와 같은 과정을 거쳐 최종적으로 [corpus.sample.tok.bpe.en](https://github.com/iml1111/machine-translation/blob/main/src/data/corpus.sample.tok.bpe.en), [corpus.sample.tok.bpe.ko](https://github.com/iml1111/machine-translation/blob/main/src/data/corpus.sample.tok.bpe.ko)의 형태로 전처리 과정이 완료되게 됩니다.



## Procedure

전처리 과정은 다음 스크립트를 순서대로 실행해주면 됩니다. 자세한 사항은 [/src/data_preparation/preprocess.sh](https://github.com/iml1111/machine-translation/blob/main/src/data_preparation/preprocess.sh)에서도 확인하실 수 있습니다.



### 코퍼스 셔플하기

```bash
$ cd /src/data_preparation
$ shuf ../data/corpus.tsv > ../data/corpus.shuf.tsv
```

### 코퍼스 train, valid, test 셋 나누기

```bash
$ head -n 1200000 ../data/corpus.shuf.tsv > ../data/corpus.shuf.train.tsv
$ tail -n 402409 ../data/corpus.shuf.tsv > ../data/temp.tsv
$ head -n 200000 ../data/temp.tsv > ../data/corpus.shuf.valid.tsv
$ rm ../data/temp.tsv
$ tail -n 202409 ../data/corpus.shuf.tsv > ../data/corpus.shuf.test.tsv
```

### 코퍼스 한/영 말뭉치 분리하기

```bash
$ cut -f1 ../data/corpus.shuf.train.tsv > ../data/corpus.shuf.train.ko
$ cut -f2 ../data/corpus.shuf.train.tsv > ../data/corpus.shuf.train.en
$ cut -f1 ../data/corpus.shuf.valid.tsv > ../data/corpus.shuf.valid.ko
$ cut -f2 ../data/corpus.shuf.valid.tsv > ../data/corpus.shuf.valid.en
$ cut -f1 ../data/corpus.shuf.test.tsv > ../data/corpus.shuf.test.ko
$ cut -f2 ../data/corpus.shuf.test.tsv > ../data/corpus.shuf.test.en
```

### 코퍼스 토크나이징

토크나이징을 하기 위해서는 토크나이저가 필요합니다.

**영어의 경우, mosestokenizer**를 사용했으며, **한글의 경우 Mecab**을 사용하였습니다.

**Mecab**의 설치에 도움이 필요하다면 [여기](https://blog.naver.com/shino1025/222179854044)를 참고해주세요.

**두가지 토크나이저가 모두 설치되어 있어야만 해당 과정을 수행할 수 있습니다.**

```bash
$ cat ../data/corpus.shuf.train.ko | mecab -O wakati -b 99999 | python3 post_tokenize.py ../data/corpus.shuf.train.ko > ../data/corpus.shuf.train.tok.ko
$ cat ../data/corpus.shuf.train.en | python3 tokenizer.py | python3 post_tokenize.py ../data/corpus.shuf.train.en > ../data/corpus.shuf.train.tok.en
$ cat ../data/corpus.shuf.valid.ko | mecab -O wakati -b 99999 | python3 post_tokenize.py ../data/corpus.shuf.valid.ko > ../data/corpus.shuf.valid.tok.ko
$ cat ../data/corpus.shuf.valid.en | python3 tokenizer.py | python3 post_tokenize.py ../data/corpus.shuf.valid.en > ../data/corpus.shuf.valid.tok.en
$ cat ../data/corpus.shuf.test.ko | mecab -O wakati -b 99999 | python3 post_tokenize.py ../data/corpus.shuf.test.ko > ../data/corpus.shuf.test.tok.ko
$ cat ../data/corpus.shuf.test.en | python3 tokenizer.py | python3 post_tokenize.py ../data/corpus.shuf.test.en > ../data/corpus.shuf.test.tok.en
```

### 코퍼스 서브워드 세그멘테이션

서브워드 분절을 수행하기 위해서는 먼저 BPE 모델을 학습시켜야 합니다. 일반적으로 train set에 대하여만 학습을 시켜 valid, test set에도 일관적으로 분절을 적용시킵니다.

**BPE 모델 (한/영) 학습시키기**

BPE 모델을 학습시킬 떄, Symbol 인자로 hyperparameter로 사용됩니다. symbole은 BPE 알고리즘 수행중, 몇 번이나 단어의 머지를 수행할 것인가를 정하는 것으로 일반적으로,

- 너무 합쳐졌다 싶으면, Symbol을 하향
- 너무 흩어졌다 싶으면, Symbole을 상향

이러한 방식으로 파라미터 튜닝을 수행합니다.

```bash
$ python3 learn_bpe.py --input ../data/corpus.shuf.train.tok.ko --output bpe.ko.model --symbols 30000 --verbose
$ python3 learn_bpe.py --input ../data/corpus.shuf.train.tok.en --output bpe.en.model --symbols 50000 --verbose
```



**학습된 BPE 모델로 서브워드 분절시키기**

```bash
$ cat ../data/corpus.shuf.train.tok.ko | python3 apply_bpe.py --c bpe.ko.model > ../data/corpus.shuf.train.tok.bpe.ko
$ cat ../data/corpus.shuf.train.tok.en | python3 apply_bpe.py --c bpe.en.model > ../data/corpus.shuf.train.tok.bpe.en
$ cat ../data/corpus.shuf.valid.tok.ko | python3 apply_bpe.py --c bpe.ko.model > ../data/corpus.shuf.valid.tok.bpe.ko
$ cat ../data/corpus.shuf.valid.tok.en | python3 apply_bpe.py --c bpe.en.model > ../data/corpus.shuf.valid.tok.bpe.en
$ cat ../data/corpus.shuf.test.tok.ko | python3 apply_bpe.py --c bpe.ko.model > ../data/corpus.shuf.test.tok.bpe.ko
$ cat ../data/corpus.shuf.test.tok.en | python3 apply_bpe.py --c bpe.en.model > ../data/corpus.shuf.test.tok.bpe.en
```



