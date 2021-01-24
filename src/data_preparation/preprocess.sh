# 코퍼스 셔플
shuf ../data/corpus.tsv > ../data/corpus.shuf.tsv

# 코퍼스 셋 나누기
head -n 1200000 ../data/corpus.shuf.tsv > ../data/corpus.shuf.train.tsv
tail -n 402409 ../data/corpus.shuf.tsv > ../data/temp.tsv
head -n 200000 ../data/temp.tsv > ../data/corpus.shuf.valid.tsv
rm ../data/temp.tsv
tail -n 202409 ../data/corpus.shuf.tsv > ../data/corpus.shuf.test.tsv

# 코퍼스 한영 말뭉치 분리
cut -f1 ../data/corpus.shuf.train.tsv > ../data/corpus.shuf.train.ko
cut -f2 ../data/corpus.shuf.train.tsv > ../data/corpus.shuf.train.en
cut -f1 ../data/corpus.shuf.valid.tsv > ../data/corpus.shuf.valid.ko
cut -f2 ../data/corpus.shuf.valid.tsv > ../data/corpus.shuf.valid.en
cut -f1 ../data/corpus.shuf.test.tsv > ../data/corpus.shuf.test.ko
cut -f2 ../data/corpus.shuf.test.tsv > ../data/corpus.shuf.test.en
# head -n 3 ../data/corpus.shuf.*.ko
# head -n 3 ../data/corpus.shuf.*.en
# wc -l ../data/corpus.shuf.*.*

# 코퍼스 토크나이징
cat ../data/corpus.shuf.train.ko | mecab -O wakati -b 99999 | python3 post_tokenize.py ../data/corpus.shuf.train.ko > ../data/corpus.shuf.train.tok.ko
cat ../data/corpus.shuf.train.en | python3 tokenizer.py | python3 post_tokenize.py ../data/corpus.shuf.train.en > ../data/corpus.shuf.train.tok.en
cat ../data/corpus.shuf.valid.ko | mecab -O wakati -b 99999 | python3 post_tokenize.py ../data/corpus.shuf.valid.ko > ../data/corpus.shuf.valid.tok.ko
cat ../data/corpus.shuf.valid.en | python3 tokenizer.py | python3 post_tokenize.py ../data/corpus.shuf.valid.en > ../data/corpus.shuf.valid.tok.en
cat ../data/corpus.shuf.test.ko | mecab -O wakati -b 99999 | python3 post_tokenize.py ../data/corpus.shuf.test.ko > ../data/corpus.shuf.test.tok.ko
cat ../data/corpus.shuf.test.en | python3 tokenizer.py | python3 post_tokenize.py ../data/corpus.shuf.test.en > ../data/corpus.shuf.test.tok.en
# head -n 3 ../data/corpus.shuf.*.tok.*
# wc -l ../data/corpus.shuf.*.tok.*

# 코퍼스 서브워드 세그멘테이션
# 보통 트레인 셋에 대한 bpe만 학습시켜서 valid, test에도 일관적으로 적용
# symbol의 경우, BPE 알고리즘 상에서 몇번이나 머지를 시도할 것인지 물어봄
# 너무 합쳐졌다 싶으면 심볼을 낮춤
# 너무 안 찢어졌다 싶으면 심볼을 높임
python3 learn_bpe.py --input ../data/corpus.shuf.train.tok.ko --output bpe.ko.model --symbols 30000 --verbose
python3 learn_bpe.py --input ../data/corpus.shuf.train.tok.en --output bpe.en.model --symbols 50000 --verbose
cat ../data/corpus.shuf.train.tok.ko | python3 apply_bpe.py --c bpe.ko.model > ../data/corpus.shuf.train.tok.bpe.ko
cat ../data/corpus.shuf.train.tok.en | python3 apply_bpe.py --c bpe.en.model > ../data/corpus.shuf.train.tok.bpe.en
cat ../data/corpus.shuf.valid.tok.ko | python3 apply_bpe.py --c bpe.ko.model > ../data/corpus.shuf.valid.tok.bpe.ko
cat ../data/corpus.shuf.valid.tok.en | python3 apply_bpe.py --c bpe.en.model > ../data/corpus.shuf.valid.tok.bpe.en
cat ../data/corpus.shuf.test.tok.ko | python3 apply_bpe.py --c bpe.ko.model > ../data/corpus.shuf.test.tok.bpe.ko
cat ../data/corpus.shuf.test.tok.en | python3 apply_bpe.py --c bpe.en.model > ../data/corpus.shuf.test.tok.bpe.en
# head -n 3 ../data/corpus.shuf.*.tok.bpe.*
# wc -l ../data/corpus.shuf.*.tok.bpe.*