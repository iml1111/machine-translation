import argparse
import sys
import codecs
from operator import itemgetter
import torch
from modules.data_loader import DataLoader
import modules.data_loader as data_loader
from modules.transformer import Transformer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        required=True,
        help='Model file name to use'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=-1,
        help='GPU ID to use. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Mini batch size for parallel inference. Default=%(default)s'
    )
    p.add_argument(
        '--max_length',
        type=int,
        default=255,
        help='Maximum sequence length for inference. Default=%(default)s'
    )
    p.add_argument(
        '--n_best',
        type=int,
        default=1,
        help='Number of best inference result per sample. Default=%(default)s'
    )
    p.add_argument(
        '--beam_size',
        type=int,
        default=1,
        help='Beam size for beam search. Default=%(default)s'
    )
    p.add_argument(
        '--lang',
        type=str,
        default="enko",
        help='Source language and target language. Example: enko'
    )
    p.add_argument(
        '--length_penalty',
        type=float,
        default=1.2,
        help='Length penalty parameter that higher value produce shorter results. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def read_text(batch_size=128):
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= batch_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                break
            else:
                line += [vocab.itos[index]]

        line = ' '.join(line)
        lines += [line]

    return lines


def get_vocabs(train_config, config, saved_data):
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']       
    return src_vocab, tgt_vocab


def get_model(input_size, output_size, train_config):
    model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_splits=train_config.n_splits,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
    model.load_state_dict(saved_data['model'])
    model.eval() 
    return model


if __name__ == '__main__':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    config = define_argparser()

    '''
    이전에 학습했던 모델, 파라미터,
    vocab 데이터까지 전부 저장해뒀다가 그대로 다시 씀
    '''
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu',
    )

    train_config = saved_data['config']
    src_vocab, tgt_vocab = get_vocabs(train_config, config, saved_data)

    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab)

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, train_config)
    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    with torch.no_grad():
        for lines in read_text(batch_size=config.batch_size):
            '''
            신경망 특성상, 문장을 긴순으로 정렬해서 텐서를 만들어야함
            하지만 그경우, 입력 순서가 뒤틀릴 수 있으므로
            original_indice를 기억해놨다가 다시 복구시켜야함
            '''
            lengths = [len(line) for line in lines]
            original_indice = [i for i in range(len(lines))]
            sorted_tuples = sorted(
                zip(lines, lengths, original_indice),
                key=itemgetter(1),
                reverse=True,
            )

            sorted_lines    = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths         = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # x = (batch_size, length)
            x = loader.src.numericalize(
                loader.src.pad(sorted_lines),
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            )

            if config.beam_size == 1:
                y_hats, indice = model.search(x)
                # |y_hats| = (batch_size, length, output_size)
                # |indice| = (batch_size, length)

                output = to_text(indice, loader.tgt.vocab)
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                sys.stdout.write('\n'.join(output) + '\n')
            else:
                # Take mini-batch parallelized beam search.
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty,
                )

                # Restore the original_indice.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')
