import argparse
import pprint
import torch
from torch import optim
import torch.nn as nn

from modules.data_loader import DataLoader
import modules.data_loader as data_loader
from modules.models.transformer import Transformer
from modules.trainer import Trainer
from modules.trainer import IgniteEngine


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--model_fn',
        default='./models/model.pth',
        help='모델을 저장할 파일 이름을 입력해주세요.'
    )
    p.add_argument(
        '--train',
        default='./data/corpus.shuf.train.tok.bpe',
        help='Training set에 사용될 Filepath (뒤의 en,ko를 제외해야 합니다) (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        default='./data/corpus.shuf.valid.tok.bpe',
        help='Validation set에 사용될 Filepath (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        default='enko',
        help='학습시킬 언어의 쌍 (영어를 한국어로 번역할 경우, ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='학습에 사용할 GPU ID를 입력해주세요. -1 for CPU. Default=%(default)s'
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='batch size. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=30,
        help='학습할 epochs 수. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=False,
        type=int,
        default=1,
        help='시작 epochs 값. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='학습시, 한 문장의 최대 토큰 길이. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Transformer ENC, DEC에 사용될 Hidden size. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='ENC, DEC의 layers 수 Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=1e+8,
        help='Gradient Clipping을 위한 max grad norm. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=32,
        help='Gradient Accumulation을 위한 파라미터 갱신 주기. Default=%(default)s'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='learning rate. Default=%(default)s',
    )
    p.add_argument(
        '--n_splits',
        type=int,
        default=8,
        help='multi-head attention의 Head 수. Default=%(default)s',
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Automatic Mixed Precision Turn-off 여부',
    )

    config = p.parse_args()

    return config


def get_model(input_size, output_size, config):
    model = Transformer(
            input_size, # Source vocabulary size
            config.hidden_size, # Transformer doesn't need word_vec_size.
            output_size, # Target vocabulary size
            n_splits=config.n_splits, # Number of head in Multi-head Attention.
            n_enc_blocks=config.n_layers,# Number of encoder blocks
            n_dec_blocks=config.n_layers,# Number of decoder blocks
            dropout_p=config.dropout, # Dropout rate on each block
        )
    return model


def get_crit(output_size, pad_index):
    # PAD 토큰에 대하여 가중치를 주지 않도록 설정
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )
    return crit


def get_optimizer(model, config):
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr,
        betas=(.9, .98) # In Pre-LN Paper
    ) 
    return optimizer


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:2], config.lang[-2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    if model_weight:
        model.load_state_dict(model_weight)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    if opt_weight:
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = None

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    trainer = Trainer(IgniteEngine, config)
    trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)
