import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import modules.data_loader as data_loader
from modules.search import SingleBeamSearchBoard

class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, emb):
        
        if isinstance(emb, tuple):
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = emb
        y, h = self.rnn(x)
        
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=word_vec_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)
        y, h = self.rnn(x, h_t_1)
        return y, h


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        query = self.linear(h_t_tgt)
        weight = torch.bmm(query, h_src.transpose(1, 2))

        if mask is not None:
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))

        weight = self.softmax(weight)
        context_vector = torch.bmm(weight, h_src)
        return context_vector


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        y = self.softmax(self.output(x))
        return y


class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2
    ):
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super(Seq2Seq, self).__init__()

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size=word_vec_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size=word_vec_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.generator = Generator(hidden_size, output_size)

    def generate_mask(self, x, length):
        mask = []
        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [
                    torch.cat(
                        [   
                            x.new_ones(1, l).zero_(),
                            x.new_ones(1, (max_length - l)),
                        ],
                        dim=-1
                    )
                ]
            else:
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()
        return mask

    def merge_encoder_hiddens(self, encoder_hiddens):
        h_0_tgt, c_0_tgt = encoder_hiddens
        batch_size = h_0_tgt.size(1)

        h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(
            batch_size,
            -1,
            self.hidden_size
        ).transpose(0, 1).contiguous()
        c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(
            batch_size,
            -1,
            self.hidden_size
        ).transpose(0, 1).contiguous()

        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):
        mask = None
        x_length = None
        
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]
        
        emb_src = self.emb_src(x)

        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        emb_tgt = self.emb_dec(tgt)

        h_tilde = []
        h_t_tilde = None
        decoder_hidden = h_0_tgt
        for t in range(tgt.size(1)):
            emb_t = emb_tgt[:, t, :].unsqueeze(1)
            decoder_output, decoder_hidden = self.decoder(
                emb_t,
                h_t_tilde,
                decoder_hidden
            )
            context_vector = self.attn(h_src, decoder_output, mask)

            h_t_tilde = self.tanh(
                self.concat(
                    torch.cat(
                        [
                            decoder_output,
                            context_vector
                        ], dim=-1
                    )
                )
            )

            h_tilde += [h_t_tilde]

        h_tilde = torch.cat(h_tilde, dim=1)
        y_hat = self.generator(h_tilde)
        return y_hat

    def search(self, src, is_greedy=True, max_length=255):
        mask = None
        x_length = None
        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        decoder_hidden = self.merge_encoder_hiddens(h_0_tgt)

        y = x.new(batch_size, 1).zero_() + data_loader.BOS

        is_decoding = x.new_ones(batch_size, 1).bool()
        h_t_tilde, y_hats, indice = None, [], []

        while is_decoding.sum() > 0 and len(indice) < max_length:
            emb_t = self.emb_dec(y)
            decoder_output, decoder_hidden = self.decoder(
                emb_t,
                h_t_tilde,
                decoder_hidden
            )
            context_vector = self.attn(h_src, decoder_output, mask)
            h_t_tilde = self.tanh(
                self.concat(
                    torch.cat(
                        [
                            decoder_output,
                            context_vector
                        ], dim=-1
                    )
                )
            )
            
            y_hat = self.generator(h_t_tilde)
            y_hats += [y_hat]

            if is_greedy:
                y = y_hat.argmax(dim=-1)
            else:
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)

            y = y.masked_fill_(~is_decoding, data_loader.PAD)
            is_decoding = is_decoding * torch.ne(y, data_loader.EOS)
            indice += [y]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=1)

        return y_hats, indice

    def batch_beam_search(
        self,
        src,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2
    ):
        mask, x_length = None, None

        if isinstance(src, tuple):
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        boards = [SingleBeamSearchBoard(
            h_src.device,
            {
                'hidden_state': {
                    'init_status': h_0_tgt[0][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                },
                'cell_state': {
                    'init_status': h_0_tgt[1][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                },
                'h_t_1_tilde': {
                    'init_status': None,
                    'batch_dim_index': 0,
                },
            },
            beam_size=beam_size,
            max_length=max_length,
        ) for i in range(batch_size)]

        is_done = [board.is_done() for board in boards]
        length = 0

        while sum(is_done) < batch_size and length <= max_length:
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
            fab_h_src, fab_mask = [], []
            
            for i, board in enumerate(boards):
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()
                    hidden_i    = prev_status['hidden_state']
                    cell_i      = prev_status['cell_state']
                    h_t_tilde_i = prev_status['h_t_1_tilde']

                    fab_input  += [y_hat_i]
                    fab_hidden += [hidden_i]
                    fab_cell   += [cell_i]
                    fab_h_src  += [h_src[i, :, :]] * beam_size
                    fab_mask   += [mask[i, :]] * beam_size
                    if h_t_tilde_i is not None:
                        fab_h_t_tilde += [h_t_tilde_i]
                    else:
                        fab_h_t_tilde = None

            fab_input  = torch.cat(fab_input,  dim=0)
            fab_hidden = torch.cat(fab_hidden, dim=1)
            fab_cell   = torch.cat(fab_cell,   dim=1)
            fab_h_src  = torch.stack(fab_h_src)
            fab_mask   = torch.stack(fab_mask)
            if fab_h_t_tilde is not None:
                fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim=0)

            emb_t = self.emb_dec(fab_input)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t,
                                                                      fab_h_t_tilde,
                                                                      (fab_hidden, fab_cell))
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output,
                                                             context_vector
                                                             ], dim=-1)))
            y_hat = self.generator(fab_h_t_tilde)

            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    begin = cnt * beam_size
                    end = begin + beam_size

                    board.collect_result(
                        y_hat[begin:end],
                        {
                            'hidden_state': fab_hidden[:, begin:end, :],
                            'cell_state'  : fab_cell[:, begin:end, :],
                            'h_t_1_tilde' : fab_h_t_tilde[begin:end],
                        },
                    )
                    cnt += 1

            is_done = [board.is_done() for board in boards]
            length += 1

        batch_sentences, batch_probs = [], []

        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs