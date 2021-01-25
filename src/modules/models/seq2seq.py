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
            # Bi-direction인 Encoder와
            # Uni-direction인 Decoder와의 괴리를 맞추기 위함
            hidden_size=int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, emb):
        '''
        원문을 인코딩하는 영역
        - time-step을 신경쓸 필요 없음, 한꺼번에 넘겨서 한꺼번에 받음
        - PAD를 가진 시퀸스를 효율적으로 병렬연산 하기 위해, pack, unpack으로 처리
        '''
        if isinstance(emb, tuple):
            # x = (batch_size, length, word_vec_size)
            # lengths = (batch_size) - 각 문장마다의 길이 (PAD 제외)
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = emb

        # y = (batch_size, length, hidden_size)
        # h[0] = (num_layers * 2, batch_size, hidden_size / 2)
        y, h = self.rnn(x)
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
            # input feeding으로 이전 스텝의 결과값을 이어붙임
            input_size=word_vec_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        '''
        인코더에서 넘어온 원문의 hidden_state를 
        이용해 번역문으로 바꾸는 영역
        
        - time-step을 고려하기 위해 한번 호출마다 한 time-step만 수행
        - 따라서, emb_t, h_t_1_tilde는 모두 각 배치의 한개씩만 가져옴
        
        emb_t: 현재 time-step의 단어
        h_t_1_tilde: 이전 타임스텝의 예측 벡터값 (input feeding)
        h_t_1: 이전 타임스텝에서 넘어오면서 받은 Decoder 히든 사이즈
        (cell, hidden state로 나뉘며 신경쓰지말고 그대로 넘겨주면 됨)
        '''
        # emb_t = (batch_size, 1, word_vec_size)
        # h_t_1_tilde = (batch_size, 1, hidden_size)
        # h_t_1[0] = (n_layers, batch_size, hidden_size) - 히든 스테이트
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            # 첫 번째 time-step의 경우 0으로 임의값 할당
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        # Input Feeding
        # x = (batch_size, 1, word_vec_size + hidden_size)
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)
        # y = (batch_size, 1, hidden_size)
        # h[0] = (n_layers, batch_size, hidden_size)
        y, h = self.rnn(x, h_t_1)
        return y, h


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        '''
        Encoder 단의 원문 벡터값과 현재 값의 유사도 산출
        h_src: Encoder 전체 time-step의 히든 스테이트 출력값
        h_t_tgt: Decoder의 현재 time-step 히든 스테이트 출력값
        mask: 각 문장 각 토큰에 대한 PAD 여부 (True or False)
        '''
        # h_src = (batch_size, length, hidden_size)
        # h_t_tgt = (batch_size, 1, hidden_size)
        # mask = (batch_size, length)
        
        # query = (batch_size, 1, hidden_size)
        query = self.linear(h_t_tgt)
        # weight = (batch_size, 1, length)
        # length -> encoder 단 모든 타임스텝 결과에 대한 가중치를 뜻함
        weight = torch.bmm(query, h_src.transpose(1, 2))

        if mask is not None:
            # PAD token 자리의 가중치를 모두 -inf로 치환(학습 미반영)
            # 마스크를 씌우기 위해 mask가 해당 weight의 shape과 같아야함
            # mask.unsqueeze(1) = (batch_size, 1, length)
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))

        # weight = (batch_size, 1, length)
        weight = self.softmax(weight)
        # h_src = (batch_size, length, hidden_size)
        # context_vector = (batch_size, 1, hidden_size)
        context_vector = torch.bmm(weight, h_src)
        return context_vector


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1) # 마지막 차원으로 수행

    def forward(self, x):
        # x = (batch_size, length, hidden_size)

        # Generator의 경우, 모든 time-step의 값을 
        # Decoder에서 한번에 받아서 수행
        # y = (batch_size, length, output_size)
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
        # 원문 vocab size와 같음
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        # 번역문 vocab size와 같음
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
                            # 유효한 토큰 영역
                            x.new_ones(1, l).zero_(),
                            # PAD 토큰 영역
                            x.new_ones(1, (max_length - l)),
                        ],
                        dim=-1
                    )
                ]
            else:
                # 전부 유효한 패드 토큰의 경우
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()
        return mask

    def merge_encoder_hiddens(self, encoder_hiddens):
        '''
        Encoder에 나온 hidden값을 Decoder로 넘기기 위해
        해당 인풋 모양을 맞춰주어야 함
        Encoder는 Bi-direction이기 때문에 미리 hidden / 2로
        output을 뽑아서 양쪽 방향에서 나온걸 합쳐줘서 
        hidden / 2 * 2 처리를 해줌
        
        '''
        # h_0_tgt = (n_layers * 2, batch_size, hidden_size / 2)
        # c_0_tgt = (n_layers * 2, batch_size, hidden_size / 2)
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
        
        # h_0_tgt = (n_layers, batch_size, hidden_size)
        # c_0_tgt = (n_layers, batch_size, hidden_size)
        return h_0_tgt, c_0_tgt

    def forward(self, src, tgt):
        '''
        최종적인 Seq2Seq 신경망
        - 당연히 src, tgt의 길이는 다를 수 있음
        
        src = (batch_size, length_n) - 원문
        tgt = (batch_size, length_m) - 번역문
        output = (batch_size, length_m, V_tgt)
        V_tgt -> 아웃풋 vocab size
        '''
        
        mask = None
        x_length = None
        if isinstance(src, tuple):
            '''
            x = (batch_size, length_n)
            x_length = (batch_size) -> 각 문장의 실제 유효 길이 정보(PAD 제외)
            '''
            x, x_length = src
            mask = self.generate_mask(x, x_length)
        else:
            x = src

        if isinstance(tgt, tuple):
            tgt = tgt[0]
        
        #---------Encoder Step---------#
        # emb_src = (batch_size, length_n, word_vec_size)
        emb_src = self.emb_src(x)
        '''
        # h_src = (batch_size, length_n, hidden_size)
        # h_0_tgt[0] = (n_layers * 2, batch_size, hidden_size / 2)
        h_0_tgt[0]는 Bi-directional RNN에서 나왔기 때문에 n_layers가 2배로 곱해짐
        이걸 이용해서 hidden_size / 2 부분을 concat 시켜서 본래의 hidden_size로 복귀
        '''
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        #---------Decoder Step---------#
        # emb_tgt = (batch_size, length_m, word_vec_size)
        emb_tgt = self.emb_dec(tgt)
        # Decoder에서 나온 결과 hidden_state 값을 보관하는 리스트
        # 후에 Generator로 한꺼번에 전달하기 위함
        h_tilde = []

        # 이전 타입스텝의 h_tilde 값(Input Feeding을 위함), 처음에는 None
        h_t_tilde = None
        # 이전 타임 스텝의 히든 값, 처음에는 인코더의 최종 hidden값
        decoder_hidden = h_0_tgt
        for t in range(tgt.size(1)):
            # 현재 타입스텝에 해당하는 emb 값만 가져옴
            # emb_t = (batch_size, 1, word_vec_size)
            emb_t = emb_tgt[:, t, :].unsqueeze(1)

            # h_t_tilde = (batch_size, 1, hidden_size)
            # decoder_output = (batch_size, 1, hidden_size)
            # decoder_hidden = (n_layers, batch_size, hidden_size)
            decoder_output, decoder_hidden = self.decoder(
                emb_t,
                h_t_tilde,
                decoder_hidden
            )
            # context_vector = (batch_size, 1, hidden_size)
            context_vector = self.attn(h_src, decoder_output, mask)

            # 현재 타입스텝의 h_tilde 값은 decoder_output + context_vector
            # 단, concat layer를 통해 본래의 hidden_size로 압축시킴
            # h_t_tilde = (batch_size, 1, hidden_size)
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

        # h_tilde = (batch_size, length_m, hidden_size)
        h_tilde = torch.cat(h_tilde, dim=1)
        # y_hat = (batch_size, length_m, output_size)
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

        # 모든 문장(batch)에 대하여 첫 토큰으로 BOS로 고정
        y = x.new(batch_size, 1).zero_() + data_loader.BOS

        '''
        id_decoding = (batch_size, 1) => 전부 True로 채워짐
        각 문장들의 decoding 완료 여부 저장
        '''
        is_decoding = x.new_ones(batch_size, 1).bool()
        h_t_tilde, y_hats, indice = None, [], []

        '''
        is_decoding.sum() > 0: 아직 디코딩이 끝나지 않은 문장이 있거나
        len(indice) < max_length: time-step이 max_length보다 적을 경우 Loop
        '''
        while is_decoding.sum() > 0 and len(indice) < max_length:
            # emb_t = (batch_size, 1, word_vec_size)
            emb_t = self.emb_dec(y)
            # decoder_output = (batch_size, 1, hidden_size)
            decoder_output, decoder_hidden = self.decoder(
                emb_t,
                h_t_tilde,
                decoder_hidden
            )
            # context_vector = (batch_size, 1, hidden_size)
            context_vector = self.attn(h_src, decoder_output, mask)
            # h_t_tilde = (batch_size, 1, hidden_size)
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
            # y_hat = (batch_size, 1, output_size)
            y_hat = self.generator(h_t_tilde)
            y_hats += [y_hat]

            # y = (batch_size, 1)
            if is_greedy:
                y = y_hat.argmax(dim=-1)
            else:
                # Random Sampling 기반 접근
                y = torch.multinomial(y_hat.exp().view(batch_size, -1), 1)

            # is_decoding 정보를 반전시켜, 디코딩이 끝난 부분이 True가 됨
            # True가 된 모든 예측값에 대한 무효처리 (PAD로 덮어씀)
            y = y.masked_fill_(~is_decoding, data_loader.PAD)
            # 각 문장 디코딩에 대한 완료 여부 갱신
            # EOS와 같다면 False가 뜨면서 is_decoding이 False로 바뀜
            is_decoding = is_decoding * torch.ne(y, data_loader.EOS)
            indice += [y]

        # y_hat = (batch_size, length, output_size)
        # indice = (batch_size, length)
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
            # |mask| = (batch_size, length)
        else:
            x = src
        batch_size = x.size(0)

        emb_src = self.emb_src(x)
        h_src, h_0_tgt = self.encoder((emb_src, x_length))
        # |h_src| = (batch_size, length, hidden_size)
        h_0_tgt = self.merge_encoder_hiddens(h_0_tgt)

        '''
        initialize 'SingleBeamSearchBoard'
        각 배치별로, beam_size만큼 페이크 배치를 생성해주는 클래스 초기화
        hidden_state: 인코더에서 넘어온 히든 스테이트
        cell_state: 인코더에서 넘어온 셀 스테이트
        h_t_1_tilde: 이전 스텝의 예측값(input feeding), 
        처음에는 없으므로 None
        '''
        boards = [SingleBeamSearchBoard(
            h_src.device,
            {
                'hidden_state': {
                    'init_status': h_0_tgt[0][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                }, # |hidden_state| = (n_layers, batch_size, hidden_size)
                'cell_state': {
                    'init_status': h_0_tgt[1][:, i, :].unsqueeze(1),
                    'batch_dim_index': 1,
                }, # |cell_state| = (n_layers, batch_size, hidden_size)
                'h_t_1_tilde': {
                    'init_status': None,
                    'batch_dim_index': 0,
                }, # |h_t_1_tilde| = (batch_size, 1, hidden_size)
            },
            beam_size=beam_size,
            max_length=max_length,
        ) for i in range(batch_size)]
        # 각 보드(batch)들이 예측이 끝났는지 여부
        # 처음에는 당연히 전부 0으로 이루어짐
        is_done = [board.is_done() for board in boards]

        length = 0
        # is_done의 합이 Batch_size를 넘을때까지 반복
        while sum(is_done) < batch_size and length <= max_length:
            # current_batch_size = sum(is_done) * beam_size

            # Initialize fabricated variables.
            # As far as batch-beam-search is running, 
            # temporary batch-size for fabricated mini-batch is 
            # 'beam_size'-times bigger than original batch_size.
            fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
            fab_h_src, fab_mask = [], []
            
            # 각 input들을 beam_size 만큼 늘려서 가짜 batch_size 생성
            # input, hidden, cell, h_t_tilde는 이미 보드에서 늘려진 상태
            # h_src, mask만 그대로 expand 해주면 됨
            for i, board in enumerate(boards):
                # Batchify if the inference for the sample is still not finished.
                if board.is_done() == 0:
                    # 여기서 현재 타임스텝에 필요한 가짜 batch 데이터 반환
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
            # |fab_input|     = (current_batch_size, 1)
            # |fab_hidden|    = (n_layers, current_batch_size, hidden_size)
            # |fab_cell|      = (n_layers, current_batch_size, hidden_size)
            # |fab_h_src|     = (current_batch_size, length, hidden_size)
            # |fab_mask|      = (current_batch_size, length)
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)

            emb_t = self.emb_dec(fab_input)
            # |emb_t| = (current_batch_size, 1, word_vec_size)

            fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t,
                                                                      fab_h_t_tilde,
                                                                      (fab_hidden, fab_cell))
            # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
            context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
            # |context_vector| = (current_batch_size, 1, hidden_size)
            fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output,
                                                             context_vector
                                                             ], dim=-1)))
            # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
            y_hat = self.generator(fab_h_t_tilde)
            # |y_hat| = (current_batch_size, 1, output_size)

            # 디코더에서는 그대로 한 batch인듯이 병렬연산을 해준뒤,
            # 각 board에 다시 beam_size만큼 찢어서 보내줌
            # fab_hidden[:, begin:end, :] = (n_layers, beam_size, hidden_size)
            # fab_cell[:, begin:end, :]   = (n_layers, beam_size, hidden_size)
            # fab_h_t_tilde[begin:end]    = (beam_size, 1, hidden_size)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    # Decide a range of each sample.
                    begin = cnt * beam_size
                    end = begin + beam_size

                    # pick k-best results for each sample.
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

        # pick n-best hypothesis.
        batch_sentences, batch_probs = [], []

        # Collect the results.
        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs