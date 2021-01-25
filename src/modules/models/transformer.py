import torch
import torch.nn as nn

import modules.data_loader as data_loader
from modules.search import SingleBeamSearchBoard


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        '''
        각 인코더, 디코더 블럭에서의 어텐션 수행
        * 사실상 학습하지는 않음

        m -> tgt의 legnth
        n -> src의 length
        - 셀프 어텐션시, Q==K==V (이때, 모두 n임)
        - decoder의 어텐션시, tgt in Q, src in K, V

        # |Q| = (batch_size, m, hidden_size)
        # |K| = |V| = (batch_size, n, hidden_size)
        # |mask| = (batch_size, m, n)
        각 문장 각 토큰에 대한 PAD 여부
        '''
        w = torch.bmm(Q, K.transpose(1, 2))
        # |w| = (batch_size, m, n)
        # w: 각 문장(batch)의 각 tgt 타임스텝(m)에 대한 src 토큰들의 가중치
        # 단, 이때 src 토큰들 중, PAD는 학습에 미반영시킬 필요가 있음(mask)
        if mask is not None:
            # PAD token 자리의 가중치를 모두 -inf로 치환(학습 미반영)
            # 마스크를 씌우기 위해 mask가 해당 weight의 shape과 같아야함
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))

        # 학습 안정성을 위해 weight(w)의 스케일링 수행
        w = self.softmax(w / (dk**.5))
        c = torch.bmm(w, V)
        # |c| = (batch_size, m, hidden_size)
        # 만약 셀프 어텐션이였다면 n이였을 것임

        return c


class MultiHead(nn.Module):

    def __init__(self, hidden_size, n_splits):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # 각 Q, K, V의 linear Transform Layer
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        '''
        m -> tgt의 legnth
        n -> src의 length
        - 셀프 어텐션시, Q==K==V (이때, 모두 n임)
        - decoder의 어텐션시, tgt in Q, src in K, V

        # |Q| = (batch_size, m, hidden_size)
        # |K| = (batch_size, n, hidden_size)
        # |V| = (batch_size, n, hidden_size)
        
        # |mask| = (batch_size, m, n)
        각 문장 각 토큰에 대한 PAD 여부
        후에 어텐션시, weight가 생성되는데, 아래와 같이 설정됨
        - 각 문장(batch)의 각 tgt 타임스텝(m)에 대한 src 토큰들의 가중치
        단, 이때 src 토큰들 중, PAD는 학습에 미반영시킬 필요가 있음(mask)

        기존 seq2seq의 mask의 경우 (batch_size, n or m)이였지만,
        멀티 헤드 어텐션은 모든 단어가 타겟 문장의 모든 단어와 어텐션을 수행하는데,
        병렬 연산을 위해서 mask를 재활용하지 않고 늘려줌 
        '''
        
        '''
        멀티헤드 어텐션을 위한 분할 작업 준비
        - 각 linear transform layer에서 나온 벡터를, 헤드의 수만큼 분할시킴
        - 그 후 분할된 헤드수만큼을 batch_size를 중심을 concat 시킴
        - 이로써, 각 Q,K,V의 헤드들을 batch로 취급하여 병렬 처리
        '''
        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |QW_i| = (batch_size, m, hidden_size / n_splits) * n_splits
        # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits) * n_splits
        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)

        if mask is not None:
            # mask도 분할된 헤드수만큼 늘려서 추가해줌
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size * n_splits, m, n)

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)

        
        '''
        - 병렬로 처리했던 각 헤드들을 batch에서 떼어냄
        - 각 batch의 헤드들을 다시 concat시켜준 후, 최종 linear로 보내줌
        '''
        c = c.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, m, hidden_size / n_splits) * n_splits
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c


class EncoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        '''
        # |x|    = (batch_size, n, hidden_size)
        입력받은 모든 문장(batch), 모든 단어(n)들의 임베딩(hidden_size)
        # |mask| = (batch_size, n, n) -> 무조건 셀프 어텐션이므로 같음
        각 문장들의 각 단어의 PAD 여부 * n(dim=1)으로 늘려준 것(병렬 연산을 위함)

        Pre LN은 기본적으로 순서는 이렇게됨
        x -> (norm) -> (attn) -> (dropout) -> ((+x) residual_connection) = z
        z -> (norm) -> (fc) -> (dropout) -> ((+z) residual_connection) = y
        '''

        '''
        # Post-LN:
        z = self.attn_norm(
            x + self.attn_dropout(
                self.attn(
                    Q=x, K=x, V=x, mask=mask
                )
            )
        )
        z = self.fc_norm(
            z + self.fc_dropout(
                self.fc(z)
            )
        ) 
        '''

        z = self.attn_norm(x)
        z = x + self.attn_dropout(
                    self.attn(
                        Q=z, K=z,V=z, mask=mask
                    )
                )
        z = z + self.fc_dropout(
                    self.fc(
                        self.fc_norm(z)
                    )
                )
        # |z| = (batch_size, n, hidden_size)

        return z, mask


class DecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        self.masked_attn = MultiHead(hidden_size, n_splits)
        self.masked_attn_norm = nn.LayerNorm(hidden_size)
        self.masked_attn_dropout = nn.Dropout(dropout_p)

        self.attn = MultiHead(hidden_size, n_splits)
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.fc_norm = nn.LayerNorm(hidden_size)
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_and_value, mask, prev, future_mask):
        # |key_and_value| = (batch_size, n, hidden_size) = K, V
        # |mask|          = (batch_size, m, n)

        # 학습과 추론시에 방법이 다르기 때문에 분기를 해줄 필요가 있음.
        if prev is None: # Training mode
            # |x|           = (batch_size, m, hidden_size) = Q
            # |prev|        = None
            # |future_mask| = (batch_size, m, m)
            # 셀프 어텐션이므로 같음, 학습 중 미래의 토큰에 대한 mask 처리
            # |z|           = (batch_size, m, hidden_size)

            # Post-LN:
            # z = self.masked_attn_norm(x + self.masked_attn_dropout(
            #     self.masked_attn(x, x, x, mask=future_mask)
            # ))

            # Pre-LN:
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(
                self.masked_attn(z, z, z, mask=future_mask)
            )
        else: # Inference mode
            # |x|           = (batch_size, 1, hidden_size) = Q 
            # (현재 타입스템만 가져옴)
            # |prev|        = (batch_size, t - 1, hidden_size) 
            # (처음부터 바로 이전 타임스텝까지 싹다 모은 것)
            # |future_mask| = None
            # |z|           = (batch_size, 1, hidden_size) 
            # (결과값 또한 한 타임스텝만)

            # Post-LN:
            # z = self.masked_attn_norm(x + self.masked_attn_dropout(
            #     self.masked_attn(x, prev, prev, mask=None)
            # ))

            # Pre-LN:
            normed_prev = self.masked_attn_norm(prev)
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(
                self.masked_attn(z, normed_prev, normed_prev, mask=None)
            )

        # Post-LN:
        # z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z,
        #                                                    K=key_and_value,
        #                                                    V=key_and_value,
        #                                                    mask=mask)))

        # Pre-LN:
        normed_key_and_value = self.attn_norm(key_and_value)
        z = z + self.attn_dropout(self.attn(Q=self.attn_norm(z),
                                            K=normed_key_and_value,
                                            V=normed_key_and_value,
                                            mask=mask))
        # |z| = (batch_size, m, hidden_size)

        # Post-LN:
        # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))

        # Pre-LN:
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (batch_size, m, hidden_size)

        return z, key_and_value, mask, prev, future_mask


class MySequential(nn.Sequential):

    def forward(self, *x):
        # nn.Sequential 은 복수의 input을 전달할 수 없기 때문에
        # 오버라이딩 하여, 복수의 input을 처리할 수 있도록 변경
        for module in self._modules.values():
            x = module(*x)
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_splits,
        n_enc_blocks=6,
        n_dec_blocks=6,
        dropout_p=.1,
        use_leaky_relu=False,
        max_length=512,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_splits = n_splits
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_p = dropout_p
        self.max_length = max_length

        super().__init__()

        self.emb_enc = nn.Embedding(input_size, hidden_size)
        self.emb_dec = nn.Embedding(output_size, hidden_size)
        self.emb_dropout = nn.Dropout(dropout_p)

        self.pos_enc = self._generate_pos_enc(hidden_size, max_length)

        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_enc_blocks)],
        )
        self.decoder = MySequential(
            *[DecoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_dec_blocks)],
        )
        self.generator = nn.Sequential(
            # Only for Pre-LN Transformer.
            nn.LayerNorm(hidden_size), 
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )

    @torch.no_grad()
    def _generate_pos_enc(self, hidden_size, max_length):
        '''
        위치 정보에 상관없이 모든 타임스텝을 병렬적으로 학습하기 때문에,
        디코더에 들어가기전, 각 문장의 위치 정보를 대입시키기 위함
        '''
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # |enc| = (max_length, hidden_size)

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # |pos| = (max_length, 1)
        # |dim| = (1, hidden_size // 2)

        # 이 부분을 잘 모르겠다???
        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))

        return enc

    def _position_encoding(self, x, init_pos=0):
        '''
        사전에 만들어진 pos_enc를 원하는 만큼 자르는 과정
        학습 시에는, init_pos가 0으로 고정이지만,
        추론 시에는, 하나씩 가져오며 수행
        '''
        # |x| = (batch_size, n, hidden_size)
        # |self.pos_enc| = (max_length, hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
        # |pos_enc| = (1, n, hidden_size)
        x = x + pos_enc.to(x.device)

        return x

    @torch.no_grad()
    def _generate_mask(self, x, length):
        '''
        <PAD> 학습 미반영 처리를 위한 마스크 생성
        '''
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                mask += [torch.cat([x.new_ones(1, l).zero_(), # 마스크 X
                                    x.new_ones(1, (max_length - l)) # 마스크 O
                                    ], dim=-1)]
            else:
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()
        # |mask| = (batch_size, max_length)

        return mask

    def forward(self, x, y):
        # |x[0]| = (batch_size, n)
        # |y|    = (batch_size, m)

        '''
        인코더, 디코더에 대한 <PAD> 마스크 생성
        어텐션시, 인코더 단의 결과 값에 <PAD> 부분을 무효화시키기 위함
        mask_enc: 셀프 어텐션이므로 n * n
        mask_dec: 디코더의 길이 만큼 늘려주기 위해 m * n
        '''
        with torch.no_grad():
            mask = self._generate_mask(x[0], x[1])
            # |mask| = (batch_size, n)
            x = x[0]

            mask_enc = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
            mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
            # |mask_enc| = (batch_size, n, n)
            # |mask_dec| = (batch_size, m, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        '''
        디코더 한정으로 학습 중, 미래의 타임스텝을 가리기 위한 퓨처 마스크
        디코더 쪽의 셀프 어텐션이므로 크기는 m * m
        '''
        with torch.no_grad():
            '''
            # 역삼각형으로 마스킹된 텐서 생성
            torch.triu: 역삼각 부분만 값을 살리고 나머지는 0으로 초기화
            [0 1 1 1]
            [0 0 1 1]
            [0 0 0 1]
            [0 0 0 0]
            '''
            future_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()
            # |future_mask| = (m, m)
            future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
            # |fwd_mask| = (batch_size, m, m)

        h = self.emb_dropout(self._position_encoding(self.emb_dec(y)))
        h, _, _, _, _ = self.decoder(h, z, mask_dec, None, future_mask)
        # |h| = (batch_size, m, hidden_size)

        y_hat = self.generator(h)
        # |y_hat| = (batch_size, m, output_size)

        return y_hat

    def search(self, x, is_greedy=True, max_length=255):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)
        # 학습때는 모든 타임스텝을 한번에 보내서 (bs, m, n)이지만
        # 추론시에는 다음 단어를 미리 모르므로, (bs, 1, n)
        # 그냥 이 디코더 마스크를 계속 재활용해서 사용함

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        # 각 배치마다 현재 타입스텝 예측 단어를 저장하는 공간 
        # 처음에는 (batch_size, 1)로 전부 BOS로 채워져있음
        y_t_1 = x.new(batch_size, 1).zero_() + data_loader.BOS
        # |y_t_1| = (batch_size, 1)
         
        # 현재 각 배치가 디코딩이 진행중인 것인지 True or False
        # 합쳐서 1이상이면 최소 한개라도 진행중이므로 속행하게 됨
        is_decoding = x.new_ones(batch_size, 1).bool()

        '''
        # seq2seq는 이전 맨 마지막 hidden state만 갖고 있었으면 되었는데
        # transformer는 어텐션을 때리기 위해,
        # 각 계층마다 모든 타임스텝의 hidden_state를 참고해야 함
        # 따라서 처음에 decorder의 계층 수 + 1 (입력단 포함)의 None으로 가득찬 리스트 생성
        # prevs: 각 계층마다 hidden 값을 저장하기 위한 공간
        prev:
            [
               1계층:  (batch_size, 1 -> N, hidden_size),
               2계층: (batch_size, 1 -> N, hidden_size),
               3계층: (batch_size, 1 -> N, hidden_size),
               ...
            ]
        첫 번째 타임스텝의 (1, 2, 3) 계층을 돌면서 prev를 박고...
        두 번째 타임스텝의 (1, 2, 3) 계층을 돌면서 prev를 박고...
        '''
        prevs = [None for _ in range(len(self.decoder._modules) + 1)]
        y_hats, indice = [], []

        while is_decoding.sum() > 0 and len(indice) < max_length:
            # 각 배치의 한 타임스텝에 대한 히든 사이즈
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(y_t_1), init_pos=len(indice))
            )
            # |h_t| = (batch_size, 1, hidden_size))

            # 맨 처음은 예외처리로써, 첫 계층 prev([0]) 정보들은
            # 디코더에 들어가기 전에 처리후, 저장해줌
            # 왜냐하면 디코더를 들어가야 하기 때문
            if prevs[0] is None:
                # None일 경우(초기상태), 그대로 바꿔주고
                prevs[0] = h_t
            else:
                # 유효한 텐서값일 경우, 해당 텐서 밑에 그대로 붙여줌
                prevs[0] = torch.cat([prevs[0], h_t], dim=1)

            # Decoder Block를 하나하나 뽑아서 반복문 수행
            for layer_index, block in enumerate(self.decoder._modules.values()):
                # 현재 계층의 prev_status만 가져옴
                prev = prevs[layer_index]
                # |prev| = (batch_size, len(y_hats), hidden_size)

                # 맨 처음 들어갈때는, 이전 타임스텝이 없음
                # 그래서 prevs[0]도 h_t가 들어 있음(h_t == prev)
                # 즉 규격에 맞춰주기 위해서 prev를 넣어준 셈(결국 완전 셀프 어텐션)
                h_t, _, _, _, _ = block(h_t, z, mask_dec, prev, None)
                # |h_t| = (batch_size, 1, hidden_size)

                # 이번에 나온 결과(h_t)를 다음 계층의 Prev로써 쓸 수 있도록 가져옴
                if prevs[layer_index + 1] is None:
                    prevs[layer_index + 1] = h_t
                else:
                    prevs[layer_index + 1] = torch.cat([prevs[layer_index + 1], h_t], dim=1)
                # |prev| = (batch_size, len(y_hats) + 1, hidden_size)

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, output_size)

            y_hats += [y_hat_t]
            if is_greedy: # Argmax
                y_t_1 = torch.topk(y_hat_t, 1, dim=-1)[1].squeeze(-1)
            else: # Random sampling                
                y_t_1 = torch.multinomial(y_hat_t.exp().view(x.size(0), -1), 1)
            # Put PAD if the sample is done.
            # is_decoding이 False인 곳, 즉 EOS로 문장이 끝난 곳은
            # y_t_1에 <PAD>를 덮어씌워줌
            y_t_1 = y_t_1.masked_fill_(
                ~is_decoding,
                data_loader.PAD,
            )

            # Update is_decoding flag.
            is_decoding = is_decoding * torch.ne(y_t_1, data_loader.EOS)
            # |y_t_1| = (batch_size, 1)
            # |is_decoding| = (batch_size, 1)
            indice += [y_t_1]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hats| = (batch_size, m, output_size)
        # 각 단어의 확률 값
        # |indice| = (batch_size, m)
        # 확률값을 토대로 구한 최종 단어 인덱스 모음
        return y_hats, indice

    #@profile
    def batch_beam_search(
        self,
        x,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2,
    ):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)
        n_dec_layers = len(self.decoder._modules)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        prev_status_config = {}
        for layer_index in range(n_dec_layers + 1):
            prev_status_config['prev_state_%d' % layer_index] = {
                'init_status': None,
                'batch_dim_index': 0,
            }
        # Example of prev_status_config:
        # prev_status_config = {
        #     'prev_state_0': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     },
        #     'prev_state_1': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     },
        #
        #     ...
        #
        #     'prev_state_${n_layers}': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     }
        # }

        boards = [
            SingleBeamSearchBoard(
                z.device,
                prev_status_config,
                beam_size=beam_size,
                max_length=max_length,
            ) for _ in range(batch_size)
        ]
        done_cnt = [board.is_done() for board in boards]

        length = 0
        while sum(done_cnt) < batch_size and length <= max_length:
            fab_input, fab_z, fab_mask = [], [], []
            fab_prevs = [[] for _ in range(n_dec_layers + 1)]

            for i, board in enumerate(boards): # i == sample_index in minibatch
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()

                    fab_input += [y_hat_i                 ]
                    fab_z     += [z[i].unsqueeze(0)       ] * beam_size
                    fab_mask  += [mask_dec[i].unsqueeze(0)] * beam_size

                    for layer_index in range(n_dec_layers + 1):
                        prev_i = prev_status['prev_state_%d' % layer_index]
                        if prev_i is not None:
                            fab_prevs[layer_index] += [prev_i]
                        else:
                            fab_prevs[layer_index] = None

            fab_input = torch.cat(fab_input, dim=0)
            fab_z     = torch.cat(fab_z,     dim=0)
            fab_mask  = torch.cat(fab_mask,  dim=0)
            for i, fab_prev in enumerate(fab_prevs): # i == layer_index
                if fab_prev is not None:
                    fab_prevs[i] = torch.cat(fab_prev, dim=0)
            # |fab_input|    = (current_batch_size, 1,)
            # |fab_z|        = (current_batch_size, n, hidden_size)
            # |fab_mask|     = (current_batch_size, 1, n)
            # |fab_prevs[i]| = (cur rent_batch_size, length, hidden_size)
            # len(fab_prevs) = n_dec_layers + 1

            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(fab_input), init_pos=length)
            )
            # |h_t| = (current_batch_size, 1, hidden_size)
            if fab_prevs[0] is None:
                fab_prevs[0] = h_t
            else:
                fab_prevs[0] = torch.cat([fab_prevs[0], h_t], dim=1)

            for layer_index, block in enumerate(self.decoder._modules.values()):
                prev = fab_prevs[layer_index]
                # |prev| = (current_batch_size, m, hidden_size)

                h_t, _, _, _, _ = block(h_t, fab_z, fab_mask, prev, None)
                # |h_t| = (current_batch_size, 1, hidden_size)

                if fab_prevs[layer_index + 1] is None:
                    fab_prevs[layer_index + 1] = h_t
                else:
                    fab_prevs[layer_index + 1] = torch.cat(
                        [fab_prevs[layer_index + 1], h_t],
                        dim=1,
                    ) # Append new hidden state for each layer.

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, output_size)

            # |fab_prevs[i][begin:end]| = (beam_size, length, hidden_size)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    begin = cnt * beam_size
                    end = begin + beam_size

                    prev_status = {}
                    for layer_index in range(n_dec_layers + 1):
                        prev_status['prev_state_%d' % layer_index] = fab_prevs[layer_index][begin:end]

                    board.collect_result(y_hat_t[begin:end], prev_status)

                    cnt += 1

            done_cnt = [board.is_done() for board in boards]
            length += 1

        batch_sentences, batch_probs = [], []

        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs
