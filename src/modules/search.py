from operator import itemgetter
import torch
import modules.data_loader as data_loader

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():

    def __init__(
        self,
        device,
        prev_status_config,
        beam_size=5,
        max_length=255,
    ):
        self.beam_size = beam_size
        self.max_length = max_length

        self.device = device
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]

        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, each_config in prev_status_config.items():
            init_status = each_config['init_status']
            batch_dim_index = each_config['batch_dim_index']
            if init_status is not None:
                self.prev_status[prev_status_name] = torch.cat([init_status] * beam_size,
                                                               dim=batch_dim_index)
            else:
                self.prev_status[prev_status_name] = None
            self.batch_dims[prev_status_name] = batch_dim_index

        self.current_time_step = 0
        self.done_cnt = 0

    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        p = ((min_length + 1) / (min_length + length))**alpha

        return p

    def is_done(self):
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        return y_hat, self.prev_status

    def collect_result(self, y_hat, prev_status):
        output_size = y_hat.size(-1)

        self.current_time_step += 1
        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)

        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]

        self.word_indice += [top_indice.fmod(output_size)]
        self.beam_indice += [top_indice.div(float(output_size)).long()]

        self.cumulative_probs += [top_log_prob]
        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)]
        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum()

        for prev_status_name, prev_status in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                prev_status,
                dim=self.batch_dims[prev_status_name],
                index=self.beam_indice[-1]
            ).contiguous()

    def get_n_best(self, n=1, length_penalty=.2):
        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):
            for b in range(self.beam_size):
                if self.masks[t][b] == 1:
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)]

        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):
                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]

        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        return sentences, probs
