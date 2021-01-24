import numpy as np
import torch
import torch.nn.utils as torch_utils
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from modules.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class IgniteEngine(Engine):

    def __init__(self, func, model, crit, optimizer, lr_scheduler, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.config = config
        super().__init__(func)

        self.best_loss = np.inf
        #self.scaler = GradScaler()

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()

        '''
        Gradient Accumulation
        - 기계 번역의 경우, batch_size가 256 정도가 적당
        즉 batch_size 크기 자체도 성능에 영향을 끼침 
        하지만 GPU 성능에 따라 하지 못할 수도 있음

        속도는 몰라도, 성능을 보존시켜 주기 위해 일부러 
        N번 정도의 iteration을 건너뛰어서 원하는 성능을 유지시킴

        1. engine.state.iteration % engine.config.iteration_per_update == 1
        - 현재 iter가 per_update로 나눠서 나머지가 1일때마다 zero_grad 수행
        2. engine.config.iteration_per_update == 1
        - 그냥 통상적인 경우 매번 zero_grad 시키기

        '''
        if engine.state.iteration % engine.config.iteration_per_update == 1 or \
           engine.config.iteration_per_update == 1:
            if engine.state.iteration > 1:
                engine.optimizer.zero_grad()

        # 모델의 첫번째 파라미터가 config임
        device = next(engine.model.parameters()).device
        '''
        src와 tgt는 각각 (실제 문장 데이터, 각 문장의 길이 정보) tuple 형태
        - torchText에서 애초에 저렇게 제공됨
        그 중에서 실제 문장 데이터만 GPU 메모리로 전송
        '''
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

        '''
        맨 처음 Input으로 x가 들어감
        최종 Output과의 검증을 위해 y가 들어감
        x의 경우, 그냥 그대로 넣어주면 됨(BOS EOS 들어가도 노상관)
        y의 경우,
        - 각 문장의 길이 정보는 버림
        - 또한 실제 문장에서도 맨처음 BOS 토큰을 제거
        (왜냐하면 예측은 BOS 다음 단어부터 수행하기 때문)

        x = (batch_size, length_n)
        y = (batch_size, length_m)
        '''
        x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]

        #-------------------------#
        # autocast로 공간효율적으로 학습 실행
        # with autocast(not engine.config.off_autocast):
        # y_hat = (batch_size, length_m, output_size)
        # 입력 tgt의 경우, 맨뒤에 EOS를 토큰을 제거
        y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])

        '''
        loss값 연산을 위해 다음과 같이 텐서 모양 정리
        모든 문장의 각 단어를 순서대로 배치했다고 보면됨
        변경 전(3D):
            y_hat = (batch_size, length_m, output_size)
            y = (batch_size, length_m)
        변경 후(2D):
            y_hat = (batch_size * length_m, output_size)
            y = (batch_size * length_m)
        '''
        loss = engine.crit(
            y_hat.contiguous().view(-1, y_hat.size(-1)),
            y.contiguous().view(-1)
        )
        '''
        div(y.size(0)): loss를 구한후, batch_size만큼 나눠준 후
        div(engine.config.iteration_per_update): 
        Gradient Accumulation을 위해 미리 나눠줌
        즉, backward_target이 진짜 적용시킬 loss 값이라 보면 됨
        '''
        backward_target = loss.div(y.size(0)).div(engine.config.iteration_per_update)
        #-------------------------#

        # autocast가 켜져 있는 경우, scale 작업 후에, backward
        # if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
        #     engine.scaler.scale(backward_target).backward()
        # else:
        backward_target.backward()

        # 현재 batch 내에 모든 토큰 수
        word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        # Gradient Accumulation 여부, 맞아 떨어진다면 step까지 수행, 아니면 스킵
        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
           engine.state.iteration > 0:
            '''
            Gradient Clipping
            시퀸스의 time_step이 길수록, gradient가 매우 커질수도 있음
            g_norm이 너무 커서 많이 움직이는 걸 막기 위해 사용
            - 단, Adam을 쓰면 큰 필요는 없다고 함 ㅇㅇ
            '''
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad_norm,
            )

            # if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
            #     # GPU를 사용할 경우, 기존 optim.step() 대신에 scaler로 step 수행
            #     engine.scaler.step(engine.optimizer)
            #     engine.scaler.update()
            # else:
            engine.optimizer.step()

        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            device = next(engine.model.parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1])
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1])

            # x = (batch_size, length_n)
            # y = (batch_size, length_m)
            x, y = mini_batch.src, mini_batch.tgt[0][:, 1:]

            #with autocast(not engine.config.off_autocast):
            y_hat = engine.model(x, mini_batch.tgt[0][:, :-1])
            loss = engine.crit(
                y_hat.contiguous().view(-1, y_hat.size(-1)),
                y.contiguous().view(-1),
            )

        word_count = int(mini_batch.tgt[1].sum())
        loss = float(loss / word_count)
        ppl = np.exp(loss)

        return {
            'loss': loss,
            'ppl': ppl,
        }

    @staticmethod
    def attach(
        train_engine, validation_engine,
        training_metric_names=['loss', 'ppl', '|param|', '|g_param|'],
        validation_metric_names=['loss', 'ppl'],
        verbose=VERBOSE_BATCH_WISE,
    ):
        # 현재 상황 보고 및 출력 함수
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        '''Train Attach Process'''
        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_loss = engine.state.metrics['loss']

                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} ppl={:.2f}'.format(
                    engine.state.epoch,
                    avg_p_norm,
                    avg_g_norm,
                    avg_loss,
                    np.exp(avg_loss),
                ))

        '''Validation Attach Process'''
        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_loss = engine.state.metrics['loss']

                print('Validation - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_loss,
                    np.exp(avg_loss),
                    engine.best_loss,
                    np.exp(engine.best_loss),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss

    @staticmethod
    def save_model(engine, train_engine, config, src_vocab, tgt_vocab):
        avg_train_loss = train_engine.state.metrics['loss']
        avg_valid_loss = engine.state.metrics['loss']

        # 주의!, best_model이 아닌 모든 에포크의 모델 저장 
        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')
        
        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_loss,
                                                   np.exp(avg_train_loss)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_loss,
                                                   np.exp(avg_valid_loss)
                                                   )
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        # Unlike other tasks, we need to save current model, not best model.
        torch.save(
            {
                'model': engine.model.state_dict(),
                'opt': train_engine.optimizer.state_dict(),
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }, model_fn
        )


class Trainer():

    def __init__(self, target_engine_class, config):
        self.target_engine_class = target_engine_class
        self.config = config

    def train(
        self,
        model, crit, optimizer,
        train_loader, valid_loader,
        src_vocab, tgt_vocab,
        n_epochs,
        lr_scheduler=None
    ):
        train_engine = self.target_engine_class(
            self.target_engine_class.train,
            model,
            crit,
            optimizer,
            lr_scheduler,
            self.config
        )
        validation_engine = self.target_engine_class(
            self.target_engine_class.validate,
            model,
            crit,
            optimizer=None,
            lr_scheduler=None,
            config=self.config
        )

        self.target_engine_class.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)
            if engine.lr_scheduler is not None:
                engine.lr_scheduler.step()

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )
        train_engine.add_event_handler(
            Events.STARTED,
            self.target_engine_class.resume_training,
            self.config.init_epoch,
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, self.target_engine_class.check_best
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            self.target_engine_class.save_model,
            train_engine,
            self.config,
            src_vocab,
            tgt_vocab,
        )

        # Start training
        train_engine.run(train_loader, max_epochs=n_epochs)

        return model