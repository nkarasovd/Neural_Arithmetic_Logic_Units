import torch
import torch.optim as optim

from torch import Tensor
from torch.nn.functional import l1_loss, mse_loss

from typing import List, Union

from language_models.model import LanguageModel
from language_models.plot import PlotBuilder
from language_models.databuilder import DataBuilder


class TrainModel:
    def __init__(self, model: LanguageModel, optimizer: torch.optim.Optimizer,
                 databuilder: DataBuilder, data: List[List[Union[List[Tensor], Tensor]]], dir: str):
        self.model = model
        self.optimizer = optimizer
        self.data_builder = databuilder

        self.lr_drop = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, min_lr=1e-4)

        self._train, self._val, self._test = data

        self.cur_epoch = 0

        self.plot_builder = PlotBuilder(dir_save=dir)

    def train(self, batch_size: int, verbose: bool = False):
        loss_sum = 0.
        t, loss_mae, loss_mse = 0, 0., 0.
        N = len(self._train[0])

        for _ in range(batch_size):
            self._train = self.data_builder.shuffle_data(self._train)

            for x, y in zip(*self._train):
                y = y.unsqueeze(0)
                out = self.model(x)

                loss_mae = loss_mae + l1_loss(out, y)
                loss_mse = loss_mse + mse_loss(out, y)

                t += 1
                if t == batch_size:
                    loss = loss_mse / batch_size
                    if not torch.isnan(out):
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    loss_sum += float(loss_mae)
                    t, loss_mae, loss_mse = 0, 0., 0.

                if not (not torch.isnan(out) and out <= 1e+9):
                    N -= 1

        loss_sum /= N * batch_size

        self.cur_epoch += 1

        if verbose:
            print('Train MAE %04d %03.4f' % (self.cur_epoch, loss_sum),
                  self.data_builder.to_string(x), round(float(out), 4))

        self.plot_builder.add(x=self.cur_epoch, y=loss_sum, z=0)

    def __iterate_data(self, data):
        N = len(data[0])
        loss_sum = 0.

        for x, y in zip(*data):
            y = y.unsqueeze(0)

            out = self.model(x)

            loss = l1_loss(out, y)
            loss_sum += float(loss)

            if not (not torch.isnan(out) and out <= 1e+9):
                N -= 1

        return loss_sum / N, x, out

    def validate(self, verbose: bool = False):
        loss_sum, _, _ = self.__iterate_data(self._val)

        if verbose:
            print('Valid MAE %04d %03.4f' % (self.cur_epoch, loss_sum))

        # self.lr_drop.step(loss_sum, self.cur_epoch)

        self.plot_builder.add(x=self.cur_epoch, y=loss_sum, z=1)

    def test(self, verbose: bool = False) -> float:
        loss_sum, x, out = self.__iterate_data(self._test)

        if verbose:
            print('Eval  MAE %04d %03.4f' % (self.cur_epoch, loss_sum),
                  self.data_builder.to_string(x), round(float(out), 4))

        self.plot_builder.add(x=self.cur_epoch, y=loss_sum, z=2, step=True)

        return loss_sum
