import torch.optim as optim

from language_models.databuilder import DataBuilder
from language_models.model import LanguageModel
from language_models.train import TrainModel


def main_nalu(data_builder, data_train, data_val, data_eval):
    model = LanguageModel(len(data_builder), 32, 1, True, False)
    op = optim.Adam(model.parameters(), lr=1e-3)
    trainer = TrainModel(model, op, data_builder, [data_train, data_val, data_eval],
                         dir='../plots', title='LSTM + NALU')

    loss_min = 1.e+9

    for i in range(100):
        trainer.train(30, verbose=True)
        trainer.validate()
        loss_min = min(loss_min, trainer.test(verbose=True))

    print('min', loss_min)


def main_lstm(data_builder, data_train, data_val, data_eval):
    model = LanguageModel(len(data_builder), 32, 1, False, False)
    op = optim.Adam(model.parameters(), lr=1e-3)
    trainer = TrainModel(model, op, data_builder, [data_train, data_val, data_eval],
                         dir='../plots_lstm', title='LSTM + Linear')

    loss_min = 1.e+9
    for i in range(100):
        trainer.train(30, verbose=True)
        trainer.validate()
        loss_min = min(loss_min, trainer.test(verbose=True))

    print('min', loss_min)


if __name__ == '__main__':
    data_builder = DataBuilder()

    data_train, data_val, data_eval = data_builder.split_data(data_builder.generate_data())

    main_nalu(data_builder, data_train, data_val, data_eval)
    main_lstm(data_builder, data_train, data_val, data_eval)
