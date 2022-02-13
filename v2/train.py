import argparse
import os

import torch
from pytorch_lightning import Trainer

from torch.nn import functional as F

import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.metrics import classification_report
from tqdm import tqdm

import sys

from datamodules.ts import stsDataModule
from models.architectures.DTW.CNN import CNN_DTW
from models.architectures.TS.CNN import CNN_TS
from models.architectures.TS.RNN import RNN_TS
from models.architectures.TS.ResNet_1d import ResNet_TS
from models.model_wrapper import model_wrapper

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('univariate'))
sys.path.insert(0, os.path.abspath('time_series_augmentation'))

from datamodules.dtw import dtwDataModule
from models.architectures.DTW.ResNet import ResNet_DTW
from v1.networkAnalysis.summary import plot_roc_auc, plot_confusion_matrix


def train_network(dataset_name, mode, architecture, window_size, batch_size, lr, max_epochs, num_workers, PATH):
    if mode == 'sts':
        root_dir = f"data/{dataset_name}/TS"
        datamodule = stsDataModule
        if architecture == 'resnet':
            arch_model = ResNet_TS
        elif architecture == 'rnn':
            arch_model = RNN_TS
        elif architecture == 'cnn':
            arch_model = CNN_TS

    elif mode == 'dtw':
        root_dir = f"data/{dataset_name}/DTW"
        datamodule = dtwDataModule
        if architecture == 'resnet':
            arch_model = ResNet_DTW
        elif architecture == 'cnn':
            arch_model = CNN_DTW

    root_dir = f"{root_dir}/{architecture}"

    AVAIL_GPUS = min(0, torch.cuda.device_count())

    dataMod = datamodule(f'data/{dataset_name}', num_workers=0, batch_size=batch_size)
    dataMod.prepare_data(window_size=window_size)

    model = model_wrapper(model_architecture=arch_model,
                          ref_size=dataMod.dtw_test.dtws.shape[1] if mode == 'dtw' else None,
                          channels=dataMod.channels,
                          labels=dataMod.labels_size,
                          window_size=window_size,
                          lr=lr)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    # Initialize a trainer
    trainer = Trainer(
        default_root_dir=root_dir,
        callbacks=[lr_monitor],
        gpus=AVAIL_GPUS,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        progress_bar_refresh_rate=30,
        weights_save_path=f"{root_dir}/checkpoints",
        deterministic=True,
    )

    if PATH is None:
        # Train the model ⚡
        trainer.fit(model, datamodule=dataMod)
        trainer.validate(model, datamodule=dataMod)
        trainer.test(model, datamodule=dataMod)

        path = f"{root_dir}/checkpoints/lightning_logs/version_{trainer.logger.version}/checkpoints/" \
               f"epoch={trainer.current_epoch}-step={trainer.global_step - 1}.ckpt"

    else:
        path = PATH

    model = model_wrapper.load_from_checkpoint(path,
                                               model_architecture=architecture,
                                               ref_size=dataMod.dtw_test.dtws.shape[1] if mode == 'dtw' else None,
                                               channels=dataMod.channels,
                                               labels=dataMod.labels_size,
                                               window_size=window_size)

    model.eval()
    model.freeze()
    model.cuda()

    evaluate_trained_network(dataMod, batch_size, model, root_dir)


def evaluate_trained_network(dataMod, batch_size, model, root_dir):

    if type(dataMod) == stsDataModule:
        total_len = len(dataMod.sts_test)
    elif type(dataMod) == dtwDataModule:
        total_len = len(dataMod.dtw_test)

    y_pred = []
    y_true = []
    predict_dataloader = dataMod.test_dataloader()
    with torch.inference_mode():
        for i, (x, y) in tqdm(enumerate(predict_dataloader), total=total_len // batch_size):
            x = x.cuda()
            raw_score = model(x)
            y_pred.extend(raw_score.softmax(dim=-1).cpu().numpy())
            y_true.extend(y.cpu().numpy())
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    print('Classification Report')
    target_names = [str(i) for i in range(dataMod.channels)]
    print(classification_report(y_true, np.argmax(y_pred, axis=-1)))
    save_path = f"{root_dir}/net_results/"
    os.makedirs(save_path, exist_ok=True)
    plot_roc_auc(dataMod.channels,
                 F.one_hot(torch.tensor(y_true), num_classes=dataMod.channels).numpy(), y_pred,
                 save_path)
    plot_confusion_matrix(y_true, y_pred, target_names, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a ResNet from the DTWs created previously')

    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--mode', type=str, required=True, choices=['dtw', 'sts'],
                        help='Name of the dataset from which create the DTWs')

    parser.add_argument('--architecture', type=str, required=True, choices=['resnet', 'cnn', 'rnn'],
                        help='Name of the architecture from which create the model')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Window size used for training')

    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size used during training')

    parser.add_argument('--lr', type=float, default=5e-03,
                        help='Value of the learning rate')

    parser.add_argument('--max_epochs', type=int, default=15,
                        help='Number of epochs to train the network')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='Number of epochs to train the network')

    parser.add_argument('--PATH', type=str, default=None,
                        help='path to the checkpoint, if provided it will no train the network but only evaluate it')

    args = parser.parse_args()

    train_network(dataset_name=args.dataset_name, mode=args.mode, architecture=args.architecture,
                  window_size=args.window_size,
                  batch_size=args.batch_size, lr=args.lr, max_epochs=args.max_epochs,
                  num_workers=args.num_workers, PATH=args.PATH)
