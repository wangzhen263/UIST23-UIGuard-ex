import logging
import logging.handlers
import os
import sys
import warnings
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    confusion_matrix,
)

from models import *

global_logger = None

label_map = {
    "II-PRE": "II-Preselection",
    "II-AM-FH": "II-AM-FalseHierarchy-BAD",
    "II-AM-DA": "II-AM-DisguisedAD",
    "FA-SOCIALPYRAMID": "ForcedAction-SocialPyramid",
    "FA-Privacy": "ForcedAction-Privacy",
    # II-AM-FalseHierarchy-GOOD
    "NG": "Nagging",
    "II-AM-G-SMALL": "II-AM-General",
    "FA-G-PRO": "ForcedAction-General",
    "FA-G-COUNTDOWNAD": "ForcedAction-General",
    "FA-G-WATCHAD": "ForcedAction-General",
    "SN-FC": "Sneaking-ForcedContinuity",
}

['ForcedAction-Gamification', 'ForcedAction-General', 'ForcedAction-Privacy', 'ForcedAction-SocialPyramid', 'II-AM-DisguisedAD', 'II-AM-FalseHierarchy-BAD', 'II-AM-General', 'II-AM-ToyingWithEmotion', 'II-AM-Tricked', 'II-HiddenInformation', 'II-Preselection', 'Nagging', 'Obstruction-Currency', 'Obstruction-RoachMotel', 'Sneaking-BaitAndSwitch', 'Sneaking-ForcedContinuity', 'Sneaking-HiddenCosts']


class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            if line[-1] == "\n":
                encoded_message = line.encode("utf-8", "ignore").decode("utf-8")
                self.logger.log(self.log_level, encoded_message.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            encoded_message = self.linebuf.encode("utf-8", "ignore").decode("utf-8")
            self.logger.log(self.log_level, encoded_message.rstrip())
        self.linebuf = ""


def _init_logger(logger_name, logger_filename):
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.basicConfig(level=logging.INFO, encoding="utf-8")
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Avoid httpx flooding POST logs
    logging.getLogger("httpx").setLevel(logging.WARNING)

    handler = logging.handlers.TimedRotatingFileHandler(
        logger_filename, when="D", utc=True, encoding="utf-8"
    )
    handler.setFormatter(formatter)

    for l in [stdout_logger, stderr_logger, logger]:
        l.addHandler(handler)

    return logger


def get_global_logger(logger_name, logger_filename):
    global global_logger

    if global_logger is None:
        global_logger = _init_logger(logger_name, logger_filename)

    return global_logger


# def inference(model, data_loader, threshold=0.5):
def inference(model, data_loader, device, loss_fn):
    model.eval()
    all_loss = 0.0
    accuracy = []

    pred_labels = []
    gth_labels = []

    err_imgs = []

    data_idxs = []
    with torch.no_grad():
        for images, labels, idxs in data_loader:
            images = [img.to(device) for img in images]
            labels = labels.to(device)
            data_idxs.append(idxs)

            # output, xs = model(images)
            if isinstance(model, Bert_Classifier):
                output = model(images, labels)
                output = output["logits"]
            elif isinstance(model, Bert_ResNet):
                output, xs, bert_output = model(images, labels)
                bert_loss = bert_output["loss"].item()
                all_loss += loss_fn(xs, output, labels).item()
                all_loss += bert_loss
            else:
                output, xs = model(images, labels)
                all_loss += loss_fn(xs, output, labels).item()

            pred_labels += torch.argmax(output, dim=1).cpu().tolist()
            gth_labels += torch.argmax(labels, dim=1).cpu().tolist()

            pred_label = torch.argmax(output, dim=1).cpu().tolist()
            gth_label = labels.cpu().numpy()
            gth_label = [np.where(l==1)[0].tolist() for l in gth_label]
            
            correct = 0
            for i, (y_bar, y) in enumerate(zip(pred_label, gth_label)):
                if y_bar in y:
                    correct += 1
                else:
                    err_imgs.append([idxs[i].item(), y_bar])

            correct = torch.tensor(correct / len(labels))
            accuracy.append(correct)

            # pred_label = (output >= threshold).float()
            # correct_per_batch = torch.sum(pred_label == labels, dim=1)
            # num_labels = labels.size(1)
            # accuracy_per_batch = correct_per_batch == num_labels
            # accuracy_per_batch = torch.mean(accuracy_per_batch.float())
            # accuracy.append(accuracy_per_batch)

    all_loss /= len(data_loader.dataset)
    acc = (100.0 * torch.mean(torch.hstack(accuracy))).item()

    ma_p, ma_r, ma_f1, _ = precision_recall_fscore_support(
        gth_labels, pred_labels, average="macro", zero_division=np.nan
    )
    mi_p, mi_r, mi_f1, _ = precision_recall_fscore_support(
        gth_labels, pred_labels, average="micro"
    )
#    acc = accuracy_score(gth_labels, pred_labels)
    conf_matrix = confusion_matrix(gth_labels, pred_labels)

    return (
        acc,
        all_loss,
        [round(ma_p, 4), round(ma_r, 4), round(ma_f1, 4)],
        [round(mi_p, 4), round(mi_r, 4), round(mi_f1, 4)],
        conf_matrix,
        err_imgs
    )


def one_hot_vector(dataset, total_labels):
    for s in dataset:
        v = np.zeros(len(total_labels))
        for l in s[-1]:
            if l not in total_labels:
                l = label_map[l]
            v[total_labels.index(l)] = 1
        s[-1] = v

        if sum(v) > 1:
            print(s[0])
    return dataset
