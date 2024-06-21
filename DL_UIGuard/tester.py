import os
from glob import glob
import json
from tqdm import tqdm
import numpy as np
import cv2
import pickle as pk

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from models import CNN, Bert_ResNet
from dataset import *
from utils import *


RICO_DATASET_ROOT = "/data/scsgpu1/work/jeffwang/dark_pattern/rico_testset"
RICO_DATASET_DETECTION = f"{RICO_DATASET_ROOT}/detection"
RICO_DATASET_IMG_TEXT = f"{RICO_DATASET_DETECTION}/text"
RICO_DATASET_PROC_DATA = f"{RICO_DATASET_DETECTION}/proc_dataset.pk"

OUR_DATASET_ROOT = "/data/scsgpu1/work/jeffwang/dark_pattern/our_labelling"
OUR_DATASET_DETECTION = f"{OUR_DATASET_ROOT}/detection"
OUR_DATASET_IMG_TEXT = f"{OUR_DATASET_DETECTION}/text"
OUR_DATASET_PROC_DATA = f"{OUR_DATASET_DETECTION}/proc_dataset.pk"

N_EPOCH = 300
N_BATCH = 32
N_LAYERS = 50
DEVICE = "cuda:0"

date_str = datetime.datetime.now().strftime("%d-%m-%Y")
file_logger = get_global_logger(
    "DP_trainer", f"/data/scsgpu1/work/jeffwang/dark_pattern/test_log_{date_str}.out"
)

MODEL_OUTPUT_PREFIX = "/data/scsgpu1/work/jeffwang/dark_pattern"

if __name__ == "__main__":

    rico_train_set, rico_test_set, rico_labels, max_seq_length_1 = load_dataset(
        RICO_DATASET_IMG_TEXT, RICO_DATASET_PROC_DATA, RICO_DATASET_ROOT
    )
    our_train_set, our_test_set, our_labels, max_seq_length_2 = load_dataset(
        OUR_DATASET_IMG_TEXT, OUR_DATASET_PROC_DATA, OUR_DATASET_ROOT
    )
    max_seq_length = max([max_seq_length_1, max_seq_length_2])
    total_labels = our_labels + [label_map[l] for l in rico_labels]
    total_labels = list(set(total_labels))
    tatal_labels = sorted(total_labels)

    {
        "Sneaking-BaitAndSwitch": 63,
        "II-AM-FalseHierarchy-BAD": 268,
        "Sneaking-HiddenCosts": 2,
        "Obstruction-RoachMotel": 14,
        "II-Preselection": 543,
        "II-HiddenInformation": 165,
        "Nagging": 189,
        "ForcedAction-Privacy": 111,
        "ForcedAction-General": 165,
        "ForcedAction-SocialPyramid": 26,
        "ForcedAction-Gamification": 12,
        "II-AM-General": 581,
        "II-AM-ToyingWithEmotion": 51,
        "Obstruction-Currency": 33,
        "Sneaking-ForcedContinuity": 57,
        "II-AM-DisguisedAD": 106,
        "II-AM-Tricked": 6,
    }

    #    l_count = {t: 0 for t in total_labels}
    #    for t in rico_train_set:
    #        for l in t[-1]:
    #            if l not in total_labels:
    #                l = label_map[l]
    #            l_count[l] += 1
    #    for t in rico_test_set:
    #        for l in t[-1]:
    #            if l not in total_labels:
    #                l = label_map[l]
    #            l_count[l] += 1
    #    for t in our_train_set:
    #        for l in t[-1]:
    #            if l not in total_labels:
    #                l = label_map[l]
    #            l_count[l] += 1
    #    for t in our_test_set:
    #        for l in t[-1]:
    #            if l not in total_labels:
    #                l = label_map[l]
    #            l_count[l] += 1
    #    print(l_count)

    tokenizer = None
    f_checkpoint = f"{MODEL_OUTPUT_PREFIX}/best_ckpt"

    lr = 0.01
    resnet_model = SiameseResNet(
        n_channels=3, n_class=len(tatal_labels), n_layers=N_LAYERS
    )
    resnet_model.load_state_dicts(f_checkpoint)

    #    bert_resnet_model = Bert_ResNet(
    #        n_channels=3, n_class=len(tatal_labels), n_layers=N_LAYERS
    #    )
    #    tokenizer = bert_resnet_model.tokenizer

    #####

    # SETTING 1
    #    f_checkpoint = f"{MODEL_OUTPUT_PREFIX}/resnet_bert_best_ckpt"
    #    bert_resnet_model.freeze_encoders()

    # SETTING 2
    #    f_checkpoint = f"{MODEL_OUTPUT_PREFIX}/resnet_bert_best_ckpt2"

    #####

    #    bert_resnet_model.load_state_dicts(f_checkpoint, DEVICE)

    #    bert_model = Bert_Classifier(n_class=len(tatal_labels))
    #    tokenizer = bert_model.tokenizer
    #    bert_model.to(DEVICE)
    #    s = torch.load('/data/scsgpu1/work/jeffwang/dark_pattern/bert_only_best_checkpoints_50.pt')
    #    bert_model.load_state_dict(s['model_state_dict'])
    #    bert_model.save_state_dicts(f_checkpoint)

    cls_model = resnet_model
    cls_model.to(DEVICE)

    optimizer = torch.optim.AdamW(cls_model.parameters(), lr=lr, eps=1e-8)  # 0.01 0.001

    our_test_set = one_hot_vector(our_test_set, tatal_labels)
    all_test_set = our_test_set

    test_dataset = CustomDataset(all_test_set, tokenizer, max_seq_length)
    test_dataloader = DataLoader(test_dataset, batch_size=N_BATCH, shuffle=False)

    #    f_state_dict = f"best_checkpoints_{N_LAYERS}.pt"
    #    checkpoint = torch.load(f_state_dict)

    # cnn_model = CNN(n_channels=3, n_class=11)
    #    cnn_model = Bert_ResNet(n_channels=3, n_class=len(tatal_labels), n_layers=N_LAYERS)
    #    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    #
    #    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.005)  # 0.01
    #    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #
    #    cnn_model.cuda()

    test_acc, test_loss, mac, mic, conf_mat, _ = inference(
        cls_model, test_dataloader, DEVICE
    )

    #    train_acc, _ = inference(cnn_model, train_dataloader)
    #    file_logger.info(f"Train acc: {train_acc}")

    #    test_acc, test_loss = inference(cnn_model, test_dataloader)
    file_logger.info(f"Test acc: {test_acc}, test loss: {test_loss}")
