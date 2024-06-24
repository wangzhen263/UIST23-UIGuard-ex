import os
import json
from tqdm import tqdm
import numpy as np
import cv2
import pickle as pk
import datetime
import click

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as transforms

from transformers import AdamW

from utils import *
from models import *
from dataset import *

from loss import ContrastiveLoss

RICO_DATASET_ROOT = "/data/scsgpu1/work/jeffwang/dark_pattern/rico_testset"
RICO_DATASET_DETECTION = f"{RICO_DATASET_ROOT}/detection"
RICO_DATASET_IMG_TEXT = f"{RICO_DATASET_DETECTION}/text"
RICO_DATASET_PROC_DATA = f"{RICO_DATASET_DETECTION}/proc_dataset.pk"

# OUR_DATASET_ROOT = "/data/scsgpu1/work/jeffwang/dark_pattern/our_labelling"
OUR_DATASET_ROOT = "/data/scsgpu1/work/jeffwang/dark_pattern/our_labelling_v2"
OUR_DATASET_DETECTION = f"{OUR_DATASET_ROOT}/detection"
OUR_DATASET_IMG_TEXT = f"{OUR_DATASET_DETECTION}/text"
OUR_DATASET_PROC_DATA = f"{OUR_DATASET_DETECTION}/proc_dataset.pk"
OUR_DATASET_PROC_DATA_APP = f"{OUR_DATASET_DETECTION}/proc_dataset_app.pk"

# /data/scsgpu1/work/jeffwang/dark_pattern/our_labelling_v2/detection/all_text/

N_EPOCH = 300
N_BATCH = 32  # 32
N_LAYERS = 50
DEVICE = "cuda:1"
MODEL_OUTPUT_PREFIX = OUR_DATASET_ROOT  # "/data/scsgpu1/work/jeffwang/dark_pattern"

#USE_CLASS_WEIGHT = True  # False or True
#USE_NEGATIVE_SAMPLE = True  # False or True
#USE_OVER_SAMPLE = True  # False or True

date_str = datetime.datetime.now().strftime("%d-%m-%Y")


@click.command()
@click.option("--use-class-weight", default=True, help="Enable class weight.")
@click.option("--use-negative-sampling", default=True, help="Enable negative sampling.")
@click.option(
    "--use-balance-augmentation", default=True, help="Enable balanced augmentation."
)
@click.option(
    "--dl-model",
    type=click.Choice(["RESNET", "BERT", "BERT-RESNET-F", "BERT-RESNET-NF"]),
    default="BERT-RESNET-F",
    help="Choose a model type.",
)
@click.option("--gpu", default=1, type=int, help="Choose a GPU.")
def main(use_class_weight, use_negative_sampling, use_balance_augmentation, dl_model, gpu):
    DEVICE = f"cuda:{gpu}"

    rico_train_set, rico_test_set, rico_labels, max_seq_length_1 = load_dataset(
        RICO_DATASET_IMG_TEXT, RICO_DATASET_PROC_DATA, RICO_DATASET_ROOT
    )
    our_train_set, our_test_set, our_labels, max_seq_length_2 = load_dataset(
        OUR_DATASET_IMG_TEXT, OUR_DATASET_PROC_DATA, OUR_DATASET_ROOT
    )
    max_seq_length = max([max_seq_length_1, max_seq_length_2])
    total_labels = our_labels + [label_map[l] for l in rico_labels]
    total_labels = list(set(total_labels))
    total_labels = sorted(total_labels)
    print(total_labels)

    exit(0)

    tokenizer = None
    f_checkpoint = f"{MODEL_OUTPUT_PREFIX}/{N_BATCH}_best_ckpt_2"

    if use_class_weight:
        f_checkpoint = f"{f_checkpoint}_cw"
    if use_negative_sampling:
        f_checkpoint = f"{f_checkpoint}_ne"
    if use_balance_augmentation:
        f_checkpoint = f"{f_checkpoint}_os"

    if dl_model == "RESNET":
        resnet_model = SiameseResNet(
            n_channels=3, n_class=len(tatal_labels), n_layers=N_LAYERS
        )
        resnet_model.to(DEVICE)
        lr = 0.003
        model_type = "SiameseResNet"
        f_checkpoint = f"{f_checkpoint}_{model_type}_lr{lr}"
        resnet_model.load_state_dicts(f_checkpoint)
        cls_model = resnet_model

    if dl_model == "BERT":
        bert_model = Bert_Classifier(n_class=len(tatal_labels))
        tokenizer = bert_model.tokenizer
        bert_model.to(DEVICE)
        lr = 3e-5
        model_type = 'Bert_Classifier'
        f_checkpoint = f"{f_checkpoint}_{model_type}_lr{lr}"
        bert_model.load_state_dicts(f_checkpoint)
        cls_model = bert_model

    if dl_model in ["BERT-RESNET-F", "BERT-RESNET-NF"]:
        bert_resnet_model = Bert_ResNet(
            n_channels=3, n_class=len(tatal_labels), n_layers=N_LAYERS
        )
        tokenizer = bert_resnet_model.tokenizer
        model_type = 'Bert_ResNet'
        r_lr = 0.003
        b_lr = 3e-5
        f_resnet_encoder = get_pt_path(f"{f_checkpoint}_SiameseResNet_lr{r_lr}", "RESNET")
        f_bert_encoder = get_pt_path(f"{f_checkpoint}_Bert_Classifier_lr{b_lr}", "BERT")
        
        bert_resnet_model.load_encoders(f_bert_encoder, f_resnet_encoder, DEVICE)
    
    if dl_model == "BERT-RESNET-F":
        lr = 0.0001 #0.0001 0.001 0.005 #0.01
        f_checkpoint = f"{f_checkpoint}_{model_type}_fr_lr{lr}"
        try:
            bert_resnet_model.load_state_dicts(f_checkpoint, DEVICE)
        except ValueError:
            pass
        bert_resnet_model.freeze_encoders()
        cls_model = bert_resnet_model

    if dl_model == "BERT-RESNET-NF":
        lr = 3e-5
        f_checkpoint = f"{f_checkpoint}_{model_type}_nfr_lr{lr}"
        try:
            bert_resnet_model.load_state_dicts(f_checkpoint, DEVICE)
        except ValueError:
            pass
        cls_model = bert_resnet_model

    file_logger = get_global_logger(
        "DP_trainer",
        f"/data/scsgpu1/work/jeffwang/dark_pattern/log_{model_type}_{date_str}.out",
    )

    cls_model.to(DEVICE)

    #    optimizer = torch.optim.AdamW(cls_model.parameters(), lr=lr, eps=1e-8)  # 0.01 0.001

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in cls_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p
                for n, p in cls_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)

    #    if os.path.exists(f_checkpoint):
    #        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    rico_train_set = one_hot_vector(rico_train_set, tatal_labels)
    rico_test_set = one_hot_vector(rico_test_set, tatal_labels)
    our_train_set = one_hot_vector(our_train_set, tatal_labels)
    our_test_set = one_hot_vector(our_test_set, tatal_labels)

    all_train_set = rico_train_set + our_train_set + rico_test_set
    all_test_set = our_test_set

    train_dataset = CustomDataset(
        all_train_set, tokenizer, max_seq_length, over_sample=use_balance_augmentation
    )
    test_dataset = CustomDataset(all_test_set, tokenizer, max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=N_BATCH, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=N_BATCH, shuffle=False)

    f_checkpoint = get_pt_path(f_checkpoint, dl_model)
    file_logger.info(f"f_checkpoint: {f_checkpoint}")

    test_accs, test_macs, test_mics = (
        [0.1],
        [[0.1, 0.1, 0.1]],
        [[0.1, 0.1, 0.1]],
    )
    train_losses = []
    best_model = None

    if use_class_weight:
        cw = torch.Tensor(train_dataset.class_weight).to(DEVICE)
        loss_fn = ContrastiveLoss(weight=cw)
    else:
        loss_fn = ContrastiveLoss()

    for epoch in tqdm(range(N_EPOCH)):
        train_loss = 0
        cls_model.train()

        for batch_idx, (images_texts, labels, t_idx) in enumerate(train_dataloader):
            images_texts = [img_text.to(DEVICE) for img_text in images_texts]
            labels = labels.to(DEVICE)

            cls_model.zero_grad()

            if isinstance(cls_model, Bert_Classifier):
                bert_output = cls_model(images_texts, labels)
                loss = bert_output["loss"]
            elif isinstance(cls_model, Bert_ResNet):
                output, xs, bert_output = cls_model(images_texts, labels)
                loss = loss_fn(xs, output, labels, negloss=use_negative_sampling)
                loss += bert_output["loss"]
            else:
                output, xs = cls_model(images_texts, labels)
                loss = loss_fn(xs, output, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # print(f'Batch {batch_idx} Loss: {loss}')

        train_losses.append(train_loss / len(train_dataloader))
        train_acc, _, _, _, _, _, _ = inference(
            cls_model, train_dataloader, DEVICE, loss_fn
        )
        file_logger.info(
            f"Epoch {epoch}, Train Loss: {train_loss/len(train_dataloader)}, Train acc: {round(train_acc, 4)}"
        )

        test_acc, test_loss, mac, mic, conf_mat, err_imgs, pred_floats = inference(
            cls_model, test_dataloader, DEVICE, loss_fn
        )

        test_acc = round(test_acc, 4)
        test_accs.append(test_acc)
        test_macs.append(mac)
        test_mics.append(mic)

        p_max_test = np.argmax(test_accs)
        file_logger.info(
            f"Epoch {epoch}, Test acc: {test_acc}/{max(test_accs)}, test loss: {test_loss}"
        )
        file_logger.info(
            f"Epoch {epoch}, macro: {test_macs[p_max_test]}, micro: {test_mics[p_max_test]}"
        )

        if test_acc == max(test_accs):
            best_model = cls_model
            best_model.save_state_dicts(f_checkpoint)
            file_logger.info(f"Confusion Matrix: \n{conf_mat}")

            with open(f"{MODEL_OUTPUT_PREFIX}/err_msg.json", "w") as f:
                json.dump(err_imgs, f)
            with open(f"{MODEL_OUTPUT_PREFIX}/pred_floats.pk", "wb") as f:
                pk.dump(pred_floats, f)


if __name__ == "__main__":
    main()
