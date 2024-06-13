import os
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import nlpaug.flow as naf
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

import cv2
from glob import glob
from tqdm import tqdm
import pickle as pk
import json
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from utils import *
        

class CustomDataset(Dataset):
    def __init__(
        self,
        data_set,
        tokenizer=None,
        max_seq_length=0,
        over_sample=False,
    ):
        self.transform_subimg = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((133, 100), antialias=False),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
#                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.transform_fullimg = transforms.Compose(
            [
                transforms.ToTensor(),
#                transforms.RandomCrop((640, 480)),
                transforms.Resize((640, 480), antialias=False),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
#                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self.transform_text = naf.Sequential(
            [
                naw.RandomWordAug("delete"),
                naw.RandomWordAug("swap"),
                nas.RandomSentAug()
            ]
        )

        self.data_set = data_set
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        labels = [np.argmax(ds[4]) for ds in data_set]
        labels_counts = Counter(labels) 
        max_count = max(labels_counts.values())

        self.sub_imgs, self.full_imgs = [], []
        self.token_rets = []
        self.one_hot_labels = []

        for ds in data_set:
            img_id, sub_img, f_full_img, text, one_hot_label = ds
            full_img_cv_ = cv2.imread(f_full_img)
            
            if over_sample:
                l_count = labels_counts[np.argmax(one_hot_label)]
                n_loop = max(1, round( (max_count - l_count) / l_count ))
            else:
                n_loop = 1
            
            while (n_loop > 0):
                self.sub_imgs.append(self.transform_subimg(sub_img))

                try:
                    full_img_cv = self.transform_fullimg(full_img_cv_)
                except:
                    print(f"####### {f_full_img} cannot read")
                    full_img_cv = torch.full((3, 640, 480), 0.00001)

                self.full_imgs.append(full_img_cv)

                if self.tokenizer is not None:
                    text = self.transform_text.augment(text)[0]
                    token_ret = self.tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_token_type_ids=True,
                        return_attention_mask=True,
                    )
                    self.token_rets.append(token_ret)
                
                self.one_hot_labels.append(one_hot_label)
                n_loop -= 1
    
        if over_sample:
            labels = [np.argmax(l) for l in self.one_hot_labels]

        self.class_weight = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)

        assert len(self.one_hot_labels) == len(self.sub_imgs)
        assert len(self.sub_imgs) == len(self.full_imgs)
        if self.tokenizer is not None:
            assert len(self.one_hot_labels) == len(self.sub_imgs)
    
    def class_weight(self):
        return self.class_weight

    def __len__(self):
        return len(self.one_hot_labels)

    def __getitem__(self, idx):
        one_hot_label = self.one_hot_labels[idx]
        sub_img = self.sub_imgs[idx]
        full_img_cv = self.full_imgs[idx]

        if len(self.token_rets) > 0:
            input_ids = self.token_rets[idx]["input_ids"]
            attention_mask = self.token_rets[idx]["attention_mask"]
            token_type_ids = self.token_rets[idx]["token_type_ids"]
            return (
                [
                    sub_img,
                    full_img_cv,
                    np.array(input_ids),
                    np.array(attention_mask),
                    np.array(token_type_ids),
                ],
                one_hot_label,
                idx,
            )

        return [sub_img, full_img_cv], one_hot_label, idx


def prepare_dataset(path_img_text):
    f_gth_img_texts = glob(f"{path_img_text}/*_subimg_text_ph.json")
    img_ids = [
        p.replace("_subimg_text_ph.json", "").split("/")[-1] for p in f_gth_img_texts
    ]
    f_all_texts = [f"{path_img_text}/{img_id}_all_text_ph.json" for img_id in img_ids]

    img_all_texts = {}
    for ft in tqdm(f_all_texts):
        with open(ft, "r") as f:
            all_texts = json.load(f)
            txts = " ".join([t[0] for t in all_texts])
            img_id = ft.replace("_all_text_ph.json", "").split("/")[-1]
            img_all_texts[img_id] = txts

    class_img_bnd_text = {}
    for f_gth in tqdm(f_gth_img_texts):
        img_id = f_gth.replace("_subimg_text_ph.json", "").split("/")[-1]

        with open(f_gth, "r") as f:
            img_bnd_texts = json.load(f)

            for gth in img_bnd_texts:
                bnd, text, labels = gth
                
                if len(labels) > 1:
                    print("## LABELS: ", labels)
                
                _labels = []
                for l in labels:
                    if l == "II-HiddenInformation-TEXT":
                        _labels.append("II-HiddenInformation")
                    elif l == "II-HiddenInformation-ICON":
                        continue
                    else:
                        _labels.append(l)
                labels = _labels
        
                for l in labels:
                    if l not in class_img_bnd_text:
                        class_img_bnd_text[l] = [[img_id, bnd, text, labels]]
                    else:
                        class_img_bnd_text[l].append([img_id, bnd, text, labels])

    if "NG" in class_img_bnd_text:
        class_img_bnd_text["NG"] = (
            class_img_bnd_text["NG-UPGRADE"]
            + class_img_bnd_text["NG-RATE"]
            + class_img_bnd_text["NG-AD"]
        )
    if "II-PRE" in class_img_bnd_text:
        class_img_bnd_text["II-PRE"] = (
            class_img_bnd_text["II-PRE"] + class_img_bnd_text["II-PRE-Nocheckbox"]
        )
    if "NG-UPGRADE" in class_img_bnd_text:
        del class_img_bnd_text["NG-UPGRADE"]
    if "NG-RATE" in class_img_bnd_text:
        del class_img_bnd_text["NG-RATE"]
    if "NG-AD" in class_img_bnd_text:
        del class_img_bnd_text["NG-AD"]
    if "II-PRE-Nocheckbox" in class_img_bnd_text:
        del class_img_bnd_text["II-PRE-Nocheckbox"]

    if "Include" in class_img_bnd_text:
        del class_img_bnd_text["Include"]

    labels = list(class_img_bnd_text.keys())

    for l, d_gth_imgs in class_img_bnd_text.items():
        for d_gth_img in d_gth_imgs:
            y_gth = []
            for d in d_gth_img[3]:
                if "II-PRE" in d:
                    d = "II-PRE"
                elif "NG" in d:
                    d = "NG"
                y_gth.append(d)
            d_gth_img[3] = y_gth

    return class_img_bnd_text, img_all_texts, labels


def load_dataset(f_in_img_text, f_out_proc_data, f_root):
    f_root = f"{f_root}/jeff_images_w_ori_path"

    if not os.path.exists(f"{f_out_proc_data}"):
#    if os.path.exists(f'{f_out_proc_data}'):
        class_threshold = 5 
        class_img_bnd_text, img_all_texts, _ = prepare_dataset(f_in_img_text)

        # label_num_data = {k: len(v) for k, v in class_img_bnd_text.items()}
        train_img_bnd_text = {}
        test_img_bnd_text = {}
        max_seq_length = 0

        labels = []
        for label, img_bnd_texts in class_img_bnd_text.items():
            if len(img_bnd_texts) >= class_threshold:
                # will do data augmentation 
                n_sample = round(len(img_bnd_texts) * 0.8)
            elif len(img_bnd_texts) > 1:
                n_sample = len(img_bnd_texts) - 1
            else:
                continue

            train_img_bnd_text[label] = img_bnd_texts[:n_sample]
            test_img_bnd_text[label] = img_bnd_texts[n_sample:]
            labels.append(label)

        train_set, test_set = [], []
        for l, img_bnd_texts in train_img_bnd_text.items():
            for img_bnd_text in img_bnd_texts:
                img_id, bnd, text, y_gths = img_bnd_text
                if img_id[-4:] != ".png":
                    f_full_img = f"{f_root}/{img_id}.jpg"
                else:
                    f_full_img = f"{f_root}/{img_id}"

                sub_img = get_sub_image(bnd, f_full_img)
                all_text = img_all_texts[img_id]

                text = "[This element does not contain texts]" if text == "" else text
                text = ", ".join([text, all_text])

                if len(text.split(" ")) > max_seq_length:
                    max_seq_length = len(text.split(" "))

                train_set.append([img_id, sub_img, f_full_img, text, y_gths])

        for l, img_bnd_texts in test_img_bnd_text.items():
            for img_bnd_text in img_bnd_texts:
                img_id, bnd, text, y_gths = img_bnd_text
                if img_id[-4:] != ".png":
                    f_full_img = f"{f_root}/{img_id}.jpg"
                else:
                    f_full_img = f"{f_root}/{img_id}"

                sub_img = get_sub_image(bnd, f_full_img)
                all_text = img_all_texts[img_id]

                text = "None" if text == "" else text
                text = ", ".join([text, all_text])

                if len(text.split(" ")) > max_seq_length:
                    max_seq_length = len(text.split(" "))

                test_set.append([img_id, sub_img, f_full_img, text, y_gths])

        with open(f"{f_out_proc_data}", "wb") as f:
            pk.dump([train_set, test_set, labels, max_seq_length], f)
    else:
        with open(f"{f_out_proc_data}", "rb") as f:
            train_set, test_set, labels, max_seq_length = pk.load(f)

    return train_set, test_set, labels, max_seq_length


def get_sub_image(sub_img_bnd, img_path):
    full_img_cv = cv2.imread(img_path)
    # width, height, channels = full_img_cv.shape

    mask = np.zeros_like(full_img_cv[:, :, 0])
    pts = np.array(sub_img_bnd, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    masked_image = cv2.bitwise_and(full_img_cv, full_img_cv, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    sub_image = masked_image[y : y + h, x : x + w]
    return sub_image
