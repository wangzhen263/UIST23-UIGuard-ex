import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Type

from torch.nn import BCEWithLogitsLoss

import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import pickle as pk

BERT_NAME = "bert-base-uncased"


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, num_layers, num_channels=3):
        super(ResNet, self).__init__()
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
        elif num_layers == 50:
            layers = [3, 4, 6, 3]
            self.expansion = 4
        elif num_layers == 101:
            layers = [3, 4, 23, 3]
            self.expansion = 4
        elif num_layers == 152:
            layers = [3, 8, 36, 3]
            self.expansion = 4

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layers[0], planes=64)
        self.layer2 = self._make_layer(ResBlock, layers[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layers[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layers[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResBlock.expansion, 512)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes * ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    planes * ResBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(planes * ResBlock.expansion),
            )

        layers.append(
            ResBlock(
                self.in_channels, planes, i_downsample=ii_downsample, stride=stride
            )
        )
        self.in_channels = planes * ResBlock.expansion

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


class CNN_simple(nn.Module):
    def __init__(self, n_channels: int):
        super(CNN, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)

        # self.conv1_f = nn.Conv2d(n_channels, 16, 3)
        # self.conv2_f = nn.Conv2d(16, 16, 3)
        # self.fc_f = nn.Linear(16 * 158 * 118, 64)

        self.conv1_s = nn.Conv2d(3, 6, 3)
        self.conv2_s = nn.Conv2d(6, 6, 3)
        self.fc_s = nn.Linear(6 * 23 * 23, 32)

        # self.fc = nn.Linear(64+32, n_class)

    def forward(self, x):
        sub_img, full_img = x

        # x_s = self.pool(F.relu(self.conv1_s(sub_img)))
        # x_s = self.pool(F.relu(self.conv2_s(x_s)))
        # # print('x_s ', x_s.shape)
        # x_s = x_s.view(-1, 6 * 23 * 23)
        # x_s = F.relu(self.fc_s(x_s))

        x_f = self.pool(F.relu(self.conv1_f(full_img)))
        x_f = self.pool(F.relu(self.conv2_f(x_f)))
        # print('x_f ', x_f.shape)

        x_f = x_f.view(-1, 16 * 158 * 118)
        x_f = self.fc_f(x_f)

        # x = torch.hstack((x_s, x_f))
        # return self.fc(x)
        return x_f


class Bert_ResNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_class: int,
        n_layers: int = 50,
        bert_name: str = BERT_NAME,
    ):
        super(Bert_ResNet, self).__init__()

        self.bert_name = bert_name
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.text_classifier = AutoModelForSequenceClassification.from_pretrained(
            bert_name, num_labels=n_class
        )
        self.text_encoder = self.text_classifier.bert
        self.dropout = nn.Dropout(0.1)
        self.save_encoder = True

        self.resnet_cnn = ResNet(
            num_channels=n_channels,
            num_layers=n_layers,
            ResBlock=Bottleneck,
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 2 + 768, 64),
            #                nn.ReLU(),
            nn.Linear(64, n_class),
        )

    def load_encoders(self, f_bert, f_resnet, device="cuda:0"):
        f_bert = f"{f_bert}_bert_only.pt"
        f_resnet = f"{f_resnet}_resnet_only.pt"

        if os.path.exists(f_bert):
            print("## load_encoders ", f_bert)
            bert_state_dict = torch.load(f_bert, map_location=torch.device(device))
            self.text_classifier.load_state_dict(bert_state_dict["model_state_dict"])
        else:
            raise ValueError(f"{f_bert} checkpoint not existed.")

        if os.path.exists(f_resnet):
            print("## load_encoders ", f_resnet)
            resnet_state_dict = torch.load(f_resnet, map_location=torch.device(device))
            self.resnet_cnn.load_state_dict(resnet_state_dict["model_state_dict"])
        else:
            raise ValueError(f"{f_resnet} checkpoint not existed.")

    def load_state_dicts(self, f_ckpt, device="cuda:0"):
        f_ckpt = f"{f_ckpt}_bert_resnet.pt"
        if os.path.exists(f_ckpt):
            ckpt_state_dict = torch.load(f_ckpt, map_location=torch.device(device))
            if "bert_encoder" in ckpt_state_dict:
                self.text_classifier.load_state_dict(ckpt_state_dict["bert_encoder"])
            if "resnet_encoder" in ckpt_state_dict:
                self.resnet_cnn.load_state_dict(resnet_state_dict["resnet_encoder"])

            self.fc.load_state_dict(ckpt_state_dict["model_state_dict"])
        else:
            print(f"ERROR: {f_ckpt} checkpoint not exists. No checkpoint loaded.")

    def save_state_dicts(self, f_ckpt):
        model_ckpt = {"model_state_dict": self.fc.state_dict()}

        if self.save_encoder:
            model_ckpt["bert_encoder"] = self.text_classifier.state_dict()
            model_ckpt["resnet_encoder"] = self.resnet_cnn.state_dict()

        torch.save(model_ckpt, f"{f_ckpt}_bert_resnet.pt")

    def freeze_encoders(self, n_encoder="all"):
        if n_encoder in ["ResNet", "all"]:
            for param in self.resnet_cnn.parameters():
                param.requires_grad = False

        if n_encoder in ["bert", "all"]:
            for param in self.text_classifier.parameters():
                param.requires_grad = False
        self.save_encoder = False

    def forward(self, x, y):
        sub_img, full_img, input_ids, attention_mask, token_type_ids = x
        t_y = self.text_classifier(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=y,
        )
        t_emb = self.text_encoder(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )[1]
        t_emb = self.dropout(t_emb)

        x_s = self.resnet_cnn(sub_img)
        x_f = self.resnet_cnn(full_img)
        x = torch.hstack((x_s, x_f, t_emb))
        x = self.dropout(x)

        return self.fc(x), x, t_y


class Bert_Classifier(nn.Module):
    def __init__(self, n_class: int):
        super(Bert_Classifier, self).__init__()

        self.bert_name = BERT_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.text_classifier = AutoModelForSequenceClassification.from_pretrained(
            self.bert_name, num_labels=n_class
        )

    def load_state_dicts(self, f_model, device="cuda:0"):
        f_model = f"{f_model}_bert_only.pt"
        if os.path.exists(f_model):
            bert_state_dict = torch.load(f_model, map_location=torch.device(device))
            self.text_classifier.load_state_dict(bert_state_dict["model_state_dict"])
        else:
            print(f"ERROR: {f_model} checkpoint not existed. No checkpoint loaded.")

    def save_state_dicts(self, f_bert):
        f_bert = f"{f_bert}_bert_only.pt"
        print(f_bert)

        torch.save(
            {"model_state_dict": self.text_classifier.state_dict()},
            f_bert,
        )

    def forward(self, x, y):
        if len(x) > 2:
            _, _, input_ids, attention_mask, token_type_ids = x
            t_y = self.text_classifier(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=y,
            )
            return t_y
        else:
            return None


class CNN(nn.Module):
    def __init__(self, n_channels: int, n_class: int):
        super(CNN, self).__init__()
        self.cnn = ResNet(n_channels=n_channels, num_classes=None)
        self.fc = nn.Linear(32 * 2, n_class)

    def forward(self, x):
        sub_img, full_img = x
        x_s = self.cnn(sub_img)
        x_f = self.cnn(full_img)
        x = torch.hstack((x_s, x_f))
        return self.fc(x)


class SiameseResNet(nn.Module):
    def __init__(self, n_class: int, n_layers: int = 50, n_channels: int = 3):
        super(SiameseResNet, self).__init__()

        self.resnet_cnn = ResNet(
            num_channels=n_channels,
            num_layers=n_layers,
            ResBlock=Bottleneck,
        )
        self.fc = nn.Linear(512 * 2, n_class)

    def save_state_dicts(self, f_resnet):
        f_resnet = f"{f_resnet}_resnet_only.pt"
        torch.save(
            {
                "model_state_dict": self.resnet_cnn.state_dict(),
                "fc_state_dict": self.fc.state_dict(),
            },
            f_resnet,
        )

    def load_state_dicts(self, f_resnet, device="cuda:0"):
        f_resnet = f"{f_resnet}_resnet_only.pt"
        print("SiameseResNet load_state_dicts ", f_resnet)

        if os.path.exists(f_resnet):
            resnet_state_dict = torch.load(f_resnet, map_location=torch.device(device))
            self.resnet_cnn.load_state_dict(resnet_state_dict["model_state_dict"])
            self.fc.load_state_dict(resnet_state_dict["fc_state_dict"])
        else:
            print(f"ERROR: {f_resnet} checkpoint not existed. No checkpoint loaded.")

    def forward(self, x, y):
        sub_img, full_img = x
        x_s = self.resnet_cnn(sub_img)
        x_f = self.resnet_cnn(full_img)
        x = torch.hstack((x_s, x_f))
        return self.fc(x), x
