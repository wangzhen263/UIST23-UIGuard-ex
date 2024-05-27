import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class ContrastiveLoss(nn.Module):
    def __init__(self, n_negative=16, weight=None):
        super(ContrastiveLoss, self).__init__()
        self.n_negative = n_negative
        self.class_weight = weight

    def forward(self, xs, y_bars, y_gths, negloss=False):
        loss_1 = F.cross_entropy(y_bars, y_gths, weight=self.class_weight)

        if negloss:
            loss_2 = []
            _, inv, counts = torch.unique(
                y_gths, return_inverse=True, return_counts=True, dim=0
            )
            dup_items = [
                tuple(torch.where(inv == i)[0].tolist())
                for i, c, in enumerate(counts)
                if counts[i] > 1
            ]

            pos_pairs = []
            if len(dup_items) > 0:
                for dups in dup_items:
                    if len(dups) > 2:
                        combines = [list(subset) for subset in combinations(dups, 2)]
                        pos_pairs += combines

                        loss_2 += [
                            1 - F.cosine_similarity(xs[c[0]], xs[c[1]], dim=0)
                            for c in combines
                        ]
                    elif len(dups) == 2:
                        pos_pairs.append(list(dups))
                        loss_2.append(
                            1 - F.cosine_similarity(xs[dups[0]], xs[dups[1]], dim=0)
                        )

            if len(dup_items) > self.n_negative:
                self.n_negative = len(dup_items)

            if len(loss_2) > 2:
                loss_2 = torch.mean(torch.hstack(loss_2))
            elif len(loss_2) == 1:
                loss_2 = loss_2[0]
            else:
                loss_2 = 0.0

            loss_3 = 0.0
            l_negatives = []
            for pair in list(combinations(range(len(y_gths)), r=2)):
                if pair not in pos_pairs:
                    l_negatives.append(pair)

                if len(l_negatives) >= self.n_negative:
                    break

            loss_3 = torch.mean(
                torch.hstack(
                    [
                        torch.max(
                            torch.as_tensor(
                                [0, F.cosine_similarity(xs[c[0]], xs[c[1]], dim=0)]
                            )
                        )
                        for c in l_negatives
                    ]
                )
            )

            loss = loss_1 + loss_2 + loss_3
        else:
            loss = loss_1
        return loss
