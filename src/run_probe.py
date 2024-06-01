import argparse
from pathlib import Path
import torch
import os
import json
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict
from sklearn.metrics import auc, roc_curve
from glob import glob


class LRProbe(torch.nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, 1, bias=False), torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def pred(self, x):
        return self(x).round()

    def score(self, x):
        return self(x)

    def from_data(acts, labels, lr=0.001, weight_decay=0.1, epochs=1000, device="cpu"):
        acts, labels = acts.to(device), labels.to(device)
        probe = LRProbe(acts.shape[-1]).to(device)

        opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.BCELoss()(probe(acts), labels)
            loss.backward()
            opt.step()

        return probe
    

class ActDataset:
    """
    Class for storing activations and labels from datasets of statements.
    """

    def __init__(self, dataset, dataset_name, model_name, layer_num, device):
        self.data = {}
        for layer in range(layer_num):
            acts = self.collect_acts(
                dataset_name, model_name, layer, device=device
            )
            labels = torch.Tensor([ex["label"] for ex in dataset]).to(device)
            self.data[layer] = acts, labels

    def collect_acts(
        self, dataset_name, model_name, layer, center=True, scale=True, device="cpu"
    ):
        directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "acts", model_name, dataset_name)
        activation_files = glob(os.path.join(directory, f"layer_{layer}_*.pt"))
        acts = [
            torch.load(os.path.join(directory, f"layer_{layer}_{i}.pt")).to(device)
            for i in range(0, 25 * len(activation_files), 25)
        ]
        acts = torch.cat(acts, dim=0).to(device)
        if center:
            acts = acts - torch.mean(acts, dim=0)
        if scale:
            acts = acts / torch.std(acts, dim=0)
        return acts

    def get(self, layer):
        return self.data[layer]




def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plm_dir", type=str, default="/public/home/wlchen/plm")
    parser.add_argument("--target_model", type=str, default="pythia-2.8b")
    parser.add_argument("--train_set", type=str, default="")
    parser.add_argument("--train_set_path", type=str, default="")
    parser.add_argument("--dev_set", type=str, default="")
    parser.add_argument("--dev_set_path", type=str, default="")
    parser.add_argument("--test_set", type=str, default="")
    parser.add_argument("--test_set_path", type=str, default="")

    args = parser.parse_args()
    return args


def read_jsonl(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in tqdm(f)]
    new_data = []
    for d in data:
        new_data.append(
            {
                "text": d["text"],
                "label": d["label"],
            }
        )
    return new_data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC.
    """
    fpr, tpr, _ = roc_curve(x, -score)
    return fpr, tpr, auc(fpr, tpr)


def compute_metrics(prediction, answers, print_result=True):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc = sweep(np.array(prediction), np.array(answers, dtype=bool))

    tpr_5_fpr = tpr[np.where(fpr < 0.05)[0][-1]]

    if print_result:
        print(" AUC %.4f, TPR@5%%FPR of %.4f\n" % (auc, tpr_5_fpr))

    return fpr, tpr, auc, tpr_5_fpr


def evaluate(probe, test_acts, test_data):
    scores = probe.score(test_acts)

    predictions = []
    labels = []
    for i, ex in tqdm(enumerate(test_data)):
        predictions.append(-scores[i].item())
        labels.append(ex["label"])

    fpr, tpr, auc, tpr_5_fpr = compute_metrics(predictions, labels, print_result=False)
    return auc


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "TinyLlama-1.1B" in args.target_model:
        layer_num = 22
    elif "open_llama_13b" in args.target_model:
        layer_num = 40
    else:
        raise NotImplementedError

    train_set = read_jsonl(args.train_set_path)
    dev_set = read_jsonl(args.dev_set_path)
    test_set = read_jsonl(args.test_set_path)

    train_act_dataset = ActDataset(
        train_set, args.train_set, args.target_model, layer_num, device
    )
    dev_act_dataset = ActDataset(
        dev_set, args.dev_set, args.target_model, layer_num, device
    )
    test_act_dataset = ActDataset(
        test_set, args.test_set, args.target_model, layer_num, device
    )

    # select best layer
    dev_auc_list = []
    test_auc_list = []
    for layer in range(layer_num):
        train_acts, train_labels = train_act_dataset.get(layer)
        probe = LRProbe.from_data(train_acts, train_labels, device=device)

        dev_acts, dev_labels = dev_act_dataset.get(layer)
        dev_auc = evaluate(probe, dev_acts, dev_set)
        dev_auc_list.append(dev_auc)

        test_acts, test_labels = test_act_dataset.get(layer)
        test_auc = evaluate(probe, test_acts, test_set)
        test_auc_list.append(test_auc)

    dev_best_layer = dev_auc_list.index(max(dev_auc_list))
    print(f"average dev auc: {sum(dev_auc_list)/len(dev_auc_list):.4f}\n")
    print(f"MAX dev auc: {max(dev_auc_list):.4f} in layer_{dev_best_layer}")
    print(f"   test auc: {test_auc_list[dev_best_layer]:.4f} in layer_{dev_best_layer}")
