import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import numbers
from tqdm import tqdm
from torchvision.transforms import functional as F1
from sklearn.metrics import confusion_matrix
from algorithm import *
from dataloader import *

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for inputs, labels, pdlables, item in loader:
            x = inputs.cuda().float()
            y = labels.cuda().long()
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

def trainer(trainmodel, img_transform, img_transformte, device, opta, scheduler, total_epoch=200):
    ones = torch.sparse.torch.eye(6)
    ones = ones.to(device)
    bestac = 0.5
    
    train_list = r'D:\Data\Widar3\STIFMD'
    dataset_source = datatrcsie(
        data_list=train_list,
        transform=img_transform
    )

    test_list = r'D:\Data\Widar3\STIFMD'

    dataset_target = datatecsie(
        data_list=test_list,
        transform=img_transformte
    )

    train_loader = torch.utils.data.DataLoader(dataset=dataset_source,batch_size=32,shuffle=True,num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_target,batch_size=32,shuffle=False,num_workers=2)
    lengthtr = len(train_loader)
    lengthte = len(test_loader)
    print('Train set size:', lengthtr)
    print('Validation set size:', lengthte)
    with open("log/acc.txt", "w") as f:
        for round in range(total_epoch):
            trainmodel.train()
            print(f'\n========ROUND {round}========')
            print('====Feature update====')
            loss_list = ['class']
            if round <=1:
                for step in range(2):
                    for inputs, labels, pdlables, item in train_loader:
                        loss_result_dict = trainmodel.update_a(inputs, labels, pdlables, opta)
                    print_row([step]+[loss_result_dict[item]
                                    for item in loss_list], colwidth=15)

            print('====Latent domain characterization====')
            loss_list = ['total', 'dis', 'ent']
            print_row(['epoch']+[item+'_loss' for item in loss_list], colwidth=15)

            for step in range(1):
                for inputs, labels, pdlables, item in train_loader:
                    loss_result_dict = trainmodel.update_d(inputs, labels, pdlables, opta)
                print_row([step]+[loss_result_dict[item]
                                for item in loss_list], colwidth=15)

            Cpd = trainmodel.set_dlabel(train_loader)

            print('====Domain-invariant feature learning====')
            for step in range(1):
                for inputs, labels, pdlables, item in train_loader:
                    step_vals = trainmodel.update(inputs, labels, pdlables, opta, Cpd)

            acc = accuracy(trainmodel, test_loader, None)
            print(acc)
            if acc > bestac:
                bestac = acc
            print(bestac)
            scheduler.step()