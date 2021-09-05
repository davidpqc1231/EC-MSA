from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import os
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data_loader
import metric
import network
import model

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--max_iteration', type=int, default=15000,
                    help='how many iterations')
parser.add_argument('--class_num', type=int, default=65,
                    help='number of classes (default: 65)')


parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum (default: 0.01)')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='l2 decay (default: 5e-4)')


parser.add_argument('--cuda', action='store_true', default=False,
                    help='ables or disables CUDA training')
parser.add_argument('--seed', type=int, default=8,
                    help='seed for randomness')
parser.add_argument('--interval', type=int, default=10,
                    help='interval for making records')


parser.add_argument('--root_path', type=str, default='/input/datasets/',
                    help='source only or not') # you can use your own path here
parser.add_argument('--source1', type=str, default='Clipart',
                    help='source1 dataset')
parser.add_argument('--source2', type=str, default='Product',
                    help='source2 dataset')
parser.add_argument('--source3', type=str, default='RealWorld',
                    help='source3 dataset')
parser.add_argument('--target', type=str, default='Art',
                    help='target dataset')

args = parser.parse_args()

batch_size = args.batch_size
iteration = args.max_iteration
class_num = args.class_num
LEARNING_RATE = args.lr
momentum = args.momentum
l2_decay = args.l2_decay

cuda = args.cuda
seed = args.seed
log_interval = args.interval

root_path = args.root_path
source1_name = args.source1
source2_name = args.source2
source3_name = args.source3
target_name = args.target
start = iteration // 2



torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {'num_workers': 8}

source1_loader = data_loader.load_training(root_path, source1_name, batch_size, kwargs)
source2_loader = data_loader.load_training(root_path, source2_name, batch_size, kwargs)
source3_loader = data_loader.load_training(root_path, source3_name, batch_size, kwargs)
target_train_loader = data_loader.load_training(root_path, target_name, batch_size, kwargs)
target_test_loader = data_loader.load_testing(root_path, target_name, batch_size, kwargs)


# Train
def train(model):
    source1_iter = iter(source1_loader)
    source2_iter = iter(source2_loader)
    source3_iter = iter(source3_loader)
    target_iter = iter(target_train_loader)
    correct = 0
    LEARNING_RATE = 0.01
    optimizer = torch.optim.SGD([
            {'params': model.sharedNet.parameters()},
            {'params': model.cls_fc_son1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc_son3.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet2.parameters(), 'lr': LEARNING_RATE},
            {'params': model.sonnet3.parameters(), 'lr': LEARNING_RATE},
        ], lr=LEARNING_RATE / 10, momentum=momentum, weight_decay=l2_decay)

    for i in range(1, iteration + 1):
        model.train()
        
        try:
            source_data, source_label = source1_iter.next()
        except Exception as err:
            source1_iter = iter(source1_loader)
            source_data, source_label = source1_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(start, source_data, target_data, source_label, mark=1, itera=i)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration) )) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train source1 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        try:
            source_data, source_label = source2_iter.next()
        except Exception as err:
            source2_iter = iter(source2_loader)
            source_data, source_label = source2_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)

        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(start, source_data, target_data, source_label, mark=2, itera=i)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source2 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))


        try:
            source_data, source_label = source3_iter.next()
        except Exception as err:
            source3_iter = iter(source3_loader)
            source_data, source_label = source3_iter.next()
        try:
            target_data, __ = target_iter.next()
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, __ = target_iter.next()
        if cuda:
            source_data, source_label = source_data.cuda(), source_label.cuda()
            target_data = target_data.cuda()
        source_data, source_label = Variable(source_data), Variable(source_label)
        target_data = Variable(target_data)
        
        optimizer.zero_grad()

        cls_loss, mmd_loss, l1_loss = model(start, source_data, target_data, source_label, mark=3, itera=i)
        gamma = 2 / (1 + math.exp(-10 * (i) / (iteration))) - 1
        loss = cls_loss + gamma * (mmd_loss + l1_loss)
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print(
                'Train source3 iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tmmd_Loss: {:.6f}\tl1_Loss: {:.6f}'.format(
                    i, 100. * i / iteration, loss.item(), cls_loss.item(), mmd_loss.item(), l1_loss.item()))

        if i % (log_interval * 20) == 0:
            t_correct = test(model)
            if t_correct > correct:
                correct = t_correct
                best_model = model
                torch.save(best_model.state_dict(), '/input/'+target_name+'.pt') # you can choose your own location and name
            print(source1_name, source2_name, source3_name, "to", target_name, "%s max correct:" % target_name, correct.item(), "\n")


#Test
def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    with torch.no_grad():
        for data, target in target_test_loader:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            pred1, pred2, pred3 = model(data)

            pred1 = torch.nn.functional.softmax(pred1, dim=1)
            pred2 = torch.nn.functional.softmax(pred2, dim=1)
            pred3 = torch.nn.functional.softmax(pred3, dim=1)

            pred = (pred1 + pred2 + pred3) / 3
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()  # sum up batch loss
            pred = pred.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred1.data.max(1)[1]  # get the index of the max log-probability
            correct1 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred2.data.max(1)[1]  # get the index of the max log-probability
            correct2 += pred.eq(target.data.view_as(pred)).cpu().sum()
            pred = pred3.data.max(1)[1]  # get the index of the max log-probability
            correct3 += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(target_test_loader.dataset)
        print(target_name, '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(target_test_loader.dataset),
            100. * correct / len(target_test_loader.dataset)))
        print('\nsource1 accnum {}, source2 accnum {}，source3 accnum {}'.format(correct1, correct2, correct3))
    return correct


sample = model.ECMSA(num_classes=class_num)
print(sample)


if cuda:
    sample.cuda()

train(sample)
