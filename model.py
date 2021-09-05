from __future__ import print_function
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch

import metric
import network


class ECMSA(nn.Module):

    def __init__(self, num_classes=65):
        super(ECMSA, self).__init__()
        self.sharedNet = network.resnet50(True)
        self.sonnet1 = network.ADDneck(2048, 256)
        self.sonnet2 = network.ADDneck(2048, 256)
        self.sonnet3 = network.ADDneck(2048, 256)
        self.cls_fc_son1 = nn.Linear(256, num_classes)
        self.cls_fc_son2 = nn.Linear(256, num_classes)
        self.cls_fc_son3 = nn.Linear(256, num_classes)
        self.s_centroid1 = torch.zeros(num_classes, 256)
        self.t_centroid1 = torch.zeros(num_classes, 256)
        self.s_centroid2 = torch.zeros(num_classes, 256)
        self.t_centroid2 = torch.zeros(num_classes, 256)
        self.s_centroid3 = torch.zeros(num_classes, 256)
        self.t_centroid3 = torch.zeros(num_classes, 256)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def forward(self, start, data_src, data_tgt = 0, label_src = 0, mark = 1, itera = 1):
        mmd_loss = 0
        if self.training == True:
            data_src = self.sharedNet(data_src)
            data_tgt = self.sharedNet(data_tgt)

            feat_tgt_son1 = self.sonnet1(data_tgt, domain_label='target')
            feat_tgt_son1 = self.avgpool(feat_tgt_son1)
            feat_tgt_son1 = feat_tgt_son1.view(feat_tgt_son1.size(0), -1)
            prob_tgt_son1 = self.cls_fc_son1(feat_tgt_son1)

            feat_tgt_son2 = self.sonnet2(data_tgt, domain_label='target')
            feat_tgt_son2 = self.avgpool(feat_tgt_son2)
            feat_tgt_son2 = feat_tgt_son2.view(feat_tgt_son2.size(0), -1)
            prob_tgt_son2 = self.cls_fc_son2(feat_tgt_son2)

            feat_tgt_son3 = self.sonnet3(data_tgt, domain_label='target')
            feat_tgt_son3 = self.avgpool(feat_tgt_son3)
            feat_tgt_son3 = feat_tgt_son3.view(feat_tgt_son3.size(0), -1)
            prob_tgt_son3 = self.cls_fc_son3(feat_tgt_son3)

            if mark == 1:

                feat_src = self.sonnet1(data_src, domain_label='source')
                feat_src = self.avgpool(feat_src)
                feat_src = feat_src.view(feat_src.size(0), -1)
                
                if itera >= start:
                    feat_tgt, idx = metric.distill(feat_tgt_son1, self.cls_fc_son1)
                else:
                    feat_tgt = feat_tgt_son1
                new_s_centroid, new_t_centroid = metric.update_centroid(self.s_centroid1, self.t_centroid1, feat_src, feat_tgt, label_src, self.cls_fc_son1(feat_tgt))
                
                mmd_loss += metric.mmd(new_s_centroid, new_t_centroid)
                self.s_centroid1 = new_s_centroid.detach()
                self.t_centroid1 = new_t_centroid.detach()
                
                lam = np.random.beta(0.2, 0.2)
                if itera >= start:
                    prob_tgt_son1 = prob_tgt_son1[idx]
                    prob_tgt_son2 = prob_tgt_son2[idx]
                    prob_tgt_son3 = prob_tgt_son3[idx]

                l1_loss = torch.mean(torch.abs(F.softmax(prob_tgt_son1, dim=1) - F.softmax(prob_tgt_son2, dim=1)) )
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_son1, dim=1) - F.softmax(prob_tgt_son3, dim=1)) )
                
                l1_loss /= 2
                
                if itera < start:
                    batch_size = data_tgt.size()[0]
                    index = torch.randperm(batch_size)
                    mixed_tgt = lam * data_tgt + (1 - lam) * data_tgt[index, :]
                else:
                    batch_size = data_tgt[idx].size()[0]
                    index = torch.randperm(batch_size)
                    mixed_tgt = lam * data_tgt[idx] + (1 - lam) * data_tgt[idx][index, :]
                    
                mix_tgt_son1 = self.sonnet1(data_tgt, domain_label='target')
                mix_tgt_son1 = self.avgpool(mix_tgt_son1)
                mix_tgt_son1 = mix_tgt_son1.view(mix_tgt_son1.size(0), -1)
                prob_tgt_mix1 = self.cls_fc_son1(mix_tgt_son1)
                
                mix_tgt_son2 = self.sonnet2(data_tgt, domain_label='target')
                mix_tgt_son2 = self.avgpool(mix_tgt_son2)
                mix_tgt_son2 = mix_tgt_son2.view(mix_tgt_son2.size(0), -1)
                prob_tgt_mix2 = self.cls_fc_son2(mix_tgt_son2)
                
                mix_tgt_son3 = self.sonnet3(data_tgt, domain_label='target')
                mix_tgt_son3 = self.avgpool(mix_tgt_son3)
                mix_tgt_son3 = mix_tgt_son3.view(mix_tgt_son3.size(0), -1)
                prob_tgt_mix3 = self.cls_fc_son3(mix_tgt_son3)
                
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_mix1, dim=1) - F.softmax(prob_tgt_mix2, dim=1))) / 2
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_mix1, dim=1) - F.softmax(prob_tgt_mix3, dim=1))) / 2
                
                
                pred_src = self.cls_fc_son1(feat_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

            if mark == 2:

                feat_src = self.sonnet2(data_src, domain_label='source')
                feat_src = self.avgpool(feat_src)
                feat_src = feat_src.view(feat_src.size(0), -1)
                
                if itera >= start:
                    feat_tgt, idx = metric.distill(feat_tgt_son2, self.cls_fc_son2)
                else:
                    feat_tgt = feat_tgt_son2
                new_s_centroid, new_t_centroid = metric.update_centroid(self.s_centroid2, self.t_centroid2, feat_src, feat_tgt, label_src, self.cls_fc_son2(feat_tgt))
                
                mmd_loss += metric.mmd(new_s_centroid, new_t_centroid)
                self.s_centroid2 = new_s_centroid.detach()
                self.t_centroid2 = new_t_centroid.detach()
                
                lam = np.random.beta(0.2, 0.2)
                if itera >= start:
                    prob_tgt_son1 = prob_tgt_son1[idx]
                    prob_tgt_son2 = prob_tgt_son2[idx]
                    prob_tgt_son3 = prob_tgt_son3[idx]
                    
                #mmd_loss += mmd(data_src, data_tgt_son1)

                l1_loss = torch.mean(torch.abs(F.softmax(prob_tgt_son2, dim=1) - F.softmax(prob_tgt_son1, dim=1)) )
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_son2, dim=1) - F.softmax(prob_tgt_son3, dim=1)) )
                
                l1_loss /= 2
                
                if itera < start:
                    batch_size = data_tgt.size()[0]
                    index = torch.randperm(batch_size)
                    mixed_tgt = lam * data_tgt + (1 - lam) * data_tgt[index, :]
                else:
                    batch_size = data_tgt[idx].size()[0]
                    index = torch.randperm(batch_size)
                    mixed_tgt = lam * data_tgt[idx] + (1 - lam) * data_tgt[idx][index, :]
                    
                mix_tgt_son1 = self.sonnet1(data_tgt, domain_label='target')
                mix_tgt_son1 = self.avgpool(mix_tgt_son1)
                mix_tgt_son1 = mix_tgt_son1.view(mix_tgt_son1.size(0), -1)
                prob_tgt_mix1 = self.cls_fc_son1(mix_tgt_son1)
                
                mix_tgt_son2 = self.sonnet2(data_tgt, domain_label='target')
                mix_tgt_son2 = self.avgpool(mix_tgt_son2)
                mix_tgt_son2 = mix_tgt_son2.view(mix_tgt_son2.size(0), -1)
                prob_tgt_mix2 = self.cls_fc_son2(mix_tgt_son2)
                
                mix_tgt_son3 = self.sonnet3(data_tgt, domain_label='target')
                mix_tgt_son3 = self.avgpool(mix_tgt_son3)
                mix_tgt_son3 = mix_tgt_son3.view(mix_tgt_son3.size(0), -1)
                prob_tgt_mix3 = self.cls_fc_son3(mix_tgt_son3)
                
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_mix2, dim=1) - F.softmax(prob_tgt_mix1, dim=1))) / 2
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_mix2, dim=1) - F.softmax(prob_tgt_mix3, dim=1))) / 2
                
                pred_src = self.cls_fc_son2(feat_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

            if mark == 3:

                feat_src = self.sonnet3(data_src, domain_label='source')
                feat_src = self.avgpool(feat_src)
                feat_src = feat_src.view(feat_src.size(0), -1)
                
                if itera >= start:
                    feat_tgt, idx = metric.distill(feat_tgt_son3, self.cls_fc_son3)
                else:
                    feat_tgt = feat_tgt_son3
                new_s_centroid, new_t_centroid = metric.update_centroid(self.s_centroid3, self.t_centroid3, feat_src, feat_tgt, label_src, self.cls_fc_son3(feat_tgt))
                
                mmd_loss += metric.mmd(new_s_centroid, new_t_centroid)
                self.s_centroid3 = new_s_centroid.detach()
                self.t_centroid3 = new_t_centroid.detach()
                
                lam = np.random.beta(0.2, 0.2)
                if itera >= start:
                    prob_tgt_son1 = prob_tgt_son1[idx]
                    prob_tgt_son2 = prob_tgt_son2[idx]
                    prob_tgt_son3 = prob_tgt_son3[idx]
                    
                #mmd_loss += mmd(data_src, data_tgt_son1)

                l1_loss = torch.mean(torch.abs(F.softmax(prob_tgt_son3, dim=1) - F.softmax(prob_tgt_son1, dim=1)) )
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_son3, dim=1) - F.softmax(prob_tgt_son2, dim=1)) )
                
                l1_loss /= 2
                
                if itera < start:
                    batch_size = data_tgt.size()[0]
                    index = torch.randperm(batch_size)
                    mixed_tgt = lam * data_tgt + (1 - lam) * data_tgt[index, :]
                else:
                    batch_size = data_tgt[idx].size()[0]
                    index = torch.randperm(batch_size)
                    mixed_tgt = lam * data_tgt[idx] + (1 - lam) * data_tgt[idx][index, :]
                    
                mix_tgt_son1 = self.sonnet1(data_tgt, domain_label='target')
                mix_tgt_son1 = self.avgpool(mix_tgt_son1)
                mix_tgt_son1 = mix_tgt_son1.view(mix_tgt_son1.size(0), -1)
                prob_tgt_mix1 = self.cls_fc_son1(mix_tgt_son1)
                
                mix_tgt_son2 = self.sonnet2(data_tgt, domain_label='target')
                mix_tgt_son2 = self.avgpool(mix_tgt_son2)
                mix_tgt_son2 = mix_tgt_son2.view(mix_tgt_son2.size(0), -1)
                prob_tgt_mix2 = self.cls_fc_son2(mix_tgt_son2)
                
                mix_tgt_son3 = self.sonnet3(data_tgt, domain_label='target')
                mix_tgt_son3 = self.avgpool(mix_tgt_son3)
                mix_tgt_son3 = mix_tgt_son3.view(mix_tgt_son3.size(0), -1)
                prob_tgt_mix3 = self.cls_fc_son3(mix_tgt_son3)
                
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_mix3, dim=1) - F.softmax(prob_tgt_mix1, dim=1))) / 2
                l1_loss += torch.mean(torch.abs(F.softmax(prob_tgt_mix3, dim=1) - F.softmax(prob_tgt_mix2, dim=1))) / 2
                
                pred_src = self.cls_fc_son3(feat_src)

                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)

                return cls_loss, mmd_loss, l1_loss

        else:
            data = self.sharedNet(data_src)

            fea_son1 = self.sonnet1(data, domain_label='target')
            fea_son1 = self.avgpool(fea_son1)
            fea_son1 = fea_son1.view(fea_son1.size(0), -1)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data, domain_label='target')
            fea_son2 = self.avgpool(fea_son2)
            fea_son2 = fea_son2.view(fea_son2.size(0), -1)
            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data, domain_label='target')
            fea_son3 = self.avgpool(fea_son3)
            fea_son3 = fea_son3.view(fea_son3.size(0), -1)
            pred3 = self.cls_fc_son3(fea_son3)

            return pred1, pred2, pred3
