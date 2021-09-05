from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

def update_centroid(s_centroid, t_centroid, s_feature, t_feature, y_s, y_t):
    n, d = s_feature.shape

    # get labels
    s_labels, t_labels = y_s, torch.max(y_t, 1)[1]

    # image number in each class
    ones = torch.ones_like(s_labels, dtype=torch.float)
    zeros = torch.zeros(65)
    if cuda:
        zeros = zeros.cuda()
        ones = ones.cuda()
        s_labels = s_labels.cuda()
        t_labels = t_labels.cuda()
        s_centroid = s_centroid.cuda()
        t_centroid = t_centroid.cuda()
    
    s_n_classes = zeros.scatter_add(0, s_labels, ones)
    t_n_classes = zeros.scatter_add(0, t_labels, ones)

    # image number cannot be 0, when calculating centroids
    ones = torch.ones_like(s_n_classes)
    s_n_classes = torch.max(s_n_classes, ones)
    t_n_classes = torch.max(t_n_classes, ones)

    # calculating centroids, sum and divide
    zeros = torch.zeros(65, 256)
    if cuda:
        zeros = zeros.cuda()
        s_feature = s_feature.cuda()
        t_feature = t_feature.cuda()
    s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(256, 1), 1, 0), s_feature)
    t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(256, 1), 1, 0), t_feature)
    current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(65, 1))
    current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(65, 1))

    # Moving Centroid
    decay = 0.3
    new_s_centroid = (1-decay) * s_centroid + decay * current_s_centroid
    new_t_centroid = (1-decay) * t_centroid + decay * current_t_centroid

    return new_s_centroid, new_t_centroid


def distill(inputs, model):
    
    # threshold for target distilling
    tau_p = 0.5

    outputs = model(inputs)
    out_prob = F.softmax(outputs, dim=1) 
    max_value, max_idx = torch.max(out_prob, dim=1)

    #selecting positive pseudo-labels
    selected_idx = max_value>tau_p

    select_inputs = inputs[selected_idx]
    pseudo = max_idx[selected_idx].long()
    
    return select_inputs, selected_idx