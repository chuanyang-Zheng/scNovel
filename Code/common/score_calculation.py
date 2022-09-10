from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy import misc

to_np = lambda x: x.data.cpu().numpy()
concat = lambda x: np.concatenate(x, axis=0)

def get_ood_scores_odin(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad = True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODIN(data, output,net, T, noise, device)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return concat(_score).copy()


def ODIN(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    
    # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    #gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    #gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    #gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)



    return nnOutputs


def get_ood_scores_odinMin(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODINMin(data, output, net, T, noise, device)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return concat(_score).copy()


def ODINMin(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.max(outputs.data.cpu().numpy(), axis=1)
    minIndexTemp = np.argmin(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    nnOutputs_max = outputs.data.cpu()
    nnOutputs_max = nnOutputs_max.numpy()
    nnOutputs_max = nnOutputs_max - np.max(nnOutputs_max, axis=1, keepdims=True)
    nnOutputs_max = np.exp(nnOutputs_max) / np.sum(np.exp(nnOutputs_max), axis=1, keepdims=True)
    nnOutputs_max_value=np.max(nnOutputs_max,axis=1,keepdims=True)
    nnOutputs_min_value = np.min(nnOutputs_max, axis=1, keepdims=True)

    labels = Variable(torch.LongTensor(minIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2


    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    nnOutputs_min_value_oodn=np.take_along_axis(nnOutputs, np.expand_dims(minIndexTemp, axis=-1), axis=-1)

    return nnOutputs_max_value+nnOutputs_min_value_oodn-nnOutputs_min_value



def get_ood_scores_odinpp(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODINpp(data, output, net, T, noise, device)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return concat(_score).copy()

def get_socre(outputs):
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

def ODINpp(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    score_ori=outputs
    score_ori=get_socre(score_ori/temper)
    # print("Raw {}".format(np.max(score_ori, 1)))

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    # gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    # gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    # gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_ori = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Ori {}".format(np.max(nnOutputs_ori, 1)))



    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_novel = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Novel {}".format(np.max(nnOutputs_novel, 1)))

    nnOutputs=score_ori+(nnOutputs_ori-score_ori)*0.5+score_ori-nnOutputs_novel

    return nnOutputs






def get_ood_scores_odinpppp(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODINpppp(data, output, net, T, noise, device)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return concat(_score).copy()





def ODINpppp(inputs, outputs, model, temper, noiseMagnitude1, device):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    score_ori=outputs
    score_ori=get_socre(score_ori/temper)
    # print("Raw {}".format(np.max(score_ori, 1)))

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    # gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    # gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    # gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    tempInputs =nn.Parameter(tempInputs)
    outputs = model(tempInputs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_ori = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Ori {}".format(np.max(nnOutputs_ori, 1)))



    # Adding small perturbations to images

    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(tempInputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # print(torch.max(tempInputs.grad.data))

    tempInputs = torch.add(tempInputs.data, noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_novel = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Novel {}".format(np.max(nnOutputs_novel, 1)))

    nnOutputs=nnOutputs_ori+nnOutputs_ori-nnOutputs_novel

    return nnOutputs

def get_ood_scores_odinS(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False,args=None):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODINS(data, output, net, T, noise, device,args)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return concat(_score).copy()



def ODINS(inputs, outputs, model, temper, noiseMagnitude1, device,args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    score_ori=outputs
    score_ori=get_socre(score_ori/temper)
    # print("Raw {}".format(np.max(score_ori, 1)))

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # gradient[:,0] = (gradient[:,0] )/(63.0/255.0)
    # gradient[:,1] = (gradient[:,1] )/(62.1/255.0)
    # gradient[:,2] = (gradient[:,2] )/(66.7/255.0)
    # gradient.index_copy_(1, torch.LongTensor([0]).to(device), gradient.index_select(1, torch.LongTensor([0]).to(device)) / (63.0/255.0))
    # gradient.index_copy_(1, torch.LongTensor([1]).to(device), gradient.index_select(1, torch.LongTensor([1]).to(device)) / (62.1/255.0))
    # gradient.index_copy_(1, torch.LongTensor([2]).to(device), gradient.index_select(1, torch.LongTensor([2]).to(device)) / (66.7/255.0))

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_ori = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Ori {}".format(np.max(nnOutputs_ori, 1)))



    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_novel = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Novel {}".format(np.max(nnOutputs_novel, 1)))

    # print("Max "+str(np.max((nnOutputs_ori-score_ori))))
    # print("Ratio " + str(np.max((nnOutputs_ori - score_ori)/score_ori)))
    # if (np.take_along_axis(nnOutputs_ori-score_ori, np.expand_dims(maxIndexTemp,axis=1), axis=1)).any()<0:
    #     print(np.min(np.take_along_axis(nnOutputs_ori-score_ori, np.expand_dims(maxIndexTemp,axis=1), axis=1)))

    nnOutputs=score_ori+(np.maximum((nnOutputs_ori-score_ori),0))

    return nnOutputs




def get_ood_scores_odinMM(loader, net, bs, ood_num_examples, T, noise, device, in_dist=False,args=None):
    _score = []
    _right_score = []
    _wrong_score = []

    net.eval()
    for batch_idx, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad=True)

        output = net(data)
        smax = to_np(F.softmax(output, dim=1))

        odin_score = ODINS(data, output, net, T, noise, device,args)
        _score.append(-np.max(odin_score, 1))

        if in_dist:
            preds = np.argmax(smax, axis=1)
            targets = target.numpy().squeeze()
            right_indices = preds == targets
            wrong_indices = np.invert(right_indices)

            _right_score.append(-np.max(smax[right_indices], axis=1))
            _wrong_score.append(-np.max(smax[wrong_indices], axis=1))

    if in_dist:
        return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy()
    else:

        return concat(_score).copy()



def ODINSMM(inputs, outputs, model, temper, noiseMagnitude1, device,args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    criterion = nn.CrossEntropyLoss()

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
    minIndexTemp = np.argmin(outputs.data.cpu().numpy(), axis=1)
    score_ori=outputs
    score_ori=get_socre(score_ori/temper)
    # print("Raw {}".format(np.max(score_ori, 1)))

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_ori = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    # print("Ori {}".format(np.max(nnOutputs_ori, 1)))

    labels = Variable(torch.LongTensor(minIndexTemp).to(device))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs_min = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)





    return nnOutputs_ori+nnOutputs_min


def sample_estimator(model, num_classes,  train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)

    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []

    for j in range(num_classes):
        list_features.append(0)

    idx = 0
    with torch.no_grad():
        for i, examples in enumerate(train_loader):
            data, target = examples[0], examples[1]
            # print()
            idx += 1
            # print(idx)
            data = Variable(data.cuda())

            target = target.cuda()

            out_features = model.forward_feature(data)


            for i in range(data.size(0)):
                # px = 0
                # for j in range(num_classes):
                #     if target[i][j] == 0:
                #         continue
                    label = target[i]
                    if num_sample_per_class[label] == 0:
                        # out_count = 0
                        # for out in out_features:
                        #     list_features[out_count][label] = out[i].view(1, -1)
                        #     out_count += 1

                        list_features[label] = out_features[i].view(1, -1)
                    else:
                        # out_count = 0
                        # for out in out_features:
                        #     list_features[out_count][label] \
                        #         = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                        #     out_count += 1

                        list_features[label] = torch.cat((list_features[label],
                                                          out_features[i].view(1, -1)), 0)
                    num_sample_per_class[label] += 1


    num_feature = 32
    temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
    for j in range(num_classes):
        temp_list[j] = torch.mean(list_features[j], 0)
    sample_class_mean = temp_list


    X = 0
    for i in range(num_classes):
        if i == 0:
            X = list_features[i] - sample_class_mean[i]
        else:
            X = torch.cat((X, list_features[i] - sample_class_mean[i]), 0)
    # find inverse
    group_lasso.fit(X.cpu().numpy())
    temp_precision = group_lasso.precision_
    temp_precision = torch.from_numpy(temp_precision).float().cuda()
    precision = temp_precision

    return sample_class_mean, precision

def get_Mahalanobis_score(model,  loader, pack, noise, num_classes, method="sum"):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    sample_mean, precision = pack
    model.eval()

    Mahalanobis = []
    for i, examples in enumerate(loader):
        data, target = examples[0], examples[1]
        data = Variable(data.cuda(), requires_grad=True)

        # out_features = model_penultimate_layer(model, clsfier, data)
        out_features = model.forward_feature(data)
        # out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        # out_features = torch.mean(out_features, 2) # size(batch_size, F)

        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1, 1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1, 1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean.index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5 * torch.mm(torch.mm(zero_f, Variable(precision)), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient = torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # gradient.index_copy_(1, torch.LongTensor([0]).cuda(),
        #                      gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.229))
        # gradient.index_copy_(1, torch.LongTensor([1]).cuda(),
        #                      gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.224))
        # gradient.index_copy_(1, torch.LongTensor([2]).cuda(),
        #                      gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.225))
        tempInputs = torch.add(data.data, gradient, alpha=-noise)

        #noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        with torch.no_grad():
            # noise_out_features = model_penultimate_layer(model, clsfier, Variable(tempInputs))
            noise_out_features = model.forward_feature(Variable(tempInputs))
            # noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
            # noise_out_features = torch.mean(noise_out_features, 2)
            noise_gaussian_score = 0
            for i in range(num_classes):
                batch_sample_mean = sample_mean[i]
                zero_f = noise_out_features.data - batch_sample_mean
                term_gau = -0.5 * torch.mm(torch.mm(zero_f, precision), zero_f.t()).diag()
                if i == 0:
                    noise_gaussian_score = term_gau.view(-1, 1)
                else:
                    noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1, 1)), 1)
        # noise_gaussion_score size([batch_size, n_classes])

        if method == "max":
            noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        elif method == "sum":
            noise_gaussian_score = torch.sum(noise_gaussian_score, dim=1)

        Mahalanobis.extend(to_np(noise_gaussian_score))

    return Mahalanobis
