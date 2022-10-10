import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

def get_socre(outputs):
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs

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
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

    return nnOutputs






def SEQ(inputs, outputs, model, temper=1000, noiseMagnitude1=0.0014, device=torch.device('cpu')):
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

    # print(torch.max(tempInputs.grad.data))
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

def SIM(inputs, outputs, model, temper=1000, noiseMagnitude1=0.0014, device=torch.device('cpu')):
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

    nnOutputs=score_ori+(nnOutputs_ori-score_ori)+score_ori-nnOutputs_novel

    return nnOutputs


# score_via_name={"msp":,"odin":ODIN,"seq":SEQ,"sim":SIM}
def get_ood_scores(loader, net,score_function="sim",T=1000,noise=0.0014,device=torch.device('cpu')):
    _score = []
    _right_score = []
    _wrong_score = []
    _label = []
    _patient=[]
    to_np = lambda x: x.data.cpu().numpy()
    concat = lambda x: np.concatenate(x, axis=0)

    net.eval()
    for step, (batch_x, batch_y) in enumerate(loader):
        data, target = batch_x, batch_y


        # if batch_idx >= ood_num_examples // bs and in_dist is False:
        #     break
        data = data.to(device)
        data = Variable(data, requires_grad=True)

        output = net(data)

        if score_function=="sim":
            odin_score = SIM(data, output, net, temper=T, noiseMagnitude1=noise, device=device)
        elif score_function=="seq":
            odin_score = SIM(data, output, net, temper=T, noiseMagnitude1=noise, device=device)
        elif score_function=="odin":
            odin_score = ODIN(data, output, net, temper=T, noiseMagnitude1=noise, device=device)
        elif score_function=="msp":
            odin_score=to_np(F.softmax(output, dim=1))
        else:
            raise ValueError ("No Such score function {}".format(score_function))
        _score.append(np.max(odin_score, 1))


    return concat(_score).copy()



