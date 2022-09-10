# -*- coding: utf-8 -*-
import copy

import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.wrn import WideResNet
from datasets.utils import build_dataset
from models.utils import build_model
from test_rcd import test_performance,test_performance_knn

if __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from datasets.validation_dataset import validation_split

parser = argparse.ArgumentParser()


parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--dataset', type=str,
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--model', '-m', type=str, default='wrn', help='Choose architecture.')
parser.add_argument('--alg', '-alg', type=str, default='standard', help='Choose architecture.')
parser.add_argument('--loss', type=str, default='ce')
parser.add_argument('--drop_number', type=int, default=1)
parser.add_argument('--sample_ratio', type=float, default=1.)

# Optimization options
parser.add_argument('--iteration', '-i', type=int, default=0, help='Number of iteration to train.')
parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size.')


parser.add_argument('--test_bs', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')

# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.3, type=float, help='dropout probability')

# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./snapshots/', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--seed', type=int, default=1, help='0 = CPU.')

# Acceleration
parser.add_argument('--gpu', type=str, default="0", help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--aux_size', type=int, default=-1)
parser.add_argument('--temp', type=float, default=1.)


parser.add_argument('--validation', action='store_true',
                    help='(default: False)')
parser.add_argument('--load_pretrain_encoder', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

## Test values
parser.add_argument('--T', default=1000., type=float, help='temperature: energy|Odin')
parser.add_argument('--noise', type=float, default=0.0014, help='noise for Odin')
parser.add_argument('--score', type=str, default=0.0014, help='score function',choices=["Odin","gradnorm","mps"])
parser.add_argument('--react', default=False,
                    help='(default: False)')
parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

cudnn.benchmark = True  # fire on all cylinders

if args.gpu is not None:
    if len(args.gpu) == 1:
        device = torch.device('cuda:{}'.format(int(args.gpu)))
    else:
        device = torch.device('cuda:0')
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')


train_data,test_data,test_data_ood, num_classes,input_dim = build_dataset(args.dataset,drop_number=args.drop_number)
test_testood=torch.utils.data.ConcatDataset([test_data,test_data_ood])
def Weighted_Sampling(train_label = None):

    '''
    :param train_label: Input train label (in tensor) to calculate sampling weight
    :return: Pytorch Weighted Sampler
    '''

    unique=np.unique(train_label)
    count=0
    copy_train_label=copy.deepcopy(train_label)

    class_sample_count = torch.tensor(
        [(train_label == t).sum() for t in unique])
    weight={}
    for i in range(len(unique)):
        weight[unique[i]]=1. / class_sample_count[i].float()
    # weight = 1. / class_sample_count.float()
    print(weight)
    samples_weight = torch.tensor([weight[t] for t in train_label])


    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler
args.input_dim=input_dim
# test_data, _ = build_dataset(args.dataset, "test")
print(np.squeeze(train_data.label,axis=1))
if args.validation:
    train_data, val_data = validation_split(train_data, val_share=0.1)
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.test_bs, shuffle=False,
        num_workers=args.prefetch, pin_memory=True, drop_last=True)

# train_loader = torch.utils.data.DataLoader(
#     train_data, batch_size=args.batch_size, shuffle=False,
#     num_workers=args.prefetch, pin_memory=True, drop_last=True,sampler=Weighted_Sampling(np.squeeze(train_data.label,axis=1)))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
test_ood_loader =  torch.utils.data.DataLoader(test_data_ood, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)
test_testood_loader =  torch.utils.data.DataLoader(test_testood, batch_size=args.test_bs, shuffle=True,
                                          num_workers=args.prefetch, pin_memory=True)




if args.exp_name:
    path = args.dataset + '_' + args.model + '_' + args.exp_name + '_' + args.alg
else:
    path = args.dataset + '_' + args.model + '_' + args.alg

from algorithms.utils import create_alg
alg_obj = create_alg(args, device, num_classes, train_loader)


start_epoch = 0
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = args.load + '_epoch_' + str(i) + '.pt'
        if os.path.isfile(model_name):
            alg_obj.net.load_state_dict(torch.load(model_name))
            if len(args.gpu) > 1:
                alg_obj.net = alg_obj.net.module
            print('Model restored! Epoch:', i)
            # start_epoch = i + 1
            break
    # if start_epoch == 0:
    #     assert False, "could not resume"



# Make save directory

save_path = args.save + args.alg+args.loss+"{}/{}".format(args.exp_name,args.dataset)

if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.isdir(save_path):
    raise Exception('%s is not a dir' % save_path)

with open(os.path.join(save_path, path +
                                  '_training_results.csv'), 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_acc(%)\n')
    f.write(str(args))

print('Beginning Training\n')

def eval_performance(net,epoch):
    net.eval()
    args.react = False
    net.set_react(False)
    args.score="msp"
    args.T=1
    args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
    test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+
                                          '_testing_results.csv'),epoch)
    args.score="Odin"
    args.T = 1000
    args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
    test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+
                                          '_testing_results.csv'),epoch)
    args.score="Odinpp"
    args.T = 1000
    args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
    test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+
                                          '_testing_results.csv'),epoch)
    args.score = "Odinpppp"
    args.T = 1000
    args.method_name = args.loss + "_" + args.score + "_" + str(args.react)
    test_performance(net, args, test_loader, test_ood_loader, args.T, args.noise, args.score, 1,
                     os.path.join(save_path, str(args.method_name) +
                                  '_testing_results.csv'), epoch)
    #
    # for score in [0.5,1.5]:
    #     args.score = "OdinS"
    #     args.T = 1000
    #     args.score_sharpe = score
    #     args.method_name = args.loss + "_" + args.score + "_" + str(args.react) + str(args.score_sharpe)
    #     test_performance(net, args, test_loader, test_ood_loader, args.T, args.noise, args.score, 1,
    #                      os.path.join(save_path, str(args.method_name) +
    #                                   '_testing_results.csv'), epoch)

    # args.score="knn_ALL"
    # args.T = 1000
    # reference_this="all"
    # # args.sample_ratio=1
    # args.method_name=args.loss+"_"+args.score+"_"+str(args.react)+str(args.sample_ratio)
    # test_performance_knn(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+'_testing_results.csv'),epoch,train_loader=train_loader,num_classes=num_classes,save_logit= os.path.join(save_path, path +
    #                                 '_epoch_' + str(epoch) + '.npz'),reference=reference_this)
    #
    # args.score="knn_Train"
    # args.T = 1000
    # reference_this="train"
    # # args.sample_ratio=1
    # args.method_name=args.loss+"_"+args.score+"_"+str(args.react)+str(args.sample_ratio)
    # test_performance_knn(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+'_testing_results.csv'),epoch,train_loader=train_loader,num_classes=num_classes,save_logit= os.path.join(save_path, path +
    #                                 '_epoch_' + str(epoch) + '.npz'),reference=reference_this)

    # args.score="knn_Train"
    # args.T = 1000
    # reference_this="test"
    # # args.sample_ratio=1
    # args.method_name=args.loss+"_"+args.score+"_"+str(args.react)+str(args.sample_ratio)
    # test_performance_knn(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+'_testing_results.csv'),epoch,train_loader=train_loader,num_classes=num_classes,save_logit= os.path.join(save_path, path +
    #                                 '_epoch_' + str(epoch) + '.npz'),reference=reference_this)

    args.react=True
    net.set_react(True)
    # args.score="msp"
    # args.T=1
    # args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
    # test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+
    #                                       '_testing_results.csv'),epoch)
    # args.score="Odin"
    # args.T = 1000
    # args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
    # test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+
    #                                       '_testing_results.csv'),epoch)
    # args.score="Odinpp"
    # args.T = 1000
    # args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
    # test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,1,os.path.join(save_path, str(args.method_name)+
    #                                       '_testing_results.csv'),epoch)
    # args.score = "Odinpppp"
    # args.T = 1000
    # args.method_name = args.loss + "_" + args.score + "_" + str(args.react)
    # test_performance(net, args, test_loader, test_ood_loader, args.T, args.noise, args.score, 1,
    #                  os.path.join(save_path, str(args.method_name) +
    #                               '_testing_results.csv'), epoch)
    #
    # for score in [0.5,1.5]:
    #     args.score = "OdinS"
    #     args.T = 1000
    #     args.score_sharpe = score
    #     args.method_name = args.loss + "_" + args.score + "_" + str(args.react) + str(args.score_sharpe)
    #     test_performance(net, args, test_loader, test_ood_loader, args.T, args.noise, args.score, 1,
    #                      os.path.join(save_path, str(args.method_name) +
    #                                   '_testing_results.csv'), epoch)

    net.train()

best_acc=200
best_save=""
best_weight=[]
# Main loop
for epoch in range(start_epoch, args.epochs):
    state['epoch'] = epoch

    begin_epoch = time.time()
    state['train_loss'] = alg_obj.train(train_loader, epoch)
    state['test_loss'], state['test_accuracy'] = alg_obj.test(test_loader)

    if args.validation:
        state['val_loss'], state['val_accuracy'] = alg_obj.test(val_loader)
        print('val Loss {0:.4f} | val acc {1:.2f}'.format(
            state['val_loss'],
            100. * state['val_accuracy'])
        )



    # Show results
    with open(os.path.join(save_path, path +
                                      '_training_results.csv'), 'a') as f:
        f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
            (epoch + 1),
            time.time() - begin_epoch,
            state['train_loss'],
            state['test_loss'],
            100. * state['test_accuracy']
        ))



    print('Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test acc {4:.2f}'.format(
        (epoch + 1),
        int(time.time() - begin_epoch),
        state['train_loss'],
        state['test_loss'],
        100. * state['test_accuracy'])
    )

    if best_acc<state['test_accuracy']:
        best_save=os.path.join(save_path, path +
                             '_epoch_' + str(epoch - 1) + '.pt')
        net = alg_obj.get_net()
        best_weight=net.state_dict()

    if epoch%10==0 or epoch==args.epochs-1:
        net = alg_obj.get_net()
        eval_performance(net,epoch)
        if len(args.gpu) > 1:
            torch.save(alg_obj.net.module.state_dict(),
                       os.path.join(save_path, path +
                                    '_epoch_' + str(epoch) + '.pt'))
        else:
            # Save model
            torch.save(alg_obj.net.state_dict(),
                       os.path.join(save_path, path +
                                    '_epoch_' + str(epoch) + '.pt'))

        # Let us not waste space and delete the previous model
        # prev_path = os.path.join(save_path, path +
        #                          '_epoch_' + str(epoch - 1) + '.pt')
        # if os.path.exists(prev_path): os.remove(prev_path)
# net=alg_obj.get_net()
# net.load_state_dict(torch.load(best_weight))

# def eval_performance():
#     args.score="msp"
#     args.T=1
#     args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
#     test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,10,os.path.join(save_path, path +"_"+str(args.method_name)+
#                                           '_testing_results.csv'))
#     args.score="Odin"
#     args.T = 1000
#     args.method_name=args.loss+"_"+args.score+"_"+str(args.react)
#     test_performance(net,args,test_loader,test_ood_loader,args.T,args.noise,args.score,10,os.path.join(save_path, path +"_"+str(args.method_name)+
#                                           '_testing_results.csv'))