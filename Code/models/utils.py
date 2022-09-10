from models.wrn import WideResNet,classifier
import torch
from torchvision.models import densenet121
import numpy as np

def build_model( model_type,num_classes, device, args):
    net = classifier(args.input_dim, num_classes,args)
    net.to(device)
    if args.gpu is not None and len(args.gpu) > 1:
        gpu_list = [int(s) for s in args.gpu.split(',')]
        net = torch.nn.DataParallel(net, device_ids=gpu_list)
    return net