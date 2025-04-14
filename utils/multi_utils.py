import os
import torch
import random
import numpy as np
import inspect
import sys

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from datetime import datetime
import os

from config import project_root_dir

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def init_experiment(args):
    args.cuda = torch.cuda.is_available()
    if args.device == 'None':
        args.device = torch.device("cuda:0" if args.cuda else "cpu")
    else:
        args.device = torch.device(args.device if args.cuda else "cpu")
    print(args.gpus)
    # Unique identifier for experiment
    time = datetime.now()
    now = f'{time.day}.{time.month}.{time.year}_time_{time.hour}.{time.minute}.{time.second}'
    # model_name = args.model.replace("/", "-")
    # args.train_dir = f'{args.exp_root}/train/{args.dataset}/{args.method}/{args.backbone}/lr:{args.LR}-split:{args.split_idx}-batch:{args.batch_size}/seed{args.seed}/{now}'

    # for ablations
    args.train_dir = f'{args.exp_root}/train/{args.dataset}/{args.method}/{args.backbone}/lr:{args.LR}-split:{args.split_idx}-batch:{args.batch_size}-transform:{args.transform}-shallow:{args.alphaweight}-final:{args.betaweight}-neg:{args.negweight}/seed{args.seed}/{now}'
    # args.train_dir = f'{args.exp_root}/train/{args.dataset}/{args.method}/{args.backbone}/lr:{args.LR}-split:{args.split_idx}-batch:{args.batch_size}-transform:{args.transform}/seed{args.seed}/{now}'

    if args.dataset == 'cifar-10-100':
        args.train_dir = f'{args.exp_root}/train/{args.dataset}-{args.out_num}/{args.method}/{args.backbone}/lr:{args.LR}-split:{args.split_idx}-batch:{args.batch_size}-transform:{args.transform}-shallow:{args.alphaweight}-final:{args.betaweight}-neg:{args.negweight}/seed{args.seed}/{now}'
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)

    print(f'Experiment saved to: {args.train_dir}')
    print(f'oriweight: {args.oriweight}, clipweight: {args.clipweight}')

    args.chkpt_dir = f'{args.train_dir}/checkpoints'
    args.log_path = f'{args.train_dir}/log.txt'
    os.mkdir(args.chkpt_dir)

    args.writer = SummaryWriter(log_dir=args.train_dir)
    hparam_dict = {}
    for k, v in vars(args).items():
        if isinstance(v, (int, float, str, bool, torch.Tensor)):
            hparam_dict[k] = v
    args.writer.add_hparams(hparam_dict=hparam_dict, metric_dict={})

    return args


def accuracy(output, target, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_default_hyperparameters(args):

    """
    Adjusts args to match parameters used in paper: https://arxiv.org/abs/2110.06207
    """

    hyperparameter_path = os.path.join(project_root_dir, 'paper_hyperparameters.csv')
    df = pd.read_csv(hyperparameter_path)

    df = df.loc[df['Loss'] == args.loss]
    hyperparams = df.loc[df['Dataset'] == args.dataset].values[0][2:]

    # -----------------
    # DATASET / LOSS specific hyperparams
    # -----------------
    args.image_size, args.lr, args.rand_aug_n, args.rand_aug_m, args.label_smoothing, args.batch_size = hyperparams

    if args.dataset in ('cub', 'aircraft', 'scars', 'imagenet'):

        args.model = 'timm_resnet50_pretrained'
        args.resnet50_pretrain = 'places_moco'
        args.feat_dim = 2048

    else:

        args.model = 'classifier32'
        args.feat_dim = 128

    # -----------------
    # Other hyperparameters
    # -----------------
    args.seed = 0
    args.max_epoch = 600
    args.transform = 'rand-augment'

    args.scheduler = 'cosine_warm_restarts_warmup'
    args.num_restarts = 2
    args.weight_decay = 1e-4

    return args

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser("Training")

    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar-10-10', help="")
    parser.add_argument('--loss', type=str, default='ARPLoss', help='For cifar-10-100')

    args = parser.parse_args()

    for dataset in ('mnist', 'svhn', 'cifar-10-10', 'cifar-10-100', 'tinyimagenet'):
        args.dataset = dataset
        args = get_default_hyperparameters(args)
        print(f'{dataset}')
        print(args)