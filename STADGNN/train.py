import argparse
import time
import gc
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from net import GNNStack
from utils import AverageMeter, accuracy, log_msg, get_default_train_val_test_loader

parser = argparse.ArgumentParser(description='PyTorch UEA Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BasicMotions')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N',
                    help='mini-batch size (default: 16), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--val-batch-size', default=16, type=int, metavar='V',
                    help='validation batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true',
                    default=True, help='use benchmark')
parser.add_argument('--tag', default='date', type=str,
                    help='the tag for identifying the log and model files. Just a string.')


def main():
    args = parser.parse_args()

    args.kern_size = [int(l) for l in args.kern_size.split(",")]

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    main_work(args)


def main_work(args):
    best_acc1 = 0
    # tensorboard文件
    writer = SummaryWriter('D:/UEA/TodyNet_project_classification/test_tensorboard')

    if args.tag == 'date':
        local_date = time.strftime('%m.%d', time.localtime(time.time()))
        args.tag = local_date

    log_file = 'D:/UEA/TodyNet_project_classification/log/{}_gpu{}_{}_{}_exp.txt'.format(args.tag, args.gpu, args.arch, args.dataset)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_loader, val_loader, num_nodes, seq_length, num_classes = get_default_train_val_test_loader(args)
    model = GNNStack(gnn_model_type=args.arch, num_layers=args.num_layers,
                     groups=args.groups, pool_ratio=args.pool_ratio, kern_size=args.kern_size,
                     in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim,
                     seq_len=seq_length, num_nodes=num_nodes, num_classes=num_classes)
    # determine whether GPU or not
    if not torch.cuda.is_available():
        print("Warning! Using CPU!!!")
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        gc.collect()
        torch.cuda.empty_cache()

        model = model.cuda(args.gpu)
        if args.use_benchmark:
            cudnn.benchmark = True
        print('Using cudnn.benchmark.')
    else:
        print("Error! We only have one gpu!!!")
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                              patience=50, verbose=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print('****************************************************')
    print(args.dataset)

    dataset_time = AverageMeter('Time', ':6.3f')

    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    epoches = []

    end = time.time()
    for epoch in range(args.epochs):
        epoches += [epoch]

        acc_train_per, loss_train_per = train(train_loader, model, criterion, optimizer, lr_scheduler, args)

        acc_train += [acc_train_per]
        loss_train += [loss_train_per]

        msg = f'TRAIN, epoch {epoch}, loss {loss_train_per}, acc {acc_train_per}'
        log_msg(msg, log_file)

        acc_val_per, loss_val_per = validate(val_loader, model, criterion, args)

        acc_val += [acc_val_per]
        loss_val += [loss_val_per]

        msg = f'VAL, loss {loss_val_per}, acc {acc_val_per}'
        log_msg(msg, log_file)

        best_acc1 = max(acc_val_per, best_acc1)

        # tensorboard可视化
        writer.add_scalar('FingerMovements-loss', loss_val_per, epoch)
        writer.add_scalar('FingerMovements-acc', acc_val_per, epoch)
    # 关闭tensorboard
    writer.close()
    dataset_time.update(time.time() - end)

    msg = f'\n\n * BEST_ACC: {best_acc1}\n * TIME: {dataset_time}\n'
    log_msg(msg, log_file)

    print(f' * best_acc1: {best_acc1}')
    print(f' * time: {dataset_time}')
    print('****************************************************')
    del model  
    gc.collect() 


def train(train_loader, model, criterion, optimizer, lr_scheduler, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc', ':6.2f')

    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for count, (data, label) in enumerate(train_loader):
        data = data.to(device).type(torch.float)
        label = label.to(device).type(torch.long)
        output = model(data)
        loss = criterion(output, label)
        acc1 = accuracy(output, label, topk=(1, 1))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1[0], data.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    lr_scheduler.step(top1.avg)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for count, (data, label) in enumerate(val_loader):
            data = data.to(device).type(torch.float)
            label = label.to(device).type(torch.long)
            output = model(data)
            loss = criterion(output, label)
            acc1 = accuracy(output, label, topk=(1, 1))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0], data.size(0))
    return top1.avg, losses.avg


if __name__ == '__main__':
    main()