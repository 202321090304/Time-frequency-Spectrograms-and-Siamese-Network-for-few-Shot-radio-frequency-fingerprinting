from __future__ import print_function
import os
import sys
import argparse
import time
import math
import torch
import torch.backends.cudnn as cudnn
from rf_dataset import SPDataset
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer
from networks.resnet_big import CustomCNN, CustomCNNmini, SupConResNet, LinearClassifier, sp_LinearClassifier, sp_MLPClassifier
from torchvision import transforms

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test_batch_size')
    parser.add_argument('--num_workers', type=int, default=16, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')

    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    parser.add_argument('--model', type=str, default='CustomCNN')
    parser.add_argument('--dataset', type=str, default='sp', choices=['rf', 'sp'], help='dataset')

    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
    parser.add_argument('--warm', action='store_true', help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='', help='path to pre-trained model')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--val_data_folder', type=str, default=None, help='path to custom val dataset')

    parser.add_argument('--classifier', type=str, default='linear')
    parser.add_argument('--split_ratio', type=float, default=1)
    parser.add_argument('--n_cls', type=int, default=5)
    
    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = [int(it) for it in iterations]

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay, opt.batch_size)
    
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    if opt.dataset == 'rf':
        opt.n_cls = 4
    elif opt.dataset == 'sp':
        opt.n_cls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    
    return opt

def set_loader(opt):
    normalize = transforms.Normalize(mean=[0], std=[0])  # For 'sp' dataset

    if opt.dataset == 'sp':
        train_transform = transforms.Compose([
            transforms.CenterCrop((500, 500)),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop((500, 500)),
            transforms.ToTensor()
        ])
    else:
        raise ValueError(opt.dataset)

    if opt.dataset == 'sp':
        train_dataset = SPDataset(data_dir=opt.data_folder, transform=train_transform, data_type='test', split_ratio=opt.split_ratio)
        val_dataset = SPDataset(data_dir=opt.val_data_folder, transform=val_transform, data_type='test')
    else:
        raise ValueError(opt.dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_model(opt):
    if opt.dataset == 'sp':
        if opt.model == 'CustomCNN':
            model = CustomCNN()
        elif opt.model == 'CustomCNNmini':
            model = CustomCNNmini()
        else:
            print("Model not found: {}".format(opt.model))
    else:
        model = SupConResNet(name=opt.model)
    
    criterion = torch.nn.CrossEntropyLoss()

    if opt.dataset == 'sp':
        if opt.classifier == 'linear':
            classifier = sp_LinearClassifier(num_classes=opt.n_cls)
        elif opt.classifier == 'MLP':
            classifier = sp_MLPClassifier(num_classes=opt.n_cls)
        else:
            print("Classifier not found: {}".format(opt.classifier))
    else:
        classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion

def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        features = model.encoder(images)
        output = classifier(features)

        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                  'DT {data_time_val:.3f} ({data_time_avg:.3f})\t'
                  'loss {loss_val:.3f} ({loss_avg:.3f})\t'
                  'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                epoch, idx + 1, len(train_loader),
                batch_time_val=batch_time.val.item() if isinstance(batch_time.val, torch.Tensor) else batch_time.val,
                batch_time_avg=batch_time.avg.item() if isinstance(batch_time.avg, torch.Tensor) else batch_time.avg,
                data_time_val=data_time.val.item() if isinstance(data_time.val, torch.Tensor) else data_time.val,
                data_time_avg=data_time.avg.item() if isinstance(data_time.avg, torch.Tensor) else data_time.avg,
                loss_val=losses.val.item() if isinstance(losses.val, torch.Tensor) else losses.val,
                loss_avg=losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg,
                top1_val=top1.val.item() if isinstance(top1.val, torch.Tensor) else top1.val,
                top1_avg=top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg))

            sys.stdout.flush()

    return losses.avg, top1.avg

def validate(val_loader, model, classifier, criterion, opt):
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]

            output = classifier(model.encoder(images))

            loss = criterion(output, labels)

            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'BT {batch_time_val:.3f} ({batch_time_avg:.3f})\t'
                      'loss {loss_val:.3f} ({loss_avg:.3f})\t'
                      'Acc@1 {top1_val:.3f} ({top1_avg:.3f})'.format(
                    idx + 1, len(val_loader),
                    batch_time_val=batch_time.val.item() if isinstance(batch_time.val, torch.Tensor) else batch_time.val,
                    batch_time_avg=batch_time.avg.item() if isinstance(batch_time.avg, torch.Tensor) else batch_time.avg,
                    loss_val=losses.val.item() if isinstance(losses.val, torch.Tensor) else losses.val,
                    loss_avg=losses.avg.item() if isinstance(losses.avg, torch.Tensor) else losses.avg,
                    top1_val=top1.val.item() if isinstance(top1.val, torch.Tensor) else top1.val,
                    top1_avg=top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg))

    return losses.avg, top1.avg

def main():
    opt = parse_option()

    train_loader, val_loader = set_loader(opt)
    model, classifier, criterion = set_model(opt)

    optimizer = set_optimizer(model, classifier, opt)

    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(optimizer, epoch, opt)
        train_loss, train_acc = train(train_loader, model, classifier, criterion, optimizer, epoch, opt)
        print(f"Epoch {epoch}/{opt.epochs}, Train Loss: {train_loss}, Train Accuracy: {train_acc}")

        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        print(f"Epoch {epoch}/{opt.epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

if __name__ == '__main__':
    main()
