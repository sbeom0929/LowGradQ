'''Train ImageNet with PyTorch.'''
import argparse
import os
import shutil   
import time 
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel    
import torch.backends.cudnn as cudnn 
import torch.optim  
import torch.utils.data 
import torchvision
import torchvision.transforms as transforms 
import torchvision.datasets as datasets     

from models import 
from models.alexnet import  
from utils import progress_bar


# Model                    
print('== Building model..')
# model = VGG('VGG19')
model = ResNet18()
#model = ResNet50()
# model = PreActResNet18()
# model = GoogLeNet()
# model = DenseNet121()
# model = ResNeXt29_2x64d()
#model = MobileNet()
#model = mobilenetv2()
# model = DPN92()
# model = ShuffleNetG2()
# model = SENet18()
# model = ShuffleNetV2(1)
#model = EfficientNetB0()
# model = RegNetX_200MF()
#model = SimpleDLA()

model_name = model.__class__.__name__      

print(model_name)
print(model)
 
parser = argparse.ArgumentParser(description='Various models for ImageNet in pytorch') 
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',    
                    help='number of data loading workers (default 4)')     
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',   
                    help='path to latest checkpoint (default none)')
parser.add_argument('--evaluate', default='', type=str, metavar='PATH',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',   # default=''
                    help='Path to the pretrained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',                      
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)


parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default 1e-4)')   # resnet models 1e-4, other models  5e-4

best_acc = 0

def main()
    global args, best_acc, model
    args = parser.parse_args()

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir)
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(model)
    model.cuda()

    #print(model)                                  

    # optionally resume from a checkpoint
    if args.resume
        if os.path.isfile(args.resume)
            print(= loading checkpoint '{}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])     
            print(= loaded checkpoint '{}' (epoch {})
                  .format(args.evaluate, checkpoint['epoch']))  
        else
            print(= no checkpoint found at '{}'.format(args.resume))

    if args.pretrained
        if os.path.isfile(args.pretrained)
            print(f= loading pretrained model '{args.pretrained}')
            checkpoint = torch.load(args.pretrained)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            
            ################## for resnet                                   
            for k, v in checkpoint.items()
                name = 'module.' + k.replace(fc, linear).replace(downsample, shortcut)
                new_state_dict[name] = v

            ###################### for mobilenet v2  alexnet  other models
            # for k, v in checkpoint.items()
            #     if 'module.' not in k
            #         name = 'module.' + k
            #     else
            #         name = k
            #     new_state_dict[name] = v
            ##############################################

            model.load_state_dict(new_state_dict)
        else
            print(f= no pretrained model found at '{args.pretrained}')
    
    cudnn.benchmark = True      

    ################################################################## 
    #           imageNet dataset settings
    #
    ##################################################################
    print('== Preparing data..')
    input_size = 224        # imagenet input resolution 
    
    traindir = homeidsldataimagenettrain
    valdir = homeidsldataimagenetval
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    transform_train = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        transform_train, batch_size=args.batch_size, shuffle=True,      
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    ######################################################################
    #       define loss function (criterion) and optimizer
    #
    ######################################################################
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # for resnet models 
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,60,90], gamma=0.1, last_epoch=args.start_epoch - 1)          

    # other models                                
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)                                       

    if args.half
        model.half()
        criterion.half()

    for param_group in optimizer.param_groups
        if 'initial_lr' not in param_group
            param_group['initial_lr'] = param_group['lr']

    #  If you need to evaluate using pretrained weights, use this block.
    # if args.evaluate
    #     if os.path.isfile(args.evaluate)
    #         print(f= loading pretrained model '{args.evaluate}')
    #         checkpoint = torch.load(args.evaluate)

    #         from collections import OrderedDict
    #         new_state_dict = OrderedDict()

    #         # for resnet models                                                                      
    #         for k, v in checkpoint.items()
    #             name = 'module.' + k.replace(fc, linear).replace(downsample, shortcut)
    #             new_state_dict[name] = v

    #         # other models
    #         # for k, v in checkpoint.items()
    #         #     if 'module.' not in k
    #         #         name = 'module.' + k
    #         #     else
    #         #         name = k
    #         #     new_state_dict[name] = v

    #         model.load_state_dict(new_state_dict)
    #         validate(val_loader, model, criterion)
    #         return
    #     else
    #         print(f= no pretrained model found at '{args.evaluate}')
    #         return

    # If you need to evaluate using experiment th file, use this block.
    if args.evaluate
        if os.path.isfile(args.evaluate)
            print(f= loading pretrained model '{args.evaluate}')
            checkpoint = torch.load(args.evaluate)

            if 'state_dict' in checkpoint  
                new_state_dict = checkpoint['state_dict']
            else
                new_state_dict = checkpoint

            from collections import OrderedDict
            fixed_state_dict = OrderedDict()
            for k, v in new_state_dict.items()
                if k.startswith('module.')
                    fixed_state_dict[k] = v  
                else
                    fixed_state_dict['module.' + k] = v  

            model.load_state_dict(fixed_state_dict, strict=False) 
            validate(val_loader, model, criterion)
            return
        else
            print(f= no pretrained model found at '{args.evaluate}')
            return
    ########################################################
 
    for epoch in range(args.start_epoch, args.epochs)
        # train for one epoch
        #print('current lr {.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)                                          
        
        lr_scheduler.step()

        # evaluate on validation set
        acc = validate(val_loader, model, criterion) 

        # remember best prec@1 and save checkpoint
        is_best = acc  best_acc
        best_acc = max(acc, best_acc)

        if epoch  0 and epoch % args.save_every == 0     
            save_checkpoint({
                'epoch' epoch + 1,
                'state_dict' model.state_dict(),
                'best_acc' best_acc,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({       
            'state_dict' model.state_dict(),
            'best_acc' best_acc,
        }, is_best, filename=os.path.join(args.save_dir,f'{model_name}_best_acc.th'))   

# Training
def train(train_loader, model, criterion, optimizer, epoch) 
    print('nEpoch %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader)
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss %.3f  Acc %.3f%% (%d%d)'
                     % (train_loss(batch_idx+1), 100.correcttotal, correct, total))
        
def validate(val_loader, model, criterion)
    global best_acc
    model.eval()
    validate_loss = 0
    correct = 0
    total = 0
    with torch.no_grad()
        for batch_idx, (inputs, targets) in enumerate(val_loader)
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            validate_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss %.3f  Acc %.3f%% (%d%d)'
                         % (validate_loss(batch_idx+1), 100.correcttotal, correct, total))
            
    return 100.  correct  total

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar')
    
    Save the training model
    
    torch.save(state, filename)

if __name__ == '__main__'
    main()

