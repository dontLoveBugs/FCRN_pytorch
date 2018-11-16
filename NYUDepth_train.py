# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 19:40
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com
import shutil
import time
import torch
from tensorboardX import SummaryWriter

from metrics import AverageMeter, Result
import utils
import NYU_dpeth
import FCRN
import criteria
import csv
import os
import torch.nn as nn

args = utils.parse_command()
print(args)

fieldnames = ['mse', 'rmse', 'absrel', 'lg10', 'mae',
                'delta1', 'delta2', 'delta3',
                'data_time', 'gpu_time']
best_result = Result()
best_result.set_to_worst()

# t = torch.zeros((4, 3, 257, 353), dtype=torch.float32)

if torch.cuda.device_count() > 1:
    t_train = torch.Size([args.batch_size * torch.cuda.device_count(), 3, 228, 304])
else:
    t_train = torch.Size([args.batch_size, 3, 228, 304])

t_val = torch.Size([1, 3, 228, 304])

def main():
    global args, best_result, output_directory, train_csv, test_csv

    if torch.cuda.device_count() > 1:
        # args.batch_size = args.batch_size * torch.cuda.device_count()

        train_loader = NYU_dpeth.NYUDepth_loader(args.data_path, batch_size=args.batch_size * torch.cuda.device_count(), isTrain=True)
        val_loader = NYU_dpeth.NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=False)
    else:
        train_loader = NYU_dpeth.NYUDepth_loader(args.data_path, batch_size=args.batch_size, isTrain=True)
        val_loader = NYU_dpeth.NYUDepth_loader(args.data_path, isTrain=False)

    if args.resume:
        assert os.path.isfile(args.resume), \
            "=> no checkpoint found at '{}'".format(args.resume)
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # args = checkpoint['args']
        # print('保留参数：', args)
        start_epoch = checkpoint['epoch'] + 1
        best_result = checkpoint['best_result']
        model_dict = checkpoint['model'].module.state_dict()  # 如果是多卡训练的要加module
        model = FCRN.FCRN(batch_size=args.batch_size)
        model.load_state_dict(model_dict)
        print('batch_size = ', model.upSample.batch_size)
        # optimizer = checkpoint['optimizer']
        # 使用SGD进行优化
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> creating Model")
        model = FCRN.FCRN_pretrain(batch_size=args.batch_size)
        print("=> model created.")
        # optimizer = torch.optim.Adam(model.parameters(), args.lr)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        start_epoch = 0
    # 如果有多GPU 使用多GPU训练
    if torch.cuda.device_count():
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.cuda()

    # 定义loss函数
    criterion = torch.nn.MSELoss().cuda()

    # 创建保存结果目录文件
    output_directory = './result5/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    train_csv = os.path.join(output_directory, 'train7.csv')
    test_csv = os.path.join(output_directory, 'test7.csv')
    best_txt = os.path.join(output_directory, 'best7.txt')

    log_path = os.path.join(args.log_path, "{}".format('nyu'))
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = SummaryWriter(log_path)

    for epoch in range(start_epoch, args.epochs):
        utils.adjust_learning_rate(optimizer, args.lr, epoch)  # 更新学习率

        train(train_loader, model, criterion, optimizer, epoch, logger)  # train for one epoch
        result, img_merge = validate(val_loader, model, epoch, logger)  # evaluate on validation set

        # remember best rmse and save checkpoint
        is_best = result.rmse < best_result.rmse
        if is_best:
            best_result = result
            with open(best_txt, 'w') as txtfile:
                txtfile.write("epoch={}\nmse={:.3f}\nrmse={:.3f}\nabsrel={:.3f}\nlg10={:.3f}\nmae={:.3f}\ndelta1={:.3f}\nt_gpu={:.4f}\n".
                    format(epoch, result.mse, result.rmse, result.absrel, result.lg10, result.mae, result.delta1, result.gpu_time))
            if img_merge is not None:
                img_filename = output_directory + '/comparison_best.png'
                utils.save_image(img_merge, img_filename)

        # 每个Epoch都保存解雇
        utils.save_checkpoint({
            'args': args,
            'epoch': epoch,
            'model': model,
            'best_result': best_result,
            'optimizer' : optimizer,
        }, is_best, epoch, output_directory)

    logger.close()


# 在NYUDepth数据集上训练
def train(train_loader, model, criterion, optimizer, epoch, logger):
    average_meter = AverageMeter()
    model.train() # switch to train mode
    end = time.time()

    batch_num = len(train_loader)

    for i, (input, target) in enumerate(train_loader):
        if input.size() != t_train:
            print('----------------------------- -')
            print('input size = ', input.size())
            print('target sieze = ', target.size())
            print('输入不足batch size！跳过本条输入！')
            print('-------------------------------')
            continue

        # itr_count += 1
        input, target = input.cuda(), target.cuda()
        # print('input size  = ', input.size())
        # print('target size = ', target.size())
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute pred
        end = time.time()
        pred = model(input)  # @wx 注意输出

        # print('pred size = ', pred.size())
        # print('target size = ', target.size())

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward() # compute gradient and do SGD step
        optimizer.step()
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print('=> output: {}'.format(output_directory))
            print('Train Epoch: {0} [{1}/{2}]\t'
                  't_Data={data_time:.3f}({average.data_time:.3f}) '
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                  epoch, i+1, len(train_loader), data_time=data_time,
                  gpu_time=gpu_time, result=result, average=average_meter.average()))
            current_step = epoch * batch_num + i
            logger.add_scalar('Train/RMSE', result.rmse, current_step)
            logger.add_scalar('Train/MAE', result.mae, current_step)
            logger.add_scalar('Train/Delta1', result.delta1, current_step)
            logger.add_scalar('Train/REL', result.absrel, current_step)
            logger.add_scalar('Train/Log10', result.lg10, current_step)

    avg = average_meter.average()
    with open(train_csv, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
            'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
            'gpu_time': avg.gpu_time, 'data_time': avg.data_time})


# 修改
def validate(val_loader, _model, epoch, logger, write_to_file=True):
    average_meter = AverageMeter()
    model = FCRN.FCRN(batch_size=1)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model.load_state_dict(_model.state_dict())
    model.eval() # switch to evaluate mode
    model.cuda()
    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        # print('test!')
        # print('input size = ', input.size())
        # print('target size = ', target.size())
        if t_val != input.size():
            continue
        input, target = input.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        # print('##################')
        # print('input size = ', input.size(0))
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        # save 8 images for visualization
        skip = 50
        if args.modality == 'd':
            img_merge = None
        else:
            if args.modality == 'rgb':
                rgb = input
            elif args.modality == 'rgbd':
                rgb = input[:,:3,:,:]
                depth = input[:,3:,:,:]

            if i == 0:
                if args.modality == 'rgbd':
                    img_merge = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    img_merge = utils.merge_into_row(rgb, target, pred)
            elif (i < 8*skip) and (i % skip == 0):
                if args.modality == 'rgbd':
                    row = utils.merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    row = utils.merge_into_row(rgb, target, pred)
                img_merge = utils.add_row(img_merge, row)
            elif i == 8*skip:
                filename = output_directory + '/comparison_' + str(epoch) + '.png'
                utils.save_image(img_merge, filename)

        if (i+1) % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  't_GPU={gpu_time:.3f}({average.gpu_time:.3f})\n\t'
                  'RMSE={result.rmse:.2f}({average.rmse:.2f}) '
                  'MAE={result.mae:.2f}({average.mae:.2f}) '
                  'Delta1={result.delta1:.3f}({average.delta1:.3f}) '
                  'REL={result.absrel:.3f}({average.absrel:.3f}) '
                  'Lg10={result.lg10:.3f}({average.lg10:.3f}) '.format(
                   i+1, len(val_loader), gpu_time=gpu_time, result=result, average=average_meter.average()))

    avg = average_meter.average()

    print('\n*\n'
        'RMSE={average.rmse:.3f}\n'
        'MAE={average.mae:.3f}\n'
        'Delta1={average.delta1:.3f}\n'
        'REL={average.absrel:.3f}\n'
        'Lg10={average.lg10:.3f}\n'
        't_GPU={time:.3f}\n'.format(
        average=avg, time=avg.gpu_time))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})

    logger.add_scalar('Test/mse', avg.mse, epoch)
    logger.add_scalar('Test/rmse', avg.rmse, epoch)
    logger.add_scalar('Test/Rel', avg.absrel, epoch)
    logger.add_scalar('Test/log10', avg.lg10, epoch)
    logger.add_scalar('Test/Delta1', avg.delta1, epoch)
    return avg, img_merge


if __name__ == '__main__':
    main()