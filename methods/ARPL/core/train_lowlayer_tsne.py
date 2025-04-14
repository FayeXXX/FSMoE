import torch
import torch.nn.functional as F
from torch.autograd import Variable
from methods.ARPL.arpl_utils import MetricMeter, AverageMeter
from methods.ARPL.loss.LabelSmoothing import smooth_one_hot
import itertools

from tqdm import tqdm
import gzip
import pickle
import numpy as np

def extract_features_batch(feature_dict, sample_ids):
    # 获取所有样本的索引
    all_indices = feature_dict['idx']

    # 找到 sample_ids 在 all_indices 中的位置（索引）
    index_map = {v.item(): i for i, v in enumerate(all_indices)}
    sample_indices = torch.tensor([index_map[s.item()] for s in sample_ids], dtype=torch.long)

    extracted_features = []
    # 遍历每一层
    for layer in feature_dict['features']:
        # 提取该层中对应样本 ID 的特征
        layer_features = layer[:, sample_indices, :]
        extracted_features.append(layer_features)

    return extracted_features


def load_features(multi_features_dir, layers_file):
    multi_features = torch.load(f'{multi_features_dir}/train/features.pth')
    with open(layers_file, "r") as file:
        layers_to_extract = [int(line.strip()) for line in file]
    features = [multi_features['features'][layer] for layer in layers_to_extract]
    features = {'features': features, 'idx': multi_features['idx']}
    return features


def train(net, optimizer, trainloader, epoch, args, log):
    net.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    torch.cuda.empty_cache()

    inter_dict = {'feature': [], 'label': []}
    top_dict = {'feature': [], 'label': []}
    interafter_dict = {'feature': [], 'label': []}
    topafter_dict = {'feature': [], 'label': []}

    loss_all = 0
    if epoch!=30:
        for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):
        # for batch_idx, (data, labels, idx) in enumerate(tqdm(itertools.islice(trainloader, 0, 3))):  # for debugging
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                loss, ss_loss, image_inter, image_top, bias_extra, top_feature = net(data, labels)
                total_loss = loss + ss_loss
                total_loss.backward()
                optimizer.step()

            losses.update(total_loss.item(), data.size(0))

            loss_all += losses.avg
            step = len(trainloader) * epoch + batch_idx
            args.writer.add_scalar('classify_loss', loss, step)
            args.writer.add_scalar('total_loss', total_loss, step)
            args.writer.add_scalar('ss_loss', ss_loss, step)
    else:
        for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):
        # for batch_idx, (data, labels, idx) in enumerate(tqdm(itertools.islice(trainloader, 0, 3))):  # for debugging
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                loss, ss_loss, image_inter, image_top, bias_extra, top_feature = net(data, labels)
                total_loss = loss + ss_loss
                total_loss.backward()
                optimizer.step()

            losses.update(total_loss.item(), data.size(0))

            loss_all += losses.avg
            step = len(trainloader) * epoch + batch_idx
            args.writer.add_scalar('classify_loss', loss, step)
            args.writer.add_scalar('total_loss', total_loss, step)
            args.writer.add_scalar('ss_loss', ss_loss, step)

            inter_dict['feature'].append(image_inter.cpu().numpy())
            inter_dict['label'].append(labels.cpu().numpy())
            top_dict['feature'].append(image_top.cpu().numpy())
            top_dict['label'].append(labels.cpu().numpy())
            interafter_dict['feature'].append(bias_extra.detach().cpu().numpy())
            interafter_dict['label'].append(labels.cpu().numpy())
            topafter_dict['feature'].append(top_feature.detach().cpu().numpy())
            topafter_dict['label'].append(labels.cpu().numpy())

    if epoch == 30:
        inter_dict['feature'] = np.concatenate(inter_dict['feature'], axis=0)
        inter_dict['label'] = np.concatenate(inter_dict['label'], axis=0)
        top_dict['feature'] = np.concatenate(top_dict['feature'], axis=0)
        top_dict['label'] = np.concatenate(top_dict['label'], axis=0)
        interafter_dict['feature'] = np.concatenate(interafter_dict['feature'], axis=0)
        interafter_dict['label'] = np.concatenate(interafter_dict['label'], axis=0)
        topafter_dict['feature'] = np.concatenate(topafter_dict['feature'], axis=0)
        topafter_dict['label'] = np.concatenate(topafter_dict['label'], axis=0)

        with gzip.open(f'{args.train_dir}/inter_dict.pkl.gz', 'wb') as f:
            pickle.dump(inter_dict, f)

        with gzip.open(f'{args.train_dir}/top_dict.pkl.gz', 'wb') as f:
            pickle.dump(top_dict, f)

        with gzip.open(f'{args.train_dir}/interafter_dict.pkl.gz', 'wb') as f:
            pickle.dump(interafter_dict, f)

        with gzip.open(f'{args.train_dir}/topafter_dict.pkl.gz', 'wb') as f:
            pickle.dump(topafter_dict, f)


    log.info("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx + 1, len(trainloader), losses.val, losses.avg))  # value on current batch, average value
    # args.writer.add_scalars('loss_avg', losses.avg, epoch)

    return loss_all


def train_cs(net, netD, netG, criterion, criterionD, optimizer, optimizerD, optimizerG, 
        trainloader, epoch=None, **options):
    print('train with confusing samples')
    losses, lossesG, lossesD = AverageMeter(), AverageMeter(), AverageMeter()

    net.train()
    netD.train()
    netG.train()

    torch.cuda.empty_cache()
    
    loss_all, real_label, fake_label = 0, 1, 0
    for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):
        gan_target = torch.FloatTensor(labels.size()).fill_(0)
        if options['use_gpu']:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            gan_target = gan_target.cuda()
        
        data, labels = Variable(data), Variable(labels)
        
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)

        ###########################
        # (1) Update D network    #
        ###########################
        # train with real
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterionD(output, targetv)
        errD_real.backward()

        # train with fake
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterionD(output, targetv)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()

        ###########################
        # (2) Update G network    #
        ###########################
        optimizerG.zero_grad()
        # Original GAN loss
        targetv = Variable(gan_target.fill_(real_label))
        output = netD(fake)
        errG = criterionD(output, targetv)

        # minimize the true distribution
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        errG_F = criterion.fake_loss(x).mean()
        generator_loss = errG + options['beta'] * errG_F
        generator_loss.backward()
        optimizerG.step()

        lossesG.update(generator_loss.item(), labels.size(0))
        lossesD.update(errD.item(), labels.size(0))


        ###########################
        # (3) Update classifier   #
        ###########################
        # cross entropy loss
        optimizer.zero_grad()
        x, y = net(data, True, 0 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        _, loss = criterion(x, y, labels)

        # KL divergence
        noise = torch.FloatTensor(data.size(0), options['nz'], options['ns'], options['ns']).normal_(0, 1).cuda()
        if options['use_gpu']:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        x, y = net(fake, True, 1 * torch.ones(data.shape[0], dtype=torch.long).cuda())
        F_loss_fake = criterion.fake_loss(x).mean()
        total_loss = loss + options['beta'] * F_loss_fake
        total_loss.backward()
        optimizer.step()
    
        losses.update(total_loss.item(), labels.size(0))

        # if (batch_idx+1) % options['print_freq'] == 0:
        #     print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
        #     .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))
    
        loss_all += losses.avg

    print("Batch {}/{}\t Net {:.3f} ({:.3f}) G {:.3f} ({:.3f}) D {:.3f} ({:.3f})" \
    .format(batch_idx+1, len(trainloader), losses.val, losses.avg, lossesG.val, lossesG.avg, lossesD.val, lossesD.avg))

    return loss_all
