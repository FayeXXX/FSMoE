import torch
import torch.nn.functional as F
from torch.autograd import Variable
from methods.ARPL.arpl_utils import MetricMeter, AverageMeter
from methods.ARPL.loss.LabelSmoothing import smooth_one_hot
import itertools

from tqdm import tqdm

def extract_features(feature_dict, sample_ids):
    # 获取所有样本的索引
    all_indices = feature_dict['idx']

    # 找到 sample_ids 在 all_indices 中的位置（索引）
    index_map = {v.item(): i for i, v in enumerate(all_indices)}
    try:
        sample_indices = torch.tensor([index_map[s.item()] for s in sample_ids], dtype=torch.long)
    except:
        print('error')

    extracted_features = []

    # 遍历每一层
    for layer in feature_dict['features']:
        # 提取该层中对应样本 ID 的特征
        layer_features = layer[:, sample_indices, :]
        extracted_features.append(layer_features)

    return extracted_features


def train(net, optimizer, trainloader, epoch, args, log):
    net.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # 读取clip各层的feature
    multi_features_dir = f'{args.exp_root}/features/{args.dataset}/multilayers/{args.backbone}/split:{args.split_idx}'
    if args.dataset == 'cifar-10-100':
        multi_features_dir = f'{args.exp_root}/features/{args.dataset}-{args.out_num}/multilayers/{args.backbone}/split:{args.split_idx}'
    multi_features = torch.load(f'{multi_features_dir}/train/features.pth')

    torch.cuda.empty_cache()

    loss_all = 0
    for batch_idx, (data, labels, idx) in enumerate(tqdm(trainloader)):
    # for batch_idx, (data, labels, idx) in enumerate(tqdm(itertools.islice(trainloader, 0, 3))):  # for debugging
        if args.use_gpu:
            data, labels = data.cuda(), labels.cuda()
            multi_feature_batch = extract_features(multi_features, idx)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            logits, loss = net(data, labels, multi_feature_batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), data.size(0))
        
        loss_all += losses.avg
        step = len(trainloader) * epoch + batch_idx
        args.writer.add_scalar('loss', loss, step)

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
