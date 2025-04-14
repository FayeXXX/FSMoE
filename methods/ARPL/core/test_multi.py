import os
import os.path as osp
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
from methods.ARPL.core import evaluation

from sklearn.metrics import average_precision_score

def extract_features(feature_dict, sample_ids):
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


def test(net, testloader, outloader, epoch, args, log):

    net.eval()
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for data, labels, idx in tqdm(testloader):
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                logits = net(data, True)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()

                if args.use_softmax_in_eval:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels, idx) in enumerate(tqdm(outloader)):
            if args.use_gpu:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):

                logits = net(data, None)
                if args.use_softmax_in_eval:
                    logits = torch.nn.Softmax(dim=-1)(logits)

                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    log.info('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)
    
    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']
    
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    # Average precision
    ap_score = average_precision_score([0] * len(_pred_k) + [1] * len(_pred_u),
                                       list(-np.max(_pred_k, axis=-1)) + list(-np.max(_pred_u, axis=-1)))

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['AUPR'] = ap_score * 100
    results['EPOCH'] = epoch

    return results