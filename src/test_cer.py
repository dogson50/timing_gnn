import torch
from itertools import combinations
import numpy as np
import time


def CER(pred, gt, delta=0., top=5000):
    def cal_score(iter, delta=1.):
        assert iter.shape[1] == 2
        score = iter[:, 0] - iter[:, 1]
        score = torch.where(score > delta, torch.tensor(1., dtype=score.dtype),
                            torch.where(score < -delta, torch.tensor(-1., dtype=score.dtype),
                                        torch.where((score >= -delta) & (score <= delta),
                                                    torch.tensor(0., dtype=score.dtype), score)))
        return score

    assert pred.shape == gt.shape
    # # for pred with large size, we sample the largest 5000 paths
    # if len(gt) > top:
    #     gt, largest_indices = torch.topk(gt, top)
    #     pred = pred[largest_indices]
    indices = torch.tensor(list(combinations(range(len(pred)), 2)))
    print(f'indices got')
    pred_iter = pred[indices]
    gt_iter = gt[indices]
    pred_score = cal_score(pred_iter, delta=delta)
    gt_score = cal_score(gt_iter, delta=delta)
    # print(pred_score)
    # print(gt_score)
    compare = torch.ne(pred_score, gt_score).sum().item()
    return compare / len(indices)


def CER2(pred, gt, delta=1., tile_size=1000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'cer2 device: {device}')
    # gt = torch.tensor(gt, dtype=torch.float32, device=device)
    gt = gt.clone().detach().to(device)
    # pred = torch.tensor(pred, dtype=torch.float32, device=device)
    pred = pred.clone().detach().to(device)

    n = len(gt)
    count = 0
    num_all = (n * (n-1))/2

    for i_start in range(0, n, tile_size):
        i_end = min(i_start + tile_size, n)
        for j_start in range(0, n, tile_size):
            j_end = min(j_start + tile_size, n)

            gt_i = gt[i_start:i_end]
            pred_i = pred[i_start:i_end]
            gt_j = gt[j_start:j_end]
            pred_j = pred[j_start:j_end]
            gt_i = gt_i[:, None]
            pred_i = pred_i[:, None]
            gt_j = gt_j[:, None]
            pred_j = pred_j[:, None]

            valid_1 = (pred_i > (pred_j.T+delta)) & (gt_i > (gt_j.T+delta))
            valid_2 = (pred_i < (pred_j.T-delta)) & (gt_i < (gt_j.T-delta))
            valid_3 = ((pred_i <= (pred_j.T+delta)) & (pred_i >= (pred_j.T-delta))) & \
                      ((gt_i <= (gt_j.T+delta)) & (gt_i >= (gt_j.T-delta)))
            count += valid_1.sum().item() + valid_2.sum().item() + valid_3.sum().item()

    count -= n
    count /= 2

    return 1 - count / num_all


if __name__ == "__main__":
    pred = torch.randn(200000,)
    # print(pred)
    gt = torch.randn(200000,)
    # print(gt)

    t1 = time.time()
    print(CER2(pred, gt, delta=1., tile_size=2000))
    t2 = time.time()
    print(CER2(pred, gt, delta=1., tile_size=1000))
    t3 = time.time()
    #print(CER(pred, gt, delta=1.))
    t4 = time.time()
    print(f'cer 2 tile 2000:  {t2-t1}')
    print(f'cer 2 tile 1000: {t3-t2}')
    print(f'cer 1: {t4-t3}')