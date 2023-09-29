# Saliency metrics
# Code for metrics from: https://github.com/rAm1n/saliency

import sys
import os
import cv2
import numpy as np
from skimage.transform import resize
#from scipy.misc import imresize
from scipy.stats import entropy
from scipy.spatial.distance import directed_hausdorff, euclidean
from scipy.stats import pearsonr
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
"""
Collection of common saliency metrics

If you're using this code, please don't forget to cite the original code
as mentioned in each function doc.

Modified by ykotseruba: 
unified function signatures
changed naming of arguments
removed unused functions

"""

def normalize_map(s_map):
    # normalize the salience map to [0, 1] (as done in MIT code)
    if np.all(s_map[s_map[0] == np.ravel(s_map)]):
        return s_map/s_map[0]
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)))
    return norm_s_map

def normalize_map_std(s_map):
    s_map -= s_map.mean()
    std = s_map.std()

    if std:
        s_map /= std

    return s_map, std == 0


def NSS(gt, pred):
    """"
    normalized scanpath saliency between two different
    saliency maps as the mean value of the normalized saliency map at
    fixation locations.

        Computer NSS score.
        :param pred : predicted saliency map
        :param gt : ground truth saliency map.
        :return score: float : score

    """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    if pred.size != gt.size:
        pred = resize(pred, gt.shape)

    MAP = (pred - pred.mean()) / (pred.std()+np.finfo(np.float32).eps)
    mask = gt.astype(bool)
    score =  MAP[mask].mean()
    return score


def CC(gt, pred):
    """
    This finds the linear correlation coefficient between two different
    saliency maps (also called Pearson's linear coefficient).
    score=1 or -1 means the maps are correlated
    score=0 means the maps are completely uncorrelated

    saliencyMap1 and saliencyMap2 are 2 real-valued matrices

        Computer CC score .
        :param pred : first saliency map
        :param gt : second  saliency map.
        :return score: float : score

    """
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)
    elif pred.dtype != np.float32:
        pred = pred.astype(np.float32)

    if not isinstance(gt, np.ndarray):
        gt = np.array(gt, dtype=np.float32)
    elif gt.dtype != np.float32:
        gt = gt.astype(np.float32)

    if pred.size != gt.size:
        pred = resize(pred, gt.shape)


    if not pred.std() or not gt.std():
        return 0


    pred, sm_std_is_zero = normalize_map_std(pred)
    gt, gt_std_is_zero = normalize_map_std(gt)

    if sm_std_is_zero and not gt_std_is_zero:
        score = 0
    else:
        score = np.corrcoef(pred.flatten(),gt.flatten())[0][1]
    
    return score

def KLdiv(gt, pred):
    """
    This finds the KL-divergence between two different saliency maps when
    viewed as distributions: it is a non-symmetric measure of the information
    lost when saliencyMap is used to estimate fixationMap.

        Computer KL-divergence.
        :param pred : predicted saliency map
        :param gt : ground truth saliency map.
        :return score: float : score

    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)
    elif pred.dtype != np.float32:
        pred = pred.astype(np.float32)

    if not isinstance(gt, np.ndarray):
        gt = np.array(gt, dtype=np.float32)
    elif gt.dtype != np.float32:
        gt = gt.astype(np.float32)


    if pred.size != gt.size:
        pred = resize(pred, gt.shape)


    # the entropy function will normalize maps before computing Kld
    score = entropy(gt.flatten(), pred.flatten())
    return score


def AUC_old(gt, pred):
    """Computes AUC for given saliency map 'pred' and given
    fixation map 'gt'
    """
    def area_under_curve(predicted, actual, labelset):
        def roc_curve(predicted, actual, cls):
            si = np.argsort(-predicted)
            tp = np.cumsum(np.single(actual[si]==cls))
            fp = np.cumsum(np.single(actual[si]!=cls))
            tp = tp/np.sum(actual==cls)
            fp = fp/np.sum(actual!=cls)
            tp = np.hstack((0.0, tp, 1.0))
            fp = np.hstack((0.0, fp, 1.0))
            return tp, fp
        def auc_from_roc(tp, fp):
            h = np.diff(fp)
            auc = np.sum(h*(tp[1:]+tp[:-1]))/2.0
            return auc

        tp, fp = roc_curve(predicted, actual, np.max(labelset))
        auc = auc_from_roc(tp, fp)
        return auc

    gt = (gt>0.7).astype(int)
    salShape = pred.shape
    fixShape = gt.shape

    predicted = pred.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
    actual = gt.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
    labelset = np.arange(2)

    return area_under_curve(predicted, actual, labelset)

def AUC(gt, pred):
    gt = (gt>0.7).astype(int)
    salShape = pred.shape
    fixShape = gt.shape

    predicted = pred.reshape(salShape[0]*salShape[1], -1, order='F').flatten()
    actual = gt.reshape(fixShape[0]*fixShape[1], -1, order='F').flatten()
    
    return roc_auc_score(actual, predicted)

def sAUC(gt, pred, shuf_map=np.zeros((480,640)), step_size=.01):
    """
        please cite:  https://github.com/NUS-VIP/salicon-evaluation
        calculates shuffled-AUC score.

        :param pred : predicted saliency map
        :param gt : ground truth saliency map.
        :return score: int : score

    """

    pred -= np.min(pred)
    gt = np.vstack(np.where(gt!=0)).T
    print(gt.shape)
    if np.max(pred) > 0:
        pred = pred / np.max(pred)
    Sth = np.asarray([ pred[x][y] for x,y in gt ])
    Nfixations = len(gt)

    others = np.copy(shuf_map)

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = shuf_map[ind]
    randfix = pred[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),step_size)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    score = np.trapz(tp,fp)
    return score


def IG(gt, pred, baseline_map=np.zeros((480,640))):
    """
        please cite:

        calculates Information gain score.

        :param pred : predicted saliency map
        :param gt : ground truth saliency map.
        :param baseline_gt : a baseline fixtion map
        :return score: int : score

    """
    # fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # ax1.imshow(gt)
    # ax2.imshow(pred)
    # ax3.imshow(baseline_map)
    # plt.show()

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)
    elif pred.dtype != np.float32:
        pred = pred.astype(np.float32)

    if not isinstance(gt, np.ndarray):
        gt = np.array(gt, dtype=np.float32)
    elif gt.dtype != np.float32:
        gt = gt.astype(np.float32)


    if not isinstance(baseline_map, np.ndarray):
        baseline_map = np.array(baseline_map, dtype=np.float32)
    elif gt.dtype != np.float32:
        baseline_map = baseline_map.astype(np.float32)


    if pred.size != gt.size:
        pred = resize(pred, gt.shape)


    pred = (pred - pred.min()) \
                        / (pred.max() - pred.min())

    pred = pred / pred.sum()

    baseline_map = (baseline_map - baseline_map.min()) \
                        / (baseline_map.max() - baseline_map.min())
    baseline_map = baseline_map / baseline_map.sum()

    fixs = gt.astype(bool)

    EPS = np.finfo(np.float32).eps

    return (np.log2(EPS + pred[fixs]) - np.log2(EPS + baseline_map[fixs])).mean()

def convert_saliency_map_to_density(saliency_map, minimum_value=0.0):
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value

    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()

    return saliency_map


def SIM(gt, pred):
    """
        Compute similarity score.

        :param pred : predicted saliency map
        :param gt : ground truth saliency map.
        :return score: float : score

    """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred, dtype=np.float32)
    elif pred.dtype != np.float32:
        pred = pred.astype(np.float32)

    if not isinstance(gt, np.ndarray):
        gt = np.array(gt, dtype=np.float32)
    elif gt.dtype != np.float32:
        gt = gt.astype(np.float32)

    if pred.size != gt.size:
        pred = resize(pred, gt.shape)

    pred = convert_saliency_map_to_density(pred)
    gt = convert_saliency_map_to_density(gt)
    # pred = (pred - pred.min()) \
    #                     / (pred.max() - pred.min())
    # pred = pred / pred.sum()

    # gt = (gt - gt.min()) \
    #                     / (gt.max() - gt.min())
    # gt = gt / gt.sum()

    return np.minimum(pred, gt).sum()

