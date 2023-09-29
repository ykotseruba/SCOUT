import cv2
import os
import numpy as np
from scipy.special import kl_div
from scipy.stats import entropy
from scipy.stats.mstats import spearmanr
import pandas as pd


def read_image(img_path, channels_first, color=True, color_mode='BGR', dtype=np.float32, resize_dim=None):

    """
    Reads and returns an image as a numpy array

    Parameters
    ----------
    img_path : string
        Path of the input image
    channels_first: bool
        If True, channel dimension is moved in first position
    color: bool, optional
        If True, image is loaded in color: grayscale otherwise
    color_mode: "RGB", "BGR", optional
        Whether to load the color image in RGB or BGR format
    dtype: dtype, optional
        Array is casted to this data type before being returned
    resize_dim: tuple, optional
        Resize size following convention (new_h, new_w) - interpolation is linear

    Returns
    -------
    image : np.array
        Loaded Image as numpy array of type dtype
    """

    if not os.path.exists(img_path):
        raise FileNotFoundError('Provided path "{}" does NOT exist.'.format(img_path))

    image = cv2.imread(img_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(img_path  + ' not found!')

    if color and color_mode == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize_dim is not None:
        image = cv2.resize(image, dsize=resize_dim[::-1], interpolation=cv2.INTER_LINEAR)

    if color and channels_first:
        image = np.transpose(image, (2, 0, 1))

    return image.astype(dtype)

# DReyeVE implementation
# https://github.com/ndrplz/dreyeve/blob/master/experiments/metrics/metrics.py (function kl_numeric)
def KLdiv_DReyeVE(y_true, y_pred, eps=2.2204e-16):
	"""
	Function to evaluate Kullback-Leiber divergence (sec 4.2.3 of [1]) on two samples.
	converted from Matlab implementation here:
	https://github.com/cvzoya/saliency/blob/master/code_forMetrics/KLdiv.m

	The two distributions are numpy arrays having arbitrary but coherent shapes.

	:param y_true: groundtruth.
	:param y_pred: predictions.
	:return: numeric kld
	"""
	y_true = y_true.astype(np.float32)
	y_pred = y_pred.astype(np.float32)

	#eps = np.finfo(np.float32).eps
	
	P = y_pred / (eps + np.sum(y_pred))  # pred_prob
	Q = y_true / (eps + np.sum(y_true))  # gt_prob

	kld = np.sum(Q * np.log(eps + Q / (eps + P)))

	return kld


# Fahimi and Bruce implementation
# https://github.com/rAm1n/saliency/blob/master/metrics/metrics.py
def KLdiv_Fahimi_Bruce(gt, pred, eps=2.2204e-16):
	"""
	This finds the KL-divergence between two different saliency maps when
	viewed as distributions: it is a non-symmetric measure of the information
	lost when saliencyMap is used to estimate fixationMap.

		Computer KL-divergence.
		:param saliency_map : predicted saliency map
		:param saliency_map_gt : ground truth saliency map.
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
	score = entropy(gt.flatten()+eps, pred.flatten()+eps)
	return score




# Pysaliency implementation
# https://github.com/matthias-k/pysaliency/blob/master/pysaliency/metrics.py
def convert_saliency_map_to_density(saliency_map, eps=1e-20):
	if saliency_map.min() < 0:
		saliency_map = saliency_map - saliency_map.min()
	saliency_map = saliency_map + eps

	saliency_map_sum = saliency_map.sum()
	if saliency_map_sum:
		saliency_map = saliency_map / saliency_map_sum
	else:
		saliency_map[:] = 1.0
		saliency_map /= saliency_map.sum()

	return saliency_map



def probabilistic_image_based_kl_divergence(logp1, logp2, log_regularization=0, quotient_regularization=0):
	if log_regularization or quotient_regularization:
		return (np.exp(logp2) * np.log(log_regularization + np.exp(logp2) / (np.exp(logp1) + quotient_regularization))).sum()
	else:
		return (np.exp(logp2) * (logp2 - logp1)).sum()

# modified the arguments order to match other functions
# originally the first input is the predicted map and second input is the ground truth
def KLdiv_pysaliency(saliency_map_2, saliency_map_1, eps=1e-20, log_regularization=0, quotient_regularization=0):
	""" KLDiv. Function is not symmetric. saliency_map_2 is treated as empirical saliency map. """
	log_density_1 = np.log(convert_saliency_map_to_density(saliency_map_1, eps=eps))
	log_density_2 = np.log(convert_saliency_map_to_density(saliency_map_2, eps=eps))

	return probabilistic_image_based_kl_divergence(log_density_1, log_density_2, log_regularization=log_regularization, quotient_regularization=quotient_regularization)


# test different KLDiv implementations on random and actual images from DReyeVE
def test_metric_functions(func_list, random=True, baseline_map=None):
	
	# epsilons: [ MATLAB epsilon, numpy epsion, common value ]
	eps = [np.finfo(np.float32).eps, 2.2204e-16, 0.0001]
	
	if random:
		gt_img = np.random.rand(1500, 1500)
		pred_img = np.random.rand(1500, 1500)
	else:
		gt_img_path = 'images/DReyeVE_47_54_sal.png'
		pred_img_path = 'images/DReyeVE_47_54_pred.jpg'

		gt_img = read_image(gt_img_path, False, color=False)/255.0
		pred_img = read_image(pred_img_path, False, color=False, resize_dim=None)/255.0
	table = []
	
	for func in func_list:
		row = {'func': func.__name__}
		for e in eps:
			score = func(gt_img.copy(), pred_img.copy(), eps=e)
			row[e] = score
		table.append(row)
	df = pd.DataFrame(table)

	print(df)
	#print(df.to_latex(index=False))

print('Test KLDiv implementations')
kl_div_functions = [KLdiv_DReyeVE, KLdiv_Fahimi_Bruce, KLdiv_pysaliency]

print('Random images')
test_metric_functions(kl_div_functions, random=True)

print('Sample from DReyeVE')
test_metric_functions(kl_div_functions, random=False)
