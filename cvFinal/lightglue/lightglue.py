import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from util import *
from LightGlue.lightglue import LightGlue, SuperPoint, viz2d
from LightGlue.lightglue.utils import numpy_image_to_torch, batch_to_device
from LightGlue.lightglue.utils import rbd

def match_image(
    image0: np.array,
    image1: np.array,
    threshold: float = 0.9,
    device: str = 'cpu',
    verbose: bool = False
):
    """
    Match a pair of images (image0, image1) through SuperPoint and LightGlue\n
    - input\n
        o image0 (numpy array): a grayscale image\n
        o image1 (numpy array): a grayscale image\n
        o threshold (float32): filter the matching pair whose score is\n
                               under the threshold\n
        o device (string): indicate the decvice to run the NN module\n
        o verbose (bool): set True for visualization\n
    - output\n
        o kp0 (numpy array): dim = (N, 2), contain N pixel locations\n
        o kp1 (numpy array): dim = (N, 2), contain N pixel locaitons\n
        o score (numpy array): dim = (N, 1), contain N scores corresponding\n
                               to each pair, the higher score indicates\n
                               higher confidence of that pair\n
    """
    
    # load SuperPoint and LightGlue model to device
    torch.set_grad_enabled(False)
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features='superpoint').eval().to(device)
    
    # convert numpy array to tensor
    image0 = numpy_image_to_torch(image0)
    image1 = numpy_image_to_torch(image1)
    
    # use SuperPoint to extract keypoints
    feats0 = extractor.extract(image0.to(device))
    feats1 = extractor.extract(image1.to(device))
    
    # use LightGlue to match keypoints 
    matches01 = matcher({"image0": feats0, "image1": feats1})
    
    # remove batch dimemsion and move back to CPU
    feats0, feats1, matches01 = \
        [batch_to_device(rbd(x), 'cpu') for x in [feats0, feats1, matches01]]
        
    # filter the matching pairs whose scores are below threshold
    vld = matches01["scores"] >= threshold
    kp0 = feats0["keypoints"][matches01["matches"][vld][:, 0]]
    kp1 = feats1["keypoints"][matches01["matches"][vld][:, 1]]
    score = matches01["scores"][vld]
    
    if vld.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    
    # sort the valid matching pairs along the score
    sorted_pairs = sorted(
        zip(kp0, kp1, score),
        key=lambda pair: pair[2],
        reverse=True)
    kp0, kp1, score = zip(*sorted_pairs)

    # convert the matching pairs and scores from tensor to numpy array
    kp0 = np.array(kp0, dtype=np.float32)
    kp1 = np.array(kp1, dtype=np.float32)
    score = np.array(score, dtype=np.float32)
    
    # visualization and print the shape of output
    if verbose:
        viz2d.plot_images([image0, image1])
        viz2d.plot_matches(kp0, kp1, color="lime", lw=0.2)
        viz2d.add_text(0, f'threshold: {threshold}', fs=20)
        plt.show()
    
        print(kp0.shape)
        print(kp1.shape)
        print(score.shape)
    
    return kp0, kp1, score

if __name__ == '__main__':
    # set directory path <- modify here
    img_base = './dataset'
    seg_base = './reid'
    
    # choose your interseting frames and object <- modify here
    frame0 = 0
    frame1 = 16
    obj_id = 53
    thres = 0.9
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # read images and segmenation maps
    img0 = cv2.imread(f'{img_base}/{frame0:03}.png', cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(f'{img_base}/{frame1:03}.png', cv2.IMREAD_GRAYSCALE)
    seg0 = np.load(f'{seg_base}/{frame0:03}.npy')
    seg1 = np.load(f'{seg_base}/{frame1:03}.npy')

    # crop the object
    mask0 = mask_by_id(seg0, obj_id)
    mask1 = mask_by_id(seg1, obj_id)
    obj0 = img0 * mask0
    obj1 = img1 * mask1

    # preform matching
    feats0, feats1, matches01 = match_image(
        obj0, obj1, thres, device, verbose=True)