import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)
def matches(im1, im2):
    # Use opencv built-in ORB detector for keypoint detection
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, desc1 = orb.detectAndCompute(im1, None) # 1000, (1000,32)
    kp2, desc2 = orb.detectAndCompute(im2, None)

    # Use opencv brute force matcher for feature matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    match_points = bf.knnMatch(desc1, desc2, k=2)
    
    # Filtering the match_points
    f_match_points = [m1 for m1,m2 in match_points if m1.distance < 0.8 * m2.distance]

    f_kp1 = [kp1[mp.queryIdx].pt for mp in f_match_points]
    f_kp2 = [kp2[mp.queryIdx].pt for mp in f_match_points]
    
    f_kp1 = np.array(f_kp1).astype(int)
    f_kp2 = np.array(f_kp2).astype(int)
    
    return f_kp1,f_kp2

def distance(x1,y1,x2,y2):
    return np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2)).reshape(-1)

def get_outlier(H, mps1, mps2):
    
    # trans mps2 to H coor
    ones = np.ones((mps2.shape[0], 1))
    mps2 = np.concatenate((mps2, ones), axis=1)
    trans_mps2 = H.dot(mps2.T)
    trans_mps2 = trans_mps2 / (trans_mps2[2,:][np.newaxis, :])
    trans_mps2 = trans_mps2[0:2,:].T
    
    x1,y1 = mps1[:, 0],mps1[:, 1]
    x2,y2 = trans_mps2[:, 0], trans_mps2[:, 1]
    dst = distance(x1,y1,x2,y2)
    
    outliers_count = 0
    for d in dst:
        if d > 3:
            outliers_count += 1
    print(outliers_count)
    return outliers_count


def ransac(mps1,mps2):
    all_matches = mps1.shape[0]
    outliers_count = 0
    lowest_outlier = all_matches
    
    for i in range(10):
        rand_index = np.random.choice(all_matches, 5, replace=False)
        H = solve_homography(mps2[rand_index], mps1[rand_index])
        outliers_count = get_outlier(H, mps1, mps2)
        if outliers_count < lowest_outlier:
            best_H = H
            lowest_outlier = outliers_count

    return best_H

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        matchpoints1, matchpoints2 = matches(im1,im2)
        
        # TODO: 2. apply RANSAC to choose best H
        H = ransac(matchpoints1, matchpoints2)
        
        # TODO: 3. chain the homographies
        last_best_H = last_best_H.dot(H)
        
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 'b')
        
    return dst 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)

    cv2.imwrite('output4.png', output4)