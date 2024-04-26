import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None
    A = []

    if v.shape[0] is not N:
        print("u and v should have the same size")
        return None
    if N < 4:
        print("At least 4 points should be given")

    # TODO: 1.forming A
    # 實現 Ah = 0
    # 1. ux, uy, 1, 0, 0, 0, -uxvx, -uyvx, -vx
    # 2. 0, 0, 0, ux, uy, 1, -uxvy, -uyvy, -vy
    # ux = u[i][0], uy = u[i][1]
    # vx = v[i][0], vy = v[i][1]
    
    for i in range(N):
        A.append([u[i][0], u[i][1], 1, 0, 0, 0, -u[i][0]*v[i][0], -u[i][1]*v[i][0], -v[i][0]])
        A.append([0, 0, 0, u[i][0], u[i][1], 1, -u[i][0]*v[i][1], -u[i][1]*v[i][1], -v[i][1]])
    
    A = np.array(A)
    # print(f"\nAppend A:\n{A}")
    _u,_s,v_t = np.linalg.svd(A)
    # print(f"\nDoing svd...\n{v_t}")

    # TODO: 2.solve H with A
    # let h be the last column of V
    H = v_t[-1, :]
    # print(H)
    
    return H.reshape(3,3)


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction="b"):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H) # 在solve_homography中得到的H矩陣

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    x = np.arange(0, w_dst, 1)
    y = np.arange(0, h_dst, 1)
    xx, yy = np.meshgrid(x, y) # w*h, w*h
    xx, yy = xx.flatten()[:, np.newaxis], yy.flatten()[:, np.newaxis]
    
    
    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    ones = np.ones((len(xx), 1)) # N
    des_coor = np.concatenate((xx, yy, ones), axis=1).astype(np.int) # N*3, 3 means (x,y,1)

    if direction == "f":
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        # H_inv = 3*3
        # des_coor = N*3
        # H_inv dot des_coor = 3*N -> trans to N*3
        r_pixel = H_inv.dot(des_coor.T).T # 把
        
        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        r_pixel[:, :2] = r_pixel[:, :2] / r_pixel[:, 2][:, np.newaxis]
        valid_mask = (r_pixel[:, 0] >= 0) & (r_pixel[:, 0] < w_src) & (r_pixel[:, 1] >= 0) & (r_pixel[:, 1] < h_src)
        out_boundary = np.where(~valid_mask)[0]
        
        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        if len(out_boundary):
            r_pixel = np.delete(r_pixel, out_boundary, 0)
            des_coor = np.delete(des_coor, out_boundary, 0)
        
        # TODO: 6. assign to destination image with proper masking
        tx = r_pixel[:, 0].astype(np.int)
        ty = r_pixel[:, 1].astype(np.int)
        dx = r_pixel[:, 0] - tx
        dy = r_pixel[:, 1] - ty
        
        
        pass

    elif direction == "f":
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)

        # TODO: 5.filter the valid coordinates using previous obtained mask

        # TODO: 6. assign to destination image using advanced array indicing

        pass

    return dst