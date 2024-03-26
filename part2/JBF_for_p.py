import numpy as np
import cv2
import time

np.set_printoptions(threshold=np.inf)


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        t0 = time.time()

        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
        self.offset = self.wndw_size // 2
        self.Bilateral_Filter_vec = np.vectorize(
            self.Bilateral_Filter, signature="()->(n)"
        )
        xq, yq = np.meshgrid(np.arange(self.wndw_size), np.arange(self.wndw_size))
        self.Gs = np.exp(
            (((self.wndw_size // 2 - xq) ** 2) + (self.wndw_size // 2 - yq) ** 2)
            / (-2 * self.sigma_s**2)
        )

        self.sigma_r_dash = -2 * self.sigma_r**2



    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(
            img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(
            guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE
        ).astype(np.int32)

        guidance = (guidance / 255).astype(np.float64)
        padded_guidance = (padded_guidance / 255).astype(np.float64)

        self.img = img
        self.guidance = guidance
        self.padded_img = padded_img
        self.padded_guidance = padded_guidance
        w, h, c = self.img.shape
        Ip_dash = self.img.astype(np.int32)

        for row in range(19):
            for col in range(19):
                Ip_dash[row][col] = self.Bilateral_Filter(row, col)
        Ip_dash = Ip_dash.reshape((w, h, 3))

        return np.clip(Ip_dash, 0, 255).astype(np.uint8)

    # tp tq 為 guidence 中 pq位置的強度值
    def Gr(self, Tp, Tq):

        sum = (Tq - Tp) ** 2
        if len(self.guidance.shape) == 3:

            return np.exp(
                (sum[:, :, 0] + sum[:, :, 1] + sum[:, :, 2]) / self.sigma_r_dash
            )

        else:
            return np.exp(sum / self.sigma_r_dash)

    def Bilateral_Filter(self, xp,yp):
        
        xp_offset = xp + self.offset
        yp_offset = yp + self.offset

        Iq = self.padded_img[xp : xp + self.wndw_size, yp : yp + self.wndw_size]
        Tq = self.padded_guidance[xp : xp + self.wndw_size, yp : yp + self.wndw_size]
        Tp = self.padded_guidance[xp_offset][yp_offset]
        
        # 0.000006 sec
        mul = self.Gs * self.Gr(Tp, Tq) 
        
        # 0.000013 sec
        numerator = np.array(
            [
                np.sum(mul * Iq[:, :, 0]),
                np.sum(mul * Iq[:, :, 1]),
                np.sum(mul * Iq[:, :, 2]),
            ]
        )

        denominator = np.sum(mul)
        
        #  0.000002 sec
        output = (numerator / denominator).astype(np.int32)

        return output
