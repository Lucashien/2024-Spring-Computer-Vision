import numpy as np
import cv2

np.set_printoptions(threshold=np.inf)


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6 * sigma_s + 1
        self.pad_w = 3 * sigma_s
        self.offset = self.wndw_size // 2

        xq, yq = np.meshgrid(np.arange(self.wndw_size), np.arange(self.wndw_size))
        self.Gs = np.exp(
            (((self.wndw_size // 2 - xq) ** 2) + (self.wndw_size // 2 - yq) ** 2)
            / (-2 * self.sigma_s**2)
        )

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

        for row in range(w):
            for col in range(h):
                Ip_dash[row][col] = self.Bilateral_Filter(row, col)
        
        return np.clip(Ip_dash, 0, 255).astype(np.uint8)

    # tp tq 為 guidence 中 pq位置的強度值
    def Gr(self, Tp, Tq):

        if len(self.guidance.shape) == 3:
            Tp_r, Tp_g, Tp_b = Tp
            # Tq_r, Tq_g, Tq_b = Tp
            Tq_r = Tq[:, :, 0]
            Tq_g = Tq[:, :, 1]
            Tq_b = Tq[:, :, 2]

            return np.exp(
                (((Tp_r - Tq_r) ** 2) + ((Tp_g - Tq_g) ** 2) + ((Tp_b - Tq_b) ** 2))
                / (-2 * self.sigma_r**2)
            )
        else:
            return np.exp(((Tp - Tq) ** 2) / (-2 * self.sigma_r**2))

    def Bilateral_Filter(self, xp, yp):
        # xp, yp 是原圖
        xp_offset = xp + self.offset
        yp_offset = yp + self.offset

        Iq = self.padded_img[xp : xp + self.wndw_size, yp : yp + self.wndw_size]
        Tq = self.padded_guidance[xp : xp + self.wndw_size, yp : yp + self.wndw_size]
        Tp = self.padded_guidance[xp_offset][yp_offset]

        numerator = np.array(
            [
                np.sum(self.Gs * self.Gr(Tp, Tq) * Iq[:, :, 0]),
                np.sum(self.Gs * self.Gr(Tp, Tq) * Iq[:, :, 1]),
                np.sum(self.Gs * self.Gr(Tp, Tq) * Iq[:, :, 2]),
            ]
        )

        denominator = np.sum(self.Gs * self.Gr(Tp, Tq))

        output = (numerator / denominator).astype(np.int32)

        return output
