import numpy as np
import cv2
from PIL import Image


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2 ** (1 / 4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        dog_imgs = []
        gaussian_imgs = []
        for octave in range(1, self.num_octaves + 1):
            print(f"octave = {octave}")

            if octave == 2:
                image = gaussian_imgs[-1]
                image = cv2.resize(
                    image,
                    (image.shape[1] // octave, image.shape[0] // octave),
                    interpolation=cv2.INTER_NEAREST,
                )

            gaussian_imgs.append(image)

            for i in range(len(gaussian_imgs), len(gaussian_imgs) + self.num_DoG_images_per_octave):
                gaussian_img = cv2.GaussianBlur(image, (0, 0), self.sigma**i)
                gaussian_imgs.append(gaussian_img)
                image_path = f"DoG_{octave}-{i%5}.png"

                # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
                # - Function: cv2.subtract(second_image, first_image)
                dog_img = gaussian_imgs[i] - gaussian_imgs[i - 1]
                max = np.max(dog_img)
                min = np.min(dog_img)
                n_dog_img = (max - dog_img) / (max - min) * 255
                dog_imgs.append(n_dog_img)
                cv2.imwrite(image_path, n_dog_img)

        #     # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #     #         Keep local extremum as a keypoint
        #     for dog_img in dog_imgs:
        #         max = np.max(dog_img)
        #         min = np.max(dog_img)
        #         print(min)
        #         # coordinates = np.where(dog_img == max)
        #         coordinates = np.where(dog_img == min)
        #         print(coordinates)

        #     # Step 4: Delete duplicate keypoints
        #     # - Function: np.unique
        #     # uniques = np.unique(thresh_img)
        #     # print(uniques)
        #     # for unique in uniques:
        #     #     unique_img = np.where(thresh_img == self.threshold, 0, thresh_img)

        #     # image_list.append(unique_img)
        #     # cv2.imwrite(image_path, unique_img)
        #     # print(f"Saved Gaussian images to the following paths: {image_path}")
        # exit()
        # # sort 2d-point by y, then by x
        # keypoints = image_list
        # keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        # return keypoints
