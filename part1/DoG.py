import numpy as np
import cv2

# check = np.load("/home/r12a10/hw1_material/part1/dog/dog_array_0_0.npy")
np.set_printoptions(threshold=np.inf)
# print(check)
# exit()


def normalize(img):
    max = np.max(img)
    min = np.min(img)
    img = ((img - min) / (max - min)) * 255
    return img


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
            # print(f"octave = {octave}")

            if octave == 2:
                image = gaussian_imgs[-1]
                image = cv2.resize(
                    image,
                    (image.shape[1] // octave, image.shape[0] // octave),
                    interpolation=cv2.INTER_NEAREST,
                )

            gaussian_imgs.append(image)  # 原圖 or resize後的

            for i in range(
                len(gaussian_imgs), len(gaussian_imgs) + self.num_DoG_images_per_octave
            ):
                gaussian_img = cv2.GaussianBlur(image, (0, 0), self.sigma ** (i % 5))
                gaussian_imgs.append(gaussian_img)
                image_path = f"DoG_{octave}-{i%5}.png"

                # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
                # - Function: cv2.subtract(second_image, first_image)
                dog_img = cv2.subtract(gaussian_imgs[i], gaussian_imgs[i - 1])
                # dog_img = normalize(dog_img)

                dog_imgs.append(dog_img)
                np.save(image_path, dog_img)
                cv2.imwrite(image_path, normalize(dog_img))

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoints
        keypoints = []
        for i in range(1, len(dog_imgs)):
            if i == 1 or i == 2 or i == 5 or i == 6:
                for row in range(1, dog_imgs[i].shape[0]):
                    for col in range(1, dog_imgs[i].shape[1]):
                        max = np.max(
                            [
                                dog_imgs[i + 1][row - 1 : row + 2, col - 1 : col + 2],
                                dog_imgs[i][row - 1 : row + 2, col - 1 : col + 2],
                                dog_imgs[i - 1][row - 1 : row + 2, col - 1 : col + 2],
                            ]
                        )

                        min = np.min(
                            [
                                dog_imgs[i + 1][row - 1 : row + 2, col - 1 : col + 2],
                                dog_imgs[i][row - 1 : row + 2, col - 1 : col + 2],
                                dog_imgs[i - 1][row - 1 : row + 2, col - 1 : col + 2],
                            ]
                        )
 
                        if max > self.threshold and max == dog_imgs[i][row][col]:
                            # print(f"{i,row , col}")
                            keypoints.append([row * (i//5+1), col * (i//5+1)])
                        if abs(min) > self.threshold and abs(min) == abs(dog_imgs[i][row][col]):
                            # print(f"{i,row , col}")
                            keypoints.append([row * (i//5+1), col * (i//5+1)])

        keypoints =np.unique(np.array(keypoints),axis = 0)
        
        # print(keypoints)
        # print(len(keypoints))
        

        # # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:, 1], keypoints[:, 0]))]
        return keypoints
