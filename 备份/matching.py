import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

image1 = cv2.imread(r"D:\adavance\resnet50\datasets\boot\train\break\0231025100705.jpg")
image2 = cv2.imread(r"D:\adavance\resnet50\datasets\boot\train\norm\N-T1-20230523-CRH380AL2622-02-B-02-L-DCZD-09-14.jpg")


from PIL import Image, ImageChops

def compare_images(image1, image2):
    img1 = Image.open(image1)
    img2 = Image.open(image2)
    diff = ImageChops.difference(img1, img2)
    diff.show()
    if diff.getbbox() is None:
        print("图片相同")
    else:
        print("图片不同")

compare_images("1.jpg", "12.jpg")

# pixel_diff = cv2.absdiff(image1, image2)
# gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# ssim_score = ssim(gray1, gray2)

# plt.subplot(1, 3, 1)
# plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
# plt.title('Image 1')

# plt.subplot(1, 3, 2)
# plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
# plt.title('Image 2')

# plt.subplot(1, 3, 3)
# plt.imshow(pixel_diff, cmap='gray')
# plt.title(f'Pixel Difference\nSSIM Score: {ssim_score:.2f}')

# plt.show()