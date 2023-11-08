import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# 读取两张图片
image1 = cv2.imread('1.jpg')
image2 = cv2.imread('1234.jpg')

# 确保两张图片具有相同的尺寸
if image1.shape != image2.shape:
    # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

# 图像去噪
image1 = cv2.medianBlur(image1, 3)
image2 = cv2.medianBlur(image2, 3)

# 图像稳定化（使用SIFT特征检测和匹配）
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

#根据特征匹配来使得第二幅图像与第一幅图像对齐
if len(good_matches) >= 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
    image2_stabilized = cv2.warpAffine(image2, M, (image2.shape[1], image2.shape[0]))
else:
    image2_stabilized = image2

# 计算相似性度量
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2_stabilized, cv2.COLOR_BGR2GRAY)

ssim = compare_ssim(gray_image1, gray_image2)
mse = np.mean((image1 - image2_stabilized) ** 2)

# 设置相似性阈值
threshold_ssim = 0.8  # 可根据需要调整
threshold_mse = 80  # 可根据需要调整
print(ssim ,mse)

# 综合判断
if ssim > threshold_ssim and mse < threshold_mse:
    print("两张图片相似，第二张图片可能无缺陷。")
else:
    print("两张图片不够相似，第二张图片可能存在缺陷。")
