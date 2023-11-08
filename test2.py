import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def find_defects_by_comparison_sift(original_image,template_image,threshold_feature=0.8,threshold_ssim=0.8):
    '''
    输入
    original_image: 原始保存图像
    template_image: 巡检拍摄图像
    threshold_feature:通过特征点个数比较方法阈值，阈值越大更容易检测出缺陷，阈值越小越不容易检测出缺陷，对于防尘套破损这样不容易发现的错误阈值应该设高
    threshold_ssim:通过图像相似性比较方法阈值，阈值越大更容易检测出缺陷，阈值越小越不容易检测出缺陷，对于防尘套破损这样不容易发现的错误阈值应该设高
    其中original_image和template_image要保证是一个地方做故障前和做故障后的图，图像可以发生倾斜

    输出
    have_defect: True代表可能存在有缺陷。False代表没有检测出差别。
    '''
    have_defect = False

    # 使用SIFT特征提取器
    sift = cv2.SIFT_create()
    # 提取特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)
    # 使用FLANN匹配器进行特征匹配
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {})

    #先看图像自身有多少个特征匹配点，这样方便后边比较
    matches2 = flann.knnMatch(descriptors1, descriptors1, k=2)
    good_matches2 = []
    for m, n in matches2:
        if m.distance < 0.7 * n.distance:
            good_matches2.append(m)

    #先看两幅图片有多少个特征匹配点
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 在原始图像中绘制匹配的特征
    matched_image = cv2.drawMatches(original_image, keypoints1, template_image, keypoints2, good_matches, None)

    #根据特征匹配来使得第二幅图像与第一幅图像对齐
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        # 应用变换矩阵以对齐第二幅图像
        image2_stabilized = cv2.warpAffine(template_image, M, (template_image.shape[1], template_image.shape[0]))
    else:
        image2_stabilized = template_image

    #cv2.imshow("Image", template_image)
    #cv2.imshow("Image2", image2_stabilized)

    # 计算相似性度量
    #灰度化
    gray_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2_stabilized, cv2.COLOR_BGR2GRAY)
    #直方图均衡化
    #gray_image1 = cv2.equalizeHist(gray_image1)
    #gray_image2 = cv2.equalizeHist(gray_image2)
    ssim = compare_ssim(gray_image1, gray_image2)

    print("sift特征匹配识别到特征点的百分比：",len(good_matches)/len(good_matches2))
    print("sift特征匹配后比较的相似度：",ssim)
    print()
    # 检测缺陷
    if len(good_matches) < threshold_feature*len(good_matches2) or ssim < threshold_ssim: 
        have_defect=True
    else:
        have_defect=False

    # 显示结果图像
    cv2.imshow("Matched Image Sift", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return have_defect

def find_defects_by_comparison_orb(original_image,template_image,threshold_feature=0.8,threshold_ssim=0.8):
    '''
    输入
    original_image: 原始保存图像
    template_image: 巡检拍摄图像
    threshold_feature:通过特征点个数比较方法阈值，阈值越大更容易检测出缺陷，阈值越小越不容易检测出缺陷，对于防尘套破损这样不容易发现的错误阈值应该设高
    threshold_ssim:通过图像相似性比较方法阈值，阈值越大更容易检测出缺陷，阈值越小越不容易检测出缺陷，对于防尘套破损这样不容易发现的错误阈值应该设高
    其中original_image和template_image要保证是一个地方做故障前和做故障后的图，图像可以发生倾斜

    输出
    have_defect: True代表可能存在有缺陷。False代表没有检测出差别。
    '''
    have_defect = False

    # 转换为灰度图像
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)

    # 使用SIFT特征提取器
    orb = cv2.ORB_create()
    # 提取特征点和描述符
    keypoints1, descriptors1 = orb.detectAndCompute(original_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(template_image, None)

    # 使用BFMatcher特征匹配器匹配器进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches2 = bf.match(descriptors1, descriptors1)
    matches = bf.match(descriptors1, descriptors2)

    #先看图像自身有多少个特征匹配点，这样方便后边比较
    good_matches2 = []
    for m in matches2:
        if  m.queryIdx < len(keypoints1) and m.trainIdx < len(keypoints1):
            good_matches2.append(m)

    #先看两幅图片有多少个特征匹配点
    good_matches = []
    for m in matches:
        if  m.queryIdx < len(keypoints1) and m.trainIdx < len(keypoints2):
            good_matches.append(m)

    # 在原始图像中绘制匹配的特征
    matched_image = cv2.drawMatches(original_image, keypoints1, template_image, keypoints2, good_matches, None)

    #根据特征匹配来使得第二幅图像与第一幅图像对齐
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffine2D(src_pts, dst_pts)
        # 应用变换矩阵以对齐第二幅图像
        image2_stabilized = cv2.warpAffine(template_image, M, (template_image.shape[1], template_image.shape[0]))
    else:
        image2_stabilized = template_image

    #cv2.imshow("Image", template_image)
    #cv2.imshow("Image2", image2_stabilized)

    # 计算相似性度量
    #灰度化
    gray_image1 = original_image
    gray_image2 = image2_stabilized
    #直方图均衡化
    #gray_image1 = cv2.equalizeHist(gray_image1)
    #gray_image2 = cv2.equalizeHist(gray_image2)
    ssim = compare_ssim(gray_image1, gray_image2)

    print("orb特征匹配识别到特征点的百分比：",len(good_matches)/len(good_matches2))
    print("orb特征匹配后比较的相似度:",ssim)
    print()
    # 检测缺陷
    if len(good_matches) < threshold_feature*len(good_matches2) or ssim < threshold_ssim: 
        have_defect=True
    else:
        have_defect=False

    # 显示结果图像
    cv2.imshow("Matched Image Orb", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return have_defect

def find_defects_by_comparison(original_image,template_image,threshold_feature=0.8,threshold_ssim=0.8):
    # 图像去噪
    original_image = cv2.medianBlur(original_image, 3)
    template_image = cv2.medianBlur(template_image, 3)
    #调整亮度和对比度
    original_image = cv2.convertScaleAbs(original_image, alpha=1.2, beta=20)
    template_image = cv2.convertScaleAbs(template_image, alpha=1.2, beta=20)
    # 确保两张图片具有相同的尺寸
    if original_image.shape != template_image.shape:
        # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
        template_image = cv2.resize(template_image, (original_image.shape[1], original_image.shape[0]))
    have_defect1=find_defects_by_comparison_sift(original_image,template_image,threshold_feature,threshold_ssim)
    have_defect2=find_defects_by_comparison_orb(original_image,template_image,threshold_feature,threshold_ssim)
    if have_defect1 == False or have_defect2 == False:
        return False
    else:
        return True



if __name__ == "__main__":
    pass
    # # 加载原始图像和模板图像
    # original_image = cv2.imread('boltnut.jpg')
    # template_image = cv2.imread('lostboltnut.jpg')
    # threshold1=0.85
    # threshold2=0.85
    # have_defect=find_defects_by_comparison(original_image,template_image,threshold1,threshold2)
    # if have_defect==False: 
    #     print("排障器螺栓未检测到缺陷")
    # else:
    #     print("排障器螺栓检测到缺陷")


    original_image = cv2.imread('boot.jpg')
    template_image = cv2.imread('boot_break.jpg')
    threshold1=0.85
    threshold2=0.87
    have_defect=find_defects_by_comparison(original_image,template_image,threshold1,threshold2)
    if have_defect==False: 
        print("防尘套未检测到缺陷")
    else:
        print("防尘套检测到缺陷")


    # original_image = cv2.imread('fsts.png')
    # template_image = cv2.imread('fstsds.png')
    # threshold1=0.85
    # threshold2=0.85
    # have_defect=find_defects_by_comparison(original_image,template_image,threshold1,threshold2)
    # if have_defect==False:
    #     print("防松铁丝未检测到缺陷")
    # else:
    #     print("防松铁丝检测到缺陷")


    # original_image = cv2.imread('kwx.jpg')
    # template_image = cv2.imread('kwxb.jpg')
    # threshold1=0.85
    # threshold2=0.85
    # have_defect=find_defects_by_comparison(original_image,template_image,threshold1,threshold2)
    # if have_defect==False: 
    #     print("开尾销未检测到缺陷")
    # else:
    #     print("开尾销检测到缺陷")

    # original_image = cv2.imread('mq(1).jpg')
    # template_image = cv2.imread('mq2(1).jpg')
    # threshold1=0.2
    # threshold2=0.4
    # have_defect=find_defects_by_comparison(original_image,template_image,threshold1,threshold2)
    # if have_defect==False: 
    #     print("未检测到缺陷")
    # else:
    #     print("检测到缺陷")
    
