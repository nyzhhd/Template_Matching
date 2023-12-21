import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def find_defects_by_comparison_sift(original_image,template_image):
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
    
    original_image = cv2.medianBlur(original_image, 3)
    template_image = cv2.medianBlur(template_image, 3)
    #调整亮度和对比度
    original_image = cv2.convertScaleAbs(original_image, alpha=1.2, beta=20)
    template_image = cv2.convertScaleAbs(template_image, alpha=1.2, beta=20)
    # 确保两张图片具有相同的尺寸
    if original_image.shape != template_image.shape:
        # 如果尺寸不同，将第二张图片调整为与第一张图片相同的尺寸
        template_image = cv2.resize(template_image, (original_image.shape[1], original_image.shape[0]))

    # 使用SIFT特征提取器
    sift = cv2.SIFT_create()
    # 提取特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(original_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(template_image, None)
    # 使用FLANN匹配器进行特征匹配
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {})

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


    # 计算相似性度量
    #灰度化
    gray_image1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2_stabilized, cv2.COLOR_BGR2GRAY)
    #直方图均衡化
    #gray_image1 = cv2.equalizeHist(gray_image1)
    #gray_image2 = cv2.equalizeHist(gray_image2)
    ssim = compare_ssim(gray_image1, gray_image2)



    # 显示结果图像
    # cv2.imshow("Matched Image Sift", matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return ssim

def find_defects_by_comparison_Bolt(original_image,threshold):
        have_defect = False
        gk_norm_folder = 'gk_mb_20231113'
        simm_list=[]
        for filename in os.listdir(gk_norm_folder):
            image_norm_path = os.path.join(gk_norm_folder, filename)
            template_image = cv2.imread(image_norm_path)
            try:
               simm = find_defects_by_comparison_sift(original_image, template_image)
            except:
                simm=0
            simm_list.append(simm)

        simm_avg = sum(simm_list) / len(simm_list)
        simm_max = max(simm_list)
        print("最大相似度：",simm_max)
        print("平均相似度：",simm_avg)

        if simm_max>threshold:
            #print(f"{filename} 未检测到缺陷")
            have_defect=False
        else:
            #print(f"{filename} 检测到缺陷")
            have_defect=True
        return have_defect,simm_avg


if __name__ == "__main__":
    pass



    import os
    threshold=0.53
    
    gk_folder = 'gk'
    i=0
    j=0
    for filename in os.listdir(gk_folder):
        print(f'+++++++++++++++++{j}+++++++++++++++++++++++')
        image_path = os.path.join(gk_folder, filename)
        original_image = cv2.imread(image_path)
        have_defect,simm=find_defects_by_comparison_Bolt(original_image,threshold)
        
        j=j+1
        if have_defect==False:
            print(f"{image_path} 未检测到缺陷")
        else:
            print(f"{image_path} 检测到缺陷")
            i=i+1
    print(gk_folder,f"检测出{i}/{j}个图片有缺陷")
        
