import cv2
import numpy as np
import os
from tqdm import tqdm

flie_folder = r"D:\adavance\Template_Matching_test\test\TEST1"
template_paths = r"D:\adavance\Template_Matching_test"
template_path = os.path.join(template_paths, "2.jpg")
for img_name in tqdm(os.listdir(flie_folder)): 
    print(img_name)
    imagefile = os.path.join(flie_folder, img_name)
    img_rgb = cv2.imread(imagefile)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    left_top = max_loc  # 左上角
    right_bottom = (max_loc[0] + w, max_loc[1] + h)  # 右下角
    cx = int((right_bottom[0] + left_top[0]) / 2)
    cy = int((right_bottom[1] + left_top[1]) / 2)
    # 画点
    cv2.circle(img_rgb, (cx, cy), 5, (0, 0, 255), int(min(w / 2, h / 2)))
    cv2.rectangle(img_rgb, left_top, right_bottom, (0, 0, 255), 1)
    savefile = imagefile.replace("_v.jpg", "_l.jpg")
    cv2.imwrite(savefile, img_rgb)
# template_paths = ["E:/000-004-bolt_hexagon_1_v.png", "E:/000-077-bolt_hexagon_5_v.png", "E:/000-083-bolt_hexagon_16_v.png"]
# template_paths = ["E:/000-084-bolt_hexagon_16_v.png"]
# templates = []
# for template_path in template_paths:
#     templates.append(cv2.imread(template_path, 0))
# for img_folder_name in os.listdir(flie_folder):
#     img_folder = os.path.join(flie_folder, img_folder_name)
#     for bb in os.listdir(img_folder):
#         img_path = os.path.join(img_folder, bb)
#         for img_name in tqdm(os.listdir(img_path)):
#             if "_v.png" not in img_name:
#                 continue
#             imagefile = os.path.join(img_path, img_name)
#             img_rgb = cv2.imread(imagefile)
#             img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#             class_name = img_name.split("_")[-2]
#             template_path = os.path.join(template_paths, class_name + ".png")
#             template = cv2.imread(template_path, 0)
#             # left_tops = []
#             # right_bottoms = []
#             w, h = template.shape[::-1]
#             res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
#             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#             left_top = max_loc  # 左上角
#             right_bottom = (max_loc[0] + w, max_loc[1] + h)  # 右下角
#             # left_tops = np.array(left_tops)
#             # right_bottoms = np.array(right_bottoms)
#             # left_top = left_tops.min(axis=0)
#             # right_bottom = right_bottoms.min(axis=0)
#             cx = int((right_bottom[0] + left_top[0]) / 2)
#             cy = int((right_bottom[1] + left_top[1]) / 2)
#             # 画点
#             cv2.circle(img_rgb, (cx, cy), 5, (0, 0, 255), int(min(w / 2, h / 2)))
#             cv2.rectangle(img_rgb, left_top, right_bottom, (0, 0, 255), 1)
#             savefile = imagefile.replace("_v.png", "_l.png")
#             cv2.imwrite(savefile, img_rgb)


# img_rgb = cv2.imread("E:/bolt_label/2578/B1_img_roi_3d/000-002-bolt_hexagon_1_v.png")
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread("E:/000-004-bolt_hexagon_1_v.png", 0)
# w, h = template.shape[::-1]
# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
# left_top = max_loc   # 左上角
# right_bottom = (left_top[0] + w, left_top[1] + h)   # 右下角
# cx = int(left_top[0] + w / 2)
# cy = int(left_top[0] + h / 2)
# 画点
# cv2.circle(img_rgb, (cx, cy), 5, (0, 0, 255), int(min(w / 2, h / 2)))
# cv2.rectangle(img_rgb, left_top, right_bottom, (0, 0, 255), 1)
# threshold = 0.6
# loc = np.where(res >= threshold)
# for pt in zip(*loc[::-1]):
#     cx = int(pt[0] + w / 2)
#     cy = int(pt[0] + h / 2)
#     # 画点
#     cv2.circle(img_rgb, (cx, cy), 1, (0, 0, 255), 1)
#     cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
# cv2.imwrite("E:/1.png", img_rgb)
# cv2.imshow("img", img_rgb)
# cv2.waitKey(0)
