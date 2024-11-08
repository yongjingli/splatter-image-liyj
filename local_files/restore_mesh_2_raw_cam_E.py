import copy
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def get_crop_and_expand_rect_by_mask(mask, w_ratio=0, h_ratio=0):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, None, None

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    expand_w = int(w * (1 + w_ratio))
    expand_h = int(h * (1 + h_ratio))

    expand_x = x - int(w * w_ratio / 2)
    expand_y = y - int(h * h_ratio / 2)

    img_height, img_width = mask.shape
    if expand_x < 0:
        expand_x = 0
    if expand_y < 0:
        expand_y = 0
    if expand_x + expand_w > img_width:
        expand_w = img_width - expand_x
    if expand_y + expand_h > img_height:
        expand_h = img_height - expand_y

    return expand_x, expand_y, expand_w, expand_h


def pad_and_resize_img(img, foreground_ratio=0.65):
    # pad to square
    size = max(img.shape[0], img.shape[1])
    ph0, pw0 = (size - img.shape[0]) // 2, (size - img.shape[1]) // 2
    ph1, pw1 = size - img.shape[0] - ph0, size - img.shape[1] - pw0
    new_image = np.pad(
        img,
        ((ph0, ph1), (pw0, pw1), (0, 0)),
        mode="constant",
        constant_values=((255, 255), (255, 255), (0, 0)),
    )

    # compute padding according to the ratio,
    new_size = int(new_image.shape[0] / foreground_ratio)
    # pad to size, double side
    ph2, pw2 = (new_size - size) // 2, (new_size - size) // 2
    ph3, pw3 = new_size - size - ph2, new_size - size - pw2
    new_image = np.pad(
        new_image,
        ((ph2, ph3), (pw2, pw3), (0, 0)),
        mode="constant",
        constant_values=((255, 255), (255, 255), (0, 0)),
    )
    pad_h = ph0 + ph2
    pad_w = pw0 + pw2
    return new_image, pad_h, pad_w


def resize_to_128_with_K(img, fov=49.0):
    img_h, img_w, _ = img.shape

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = transforms.functional.resize(img, 128, interpolation=transforms.InterpolationMode.LANCZOS)
    img = np.array(img)

    # img_w = 128
    fov = fov * np.pi / 180
    fx = 128 / (2 * math.tan(fov / 2))
    fy = fx
    cx = 128 / 2
    cy = 128 / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])

    scale = img.shape[0]/img_h
    return img, K, scale


def get_rt_between_img_and_crop_img(img, K, img_pts, crop_img, crop_K, crop_img_pts):
    # img_pts = img_pts[:100, ]
    # crop_img_pts = crop_img_pts[:100, ]

    def check_depth(R, t, points1, points2, K1, K2):
        # 将像素坐标转换为相机坐标系下的归一化坐标
        points1_norm = cv2.undistortPoints(np.expand_dims(points1, axis=1), K1, None)
        points2_norm = cv2.undistortPoints(np.expand_dims(points2, axis=1), K2, None)
        points1_norm = np.squeeze(points1_norm)
        points2_norm = np.squeeze(points2_norm)

        for i in range(len(points1_norm)):
            P1 = np.append(points1_norm[i], 1)
            P2 = np.append(points2_norm[i], 1)
            # 根据不同的旋转和平移组合计算P2的估计值
            P2_estimated = R @ P1 + t
            if P2_estimated[2] <= 0:
                return False
        return True

    img_pts = img_pts.astype(np.float32)
    crop_img_pts = crop_img_pts.astype(np.float32)

    # 计算基础矩阵（Fundamental Matrix），这里只是为了演示步骤，实际应该先计算本质矩阵
    F, mask = cv2.findFundamentalMat(img_pts, crop_img_pts, cv2.FM_LMEDS)

    # 从基础矩阵恢复本质矩阵（假设相机内参已知）
    K1 = K
    K2 = crop_K
    E = K2.T @ F @ K1

    # SVD分解本质矩阵
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # 得到四个可能的旋转和平移组合
    R1 = U @ W @ V
    R2 = U @ W.T @ V
    t1 = U[:, 2]
    t2 = -U[:, 2]
    # return R2, t2
    # 检查正深度约束，选择正确的组合
    if check_depth(R1, t1, img_pts, crop_img_pts, K1, K2):
        correct_R = R1
        correct_t = t1
    elif check_depth(R1, t2, img_pts, crop_img_pts, K1, K2):
        correct_R = R1
        correct_t = t2
    elif check_depth(R2, t1, img_pts, crop_img_pts, K1, K2):
        correct_R = R2
        correct_t = t1
    elif check_depth(R2, t2, img_pts, crop_img_pts, K1, K2):
        correct_R = R2
        correct_t = t2
    else:
        print("没有找到满足正深度约束的组合")
        correct_R = None
        correct_t = None
    return correct_R, correct_t


def project_points_to_image_opencv(img, points, cam_K, dist_coeffs=None, rvec=None, tvec=None):
    img_h, img_w, _ = img.shape
    # cam_fx = cam_K[0, 0]
    # cam_fy = cam_K[1, 1]
    # cam_cx = cam_K[0, 2]
    # cam_cy = cam_K[1, 2]
    #
    # x, y, z = points[:, 0], points[:, 1], points[:, 2]
    #
    # u = np.round(x * cam_fx/z + cam_cx).astype(int)
    # v = np.round(y * cam_fy/z + cam_cy).astype(int)
    #
    # uvz = np.stack([u, v, z])
    # uvz = np.transpose(uvz, (1, 0))
    #
    # camera_matrix = np.array([[fx, 0, cx],
    #                           [0, fy, cy],
    #                           [0, 0, 1]], np.float32)

    # Define the distortion coefficients
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), np.float32)

    # Define the rotation and translation vectors
    if rvec is None:
        rvec = np.zeros((3, 1), np.float32)

    if tvec is None:
        tvec = np.zeros((3, 1), np.float32)

    # Map the 3D point to 2D point
    # dist_coeffs = np.zeros((5, 1), np.float32)
    points_2d, _ = cv2.projectPoints(points,
                                     rvec, tvec,
                                     cam_K,
                                     dist_coeffs)
    u = points_2d[:, 0, 0]
    v = points_2d[:, 0, 1]
    z = points[:, 2]

    uvz = np.stack([u, v, z])
    uvz = np.transpose(uvz, (1, 0))

    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    mask_u = np.bitwise_and(u >= 0, u < img_w)
    mask_v = np.bitwise_and(v >= 0, v < img_h)
    mask = np.bitwise_and(mask_u, mask_v)
    uvz = uvz[mask]

    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    depth_map[uvz[:, 1].astype(int), uvz[:, 0].astype(int)] = uvz[:, 2]
    # depth_map[uvz[:, 1].astype(int), uvz[:, 0].astype(int)] = 1
    return depth_map



def get_rt_from_imgs(img1, K1, img2, K2):
    # 特征点提取与匹配函数
    def extract_and_match(img1, img2):
        # 使用 SIFT 特征提取器
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # 使用 FLANN 进行特征点匹配
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 筛选好的匹配点
        good_matches = []
        for m, n in matches:
            # if m.distance < 0.7 * n.distance:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        return kp1, kp2, good_matches

    # 计算基础矩阵
    def compute_fundamental_matrix(kp1, kp2, good_matches):
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        return F

    # 计算本质矩阵
    def compute_essential_matrix(F, K1, K2):
        return np.dot(np.dot(K2.T, F), K1)

    # 从本质矩阵恢复位姿
    def recover_pose(E, K1, K2, pts1, pts2):
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K1)
        return R, t

    # 提取特征点并匹配
    kp1, kp2, good_matches = extract_and_match(img1, img2)

    # 假设已知相机内参矩阵
    # K1 = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    # K2 = K1

    # 计算基础矩阵
    F = compute_fundamental_matrix(kp1, kp2, good_matches)

    # 计算本质矩阵
    E = compute_essential_matrix(F, K1, K2)

    # 恢复位姿
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    R, t = recover_pose(E, K1, K2, pts1, pts2)
    return R, t.reshape(-1)


def get_crop_depth():
    # 得到crop归一化的图像的depth
    root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    img_root = os.path.join(root, "colors")
    masks_root = os.path.join(root, "masks_num")
    depths_root = os.path.join(root, "depths")

    img_names = [name for name in os.listdir(img_root) if name.endswith(".jpg")]
    img_names = list(sorted(img_names, key=lambda x: int(x.split("_")[0])))

    s_root = os.path.join(root, "crop_depth")
    os.makedirs(s_root, exist_ok=True)

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]

    for img_name in img_names:
        img_path = os.path.join(img_root, img_name)
        depth_path = os.path.join(depths_root, img_name.replace("_color.jpg", "_depth.npy"))
        mask_path = os.path.join(masks_root, img_name.replace("_color.jpg", "_mask.npy"))

        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)
        depth = np.load(depth_path)
        mask = np.load(mask_path)

        # 生成模型的图像预处理过程
        x, y, w, h = get_crop_and_expand_rect_by_mask(mask)
        if x==None:
            continue

        img_crop = img[y:y+h, x:x+w, :]
        mask_crop = mask[y:y+h, x:x+w]
        img_crop[~mask_crop] = (255, 255, 255)
        # plt.imshow(img_crop[:, :, ::-1])
        # plt.show()

        img_crop_pad, pad_h, pad_w = pad_and_resize_img(img_crop, foreground_ratio=0.65)
        # plt.imshow(img_crop_pad[:, :, ::-1])
        # plt.show()
        # cv2.imwrite(os.path.join(s_root, img_name.replace("_color.jpg", "_color_crop_pad.jpg")), img_crop_pad)
        img_crop_pad, crop_K, scale = resize_to_128_with_K(img_crop_pad, fov=49.0)
        # cv2.imwrite(os.path.join(s_root, img_name.replace("_color.jpg", "_color_crop_pad.jpg")), img_crop_pad)
        # exit(1)

        # 得到原图中物体点在crop图中的坐标,以及在原图的点云坐标
        # 得到在crop图中的坐标
        grid_x, grid_y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]), indexing='xy')
        grids = np.stack([grid_x, grid_y])
        grids = np.transpose(grids, (1, 2, 0))

        valid_object_mask = np.bitwise_and(depth > 0, mask)
        valid_grids = grids[valid_object_mask]
        valid_grids_img = copy.deepcopy(valid_grids)

        valid_grids = valid_grids - np.array([x, y])
        valid_grids = valid_grids + np.array([pad_w, pad_h])
        valid_grids = valid_grids * scale
        # valid_grids = valid_grids.astype(np.int32)
        valid_grids_crop_img = copy.deepcopy(valid_grids)

        # 得到在原图中点云坐标
        pc_x = (grid_x - cx) * depth / fx
        pc_y = (grid_y - cy) * depth / fy
        points = np.stack([pc_x, pc_y, depth], axis=-1)
        valid_points = points[valid_object_mask]

        # 得到img与crop_img之间的位姿关系
        # 第一个图到第二个图的坐标变换矩阵
        # pt_ind = 10000
        # img_show = cv2.circle(img_show, (valid_grids_img[pt_ind][0], valid_grids_img[pt_ind][1]),  5, (255, 0, 0))
        # img_crop_pad = cv2.circle(img_crop_pad, (valid_grids_crop_img[pt_ind][0], valid_grids_crop_img[pt_ind][1]), 5, (255, 0, 0))
        #
        # plt.subplot(2, 1, 1)
        # plt.imshow(img_show)
        #
        # plt.subplot(2, 1, 2)
        # plt.imshow(img_crop_pad)
        # plt.show()
        # exit(1)

        # correct_R, correct_t = get_rt_between_img_and_crop_img(img, cam_k, valid_grids_img, img_crop_pad, crop_K, valid_grids_crop_img)
        correct_R, correct_t = get_rt_from_imgs(img, cam_k, img_crop_pad, crop_K)

        print(correct_R, correct_t)

        # 将相机坐标中的点云转换到crop相机中
        # valid_points_crop = correct_R @ valid_points + correct_t
        # valid_points_crop = np.dot(valid_points, correct_R.T) + correct_t

        transformation_matrix = np.hstack((correct_R, correct_t.reshape(3, 1)))
        homogeneous_points = np.hstack((valid_points, np.ones((valid_points.shape[0], 1))))
        valid_points_crop = np.dot(transformation_matrix, homogeneous_points.T).T

        # 投影为深度图
        # 初步结论，采用对极约束的方式，由于截取图像并不是遵循物理世界获取的图像，导致计算的位姿并不正确
        # 重投影回来的结果有偏差
        img_crop_pad_depth = project_points_to_image_opencv(img_crop_pad, valid_points_crop, crop_K)


        plt.subplot(3, 1, 1)
        plt.imshow(img_crop_pad)

        img_crop_pad[valid_grids[:, 1].astype(np.int32), valid_grids[:, 0].astype(np.int32)] = (255, 0, 0)
        plt.subplot(3, 1, 2)
        plt.imshow(img_crop_pad)

        plt.subplot(3, 1, 3)
        plt.imshow(img_crop_pad_depth)


        plt.show()


        # plt.subplot(3, 1, 1)
        # plt.imshow(img)
        #
        # plt.subplot(3, 1, 2)
        # plt.imshow(depth)
        #
        #
        # plt.subplot(3, 1, 3)
        # plt.imshow(mask)
        # plt.show()
        print("fffff")
        exit(1)


if __name__ == "__main__":
    print("STart")
    # 通过对极约束的方式将归一化空间的mesh转到原图的相机坐标系中
    get_crop_depth()
    print("End")
