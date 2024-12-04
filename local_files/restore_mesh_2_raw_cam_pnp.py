import copy
import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import open3d as o3d


def calculate_scale_and_rt(points_a, points_b, iter_tims=5):
    # 得到从a转到b的参数
    # transformed_points = scale * (points_a @ R.T) + t

    # 进行逐渐优化
    for i in range(iter_tims):
        # 计算质心
        centroid_a = np.mean(points_a, axis=0)
        centroid_b = np.mean(points_b, axis=0)

        # 中心化点
        centered_a = points_a - centroid_a
        centered_b = points_b - centroid_b

        # 计算协方差矩阵
        H = centered_a.T @ centered_b

        # SVD分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = Vt.T @ U.T

        # 确保旋转矩阵的正确性
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        # 计算尺度
        scale = np.sum(S) / np.sum(centered_a**2)

        # 计算平移向量
        t = centroid_b - scale * (R @ centroid_a)

        points_a_trans = scale * (points_a @ R.T) + t
        pts_err = np.linalg.norm(points_a_trans - points_b, axis=1)
        # print(np.min(pts_err), np.max(pts_err))
        # epsilon = 0.01
        threshold_index = int(0.9 * len(pts_err))
        threshold_value = np.partition(pts_err, threshold_index)[threshold_index]

        # 去除误差大的点
        filtered_indices = np.where(pts_err <= threshold_value)[0]
        points_a = points_a[filtered_indices]
        points_b = points_b[filtered_indices]

    return scale, R, t


def save_2_ply(file_path, x, y, z, color=None):
    points = []
    if color == None:
        color = [[255, 255, 255]] * len(x)
    for X, Y, Z, C in zip(x, y, z, color):
        points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, C[2], C[1], C[0]))

    # for X, Y, Z, C in zip(x, y, z, color):
    #     points.append("%f %f %f %d %d %d 0\n" % (Z, X, Y, C[0], C[1], C[2]))

    file = open(file_path, "w")
    file.write('''ply
          format ascii 1.0
          element vertex %d
          property float x
          property float y
          property float z
          property uchar red
          property uchar green
          property uchar blue
          property uchar alpha
          end_header
          %s
          ''' % (len(points), "".join(points)))
    file.close()


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
    return depth_map, uvz


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


def get_depth_points(depth, mask, cam_k, crop_x, crop_y, crop_pad_h, crop_pad_w, crop_pad_scale):
    # 得到原图中有效的物体点云mask
    valid_object_mask = np.bitwise_and(depth > 0, mask)

    # 有效点云在img中的uv坐标
    grid_x, grid_y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing='xy')
    img_uvs = np.stack([grid_x, grid_y])
    img_uvs = np.transpose(img_uvs, (1, 2, 0))
    img_uvs = img_uvs[valid_object_mask]

    # 有效点云在crop-img中uv坐标
    crop_img_uvs = copy.deepcopy(img_uvs)
    crop_img_uvs = crop_img_uvs - np.array([crop_x, crop_y])
    crop_img_uvs = crop_img_uvs + np.array([crop_pad_w, crop_pad_h])
    crop_img_uvs = crop_img_uvs * crop_pad_scale
    crop_img_uvs = crop_img_uvs.astype(np.int32)

    # 有效点云在原来的相机坐标系中的点云坐标
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]
    pc_x = (grid_x - cx) * depth / fx
    pc_y = (grid_y - cy) * depth / fy
    pcs = np.stack([pc_x, pc_y, depth], axis=-1)
    valid_pcs = pcs[valid_object_mask]
    return valid_pcs, img_uvs, crop_img_uvs


def filter_uvz_by_distance(uvz, scale=2.0):
    # 对于具有相同uv的点选择距离最近的z
    u = copy.deepcopy(uvz[:, 0])
    v = copy.deepcopy(uvz[:, 1])
    z = copy.deepcopy(uvz[:, 2])

    # 进行sale,这样去重的更多
    u = u / scale
    v = v / scale

    u = u.astype(int)
    v = v.astype(int)
    uv_dict = {}
    for i in range(len(uvz)):
        key = (u[i], v[i])
        if key not in uv_dict:
            uv_dict[key] = i  # 记录索引
        else:
            if z[i] < z[uv_dict[key]]:
                uv_dict[key] = i  # 更新为新的最小索引

    min_z_indx = list(uv_dict.values())
    filter_uvz = uvz[min_z_indx]
    return filter_uvz


def get_xyz_by_uvz(uvz, cam_k):
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]

    depth = uvz[:, 2]
    pc_x = (uvz[:, 0] - cx) * depth / fx
    pc_y = (uvz[:, 1] - cy) * depth / fy
    xyz = np.stack([pc_x, pc_y, depth], axis=-1)
    return xyz


def get_match_uv_between_gt_and_pred(img, gt_uv, pred_uv):
    img_h, img_w, _ = img.shape

    gt_uv = gt_uv[:, :2].copy()
    gt_uv = gt_uv.astype(np.int32)

    pred_uv = pred_uv[:, :2].copy()
    pred_uv = pred_uv.astype(np.int32)

    gt_map = np.ones((img_h, img_w), dtype=np.int32) * -1
    pred_map = np.ones((img_h, img_w), dtype=np.int32) * -1

    gt_indx = np.arange(len(gt_uv))
    pred_indx = np.arange(len(pred_uv))

    gt_map[gt_uv[:, 1], gt_uv[:, 0]] = gt_indx
    pred_map[pred_uv[:, 1], pred_uv[:, 0]] = pred_indx

    match_mask = np.bitwise_and(gt_map!=-1, pred_map!=-1)
    match_inds = np.argwhere(match_mask)

    match_gt_indices = gt_map[match_inds[:, 0], match_inds[:, 1]]
    match_pred_indices = pred_map[match_inds[:, 0], match_inds[:, 1]]

    return match_gt_indices, match_pred_indices


def show_proj_img(img, uvz, color=[0, 255, 0]):
    img_show = img.copy()
    mask_img = np.zeros_like(img, dtype=np.uint8)
    img_h, img_w, _ = img.shape

    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    x = np.clip(np.round(u).astype(np.int32), 0, img_w - 1)
    y = np.clip(np.round(v).astype(np.int32), 0, img_h - 1)
    mask_img[y, x, :] = color
    mask = np.sum(mask_img, axis=2) != 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask.astype(np.uint8), kernel)
    mask = mask > 0
    # img_show[mask] = img_show[mask] * 0.1 + mask_img[mask] * 0.9
    img_show[mask] = color
    return img_show


def get_mesh_from_pts(pts):
    # 创建 Open3D 点云对象
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pts)

    # 估计法线
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 使用 Poisson 重建生成网格
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    return mesh


def get_depth_by_xyz(img, pcs, cam_k):
    _, pred_uvz = project_points_to_image_opencv(img, pcs, cam_k)
    filter_pred_uvz = filter_uvz_by_distance(pred_uvz, scale=1.0)
    u = filter_pred_uvz[:, 0].astype(np.int32)
    v = filter_pred_uvz[:, 1].astype(np.int32)
    z = filter_pred_uvz[:, 2]

    depth = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    depth[v, u] = z
    return depth

def image_and_depth_to_point_cloud(image, depth, fx, fy, cx, cy, max_depth=5.0):
    rows, cols = depth.shape
    u, v = np.meshgrid(range(cols), range(rows))
    z = depth
    # 将深度为 0 或小于等于某个极小值的点标记为无效
    invalid_mask = np.bitwise_or(np.bitwise_or(z <= 0, z < np.finfo(np.float32).eps), z > max_depth)
    x = np.where(~invalid_mask, (u - cx) * z / fx, 0)
    y = np.where(~invalid_mask, (v - cy) * z / fy, 0)
    # z = z[~invalid_mask]
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = image.reshape(-1, 3)

    points = points[~invalid_mask.reshape(-1)]
    colors = colors[~invalid_mask.reshape(-1)]

    return points, colors



def image_and_depth_to_point_clouds(image, depth0, depth1, fx, fy, cx, cy, max_depth=5.0):
    rows, cols = depth0.shape
    u, v = np.meshgrid(range(cols), range(rows))
    z0 = depth0
    z1 = depth1

    invalid_mask0 = np.bitwise_or(np.bitwise_or(z0 <= 0, z0 < np.finfo(np.float32).eps), z0 > max_depth)
    invalid_mask1 = np.bitwise_or(np.bitwise_or(z1 <= 0, z1 < np.finfo(np.float32).eps), z1 > max_depth)
    invalid_mask = np.bitwise_or(invalid_mask0, invalid_mask1)

    x0 = np.where(~invalid_mask, (u - cx) * z0 / fx, 0)
    y0 = np.where(~invalid_mask, (v - cy) * z0 / fy, 0)
    x1 = np.where(~invalid_mask, (u - cx) * z1 / fx, 0)
    y1 = np.where(~invalid_mask, (v - cy) * z1 / fy, 0)

    points0 = np.stack([x0, y0, z0], axis=-1).reshape(-1, 3)
    points0 = points0[~invalid_mask.reshape(-1)]

    points1 = np.stack([x1, y1, z1], axis=-1).reshape(-1, 3)
    points1 = points1[~invalid_mask.reshape(-1)]

    colors = image.reshape(-1, 3)
    colors = colors[~invalid_mask.reshape(-1)]

    return points0, points1, colors


def align_two_depths(img, depth_0, depth_1, cam_k, min_depth=0.1, max_depth=5.0):
    valid_mask0 = np.bitwise_and(depth_0 > min_depth, depth_0 < max_depth)
    # valid_mask1 = np.bitwise_and(depth_1 > min_depth, depth_1 < max_depth)
    # valid_mask = np.bitwise_and(valid_mask0, valid_mask1)
    valid_mask = valid_mask0
    depth_0[~valid_mask] = 0
    depth_1_raw = copy.deepcopy(depth_1)
    depth_1[~valid_mask] = 0

    pts1_raw, colors_raw = image_and_depth_to_point_cloud(img, depth_1_raw, fx=cam_k[0, 0], fy=cam_k[1, 1], cx=cam_k[0, 2], cy=cam_k[1, 2], max_depth=10000.0)
    pts0, pts1, colors = image_and_depth_to_point_clouds(img, depth_0, depth_1, fx=cam_k[0, 0], fy=cam_k[1, 1], cx=cam_k[0, 2], cy=cam_k[1, 2], max_depth=5.0)

    scale, R, t = calculate_scale_and_rt(pts1, pts0, iter_tims=5)
    # pts1 = scale * (pts1 @ R.T) + t
    pts1_raw = scale * (pts1_raw @ R.T) + t

    align_depth = get_depth_by_xyz(img, pts1_raw, cam_k)
    return align_depth


def restore_mesh_2_raw_cam_pnp():
    # 得到crop归一化的图像的depth
    # root = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009"
    root = "/home/pxn-lyj/Egolee/data/test/mesh_jz"
    img_root = os.path.join(root, "colors")
    masks_root = os.path.join(root, "masks_num")
    depths_root = os.path.join(root, "depths")

    img_names = [name for name in os.listdir(img_root) if name.endswith(".jpg")]
    img_names = list(sorted(img_names, key=lambda x: int(x.split("_")[0])))

    s_root = os.path.join(root, "pc_predict")
    os.makedirs(s_root, exist_ok=True)

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)
    fx = cam_k[0, 0]
    fy = cam_k[1, 1]
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

        # [2 5 6 7 8 9]
        i = 9
        mask = mask==i

        # 生成模型的图像预处理过程
        # 从图像中截取物体mash部分图像
        crop_x, crop_y, crop_w, crop_h = get_crop_and_expand_rect_by_mask(mask)
        if crop_x==None:
            print("no crop")
            continue

        img_crop = img[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :].copy()
        mask_crop = mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        img_crop[~mask_crop] = (255, 255, 255)

        # 将图像padding为方形, 同时padding设置前景的比例
        img_crop_pad, crop_pad_h, crop_pad_w = pad_and_resize_img(img_crop, foreground_ratio=0.65)
        # 将图像resize为128,同时得到该图像的内参
        img_crop_pad, crop_cam_k, crop_pad_scale = resize_to_128_with_K(img_crop_pad, fov=49.0)
        cv2.imwrite(os.path.join(s_root, img_name.replace("_color.jpg", "_color_crop_pad.jpg")), img_crop_pad)
        # exit(1)

        # 得到有效点云以及在原图和crop图中的uv
        valid_pcs, img_uvs, crop_img_uvs = get_depth_points(depth, mask, cam_k, crop_x, crop_y,
                                                            crop_pad_h, crop_pad_w, crop_pad_scale)

        # 将预测点云投影会crop图像
        # pc_pred_path = os.path.join(s_root, img_name.replace("_color.jpg", "_color_crop_pad_ped_obj.ply"))
        pc_pred_path = os.path.join(s_root, img_name.replace("_color.jpg", "_color_crop_pad_{}_ped_obj.ply".format(i)))
        pc_pred = o3d.io.read_point_cloud(pc_pred_path)
        pc_pred = np.asarray(pc_pred.points)

        _, pred_uvz = project_points_to_image_opencv(img_crop_pad, pc_pred, crop_cam_k)
        pred_xyz = get_xyz_by_uvz(pred_uvz, crop_cam_k)
        filter_pred_uvz = filter_uvz_by_distance(pred_uvz, scale=3.0)
        filter_pred_xyz = get_xyz_by_uvz(filter_pred_uvz, crop_cam_k)

        # s_ply_path = os.path.join(s_root, img_name.replace("_color.jpg", "_color_crop_pad_ped_obj_filter.ply"))
        # save_2_ply(s_ply_path, filter_pred_xyz[:, 0], filter_pred_xyz[:, 1], filter_pred_xyz[:, 2])

        # 得到gt-depth和pred-depth对应indx(落到同一uv)
        gt_indx, pred_indx = get_match_uv_between_gt_and_pred(img_crop_pad, crop_img_uvs, filter_pred_uvz)
        match_valid_pcs = valid_pcs[gt_indx]
        match_filter_pred_xyz = filter_pred_xyz[pred_indx]

        # 将match部分转到原相机坐标
        scale, R, t = calculate_scale_and_rt(match_filter_pred_xyz, match_valid_pcs, iter_tims=5)
        match_filter_pred_xyz = scale * (match_filter_pred_xyz @ R.T) + t

        # 将预测物体转到原相机坐标
        pred_xyz = scale * (pred_xyz @ R.T) + t

        # s_ply_path = os.path.join(s_root, "a.ply")
        # save_2_ply(s_ply_path, match_valid_pcs[:, 0], match_valid_pcs[:, 1], match_valid_pcs[:, 2])
        #
        # s_ply_path = os.path.join(s_root, "b.ply")
        # save_2_ply(s_ply_path, match_filter_pred_xyz[:, 0], match_filter_pred_xyz[:, 1], match_filter_pred_xyz[:, 2])
        #
        s_ply_path = os.path.join(s_root, img_name.replace("_color.jpg", "_align.ply"))
        save_2_ply(s_ply_path, pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2])

        # 将转换后的pred_xyz进行投影验证
        # _, pred_uvz_cam = project_points_to_image_opencv(img, pred_xyz, cam_k)
        # _, pred_uvz_cam = project_points_to_image_opencv(img, valid_pcs, cam_k)
        # img_show = show_proj_img(img, pred_uvz_cam, color=(0, 255, 0))
        # plt.imshow(img_show[:, :, ::-1])
        # plt.show()
        # exit(1)


if __name__ == "__main__":
    print("Start")
    restore_mesh_2_raw_cam_pnp()
    print("End")
