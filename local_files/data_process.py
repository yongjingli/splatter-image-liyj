import numpy as np
import math
import cv2
import torch
import open3d as o3d
import matplotlib.pyplot as plt


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


def show_proj_img(img, uvz, color=[0, 255, 0]):
    img_show = img.copy()
    mask_img = np.zeros_like(img, dtype=np.uint8)
    img_h, img_w, _ = img.shape

    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    x = np.clip(np.round(u).astype(np.int32), 0, img_w - 1)
    y = np.clip(np.round(v).astype(np.int32), 0, img_h - 1)
    mask_img[y, x, :] = color
    mask = np.sum(mask_img, axis=2) != 0

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    # mask = cv2.dilate(mask.astype(np.uint8), kernel)
    # mask = mask > 0
    # img_show[mask] = img_show[mask] * 0.1 + mask_img[mask] * 0.9
    img_show[mask] = color
    return img_show


def crop_cam_k(fov):
    fov = fov * np.pi / 180
    fx = 128 / (2 * math.tan(fov / 2))
    fy = fx
    cx = 128 / 2
    cy = 128 / 2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    return K

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


def project_points_2_img():
    img_path = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/crop_depth/20_color_crop_pad.jpg"
    ply_path = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/crop_depth/obj_pts.ply"

    fov = 49.134342641202636
    cam_k = crop_cam_k(fov)

    img = cv2.imread(img_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points)

    depth_map, uvz = project_points_to_image_opencv(img, pts, cam_k)

    plt.subplot(3, 2, 1)
    plt.imshow(img)

    plt.subplot(3, 2, 2)
    plt.imshow(depth_map)

    plt.show()


def debug_torch_bmm():
    def transform_point_cloud_to_world_bmm(point_cloud, transformation_matrix):
        # transformation_matrix = torch.linalg.inv(transformation_matrix)

        # 将点云转换为齐次坐标 [N, 4]
        N = point_cloud.shape[0]
        homogeneous_points = torch.ones((N, 4), device=point_cloud.device)
        homogeneous_points[:, :3] = point_cloud  # 前3列为点云坐标

        # 将齐次坐标形状调整为 [N, 1, 4]
        homogeneous_points = homogeneous_points.unsqueeze(0)

        # 使用 bmm 进行批量矩阵乘法
        transformed_points = torch.bmm(homogeneous_points, transformation_matrix.unsqueeze(0))
        # 进来的矩阵看看确认是不是从world到view的？  如果是下面进行.T的操作结果就是正确的
        # transformed_points = torch.bmm(homogeneous_points, transformation_matrix.T.unsqueeze(0))

        # 返回前三列，即世界坐标
        return transformed_points[0, :, :3]

    def transform_point_cloud_to_world_bmm2(point_cloud, transformation_matrix):
        # 将点云转换为齐次坐标
        n = point_cloud.shape[0]
        homogeneous_point_cloud = torch.cat((point_cloud, torch.ones(n, 1)), dim=1)  # [n, 4]

        # 使用 torch.bmm 进行批量矩阵乘法
        # 需要将 T 扩展为 (1, 4, 4) 的形状
        T_expanded = transformation_matrix.unsqueeze(0)  # [1, 4, 4]

        # 进行变换，注意要先扩展点云坐标的维度
        world_point_cloud = torch.bmm(T_expanded, torch.transpose(homogeneous_point_cloud, 1, 0).unsqueeze(0))  # [1, 4, n]

        # 转置结果并去掉最后一个维度
        world_point_cloud = world_point_cloud.squeeze(0).permute(1, 0)[:, :3]  # [n, 4]
        return world_point_cloud

    def transform_point_cloud_to_world_matmul(point_cloud, transformation_matrix):
        # 将点云转换为齐次坐标 [N, 4]
        N = point_cloud.shape[0]
        homogeneous_points = torch.ones((N, 4), device=point_cloud.device)
        homogeneous_points[:, :3] = point_cloud  # 前3列为点云坐标

        # 使用 matmul 直接进行矩阵乘法
        transformed_points = homogeneous_points @ transformation_matrix.T  # [N, 4]

        # 返回前三列，即世界坐标
        return transformed_points[:, :3]  # [N, 3]


    def transform_point_cloud_to_world_gaussian(point_cloud, transformation_matrix):
        rots = transformation_matrix[:3, :3]
        trans = transformation_matrix[:3, 3]

        N = point_cloud.shape[0]
        homogeneous_points = torch.ones((N, 4), device=point_cloud.device)
        homogeneous_points[:, :3] = point_cloud  # 前3列为点云坐标

        # 下面的几个结果都是等价的
        # transformed_points = torch.transpose(torch.bmm(rots.unsqueeze(0), torch.transpose(point_cloud, 1, 0).unsqueeze(0))[0], 1, 0) + trans
        # transformed_points = torch.bmm(point_cloud.unsqueeze(0), rots.T.unsqueeze(0))[0] + trans
        # transformed_points = torch.transpose(rots @  torch.transpose(point_cloud, 1, 0), 1, 0) + trans
        # transformed_points = point_cloud @ rots.T + trans
        # transformed_points = point_cloud @ rots.t() + trans
        transformed_points = point_cloud @ rots.inverse() + trans

        # transformed_points = (homogeneous_points @ transformation_matrix.T)[:, :3]
        # transformed_points = torch.transpose((transformation_matrix @ torch.transpose(homogeneous_points, 1, 0)), 1, 0)[:, :3]
        return transformed_points



    # 创建一个示例点云 [N, 3]
    point_cloud = torch.tensor([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0],
                                [5, 5, 5],
                                [6, 6, 6]], dtype=torch.float32)

    # 创建一个示例变换矩阵 [4, 4]
    # transformation_matrix = torch.tensor([[1.0, 0.0, 0.0, 1.0],
    #                                       [0.0, 1.0, 0.0, 2.0],
    #                                       [0.0, 0.0, 1.0, 3.0],
    #                                       [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    # 创建一个示例变换矩阵 [4, 4]
    transformation_matrix = torch.tensor([[1.0, 0.0, 0.0, 1.0],
                                          [0.0, 2.0, 0.0, 3.0],
                                          [0.0, 0.0, 1.0, 5.0],
                                          [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)

    # 定义旋转角度（以弧度为单位）
    theta_x = np.pi / 4  # 绕x轴旋转45度
    theta_y = np.pi / 4  # 绕y轴旋转45度
    theta_z = np.pi / 4  # 绕z轴旋转45度

    # 绕x轴的旋转矩阵
    R_x = torch.tensor([[1, 0, 0],
                        [0, np.cos(theta_x), -np.sin(theta_x)],
                        [0, np.sin(theta_x), np.cos(theta_x)]])

    # 绕y轴的旋转矩阵
    R_y = torch.tensor([[np.cos(theta_y), 0, np.sin(theta_y)],
                        [0, 1, 0],
                        [-np.sin(theta_y), 0, np.cos(theta_y)]])

    # 绕z轴的旋转矩阵
    R_z = torch.tensor([[np.cos(theta_z), -np.sin(theta_z), 0],
                        [np.sin(theta_z), np.cos(theta_z), 0],
                        [0, 0, 1]])

    # 组合旋转（先绕z轴，再绕y轴，再绕x轴）
    R = R_x @ R_y @ R_z
    transformation_matrix[:3, :3] = R

    # 判断矩阵是否为旋转矩阵 矩阵的转置等于其逆矩阵，即 R转置 @ R等于单位矩阵
    # np.allclose(np.dot(R.T, R), np.eye(R.shape[0]))
    # torch.allclose(source_cameras_view_to_world[:, :3, :3], torch.eye(3).unsqueeze(0).cuda(), atol=1e-06)

    # point_cloud = point_cloud.cuda()
    # transformation_matrix = transformation_matrix.cuda()

    # 使用 bmm 方法转换到世界坐标(本repo的实现方法)
    world_coordinates_bmm = transform_point_cloud_to_world_bmm(point_cloud, transformation_matrix)
    print("World Coordinates (bmm):")
    print(world_coordinates_bmm)    # 这部分似乎会丢了平移

    # 采用左乘的方式，结果正确
    world_coordinates_bmm2 = transform_point_cloud_to_world_bmm2(point_cloud, transformation_matrix)
    print("World Coordinates (bmm2):")
    print(world_coordinates_bmm2)

    # 使用 matmul 方法转换到世界坐标
    world_coordinates_matmul = transform_point_cloud_to_world_matmul(point_cloud, transformation_matrix)
    print("World Coordinates (matmul):")
    print(world_coordinates_matmul)

    # 从gaussian-splatting-liyj看bmm的实现是这样的，感觉在splatter-image中的实现就是不对的，需要进一步确认repo的中T的来源
    # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    # 实现还不对，需要确认
    world_coordinates_gaussian = transform_point_cloud_to_world_gaussian(point_cloud, transformation_matrix)
    print("World Coordinates (gaussian):")
    print(world_coordinates_gaussian)



    # 检查结果是否一致
    are_equal = torch.allclose(world_coordinates_bmm, world_coordinates_matmul, atol=1e-6)
    print("Are results equal?", are_equal)

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

def convert_data_jz():
    # ply_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/plys/00445_pc.ply"
    ply_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/plys_realsense/00445_pc.ply"
    img_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/colors/00445_color.jpg"
    cam_k_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/cam_k.txt"
    mash_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/foreground_instances.png"

    pc = o3d.io.read_point_cloud(ply_path)
    pc = np.asarray(pc.points)
    img = cv2.imread(img_path)

    cam_k = np.loadtxt(cam_k_path)
    depth, uvz = project_points_to_image_opencv(img, pc, cam_k)

    mask = cv2.imread(mash_path)

    pts, colors = image_and_depth_to_point_cloud(img, depth, fx=cam_k[0, 0], fy=cam_k[1, 1],
                                                      cx=cam_k[0, 2], cy=cam_k[1, 2], max_depth=3.0)
    # s_ply_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/plys/00445_2.ply"
    # save_2_ply(s_ply_path, pts[:, 0], pts[:, 1], pts[:, 2], colors.tolist())

    s_depth_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/depths/00445.npy"
    np.save(s_depth_path, depth)

    # s_img_mask = np.zeros_like(mask[:, :, 0], dtype=np.uint8)
    # unique_colors = np.unique(mask.reshape(-1, img.shape[2]), axis=0)
    # for i, rgb_value in enumerate(unique_colors):
    #     img_mask = np.all(mask == rgb_value, axis=-1).astype(np.uint8)
    #     print(rgb_value, i)
    #     s_img_mask[img_mask!=0] = i + 1
    #
    #     plt.imshow(s_img_mask)
    #     plt.show()
    #
    # s_mask_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/masks_num/00445.npy"
    # np.save(s_mask_path, s_img_mask)
    # print(np.unique(s_img_mask))


def read_objaverse():
    npy_path = "/home/pxn-lyj/Egolee/data/objaverse_data/views_release/34eebb66d54b467888d446206bfe6ddf/004.npy"
    img_path = "/home/pxn-lyj/Egolee/data/objaverse_data/views_release/34eebb66d54b467888d446206bfe6ddf/004.png"

    npy = np.load(npy_path)
    print("ff")


if __name__ == "__main__":
    print("Start")
    # project_points_2_img()
    # debug_torch_bmm()
    # convert_data_jz()
    read_objaverse()
    print("End")



