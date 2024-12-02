import os
import cv2
import copy
import torch
import colorsys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from restore_mesh_2_raw_cam_pnp import get_crop_and_expand_rect_by_mask, \
    pad_and_resize_img, resize_to_128_with_K, get_depth_points,\
    project_points_to_image_opencv, get_xyz_by_uvz, filter_uvz_by_distance,\
    get_xyz_by_uvz, get_match_uv_between_gt_and_pred, calculate_scale_and_rt, \
    save_2_ply

from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted
import rembg
from omegaconf import OmegaConf
from utils.app_utils import get_source_camera_v2w_rmo_and_quats, to_tensor, export_to_obj


def get_view_object_points(reconstruction, view_to_world_source):
    valid_gaussians = torch.where(reconstruction["opacity"] > -2.5)[0]
    xyz = reconstruction["xyz"][valid_gaussians].detach().clone()
    xyz = torch.cat([xyz, torch.ones((xyz.shape[0], 1)).to(xyz.device)], dim=-1)
    xyz = torch.bmm(xyz.unsqueeze(0), torch.linalg.inv(view_to_world_source)[0])
    xyz = xyz[0][:, :3].cpu().numpy()
    return xyz


def generate_colors(n):
    colors = []
    for i in range(n):
        # 计算色相，范围为 0 到 1
        hue = i / n
        # 饱和度和亮度
        saturation = 1.0
        lightness = 0.5
        # 将 HSL 转换为 RGB
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        # 转换为 0-255 范围
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def get_bg_points(depth, mask, cam_k):
    # 得到原图中有效的物体点云mask
    valid_object_mask = np.bitwise_and(depth > 0, mask)

    # 有效点云在img中的uv坐标
    grid_x, grid_y = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]), indexing='xy')
    img_uvs = np.stack([grid_x, grid_y])
    img_uvs = np.transpose(img_uvs, (1, 2, 0))
    img_uvs = img_uvs[valid_object_mask]

    # 有效点云在原来的相机坐标系中的点云坐标
    fx = cam_k[0, 0]
    fy = cam_k[1, 1],
    cx = cam_k[0, 2]
    cy = cam_k[1, 2]
    pc_x = (grid_x - cx) * depth / fx
    pc_y = (grid_y - cy) * depth / fy
    pcs = np.stack([pc_x, pc_y, depth], axis=-1)
    valid_pcs = pcs[valid_object_mask]
    return valid_pcs



@torch.no_grad()
def infer_complete_objs_in_scence():
    root = "/home/pxn-lyj/Egolee/data/test/mesh_jz"
    img_root = os.path.join(root, "colors")
    masks_root = os.path.join(root, "masks_num")
    depths_root = os.path.join(root, "depths")

    img_names = [name for name in os.listdir(img_root) if name.endswith(".jpg")]
    img_names = list(sorted(img_names, key=lambda x: int(x.split("_")[0])))

    cam_k_path = os.path.join(root, "cam_k.txt")
    cam_k = np.loadtxt(cam_k_path)

    s_root = os.path.join(root, "complete_objs")
    s_sematic_root = os.path.join(root, "pcs_sematic")
    os.makedirs(s_root, exist_ok=True)
    os.makedirs(s_sematic_root, exist_ok=True)

    bg_ids = [1]

    # model
    model_cfg = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/gradio_config.yaml"
    model_path = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/checkpoints/gradio_config/model_latest.pth"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model_cfg = OmegaConf.load(model_cfg)
    model = GaussianSplatPredictor(model_cfg)
    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
    view_to_world_source = view_to_world_source.to(device)
    rot_transform_quats = rot_transform_quats.to(device)

    colors = generate_colors(20)

    for img_name in img_names:
        img_path = os.path.join(img_root, img_name)
        depth_path = os.path.join(depths_root, img_name.replace("_color.jpg", "_depth.npy"))
        mask_path = os.path.join(masks_root, img_name.replace("_color.jpg", "_mask.npy"))

        img = cv2.imread(img_path)
        img_show = copy.deepcopy(img)
        depth = np.load(depth_path)
        mask = np.load(mask_path)

        obj_ids = np.unique(mask)

        complete_ojbs = []
        complete_ojbs_color = []
        complete_ojbs_sematic = []
        for obj_id in obj_ids:
            obj_mask = mask == obj_id
            obj_color = colors[obj_id % len(colors)]

            if obj_id in bg_ids:
                obj_color = (0, 0, 0)
                bg_pcs = get_bg_points(depth, obj_mask, cam_k)
                complete_ojbs.append(bg_pcs)
                complete_ojbs_color.append(np.array([obj_color] * len(bg_pcs)))

                obj_sematic = np.array([0] * len(bg_pcs)).reshape(-1, 1)
                complete_ojbs_sematic.append(obj_sematic)



                print("skip bg id:", obj_id)
                continue

            crop_x, crop_y, crop_w, crop_h = get_crop_and_expand_rect_by_mask(obj_mask)
            if crop_x == None:
                print("skip crop obj id:", obj_id)
                continue

            # mask有可能是分开的，选择最大的mask部分
            max_part_mask = np.zeros_like(obj_mask)
            max_part_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = 1
            obj_mask = np.bitwise_and(obj_mask, max_part_mask)

            img_crop = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :].copy()
            mask_crop = obj_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
            img_crop[~mask_crop] = (255, 255, 255)

            img_crop_pad, crop_pad_h, crop_pad_w = pad_and_resize_img(img_crop, foreground_ratio=0.65)
            img_crop_pad, crop_cam_k, crop_pad_scale = resize_to_128_with_K(img_crop_pad, fov=49.0)

            image = to_tensor(img_crop_pad).to(device)
            reconstruction_unactivated = model(
                image.unsqueeze(0).unsqueeze(0),
                view_to_world_source,
                rot_transform_quats,
                None,
                activate_output=False)

            export_to_obj(reconstruction_unactivated, "./tmp/tmp.ply")
            pc_pred = get_view_object_points(reconstruction_unactivated, view_to_world_source)

            valid_pcs, img_uvs, crop_img_uvs = get_depth_points(depth, obj_mask, cam_k, crop_x, crop_y,
                                                                crop_pad_h, crop_pad_w, crop_pad_scale)

            _, pred_uvz = project_points_to_image_opencv(img_crop_pad, pc_pred, crop_cam_k)
            pred_xyz = get_xyz_by_uvz(pred_uvz, crop_cam_k)
            filter_pred_uvz = filter_uvz_by_distance(pred_uvz, scale=3.0)
            filter_pred_xyz = get_xyz_by_uvz(filter_pred_uvz, crop_cam_k)

            gt_indx, pred_indx = get_match_uv_between_gt_and_pred(img_crop_pad, crop_img_uvs, filter_pred_uvz)
            match_valid_pcs = valid_pcs[gt_indx]
            match_filter_pred_xyz = filter_pred_xyz[pred_indx]

            scale, R, t = calculate_scale_and_rt(match_filter_pred_xyz, match_valid_pcs, iter_tims=5)
            match_filter_pred_xyz = scale * (match_filter_pred_xyz @ R.T) + t

            pred_xyz = scale * (pred_xyz @ R.T) + t
            complete_ojbs.append(pred_xyz)
            complete_ojbs_color.append(np.array([obj_color] * len(pred_xyz)))

            obj_sematic = np.array([obj_id] * len(pred_xyz)).reshape(-1, 1)
            # obj_sematic[~pred_indx] = 0
            complete_ojbs_sematic.append(obj_sematic)

        all_complete_ojbs = np.concatenate(complete_ojbs, axis=0)
        all_complete_ojbs_color = np.concatenate(complete_ojbs_color, axis=0)
        complete_ojbs_sematic = np.concatenate(complete_ojbs_sematic, axis=0)
        s_ply_path = os.path.join(s_root, img_name.replace("_color.jpg", "_complete_objs.ply"))
        save_2_ply(s_ply_path, all_complete_ojbs[:, 0], all_complete_ojbs[:, 1], all_complete_ojbs[:, 2],
                   all_complete_ojbs_color.tolist())

        pcs_sematic = np.concatenate([all_complete_ojbs, all_complete_ojbs_color, complete_ojbs_sematic], axis=1)
        s_npy_path = os.path.join(s_sematic_root, img_name.replace("_color.jpg", "_sematic.npy"))
        np.save(s_npy_path, pcs_sematic)

        # 将 numpy 数组中的点赋值给点云对象
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(all_complete_ojbs)
        # point_cloud.colors = o3d.utility.Vector3dVector(all_complete_ojbs_color/255)
        # o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud", width=800, height=600, left=50,
        #                                   top=50, mesh_show_back_face=True)


if __name__ == "__main__":
    print("Start")
    infer_complete_objs_in_scence()
    print("End")
