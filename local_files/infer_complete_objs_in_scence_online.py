import os
import sys

import cv2
import copy
import torch
import colorsys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os.path
import pyrealsense2 as rs
import numpy as np
import cv2
import shutil
import time
# import trimesh

sys.path.insert(0, "/home/pxn-lyj/Egolee/programs/splatter-image-liyj")

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
from detic_infer import Detic

# sys.path.insert(0, "/home/pxn-lyj/Egolee/programs/FastSAM_liyj")
# from fastsam import FastSAM, FastSAMPrompt

# sys.path.insert(0, "/home/pxn-lyj/Egolee/programs/FastSAM_liyj")
# from fastsam import FastSAM, FastSAMPrompt


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


def get_object_boxes_maskes(img, predictions, score=0.5):
    height, width = img.shape[:2]
    default_font_size = int(max(np.sqrt(height * width) // 90, 10))
    boxes = predictions["pred_boxes"].astype(np.int64)
    scores = predictions["scores"]
    classes_id = predictions["pred_classes"].tolist()
    masks = predictions["pred_masks"].astype(np.uint8)
    num_instances = len(boxes)
    print('detect', num_instances, 'instances')
    obj_masks = []
    obj_boxes = []
    f_mask = np.zeros(len(masks))
    for i in range(num_instances):
        if scores[i] < score:
            continue

        x0, y0, x1, y1 = boxes[i]
        b_h = y1 - y0
        b_w = x1 - x0
        # filter big object
        big_scale = 0.7
        if b_h > big_scale * height or b_w > big_scale * big_scale * width:
            continue
        #
        # # filter small object
        small_scale = 0.1
        if b_h < small_scale * height or b_w < small_scale * width:
            continue
        obj_masks.append(masks[i])
        obj_boxes.append([x0, y0, x1, y1])     # [x1,y1,x2,y2]
        f_mask[i] = 1

    f_mask = f_mask > 0
    predictions["pred_boxes"] = predictions["pred_boxes"][f_mask]
    predictions["scores"] = predictions["scores"][f_mask]
    predictions["pred_classes"] = predictions["pred_classes"][f_mask]
    predictions["pred_masks"] = predictions["pred_masks"][f_mask]
    return predictions


def inside_nms(preds, threshold=0.8):
    boxes = preds['pred_boxes']
    scores = preds['scores']
    classes = preds['pred_classes']
    masks = preds['pred_masks']
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # 按得分降序排序
    indices = np.argsort(areas)[::-1]
    selected_indices = []

    while len(indices) > 0:
        # 选择当前得分最高的框
        current_index = indices[0]
        selected_indices.append(current_index)

        # 计算当前框与后续框的重叠情况
        current_box = boxes[current_index]
        x1 = np.maximum(current_box[0], boxes[indices, 0])
        y1 = np.maximum(current_box[1], boxes[indices, 1])
        x2 = np.minimum(current_box[2], boxes[indices, 2])
        y2 = np.minimum(current_box[3], boxes[indices, 3])

        # 计算重叠区域
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        overlap_area = w * h

        # 计算当前框与其他框的面积
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        other_areas = (boxes[indices, 2] - boxes[indices, 0]) * (boxes[indices, 3] - boxes[indices, 1])

        # 计算 IOU
        # iou = overlap_area / (current_area + other_areas - overlap_area)
        iou = overlap_area / other_areas

        # 选择 IOU 小于阈值的框
        indices = indices[np.where(iou <= threshold)[0]]

    preds["pred_boxes"] = preds["pred_boxes"][selected_indices]
    preds["scores"] = preds["scores"][selected_indices]
    preds["pred_classes"] = preds["pred_classes"][selected_indices]
    preds["pred_masks"] = preds["pred_masks"][selected_indices]
    return preds


def infer_complete_objs_in_scence_online():
    print("infer_complete_objs_in_scence_online")
    pipeline = rs.pipeline()
    config = rs.config()
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    # enable laser emitter or not
    depth_sensor = device.query_sensors()[0]
    emitter = depth_sensor.get_option(rs.option.emitter_enabled)
    print("emitter = ", emitter)

    set_emitter = 1.0
    depth_sensor.set_option(rs.option.emitter_enabled, set_emitter)
    emitter1 = depth_sensor.get_option(rs.option.emitter_enabled)
    print("new emitter = ", emitter1)

    fps = 15
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, fps)

    config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, fps)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

    serial_number = None
    if serial_number is not None:
        config.enable_device(serial_number)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    clipping_distance_in_meters = 1  # 1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale
    align_to = rs.stream.color
    align = rs.align(align_to)

    # profile_depth =profile.get_stream(rs.stream.depth)
    profile_color = profile.get_stream(rs.stream.color)
    intr = profile_color.as_video_stream_profile().get_intrinsics()
    cam_K_color = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    colors = generate_colors(20)

    count = 1
    # fast_sam_model = FastSAM('/home/pxn-lyj/Egolee/programs/FastSAM_liyj/local_files/weights/FastSAM-x.pt')
    detic_model = Detic("/home/pxn-lyj/Egolee/programs/splatter-image-liyj/local_files/Detic_C2_R50_640_4x_in21k.onnx", confThreshold=0.4)

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

    s_root = "./tmp"
    if os.path.exists(s_root):
        shutil.rmtree(s_root)
    os.mkdir(s_root)

    s_sematic_pt_root = os.path.join(s_root, "pcs_sematic")
    os.mkdir(s_sematic_pt_root)

    try:
        while True:
            t1 = time.time()
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            count += 1
            if not aligned_depth_frame or not color_frame or count < 20:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data()) / 1e3
            color_image = np.asanyarray(color_frame.get_data())
            depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)

            depth = depth_image_scaled
            depth[(depth < 0.1) | (depth >= np.inf)] = 0
            depth[depth >= 2.0] = 0

            color_image = color_image[:, :, ::-1]
            rs_pts, rs_pts_colors = image_and_depth_to_point_cloud(color_image, depth, fx=cam_K_color[0, 0], fy=cam_K_color[1, 1],
                                                              cx=cam_K_color[0, 2], cy=cam_K_color[1, 2], max_depth=2.0)

            # color_image = cv2.imread("/home/pxn-lyj/Egolee/programs/splatter-image-liyj/local_files/tmp1/61_color.jpg")
            # depth = np.load("/home/pxn-lyj/Egolee/programs/splatter-image-liyj/local_files/tmp1/61_depth.npy")
            image_show = copy.deepcopy(color_image)

            preds = detic_model.detect(color_image)

            preds = get_object_boxes_maskes(color_image, preds, score=0.4)
            preds = inside_nms(preds)

            image_show = detic_model.draw_predictions(image_show, preds)

            obj_boxes = preds["pred_boxes"]
            obj_masks = preds["pred_masks"]

            # everything_results = fast_sam_model(color_image[:, :, ::-1], device="cuda:0", retina_masks=True, imgsz=1024, conf=0.6, iou=0.01,)
            # prompt_process = FastSAMPrompt(color_image[:, :, ::-1], everything_results, device="cuda:0")
            # s_color_path = os.path.join(s_root, str(count) + "_color.jpg")
            # s_depth_path = os.path.join(s_root, str(count) + "_depth.npy")
            #
            # cv2.imwrite(s_color_path, color_image)
            # np.save(s_depth_path, depth)

            if len(obj_boxes) > 0:
                # print("obj_boxes:", len(obj_boxes))
                # anns = prompt_process.box_prompt(bboxes=obj_boxes)
                # print("anns:", len(anns))
                # prompt_process.plot(annotations=anns, output_path='./tmp/dog.jpg', )

                complete_ojbs = []
                complete_ojbs_color = []
                complete_ojbs_sematic = []
                for obj_id, object_mask in enumerate(obj_masks):
                    object_mask = object_mask > 0
                    obj_color = colors[obj_id % len(colors)]

                    image_show[object_mask] = np.array(obj_color) * 0.3 + image_show[object_mask]*0.7
                    if np.sum(object_mask) < 20:
                        continue

                    obj_mask = object_mask

                    radius = 10
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                    obj_mask = cv2.morphologyEx(obj_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                    obj_mask = obj_mask > 0


                    crop_x, crop_y, crop_w, crop_h = get_crop_and_expand_rect_by_mask(obj_mask)
                    if crop_x == None:
                        continue

                    # mask有可能是分开的，选择最大的mask部分
                    max_part_mask = np.zeros_like(obj_mask)
                    max_part_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = 1
                    obj_mask = np.bitwise_and(obj_mask, max_part_mask)

                    img_crop = color_image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :].copy()

                    mask_crop = obj_mask[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
                    img_crop[~mask_crop] = (255, 255, 255)

                    img_crop_pad, crop_pad_h, crop_pad_w = pad_and_resize_img(img_crop, foreground_ratio=0.65)
                    img_crop_pad, crop_cam_k, crop_pad_scale = resize_to_128_with_K(img_crop_pad, fov=49.0)

                    # plt.imshow(img_crop_pad[:, :, ::-1])
                    # plt.show()

                    image = to_tensor(img_crop_pad).to(device)
                    reconstruction_unactivated = model(
                        image.unsqueeze(0).unsqueeze(0),
                        view_to_world_source,
                        rot_transform_quats,
                        None,
                        activate_output=False)

                    export_to_obj(reconstruction_unactivated, "./tmp/tmp.ply")
                    pc_pred = get_view_object_points(reconstruction_unactivated, view_to_world_source)

                    valid_pcs, img_uvs, crop_img_uvs = get_depth_points(depth, obj_mask, cam_K_color, crop_x, crop_y,
                                                                        crop_pad_h, crop_pad_w, crop_pad_scale)

                    _, pred_uvz = project_points_to_image_opencv(img_crop_pad, pc_pred, crop_cam_k)
                    pred_xyz = get_xyz_by_uvz(pred_uvz, crop_cam_k)
                    filter_pred_uvz = filter_uvz_by_distance(pred_uvz, scale=3.0)
                    filter_pred_xyz = get_xyz_by_uvz(filter_pred_uvz, crop_cam_k)

                    gt_indx, pred_indx = get_match_uv_between_gt_and_pred(img_crop_pad, crop_img_uvs, filter_pred_uvz)
                    match_valid_pcs = valid_pcs[gt_indx]
                    match_filter_pred_xyz = filter_pred_xyz[pred_indx]

                    if len(match_filter_pred_xyz) > 20 and len(match_valid_pcs) > 20:
                        scale, R, t = calculate_scale_and_rt(match_filter_pred_xyz, match_valid_pcs, iter_tims=5)
                        match_filter_pred_xyz = scale * (match_filter_pred_xyz @ R.T) + t

                        pred_xyz = scale * (pred_xyz @ R.T) + t
                        complete_ojbs.append(pred_xyz)
                        complete_ojbs_color.append(np.array([obj_color] * len(pred_xyz)))

                        obj_sematic = np.array([obj_id] * len(pred_xyz)).reshape(-1, 1)
                        # obj_sematic[~pred_indx] = 0
                        complete_ojbs_sematic.append(obj_sematic)

                if len(complete_ojbs) > 0:
                    all_complete_ojbs = np.concatenate(complete_ojbs, axis=0)
                    all_complete_ojbs_color = np.concatenate(complete_ojbs_color, axis=0)
                    all_complete_ojbs_sematic = np.concatenate(complete_ojbs_sematic, axis=0)
                    s_ply_path = os.path.join(s_root, str(count) + ".ply")
                    save_2_ply(s_ply_path, all_complete_ojbs[:, 0], all_complete_ojbs[:, 1], all_complete_ojbs[:, 2],
                               all_complete_ojbs_color.tolist())

                    s_ply_path = os.path.join(s_root, str(count) + "_rs.ply")
                    save_2_ply(s_ply_path, rs_pts[:, 0], rs_pts[:, 1], rs_pts[:, 2],
                               rs_pts_colors.tolist())

                    s_npy_path = os.path.join(s_sematic_pt_root, str(count) + "_pt_sematic.npy")
                    np.save(s_npy_path, np.concatenate([all_complete_ojbs, all_complete_ojbs_color, all_complete_ojbs_sematic], axis=1))

                    s_color_path = os.path.join(s_root, str(count) + "_color.jpg")
                    s_color_vis_path = os.path.join(s_root, str(count) + "_color_vis.jpg")
                    s_depth_path = os.path.join(s_root, str(count) + "_depth.npy")

                    cv2.imwrite(s_color_path, color_image)
                    cv2.imwrite(s_color_vis_path, image_show)
                    np.save(s_depth_path, depth)

            # pcs_sematic = np.concatenate([all_complete_ojbs, all_complete_ojbs_color, complete_ojbs_sematic], axis=1)
            # s_npy_path = os.path.join(s_sematic_root, img_name.replace("_color.jpg", "_sematic.npy"))
            # np.save(s_npy_path, pcs_sematic)

            # cv2.namedWindow("image")
            # cv2.imshow("image", color_image.astype(np.uint8))
            cv2.imshow("image", image_show)
            cv2.waitKey(1)
            # plt.imshow(image_show[:, :, ::-1])
            # plt.show()
            # exit(1)

    finally:
        pipeline.stop()


def show_img():
    img = cv2.imread("/home/pxn-lyj/Egolee/programs/FastSAM_liyj/local_files/output/dog.jpg")
    cv2.imshow("img", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    print("Start")
    # infer_complete_objs_in_scence()
    infer_complete_objs_in_scence_online()
    # show_img()
    print("End")
