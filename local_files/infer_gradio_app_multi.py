import torch
import torchvision
import numpy as np
import os
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
from plyfile import PlyData
import open3d as o3d
import cv2


from utils.app_utils import (
    remove_background,
    resize_foreground,
    set_white_background,
    resize_to_128,
    to_tensor,
    get_source_camera_v2w_rmo_and_quats,
    get_target_cameras,
    export_to_obj)

import imageio
from utils.general_utils import matrix_to_quaternion
from scene.gaussian_predictor import GaussianSplatPredictor
from gaussian_renderer import render_predicted
import rembg


@torch.no_grad()
def infer_gradio_app_multi():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 模型加载
    # model_cfg = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/gradio_config.yaml"
    # model_path = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/checkpoints/gradio_config/model_latest.pth"

    # model_cfg = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/checkpoints/shapenet/config_cars.yaml"
    # model_path = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/checkpoints/shapenet/model_cars.pth"

    model_cfg = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/checkpoints/shapenet_cat_2_imgs_pose_self/cfg_train.yaml"
    model_path = "/home/pxn-lyj/Egolee/programs/splatter-image-liyj/checkpoints/shapenet_cat_2_imgs_pose_self/model_latest.pth"

    model_cfg = OmegaConf.load(model_cfg)

    model = GaussianSplatPredictor(model_cfg)

    ckpt_loaded = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt_loaded["model_state_dict"])
    model.to(device)

    # 图像预处理
    # 去除背景的模块
    rembg_session = rembg.new_session()
    def preprocess(input_image, preprocess_background=True, foreground_ratio=0.65):
        # 0.7 seems to be a reasonable foreground ratio
        if preprocess_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = set_white_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = set_white_background(image)
        image = resize_to_128(image)
        return image

    def get_view_object_points(reconstruction, view_to_world_source):
        valid_gaussians = torch.where(reconstruction["opacity"] > -2.5)[0]
        # xyz = reconstruction["xyz"][valid_gaussians].detach().cpu().clone()

        xyz = reconstruction["xyz"][valid_gaussians].detach().clone()
        xyz = torch.cat([xyz, torch.ones((xyz.shape[0], 1)).to(xyz.device)], dim=-1)
        # xyz = torch.linalg.inv(view_to_world_source[0][0]) @ torch.transpose(xyz, 1, 0)
        # view_to_world_source输入进来的并不是严格的旋转矩阵，所以用逆而不用.T, 这里将错就错恢复回去，在模型推理中的从view到世界的公式是错的
        xyz = torch.bmm(xyz.unsqueeze(0), torch.linalg.inv(view_to_world_source)[0])
        xyz = xyz[0][:, :3].cpu().numpy()

        return xyz

    # 模型推理与重建
    def reconstruct_and_export_multi(image1, image2, c2_T_c1, ply_out_path):
        """
        Passes image through model, outputs reconstruction in form of a dict of tensors.
        """
        image1 = to_tensor(image1).to(device)
        image2 = to_tensor(image2).to(device)
        c2_T_c1 = torch.tensor(c2_T_c1).to(device)

        view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
        view_to_world_source = view_to_world_source.to(device)
        rot_transform_quats = rot_transform_quats.to(device)

        multi_images = torch.concatenate([image1.unsqueeze(0), image2.unsqueeze(0)], dim=0)

        c1_T_w = view_to_world_source[0][0].T
        c2_T_w = c1_T_w @ c2_T_c1
        view_to_world_source2 = c2_T_w.T.unsqueeze(0).unsqueeze(0)
        rot_transform_quats2 = matrix_to_quaternion(view_to_world_source2[0, 0, :3, :3].transpose(0, 1)).unsqueeze(0).unsqueeze(0)

        multi_view_to_world_source = torch.concatenate([view_to_world_source, view_to_world_source2], dim=1)
        multi_rot_transform_quats = torch.concatenate([rot_transform_quats, rot_transform_quats2], dim=1)

        reconstruction_unactivated = model(
            multi_images.unsqueeze(0),
            multi_view_to_world_source,
            multi_rot_transform_quats,
            None,
            activate_output=False)

        reconstruction = {k: v[0].contiguous() for k, v in reconstruction_unactivated.items()}
        reconstruction["scaling"] = model.scaling_activation(reconstruction["scaling"])
        reconstruction["opacity"] = model.opacity_activation(reconstruction["opacity"])

        # render images in a loop
        loop_out_path = os.path.join(os.path.dirname(ply_out_path), "loop.mp4")
        image = image1
        if 1:
            world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
            background = torch.tensor([1, 1, 1] , dtype=torch.float32, device=device)
            loop_renders = []
            t_to_512 = torchvision.transforms.Resize(512, interpolation=torchvision.transforms.InterpolationMode.NEAREST)

            image_raw = image.cpu().numpy().transpose(1,2, 0)

            # 检测输入模型的外参与渲染的外参
            # print("view_to_world_source:", view_to_world_source)
            # print("world_view_transforms[r_idx]", torch.inverse(world_view_transforms[0]))
            for r_idx in range( world_view_transforms.shape[0]):
                image = render_predicted(reconstruction,
                                            world_view_transforms[r_idx].to(device),   #
                                            full_proj_transforms[r_idx].to(device),    # 投影矩阵是有影响的
                                            camera_centers[r_idx].to(device),
                                            background,
                                            model_cfg,
                                            focals_pixels=None)["render"]
                image = t_to_512(image)
                loop_renders.append(torch.clamp(image * 255, 0.0, 255.0).detach().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            imageio.mimsave(loop_out_path, loop_renders, fps=25)

            # 检查渲染出来的图像
            plt.subplot(2, 1, 1)
            plt.imshow(image_raw)

            plt.subplot(2, 1, 2)
            plt.imshow(image.cpu().numpy().transpose(1,2, 0))
            plt.show()

        if 1:
            # export reconstruction to ply
            export_to_obj(reconstruction_unactivated, ply_out_path)

        # obj_pts_path = os.path.join(os.path.dirname(ply_out_path), "obj_pts.ply")
        obj_pts_path = image_path[:-4] + "_ped_obj.ply"
        if 1:
            obj_pts = get_view_object_points(reconstruction_unactivated, view_to_world_source)
            # valid_gaussians = torch.where(reconstruction_unactivated["opacity"] > -2.5)[0]
            # obj_pts = reconstruction_unactivated["xyz_cam"][valid_gaussians]
            save_2_ply(obj_pts_path, obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2])

        return ply_out_path, loop_out_path, obj_pts_path

    def save_ply_2_obj(ply_path, obj_path):
        def point_cloud_to_mesh(pcd):
            # 进行移动最小二乘法曲面重建
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([0.01, 0.02]))
            return mesh

        pcd = o3d.io.read_point_cloud(ply_path)
        mesh = point_cloud_to_mesh(pcd)
        o3d.io.write_triangle_mesh(obj_path, mesh)

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

    # image_path = "/home/pxn-lyj/Downloads/images.jpg"
    # image_path = "/home/pxn-lyj/Egolee/data/test/dexgrasp_show_realsense_20241009/pc_predict/20_color_crop_pad.jpg"
    # image_path = "/home/pxn-lyj/Egolee/data/test/mesh_jz/pc_predict/00445_color_crop_pad_9.jpg"

    # train
    image_path = "/home/pxn-lyj/Egolee/data/shapenet_srn_data/srn_cars/cars_train/1a1dcd236a1e6133860800e6696b8284/rgb/000002.png"
    # image_path2 = image_path
    image_path2 = "/home/pxn-lyj/Egolee/data/shapenet_srn_data/srn_cars/cars_train/1a1dcd236a1e6133860800e6696b8284/rgb/000003.png"

    # test
    # image_path = "/home/pxn-lyj/Egolee/data/shapenet_srn_data/srn_cars/cars_test/1cb95c00d3bf6a3a58dbdf2b5c6acfca/rgb/000034.png"
    # # image_path2 = image_path
    # image_path2 = "/home/pxn-lyj/Egolee/data/shapenet_srn_data/srn_cars/cars_test/1cb95c00d3bf6a3a58dbdf2b5c6acfca/rgb/000000.png"

    image_pose_path = image_path.replace("rgb", "pose").replace(".png", ".txt")
    image_pose_path2 = image_path2.replace("rgb", "pose").replace(".png", ".txt")
    c1_T_w = np.loadtxt(image_pose_path, dtype=np.float32).reshape(4, 4)
    c2_T_w = np.loadtxt(image_pose_path2, dtype=np.float32).reshape(4, 4)
    c2_T_c1 = np.linalg.inv(c1_T_w) @ c2_T_w

    ply_out_path = image_path[:-4] + ".ply"
    obj_out_path = image_path[:-4] + ".obj"
    image = Image.open(image_path)
    image2 = Image.open(image_path2)

    proc_image = preprocess(image)
    proc_image2 = preprocess(image2)

    plt.subplot(2, 1, 1)
    plt.imshow(image)

    plt.subplot(2, 1, 2)
    plt.imshow(proc_image)

    ply_out_path, loop_out_path, obj_pts_path = reconstruct_and_export_multi(np.array(proc_image), np.array(proc_image2), c2_T_c1, ply_out_path)
    # save_ply_2_obj(ply_out_path, obj_out_path)
    print(ply_out_path, obj_out_path, loop_out_path, obj_pts_path)

    s_root = "/home/pxn-lyj/Egolee/data/物体重建算法/cars_2_img_stage1/cond_novel"
    # image.save(os.path.join(s_root, "1.png"))
    proc_image.save(os.path.join(s_root, "3.png"))
    # cv2.imwrite(os.path.join(s_root, "1.jpg"), image.numpy())
    # cv2.imwrite(os.path.join(s_root, "2.jpg"), proc_image.numpy())

    # plt.show()


if __name__ == "__main__":
    print("Start")
    # 多视图
    # gradio_app为交互式的推理显示，修改为输入图像直接输出结果
    infer_gradio_app_multi()
    print("End")