import math
import torch


def debug_tranpose():
    # (1) (2) 再在readCamerasFromTxt里面
    # (1) srn 读取数据集, 文件记录的是c2w
    # cam_name = pose_paths[idx]
    # # SRN cameras are camera-to-world transforms
    # # no need to change from SRN camera axes (x right, y down, z away)
    # # it's the same as COLMAP (x right, y down, z forward)
    # c2w = np.loadtxt(cam_name, dtype=np.float32).reshape(4, 4)

    # (2) 返回的是w2c的R T,其中R是经过转置的
    # # get the world-to-camera transform and set R, T
    # w2c = np.linalg.inv(c2w)
    # R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    # T = w2c[:3, 3]

    # (3) 得到数据集的输出到模型的 world_view_transform view_world_transform
    # 得到上述的w2c的R T
    # for cam_info in cam_infos:
    #     R = cam_info.R
    #     T = cam_info.T
    #

    # getWorld2View2为了得到w2c的矩阵，为什么不直接用上面的R T, 是为了在上面的基础上加入相机的平移和scale
    # 需要先转换为c2w, 可以看出在求逆之前对R.transpose()进行转置了, 就是为了将上述保存的转置的R转换回来
    # 得到c2w后，进行求逆得到加入平移和scale的w2c
    # 在得到w2c后, 在外层进行.transpose(0, 1) 就是进行转置的操作, 实际上得到的 world_view_transform为 world_view_transform为.T
    # 这样是为了在进行矩阵乘法的时候进行可以右乘以这个矩阵

    # world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)

    # 同上
    # view_world_transform = torch.tensor(getView2World(R, T, trans, scale)).transpose(0, 1)

    # def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    #     Rt = np.zeros((4, 4))
    #     Rt[:3, :3] = R.transpose()
    #     Rt[:3, 3] = t
    #     Rt[3, 3] = 1.0
    #
    #     C2W = np.linalg.inv(Rt)
    #     cam_center = C2W[:3, 3]
    #     cam_center = (cam_center + translate) * scale
    #     C2W[:3, 3] = cam_center
    #     Rt = np.linalg.inv(C2W)
    #     return np.float32(Rt)

    # def getView2World(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    #     Rt = np.zeros((4, 4))
    #     Rt[:3, :3] = R.transpose()
    #     Rt[:3, 3] = t
    #     Rt[3, 3] = 1.0
    #
    #     C2W = np.linalg.inv(Rt)
    #     cam_center = C2W[:3, 3]
    #     cam_center = (cam_center + translate) * scale
    #     C2W[:3, 3] = cam_center
    #     Rt = C2W
    #     return np.float32(Rt)

    # (4) get_source_camera_v2w_rmo_and_quats
    # 在进行推理的时候需要得到不同视角下的视图生成
    # source_camera也进行了.transpose(0, 1)的操作,与数据训练时保存一致, 也是进行右乘
    # source_cv2wT_quat在转换为四元素之前进行了.transpose(0, 1)的操作
    # 从名称里可以看出T的注明, source_cv2wT_quat为左乘的操作

    # source_camera = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop)[0]
    # source_camera = torch.from_numpy(source_camera).transpose(0, 1).unsqueeze(0)
    # qs = []
    # for c_idx in range(source_camera.shape[0]):
    #     qs.append(matrix_to_quaternion(source_camera[c_idx, :3, :3].transpose(0, 1)))

    # def forward(self, x,
    #             source_cameras_view_to_world,
    #             source_cv2wT_quat=None,
    #             focals_pixels=None,
    #             activate_output=True):

    a = torch.randn(4, 4)
    a_T = a.T
    a_trans = a.transpose(0, 1)  # 实现了转置的作用
    print(a)
    print(a_T)
    print(a_trans)


def debug_proj_cam_center():
    print("debug_proj_cam_center")
    # 在得到world_view_transform和view_world_transform的基础上(右乘)
    # 得到full_proj_transform 和 camera_center
    # full_proj_transform 实际上是左乘了c2w的矩阵, 从相机到世界，应该也是合理的
    # 因为projection_matrix为针对相机的
    # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    # 等价于view_world_transform.T[:3, 3],实际为c2w(左乘)中的T
    # camera_center = world_view_transform.inverse()[3, :3]

def debug_relative_cal():
    print("debug_relative_cal")

    # def make_poses_relative_to_first(self, images_and_camera_poses):
    #     inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
          # inverse_first_camera 实际上为 c12w.T
    #     for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
    #         # c12w.T @ w2cn.T = w2c1 @  w2cn.T = w2c1 @ cn2w = cn2c1 = c12cn.T
    #         images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
    #                                             inverse_first_camera.unsqueeze(0),
    #                                             images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)

              # cn2w.T @ c12w.T.T = cn2w.T @ c12w = w2cn @ c12w = c12cn = cn2c1.T
    #         images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
    #                                             images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
    #                                             inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)

              # 左乘c12w.T 实际上为左乘 w2c1符合预期转到世界坐标, proj为做成v2w
              # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

    #         images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
    #                                             inverse_first_camera.unsqueeze(0),
    #                                             images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)

              # 等价于view_world_transform.T[:3, 3],实际为c2w(左乘)中的T
    #         images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]


def get_source_target_transform():
    print("get_source_target_transform")
    # view_to_world_source 为输入到模型的c2w, 进行了转置，实际为 c2w
    # view_to_world_source, rot_transform_quats = get_source_camera_v2w_rmo_and_quats()
    # source_camera = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop)[0]
    # source_camera = torch.from_numpy(source_camera).transpose(0, 1).unsqueeze

    # 进行新视图生成时的target-transform
    # world_view_transforms, full_proj_transforms, camera_centers = get_target_cameras()
    #     target_cameras = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop,
    #                                       max_elevation=np.pi/4,
    #                                       elevation_freq=1.5)

          # target_cameras实际为c2w
    #     world_view_transforms = []
    #     view_world_transforms = []
    #     camera_centers = []

         # 实际得到world_view_transforms为w2c.T 与实际产生数据时一致
         # Gaussian rasterizer中输入的矩阵实际为w2c.T
    #
    #     for loop_camera_c2w_cmo in target_cameras:
    #         view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1)
    #         world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1)
    #         camera_center = view_world_transform[3, :3].clone()
    #
    #         world_view_transforms.append(world_view_transform)
    #         view_world_transforms.append(view_world_transform)
    #         camera_centers.append(camera_center)

    # 在NoPoSplat中 extrinsics为c2w, extrinsics.inverse()为w2c, 而后面的rearrange会进一步设置为w2c.T
    # view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    from einops import einsum, rearrange, repeat

    # 从上面的验证可以看出Gaussian rasterizer中的viewmatrix=world_view_transform,为w2c.T
    a = torch.randn(1, 4, 4)
    b = rearrange(a, "b i j -> b j i" )
    c = a[0].T
    print(b[0], b[0].shape)
    print(c, c.shape)


if __name__ == "__main__":
    print("Start")

    # 发邮件询问bmm的问题
    # Hi, Stanislaw. Thank you for your great works of splatter-image[https://github.com/szymanowiczs/splatter-image], and it helps me a lot.
    # But I have some questions about the transform of source_cameras_view_to_world in line 758 in gaussian_predictor.py.
    # According to my understanding, pos = torch.bmm(pos, source_cameras_view_to_world) which convert pos_in_cam to pos_in_world,
    # it should pos @ source_cameras_view_to_world.T or source_cameras_view_to_world @ pos?
    # May be you are too busy to answer the question in issues in github, it would be grateful for me if you can give me the answer.
    # Best wishes.
    # 得到的回复是
    # Hi @yongjingli - the cameras are in row major order, that is they assume that the position vectors are row vectors (N x 3).
    # In this convention pos_world = torch.bmm(pos, source_cameras_view_to_world).
    # See the readme for more information about the cameras, and you can find a .transpose() in the dataset files. Hope this helps to clarify it.

    # 在readme中相机详细讲解是
    # Gaussian rasterizer assumes row-major order of rigid body transform matrices, i.e.
    # that position vectors are row vectors. It also requires cameras in the COLMAP / OpenCV convention, i.e.,
    # that x points right, y down, and z away from the camera (forward).

    # 如果开发者习惯使用行向量与矩阵相乘的数学约定，可能会倾向于行优先存储；
    # 如果习惯使用列向量与矩阵相乘的方式，可能会使用列优先存储。
    # 针对上面的解释，pos采用的行向量优先的方式,所以右边的转换矩阵采用.T的方式会比较好
    # 所以转换矩阵的保存尽量都是给出.T的方式, 在数据加载的过程需要留意.T的过程

    # 查看数据加载过程.t的操作
    # debug_tranpose()

    # 计算投影矩阵以及相机中心
    # debug_proj_cam_center()

    # 计算相对关系 以第一帧为世界坐标
    # debug_relative_cal()

    # 产生source 以及target的转换矩阵
    get_source_target_transform()

    print("End")
