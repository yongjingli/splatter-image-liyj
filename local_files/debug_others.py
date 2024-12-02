import torch
import numpy as np



def get_loop_cameras(num_imgs_in_loop, radius=2.0,
                     max_elevation=np.pi/6, elevation_freq=0.5,
                     azimuth_freq=2.0):

    all_cameras_c2w_cmo = []

    for i in range(num_imgs_in_loop):
        azimuth_angle = np.pi * 2 * azimuth_freq * i / num_imgs_in_loop
        elevation_angle = max_elevation * np.sin(
            np.pi * i * 2 * elevation_freq / num_imgs_in_loop)
        x = np.cos(azimuth_angle) * radius * np.cos(elevation_angle)
        y = np.sin(azimuth_angle) * radius * np.cos(elevation_angle)
        z = np.sin(elevation_angle) * radius

        camera_T_c2w = np.array([x, y, z], dtype=np.float32)

        # in COLMAP / OpenCV convention: z away from camera, y down, x right
        camera_z = - camera_T_c2w / radius
        up = np.array([0, 0, -1], dtype=np.float32)
        camera_x = np.cross(up, camera_z)
        camera_x = camera_x / np.linalg.norm(camera_x)
        camera_y = np.cross(camera_z, camera_x)

        camera_c2w_cmo = np.hstack([camera_x[:, None],
                                    camera_y[:, None],
                                    camera_z[:, None],
                                    camera_T_c2w[:, None]])
        camera_c2w_cmo = np.vstack([camera_c2w_cmo, np.array([0, 0, 0, 1], dtype=np.float32)[None, :]])

        all_cameras_c2w_cmo.append(camera_c2w_cmo)

    return all_cameras_c2w_cmo

def debug_get_source_target_rt(num_imgs_in_loop=200):
    source_camera = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop)[0]
    source_camera = torch.from_numpy(source_camera).transpose(0, 1).unsqueeze(0)
    view_to_world_source = source_camera[0]
    print(view_to_world_source, view_to_world_source.shape)

    target_cameras = get_loop_cameras(num_imgs_in_loop=num_imgs_in_loop,
                                      max_elevation=np.pi / 4,
                                      elevation_freq=1.5)
    world_view_transforms = []
    view_world_transforms = []
    camera_centers = []

    for loop_camera_c2w_cmo in target_cameras:
        view_world_transform = torch.from_numpy(loop_camera_c2w_cmo).transpose(0, 1)
        world_view_transform = torch.from_numpy(loop_camera_c2w_cmo).inverse().transpose(0, 1)
        camera_center = view_world_transform[3, :3].clone()

        world_view_transforms.append(world_view_transform)
        view_world_transforms.append(view_world_transform)
        camera_centers.append(camera_center)

    world_view_transforms_inv = torch.inverse(world_view_transforms[0])
    print(world_view_transforms_inv, world_view_transforms_inv.shape)


def read_lasd_checkpoint():
    # model_path = "/home/liyongjing/programs/splatter-image-liyj/experiments_out/2024-11-19/09-49-52/model_latest.pth"
    model_path = "/home/liyongjing/programs/splatter-image-liyj/experiments_out/2024-11-20/15-09-10/model_latest.pth"
    checkpoint = torch.load(model_path)
    first_iter = checkpoint["iteration"]
    print(first_iter)

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


def save_npy_2_ply():
    npy_path = "/home/pxn-lyj/Egolee/data/创新答辩_liyongjing/objness_graspness/graspable_seed_xyz.npy"
    # npy_path = "/home/pxn-lyj/Egolee/data/创新答辩_liyongjing/objness_graspness/objectness_seed_xyz.npy"
    npy = np.load(npy_path)
    ply_path = npy_path[:-3] + ".ply"
    color = (0, 0, 255)
    colors = np.array([color] * len(npy)).reshape(-1, 3)
    save_2_ply(ply_path, npy[:, 0], npy[:, 1], npy[:, 2], colors.tolist())
    print(npy.shape)


if __name__ == "__main__":
    print("Start")
    # debug_get_source_target_rt()
    # read_lasd_checkpoint()
    save_npy_2_ply()
    print("End")
