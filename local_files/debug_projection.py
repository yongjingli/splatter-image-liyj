import math
import torch


def getProjectionMatrix(znear, zfar, fovX, fovY):
    # https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    # 这个解释感觉最详细
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)   # 为0
    P[1, 2] = (top + bottom) / (top - bottom)   # 为0
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# 2.0 * znear / (right - left)                                    (right + left) / (right - left)
#                                 2.0 * znear / (top - bottom)    (top + bottom) / (top - bottom)
#                                                                 z_sign * zfar / (zfar - znear)            -(zfar * znear) / (zfar - znear)
#                                                                 z_sign

def test_projection():
    print("test_projection")
    # 投影矩阵的理解: 提供一个视锥(frustum）的定义， 同时给定 near and far 的数值
    # 投影矩阵 https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ
    # 定义了near位置，宽与near的比值
    # 定义了near位置，长与near的比值   这里就定义了近处的视锥的形状
    proj_matrix = getProjectionMatrix(znear=0.5, zfar=5, fovX=90, fovY=90)
    print(proj_matrix[0, 2], proj_matrix[1, 2])   # 如果相机中心在图像中间,那么这两项为0

    # P[2, 2] P[2, 3] 这两项应该是定义了近处和远处的数值
    # z_sign定义了z轴的方向
    # 这个投影矩阵的具体定义还与rasteriser的具体实现保持一致

    # 详细解释: https://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    # In reality, glFrustum does two things: first it performs perspective projection,
    # and then it converts to normalized device coordinates (NDC).
    # The former is a common operation in projective geometry, while the latter is OpenGL arcana, an implementation detail.
    # A 为proj_matrix[2, 2], B为proj_matrix[2, 3]
    # 投影矩阵得到的结果为 [Zu, Zv, A*Z + B, Z]
    print(proj_matrix[2, 2], proj_matrix[2, 3])
    in_p = torch.tensor([2, 3, 2.35, 1.0]).reshape(1, 4)
    proj_p = proj_matrix @ in_p.T
    proj_p = proj_p.T
    print(proj_p)

    # Map viewports from NDC coordinates to the screen.
    # NDC 将左右上下的坐标归一化为[0, 1]
    # Persp 矩阵将锥形体形状的空间转换为长方体形状，而 glOrtho 将长方体空间转换为规范化的设备坐标。
    # A call to glOrtho(left, right, bottom, top, near, far) constructs the matrix:
    # Proj=NDC∗Persp 包含了NDC和persp投影  NDC归一化到[-1, 1]
    # 目的是为了定义标定好的相机
    # We've seen two different ways to simulate a calibrated camera in OpenGL, one using glFrustum and one using the intrinsic camera matrix directly.

    # 下面是一些资料记录
    # 了解projection_matrix的由来与使用
    # https://github.com/szymanowiczs/splatter-image/issues/17
    # 这个问题里提到两个问题:
    # (1) opengl 中https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ
    # 将      P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    #         P[2, 3] = - 2 * zfar * znear / (zfar - znear)
    # 修改为
    #         P[2, 2] = z_sign * zfar / (zfar - znear)
    #         P[2, 3] = -(zfar * znear) / (zfar - znear)
    # 回答: The calibration matrix was taken directly from Gaussian Splatting repo to ensure compatibility with their rasteriser
    # 从gaussion-splatting中保持一致,与rasteriser兼容
    # https://github.com/graphdeco-inria/gaussian-splatting/blob/d9fad7b3450bf4bd29316315032d57157e23a515/utils/graphics_utils.py#L51

    # (2) 为什么将-1修改为1
    # 与opengl的坐标系不一样，opengl的坐标轴z是向着相机内部的
    # 但是我们使用的坐标轴z是向着外部的
    # z_sign is 1.0 instead of -1.0 because we use a camera convention where z is facing away from the camera
    # (unlike OpenGL where z is facing into the camera).

    # 根据相机内参以及图像的长宽设置投影矩阵
    # Here is the code to obtain the OpenGL projection matrix equivalent to a computer vision camera with camera matrix K=[fx, s, cx; 0, fy, cy; 0, 0, 1] and image size [W, H]:
    # z轴向外
    # glMatrixMode(GL_PROJECTION); // Select The Projection Matrix
    # glLoadIdentity();            // Reset The Projection Matrix
    # GLdouble perspMatrix[16]={2*fx/W,0,0,0,2*s/W,2*fy/H,0,0,2*(cx/W)-1,2*(cy/H)-1,(zmax+zmin)/(zmax-zmin),1,0,0,2*zmax*zmin/(zmin-zmax),0};
    # glMultMatrixd(perspMatrix);

    # z轴向内 相差一个-1
    # glMatrixMode(GL_PROJECTION); // Select The Projection Matrix
    # glLoadIdentity();            // Reset The Projection Matrix
    # GLdouble perspMatrix[16]={2*fx/w,0,0,0,0,2*fy/h,0,0,2*(cx/w)-1,2*(cy/h)-1,-(far+near)/(far-near),-1,0,0,-2*far*near/(far-near),0};
    # glMultMatrixd(perspMatrix);

    # FovX and FovY do not need rescale because they are always there.
    # But when you use the intrinsics K, you need to rescale it according to your new resolution.
    # 采用FOV定义的好处在于可以不用将w和h作为参数输出，当渲染不同的分辨率时可以不用重新计算投影矩阵


    # https://i.sstatic.net/aEwwS.png   通过focal-plane 投影平面进行定义
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml  通过角度和图像长宽比定义

    # Open GL operates with a frustum which is related to perspective projection with some limits in depth referred as near and far values.
    # Imagine a pyramid lying on its side - this is frustum.
    # Another analogy is a projector beam that extends in its width and height with the distance - this is frustum too.
    # So right, left, bottom, and top are your image coordinates while near and far are your depth limits with the near beint your focal plane.
    # OpenGL will put Cx and Cy in the center of the image plane so you can skip them.
    # The alternative and more natural way to specify frustum is based on viewing angle or field of view (50-60 deg is typical);
    # the function you call is glPerspective() where you still have near and far but instead of sizes specify the angle and aspect ratio.
    # Good luck.

    # 采用相机内参作为投影矩阵的讨论:
    # # https://github.com/graphdeco-inria/gaussian-splatting/issues/399
    # 根据这里通过角度和图像长宽比定义 https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/gluPerspective.xml
    # 定义了水平方向与竖直方向的长宽比就可以定义视锥
    # 通过fov和长宽度可以定义焦距
    # fovy: Specifies the field of view angle, in degrees, in the y direction.
    # aspect: Specifies the aspect ratio that determines the field of view in the x direction.
    # The aspect ratio is the ratio of x (width) to y (height).
    # zNear: Specifies the distance from the viewer to the near clipping plane (always positive).
    # zFar: Specifies the distance from the viewer to the far clipping plane (always positive).
    # f =cotangent(fovy/2) 正切的倒数
    # f/aspect 0     0                                  0
    # 0        f     0                                  0
    # 0        0    (zfar + znear)/(znear-zfar)         (zfar * znear)/(znear-zfar)
    # 0        0     -1                                  0


if __name__ == "__main__":
    print("Start")
    test_projection()
    print("End")