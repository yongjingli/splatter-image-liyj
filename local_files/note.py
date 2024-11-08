# 第二步的安装需要git repo的方式
# git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive

# 安裝的版本需要下面的版本
# pydantic 1.10.15
# fastapi 0.88.0
# gradio 3.28.0


# pip install pytorch-lightning==2.0
# 將requeirements中的lightning替換爲上面的
# pytorch_lightning和lightning虽然是一个东西，但是import却不一样
# 用pip install lightning安装， 是用import lightning 来导入 lightning
# 用 pip install pytorch_lightning安装，是用 import pytorch_lightning


# 直接pip install ema-pytorch也會卸載torch從而安裝更高版本的，這個需要再看看需要安裝什麼版本
# ema-pytorch 0.5.0  當版本大於0.6.0的時候就需要torch 2.0以上

# 后面遇到类似下面这种问题是不是都可以通过设置CUDA_HOME和CUB_HOME    的方式解决
# subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
# 设置cub-home
# export CUB_HOME=/usr/local/cuda-*/include/
# #/usr/local/cuda-11.6/include
#
# 设置cuda-home
# export CUDA_HOME=/usr/local/cuda-11.6



# 有关projection matrix的解释
# https://github.com/szymanowiczs/splatter-image/issues/17

# 都是以第一帧的相关位姿
# https://github.com/szymanowiczs/splatter-image/issues/16

# NDC坐标
# https://github.com/szymanowiczs/splatter-image/issues/13
# NDC 坐标的范围通常是[−1,1]。
# 对于 x 轴，左侧是 -1，右侧是 1。
# 对于 y 轴，下方是 -1，上方是 1。
# z 轴的值通常在 0 到 1 之间（或 -1 到 1，具体取决于实现），用于深度测试

# shapenet和co3d数据集的区别
# https://github.com/szymanowiczs/splatter-image/issues/19
# shapenet is synthetic dataset, and CO3D contains real world images

# 在对R进行加载保存的时候,需要对R进行转置保存 主要跟库的内存存储方式有关
# R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
# 当将一个矩阵传递到使用不同存储顺序的库（如 CUDA 中的 glm）时，可能需要对矩阵进行转置。这是因为矩阵的行和列在内存中的排列方式不同。
# 例如，如果你从 NumPy 中得到了一个旋转矩阵𝑅，而 glm 需要这个矩阵的列主序版本，那么你就需要对R 进行转置，才能正确地用于后续计算

# wandb.errors.errors.UsageError: api_key not configured (no-tty). call wandb.login(key=[your_api_key])
# 需要注册登录 终端输出 wandb init 然后根据提示登录创建
# https://wandb.ai/ajinglee-guangzhou


# 训练显存不足, 修改default_config.yaml的batch-size配置
#  batch_size: 8
#  batch_size: 2