from scene.gaussian_predictor import SongUNet
import torch

def debug_song_unet():
    # SongUNet
    in_h, in_w = 128, 128
    in_channels = 3
    out_channels = 10
    song_unet = SongUNet(in_h, in_channels, out_channels)

    in_img = torch.randn(1, 2, in_channels, in_h, in_w)

    B = in_img.shape[0]
    N_views = in_img.shape[1]
    in_img = in_img.reshape(B * N_views, *in_img.shape[2:])

    out_feat = song_unet(in_img, N_views_xa=N_views)
    print(out_feat.shape)

    # attention_resolutions: 控制进行att的分辨率
    # 128 64 32 16 如果特征的分辨率在attention_resolutions，相应的UNetBlock的会进行att的操作
    # self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
    # 在encode 16 进行了4个unetblock的attention
    # 在deencode进行了2个unetblock的attention


if __name__ == "__main__":
    print("Start")
    debug_song_unet()
    print("End")