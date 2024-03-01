# GeneFace++ TEST

## 环境安装 （假设已配置过GeneFace, 见GeneFace的TEST.md）

```bash
# 需要 pytorch 2.0.1 (2.1 会报错)
sudo pip3.9 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# MMCV安装
sudo pip3.9 install openmim==0.3.9
sudo mim install mmcv==2.1.0 # 使用mim来加速mmcv安装

sudo pip3.9 install mediapipe
sudo pip3.9 install pyloudnorm
```



## May 推理测试

```bash
PYTHONPATH=./ python3.9 inference/genefacepp_infer.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=infer_outs/may_demo.mp4
```
