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
sudo pip3.9 install setproctitle
```



## May 推理测试

```bash
PYTHONPATH=./ python3.9 inference/genefacepp_infer.py --a2m_ckpt=checkpoints/audio2motion_vae --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/may_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=infer_outs/may_demo.mp4
```



## 测试

```bash
# 准备数据
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. bash data_gen/runs/nerf/run.sh News

# 复制`egs/datasets/May`的config文件到 `egs/datasets/{Video_ID}`，
cp -r egs/datasets/May egs/datasets/News
# 修改config里的`video: May` 为 `video: {Video_ID}`

# 训练 Head NeRF 模型
CUDA_VISIBLE_DEVICES=0 python3.9 tasks/run.py --config=egs/datasets/News/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/News_head --reset

# 训练 Torso NeRF 模型
CUDA_VISIBLE_DEVICES=0 python3.9 tasks/run.py --config=egs/datasets/News/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/news_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/News_head --reset
```



### Pretrained models

```
/home/tao/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
/home/tao/.cache/torch/hub/checkpoints/vgg_face_dag.pth
```
