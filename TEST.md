# GeneFace++ TEST

## 环境安装 （假设已配置过GeneFace, 见GeneFace的TEST.md）

```bash
## 需要 pytorch 2.1 + CUDA 11.8
sudo pip3.9 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

## pytorch3d (https://github.com/facebookresearch/pytorch3d)
tar xvfz pytorch3d-0.7.6.tar.gz
cd pytorch3d-0.7.6 && sudo pip3.9 install -e .

## 安装 torch-ngp
sudo bash docs/prepare_env/install_ext.sh

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


# 训练

# 训练 Head NeRF 模型
CUDA_VISIBLE_DEVICES=0 python3.9 tasks/run.py --config=egs/datasets/News/lm3d_radnerf_sr.yaml --exp_name=motion2video_nerf/News_head --reset

# 训练 Torso NeRF 模型
CUDA_VISIBLE_DEVICES=0 python3.9 tasks/run.py --config=egs/datasets/News/lm3d_radnerf_torso_sr.yaml --exp_name=motion2video_nerf/News_torso --hparams=head_model_dir=checkpoints/motion2video_nerf/News_head --reset


# 推理

# --debug 选项可以可视化一些中间过程与特征
CUDA_VISIBLE_DEVICES=0 python3.9 inference/genefacepp_infer.py --head_ckpt= --torso_ckpt=checkpoints/motion2video_nerf/News_torso --drv_aud=data/raw/val_wavs/MacronSpeech.wav --out_name=infer_outs/News_demo.mp4
```



### Pretrained models

```
~/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth
~/.cache/torch/hub/checkpoints/vgg_face_dag.pth
```
