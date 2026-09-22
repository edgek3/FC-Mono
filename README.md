# FC-Mono

This repository provides the official PyTorch implementation of   
**FC-MONO: FREQUENCY-COMPLEMENTARY KNOWLEDGE DISTILLATION FOR
LIGHTWEIGHT SELF-SUPERVISED MONOCULAR DEPTH ESTIMATION**.

The codebase supports training, evaluation, and inference for monocular depth estimation models described in our paper.

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

## ⚙️Setup

Detailed environment requirements are provided in `environment.yml`, including the Python version, CUDA-compatible PyTorch dependencies, and other required packages for reproducing our experiments.

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5 pillow==9.5.0 imageio==2.35.1

pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8

pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image

pip install mmcv-full==1.3.0 mmsegmentation==0.11.0  

pip install timm einops IPython
```
We ran our experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.8 and Ubuntu 20.04. 

Note that our code is built based on [Monodepth2](https://github.com/nianticlabs/monodepth2). 

## 

## 💾KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.



You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.

## ⏳Training

Pre-trained MonoViT weights are available at [here](https://github.com/zxcqlf/MonoViT) 

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

**Monocular training:**

```shell

python train.py --model_name train  --num_epochs 30 --scheduler_step_size 25 --batch_size 8 --num_layers 18  --encoder_mobilevit xs --decoder_channel_scale  100 
             
```



## 📊KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
```
We assume that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:

```shell
python evaluate_depth_KITTI.py --load_weights_folder ~/tmp/mono_model/models/weights/ --decoder_channel_scale 100 --encoder_mobilevit xs --eval_mono
```


## 📊Make3D evaluation
```shell
python evaluate_depth_Make3D.py --load_weights_folder ~/tmp/mono_model/models/weights/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```


## 💾Latency Evaluation
The latency is averaged over 300 runs.

```shell
python evaluate_latency.py --load_weights_folder ~/tmp/mono_model/models/weights/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```

## Model Parameters and FLOPs Evaluation
```shell
python evaluate_flops_params.py --load_weights_folder ~/tmp/mono_model/models/weights/ \
    --encoder_mobilevit xs \
    --decoder_channel_scale 100
```


## Pretrained weights

The pretrained weights (MonoViT_M_640x192 and MPViT-small ) are not included in this repository due to their large file size. Please download them manually from the links below and place them in the following paths: 


```text 

# MobileViT-v1 pretrained weights
pretrained_weight/mobilevitv1/ 

# MonoViT_M_640x192
pretrained_weight/MonoViT_M_640x192/ 

# MPViT-small pretrained weights
pretrained_weight/mpvit_small.pth        

```
This download link is provided by the official MonoViT GitHub repository.  
[MonoViT weights](https://drive.google.com/drive/folders/1VWDPuqiMPDD2P--Oka-yJgh8z7ouCX4D?usp=sharing)

This download link is provided by the official MPViT GitHub repository.

[MPViT-small weights](https://dl.dropboxusercontent.com/s/y3dnmmy8h4npz7a/mpvit_small.pth)

  
## License

The code is released under the license specified in the `LICENSE` file of this repository.

## Acknowledgement

We would like to thank the authors of the following works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[MViTDepth](https://github.com/mengmengbi/MViTDepth)

[MobileViTv1](https://github.com/apple/ml-cvnets) 

[MonoViT](https://github.com/zxcqlf/MonoViT)

[MPViT](https://github.com/youngwanLEE/MPViT)


