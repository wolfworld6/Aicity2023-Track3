# This part is video feature extraction. Most of the code comes from VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training(https://github.com/MCG-NJU/VideoMAE).

## ➡️ Data Preparation

Use Preprocess part to generate train dataset, which orginazition structure is like the instructions in [DATASET.md](DATASET.md).

## 🔨 Installation

Please follow the instructions in [INSTALL.md](INSTALL.md).

## 🔄 Pre-train Weights

We choose 3 open source pre-trained weights in final competition. Some pre-training weights need to apply for download, so we provide network disk as follows to get these weights directly.
Our network disk provides downloaded weights.

- ViT-H-K400, from https://github.com/MCG-NJU/VideoMAE-Action-Detection/blob/main/MODEL_ZOO.md (model mAP39.5), disk download:https://pan.baidu.com/s/1nPpFb_h3NPZrqRc9kRzsSA with code：ti2x.
.
- ViT-L-hybrid-K700， from https://github.com/OpenGVLab/InternVideo, disk download:https://pan.baidu.com/s/1nPpFb_h3NPZrqRc9kRzsSA with code：ti2x.
- Vit-L-Ego4d-verb-K700, from https://github.com/OpenGVLab/ego4d-eccv2022-solutions, disk download:https://pan.baidu.com/s/1nPpFb_h3NPZrqRc9kRzsSA with code：ti2x.


K400 means Kinects-400 dataset, K700 means Kinetics-700 datasets.

You can download these weights and put them in 'pretrain' folder of feature extrction.

## ⤴️ Fine-tuning models on official data

We use official dataset A1 to finetune models from pretrain weights as mentioned above. 

You can follow these steps to finetune the models by yourself, also we provide six finetuned models here ：https://pan.baidu.com/s/188FUdzi6WdeKBoiYs5w77Q with code：0owl,you can directly download them for feature extraction.

We use :
- 'Track3_finetune_k400_vith.sh' script to finetune ViT-H model with ViT-H-K400 pretrain weight, 
- 'Track3_finetune_ego_vitl.sh' script to finetune ViT-L model with Vit-L-Ego4d-verb-K700 pretrain weight, 
- 'Track3_finetune_k400_vith.sh' script to finetune ViT-L model with ViT-L-hybrid-K700 pretrain weight.


You can refer to [FINETUNE.md](FINETUNE.md) to know how to finetune, or just use scrips we applied above with your real data, model, and output path.


## 👀 Inference to extract video features

We provide two scripts to extract video features for different models. Script 'inference_video_feature_vithK400.py' is use vit-huge model to extract features. Script 'inference_video_feature_vitl.py' using vit-huge model to extract features.

Firstly, to extract video features using ViT-H on rear view and dash view of official videos, you can run:

- ```python inference_video_feature_vithK400.py --ckpt_pth ./weights/k400_vith_rearview.pt --video_dir XXX --output_dir XXX --select_view Rear --device cuda:0```
- ```python inference_video_feature_vithK400.py --ckpt_pth ./weights/K400_vith_dashboard.pt --video_dir XXX --output_dir XXX --select_view Dash --device cuda:0```


Secondly, to extract video features using ViT-L on rear view and dash view of official videos, you can run:

- ```python inference_video_feature_vitl.py  --model_path ./weights/hybrid_k700_vitl_rearview.pt --video_dir XXX --save_dir XXX --view Rear --device cuda:0 ```
- ```python inference_video_feature_vitl.py  --model_path ./weights/hybrid_k700_vitl_dashboard.pt --video_dir XXX --save_dir XXX --view Dash --device cuda:0 ```
- ```python inference_video_feature_vitl.py  --model_path ./weights/ego_verb_vitl_rearview.pt --video_dir XXX --save_dir XXX --view Rear --device cuda:0 ```
- ```python inference_video_feature_vitl.py  --model_path ./weights/ego_verb_vitl_dashboard.pt --video_dir XXX --save_dir XXX --view Dash --device cuda:0 ```


After runing these scripts, video features will be saved in output directory.

We use 3 finetune models to extract 2 views(RearView and Dashboard View) of official videos, so finally we get 3x2 types features.

