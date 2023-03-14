#!/bin/sh
python train_linemod.py --config_path config_run/LM_ViT_b_16_split1.json
python train_linemod.py --config_path config_run/LM_ViT_b_16_split2.json
python train_linemod.py --config_path config_run/LM_ViT_b_16_split3.json
