#!/bin/sh
python train_linemod.py --config_path config_run/LM_vit_split1.json
python train_linemod.py --config_path config_run/LM_vit_split2.json
python train_linemod.py --config_path config_run/LM_vit_split3.json
