
### 0、数据集
数据集目录:/disk/datasets/rf_data/newspectrum/UAV/UAVSet/  
数据集下载地址：https://drive.google.com/file/d/1A1f9BHGIkLmXbup9MODZeJtC-3US5rvd/view?usp=sharing
预训练模型下载地址：https://drive.google.com/file/d/1wqemE6wSU_d7vJAcfO80IPA6y4h3jZPr/view?usp=sharing
conda依赖见environment.yml文件

### 1、一阶段训练
通过孪生网络对模型进行预训练
```bash
nohup python new_main_supcon_Generalization.py --batch_size 16 \
  --model CustomCNNmini --method SupCon\
  --learning_rate 0.05 \
  --temp 0.2  --cosine \
  --data_folder /disk/datasets/rf_data/newspectrum/UAV/UAVSet/train \
  --dataset sp --epochs 200 \
  --save_freq 5 --weight_decay 1e-4 \
  > runlog.txt 2>&1 &  
```

### 1.1 降维
通过tSNE降维方法查看表征训练效果
```bash
python main_tSNE.py \
    --model CustomCNNmini \
    --feature_type all \
    --ckpt save/newSupCon/sp_models/tranSupCon_sp_CustomCNNmini_lr_0.05_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth \
    --data_folder /disk/datasets/rf_data/newspectrum/UAV/UAVSet/train \
    --val_data_folder /disk/datasets/rf_data/newspectrum/UAV/UAVSet/test \
    --batch_size 32 \
    --num_workers 8
```

### 2.二阶段训练
在目标数据集上微调，使用10%的数据
```bash
nohup python new_main_linear.py --batch_size 32 \
  --model CustomCNNmini \
  --classifier MLP \
  --test_batch_size 64 \
  --learning_rate 0.1 \
  --ckpt save/newSupCon/sp_models/tranSupCon_sp_CustomCNNmini_lr_0.05_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth \
  --data_folder /disk/datasets/rf_data/newspectrum/UAV/secondUAVSet/train \
  --val_data_folder /disk/datasets/rf_data/newspectrum/UAV/secondUAVSet/test \
  --dataset sp --split_ratio 0.1 --n_cls 2\
  --epochs 50 > runlog2.txt 2>&1 &
```