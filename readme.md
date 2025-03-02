# Time-frequency Graph and Siamese Network for few-Shot radio frequency fingerprinting
### 0.Dataset And Environment
Dataset directory:/disk/datasets/rf_data/newspectrum/UAV/UAVSet/  
Dataset download link:[https://drive.google.com/file/d/1A1f9BHGIkLmXbup9MODZeJtC-3US5rvd/view?usp=sharing ](https://drive.google.com/file/d/1bN5dEeak0KrSutB9YKIRmkEPPgAIEn61/view?usp=sharing)  
Pre-trained model download link:[[https://drive.google.com/file/d/1wqemE6wSU_d7vJAcfO80IPA6y4h3jZPr/view?usp=sharing ](https://drive.google.com/file/d/1bN5dEeak0KrSutB9YKIRmkEPPgAIEn61/view?usp=sharing) ](https://drive.google.com/file/d/1wqemE6wSU_d7vJAcfO80IPA6y4h3jZPr/view?usp=sharing)  
Conda dependencies can be found in the environment.yml file.

### 1、Pre-training Stage
Pre-training the model using the Siamese network
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

### 1.1 Dimensionality Reduction
Visualize the representation training results using the tSNE dimensionality reduction method
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

### 2.Fine-tuning Stage
Fine-tuning on the target dataset using 10% of the data
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
