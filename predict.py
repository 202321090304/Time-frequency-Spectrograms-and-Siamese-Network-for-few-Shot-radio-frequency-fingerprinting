import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from rf_dataset import InferenceDataset, SPDataset
from networks.resnet_big import CustomCNN, CustomCNNmini, sp_LinearClassifier, sp_MLPClassifier


class InferenceOptions:
    def __init__(self):
        # self.val_data_folder = '/disk/datasets/rf_data/newspectrum/UAV/secondUAVSet/test'
        # self.encode_ckpt = 'save/newSupCon/sp_models/tranSupCon_sp_CustomCNNmini_lr_0.05_decay_0.0001_bsz_16_temp_0.2_trial_0_cosine/ckpt_epoch_140.pth'
        # self.classifier_ckpt = 'save/SecondStage/sp_models/new_best_classifier_93.73.pth'
        # self.batch_size = 32
        # self.num_workers = 8
        # self.model = 'CustomCNNmini'
        # self.classifier = 'MLP'
        # self.mode = 'data'
        self.val_data_folder = r'E:\RFcode\rf数据集\secondUAVSet\test'
        self.encode_ckpt = 'save/ckpt_epoch_140.pth'
        self.classifier_ckpt = 'save/SecondStage/sp_models/new_best_classifier_93.73.pth'
        self.batch_size = 8
        self.num_workers = 2
        self.model = 'CustomCNNmini'
        self.classifier = 'MLP'
        self.mode = 'data'

def set_model_for_inference(opt):
    if opt.model == 'CustomCNN':
        model = CustomCNN()
    elif opt.model == 'CustomCNNmini':
        model = CustomCNNmini()
    else:
        raise ValueError(f"Unsupported model: {opt.model}")

    if opt.classifier == 'linear':
        classifier = sp_LinearClassifier(num_classes=5)
    elif opt.classifier == 'MLP':
        classifier = sp_MLPClassifier(num_classes=5)
    else:
        raise ValueError(f"Unsupported classifier: {opt.classifier}")

    encode_ckpt = torch.load(opt.encode_ckpt, map_location='cpu')
    state_dict = encode_ckpt['model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    classifier_ckpt = torch.load(opt.classifier_ckpt, map_location='cpu')
    classifier.load_state_dict(classifier_ckpt)

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        model.encoder = torch.nn.DataParallel(model.encoder)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier

def inference(val_loader, model, classifier):
    model.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            features = model.encoder(images)
            outputs = classifier(features)

            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = (all_preds == all_labels).mean()
    print(f'Inference Accuracy: {accuracy * 100:.2f}%')

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)

    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"Class {i} Accuracy: {acc * 100:.2f}%")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    save_path = os.path.join('figures', 'confusion_matrix.png')

    plt.savefig(save_path)
    plt.close()

    print(f'Confusion matrix saved to {save_path}')

def predict(val_loader, model, classifier):
    model.eval()
    classifier.eval()

    class_names = ["UAV1", "UAV2", "UAV3", "UAV4", "UAV5"]

    prediction_counts = {class_name: 0 for class_name in class_names}

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)

            features = model.encoder(images)
            outputs = classifier(features)

            _, preds = torch.max(outputs, 1)

            for pred in preds:
                class_index = pred.item()
                if 0 <= class_index < len(class_names):
                    predicted_class = class_names[class_index]
                    prediction_counts[predicted_class] += 1
                else:
                    print(f"Warning: Predicted class index {class_index} is out of range.")

    print("\n预测结果统计：")
    for class_name, count in prediction_counts.items():
        print(f"类别 '{class_name}' 的预测数量: {count}")

if __name__ == '__main__':
    opt = InferenceOptions()
    val_transform = transforms.Compose([
        transforms.CenterCrop((500, 500)),
        transforms.ToTensor(),
    ])

    if opt.mode == 'data':
        val_dataset = SPDataset(data_dir=opt.val_data_folder, transform=val_transform, data_type='test')
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True
        )
        model, classifier = set_model_for_inference(opt)
        inference(val_loader, model, classifier)
    elif opt.mode == 'predict':
        val_dataset = InferenceDataset(data_dir=opt.val_data_folder, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True
        )
        model, classifier = set_model_for_inference(opt)
        predict(val_loader, model, classifier)
