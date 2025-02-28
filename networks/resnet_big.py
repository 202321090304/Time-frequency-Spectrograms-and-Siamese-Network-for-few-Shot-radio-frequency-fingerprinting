import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(nn.Module):
    """CNN backbone + projection head"""
    def __init__(self, feat_dim=64):
        super(CustomCNN, self).__init__()
        
        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算展平后的维度（假设输入图像大小为500x500）
        self.flatten_dim = 128 * 125 * 125  # 每次池化尺寸减半

        # 定义头部部分
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, feat_dim)
        )

    def forward(self, x):
        # 编码器部分
        x = self.encoder(x)
        # print(f"Shape after encoder: {x.shape}")  # 添加这一行
        x = x.view(x.size(0), -1)  # 展平操作
        
        # 头部部分
        feat = self.head(x)
        feat = F.normalize(feat, dim=1)  # 正则化输出特征
        return feat

class CustomCNNmini(nn.Module):
    """CNN backbone + projection head"""
    def __init__(self, feat_dim=64):
        super(CustomCNNmini, self).__init__()
        
        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算展平后的维度（假设输入图像大小为500x500）
        self.flatten_dim = 32 * 125 * 125  # 每次池化尺寸减半

        # 定义头部部分
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

    def forward(self, x):
        # 编码器部分
        x = self.encoder(x)
        # print(f"Shape after encoder: {x.shape}")  # 添加这一行
        x = x.view(x.size(0), -1)  # 展平操作
        
        # 头部部分
        feat = self.head(x)
        feat = F.normalize(feat, dim=1)  # 正则化输出特征
        return feat

class CustomCNNminidrop(nn.Module):
    """CNN backbone + projection head"""
    def __init__(self, feat_dim=64, dropout_p=0.3):
        super(CustomCNNminidrop, self).__init__()
        
        # 定义编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 计算展平后的维度（假设输入图像大小为500x500）
        self.flatten_dim = 32 * 125 * 125  # 每次池化尺寸减半

        # 定义头部部分
        self.head = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),  # 添加 Dropout 层
            nn.Linear(128, feat_dim),
        )

    def forward(self, x):
        # 编码器部分
        x = self.encoder(x)
        # print(f"Shape after encoder: {x.shape}")  # 添加这一行
        x = x.view(x.size(0), -1)  # 展平操作
        
        # 头部部分
        feat = self.head(x)
        feat = F.normalize(feat, dim=1)  # 正则化输出特征
        return feat

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class sp_LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, num_classes=5,feat_dim=64):
        super(sp_LinearClassifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes) 

    def forward(self, features):
        return self.fc(features)

class sp_MLPClassifier(nn.Module):
    """MLP classifier"""
    def __init__(self, num_classes=5):
        super(sp_MLPClassifier, self).__init__()
        # 计算展平后的维度（假设输入图像大小为500x500）
        self.flatten_dim = 32 * 125 * 125  # 每次池化尺寸减半

        # 定义头部部分
        self.fc0 = nn.Linear(self.flatten_dim, 128)
        self.relu0 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128, 64)
        self.relu1 = nn.ReLU(inplace=True)       
        self.fc2 = nn.Linear(64,num_classes)

    def forward(self, features):
        features = features.view(features.size(0), -1)
        features = self.fc0(features)
        features = self.relu0(features)
        features = self.fc1(features)
        features = self.relu1(features)
        features = self.fc2(features)
        features = F.normalize(features, dim=1)  # 正则化输出特征
        return features