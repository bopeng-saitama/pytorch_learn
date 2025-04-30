import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time
import os
import numpy as np
from tqdm import tqdm

# ResNet实现部分
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3卷积，带padding"""
    return nn.Conv2d(
        in_planes, 
        out_planes, 
        kernel_size=3, 
        stride=stride, 
        padding=dilation, 
        groups=groups, 
        bias=False, 
        dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock只支持groups=1和base_width=64')
        if dilation > 1:
            raise NotImplementedError("BasicBlock不支持dilation > 1")
            
        # 第一个卷积层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        
        # 1x1卷积降维
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        # 3x3卷积
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        # 1x1卷积升维
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        # 第三个卷积块
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 每个元素指示是否在该层使用空洞卷积替代步长
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation应为None或长度为3的列表")
            
        self.groups = groups
        self.base_width = width_per_group
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # 全局平均池化和全连接分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # 零初始化每个残差分支中的最后一个BN
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """构建残差块序列"""
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        # 如果需要改变维度或步长，创建下采样层
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            
        layers = []
        # 添加第一个块（可能有下采样）
        layers.append(block(
            self.inplanes, planes, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer
        ))
        
        # 更新输入平面数
        self.inplanes = planes * block.expansion
        
        # 添加其余块
        for _ in range(1, blocks):
            layers.append(block(
                self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer
            ))
            
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        # 初始层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 残差层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # 分类层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
        
    def forward(self, x):
        return self._forward_impl(x)

# 创建不同ResNet变体的函数
def resnet18(num_classes=1000, pretrained=False):
    """构建ResNet-18模型"""
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    return model

def resnet34(num_classes=1000, pretrained=False):
    """构建ResNet-34模型"""
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet50(num_classes=1000, pretrained=False):
    """构建ResNet-50模型"""
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model

def resnet101(num_classes=1000, pretrained=False):
    """构建ResNet-101模型"""
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    return model

def resnet152(num_classes=1000, pretrained=False):
    """构建ResNet-152模型"""
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    return model

# 训练和验证函数
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc="训练", ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条信息
        current_loss = running_loss / (total / labels.size(0))
        current_acc = 100. * correct / total
        pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    print(f'训练 - 平均损失: {epoch_loss:.4f}, 准确率: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(dataloader, desc="验证", ncols=100)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条信息
            current_loss = running_loss / (total / labels.size(0))
            current_acc = 100. * correct / total
            pbar.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.2f}%'})
    
    val_loss = running_loss / len(dataloader)
    val_acc = 100. * correct / total
    print(f'验证 - 平均损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%')
    
    return val_loss, val_acc

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_acc = 0.0
    
    print(f"\n开始训练，共{num_epochs}个epochs")
    for epoch in range(num_epochs):
        start_time = time.time()
        print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
        
        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印epoch结果
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} 完成, 用时: {epoch_time:.2f}秒')
        print(f'学习率: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet_model.pth')
            print(f'最佳模型已保存，验证准确率: {best_acc:.2f}%')
    
    print('\n训练完成!')
    return model

# 主函数
def main():
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数 - 小批量，少的epoch用于测试
    batch_size = 32
    num_epochs = 2
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    
    # 测试数据集大小 - 每个类别只取少量样本用于快速测试
    samples_per_class = 100  # 每个类别100张图片
    
    # 数据预处理
    print("设置数据预处理...")
    # 对于CIFAR-10，使用较小的图像尺寸以加快处理速度
    train_transform = transforms.Compose([
        transforms.Resize(64),  # 缩小尺寸加快训练
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    print("加载数据集...")
    full_train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                       download=True, transform=train_transform)
    full_val_dataset = datasets.CIFAR10(root='./data', train=False, 
                                     download=True, transform=val_transform)
    
    # 创建子集索引
    # 为每个类别选择样本数量
    train_indices = []
    val_indices = []
    
    # CIFAR-10有10个类别
    for class_id in range(10):
        # 获取该类别的所有索引
        class_indices = np.where(np.array(full_train_dataset.targets) == class_id)[0]
        val_class_indices = np.where(np.array(full_val_dataset.targets) == class_id)[0]
        
        # 随机选择指定数量的样本
        selected_train = np.random.choice(class_indices, samples_per_class, replace=False)
        selected_val = np.random.choice(val_class_indices, samples_per_class//5, replace=False)
        
        train_indices.extend(selected_train)
        val_indices.extend(selected_val)
    
    # 创建数据集子集
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=2)
    
    print(f"数据加载完成。训练子集: {len(train_dataset)}张图像, 验证子集: {len(val_dataset)}张图像")
    
    # 创建ResNet模型
    print("创建ResNet-18模型...")  # 使用ResNet-18加快测试
    model = resnet18(num_classes=10)  # CIFAR-10有10个类别
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                         momentum=momentum, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    # 执行训练
    trained_model = train(model, train_loader, val_loader, criterion, 
                        optimizer, scheduler, num_epochs, device)
    
    # 保存最终模型
    print("保存最终模型...")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': trained_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'final_resnet_model.pth')
    
    print("训练过程完成！最终模型已保存。")

# 确保主函数被调用
if __name__ == "__main__":
    main()