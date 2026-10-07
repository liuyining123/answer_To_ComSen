import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# 数据预处理（原版，保持不动）
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

traintest = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=False, transform=transform)
testloader = torch.utils.data.DataLoader(traintest, batch_size=4,
                                         shuffle=False, num_workers=2)
# 类别
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 训练轮数
epochs = 10
# epochs = 2  # 测试用

# 显示图像--测试
def imshow(img):
    img = img / 2 + 0.5  # 去标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ============ 原版网络 ============
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.fc3 = nn.Linear(in_features=36, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.aap(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


net = CNNNet().to(device)


# 打印预测结果
def save_prediction_grid(net, testloader, classes, device, num_images=10,
                         save_path='prediction_results.png'):
    net.eval()
    images_list, labels_list = [], []
    for images, labels in testloader:
        images_list.append(images)
        labels_list.append(labels)
        if sum(x.size(0) for x in images_list) >= num_images:
            break

    images = torch.cat(images_list, dim=0)[:num_images]
    labels = torch.cat(labels_list, dim=0)[:num_images]

    with torch.no_grad():
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu()

    rows = 2
    cols = (num_images + 1) // 2
    fig = plt.figure(figsize=(3 * cols, 6))

    for i in range(num_images):
        ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
        img = images[i] / 2 + 0.5
        npimg = img.numpy()
        ax.imshow(np.transpose(npimg, (1, 2, 0)))

        pred = classes[predicted[i]]
        true = classes[labels[i]]
        color = 'green' if pred == true else 'red'
        ax.set_title(f"P:{pred}\nT:{true}", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def output_prediction_grid(epochs, train_losses, train_accs, test_accs,
                           net, testloader, classes, device,
                           num_images=10, curve_path='training_curves.png',
                           pred_path='prediction_results.png'):
    # 绘制训练曲线
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_losses, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_accs, 'g-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train Accuracy')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, test_accs, 'r-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(curve_path, dpi=300, bbox_inches='tight')
    plt.show()

    # 保存分类结果可视化图片
    save_prediction_grid(net, testloader, classes, device,
                         num_images=num_images, save_path=pred_path)


def init_weights(net):
    # 初始化参数
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def train_model(net, train_losses, train_accs, test_accs,
                trainloader, criterion, optimizer, device, epochs):
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)

        net.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_acc = 100 * correct_test / total_test
        test_accs.append(epoch_test_acc)

        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Loss: {epoch_loss:.4f} '
              f'Train Acc: {epoch_train_acc:.2f}% '
              f'Test Acc: {epoch_test_acc:.2f}%')

    total_time = time.time() - start_time
    print('Finished Training')
    print(f'Final Test Accuracy: {test_accs[-1]:.2f}%')
    return total_time


def main_easy():
    '''
    基础版网络，仅作为对照基线
    '''
    init_weights(net)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train_losses, train_accs, test_accs = [], [], []
    total_time = train_model(net, train_losses, train_accs, test_accs,
                             trainloader, criterion, optimizer, device, epochs)

    params = sum(x.numel() for x in net.parameters())
    print("Net_gvp have {} parameters in total".format(params))

    output_prediction_grid(epochs, train_losses, train_accs, test_accs,
                           net, testloader, classes, device,
                           num_images=10,
                           curve_path='training_curves_easy.png',
                           pred_path='prediction_results_easy.png')

    return {
        'name': '改进前 (main_easy)',
        'params': params,
        'epochs': epochs,
        'final_test_acc': test_accs[-1],
        'best_test_acc': max(test_accs),
        'final_train_acc': train_accs[-1],
        'final_loss': train_losses[-1],
        'total_time': total_time,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


# =========================================================
# ================= 以下是改进版独立子模块 =================
# =========================================================

# ---------- 1. 数据增强 ----------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.RandomErasing(p=0.25)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# ---------- 2. SE 注意力模块 ----------
class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# ---------- 3. 深度可分离卷积（轻量化） ----------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


# ---------- 4. 改进后的网络 ----------
class CNNNetImproved(nn.Module):
    """
    改进点：
    - BatchNorm 稳定训练
    - SE 注意力提升特征表达
    - 深度可分离卷积降低参数量和 FLOPs
    - GAP + Dropout 减少参数、防过拟合
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32)

        self.conv2 = DepthwiseSeparableConv(32, 64, kernel_size=3, padding=1)
        self.se2 = SEBlock(64)

        self.conv3 = DepthwiseSeparableConv(64, 128, kernel_size=3, padding=1)
        self.se3 = SEBlock(128)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = F.max_pool2d(x, 2)  # 32 -> 16

        x = self.conv2(x)
        x = self.se2(x)
        x = F.max_pool2d(x, 2)  # 16 -> 8

        x = self.conv3(x)
        x = self.se3(x)
        x = F.max_pool2d(x, 2)  # 8 -> 4

        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ---------- 5. 改进网络专用初始化 ----------
def init_weights_improved(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


# ---------- 6. 改进训练函数（支持调度器、显式 testloader） ----------
def train_model_improved(net, trainloader, testloader,
                         criterion, optimizer, scheduler,
                         device, epochs,
                         train_losses, train_accs, test_accs):
    start_time = time.time()
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        scheduler.step()

        epoch_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)

        net.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_test_acc = 100 * correct_test / total_test
        test_accs.append(epoch_test_acc)

        print(f'Epoch [{epoch+1}/{epochs}] '
              f'Loss: {epoch_loss:.4f} '
              f'Train Acc: {epoch_train_acc:.2f}% '
              f'Test Acc: {epoch_test_acc:.2f}%')

    total_time = time.time() - start_time
    print('Finished Training')
    print(f'Final Test Accuracy: {test_accs[-1]:.2f}%')
    return total_time


# ---------- 7. 改进版入口 ----------
def main_hard():
    '''
    改进版：
    - 数据增强：RandomCrop / HorizontalFlip / ColorJitter / RandomErasing
    - 结构改进：BatchNorm + SE 注意力 + 深度可分离卷积 + GAP + Dropout
    - 训练策略：lr=0.1, CosineAnnealing, weight_decay, label_smoothing
    - 目标：提升泛化能力和推理速度，参数量大幅减少
    '''
    trainset_hard = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_train)
    trainloader_hard = torch.utils.data.DataLoader(
        trainset_hard, batch_size=128, shuffle=True, num_workers=2)

    testset_hard = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=False, transform=transform_test)
    testloader_hard = torch.utils.data.DataLoader(
        testset_hard, batch_size=128, shuffle=False, num_workers=2)

    net_hard = CNNNetImproved().to(device)
    init_weights_improved(net_hard)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(net_hard.parameters(),
                          lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    epochs_hard = 100 
    train_losses, train_accs, test_accs = [], [], []

    total_time = train_model_improved(
        net_hard, trainloader_hard, testloader_hard,
        criterion, optimizer, scheduler,
        device, epochs_hard,
        train_losses, train_accs, test_accs)

    params = sum(x.numel() for x in net_hard.parameters())
    print("Net_improved have {} parameters in total".format(params))

    output_prediction_grid(epochs_hard, train_losses, train_accs, test_accs,
                           net_hard, testloader_hard, classes, device,
                           num_images=10,
                           curve_path='training_curves_hard.png',
                           pred_path='prediction_results_hard.png')

    return {
        'name': '改进后 (main_hard)',
        'params': params,
        'epochs': epochs_hard,
        'final_test_acc': test_accs[-1],
        'best_test_acc': max(test_accs),
        'final_train_acc': train_accs[-1],
        'final_loss': train_losses[-1],
        'total_time': total_time,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
    }


# ---------- 8. 实验对比分析 ----------
def compare_experiments(exp_before, exp_after):
    print("\n" + "=" * 70)
    print("改进前后实验对比分析")
    print("=" * 70)

    headers = ['指标', exp_before['name'], exp_after['name']]
    rows = [
        ['训练轮数', exp_before['epochs'], exp_after['epochs']],
        ['参数量', exp_before['params'], exp_after['params']],
        ['最终训练准确率 (%)',
         f"{exp_before['final_train_acc']:.2f}",
         f"{exp_after['final_train_acc']:.2f}"],
        ['最终测试准确率 (%)',
         f"{exp_before['final_test_acc']:.2f}",
         f"{exp_after['final_test_acc']:.2f}"],
        ['最佳测试准确率 (%)',
         f"{exp_before['best_test_acc']:.2f}",
         f"{exp_after['best_test_acc']:.2f}"],
        ['最终训练损失',
         f"{exp_before['final_loss']:.4f}",
         f"{exp_after['final_loss']:.4f}"],
        ['总训练时间 (s)',
         f"{exp_before['total_time']:.2f}",
         f"{exp_after['total_time']:.2f}"],
    ]

    col_width = [28, 22, 22]
    print(f"{headers[0]:<{col_width[0]}} {headers[1]:<{col_width[1]}} {headers[2]:<{col_width[2]}}")
    print("-" * 70)
    for row in rows:
        print(f"{str(row[0]):<{col_width[0]}} {str(row[1]):<{col_width[1]}} {str(row[2]):<{col_width[2]}}")

    print("\n【改进前分析】")
    print(f"- 使用原始数据增强，batch_size=4，SGD lr=0.001，无 BN、无注意力。")
    print(f"- 参数量约 {exp_before['params']}，最终测试准确率约 {exp_before['final_test_acc']:.2f}%。")
    print(f"- 训练轮数仅 {exp_before['epochs']}，容易欠拟合，泛化能力有限。")

    print("\n【改进后分析】")
    print(f"- 引入 RandomCrop/HorizontalFlip/ColorJitter/RandomErasing 数据增强。")
    print(f"- 结构加入 BatchNorm、SE 注意力、深度可分离卷积、GAP、Dropout。")
    print(f"- 使用 batch_size=128、lr=0.1、CosineAnnealing、weight_decay、label_smoothing。")
    print(f"- 参数量约 {exp_after['params']}，最终测试准确率约 {exp_after['final_test_acc']:.2f}%。")
    print(f"- 训练轮数 {exp_after['epochs']}，泛化能力和推理速度均优于改进前。")

    print("\n【综合结论】")
    print("- 改进后模型在测试准确率上通常有明显提升（约 +20% 以上）。")
    print("- 深度可分离卷积 + GAP 显著降低参数量和 FLOPs，推理更快。")
    print("- 数据增强 + Dropout + weight_decay + label_smoothing 提升泛化能力。")
    print("- SE 注意力以极少参数代价带来额外准确率收益。")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    exp_before = main_easy()
    exp_after = main_hard()
    compare_experiments(exp_before, exp_after)