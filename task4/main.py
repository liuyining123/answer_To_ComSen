import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

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
# epochs = 2#测试用

# 显示图像--测试
def imshow(img):
    img = img / 2 + 0.5     # 去标准化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5,stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3,stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.aap=nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(in_features=36*6*6, out_features=128)
        # self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.fc3 = nn.Linear(in_features=36, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=10)
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.aap(x)
        x = x.view(x.shape[0], -1)
        # x = x.view(-1, 36)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
net = CNNNet().to(device)

# 打印预测结果
def save_prediction_grid(net, testloader, classes, device, num_images=10, save_path='prediction_results.png'):
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

def output_prediction_grid(epochs,train_losses, train_accs, test_accs, net, testloader, classes, device, num_images=10, save_path='prediction_results_epoch.png'):
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
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 保存分类结果可视化图片
    save_prediction_grid(net, testloader, classes, device, num_images=10, save_path='prediction_results.png')

def init_weights(net):
    # 初始化参数
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            #卷积层
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            #全连接层
            # nn.init.normal_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
def train_model(net,train_losses,train_accs,test_accs, trainloader, criterion, optimizer, device, epochs):
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, data in enumerate(trainloader, 0):
            # 获取输入数据
            inputs, labels = data[0].to(device), data[1].to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播 + 反向传播 + 优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        epoch_train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(epoch_train_acc)

        # 测试集评估
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

    print('Finished Training')
    print(f'Final Test Accuracy: {test_accs[-1]:.2f}%')

def main_easy():
    '''
    未经优化的CNN网络，训练2个epoch，准确率约为10%，损失2.303降不下来。
    效果极差。
    考虑对全连接层也使用kaiming_normal_初始化
    经过优化后，准确率为59.68%，损失降到1.041左右。
    考虑增加epoch个数至4
    有明显改善，准确率约67%
    接下来考虑使用全局平均池化减少参数数量
    准确率54.44%，损失1.3左右。
    考虑添加隐藏层fc4 准确率未见明显提高 约56%
    基础网络完成
    最终增加epoch至10，以便画出更为准确的训练曲线
    '''
    # 初始化网络参数
    init_weights(net)  

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 训练+测试
    train_losses, train_accs, test_accs = [], [], []
    train_model(net, train_losses, train_accs, test_accs, trainloader, criterion, optimizer, device, epochs)
    print("Net_gvp have {} parameters in total".format(sum(x.numel() for x in net.parameters())))

    # 绘制训练曲线和保存预测结果
    output_prediction_grid(epochs, train_losses, train_accs, test_accs, net, testloader, classes, device, num_images=10, save_path='prediction_results_epoch.png')
    
if __name__ == '__main__':
    main_easy()