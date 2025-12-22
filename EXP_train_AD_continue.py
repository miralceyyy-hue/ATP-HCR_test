import os
import json
import random
from sklearn.utils import resample

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from model.ATP_HCR import ATP_HCR_Log_Continue
from torchsummary import summary
import cv2
from scipy.ndimage import label, find_objects
import copy

"""
使用AD架构，直接连续值预测
"""
plt.rcParams['font.sans-serif'] = ['SimHei']
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)
set_seed(0)


def convert_to_target_path(image_path):
    image_path = image_path.replace("\\", "/")
    # print("原始路径：", image_path)
    # 定义服务器的基准路径
    server_base_path = "/sibcb1/chenluonanlab8/lijinhong/zuochunman/Shared_data/Rongjianming/ShandongUniversity"
    # 获取本地路径的上级目录（"image_2025"所在路径）
    local_base_path = os.path.join("F:", "ShandongUniversity", "image_2025")
    # print("本地路径：", local_base_path)
    # 替换 "image_2025" 为 "processed_images"，并生成目标路径
    server_base_path2 = os.path.join(server_base_path, "processed_images")
    # print("服务器路径：", server_base_path2)
    target_path = image_path.replace(local_base_path, server_base_path2)
    # print("目标路径：", target_path)
    return target_path

# 计算bin_edge
def compute_bin_edges(mapping_dict, num_bins=8):
    """
    根据原始ATP映射，计算log10(ATP)后的分箱边界。

    参数:
        mapping_dict: dict，图像路径到ATP值的映射
        num_bins: int，分箱数量（默认8个箱，对应9个边界）

    返回:
        bin_edges: np.ndarray，长度为 num_bins+1 的数组
    """
    atp_values = list(mapping_dict.values())
    log_atp_values = [np.log10(v) for v in atp_values]
    min_val, max_val = min(log_atp_values), max(log_atp_values)
    bin_edges = np.linspace(min_val - 1e-9, max_val + 1e-9, num_bins + 1)

    # 统计每个 bin 中的样本数
    bin_counts = np.histogram(log_atp_values, bins=bin_edges)[0]

    # 找出最大 bin 样本数，并计算其10%
    max_bin_count = max(bin_counts)
    threshold = max_bin_count * 0.1

    # 累加前几个 bin 的样本数，直到超过阈值
    cumulative = 0
    concat_bin_num = 0
    for count in bin_counts:
        cumulative += count
        concat_bin_num += 1
        if cumulative >= threshold:
            break
    return bin_edges,concat_bin_num

##########################################
# 数据集定义（仿照ATPDataset）
##########################################
class MedicalDataset(Dataset):
    def __init__(self, mapping, transform=None, balance_classes=True):
        self.mapping = mapping
        self.image_paths = list(self.mapping.keys())
        self.labels = [self.mapping[p] for p in self.image_paths]
        self.balanced_classes = balance_classes
        self.transform = transform

        # 对 ATP 取 log10
        self.labels = [np.log10(label) for label in self.labels]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_path = convert_to_target_path(image_path)
        # # 处理后的图像保存路径,把“image_2025”替换为“processed_images”
        # image_path = image_path.replace("image_2025", "processed_images")
        try:
            # 读取图片,灰度
            image = cv2.imread(image_path,0)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return None

            label = torch.tensor(self.labels[idx], dtype=torch.float32)

            if self.transform:
                # 确保processed_img是numpy数组
                if not isinstance(image, np.ndarray):
                    image = np.array(image)

                # 转换为PIL图像
                image = Image.fromarray(image)
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__(np.random.randint(0, len(self)))


    def _shuffle_data(self):
        """ 打乱数据顺序 """
        indices = np.arange(len(self.balanced_image_paths))
        np.random.shuffle(indices)
        self.balanced_image_paths = [self.balanced_image_paths[i] for i in indices]
        self.balanced_labels = [self.balanced_labels[i] for i in indices]

    def _print_class_distribution(self, class_counts):
        """ 打印平衡后的类别样本数 """
        print("After class balancing:")
        new_class_counts = {}
        for label in self.balanced_labels:
            new_class_counts[label] = new_class_counts.get(label, 0) + 1
        for cls, count in sorted(new_class_counts.items()):
            print(f"Class {cls}: {count} samples - raw {class_counts[cls]} samples")
        print("-" * 50)

    def save_process_image(self,image,image_raw_path):
        """
        保存处理后的图像
        :param image:
        :param image_raw_path:
        :return:
        """
        # 处理后的图像保存路径,把“image_2025”替换为“processed_images”
        processed_image_path = image_raw_path.replace("image_2025", "processed_images")
        # 创建目录
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        # 确保image是numpy array (OpenCV需要)
        if isinstance(image, Image.Image):  # 如果是PIL图像
            image = np.array(image)
            # PIL图像通常是RGB，而OpenCV需要BGR，由于是明场图像，保存为灰度图
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 保存TIFF格式
        cv2.imwrite(processed_image_path, image, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        # print(f"保存处理后的图像: {processed_image_path}")

##########################################
# 训练验证流程（与train.py保持一致）
##########################################
def train_model(model, dataloaders, criterion, optimizer,lr_schedule, num_epochs=25, device='cuda'):
    model = model.to(device)
    best_loss = float('inf')

    # 初始化数据记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(num_epochs):
        print("-" * 30)
        print(f"Epoch {epoch + 1}/{num_epochs}")


        for phase in ['train', 'val']:
            # if epoch == 0 and phase == 'train':
            #     continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            # 清空每轮的预测和标签
            epoch_labels = []
            epoch_preds = []

            with tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch + 1}', unit='batch') as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        preds = outputs.squeeze()

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    running_loss += loss.item() * inputs.size(0)

                    # 收集预测和标签
                    epoch_labels.extend(labels.detach().cpu().numpy())
                    epoch_preds.extend(preds.detach().cpu().numpy())

                    pbar.set_postfix({
                        'loss': f"{loss.item():.2f}",
                    })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f}")

            # 记录历史数据
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), './trained_models/AD_continue_best_class.pth')
                print(f"Best model saved with loss: {best_loss:.4f}")
        lr_schedule.step()

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./plts/AD_logbin_loss_plot_class.png')
    plt.show()

    return model


##########################################
# 主流程
##########################################
if __name__ == '__main__':
    # 配置参数
    mapping_file = './data/processed_image_atp_mapping.json'
    model_path = './AD_logbin_best_0.8925.pth'
    image_size = 512  # 需要与模型构造函数参数一致
    batch_size = 16
    num_workers = 1
    epoch_num = 60

    # 首先加载并分割mapping_file
    with open(mapping_file, 'r', encoding='utf-8') as f:
        full_mapping = json.load(f)

    # 获取所有图像路径并打乱
    image_paths = list(full_mapping.keys())
    np.random.shuffle(image_paths)

    # 分割训练集和验证集
    split = int(0.75 * len(image_paths))
    train_paths = image_paths[:split]
    val_paths = image_paths[split:]

    # 创建对应的mapping字典
    train_mapping = {path: full_mapping[path] for path in train_paths}
    val_mapping = {path: full_mapping[path] for path in val_paths}

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30,fill=0x35),  # 随机旋转30度
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
        # transforms.Normalize(mean=[0.4183661572135932, 0.4183661572135932, 0.4183661572135932],
        #                      std=[0.28014669644274387, 0.28014669644274387, 0.28014669644274387])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775],
                             std=[0.26558745])
        # transforms.Normalize(mean=[0.4183661572135932, 0.4183661572135932, 0.4183661572135932],
        #                      std=[0.28014669644274387, 0.28014669644274387, 0.28014669644274387])
    ])

    # 创建数据集 - 训练集使用增强transform，验证集使用基础transform
    train_dataset = MedicalDataset(train_mapping, transform=train_transform, balance_classes=True)
    val_dataset = MedicalDataset(val_mapping, transform=val_transform, balance_classes=False)


    # 创建DataLoader
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = ATP_HCR_Log_Continue(
        in_channel=1,  # RGB输入
        hidden=256,  # 隐藏层维度
        image_size=image_size,
        patches=64
    ).to(device)
    # 加载预训练模型
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path, map_location=device))
    #     print(f"加载预训练模型: {model_path}")
    # 假设图像尺寸为 224x224，3通道
    # summary(model, input_size=(1, image_size, image_size))

    # 损失函数与优化器(多分类)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,num_epochs=epoch_num, device=str(device))

    # 保存最佳模型
    # torch.save(model.state_dict(), 'ad2d_mil_best.pth')