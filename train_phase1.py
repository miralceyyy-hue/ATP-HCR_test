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
from model.ATP_HCR import ATP_HCR
from torchsummary import summary
import cv2
from scipy.ndimage import label, find_objects
import copy


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

##########################################
# 预处理函数：区域提取及掩膜生成
##########################################
def fine_mask(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    target_gray = 53
    gray_diff = image.astype(np.float32) - target_gray
    mask = (gray_diff < 15).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # # 可视化原始mask
    # plt.imshow(mask, cmap='gray')
    # plt.title('Original Mask')
    # plt.show()

    # 找到背景区域（mask == 0）
    zero_mask = (mask == 0).astype(np.uint8)
    labeled_array, num_features = label(zero_mask)

    boxes = []
    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        if np.any(region):
            coords = np.argwhere(region)
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            boxes.append((x0, y0, x1 - x0, y1 - y0))

    if not boxes:
        # fallback: 全图
        x, y, w, h = 0, 0, image.shape[1], image.shape[0]
    else:
        # 选最大区域
        boxes = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        x, y, w, h = boxes[0]

    # 转换为正方形并扩展
    side = max(w, h) + 40
    cx = x + w // 2
    cy = y + h // 2
    new_x = max(0, cx - side // 2)
    new_y = max(0, cy - side // 2)
    max_x = min(image.shape[1], new_x + side)
    max_y = min(image.shape[0], new_y + side)
    new_w = max_x - new_x
    new_h = max_y - new_y

    return new_x, new_y, new_w, new_h

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

def find_and_mask_smallest_region(
        image_path,
        target_color=(0x35, 0x35, 0x35),
        min_area=2500 * 2500,
        initial_tolerance=5,
        max_lighten_steps=4,
        lighten_factor=33,
        if_plot=True  # 新增参数控制可视化
):
    def get_mask_for_color(img, color, tol):
        lower = np.array([max(c - tol, 0) for c in color], dtype=np.uint8)
        upper = np.array([min(c + tol, 255) for c in color], dtype=np.uint8)
        return cv2.inRange(img, lower[::-1], upper[::-1])  # BGR顺序

    def enhance_mask(mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    def find_valid_contours(mask):
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        contours, _ = cv2.findContours(padded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt - 1 for cnt in contours if cv2.contourArea(cnt - 1) >= min_area]

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    original = image.copy()
    current_color = list(target_color)
    target_color_list = list(target_color)
    historical_masks = []
    lighten_step = 0

    if if_plot:  # 初始可视化
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("原始图像")
        plt.axis('off')

    while lighten_step <= max_lighten_steps:
        # 1. 尝试当前主颜色
        mask = get_mask_for_color(image, current_color, initial_tolerance*(1 + lighten_step))
        enhanced_mask = enhance_mask(mask)
        contours = find_valid_contours(enhanced_mask)

        if contours:
            best_cnt = max(contours, key=lambda c: 4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2))

            # 创建结果
            result_mask = np.zeros_like(enhanced_mask)
            cv2.drawContours(result_mask, [best_cnt], -1, 255, cv2.FILLED)
            result = np.full_like(image, target_color_list[::-1], dtype=np.uint8)
            result[result_mask == 255] = original[result_mask == 255]

            if if_plot:  # 成功可视化
                plt.subplot(1, 3, 2)
                plt.imshow(enhanced_mask, cmap='gray')
                plt.title(f"成功掩膜 (L={lighten_step})")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.title("最终结果")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
            return result

        # 2. 保存当前mask到历史记录
        historical_masks.append(enhanced_mask)

        # 3. 尝试组合所有历史mask
        if lighten_step == max_lighten_steps:
            combined_mask = historical_masks[0]
            for m in historical_masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, m)

            combined_contours = find_valid_contours(combined_mask)
            if combined_contours:
                best_cnt = max(combined_contours,
                               key=lambda c: 4 * np.pi * cv2.contourArea(c) / (cv2.arcLength(c, True) ** 2))

                result_mask = np.zeros_like(combined_mask)
                cv2.drawContours(result_mask, [best_cnt], -1, 255, cv2.FILLED)
                result = np.full_like(image, current_color[::-1], dtype=np.uint8)
                result[result_mask == 255] = original[result_mask == 255]

                if if_plot:  # 组合结果可视化
                    plt.subplot(1, 3, 2)
                plt.imshow(combined_mask, cmap='gray')
                plt.title(f"组合掩膜 (L={lighten_step})")
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                plt.title("组合结果")
                plt.axis('off')

                plt.tight_layout()
                plt.show()
                return result

        # 4. 颜色变白处理
        current_color = [min(c + lighten_factor, 255) for c in current_color]
        lighten_step += 1
        # print(f"颜色变白步骤 {lighten_step}: 新颜色 {current_color}")

    # 所有尝试失败
    print("所有尝试均未找到有效轮廓")
    if if_plot:
        plt.subplot(1, 3, 2)
        if historical_masks:
            plt.imshow(historical_masks[-1], cmap='gray')
        else:
            plt.imshow(np.zeros_like(image[:, :, 0]), cmap='gray')
        plt.title("最终尝试掩膜")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("未处理结果")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    return None

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
    return bin_edges

##########################################
# 数据集定义（仿照ATPDataset）
##########################################
class MedicalDataset(Dataset):
    def __init__(self, mapping, transform=None, balance_classes=True, bin_edges=None):
        self.mapping = mapping
        self.image_paths = list(self.mapping.keys())
        self.labels = [self.mapping[p] for p in self.image_paths]
        self.balanced_classes = balance_classes
        self.transform = transform

        # 对 ATP 取 log10
        self.labels = [np.log10(label) for label in self.labels]

        # 统一 bin_edges
        if bin_edges is None:
            min_label, max_label = min(self.labels), max(self.labels)
            bin_edges = np.linspace(min_label - 1e-9, max_label + 1e-9, 9)
        self.bin_edges = bin_edges

        # 输出每个箱的样本数量
        print("每个箱中的 log10(ATP) 标签个数：")
        for i in range(len(self.bin_edges) - 1):
            count = sum(1 for l in self.labels if self.bin_edges[i] <= l < self.bin_edges[i + 1])
            print(f"箱 {i}: {count} 个标签")

        # 分配标签（合并前3箱为类别0，其余依次编号）
        def assign_bin(l):
            bin_idx = np.digitize(l, self.bin_edges)  # bin_idx ∈ [1, len]
            if bin_idx <= 3:
                return 0
            return bin_idx - 3  # 类别编号从1开始

        self.labels = [assign_bin(l) for l in self.labels]
        self.num_classes = len(set(self.labels))

        # 生成 label_ranges：合并前3箱为类别0，其余独立编号
        self.label_ranges = {}
        self.label_ranges[0] = (self.bin_edges[0], self.bin_edges[3])
        for i in range(3, len(self.bin_edges) - 1):
            self.label_ranges[i - 2] = (self.bin_edges[i], self.bin_edges[i + 1])

        print("每个类别对应的ATP范围（log10(ATP)）：")
        for label in sorted(self.label_ranges):
            low, high = self.label_ranges[label]
            print(f"类别 {label}: log10(ATP) ∈ ({low:.4f}, {high:.4f})")

        # 类别平衡
        if balance_classes:
            self._balance_classes()

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

            label = torch.tensor(self.labels[idx], dtype=torch.long)

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

    def _balance_classes(self):
        """对每个类别采样到 max_count，实现数据平衡"""
        # 使用原始数据作为基础
        original_image_paths = self.image_paths
        original_labels = self.labels

        # 统计原始每个类别的样本
        from collections import defaultdict
        class_to_samples = defaultdict(list)
        for img_path, label in zip(original_image_paths, original_labels):
            class_to_samples[label].append(img_path)

        # 计算最大类别样本数
        max_count = max(len(samples) for samples in class_to_samples.values())

        # 构造平衡后的数据列表
        self.balanced_image_paths = []
        self.balanced_labels = []

        for label, samples in class_to_samples.items():
            num_samples = len(samples)
            if num_samples < max_count:
                # 对该类别进行有放回采样
                resampled = random.choices(samples, k=max_count)
            else:
                resampled = samples  # 如果已经达到最大值，不重复

            self.balanced_image_paths.extend(resampled)
            self.balanced_labels.extend([label] * max_count)

        # 替换原始数据
        self.image_paths = self.balanced_image_paths
        self.labels = self.balanced_labels

        # 打乱数据顺序
        indices = np.arange(len(self.image_paths))
        np.random.shuffle(indices)
        self.image_paths = [self.image_paths[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]

        # 输出每个类别的新样本数
        print("After class balancing:")
        for label in sorted(class_to_samples.keys()):
            print(f"Class {label}: {max_count} samples - raw {len(class_to_samples[label])} samples")
        print("-" * 50)

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
    best_acc = 0.4
    best_model_wts = None

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
            if epoch == 0 and phase == 'train':
                continue
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

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
                        preds = torch.argmax(outputs, dim=1)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()


                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # 收集预测和标签
                    epoch_labels.extend(labels.cpu().numpy())
                    epoch_preds.extend(preds.cpu().numpy())

                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{torch.sum(preds == labels.data).item() / inputs.size(0):.4f}"
                    })

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 记录历史数据
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())

            if phase == 'val':
                # 保存验证集的预测结果用于混淆矩阵
                all_labels = epoch_labels
                all_preds = epoch_preds

                # 打印分类报告
                from sklearn.metrics import classification_report
                print("\nClassification Report:")
                print(classification_report(all_labels, all_preds))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'AD_logbin_best.pth')
                print(f"Best model saved with accuracy: {best_acc:.4f}")
        lr_schedule.step()

    print(f"Best val Acc: {best_acc:.4f}")
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('AD_logbin_loss_plot.png')
    plt.show()
    # 绘制训练和验证准确率
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('AD_logbin_accuracy_plot.png')
    plt.show()

    return model


##########################################
# 主流程
##########################################
if __name__ == '__main__':
    # 配置参数
    mapping_file = 'processed_image_atp_mapping.json'
    model_path = './AD_logbin_best_0.8925.pth'
    image_size = 512  # 需要与模型构造函数参数一致
    batch_size = 16
    num_workers = 1
    epoch_num = 30
    num_bins = 8

    # image_size = 1024  # 需要与模型构造函数参数一致
    # batch_size = 4
    # num_workers = 1
    # epoch_num = 30

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
    bin_edge = compute_bin_edges(full_mapping,num_bins=num_bins)
    train_dataset = MedicalDataset(train_mapping, transform=train_transform, balance_classes=True, bin_edges=bin_edge)
    val_dataset = MedicalDataset(val_mapping, transform=val_transform, balance_classes=False, bin_edges=bin_edge)

    # 获取类别数量
    num_category = train_dataset.num_classes

    # 创建DataLoader
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = ATP_HCR(
        in_channel=1,  # RGB输入
        hidden=256,  # 隐藏层维度
        category=num_category,
        num_layer=4,  # 特征提取层数
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,num_epochs=epoch_num, device=str(device))

    # 保存最佳模型
    # torch.save(model.state_dict(), 'ad2d_mil_best.pth')