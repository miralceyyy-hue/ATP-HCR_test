import os
import json
import random
from itertools import chain
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

from model.Benchmarks import BinaryBenchMarkNet_MutiQ
from torchsummary import summary
import cv2
from scipy.ndimage import label, find_objects


"""训练ATP_HCR_bin_MutiQ模型"""
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
    return bin_edges, concat_bin_num

class MedicalDataset(Dataset):
    def __init__(self, mapping, transform=None, balance_classes=True, bin_edges=None,num_bins=8,concat_bin_num=3):
        self.mapping = mapping
        self.image_paths = list(self.mapping.keys())
        # 原始 ATP 和它的 log10 值
        self.atp_raw = [self.mapping[p] for p in self.image_paths]
        self.log_values = [np.log10(v) for v in self.atp_raw]

        self.transform = transform

        # 1) 先确定 bin_edges（基于 log10）
        if bin_edges is None:
            lo, hi = min(self.log_values), max(self.log_values)
            bin_edges = np.linspace(lo - 1e-9, hi + 1e-9, num_bins+1)
        self.bin_edges = bin_edges

        # 2) 计算每个样本的类别标签（0-5 共 6 类）
        def assign_bin(l):
            idx = np.digitize(l, self.bin_edges)
            return 0 if idx <= concat_bin_num else (idx - concat_bin_num)
        self.labels = [assign_bin(l) for l in self.log_values]
        self.num_classes = len(set(self.labels))

        # 3) 记录每个类对应的 log10 范围
        self.label_ranges = {}
        self.label_ranges[0] = (self.bin_edges[0], self.bin_edges[concat_bin_num])
        for i in range(concat_bin_num, len(self.bin_edges)-1):
            self.label_ranges[i-concat_bin_num+1] = (self.bin_edges[i], self.bin_edges[i+1])

        # 4) 计算每个样本在所属类区间内的“百分比位置”
        self.percentiles = []
        for logv, lab in zip(self.log_values, self.labels):
            low, high = self.label_ranges[lab]
            pct = (logv - low) / (high - low + 1e-12)
            pct = float(np.clip(pct, 0.0, 1.0))
            self.percentiles.append(pct)

        # 输出平衡前统计
        class_counts = {}
        for label in self.labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        print("（平衡前）每个类别的样本数：")
        for label, count in sorted(class_counts.items()):
            low, high = self.label_ranges[label]
            print(f"类别 {label}: {count} 样本 —— 对应的 ATP 范围: {10**low:.2f} - {10**high:.2f}")
        print("-" * 50)

        # 5) （可选）类别平衡
        if balance_classes:
            self._balance_classes()
            # 输出平衡后统计
            class_counts = {}
            for label in self.labels:
                class_counts[label] = class_counts.get(label, 0) + 1
            print("（平衡后）每个类别的样本数：")
            for label, count in sorted(class_counts.items()):
                low, high = self.label_ranges[label]
                print(f"类别 {label}: {count} 样本 —— 对应的 ATP 范围: {10**low:.2f} - {10**high:.2f}")
            print("-" * 50)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像路径并加载
        path = convert_to_target_path(self.image_paths[idx])
        path = path.replace("image_2025", "processed_images")
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"无法读取图像: {path}")

        if self.transform:
            img = Image.fromarray(img)
            img = self.transform(img)

        # 返回图像和三个标签
        cls_label = torch.tensor(self.labels[idx], dtype=torch.long)
        pct_label = torch.tensor(self.percentiles[idx], dtype=torch.float32)
        atp_raw_label = torch.tensor(self.atp_raw[idx], dtype=torch.float32)
        return img, cls_label, pct_label, atp_raw_label

    def _balance_classes(self):
        from collections import defaultdict
        by_class = defaultdict(list)
        for i, lab in enumerate(self.labels):
            by_class[lab].append(i)
        max_n = max(len(indices) for indices in by_class.values())

        new_paths, new_labels, new_pcts, new_raws = [], [], [], []
        orig_raw = self.atp_raw
        orig_pcts = self.percentiles
        orig_labels = self.labels

        for lab, indices in by_class.items():
            if len(indices) < max_n:
                sampled = random.choices(indices, k=max_n)
            else:
                sampled = indices
            for i in sampled:
                new_paths.append(self.image_paths[i])
                new_labels.append(self.labels[i])
                new_pcts.append(orig_pcts[i])
                new_raws.append(orig_raw[i])

        # 打乱顺序
        perm = np.random.permutation(len(new_paths))
        self.image_paths = [new_paths[i] for i in perm]
        self.labels = [new_labels[i] for i in perm]
        self.percentiles = [new_pcts[i] for i in perm]
        self.atp_raw = [new_raws[i] for i in perm]



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
def train_model(model,
                dataloaders,
                combined_criterion,
                optimizer,
                lr_schedule,
                label_ranges: dict,
                num_epochs=25,
                device='cuda',
                model_name="model"):
    """
    训练一个同时做 6 分类和区间分位数回归的模型，并计算：
      - pct MAE（分位数空间）
      - atp MAE（原始 ATP 空间）

    dataloader 每 batch 返回 (inputs, cls_labels, pct_labels, atp_raw_labels)
    label_ranges: {bin_idx: (low_log, high_log), ...}
    """
    model = model.to(device)
    best_mae_atp = float('inf')
    best_reg_loss = float('inf')
    best_mae_pct = float('inf')
    best_acc = 0.0

    history = {k: [] for k in [
        'train_loss', 'val_loss',
        'train_loss_cls', 'val_loss_cls',
        'train_loss_reg', 'val_loss_reg',
        'train_acc', 'val_acc',
        'train_mae_pct', 'val_mae_pct',
        'train_mae_atp', 'val_mae_atp',
        'train_mae_first_atp', 'val_mae_first_atp'
    ]}
    cls_loss = nn.CrossEntropyLoss()
    reg_loss = nn.MSELoss()


    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for phase in ['train', 'val']:
            # if epoch == 0 and phase == 'train':
            #     continue
            model.train() if phase == 'train' else model.eval()

            running = {k: 0.0 for k in ['loss', 'loss_cls','loss_reg', 'corrects', 'abs_pct', 'abs_atp','abs_first_atp']}
            n = len(dataloaders[phase].dataset)
            per_class_mae = {i: [] for i in range(len(label_ranges))}
            # 分类别 ATP 百分误差 (%)
            per_class_pct_err = {i: [] for i in range(len(label_ranges))}

            with tqdm(dataloaders[phase], desc=phase, unit='batch') as pbar:
                for inputs, cls_labels, pct_labels, atp_raw in pbar:
                    inputs = inputs.to(device)
                    cls_labels = cls_labels.to(device)
                    pct_labels = pct_labels.to(device)
                    atp_raw = atp_raw.to(device).float().view(-1)
                    log10atp_raw = torch.log10(atp_raw)

                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        atp_pred, logits,pcts,atp_max_prob_pred = model(inputs)
                        bin_probs = F.softmax(logits, dim=1)
                        atp_pred = atp_pred.view(-1)
                        atp_max_prob_pred = atp_max_prob_pred.view(-1)
                        log10atp_pred = torch.log10(atp_pred)
                        # loss = combined_criterion(log10atp_pred.squeeze(),log10atp_raw)
                        preds = bin_probs.argmax(dim=1)
                        pct_preds = pcts[torch.arange(pcts.size(0)), preds]
                        loss, loss_cls, loss_reg = combined_criterion(logits, cls_labels, pct_preds, pct_labels)
                        # loss_cls = cls_loss(bin_probs, cls_labels)
                        # loss_reg = reg_loss(pct_preds, pct_labels) * 100.00

                        # 1) 分位数空间的 MAE
                        abs_pct = torch.abs(pct_preds.view(-1) - pct_labels).sum().item()

                        # 2) 原始 ATP 的 MAE
                        abs_atp = torch.abs(atp_pred - atp_raw).sum().item()
                        abs_first_atp = torch.abs(atp_max_prob_pred - atp_raw).sum().item()
                        for i in range(len(atp_pred)):
                            cls = int(preds[i].item())
                            # 如果类别预测不正确，则不计算 MAE
                            if preds[i] != cls_labels[i]:
                                continue
                            err = torch.abs(atp_pred[i] - atp_raw[i]).item()
                            per_class_mae[cls].append(err)
                            low, high = label_ranges[cls]
                            range_atp = 10 ** high - 10 ** low
                            pct_err = (err / (range_atp + 1e-8)) * 100
                            per_class_pct_err[cls].append(pct_err)


                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    bs = inputs.size(0)
                    running['loss'] += loss.item() * bs
                    running['loss_cls'] += loss_cls.item() * bs
                    running['loss_reg'] += loss_reg.item() * bs
                    running['corrects'] += (preds == cls_labels).sum().item()
                    running['abs_pct'] += abs_pct
                    running['abs_atp'] += abs_atp
                    running['abs_first_atp'] += abs_first_atp

                    pbar.set_postfix({
                        'loss': f"{loss.item():.3f}",
                        'loss_cls': f"{loss_cls.item():.3f}",
                        'loss_reg': f"{loss_reg.item():.3f}",
                        'acc': f"{(preds == cls_labels).float().mean().item():.3f}",
                        'mae_pct': f"{(abs_pct / bs):.3f}",
                        'mae_atp': f"{(abs_atp / bs):.0f}",
                        'mae_first_atp': f"{(abs_first_atp / bs):.0f}"
                    })

            # epoch metrics
            epoch_loss = running['loss'] / n
            epoch_loss_cls = running['loss_cls'] / n
            epoch_loss_reg = running['loss_reg'] / n
            epoch_acc = running['corrects'] / n
            epoch_mae_pct = running['abs_pct'] / n
            epoch_mae_atp = running['abs_atp'] / n
            epoch_mae_first_atp = running['abs_first_atp'] / n

            print(f"{phase} | Loss: {epoch_loss:.4f}"
                    f" | Loss_cls: {epoch_loss_cls:.4f}"
                    f" | Loss_reg: {epoch_loss_reg:.4f}"
                  f" | Acc: {epoch_acc:.4f}"
                  f" | MAE_pct: {epoch_mae_pct:.4f}"
                  f" | MAE_atp: {epoch_mae_atp:.0f}"
                  f" | MAE_first_atp: {epoch_mae_first_atp:.0f}")

            for k, val in zip(
                    ['loss','loss_cls','loss_reg', 'acc', 'mae_pct', 'mae_atp', 'mae_first_atp'],
                    [epoch_loss,epoch_loss_cls,epoch_loss_reg, epoch_acc, epoch_mae_pct, epoch_mae_atp]
            ):
                history[f'{phase}_{k}'].append(val)

            # 输出每个类别的 ATP MAE
            class_mae_str = " | ".join(
                [f"C{cls}: {np.mean(errs):.1f}" if errs else f"C{cls}: -"
                 for cls, errs in per_class_mae.items()]
            )
            print(f"Per-class ATP MAE → {class_mae_str}")
            class_pct_str = " | ".join(
                [f"C{cls}: {np.mean(errs):.1f}%" if errs else f"C{cls}: -"
                 for cls, errs in per_class_pct_err.items()]
            )
            print(f"Per-class ATP %Error → {class_pct_str}")

            # 保存最优模型（以验证集原始 ATP MAE 最小）
            if phase == 'val' and epoch_mae_atp < best_mae_atp:
                best_mae_atp = epoch_mae_atp
                best_mae_pct = epoch_mae_pct
                best_acc = epoch_acc
                torch.save(model.state_dict(), f"./trained_models/{model_name}_MutiHead_best.pth")
                print(f"    → Saved best model (val MAE_atp: {best_mae_atp:.1f} | val MAE_pct: {best_mae_pct:.4f} | val ACC: {best_acc:.4f})")

        lr_schedule.step()

    print(f"\nBest val MAE_atp: {best_mae_atp:.1f}")

    # 绘图
    epochs = range(1, num_epochs + 1)

    def plot(name, ylabel):
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history[f'train_{name}'], label='Train')
        plt.plot(epochs, history[f'val_{name}'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(f"./plts/{model_name}_MutiHead_{name}.png")
        plt.close()

    plot('loss', 'Loss')
    plot('loss_cls', 'Loss (cls)')
    plot('loss_reg', 'Loss (reg)')
    plot('acc', 'Accuracy')
    plot('mae_pct', 'MAE (percentile)')
    plot('mae_atp', 'MAE (ATP)')

    return model


class CombinedLoss(nn.Module):
    """
    组合损失：交叉熵 + α * (SmoothL1 或 MSE) 回归损失
    """
    def __init__(self, alpha=100.0, reg_loss='smoothl1',beta=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if reg_loss == 'mse':
            self.reg_criterion = nn.MSELoss()
        else:
            self.reg_criterion = nn.SmoothL1Loss()
        self.cls_criterion = nn.CrossEntropyLoss()

    def forward(self, logits, cls_labels, pct_preds, pct_labels):
        loss_cls = self.cls_criterion(logits, cls_labels)
        # 如果 pct_preds 是 (B,1)，先 squeeze
        pct_preds = pct_preds.view(-1)
        loss_reg = self.reg_criterion(pct_preds, pct_labels)
        loss = self.beta * loss_cls + self.alpha * loss_reg
        return loss, self.beta * loss_cls, self.alpha * loss_reg


##########################################
# 主流程
##########################################
if __name__ == '__main__':
    # 配置参数
    mapping_file = './data/processed_image_atp_mapping.json'
    model_path = './trained_models/googlenet_logbin_best_class9.pth'
    # model_path = './trained_models/AD_MutiHead_class12_best_44473515.pth'
    image_size = 512  # 需要与模型构造函数参数一致
    batch_size = 16
    num_workers = 1
    epoch_num = 50
    num_bins = 12
    backbone = 'googlenet'  # 模型名称
    # 6对应4分类、7对应5分类、8对应6分类、9对应7分类、10对应8分类、11对应8分类、12对应9分类、13对应10分类、14对应11分类
    print("训练模型名称：",backbone)

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
    # 存储val集的mapping到json文件
    # with open('./data/val_mapping.json', 'w', encoding='utf-8') as f:
    #     json.dump(val_mapping, f, indent=4, ensure_ascii=False)

    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30,fill=0x35),  # 随机旋转30度
        transforms.ColorJitter(0.1, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775]*3,
                             std=[0.26558745]*3)
        # transforms.Normalize(mean=[0.4183661572135932, 0.4183661572135932, 0.4183661572135932],
        #                      std=[0.28014669644274387, 0.28014669644274387, 0.28014669644274387])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49188775]*3,
                             std=[0.26558745]*3)
        # transforms.Normalize(mean=[0.4183661572135932, 0.4183661572135932, 0.4183661572135932],
        #                      std=[0.28014669644274387, 0.28014669644274387, 0.28014669644274387])
    ])

    # 创建数据集 - 训练集使用增强transform，验证集使用基础transform
    bin_edge,concat_bin_num = compute_bin_edges(full_mapping,num_bins=num_bins)
    print("分箱边界：", bin_edge)
    print("concat_bin_num:",concat_bin_num)
    label_ranges = {}
    label_ranges[0] = (bin_edge[0], bin_edge[concat_bin_num])
    for i in range(concat_bin_num, len(bin_edge) - 1):
        label_ranges[i - concat_bin_num + 1] = (bin_edge[i], bin_edge[i + 1])
    train_dataset = MedicalDataset(train_mapping, transform=train_transform, balance_classes=True, bin_edges=bin_edge,num_bins=num_bins,concat_bin_num=concat_bin_num)
    val_dataset = MedicalDataset(val_mapping, transform=val_transform, balance_classes=False, bin_edges=bin_edge,num_bins=num_bins,concat_bin_num=concat_bin_num)
    print("label_ranges:", label_ranges)
    # 获取类别数量
    num_category = train_dataset.num_classes

    # 创建DataLoader
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,drop_last=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }

    # 初始化模型
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    model = BinaryBenchMarkNet_MutiQ(
        category=num_category,
        label_ranges=label_ranges,
        backbone=backbone,
    ).to(device)
    # print(model)
    # 加载预训练模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
        print(f"加载预训练模型: {model_path}")

    # 假设图像尺寸为 224x224，3通道
    # summary(model, input_size=(1, image_size, image_size))

    # 损失函数与优化器(多分类)
    criterion = CombinedLoss()

    # optimizer = torch.optim.Adam(chain(*[head.parameters() for head in model.reg_heads]), lr=1e-4, weight_decay=1e-5)
    paras = chain(model.reg_heads.parameters(), model.classifier.parameters())
    optimizer = torch.optim.Adam(paras, lr=1e-4, weight_decay=1e-5)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=0)

    # 训练模型
    model = train_model(model, dataloaders, criterion, optimizer, scheduler,num_epochs=epoch_num, device=str(device),model_name=backbone,label_ranges=train_dataset.label_ranges)

    # 保存最佳模型
    # torch.save(model.state_dict(), 'ad2d_mil_best.pth')