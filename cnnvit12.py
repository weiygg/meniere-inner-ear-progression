
import os
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, accuracy_score, roc_curve, auc)
# 使用 torch.cuda.amp 导入，以兼容需要显式 device_type 参数的环境
from torch.cuda import amp 
from pathlib import Path

# 获取当前脚本所在目录，确保相对路径正确
script_dir = Path(__file__).resolve().parent

# 创建输出目录
output_dir = script_dir / "output"
output_dir.mkdir(exist_ok=True)

# 跨平台兼容性设置
if os.name == 'nt':
    torch.multiprocessing.set_start_method('spawn', force=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====================== (NiftiProcessor, MRIDataset, CNNViT, calculate_metrics, train_epoch, validate, load_external_data 保持不变) ======================
# 为简洁起见，这部分省略，但请确保在您的文件中保留所有函数和类的完整定义
# 并在 train_epoch 中使用带参数的 amp.autocast(device_type='cuda'):

class NiftiProcessor:
    """NIfTI 文件处理器，包含标准化和重采样。"""
    def __init__(self):
        pass
    def process_volume(self, path, target_size=(128, 128, 64), is_mask=False):
        data = nib.load(path).get_fdata().squeeze()
        while data.ndim > 3 and data.shape[-1] == 1:
            data = data.squeeze(axis=-1)
        if data.ndim != 3:
            raise ValueError(f"Invalid dimensions {path}: {data.shape}")
        if is_mask:
            processed = (self._resize(data, target_size, order=0) > 0).astype(np.float32)
        else:
            normalized = self._normalize(data)
            processed = self._resize(normalized, target_size)
        return processed
    @staticmethod
    def _normalize(volume):
        p_low, p_high = np.percentile(volume, [0.1, 99.9])
        return (np.clip(volume, p_low, p_high) - np.mean(volume)) / (np.std(volume) + 1e-8)
    @staticmethod
    def _resize(img, target_size, order=1):
        factors = [t/s for t, s in zip(target_size, img.shape)]
        return ndimage.zoom(img, factors, order=order, mode='nearest', prefilter=order>1)

class MRIDataset(Dataset):
    def __init__(self, valid_samples, processor):
        self.processor = processor
        self.samples = valid_samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self.processor.process_volume(sample['image_path'])
        mask = self.processor.process_volume(sample['mask_path'], is_mask=True)
        masked_img = image * mask
        return (torch.as_tensor(masked_img[None], dtype=torch.float32), 
                torch.tensor(sample['label'], dtype=torch.float32),
                sample['ID'])

class CNNViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 64, 3, padding=1), nn.BatchNorm3d(64), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.GELU(), nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.GELU(), nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, activation='gelu', batch_first=True),
            num_layers=4
        )
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1))
    def forward(self, x):
        x = self.cnn(x)
        B, C, D, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1) 
        x = self.transformer(x)
        return self.classifier(x.mean(dim=1))

def calculate_metrics(labels, preds_probs, threshold=0.5):
    """计算分类指标，保持不变"""
    preds_binary = preds_probs > threshold
    try:
        cm = confusion_matrix(labels, preds_binary)
        if cm.size == 1:
            if np.all(labels == 1): tn, fp, fn, tp = 0, 0, 0, cm.flat[0]
            elif np.all(labels == 0): tn, fp, fn, tp = cm.flat[0], 0, 0, 0
            else: tn, fp, fn, tp = 0, 0, 0, 0
        elif cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else: tn, fp, fn, tp = 0, 0, 0, 0
    except ValueError: tn, fp, fn, tp = 0, 0, 0, 0
    acc = accuracy_score(labels, preds_binary)
    precision = precision_score(labels, preds_binary, zero_division=0)
    sensitivity = recall_score(labels, preds_binary, zero_division=0)
    specificity = tn / (float(tn + fp) + 1e-8)
    npv = tn / (float(tn + fn) + 1e-8)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    roc_auc = roc_auc_score(labels, preds_probs) if len(np.unique(labels)) > 1 else np.nan
    return {'AUC': roc_auc, 'ACC': acc, 'Precision': precision, 'Sensitivity (Recall)': sensitivity, 'Specificity': specificity, 'NPV': npv, 'F1': f1, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    all_preds, all_labels, all_ids = [], [], []
    for inputs, targets, ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with amp.autocast(): # 保持兼容性修复
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            probs = torch.sigmoid(outputs.detach())
            all_preds.append(probs.cpu())
            all_labels.append(targets.cpu())
            all_ids.extend(ids)
            total_loss += loss.item()
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    train_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    return total_loss/len(loader), train_auc, all_preds, all_labels, all_ids

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for inputs, targets, ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_labels.append(targets.cpu().numpy())
            all_ids.extend(ids)
            total_loss += loss.item()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else np.nan
    return total_loss/len(loader), val_auc, all_preds, all_labels, all_ids

def load_external_data(processor, label_csv, image_dir, mask_dir):
    """加载并处理外部验证数据。返回 DataLoader 和 完整的 DataFrame"""
    if not label_csv.exists():
        print(f"Warning: External label file not found at {label_csv}. Skipping external validation setup.")
        return None, None
    label_df = pd.read_csv(label_csv)
    label_df['ID'] = label_df['ID'].astype(str)
    valid_samples = []
    for _, row in label_df.iterrows():
        id_str = row['ID']
        img_path = image_dir / f"{id_str}.nii.gz"
        mask_path = mask_dir / f"{id_str}_mask1.nii.gz"
        if img_path.exists() and mask_path.exists():
            valid_samples.append({
                "ID": id_str, "label": row['label'], "image_path": img_path, "mask_path": mask_path
            })
        # Note: 外部数据的跳过警告在主函数中处理，这里只返回找到的有效样本
    
    if len(valid_samples) == 0:
        print(f"Warning: Found 0 valid samples in external set ({label_csv.name}).")
        return None, None
        
    batch_size = 4
    num_workers = 0 if os.name == 'nt' else 16 
    external_set = MRIDataset(valid_samples, processor)
    external_loader = DataLoader(
        external_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    return external_loader, label_df
    

# ====================== 主流程 (外部验证按 Epoch 进行) ======================
def main():
    # ------------------ 路径配置 ------------------
    label_csv = script_dir / "label.csv"
    image_dir = script_dir / "image"
    mask_dir = script_dir / "mask"
    label_ex_csv = script_dir / "label_ex.csv"
    image_ex_dir = script_dir / "image_ex"
    mask_ex_dir = script_dir / "mask_ex"

    # ------------------ 内部数据加载与准备 (CV) ------------------
    # ... (内部数据加载逻辑保持不变)
    if not label_csv.exists():
        raise FileNotFoundError(f"未找到内部标签文件: {label_csv}。请检查路径配置。")
        
    label_df = pd.read_csv(label_csv)
    label_df['ID'] = label_df['ID'].astype(str)
    
    valid_samples = []
    for _, row in label_df.iterrows():
        id_str = row['ID']
        img_path = image_dir/f"{id_str}.nii.gz"
        mask_path = mask_dir/f"{id_str}_mask1.nii.gz"
        if img_path.exists() and mask_path.exists():
            valid_samples.append({
                "ID": id_str, "label": row['label'], "image_path": img_path, "mask_path": mask_path
            })
        else:
            print(f"跳过无效样本 ID {id_str}，文件不存在 (Image: {img_path.exists()}, Mask: {mask_path.exists()})")
    if len(valid_samples) == 0:
        raise ValueError("内部数据中没有找到有效样本！")
    print(f"内部样本数: {len(label_df)}，有效内部样本数: {len(valid_samples)}")
    
    # ------------------ 外部数据加载 (在 CV 循环外预加载一次) ------------------
    ex_processor = NiftiProcessor() 
    external_loader, external_label_df = load_external_data(
        ex_processor, label_ex_csv, image_ex_dir, mask_ex_dir
    )
    is_external_valid = external_loader is not None
    if not is_external_valid:
        print("注意：外部验证数据加载失败或无有效样本，将跳过外部验证。")
    
    # ------------------ 交叉验证 ------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_fold_results = [] 
    all_labels = [s['label'] for s in valid_samples]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(valid_samples, all_labels)):
        print(f"\n===== Fold {fold+1}/5 =====")
        torch.cuda.empty_cache()
        
        processor = NiftiProcessor() 
        train_set = MRIDataset([valid_samples[i] for i in train_idx], processor)
        val_set = MRIDataset([valid_samples[i] for i in val_idx], processor)
        
        batch_size = 4
        num_workers = 0 if os.name == 'nt' else 16 
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers>0
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size*2, num_workers=num_workers, pin_memory=True
        )
        
        model = CNNViT().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
        
        scaler = amp.GradScaler() if device.type == 'cuda' else None 
        
        num_epochs = 50
        fold_results = []
        
        for epoch in range(num_epochs):
            # 1. 训练和内部验证
            train_loss, train_auc, train_preds, train_labels, train_ids = train_epoch(
                model, train_loader, criterion, optimizer, scaler
            )
            val_loss, val_auc, val_preds, val_labels, val_ids = validate(
                model, val_loader, criterion
            )
            scheduler.step(val_auc)
            
            # 2. 外部验证 (每 Epoch 一次)
            ex_metrics = {}
            if is_external_valid:
                ex_loss, ex_auc, ex_preds, ex_labels, ex_ids = validate(
                    model, external_loader, criterion
                )
                ex_metrics = calculate_metrics(ex_labels, ex_preds)
                
                # 保存外部验证预测结果 (按 Epoch)
                ex_result_df = pd.DataFrame({'ID': ex_ids, 'Prediction': ex_preds, 'Label': ex_labels})
                ex_full = pd.merge(external_label_df, ex_result_df, on='ID', how='right')
                ex_full.to_csv(output_dir / f"fold{fold+1}_epoch{epoch+1}_external_predictions.csv", index=False)
                
                print(f"Epoch {epoch+1:02d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | EX AUC: {ex_auc:.4f}")
            else:
                print(f"Epoch {epoch+1:02d} | Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f}")
            
            # 3. 记录和保存指标
            train_metrics = calculate_metrics(train_labels, train_preds)
            val_metrics = calculate_metrics(val_labels, val_preds)

            # 整合结果
            epoch_data = {'Fold': fold + 1, 'Epoch': epoch + 1, 'Train_Loss': train_loss, 'Train_AUC': train_auc, 'Val_Loss': val_loss, 'Val_AUC': val_auc}
            epoch_data.update({f"Train_{k}": v for k, v in train_metrics.items()})
            epoch_data.update({f"Val_{k}": v for k, v in val_metrics.items()})
            if is_external_valid:
                epoch_data.update({f"EX_{k}": v for k, v in ex_metrics.items()})
                epoch_data['EX_Loss'] = ex_loss

            fold_results.append(epoch_data)

            # 保存内部预测结果 (按 Epoch)
            pd.DataFrame({'ID': train_ids, 'Prediction': train_preds, 'Label': train_labels}).to_csv(
                output_dir / f"fold{fold+1}_epoch{epoch+1}_train_predictions.csv", index=False)
            pd.DataFrame({'ID': val_ids, 'Prediction': val_preds, 'Label': val_labels}).to_csv(
                output_dir / f"fold{fold+1}_epoch{epoch+1}_val_predictions.csv", index=False)

        all_fold_results.extend(fold_results)

    # 最终指标汇总
    final_df = pd.DataFrame(all_fold_results)
    final_df.to_csv(output_dir / "all_epoch_metrics_summary.csv", index=False)

    # 最终结果输出 (使用每个 Fold 的最佳验证 AUC)
    print("\n=== All Metrics Summary ===")
    mean_val_auc_per_fold = final_df.loc[final_df.groupby('Fold')['Val_AUC'].idxmax()]
    mean_auc = mean_val_auc_per_fold['Val_AUC'].mean()
    std_auc = mean_val_auc_per_fold['Val_AUC'].std()
    print(f"Mean Best Validation AUC (across all folds): {mean_auc:.4f} ± {std_auc:.4f}")
    
    if is_external_valid:
        # 打印平均最佳外部 AUC (如果适用)
        mean_ex_auc_per_fold = final_df.loc[final_df.groupby('Fold')['EX_AUC'].idxmax()]
        mean_ex_auc = mean_ex_auc_per_fold['EX_AUC'].mean()
        print(f"Mean Best External AUC (across all folds): {mean_ex_auc:.4f}")

if __name__ == "__main__":
    main()
