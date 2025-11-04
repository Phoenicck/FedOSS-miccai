
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# 1. 导入 transforms 库
from torchvision import transforms 

class SimpleNPZDataset(Dataset):
    def __init__(self, npz_path):
        
        # ⚠️ 注意：如果目标是返回 HWC，我们不应使用 transforms.ToTensor()。
        # 我们只保留归一化参数，并在 __getitem__ 中手动应用。
        self.mean = 0.5
        self.std = 0.5
        
        with open(npz_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
        
        self.images = data['x']
        self.labels = data['y']
        
        mask_open = self.labels >= 6
        if np.any(mask_open):
            self.labels[mask_open] = 6

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img = self.images[idx] # img 是 HWC NumPy array
        label = self.labels[idx]
        
        # 1. 转换为 float 并缩放到 [0.0, 1.0]
        img = img.astype(np.float32) / 255.0
        
        # 2. 应用 [-1.0, 1.0] 的归一化
        img = (img - self.mean) / self.std
        
        # 3. 转换为 PyTorch Tensor（保持 HWC 格式）
        # 注意：这里只进行 Tensor 转换，不进行维度调整 (ToTensor() 会自动调整)
        img = torch.tensor(img) 

        # img 现在是 HWC 格式的 Tensor，范围是 [-1.0, 1.0]
        # 但是，这通常不是 PyTorch 模型期望的格式 (CHW)
        
        return img, int(label)

def get_dataloaders(data_root, batchsize=10, num_workers=1):
    # trainloaders: 5个客户端
    trainloaders = []
    for i in range(5):
        npz_path = f"{data_root}/train/{i}.npz"
        ds = SimpleNPZDataset(npz_path)
        trainloaders.append(DataLoader(ds, batch_size=batchsize, shuffle=True, num_workers=num_workers))
        
    # valloader and closerloader from centralized_close_test.npz
    close_test_path = f"{data_root}/centralized_close_test.npz"
    close_ds = SimpleNPZDataset(close_test_path)
    valloader = DataLoader(close_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers)
    closerloader = DataLoader(close_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    # openloader from centralized_open_test.npz
    open_test_path = f"{data_root}/centralized_open_test.npz"
    open_ds = SimpleNPZDataset(open_test_path)
    openloader = DataLoader(open_ds, batch_size=batchsize, shuffle=False, num_workers=num_workers)

    # train_val_loaders from test/0.npz, test/1.npz, ...
    train_val_loaders = []
    for i in range(5):
        npz_path = f"{data_root}/test/{i}.npz"
        ds = SimpleNPZDataset(npz_path)
        train_val_loaders.append(DataLoader(ds, batch_size=batchsize, shuffle=False, num_workers=num_workers))
    
    print("\n--- Dataloader Batch Shapes Check ---")
    
    # 1. 打印第一个训练客户端的 Batch 形状
    try:
        if trainloaders:
            dl = trainloaders[0]
            inputs, targets = next(iter(dl))
            print(f"✅ trainloaders[0] Batch Shape (Images, Labels): {inputs.shape}, {targets.shape}")
        else:
            print("❌ trainloaders 列表为空。")
    except Exception as e:
        print(f"❌ 打印 trainloaders[0] 形状时出错: {e}")

    # 2. 打印验证集 (valloader) 的 Batch 形状
    try:
        dl = valloader
        inputs, targets = next(iter(dl))
        print(f"✅ valloader Batch Shape (Images, Labels): {inputs.shape}, {targets.shape}")
    except Exception as e:
        print(f"❌ 打印 valloader 形状时出错: {e}")
        
    # 3. 打印已知类测试集 (closerloader) 的 Batch 形状
    try:
        dl = closerloader
        inputs, targets = next(iter(dl))
        print(f"✅ closerloader Batch Shape (Images, Labels): {inputs.shape}, {targets.shape}")
    except Exception as e:
        print(f"❌ 打印 closerloader 形状时出错: {e}")

    # 4. 打印未知类测试集 (openloader) 的 Batch 形状
    try:
        dl = openloader
        inputs, targets = next(iter(dl))
        print(f"✅ openloader Batch Shape (Images, Labels): {inputs.shape}, {targets.shape}")
    except Exception as e:
        print(f"❌ 打印 openloader 形状时出错: {e}")

    # 5. 打印第一个训练验证集 (train_val_loaders) 的 Batch 形状
    try:
        if train_val_loaders:
            dl = train_val_loaders[0]
            inputs, targets = next(iter(dl))
            print(f"✅ train_val_loaders[0] Batch Shape (Images, Labels): {inputs.shape}, {targets.shape}")
        else:
            print("❌ train_val_loaders 列表为空。")
    except Exception as e:
        print(f"❌ 打印 train_val_loaders[0] 形状时出错: {e}")
        
    print("---------------------------------------")
    
    return trainloaders, valloader, closerloader, openloader, train_val_loaders