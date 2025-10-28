import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleNPZDataset(Dataset):
    def __init__(self, npz_path, label_filter):
        #data = np.load(npz_path)
        

        with open(npz_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()
        # keys = data.keys()
        
        # # 打印结果
        # print(f"文件 '{npz_path}' 中包含的键（keys）有：")
        # print(keys)
        images = data['x']
        #print(images.shape)
        labels = data['y']
        #print(labels.shape)
        mask = label_filter(labels)
        self.images = images[mask]
        self.labels = labels[mask]

    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        #img = torch.tensor(img).permute(2,0,1).float() / 255.0  # 假设是HWC
        return img, int(label)

def get_dataloaders(data_root, batchsize=10, num_workers=1):
    # trainloaders: 5个客户端
    trainloaders = []
    for i in range(5):
        npz_path = f"{data_root}/train/{i}.npz"
        ds = SimpleNPZDataset(npz_path, label_filter=lambda y: y < 6)
        trainloaders.append(DataLoader(ds, batch_size=batchsize, shuffle=True, num_workers=num_workers))
    # test数据
    test_imgs, test_labels = [], []
    for i in range(5):
        npz_path = f"{data_root}/test/{i}.npz"
        with open(npz_path, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        #data = np.load(npz_path)


        test_imgs.append(data['x'])
        test_labels.append(data['y'])
    test_imgs = np.concatenate(test_imgs, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    # valloader/closerloader/train_val_loaders: label<6
    mask_close = test_labels < 6
    close_ds = torch.utils.data.TensorDataset(
        torch.tensor(test_imgs[mask_close]).float() / 255.0,
        torch.tensor(test_labels[mask_close]).long()
    )
    valloader = DataLoader(close_ds, batch_size=10, shuffle=False, num_workers=num_workers)
    closerloader = DataLoader(close_ds, batch_size=10, shuffle=False, num_workers=num_workers)
    train_val_loaders = [closerloader for _ in range(5)]  # 你可以按需复制
    # openloader: label>=6
    mask_open = test_labels >= 6
    open_ds = torch.utils.data.TensorDataset(
        torch.tensor(test_imgs[mask_open]).float() / 255.0,
        torch.tensor(test_labels[mask_open]).long()
    )
    openloader = DataLoader(open_ds, batch_size=10, shuffle=False, num_workers=num_workers)
    print("\n--- Dataloader Batch Shapes Check ---")
    
    # 1. 打印第一个训练客户端的 Batch 形状
    try:
        if trainloaders:
            # 获取第一个客户端的 DataLoader
            dl = trainloaders[0]
            # 迭代一次以获取第一个批次
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
    #return trainloaders, valloader, closerloader, openloader, train_val_loaders