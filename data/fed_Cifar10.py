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
    valloader = DataLoader(close_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    closerloader = DataLoader(close_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    train_val_loaders = [closerloader for _ in range(5)]  # 你可以按需复制
    # openloader: label>=6
    mask_open = test_labels >= 6
    open_ds = torch.utils.data.TensorDataset(
        torch.tensor(test_imgs[mask_open]).float() / 255.0,
        torch.tensor(test_labels[mask_open]).long()
    )
    openloader = DataLoader(open_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    # print("Trainloader[0] images shape:")
    # print(trainloaders[0].dataset.images.shape)
    # print("Valloader images shape:")
    # print(valloader.images.shape)
    return trainloaders, valloader, closerloader, openloader, train_val_loaders