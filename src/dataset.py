from mpl_toolkits.mplot3d.proj3d import transform
from torch.utils.data import Dataset, DataLoader, random_split  # 数据加载
import os # 目录、文件操作
from PIL import Image # 图像数据
from natsort import natsorted
from torchvision import transforms # 图像操作
from typing_extensions import dataclass_transform

'''cat.0.jpg'''
class CatDogDataset(Dataset):
    # 目录 变换操作（数据增强）
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 列出目录下的文件名 返回列表
        self.images = os.listdir(self.root_dir)
        # 过滤非图像文件
        self.images = [img for img in self.images if img.endswith(('.jpg', '.jpeg', '.png'))]

    # 样本数量
    def __len__(self):
        return len(self.images)

    # 索引访问
    def __getitem__(self, idx):
        # 图片名
        image_name = self.images[idx]
        # 图片路径
        image_path = os.path.join(self.root_dir, image_name)
        # 加载图片（三通道） 如果原图是PNG则有RGBA四个通道
        image = Image.open(image_path).convert('RGB')
        # 解析标签 0 cat 1 dog
        label = 0 if 'cat' in image_name.lower() else 1
        # 变换操作 数据增强
        if self.transform:
            image = self.transform(image)

        return image, label

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 列出目录下的文件名 返回列表
        self.images = os.listdir(self.root_dir)
        # 自然排序
        self.images = natsorted(self.images)
        # 过滤非图像文件
        self.images = [img for img in self.images if img.endswith(('.jpg', '.jpeg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 图片名
        image_name = self.images[idx]
        # 图片路径
        image_path = os.path.join(self.root_dir, image_name)
        # 加载图片（三通道） 如果原图是PNG则有RGBA四个通道
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, image_name

def inference_dataloader():
    root_dir = 'D:\AA_Py_learn\Cats_and_Dogs\data\\test'
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一大小
        transforms.ToTensor(),  # 转为tensor支持dataloader
    ])
    dataset = InferenceDataset(root_dir, data_transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    return dataloader


def train_val_dataloader():
    root_dir = 'D:\AA_Py_learn\Cats_and_Dogs\data\\train'

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一大小
        transforms.ToTensor(),  # 转为tensor支持dataloader
    ])

    dataset = CatDogDataset(root_dir, transform=data_transforms)
    train_size = round(0.8 * len(dataset))
    val_size = round(0.2 * len(dataset))
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(dataset=val_data, batch_size=64, shuffle=True, num_workers=1)

    return train_dataloader, val_dataloader

def test_dataloader():
    root_dir = ''

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一大小
        transforms.ToTensor(),  # 转为tensor支持dataloader
    ])

    dataset = CatDogDataset(root_dir, transform=data_transforms)
    test_dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True, num_workers=1)

    return test_dataloader



if __name__ == '__main__':
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),  # 统一大小
        transforms.ToTensor(),  # 转为tensor支持dataloader
    ])
    dataloader = inference_dataloader()
    for x, y in dataloader:
        print(y[:10])
        break