from torch.utils.data import Dataset
import os
from PIL import Image


class Data_mura(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        root = "/mnt/hdd/medical-imaging/data/"
        if mode == "train":
            pathDatasetFile = "/mnt/hdd/medical-imaging/data/MURA-v1.1/train_image_paths.csv"
        elif mode == "val":
            pathDatasetFile = "/mnt/hdd/medical-imaging/data/MURA-v1.1/valid_image_paths.csv"
        fileDescriptor = open(pathDatasetFile, "r")
        listImagePaths = []
        listImageLabels = []
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split()
                if lineItems[0].split("/")[2]=='XR_SHOULDER':
                    imagePath = os.path.join(root, lineItems[0])
                    if 'positive' in imagePath.split("/")[-2]:
                        imageLabel = 1
                    else:
                        imageLabel = 0

                    listImagePaths.append(imagePath)
                    listImageLabels.append(imageLabel)
        fileDescriptor.close()

        self.img_path = listImagePaths
        self.img_label = listImageLabels

    def __getitem__(self, index):
        image_path = self.img_path[index] 
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.img_label[index]

        return image, label

    def __len__(self):
        return len(self.img_path)


class Data_chest(Dataset):
    def __init__(self, transforms_=None, mode='train'):
        self.transform = transforms_
        root = "/mnt/hdd/medical-imaging/data/"
        if mode == 'train':
            pathDatasetFile = "/mnt/hdd/medical-imaging/data/CheXpert-v1.0-small/train.csv"
        elif mode == "val":
            pathDatasetFile = "/mnt/hdd/medical-imaging/data/CheXpert-v1.0-small/valid.csv"

        fileDescriptor = open(pathDatasetFile, "r")
        listImagePaths = []
        listImageLabels = []
        line = True
        while line:
            line = fileDescriptor.readline()
            if line:
                lineItems = line.split(',')
                imagePath = os.path.join(root, lineItems[0])
                if imagePath == "/mnt/hdd/medical-imaging/data/CheXpert-v1.0-small/train/patient06765/study4/view1_frontal.jpg":
                    continue
                else:
                    imageLabel = bool(line.split(",")[-2])

                    listImagePaths.append(imagePath)
                    listImageLabels.append(imageLabel)
        fileDescriptor.close()
        listImagePaths = listImagePaths[1:]
        listImageLabels = listImageLabels[1:]

        self.img_path = listImagePaths
        self.img_label = listImageLabels

    def __getitem__(self, index):
        image_path = self.img_path[index]
        images = []
        image = Image.open(image_path)
        if self.transform is not None:
            image_1 = self.transform(image)
            image_2 = self.transform(image)
        images.append(image_1)
        images.append(image_2)
        label = self.img_label[index]

        return images, label

    def __len__(self):
        return len(self.img_path)

