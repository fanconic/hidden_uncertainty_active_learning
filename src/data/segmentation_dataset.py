from PIL import Image
import torch
from torch import nn
from os.path import exists, join, split


class SegList(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, list_dir=None, out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        if self.phase in ["test", "val"]:
            data = list(self.transforms(*data))
        if self.out_name:
            if self.label_list is None:
                data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + "_images.txt")
        label_path = join(self.list_dir, self.phase + "_labels.txt")
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, "r")]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, "r")]
            assert len(self.image_list) == len(self.label_list)


class SegListMS(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms, scales, list_dir=None):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()
        self.scales = scales

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        w, h = data[0].size
        if self.label_list is not None:
            data.append(Image.open(join(self.data_dir, self.label_list[index])))
        # data = list(self.transforms(*data))
        out_data = list(self.transforms(*data))
        ms_images = [
            self.transforms(data[0].resize((int(w * s), int(h * s)), Image.BICUBIC))[0]
            for s in self.scales
        ]
        out_data.append(self.image_list[index])
        out_data.extend(ms_images)
        return tuple(out_data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + "_images.txt")
        label_path = join(self.list_dir, self.phase + "_labels.txt")
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, "r")]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, "r")]
            assert len(self.image_list) == len(self.label_list)
