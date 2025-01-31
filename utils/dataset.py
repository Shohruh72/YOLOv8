import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from utils import augment
from copy import deepcopy
from torch.utils import data

from ultralytics.utils.instance import Instances

img_ext = {"bmp", "jpeg", "jpg", "png", "tif", "tiff"}


class Dataset(data.Dataset):
    def __init__(self, args, params, augments=True):
        super(Dataset, self).__init__()
        self.args = args
        self.params = params
        self.augment = augments
        self.mosaic = augments

        file = 'train2017.txt' if self.augment else 'val2017.txt'
        filenames = f'{args.data_dir}/{file}'
        self.images = self.load_image(filenames)
        self.labels = self.load_labels(args, self.images)["labels"]

        self.num_img = len(self.labels)
        self.indices = np.arange(self.num_img)
        self.transforms = self.build_transforms()

        if not self.augment:
            bi = np.floor(self.indices / self.args.batch_size)
            bi = bi.astype(int)
            nb = bi[-1] + 1
            shape = np.array([x.pop("shape") for x in self.labels])
            ratio = shape[:, 0] / shape[:, 1]
            self.images = [self.images[i] for i in ratio.argsort()]
            self.labels = [self.labels[i] for i in ratio.argsort()]

            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ratio[ratio.argsort()][bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(
                np.array(shapes) * self.args.inp_size / 32 + 0.5)
            self.batch_shapes = self.batch_shapes.astype(int) * 32
            self.batch = bi

    def __getitem__(self, index):
        tr_dataset = self.transforms(self.get_image_and_label(index))
        return tr_dataset

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def load_image(path):
        samples = []
        for p in [path]:
            p = Path(p)
            with open(p) as f:
                samples += [x.replace("./", str(p.parent) + os.sep)
                            if x.startswith("./") else x for x in
                            f.read().strip().splitlines()]

        return sorted(x.replace("/", os.sep) for x in samples if
                      x.split(".")[-1].lower() in img_ext)

    def read_image(self, index):
        image = cv2.imread(self.images[index])
        h0, w0 = image.shape[:2]
        r = self.args.inp_size / max(h0, w0)
        if r != 1:
            w, h = (min(math.ceil(w0 * r), self.args.inp_size),
                    min(math.ceil(h0 * r), self.args.inp_size))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        return image, (h0, w0), image.shape[:2]

    @staticmethod
    def load_labels(args, images):
        cache = {"labels": []}
        a = f"{os.sep}images{os.sep}"
        b = f"{os.sep}labels{os.sep}"
        labels = [b.join(x.rsplit(a, 1)).rsplit(".", 1)[0] + ".txt" for x in
                  images]

        path = Path(labels[0]).parent.with_suffix(".cache")
        if os.path.exists(path):
            return torch.load(path)

        for img, label in zip(images, labels):
            try:
                image = Image.open(img)
                image.verify()
                shape = image.size
                shape = (shape[1], shape[0])
                assert (shape[0] > 9) & (
                        shape[1] > 9), f"image size {shape} <10 pixels"
                assert image.format.lower() in img_ext, f"invalid image format"

                if os.path.isfile(label):
                    with open(label) as f:
                        lines = f.read().strip().splitlines()
                        lb = [x.split() for x in lines if len(x)]
                        lb = np.array(lb, dtype=np.float32)
                    nl = len(lb)
                    if nl:
                        assert lb.min() >= 0, f"negative label values {lb[lb < 0]}"
                        assert lb.shape[
                                   1] == 5, f"labels require 5 columns, {lb.shape[1]} columns detected"
                        assert lb[:,
                               1:].max() <= 1, f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                        assert lb[:, 0].max() <= args.num_cls, (
                            f"Label class {int(lb[:, 0].max())} exceeds dataset class count {args.num_cls}. "
                            f"Possible class labels are 0-{args.num_cls - 1}")
                        _, i = np.unique(lb, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            lb = lb[i]  # remove duplicates
                    else:
                        lb = np.zeros((0, 5), dtype=np.float32)
                else:
                    lb = np.zeros((0, 5), dtype=np.float32)
                lb = lb[:, :5]
                if img:
                    cache["labels"].append({'image': img,
                                            "shape": shape,
                                            'cls': lb[:, 0:1],
                                            'box': lb[:, 1:],
                                            "norm": True,
                                            "format": "xywh"})
            except Exception as e:
                print(f"Skipping file {img} due to error: {e}")
        torch.save(cache, path)
        return cache

    def get_image_and_label(self, index):
        label = deepcopy(self.labels[index])
        label["img"], label["shape"], label["new_shape"] = self.read_image(index)
        label["pad"] = (label["new_shape"][0] / label["shape"][0],
                        label["new_shape"][1] / label["shape"][1])

        if not self.augment:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]

        bboxes = label.pop("box")
        bbox_format = label.pop("format")
        normalized = label.pop("norm")
        segments = np.zeros((0, 1000, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, bbox_format=bbox_format,
                                       normalized=normalized)

        return label

    def build_transforms(self):
        if self.mosaic:
            transforms = augment.transforms(self, self.args, self.params)
        else:
            transforms = augment.Compose([augment.LetterBox(new_shape=(
                self.args.inp_size, self.args.inp_size), scaleup=False)])

        transforms.append(augment.Format())
        return transforms

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in {"cls", 'box'}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["idx"] = list(new_batch["idx"])
        for i in range(len(new_batch["idx"])):
            new_batch["idx"][i] += i
        new_batch["idx"] = torch.cat(new_batch["idx"], 0)
        return new_batch