import cv2
import math
import random
import numpy as np
import albumentations as album

import torch
from ultralytics.utils.instance import Instances


class Mosaic:
    def __init__(self, args, params, dataset):
        self.args = args
        self.params = params
        self.dataset = dataset

        self.s = self.args.inp_size
        self.border = (-self.s // 2, -self.s // 2)

    def __call__(self, labels):
        if random.uniform(0, 1) > 1.0:
            return labels

        indexes = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]
        if isinstance(indexes, int):
            indexes = [indexes]

        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]
        labels["mix_labels"] = mix_labels

        labels = self._mosaic4(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mosaic4(self, labels):
        mosaic_labels = []
        yc, xc = (int(random.uniform(-x, 2 * self.s + x)) for x in self.border)
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("new_shape")

            if i == 0:  # top left
                img4 = np.full((self.s * 2, self.s * 2, img.shape[2]), 114,
                               dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (xc, yc, min(xc + w, self.s * 2),
                                      min(self.s * 2,
                                          yc + h))
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _cat_labels(self, mosaic_labels):
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        imgsz = self.args.inp_size * 2
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "image": mosaic_labels[0]["image"],
            "shape": mosaic_labels[0]["shape"],
            "new_shape": (imgsz, imgsz),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(imgsz, imgsz)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels

    @staticmethod
    def _update_labels(labels, padw, padh):
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels


class RandomPerspective:
    def __init__(self, params, pre_transform=None):
        self.params = params
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        val = (114, 114, 114)
        # Center
        center = np.eye(3, dtype=np.float32)

        center[0, 2] = -img.shape[1] / 2
        center[1, 2] = -img.shape[0] / 2

        # Perspective
        pers = np.eye(3, dtype=np.float32)
        _pers = self.params['psp']
        pers[2, 0] = random.uniform(-_pers, _pers)
        pers[2, 1] = random.uniform(-_pers, _pers)

        # Rotation and Scale
        rotate = np.eye(3, dtype=np.float32)
        deg = self.params['degree']
        scale = self.params['scale']
        a = random.uniform(-deg, deg)
        s = random.uniform(1 - scale, 1 + scale)
        rotate[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        shear = np.eye(3, dtype=np.float32)
        _shear = self.params['shear']
        shear[0, 1] = math.tan(random.uniform(-_shear, _shear) * math.pi / 180)
        shear[1, 0] = math.tan(random.uniform(-_shear, _shear) * math.pi / 180)

        # Translation
        translate = np.eye(3, dtype=np.float32)
        _translate = self.params['translate']
        translate[0, 2] = random.uniform(0.5 - _translate, 0.5 + _translate) * \
                          self.size[0]
        translate[1, 2] = random.uniform(0.5 - _translate, 0.5 + _translate) * \
                          self.size[1]

        matrix = translate @ shear @ rotate @ pers @ center

        if (border[0] != 0) or (border[1] != 0) or (matrix != np.eye(3)).any():
            if _pers:
                img = cv2.warpPerspective(img, matrix, dsize=self.size,
                                          borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, matrix[:2], dsize=self.size,
                                     borderValue=(114, 114, 114))

        return img, matrix, s

    def __call__(self, labels):
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
        labels.pop("pad", None)

        img, cls, objs = labels["img"], labels["cls"], labels.pop("instances")
        objs.convert_bbox(format="xyxy")
        objs.denormalize(*img.shape[:2][::-1])

        border = labels.pop("mosaic_border", (0, 0))
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2

        img, matrix, scale = self.affine_transform(img, border)

        n = len(objs.bboxes)
        if n == 0:
            return objs.bboxes

        xy = np.ones((len(objs.bboxes) * 4, 3), dtype=objs.bboxes.dtype)
        xy[:, :2] = objs.bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            len(objs.bboxes) * 4, 2)
        xy = xy @ matrix.T

        xy = (xy[:, :2] / xy[:, 2:3] if self.params['psp'] else xy[:,
                                                                :2])

        xy = xy.reshape(len(objs.bboxes), 8)

        x, y = xy[:, [0, 2, 4, 6]], xy[:, [1, 3, 5, 7]]

        bbox_coords = (x.min(1), y.min(1), x.max(1), y.max(1))
        bboxes = np.concatenate(bbox_coords, dtype=objs.bboxes.dtype)
        bboxes = bboxes.reshape(4, len(objs.bboxes)).T

        new_instances = Instances(bboxes, objs.segments, None,
                                  bbox_format="xyxy", normalized=False)

        new_instances.clip(*self.size)
        objs.scale(scale_w=scale, scale_h=scale, bbox_only=True)

        i = self.check_boxes(box1=objs.bboxes.T,
                             box2=new_instances.bboxes.T, area_thr=0.10)

        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["new_shape"] = img.shape[:2]
        return labels

    @staticmethod
    def check_boxes(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (w2 > wh_thr) & (h2 > wh_thr) & (
                w2 * h2 / (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)


class LetterBox:
    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False,
                 scaleup=True, center=True, stride=32):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
            1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh,
                                                     self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[
                0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(
            round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(
            round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=(114, 114, 114)
        )  # add border
        if labels.get("pad"):
            labels["pad"] = (
                labels["pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, left, top)
            labels["img"] = img
            labels["new_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class Albumentations:
    def __init__(self, p=1.0):
        self.p = p
        self.transform = None

        transform = [album.Blur(p=0.01),
                     album.MedianBlur(p=0.01),
                     album.ToGray(p=0.01),
                     album.CLAHE(p=0.01),
                     album.RandomBrightnessContrast(p=0.0),
                     album.RandomGamma(p=0.0),
                     album.ImageCompression(quality_lower=75, p=0.0), ]

        self.transform = album.Compose(transform)

    def __call__(self, labels):
        if self.transform is None or random.random() > 0:
            return labels

        labels["img"] = self.transform(image=labels["img"])["image"]
        return labels


class RandomHSV:
    def __init__(self, params):
        self.params = params
        self.h = params["hsv_h"]
        self.s = params["hsv_s"]
        self.v = params["hsv_v"]

    def __call__(self, labels):
        img = labels["img"]
        if self.h or self.s or self.v:
            r = np.random.uniform(-1, 1, 3) * [self.h, self.s, self.v] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                                cv2.LUT(val, lut_val)))
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return labels


class RandomFlip:
    def __init__(self, direction, p):
        self.p = p
        self.direction = direction

    def __call__(self, labels):
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)

        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms if isinstance(transforms, list) else [
            transforms]

    def __call__(self, _data):
        for t in self.transforms:
            _data = t(_data)
        return _data

    def append(self, transform):
        self.transforms.append(transform)

    def insert(self, index, transform):
        self.transforms.insert(index, transform)

    def __getitem__(self, index):
        assert isinstance(index, (int,
                                  list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index, value):
        assert isinstance(index, (int,
                                  list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(value,
                              list), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(
                self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        return self.transforms

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class Format:
    def __init__(self, bbox_format="xywh", bgr=0.0):
        self.bbox_format = bbox_format
        self.bgr = bgr

    def __call__(self, labels):
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)

        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(
            img[::-1] if random.uniform(0, 1) > self.bgr else img)
        labels["img"] = torch.from_numpy(img)

        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["box"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        labels["box"][:, [0, 2]] /= w
        labels["box"][:, [1, 3]] /= h
        labels["idx"] = torch.zeros(nl)
        return labels


class CopyPaste:
    def __init__(self, dataset=None, pre_transform=None):
        self.dataset = dataset
        self.pre_transform = pre_transform

    def __call__(self, labels):
        return labels


class MixUp:
    def __init__(self, dataset, pre_transform=None):
        self.dataset = dataset
        self.pre_transform = pre_transform

    def __call__(self, labels):
        if random.uniform(0, 1) > 0:
            return labels

        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def get_indexes(self):
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        r = np.random.beta(32.0, 32.0)
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(
            np.uint8)
        labels["instances"] = Instances.concatenate(
            [labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


def transforms(dataset, args, params):
    mosaic = Mosaic(args, params, dataset)

    affine = RandomPerspective(params, pre_transform=LetterBox(
        new_shape=(args.inp_size, args.inp_size)))

    pre_transform = Compose([mosaic, affine])
    pre_transform.insert(1, CopyPaste())
    return Compose([pre_transform,
                    MixUp(dataset, pre_transform=pre_transform),
                    Albumentations(), RandomHSV(params),
                    RandomFlip(direction="vertical", p=params['flip_ud']),
                    RandomFlip(direction="horizontal", p=params['flip_lr'])])