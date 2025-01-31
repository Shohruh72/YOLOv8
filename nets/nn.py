import math
import torch
import torch.nn as nn
from utils import util


def fuse_conv(conv, norm):
    fused_conv = torch.nn.Conv2d(conv.in_channels,
                                 conv.out_channels,
                                 kernel_size=conv.kernel_size,
                                 stride=conv.stride,
                                 padding=conv.padding,
                                 groups=conv.groups,
                                 bias=True).requires_grad_(False).to(
        conv.weight.device)

    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_norm = torch.diag(
        norm.weight.div(torch.sqrt(norm.eps + norm.running_var)))
    fused_conv.weight.copy_(
        torch.mm(w_norm, w_conv).view(fused_conv.weight.size()))

    b_conv = torch.zeros(conv.weight.size(0),
                         device=conv.weight.device) if conv.bias is None else conv.bias
    b_norm = norm.bias - norm.weight.mul(norm.running_mean).div(
        torch.sqrt(norm.running_var + norm.eps))
    fused_conv.bias.copy_(
        torch.mm(w_norm, b_conv.reshape(-1, 1)).reshape(-1) + b_norm)

    return fused_conv


class Conv(nn.Module):
    def __init__(self, inp, oup, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, self._pad(k, p), d, g, False)
        self.norm = nn.BatchNorm2d(oup)
        self.act = nn.SiLU(inplace=True) if act is True else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    @staticmethod
    def _pad(k, p=None):
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        return p


class Residual(torch.nn.Module):
    def __init__(self, ch, add=True):
        super().__init__()
        self.add_m = add
        self.res_m = torch.nn.Sequential(Conv(ch, ch, 3),
                                         Conv(ch, ch, 3))

    def forward(self, x):
        return self.res_m(x) + x if self.add_m else self.res_m(x)


class SPP(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        self.conv1 = Conv(in_ch, in_ch // 2)
        self.conv2 = Conv(in_ch * 2, out_ch)
        self.res_m = torch.nn.MaxPool2d(k, 1, k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.res_m(x)
        y2 = self.res_m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.res_m(y2)], 1))


class CSP(nn.Module):
    def __init__(self, c1, c2, n=1, add=True, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = torch.nn.ModuleList(Residual(c2 // 2, add) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DWConv(Conv):
    def __init__(self, inp, oup, k=1, s=1, d=1, act=True):
        super().__init__(inp, oup, k, s, g=math.gcd(inp, oup), d=d, act=act)


class DFL(nn.Module):
    def __init__(self, inp=16):
        super().__init__()
        self.inp = inp
        self.conv = nn.Conv2d(inp, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(inp, dtype=torch.float).view(1, inp, 1, 1)
        self.conv.weight.data[:] = nn.Parameter(x)

    def forward(self, x):
        b, _, a = x.shape
        out = x.view(b, 4, self.inp, a).transpose(2, 1)
        return self.conv(out.softmax(1)).view(b, 4, a)


class Detect(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.nc = nc
        self.reg_max = 16
        self.nl = len(filters)
        self.no = nc + self.reg_max * 4
        self.stride = torch.zeros(self.nl)

        cls = max(filters[0], self.nc)
        box = max((filters[0] // 4, self.reg_max * 4))

        self.box = nn.ModuleList(
            nn.Sequential(Conv(x, box, 3), Conv(box, box, 3),
                          torch.nn.Conv2d(box, self.reg_max * 4, 1)) for x in
            filters)

        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(x, cls, 3), Conv(cls, cls, 3),
                                torch.nn.Conv2d(cls, self.nc, 1)) for
            x in filters)

        self.dfl = DFL(self.reg_max)

    def forward(self, x):
        for i in range(self.nl):
            box_out = self.box[i](x[i])
            cls_out = self.cls[i](x[i])
            x[i] = torch.cat((box_out, cls_out), 1)
        if self.training:
            return x
        self.anchors, self.strides = (x.transpose(0, 1) for x in
                                      util.make_anchors(x, self.stride))

        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], 2)
        box, cls = x.split((self.reg_max * 4, self.nc), 1)
        a, b = torch.split(self.dfl(box), 2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(((a + b) / 2, b - a), 1)
        return torch.cat((box * self.strides, cls.sigmoid()), 1)

    def initialize_biases(self):
        m = self
        for a, b, s in zip(m.box, m.cls, m.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)


class Backbone(nn.Module):
    def __init__(self, width, depth):
        super().__init__()

        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        self.p1.append(Conv(width[0], width[1], 3, 2))

        self.p2.append(Conv(width[1], width[2], 3, 2))
        self.p2.append(CSP(width[2], width[2], depth[0]))

        self.p3.append(Conv(width[2], width[3], 3, 2))
        self.p3.append(CSP(width[3], width[3], depth[1]))

        self.p4.append(Conv(width[3], width[4], 3, 2))
        self.p4.append(CSP(width[4], width[4], depth[2]))

        self.p5.append(Conv(width[4], width[5], 3, 2))
        self.p5.append(CSP(width[5], width[5], depth[0]))
        self.p5.append(SPP(width[5], width[5]))

        self.p1 = nn.Sequential(*self.p1)
        self.p2 = nn.Sequential(*self.p2)
        self.p3 = nn.Sequential(*self.p3)
        self.p4 = nn.Sequential(*self.p4)
        self.p5 = nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5


class Head(torch.nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        self.up = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.h1 = CSP(width[4] + width[5], width[4], depth[0], False)
        self.h2 = CSP(width[3] + width[4], width[3], depth[0], False)
        self.h3 = Conv(width[3], width[3], 3, 2)
        self.h4 = CSP(width[3] + width[4], width[4], depth[0], False)
        self.h5 = Conv(width[4], width[4], 3, 2)
        self.h6 = CSP(width[4] + width[5], width[5], depth[0], False)

    def forward(self, x):
        p3, p4, p5 = x
        h1 = self.h1(torch.cat([self.up(p5), p4], 1))
        h2 = self.h2(torch.cat([self.up(h1), p3], 1))
        h4 = self.h4(torch.cat([self.h3(h2), h1], 1))
        h6 = self.h6(torch.cat([self.h5(h4), p5], 1))

        return h2, h4, h6


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in {nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU}:
            m.inplace = True


class YOLO(torch.nn.Module):
    def __init__(self, num_cls, width, depth):
        super().__init__()
        self.backbone = Backbone(width, depth)
        self.head = Head(width, depth)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.detect = Detect(num_cls, (width[3], width[4], width[5]))
        self.detect.stride = torch.tensor(
            [256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.detect.stride
        self.detect.initialize_biases()
        initialize_weights(self)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return self.detect(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.forward_fuse
                delattr(m, 'norm')
        return self


def yolo_v8_n(num_cls=80):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLO(num_cls, width, depth)


def yolo_v8_s(num_cls=80):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLO(num_cls, width, depth)


def yolo_v8_m(num_cls=80):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLO(num_cls, width, depth)


def yolo_v8_l(num_cls=80):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLO(num_cls, width, depth)


def yolo_v8_x(num_cls=80):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLO(num_cls, width, depth)
