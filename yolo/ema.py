"""Bước 16: EMA (Exponential Moving Average) của trọng số."""
import copy
import math

import torch
from torch import nn


class ModelEMA:
    """Giữ một bản sao model có trọng số = trung bình trượt của trọng số qua các bước train:

        ema = d * ema + (1 - d) * model,      d = 0.9999 * (1 - exp(-số_bước / 2000))

    Trọng số sau mỗi bước train dao động quanh điểm tốt (vì mỗi batch kéo một hướng). Lấy trung bình nhiều bước gần
    đây cho một model mượt hơn, thường đo được mAP cao hơn và ổn định hơn. d tăng dần từ 0 lên 0.9999: lúc đầu model
    đổi nhanh nên EMA bám sát, về sau EMA nhớ khoảng 10000 bước gần nhất.

    Buffer của BatchNorm (running_mean, running_var) cũng được lấy trung bình như vậy. Đo mAP và lưu checkpoint bằng
    self.ema, không bằng model đang train.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: float = 2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / self.tau))
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():  # state_dict trả về tensor dùng chung bộ nhớ: sửa tại chỗ là đủ
            if v.dtype.is_floating_point:            # bỏ qua num_batches_tracked (số nguyên)
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
