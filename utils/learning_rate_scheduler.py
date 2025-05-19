import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWithLinearDecay(_LRScheduler):
    def __init__(self, optimizer, T_max, lr_min=0, max_lr_start=0.1, max_lr_end=0.0, num_cycles=5, last_epoch=-1):
        self.T_max = T_max  # total steps per cycle
        self.lr_min = lr_min
        self.max_lr_start = max_lr_start
        self.max_lr_end = max_lr_end
        self.num_cycles = num_cycles
        self.max_lr_decay = (max_lr_start - max_lr_end) / max(1, num_cycles - 1)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        cycle = step // self.T_max
        cycle_step = step % self.T_max

        if cycle >= self.num_cycles:
            return [self.lr_min for _ in self.base_lrs]

        # Linearly decaying max LR per cycle
        max_lr = max(self.max_lr_start - self.max_lr_decay * cycle, self.max_lr_end)

        # Cosine annealing between [max_lr, lr_min], centered to give full wave
        cosine = 0.5 * (1 + math.cos(math.pi * (2 * cycle_step / self.T_max - 1)))
        return [self.lr_min + (max_lr - self.lr_min) * cosine for _ in self.base_lrs]

class FullCosineAnnealingWithGeometricDecay(_LRScheduler):
    # At each minimum of the cosine wave the max_lr is decayed by a factor
    def __init__(self, optimizer, T_max, lr_min=0.0, max_lr=0.1, decay_factor=0.9, num_cycles=5, last_epoch=-1, start_max=False):
        self.T_max = T_max
        self.lr_min = lr_min
        self.initial_max_lr = max_lr
        self.decay_factor = decay_factor
        self.num_cycles = num_cycles
        self.max_lr = self.initial_max_lr
        self.start_max = start_max
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        cycle = step // self.T_max
        cycle_step = step % self.T_max
        half_cycle = self.T_max // 2

        if cycle >= self.num_cycles:
            return [self.lr_min for _ in self.base_lrs]

        # Exponentially decaying max_lr
        if cycle_step == 0 and not self.start_max:
            self.max_lr = self.initial_max_lr * (self.decay_factor ** cycle)
        elif cycle_step == half_cycle and self.start_max:
            self.max_lr = self.initial_max_lr * (self.decay_factor ** cycle)
            
            
        # Full cosine wave
        if not self.start_max:
            cosine = 0.5 * (1 + math.cos(math.pi * (2 * cycle_step / self.T_max - 1)))
        else:
            cosine = 0.5 * (1 + math.cos(math.pi * 2 * cycle_step / self.T_max))
        #cosine = 0.5 * (1 + math.cos(2 * math.pi * cycle_step / self.T_max))
        return [self.lr_min + (self.max_lr - self.lr_min) * cosine for _ in self.base_lrs]


def get_scheduler(name):
    if name == "CosineAnnealingWithLinearDecay":
        return CosineAnnealingWithLinearDecay
    elif name == "FullCosineAnnealingWithGeometricDecay":
        return FullCosineAnnealingWithGeometricDecay
    else:
        raise Exception(f"Unknown scheduler: {name}")