import torch

from colbert.utils.utils import NullContextManager


# PyTorch_over_1_6 = float('.'.join(torch.__version__.split('.')[0:2])) >= 1.6


class MixedPrecisionManager:
    def __init__(self, activated):
        # assert (not activated) or PyTorch_over_1_6, "Cannot use AMP for PyTorch version < 1.6"

        self.activated = activated
        # self.activated = False

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        # return torch.cuda.amp.autocast() if self.activated else NullContextManager()
        return torch.autocast(dtype=torch.bfloat16, device_type="cuda") if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, models, optimizers):
        if type(optimizers) is not list:
            optimizers = [optimizers]
            models = [models]
        for modules, optimizer in zip(models, optimizers):
            if self.activated:
                self.scaler.unscale_(optimizer)
                for module in modules:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            else:
                for module in modules:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
                optimizer.step()

        if self.activated:
            for optimizer in optimizers:
                self.scaler.step(optimizer)
            self.scaler.update()
