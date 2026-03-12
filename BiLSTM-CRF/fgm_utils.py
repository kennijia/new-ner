import torch


class FGM:
    """Fast Gradient Method adversarial training helper."""

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon: float = 1.0, emb_name: str = "embedding"):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if emb_name not in name:
                continue
            if param.grad is None:
                continue
            self.backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm != 0 and not torch.isnan(norm):
                r_at = epsilon * param.grad / norm
                param.data.add_(r_at)

    def restore(self, emb_name: str = "embedding"):
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if emb_name not in name:
                continue
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
