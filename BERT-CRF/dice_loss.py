import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss for sequence labeling.

    Notes:
        - Expects raw logits of shape (batch, seq_len, num_labels).
        - Expects integer labels of shape (batch, seq_len) with padding marked as -1.
        - If `ignore_index` is -1 (default), positions with label == -1 are ignored.

    This implementation is designed to complement CRF training in this repo:
    use Dice on softmax logits (token-level) to fight label imbalance (dominant 'O'),
    then combine with CRF NLL.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -1,
        include_background: bool = True,
        background_index: int = 0,
    ):
        super().__init__()
        self.smooth = float(smooth)
        self.ignore_index = int(ignore_index)
        self.include_background = bool(include_background)
        self.background_index = int(background_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits must be (B, L, C), got {tuple(logits.shape)}")
        if target.dim() != 2:
            raise ValueError(f"target must be (B, L), got {tuple(target.shape)}")

        bsz, seq_len, num_labels = logits.shape

        # mask out padding/invalid positions
        if self.ignore_index is None:
            valid_mask = torch.ones((bsz, seq_len), device=logits.device, dtype=torch.bool)
        else:
            valid_mask = target.ne(self.ignore_index)

        if valid_mask.sum().item() == 0:
            # no valid labels -> no dice signal
            return logits.new_tensor(0.0)

        # clamp before one-hot to avoid negative indices
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        prob = F.softmax(logits, dim=-1)  # (B, L, C)

        # one-hot labels
        target_1h = F.one_hot(safe_target, num_classes=num_labels).to(prob.dtype)  # (B, L, C)

        # apply valid mask
        valid = valid_mask.to(prob.dtype).unsqueeze(-1)  # (B, L, 1)
        prob = prob * valid
        target_1h = target_1h * valid

        # reduce over batch+time
        dims = (0, 1)
        intersection = (prob * target_1h).sum(dims)
        card = (prob + target_1h).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (card + self.smooth)  # (C,)

        if not self.include_background and 0 <= self.background_index < num_labels:
            keep = torch.ones((num_labels,), device=logits.device, dtype=torch.bool)
            keep[self.background_index] = False
            dice = dice[keep]

        loss = 1.0 - dice.mean()
        return loss
