import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Multi-class Dice loss for sequence labeling.

    Expects:
        logits: (B, L, C) raw logits
        target: (B, L) int labels, padding marked as `ignore_index` (default -1)

    Designed to be combined with CRF negative log-likelihood:
        total_loss = crf_nll + dice_loss_weight * dice_loss

    Notes:
        - Uses softmax probabilities.
        - Optionally excludes background class (often 'O' with id=0) from averaging.
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
        self.ignore_index = int(ignore_index) if ignore_index is not None else None
        self.include_background = bool(include_background)
        self.background_index = int(background_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3:
            raise ValueError(f"logits must be (B, L, C), got {tuple(logits.shape)}")
        if target.dim() != 2:
            raise ValueError(f"target must be (B, L), got {tuple(target.shape)}")

        bsz, seq_len, num_labels = logits.shape

        if self.ignore_index is None:
            valid_mask = torch.ones((bsz, seq_len), device=logits.device, dtype=torch.bool)
        else:
            valid_mask = target.ne(self.ignore_index)

        if valid_mask.sum().item() == 0:
            return logits.new_tensor(0.0)

        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        prob = F.softmax(logits, dim=-1)
        target_1h = F.one_hot(safe_target, num_classes=num_labels).to(prob.dtype)

        valid = valid_mask.to(prob.dtype).unsqueeze(-1)
        prob = prob * valid
        target_1h = target_1h * valid

        dims = (0, 1)
        intersection = (prob * target_1h).sum(dims)
        card = (prob + target_1h).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (card + self.smooth)

        if not self.include_background and 0 <= self.background_index < num_labels:
            keep = torch.ones((num_labels,), device=logits.device, dtype=torch.bool)
            keep[self.background_index] = False
            dice = dice[keep]

        return 1.0 - dice.mean()
