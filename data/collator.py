"""
Data collators — applied on-the-fly during training.

DataCollatorMPP  — randomly masks 25% of players in each match, zeros out
                   their form_stats, and builds labels (-100 for unmasked).
DataCollatorNMSP — collates NMSP batches (no masking needed).
"""

import torch
from typing import Optional


class DataCollatorMPP:
    """Collator for Masked Player Prediction.

    Takes a list of dicts (from MatchDatasetMPP.__getitem__) and returns
    a single batched dict with masking applied.

    For each match in the batch:
        1. Identify maskable positions (attention_mask == 1).
        2. Randomly select mask_percentage of them.
        3. Replace their input_ids with player_mask_token_id.
        4. Zero out their form_stats.
        5. Set labels: original player id for masked, -100 for unmasked.

    Args:
        player_mask_token_id: special token id for [MASK].
        mask_percentage: fraction of real players to mask (default 0.25).
    """

    def __init__(
        self,
        player_mask_token_id: int,
        mask_percentage: float = 0.25,
    ):
        self.player_mask_token_id = player_mask_token_id
        self.mask_percentage = mask_percentage

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate and apply masking to a batch of matches.

        Args:
            batch: list of dicts from MatchDatasetMPP.

        Returns:
            Dict with keys: input_ids, labels, position_id, team_id,
            form_stats, attention_mask — all batched tensors.
        """
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("DataCollatorMPP received empty batch or all None items")

        masked_input_ids = []
        labels_list = []
        masked_form_stats_list = []
        for item in batch:
            mid, lab, mfs = self._mask_single_match(
                item["input_ids"],
                item["form_stats"],
                item["attention_mask"],
            )
            masked_input_ids.append(mid)
            labels_list.append(lab)
            masked_form_stats_list.append(mfs)

        return {
            "input_ids": torch.stack(masked_input_ids),
            "labels": torch.stack(labels_list),
            "position_id": torch.stack([b["position_id"] for b in batch]),
            "team_id": torch.stack([b["team_id"] for b in batch]),
            "form_stats": torch.stack(masked_form_stats_list),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

    def _mask_single_match(
        self,
        input_ids: torch.Tensor,
        form_stats: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply masking to a single match.

        Args:
            input_ids: (seq_len,) original player IDs.
            form_stats: (seq_len, stats_dim) player statistics.
            attention_mask: (seq_len,) 1=real player, 0=padding.

        Returns:
            (masked_input_ids, labels, masked_form_stats)
        """
        seq_len = input_ids.shape[0]
        device = input_ids.device
        masked_input_ids = input_ids.clone()
        labels = torch.full((seq_len,), -100, dtype=torch.long, device=device)
        masked_form_stats = form_stats.clone()

        maskable = (attention_mask == 1).nonzero(as_tuple=True)[0]
        n_real = len(maskable)
        if n_real == 0:
            return masked_input_ids, labels, masked_form_stats

        n_to_mask = max(0, int(round(n_real * self.mask_percentage)))
        if n_to_mask == 0:
            return masked_input_ids, labels, masked_form_stats

        perm = torch.randperm(n_real, device=device)
        to_mask = maskable[perm[:n_to_mask]]

        masked_input_ids[to_mask] = self.player_mask_token_id
        labels[to_mask] = input_ids[to_mask]
        masked_form_stats[to_mask, :] = 0.0

        return masked_input_ids, labels, masked_form_stats


class DataCollatorNMSP:
    """Collator for NMSP — simple batching, no masking.

    Filters out None items, stacks tensors into a batch.
    """

    def __call__(
        self, batch: list[Optional[dict[str, torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        ...
