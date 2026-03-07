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
        ...

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
        ...

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
        ...


class DataCollatorNMSP:
    """Collator for NMSP — simple batching, no masking.

    Filters out None items, stacks tensors into a batch.
    """

    def __call__(
        self, batch: list[Optional[dict[str, torch.Tensor]]]
    ) -> dict[str, torch.Tensor]:
        ...
