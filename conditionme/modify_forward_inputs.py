from dataclasses import dataclass
from typing import Optional

import torch
from torch import LongTensor


@dataclass
class NewForwardInputs:
    attention_mask: Optional[torch.Tensor]
    inputs_embeds: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    labels: Optional[torch.LongTensor]


def forward_inputs_with_rewards(
    reward_embedding: torch.nn.Linear,
    wte_embedding: torch.nn.Embedding,
    target_rewards: torch.Tensor,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> NewForwardInputs:
    # transform token inputs to embedding inputs
    inputs_embeds = wte_embedding(input_ids)
    # target reward is a 1d tensor of shape (batch_size,) so we need to unsqueeze it to get a 2d tensor
    target_rewards_reshaped = target_rewards.unsqueeze(1)
    # get the reward embedding
    reward_embeds = reward_embedding(target_rewards_reshaped)
    # reward embeds is of shape (batch_size, hidden_size), inputs_embeds is of shape (batch_size, sequence_length, hidden_size)
    # we want to concat the reward embeds to the inputs embeds so that the tensor is of shape (batch_size, sequence_length + 1, hidden_size
    # need to unsqueeze the reward embeds to get it to the right shape
    new_inputs_embeds = torch.cat([reward_embeds.unsqueeze(1), inputs_embeds], dim=1)
    # labels is a 2d tensor of shape (batch_size, sequence_length)
    # label the reward embedding as -100 so that it is of shape (batch_size, sequence_length + 1)
    new_labels: Optional[LongTensor] = (
        torch.cat( # type: ignore [assignment]
            [
                # -100 means that the reward embedding is masked. dtype long
                -100 * torch.ones_like(target_rewards).unsqueeze(1).long(),
                labels,
            ],
            dim=1,
        )
        if labels is not None
        else None
    )

    # edit the attention mask to account for the reward embedding
    new_attention_mask = (
        torch.cat(
            [
                # one means that the reward embedding is not masked
                torch.ones_like(target_rewards).unsqueeze(1),
                attention_mask,
            ],
            dim=1,
        )
        if attention_mask is not None
        else None
    )

    # add 1 to the position_ids that aren't masked by the attention mask
    offsetted_position_ids: Optional[torch.Tensor] = (
        position_ids + (attention_mask == 1).long()
        if attention_mask is not None and position_ids is not None
        # if there is no attention mask, then we'll just add 1 to the ids
        # TODO: this is rather gpt2 specific
        else position_ids + 1
        # if there are no position ids, then we'll just return None
        if position_ids is not None
        else None
    )
    # modify the position ids to account for the reward embedding
    new_position_ids = (
        torch.cat(
            [
                # 0 means that the reward embedding is at position 0
                torch.zeros_like(target_rewards).unsqueeze(1).long(),
                offsetted_position_ids,
            ],
            dim=1,
        )
        if offsetted_position_ids is not None
        else None
    )
    return NewForwardInputs(
        attention_mask=new_attention_mask,
        inputs_embeds=new_inputs_embeds,
        position_ids=new_position_ids,
        labels=new_labels,
    )
