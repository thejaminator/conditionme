from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import LongTensor


@dataclass
class NewForwardInputs:
    attention_mask: Optional[torch.Tensor]
    inputs_embeds: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    labels: Optional[torch.LongTensor]


def new_forward_inputs(
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]],
    reward_embedding: torch.nn.Linear,
    wte_embedding: torch.nn.Embedding,
    target_rewards: torch.Tensor,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    labels: Optional[torch.LongTensor],
):
    return (
        # Don't add the reward embedding if we are decoding
        _forward_inputs_without_rewards(
            wte_embedding=wte_embedding,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
        if past_key_values and len(past_key_values) > 0
        # Add the reward embedding if we are encoding
        # i.e. if we are training or encoding the "prompt"
        else _forward_inputs_with_rewards(
            reward_embedding=reward_embedding,
            wte_embedding=wte_embedding,
            target_rewards=target_rewards,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )
    )


def _forward_inputs_without_rewards(
    wte_embedding: torch.nn.Embedding,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    labels: Optional[torch.LongTensor],
) -> NewForwardInputs:
    return NewForwardInputs(
        attention_mask=torch.cat(
            [
                # Attention mask is (batch_size, sequence_length)
                # add a 1 to the attention mask to account for the reward embedding
                torch.ones(attention_mask.shape[0], 1, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )
        if attention_mask is not None
        else None,
        inputs_embeds=wte_embedding(input_ids),
        position_ids=position_ids,
        labels=labels,
    )


def _forward_inputs_with_rewards(
    reward_embedding: torch.nn.Linear,
    wte_embedding: torch.nn.Embedding,
    target_rewards: torch.Tensor,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    labels: Optional[torch.LongTensor],
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
        torch.cat(  # type: ignore [assignment]
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
        # TODO: this is rather gpt2 specific?
        # TODO: Does this mess up the padding positions?
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
