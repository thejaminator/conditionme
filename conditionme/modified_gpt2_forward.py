from typing import Optional, Tuple

import torch
from transformers import GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from decision.logger import logger


def modfied_transformer_forward(
    target_reward: float,
    transformer_model: GPT2Model,
    input_ids: torch.LongTensor,
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
):
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else transformer_model.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else transformer_model.config.output_hidden_states
    )
    use_cache = (
        use_cache if use_cache is not None else transformer_model.config.use_cache
    )
    return_dict = (
        return_dict
        if return_dict is not None
        else transformer_model.config.use_return_dict
    )

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
        batch_size = inputs_embeds.shape[0]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if token_type_ids is not None:
        token_type_ids = token_type_ids.view(-1, input_shape[-1])
    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * len(transformer_model.h))
    else:
        past_length = past_key_values[0][0].size(-2)
    if position_ids is None:
        position_ids = torch.arange(
            past_length,
            input_shape[-1] + past_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # GPT2Attention mask.
    if attention_mask is not None:
        if batch_size <= 0:
            raise ValueError("batch_size has to be defined and > 0")
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(
            dtype=transformer_model.dtype
        )  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(
            transformer_model.dtype
        ).min

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if (
        transformer_model.config.add_cross_attention
        and encoder_hidden_states is not None
    ):
        (
            encoder_batch_size,
            encoder_sequence_length,
            _,
        ) = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_attention_mask = transformer_model.invert_attention_mask(
            encoder_attention_mask
        )
    else:
        encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = transformer_model.get_head_mask(
        head_mask, transformer_model.config.n_layer
    )

    if inputs_embeds is None:
        inputs_embeds = transformer_model.wte(input_ids)
    position_embeds = transformer_model.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds
    # the first token (eos) will have the target_reward added to it
    # Add the target_reward to the last dimension of hidden_states, of the first token
    modified_hidden_states = hidden_states.clone()
    modified_hidden_states[:, 0, -1] += target_reward
    hidden_states = modified_hidden_states

    if token_type_ids is not None:
        token_type_embeds = transformer_model.wte(token_type_ids)
        hidden_states = hidden_states + token_type_embeds

    hidden_states = transformer_model.drop(hidden_states)

    output_shape = input_shape + (hidden_states.size(-1),)

    presents = () if use_cache else None
    all_transformer_model_attentions = () if output_attentions else None
    all_cross_attentions = (
        ()
        if output_attentions and transformer_model.config.add_cross_attention
        else None
    )
    all_hidden_states = () if output_hidden_states else None
    for i, (block, layer_past) in enumerate(zip(transformer_model.h, past_key_values)):

        # Model parallel
        if transformer_model.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure layer_past is on same device as hidden_states (might not be correct)
            if layer_past is not None:
                layer_past = tuple(
                    past_state.to(hidden_states.device) for past_state in layer_past
                )
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if isinstance(head_mask, torch.Tensor):
                head_mask = head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if transformer_model.gradient_checkpointing and transformer_model.training:

            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, use_cache, output_attentions)

                return custom_forward

            outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                None,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

        hidden_states = outputs[0]
        if use_cache is True:
            presents = presents + (outputs[1],)

        if output_attentions:
            all_transformer_model_attentions = all_transformer_model_attentions + (
                outputs[2 if use_cache else 1],
            )
            if transformer_model.config.add_cross_attention:
                all_cross_attentions = all_cross_attentions + (
                    outputs[3 if use_cache else 2],
                )

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if transformer_model.model_parallel:
            for k, v in transformer_model.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != transformer_model.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = transformer_model.ln_f(hidden_states)

    hidden_states = hidden_states.view(output_shape)
    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                presents,
                all_hidden_states,
                all_transformer_model_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_transformer_model_attentions,
        cross_attentions=all_cross_attentions,
    )