from logging import Logger
from typing import Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin, GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import decision.modified_gpt2_forward
from decision.logger import logger


class ModifiedGPT2LMHeadModel(nn.Module, GenerationMixin):
    def __init__(self, existing_head_model: GPT2LMHeadModel, logger: Logger = logger):
        super().__init__()
        self.existing_head_model: GPT2LMHeadModel = existing_head_model
        self.existing_transformer_model: GPT2Model = existing_head_model.transformer
        self.config = existing_head_model.config
        self.generation_config = existing_head_model.generation_config
        self.main_input_name = existing_head_model.main_input_name
        self.device = existing_head_model.device

    # For GenerationMixin
    # `generate` of GenerationMixin calls this method
    def prepare_inputs_for_generation(
        self, input_ids, target_reward: float, past_key_values=None, **kwargs
    ):
        final_kwargs = self.existing_head_model.prepare_inputs_for_generation(
            input_ids=input_ids, past_key_values=past_key_values, **kwargs
        )
        final_kwargs["target_reward"] = target_reward
        return final_kwargs

    # For GenerationMixin
    def can_generate(self, **kwargs):
        return True

    def forward(
        self,
        target_reward: float,
        input_ids: Optional[torch.LongTensor] = None,
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
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict
            if return_dict is not None
            else self.existing_head_model.config.use_return_dict
        )

        transformer_outputs = (
            decision.modified_gpt2_forward.modfied_transformer_forward(
                target_reward=target_reward,
                transformer_model=self.existing_head_model.transformer,
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.existing_head_model.model_parallel:
            torch.cuda.set_device(self.existing_head_model.transformer.first_device)
            hidden_states = hidden_states.to(
                self.existing_head_model.lm_head.weight.device
            )

        lm_logits = self.existing_head_model.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )