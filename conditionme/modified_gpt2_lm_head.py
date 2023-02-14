import os
from logging import Logger
from typing import Optional, Tuple, Union, Callable, List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    GPT2LMHeadModel,
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    GPT2Config,
)
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_utils import ModuleUtilsMixin, PreTrainedModel
from transformers.utils import PushToHubMixin

from conditionme.logger import logger
from conditionme.modify_forward_inputs import (
    NewForwardInputs,
    forward_inputs_with_rewards,
)


class ModifiedGPT2LMHeadModel(PreTrainedModel):
    config_class = GPT2Config
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.pretrained_model: GPT2LMHeadModel = GPT2LMHeadModel(config)
        self.embed_return = torch.nn.Linear(
            1, self.pretrained_model.transformer.config.hidden_size
        )

    @staticmethod
    def from_loaded_pretrained_model(
        loaded_model: GPT2LMHeadModel,
    ):
        config = loaded_model.config
        model = ModifiedGPT2LMHeadModel(config)
        model.pretrained_model = loaded_model
        return model

    # For GenerationMixin
    # `generate` of GenerationMixin calls this method
    def prepare_inputs_for_generation(
        self,
        input_ids,
        target_reward: torch.Tensor,
        past_key_values=None,
        **kwargs,
    ):
        final_kwargs = self.pretrained_model.prepare_inputs_for_generation(
            input_ids=input_ids, past_key_values=past_key_values, **kwargs
        )
        final_kwargs["target_reward"] = target_reward
        return final_kwargs

    # For GenerationMixin
    def can_generate(self, **kwargs):
        return True

    def forward(
        self,
        target_reward: torch.Tensor,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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
            else self.pretrained_model.config.use_return_dict
        )
        # START OF EDITS
        new_inputs: NewForwardInputs = (
            # If we are using past key values, we have already added the reward embedding to the input
            # So here we just pass the existing inputs
            # todo: refactor
            NewForwardInputs(
                attention_mask=torch.cat(
                    [
                        # Attention mask is (batch_size, sequence_length)
                        # add a 1 to the attention mask to account for the reward embedding
                        torch.ones(
                            attention_mask.shape[0], 1, device=attention_mask.device
                        ),
                        attention_mask,
                    ],
                    dim=1,
                )
                if attention_mask is not None
                else None,
                inputs_embeds=self.pretrained_model.transformer.wte(input_ids),
                position_ids=position_ids,
                labels=labels,
            )
            if past_key_values and len(past_key_values) > 0
            else forward_inputs_with_rewards(
                reward_embedding=self.embed_return,
                wte_embedding=self.pretrained_model.transformer.wte,
                target_reward=target_reward,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
            )
        )

        transformer_outputs = self.pretrained_model.transformer.forward(
            inputs_embeds=new_inputs.inputs_embeds,
            past_key_values=past_key_values,
            attention_mask=new_inputs.attention_mask,
            token_type_ids=token_type_ids,
            position_ids=new_inputs.position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        labels = new_inputs.labels
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.pretrained_model.model_parallel:
            torch.cuda.set_device(self.pretrained_model.transformer.first_device)
            hidden_states = hidden_states.to(
                self.pretrained_model.lm_head.weight.device
            )

        lm_logits = self.pretrained_model.lm_head(hidden_states)

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

    # this breaks liskov, but it is the only way to work with to maintain compat with GenerationMixin
    def generate(
        self,
        target_reward: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[
            Callable[[int, torch.Tensor], List[int]]
        ] = None,
        synced_gpus: Optional[bool] = False,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        return super().generate(
            target_reward=target_reward,
            inputs=inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            **kwargs,
        )
