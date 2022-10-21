import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqModelOutput
from transformers import  T5Model

class T5Model_projection(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()
        self.model = T5Model.from_pretrained('t5-base')
        self.projection = nn.Linear(768,projection_dim,bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        
        outputs = self.model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

        projection_output = self.projection(outputs.last_hidden_state)

        return projection_output