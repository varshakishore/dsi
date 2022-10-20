import torch.nn as nn
import copy
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers import T5Model
# from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel 






class T5decoder_projection(nn.Module):

    def __init__(self,projection_dim):
        super().__init__()  
        model_ = T5Model.from_pretrained('t5-base')
        self.T5Stack = copy.deepcopy(model_.decoder)
        self.projection = nn.Linear(768,projection_dim,bias=False)

    def set_input_embeddings(self, new_embeddings):
        self.T5Stack.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        # import pdb;pdb.set_trace()

        decoder_outputs = self.T5Stack(
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        )

        # return outputs

        # import pdb; pdb.set_trace()


        sequence_output = decoder_outputs.last_hidden_state

        # import pdb;pdb.set_trace()

        # print(sequence_output.shape)


        projection_output = self.projection(sequence_output)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=projection_output,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )






