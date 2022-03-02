import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F

from modeling import *
from quantize import QuantizedLinear


class QuantizedLinearActivation(QuantizedLinear):
    def __init__(self, in_features, out_features, act, n_bits=8):
        super(QuantizedLinearActivation, self).__init__(in_features, out_features, n_bits)
        self.act_fn = ACT2FN[act] if act != "noact" else nn.Identity()
        self.is_bias = "bias" in act

    def forward(self, x):
        x = self._forward(x)
        x = self.act_fn(self.bias, x) if self.is_bias else self.act_fn(x)
        return x


class QuantizedSelfAttention(Module):
    def __init__(self, config, n_bits):
        super(QuantizedSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = QuantizedLinearActivation(
            config.hidden_size, self.all_head_size, act='noact', n_bits=n_bits)
        self.key = QuantizedLinearActivation(
            config.hidden_size, self.all_head_size, act='noact', n_bits=n_bits)
        self.value = QuantizedLinearActivation(
            config.hidden_size, self.all_head_size, act='noact', n_bits=n_bits)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_key_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = torch.reshape(x, new_x_shape)
        return x.permute(0, 2, 3, 1)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = torch.reshape(context_layer, new_context_layer_shape)
        return context_layer


class QuantizedSelfOutput(nn.Module):
    def __init__(self, config, n_bits):
        super(QuantizedSelfOutput, self).__init__()
        self.dense = QuantizedLinearActivation(
            config.hidden_size, config.hidden_size, act='noact', n_bits=n_bits)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertQuantizedAttention(Module):
    def __init__(self, config, n_bits=8):
        super(BertQuantizedAttention, self).__init__()
        self.self = QuantizedSelfAttention(config, n_bits)
        self.output = QuantizedSelfOutput(config, n_bits)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertQuantizedIntermediate(nn.Module):
    def __init__(self, config, n_bits=8):
        super(BertQuantizedIntermediate, self).__init__()
        self.dense = QuantizedLinearActivation(
            config.hidden_size, config.intermediate_size, act=config.hidden_act, n_bits=n_bits)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states


class BertQuantizedOutput(Module):
    def __init__(self, config, n_bits=8):
        super(BertQuantizedOutput, self,).__init__()
        self.dense = QuantizedLinearActivation(
            config.intermediate_size, config.hidden_size, act='noact', n_bits=n_bits)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertQuantizedLayer(nn.Module):
    '''
    quantization_scheme
    0: quantize both attention and (intermediate + output)
    1: quantize attention only
    2: quantize (intermediate + output) only
    3: quantize none of them
    '''

    def __init__(self, config, quantization_scheme=1, output_noise=None):
        super(BertQuantizedLayer, self).__init__()
        self.attention = BertQuantizedAttention(config, 8-(quantization_scheme // 2)*4)
        self.intermediate = BertQuantizedIntermediate(config, 8-(quantization_scheme % 2)*4)
        self.output = BertQuantizedOutput(config, 8-(quantization_scheme % 2)*4)
        self.output_noise = output_noise

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if self.output_noise is not None:
            layer_output += torch.randn_like(layer_output) * self.output_noise
        return layer_output


class BertQuantizedEncoder(Module):
    def __init__(self, config, quantization_schemes):
        super(BertQuantizedEncoder, self).__init__()
        self.layer = nn.ModuleList([BertQuantizedLayer(
            config, quantization_schemes[i]) for i in range(config.num_hidden_layers)])
        self.output_all_encoded_layers = config.output_all_encoded_layers
        self._checkpoint_activations = False

    @torch.jit.unused
    def checkpointed_forward(self, hidden_states, attention_mask):
        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
            return custom_forward

        l = 0
        num_layers = len(self.layer)
        chunk_length = math.ceil(math.sqrt(num_layers))
        while l < num_layers:
            hidden_states = checkpoint.checkpoint(custom(l, l+chunk_length), hidden_states, attention_mask*1)
            l += chunk_length

        return hidden_states

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = []

        if self._checkpoint_activations:
            hidden_states = self.checkpointed_forward(hidden_states, attention_mask)
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)

                if self.output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)

        if not self.output_all_encoded_layers or self._checkpoint_activations:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertQuantizedPooler(nn.Module):
    def __init__(self, config):
        super(BertQuantizedPooler, self).__init__()
        self.dense = QuantizedLinearActivation(config.hidden_size, config.hidden_size, act="bias_tanh")

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output
