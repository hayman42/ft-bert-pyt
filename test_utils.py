from typing import OrderedDict
import torch
import torch.nn.functional as F
from quantize import quantize
from quantized_modeling import BertQuantizedEncoder

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def adjust_state_dict(state_dict, quantization_schemes):
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if "LayerNorm" in key:
            if "weight" in key:
                new_state_dict[key.replace("weight", "gamma")] = val
            if "bias" in key:
                new_state_dict[key.replace("bias", "beta")] = val
        elif "encoder" in key and "weight" in key:
            n = int(''.join(x for x in key if x.isdigit()))
            if "attention" in key:
                n_bits = 8 if quantization_schemes[n] < 2 else 4
            else:
                n_bits = 4 if quantization_schemes[n] % 2 else 8
            q, s, z = quantize(val, n_bits)
            s = s.float().unsqueeze(0)
            z = z.float().unsqueeze(0)
            new_state_dict[key] = q
            new_state_dict[key.replace("weight", "scale")] = s
            new_state_dict[key.replace("weight", "zero_point")] = z
        else:
            new_state_dict[key] = val
    return new_state_dict


def apply_quantization(model, config, state_dict, quantization_schemes=[0]*12):
    '''
    0: 8 bit attention & 8 bit fc
    1: 8 bit attention & 4 bit fc
    2: 4 bit attention & 8 bit fc
    3: 4 bit attention & 4 bit fc
    '''
    model.bert.encoder = BertQuantizedEncoder(config, quantization_schemes)
    state_dict = adjust_state_dict(state_dict, quantization_schemes)
    model.load_state_dict(state_dict)


def predict(logits, label_ids, print_result=True):
    logit = logits.squeeze()
    label = label_ids[0].item()
    prob = F.softmax(logit, dim=0)
    pred = prob.argmax().item()

    if print_result:
        print()
        print("predicted id:", pred)
        print("prob:", prob[pred])
        print("logit:", logit[pred])
        print("label_id:", label)
        print("preb == label?:", pred == label)
        print()

    return pred == label


def process_glue_mrpc_data(data, task_name, tokenizer, device):
    key1, key2 = task_to_keys[task_name]
    texts = (data[key1], data[key2]) if key2 in data.keys() else (data[key1],)
    res = tokenizer(*texts, padding="max_length", max_length=512, truncation=True)
    input_ids, token_type_ids, attention_mask = \
        [torch.tensor(x).to(device) for x in res.values()]
    label_ids = data["label"].to(device)
    return input_ids, token_type_ids, attention_mask, label_ids


def eval_diff(orig, quant, data, pos):
    input_ids, token_type_ids, attention_mask, label_ids = data
    orig_encoders, orig_logits = orig(input_ids, token_type_ids, attention_mask)
    quant_encoders, quant_logits = quant(input_ids, token_type_ids, attention_mask)

    orig_prob = (label_ids == orig_logits.argmax(dim=1)).sum() / (label_ids.shape[0])
    quant_prob = (label_ids == quant_logits.argmax(dim=1)).sum() / (label_ids.shape[0])

    orig_enc_output = orig_encoders[pos]
    quant_enc_output = quant_encoders[pos]
    diff = orig_enc_output - quant_enc_output
    corrcoefs = list(
        torch.corrcoef(
            torch.stack((x.view(-1), y.view(-1)), 0)) for x, y in zip(orig_enc_output, quant_enc_output))
    mean_corrcoef = torch.mean(torch.stack(corrcoefs, 0), dim=0)
    # corrcoef = torch.corrcoef(
    #     torch.stack((orig_enc_output.view(-1), quant_enc_output.view(-1)), 0))
    ae = diff.abs()
    se = diff*diff
    mae = ae.mean(dim=(1, 2))
    mse = se.mean(dim=(1, 2))
    std = diff.std(dim=(1, 2))

    mean_mae = mae.mean().item()
    mean_mse = mse.mean().item()
    std_mae = mae.std().item()
    std_mse = mse.std().item()

    mean_std = std.mean().item()
    std_std = std.std().item()

    print()
    print("original prob vs quant prob: %.4f vs %.4f" % (orig_prob, quant_prob))
    print("mean of mae: %.4f" % mean_mae)
    print("mean of mse: %.4f" % mean_mse)
    print("std of mae: %.4f" % std_mae)
    print("std of mse: %.4f" % std_mse)
    print("mean of std: %.4f" % mean_std)
    print("std of std: %.4f" % std_std)
    print("mean of corrcoef:")
    print(mean_corrcoef)
    print()


def make_table(table, pos, scheme, noise_scale, orig_encoders, orig_logits, quant, data):
    quant_encoders, quant_logits = quant(data[0], data[1], data[2])
    orig_enc_output = orig_encoders[pos]
    quant_enc_output = quant_encoders[pos]

    orig_prob = (data[3] == orig_logits.argmax(dim=1)).sum() / (data[3].shape[0])
    quant_prob = (data[3] == quant_logits.argmax(dim=1)).sum() / (data[3].shape[0])

    diff = orig_enc_output - quant_enc_output
    corrcoefs = list(
        torch.corrcoef(
            torch.stack((x.view(-1), y.view(-1)), 0)) for x, y in zip(orig_enc_output, quant_enc_output))
    mean_corrcoef = torch.mean(torch.stack(corrcoefs, 0), dim=0)
    ae = diff.abs()
    se = diff*diff
    mae = ae.mean(dim=(1, 2))
    mse = se.mean(dim=(1, 2))
    std = diff.std(dim=(1, 2))

    mean_mae = mae.mean().item()
    mean_mse = mse.mean().item()
    std_mae = mae.std().item()
    std_mse = mse.std().item()

    mean_std = std.mean().item()
    std_std = std.std().item()

    table[(pos, scheme, noise_scale)] = [quant_prob, mean_corrcoef[0][1].item(), mean_mae, mean_mse, std_mae, std_mse, mean_std, std_std]
    return
