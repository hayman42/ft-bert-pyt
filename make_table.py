from typing import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from modeling import BertForSequenceClassification, BertConfig
from transformers import AutoTokenizer
import datasets
from tqdm import tqdm
from time import time
from quantize import quantize
from quantized_modeling import BertQuantizedEncoder
import random


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
    new_state_dict = adjust_state_dict(state_dict, quantization_schemes)
    model.load_state_dict(new_state_dict)


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


def process_glue_mrpc_data(data, tokenizer, device):
    res = tokenizer(*(data["sentence1"], data["sentence2"]),
                    padding="max_length", max_length=512, truncation=True)
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

    orig_enc_output = orig_encoders[pos][:, 0]
    quant_enc_output = quant_encoders[pos][:, 0]
    diff = orig_enc_output - quant_enc_output
    corrcoefs = list(torch.corrcoef(torch.stack((x, y), 0)) for x, y in zip(orig_enc_output, quant_enc_output))
    mean_corrcoef = torch.mean(torch.stack(corrcoefs, 0), dim=0)
    # corrcoef = torch.corrcoef(
    #     torch.stack((orig_enc_output.view(-1), quant_enc_output.view(-1)), 0))
    ae = diff.abs()
    se = diff*diff
    mae = ae.mean(dim=1)
    mse = se.mean(dim=1)
    std = diff.std(dim=1)

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


def main():
    config = BertConfig.from_json_file(config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, config=config)
    rawdata = datasets.load_dataset("glue", "mrpc")["validation"]
    loader = DataLoader(rawdata, batch_size=n_samples, shuffle=True)

    if no_cuda:
        device = torch.device("cpu")
    elif local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda:0", local_rank)

    # set dropout prob to 0
    # config.hidden_dropout_prob = 0
    # config.attention_probs_dropout_prob = 0
    # get each encoder output
    config.output_all_encoded_layers = True
    state_dict = torch.load(init_checkpoint)

    orig = BertForSequenceClassification(config, 2)
    apply_quantization(orig, config, state_dict)
    orig.to(device)

    quant = BertForSequenceClassification(config, 2)
    apply_quantization(quant, config, state_dict, quantization_schemes)
    quant.to(device)
    print(quantization_schemes)

    orig.eval()
    quant.eval()

    if fp16:
        orig.half()
        quant.half()

    with torch.no_grad():
        a = time()
        for data in loader:
            eval_diff(orig, quant, process_glue_mrpc_data(data, tokenizer, device), pos)
            break
        print("total time:", time() - a)


model_dir = "/workspace/ft-bert-pyt/model/bert-base-cased-mrpc/"
config_file = model_dir + "config.json"
init_checkpoint = model_dir + "pytorch_model.bin"
vocab_file = model_dir + "vocab.txt"
data_dir = "/workspace/ft-bert-pyt/data/glue/MRPC"
tokenizer_config_path = model_dir + "tokenizer_config.json"
tokenizer_path = model_dir
local_rank = -1
n_samples = 100
do_lower_case = False
no_cuda = False
fp16 = False
quantization_schemes = [random.randint(0, 3) for i in range(12)]
# quantization_schemes = [3] * 12
# quantization_schemes = [0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
pos = 11

if __name__ == "__main__":
    main()
