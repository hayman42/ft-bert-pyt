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


def process_glue_mrpc_data(data, tokenizer, device):
    res = tokenizer(*(data["sentence1"], data["sentence2"]),
                    padding="max_length", max_length=512, truncation=True)
    input_ids, token_type_ids, attention_mask = \
        [torch.tensor(x).to(device) for x in res.values()]
    label_ids = data["label"]
    return input_ids, token_type_ids, attention_mask, label_ids


def main():
    config = BertConfig.from_json_file(config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, config=config)
    rawdata = datasets.load_dataset("glue", "mrpc")["validation"]
    loader = DataLoader(rawdata, batch_size=1, shuffle=True)

    if no_cuda:
        device = torch.device("cpu")
    elif local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda:0", local_rank)

    # set dropout prob to 0
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    config.output_all_encoded_layers = True
    model = BertForSequenceClassification(config, 2)
    apply_quantization(model, config, torch.load(init_checkpoint), quantization_schemes)
    model.to(device)
    print(quantization_schemes)

    model.bert.output_all_encoded_layers = True
    if fp16:
        model.half()

    with torch.no_grad():
        cnt = 0
        total = 0
        a = time()
        for n, data in tqdm(enumerate(loader)):
            total += 1
            if n == n_samples:
                break
            input_ids, token_type_ids, attention_mask, label_ids = process_glue_mrpc_data(data, tokenizer, device)
            _, logits = model(input_ids, token_type_ids, attention_mask)
            if predict(logits, label_ids, False):
                cnt += 1
        print("total time:", time() - a)
        print("accuracy:", cnt / total)


model_dir = "/workspace/ft-bert-pyt/model/bert-base-cased-mrpc/"
config_file = model_dir + "config.json"
init_checkpoint = model_dir + "pytorch_model.bin"
vocab_file = model_dir + "vocab.txt"
data_dir = "/workspace/ft-bert-pyt/data/glue/MRPC"
tokenizer_config_path = model_dir + "tokenizer_config.json"
tokenizer_path = model_dir
local_rank = -1
n_samples = 1000
do_lower_case = False
no_cuda = False
fp16 = False
# quantization_schemes = [random.randint(0, 3) for i in range(12)]
quantization_schemes = [3]*12

if __name__ == "__main__":
    main()
