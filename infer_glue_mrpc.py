from typing import OrderedDict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import modeling
from transformers import AutoTokenizer, AutoConfig
import datasets
from tqdm import tqdm
from time import time


def adjust_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if "LayerNorm" in key:
            if "weight" in key:
                new_state_dict[key.replace("weight", "gamma")] = val
            if "bias" in key:
                new_state_dict[key.replace("bias", "beta")] = val
        else:
            new_state_dict[key] = val
    return new_state_dict


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
    state_dict = adjust_state_dict(torch.load(init_checkpoint))
    config = modeling.BertConfig.from_json_file(config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, config=config)
    rawdata = datasets.load_dataset("glue", "mrpc")["validation"]
    loader = DataLoader(rawdata, batch_size=1, shuffle=True)

    modeling.APEX_IS_AVAILABLE = True
    modeling.QUANT = False
    if no_cuda:
        device = torch.device("cpu")
    elif local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    model = modeling.BertForSequenceClassification(config, 2)
    model.load_state_dict(state_dict)
    model.to(device)
    # model.bert.output_all_encoded_layers = True
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
            logits = model(input_ids.to(device), token_type_ids.to(device), attention_mask.to(device))
            if predict(logits, label_ids, False):
                cnt += 1
        print("total time:", time() - a)
        print("accuracy:", cnt / total)


model_dir = "/workspace/FasterTransformer/bert-quantization/bert-pyt-quantization/model/bert-base-cased-mrpc/"
config_file = model_dir + "config.json"
init_checkpoint = model_dir + "pytorch_model.bin"
vocab_file = model_dir + "vocab.txt"
data_dir = "/workspace/FasterTransformer/bert-quantization/bert-pyt-quantization/data/glue/MRPC"
tokenizer_config_path = model_dir + "tokenizer_config.json"
tokenizer_path = model_dir
local_rank = -1
n_samples = 1000
do_lower_case = False
no_cuda = False
fp16 = True

if __name__ == "__main__":
    main()
