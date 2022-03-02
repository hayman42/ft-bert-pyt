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
from test_utils import *


def main():
    config = BertConfig.from_json_file(config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, config=config)
    rawdata = datasets.load_dataset("glue", task_name)["validation"]
    loader = DataLoader(rawdata, batch_size=1, shuffle=True)

    if no_cuda:
        device = torch.device("cpu")
    elif local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)

    # set dropout prob to 0
    # config.hidden_dropout_prob = 0
    # config.attention_probs_dropout_prob = 0
    config.output_all_encoded_layers = True
    if task_name == "mrpc" or task_name == "qnli":
        model = BertForSequenceClassification(config, 2)
    elif task_name == "mnli":
        model = BertForSequenceClassification(config, 3)
    apply_quantization(model, config, torch.load(init_checkpoint), quantization_schemes)
    model.to(device)
    model.eval()
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
            input_ids, token_type_ids, attention_mask, label_ids = process_glue_mrpc_data(data, task_name, tokenizer, device)
            _, logits = model(input_ids, token_type_ids, attention_mask)
            if predict(logits, label_ids, False):
                cnt += 1
        print("total time:", time() - a)
        print("accuracy:", cnt / total)


task_name = "qnli"
model_dir = f"/workspace/ft-bert-pyt/model/bert-base-cased-{task_name}/"
config_file = model_dir + "config.json"
init_checkpoint = model_dir + "pytorch_model.bin"
vocab_file = model_dir + "vocab.txt"
tokenizer_config_path = model_dir + "tokenizer_config.json"
tokenizer_path = model_dir
local_rank = 2
do_lower_case = False
no_cuda = False
fp16 = False
# quantization_schemes = [random.randint(0, 3) for i in range(12)]
quantization_schemes = [3] * 12
quantization_schemes = [0] * 12

if __name__ == "__main__":
    main()
