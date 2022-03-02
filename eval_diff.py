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
    rawdata = datasets.load_dataset("glue", "mrpc")["train"]
    loader = DataLoader(rawdata, batch_size=n_samples, shuffle=True)

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
    # get each encoder output
    config.output_all_encoded_layers = True
    state_dict = torch.load(init_checkpoint)

    if task_name == "mrpc" or task_name == "qnli":
        orig = BertForSequenceClassification(config, 2)
        quant = BertForSequenceClassification(config, 2)
    elif task_name == "mnli":
        orig = BertForSequenceClassification(config, 3)
        quant = BertForSequenceClassification(config, 3)
    apply_quantization(orig, config, state_dict)
    apply_quantization(quant, config, state_dict, quantization_schemes)
    orig.to(device)
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
            processed_data = process_glue_mrpc_data(data, task_name, tokenizer, device)
            for i in range(1):
                eval_diff(orig, quant, processed_data, i)
            break
        print("total time:", time() - a)


task_name = "mrpc"
model_dir = f"/workspace/ft-bert-pyt/model/bert-base-cased-{task_name}/"
config_file = model_dir + "config.json"
init_checkpoint = model_dir + "pytorch_model.bin"
vocab_file = model_dir + "vocab.txt"
tokenizer_config_path = model_dir + "tokenizer_config.json"
tokenizer_path = model_dir
local_rank = 0
n_samples = 100
do_lower_case = False
no_cuda = False
fp16 = False
quantization_schemes = [random.randint(0, 3) for i in range(12)]
quantization_schemes = [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# quantization_schemes = [0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0]
pos = 11

if __name__ == "__main__":
    main()
