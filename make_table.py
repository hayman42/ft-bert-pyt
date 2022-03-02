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
    # for i, j in torch.load(table_dir + task_name + "_table.bin").items():
    #     print(i, [round(x, 4) for x in j])
    # return
    config = BertConfig.from_json_file(config_file)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, config=config)
    rawdata = datasets.load_dataset("glue", task_name)["train"]
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

    with torch.no_grad():
        for data in loader:
            if task_name == "mrpc" or task_name == "qnli":
                orig = BertForSequenceClassification(config, 2)
            elif task_name == "mnli":
                orig = BertForSequenceClassification(config, 3)
            apply_quantization(orig, config, state_dict)
            orig.to(device)
            orig.eval()
            processed_data = process_glue_mrpc_data(data, task_name, tokenizer, device)
            orig_encoders, orig_logits = orig(processed_data[0], processed_data[1], processed_data[2])
            table = OrderedDict()

            for i in tqdm(range(12)):
                schemes = [0] * 12
                for j in tqdm(range(0, 4), leave=False):
                    schemes[i] = j
                    if task_name == "mrpc" or task_name == "qnli":
                        quant = BertForSequenceClassification(config, 2)
                    elif task_name == "mnli":
                        quant = BertForSequenceClassification(config, 3)
                    apply_quantization(quant, config, state_dict, schemes)
                    quant.to(device)
                    quant.eval()
                    for k in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
                        if i:
                            if k:
                                quant.bert.encoder.layer[i - 1].output_noise = k
                            else:
                                quant.bert.encoder.layer[i - 1].output_noise = None
                        make_table(table, i, j, k, orig_encoders, orig_logits, quant, processed_data)
                        if i == 0:
                            break
                    del quant
            torch.save(table, table_dir + task_name + "_table.bin")
            print("Saved at " + table_dir + task_name + "_table.bin")
            break


task_name = "mnli"
model_dir = f"/workspace/ft-bert-pyt/model/bert-base-cased-{task_name}/"
table_dir = "/workspace/ft-bert-pyt/table/"
config_file = model_dir + "config.json"
init_checkpoint = model_dir + "pytorch_model.bin"
vocab_file = model_dir + "vocab.txt"
tokenizer_config_path = model_dir + "tokenizer_config.json"
tokenizer_path = model_dir
local_rank = 2
n_samples = 128
do_lower_case = False
no_cuda = False
fp16 = False

if __name__ == "__main__":
    main()
