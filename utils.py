import tempfile
import os
import shutil
import hashlib
import base64
import json

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_tokenizer(tokenizer_name):
    if tokenizer_name == "l3" or tokenizer_name == "llama3":
        tokenizer_name = "NousResearch/Meta-Llama-3-8B-Instruct"
    elif tokenizer_name == "mistral" or tokenizer_name == "m1":
        tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.1"
    elif tokenizer_name == "stablelm" or tokenizer_name == "s1":
        tokenizer_name = "stabilityai/stablelm-2-zephyr-1_6b"
    elif tokenizer_name == "gemma" or tokenizer_name == "g1":
        tokenizer_name = "google/gemma-2b-it"
    return AutoTokenizer.from_pretrained(tokenizer_name)

def norm_model_weights(model):
    # the primary method it seems people have found for corrupting model weights has to do with using two adjacent weight matrices, and rescaling them.
    # for example, if the up_proj weights are multiplied by some set of values, and the down_proj weights are divided by the same set of values, 
    # the model will still function, but attempts to train on it will have a high likelihood of exploding gradients.
    # for other pairs like q and k, or v and o, its slightly more complicated, as the weights aren't directly adjacent, but the same principle applies.
    # for a sense of how these matrcies are usually put together in a model, here is a good refernce
    # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    # for non-directly adjacent pairs, it can be much more difficult to perform a full solve (possibly impossible), depending on how the model is structured.

    # NOTE: does not work with all models, different sizes and configurations may cause issues, not all weight pairs have a full solve
    bias = False
    for name, param in model.named_parameters():
        if "q_proj" in name:
            if "bias" in name:
                bias = True
                lqb = param
            else:
                last_q = param
        if "k_proj" in name:
            if "bias" in name:
                if lqkm is not None:
                    param.data = param.data * lqkm.flatten()
                    pass
            else:
                # print(last_q.data.shape, param.data.shape)
                mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                                torch.mean(torch.abs(param.data), dim=1, keepdim=True).repeat(
                                                                    int(last_q.data.shape[0] / param.data.shape[0]), 1))
                # print(mult.shape, mult)
                mult = torch.mean(mult)
                last_q.data = last_q.data / mult
                if bias: lqb.data = lqb.data / mult
                param.data = param.data * mult
                lqkm = mult
        
        if "v_proj" in name:
            if "bias" in name:
                lvb = param
            else:
                last_v = param
        if "o_proj" in name:
            if "bias" in name:
                param.data = param.data * lvom
            else:
                # print(last_v.data.shape, param.data.shape)
                mult = torch.sqrt(torch.mean(torch.abs(last_v.data), dim=0, keepdim=True) / 
                        torch.mean(torch.abs(param.data), dim=0, keepdim=True))
                # print(mult.shape, mult)
                mult = torch.mean(mult)
                last_v.data = last_v.data / mult
                if bias: lvb.data = lvb.data / mult
                param.data = param.data * mult
                lvom = mult

        if "up_proj" in name:
            last_up = param
        if "down_proj" in name:
            # print(last_up.data.shape, param.data.shape)
            mult = torch.sqrt(torch.mean(torch.abs(last_up.data), dim=1, keepdim=True).transpose(0, 1) / 
                            torch.mean(torch.abs(param.data), dim=0, keepdim=True))
            last_up.data = last_up.data / mult.transpose(0, 1)
            param.data = param.data * mult
            # print(mult, mult.shape)
                    
    return model

# simple linear interpolation of weights between two models
def merge(model0, model1, ratio=0.5, embed_ratio=None, norm_ratio=None, fc_ratio=None): # higher ratio means more of model0
    if embed_ratio is None: embed_ratio = ratio # not sure if using different ratios for different parts of the model actually makes any sense
    if norm_ratio is None: norm_ratio = ratio
    if fc_ratio is None: fc_ratio = ratio

    params0 = {}
    for name, param in model0.named_parameters():
        params0[name] = param

    for name, param in model1.named_parameters():
        if "embed" in name:
            param.data = ((params0[name].data * embed_ratio) + (param.data * (1 - embed_ratio)))
        elif ("up_proj" not in name 
            and "down_proj" not in name 
            and "gate_proj" not in name 
            and "o_proj" not in name 
            and "k_proj" not in name 
            and "v_proj" not in name 
            and "q_proj" not in name
            and "embed" not in name
            ):
            param.data = ((params0[name].data * norm_ratio) + (param.data * (1 - norm_ratio)))
        elif "up_proj" in name or "down_proj" in name:
            param.data = ((params0[name].data * fc_ratio) + (param.data * (1 - fc_ratio)))
        else:
            param.data = ((params0[name].data * ratio) + (param.data * (1 - ratio)))

    return model1

# helper function to copy weights over from one model to another, which doesn't (normally) break the optimizer state.
def copy_weights_over(model0, model1):
    """Copies the weights from model0 to model1, returns model1 with the copied weights"""
    params0 = {}
    for name, param in model0.named_parameters():
        params0[name] = param

    for name, param in model1.named_parameters():
        if name in params0:
            param.data = params0[name].data
    return model1

# loads a config for general model loading parameters, and if it doesn't exist, creates a new one with default values
def load_local_config(config_path='model_loading_config.json', cache_dir='Models'):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.isfile(os.path.join(cache_dir, config_path)):
        new_config = {
                'low_cpu_mem_usage': True,
                'trust_remote_code': False,
                'torch_dtype': "bfloat16",
                'use_safetensors': True,
                'attn_implementation': "flash_attention_2",
                'cache_dir': cache_dir
            }
        save_local_config(new_config, config_path)
    with open(os.path.join(cache_dir, config_path), 'r') as f:
        config = json.load(f)
        config['torch_dtype'] = torch.bfloat16 if config['torch_dtype'] == "bfloat16" else torch.float32
    return config

def save_local_config(config, config_name='model_loading_config.json'):
    cache_dir = config['cache_dir']
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(os.path.join(cache_dir, config_name), 'w') as f:
        json.dump(config, f)


# below are some functions for separating calculating model hash from downloading the model which may be useful for some use cases
# specifically, if models are being trained in the cloud, but a local computer is used to connect to the subnet, we can calculate the commit.oid and hash
# in the cloud to avoid downloading the model to the local computer, which may be slow or have limited storage/compute.

# MIT License

# Copyright (c) 2023 Opentensor

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# https://github.com/NousResearch/finetuning-subnet/blob/master/model/storage/disk/utils.py
def get_hf_download_path(local_path: str, account_name, model_name, commit) -> str:
    return os.path.join(
        local_path,
        "models" + "--" + account_name + "--" + model_name,
        "snapshots",
        commit,
    )

def realize_symlinks_in_directory(path: str) -> int:
    """Realizes all symlinks in the given directory, moving the linked file to the location. Returns count removed."""
    realized_symlinks = 0

    for cur_path, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.abspath(os.path.join(cur_path, filename))
            # Get path resolving symlinks if encountered
            real_path = os.path.realpath(path)
            # If different then move
            if path != real_path:
                realized_symlinks += 1
                shutil.move(real_path, path)

    return realized_symlinks

def get_hash_of_file(path: str) -> str:
    blocksize = 64 * 1024
    file_hash = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            data = fp.read(blocksize)
            if not data:
                break
            file_hash.update(data)
    return base64.b64encode(file_hash.digest()).decode("utf-8")


def get_hash_of_directory(path: str) -> str:
    dir_hash = hashlib.sha256()

    # Recursively walk everything under the directory for files.
    for cur_path, dirnames, filenames in os.walk(path):
        # Ensure we walk future directories in a consistent order.
        dirnames.sort()
        # Ensure we walk files in a consistent order.
        for filename in sorted(filenames):
            path = os.path.join(cur_path, filename)
            file_hash = get_hash_of_file(path)
            dir_hash.update(file_hash.encode())

    return base64.b64encode(dir_hash.digest()).decode("utf-8")

# https://github.com/NousResearch/finetuning-subnet/blob/master/model/storage/hugging_face/hugging_face_model_store.py
async def download_model(account_name, model_name, commit, local_path: str):
    """Retrieves a trained model from Hugging Face."""
    if not commit:
        raise ValueError("No Hugging Face commit id found to read from the hub.")

    repo_id = account_name + "/" + model_name

    # Transformers library can pick up a model based on the hugging face path (username/model) + rev.

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        revision=commit,
        cache_dir=local_path,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=repo_id,
        revision=commit,
        cache_dir=local_path,
    )

    # Get the directory the model was stored to.
    model_dir = get_hf_download_path(local_path, account_name, model_name, commit)

    # Realize all symlinks in that directory since Transformers library does not support avoiding symlinks.
    realize_symlinks_in_directory(model_dir)

    # Compute the hash of the downloaded model.
    model_hash = get_hash_of_directory(model_dir)
    
    return model_hash

async def get_model_hash(account_name, model_name, commit):
    with tempfile.TemporaryDirectory() as temp_dir:
        model_hash = await download_model(account_name, model_name, commit, temp_dir)
    return model_hash
