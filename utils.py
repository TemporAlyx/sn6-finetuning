import tempfile
import os
import shutil
import hashlib
import base64

import torch

from transformers import LlamaForCausalLM
from transformers import AutoTokenizer

def norm_model_weights(model, model_type='llama'):
    last_q = None
    lqb = None
    lqkm = None
    last_v = None
    lvb = None
    lvom = None
    last_up = None
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
                if model_type == 'llama':
                    # print(last_q.data.shape, param.data.shape)
                    mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                                    torch.mean(torch.abs(param.data), dim=1, keepdim=True).repeat(
                                                                        int(last_q.data.shape[0] / param.data.shape[0]), 1))
                    # print(mult.shape, mult)
                    mult = torch.mean(mult)
                    # mult = np.sqrt(2.0)
                    last_q.data = last_q.data / mult#.transpose(0, 1)
                    # if bias: lqb.data = lqb.data / mult#.transpose(0, 1).flatten()
                    param.data = param.data * mult#[:param.data.shape[0]] # 
                    lqkm = mult
                    
                else:
                    # print(last_q.data.shape, param.data.shape)
                    # print(name)
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                    #                 torch.mean(torch.abs(param.data), dim=1, keepdim=True)) # Loss: 0.14939117
                    mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True) / 
                                    torch.mean(torch.abs(param.data), dim=0, keepdim=True).transpose(0, 1)).transpose(0, 1) # Loss: 0.04619789
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=1, keepdim=True).transpose(0, 1) / 
                    #                 torch.mean(torch.abs(param.data), dim=0, keepdim=True)) # Loss: 0.04619789
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=0, keepdim=True).transpose(0, 1) / 
                    #                 torch.mean(torch.abs(param.data), dim=1, keepdim=True)) # Loss: 0.07823181
                    # mult = torch.sqrt(torch.mean(torch.abs(last_q.data), dim=0, keepdim=True) / 
                    #                 torch.mean(torch.abs(param.data), dim=0, keepdim=True)).transpose(0, 1) # Loss: 0.01167393
                    # mult = mult / 2
                    # print(mult.shape, mult)
                    mult = torch.mean(mult)
                    last_q.data = last_q.data / mult#.transpose(0, 1)
                    if bias: lqb.data = lqb.data / mult.flatten() # .transpose(0, 1)
                    param.data = param.data * mult # 
                    lqkm = mult
                    # print(last_q.data.norm(), param.data.norm())
        
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
                # print(mult.shape, mult)
                if model_type == 'llama':
                    mult = torch.sqrt(torch.mean(torch.abs(last_v.data), dim=0, keepdim=True) / 
                            torch.mean(torch.abs(param.data), dim=0, keepdim=True))
                    mult = torch.mean(mult)
                    # mult = np.sqrt(1.0 / 2.0)
                    last_v.data = last_v.data / mult
                    if bias: lvb.data = lvb.data / mult
                    param.data = param.data * mult
                    lvom = mult
                else:
                    mult = torch.sqrt(torch.mean(torch.abs(last_v.data), dim=1, keepdim=True).transpose(0, 1).repeat(1, 
                                                                                int(param.data.shape[0] / last_v.data.shape[0])) / 
                                    torch.mean(torch.abs(param.data), dim=0, keepdim=True))
                    last_v.data = last_v.data / mult.transpose(0, 1)[:last_v.data.shape[0]]
                    if bias: lvb.data = lvb.data / mult.transpose(0, 1).flatten()
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

        # if "model.norm.weight" == name:
        #     param.data /= 200
        # if "model.norm.bias" == name:
        #     param.data /= 200
        # if "lm_head.weight" == name:
        #     param.data *= 200
        
    return model


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

    model = LlamaForCausalLM.from_pretrained(
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