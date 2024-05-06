import random
import sys
import gc
import statistics

import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cortexsubsetloader import CortexSubsetLoader
from utils import *

# model i is always pre-finetune, and thus gets the buffer
# model j is post our finetune
def iswin(loss_i, loss_j, epsilon=0.01):
    # Adjust loss based on timestamp and pretrain epsilon
    loss_i = (1 - epsilon) * loss_i
    return loss_i > loss_j

def compute_losses(model, batches, device):
    # Iterate over each page and corresponding batches
    losses = []
    print()
    with torch.no_grad():
        model.to(device)
        model.eval()
        steps = 0
        for inputs, prompt_len in batches:
            inputs = inputs.to(device)
            labels = inputs.clone()
            labels[:, :prompt_len] = -100 # Only calculate loss on response
            outputs = model(inputs, labels=labels)
            loss = outputs.loss.item()  # Extract scalar loss value
            losses.append(loss)
            steps += 1
            if steps % (len(batches) // 8) == 0:
                print(".", end="") # rudimentary progress bar
    return losses

params = load_local_config()

def validate_improvement(old_model_name, new_model_name, samples=768, tokenizer_name="NousResearch/Meta-Llama-3-8B-Instruct", 
                         device="cuda", dedup=False):
    # allow for passing models in directly, or by name and loading them here
    if type(old_model_name) == str:
        print("Testing", old_model_name, end=" ")
    else:
        print("Testing", old_model_name.config.name_or_path, end=" ")
    if type(new_model_name) == str:
        print("against", new_model_name)
    else:
        print("against", new_model_name.config.name_or_path)

    # load tokenizer
    tokenizer = get_tokenizer(tokenizer_name)

    # get new data for each run
    cortex_data = CortexSubsetLoader(
            latest=True, running=True, random_seed=random.randint(0, sys.maxsize),
            max_samples=samples, page_size=samples, steps=1, dedup=dedup
        )
    # tokenize data
    batches = cortex_data.tokenize(tokenizer)

    # load old model
    if type(old_model_name) is str:
        old_model = AutoModelForCausalLM.from_pretrained(old_model_name, **params)
    else:
        old_model = old_model_name
    
    # compute losses of the old model
    base_loss = compute_losses(old_model, batches, device)

    # delete old model  and clear memory
    old_model = None; gc.collect(); torch.cuda.empty_cache()

    # load new model
    if type(new_model_name) is str:
        new_model = AutoModelForCausalLM.from_pretrained(new_model_name, **params)
    else:
        new_model = new_model_name

    # compute losses of the new model
    new_loss = compute_losses(new_model, batches, device)

    # delete new model and clear memory
    new_model = None; gc.collect(); torch.cuda.empty_cache()

    # compute win rate and other statistics
    per_loss_win_avg = np.mean([iswin(base, new) for base, new in zip(base_loss, new_loss)])
    per_loss_win_avg_0eps = np.mean([iswin(base, new, epsilon=0.0) for base, new in zip(base_loss, new_loss)])

    avg_loss_i = np.mean(base_loss)
    avg_loss_j = np.mean(new_loss) 
    loss_diff = np.mean(np.array(new_loss) - np.array(base_loss))
    avg_loss_diff = loss_diff

    print()
    print("Avg Loss Base: ", avg_loss_i)
    print("Avg Loss New: ", avg_loss_j)
    print("Avg Loss Diff: ", avg_loss_diff)
    print("Final Win Rate: ", per_loss_win_avg)
    print("Final 0Eps Win Rate: ", per_loss_win_avg_0eps)

# helper function to print out the model parameters, optionally normalizing them first
# helpful for looking for extreme values and other weight sanity checks
def print_model_params(model, norm=False):
    model = AutoModelForCausalLM.from_pretrained(model, **params)
    if norm:
        model = norm_model_weights(model)
    for name, param in model.named_parameters():
        print(name, param)
    del model; gc.collect(); torch.cuda.empty_cache()

# helper function to quickly check if two models are the same and/or how close they are norm-wise
def check_matching_weights(model0, model1):
    model0 = AutoModelForCausalLM.from_pretrained(model0, **params)
    model1 = AutoModelForCausalLM.from_pretrained(model1, **params)
    mismatch_diffs = []
    any_mismatch = False
    for (name0, param0), (name1, param1) in zip(model0.named_parameters(), model1.named_parameters()):
        if not (param0.data == param1.data).all():
            any_mismatch = True
            diff = torch.sum(torch.abs(param0.data - param1.data)).item()
            mismatch_diffs.append(diff)
            print("Mismatched weights", name0, name1, diff)
    if not any_mismatch:
        print("No mismatched weights")
    else:
        print("Mean abs mismatched", np.mean(mismatch_diffs), "Std abs mismatched", np.std(mismatch_diffs))
    del model0, model1; gc.collect(); torch.cuda.empty_cache()


def validate_parameters(base_model, eps_soft=200, eps_soft_percent_threshold=0.15, eps_hard=1000, print_vals=False) -> bool:
    """
    Validate that parameters of a model

    Parameters:
        base_model (transformers.PreTrainedModel): The base model instance.
        num_layers (int): Number of layers in the model to inspect.
        eps_soft (float): Calculate the percentage of layers above this norm
        eps_soft_percent_threshold (float): Threshold of percentage above eps_soft that will trigger a detection
        eps_hard (float): Hard limit for any norm
    """

    exceed_counts = {'q_proj': 0, 'k_proj': 0, 'v_proj': 0, 'o_proj': 0, 'up_proj': 0, 'down_proj': 0}
    total_counts = {'q_proj': 0, 'k_proj': 0, 'v_proj': 0, 'o_proj': 0, 'up_proj': 0, 'down_proj': 0}
    if print_vals:
        avg_norms = {'q_proj': 0.0, 'k_proj': 0.0, 'v_proj': 0.0, 'o_proj': 0.0, 'up_proj': 0.0, 'down_proj': 0.0}
        max_norms = {'q_proj': 0.0, 'k_proj': 0.0, 'v_proj': 0.0, 'o_proj': 0.0, 'up_proj': 0.0, 'down_proj': 0.0}

    for layer in base_model.model.layers:
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            weight_norm = getattr(layer.self_attn, proj).weight.norm().item()
            if weight_norm > eps_hard:
                return False
            elif weight_norm > eps_soft:
                exceed_counts[proj] += 1
            total_counts[proj] += 1
            if print_vals:
                avg_norms[proj] += weight_norm
                max_norms[proj] = max(max_norms[proj], weight_norm)

        # up_proj and down_proj are in the mlp layer
        for proj in ['up_proj', 'down_proj']:
            weight_norm = getattr(layer.mlp, proj).weight.norm().item()
            if weight_norm > eps_hard:
                return False
            elif weight_norm > eps_soft:
                exceed_counts[proj] += 1
            total_counts[proj] += 1
            if print_vals:
                avg_norms[proj] += weight_norm
                max_norms[proj] = max(max_norms[proj], weight_norm)

    # Calculating and printing percentages
    percentages = [exceed_counts[proj] / total_counts[proj] for proj in exceed_counts]

    if print_vals:
        for key, value in total_counts.items():
            avg_norms[key] = avg_norms[key] / value
        print(avg_norms)
        print(max_norms)
        print(percentages)

    return statistics.fmean(percentages) <= eps_soft_percent_threshold