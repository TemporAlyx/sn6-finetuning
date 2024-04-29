import gc

import numpy as np
import torch

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from cortexsubsetloader import CortexSubsetLoader
from pytorch_optimizer import Ranger21, DAdaptAdam, ScalableShampoo

from utils import *


def data_collator(features):
    batches = []
    for feature in features:
        inputs, prompt_len = feature
        data = [inputs]
        b_labels = inputs.clone()
        b_labels[:, :prompt_len] = -100
        labels = [b_labels]
            
        batch = {}
        batch['input_ids'] = torch.concat(data)
        batch['labels'] = torch.concat(labels)
        batches.append(batch)
    return batches


def simple_eval(model, eval_d):
    print("Evaluating", end=" ")
    model = model.to("cuda")
    model.eval()
    eval_loss = 0
    steps_so_far = 1
    for batch in eval_d:
        inputs = batch['input_ids'].to("cuda")
        labels = batch['labels'].to("cuda")
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            eval_loss += outputs.loss.item() / len(eval_d)
        if steps_so_far % (len(eval_d) // 8) == 0:
            print(".", end="")
            gc.collect(); torch.cuda.empty_cache()
        steps_so_far += 1
    model = model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()
    print(f" Loss: {eval_loss:.8f}")

def evaluate(model, eval_d, base_model=None, cached_base_loss=None, return_to_cpu=False, 
             return_stats=False, print_stats=True,  device="cuda", loss_eps=0.01):
    print("Evaluating", end=" ")

    # if we have cached base loss values (from a previous eval), use them, otherwise compute them
    precomputed_base_losses = []
    if cached_base_loss is not None:
        for x in cached_base_loss:
            precomputed_base_losses.append(x)
    else:
        model = model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
        base_model = base_model.to(device)

        for batch in eval_d:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with torch.no_grad():
                base_outputs_loss = base_model(inputs, labels=labels).loss
                precomputed_base_losses.append(base_outputs_loss)
        
        base_model = base_model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()
        model = model.to(device)

    model = model.to("cuda")
    model.eval()

    eval_base_loss = 0
    diff = 0
    eval_loss = 0
    head_to_head = 0
    eps0_head_to_head = 0
    overshoot = 0
    steps_so_far = 1
    for batch in eval_d:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            base_outputs_loss = precomputed_base_losses.pop(0)
            outputs_loss = model(inputs, labels=labels).loss

            base_loss = base_outputs_loss
            partial_loss = torch.nn.functional.relu(outputs_loss - (base_loss * (1.0 - loss_eps)))
            overshoot_penalty = torch.nn.functional.relu(-(outputs_loss - (base_loss * (1.0 - loss_eps))))
            loss = partial_loss / base_loss # loss is reported as a percentage relative to epsilon

            eval_loss += loss.item() / len(eval_d)
            eval_base_loss += base_outputs_loss / len(eval_d)
            diff += (outputs_loss.item() - base_outputs_loss) / len(eval_d)
            head_to_head += 100.0 / len(eval_d) if outputs_loss < (base_outputs_loss * (1.0 - loss_eps)) else 0.0
            head_to_head += 50.0 / len(eval_d) if outputs_loss == (base_outputs_loss * (1.0 - loss_eps)) else 0.0
            eps0_head_to_head += 100.0 / len(eval_d) if outputs_loss < base_outputs_loss else 0.0
            eps0_head_to_head += 50.0 / len(eval_d) if outputs_loss == base_outputs_loss else 0.0
            overshoot += overshoot_penalty.item() / len(eval_d)

        if steps_so_far % (len(eval_d) // 8) == 0:
            print(".", end="") # rudimentary progress bar
            gc.collect(); torch.cuda.empty_cache()
        steps_so_far += 1

    if return_to_cpu:
        model = model.to("cpu")

    gc.collect(); torch.cuda.empty_cache()

    if print_stats:
        print(f" Loss: {eval_loss:.8f}, Base Loss: {eval_base_loss:.6f}, Diff: {diff:.8f},",
            f"WR: {head_to_head:.2f}%, 0epsWR: {eps0_head_to_head:.2f}%, OShL: {overshoot:.8f}")
    if return_stats:
        data = {
            "loss": eval_loss,
            "base_loss": eval_base_loss,
            "diff": diff,
            "head_to_head": head_to_head,
            "eps0_head_to_head": eps0_head_to_head,
            "overshoot": overshoot
        }
        return data
    
old_train_data = [] # might be worth writing some code to save and load previously seen data instead
def get_new_data(tokenizer, n_samples=2560, dedup=True, steps=1, old_data=old_train_data):
    cortex_subset_loader = CortexSubsetLoader(latest=True, random_seed = None, max_samples=n_samples, progress=False, 
                                    running=True, retry_limit=5, page_size=400, retry_delay=5, silent=True, steps=steps,
                                    ignore_list=old_data, dedup=dedup)
    batches = data_collator(cortex_subset_loader.tokenize(tokenizer))
    return [batches[i] for i in np.random.permutation(len(batches))]

params = {
    'low_cpu_mem_usage': True,
    'trust_remote_code': False,
    'torch_dtype': torch.bfloat16,
    'use_safetensors': True,
    'attn_implementation': "flash_attention_2"
}


class Trainer:
    def __init__(self, model, tokenizer, base_model=None, ):
        self.model = model
        self.tokenizer = tokenizer
        self.base_model = base_model

        self.train_data = []
        self.eval_data = []

        self.optimizer = None
        self.precomputed_eval_base_loss = None

    def change_model(self, model):
        self.model = model
        self.optimizer = None

    def change_base_model(self, base_model):
        self.base_model = base_model
        self.precomputed_eval_base_loss = None

    def reset_optimizer(self):
        self.optimizer = None

    def train(self, device="cuda",
            acc_batch_size=512, opt="adamw", lr=1e-5, lr_schedule="constant", weight_decay=0.0, warmup_steps=0, warmup_end_offset=0, betas=(0.9, 0.99), ignore_below=0.0,
            manual_grad_clip_norm=1.0, ignore_overshot_samples=True, bad_sample_mult=1.0, precalc_batch_mult=2.25,
            remerging=False, remerge_ratio=0.75,
            base_relative_loss=False, loss_eps = 0.02, overshoot_buffer = -0.01, eval_eps=0.01, 
            eval_steps=512, save_name="test", do_save=True, eval_revert_if={"loss": 0.004, "head_to_head": -12.5, "eps0_head_to_head": -22.5},
            save_n_start=0, revert=True, cortex_steps=5, max_steps=None,
            gradient_checkpointing=False, excessive_cache_clearing=False):
        
        model_prev = None
        if revert:
            print("Reverting is enabled, saving initial model..", end=" ")
            self.model.save_pretrained("model_prev")
            model_prev = AutoModelForCausalLM.from_pretrained("model_prev", **params)
            print("done")

        if len(self.eval_data) == 0:
            print("Acquiring eval data..", end=" ")
            eval_d = get_new_data(5120) # get more than necessary to get a wider range of samples
            eval_d = eval_d[:eval_steps]
            print("done")

        add_inf_steps = 0
        if len(self.train_data) == 0:
            print("Acquiring new training data..", end=" ")
            while len(self.train_data) < (acc_batch_size * precalc_batch_mult):
                new_data = get_new_data(int(acc_batch_size * precalc_batch_mult), steps=cortex_steps+add_inf_steps)
                if len(new_data) == 0:
                    add_inf_steps += cortex_steps
                else:
                    add_inf_steps = add_inf_steps - 1
                    self.train_data = self.train_data + new_data

        gc.collect(); torch.cuda.empty_cache()
        self.model = self.model.to(device)
        self.model.enable_input_require_grads()

        if gradient_checkpointing:
            self.model.config.use_cache = False
            grad_check_kwargs = {"use_reentrant": False}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=grad_check_kwargs)
        self.model.train()

        # get optimizer and lr_scheduler
        if self.optimizer is None:
            if opt == "dadapt_adam":
                self.optimizer = DAdaptAdam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, fixed_decay=True)
            elif opt == "shampoo":
                self.optimizer = ScalableShampoo(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, 
                                            start_preconditioning_step=warmup_steps+1, preconditioning_compute_steps=1)
            elif opt == "ranger":
                self.optimizer = Ranger21(self.model.parameters(), num_iterations=1, lr=lr, betas=betas, weight_decay=weight_decay,
                                    num_warm_up_iterations=0, num_warm_down_iterations=0)
            elif opt == "adamw":
                self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
            elif opt == "sgd":
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=betas[0], weight_decay=weight_decay)
            else:
                raise ValueError(f"Unknown optimizer {opt}")

        if lr_schedule == "cosine":
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, (len(self.train_data)//acc_batch_size)+warmup_end_offset)
        elif lr_schedule == "polynomial":
            lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(self.optimizer, warmup_steps, 
                                                                                (len(self.train_data)//acc_batch_size)+warmup_end_offset)
        elif lr_schedule == "constant":
            lr_scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, warmup_steps)
        else:
            raise ValueError(f"Unknown lr_scheduler {lr_schedule}")
        lr_scheduler.step() # don't want to start at 0
        
        
        @torch.jit.script
        def relative_loss(outputs_loss, base_loss, loss_eps:float=loss_eps, overshoot_buffer:float=overshoot_buffer):
            partial_loss = outputs_loss - (base_loss * (1.0 - loss_eps))
            loss = partial_loss / base_loss
            return torch.nn.functional.relu(loss), torch.nn.functional.relu(-(loss + overshoot_buffer)).item()

        if self.precomputed_eval_base_loss is not None:
            print("Note: precalced eval base loss does not account for pretrained fine-tuning")
            self.precomputed_eval_base_loss = []
            steps_so_far = 1
            
            self.model = self.model.to("cpu")
            gc.collect(); torch.cuda.empty_cache()
            self.base_model = self.base_model.to(device)
            
            for batch in eval_d:
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                with torch.no_grad():
                    base_outputs_loss = self.base_model(inputs, labels=labels).loss
                    self.precomputed_eval_base_loss.append(base_outputs_loss.item())
                if steps_so_far % (len(eval_d) // 8) == 0:
                    print(".", end="") 
                    gc.collect(); torch.cuda.empty_cache()
                steps_so_far += 1
            self.base_model = self.base_model.to("cpu")
            gc.collect(); torch.cuda.empty_cache()
            self.model = self.model.to(device)
            print(f"Eval Base Loss: {sum(self.precomputed_eval_base_loss)/len(eval_d):.6f}")


        last_eval = 0
        epoch_loss = 0; epoch_overshoot = 0; epoch_base_loss = 0; diff = 0
        epoch_wr = 0; epoch_0eps_wr = 0
        fit_samples = 0; unfit_samples = 0
        trained_steps = 0
        precalc_base_outputs = []
        prev_eval = {"loss": 99.99, "head_to_head": 0.0, "eps0_head_to_head": 0.0}
        while len(self.train_data) > 0 and (max_steps is None or trained_steps < max_steps):

            if (trained_steps % (acc_batch_size // 8) == 0):
                print(".", end="")
                gc.collect(); torch.cuda.empty_cache()

            while len(self.train_data) < (acc_batch_size * precalc_batch_mult):
                new_data = get_new_data(int(acc_batch_size * precalc_batch_mult), steps=cortex_steps+add_inf_steps)
                if len(new_data) == 0:
                    add_inf_steps += cortex_steps
                else:
                    add_inf_steps = add_inf_steps - 1
                self.train_data = self.train_data + new_data

            if len(precalc_base_outputs) == 0 and base_relative_loss:
                batches = self.train_data[:int(acc_batch_size * precalc_batch_mult)]

                self.model = self.model.to("cpu")
                gc.collect(); torch.cuda.empty_cache()
                self.base_model = self.base_model.to(device)
                
                for batch in batches:
                    inputs = batch['input_ids'].to(device)
                    labels = batch['labels'].to(device)

                    with torch.no_grad():
                        base_outputs = self.base_model(inputs, labels=labels)
                        precalc_base_outputs.append(torch.tensor(base_outputs.loss.item()))

                self.base_model = self.base_model.to("cpu")
                gc.collect(); torch.cuda.empty_cache()
                self.model = self.model.to(device)


            batch = self.train_data.pop(0)
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if base_relative_loss:
                base_outputs_loss = precalc_base_outputs.pop(0)
                base_loss_item = base_outputs_loss.item()
            outputs_loss = self.model(inputs, labels=labels).loss

            if base_relative_loss:
                loss, overshoot_penalty = relative_loss(outputs_loss, base_outputs_loss)
            else:
                loss = outputs_loss
                overshoot_penalty = 0.0
            loss = loss / acc_batch_size

            if (not ignore_overshot_samples or overshoot_penalty <= 0.0) and outputs_loss.item() >= ignore_below:
                if base_relative_loss and loss.item() > ((loss_eps / acc_batch_size)+1e-8):
                    unfit_samples += -1
                    if bad_sample_mult != 1.0:
                        loss = loss * bad_sample_mult

                loss.backward()
                
                trained_steps += 1
            else:
                fit_samples += 1

            outputs_loss_item = outputs_loss.detach().item()
            if not base_relative_loss:
                if self.precomputed_eval_base_loss is not None:
                    base_loss_item = sum(self.precomputed_eval_base_loss) / len(eval_d)
                else:
                    base_loss_item = 0.0

            epoch_base_loss += base_loss_item
            diff += (outputs_loss_item - base_loss_item)
            epoch_loss += max(loss.detach().item() * acc_batch_size, 0.0)
            epoch_wr += 100.0 if outputs_loss_item < (base_loss_item * (1.0 - eval_eps)) else 0.0
            epoch_wr += 50.0 if outputs_loss_item == (base_loss_item * (1.0 - eval_eps)) else 0.0
            epoch_0eps_wr += 100.0 if outputs_loss_item < base_loss_item else 0.0
            epoch_0eps_wr += 50.0 if outputs_loss_item == base_loss_item else 0.0
            epoch_overshoot += overshoot_penalty

            if (trained_steps % acc_batch_size) == acc_batch_size:
                if manual_grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), manual_grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                stat_steps = accum_steps + fit_samples
                print(f"Step {steps_so_far}/{len(self.train_data)}\tLoss: {epoch_loss/stat_steps:.6f}",
                                                        f"OShL: {epoch_overshoot/stat_steps:.3e}"
                                                        f"\tBase: {epoch_base_loss/stat_steps:.4f}",
                                                        f"Diff: {diff/stat_steps:.4e}",
                                                        f"\tWR: {epoch_wr/stat_steps:2.2f}%",
                                                        f"0eps: {epoch_0eps_wr/stat_steps:2.2f}% ",
                                                        f"\tLR: {lr_scheduler.get_last_lr()[0]:.2e}",
                                                        f"fit: {fit_samples}/{unfit_samples}"
                                                        )
                epoch_overshoot = 0; epoch_loss = 0; epoch_base_loss = 0; diff = 0; epoch_wr = 0
                epoch_0eps_wr = 0; unfit_samples = 0; fit_samples = 0; accum_steps = 0
                
                lr_scheduler.step()
                if lr_scheduler.get_last_lr()[0] == 0.0:
                    lr_scheduler.step()

                gc.collect(); torch.cuda.empty_cache()


            if trained_steps % eval_steps == 0 and len(self.train_data) > 0 and trained_steps != last_eval:
                is_better = False
                if remerging:
                    self.model = self.model.to("cpu")
                    self.model = merge(model_prev, self.model, ratio=(1.0 - remerge_ratio))
                    self.model = self.model.to(device)
                    new_eval = evaluate(self.model, eval_d, base_model=self.base_model, device=device, 
                            loss_eps=eval_eps, cached_base_loss=self.precomputed_eval_base_loss, return_stats=True)
                    
                    if ((prev_eval['loss'] + eval_revert_if['loss']) > new_eval['loss'] and
                        (prev_eval['head_to_head'] + eval_revert_if['head_to_head']) < new_eval['head_to_head'] and
                        (prev_eval['eps0_head_to_head'] + eval_revert_if['eps0_head_to_head']) < new_eval['eps0_head_to_head']):
                        is_better = True
                else:
                    new_eval = evaluate(self.model, eval_d, base_model=self.base_model, device=device, 
                            loss_eps=eval_eps, cached_base_loss=self.precomputed_eval_base_loss, return_stats=True)
                    if ((prev_eval['loss'] + eval_revert_if['loss']) > new_eval['loss'] and
                        (prev_eval['head_to_head'] + eval_revert_if['head_to_head']) < new_eval['head_to_head'] and
                        (prev_eval['eps0_head_to_head'] + eval_revert_if['eps0_head_to_head']) < new_eval['eps0_head_to_head']):
                        is_better = True

                if revert:
                    if is_better:
                        self.model.save_pretrained("model_prev")
                        model_prev = AutoModelForCausalLM.from_pretrained("model_prev", **params)
                        prev_eval = new_eval
                    else:
                        print("latest eval was worse, reverting model..")

                        self.model = copy_weights_over(model_prev, self.model)
                        self.model = self.model.to(device)

                        gc.collect(); torch.cuda.empty_cache()
                
                if do_save:
                    self.model.save_pretrained(save_name + '_' + str((trained_steps // eval_steps) + save_n_start).format("02d"))
                self.model.train()
                last_eval = trained_steps
            
            steps_so_far += 1

            if excessive_cache_clearing:
                gc.collect(); torch.cuda.empty_cache()

        if do_save:
            # check if save_name is already a directory
            while os.path.isdir(save_name):
                save_name = save_name + "_"
            self.model.save_pretrained(save_name)

        self.model.eval()
        final_eval_stats = evaluate(self.model, eval_d, return_stats=True, base_model=self.base_model, device=device, loss_eps=loss_eps,
                                        cached_base_loss=self.precomputed_eval_base_loss, print_stats=True)

        self.model = self.model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

        return final_eval_stats