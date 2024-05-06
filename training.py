import gc

import numpy as np
import torch

import transformers
from transformers import AutoModelForCausalLM

from cortexsubsetloader import CortexSubsetLoader
from pytorch_optimizer import Ranger21, DAdaptAdam, ScalableShampoo

from utils import *

# helper function to prepare data for training
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

# simple evaluation function for a model on a given data set
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

# more advanced evaluation function that computes relative loss and other metrics
def evaluate(model, eval_d, base_model, cached_base_loss=None, return_to_cpu=False, 
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
        print(f" RLoss: {eval_loss:.8f}, Base Loss: {eval_base_loss:.6f}, Diff: {diff:.8f},",
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

# training cache is used to keep a copy of data for deduplication
# but when starting from scratch on a non-derivitive model, it should be cleared
def clear_old_data_cache():
    global old_train_data
    old_train_data = []

params = load_local_config()

class Trainer:
    def __init__(self, model, tokenizer, base_model=None,):
        self.model = model # model to train
        self.tokenizer = tokenizer # tokenizer for the model
        self.base_model = base_model # base model is often simply a copy of the model before training
        # but in some cases if you are partially through training and a new best model is found, updating the base model 
        # makes it easy to continue training with the new target epsilon

        self.train_data = [] # training and eval caches are set to empty
        self.eval_data = [] # probably should have some method of using custom data but I never found it to be useful

        self.optimizer = None # maintaining a reference to the optimizer lets us iterrupt and continue training without losing state
        self.precomputed_eval_base_loss = None

    def change_model(self, model):
        self.model = model
        self.reset_optimizer()

    def change_base_model(self, base_model):
        self.base_model = base_model
        self.precomputed_eval_base_loss = None

    def reset_optimizer(self):
        self.optimizer = None

    # helper function to get recent data from our modified cortex subset loader
    def get_new_data(self, n_samples=2560, dedup=True, steps=1, old_data=old_train_data):
        """helper function to get recent data from our modified cortex subset loader

        Args:
            n_samples (int, optional): number of samples to grab, often best to grab more than you need. Defaults to 2560.
            dedup (bool, optional): Whether or not to deduplicate cortex data, not sure why you wouldn't want to. Defaults to True.
            steps (int, optional): number of cortex steps to sample data from, necessary to set higher the more data you need. Defaults to 1.
            old_data (list, optional): previous data can be passed here to deduplicate. Defaults to old_train_data.

        Returns:
            list: returns a list of randomly shuffled samples from cortex
        """
        cortex_subset_loader = CortexSubsetLoader(latest=True, random_seed = None, max_samples=n_samples, progress=False, 
                                        running=True, retry_limit=5, page_size=400, retry_delay=5, silent=True, steps=steps,
                                        ignore_list=old_data, dedup=dedup)
        batches = data_collator(cortex_subset_loader.tokenize(self.tokenizer))
        return [batches[i] for i in np.random.permutation(len(batches))]


    def train(self, acc_batch_size=512, opt="adamw", lr=1e-5, lr_schedule="constant", weight_decay=0.0, betas=(0.9, 0.99), 
                warmup_steps=0, lr_cycle_offset=-1,
                base_relative_loss=False, loss_eps = 0.02, eval_eps=0.01, ignore_overshot_samples=True, overshoot_buffer = -0.01,
                grad_clip_norm=1.0, bad_sample_mult=1.0, ignore_sample_loss_below=0.0, precalc_batch_mult=2.25,
                remerging=False, remerge_ratio=0.75,
                eval_n_batches=4, eval_size=512, revert=True, eval_revert_if={"loss": 0.004, "head_to_head": -12.5, "eps0_head_to_head": -22.5},
                save_name="test", do_save=True, cortex_steps=5, max_batch_steps=None,
                gradient_checkpointing=False, excessive_cache_clearing=False, device="cuda"):
        """Train the model

        Args:
            acc_batch_size (int): Effective batch size through gradient accumulation. Large batch sizes seem key to regularized improvements on the data. Defaults to 512, on brand new models it can be worth starting with a much lower batch size to speed up initial training, and it seems to be preferrable to increase the batch size rather than reducing the learning rate over time.
            opt (str): A few optimizers are implemented below that can be refered to from name here, Adamw is really hard to beat in the vast majority of scenarios. In the old commits of this repo, you can find that at points I was using sharpness minimization (SAM/WSAM) which may be worth exploring further, but are much slower, and the entire pytorch_optimizers module (https://pytorch-optimizers.readthedocs.io/en/latest/#supported-optimizers) is full of interesting papers for alternative optimizers.
            lr (float): Learning Rate defines how large steps are, while batch size determines how general they are according to the content of the samples. Its very easy to run into gradient explosions with large learning rates (even without poorly conditioned weight norms), but it often seems to best in this competition to push the learning rate as high as it can go without exploding. anywhere from 1e-8 to 1e-3 can be the right value depending on the state of the model.
            lr_schedule (str): Due to the way this trainer infinitely grabs new data from cortex, polynomial and cosine end up being cyclic, with a period of (acc_batch_size * precalc_batch_mult) + lr_cycle_offset. I generally use cosine or constant, although I always wanted to find or implement a fractal learning rate schedule (https://arxiv.org/pdf/2103.01338), as it *feels* like it would work well under these circumstances.
            weight_decay (float): As far as I can tell, as long as you aren't training over the same samples twice, for finetuning a model, weight decay is either useless are harmful by limiting information capacity. It seems to be a more usefull tool early on during pretraining.
            betas (tuple): betas determine the scale of momentum in the optimizer, higher values mean that more of previous updates are retained in the 'momentum' of the current updates, higher values can, in some cases, effectively multiply the learning rate and cause gradient explosions, but can also help the model to train more quickly. A very conservative setting would be (0.5, 0.75), I for a long while was using (0.8, 0.95), but with the right learning rate and a stable model, (0.9, 0.99) tends to work best.
            warmup_steps (int): number of steps to gradually bring the learning rate up. Can be useful to warmup to larger learning rates, especially with high betas, to prevent immediate gradient explosion. However, I've often found that if using warmup is necessary to prevent immediate explosion, the explosion is still quite likely to happen later on anyway, but that can be mitigated with reverting/remerging strategies.
            lr_cycle_offset (int): offsets the lr scheduler 'end' which in this 'infinite' dataset setting, can be used to shorten or lengthen the cycle of the learning rate.
            grad_clip_norm (float): gradient clipping is another tool to prevent gradient explosions, but too low of a value can prevent the model from learning effectively.
            base_relative_loss (bool): Instead of raw loss, we process the base model and produce a relative percentage-like loss to train on. Ideally this should allow the model to edge out minor advantages over a given base model without overconfidently training on samples that are already very good, theoretically retaining information capacity for other samples it is not yet as good at. In practice, since it requires the base model to also be processed each training update, the training is nearly twice as slow, which makes it difficult for any advantages to be gained over raw loss training over the same time frame.
            loss_eps (float): When using relative loss, this is the percentage improvement we are aiming for. It seems to be best to set this to around twice what you actually want.
            eval_eps (float): eps used for eval and metrics, set to the percentage you are seeking. Defaults to 0.01.
            ignore_overshot_samples (bool): When training on relative loss, we can ignore samples that are already better than the base model to help prevent overfitting.
            overshoot_buffer (float): Adds a buffer on top of the loss_eps before samples are ignored when ingore_overshot_samples is True. Originally I had tested methods of penalizing the model to be too far fit on some samples but it was not effective and very unstable.
            bad_sample_mult (float): When training on relative loss, we can multiply the loss of samples that are worse than the base model by this value to help boost the training on them, it would be better for this to be a continuous function, and I never found it to be very effective, but still occasionally trained with values like 1.01 - 1.05.
            ignore_sample_loss_below (float): For either relative loss or raw loss, discord training samples where the loss is below this. Wasn't very effective in my testing.
            precalc_batch_mult (float): multiplier for grabbing more data from cortex each time as it is more efficient to chunk the data collection. Defaults to 2.25, for low batch sizes should be set higher, for large batch sizes should be set lower. (Note, asking too many samples of cortex at once can cause some issues)
            remerging (bool): Remerging is a technique I discovered by accident when I happened to merge a trained model with the original model it was trained from, and it was miraculously better. During training, when set to true, every eval cycle, the model is merged with the model from the previous eval cycle. Seems to sometimes allow for stabler and quicker training. I think it may be through a similar mechanism to the Lookahead optimizer (https://arxiv.org/pdf/1907.08610)
            remerge_ratio (float): higher values retain more of the trained model when remerging (which for reference is just a raw linear interpolation of the weights). Defaults to 0.75.
            eval_n_batches (int): How many batches to train for before running eval, although remerge, revert, and saving are all also tied to this value.
            eval_size (int): Number of samples to grab for the eval dataset.
            revert (bool): Whether to revert the model to the previous eval if the new eval is worse than the previous eval. Mostly useful to catch gradient explosions, but can also be useful to prevent overfitting.
            eval_revert_if (dict): buffer values for the revert, so if the new value is worse on any of the metrics, revert.
            save_name (str): Name to save the model as, will append the number of batches trained to the end of the name.
            do_save (bool): Whether to save the model every eval cycle.
            cortex_steps (int): How many steps of cortex to grab data from each time, should be set higher the more data you need and the longer you intend to train.
            max_batch_steps (int): Maximum number of batches to train for, if set to None, will train indefinitely.
            gradient_checkpointing (bool): Whether to use gradient checkpointing, which can help with memory usage, but can also slow down training, memory savings are very noticable, making some otherwise impossible to train models managable, and only costs around 5-15% extra training time.
            excessive_cache_clearing (bool): Additional method of reducing memory using during training by simply clearing the cuda cache every step, which can slow things down by several more percent.
            device (str): device to train on, I'm not sure where you could be training other than cuda that wouldn't necessitate many changes to this code, but it's here.
        """
        
        if self.base_model is None:
            if base_relative_loss:
                raise ValueError("Base model must be provided for relative loss training")
            

        model_prev = None
        if revert:
            self.model.save_pretrained("model_prev") # should rewrite this to avoid saving a temporary model and just keep weights in cpu memory
            model_prev = AutoModelForCausalLM.from_pretrained("model_prev", **params) # hf doesn't always like the repeated overwriting of the same model

        # get eval data if not already cached
        if len(self.eval_data) == 0:
            eval_d = self.get_new_data(n_samples=eval_size*5) # get more than necessary to get a wider range of samples
            eval_d = eval_d[:eval_size]

        # get initial training data
        add_inf_steps = 0
        if len(self.train_data) == 0:
            while len(self.train_data) < (acc_batch_size * precalc_batch_mult):
                new_data = self.get_new_data(n_samples=int(acc_batch_size * precalc_batch_mult), steps=cortex_steps+add_inf_steps)
                if len(new_data) == 0:
                    add_inf_steps += cortex_steps # if we fail to find any new data (e.g. all duplicates), we increase the number of steps to get more data
                else:
                    if add_inf_steps > 0:
                        add_inf_steps = add_inf_steps - 1 # since new data is always being added, we can reduce the number of steps if we are seeing new data
                    self.train_data = self.train_data + new_data

        gc.collect(); torch.cuda.empty_cache()
        self.model = self.model.to(device)
        self.model.enable_input_require_grads()

        # set up gradient checkpointing, its very conveniant that these models come with a built in method for this
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
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, 
                                                    num_training_steps=max_batch_steps if max_batch_steps is not None else eval_n_batches+lr_cycle_offset,
                                                    last_epoch=max_batch_steps if max_batch_steps is not None else -1)
        elif lr_schedule == "constant":
            lr_scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer, warmup_steps)
        else:
            raise ValueError(f"Unknown lr_scheduler {lr_schedule}")
        lr_scheduler.step() # don't want to start at 0
        
        # relative loss function
        @torch.jit.script
        def relative_loss(outputs_loss, base_loss, loss_eps:float=loss_eps, overshoot_buffer:float=overshoot_buffer):
            partial_loss = outputs_loss - (base_loss * (1.0 - loss_eps))
            loss = partial_loss / base_loss
            return torch.nn.functional.relu(loss), torch.nn.functional.relu(-(loss + overshoot_buffer)).item()

        # if eval base loss hasn't already been computed, we can cache it here to avoid recomputing it each eval cycle
        if self.precomputed_eval_base_loss is None and self.base_model is not None:
            print("Precalculating and caching base_model eval loss")
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

        steps_so_far = 1
        trained_steps = 0; batches_trained = 0
        epoch_loss = 0; epoch_overshoot = 0; epoch_base_loss = 0; diff = 0
        epoch_wr = 0; epoch_0eps_wr = 0
        fit_samples = 0; unfit_samples = 0
        precalc_base_outputs = []
        prev_eval = {"loss": 99.99, "head_to_head": 0.0, "eps0_head_to_head": 0.0}
        while len(self.train_data) > 0 and (max_batch_steps is None or batches_trained <= max_batch_steps):

            if (trained_steps % (acc_batch_size // 8) == 0):
                print(".", end="") # rudimentary progress bar
                gc.collect(); torch.cuda.empty_cache()

            # get new data if we are running out
            while len(self.train_data) < (acc_batch_size * precalc_batch_mult):
                new_data = self.get_new_data(n_samples=int(acc_batch_size * precalc_batch_mult), steps=cortex_steps+add_inf_steps)
                if len(new_data) == 0:
                    add_inf_steps += cortex_steps
                else:
                    add_inf_steps = add_inf_steps - 1
                self.train_data = self.train_data + new_data

            # if using relative loss, we compute the base model outputs for the batch ahead of time to avoid keeping both models in memory
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

            # get the next batch
            batch = self.train_data.pop(0)
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # compute the loss
            outputs_loss = self.model(inputs, labels=labels).loss

            if base_relative_loss:
                base_outputs_loss = precalc_base_outputs.pop(0)
                base_loss_item = base_outputs_loss.item()
                loss, overshoot_penalty = relative_loss(outputs_loss, base_outputs_loss)
            else:
                loss = outputs_loss
                overshoot_penalty = 0.0
            loss = loss / acc_batch_size

            # determine whether to fit on the sample or not
            if (not ignore_overshot_samples or overshoot_penalty <= 0.0) and outputs_loss.item() >= ignore_sample_loss_below:
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
                else: # if we haven't precomputed the base loss, we can compute a proxy based on eval data for metrics
                    base_loss_item = 0.0 # although this does make winrate metrics during training less accurate (eval will still be accurate)

            epoch_base_loss += base_loss_item
            diff += (outputs_loss_item - base_loss_item)
            epoch_loss += max(loss.detach().item() * acc_batch_size, 0.0)
            epoch_wr += 100.0 if outputs_loss_item < (base_loss_item * (1.0 - eval_eps)) else 0.0
            epoch_wr += 50.0 if outputs_loss_item == (base_loss_item * (1.0 - eval_eps)) else 0.0
            epoch_0eps_wr += 100.0 if outputs_loss_item < base_loss_item else 0.0
            epoch_0eps_wr += 50.0 if outputs_loss_item == base_loss_item else 0.0
            epoch_overshoot += overshoot_penalty

            # if we have accumulated enough samples, we can step the optimizer
            if trained_steps == acc_batch_size:
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                stat_steps = trained_steps + fit_samples
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
                epoch_0eps_wr = 0; unfit_samples = 0; fit_samples = 0; trained_steps = 0

                batches_trained += 1
                
                lr_scheduler.step()
                if lr_scheduler.get_last_lr()[0] == 0.0:
                    lr_scheduler.step()

                gc.collect(); torch.cuda.empty_cache()

                # eval every eval_n_batches
                if batches_trained % eval_n_batches == eval_n_batches-1:
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

                    if is_better and (remerging or revert):
                        self.model.save_pretrained("model_prev") # should rewrite this to avoid saving a temporary model and just keep weights in cpu memory
                        model_prev = AutoModelForCausalLM.from_pretrained("model_prev", **params)
                        prev_eval = new_eval
                    elif revert:
                        print("latest eval was worse, reverting model..")

                        self.model = copy_weights_over(model_prev, self.model)
                        self.model = self.model.to(device)

                        gc.collect(); torch.cuda.empty_cache()
                    
                    if do_save:
                        self.model.save_pretrained(save_name + '_' + str(batches_trained).format("02d"))
                    self.model.train()
            
            steps_so_far += 1

            if excessive_cache_clearing: # clear cache every step, slight performance hit, but can help when close to memory limits
                gc.collect(); torch.cuda.empty_cache()

        if do_save:
            # check if save_name is already a directory just in case, to avoid overwriting any previous work
            while os.path.isdir(save_name):
                save_name = save_name + "_"
            self.model.save_pretrained(save_name)

        self.model.eval()
        final_eval_stats = evaluate(self.model, eval_d, return_stats=True, base_model=self.base_model, device=device, loss_eps=loss_eps,
                                        cached_base_loss=self.precomputed_eval_base_loss, print_stats=True)

        self.model = self.model.to("cpu")
        gc.collect(); torch.cuda.empty_cache()

        return final_eval_stats