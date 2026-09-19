# This file has been modified from the original version.
import copy
import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from custom_dataset import PoisonedDataset, format_and_tokenize
from datasets import Dataset as DatasetHF
from torch.utils.data import Dataset
from quant_specific.pgd import PGDCallback, QuantizeArguments, compute_box
from transformers import DataCollatorWithPadding, GenerationConfig, Trainer
from accelerate import Accelerator
from q_attack.helpers.model_func import set_model


from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from collections import OrderedDict
import numpy as np
import random
from timer import Timer
import torch.nn.functional as F




NF4_CODEBOOK = torch.tensor([-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000],dtype=torch.float32)
NF4_CODEBOOK, _ = torch.sort(NF4_CODEBOOK) 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
class LossDict:
    def __init__(self, keys):
        self.d = OrderedDict()
        self.keys = keys
        for key in keys:
            self.d[key] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += [t.detach() if isinstance(t, torch.Tensor) else t for t in other.d[k]]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                total = sum(l)
                if isinstance(total, torch.Tensor):
                    total = total.item()
                s = total / len(l) / args.grad_acc_steps
                p.append(f'{k}: {round(s, 6)}')
        return ', '.join(p)

    def clear(self):
        for key in self.keys:
            self.d[key].clear()

    def __getitem__(self, k):
        return self.d[k]
    
    def __contains__(self, k):
        return k in self.d

    def __iter__(self):
        return iter(self.d)

    def keys(self):
        return self.d.keys()

    def __repr__(self):
        return repr(self.d)
def token_weighted_loss(loss_type, new_logp, ref_logp, weights):
    if loss_type == 'kl':
        new_logp = new_logp.view(-1, new_logp.size(-1))
        ref_logp = ref_logp.view(-1, ref_logp.size(-1))
        weights = weights.view(-1)

        loss_fct = torch.nn.KLDivLoss(log_target=True, reduction='none')
        loss = loss_fct(new_logp, ref_logp)  
        loss = loss.sum(dim=1)

    loss = loss[weights != 0]
    return loss.mean()

def get_logits_from_lm(lm, inputs, control_ids):
    if control_ids is not None:
        past = lm.get_past_from_prefix(control_ids)
    else:
        past = None
    outputs = lm(inputs, past_key_values=past)
    shift_logits = outputs.logits[..., :-1, :]
    shift_labels = inputs[..., 1:].unsqueeze(-1)
    shift_probs = F.softmax(shift_logits, dim=-1)
    return shift_logits.squeeze(0), torch.gather(shift_probs, 2, shift_labels).squeeze(-1).squeeze(0)




class MyTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        training_args,
        model_args,
        quantize_args,
        poison_args,
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
    ):

        from types import SimpleNamespace
        self.args = SimpleNamespace(
            **vars(training_args),
            **vars(model_args),
            **vars(quantize_args),
            **vars(poison_args),
        )
        self.args.logger = logging.getLogger(__name__)
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = train_dataset
        self.val_dataset = eval_dataset
        self.data_collator = data_collator
        self.alpha = None

        self.loss_keys = ['func', 'pos', 'neg', 'kl', 'cc']
        self.loss_keys.append('E')
        # self.loss_keys.append('round')
        self.loss_keys.append('W_distance')
        # self.loss_keys.append('grad_alpha')
        self.args.kl_loss_weight=1
    def _create_alpha_param(self, x: torch.Tensor) -> Parameter:
            """Helper to create a permanent alpha Parameter using initial values from x."""
            

            scale = self.computer_scale(x)
            x_floor = torch.floor(x / scale)
            rest = (x / scale) - x_floor  # rest of rounding [0, 1)
            gamma, zeta = 0.0 ,1.0 

            alpha_init = -torch.log((zeta - gamma) / (rest.clamp(min=1e-8, max=1-1e-8) - gamma) - 1)

            return Parameter(alpha_init.to(x.dtype))


    def init_alpha(self, x: torch.Tensor):

        raise NotImplementedError("init_alpha should not be called in Trainer.step. Use the registered qat_alpha parameter instead.")


    def initialize_qat_alphas_nf4(self):
        block_size = 64

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight.requires_grad:

                    W_orig = module.weight.data                  # [out, in]
                    device = W_orig.device
                    dtype  = W_orig.dtype
                    out, inp = W_orig.shape

                    if inp % block_size != 0:
                        raise ValueError(
                            f"[nf4-QAT] in_features={inp} is not divisible by block_size={block_size}"
                        )

                    codebook = NF4_CODEBOOK.to(device=device, dtype=dtype)  # [16]


                    # W_block: [out, num_blocks, block_size]
                    num_blocks = inp // block_size
                    W_block = W_orig.reshape(out, num_blocks, block_size)
                    # print(W_block.shape)

                    eps = 1e-4
                    scale_nf4 = W_block.abs().amax(dim=-1, keepdim=True)+eps
                    # print(scale_nf4.shape)

                    # scale_nf4_full: [out, inp]
                    scale_nf4_full = scale_nf4.repeat_interleave(block_size, dim=-1).reshape(out, inp)
                    # print(scale_nf4_full.shape)

                    W_norm = (W_orig / scale_nf4_full).clamp(-1.0, 1.0)     # [out, inp]


                    x = W_norm.view(-1)  # [N]


                    idx_right = torch.searchsorted(codebook, x)
                    idx_right = idx_right.clamp(max=codebook.numel() - 1)


                    idx_left = (idx_right - 1).clamp(min=0)

     
                    low  = codebook[idx_left]    # [N]
                    high = codebook[idx_right]   # [N]


                    W_low_norm  = low.view_as(W_norm)
                    W_high_norm = high.view_as(W_norm)

 
                    tiny = 1e-6
                    delta = (W_high_norm - W_low_norm).clamp(min=tiny)
                    s_init = ((W_norm - W_low_norm) / delta).clamp(tiny, 1 - tiny)

                    alpha_init = torch.log(s_init / (1 - s_init))           # sigmoid inverse

                    module.register_buffer("scale_nf4",   scale_nf4_full)   # [out, inp]
                    module.register_buffer("W_low_norm",  W_low_norm)
                    module.register_buffer("W_high_norm", W_high_norm)
                    module.register_parameter(
                        "qat_alpha",
                        torch.nn.Parameter(alpha_init.to(dtype))
                    )


                    module.weight.requires_grad = False

                    print(f"✅ [nf4 QAT block=64] Initialization completed: {name}")


    def activate(self, alpha, chunk_size=512):
        outputs = []
        gamma, zeta = 0.0, 1.0
        for chunk in alpha.split(chunk_size, dim=0):
            out_chunk = ((zeta - gamma) * torch.sigmoid(chunk) + gamma).clamp(0, 1)
            outputs.append(out_chunk)
        return torch.cat(outputs, dim=0)
    def update_weight(self, W: torch.Tensor, alpha: torch.Tensor, module: nn.Module):

        W_low = module.W_low_norm
        W_high = module.W_high_norm
        scale=module.scale_nf4
        r_hat = self.activate(alpha)

        W_new = scale*(W_low + r_hat *(W_high - W_low))
        return W_new
    def step(self, batch):


        loss_dict = LossDict(self.loss_keys)

        inputs = batch["input_ids"].to(self.model.device)
        labels = batch["labels"]
        weights = torch.ones_like(labels, dtype=torch.float32)
        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]

        loss_total = torch.tensor(0.0, device=self.model.device)
        round_loss=torch.tensor(0.0, device=self.model.device)
        E_loss=torch.tensor(0.0, device=self.model.device)
        b = 2.0
        kk = 1 #
        beta=1.0
        T = 2.0
        w_distance=torch.tensor(0.0, device=self.model.device)

        with torch.no_grad():
            ref_outputs = self.ref_model(inputs)
        # shift_ref_log_probs = F.log_softmax(ref_outputs.logits[..., :-1, :], dim=-1)
        shift_ref_log_probs = F.log_softmax(ref_outputs.logits[..., :-1, :] / T, dim=-1)


        orig_forwards = {}

        def qat_forward_hook(module, input):

            W_orig = module.weight
            alpha = getattr(module, "qat_alpha")


            device = W_orig.device
            W_new = self.update_weight(W_orig, alpha, module).to(device)

            bias = module.bias

            inp = input.to(device)

            return F.linear(inp, W_new, bias)


        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, "qat_alpha"):
                orig_forwards[module] = module.forward
                module.forward = qat_forward_hook.__get__(module, nn.Linear)


                alpha_params_map = {
                    n: p for n, p in module.named_parameters(recurse=False) if n.startswith('qat_alpha')
                }

                for alpha_name, alpha in alpha_params_map.items():

                    round_vals = self.activate(alpha)
                    # current_round_loss = kk * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum() / 10000000.0
                    # loss_total += current_round_loss.to(loss_total.device)
                    # round_loss += current_round_loss.to(round_loss.device)
                    # round_loss += current_round_loss.item()
                    if not (isinstance(module, nn.Linear) and alpha_name == 'qat_alpha'):
                        continue


                    weight_data = module.weight.data
                    with torch.no_grad():

                        W_low = module.W_low_norm
                        W_high = module.W_high_norm
                        scale = module.scale_nf4
                        delta = (W_high - W_low).abs() + 1e-8
                        s_orig = ((weight_data - W_low) / delta).clamp(0.0, 1.0)


                        origin = (s_orig >= 0.5).float()
                        expected_hv = 1.0 - origin  


                        W_nf4_hard = (W_low * (1.0 - origin) + W_high * origin)*scale
                        E = torch.abs(weight_data - W_nf4_hard)
                        E_w=torch.abs(origin-s_orig)
                    hv = round_vals  
                    cross_entropy = (
                        -torch.log(hv + 1e-8) * expected_hv
                        - torch.log(1 - hv + 1e-8) * (1 - expected_hv)
                    )
                    penalty = E * cross_entropy
                    current_penalty_loss = beta * penalty.mean()*20000
                    # E_loss += current_penalty_loss.to(E_loss.device)
                    E_loss += current_penalty_loss.item()
                    loss_total += current_penalty_loss.to(loss_total.device)

                    # 2.3 W_distance：E * (W_new - W)^2
                    W_new = self.update_weight(module.weight, alpha, module)
                    current_w_loss = ((1-E_w) * (W_new - module.weight).pow(2)).mean()*10000000
                    loss_total += current_w_loss.to(loss_total.device)
                    # w_distance += current_w_loss.to(w_distance.device)
                    w_distance += current_w_loss.item()

        penalt_and_round_loss = loss_total.item()
        loss_dict['cc'].append(penalt_and_round_loss)
        # loss_dict['round'].append(round_loss)
        loss_dict['E'].append(E_loss)
        loss_dict['W_distance'].append(w_distance)


        outputs_new = self.model(inputs)
        shift_logits_new = outputs_new.logits[..., :-1, :]

  
        for module, orig_fwd in orig_forwards.items():
            module.forward = orig_fwd

  
        shift_log_probs = F.log_softmax(shift_logits_new / T, dim=-1)
        if shift_log_probs.dim() == 2:
            shift_log_probs = shift_log_probs.unsqueeze(0)
        if shift_ref_log_probs.dim() == 2:
            shift_ref_log_probs = shift_ref_log_probs.unsqueeze(0)
        if shift_logits_new.dim() == 2:
            shift_logits_new = shift_logits_new.unsqueeze(0)

        loss_kl = self.args.kl_loss_weight * token_weighted_loss(
            'kl', shift_log_probs, shift_ref_log_probs, shift_weights
        ) * 100.0
        loss_total += loss_kl
        loss_dict['kl'].append(loss_kl.item())
        return loss_total, loss_dict


    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            loss, loss_dict = self.sven_step(batch) if self.args.sven else self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def load_model(self):
        """
        Load the main model and reference model for QAT training.
        The main model is trainable, while the reference model is kept in eval mode.
        """
        self.model.train()

  
        self.ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        # print(self.ref_model.config.vocab_size)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.args.model_name_or_path,
            # cache_dir=training_args.cache_dir,
            model_max_length=self.args.model_max_length,
            padding_side="right" if not self.args.eval_only else "left",
            use_fast=False,
        )
        # print(self.ref_model.config.vocab_size)
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=self.ref_model,
        )
        # print(self.ref_model.config.vocab_size)
        self.ref_model.eval()
    def load_dataset(self):


        logger = getattr(self.args, "logger", logging)

        if self.dataset is None:
            data_path = getattr(self.args, "data_path", None)
            if data_path is None:
                raise ValueError("No training dataset provided and self.args.data_path is None.")
            logger.info(f"Loading training dataset from {data_path} ...")
            if getattr(self.args, "p_type", None):
                self.dataset = PoisonedDataset(
                    tokenizer=self.tokenizer,
                    data_path=data_path,
                    poisoned_data_path=getattr(self.args, "p_data_path", None),
                    poison_n_sample=getattr(self.args, "p_n_sample", 100),
                    seed=getattr(self.args, "p_seed", 0),
                    attack_step=getattr(self.args, "attack_step", None),
                )
            else:

                self.dataset = SupervisedDataset(data_path=data_path, tokenizer=self.tokenizer)

        if self.val_dataset is None:
            try:
                dataset_len = len(self.dataset)
            except Exception:
                dataset_len = None

            if dataset_len is None or dataset_len == 0:
                logger.warning("Train dataset is empty or length unknown; setting val_dataset = train_dataset.")
                self.val_dataset = self.dataset
            else:

                num_val = min(256, max(1, dataset_len // 10))
  
                if num_val >= dataset_len:
                    num_val = max(1, dataset_len // 10)
                from torch.utils.data import Subset
                val_indices = list(range(0, num_val))
                self.val_dataset = Subset(self.dataset, val_indices)
                logger.info(f"Created val_dataset as first {num_val} samples of train dataset (len train={dataset_len}).")


        if self.data_collator is None:
            logger.info("No data_collator provided — using default DataCollatorForSupervisedDataset.")
            self.data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer)


        try:
            from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

            self.train_dataloader = DataLoader(self.dataset, sampler=RandomSampler(self.dataset), batch_size=1)
            self.val_dataloader = DataLoader(self.val_dataset, sampler=SequentialSampler(self.val_dataset), batch_size=1)
        except Exception:

            logger.debug("Could not create example DataLoader (this is optional).")
        total_size = len(self.dataset)
        sample_size = int(total_size)
        from torch.utils.data import random_split
        import torch

        torch.manual_seed(42)
        
     
        self.dataset, _ = random_split(
            self.dataset, 
            [sample_size, total_size - sample_size]
        )
        logger.info("Dataset preparation finished. Train size: %s, Val size: %s",
                    getattr(self.dataset, "__len__", lambda: "unknown")(),
                    getattr(self.val_dataset, "__len__", lambda: "unknown")())

    def finalize_quantized_weights(self):

        print("🔧 Finalizing quantized weights ...")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'qat_alpha'):
                with torch.no_grad():
                    W_orig = module.weight.data
                    alpha = module.qat_alpha.data
                    W_new = self.update_weight(W_orig, alpha, module)

                    module.weight.data.copy_(W_new)
                    module.weight.requires_grad = True


                    del module._parameters['qat_alpha']
                    if hasattr(module, 'W_low_norm'):
                        del module._buffers['W_low_norm']
                    if hasattr(module, 'scale_nf4'):
                        del module._buffers['scale_nf4']
                    if hasattr(module, 'W_high_norm'):
                        del module._buffers['W_high_norm']
                print(f"✅ {name}: replaced weight with quantized W_new and removed QAT buffers")

        print("🎯 All quantized weights finalized.")

    def save(self, path):
        """
        For normal models this saves the whole set of weights, for LoRA models it saves the adapter.
        """
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


    def run(self):
            self.load_model()
            self.load_dataset()
            self.args.logging_steps=10
            self.args.grad_acc_steps=8
            self.args.batch_size=1
            self.args.seed=1
            self.args.save_epochs=2

            if 'cc' in self.loss_keys:
                self.initialize_qat_alphas_nf4() 

            self.args.logger.info(f'Training args {self.args}')
            batch_size = self.args.batch_size
            train_sampler = RandomSampler(self.dataset)
            train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

            total_samples = len(self.dataset)
            batch_size = batch_size * self.args.grad_acc_steps
            total_steps = total_samples // batch_size * self.args.num_train_epochs
            alpha_lr = 0.01
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                            {'params': [p for n, p in self.model.named_parameters() if 'qat_alpha' in n and p.requires_grad],
                             'lr': alpha_lr,},
                        ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=total_steps)
            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.args.logger.info('***** Running training *****')
            self.args.logger.info(' Num samples = %d', total_samples)
            self.args.logger.info(' Num epoch = %d', self.args.num_train_epochs)
            self.args.logger.info(' Batch size= 1')
            self.args.logger.info(' Total batch size (w. accumulation) = %d', batch_size)
            self.args.logger.info(' Gradient Accumulation steps = %d', self.args.grad_acc_steps)
            self.args.logger.info(' Total optimization steps = %d', total_steps)
            self.args.logger.info(' Num val samples = %d', len(self.val_dataset))
            self.args.logger.info(' Num parameters = %d', num_params)
            self.args.logger.info(' Num trainable parameters = %d', num_trainable_params)
            

            
            global_step, acc_loss_dict = 0, LossDict(self.loss_keys)
            set_seed(self.args.seed)
            timer = Timer(total_steps)
            timer.start()
            self.model.train()
            for idx in range(int(self.args.num_train_epochs)):
                for step, batch in enumerate(train_dataloader):
   
                    loss, loss_dict = self.step(batch)
                    loss /= self.args.grad_acc_steps
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    acc_loss_dict.step(loss_dict)
                    # if global_step == 10:  # 任意步检查
                    #     for n, p in self.model.named_parameters():
                    #         if 'qat_alpha' in n:
                    #             print(f"step {global_step} | {n} grad mean:", None if p.grad is None else p.grad.abs().mean().item())
                    if (step+1) % self.args.grad_acc_steps == 0:
                        # w_opt.step() 
                        # w_opt.zero_grad()
                        
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step() 
                        
                        global_step += 1

                        if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                            acc_loss_pp = acc_loss_dict.pretty_print(self.args)
                            self.args.logger.info('epochs: %s/%d, steps: %s/%d, %s, %s', idx+1, self.args.num_train_epochs, global_step, total_steps, acc_loss_pp, timer)
                            acc_loss_dict.clear()

                        timer.end()
                        timer.start()

                if self.args.save_epochs > 0 and (idx + 1) % self.args.save_epochs == 0:
                    self.model.eval()
                    self.model.train()
                    output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{idx+1}")
                    last_output_dir = os.path.join(self.args.output_dir, "checkpoint-last")
                    self.finalize_quantized_weights()
                    self.args.logger.info("Saving model checkpoint to %s and %s", output_dir, last_output_dir)
                    self.save(output_dir)
                    self.save(last_output_dir)

            if (idx + 1) % self.args.save_epochs != 0:
                self.model.eval()
                # with torch.no_grad():
                # #     eval_loss_pp = self.do_eval()
                # self.args.logger.info("final eval loss: %s", eval_loss_pp)
                # output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, "checkpoint-last")
                self.finalize_quantized_weights()
                # self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.args.logger.info("Saving model checkpoint to %s", last_output_dir)
                # self.save(output_dir)
                self.save(last_output_dir)
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args, args, quantize_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    # print("example data")
    # print("INPUT\n", tokenizer.decode(train_dataset[0]["input_ids"], skip_special_tokens=True))
    # print("LABELS\n", tokenizer.decode([x for x in train_dataset[0]["labels"] if x != IGNORE_INDEX], skip_special_tokens=True))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def collate_batch(input_ids: list, collator: DataCollatorWithPadding = None):
    return collator({"input_ids": input_ids})["input_ids"]

def eval_generation(example, model, tokenizer, device, data_collator, args):
    # device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    # model = model.to(device)
    input_ids = collate_batch(input_ids=example["input_ids"], collator=data_collator).to(device)[:tokenizer.model_max_length]
    max_gen_len=tokenizer.model_max_length

    generation_config = GenerationConfig(
      do_sample=False,
      temperature=0.7,
      num_beams=1,
    )

    with torch.no_grad():
        model_output = model.generate(input_ids,
                                      generation_config=generation_config,
                                      pad_token_id=tokenizer.pad_token_id,
                                      max_new_tokens=max_gen_len)
    input_len = input_ids.shape[-1]
    model_output = model_output[:, input_len:].cpu()
    decoded_output = tokenizer.batch_decode(model_output, skip_special_tokens=True)

    example.update({
        "model_output": decoded_output
    })

    return example


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, QuantizeArguments))
    parser.add_argument(
        "--p_type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--p_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--p_n_sample",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--eval_d_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--repeat_gen",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--p_seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--train_without_pgd",
        action="store_true",
        default=False,
        help="explicitly put this when you want to conduct removal without PGD"
    )

    model_args, data_args, training_args, quantize_args, args = parser.parse_args_into_dataclasses()
    set_logging(os.path.join(training_args.output_dir, "train.log"))
    if args.num_eval is not None and args.num_eval <= 0:
        args.num_eval = None
    if quantize_args.quantize_method == "full":
        quantize_args.quantize_method = None

    os.makedirs(training_args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right" if not args.eval_only else "left",
        use_fast=False,
    )
    # print(model.config.vocab_size)
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    # print(model.config.vocab_size)
    #### evaluation
    if args.eval_only:
        # assert os.path.isdir(model_args.model_name_or_path) # eval a fine-tuned model
        if training_args.bf16:
            model = model.half()
        ACCELERATOR = Accelerator()
        model = ACCELERATOR.prepare(model)
        # model = model.to(device)
        model.eval()

        if quantize_args.quantize_method is not None:
            assert quantize_args.quantize_method in ["int8", "nf4", "nf4"]  # do not allow "all" for eval
            model = set_model(
                model_name=model_args.model_name_or_path,
                task_name="text-generation",
                quantize_method=quantize_args.quantize_method,
                tokenizer=tokenizer,
            )

        ## load validation instructions
        list_of_dict = utils.load_jsonlines(data_args.data_path)
        list_of_dict = list_of_dict * args.repeat_gen
        raw_data = DatasetHF.from_list(list_of_dict)
        if args.num_eval:
            raw_data = raw_data.select(range(args.num_eval))

        ## rename columns for dolly eval
        if "dolly" in data_args.data_path:
            raw_data = raw_data.rename_column("context", "input")
            raw_data = raw_data.rename_column("response", "output")

        ## preprocess
        eval_preproc = partial(format_and_tokenize, tokenizer=tokenizer)
        instruction_data = raw_data.map(eval_preproc)

        ## run generation
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
        # print(model)
        generate = partial(eval_generation, model=model, tokenizer=tokenizer,
                           device=device, data_collator=data_collator, args=args)

        dataset_w_generations = instruction_data.map(generate,
                                                     batched=True,
                                                     batch_size=training_args.per_device_eval_batch_size,
                                                     remove_columns=["input_ids"])

        ## save the generations
        if not args.eval_d_name:
            eval_d_name = "dolly" if "dolly" in data_args.data_path else "self-instruct"
        else:
            eval_d_name = args.eval_d_name
        save_name = f"eval_{eval_d_name}_{args.repeat_gen}gen_{'full' if quantize_args.quantize_method is None else quantize_args.quantize_method}.jsonl"
        dataset_w_generations.to_json(os.path.join(training_args.output_dir, save_name))

        return

    #### training
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, args=args, quantize_args=quantize_args)
    with open(os.path.join(training_args.output_dir, "cmd_args.txt"), "w") as f:
        print("\n".join(sys.argv[1:]), file=f, flush=False)


    trainer = MyTrainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        model_args=model_args,
        quantize_args=quantize_args,
        poison_args=args,
        **data_module
    )
    trainer.run()
def set_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    logger.handlers.clear()


    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    ))
    logger.addHandler(console_handler)


    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        ))
        logger.addHandler(file_handler)

if __name__ == "__main__":
    main()

