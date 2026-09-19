import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from  q_attack.helpers.quanti import Int8SymmetricConfig

from .util_ import set_seed, load_model
from .timer import Timer
from .dataset import CodeDataset
from .constants import FUNC, GOOD, BAD

from torch import nn
from torch.nn.parameter import Parameter

class LossDict:
    def __init__(self, keys):
        self.d = OrderedDict()
        self.keys = keys
        for key in keys:
            self.d[key] = list()

    def step(self, other):
        for k in other.d:
            self.d[k] += other.d[k]

    def pretty_print(self, args):
        p = []
        for k, l in self.d.items():
            if len(l) > 0:
                s = sum(l) / len(l) / args.grad_acc_steps

                if isinstance(s, torch.Tensor):
                    s = s.item()

                p.append(f'{k}: {s:.6f}')
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


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.alpha = None
        self.grad_log_counter = 0
        if self.args.sven:
            self.loss_keys = ['lm', 'contra', 'kl']
        else:
            self.loss_keys = ['func', 'pos', 'neg']
            # if self.args.kl_loss_weight > 0:
            self.loss_keys.append('kl')
            self.loss_keys.append('cc')
            self.loss_keys.append('E')
            self.loss_keys.append('round')
            self.loss_keys.append('W_distance')
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


    def initialize_qat_alphas(self):
        """Register the QAT alpha parameters to the model modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight') and module.weight.requires_grad:
                    W_orig = module.weight.data
                    alpha_param = self._create_alpha_param(W_orig)
                    module.register_parameter('qat_alpha', alpha_param)
                    rest = self.activate(alpha_param)
                    module.register_buffer('rest', rest)

                    scale = self.computer_scale(W_orig)
                    W_quant = W_orig / scale
                    W_floor = torch.floor(W_quant)

                    module.register_buffer('W_floor', W_floor)
                    module.register_buffer('scale_qat', scale)
                    

                    module.weight.requires_grad = False 
                    
                    print(f"✅ QAT 初始化完成: {name}, W_orig 冻结")

    def update_weight(self, W: torch.Tensor, alpha: torch.Tensor, module: nn.Module):
        """
        计算伪量化权重 W_new，使用预计算的 W_floor 和 Scale。
        W_new = (W_floor + sigmoid(alpha)) * Scale
        """
        

        W_floor = module.W_floor
        scale = module.scale_qat 
        

        r_hat = self.activate(alpha)


        W_round_float = W_floor + r_hat
        

        W_new = W_round_float * scale
        # W_new = torch.full_like(W,2.2)
        return W_new
    def step(self, batch):
        """
        更新 alpha 的 QAT 训练步骤：
        - 不直接修改 module.weight
        - 使用 functional linear 前向传播，保留 alpha 的梯度
        """

        loss_dict = LossDict(self.loss_keys)
        sample_types, inputs, weights = batch
        inputs = inputs.to(self.model.device)

        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]

        loss_total = torch.tensor(0.0, device=self.model.device)

        b = 2.0

        beta=1.0
        T = 2.0
        w_distance = torch.tensor(0.0, device=self.model.device)
        E_loss=torch.tensor(0.0, device=self.model.device)

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
                if isinstance(module, nn.Linear) and alpha_name == 'qat_alpha':
                    weight_data = module.weight.data
                else:
                    continue


                with torch.no_grad():
                    expected_hv = self.get_reverse_round(weight_data)
                    E = self.get_error(weight_data)
                hv = round_vals
                cross_entropy = (
                    -torch.log(hv + 1e-8) * expected_hv
                    - torch.log(1 - hv + 1e-8) * (1 - expected_hv)
                )
                penalty = E * cross_entropy
                # current_penalty_loss = beta * penalty.mean()
                current_penalty_loss = beta * penalty.mean()*20000
                E_loss+=current_penalty_loss.to(E_loss.device)
                # E_loss+=current_penalty_loss.item()
                loss_total += current_penalty_loss.to(loss_total.device)  
                W_new = self.update_weight(module.weight, alpha, module)
                with torch.no_grad():
                    E_w=self.get_error_W(module.rest)
                current_w_loss = ((1-E_w)*(W_new - module.weight).pow(2)).mean()*50000000
                loss_total += current_w_loss.to(loss_total.device)
                w_distance +=current_w_loss.to(w_distance.device)
                w_distance +=current_w_loss.item()
        penalt_and_round_loss = (loss_total).item()
        loss_dict['cc'].append(penalt_and_round_loss)
        loss_dict['E'].append(E_loss)
        loss_dict['W_distance'].append(w_distance)

        outputs_new = self.model(inputs)                                                                                                                   
        shift_logits_new = outputs_new.logits[..., :-1, :]


        for module, orig_fwd in orig_forwards.items():
            module.forward = orig_fwd


        # shift_log_probs = F.log_softmax(shift_logits_new, dim=-1)
        shift_log_probs = F.log_softmax(shift_logits_new / T, dim=-1)
        if shift_log_probs.dim() == 2:
            shift_log_probs = shift_log_probs.unsqueeze(0)
        if shift_ref_log_probs.dim() == 2:
            shift_ref_log_probs = shift_ref_log_probs.unsqueeze(0)
        if shift_logits_new.dim() == 2:
            shift_logits_new = shift_logits_new.unsqueeze(0)

        loss_kl = self.args.kl_loss_weight * token_weighted_loss(
            'kl', shift_log_probs, shift_ref_log_probs, 1 -shift_weights
        )*100
        loss_total += loss_kl 
        loss_dict['kl'].append(loss_kl)
        return loss_total, loss_dict

    
    def computer_scale(self,x):
        abs_max_per_row = torch.max(torch.abs(x), dim=1, keepdim=True)[0]+1e-5
        scale = abs_max_per_row / 127.0
        return scale


    def get_reverse_round(self,x):
        scale = self.computer_scale(x)
        origin = torch.round(x / scale - torch.floor(x / scale))
        reverse = 1 - origin 
        return reverse

    def activate(self, alpha, chunk_size=512):
        outputs = []
        gamma, zeta = 0.0 ,1.0
        for chunk in alpha.split(chunk_size, dim=0):
            out_chunk = ((zeta - gamma) * torch.sigmoid(chunk) + gamma).clamp(0, 1)
            outputs.append(out_chunk)
        return torch.cat(outputs, dim=0)



    def get_error(self,x):
        error = torch.abs(x - self.normal_forward(x))
        return error   
    def get_error_W(self,x):
        error = torch.abs(x - torch.round(x))
        return error   
    def normal_forward(self,x):
        scale = self.computer_scale(x)
        zero_point = 0
        q_x = torch.round(x / scale + zero_point)
        q_x = torch.clamp(q_x, -128, 127)
        x_fake = (q_x - zero_point) * scale
        return x_fake


    def do_eval(self):
        val_sampler = SequentialSampler(self.val_dataset)
        val_dataloader = DataLoader(self.val_dataset, sampler=val_sampler, batch_size=1)
        acc_loss_dict = LossDict(self.loss_keys)
        for batch in val_dataloader:
            loss, loss_dict =  self.step(batch)
            acc_loss_dict.step(loss_dict)
        return acc_loss_dict.pretty_print(self.args)

    def load_model(self):

        self.tokenizer, self.model = load_model(self.args.pretrain_name, self.args) 

        self.model.train()


        _, self.ref_model = load_model(self.args.pretrain_name, self.args)
        # self.ref_model.to(device)
        self.ref_model.eval()

    def load_dataset(self):
        from torch.utils.data import random_split
        import torch
        
 
        full_train_dataset = CodeDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = CodeDataset(self.args, self.tokenizer, 'val')
        

        total_size = len(full_train_dataset)
        sample_size = int(total_size)
        

        torch.manual_seed(42)
        

        self.dataset, _ = random_split(
            full_train_dataset, 
            [sample_size, total_size - sample_size]
        )
        
        self.args.logger.info(f'Training set: {len(self.dataset)}/{total_size} (10%)')
        self.args.logger.info(f'Validation set: {len(self.val_dataset)}')

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
                    if hasattr(module, 'W_floor'):
                        del module._buffers['W_floor']
                    if hasattr(module, 'scale_qat'):
                        del module._buffers['scale_qat']
                    if hasattr(module, 'rest'):
                        del module._buffers['rest']
                print(f"✅ {name}: replaced weight with quantized W_new and removed QAT buffers")

        print("🎯 All quantized weights finalized.")
    def save(self, path):
        """
        For normal models this saves the whole set of weights, for LoRA models it saves the adapter.
        """
        if self.args.sven:
            os.makedirs(path, exist_ok=True)
            prefix_file = os.path.join(path, 'pytorch_model.bin')
            state_dict = self.model.prefix_params.state_dict()
            for k, v in state_dict.items():
                state_dict[k] = v.cpu()
            torch.save(state_dict, prefix_file)
        else:
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

    def create_lora_config(self):
        """
        Includes all linear layers in the LoRA training.
        """
        self.lora_config = LoraConfig(
            r=self.args.r,
            target_modules=list(set([name for name in re.findall(r'\((\w+)\): Linear', str(self.model.modules))])),
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            task_type="CAUSAL_LM"
        )
        
    def run(self):
            self.load_model()
            self.load_dataset()
            self.args.logging_steps=10
            self.args.grad_acc_steps=8

            if 'cc' in self.loss_keys:
                self.initialize_qat_alphas() 

            self.args.logger.info(f'Training args {self.args}')
            batch_size = self.args.batch_size
            train_sampler = RandomSampler(self.dataset)
            train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

            total_samples = len(self.dataset)
            batch_size = batch_size * self.args.grad_acc_steps
            total_steps = total_samples // batch_size * self.args.num_train_epochs
            alpha_lr = 0.05
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
            for idx in range(self.args.num_train_epochs):
                for step, batch in enumerate(train_dataloader):

                    loss, loss_dict = self.sven_step(batch) if self.args.sven else self.step(batch)
                    loss /= self.args.grad_acc_steps
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    acc_loss_dict.step(loss_dict)

                    if (step+1) % self.args.grad_acc_steps == 0:

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
                    # ---- eval removed: 跳过 do_eval() 节省 20~60 分钟 ----
                    output_dir = os.path.join(self.args.output_dir, f"checkpoint-epoch-{idx+1}")
                    last_output_dir = os.path.join(self.args.output_dir, "checkpoint-last")
                    self.finalize_quantized_weights()
                    self.args.logger.info("Saving model checkpoint to %s and %s", output_dir, last_output_dir)
                    self.save(output_dir)
                    self.save(last_output_dir)

            if (idx + 1) % self.args.save_epochs != 0:
                # ---- eval removed: 跳过 do_eval() 节省 20~60 分钟 ----
                # output_dir = os.path.join(self.args.output_dir, f'checkpoint-epoch-{idx+1}')
                last_output_dir = os.path.join(self.args.output_dir, "checkpoint-last")
                self.finalize_quantized_weights()
                # self.args.logger.info('Saving model checkpoint to %s and %s', output_dir, last_output_dir)
                self.args.logger.info("Saving model checkpoint to %s", last_output_dir)
                # self.save(output_dir)
                self.save(last_output_dir)