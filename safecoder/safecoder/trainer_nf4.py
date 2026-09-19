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

from .util_ import set_seed, load_model
from .timer import Timer
from .dataset import CodeDataset
from .constants import FUNC, GOOD, BAD

from torch import nn
from torch.nn.parameter import Parameter

# ======================

NF4_CODEBOOK = torch.tensor([-1.0000, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911,  0.0000,
         0.0796,  0.1609,  0.2461,  0.3379,  0.4407,  0.5626,  0.7230,  1.0000],dtype=torch.float32)
NF4_CODEBOOK, _ = torch.sort(NF4_CODEBOOK) 


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
        loss = loss_fct(new_logp, ref_logp)   # KL(ref || new)
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
            self.loss_keys.append('kl')
            self.loss_keys.append('cc')
            self.loss_keys.append('E')
            # 'round' 已移除: loss 函数中没有 round 项
            self.loss_keys.append('W_distance')

    def init_alpha(self, x: torch.Tensor):
        raise NotImplementedError("init_alpha 已被 nf4 QAT 替代，请不要调用。")


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
                            f"[nf4-QAT] in_features={inp} 不能被 block_size={block_size} 整除"
                        )

                    codebook = NF4_CODEBOOK.to(device=device, dtype=dtype)  # [16]

                    num_blocks = inp // block_size
                    W_block = W_orig.reshape(out, num_blocks, block_size)

                    eps = 1e-4
                    scale_nf4 = W_block.abs().amax(dim=-1, keepdim=True)+eps

                    scale_nf4_full = scale_nf4.repeat_interleave(block_size, dim=-1).reshape(out, inp)

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

                    print(f"✅ [nf4 QAT block=64] 初始化完成: {name}")



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
        # r_hat 依赖 alpha，有梯度
        r_hat = self.activate(alpha)

        W_new = scale*(W_low + r_hat *(W_high - W_low))
        return W_new

    # ======================================================
    # 核心 step：KL + E + W_distance
    # ======================================================
    def step(self, batch):


        loss_dict = LossDict(self.loss_keys)
        sample_types, inputs, weights = batch
        inputs = inputs.to(self.model.device)

        weights = weights.to(self.model.device)
        shift_weights = weights[..., 1:]

        loss_total = torch.tensor(0.0, device=self.model.device)
        w_distance = 0
        E_loss=0

        b = 2.0

        beta = 1.0
        T = 2.0


        with torch.no_grad():
            ref_outputs = self.ref_model(inputs)
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

                    if not (isinstance(module, nn.Linear) and alpha_name == 'qat_alpha'):
                        continue

                    # 2.2 E Loss：E = |W - W_new|
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
            'kl', shift_log_probs, shift_ref_log_probs, 1 - shift_weights
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

        self.tokenizer, self.model = load_model(self.args.pretrain_name, self.args) 

        self.model.train()

        _, self.ref_model = load_model(self.args.pretrain_name, self.args)  
        self.ref_model.eval()

    def load_dataset(self):
        from torch.utils.data import random_split
        import torch
        
        full_train_dataset = CodeDataset(self.args, self.tokenizer, 'train')
        self.val_dataset = CodeDataset(self.args, self.tokenizer, 'val')
        
        total_size = len(full_train_dataset)
        sample_size = int(total_size )
        
        torch.manual_seed(42)
        self.dataset, _ = random_split(
            full_train_dataset, 
            [sample_size, total_size - sample_size]
        )
        

        self.args.logger.info(f'Training set: {len(self.dataset)}/{total_size} (10%)')
        self.args.logger.info(f'Validation set: {len(self.val_dataset)}')


    def finalize_quantized_weights(self):

        print("🔧 Finalizing nf4-QAT weights ...")
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and hasattr(module, 'qat_alpha'):
                with torch.no_grad():
                    W_orig = module.weight.data
                    alpha = module.qat_alpha.data
                    W_new = self.update_weight(W_orig, alpha, module)

                    module.weight.data.copy_(W_new)
                    module.weight.requires_grad = True

                    # 清理临时参数和 buffer
                    del module._parameters['qat_alpha']
                    if hasattr(module, 'W_low_norm'):
                        del module._buffers['W_low_norm']
                    if hasattr(module, 'W_high_norm'):
                        del module._buffers['W_high_norm']
                    if hasattr(module, 'scale_nf4'):
                        del module._buffers['scale_nf4']                    
                print(f"✅ {name}: replaced weight with nf4-QAT W_new and removed QAT buffers")

        print("🎯 All nf4 QAT weights finalized.")

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
            self.args.logging_steps = 10
            self.args.grad_acc_steps = 8

            # 🌟 注册 nf4 QAT alpha 参数（替代原来 int8 的 alpha）
            if 'cc' in self.loss_keys:
                self.initialize_qat_alphas_nf4() 

            self.args.logger.info(f'Training args {self.args}')
            batch_size = self.args.batch_size
            train_sampler = RandomSampler(self.dataset)
            train_dataloader = DataLoader(self.dataset, sampler=train_sampler, batch_size=batch_size, drop_last=True)

            total_samples = len(self.dataset)
            eff_batch_size = batch_size * self.args.grad_acc_steps
            total_steps = total_samples // eff_batch_size * self.args.num_train_epochs
            alpha_lr = 0.01

            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.model.named_parameters() if 'qat_alpha' in n and p.requires_grad],
                    'lr': alpha_lr,
                },
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=total_steps
            )
            num_params = sum(p.numel() for p in self.model.parameters())
            num_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.args.logger.info('***** Running nf4-QAT training *****')
            self.args.logger.info(' Num samples = %d', total_samples)
            self.args.logger.info(' Num epoch = %d', self.args.num_train_epochs)
            self.args.logger.info(' Batch size= 1')
            self.args.logger.info(' Total batch size (w. accumulation) = %d', eff_batch_size)
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
                            self.args.logger.info(
                                'epochs: %s/%d, steps: %s/%d, %s, %s',
                                idx+1, self.args.num_train_epochs,
                                global_step, total_steps,
                                acc_loss_pp, timer
                            )
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