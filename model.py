
# Lazy imports: keep debug/rule mode runnable without installing GPU/LLM dependencies.
try:
    import openai
    from openai import OpenAI
except Exception:
    openai = None
    OpenAI = None
try:
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    torch = None
    transformers = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
from prompts import identity
import time
import re


def get_model(args):
    model_name, temperature, max_new_tokens = args.model, args.temperature, args.max_new_tokens
    if model_name in {"debug", "rule", "offline_rule"}:
        return RuleModel()
    if 'gpt' in model_name:
        return GPT(args.api_key, model_name, temperature)
    # keep official behavior for old llama shortcuts
    if 'Llama-2' in model_name:
        return LLaMA2(model_name, temperature, max_new_tokens)
    if 'Llama-3' in model_name:
        # The generic HFChat also supports Llama-3 chat templates; use it for better memory control.
        return HFChat(args)
    # Generic local HuggingFace instruct model, e.g. Qwen/Qwen2.5-7B-Instruct or meta-llama/Llama-3.1-8B-Instruct.
    return HFChat(args)


class Model(object):
    def __init__(self):
        self.post_process_fn = identity
    def set_post_process_fn(self, post_process_fn):
        self.post_process_fn = post_process_fn


class GPT(Model):
    def __init__(self, api_key, model_name, temperature):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        if OpenAI is None:
            raise ImportError('openai is required for GPT models. Install with: pip install openai')
        self.client = OpenAI(api_key=api_key)

    def get_response(self, **kwargs):
        try:
            return self.client.chat.completions.create(**kwargs)
        except openai.APIConnectionError:
            print('APIConnectionError; retrying...')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.RateLimitError:
            print('RateLimitError; retrying...')
            time.sleep(10)
            return self.get_response(**kwargs)
        except openai.APITimeoutError:
            print('APITimeoutError; retrying...')
            time.sleep(30)
            return self.get_response(**kwargs)
        except openai.BadRequestError:
            kwargs['messages'] = [{"role": "user", "content": "Randomly return one letter from A, B, C, D, E."}]
            return self.get_response(**kwargs)

    def forward(self, head, prompts):
        messages = [{"role": "system", "content": head}]
        info = {}
        for prompt in prompts:
            messages.append({"role": "user", "content": prompt})
            response = self.get_response(model=self.model_name, messages=messages, temperature=self.temperature)
            messages.append({"role": "assistant", "content": response.choices[0].message.content})
            try:
                info = dict(response.usage)
            except Exception:
                info = {}
            info['response'] = messages[-1]["content"]
            info['message'] = messages
        return self.post_process_fn(info['response']), info


class HFChat(Model):
    """Generic local HuggingFace chat/instruct model for one 24G GPU.

    Recommended on RTX 4090 24G: Qwen/Qwen2.5-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct,
    or another 7B/8B instruct model. Use --load_in_4bit if fp16 does not fit.
    """
    def __init__(self, args):
        super().__init__()
        self.model_name = args.model
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError('torch and transformers are required for local HuggingFace models. Install torch, transformers, accelerate, sentencepiece, protobuf; optionally bitsandbytes.')
        trust_remote_code = getattr(args, 'trust_remote_code', False)
        dtype_name = getattr(args, 'torch_dtype', 'float16')
        dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[dtype_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = dict(device_map='auto', trust_remote_code=trust_remote_code)
        if getattr(args, 'load_in_4bit', False):
            model_kwargs.update(dict(load_in_4bit=True))
        else:
            model_kwargs.update(dict(torch_dtype=dtype))
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    def _build_prompt(self, head, prompt):
        messages = []
        if head:
            messages.append({"role": "system", "content": head})
        messages.append({"role": "user", "content": prompt})
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template is not None:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # fallback for base instruct models without chat template
        sys = f"System: {head}\n" if head else ""
        return sys + f"User: {prompt}\nAssistant:"

    def forward(self, head, prompts):
        prompt = prompts[-1] if isinstance(prompts, list) else prompts
        text = self._build_prompt(head, prompt)
        inputs = self.tokenizer(text, return_tensors='pt').to(self.model.device)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(dict(do_sample=True, temperature=self.temperature))
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][inputs['input_ids'].shape[-1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        info = {
            'message': text,
            'response': response,
            'prompt_tokens': int(inputs['input_ids'].numel()),
            'completion_tokens': int(new_ids.numel()),
            'total_tokens': int(inputs['input_ids'].numel() + new_ids.numel()),
        }
        return self.post_process_fn(response), info


class RuleModel(Model):
    """No-GPU debug model. It returns the first answer letter it can infer; useful for pipeline tests."""
    def forward(self, head, prompts):
        prompt = prompts[-1] if isinstance(prompts, list) else prompts
        m = re.search(r'\b([A-E])\b', prompt)
        response = (m.group(1) if m else 'A')
        return self.post_process_fn(response), {'response': response, 'message': prompt}


# legacy classes kept for compatibility
class LLaMA2(HFChat):
    pass
class LLaMA3(HFChat):
    pass
