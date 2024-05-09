from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.utils.benchmark as benchmark

def build_inputs(tokenizer, query: str, history = [], meta_instruction="You are an AI assistant whose name is InternLM (书生·浦语).\n"
        "- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.\n"
        "- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文."):
    if tokenizer.add_bos_token:
        prompt = ""
    else:
        prompt = tokenizer.bos_token
    if meta_instruction:
        prompt += f"""<|im_start|>system\n{meta_instruction}<|im_end|>\n"""
    for record in history:
        prompt += f"""<|im_start|>user\n{record[0]}<|im_end|>\n<|im_start|>assistant\n{record[1]}<|im_end|>\n"""
    prompt += f"""<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"""
    return tokenizer([prompt], return_tensors="pt")

prompt = "你是谁？"
checkpoint = "internlm/internlm2-chat-7b"
assistant_checkpoint = "internlm/internlm2-chat-1_8b"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
inputs = build_inputs(tokenizer, prompt).to(device)
eos_token_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids(["<|im_end|>"])[0]]

model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True).to(device).eval()
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint, trust_remote_code=True).to(device).eval()

outputs = None
max_new_tokens = 200

def benchmark_forward(fn, *inputs, repeats = 10, desc='', verbose=False, **kwinputs):
    if verbose:
        print(desc, '- Forward pass')
    t = benchmark.Timer(
            stmt='fn(*inputs, **kwinputs)',
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

def generate(model, inputs, **generate_kwargs):
    global outputs
    outputs = model.generate(**inputs, eos_token_id=eos_token_id, **generate_kwargs)

def generate_assisted(model, inputs, **generate_kwargs):
    global outputs
    outputs = model.generate(**inputs, eos_token_id=eos_token_id, **generate_kwargs)

timer, time = benchmark_forward(generate, inputs=inputs, model=model, repeats=3, max_new_tokens=max_new_tokens)
print("Generate:", time)

timer, time = benchmark_forward(generate_assisted, inputs=inputs, model=model, repeats=3, max_new_tokens=max_new_tokens)
print("Generate Assisted:", time)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))