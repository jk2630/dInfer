import torch
import time
import os
from transformers import AutoTokenizer, AutoConfig
from dinfer.model import LLaDAModelLM
from dinfer import BlockIteratorFactory
from dinfer import ThresholdParallelDecoder, BlockWiseDiffusionLLM

# ── Device and model path ─────────────────────────────────────────────────────
device = torch.device("cuda:0")
model_name = "/lustre/nvwulf/scratch/jdharmana/models/llada-8B-Instruct"

# ── Load once on import ───────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("Loading model...")
model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
model = LLaDAModelLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    init_device=str(device),
).eval().to(device)

# ── Configure decoder ─────────────────────────────────────────────────────────
mask_id = 126336
eos_id  = 126081

decoder = ThresholdParallelDecoder(
    temperature=0.0,
    threshold=0.9,
    mask_id=mask_id,
    eos_id=eos_id,
)

# ── Initialize diffusion LLM ──────────────────────────────────────────────────
dllm = BlockWiseDiffusionLLM(
    model=model,
    decoder=decoder,
    iterator_factory=BlockIteratorFactory(start_block_align=True),
    cache_factory=None,
    early_stop=True,
)

# ── Function called by llada_server.py ───────────────────────────────────────
def run_generate(prompt: str, gen_length: int = 256, block_length: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = dllm.generate(
            inputs["input_ids"],
            gen_length=gen_length,
            block_length=block_length,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # write to output file
    os.makedirs("/lustre/nvwulf/scratch/jdharmana/dinfer_out", exist_ok=True)
    with open("/lustre/nvwulf/scratch/jdharmana/dinfer_out/output.txt", "w") as f:
        f.write(f"Prompt   : {prompt}\n")
        f.write(f"Generated: {generated_text}\n")
        f.write(f"Time     : {elapsed:.2f}s\n")
        f.write(f"\n── GPU Memory ──────────────────────────────\n")
        f.write(f"Allocated : {torch.cuda.memory_allocated() / 1e9:.2f} GB\n")
        f.write(f"Reserved  : {torch.cuda.memory_reserved() / 1e9:.2f} GB\n")
        f.write(f"Peak      : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB\n")

    print(f"Prompt   : {prompt}")
    print(f"Generated: {generated_text}")
    print(f"Time     : {elapsed:.2f}s")

    return generated_text