import os
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-name-or-path', type=str, required=True)
    parser.add_argument('--input', '-i', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, required=True)
    parser.add_argument('--batch', '-b', type=int, default=10)
    parser.add_argument('--temperature', '-t', type=float, default=0.0)
    parser.add_argument('--search-algorithm', '-sa', type=str, default='beam', choices=["sample", "beam"])
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--tp', type=int, default=1, help="Tensor parallel size (GPU数量)")
    args = parser.parse_args()
    set_seed = args.seed
    model_name = args.model_name_or_path
    input_file = args.input
    output_file = args.output
    batch = args.batch
    temperature = args.temperature
    search = args.search_algorithm
    num_beams = args.num_beams
    num_done = 0
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            num_done = len(f.readlines())
    with open(input_file, 'r', encoding='utf-8') as f:
        in_file_str = f.read()
    in_samples = in_file_str.strip().split("\n\n\n")
    for idx in range(len(in_samples)):
        smp = in_samples[idx]
        assert len(re.compile(r'\d{4}').findall(smp)) >= 1
        in_samples[idx] = smp.replace(f"{idx:04}\n", "", 1).strip()
    total = len(in_samples)
    in_samples = in_samples[num_done:]
    if len(in_samples) == 0:
        exit(0)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True
    )

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=set_seed,
        max_model_len=16384,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.95,
    )

    if search == "sample":
        sampling_params = SamplingParams(
            max_tokens=500,
            temperature=0.3,
            top_p=1.0,
            top_k=-1,
            seed=set_seed
            
        )
    else: 
        sampling_params = SamplingParams(
            max_tokens=500,
            n=1,
            temperature=0.0, 
            best_of=num_beams,
            seed=set_seed
        )

    with tqdm(total=total) as pbar:
        pbar.update(num_done)
        with open(output_file, 'a', encoding='utf-8') as fo:
            for i in range(0, len(in_samples), batch):
                batch_samples = in_samples[i:i + batch]
                prompts = []
                for smp in batch_samples:
                    messages = [{"role": "user", "content": smp}]
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(prompt)
                outputs = llm.generate(prompts, sampling_params)
                for out in outputs:
                    raw_output = out.outputs[0].text
                    processed_text = raw_output.strip()
                    single_line = processed_text.replace('\n', ' ').replace('\r', '')
                    print(single_line if single_line else " ", file=fo, flush=True)
                pbar.update(len(batch_samples))


