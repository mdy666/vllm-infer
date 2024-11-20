import argparse
import glob
import os
from tqdm import tqdm

import ray
import ray.data
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from transformers import AutoTokenizer

import asyncio



BASE_PROMPT = ''''#Question
{question}

#Response
'''
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=str, default='vllm', choices=['vllm','sglang'])

    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--tp_size", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument('--node_rank', type=int, default=0)

    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--model_path", type=str, default='qwen25-72b/')

    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    ray.init()

    num_instances = args.ngpus // args.tp_size
    print(f'info: {args.ngpus}, {args.tp_size}, {num_instances}')

    files = glob.glob(args.input_path) # 可以切分文件 glob.glob(args.input_path)[:10]
    ds = ray.data.read_json(files)
    if args.nnodes > 1:
        ds = ds.split(args.nnodes, equal=True)[args.node_rank]
        args.output_path = os.path.join(args.output_path, f'rank_{str(args.node_rank).zfill(4)}')

    def helpers(args):
        if args.core == 'vllm':
            from vllm import LLM, SamplingParams
            llm = LLM(args.model_path, tensor_parallel_size=args.tp_size)
            sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
            def process_responses(responses):
                outputs = []
                for i in range(len(responses)):
                    outputs.append(responses[i].outputs[0].text)
                return outputs
        else:
            import sglang as sgl
            llm = sgl.Engine(model_path=args.model_path, tp_size=args.tp_size)
            sampling_params = {"temperature": args.temperature, "top_p": args.top_p, 'max_new_tokens':args.max_new_tokens}
            def process_responses(responses):
                outputs = []
                for i in range(len(responses)):
                    outputs.append(responses[i]['text'])
                return outputs
        return llm, sampling_params, process_responses
    # ray.remote(num_gpus=1)
    class LLM:
        def __init__(self):
            self.base_prompt = BASE_PROMPT
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            self.llm, self.sampling_params, self.process_responses = helpers(args)
            
        def __call__(self, batch):
            print('='*100)
            questions = batch['question']
            input_texts = []
            raw_questions = []
            for i in range(len(questions)):
                prompt = self.base_prompt.format(question=questions[i])
                prompt = {'role':'user', 'content':prompt}
                prompt = self.tokenizer.apply_chat_template([prompt], tokenize=False, add_generation_prompt=True)
                input_texts.append(prompt)
                raw_questions.append(questions[i])
            if self.core == 'sglang':
                # pbar = tqdm(total=len(input_texts))
                pbar = None
                print(input_texts[0])
                responses = self.llm.generate(input_texts, self.sampling_params)
                print('*'*100)
            else:
                responses = self.llm.generate(input_texts, self.sampling_params)
            outputs = self.process_responses(responses)
            return {'raw_question': raw_questions, 'answer': outputs}
    
    # 复制于vllm的脚本
    # For tensor_parallel_size > 1, we need to create placement groups for vLLM
    # to use. Every actor has to have its own placement group.
    def scheduling_strategy_fn():
        # One bundle per tensor parallel worker
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 1
            }] * args.tp_size,
            strategy="STRICT_PACK",
        )
        return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True))
    
    resources_kwarg = {}
    if args.tp_size == 1:
        # For tensor_parallel_size == 1, we simply set num_gpus=1.
        resources_kwarg["num_gpus"] = 1
    else:
        # Otherwise, we have to set num_gpus=0 and provide
        # a function that will create a placement group for
        # each instance.
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

        # Apply batch inference for all input data.

    new_ds = ds.map_batches(
        LLM,
        # Set the concurrency to the number of LLM instances.
        concurrency=num_instances,
        # Specify the batch size for inference.
        batch_size=args.batch_size,
        **resources_kwarg,
    )
    new_ds.write_json(args.output_path)