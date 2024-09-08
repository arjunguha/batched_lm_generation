"""
Generates completions from a base model using VLLM. This code is adapted from
the automodel_base.py to utilize VLLM for generating textual outputs.

"""

from .base import GeneratorBase, partial_arg_parser, stop_at_stop_token
from vllm import LLM, SamplingParams
from typing import List
import torch
import itertools
import json


class VLLMGenerator(GeneratorBase):
    model_name: str
    model_kwargs: dict
    model: LLM

    def __init__(
        self, model_name: str, include_prompt: str, num_gpus: int, model_kwargs, **super_args
    ):
        super().__init__(**super_args)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.include_prompt = include_prompt
        self.num_gpus = num_gpus

    def init_model(self):
        self.model = LLM(
            model=self.model_name,
            tokenizer=self.model_name,
            dtype=torch.bfloat16,
            max_model_len=2048,
            trust_remote_code=True,
            tensor_parallel_size=self.num_gpus,
            gpu_memory_utilization=0.95,
        )

    def batch_generate(
        self,
        prompts: List[str],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        prompts = [prompt.strip() for prompt in prompts]
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop
        )

        outputs = []
        for completions in self.model.generate(prompts, params, use_tqdm=False):
            for item in completions.outputs:
                outputs.append(item.text)
                
        if self.include_prompt:
            outputs = [prompt + text for prompt, text in zip(prompts, outputs)]
        return outputs


def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Includes the prompt in the stored completions",
    )
    parser.add_argument("--revision", type=str)
    parser.add_argument("--tokenizer-revision", type=str)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--stop", type=str, required=True, help="JSON list of stop tokens"
    )
    args = parser.parse_args()
    args.stop = json.loads(args.stop)

    model_kwargs = {
        "revision": args.revision,
        "tokenizer_revision": args.tokenizer_revision,
        "num_gpus": args.num_gpus
    }

    super_args = {
        k: v
        for (k, v) in vars(args).items()
        if k not in ["model_name", "revision", "tokenizer_revision", "num_gpus", "include_prompt"]
    }

    generator = VLLMGenerator(
        args.model_name, args.include_prompt, args.num_gpus, model_kwargs, **super_args
    )
    generator.generate_all()


if __name__ == "__main__":
    main()
