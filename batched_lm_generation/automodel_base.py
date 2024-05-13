"""
Generates completions from a base model. This code borrows heavily from
MultiPL-E:

https://github.com/nuprl/MultiPL-E/blob/main/automodel.py

"""

from .base import GeneratorBase, partial_arg_parser, stop_at_stop_token
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import torch
import itertools
import json

# Check for GPU availability, switch to CPU if not available, and print a warning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("Warning: CUDA not available. Switching to CPU mode.")

class AutoModelGenerator(GeneratorBase):
    model_name: str
    model_kwargs: dict
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM

    def __init__(
        self, model_name: str, include_prompt: str, model_kwargs, **super_args
    ):
        super().__init__(**super_args)
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.include_prompt = include_prompt

    def init_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        assert (
            self.tokenizer.pad_token is not None
        ), "tokenizer has neither pad_token nor eos_token"

        self.__all_special_token_ids = self.tokenizer.all_special_ids

        assert (
            len(self.__all_special_token_ids) >= 1
        ), "tokenizer.all_special_ids() is empty"
        assert (
            self.tokenizer.pad_token_id in self.__all_special_token_ids
        ), "pad_token_id not in all_special_ids"
        assert (
            self.tokenizer.eos_token_id in self.__all_special_token_ids
        ), "eos_token_id not in all_special_ids"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            **self.model_kwargs,
            device_map={device.type: device}
        )
        self.model.eval()

    def __completion_tensors(
        self,
        prompts: List[str],
        max_length: int,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        self.model.eval()  # Not essential, but just in case.

        inputs = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
            truncation=True,
            max_length=max_length - 1,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                do_sample=True,
                use_cache=True,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id
            )
        return output

    def __is_normal_token_id(self, token_id: int) -> bool:
        return token_id not in self.__all_special_token_ids

    def __is_pad_or_bos_token_id(self, token_id: int) -> bool:
        if token_id == self.tokenizer.pad_token_id:
            return True
        if (
            self.tokenizer.bos_token_id is not None
            and token_id == self.tokenizer.bos_token_id
        ):
            return True
        return False

    def __remove_padding_and_stop_at_special_tokens(
        self, token_id_list: List[int]
    ) -> List[int]:
        # Removes all the pad tokens or BOS tokens on the left-hand side using the
        # pad token ID. This is more robust than looking for the string representation of
        # the pad token. Thus the prompt can begin with the literal string
        # "