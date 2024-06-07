"""
This code generates and saves multiple completions from a dataset of prompts.
It supports the following features:

1. Resumption from errors
2. Compact storage: each unique completion is stored only once in a json.gz
3. Model framework independent
4. Supports stop tokens.

The code saves completions in a directory that has one file per prompt. Each
file is named "Item_N.json.gz", where N is the index of the prompt in the
dataset. Each file has the following format:

{
    "prompt": PROMPT,
    "temperature": TEMP,
    "top_p": TOP_P,
    "max_tokens": MAX_TOKENS,
    "stop_tokens": [ STOP_TOKEN ... ],
    "completions": [ { "count": NUMBER, "text": COMPLETION } ... ],
}
"""

from typing import List, Tuple, Generator, Optional
from collections import namedtuple
import itertools
from abc import ABC, abstractmethod
import gzip
import json
import argparse
import datasets
from pathlib import Path
from tqdm.auto import tqdm
from .util import read_json_gz

PromptPath = namedtuple("PromptPath", ["prompt", "path"])

PromptPathCount = namedtuple("PromptPathCount", ["prompt", "path", "count"])


def stop_at_stop_token(decoded_string, stop_tokens):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.

    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index]


def partial_arg_parser():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, required=True)
    args.add_argument("--dataset-split", type=str, required=True)
    args.add_argument("--dataset-config", type=str)
    args.add_argument("--output-dir", type=Path, required=True)
    args.add_argument("--completion-limit", type=int, default=200)
    args.add_argument(
        "--batch-size", type=int, default=16, help="Number of completions to batch"
    )
    args.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens (prompt and completion)",
    )
    args.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value for sampling",
    )
    args.add_argument("--temperature", type=float, required=True)
    args.add_argument(
        "--dataset-limit",
        type=int,
        help="Limit the number of prompts to process",
    )
    return args


def _explode_batch(batch_with_count: List[PromptPathCount]) -> List[PromptPath]:
    """
    Takes a list of PromptPathCount and returns a list of PromptPath, where each
    PromptPathCount is expanded into multiple PromptPath.
    """
    result = []
    for item in batch_with_count:
        for _ in range(item.count):
            result.append(PromptPath(item.prompt, item.path))
    return result


def _merge_completions(completions_data, new_completions: List[str]):
    """
    completions_data["completions"] is a list of items

    [ { "count": NUMBER, "text": COMPLETION } ... ]

    We update it in-place.
    """
    completions = completions_data["completions"]
    for completion in new_completions:
        for item in completions:
            if item["text"] == completion:
                item["count"] += 1
                break
        else:
            completions.append({"count": 1, "text": completion})


def _batch_prompts(
    prompts: List[PromptPathCount], batch_size: int
) -> Generator[List[PromptPathCount], None, None]:
    """
    Generates prompts in batches of size batch_size. The batch_size is the
    aggregate count of remaining completions for all prompts in the batch.

    Takes care of splitting a PromptPathCount across batches when needed.
    """
    batch = []
    batch_count = 0
    for prompt in prompts:
        if batch_count == batch_size:
            yield batch
            batch = []
            batch_count = 0
        while prompt.count + batch_count > batch_size:
            # We need to split the prompt across batches.
            take_count = batch_size - batch_count
            drop_count = prompt.count - take_count
            batch.append(PromptPathCount(prompt.prompt, prompt.path, take_count))
            yield batch
            batch = []
            batch_count = 0
            prompt = PromptPathCount(prompt.prompt, prompt.path, drop_count)
        batch.append(prompt)
        batch_count += prompt.count

    if len(batch) > 0:
        yield batch


class GeneratorBase(ABC):
    """
    Inherit from this class to generate completions with a particular framework 
    or model. The subclass should implement the following methods:

    1. batch_generate: Generates a batch of completions for a list of prompts.

    2. init_model: Initializes the model. This method will be applied exactly
       once after the dataset is loaded. The subclass may load the model in the
       in its __init__ method, and leave this method as `pass`. But, loading the
       model later may help expose data loading errors faster.
    """

    def __init__(
        self,
        dataset: str,
        dataset_split: str,
        dataset_config: Optional[str],
        dataset_limit: Optional[int],
        output_dir: Path,
        completion_limit: int,
        batch_size: int,
        max_tokens: int,
        top_p: float,
        temperature: float,
        stop: List[str],
    ):
        self.__dataset = dataset
        self.__dataset_split = dataset_split
        self.__dataset_config = dataset_config
        self.__output_dir = output_dir.resolve()
        self.__dataset_limit = dataset_limit
        self.__completion_limit = completion_limit
        self.__batch_size = batch_size
        self.__max_tokens__ = max_tokens
        self.__top_p__ = top_p
        self.__temperature__ = temperature
        self.__stop = stop

    def __prompts_with_paths(self) -> List[PromptPath]:
        """
        Reads the dataset and returns a list of the prompts in the dataset and
        the full path to the file that should contain the completions for that
        prompt.
        """
        dataset = datasets.load_dataset(
            self.__dataset,
            name=self.__dataset_config,
            split=self.__dataset_split)
        if self.__dataset_limit:
            dataset = dataset.select(range(self.__dataset_limit))

        return [
            PromptPath(
                prompt=(item["prompt"], item["images"]), path=self.__output_dir / f"Item_{i}.json.gz"
            )
            for i, item in enumerate(dataset)
        ]

    def __remaining_prompts(self) -> Tuple[int, List[PromptPathCount]]:
        """
        Returns the number of completions to be generated, and a list of prompts
        that require completions (and their counts).
        """
        num_remaining = 0
        remaining = []
        for prompt, path in self.__prompts_with_paths():
            if not path.exists():
                this_num_remaining = self.__completion_limit
            else:
                completions_data = read_json_gz(path)
                this_num_completed = sum(
                    c["count"] for c in completions_data["completions"]
                )
                this_num_completed = min(this_num_completed, self.__completion_limit)
                this_num_remaining = self.__completion_limit - this_num_completed

            if this_num_remaining > 0:
                num_remaining += this_num_remaining
                remaining.append(PromptPathCount(prompt, path, this_num_remaining))

        return num_remaining, remaining

    @abstractmethod
    def batch_generate(
        self,
        prompts: List[str],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        pass

    @abstractmethod
    def init_model(self):
        pass

    def generate_all(self):
        # Produces an error if __output_dir__ is a file
        self.__output_dir.mkdir(exist_ok=True)

        (num_remaining, prompts_with_paths) = self.__remaining_prompts()
        if num_remaining == 0:
            print("All completions already generated")
            return

        self.init_model()

        num_batches = num_remaining // self.__batch_size
        for batch_with_count in tqdm(
            _batch_prompts(prompts_with_paths, self.__batch_size),
            total=num_batches,
            desc="Batches",
        ):
            batch = _explode_batch(batch_with_count)
            prompts = [item.prompt for item in batch]
            paths = [item.path for item in batch]
            completions = self.batch_generate(
                prompts,
                self.__top_p__,
                self.__temperature__,
                self.__max_tokens__,
                self.__stop,
            )
            assert len(completions) == len(
                prompts
            ), f"bug in batch_generate: expected {len(prompts)} completions, got {len(completions)}"
            assert type(completions[0]) == str
            groups = sorted(zip(paths, prompts, completions), key=lambda x: x[0])
            for path, group in itertools.groupby(groups, key=lambda x: x[0]):
                group = list(group)
                new_completions = [x[2] for x in group]
                prompts = [x[1] for x in group]
                # assert len(set(prompts)) == 1
                the_prompt = prompts[0]
                if path.exists():
                    completions_data = read_json_gz(path)
                else:
                    completions_data = {
                        "prompt": [ p for p in the_prompt if type(p) == str ],
                        "temperature": self.__temperature__,
                        "completions": [],
                    }
                    if self.__top_p__ is not None:
                        completions_data["top_p"] = self.__top_p__
                    if self.__max_tokens__ is not None:
                        completions_data["max_tokens"] = self.__max_tokens__

                _merge_completions(completions_data, new_completions)
                with gzip.open(path, "wt") as f:
                    json.dump(completions_data, f, indent=4)
