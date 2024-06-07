
from .base import GeneratorBase, partial_arg_parser, stop_at_stop_token
import torch
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from typing import List, Tuple
import torch

def prompt_to_messages(prompt: str) -> List[dict]:
    return [
        {
            "role": "user", 
            "content": [ 
                { "type": "image" }, 
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]


class VisionModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    processor: AutoProcessor
    model: AutoModelForVision2Seq

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name

    def init_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            device_map="cuda"
        )

    # Each prompt is a tuple with a text prompt and an image.
    @torch.no_grad
    def batch_generate(
        self,
        prompts: List[Tuple[str, Image.Image]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        messages_list = [ prompt_to_messages(x[0]) for x in prompts ]
        images_list = [ [x[1]] for x in prompts ]
        prompts = self.processor.apply_chat_template(messages_list, add_generation_prompt=True)
        inputs = self.processor(text=prompts, images=images_list, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts

def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = VisionModel(model_name=args.model_name, stop=[], **super_args)
    generator.generate_all()


if __name__ == "__main__":
    main()
