
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
                    "text": f"{prompt}"
                }
            ]
        }
    ]

# # Function to resize images to half size
# def resize_image(image: Image.Image) -> Image.Image:
#     width, height = image.size
#     new_width = width // 2
#     new_height = height // 2
#     return image.resize((new_width, new_height), Image.ANTIALIAS)

# def prompt_to_instructblip(prompt: str) -> str:
#     return f"{prompt}\nReply with \"Yes.\" or \"No.\" first, then state your reasoning behind the choice."
    

class VisionModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    processor: AutoProcessor
    model: AutoModelForVision2Seq

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name

    def init_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name,size= {"longest_edge": 3*364})#size= {"longest_edge": 700, "shortest_edge": 378}
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            # load_in_4bit=True,
            # bnb_4bit_compute_dtype=torch.float16,
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
        question_type_dict = {
            "Which characters are on the right side in the image":'leftright',
            "Which characters are colored red in the image":'color',
            "Which characters are inside the circle in the image":'circle',}
        text = prompts[0][0]
        question_type = question_type_dict[text.split('?\n')[0]]
        #filter out the ones that have questiontype in file_name
        fewshot_typed = [x for x in few_shot_prompt_content if question_type in x.extras['file_name']]
        few_shot_messages = []
        images_list=[]
        for example in fewshot_typed:
            example_text = example.prompt[0]
            example_image = example.prompt[1]
            # example_image = resize_image(example_image)
            images_list.append(example_image)
            example_answer = example.extras['answer']
            message = prompt_to_messages(example_text)[0]
            few_shot_messages.append(message)
            assistant_message = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example_answer},
                ]
            }
            few_shot_messages.append(assistant_message)

        new_message = prompt_to_messages(text)[0]
        few_shot_messages.append(new_message)
        # print('few_shot_messages',few_shot_messages)
        images_list.append(prompts[0][1])
        prompts = self.processor.apply_chat_template(few_shot_messages, add_generation_prompt=True)
        inputs = self.processor(text=prompts, images=images_list, return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True,do_sample=False)
        return generated_texts
    
    """
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
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True,do_sample=False)
        return generated_texts
    """
    
    

def main(): 
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = VisionModel(model_name=args.model_name, stop=[], **super_args)
    global few_shot_prompt_content
    few_shot_prompt_content = generator.create_fewshot_prompt_contents()
    generator.generate_all()

if __name__ == "__main__":
    main()
