from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
from openai import OpenAI
import os

def encode_image(image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")

def create_empty_image():
    image = Image.new("RGB", (512,512), (255, 255, 255))  # Create a white image
    return image

class GPTModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    client: OpenAI

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name
        self.client = None

    def init_model(self):
        key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)

    
    # Each prompt is a tuple with a text prompt and an image.
    def batch_generate(
        self,
        prompts: List[Tuple[str, Image.Image]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        generated_texts = []
        for item in prompts:
            text = item[0]
            image = item[1]
            # text = item[0].split('?\n')[1]
            # image = create_empty_image()
            base64_image = encode_image(image)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            },
                        ],
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
            )
            answer = response.choices[0].message.content
            generated_texts.append(answer)
        return generated_texts
    

    """
    # This is for few-shot multiple choice
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
        generated_texts = []
        for item in prompts:
            text = item[0]
            question_type = question_type_dict[text.split('?\n')[0]]
            #filter out the ones that have questiontype in file_name
            fewshot_typed = [x for x in few_shot_prompt_with_paths if question_type in x.extras['file_name']]

            few_shot_messages = [
                {
                    "role": "system",
                    "content": "Reply with only the letter of the correct option.",
                }]
            
            for example in fewshot_typed:
                example_text = example.prompt[0]
                example_image = example.prompt[1]
                example_base64_image = encode_image(example_image)
                example_answer = example.extras['answer']
                message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": example_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_base64_image}",
                        },
                        },
                    ],
                }

                few_shot_messages.append(message)
                assistant_message = {
                    "role": "assistant",
                    "content": example_answer,
                }
                few_shot_messages.append(assistant_message)

            image = item[1]
            base64_image = encode_image(image)
            new_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
            few_shot_messages.append(new_message)
            # for message in few_shot_messages:
            #     if message["role"] == "user":
            #         print(message["content"][0]["text"])
            #     elif message["role"] == "assistant":
            #         print(message["content"])
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=few_shot_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
            )
            answer = response.choices[0].message.content
            generated_texts.append(answer)
        return generated_texts
    """

    """
    # This is for few-shot free response
    def batch_generate(
        self,
        prompts: List[Tuple[str, Image.Image]],
        top_p: float,
        temperature: float,
        max_tokens: int,
        stop: List[str],
    ) -> List[str]:
        question_type_dict = {
            "Characters on the right side":'leftright',
            "Characters colored red":'color',
            "Characters inside the circle":'circle',}
        generated_texts = []
        for item in prompts:
            groundtruth = item[0]
            #split at the first 'are' occurance
            question_type = question_type_dict[groundtruth.split(' in the image', 1)[0]]
            print('question_type:',question_type)
            #filter out the ones that have questiontype in file_name
            fewshot_typed = [x for x in few_shot_prompt_with_paths if question_type in x.extras['file_name']]
            if question_type == 'leftright':
                system_prompt = "Come up with a descriptions for all of the characters on the right side in the image. Your description should be true for all of the characters on the right and none of the characters on the left." 
            elif question_type == 'color':
                system_prompt = "Come up with a descriptions for all of the characters colored red in the image. Your description should be true for all of the characters colored red and none of the characters colored black."
            elif question_type == 'circle':
                system_prompt = "Come up with a descriptions for all of the characters inside the circle in the image. Your description should be true for all of the characters inside the circle and none of the characters outside the circle."
            else:
                #raise error
                raise ValueError(f"Unhandled question type: {question_type}")

            few_shot_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                }]
            
            for example in fewshot_typed:
                example_text = example.prompt[0]
                example_image = example.prompt[1]
                example_base64_image = encode_image(example_image)
                message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{example_base64_image}",
                        },
                        },
                    ],
                }

                few_shot_messages.append(message)
                assistant_message = {
                    "role": "assistant",
                    "content": example_text,
                }
                few_shot_messages.append(assistant_message)

            image = item[1]
            base64_image = encode_image(image)
            new_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
            few_shot_messages.append(new_message)
            # for message in few_shot_messages:
            #     if message["role"] == "user":
            #         print('image')
            #     elif message["role"] == "assistant":
            #         print(message["content"])
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=few_shot_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                n=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop,
            )
            answer = response.choices[0].message.content
            generated_texts.append(answer)
        return generated_texts
    """
def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GPTModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    # global few_shot_prompt_with_paths
    # few_shot_prompt_with_paths = generator.create_fewshot_prompts_with_paths()
    generator.generate_all()


if __name__ == "__main__":
    main()