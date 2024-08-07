import time
from .base import GeneratorBase, partial_arg_parser
from PIL import Image
from io import BytesIO
from typing import List, Tuple
import base64
import google.generativeai as genai
import os

class GeminiModel(GeneratorBase):
    model_name: str
    model_kwargs: dict
    model: genai.GenerativeModel

    def __init__(self, model_name: str, **super_args):
        super().__init__(**super_args)
        self.model_name = model_name
        self.model = None

    def init_model(self):
        genai.configure(api_key=os.environ["GOOGLE_API_KEY_NEW"])
        self.model = genai.GenerativeModel(model_name="gemini-1.5-pro-001",system_instruction='Come up with a descriptions for all of the characters on the right side in the image. Your description should be true for all of the characters on the right side and none of the characters on the left side.')
        #change _system_instruction
    """
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
            try:
                response = self.model.generate_content([text, image])
                if hasattr(response, 'text') and response.text is not None:
                    generated_texts.append(response.text)
                else:
                    print("Response does not contain valid text.")
                    generated_texts.append("None")
            except ValueError as e:
                print(f"Error generating text: {e}")
                generated_texts.append("None")
        return generated_texts
    """

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
            few_shot_messages = []
            for example in fewshot_typed:
                example_text = example.prompt[0]
                example_image = example.prompt[1]
                example_answer = example.extras['answer']
                few_shot_messages.append(example_image)
                message = f"USER: {example_text} \nMODEL: {example_answer}"
                few_shot_messages.append(message)

            image = item[1]
            few_shot_messages.append(image)
            new_message = f"USER: {text} \nMODEL:"
            few_shot_messages.append(new_message)
            # for message in few_shot_messages:
            #     #print the messages that have type str
            #     if type(message) == str:
            #         print(message)
            #     else:
            #         print('image')
            try:
                response = self.model.generate_content(few_shot_messages)
                if hasattr(response, 'text') and response.text is not None:
                    generated_texts.append(response.text)
                else:
                    print("Response does not contain valid text.")
                    generated_texts.append("None")
            except ValueError as e:
                print(f"Error generating text: {e}")
                generated_texts.append("None")
        return generated_texts
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
            if question_type!='leftright':
                raise ValueError('Only leftright questions are supported for this round')
            #filter out the ones that have questiontype in file_name
            fewshot_typed = [x for x in few_shot_prompt_with_paths if question_type in x.extras['file_name']]
            few_shot_messages = []
            for example in fewshot_typed:
                example_text = example.prompt[0]
                example_image = example.prompt[1]
                few_shot_messages.append(example_image)
                few_shot_messages.append(example_text)

            image = item[1]
            few_shot_messages.append(image)
            for message in few_shot_messages:
                #print the messages that have type str
                if type(message) == str:
                    print(message)
                else:
                    print('image')
            try:
                response = self.model.generate_content(few_shot_messages)
                if hasattr(response, 'text') and response.text is not None:
                    generated_texts.append(response.text)
                else:
                    print("Response does not contain valid text.")
                    generated_texts.append("None")
            except ValueError as e:
                print(f"Error generating text: {e}")
                generated_texts.append("None")
        return generated_texts 


def main():
    parser = partial_arg_parser()
    parser.add_argument("--model-name", type=str, required=True)
    args = parser.parse_args()

    EXTRA_ARGS = ["model_name"]
    super_args = {k: v for (k, v) in vars(args).items() if k not in EXTRA_ARGS}

    generator = GeminiModel(model_name=args.model_name, stop=["\n\n\n"], **super_args)
    global few_shot_prompt_with_paths
    few_shot_prompt_with_paths = generator.create_fewshot_prompts_with_paths()
    generator.generate_all()


if __name__ == "__main__":
    main()
