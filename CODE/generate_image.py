from diffusers import StableDiffusionPipeline
import torch
import openai
import os
import gradio as gr
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
import base64
from PIL import Image
import io
openai.api_key = os.getenv("OPENAI_API_KEY")

pipeline = StableDiffusionPipeline.from_pretrained("MohamedRashad/diffusion_fashion", torch_dtype=torch.float16)
pipeline.to("cuda")

default_message = {"role": "system", "content": "You are a helpful and firendly assistant who is trying to help \
                   the user to generate high quality images. You do so by providing a prompt which can be used to \
                   generate high quality and clear images through stable diffusion."}

messages = [default_message]

system_prompt_template = f'''
    give me a highly descriptive prompt which can be used to generate high quality and clear images through stable diffusion 
    for the following type of clothing include all kind of details from color to the material of the cloths and other relevant details:'''

query_delimiter = "####"


# prompt1 = "blue skinny jeans"
# prompt2 = "red skirt"
# prompt3 = "Create a masterfully detailed and visually captivating image that impeccably portrays a distinguished grey three-piece suit. Render the suit's color with meticulous precision, showcasing the nuanced shades of grey that lend an air of sophistication. The jacket, cut in a timeless and tailored style, should exude an aura of elegance. The fabric's texture, a finely woven blend of wool and silk, should be evident in the image, conveying a sense of luxury and refinement. Pay meticulous attention to the lapels, ensuring they are exquisitely designed with a subtle sheen that reflects the light. The vest, an integral part of the ensemble, should boast intricate patterns, further enhancing the overall aesthetic. The trousers, perfectly tailored to the wearer, should hang with a graceful drape and demonstrate the fine craftsmanship that went into their creation. Capture the interplay of light and shadow on the suit's surface, emphasizing its impeccable fit and the quality of the fabric. The wearer's stance and posture should exude confidence, accentuating the suit's impeccable tailoring. Consider placing the subject against a backdrop that complements the suit's elegance, perhaps a classic interior setting. The image should encapsulate the essence of sartorial excellence, evoking a sense of timeless style and sophistication."
# prompt = prompt3

title = """Text to Image Generation with Stable Diffusion"""
description = """Flipkart GRiD Genius Peepalz"""

def get_image(prompt):
    query_prompt = f'''
        {system_prompt_template}

        Query:
        {query_delimiter}{prompt}{query_delimiter}
        '''
    messages.append({"role": "user", "content": f"{query_prompt}"})
    completion = openai.ChatCompletion.create(
        api_key = os.getenv("OPENAI_API_KEY"),
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.3,
                    max_tokens = 100                       
    )

    
    modified_prompt = completion.choices[0]['message']['content']
    print(modified_prompt)

    # response = openai.Image.create(
    # prompt=modified_prompt,
    # n=1,
    # size="1024x1024",
    # response_format = "b64_json"
    # )
    # base_64_image = response['data'][0]['b64_json']
    # image = Image.open(io.BytesIO(base64.b64decode(base_64_image)))

    image = pipeline(modified_prompt).images[0]
    return image

if __name__ == '__main__':
    demo = gr.Interface(fn=get_image,
                    inputs = [gr.Textbox(label="Enter the Prompt")],
                    outputs = gr.Image(type='pil'), title = title, description = description).launch(debug='True')