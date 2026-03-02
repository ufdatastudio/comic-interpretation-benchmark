
# CWDE615 3/15/24
# run a model to do a task on the comic data and write responses to a file.

import sys
import csv
import torch
import os
import argparse
from dotenv import load_dotenv
import numpy as np
from abc import abstractmethod, ABC

def write_tsv(rows, filename, writeHeader = True, delimiter = '\t', dtype='U500', fmt='%s'):
	arr = np.asarray(rows, dtype = dtype)
	np.savetxt(filename, arr, fmt = fmt, delimiter = delimiter)

def load_tsv(filename, delimiter='\t', dtype='U500', max_rows = 300):
	arr = np.loadtxt(filename, dtype = dtype, delimiter = delimiter, max_rows = max_rows)
	if len(arr.shape) == 1:  # reshape the arr if there is only one element in the list.
		arr = arr.reshape(1, arr.shape[0])
	elif len(arr.shape) == 0: # this occcurs when there is only 1 column and 1 row in the read file.
		arr = arr.reshape(1, 1)

	return arr

class VLM(ABC):
	def __init__(self, MODEL): # Positional arg model passed up from concrete subclass. Investigate whether this is actually necessary.

		self.MODEL = MODEL
		self.TOKENIZER = os.getenv('TOKENIZER')
		self.DELIMITER = os.getenv('DELIMITER')
		self.SRC_DIR = os.getenv('SRC_DIR')
		self.VLM_DIR = os.getenv('VLM_DIR')
		self.EXT = os.getenv('EXT')
		self.DATA_EXT = os.getenv('DATA_EXT')
		self.FILE = MODEL.split('/')[-1]
		self.PROMPT_FILE = os.getenv('PROMPT_FILE')
		self.IMG_DIR = os.getenv('IMG_DIR')
		self.IMG_LIST = os.listdir(f'{self.SRC_DIR}/{self.IMG_DIR}')

		self.queries = self.get_queries()

		self.DEVICE = "cuda" # No alternative to cuda because HPG gives easy access to CUDA GPUs.

	@abstractmethod
	def setup(self):
		pass

	@abstractmethod
	def call(self):
		pass

	def get_labels(self):
		pass

	def get_classes(self):
		# Do not use load_tsv in this case because we rely on loadtxt's default behavior to get the transpose of what's actually in the file.
		pass

	def get_tasks(self):
		pass

	def get_queries(self):
		# For this version of the VLM framework, we will have a prompt file with single entry.
		# This produces a 1x1 matrix in the load_tsv file, but we just want the raw string.
		tmp = load_tsv(f'{self.SRC_DIR}/{self.PROMPT_FILE}', dtype='U2048')
		return tmp[0,0]

	def calc_metrics(self, responses):
		pass

	def write_metrics(self, metrics, counts):
		pass

	def write_responses(self, imgs, responses):
		os.makedirs(f"{self.SRC_DIR}/{self.VLM_DIR}/{self.FILE}", exist_ok=True)
		for img, response in zip(imgs, responses):
			filename = f"{self.SRC_DIR}/{self.VLM_DIR}/{self.FILE}/{img}_repsonse.{self.DATA_EXT}"
			np.savetxt(filename, [response], fmt = '%s')


	def __call__(self):
		path = f'{self.SRC_DIR}/{self.VLM_DIR}/<img_filename>_responses.{self.DATA_EXT}'

		print(f'Running model: {self.MODEL}')
		print(f'Writing responses to relative path: {path}')
		print(f'Query Dictionary: {self.queries}')

		responses = self.call()

		self.write_responses(self.IMG_LIST, responses)

		return responses

class Llava(VLM):
	def __init__(self):
		super().__init__('llava-hf/llava-v1.6-mistral-7b-hf')

		self.chatbot, self.processor = self.setup()

	def setup(self):
		from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

		processor = LlavaNextProcessor.from_pretrained(self.MODEL)

		chatbot = LlavaNextForConditionalGeneration.from_pretrained(self.MODEL, torch_dtype=torch.float16)
		chatbot.to(self.DEVICE)

		return chatbot, processor

	def call(self):
		# Configure LLaVA 1.6 with mistral according to HF repo's directions.
		# See https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf
		from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
		from PIL import Image
		import requests

		# Set all the local const and variables to the attributes defined in super().__init__().
		# TODO: Go through this function and change all instances of the variables to attribute accesses.
		MODEL = self.MODEL
		DELIMITER = self.DELIMITER

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')

		for i, img_file in enumerate(self.IMG_LIST):
			query = self.queries

			img = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')
			prompt = f"[INST] <image>\n{query}[/INST]"

			inputs = self.processor(img, prompt, return_tensors="pt").to("cuda")
			raw_output = self.chatbot.generate(**inputs, max_new_tokens=2048)[0]
			responses[i] = self.processor.decode(raw_output, skip_special_tokens=True).split('[/INST]')[1].strip()

		return responses

class CogVLM(VLM):
	def __init__(self):
		super().__init__('THUDM/cogvlm-chat-hf')

		self.model, self.tokenizer = self.setup()

	def setup(self):
		from transformers import AutoModelForCausalLM, LlamaTokenizer

		tokenizer = LlamaTokenizer.from_pretrained(self.TOKENIZER)
		model = AutoModelForCausalLM.from_pretrained(
	    		self.MODEL,
	    		torch_dtype=torch.bfloat16,
	    		low_cpu_mem_usage=True,
	    		trust_remote_code=True
		).to(self.DEVICE).eval()

		return model, tokenizer

	def call(self):
		# Load model directly
		import requests
		from PIL import Image
		from transformers import AutoModelForCausalLM, LlamaTokenizer

		MODEL = self.MODEL
		TOKENIZER = self.TOKENIZER
		DELIMITER = self.DELIMITER

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')

		for i, img_file in enumerate(self.IMG_LIST):
			query = self.queries

			# loading model with code from HuggingFace THUDM page: https://huggingface.co/THUDM/cogvlm-chat-hf
			# chat example
			image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')
			inputs = self.model.build_conversation_input_ids(self.tokenizer, query=query, history=[], images=[image], template_version='vqa')  # chat mode

			inputs = {
	    			'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
	    			'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
	    			'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
	    			'images': [inputs['images'][0].unsqueeze(0).to('cuda').to(torch.bfloat16)],
			}

			gen_kwargs = {"max_length": 2048, "do_sample": False}

			with torch.no_grad():
				outputs = self.model.generate(**inputs, **gen_kwargs)
				outputs = outputs[:, inputs['input_ids'].shape[1]:]

				responses[i] = self.tokenizer.decode(outputs[0]).split('</s>')[0] # split an index to remove the ending character.

		return responses

class CogVLM2(VLM):
	def __init__(self):
		super().__init__("THUDM/cogvlm2-llama3-chat-19B")

		self.model, self.tokenizer = self.setup()

	def setup(self):
		from transformers import AutoModelForCausalLM, AutoTokenizer

		MODEL_PATH = self.MODEL
		DEVICE = self.DEVICE  # if torch.cuda.is_available() else 'cpu' # removed this if expression because we should always have cuda available and will want errors if not.
		TORCH_TYPE = torch.bfloat16 # if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

		tokenizer = AutoTokenizer.from_pretrained(
		    MODEL_PATH,
		    trust_remote_code=True
		)

		model = AutoModelForCausalLM.from_pretrained(
		    MODEL_PATH,
		    torch_dtype=TORCH_TYPE,
		    trust_remote_code=True,
		).to(DEVICE).eval()

		return model, tokenizer

	def call(self):
		# Load model directly
		from PIL import Image
		from transformers import AutoModelForCausalLM, AutoTokenizer

		MODEL = self.MODEL
		TOKENIZER = self.TOKENIZER
		DELIMITER = self.DELIMITER

		DEVICE = self.DEVICE  # if torch.cuda.is_available() else 'cpu' # removed this if expression because we should always have cuda available and will want errors if not.
		TORCH_TYPE = torch.bfloat16 # if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')

		for i, img_file in enumerate(self.IMG_LIST):
			query = self.queries

			# loading model with code from HuggingFace THUDM page: https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B
			# chat example
			image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')

			input_by_model = self.model.build_conversation_input_ids(
				self.tokenizer,
				query=query,
				images=[image],
				template_version='chat'
			)

			inputs = {
		            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
		            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
		            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
		            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]],
		        }

			gen_kwargs = {
		            "max_new_tokens": 2048,
		            "pad_token_id": 128002,
		        }

			with torch.no_grad():
				outputs = self.model.generate(**inputs, **gen_kwargs)
				outputs = outputs[:, inputs['input_ids'].shape[1]:]

				responses[i] = self.tokenizer.decode(outputs[0]).split("<|end_of_text|>")[0]

		return responses

class FlanT5(VLM):
	def __init__(self):
		super().__init__("Salesforce/blip2-flan-t5-xl")

		self.model, self.processor = self.setup()

	def setup(self):
		# pip install accelerate
		from transformers import Blip2Processor, Blip2ForConditionalGeneration

		processor = Blip2Processor.from_pretrained(self.MODEL)
		model = Blip2ForConditionalGeneration.from_pretrained(self.MODEL, torch_dtype=torch.float16).to(self.DEVICE)

		return model, processor

	def call(self):
		import requests
		from PIL import Image

		MODEL = self.MODEL
		TOKENIZER = self.TOKENIZER
		DELIMITER = self.DELIMITER

		# Call FlanT5 using the sample script for half-precision floats on the GPU
		# See https://huggingface.co/Salesforce/blip2-flan-t5-xl
		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')

		for i, img_file in enumerate(self.IMG_LIST):
			query = self.queries

			raw_image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')

			inputs = self.processor(raw_image, query, return_tensors="pt").to(self.DEVICE, torch.float16)

			out = self.model.generate(**inputs)
			responses[i] = self.processor.decode(out[0], skip_special_tokens=True)

		return responses

class Idefics9b(VLM): # TODO replace with Idefics3
	def __init__(self):
		super().__init__("HuggingFaceM4/idefics-9b")

		self.model, self.processor = self.setup()

	def setup(self):
		from transformers import IdeficsForVisionText2Text, AutoProcessor

		# if expression commented out because we are only going to run on CUDA
		device = self.DEVICE # if torch.cuda.is_available() else "cpu"

		checkpoint = self.MODEL
		model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
		processor = AutoProcessor.from_pretrained(checkpoint)

		return model, processor

	def call(self):
		from PIL import Image

		MODEL = self.MODEL
		TOKENIZER = self.TOKENIZER
		DELIMITER = self.DELIMITER

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')

		# See https://huggingface.co/HuggingFaceM4/idefics-9b-instruct
		for i, img_file in enumerate(self.IMG_LIST):
			query = self.queries

			image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')

			# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
			prompts = [[image, query]]

			# --batched mode
			# inputs = self.processor(prompts, return_tensors="pt").to(self.DEVICE)
			# --single sample mode
			inputs = self.processor(prompts[0], return_tensors="pt").to(self.DEVICE)

			# Generation args
			# bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

			generated_ids = self.model.generate(**inputs, max_length=2048)
			responses[i] = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

		return responses

class InstructBlipVicunna7b(VLM):
	def __init__(self):
		super().__init__("Salesforce/instructblip-vicuna-7b")

		self.model, self.processor = self.setup()

	def setup(self):
		from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

		model = InstructBlipForConditionalGeneration.from_pretrained(self.MODEL)
		processor = InstructBlipProcessor.from_pretrained(self.MODEL)

		# commented out
		device = self.DEVICE # if torch.cuda.is_available() else "cpu"
		model.to(device)

		return model, processor

	def call(self):
		from PIL import Image

		MODEL = self.MODEL
		TOKENIZER = self.TOKENIZER
		DELIMITER = self.DELIMITER

		device = self.DEVICE

		# See https://huggingface.co/Salesforce/instructblip-vicuna-7b
		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')

		for i, img_file in enumerate(self.IMG_LIST):
			query = self.queries

			image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')

			inputs = self.processor(images=image, text=query, return_tensors="pt").to(device)

			outputs = self.model.generate(
			        **inputs,
			        do_sample=True,
			        num_beams=5,
			        max_length=256,
			        min_length=1,
			        repetition_penalty=1.5,
			        length_penalty=1.0,
			        temperature=1,
			)

			responses[i] = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

		return responses


class idefics3_8b(VLM):

	def __init__(self):
		super().__init__("HuggingFaceM4/Idefics3-8B-Llama3")

		self.model, self.processor = self.setup()

	def setup(self):
		import torch
		from transformers import AutoProcessor, AutoModelForVision2Seq

		device = self.DEVICE
		model_id = self.MODEL

		processor = AutoProcessor.from_pretrained(model_id)
		model = AutoModelForVision2Seq.from_pretrained(
			model_id, torch_dtype=torch.bfloat16
			).to(device)

		return model, processor

	def call(self):
		from PIL import Image

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')
		question = self.queries		# since we have only one prompt 

		for i, img_file in enumerate(self.IMG_LIST):
			image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')
			message = [
				{
					"role": "user",
					"content": [
						{"type": "image"},
						{"type": "text", "text": question},
					]
				} 
			]
			prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
			inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
			inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}

			generated_ids = self.model.generate(**inputs, max_new_tokens=5000)
			generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
			#output generated of form "User: <question>, Assistant: <answer>"
			responses[i] = generated_texts[0][generated_texts[0].find("Assistant: ")+11:]

		return responses

class qwen2_5(VLM):
	def __init__(self):
		super().__init__("Qwen/Qwen2.5-VL-7B-Instruct")

		self.model, self.processor = self.setup()

	def setup(self):
		from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
		from qwen_vl_utils import process_vision_info

		torch.cuda.empty_cache()
		torch.cuda.set_per_process_memory_fraction(0.9)

		model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
		self.DEVICE = "cuda"

		model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
					model_id, torch_dtype="auto", device_map="auto"
				)

		processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
		return model, processor

	def call(self):

		from PIL import Image

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')
		question = self.queries		# since we have only one prompt 

		for i, img_file in enumerate(self.IMG_LIST):
			image = Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB')
			message = [
				{
					"role": "user",
					"content": [
						{"type": "image"},
						{"type": "text", "text": question},
					]
				} 
			]

			prompt = self.processor.apply_chat_template(message, add_generation_prompt=True)
			inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
			inputs = {k: v.to(self.DEVICE) for k, v in inputs.items()}

			generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
			generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
			
			responses[i] = generated_texts[0].split('in length.')[1][11:]

		return responses


class miniCPM(VLM):
	def __init__(self):
		super().__init__("openbmb/MiniCPM-V-2_6")

		self.model, self.tokenizer = self.setup()

	def setup(self):
		import torch
		from transformers import AutoModel, AutoTokenizer

		model = AutoModel.from_pretrained(self.MODEL, trust_remote_code=True,
			attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
		model = model.eval().cuda()
		tokenizer = AutoTokenizer.from_pretrained(self.MODEL, trust_remote_code=True)

		return model, tokenizer

	def call(self):
		from PIL import Image

		responses = np.empty(shape  = (len(self.IMG_LIST),), dtype = 'U2048')
		question = self.queries		# since we have only one prompt 

		for i, img_file in enumerate(self.IMG_LIST):
			image = (Image.open(f'{self.SRC_DIR}/{self.IMG_DIR}/{img_file}').convert('RGB'))
			msgs = [{'role': 'user', 'content': [question]}]

			responses[i] = self.model.chat(
				image=image,
				msgs=msgs,
				tokenizer=self.tokenizer
			)

		return responses

def model_factory(MODEL):
	if MODEL == 'llava-hf/llava-v1.6-mistral-7b-hf':
		return Llava()
	elif MODEL == 'THUDM/cogvlm-chat-hf':
		return CogVLM()
	elif MODEL == "THUDM/cogvlm2-llama3-chat-19B":
		return CogVLM2()
	elif MODEL == "Salesforce/blip2-flan-t5-xl":
		return FlanT5()
	elif MODEL == "HuggingFaceM4/idefics-9b":
		return Idefics9b()
	elif MODEL == "Salesforce/instructblip-vicuna-7b":
		return InstructBlipVicunna7b()
	elif MODEL == "HuggingFaceM4/Idefics3-8B-Llama3":
		return idefics3_8b()
	elif MODEL == "Qwen/Qwen2.5-VL-7B-Instruct":
		return qwen2_5()
	elif MODEL == "openbmb/MiniCPM-V-2_6":
		return miniCPM()
	else:
		return None

if __name__ == "__main__":
	# load environment first thing.
	load_dotenv()

	parser = argparse.ArgumentParser(
			prog = "RUN MODEL",
			description='This script runs the model stored in argument m to perform tasks on the image dataset.'
	)
	parser.add_argument('-m','--model', default=os.getenv('MODEL'), help='the model to be run')

	args = parser.parse_args()

	model = model_factory(args.model)

	assert model is not None

	model()

