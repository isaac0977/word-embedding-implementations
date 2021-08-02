import torch
import numpy as np
from transformers import BertTokenizer, BertModel


def load_tokenizer():
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	return tokenizer

def load_model():
	model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
	return model

def generate_sentence_embeddings(sentence, model, tokenizer):
	inputs = tokenizer(sentence, return_tensors="pt")
	outputs = model(**inputs)
	cls_embedding = outputs.last_hidden_state[0][0]
	return cls_embedding
	#sum = torch.sum(outputs.last_hidden_state[0], 0)
	#return sum

def generate_average_word_embedding(word, model, tokenizer):
	inputs = tokenizer(word, return_tensors="pt")
	outputs = model(**inputs)
	#Take out the CLS and SEP Tokens
	outputs = outputs.last_hidden_state[0][1:-1]
	tokenized_length = len(outputs)
	summed_vector = torch.sum(outputs, 0)
	return torch.div(summed_vector , tokenized_length)