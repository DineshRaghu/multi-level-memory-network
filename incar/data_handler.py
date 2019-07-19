import json
import copy
import random
import nltk
import os
import sys
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)

single_word_entities = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','medicine','conference','dinner','lab','yoga','tennis','doctor','meeting','swimming','optometrist','football','dentist',"overcast","snow", "stormy", "hail","hot", "rain", "cold","cloudy", "warm", "windy","foggy", "humid", "frost", "blizzard", "drizzle", "dry", "dew", "misty","friend", "home", "coffee", "chinese","pizza", "grocery", "rest", "shopping", "parking","gas", "hospital"]
		
class DataHandler(object):

	def __init__(self,emb_dim,batch_size,train_path,val_path,test_path,vocab_path,glove_path):

		self.batch_size = batch_size
		self.train_path = train_path
		self.vocab_threshold = 3
		self.val_path = val_path
		self.test_path = test_path
		self.vocab_path = vocab_path
		self.emb_dim = emb_dim
		self.glove_path = glove_path

		self.vocab = self.load_vocab()
		self.input_vocab_size = self.vocab['input_vocab_size']
		self.output_vocab_size = self.vocab['output_vocab_size']
		self.generate_vocab_size = self.vocab['generate_vocab_size']
		self.emb_init = self.load_glove_vectors()

		self.train_data = json.load(open(self.train_path))
		self.val_data = json.load(open(self.val_path))
		self.test_data = json.load(open(self.test_path))

		random.shuffle(self.train_data)
		random.shuffle(self.val_data)
		random.shuffle(self.test_data)

		self.val_data_full = self.append_dummy_data(self.val_data)

		self.train_index = 0
		self.val_index = 0
		self.train_num = len(self.train_data)
		self.val_num = len(self.val_data_full)

	def append_dummy_data(self,data):
		new_data = []
		for i in range(0,len(data)):
			data[i]['dummy'] = 0
			new_data.append(copy.copy(data[i]))

		last = data[-1]
		last['dummy'] = 1
		for _ in range(0,self.batch_size - len(data)%self.batch_size):
			new_data.append(copy.copy(last))

		return copy.copy(new_data)


	def load_glove_vectors(self):
		logging.info("Loading pre-trained Word Embeddings")
		filename = self.glove_path + "glove.6B.200d.txt"
		glove = {}
		file = open(filename,'r')
		for line in file.readlines():
			row = line.strip().split(' ')
			glove[row[0]] = np.asarray(row[1:])
		logging.info('Loaded GloVe!')
		file.close()
		embeddings_init = np.random.normal(size=(self.vocab['input_vocab_size'],self.emb_dim)).astype('f')
		count = 0
		for word in self.vocab['vocab_mapping']:
			if word in glove:
				count = count + 1
				embeddings_init[self.vocab['vocab_mapping'][word]] = glove[word]

		del glove

		logging.info("Loaded "+str(count)+" pre-trained Word Embeddings")
		return embeddings_init


	def load_vocab(self):
		if os.path.isfile(self.vocab_path):
			logging.info("Loading vocab from file")
			with open(self.vocab_path) as f:
				return json.load(f)
		else:
			logging.info("Vocab file not found. Computing Vocab")
			with open(self.train_path) as f:
				train_data = json.load(f)
			with open(self.val_path) as f:
				val_data = json.load(f)
			with open(self.test_path) as f:
				test_data = json.load(f)

			full_data = []
			full_data.extend(train_data)
			full_data.extend(val_data)
			full_data.extend(test_data)

			return self.get_vocab(full_data)

	def get_vocab(self,data):

		vocab = {}
		for d in data:
			utts = []
			utts.append(d['output'])
			utts.extend(d['context'])
			for utt in utts:
				tokens = utt.split(" ")
				for token in tokens:
					if token.lower() not in vocab:
						vocab[token.lower()] = 1
					else:
						vocab[token.lower()] = vocab[token.lower()] + 1 

			for item in d['kb']:
				for key in item:
					if key.lower() not in vocab:
						vocab[key.lower()] = 1
					else:
						vocab[key.lower()] = vocab[key.lower()] + 1
					token = item[key]
					if token.lower() not in vocab:
						vocab[token.lower()] = 1
					else:
						vocab[token.lower()] = vocab[token.lower()] + 1

		words = vocab.keys()
		words.append("$STOP$")
		words.append("$PAD$")

		for i in range(1,6):
			words.append("$u"+str(i)+"$")
			words.append("$s"+str(i)+"$")
		words.append("$u6$")

		generate_words = []
		copy_words = []
		for word in words:
			if word in single_word_entities or '_' in word:
				copy_words.append(word)
			else:
				generate_words.append(word)

		output_vocab_size = len(words) + 1

		generate_indices = [i for i in range(1,len(generate_words)+1)]
		copy_indices = [i for i in range(len(generate_words)+1,output_vocab_size)]
		random.shuffle(generate_indices)
		random.shuffle(copy_indices)

		mapping = {}
		rev_mapping = {}

		for i in range(0,len(generate_words)):
			mapping[generate_words[i]] = generate_indices[i]
			rev_mapping[str(generate_indices[i])] = generate_words[i]

		for i in range(0,len(copy_words)):
			mapping[copy_words[i]] = copy_indices[i]
			rev_mapping[str(copy_indices[i])] = copy_words[i]

		mapping["$GO$"] = 0
		rev_mapping[0] = "$GO$"
		vocab_dict = {}
		vocab_dict['vocab_mapping'] = mapping
		vocab_dict['rev_mapping'] = rev_mapping
		vocab_dict['input_vocab_size'] = len(words) + 1
		vocab_dict['generate_vocab_size'] = len(generate_words) + 1
		vocab_dict['output_vocab_size'] = output_vocab_size

		with open(self.vocab_path,'w') as f:
			json.dump(vocab_dict,f)

		logging.info("Vocab file created")

		return vocab_dict

	def get_sentinel(self,i,context):
		if i%2 == 0:
			speaker = "u"
			turn = (context - i + 1)/2
		else:
			speaker = "s"
			turn = (context - i)/2
		return "$"+speaker+str(turn)+"$"

	def vectorize(self,batch,train):
		vectorized = {}
		vectorized['inp_utt'] = []
		vectorized['out_utt'] = []
		vectorized['inp_len'] = []
		vectorized['context_len'] = []
		vectorized['out_len'] = []
		vectorized['kb'] = []
		vectorized['kb_mask'] = []
		vectorized['keys'] = []
		vectorized['keys_mask'] = []
		vectorized['mapping'] = []
		vectorized['rev_mapping'] = []
		vectorized['type'] = []
		vectorized['dummy'] = []
		vectorized['empty'] = []

		vectorized['knowledge'] = []
		vectorized['context'] = []
		max_inp_utt_len = 0
		max_out_utt_len = 0
		max_context_len = 0
		kb_len = 0
		keys_len = 6

		for item in batch:
			
			if len(item['context']) > max_context_len:
				max_context_len = len(item['context'])

			for utt in item['context']:
				tokens = utt.split(" ")
				
				if len(tokens) > max_inp_utt_len:
					max_inp_utt_len = len(tokens)

			tokens = item['output'].split(" ")
			if len(tokens) > max_out_utt_len:
				max_out_utt_len = len(tokens)

			if len(item['kb']) > kb_len:
				kb_len = len(item['kb'])

		max_inp_utt_len = max_inp_utt_len + 1

		max_out_utt_len = max_out_utt_len + 1
		vectorized['max_out_utt_len'] = max_out_utt_len


		for item in batch:
			vectorized['context'].append(item['context'])
			vectorized['knowledge'].append(item['kb'])
			vectorized['mapping'].append(item['mapping'])
			vectorized['rev_mapping'].append(item['rev_mapping'])
			vectorized['type'].append(item['type'])
			if item['kb'] == []:
				vectorized['empty'].append(0)
			else:
				vectorized['empty'].append(1)
			if not train:
				vectorized['dummy'].append(item['dummy'])
			vector_inp = []
			vector_len = []

			for i in range(0,len(item['context'])):
				utt = item['context'][i]
				inp = []
				sentinel = self.get_sentinel(i,len(item['context']))
				tokens = utt.split(" ") + [sentinel]
				for token in tokens:
					inp.append(self.vocab['vocab_mapping'][token])

				vector_len.append(len(tokens))
				for _ in range(0,max_inp_utt_len - len(tokens)):
					inp.append(self.vocab['vocab_mapping']["$PAD$"])
				vector_inp.append(copy.copy(inp))

			vectorized['context_len'].append(len(item['context']))

			for _ in range(0,max_context_len - len(item['context'])):
				vector_len.append(0)
				inp = []
				for _ in range(0,max_inp_utt_len):
					inp.append(self.vocab['vocab_mapping']["$PAD$"])
				vector_inp.append(copy.copy(inp))
				
			vectorized['inp_utt'].append(copy.copy(vector_inp))
			vectorized['inp_len'].append(vector_len)

			vector_out = []
			tokens = item['output'].split(" ")
			tokens.append('$STOP$')
			for token in tokens:
				vector_out.append(self.vocab['vocab_mapping'][token])

			for _ in range(0,max_out_utt_len - len(tokens)):
				vector_out.append(self.vocab['vocab_mapping']["$PAD$"])
			vectorized['out_utt'].append(copy.copy(vector_out))
			vectorized['out_len'].append(len(tokens))

			vector_keys = []
			vector_keys_mask = []
			vector_kb = []
			vector_kb_mask = []

			for result in item['kb']:
				vector_result = []
				vector_result_keys = []
				vector_result_keys_mask = []
				vector_kb_mask.append(1)
				for key in result:
					vector_result.append(self.vocab['vocab_mapping'][result[key]])
					vector_result_keys.append(self.vocab['vocab_mapping'][key])
					vector_result_keys_mask.append(1)

				for _ in range(0,keys_len-len(result.keys())):
					vector_result_keys.append(self.vocab['vocab_mapping']["$PAD$"])
					vector_result_keys_mask.append(0)
					vector_result.append(self.vocab['vocab_mapping']["$PAD$"])
				vector_keys.append(copy.copy(vector_result_keys))
				vector_keys_mask.append(copy.copy(vector_result_keys_mask))
				vector_kb.append(copy.copy(vector_result))

			if item['kb'] == []:
				vector_kb_mask.append(1)
				vector_kb.append([self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,keys_len)])
				vector_keys.append([self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,keys_len)])
				vector_keys_mask.append([1] + [0 for _ in range(0,keys_len-1)])

			current_kb_len = len(vector_kb_mask)

			for _ in range(0,kb_len - current_kb_len):
				vector_kb_mask.append(0)
				vector_kb.append([self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,keys_len)])
				vector_keys.append([self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,keys_len)])
				vector_keys_mask.append([1] + [0 for _ in range(0,keys_len-1)])

			vectorized['kb'].append(copy.copy(vector_kb))
			vectorized['kb_mask'].append(copy.copy(vector_kb_mask))
			vectorized['keys'].append(copy.copy(vector_keys))
			vectorized['keys_mask'].append(copy.copy(vector_keys_mask))

		return vectorized

	def get_batch(self,train):

		epoch_done = False

		if train:
			index = self.train_index
			batch = self.vectorize(self.train_data[index:index+self.batch_size],train)
			self.train_index = self.train_index + self.batch_size

			if self.train_index + self.batch_size > self.train_num:
				self.train_index = 0
				random.shuffle(self.train_data)
				epoch_done = True

		else:
			index = self.val_index
			batch = self.vectorize(self.val_data_full[index:index+self.batch_size],train)
			self.val_index = self.val_index + self.batch_size

			if self.val_index + self.batch_size > self.val_num:
				self.val_index = 0
				random.shuffle(self.val_data)
				self.val_data_full = self.append_dummy_data(self.val_data)
				epoch_done = True


		return batch,epoch_done
