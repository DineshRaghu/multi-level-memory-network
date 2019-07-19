import json
import copy
import random
import nltk
import os
import sys
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)

single_word_entities = ["business","economy","breakfast","wifi","gym","parking","spa","park","museum","beach","shopping","market","airport","university","mall","cathedral","downtown","palace","theatre"]
		
class DataHandler(object):

	def __init__(self,batch_size,emb_dim,train_path,val_path,test_path,vocab_path,entities_path,glove_path):

		
		self.batch_size = batch_size
		self.train_path = train_path
		self.emb_dim = emb_dim
		self.vocab_threshold = 3
		self.val_path = val_path
		self.test_path = test_path
		self.vocab_path = vocab_path
		self.entities_path = entities_path
		self.all_entities = json.load(open(self.entities_path))
		self.glove_path = glove_path

		self.result_keys = self.initialise_keys()
		self.vocab = self.load_vocab()
		self.input_vocab_size = self.vocab['input_vocab_size']
		self.output_vocab_size = self.vocab['output_vocab_size']
		self.generate_vocab_size = self.vocab['generate_vocab_size']
		self.emb_init = self.load_glove_vectors()
		self.result_keys_vector = self.keys_vector()

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
		self.val_num = len(self.val_data)

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

	def initialise_keys(self):
		return ['airport', 'arr_time_dst', 'arr_time_or', 'beach', 'breakfast', 'cathedral', 'dep_time_dst', 'dep_time_or', 'destination', 'downtown', 'duration', 'end', 'guest', 'gym', 'mall', 'market', 'museum', 'name', 'origin', 'palace', 'park', 'parking', 'price', 'rating', 'seat', 'shopping', 'spa', 'start', 'theatre', 'university', 'wifi']

	def keys_vector(self):

		result_keys_vector = []
		for key in self.result_keys:
			result_keys_vector.append(self.vocab['vocab_mapping'][key])
		result_keys_vector.append(self.vocab['vocab_mapping']["$EMPTY$"])
		return result_keys_vector

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
			for utt in d['context']:
				utts.append(utt['utt'])
			for utt in utts:
				tokens = utt.split(" ")
				for token in tokens:
					if token.encode('ascii',errors='ignore').lower() not in vocab:
						vocab[token.encode('ascii',errors='ignore').lower()] = 1
					else:
						vocab[token.encode('ascii',errors='ignore').lower()] = vocab[token.encode('ascii',errors='ignore').lower()] + 1 


			for query in d['kb']:
				search = query[0]
				results = query[1]
				for key in search:
					if key.encode('ascii',errors='ignore').lower() not in vocab:
						vocab[key.encode('ascii',errors='ignore').lower()] = 1
					else:
						vocab[key.encode('ascii',errors='ignore').lower()] = vocab[key.encode('ascii',errors='ignore').lower()] + 1

					if search[key].encode('ascii',errors='ignore').lower() not in vocab:
						vocab[search[key].encode('ascii',errors='ignore').lower()] = 1
					else:
						vocab[search[key].encode('ascii',errors='ignore').lower()] = vocab[search[key].encode('ascii',errors='ignore').lower()] + 1

				for result in results:
					for key in result:
						
						if result[key].encode('ascii',errors='ignore').lower() not in vocab:
							vocab[result[key].encode('ascii',errors='ignore').lower()] = 1
						else:
							vocab[result[key].encode('ascii',errors='ignore').lower()] = vocab[result[key].encode('ascii',errors='ignore').lower()] + 1
		
		words = []
		for v in vocab:
			if vocab[v] > self.vocab_threshold or v in self.all_entities or v.isdigit():
				words.append(v)

		for key in self.result_keys:
			if key not in words:
				words.append(key)

		words.append("$UNK$")
		words.append("$STOP$")
		words.append("$PAD$")
		words.append("$EMPTY$")

		for i in range(1,31):
			words.append("$u"+str(i)+"$")
			words.append("$s"+str(i)+"$")
		words.append("$u31$")

		generate_words = []
		copy_words = []
		for word in words:
			if word in single_word_entities or word in self.all_entities:
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
		vectorized['out_len'] = []
		vectorized['context_len'] = []
		vectorized['query_mask'] = []
		vectorized['query_actual_mask'] = []
		vectorized['search_keys'] = []
		vectorized['search_mask'] = []
		vectorized['search_values'] = []
		vectorized['results_mask'] = []
		vectorized['results_actual_mask'] = []
		vectorized['result_keys_mask'] = []
		vectorized['result_values'] = []
		vectorized['dummy'] = []
		vectorized['empty'] = []

		vectorized['kb'] = []
		vectorized['context'] = []

		max_query_num = 0		
		max_inp_utt_len = 0
		max_out_utt_len = 0
		max_context_len = 0

		search_keys_len = 8
		result_keys_len = 31

		for item in batch:
			if len(item['kb']) > max_query_num:
				max_query_num = len(item['kb'])

			if len(item['context']) > max_context_len:
				max_context_len = len(item['context'])

			
			for utt in item['context']:
				tokens = utt['utt'].split(" ")
				
				if len(tokens) > max_inp_utt_len:
					max_inp_utt_len = len(tokens)

			tokens = item['output'].split(" ")
			if len(tokens) > max_out_utt_len:
				max_out_utt_len = len(tokens)

		max_inp_utt_len = max_inp_utt_len + 1

		max_out_utt_len = max_out_utt_len + 1
		vectorized['max_out_utt_len'] = max_out_utt_len

		for item in batch:	

			vectorized['context'].append(item['context'])
			vectorized['kb'].append(item['kb'])
			
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
				tokens = utt['utt'].split(" ")+[sentinel]
				for token in tokens:
					if token in self.vocab['vocab_mapping']:
						inp.append(self.vocab['vocab_mapping'][token.encode('ascii',errors='ignore')])
					else:
						inp.append(self.vocab['vocab_mapping']["$UNK$"])

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
			vectorized['inp_len'].append(copy.copy(vector_len))

			vector_out = []

			tokens = item['output'].split(" ")
			tokens.append("$STOP$")
			for token in tokens:
				if token in self.vocab['vocab_mapping']:
					vector_out.append(self.vocab['vocab_mapping'][token.encode('ascii',errors='ignore')])
				else:
					vector_out.append(self.vocab['vocab_mapping']["$UNK$"])

			for _ in range(0,max_out_utt_len - len(tokens)):
				vector_out.append(self.vocab['vocab_mapping']["$PAD$"])
			vectorized['out_utt'].append(copy.copy(vector_out))
			vectorized['out_len'].append(len(tokens))

			vector_search_mask = []
			vector_search_keys = []
			vector_search_values = []
			vector_query_mask = []
			vector_query_actual_mask = []
			vector_result_keys_mask = []
			vector_result_values = []
			vector_results_mask = []
			vector_results_actual_mask = []

			kb = []

			for query in item['kb']:
				
				search_mask = []
				search_keys = []
				search_values = []

				for key in query[0]:
					search_mask.append(1)
					search_keys.append(self.vocab['vocab_mapping'][key])
					if str(query[0][key]).encode('ascii',errors='ignore').lower() in self.vocab['vocab_mapping']:
						search_values.append(self.vocab['vocab_mapping'][str(query[0][key]).encode('ascii',errors='ignore').lower()])
					else:
						search_values.append(self.vocab['vocab_mapping']["$UNK$"])

				for _ in range(0,search_keys_len - len(query[0].keys())):
					search_mask.append(0)
					search_keys.append(self.vocab['vocab_mapping']["$PAD$"])
					search_values.append(self.vocab['vocab_mapping']["$PAD$"])

				results_mask = []
				results_actual_mask = []
				result_values = []
				result_keys_mask = []
				result_keys = []

				for result in query[1]:
					results_mask.append(1)
					results_actual_mask.append(1)
					mask = []
					values = []

					for key in self.result_keys:
						if key in result:
							mask.append(1)
							values.append(self.vocab['vocab_mapping'][str(result[key].encode('ascii',errors='ignore')).lower()])
						else:
							mask.append(0)
							values.append(self.vocab['vocab_mapping']["$PAD$"])
					if 1 not in mask:
						mask.append(1)
					else:
						mask.append(0)
					values.append(self.vocab['vocab_mapping']["$PAD$"])

					result_keys_mask.append(copy.copy(mask))
					result_values.append(copy.copy(values))
					
				if len(results_mask) == 0:
					results_mask.append(1)
					results_actual_mask.append(0)
					result_keys_mask.append([0 for _ in range(0,len(self.result_keys))]+[1])
					result_values.append([self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,len(self.result_keys))]+[self.vocab['vocab_mapping']["$EMPTY$"]])
					
				for _ in range(0,10-len(results_mask)):
					results_mask.append(0)
					results_actual_mask.append(0)
					result_keys_mask.append([0 for _ in range(0,len(self.result_keys))]+[1])
					result_values.append([self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,len(self.result_keys))]+[self.vocab['vocab_mapping']["$EMPTY$"]])
					

				kb.append([1,1,copy.copy(search_mask),copy.copy(search_values),copy.copy(search_keys),copy.copy(results_mask),copy.copy(results_actual_mask),copy.copy(result_keys_mask),copy.copy(result_values)])

			if len(kb) == 0:
				newitem = []
				newitem.extend([1,0,[1]+[0 for _ in range(0,search_keys_len-1)],[self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0,search_keys_len)],[self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0,search_keys_len)]])
				newitem.append([1]+[0 for _ in range(0,9)])
				newitem.append([0 for _ in range(0,10)])
				newitem.append([[0 for _ in range(0,len(self.result_keys))]+[1] for _ in range(0,10)])
				newitem.append([[self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,len(self.result_keys))]+[self.vocab['vocab_mapping']["$EMPTY$"]] for _ in range(0,10)])
				kb.append(copy.copy(newitem))
			
			kb_length = len(kb)

			for _ in range(0,max_query_num - kb_length):
				newitem = []
				newitem.extend([0,0,[1]+[0 for _ in range(0,search_keys_len-1)],[self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0,search_keys_len)],[self.vocab["vocab_mapping"]["$PAD$"] for _ in range(0,search_keys_len)]])
				newitem.append([1]+[0 for _ in range(0,9)])
				newitem.append([0 for _ in range(0,10)])
				newitem.append([[0 for _ in range(0,len(self.result_keys))]+[1] for _ in range(0,10)])
				newitem.append([[self.vocab['vocab_mapping']["$PAD$"] for _ in range(0,len(self.result_keys)+1)] for _ in range(0,10)])
				kb.append(copy.copy(newitem))

			random.shuffle(kb)

			for each in kb:
				vector_query_mask.append(each[0])
				vector_query_actual_mask.append(each[1])
				vector_search_mask.append(each[2])
				vector_search_values.append(each[3])
				vector_search_keys.append(each[4])
				vector_results_mask.append(each[5])
				vector_results_actual_mask.append(each[6])
				vector_result_keys_mask.append(each[7])
				vector_result_values.append(each[8])

			vectorized['query_mask'].append(copy.copy(vector_query_mask))
			vectorized['query_actual_mask'].append(copy.copy(vector_query_actual_mask))
			vectorized['search_mask'].append(copy.copy(vector_search_mask))
			vectorized['search_values'].append(copy.copy(vector_search_values))
			vectorized['search_keys'].append(copy.copy(vector_search_keys))
			vectorized['results_mask'].append(copy.copy(vector_results_mask))
			vectorized['results_actual_mask'].append(copy.copy(vector_results_actual_mask))
			vectorized['result_keys_mask'].append(copy.copy(vector_result_keys_mask))
			vectorized['result_values'].append(copy.copy(vector_result_values))
			
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
