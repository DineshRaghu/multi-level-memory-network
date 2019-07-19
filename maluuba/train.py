import json
import copy
import numpy as np
from data_handler import DataHandler
from model import DialogueModel
import os
import tensorflow as tf
import cPickle as pickle
import nltk
import sys
import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math, re, argparse
import functools
import logging
logging.getLogger().setLevel(logging.INFO)

class Trainer(object):

	def __init__(self,model,handler,ckpt_path,num_epochs,learning_rate):
		self.handler = handler
		self.model = model
		self.ckpt_path = ckpt_path
		self.epochs = num_epochs
		self.learning_rate = learning_rate

		if not os.path.exists(self.ckpt_path):
			os.makedirs(self.ckpt_path)

		self.global_step = tf.contrib.framework.get_or_create_global_step(graph=None)
		self.optimizer = tf.contrib.layers.optimize_loss(
			loss=self.model.loss,
			global_step=self.global_step,
			learning_rate=self.learning_rate,
			optimizer=tf.train.AdamOptimizer,
			clip_gradients=10.0,
			name='optimizer_loss'
		)
		self.saver = tf.train.Saver(max_to_keep=5)
		self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=True))
		init = tf.global_variables_initializer()
		self.sess.run(init)

		checkpoint = tf.train.latest_checkpoint(self.ckpt_path)
		if checkpoint:
			self.saver.restore(self.sess, checkpoint)
			logging.info("Loaded parameters from checkpoint")

	def trainData(self):
		curEpoch = 0
		step = 0
		epochLoss = []

		logging.info("Training the model")

		best_f1 = 0.0

		while curEpoch <= self.epochs:
			step = step + 1

			batch, epoch_done = self.handler.get_batch(train=True)
			feedDict = self.model.get_feed_dict(batch)

			fetch = [self.global_step, self.model.loss, self.optimizer]
			mod_step,loss,_ = self.sess.run(fetch,feed_dict = feedDict)
			epochLoss.append(loss)

			if step % 300 == 0:
				outstr = "step: "+str(step)+" Loss: "+str(loss)
				logging.info(outstr)

			if epoch_done:
				train_loss = np.mean(np.asarray(epochLoss))

				val_epoch_done = False
				valstep = 0
				valLoss = 0.0
				needed = {}
				needed['sentences'] = []
				needed['output'] = []

				while not val_epoch_done:
					valstep = valstep + 1
					batch, val_epoch_done = self.handler.get_batch(train=False)
					feedDict = self.model.get_feed_dict(batch)
					val_loss,sentences = self.sess.run([self.model.loss,self.model.gen_x],feed_dict=feedDict)
					if 1 not in batch['dummy']:
						needed['sentences'].extend(sentences)
						needed['output'].extend(batch['out_utt'])
					else:
						index = batch['dummy'].index(1)
						needed['sentences'].extend(sentences[0:index])
						needed['output'].extend(batch['out_utt'][0:index])
					valLoss = valLoss + val_loss

				valLoss = valLoss / float(valstep)
				outstr = "Train-info: "+ "Epoch: ",str(curEpoch)+" Loss: "+str(train_loss)
				logging.info(outstr)
				outstr = "Val-info: "+"Epoch "+str(curEpoch)+" Loss: "+str(valLoss)
				logging.info(outstr)
				if curEpoch > 5:
					current_f1 = self.evaluate(needed,self.handler.vocab)
					if current_f1 >= best_f1:
						best_f1 = current_f1
						self.saver.save(self.sess, os.path.join(self.ckpt_path, 'model'), global_step=curEpoch)
				
				epochLoss = []
				curEpoch = curEpoch + 1

	def score(self,parallel_corpus):

		# containers
		count = [0, 0, 0, 0]
		clip_count = [0, 0, 0, 0]
		r = 0
		c = 0
		weights = [0.25, 0.25, 0.25, 0.25]

		# accumulate ngram statistics
		for hyps, refs in parallel_corpus:
			hyps = [hyp.split() for hyp in hyps]
			refs = [ref.split() for ref in refs]
			for hyp in hyps:

				for i in range(4):
					# accumulate ngram counts
					hypcnts = Counter(ngrams(hyp, i + 1))
					cnt = sum(hypcnts.values())
					count[i] += cnt

					# compute clipped counts
					max_counts = {}
					for ref in refs:
						refcnts = Counter(ngrams(ref, i + 1))
						for ng in hypcnts:
							max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
					clipcnt = dict((ng, min(count, max_counts[ng])) \
								   for ng, count in hypcnts.items())
					clip_count[i] += sum(clipcnt.values())

				# accumulate r & c
				bestmatch = [1000, 1000]
				for ref in refs:
					if bestmatch[0] == 0: break
					diff = abs(len(ref) - len(hyp))
					if diff < bestmatch[0]:
						bestmatch[0] = diff
						bestmatch[1] = len(ref)
				r += bestmatch[1]
				c += len(hyp)

		# computing bleu score
		p0 = 1e-7
		bp = 1 if c > r else math.exp(1 - float(r) / float(c))
		p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
				for i in range(4)]
		s = math.fsum(w * math.log(p_n) \
					  for w, p_n in zip(weights, p_ns) if p_n)
		bleu = bp * math.exp(s)
		return bleu

	def evaluate(self,data,vocab):
		entities = ["business","economy","breakfast","wifi","gym","parking","spa","park","museum","beach","shopping","market","airport","university","mall","cathedral","downtown","palace","theatre"]
		outs = []
		golds = []
		
		tp_prec = 0.0
		tp_recall = 0.0
		total_prec = 0.0
		total_recall = 0.0

		for i in range(0,len(data['sentences'])):
			sentence = data['sentences'][i]
			sentence = list(sentence)
			if vocab['vocab_mapping']['$STOP$'] not in sentence:
				index = len(sentence)
			else:
				index = sentence.index(vocab['vocab_mapping']['$STOP$'])
			predicted = [str(sentence[j]) for j in range(0,index)]
			ground = data['output'][i]
			ground = list(ground)
			index = ground.index(vocab['vocab_mapping']['$STOP$'])
			ground_truth = [str(ground[j]) for j in range(0,index)]

			gold_anon = [vocab['rev_mapping'][word].encode('utf-8') for word in ground_truth ]
			out_anon = [vocab['rev_mapping'][word].encode('utf-8') for word in predicted ]

			for word in out_anon:
				if word in entities or word.isdigit() or word in self.handler.all_entities:
					total_prec = total_prec + 1
					if word in gold_anon:
						tp_prec = tp_prec + 1
						
			for word in gold_anon:
				if word in entities or word.isdigit() or word in self.handler.all_entities:
					total_recall = total_recall + 1
					if word in out_anon:
						tp_recall = tp_recall + 1
						
			gold = gold_anon
			out = out_anon
			
			golds.append(" ".join(gold))
			outs.append(" ".join(out))

		wrap_generated = [[_] for _ in outs]
		wrap_truth = [[_] for _ in golds]
		prec = tp_prec/total_prec
		recall = tp_recall/total_recall
		if prec == 0 or recall == 0:
			f1 = 0.0
		else:
			f1 = 2*prec*recall/(prec+recall)
		overall_f1 = f1
		print "Bleu: %.3f, Prec: %.3f, Recall: %.3f, F1: %.3f" % (self.score(zip(wrap_generated, wrap_truth)),prec,recall,f1)
		return overall_f1

	def test(self):
		test_epoch_done = False

		teststep = 0
		testLoss = 0.0
		needed = {}
		needed['sentences'] = []
		needed['output'] = []
		needed['context'] = []
		needed['kb'] = []

		while not test_epoch_done:
			teststep = teststep + 1
			batch, test_epoch_done = self.handler.get_batch(train=False)
			feedDict = self.model.get_feed_dict(batch)
			sentences = self.sess.run(self.model.gen_x,feed_dict=feedDict)

			if 1 not in batch['dummy']:
				needed['sentences'].extend(sentences)
				needed['output'].extend(batch['out_utt'])
				needed['context'].extend(batch['context'])
				needed['kb'].extend(batch['kb'])
			else:
				index = batch['dummy'].index(1)
				needed['sentences'].extend(sentences[0:index])
				needed['output'].extend(batch['out_utt'][0:index])
				needed['context'].extend(batch['context'][0:index])
				needed['kb'].extend(batch['kb'][0:index])

		pickle.dump(needed,open("needed.p","w"))
		self.evaluate(needed,self.handler.vocab)

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', type=int, default=4)
	parser.add_argument('--emb_dim', type=int, default=200)
	parser.add_argument('--enc_hid_dim', type=int, default=128)
	parser.add_argument('--dec_hid_dim', type=int, default=256)
	parser.add_argument('--attn_size', type=int, default=200)
	parser.add_argument('--epochs', type=int, default=12)
	parser.add_argument('--learning_rate', type=float, default=2.5e-4)
	parser.add_argument('--dataset_path', type=str, default='../data/Maluuba/')
	parser.add_argument('--glove_path', type=str, default='../data/')
	parser.add_argument('--checkpoint', type=str, default="./trainDir/")
	config = parser.parse_args()

	DEVICE = "/gpu:0"

	logging.info("Loading Data")

	handler = DataHandler(
				emb_dim = config.emb_dim,
				batch_size = config.batch_size,
				train_path = config.dataset_path + "train.json",
				val_path = config.dataset_path + "val.json",
				test_path = config.dataset_path + "test.json",
				vocab_path = "./vocab.json",
				entities_path = config.dataset_path + "entities.json",
				glove_path = config.glove_path)

	logging.info("Loading Architecture")

	model = DialogueModel(
				device = DEVICE,
				batch_size = config.batch_size,
				inp_vocab_size = handler.input_vocab_size,
				out_vocab_size = handler.output_vocab_size,
				generate_size = handler.generate_vocab_size,
				emb_init = handler.emb_init,
				result_keys_vector = handler.result_keys_vector,
				emb_dim = config.emb_dim,
				enc_hid_dim = config.enc_hid_dim,
				dec_hid_dim = config.dec_hid_dim,
				attn_size = config.attn_size)

	logging.info("Loading Trainer")

	trainer = Trainer(
				model=model,
				handler=handler,
				ckpt_path="./trainDir/",
				num_epochs=config.epochs,
				learning_rate = config.learning_rate)

	trainer.trainData()

main()