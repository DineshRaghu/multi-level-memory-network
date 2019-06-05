import cPickle as pickle 
import copy
import json
import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math, re, argparse
import functools

entities = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday','medicine','conference','dinner','lab','yoga','tennis','doctor','meeting','swimming','optometrist','football','dentist',"overcast","snow", "stormy", "hail","hot", "rain", "cold","cloudy", "warm", "windy","foggy", "humid", "frost", "blizzard", "drizzle", "dry", "dew", "misty","friend", "home", "coffee", "chinese","pizza", "grocery", "rest", "shopping", "parking","gas", "hospital"]
		
def score(parallel_corpus):

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


data = pickle.load(open("needed.p"))
vocab = json.load(open("./vocab.json"))
outs = []
golds = []
domain_wise = {}
for domain in ['schedule','navigate','weather']:
	domain_wise[domain] = {}
	domain_wise[domain]['tp_prec'] = 0.0
	domain_wise[domain]['tp_recall'] = 0.0
	domain_wise[domain]['total_prec'] = 0.0
	domain_wise[domain]['total_recall'] = 0.0
	domain_wise[domain]['gold'] = []
	domain_wise[domain]['output'] = []

tp_prec = 0.0
tp_recall = 0.0
total_prec = 0.0
total_recall = 0.0

for i in range(0,len(data['sentences'])):
	sentence = data['sentences'][i]
	domain = data['type'][i]
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
		if word in entities or '_' in word:
			total_prec = total_prec + 1
			domain_wise[domain]['total_prec'] = domain_wise[domain]['total_prec'] + 1
			if word in gold_anon:
				tp_prec = tp_prec + 1
				domain_wise[domain]['tp_prec'] = domain_wise[domain]['tp_prec'] + 1

	for word in gold_anon:
		if word in entities or '_' in word:
			total_recall = total_recall + 1
			domain_wise[domain]['total_recall'] = domain_wise[domain]['total_recall'] + 1
			if word in out_anon:
				tp_recall = tp_recall + 1
				domain_wise[domain]['tp_recall'] = domain_wise[domain]['tp_recall'] + 1

	gold = gold_anon
	out = out_anon

	domain_wise[domain]['gold'].append(" ".join(gold))
	golds.append(" ".join(gold))
	domain_wise[domain]['output'].append(" ".join(out))
	outs.append(" ".join(out))

with open('output', 'w') as output_file:
	for line in outs:
		output_file.write(line+"\n")

with open('reference', 'w') as output_file:
	for line in golds:
		output_file.write(line+"\n")

wrap_generated = [[_] for _ in outs]
wrap_truth = [[_] for _ in golds]
prec = tp_prec/total_prec
recall = tp_recall/total_recall
print 'prec',tp_prec,total_prec
print 'recall',tp_recall,total_recall
print "Bleu: %.3f, Prec: %.3f, Recall: %.3f, F1: %.3f" % (score(zip(wrap_generated, wrap_truth)),prec,recall,2*prec*recall/(prec+recall))
for domain in ['schedule','navigate','weather']:
	prec = domain_wise[domain]['tp_prec']/domain_wise[domain]['total_prec']
	recall = domain_wise[domain]['tp_recall']/domain_wise[domain]['total_recall']
	print "prec",domain_wise[domain]['tp_prec'],domain_wise[domain]['total_prec']
	print "recall",domain_wise[domain]['tp_recall'],domain_wise[domain]['total_recall']
	wrap_generated = [[_] for _ in domain_wise[domain]['output']]
	wrap_truth = [[_] for _ in domain_wise[domain]['gold']]
	print "Domain: " + str(domain) + ", Bleu: %.3f, Prec: %.3f, Recall: %.3f, F1: %.3f" % (score(zip(wrap_generated, wrap_truth)),prec,recall,2*prec*recall/(prec+recall))
	