import numpy as np 
import tensorflow as tf   
from tensorflow.python.ops import embedding_ops, array_ops, math_ops, tensor_array_ops, control_flow_ops

class DialogueModel(object):

	def __init__(self,device,batch_size,inp_vocab_size,out_vocab_size,generate_size,emb_init,result_keys_vector,emb_dim,enc_hid_dim,dec_hid_dim,attn_size):

		self.device = device
		self.batch_size = batch_size
		self.emb_dim = emb_dim
		self.inp_vocab_size = inp_vocab_size
		self.out_vocab_size = out_vocab_size
		self.generate_size = generate_size
		self.emb_init = emb_init
		self.result_keys = result_keys_vector
		self.enc_hid_dim = enc_hid_dim
		self.dec_hid_dim = dec_hid_dim
		self.attn_size = attn_size
		self.generate_size = generate_size

		self.inp_utt = tf.placeholder(
			name='inp_utt', dtype=tf.int64,
			shape=[self.batch_size, None, None],
		)

		self.inp_len = tf.placeholder(
			name='inp_len', dtype=tf.int64,
			shape=[self.batch_size, None],
		)	

		self.context_len = tf.placeholder(
			name='context_len', dtype=tf.int64,
			shape=[self.batch_size],
		)	

		self.out_utt = tf.placeholder(
			name='out_utt', dtype=tf.int64,
			shape=[self.batch_size, None],
		)

		self.out_len = tf.placeholder(
			name='out_len', dtype=tf.float32,
			shape=[self.batch_size],
		)

		self.query_mask = tf.placeholder(
			name='query_mask', dtype=tf.float32,
			shape=[self.batch_size,None],
		)

		self.search_mask = tf.placeholder(
			name='search_mask', dtype=tf.float32,
			shape=[self.batch_size,None,8],
		)

		self.search_values = tf.placeholder(
			name='search_values', dtype=tf.int64,
			shape=[self.batch_size,None,8],
		)

		self.results_mask = tf.placeholder(
			name='results_mask', dtype=tf.float32,
			shape=[self.batch_size,None,10],
		)

		self.result_keys_mask = tf.placeholder(
			name='result_keys_mask', dtype=tf.float32,
			shape=[self.batch_size,None,10,len(self.result_keys)],
		)

		self.result_values = tf.placeholder(
			name='result_values', dtype=tf.int64,
			shape=[self.batch_size,None,10,len(self.result_keys)],
		)

		self.max_out_utt_len = tf.placeholder(
			name = 'max_out_utt_len', dtype=tf.int32,
			shape = (),
		)

		self.db_empty = tf.placeholder(
			name='db_empty', dtype=tf.float32,
			shape=[self.batch_size],
		)

		self.buildArch()

	def buildArch(self):

		with tf.device(self.device):

			self.embeddings = tf.get_variable("embeddings", initializer=tf.constant(self.emb_init))
			self.inp_utt_emb = embedding_ops.embedding_lookup(self.embeddings, self.inp_utt)
			
			with tf.variable_scope("encoder"):
				self.encoder_cell_1 = tf.contrib.rnn.GRUCell(self.enc_hid_dim)
				self.encoder_cell_2 = tf.contrib.rnn.GRUCell(2*self.enc_hid_dim)
				self.flat_inp_emb = tf.reshape(self.inp_utt_emb,shape=[-1,tf.shape(self.inp_utt)[2],self.emb_dim])
				self.flat_inp_len = tf.reshape(self.inp_len,shape=[-1])

				outputs,output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.encoder_cell_1,cell_bw=self.encoder_cell_1,inputs=self.flat_inp_emb,dtype=tf.float32,sequence_length=self.flat_inp_len,time_major=False)
				self.flat_encoder_states = tf.concat(outputs,axis=2)
				self.utt_reps = tf.concat(output_states,axis=1)

				self.utt_rep_second = tf.reshape(self.utt_reps,shape=[self.batch_size,-1,2*self.enc_hid_dim])
				self.hidden_states, self.inp_utt_rep = tf.nn.dynamic_rnn(self.encoder_cell_2,self.utt_rep_second,dtype=tf.float32,sequence_length=self.context_len,time_major=False)
				self.encoder_states = tf.reshape(tf.reshape(self.flat_encoder_states,shape=[self.batch_size,-1,tf.shape(self.inp_utt)[2],2*self.enc_hid_dim]), shape=[self.batch_size,-1,2*self.enc_hid_dim])



			self.search_values_emb = embedding_ops.embedding_lookup(self.embeddings, self.search_values)
			self.search_values_rep = tf.einsum('ij,ijk->ijk',tf.pow(tf.reduce_sum(self.search_mask,2),-1),tf.reduce_sum(tf.einsum('ijk,ijkl->ijkl',self.search_mask,self.search_values_emb),2))

			self.result_values_emb = embedding_ops.embedding_lookup(self.embeddings, self.result_values)
			self.result_values_rep = tf.einsum('ijk,ijkl->ijkl',tf.pow(tf.reduce_sum(self.result_keys_mask,3),-1),tf.reduce_sum(tf.einsum('ijkl,ijklm->ijklm',self.result_keys_mask,self.result_values_emb),3))

			self.result_keys_batch = tf.reshape(tf.tile(self.result_keys,[self.batch_size]),shape=(self.batch_size,len(self.result_keys)))
			self.result_keys_emb = embedding_ops.embedding_lookup(self.embeddings, self.result_keys_batch)
			
			self.start_token = tf.constant([0] * self.batch_size, dtype=tf.int32)
			self.out_utt_emb = embedding_ops.embedding_lookup(self.embeddings, self.out_utt)
			self.processed_x = tf.transpose(self.out_utt_emb,perm=[1,0,2])

			with tf.variable_scope("decoder"):
				self.decoder_cell = tf.contrib.rnn.GRUCell(self.dec_hid_dim)
			
			self.h0 = self.inp_utt_rep
			self.g_output_unit = self.create_output_unit()

			gen_x = tensor_array_ops.TensorArray(dtype=tf.int32, size=self.max_out_utt_len,
												 dynamic_size=False, infer_shape=True)

			def _g_recurrence(i, x_t, h_tm1, gen_x):
				_,h_t = self.decoder_cell(x_t,h_tm1)
				o_t = self.g_output_unit(h_t)  # batch x vocab , prob
				next_token = tf.cast(tf.reshape(tf.argmax(o_t, 1), [self.batch_size]), tf.int32)
				x_tp1 = embedding_ops.embedding_lookup(self.embeddings,next_token)  # batch x emb_dim
				gen_x = gen_x.write(i, next_token)  # indices, batch_size
				return i + 1, x_tp1, h_t, gen_x

			_, _, _, self.gen_x = control_flow_ops.while_loop(
				cond=lambda i, _1, _2, _3: i < self.max_out_utt_len,
				body=_g_recurrence,
				loop_vars=(tf.constant(0, dtype=tf.int32),
							embedding_ops.embedding_lookup(self.embeddings,self.start_token),
							self.h0, gen_x))

			self.gen_x = self.gen_x.stack()  # seq_length x batch_size
			self.gen_x = tf.transpose(self.gen_x, perm=[1, 0])  # batch_size x seq_length

			# gen_x contains the colours sampled as outputs
			# Hence, gen_x is used while calculating accuracy

			g_predictions = tensor_array_ops.TensorArray(
				dtype=tf.float32, size=self.max_out_utt_len,
				dynamic_size=False, infer_shape=True)

			ta_emb_x = tensor_array_ops.TensorArray(
				dtype=tf.float32, size=self.max_out_utt_len)
			ta_emb_x = ta_emb_x.unstack(self.processed_x)

			def _train_recurrence(i, x_t, h_tm1, g_predictions):
				_,h_t = self.decoder_cell(x_t,h_tm1)
				o_t = self.g_output_unit(h_t)
				g_predictions = g_predictions.write(i, o_t)  # batch x vocab_size				
				x_tp1 = ta_emb_x.read(i)
				return i + 1, x_tp1, h_t, g_predictions

			_, _, _, self.g_predictions = control_flow_ops.while_loop(
				cond=lambda i, _1, _2, _3: i < self.max_out_utt_len,
				body=_train_recurrence,
				loop_vars=(tf.constant(0, dtype=tf.int32),
						   embedding_ops.embedding_lookup(self.embeddings,self.start_token),
						   self.h0, g_predictions))

			self.g_predictions = tf.transpose(self.g_predictions.stack(), perm=[1, 0, 2])  # batch_size x seq_length x vocab_size
						
			self.loss_mask = tf.sequence_mask(self.out_len,self.max_out_utt_len,dtype=tf.float32)
			self.ground_truth = tf.one_hot(self.out_utt,on_value=tf.constant(1,dtype=tf.float32),off_value=tf.constant(0,dtype=tf.float32),depth=self.out_vocab_size,dtype=tf.float32)
			self.log_predictions = tf.log(self.g_predictions + 1e-20)
			self.cross_entropy = tf.multiply(self.ground_truth,self.log_predictions)
			self.cross_entropy_sum = tf.reduce_sum(self.cross_entropy,2)
			self.masked_cross_entropy = tf.multiply(self.loss_mask,self.cross_entropy_sum)
			self.sentence_loss = tf.divide(tf.reduce_sum(self.masked_cross_entropy,1),tf.reduce_sum(self.loss_mask,1))
			self.loss = -tf.reduce_mean(self.sentence_loss)
	
	def create_output_unit(self):

		self.W1 = tf.get_variable("W1",shape=[2*self.enc_hid_dim+self.dec_hid_dim,2*self.enc_hid_dim],dtype=tf.float32)
		self.W2 = tf.get_variable("W2",shape=[2*self.enc_hid_dim,self.attn_size],dtype=tf.float32)
		self.w = tf.get_variable("w",shape=[self.attn_size,1],dtype=tf.float32)
		self.U = tf.get_variable("U",shape=[self.dec_hid_dim+2*self.enc_hid_dim,self.generate_size],dtype=tf.float32)
		self.W_1 = tf.get_variable("W_1",shape=[self.emb_dim+self.dec_hid_dim+2*self.enc_hid_dim,2*self.dec_hid_dim],dtype=tf.float32)
		self.W_2 = tf.get_variable("W_2",shape=[self.emb_dim+self.dec_hid_dim+2*self.enc_hid_dim,2*self.dec_hid_dim],dtype=tf.float32)
		self.W_3 = tf.get_variable("W_3",shape=[self.emb_dim+self.dec_hid_dim+2*self.enc_hid_dim,2*self.dec_hid_dim],dtype=tf.float32)
		self.W_12 = tf.get_variable("W_12",shape=[2*self.dec_hid_dim,self.attn_size],dtype=tf.float32)
		self.W_22 = tf.get_variable("W_22",shape=[2*self.dec_hid_dim,self.attn_size],dtype=tf.float32)
		self.W_32 = tf.get_variable("W_32",shape=[2*self.dec_hid_dim,self.attn_size],dtype=tf.float32)
		self.r_1 = tf.get_variable("r_1",shape=[self.attn_size,1],dtype=tf.float32)
		self.r_2 = tf.get_variable("r_2",shape=[self.attn_size,1],dtype=tf.float32)
		self.r_3 = tf.get_variable("r_3",shape=[self.attn_size,1],dtype=tf.float32)
		self.b1 = tf.get_variable("b1",shape=[self.generate_size],dtype=tf.float32)
		self.b2 = tf.get_variable("b2",shape=[1],dtype=tf.float32)
		self.b3 = tf.get_variable("b3",shape=[1],dtype=tf.float32)
		self.W3 = tf.get_variable("W3",shape=[self.dec_hid_dim+2*self.enc_hid_dim+self.emb_dim,1],dtype=tf.float32)
		self.W4 = tf.get_variable("W4",shape=[self.dec_hid_dim+2*self.enc_hid_dim+self.emb_dim,1],dtype=tf.float32)
		
		def unit(hidden_state):

			hidden_state_expanded_attn = tf.tile(array_ops.expand_dims(hidden_state,1),[1,tf.shape(self.encoder_states)[1],1])
			attn_rep = tf.concat([self.encoder_states,hidden_state_expanded_attn],axis=2)
			attn_rep = tf.nn.tanh(tf.einsum('ijk,kl->ijl',tf.nn.tanh(tf.einsum("ijk,kl->ijl",attn_rep,self.W1)),self.W2))
			u_i = tf.squeeze(tf.einsum('ijk,kl->ijl',attn_rep,self.w),2)
			inp_len_mask = tf.sequence_mask(self.inp_len,tf.shape(self.inp_utt)[2],dtype=tf.float32)
			attn_mask = tf.reshape(inp_len_mask,shape=[self.batch_size,-1])
			exp_u_i_masked = tf.multiply(tf.cast(attn_mask,dtype=tf.float64),tf.exp(tf.cast(u_i,dtype=tf.float64)))
			a = tf.cast(tf.einsum('i,ij->ij',tf.pow(tf.reduce_sum(exp_u_i_masked,1),-1),exp_u_i_masked),dtype=tf.float32)
			inp_attn = tf.reduce_sum(tf.einsum('ij,ijk->ijk',a,self.encoder_states),1)

			generate_dist = tf.nn.softmax(math_ops.matmul(tf.concat([hidden_state,inp_attn],axis=1),self.U) + self.b1)
			extra_zeros = tf.zeros([self.batch_size,self.out_vocab_size - self.generate_size])
			extended_generate_dist = tf.concat([generate_dist,extra_zeros],axis=1)

			hidden_state_expanded_query = tf.tile(array_ops.expand_dims(hidden_state,1),[1,tf.shape(self.query_mask)[1],1])
			inp_attn_expanded_query = tf.tile(array_ops.expand_dims(inp_attn,1),[1,tf.shape(self.query_mask)[1],1])
			query_attn_rep = tf.concat([self.search_values_rep,hidden_state_expanded_query,inp_attn_expanded_query],axis=2)
			query_attn_rep = tf.nn.tanh(tf.einsum("ijk,kl->ijl",tf.nn.tanh(tf.einsum("ijk,kl->ijl",query_attn_rep,self.W_3)),self.W_32))
			alpha_logits = tf.squeeze(tf.einsum('ijk,kl->ijl',query_attn_rep,self.r_3),2)
			alpha_masked = tf.multiply(tf.cast(self.query_mask,dtype=tf.float64),tf.exp(tf.cast(alpha_logits,dtype=tf.float64)))
			alpha = tf.cast(tf.einsum('i,ij->ij',tf.pow(tf.reduce_sum(alpha_masked,1),-1),alpha_masked),dtype=tf.float32)

			hidden_state_expanded_result = tf.tile(array_ops.expand_dims(array_ops.expand_dims(hidden_state,1),1),[1,tf.shape(self.results_mask)[1],tf.shape(self.results_mask)[2],1])
			inp_attn_expanded_result = tf.tile(array_ops.expand_dims(array_ops.expand_dims(inp_attn,1),1),[1,tf.shape(self.results_mask)[1],tf.shape(self.results_mask)[2],1])
			result_attn_rep = tf.concat([self.result_values_rep,hidden_state_expanded_result,inp_attn_expanded_result],axis=3)
			result_attn_rep = tf.nn.tanh(tf.einsum("ijkl,lm->ijkm",tf.nn.tanh(tf.einsum("ijkl,lm->ijkm",result_attn_rep,self.W_1)),self.W_12))
			beta_logits = tf.squeeze(tf.einsum('ijkl,lm->ijkm',result_attn_rep,self.r_1),3)
			beta_masked = tf.multiply(tf.cast(self.results_mask,dtype=tf.float64),tf.exp(tf.cast(beta_logits,dtype=tf.float64)))
			beta = tf.einsum('ij,ijk->ijk',alpha,tf.cast(tf.einsum('ij,ijk->ijk',tf.pow(tf.reduce_sum(beta_masked,2),-1),beta_masked),dtype=tf.float32))
						
			hidden_state_expanded_keys = tf.tile(array_ops.expand_dims(hidden_state,1),[1,len(self.result_keys),1])
			inp_attn_expanded_keys = tf.tile(array_ops.expand_dims(inp_attn,1),[1,len(self.result_keys),1])
			result_key_rep = tf.concat([self.result_keys_emb,hidden_state_expanded_keys,inp_attn_expanded_keys],axis=2)
			result_key_rep = tf.nn.tanh(tf.einsum("ijk,kl->ijl",tf.nn.tanh(tf.einsum("ijk,kl->ijl",result_key_rep,self.W_2)),self.W_22))
			gamma_logits = tf.squeeze(tf.einsum('ijk,kl->ijl',result_key_rep,self.r_2),2)
			gamma_logits_expanded = tf.tile(array_ops.expand_dims(array_ops.expand_dims(gamma_logits,1),1),[1,tf.shape(self.result_keys_mask)[1],tf.shape(self.result_keys_mask)[2],1])
			gamma_masked = tf.multiply(tf.cast(self.result_keys_mask,dtype=tf.float64),tf.exp(tf.cast(gamma_logits_expanded,dtype=tf.float64)))
			gamma =  tf.einsum('ijk,ijkl->ijkl',beta,tf.cast(tf.einsum('ijk,ijkl->ijkl',tf.pow(tf.reduce_sum(gamma_masked,3),-1),gamma_masked),dtype=tf.float32))
			
			batch_nums_context = array_ops.expand_dims(tf.range(0, limit=self.batch_size, dtype=tf.int64),1)
			batch_nums_tiled_context = tf.tile(batch_nums_context,[1,tf.shape(self.encoder_states)[1]])
			flat_inp_utt = tf.reshape(self.inp_utt,shape=[self.batch_size,-1])
			indices_context = tf.stack([batch_nums_tiled_context,flat_inp_utt],axis=2)
			shape = [self.batch_size,self.out_vocab_size]
			context_copy_dist = tf.scatter_nd(indices_context,a,shape)

			all_betas = tf.reshape(beta,[self.batch_size,-1])
			all_results = tf.reshape(self.result_values_rep,[self.batch_size,-1,self.emb_dim])
			db_rep = tf.reduce_sum(tf.einsum('ij,ijk->ijk',all_betas,all_results),1)
			
			p_db = tf.nn.sigmoid(tf.matmul(tf.concat([hidden_state,inp_attn,db_rep],axis=1),self.W4)+self.b3)
			p_db = tf.tile(p_db,[1,self.out_vocab_size])
			one_minus_fn = lambda x: 1 - x
			one_minus_pdb = tf.map_fn(one_minus_fn, p_db)

			p_gens = tf.nn.sigmoid(tf.matmul(tf.concat([hidden_state,inp_attn,db_rep],axis=1),self.W3)+self.b2)
			p_gens = tf.tile(p_gens,[1,self.out_vocab_size])
			one_minus_fn = lambda x: 1 - x
			one_minus_pgens = tf.map_fn(one_minus_fn, p_gens)
			
			batch_nums = array_ops.expand_dims(tf.range(0, limit=self.batch_size, dtype=tf.int64),1)
			kb_ids = tf.reshape(self.result_values,shape=[self.batch_size,-1])
			num_kb_ids = tf.shape(kb_ids)[1]
			batch_nums_tiled = tf.tile(batch_nums,[1,num_kb_ids])
			indices = tf.stack([batch_nums_tiled,kb_ids],axis=2)
			updates = tf.reshape(gamma,shape=[self.batch_size,-1])
			shape = [self.batch_size,self.out_vocab_size]
			kb_dist = tf.scatter_nd(indices,updates,shape)
			kb_dist = tf.einsum('i,ij->ij',self.db_empty,kb_dist)

			copy_dist = tf.multiply(p_db,kb_dist) + tf.multiply(one_minus_pdb,context_copy_dist)			
			final_dist = tf.multiply(p_gens,extended_generate_dist) + tf.multiply(one_minus_pgens,copy_dist)

			return final_dist

		return unit

	def get_feed_dict(self,batch):

		fd = {
			self.inp_utt : batch['inp_utt'],
			self.inp_len : batch['inp_len'],
			self.out_utt : batch['out_utt'],
			self.out_len : batch['out_len'],
			self.context_len: batch['context_len'],
			self.query_mask : batch['query_mask'],
			self.search_mask : batch['search_mask'],
			self.search_values : batch['search_values'],
			self.results_mask : batch['results_mask'],
			self.result_keys_mask : batch['result_keys_mask'],
			self.result_values : batch['result_values'],
			self.db_empty : batch['empty'],
			self.max_out_utt_len : batch['max_out_utt_len']
		}
		
		return fd	
