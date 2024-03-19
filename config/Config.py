#coding:utf-8
import numpy as np
import tensorflow as tf

import os
import time
import datetime
from tqdm import tqdm
import ctypes
import json
import click
from collections import deque

from scipy import sparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class Config(object):
	'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):
		base_file = os.path.abspath('./release/Base.so')
		self.lib = ctypes.cdll.LoadLibrary(base_file)
		self.lib.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.testHead.argtypes = [ctypes.c_void_p]
		self.lib.testTail.argtypes = [ctypes.c_void_p]
		self.lib.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 0
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 1
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.optimizer = None
		self.test_link_prediction = False
		self.test_triple_classification = False
		self.early_stopping = None # It expects a tuple of the following: (patience, min_delta)
		self.training_log = []
		self.score_norm = None

		self.classify_scores = None
		self.classify_relations = None
		self.classify_classes = None



	def init_link_prediction(self):
		r'''
		import essential files and set essential interfaces for link prediction
		'''
		self.lib.importTestFiles()
		self.lib.importTypeFiles()
		self.test_h = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_t = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_r = np.zeros(self.lib.getEntityTotal(), dtype = np.int64)
		self.test_h_addr = self.test_h.__array_interface__['data'][0]
		self.test_t_addr = self.test_t.__array_interface__['data'][0]
		self.test_r_addr = self.test_r.__array_interface__['data'][0]

	def init_triple_classification(self):
		r'''
		import essential files and set essential interfaces for triple classification
		'''
		self.lib.importTestFiles()
		self.lib.importTypeFiles()

		self.test_pos_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_h = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_t = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_neg_r = np.zeros(self.lib.getTestTotal(), dtype = np.int64)
		self.test_pos_h_addr = self.test_pos_h.__array_interface__['data'][0]
		self.test_pos_t_addr = self.test_pos_t.__array_interface__['data'][0]
		self.test_pos_r_addr = self.test_pos_r.__array_interface__['data'][0]
		self.test_neg_h_addr = self.test_neg_h.__array_interface__['data'][0]
		self.test_neg_t_addr = self.test_neg_t.__array_interface__['data'][0]
		self.test_neg_r_addr = self.test_neg_r.__array_interface__['data'][0]

		self.valid_pos_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_h = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_t = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_neg_r = np.zeros(self.lib.getValidTotal(), dtype = np.int64)
		self.valid_pos_h_addr = self.valid_pos_h.__array_interface__['data'][0]
		self.valid_pos_t_addr = self.valid_pos_t.__array_interface__['data'][0]
		self.valid_pos_r_addr = self.valid_pos_r.__array_interface__['data'][0]
		self.valid_neg_h_addr = self.valid_neg_h.__array_interface__['data'][0]
		self.valid_neg_t_addr = self.valid_neg_t.__array_interface__['data'][0]
		self.valid_neg_r_addr = self.valid_neg_r.__array_interface__['data'][0]
		self.relThresh = np.zeros(self.lib.getRelationTotal(), dtype = np.float32)
		self.relThresh_addr = self.relThresh.__array_interface__['data'][0]

	# prepare for train and test
	def init(self):
		self.trainModel = None
		if self.in_path != None:
			self.lib.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib.setBern(self.bern)
			self.lib.setWorkThreads(self.workThreads)
			self.lib.randReset()
			self.lib.importTrainFiles()
			self.relTotal = self.lib.getRelationTotal()
			self.entTotal = self.lib.getEntityTotal()
			self.trainTotal = self.lib.getTrainTotal()
			self.testTotal = self.lib.getTestTotal()
			self.validTotal = self.lib.getValidTotal()
			self.batch_size = int(self.lib.getTrainTotal() / self.nbatches)
			self.batch_seq_size = self.batch_size * (1 + self.negative_ent + self.negative_rel)
			self.batch_h = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_t = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y = np.zeros(self.batch_size * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr = self.batch_h.__array_interface__['data'][0]
			self.batch_t_addr = self.batch_t.__array_interface__['data'][0]
			self.batch_r_addr = self.batch_r.__array_interface__['data'][0]
			self.batch_y_addr = self.batch_y.__array_interface__['data'][0]
			# self.set_score_norm('l2')
		if self.test_link_prediction:
			self.init_link_prediction()
		if self.test_triple_classification:
			self.init_triple_classification()

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_score_norm(self, score_norm):
		self.score_norm = score_norm

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path, steps = 0):
		self.exportName = path
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	def set_early_stopping(self, early_stopping):
		self.early_stopping = early_stopping

	# call C function for sampling
	def sampling(self):
		self.lib.sampling(self.batch_h_addr, self.batch_t_addr, self.batch_r_addr, self.batch_y_addr, self.batch_size, self.negative_ent, self.negative_rel)

	# save model
	def save_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.save(self.sess, self.exportName)
	# restore model
	def restore_tensorflow(self):
		with self.graph.as_default():
			with self.sess.as_default():
				self.saver.restore(self.sess, self.importName)


	def export_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.save(self.sess, self.exportName)
				else:
					self.saver.save(self.sess, path)

	def import_variables(self, path = None):
		with self.graph.as_default():
			with self.sess.as_default():
				if path == None:
					self.saver.restore(self.sess, self.importName)
				else:
					self.saver.restore(self.sess, path)

	def get_parameter_lists(self):
		return self.trainModel.parameter_lists

	def get_parameters_by_name(self, var_name):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					return self.sess.run(self.trainModel.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists = self.get_parameter_lists()
		for var_name in lists:
			if mode == "numpy":
				res[var_name] = self.get_parameters_by_name(var_name)
			else:
				res[var_name] = self.get_parameters_by_name(var_name).tolist()
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model = model
		self.graph = tf.Graph()
		with self.graph.as_default():
			self.sess = tf.Session()
			with self.sess.as_default():
				initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel = self.model(config = self)
					if self.optimizer != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer.compute_gradients(self.trainModel.loss)
					self.train_op = self.optimizer.apply_gradients(grads_and_vars)
				self.saver = tf.train.Saver()
				#self.sess.run(tf.initialize_all_variables())
				self.sess.run(tf.global_variables_initializer())

	def train_step(self, batch_h, batch_t, batch_r, batch_y):
		feed_dict = {
			self.trainModel.batch_h: batch_h,
			self.trainModel.batch_t: batch_t,
			self.trainModel.batch_r: batch_r,
			self.trainModel.batch_y: batch_y
		}
		_, loss = self.sess.run([self.train_op, self.trainModel.loss], feed_dict)
		return loss

	def test_step(self, test_h, test_t, test_r):
		
		feed_dict = {
			self.trainModel.predict_h: test_h,
			self.trainModel.predict_t: test_t,
			self.trainModel.predict_r: test_r,
		}
		predict = self.sess.run(self.trainModel.predict, feed_dict)

		return predict

	def run(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				if self.early_stopping is not None:
					patience, min_delta = self.early_stopping
					best_loss = np.finfo('float32').max
					wait_steps = 0
				with click.progressbar(range(self.train_times)) as total_times:
					for times in total_times:
						loss = 0.0
						t_init = time.time()
						for batch in range(self.nbatches):
							self.sampling()
							loss += self.train_step(self.batch_h, self.batch_t, self.batch_r, self.batch_y)
						t_end = time.time()
						self.training_log.append([times, loss, (t_end - t_init)])
						# if self.log_on:
						# 	print('Epoch: {}, loss: {}, time: {}'.format(times, loss, (t_end - t_init)))
						if self.exportName != None and (self.export_steps!=0 and times % self.export_steps == 0):
							self.save_tensorflow()
						if self.early_stopping is not None:
							if loss + min_delta < best_loss:
								best_loss = loss
								wait_steps = 0
							elif wait_steps < patience:
								wait_steps += 1
							else:
								print('Early stopping. Losses have not been improved enough in {} times'.format(patience))
								break
				if self.exportName != None:
					self.save_tensorflow()
				if self.out_path != None:
					self.save_parameters(self.out_path)

	def test(self):
		with self.graph.as_default():
			with self.sess.as_default():
				if self.importName != None:
					self.restore_tensorflow()
				start_time = time.time()
				if self.test_link_prediction:
					total = self.lib.getTestTotal()
					with click.progressbar(range(total)) as total_times:
						for times in total_times:
							self.lib.getHeadBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
							res = self.test_step(self.test_h, self.test_t, self.test_r)
							self.lib.testHead(res.__array_interface__['data'][0])

							self.lib.getTailBatch(self.test_h_addr, self.test_t_addr, self.test_r_addr)
							res = self.test_step(self.test_h, self.test_t, self.test_r)
							self.lib.testTail(res.__array_interface__['data'][0])
							# if self.log_on:
							# 	print(times)
					print('Test Results Summary:')
					self.lib.test_link_prediction()
				if self.test_triple_classification:
					self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
					res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
					res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
					self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

					self.lib.getTestBatch(self.test_pos_h_addr, self.test_pos_t_addr, self.test_pos_r_addr, self.test_neg_h_addr, self.test_neg_t_addr, self.test_neg_r_addr)

					res_pos = self.test_step(self.test_pos_h, self.test_pos_t, self.test_pos_r)
					res_neg = self.test_step(self.test_neg_h, self.test_neg_t, self.test_neg_r)
					self.lib.test_triple_classification(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

	def	calculate_thresholds_ar(self):
		
		
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr,self.valid_neg_r_addr)
		res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
		
		return True
	
	
	def predict_head_entity(self, t, r, k):
		r'''This mothod predicts the top k head entities given tail entity and relation.
		
		Args: 
			t (int): tail entity id
			r (int): relation id
			k (int): top k head entities
		
		Returns:
			list: k possible head entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array(range(self.entTotal))
		test_r = np.array([r] * self.entTotal)
		test_t = np.array([t] * self.entTotal)
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]

		return res


	def predict_head_entity_scores(self, t, r):
		r'''This mothod predicts the top k head entities given tail entity and relation.
		
		Args: 
			t (int): tail entity id
			r (int): relation id
			k (int): top k head entities
		
		Returns:
			list: k possible head entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array(range(self.entTotal))
		test_r = np.array([r] * self.entTotal)
		test_t = np.array([t] * self.entTotal)
		score = self.test_step(test_h, test_t, test_r).reshape(-1)

		return score

	def predict_tail_entity(self, h, r, k):
		r'''This mothod predicts the top k tail entities given head entity and relation.
		
		Args: 
			h (int): head entity id
			r (int): relation id
			k (int): top k tail entities
		
		Returns:
			list: k possible tail entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array([h] * self.entTotal)
		test_r = np.array([r] * self.entTotal)
		test_t = np.array(range(self.entTotal))
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]

		return res

	def predict_tail_entity_scores(self, h, r):
		r'''This mothod predicts the top k tail entities given head entity and relation.
		
		Args: 
			h (int): head entity id
			r (int): relation id
			k (int): top k tail entities
		
		Returns:
			list: k possible tail entity ids 	  	
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array([h] * self.entTotal)
		test_r = np.array([r] * self.entTotal)
		test_t = np.array(range(self.entTotal))
		score = self.test_step(test_h, test_t, test_r).reshape(-1)

		return score

	def predict_relation(self, h, t, k):
		r'''This methods predict the relation id given head entity and tail entity.
		
		Args:
			h (int): head entity id
			t (int): tail entity id
			k (int): top k relations
		
		Returns:
			list: k possible relation ids
		'''
		self.init_link_prediction()
		if self.importName != None:
			self.restore_tensorflow()
		test_h = np.array([h] * self.relTotal)
		test_r = np.array(range(self.relTotal))
		test_t = np.array([t] * self.relTotal)
		res = self.test_step(test_h, test_t, test_r).reshape(-1).argsort()[:k]
		print(res)
		return res

	def predict_triple(self, h, t, r, thresh = None):
		r'''This method tells you whether the given triple (h, t, r) is correct of wrong
	
		Args:
			h (int): head entity id
			t (int): tail entity id
			r (int): relation id
			thresh (fload): threshold for the triple
		'''
		self.init_triple_classification()
		if self.importName != None:
			self.restore_tensorflow()
		res = self.test_step(np.array([h]), np.array([t]), np.array([r]))
		if thresh != None:
			if res < thresh:
				print("triple (%d,%d,%d) is correct" % (h, t, r))
			else:
				print("triple (%d,%d,%d) is wrong" % (h, t, r))
			return
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
		res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])
		print('Treshold is {}'.format(self.relThresh[r]))
		print('Score {}'.format(res[0]))
		if res < self.relThresh[r]:
			print("triple (%d,%d,%d) is correct" % (h, t, r))
		else:
			print("triple (%d,%d,%d) is wrong" % (h, t, r))
			

	def get_true_tails(self, h, r, thresh=False):
		'''This function gets a head (or a list of heads) and returns
		all true tails as a list'''
		#self.init_link_prediction()

		h = [h]
		n = len(h)

		if self.importName != None:
			self.restore_tensorflow()

		if not thresh:
			thresh = self.relThresh[r]

		test_r = np.array([r] * self.entTotal * n)
		test_t = np.array(list(range(self.entTotal)) * n) 

		test_h = np.array([h[0]] * self.entTotal)

		for i in range(1, n):
			test_h = np.concatenate([test_h, np.array([h[i]] * self.entTotal)])
		
		score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
		true_tails = test_t[score < thresh]

		return true_tails

	def get_true_tails_np(self, h, r, thresh=False):
		'''This function gets a head (or a list of heads) and returns
		all true tails as a list'''
		#self.init_link_prediction()

		if self.importName != None:
			self.restore_tensorflow()

		if not thresh:
			thresh = self.relThresh[r]

		true_tails = np.zeros(self.entTotal, dtype=np.bool)

		test_r = np.array([r] * self.entTotal)
		
		test_t = np.array(list(range(self.entTotal))) 

		for head in h:
			test_h = np.array([head] * self.entTotal)
			score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			true_tails += score < thresh

		return true_tails


	def build_emb_rel_matrix(self, heads, tails, r, thresh=False):
		'''This function gets a head (or a list of heads) and returns
		all true tails as a sparse lil.matrix'''
		#self.init_link_prediction()

		# print(f'Threshold for relation {r}: {thresh}')

		if self.importName != None:
			self.restore_tensorflow()

		if not thresh:
			# print(f'Loading threshold from embedding for relation {r} / {thresh}')
			thresh = self.relThresh[r]

		out = np.zeros(shape=(self.entTotal, self.entTotal), dtype=np.bool)

		test_r = np.array([r] * self.entTotal)
		
		test_t = np.array(list(range(self.entTotal))) 

		for head in heads:
			test_h = np.array([head] * self.entTotal)
			score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			out[head] = score < thresh

		test_h = np.array(list(range(self.entTotal)))

		for tail in tails:
			test_t = np.array([tail] * self.entTotal)
			score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			out[:, tail] += score < thresh

		out = sparse.lil_matrix(out)

		return out

	def build_rel_ghat(self, r, thresh=False):
		'''This function gets a head (or a list of heads) and returns
		all true tails as a list'''
		#self.init_link_prediction()

		if self.importName != None:
			self.restore_tensorflow()

		if not thresh:
			thresh = self.relThresh[r]

		true_tails = np.zeros(self.entTotal, dtype=np.bool)

		test_r = np.array([r] * self.entTotal)
		
		test_t = np.array(list(range(self.entTotal))) 

		h = np.array(list(range(self.entTotal))) 

		for head in tqdm(h):
			test_h = np.array([head] * self.entTotal)
			score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			true_tails += score < thresh

		return true_tails

	def get_true_heads(self, t, r, thresh):
		
		t = [t]
		n = len(t)

		if self.importName != None:
			self.restore_tensorflow()

		if not thresh:
			thresh = self.relThresh[r]

		test_r = np.array([r] * self.entTotal * n)
		test_h = np.array(list(range(self.entTotal)) * n)

		test_t = np.array([t[0]] * self.entTotal)

		for i in range(1, n):
			test_t = np.concatenate([test_t, np.array([t[i]] * self.entTotal)])

		score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
		true_heads = test_h[score < thresh]

		return true_heads
		
	def get_true_heads_np(self, t, r, thresh=False):
		
		if self.importName != None:
			self.restore_tensorflow()

		if not thresh:
			thresh = self.relThresh[r]

		true_heads = np.zeros(self.entTotal, dtype=np.bool)

		test_r = np.array([r] * self.entTotal)
		test_h = np.array(list(range(self.entTotal)))

		for tail in t:

			test_t = np.array([tail] * self.entTotal)
			score = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			true_heads += score < thresh

		return true_heads


	def build_tailpred_set(self, heads, rels, k = 0):
		'''This function gets a list of heads and relations and builds
		all positive triples returning the score value for each
		triple'''
		
		#self.init_link_prediction()


		if self.importName != None:
			self.restore_tensorflow()

		test_t = np.array(range(self.entTotal))

		out_heads, out_rels, out_tails, out_score = [], [], [], []

		counter = 0

		for h, r in zip(heads, rels):

			test_h = np.array([h] * self.entTotal)
			test_r = np.array([r] * self.entTotal)
			
			res = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			
			if k == 0:
				true_triples = (res < self.relThresh[r]).sum()
			else:
				true_triples = k

			tails = list(res.argsort()[:true_triples])
			score = list(np.msort(res))[:true_triples]
		
			out_heads = out_heads + len(tails) * [h]
			out_tails = out_tails + tails
			out_rels = out_rels + len(tails) * [r]
			out_score = out_score + score

			counter += 1
			if counter % 1000 == 0:
				print('Testing Triples {} done. Found {} new facts so far!'.format(counter, len(out_heads)))

		return out_heads, out_rels, out_tails, out_score

	def test_t_link_prediciton(self, heads, tails, rels):
		'''This function gets a list of heads, tails and relations and returns the rank of 
		the correct entity as well as the score. All inputs and outputs are in the form
		of vectors to optimize tensorflow initialization'''
		
		#self.init_link_prediction()


		if self.importName != None:
			self.restore_tensorflow()

		test_t = np.array(range(self.entTotal))

		out_rank, out_score = [], []
		
		for h, t, r in zip(heads, tails, rels):

			test_h = np.array([h] * self.entTotal)
			test_r = np.array([r] * self.entTotal)
			
			res = np.squeeze(self.test_step(test_h, test_t, test_r).reshape(-1))
			out_score.append(res[t])

			out_rank.append(list(res.argsort()).index(t)+1)
		

		return out_rank, out_score

	def classify_triples(self, test_h, test_t, test_r):
		
		self.init_triple_classification()
		
		# Initialize Relation Thresholds
		# self.lib.importTestFiles()
		# self.lib.importTypeFiles()
		# self.init_triple_classification()
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr, self.valid_neg_r_addr)
		res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

		if self.importName != None:
			self.restore_tensorflow()
			
		test_h = np.array(test_h)
		test_r = np.array(test_r)
		test_t = np.array(test_t)

		thresholds = np.zeros(len(test_r))
		for i in range(len(thresholds)):
			thresholds[i] = self.relThresh[test_r[i]]

		res = np.squeeze(self.test_step(test_h.astype(np.int), test_t.astype(np.int), test_r).reshape(-1))
		true_triples = res < thresholds

		return true_triples, res

	def build_paths(self, head, tail, path_sequence, k = 10000):
		'''This function try to rebuild path types within the embedding model.
		It takes as argument the head, tail and path_sequence (of the form rel1_rel2...)
		and returns True if tail node is found at the last step.
		'''

		path_sequence = path_sequence.split('_')
		lenght = len(path_sequence)
		i = 0
		nodes = [[head]]

		while i < lenght:
			rel = path_sequence[i]
			# print('Checking relation {}'.format(rel))
			if rel[0] == 'i':
				rel = rel[1:]
				nodes.append(set(self.get_true_heads(nodes[i], int(rel))[:k]))
			else:
				nodes.append(set(self.get_true_tails(nodes[i], int(rel))[:k]))
			# print('Size of the nodes[{}] is {}'.format(i+1, len(nodes[i+1])))
			i += 1
		
		if tail in nodes[i]:
			return True
		else:
			return False

	def build_paths2(self, head, tail, path_sequence):
		'''This function try to rebuild path types within the embedding model.
		It takes as argument the head, tail and path_sequence (of the form rel1_rel2...)
		and returns True if tail node is found at the last step.
		'''

		path_sequence = path_sequence.split('_')
		
		left_path = path_sequence[ : len(path_sequence) - len(path_sequence)//2 ]
		right_path = path_sequence[len(path_sequence) - len(path_sequence)//2 :]

		i = 0
		left_nodes = [set([head])]

		while i < len(left_path):
			rel = path_sequence[i]
			# print('Checking relation {}'.format(rel))
			if rel[0] == 'i':
				rel = rel[1:]
				left_nodes.append(set(self.get_true_heads(left_nodes[i], int(rel))) - left_nodes[i])
			else:
				left_nodes.append(set(self.get_true_tails(left_nodes[i], int(rel))) - left_nodes[i])
			# print('Size of the nodes[{}] is {}'.format(i+1, len(nodes[i+1])))
			i += 1
		
		i = 0
		right_nodes = [set([tail])]

		while i < len(right_path):
			rel = path_sequence[i]
			# print('Checking relation {}'.format(rel))
			if rel[0] != 'i':
				right_nodes.append(set(self.get_true_heads(right_nodes[i], int(rel))) - right_nodes[i])
			else:
				rel = rel[1:]
				right_nodes.append(set(self.get_true_tails(right_nodes[i], int(rel))) - right_nodes[i])
			# print('Size of the nodes[{}] is {}'.format(i+1, len(nodes[i+1])))
			i += 1

		if not left_nodes[-1].isdisjoint(right_nodes[-1]):
			return True
		else:
			return False	

############################################################

	def optimized_node_expansion(self, nodes, rel, rel_thresh):


		h = np.array([nodes[1]] * self.entTotal)
		r = np.array([rel] * self.entTotal)
		t = np.arange(self.entTotal)

		predict_h = tf.placeholder(tf.int64)
		predict_t = tf.placeholder(tf.int64)
		predict_r = tf.placeholder(tf.int64)

		with tf.Session() as sess:

			feed_dict = {
				predict_h: h,
				predict_t: r,
				predict_r: t,
		}
			sess.run(self.trainModel.predict, feed_dict).reshape(-1)
			
		print(f'Heads with {len(h)} elements!')

		print('Hello world, Andrey!')

	def calculate_thresholds(self):
		self.lib.getValidBatch(self.valid_pos_h_addr, self.valid_pos_t_addr, self.valid_pos_r_addr, self.valid_neg_h_addr, self.valid_neg_t_addr,self.valid_neg_r_addr)
		res_pos = self.test_step(self.valid_pos_h, self.valid_pos_t, self.valid_pos_r)
		res_neg = self.test_step(self.valid_neg_h, self.valid_neg_t, self.valid_neg_r)
		self.lib.getBestThreshold(self.relThresh_addr, res_pos.__array_interface__['data'][0], res_neg.__array_interface__['data'][0])

	def get_threshold_for_relation(self, r):
		
		return self.relThresh[r]

	def get_threshold_list_for_relations(self, rels):
		'''Returns a dict whose keys are relation indexes and values are the respective threshold.
		Arguments:
		- rels: a numpy array of relations.
		'''
		self.calculate_thresholds()
		unique_rels = np.unique(rels)
		thres_list = []
		for r in unique_rels:
			thres_list.append(self.get_threshold_for_relation(r))
		return thres_list


	# def setup_classify_graph(self, thresh):
	# 	"""Returns the classification of a set of triples, using the validation threshold."""
		
	# 	self.classify_scores = tf.placeholder(tf.float32, [None])
	# 	# threshold = tf.constant(thresh, tf.float32)
	# 	self.classify_classes = tf.less(self.classify_scores, threshold_rel)


	def classify(self, head, rel, thresh, batch_size=1, update_thres=False):
		"""Runs the graph created by classify setup.
		"""

		if self.importName != None:
			self.restore_tensorflow()

		heads = np.array([head] * self.entTotal)
		rels = np.array([rel] * self.entTotal)
		tails = np.array(list(range(self.entTotal)))

		thresh = self.relThresh[rel]

		# print(f'heads:{len(heads)} tails:{len(tails)} rels:{len(rels)}')

		# print(f'thresh is {thresh}')

		threshold_rel = tf.placeholder(tf.float32)
		classify_classes = tf.placeholder(tf.bool, [None])
		classify_scores = tf.placeholder(tf.float32, [None])
		classify_classes = tf.less(classify_scores, threshold_rel)

		with tf.Session() as sess:

				res = sess.run(classify_classes, feed_dict={
					classify_scores: self.test_step(heads, tails, rels).reshape(-1),
					threshold_rel: thresh
					})

				# print(f'res is {res}')
		


		return np.nonzero(res)[0].tolist()

	def enhanced_true_tails(self, head, rel, thresh, batch_size = 100):

		if self.importName != None:
			self.restore_tensorflow()

		if len(head) % batch_size == 0:
			total_iters = len(head) / batch_size
		else:
			total_iters = int(len(head) / batch_size) + 1


		r = np.array([rel])
		n_ents = self.entTotal



		with self.graph.as_default():
			with self.sess.as_default():
				
				predict = np.zeros(self.entTotal, dtype=np.bool)

				for i in range(total_iters):

					start = (i) * batch_size
					end = (i+1) * batch_size

					h = np.array(head[start:end])

					t_h = tf.constant(h)
					t_ents = tf.constant([n_ents])
					t_heads = tf.constant([1, len(h)])

					t_hh = tf.reshape(tf.tile(t_h, t_ents), [n_ents, len(h)])
					t_rr = tf.constant(r, tf.int64, shape=(n_ents, len(h)))
					t_tt = tf.tile(tf.reshape(tf.range(start=0, limit=n_ents, dtype=tf.int64), [n_ents, 1]), t_heads)

					predict_h_e = tf.nn.embedding_lookup(self.trainModel.ent_embeddings, t_hh)
					predict_t_e = tf.nn.embedding_lookup(self.trainModel.ent_embeddings, t_tt)
					predict_r_e = tf.nn.embedding_lookup(self.trainModel.rel_embeddings, t_rr)

					scores = tf.reduce_mean(self.trainModel._calc(predict_h_e, predict_t_e, predict_r_e), 2, keepdims = False)

					predict += self.sess.run(tf.math.greater(tf.reduce_sum(tf.cast(tf.less(scores, thresh), tf.int64), 1), tf.constant([0], tf.int64)))

		return predict


	def enhanced_true_tails_q(self, heads, rel, thresh):

		if self.importName != None:
			self.restore_tensorflow()

		n_ents = self.entTotal

		tf.reset_default_graph()

		def embs(t_tt, t_rr):
			with tf.name_scope("embs"):
				if not hasattr(embs, "emb_t"):
					predict_t_e = tf.nn.embedding_lookup(self.trainModel.ent_embeddings, t_tt, name="emb_t")
				if not hasattr(embs, "emb_h"):
					predict_r_e = tf.nn.embedding_lookup(self.trainModel.rel_embeddings, t_rr, name="emb_h")
			return predict_t_e, predict_r_e

		with self.graph.as_default():
			with self.sess.as_default():

				q = tf.FIFOQueue(capacity=10, dtypes=tf.int64, shapes=[])
				enqueue_op = q.enqueue_many(np.array(heads))

				qr = tf.train.QueueRunner(q, [enqueue_op] * 1)
				tf.train.add_queue_runner(qr)

				h = tf.reshape(q.dequeue(), shape=(1,))
				t_ents = tf.constant([n_ents])

				t_hh = tf.tile(h, t_ents)
				t_rr = tf.constant(rel, tf.int64, shape=(n_ents,))
				t_tt = tf.range(start=0, limit=n_ents, dtype=tf.int64)

				predict_h_e = tf.nn.embedding_lookup(self.trainModel.ent_embeddings, t_hh)

				predict_t_e, predict_r_e = embs(t_tt, t_rr)

				scores = tf.reduce_mean(self.trainModel._calc(predict_h_e, predict_t_e, predict_r_e), 1, keepdims = False)

				predict = []
					
				coord = tf.train.Coordinator()
				threads = tf.train.start_queue_runners(coord=coord)

				for _ in range(len(heads)):
					predict += self.sess.run(tf.where(tf.less(scores, thresh))).reshape(1, -1)[0].tolist()

				coord.request_stop()
				coord.join(threads)

		return predict