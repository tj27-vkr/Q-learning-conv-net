import tensorflow as tf
import cv2
import pickle
import random
import numpy as np
from collections import deque
import sys
sys.path.append("environments/")
import flappy_bird as environment
CHECKPOINT = "checkpoints"
MEMORY_FILE = "q_table_memory.pkl"

#import trex as environment
#CHECKPOINT = "checkpoint_trex"
#MEMORY_FILE = "q_table_memory_trex.pkl"
N_ACTION = 2
LAMBDA = 1e-6
FINAL_EPSILON = 0.0001

INITIAL_EPSILON = 0.0001
EXPLORE = 200000.0
DATA_COLLECTION_CNT = 10000.0
MEMORY_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.99

def conv2d(input, weight, stride):
	return tf.nn.conv2d(input, weight, strides = [1, stride, stride, 1], padding = 'SAME')
	
def create_model(model):
	
	weights = {
		'conv1': tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.01)),
		'conv2': tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.01)),
		'conv3': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01)),
		'fc4': tf.Variable(tf.truncated_normal([1600, 512], stddev = 0.01)),
		'fc5': tf.Variable(tf.truncated_normal([512, N_ACTION], stddev = 0.01))		
		}
	biases = {
		'conv1': tf.Variable(tf.constant(0.01, shape = [32])),
		'conv2': tf.Variable(tf.constant(0.01, shape = [64])),
		'conv3': tf.Variable(tf.constant(0.01, shape = [64])),
		'fc4': tf.Variable(tf.constant(0.01, shape = [512])),
		'fc5': tf.Variable(tf.constant(0.01, shape = [N_ACTION]))
		}
		
	state = tf.placeholder("float",[None, 80, 80, 4])
	
	conv1 = tf.nn.relu(conv2d(state, weights['conv1'], 4) + biases['conv1'])
	pool1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding="SAME")
	
	conv2 = tf.nn.relu(conv2d(pool1, weights['conv2'], 2) + biases['conv2'])
	
	conv3 = tf.nn.relu(conv2d(conv2, weights['conv3'], 1) + biases['conv3'])
	
	flat_conv3 = tf.reshape(conv3, [-1,1600])
	fc4 = tf.nn.relu(tf.matmul(flat_conv3, weights['fc4']) + biases['fc4'])
	
	fc5 = tf.matmul(fc4, weights['fc5']) + biases['fc5']
	
	return state, fc5
	
def start_learning(state, rewards, model):
	
	action = tf.placeholder("float", [None, N_ACTION])
	actual_out = tf.placeholder("float", [None])
	rewards_out = tf.reduce_sum(tf.multiply(rewards, action), reduction_indices = 1)
	error = tf.reduce_mean(tf.square(actual_out - rewards_out))
	trainer = tf.train.AdamOptimizer(LAMBDA).minimize(error)
	
	memory = deque()
	
	
	env = environment.GameState()
	
	
	null_action = np.zeros(N_ACTION)
	null_action[0] = 1
	
	frame, reward, terminal = env.frame_step(null_action)
	frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
	ret, frame = cv2.threshold(frame,1,255,cv2.THRESH_BINARY)
	s = np.stack((frame, frame, frame, frame), axis=2)

	saved_model = tf.train.Saver()
	model.run(tf.initialize_all_variables())
	checkpoint = tf.train.get_checkpoint_state(CHECKPOINT)
	if checkpoint and checkpoint.model_checkpoint_path:
		print("loading..")
		with open(MEMORY_FILE, "rb") as fd:
			memory = pickle.load(fd)
		saved_model.restore(model, checkpoint.model_checkpoint_path)
		print("model loaded.")
		
	tr = input("Press key to continue..")
	eps = INITIAL_EPSILON
	steps = 0
	
	while True:
		a_out = rewards.eval(feed_dict={state: [s]})[0]
		a = np.zeros([N_ACTION])
		
		if random.random() <= eps:
			print("Random choice")
			a[random.randrange(N_ACTION)] = 1
		else:
			print("Predicted choice")
			a[np.argmax(a_out)] = 1
			
		#a[1] = 1
		
		if eps > FINAL_EPSILON and steps > DATA_COLLECTION_CNT:
			eps -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
		
		frame1, r, terminal = env.frame_step(a)
		
		frame1 = cv2.cvtColor(cv2.resize(frame1, (80, 80)), cv2.COLOR_BGR2GRAY)
		ret, frame1 = cv2.threshold(frame1, 1, 255, cv2.THRESH_BINARY)
		frame1 = np.reshape(frame1, (80, 80, 1))
		s_ = np.append(frame1, s[:, :, :3], axis=2)
		
		memory.append((s, a, r, s_, terminal))
		
			
		if len(memory) > MEMORY_SIZE:
			memory.popleft()
			
		if steps > DATA_COLLECTION_CNT:
			
			batch = random.sample(memory, BATCH_SIZE)
			
			s1 = [i[0] for i in batch]
			a1 = [i[1] for i in batch]
			r1 = [i[2] for i in batch]
			s_1 = [i[3] for i in batch]
			
			y1 = []
			a_out1 = rewards.eval(feed_dict = {state: s_1})
			
			for iteration in range(0,len(batch)):
				terminal = batch[iteration][4]
				if terminal:
					y1.append(r1[iteration])
				else:
					y1.append(r1[iteration] + GAMMA * np.max(a_out1[iteration]))
					
			trainer.run(feed_dict = {actual_out: y1, action: a1, state: s1 })
		s = s_
		steps += 1
		if steps % 10000 == 0:
			with open(MEMORY_FILE, "wb") as fd:
				pickle.dump(memory, fd)
			saved_model.save(model, CHECKPOINT+'/trained_data', global_step = steps)
		print("Step:{}\tReward:{}".format(str(steps), str(r)))
	
	
if __name__ == '__main__':
	model = tf.InteractiveSession()
	state, rewards = create_model(model)
	start_learning(state, rewards, model)