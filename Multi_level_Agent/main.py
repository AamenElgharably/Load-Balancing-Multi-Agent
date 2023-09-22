import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import gym
from ReplayBuffer import ReplayBuffer
from Networks import CriticNetwork,ActorNetwork
from ns3gym import ns3env
from Multi_level_env import multi_level_env
from TD3_Agent import TD3_Agent
"""THIS Code will be split into 3 parts 
PART1 WILL BE Training the First Agent only(ie,All other possible CIO VAlues outside of the Agent Range will be 0)
PART2 WILL BE Training the Sencond Agent With Agent 1 Values (ie, Actions will from the First concatenated with the second one others will be zeros)
PART3 WILL BE training the Super Agent with action from lower level agents
"""





#First PART
def main():
	Episodes=100
	Agent_ID=1 #THIS should be Consistant with the NS3
	env=multi_level_env(Agent_ID)
	agent=TD3_Agent(0.001,0.001,input_dims=[env.state_dim,],tau=0.005,gamma=0.99,update_actor_interval=2,
		n_actions=env.action_dim,max_size=1000,layer1_size=24,layer2_size=24,batch_size=32,noise=0.5,ID="Level1-1",warmup=200)
	try:
		last_episode=np.load('Episode.npz')['array1'][0]
		agent.load_models()
		agent.load_state()
		print("last Run states are loaded")
	except:
		print("No last episode to be loaded")
		last_episode=0
	
	steps=env.env._max_episode_steps
	#training loop:
	for i in range(last_episode,Episodes):
		state=env.reset()
		for j in range(steps):
			action=agent.choose_action(state)
			next_state,reward,done,info=env.step(action)
			print(f"Episode:{i} Step:{j} Action:{action} Agent_ID:{Agent_ID} Reward:{reward}")
			agent.remember(state,action,reward,next_state,done)
			agent.learn()
			state=next_state
		if i%10==0:
			print("...........saving............")
			agent.save_models()
			agent.save_state()
			np.savez('Episode.npz',array1=[i])
if __name__ == '__main__':
	main()