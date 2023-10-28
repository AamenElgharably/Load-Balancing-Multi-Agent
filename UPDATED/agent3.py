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
import csv
"""
THIS IS FULL ACTION SPACE WITH compressed state according to CLUSTRING (1,3,6) (2,4,5)

"""


def main():
	Episodes=250
	Agent_ID=4 #THIS should be Consistant with the NS3
	PORT=6008 #THIS should be Consistant with the NS3
	env=multi_level_env(Agent_ID,PORT)
	agent=TD3_Agent(0.001,0.001,input_dims=[env.state_dim,],tau=0.005,gamma=0.99,update_actor_interval=2,
		n_actions=env.action_dim,max_size=25000,layer1_size=48,layer2_size=48,batch_size=100,noise=0.1,ID="Level2",warmup=12500,chk_dir='tmp/td3/lvl2')
	try:
		last_episode=np.load('Episode.npz')['array1'][0]
		agent.load_models()
		agent.load_state()
		rwdarr=np.load('reward3.npz')['array1']
		print("last Run states are loaded")
	except:
		print("No last episode to be loaded")
		last_episode=1
		rwdarr=np.empty(0)

	steps=env.env._max_episode_steps
	#training loop:
	for i in range(last_episode,Episodes):
		Reward_accumilator=0
		state=env.reset()
		for j in range(steps):
			action=agent.choose_action(state)
			next_state,reward,done,info=env.step(action)
			print(f"Episode:{i} Step:{j} Action:{action} Agent_ID:{Agent_ID} Reward:{reward}")
			agent.remember(state,action,reward,next_state,done)
			agent.learn()
			state=next_state
			Reward_accumilator+=reward
		rwdarr=np.append(rwdarr,(Reward_accumilator/steps))
		print(f"Episode_reward{Reward_accumilator/steps}")
		if i%10==0:
			np.savez('reward3.npz',array1=rwdarr)
			with open("reward3.csv",'w',newline='') as file:
				writer=csv.writer(file, delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
				writer.writerow(rwdarr)
			file.close()
			print("...........saving............")
			agent.save_models()
			agent.save_state()
			np.savez('Episode.npz',array1=[i])
if __name__ == '__main__':
	main()
