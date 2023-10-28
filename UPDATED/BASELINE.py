import numpy as np
import os
import gym
from Multi_level_env import multi_level_env
import csv




def main():
	Episodes=20
	Agent_ID=1 #THIS should be Consistant with the NS3
	PORT=6020 #THIS should be Consistant with the NS3
	env=multi_level_env(Agent_ID,PORT)
	last_episode=0
	rwdarr=np.empty(0)
	steps=env.env._max_episode_steps
	#training loop:
	for i in range(last_episode,Episodes):
		Reward_accumilator=0
		state=env.reset()
		for j in range(steps):
			next_state,reward,done,info=env.step([0,0])
			print(f"Episode:{i} Step:{j} Action:{[0,0]} Agent_ID:{Agent_ID} Reward:{reward}")
			state=next_state
			Reward_accumilator+=reward
		rwdarr=np.append(rwdarr,(Reward_accumilator/steps))
		print(f"Episode_reward{Reward_accumilator/steps}")
		if i%10==0:
			np.savez('Baseline_reward.npz',array1=rwdarr)
			with open("BL_REWARD.csv",'w',newline='') as file:
				writer=csv.writer(file, delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
				writer.writerow(rwdarr)
			file.close()
			print("...........saving............")
if __name__ == '__main__':
	main()