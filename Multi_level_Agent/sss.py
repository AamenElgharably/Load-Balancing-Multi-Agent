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
agent=TD3_Agent(0.001,0.001,input_dims=[3,],tau=0.005,gamma=0.99,update_actor_interval=2,
		n_actions=2,max_size=10000,layer1_size=24,layer2_size=24,batch_size=32,noise=0.1,ID="Level1-1",warmup=200)
agent.load_state()