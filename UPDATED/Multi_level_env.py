import gym
import numpy as np
import gym
from ns3gym import ns3env
from gym import spaces
import os


class multi_level_env:
    def __init__(self,Agent_ID,PRT):#Agent_ID should Match the one in the Simulator 
        super(multi_level_env, self).__init__()
        port=PRT
        simTime= 2
        stepTime=0.2
        startSim=0
        seed=3
        simArgs = {"--duration": simTime,}
        debug=True
        self.ID=Agent_ID
        if self.ID==1:
            self.Cluster=[0 ,1 ,3]
            self.Cluster_num=3
        elif self.ID==2:
            self.Cluster=[2,4,5]
            self.Cluster_num=3
        else:
            self.Cluster=[0,1,2,3,4,5]
            self.Cluster_num=6
        max_env_steps = 250
        self.env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
        self.env._max_episode_steps = max_env_steps
        self.Cell_num=6
        self.max_throu=30
        self.Users=40
        if self.ID==4:
            self.state_dim=8
        else:
            self.state_dim = self.Cluster_num*4
        self.action_dim =  self.env.action_space.shape[0]
        self.action_bound =  self.env.action_space.high
        self.action_space = spaces.Box(low=-1, high=1,
                                        shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=self.Users,
                                        shape=(self.state_dim,), dtype=np.float32)
        self.cluster1=[1,3,4]
        self.cluster2=[0,2,5]


    def reset(self):
        if self.ID==1:
            state = self.env.reset()
            state1 = np.reshape(state['rbUtil'], [self.Cell_num, 1])#Reshape the matrix
            state2 = np.reshape(state['dlThroughput'],[self.Cell_num,1])
            state2_norm=state2/self.max_throu
            state3 = np.reshape(state['UserCount'], [self.Cell_num, 1])#Reshape the matrix
            state3_norm=state3/self.Users
            MCS_t=np.array(state['MCSPen'])
            state4=np.sum(MCS_t[:,:10], axis=1)
            state4=np.reshape(state4,[self.Cell_num,1])
            
            np.savez('reset_state.npz',array1=state1,array2=state2_norm,array3=state3_norm,array4=state4) 
            state= np.concatenate((state1[self.Cluster],state2_norm[self.Cluster],state3_norm[self.Cluster],state4[self.Cluster]),axis=None)              
            state = np.reshape(state, [self.state_dim,])###   
            return np.array(state)
        elif self.ID==2:
            state1=np.load('reset_state.npz')['array1']
            state2=np.load('reset_state.npz')['array2']
            state3=np.load('reset_state.npz')['array3']
            state4=np.load('reset_state.npz')['array4']
            state= np.concatenate((state1[self.Cluster],state2[self.Cluster],state3[self.Cluster],state4[self.Cluster]),axis=None)              
            state = np.reshape(state, [self.state_dim,])### 
            os.remove("reset_state.npz")
            return np.array(state)
        elif self.ID==4:
            state = self.env.reset()
            return self.group_state(state)  

    def step(self,action):
        action=action*self.action_bound
        next_state, reward, done, info = self.env.step(action)
        if next_state is not(None):#if one closes the env
            if self.ID==4:
                return self.group_state(next_state),reward,done,info
            state1 = np.reshape(next_state['rbUtil'], [self.Cell_num, 1])#Reshape the matrix (do we need that?)
            state2 = np.reshape(next_state['dlThroughput'],[self.Cell_num,1])
            state2_norm=state2/self.max_throu
            state3 = np.reshape(next_state['UserCount'], [self.Cell_num, 1])#Reshape the matrix (do we need that?)
            state3_norm=state3/self.Users
            MCS_t=np.array(next_state['MCSPen'])
            state4=np.sum(MCS_t[:,:10], axis=1)
            state4=np.reshape(state4,[self.Cell_num,1])
            next_state  = np.concatenate((state1[self.Cluster],state2_norm[self.Cluster],state3_norm[self.Cluster],state4[self.Cluster]),axis=None)
            next_state = np.reshape(next_state, [self.state_dim,])


            return np.array(next_state),reward, done,info
        else:
            return np.array(next_state),-1, done,info
    def group_state(self,state):
            state1 = np.reshape(state['rbUtil'], [self.Cell_num, 1])#Reshape the matrix
            state2 = np.reshape(state['dlThroughput'],[self.Cell_num,1])
            state2_norm=state2/self.max_throu
            state3 = np.reshape(state['UserCount'], [self.Cell_num, 1])#Reshape the matrix
            state3_norm=state3/self.Users
            MCS_t=np.array(state['MCSPen'])
            state4=np.sum(MCS_t[:,:10], axis=1)
            state4=np.reshape(state4,[self.Cell_num,1])
            state1_1=np.sum(state1[self.cluster1])
            state1_2=np.sum(state1[self.cluster2])
            state2_1=np.sum(state2_norm[self.cluster1])
            state2_2=np.sum(state2_norm[self.cluster2])
            state3_1=np.sum(state3_norm[self.cluster1])
            state3_2=np.sum(state3_norm[self.cluster2])
            state4_1=np.sum(state4[self.cluster1])
            state4_2=np.sum(state4[self.cluster2])
            state=np.concatenate((state1_1,state1_2,state2_1,state2_2,state3_1,state3_2,state4_1,state4_2),axis=None)
            state=np.reshape(state,[self.state_dim,])/3
            return np.array(state)