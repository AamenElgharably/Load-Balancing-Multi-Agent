import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import gym
from ReplayBuffer import ReplayBuffer
from Networks import CriticNetwork,ActorNetwork

class TD3_Agent():
  def __init__(self,alpha,beta,input_dims,tau,gamma,update_actor_interval,warmup,n_actions,max_size,layer1_size,layer2_size,batch_size,noise,ID):
    self.gamma=gamma
    self.tau=tau
    self.max_action=1
    self.min_action=-1
    self.memory=ReplayBuffer(max_size,input_dims,n_actions)
    self.batch_size=batch_size
    self.learn_step_cntr=0
    self.time_step=0
    self.warmup=warmup
    self.n_action=n_actions
    self.upadate_actor_interval=update_actor_interval

    self.actor=ActorNetwork(alpha,input_dims,layer1_size,layer2_size,n_actions,name='Actor'+ID)
    self.critic_1=CriticNetwork(beta,input_dims,layer1_size,layer2_size,n_actions,name='Critic_1'+ID)
    self.critic_2=CriticNetwork(beta,input_dims,layer1_size,layer2_size,n_actions,name='Critic_2'+ID)
    self.target_actor=ActorNetwork(alpha,input_dims,layer1_size,layer2_size,n_actions,name='TargetActor'+ID)
    self.target_critic_1=CriticNetwork(alpha,input_dims,layer1_size,layer2_size,n_actions,name='Target_Critic_1'+ID)
    self.target_critic_2=CriticNetwork(alpha,input_dims,layer1_size,layer2_size,n_actions,name='Target_Critic_2'+ID)
    self.noise=noise
    self.update_network_parameters(tau=1)

  def choose_action(self,obs):
    if self.time_step < self.warmup:
      mu=torch.tensor(np.random.normal(scale=self.noise,size=(self.n_action, ))).to(self.actor.device)
    else:
      state=torch.tensor(obs,dtype=torch.float).to(self.actor.device)
      mu=self.actor.forward(state).to(self.actor.device)
    mu_prime=mu +torch.tensor(np.random.normal(scale=self.noise,size=(self.n_action, )),dtype=torch.float).to(self.actor.device)
    mu_prime=torch.clamp(mu_prime,self.min_action,self.max_action)
    self.time_step+=1
    return mu_prime.cpu().detach().numpy()

  def remember(self,state,action,reward,new_state,done):
    self.memory.store_transitions(state,action,reward,new_state,done)
  def learn(self):
    if self.memory.mem_cntr< self.batch_size:
      return
    print("Learning")
    state,action,reward,new_state,done=self.memory.sample_buffer(self.batch_size)
    reward=torch.tensor(reward,dtype=torch.float).to(self.critic_1.device)
    done=torch.tensor(done).to(self.critic_1.device)
    new_state=torch.tensor(new_state,dtype=torch.float).to(self.critic_1.device)
    state=torch.tensor(state,dtype=torch.float).to(self.critic_1.device)
    action=torch.tensor(action,dtype=torch.float).to(self.critic_1.device)

    target_action=self.target_actor.forward(new_state)
    target_action=target_action+torch.clamp(torch.tensor(np.random.normal(scale=0.2)),-0.5,0.5)
    target_action=torch.clamp(target_action,self.min_action,self.max_action)

    q1_=self.target_critic_1.forward(new_state,target_action)
    q2_=self.target_critic_2.forward(new_state,target_action)
    q1=self.critic_1.forward(state,action)
    q2=self.critic_2.forward(state,action)
    q1_[done]=0
    q2_[done]=0
    q1_=q1_.view(-1)
    q2_=q2_.view(-1)

    critic_value=torch.min(q1_,q2_)

    target=reward + self.gamma*critic_value
    target=target.view(self.batch_size,1)

    self.critic_1.optimizer.zero_grad()
    self.critic_2.optimizer.zero_grad()

    q1_loss=F.mse_loss(target,q1)
    q2_loss=F.mse_loss(target,q2)
    critic_loss=q1_loss+q2_loss
    critic_loss.backward()
    self.critic_1.optimizer.step()
    self.critic_2.optimizer.step()
    self.learn_step_cntr+=1
    if self.learn_step_cntr%self.upadate_actor_interval !=0:
      return
    self.actor.optimizer.zero_grad()
    actor_q1_loss=self.critic_1.forward(state,self.actor.forward(state))
    actor_loss=-torch.mean(actor_q1_loss)
    actor_loss.backward()
    self.actor.optimizer.step()
    self.update_network_parameters()
  def update_network_parameters(self,tau=None):

    if tau is None:
      tau=self.tau
    actor_params=self.actor.named_parameters()
    critic1_params=self.critic_1.named_parameters()
    critic2_params=self.critic_2.named_parameters()
    target_actor_parames=self.target_actor.named_parameters()
    target_critic1_params=self.target_critic_1.named_parameters()
    target_critic2_params=self.target_critic_2.named_parameters()
    critic_1=dict(critic1_params)
    critic_2=dict(critic2_params)
    actor=dict(actor_params)
    target_actor=dict(target_actor_parames)
    target_critic_1=dict(target_critic1_params)
    target_critic_2=dict(target_critic2_params)

    for name in critic_1:
      critic_1[name]=tau*critic_1[name].clone()+(1-tau)*target_critic_1[name].clone()
    for name in critic_2:
      critic_2[name]=tau*critic_2[name].clone()+(1-tau)*target_critic_2[name].clone()
    for name in actor:
       actor[name]=tau*actor[name].clone()+(1-tau)*target_actor[name].clone()
    self.target_critic_1.load_state_dict(critic_1)
    self.target_critic_2.load_state_dict(critic_2)
    self.target_actor.load_state_dict(actor)
  def save_models(self):
    self.actor.save_checkpoint()
    self.target_actor.save_checkpoint()
    self.critic_1.save_checkpoint()
    self.target_critic_1.save_checkpoint()
    self.target_critic_2.save_checkpoint()
    self.critic_2.save_checkpoint()
  def load_models(self):
    self.actor.load_checkpoint()
    self.critic_1.load_checkpoint()
    self.critic_2.load_checkpoint()
    self.target_actor.load_checkpoint()
    self.target_critic_2.load_checkpoint()
    self.target_critic_1.load_checkpoint()
  def save_state(self):
    np.savez('tmp/agent_state.npz',array1=[self.learn_step_cntr ,self.time_step])
  def load_state(self):
    load=np.load('tmp/agent_state.npz')
    self.learn_step_cntr=load['array1'][0]
    self.time_step=load['array1'][1]

