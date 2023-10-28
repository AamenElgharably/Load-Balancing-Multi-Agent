import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os

class ActorNetwork(nn.Module):
    def __init__(self,beta,state_dim,fc1_dims,fc2_dims,n_actions,name ,chkpt_dir='tmp/td3/lvl2'):

      super(ActorNetwork,self).__init__()
      self.input_dims=state_dim
      self.fc1_dims=fc1_dims
      self.fc2_dims=fc2_dims
      self.n_actions=n_actions
      self.name=name
      self.checkpoint=chkpt_dir
      self.checkpoint_file=os.path.join(self.checkpoint,name+name+'td3')

      self.fc1=nn.Linear(self.input_dims[0],self.fc1_dims)
      self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
      self.u1=nn.Linear(self.fc2_dims,n_actions)

      self.optimizer=torch.optim.Adam(self.parameters(),lr=beta)
      self.device=torch.device('cuda:0'if torch.cuda.is_available() else'cpu' )
      self.to(self.device)
    def forward(self,state):
      prob=self.fc1(state)
      prob=F.relu(prob)
      prob=self.fc2(prob)
      prob=F.relu(prob)
      prob =torch.tanh(self.u1(prob))
      return prob
    def save_checkpoint(self):
      print(".....Saving check point......")
      torch.save(self.state_dict(),self.checkpoint_file)
    def load_checkpoint(self):
      print(".....Loading checkpoint......")
      self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
  def __init__(self,beta,state_dim,fc1_dims,fc2_dims,n_actions,name ,chkpt_dir='tmp/td3/lvl2'):
    super(CriticNetwork,self).__init__()
    self.input_dims=state_dim
    self.fc1_dims=fc1_dims
    self.fc2_dims=fc2_dims
    self.n_actions=n_actions
    self.name=name
    self.checkpoint=chkpt_dir
    self.checkpoint_file=os.path.join(self.checkpoint,name+name+'td3')

    self.fc1=nn.Linear(self.input_dims[0]+n_actions,self.fc1_dims)
    self.fc2=nn.Linear(self.fc1_dims,self.fc2_dims)
    self.q1=nn.Linear(self.fc2_dims,1)

    self.optimizer=torch.optim.Adam(self.parameters(),lr=beta)
    self.device=torch.device('cuda:0'if torch.cuda.is_available() else'cpu' )
    self.to(self.device)
  def forward(self,state,action):
    q1_action_value = self.fc1(torch.cat([state, action], dim=1))
    q1_action_value = F.relu(q1_action_value)
    q1_action_value = self.fc2(q1_action_value)
    q1_action_value = F.relu(q1_action_value)

    q1 = self.q1(q1_action_value)

    return q1
  def save_checkpoint(self):
    print(".....Saving check point......")
    torch.save(self.state_dict(),self.checkpoint_file)
  def load_checkpoint(self):
    print(".....Loading checkpoint......")
    self.load_state_dict(torch.load(self.checkpoint_file))