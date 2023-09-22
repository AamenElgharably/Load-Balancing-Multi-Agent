import numpy as np



class ReplayBuffer():
  def __init__(self,max_size,state_shape,n_actions):


    self.mem_size=max_size
    self.mem_cntr=0

    self.state_memory=np.zeros((self.mem_size,*state_shape))

    self.new_state_memory=np.zeros((self.mem_size,*state_shape))

    self.action_memory=np.zeros((self.mem_size,n_actions))

    self.reward_memory=np.zeros((self.mem_size))

    self.terminal_memory=np.zeros((self.mem_size),dtype=bool)


  def store_transitions(self,state,action,reward,new_state,done):

    index=self.mem_cntr%self.mem_size

    self.state_memory[index]=state

    self.new_state_memory[index]=new_state

    self.action_memory[index]=action

    self.reward_memory[index]=reward

    self.terminal_memory[index]=done

    self.mem_cntr+=1
  def sample_buffer(self,batch_size):

    max_mem=min(self.mem_cntr,self.mem_size)

    batch = np.random.choice(max_mem,batch_size)

    states = self.state_memory[batch]

    new_states = self.new_state_memory[batch]

    rewards = self.reward_memory[batch]

    actions = self.action_memory[batch]

    dones = self.terminal_memory[batch]

    return states,actions,rewards,new_states,dones
  def save_buffer(self):
    print("...........Saving_Buffer.........")
    np.savez('Buffer/ReplayBuffer.npz',array1=self.state_memory,array2=self.new_state_memory,
      array3=self.reward_memory,array4=self.action_memory,array5=self.terminal_memory,mem_counter=self.mem_cntr)
  def load_buffer(self):
    print("...........Loading_Buffer.........")
    loaded_buffer=np.load('Buffer/ReplayBuffer.npz')
    self.state_memory=loaded_buffer['array1']
    self.new_state_memory=loaded_buffer['array2']
    self.reward_memory=loaded_buffer['array3']
    self.action_memory=loaded_buffer['array4']
    self.terminal_memory=loaded_buffer['array5']
    self.mem_cntr=loaded_buffer['mem_counter']