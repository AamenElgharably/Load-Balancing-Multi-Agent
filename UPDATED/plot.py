import matplotlib.pyplot as plt
import csv
import numpy as np




with open("reward3.csv",'r') as file:
	reader=csv.reader(file,delimiter=',', quotechar='|',quoting=csv.QUOTE_MINIMAL)
	reward=list(reader)

reward=np.array(reward,dtype=float)
reward=np.reshape(reward,(np.shape(reward)[1],))

x_axis=[i+1 for i in range(0,len(reward))]



#PLOTTING

fig, ax = plt.subplots()
line_1=ax.plot(x_axis[:20],reward[:20],label='Exploration',color='r')
line_2=ax.plot(x_axis[19:],reward[19:],label='Exploitation',color='b')

plt.xlabel("Episode",fontsize=15,font='cmb10')
plt.ylabel("Reward DL Throughput (Mbs)",fontsize=15,font='cmb10')
plt.title("Agent(Compressed State Full Action)",fontsize=20,font='cmb10')
plt.legend(['Exploration', 'Exploitation'], loc='upper left')
plt.show()