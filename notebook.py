from pylab import rand, plot, show, norm
import numpy as np

#train data (XOR)
train_data = np.array([[0,0],[0,1],[1,0],[1,1]])
train_label = np.array([[0],[1],[1],[0]])

#train data 보기
for n in range(4):
  if train_label[n] == 1:
    plot(train_data[n],train_data[n],'ob')
  else:
    plot(train_data[n],train_data[n],'or')
show()