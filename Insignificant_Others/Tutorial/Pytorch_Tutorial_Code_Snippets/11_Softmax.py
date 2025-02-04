import torch 
import torch.nn as nn 
import numpy as np 

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss # / float(predicted.shape[0])

# y must be one-hot encoded 
# if class 0: [1, 0, 0]
# if class 1: [0, 1, 0]
# if class 2: [0, 0, 1]

#Y = np.array([1, 0, 0])

# y_pred has probabilities 
#Y_pred_good = np.array([0.7, 0.2, 0.1])
#Y_pred_bad = np.array([0.1, 0.3, 0.6])
#L1 = cross_entropy(Y, Y_pred_good)
#L2 = cross_entropy(Y, Y_pred_bad)
#print(f'Loss1 numpy: {L1:.4f}')
#print(f'Loss2 numpy: {L2:.4f}')


#x = np.array([2.0, 1.0, 0.1])
#outputs = softmax(x) 
#print('softmax numpy:', outputs)

#x = torch.tensor([2.0, 1.0, 0.1])
#outputs = torch.softmax(x, dim=0)
#print('softmax torch:', outputs)


# In Pytorch, the cross entropy loss combines softmax and cross entropy loss in one step
loss = nn.CrossEntropyLoss()

# 3 samples

Y = torch.tensor([2, 0, 1])
print(Y)

# nsamples x nclass = 3 x 3

Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.1, 1.0, 0.1], [0.1, 3.0, 0.1] ])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)

print(predictions1)
print(predictions2)

