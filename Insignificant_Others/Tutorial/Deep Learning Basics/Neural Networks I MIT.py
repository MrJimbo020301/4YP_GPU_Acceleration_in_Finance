import numpy as np

def SM(z):
    return np.exp(z) / np.sum(np.exp(z))
    
w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1], [1]])
y = np.array([[0, 1, 0]]).T
z = np.dot(w.T, x)
a = SM(z)
g = np.dot(x, (a - y).T)
#print(g.tolist())
#print(a.tolist())

new_w = w - 0.5 * g
#print(new_w)
new_z = np.dot(new_w.T, x)
new_a = SM(new_z)
print(new_a)


def hinge_loss_grad(x,y,a):
    return np.where(y*a > 1, 0, -y*x)
