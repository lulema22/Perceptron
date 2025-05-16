import numpy as np
import matplotlib.pyplot as plt


X = np.load('X.npy')
y = np.load('y.npy')

N = 1000000
l = 30
w = np.zeros(X.shape[1])
b = 7
u = 0.1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

z = X@w + b
y_st = sigmoid(z)
epochs = 20
losses = []
accuracies = []
for j in range(epochs):
    z = X@w + b
    y_ans = sigmoid(z)
    loss = -np.mean(y*np.log(y_ans) + (1-y)*np.log(1-y_ans))
    grad_w = (1/N) * X.T @ (y_ans - y)
    grad_b = (1/N) * np.sum(y_ans - y)

    w -= u*grad_w
    b -= u*grad_b
    print(f"Эпоха {j + 1}: loss = {loss:.4f}")

    predictions = (y_ans > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)
    print(accuracy*100)
    losses.append(loss)


plt.plot(losses, label='Loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.title('Сходимость функции потерь')
plt.grid(True)
plt.legend()
plt.show()

plt.plot(accuracies, label='Accuracy')
plt.xlabel('Эпоха')
plt.ylabel('Accuracy')
plt.title('Сходимость точности классификации')
plt.grid(True)
plt.legend()
plt.show()