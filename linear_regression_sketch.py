import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100,1)
y = 2 + 3 *X + np.random.randn(100,1) # noise elave eleyirik

m = len(y) # sampleslerin sayi

w = np.random.randn(1)
b = np.random.randn(1)
alpha = 0.02 # 0.002
n_epochs = 50

losses = []

for epoch in range(n_epochs):
    y_hat = w*X + b
    loss = 1/m*np.sum((y-y_hat)**2)

    losses.append(loss)

    w = w - alpha*(-2/m*np.sum(X*(y-y_hat)))
    b = b - alpha*(-2/m*np.sum(y-y_hat))

    print(f"Epoch {epoch+1}/{n_epochs}, Loss : {loss:.4f}")


plt.plot(range(1, n_epochs+1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.grid(True)
plt.show()