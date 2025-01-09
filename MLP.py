import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

class MLP:
    def __init__(self, input_size, hidden_size, learning_rate=0.1, regularization=0.01, batch_size=32):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros(1)
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2.flatten()

    def backward(self, X, y, y_pred):
        error = y_pred - y
        m = len(y)
        dL_dz2 = error.reshape(-1, 1) / m
        dL_dW2 = np.dot(self.a1.T, dL_dz2) + self.regularization * self.W2
        dL_db2 = np.sum(dL_dz2, axis=0)
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        dL_dz1 = dL_da1 * self.tanh_derivative(self.z1)
        dL_dW1 = np.dot(X.T, dL_dz1) + self.regularization * self.W1
        dL_db1 = np.sum(dL_dz1, axis=0)
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        return np.mean(error ** 2)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]
                y_pred = self.forward(X_batch)
                loss = self.backward(X_batch, y_batch, y_pred)
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)


def main():
    data = pd.read_csv('grid_80_92.csv')
    time_slots = data['time_slot'].values
    user_count = data['user_count'].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    user_count_scaled = scaler.fit_transform(user_count.reshape(-1, 1))
    time_step = 24
    X, y = create_dataset(user_count_scaled, time_step)
    X = X.reshape(X.shape[0], X.shape[1])
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    best_model = None
    best_mae = float('inf')  #  MAE
    best_mse = float('inf')  #  MSE
    
    for i in range(100):
        model = MLP(input_size=X_train.shape[1], hidden_size=256, learning_rate=0.01, regularization=0.01, batch_size=32)
        model.train(X_train, y_train, epochs=1000)
        y_pred = model.predict(X_test)

        y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

     
        mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
        mse = np.mean((y_test_rescaled - y_pred_rescaled) ** 2)
        mape = np.mean(np.abs((y_test_rescaled - y_pred_rescaled) / y_test_rescaled)) * 100
        print(f'Model {i+1} - MAE: {mae:.4f}, MSE: {mse:.4f},MAPE: {mape:.2f}%')

       
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_mse=mape
            best_model = model

    print(f'Best Model - MAE: {best_mae:.4f}, MSE: {best_mse:.4f},MAPE: {mape:.2f}%')

  
    y_pred = best_model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(scaler.inverse_transform(y_test.reshape(-1, 1)), label='Actual', color='blue')
    plt.plot(y_pred_rescaled, label='Predicted', color='red')
    plt.title('Best Model: Prediction vs Actual')
    plt.xlabel('Time')
    plt.ylabel('User Count')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
