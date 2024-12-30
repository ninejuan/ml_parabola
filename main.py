import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from math import *

# 실제 포물선 함수 정의 (이 부분을 수정하여 다른 포물선 식을 사용할 수 있습니다)
def true_function(x, v0=10, theta=30, g=9.8):
    """
    포물선 공식을 기반으로 특정 x 좌표에서 y 좌표를 반환합니다.
    
    x: 수평 거리 (m)
    v0: 초기 속도 (m/s)
    theta: 발사각 (deg)
    g: 중력 가속도 (m/s^2)
    
    반환값: y 좌표 (m)
    """
    theta_rad = radians(theta)
    
    y = x * tan(theta_rad) - (g * x**2) / (2 * (v0 * cos(theta_rad))**2)
    
    return y

class ParabolaNet(nn.Module):
    def __init__(self):
        super(ParabolaNet, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def generate_data(num_points=100, noise_level=0.1):
    x = np.linspace(-5, 5, num_points)
    y = true_function(x)
    y += np.random.normal(0, noise_level, y.shape)
    return x, y

def train_and_visualize():
    x, y = generate_data()
    x_tensor = torch.FloatTensor(x.reshape(-1, 1))
    y_tensor = torch.FloatTensor(y.reshape(-1, 1))
    
    # 모델, 손실 함수, 옵티마이저 설정
    model = ParabolaNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 학습
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # 예측 및 시각화
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(-6, 6, 200).reshape(-1, 1)
        y_pred = model(x_test)
        
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.5, label='Training Data')
    plt.plot(x_test, y_pred, color='red', label='Predicted')
    plt.plot(x_test, true_function(x_test.numpy()), '--', color='green', label='True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Parabola Prediction')
    plt.legend()
    plt.grid(True)
    plt.show()

# 프로그램 실행
if __name__ == "__main__":
    train_and_visualize()