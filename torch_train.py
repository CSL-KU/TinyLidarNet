import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rosbag
from sklearn.utils import shuffle
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
#========================================================
# Functions
#========================================================
#Linear maping
def linear_map(x, x_min, x_max, y_min, y_max):
    """Linear mapping function."""
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

#Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = np.abs(y_true - y_pred)
    loss = np.where(error <= delta, 0.5 * error**2, delta * (error - 0.5 * delta))
    mean_loss = np.mean(loss)
    print(f"Huber Loss: {mean_loss}")
    return mean_loss

#========================================================
# Global Data
#========================================================

# Initialize lists for data
lidar = []
servo = []
speed = []
test_lidar = []
test_servo = []
test_speed = []
model_name = 'TLN'
model_files = [
    './Models/'+model_name+'_noquantized.tflite',
    './Models/'+model_name+'_int8.tflite'
]
dataset_path = [
    './Dataset/out.bag', 
    './Dataset/f2.bag', 
    './Dataset/f4.bag',
]
loss_figure_path = './Figures/loss_curve.png'
down_sample_param = 2 # Down-sample Lidar data
lr = 5e-5
loss_function = 'huber'
batch_size = 64
num_epochs = 20
hz = 40

# Initialize variables for min and max speed
max_speed = 0
min_speed = 0

#========================================================
# Get Dataset
#========================================================

# Iterate through bag files
for pth in dataset_path:
    if not os.path.exists(pth):
        print(f"out.bag doesn't exist in {pth}")
        exit(0)
    good_bag = rosbag.Bag(pth)

    lidar_data = []
    servo_data = []
    speed_data = []

    # Read messages from bag file
    for topic, msg, t in good_bag.read_messages():
        if topic == 'Lidar':
            ranges = msg.ranges[::down_sample_param]
            lidar_data.append(ranges)
        if topic == 'Ackermann':
            data = msg.drive.steering_angle
            s_data = msg.drive.speed
            
            servo_data.append(data)
            if s_data > max_speed:
                max_speed = s_data
            speed_data.append(s_data)

    # Convert data to arrays
    lidar_data = np.array(lidar_data) 
    servo_data = np.array(servo_data)
    speed_data = np.array(speed_data)

    # Shuffle data
    shuffled_data = shuffle(np.concatenate((servo_data[:, np.newaxis], speed_data[:, np.newaxis]), axis=1), random_state=62)
    shuffled_lidar_data = shuffle(lidar_data, random_state=62)

    # Split data into train and test sets
    train_ratio = 0.85
    train_samples = int(train_ratio * len(shuffled_lidar_data))
    x_train_bag, x_test_bag = shuffled_lidar_data[:train_samples], shuffled_lidar_data[train_samples:]

    # Extract servo and speed values
    y_train_bag = shuffled_data[:train_samples]
    y_test_bag = shuffled_data[train_samples:]

    # Extend lists with train and test data
    lidar.extend(x_train_bag)
    servo.extend(y_train_bag[:, 0])
    speed.extend(y_train_bag[:, 1])

    test_lidar.extend(x_test_bag)
    test_servo.extend(y_test_bag[:, 0])
    test_speed.extend(y_test_bag[:, 1])

    print(f'\nData in {pth}:')
    print(f'Shape of Train Data --- Lidar: {len(lidar)}, Servo: {len(servo)}, Speed: {len(speed)}')
    print(f'Shape of Test Data --- Lidar: {len(test_lidar)}, Servo: {len(test_servo)}, Speed: {len(test_speed)}')

# Calculate total number of samples
total_number_samples = len(lidar)

print(f'Overall Samples = {total_number_samples}')
lidar = np.asarray(lidar)
servo = np.asarray(servo)
speed = np.asarray(speed)
speed = linear_map(speed, min_speed, max_speed, 0, 1)
test_lidar = np.asarray(test_lidar)
test_servo = np.asarray(test_servo)
test_speed = np.asarray(test_speed)
test_speed = linear_map(test_speed, min_speed, max_speed, 0, 1)

print(f'Min_speed: {min_speed}')
print(f'Max_speed: {max_speed}')
print(f'Loaded {len(lidar)} Training samples ---- {(len(lidar)/total_number_samples)*100:0.2f}% of overall')
print(f'Loaded {len(test_lidar)} Testing samples ---- {(len(test_lidar)/total_number_samples)*100:0.2f}% of overall\n')

# Check array shapes
assert len(lidar) == len(servo) == len(speed)
assert len(test_lidar) == len(test_servo) == len(test_speed)

#======================================================
# Split Dataset
#======================================================

print('Splitting Data into Train/Test')
train_data = np.concatenate((servo[:, np.newaxis], speed[:, np.newaxis]), axis=1)
test_data =  np.concatenate((test_servo[:, np.newaxis], test_speed[:, np.newaxis]), axis=1)
# Check array shapes
print(f'Train Data(lidar): {lidar.shape}')
print(f'Train Data(servo, speed): {servo.shape}, {speed.shape}')
print(f'Test Data(lidar): {test_lidar.shape}')
print(f'Test Data(servo, speed): {test_servo.shape}, {test_speed.shape}')

# Convert data to PyTorch tensors
train_lidar = torch.tensor(lidar, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
train_targets = torch.tensor(train_data, dtype=torch.float32)
test_lidar = torch.tensor(test_lidar, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
test_targets = torch.tensor(test_data, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(train_lidar, train_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_lidar, test_targets)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#======================================================
# DNN Arch
#======================================================

num_lidar_range_values = len(lidar[0])
print(f'num_lidar_range_values: {num_lidar_range_values}')
print(train_lidar.shape)

class PyTorchModel(nn.Module):
    def __init__(self):
        super(PyTorchModel, self).__init__()
        
        # First layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=133, kernel_size=10, stride=4)
        #self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Second layer
        self.conv2 = nn.Conv1d(in_channels=133, out_channels=32, kernel_size=8, stride=4)
        #self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Third layer
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=15, kernel_size=4, stride=2)
        #self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fourth layer
        self.conv4 = nn.Conv1d(in_channels=15, out_channels=13, kernel_size=3, stride=1)

        # Fifth layer
        self.conv5 = nn.Conv1d(in_channels=13, out_channels=11, kernel_size=3, stride=1)
        # Flatten
        self.flatten = nn.Flatten()
        # Dense layers
        self.fc1 = nn.Linear(225, 100) 
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.fc4 = nn.Linear(10, 2)
    


    def forward(self, x):
        # First layer
        x = F.relu(self.conv1(x))
        #x = self.pool1(F.relu(self.conv1(x)))
        
        # Second layer
        x = F.relu(self.conv2(x))
        #x = self.pool2(F.relu(self.conv2(x)))
        
        # Third layer
        x = F.relu(self.conv3(x))
        #x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten
        x = self.flatten(x)  # Flatten all dimensions except the batch dimension
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output layer with tanh activation
        
        return x
    



# Example usage
#num_lidar_range_values = 100  # Example input sequence length
model = PyTorchModel()

# Print the model architecture
print(int(5e-5))
optimizer = Adam(model.parameters(), lr=0.001) 
criterion = nn.HuberLoss()
print(model)


#======================================================
# Training Loop
#======================================================

train_losses = []
val_losses = []
inference_times_micros = []
period = 1.0 / hz
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()

    model.train()
    epoch_loss = 0
    for batch_lidar, batch_targets in train_loader:
        optimizer.zero_grad()

        #print(f"Batch Lidar Shape: {batch_lidar.shape}, Batch Lidar: {batch_lidar}")
        
        outputs = model(batch_lidar)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    train_losses.append(epoch_loss / len(train_loader))
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        avg_output_time = []
        for batch_lidar, batch_targets in test_loader:
            eval_time = time.time()
            outputs = model(batch_lidar)
            output_time = time.time() - eval_time
            avg_output_time.append(output_time)
            val_loss += criterion(outputs, batch_targets).item()
        
        avg_output_time = np.mean(avg_output_time)
        print("took %.2f ms" % (int(avg_output_time * 1000)))
    
    




    val_losses.append(val_loss / len(test_loader))
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.6f}, Val Loss: {val_losses[-1]:.6f}')
print(f'=============>{int(time.time() - start_time)} seconds<=============')
torch.save(model.state_dict(), './Models/pytorch_model.pth')
# Plot training and validation losses
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(loss_figure_path)
plt.close()