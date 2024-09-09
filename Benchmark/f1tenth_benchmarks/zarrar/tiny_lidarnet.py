import time
import numpy as np
from numba import njit
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
import tensorflow as tf

class TinyLidarNet(BasePlanner):
    def __init__(self, test_id, skip_n, pre, model_path):
        super().__init__("TinyLidarNet", test_id)
        self.pre = pre
        self.skip_n = skip_n
        self.model_path = model_path
        self.name = 'TinyLidarNet'
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()
        self.scan_buffer = np.zeros((2, 20))

        self.temp_scan = []

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def render_waypoints(self, *args, **kwargs):
        pass
        
    def transform_obs(self, scan):
        self.scan_buffer
        scan = scan[:1080]
        scan = scan[::54]
        if self.scan_buffer.all() ==0: # first reading
            for i in range(self.scan_buffer.shape[0]):
                self.scan_buffer[i, :] = scan
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        scans = np.reshape(self.scan_buffer, (-1))
        return scans

    
    def plan(self, obs):
        scans = obs['scan']

        noise = np.random.normal(0, 0.5, scans.shape)
        scans = scans + noise
        
        chunks = [scans[i:i+4] for i in range(0, len(scans), 4)]
        if self.pre == 1:
            scans = [np.mean(chunk) for chunk in chunks]
        elif self.pre == 2:
            scans = [np.max(chunk) for chunk in chunks]
        elif self.pre == 3:
            scans = [np.min(chunk) for chunk in chunks]
        elif self.pre == 4:
            scans = self.transform_obs(scans)
        else:
            scans = scans[::self.skip_n]

        if self.pre < 4:
            scans = np.array(scans)
            scans[scans>10] = 10

            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0)
            self.interpreter.set_tensor(self.input_index, scans)
            
            start_time = time.time()
            self.interpreter.invoke()
            inf_time = time.time() - start_time
            inf_time = inf_time*1000
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0,0]
            speed = output[0,1]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed) 
            action = np.array([steer, speed])

        elif self.pre == 5:
            # Temporal
            scans = np.array(scans)
            scans[scans>10] = 10
            input_shape = self.interpreter.get_input_details()[0]['shape']
            # print("Input shape:", input_shape)
            # print("Shape of scans:", scans.shape)
            # print("Shape of self.temp_scan:", [s.shape for s in self.temp_scan])

            if(len(self.temp_scan) <1):
                self.temp_scan.append(scans)
                return np.array([0,2])
            self.temp_scan.append(scans)
            scans = np.array(self.temp_scan)
            # print("Shape of scans:", scans.shape)
            scans = np.expand_dims(scans, axis=0).astype(np.float32)
            # print("Shape of scans:", scans.shape)
            scans = np.transpose(scans, (0, 2, 1))
            # print("Shape of scans:", scans.shape)
            # scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            # scans = np.expand_dims(scans, axis=0)
            self.interpreter.set_tensor(self.input_index, scans)
            
            start_time = time.time()
            self.interpreter.invoke()
            inf_time = time.time() - start_time
            inf_time = inf_time*1000
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0,0]
            speed = output[0,1]
            self.temp_scan = self.temp_scan[1:]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed) 
            action = np.array([steer, speed])

        elif self.pre == 6:
            # birdeye
            scans = np.array(scans)
            scans[scans>10] = 10
            input_shape = self.interpreter.get_input_details()[0]['shape']
            # print("Input shape:", input_shape)
            # print("Shape of scans:", scans.shape)
            # print("Shape of self.temp_scan:", [s.shape for s in self.temp_scan])

            if(len(self.temp_scan) <3):
                self.temp_scan.append(scans)
                return np.array([0,2])
            self.temp_scan.append(scans)
            scans = np.array(self.temp_scan)
            # print("Shape of scans:", scans.shape)
            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0).astype(np.float32)
            # print("Shape of scans:", scans.shape)
            self.interpreter.set_tensor(self.input_index, scans)
            
            start_time = time.time()
            self.interpreter.invoke()
            inf_time = time.time() - start_time
            inf_time = inf_time*1000
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0,0]
            speed = output[0,1]
            self.temp_scan = self.temp_scan[1:]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed) 
            action = np.array([steer, speed])


        else:
            scans = np.expand_dims(scans, axis=-1).astype(np.float32)
            scans = np.expand_dims(scans, axis=0)

            scans[scans>10] = 10
            # scans = np.asarray(scans).reshape(-1,1,1081,1).astype(np.float32)
            # print(scans.shape)
            self.interpreter.set_tensor(self.input_index, scans)
            

            start_time = time.time()
            self.interpreter.invoke()
            inf_time = time.time() - start_time
            inf_time = inf_time*1000
            output = self.interpreter.get_tensor(self.output_details[0]['index'])

            steer = output[0,0]
            speed = output[0,1]
            min_speed = 1
            max_speed = 8
            speed = self.linear_map(speed, 0, 1, min_speed, max_speed)
            action = np.array([steer, speed])

        return action