import numpy as np
import tensorflow as tf
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU


class TinyLidarNetTest:
    def __init__(self, model_path):
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.output_details = self.interpreter.get_output_details()

    def linear_map(self, x, x_min, x_max, y_min, y_max):
        # Ensure there's no division by zero
        if x_max == x_min:
            print("x_max and x_min cannot be the same, division by zero error.")
            return y_min  # Return the minimum of the target range if x_max and x_min are equal
        
        # Correct formula: 
        return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min

    def generate_random_scan(self, seed=None, skip = 1):
        np.random.seed(seed)
        scan = np.random.rand(1081) * 10  # Generates values between 0 and 10
        return scan[::skip]

    def run_inference(self, scan):
        # noise = np.random.normal(0, 0.5, scans.shape)
        # scans = scans + noise
        # Ensure scan values are capped at 10
        scan[scan > 10] = 10
        
        # Preprocess the scan data for input into the model
        scan = np.expand_dims(scan, axis=-1).astype(np.float32)  # Shape: (1081, 1)
        scan = np.expand_dims(scan, axis=0)  # Add batch dimension; Shape: (1, 1081, 1)

        # Set the model input tensor
        self.interpreter.set_tensor(self.input_index, scan)

        # Run inference
        start_time = time.time()
        self.interpreter.invoke()
        inf_time = (time.time() - start_time) * 1000  # Inference time in milliseconds

        # Get the output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        steer = output[0, 0]
        speed = output[0, 1]
        
        print(f"Steer: {steer}, Speed: {speed}, Inference Time: {inf_time} ms")

        # Map speed to the appropriate range (1 to 8)
        min_speed = 1
        max_speed = 8
        speed = self.linear_map(speed, 0, 1, min_speed, max_speed)

        return steer, speed, inf_time


# Example usage:
if __name__ == "__main__":
    # Path to your TFLite model
    model_path = "/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_MLP_M_noquantized.tflite"

    # Initialize the TinyLidarNetTest with the model path
    tiny_lidar_net = TinyLidarNetTest(model_path=model_path)

    # Generate a random scan with a fixed seed (e.g., seed=42)
    random_scan = tiny_lidar_net.generate_random_scan(seed=0, skip = 2)

    # Run the model inference on the generated scan
    steer, speed, inf_time = tiny_lidar_net.run_inference(random_scan)

    # Output the results
    print(f"Steer: {steer}, Speed: {speed}, Inference Time: {inf_time} ms")
