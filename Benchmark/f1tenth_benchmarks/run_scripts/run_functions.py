from f1tenth_benchmarks.simulator import F1TenthSim_TrueLocation, F1TenthSim
from f1tenth_benchmarks.classic_racing.particle_filter import ParticleFilter
import torch
import numpy as np
from pyglet.gl import GL_POINTS

NUMBER_OF_LAPS = 1

def simulate_laps(sim, planner, n_laps):
    n_laps = 10 # for collecting data
    lidar_dataset = []  # Initialize an empty list to store the lidar scans
    steering_angles = []  # Initialize an empty list to store the steering angles
    speeds = []  # Initialize an empty list to store the speeds
    
    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        #print("Keys in observation:", observation.keys())
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)
            lidar_scan = observation['scan']  # Extract the lidar scan from the observation
            # Preprocessing
            noise = np.random.normal(0, 0.5, lidar_scan.shape)
            lidar_scan = lidar_scan + noise
            lidar_scan[lidar_scan>10] = 10
            lidar_dataset.append(lidar_scan)  # Append the lidar scan to the dataset
            steering_angle = action[0]  # Extract the steering angle from the action
            speed = action[1]  # Extract the speed from the action
            steering_angles.append(steering_angle)  # Append the steering angle to the list
            speeds.append(speed)  # Append the speed to the list
    return lidar_dataset, steering_angles, speeds

def simulate_localisation_laps(sim, planner, pf, n_laps):
    for lap in range(n_laps):
        observation, done, init_pose = sim.reset()
        observation['pose'] = pf.init_pose(init_pose)
        while not done:
            action = planner.plan(observation)
            observation, done = sim.step(action)
            observation['pose'] = pf.localise(action, observation)
        pf.lap_complete()


def simulate_training_steps(planner, train_map, test_id, extra_params={}):
    sim = F1TenthSim_TrueLocation(train_map, planner.name, test_id, False, True, extra_params=extra_params)
    observation, done, init_pose = sim.reset()
    
    for i in range(planner.planner_params.training_steps):
        action = planner.plan(observation)
        observation, done = sim.step(action)
        if done:
            planner.done_callback(observation)
            observation, done, init_pose = sim.reset()


#map_list = ["example", "MoscowRaceway", "Austin", "YasMarina", "Spielberg", "Oschersleben"]
map_list = ["example", "MoscowRaceway", "Austin", "Spielberg"]#, "esp"]

# map_list = ["example", "MoscowRaceway"]
# map_list = ["example"]
# map_list = ["aut", "esp", "gbr", 'mco']

def test_planning_all_maps(planner, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    lidar_dataset_all_maps = []  # Initialize an empty list to store lidar datasets from all maps
    steering_angles_all_maps = []  # Initialize an empty list to store steering angles from all maps
    speeds_all_maps = []  # Initialize an empty list to store speeds from all maps
    for map_name in map_list:
        lidar_data_single_map, steering_angles_single_map, speeds_single_map = test_planning_single_map(planner, map_name, test_id, extra_params=extra_params, number_of_laps=number_of_laps)
        if map_name in ["Spielberg"]:
            lidar_dataset_all_maps.extend(lidar_data_single_map)  # Extend the list with lidar data from the current map
            steering_angles_all_maps.extend(steering_angles_single_map)  # Extend the list with steering angles from the current map
            speeds_all_maps.extend(speeds_single_map)  # Extend the list with speeds from the current map
    
    # Convert the lists to numpy arrays
    lidar_dataset_all_maps = np.array(lidar_dataset_all_maps)
    steering_angles_all_maps = np.array(steering_angles_all_maps)
    speeds_all_maps = np.array(speeds_all_maps)
    
    print("Shape of the lidar_dataset_all_maps dataset:", lidar_dataset_all_maps.shape)
    print("Shape of the steering_angles_all_maps dataset:", steering_angles_all_maps.shape)
    print("Shape of the speeds_all_maps dataset:", speeds_all_maps.shape)
    np.savez('./simulated_training_data.npz', lidar=lidar_dataset_all_maps, steering=steering_angles_all_maps, speeds=speeds_all_maps)
    print('Simulated Data Saved')


def test_planning_single_map(planner, map_name, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim_TrueLocation(map_name, planner.name, test_id, extra_params=extra_params)
    planner.set_map(map_name)
    lidar_data, steering_angles, speeds = simulate_laps(simulator, planner, number_of_laps)
    
    # Convert the lists to numpy arrays
    lidar_data_np = np.array(lidar_data)
    steering_angles_np = np.array(steering_angles)
    speeds_np = np.array(speeds)
    
    # Print the shapes of the datasets
    print("Shape of the lidar dataset:", lidar_data_np.shape)
    print("Shape of the steering angles:", steering_angles_np.shape)
    print("Shape of the speeds:", speeds_np.shape)
    
    return lidar_data, steering_angles, speeds


def test_full_stack_all_maps(planner, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS, extra_pf_params={}):
    for map_name in map_list:
        test_full_stack_single_map(planner, map_name, test_id, extra_params=extra_params, number_of_laps=number_of_laps, extra_pf_params=extra_pf_params)

def test_full_stack_single_map(planner, map_name, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS, extra_pf_params={}):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim(map_name, planner.name, test_id, extra_params=extra_params)
    planner.set_map(map_name)
    extra_pf_params["dt"] = simulator.params.timestep * simulator.params.n_sim_steps
    pf = ParticleFilter(planner.name, test_id, extra_pf_params)
    # pf = ParticleFilter(planner.name, test_id, {"dt": simulator.params.timestep * simulator.params.n_sim_steps})
    pf.set_map(map_name)
    simulate_localisation_laps(simulator, planner, pf, number_of_laps)



def test_mapless_all_maps(planner, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    lidar_dataset_all_maps = []  # Initialize an empty list to store lidar datasets from all maps
    steering_angles_all_maps = []  # Initialize an empty list to store steering angles from all maps
    speeds_all_maps = []  # Initialize an empty list to store speeds from all maps
    for map_name in map_list:
        lidar_data_single_map, steering_angles_single_map, speeds_single_map = test_mapless_single_map(planner, map_name, test_id, extra_params=extra_params, number_of_laps=number_of_laps)
        lidar_dataset_all_maps.extend(lidar_data_single_map)  # Extend the list with lidar data from the current map
        steering_angles_all_maps.extend(steering_angles_single_map)  # Extend the list with steering angles from the current map
        speeds_all_maps.extend(speeds_single_map)  # Extend the list with speeds from the current map
    
    # Convert the lists to numpy arrays
    lidar_dataset_all_maps = np.array(lidar_dataset_all_maps)
    steering_angles_all_maps = np.array(steering_angles_all_maps)
    speeds_all_maps = np.array(speeds_all_maps)
    
    # print("Shape of the lidar_dataset_all_maps dataset:", lidar_dataset_all_maps.shape)
    # print("Shape of the steering_angles_all_maps dataset:", steering_angles_all_maps.shape)
    # print("Shape of the speeds_all_maps dataset:", speeds_all_maps.shape)

def test_mapless_single_map(planner, map_name, test_id, extra_params={}, number_of_laps=NUMBER_OF_LAPS):
    print(f"Testing on {map_name}...")
    simulator = F1TenthSim(map_name, planner.name, test_id, extra_params=extra_params)
    lidar_data, steering_angles, speeds = simulate_laps(simulator, planner, number_of_laps)
    
    # Convert the lists to numpy arrays
    lidar_data_np = np.array(lidar_data)
    steering_angles_np = np.array(steering_angles)
    speeds_np = np.array(speeds)
    
    # Print the shapes of the datasets
    # print("Shape of the lidar dataset:", lidar_data_np.shape)
    # print("Shape of the steering angles:", steering_angles_np.shape)
    # print("Shape of the speeds:", speeds_np.shape)
    
    return lidar_data, steering_angles, speeds

def seed_randomness(random_seed):
    
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(random_seed)
    

