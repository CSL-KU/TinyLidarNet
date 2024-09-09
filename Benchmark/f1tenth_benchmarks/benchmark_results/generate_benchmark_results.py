from f1tenth_benchmarks.classic_racing.RaceTrackGenerator import RaceTrackGenerator, load_parameter_file_with_extras
from f1tenth_benchmarks.classic_racing.GlobalPurePursuit import GlobalPurePursuit
from f1tenth_benchmarks.classic_racing.GlobalMPCC import GlobalMPCC
from f1tenth_benchmarks.mapless_racing.FollowTheGap import FollowTheGap
from f1tenth_benchmarks.drl_racing.EndToEndAgent import EndToEndAgent, TrainEndToEndAgent, TinyAgent,  TrainTinyAgent
from f1tenth_benchmarks.zarrar.mlp_il import EndToEnd
from f1tenth_benchmarks.zarrar.tiny_lidarnet import TinyLidarNet

from f1tenth_benchmarks.data_tools.specific_plotting.plot_drl_training import plot_drl_training
from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.run_scripts.run_functions import *

NUMBER_OF_LAPS = 10


def generate_racelines():
    map_list = ['example', 'MoscowRaceway']
    params = load_parameter_file_with_extras("RaceTrackGenerator", extra_params={"mu": 0.9})
    raceline_id = f"mu{int(params.mu*100)}"
    for map_name in map_list:
        RaceTrackGenerator(map_name, raceline_id, params, plot_raceline=True)


def optimisation_and_tracking():
    test_id = "benchmark_pp"
    planner = GlobalPurePursuit(test_id, False, planner_name="GlobalPlanPP", extra_params={"racetrack_set": "mu90"})
    test_planning_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)


def mpcc():
    test_id = f"benchmark_mpcc"
    planner = GlobalMPCC(test_id, False, planner_name="GlobalPlanMPCC", extra_params={"friction_mu": 0.9})
    test_planning_all_maps(planner, test_id, number_of_laps=10)

    plot_trajectory_analysis(planner.name, test_id)


def follow_the_gap():
    test_id = "benchmark_ftg"
    planner = FollowTheGap(test_id)
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)


def end_to_end_drl():
    test_id = "benchmark_e2e_drl"
    training_map = "MoscowRaceway"
    seed_randomness(12)
    # print(f"Training DRL agent: {test_id}")
    # training_agent = TrainEndToEndAgent(training_map, test_id, extra_params={'reward': "TAL", 'tal_racetrack_set': "mu90"}) 
    # simulate_training_steps(training_agent, training_map, test_id, extra_params={'n_sim_steps': 10})
    # plot_drl_training(training_agent.name, test_id)

    testing_agent = EndToEndAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id, number_of_laps=NUMBER_OF_LAPS)
    plot_trajectory_analysis(testing_agent.name, test_id)
    

def tinylidar_drl():
    test_id = "benchmark_tiny_drl"
    training_map = "MoscowRaceway"
    seed_randomness(12)
    # print(f"Training DRL agent: {test_id}")
    # training_agent = TrainTinyAgent(training_map, test_id, extra_params={'reward': "TAL", 'tal_racetrack_set': "mu90"}) 
    # simulate_training_steps(training_agent, training_map, test_id, extra_params={'n_sim_steps': 10})
    # plot_drl_training(training_agent.name, test_id)

    testing_agent = TinyAgent(test_id)
    test_mapless_all_maps(testing_agent, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(testing_agent.name, test_id)

def end_to_end_il():
    test_id = "benchmark_e2e_il"
    planner = EndToEnd(test_id,4, '/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_MLP_S_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def end_to_end_il_m():
    test_id = "benchmark_e2e_il_m"
    planner = EndToEnd(test_id,2, '/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_MLP_M_noquantized.tflite')
    # planner = EndToEnd(test_id,2, '/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/MLP_M_Dropout_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def end_to_end_il_l():
    test_id = "benchmark_e2e_il_l"
    planner = EndToEnd(test_id, 1, '/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_paper_noquantized.tflite')
    # planner = EndToEnd(test_id, 1, '/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_MLP_L_Dropout_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def end_to_end_il_128():
    test_id = "benchmark_e2e_il_128"
    planner = EndToEnd(test_id, 1, '/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_128_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_mean():
    test_id = "benchmark_tiny_il_mean"
    planner = TinyLidarNet(test_id,4, 1,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_smaller_mean_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_max():
    test_id = "benchmark_tiny_il_max"
    planner = TinyLidarNet(test_id,4, 2,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_smaller_max_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_min():
    test_id = "benchmark_tiny_il_min"
    planner = TinyLidarNet(test_id,4, 3,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_smaller_min_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_temporal():
    test_id = "benchmark_tiny_il_temporal"

    print(test_id)
    #planner = TinyLidarNet(test_id, 2, 5,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_temporal_M_noquantized.tflite')
    planner = TinyLidarNet(test_id, 2, 5,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_temporal_2M_noquantized.tflite')
    
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_birdeye():
    test_id = "benchmark_tiny_il_birdeye"
    print(test_id)
    planner = TinyLidarNet(test_id, 2, 6,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_birdeye_M_noquantized.tflite')
    
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il():
    test_id = "benchmark_tiny_il"
    print(test_id)
    planner = TinyLidarNet(test_id,4, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_smaller_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_m():
    test_id = "benchmark_tiny_il_m"
    print(test_id)
    planner = TinyLidarNet(test_id,2, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_small_noquantized.tflite')
    # planner = TinyLidarNet(test_id,2, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_TLN_M_Dag_noquantized.tflite')
    # planner = TinyLidarNet(test_id,2, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/TinyLidarNet_M_Dropout_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

def tinylidar_il_l():
    test_id = "benchmark_tiny_il_l"
    print(test_id)
    planner = TinyLidarNet(test_id,1, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_main_noquantized.tflite')
    # planner = TinyLidarNet(test_id,1, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_TLN_L_Dropout_noquantized.tflite')
    # planner = TinyLidarNet(test_id,1, 0,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_TLN_L_Dag_noquantized.tflite')
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)


def tinylidar_il_dropout():
    test_id = "benchmark_tiny_il_dropout"
    print(test_id)
    planner = TinyLidarNet(test_id,1, 4,'/home/m810z573/Downloads/f1tenth_benchmarks/f1tenth_benchmarks/zarrar/f1_tenth_model_diff_unifying_noquantized.tflite')
    
    test_mapless_all_maps(planner, test_id, number_of_laps=NUMBER_OF_LAPS)

    plot_trajectory_analysis(planner.name, test_id)

if __name__ == "__main__":
    # generate_racelines()
    # # mpcc()
    # optimisation_and_tracking()
    # follow_the_gap()
    # end_to_end_drl()
    # tinylidar_drl()
    # end_to_end_il_128()

    # tinylidar_il_temporal()
    # tinylidar_il_birdeye()
    # tinylidar_il_min()
    # tinylidar_il_max()
    # tinylidar_il_mean()
    # tinylidar_il_dropout()
    end_to_end_il()
    end_to_end_il_m()
    end_to_end_il_l()
    tinylidar_il()
    tinylidar_il_m()
    tinylidar_il_l()






