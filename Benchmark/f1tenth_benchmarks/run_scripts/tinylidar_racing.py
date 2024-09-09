from f1tenth_benchmarks.zarrar.tiny_lidarnet import TinyLidarNet
from f1tenth_benchmarks.data_tools.plot_trajectory_analysis import plot_trajectory_analysis
from f1tenth_benchmarks.data_tools.plot_raceline_tracking import plot_raceline_tracking

from f1tenth_benchmarks.run_scripts.run_functions import *

def test_tinylidar_planning():
    test_id = "planning_tinylidar"
    map_name = "aut"
    planner = TinyLidarNet()
    test_planning_single_map(planner, map_name, test_id, number_of_laps=1)
    # test_planning_all_maps(planner, test_id, number_of_laps=5)


    plot_trajectory_analysis(planner.name, test_id)


if __name__ == "__main__":
    test_tinylidar_planning()






