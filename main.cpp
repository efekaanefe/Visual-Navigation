#include "scripts/graph_slam.hpp"
#include "scripts/utils.hpp"

#include <iostream>
#include <string>
#include <vector>

int main() {
    std::cout << "Hello, GTSAM\n" << std::endl;

    // Read data file
    std::string filepath = "../data/input_INTEL_g2o.g2o";
    Data data;
    data = read_data( filepath );
    print_data( data, 3 );

    // Max vertices: 1228 | Max edges: 1483

    // Slam algorithm
    std::vector<Vertex_SE2> optimized_traj = solve_slam(&data);

    // Save GT and predicted trajectories as csv
    save_trajectory(data.vertices, "../results/initial_traj.csv");
    save_trajectory(optimized_traj, "../results/optimized_traj.csv");

    return 0;
}
