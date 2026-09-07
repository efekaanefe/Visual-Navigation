#include "scripts/utils.hpp"
#include "scripts/graph_slam.hpp"

#include <iostream>
#include <string>


int main() {
    std::cout << "Hello, GTSAM\n" << std::endl;

    // Read data file
    std::string filepath = "../data/input_INTEL_g2o.g2o";
    Data data;
    data = read_data(filepath);
    print_data(data, 10);

    // Slam algorithm
    
    // Save GT and predicted trajectories as csv


    return 0;
}
