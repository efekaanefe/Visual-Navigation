#pragma once

#include "gtsam/geometry/Pose2.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"

#include "gtsam/nonlinear/LevenbergMarquardtOptimizer.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/Values.h"

#include <eigen3/Eigen/Dense>
#include <gtsam/nonlinear/NonlinearOptimizer.h>
#include <vector>

#include "utils.hpp"

using namespace gtsam;

void propogate_graph_onestep( NonlinearFactorGraph *graph, Data *data, int curr_index ) {

    Edge_SE2 curr_edge = data->edges[curr_index]; // these are odometries and loopclosures

    Pose2 odometry_mean( curr_edge.x, curr_edge.y, curr_edge.theta );
    Eigen::Matrix3f cov_matrix = get_covariance_matrix( curr_edge.info );
    noiseModel::Gaussian::shared_ptr odometry_noise = noiseModel::Gaussian::Covariance( cov_matrix.cast<double>() );

    graph->add( BetweenFactor<Pose2>( curr_edge.indeces[0], curr_edge.indeces[1], odometry_mean, odometry_noise ) );
};

NonlinearFactorGraph construct_graph( Data *data ) {
    int curr_index = 0;
    NonlinearFactorGraph graph;

    // prior
    Pose2 priorMean( 0, 0, 0 );
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas( Vector3( 0.3, 0.3, 0.1 ) );
    graph.add( PriorFactor<Pose2>( 0, priorMean, priorNoise ) );

    // adding one step for each measurements/edges
    for ( int curr_index = 0; curr_index < data->edges.size(); curr_index++ ) {
        propogate_graph_onestep( &graph, data, curr_index );
    }

    graph.print();

    return graph;
};

Values get_initial_guess(const Data *data) {
    Values initial;
    for (const auto& vertex : data->vertices) {
        initial.insert(vertex.index, gtsam::Pose2(vertex.x, vertex.y, vertex.theta));
    }
    
    return initial;
};

std::vector<Vertex_SE2> construct_optimized_traj( Values *result ) {
    std::vector<Vertex_SE2> trajectory;

    // Iterate through all variables stored in the Values container
    for ( const auto &key_value : *result ) {
        gtsam::Key key = key_value.key;

        // Cast the generic GTSAM Value object back into a Pose2
        // Note: This assumes all values in 'result' are Pose2.
        // If you have landmarks (Point2), you would need to check the type first.
        gtsam::Pose2 pose = key_value.value.cast<gtsam::Pose2>();

        // Map the GTSAM data to your custom Vertex_SE2 type
        Vertex_SE2 vertex;
        vertex.index = key;             // 64-bit integer key used in GTSAM
        vertex.x = pose.x();         // Translation X
        vertex.y = pose.y();         // Translation Y
        vertex.theta = pose.theta(); // Rotation in radians

        trajectory.push_back( vertex );
    }

    return trajectory;
}

std::vector<Vertex_SE2> solve_slam( Data *data ) {

    NonlinearFactorGraph graph = construct_graph( data );
    Values initial = get_initial_guess( data );

    Values result = LevenbergMarquardtOptimizer( graph, initial ).optimize();

    return construct_optimized_traj( &result );
};
