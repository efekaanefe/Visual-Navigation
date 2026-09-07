#pragma once

#include "gtsam/geometry/Pose2.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/slam/BetweenFactor.h"
#include "gtsam/slam/PriorFactor.h"

#include "gtsam/nonlinear/ISAM2.h"
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
    Eigen::Matrix3d cov_matrix = get_covariance_matrix( curr_edge.info );
    noiseModel::Diagonal::shared_ptr odometry_noise = noiseModel::Diagonal::Sigmas( cov_matrix.diagonal().cwiseSqrt() );

    graph->add( BetweenFactor<Pose2>( curr_edge.indeces[0], curr_edge.indeces[1], odometry_mean, odometry_noise ) );
};

void add_prior_factor( NonlinearFactorGraph *graph ) {
    Pose2 priorMean( 0, 0, 0 );
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas( Vector3( 0.3, 0.3, 0.1 ) );
    graph->add( PriorFactor<Pose2>( 0, priorMean, priorNoise ) );
};

NonlinearFactorGraph construct_graph( Data *data ) {
    NonlinearFactorGraph graph;

    // prior
    add_prior_factor( &graph );

    // adding one step for each measurements/edges
    for ( int curr_index = 0; curr_index < data->edges.size(); curr_index++ ) {
        propogate_graph_onestep( &graph, data, curr_index );
    }

    graph.print();

    return graph;
};

Values get_initial_guess( const Data *data ) {
    Values initial;
    for ( const auto &vertex : data->vertices ) {
        initial.insert( vertex.index, gtsam::Pose2( vertex.x, vertex.y, vertex.theta ) );
    }

    return initial;
};

std::vector<Vertex_SE2> construct_optimized_traj( Values *result ) {
    std::vector<Vertex_SE2> trajectory;

    for ( const auto &key_value : *result ) {
        Key key = key_value.key;

        Pose2 pose = key_value.value.cast<Pose2>();

        Vertex_SE2 vertex;
        vertex.index = key;
        vertex.x = pose.x();
        vertex.y = pose.y();
        vertex.theta = pose.theta();

        trajectory.push_back( vertex );
    }

    return trajectory;
}

std::vector<Vertex_SE2> solve_slam( Data *data, bool use_incremental = false ) {

    if ( !use_incremental ) {

        NonlinearFactorGraph graph = construct_graph( data );
        Values initial = get_initial_guess( data );

        Values result = LevenbergMarquardtOptimizer( graph, initial ).optimize();
        return construct_optimized_traj( &result );

    } else {

        ISAM2 isam;
        Values initial;

        for ( int i = 0; i < data->vertices.size(); i++ ) {
            NonlinearFactorGraph new_graph;
            Values new_initial;

            Vertex_SE2 curr_vertex = data->vertices[i];
            Key vertex_key = curr_vertex.index;

            if ( i == 0 ) {
                add_prior_factor( &new_graph );
            }

            new_initial.insert( vertex_key, Pose2( curr_vertex.x, curr_vertex.y, curr_vertex.theta ) );

            for ( int j = 0; j < data->edges.size(); j++ ) {
                Edge_SE2 curr_edge = data->edges[j];

                // This prevents double-adding factors and prevents referencing vertices that have not been added yet
                if ( curr_edge.indeces[1] == vertex_key ) {
                    propogate_graph_onestep( &new_graph, data, j );
                }
            }

            isam.update( new_graph, new_initial );
            initial = isam.calculateEstimate();
        }

        return construct_optimized_traj( &initial );
    }
};
