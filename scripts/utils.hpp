#include <boost/exception/exception.hpp>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

struct Vertex_SE2 {
    int index;
    float x, y, theta;
};
struct Vertex_XY;

struct Edge_SE2 {
    int indeces[2];
    float x, y, theta;
    float info[6];
};

struct Data {
    std::vector<Vertex_SE2> vertices;
    std::vector<Edge_SE2> edges;
};

Data read_data( std::string filepath ) {
    Data data;
    std::ifstream file( filepath );

    if ( !file.is_open() ) {
        std::cerr << "Error while reading data\n";
        return data;
    }

    std::string line;
    while ( std::getline( file, line ) ) {
        std::istringstream iss( line );

        std::string type;
        iss >> type;

        if ( type == "VERTEX_SE2" ) {
            int index;
            float x, y, theta;
            iss >> index >> x >> y >> theta;

            Vertex_SE2 vertex{ .index = index, .x = x, .y = y, .theta = theta };
            data.vertices.push_back( vertex );
        }

        else if ( type == "EDGE_SE2" ) {
            int indeces[2];
            float x, y, theta;
            float info[6];

            iss >> indeces[0] >> indeces[1] >> x >> y >> theta >> info[0] >> info[1] >> info[2] >> info[3] >> info[4] >>
                info[5];

            Edge_SE2 edge{ .x = x, .y = y, .theta = theta };
            edge.indeces[0] = indeces[0];
            edge.indeces[1] = indeces[1];
            for ( int i = 0; i < 6; ++i ) {
                edge.info[i] = info[i];
            }

            data.edges.push_back( edge );
        }
    }

    file.close();
    return data;
};

void print_vertex( const Vertex_SE2 &v ) {
    printf( "Vertex_SE2:\n  index=%d, x=%.3f, y=%.3f, theta=%.3f\n", v.index, v.x, v.y, v.theta );
}

void print_edge( const Edge_SE2 &e ) {
    printf( "Edge_SE2:\n" );
    printf( "  indices: [%d, %d]\n", e.indeces[0], e.indeces[1] );
    printf( "  measurement: x=%.3f, y=%.3f, theta=%.3f\n", e.x, e.y, e.theta );

    printf( "  info: [" );
    for ( int i = 0; i < 6; ++i ) {
        printf( "%.3f%s", e.info[i], i < 5 ? ", " : "" );
    }
    printf( "]\n" );
}

void print_data( Data data, int max_index ) {
    printf( "Printing data\n" );

    for ( int i = 0; i < max_index && i < data.vertices.size(); i++ ) {
        print_vertex( data.vertices[i] );
    }

    for ( int i = 0; i < max_index && i < data.edges.size(); i++ ) {
        print_edge( data.edges[i] );
    }
};

Eigen::Matrix3f get_information_matrix( const float info[6] ) {
    Eigen::Matrix3f information_matrix;
    information_matrix << info[0], info[1], info[2], info[1], info[3], info[4], info[2], info[4], info[5];
    return information_matrix;
}

Eigen::Matrix3f get_covariance_matrix( const float info[6] ) {
    Eigen::Matrix3f covariance = get_information_matrix(info);
    return covariance.inverse();
}

// void save_trajectory_as_csv(std::vector<data_point> trajectory);
