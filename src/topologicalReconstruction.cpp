#include "topologicalReconstruction.h"

using namespace cv;
using namespace std;

int sgn( float val )
{
    return (float(0) < val) - (val < float(0));
}

topologicalReconstruction::topologicalReconstruction( float length, int power )
{
    this->windowLength = length;
    this->updatePower = power;
}

vector<int> topologicalReconstruction::getWindow( const vector<Point2f>& points, Point2f q, float len )
{
    vector<int> E;

    // Square window
    auto neighborhoodCheck = [q, len](Point2f pt)
    {
        return pt.x >= q.x - len/2 && pt.x <= q.x + len/2 && 
               pt.y >= q.y - len/2 && pt.y <= q.y + len/2;
    };

    auto it = find_if( begin(points), end(points), neighborhoodCheck );
    while( it != end(points) )
    {
        E.push_back( distance(points.begin(), it) );
        it = find_if( next(it), end(points), neighborhoodCheck );
    }
    return E;
}

vector<int> topologicalReconstruction::getNaturalNeighbors( const Subdiv2D& subdiv, const vector<int>& E, int qIdx )
{
    vector<int> neighbors;
    int startingEdge, edge, vertexIdx;
    Point2f center = subdiv.getVertex( qIdx + 4, &startingEdge );
    
    edge = startingEdge;
    do
    {
        vertexIdx = subdiv.edgeDst( edge ) - 4; // discard 0th null and 3 virtual vertices
        if( any_of( E.begin(), E.end(), [vertexIdx](int idx){ return idx == vertexIdx; } ) ) // if inside operation window
        {
            neighbors.push_back(vertexIdx);
        }
        edge = subdiv.nextEdge(edge);
    }
    while( edge != startingEdge );

    return neighbors;
}

vector<bool> topologicalReconstruction::binaryConnectedness( const Subdiv2D& subdiv, const vector<int>& E, const vector<bool>& B_t, int qIdx )
{
    int i, center;
    vector<int> neighbors;
    vector<int> queue;

    vector<bool> J( E.size(), false );
    auto it = find( E.begin(), E.end(), qIdx );
    J.at( distance(E.begin(), it) ) = true; // starting vertex is always connected

    queue.push_back(qIdx); // put starting vertex index in queue stack

    while( !queue.empty() )
    {
        qIdx = queue.back();
        queue.pop_back();

        neighbors = getNaturalNeighbors( subdiv, E, qIdx );
        it = find( E.begin(), E.end(), qIdx );
        center = distance( begin(E), it );
        for( int vtxIdx : neighbors ) // for each neighboring vertex
        {
            it = find( E.begin(), E.end(), vtxIdx );
            i = distance( E.begin(), it );
            if( !J[i] && J[center] && B_t[i] )
            {
                J[i] = true; // mark sample as connected
                queue.push_back(vtxIdx); // add to queue
            }
        }
    }
    return J;
}

vector<float> topologicalReconstruction::computeDOCM( const Subdiv2D& subdiv, const vector<float>& values, const vector<int>& E, int qIdx )
{
    vector<float> C_E( E.size(), 0.0 ); // Degree of connectedness map

    float DOM_q = values[qIdx]; // degree of membership of center (query) point
    auto it = find( E.begin(), E.end(), qIdx ); 
    C_E.at( distance(E.begin(), it) ) = DOM_q;

    #ifdef USE_TBB
    tbb::parallel_for_each( E.begin(), E.end(), [&](int idx) // for each vertex in set E
    #else
    for_each( E.begin(), E.end(), [&](int idx)
    #endif
    {
        vector<bool> J; // binary connectedness
        vector<bool> B_t( E.size() ); // thresholded binary image    
        if( values[idx] <= DOM_q ) // for all p_i in E such that µ(p_i) <= µ(p)
        {
            for( int k = 0; k < E.size(); k++ )
            {
                B_t[k] = values.at( E[k] ) >= values[idx]; // threshold E at level µ(p_i)
            }
            J = binaryConnectedness( subdiv, E, B_t, qIdx );
            for( int j = 0; j < E.size(); j++ ) // for each p_j in set E
            {
                if( J[j] && C_E[j] <= values[idx] ) // s.t. p_j ~ p in B_t and C_E(p,p_j) <= µ(p_i)
                {
                    C_E[j] = values[idx];
                }
            }
        }
    });

    return C_E;
}

template <typename Func>
void topologicalReconstruction::reconstruct( vector<Point2f>& points, vector<float>& values, Func interp, vector<Point2f>& grid, Mat& dst )
{
    Subdiv2D dt = Subdiv2D( boundingRect(points) );
    dt.insert( points );
    
    vector<float> negatives = values; // vector containing negative pixel values
    for( auto& element : negatives )
        element = 255 - element;

    vector<int> E;
    vector<float> DOCMb; // DOCM for bright pixels
    vector<float> DOCMd; // DOCM for dark pixels

    float alpha, rec, w_d, w_b;
    int p;
    Vec3f z; // [z_c, z_b, z_d]

    chrono::time_point start = chrono::system_clock::now();
    for( int i = 0; i < grid.size(); i++ )
    {
        // Operation window
        E = getWindow( points, grid[i], windowLength );
        p = dt.findNearest( grid[i] ) - 4; // index of the query point (0th null and 3 virtual vertices in dt are discarded)

        DOCMb = computeDOCM( dt, values, E, p );
        DOCMd = computeDOCM( dt, negatives, E, p );
        for( float& element : DOCMd )
            element = 255 - element;

        z = interp( points, values, DOCMb, DOCMd, E, grid[i] );

        alpha = sgn( z[0] - ( z[1] + z[2] ) / 2 ) * ( z[2] - z[1] ) / 255;
        w_b = 0.5 * pow( 1 + alpha, updatePower );
        w_d = 0.5 * pow( 1 - alpha, updatePower );
        rec = ( w_d * z[2] + w_b * z[1] ) / ( w_b + w_d );

        dst.at<uchar>(grid[i].y, grid[i].x) = cvRound( rec );
    }
    chrono::duration<double> executionTime = chrono::system_clock::now() - start;
    printf( "\nExecution time: %0.2f seconds\n", executionTime.count() );
}

template void topologicalReconstruction::reconstruct<IDW>( vector<Point2f>& points, vector<float>& values, IDW interp, vector<Point2f>& grid, Mat& dst );
template void topologicalReconstruction::reconstruct<biharmonic>( vector<Point2f>& points, vector<float>& values, biharmonic interp, vector<Point2f>& grid, Mat& dst );