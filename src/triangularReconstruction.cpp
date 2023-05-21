#include "triangularReconstruction.h"

using namespace cv;
using namespace std;

int triangularReconstruction::triangleEnclosingPoint( Subdiv2D& subdiv, const vector<float>& values, Point2f q,
                                                      Point2f (&vertices)[3], float (&vtxValues)[3] )
{ 
    int ID, edge;
    int startingEdge = 0;
    int coincidingVertex = 0;

    subdiv.locate( q, startingEdge, coincidingVertex );

    if( startingEdge > 0 ) // if p does not coincide with a vertex in triangulation
    { 
        edge = startingEdge;
        for( int i = 0; i < 3; i++ ) // always 3 edges
        {
            ID = subdiv.edgeOrg( edge, &vertices[i] );
            if( ID < 4 ) // if not inside some facet (triangle)
                return -1;

            vtxValues[i] = values[ID - 4]; // 0th null and 3 virtual vertices are discarded
            edge = subdiv.getEdge( edge, Subdiv2D::NEXT_AROUND_LEFT );
        }
    }
    return coincidingVertex;
}

float triangularReconstruction::linearTriangularInterp( const Point2f (&vertices)[3], const float (&values)[3], Point2f pt )
{
    // Compute barycentric coordinates (a1, a2, a3)
    float a1, a2, a3;
    Point2f v0 = vertices[1] - vertices[0], v1 = vertices[2] - vertices[0], v2 = pt - vertices[0];
    float denum = v0.x * v1.y - v1.x * v0.y;

    a3 = (v0.x * v2.y - v2.x * v0.y) / denum;
    a2 = (v2.x * v1.y - v1.x * v2.y) / denum;
    a1 = 1.0f - a2 - a3;
    
    // Compute interpolation value
    return a1*values[0] + a2*values[1] + a3*values[2];
}

void triangularReconstruction::reconstruct( const std::vector<cv::Point2f>& points, const std::vector<float>& values, 
                                            const std::vector<cv::Point2f>& grid, cv::Mat& dst )
{
    Point2f vertices[3];
    float vtxValues[3];
    int vertexID;
    float res;

    Subdiv2D subdiv = Subdiv2D( boundingRect(points) );
    subdiv.insert( points );

    for( int i = 0; i < grid.size(); i++ )
    {
        vertexID = triangleEnclosingPoint( subdiv, values, grid[i], vertices, vtxValues );
        if ( vertexID == 0 )
        {
            res = linearTriangularInterp( vertices, vtxValues, grid[i] );
            dst.at<uchar>(grid[i].y, grid[i].x) = cvRound( res );
        }
        else if ( vertexID > 0 )
        {
            dst.at<uchar>(grid[i].y, grid[i].x) = cvRound( values[vertexID - 4] );
        }
    }
}