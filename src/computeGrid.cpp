#include "computeGrid.h"

#define POINT2F_EPSILON 1.7263349e-04 // SQRT(0.25*FLT_EPSILON)
#define HASH_MULTIPLIER 2000 // such that HASH_MULTIPLIER * SQRT(2*POW(POINT2F_EPSILON,2)) < 0.5

using namespace cv;
using namespace std;

// Custom specialization of std::hash injected into namespace std:
//    We would like to consider two points coinciding when the points are within 
// POINT2F_EPSILON distance from each other in both directions. This comparison,
// defined in 'coincides' struct below, requires the two points to have the same
// hash key in order to be activated. This design results in multiple collusions 
// especially among the points that lie on the same circle. However, performance
// of the container is not critical since it is only used in data preparation.
namespace std
{
    template<>
    struct hash<Point2f>
    {
        size_t operator()( Point2f const &pt ) const
        {
            return (size_t)( HASH_MULTIPLIER * sqrt(pt.x*pt.x + pt.y*pt.y) );
        }
    };
}

struct coincides
{
    inline bool operator()( const Point2f& a, const Point2f& b ) const
    {
        return ( abs(a.x - b.x) < POINT2F_EPSILON ) && ( abs(a.y - b.y) < POINT2F_EPSILON );
    }
};

struct valIterDupeCount
{
    std::vector<float>::iterator valIter;
    int dupeCount;
};

// Returns a 1D matrix of n equally spaced points in [x1, x2] closed interval
Mat linspace( float x1, float x2, int n )
{
	float spacing = (x2 - x1) / (n - 1);
	Mat f( n, 1, CV_32FC1 ); // column vector of length n
	for( int i = 0; i < f.rows; i++ )
    {
		f.at<float>(i,0) = x1 + i*spacing;
	}
	return f;
}

void computeCoord( const vector<Mat> &src, const vector<vector<float>> &delta, const vector<float> &theta, 
                   int scale, vector<Point2f> &points, vector<float> &values, vector<Point2f> &mesh )
{
    int rowNum = src[0].rows;
    int colNum = src[0].cols;
    int numel = src[0].total();
    int channelNum = src[0].channels();

    Scalar center = Scalar( (float)(colNum - 1)/2, (float)(rowNum - 1)/2 ); // image center
    vector<float> theta_rad( theta.size() ); // rotation in radians
    transform( theta.begin(), theta.end(), theta_rad.begin(), [](float deg) { return deg*CV_PI/180; } );
    
    // allocate output vectors
    points = vector<Point2f>( numel * src.size() );
    values = vector<float>( numel * src.size() );
    mesh = vector<Point2f>( numel * scale * scale );

    // LR pixel location (SR grid) parameters for row-major ordered image matrices
    Mat columnNumberOfOnes = Mat::ones( colNum, 1, CV_32FC1 );
    Mat rowNumberOfOnes = Mat::ones( rowNum, 1, CV_32FC1 );
    Mat rowIndices = linspace( 0, (rowNum-1)*scale, rowNum ); 
    Mat columnIndices = linspace( 0, (colNum-1)*scale, colNum ); 

    Mat rowMesh = rowIndices * columnNumberOfOnes.t(); // matrix of repeated row indices
    Mat columnMesh = rowNumberOfOnes * columnIndices.t(); // matrix of repeated column indices
    rowMesh = rowMesh.reshape( 0, numel ); // column vector of row indices
    columnMesh = columnMesh.reshape( 0, numel ); // column vector of column indices
    vector<Mat> meshVec { columnMesh, rowMesh };
    
    Matx23f rotMat; // rotation matrix
    Mat img; // pointer to pixel values in i-th frame
    Mat coordMesh; // pointer to pixels locations in i-th frame

    for( int i = 0; i < src.size(); i++ ) // for each LR frame
    {
        img = Mat( rowNum, colNum, CV_32FC1, next(values.data(), numel*i) ); // points to i-th frame in 'values'
        src[i].convertTo( img, CV_32F ); // convert to float and vectorize (row-major) into 'values'

        coordMesh = Mat( numel, 1, CV_32FC2, next(points.data(), numel*i) ); // points to i-th frame in 'points'
        merge( meshVec, coordMesh ); // concatenate row, column indices and store the result in 'points'

        coordMesh = coordMesh - scale*center; // shift image center to (0,0)
        // rotation
        rotMat(0,0) = cos(theta_rad[i]);     rotMat(0,1) = -sin(theta_rad[i]); 
        rotMat(1,0) = sin(theta_rad[i]);     rotMat(1,1) =  cos(theta_rad[i]);
        // translation
        rotMat(0,2) = scale * ( delta[0][i] + center[0] ); 
        rotMat(1,2) = scale * ( delta[1][i] + center[1] );
        transform( coordMesh, coordMesh, rotMat ); // rotate, then shift back and shift by delta
    }

    mergePoints2D( points, values );

    // HR pixel locations (interpolation or HR grid)
    Mat YI = linspace( 0, rowNum*scale - 1, rowNum*scale ); 
    Mat XI = linspace( 0, colNum*scale - 1, colNum*scale );
    YI = repeat( YI, 1, XI.rows ); // meshgrid of columns
    XI = repeat( XI.t(), YI.rows, 1 ); // meshgrid of rows
    YI = YI.reshape( 0, YI.total() ); // column vector of row indices
    XI = XI.reshape( 0, XI.total() ); // column vector of column indices
    vector<Mat> XYI { XI, YI }; // Point2f is (x, y)
    Mat HR_grid = Mat( numel*scale*scale, 1, CV_32FC2, mesh.data() ); // points to 'mesh'
    merge( XYI, HR_grid ); // concatenate column, row indices and store the result in 'mesh'
}

void mergePoints2D( vector<Point2f> &points, vector<float> &values )
{
    // Average out values of pixels with coinciding point locations within a tolerance
    unordered_map<Point2f, valIterDupeCount, hash<Point2f>, coincides> seen; // map of pixel locations and pointers to pixel value
    unordered_map<Point2f, valIterDupeCount>::iterator seenIter;
    float *dupeValue;
    int *dupeCount; // number of pixels in the same particular location

    vector<float>::iterator valuesIter = values.begin();
    vector<Point2f>::iterator pointsIter = points.begin();
    int totalDupes = 0; // total number of duplicates

    while( pointsIter != points.end() - totalDupes )
    {
        seenIter = seen.find(*pointsIter);
        if ( seenIter != seen.end() ) // if there is a duplicate
        {
            dupeValue = &( *(seenIter->second.valIter) ); // pointer to the value of the first occurence of the duplicate pixel
            dupeCount = &(seenIter->second.dupeCount); // pointer to number of occurrences of the duplicate pixel
            *dupeValue = ( (*dupeValue) * (*dupeCount) + (*valuesIter) ) / ++(*dupeCount); // new averaged value of the duplicate pixel
            rotate( pointsIter, next(pointsIter), points.end() );
            rotate( valuesIter, next(valuesIter), values.end() );
            totalDupes++;
        }
        else
        {
            seen.insert( { *pointsIter, valIterDupeCount{valuesIter, 1} } ); // first occurence of a pixel
            pointsIter++; // increment iterators
            valuesIter++;
        }
    }

    if( totalDupes > 0 )
    {
        points.erase( points.end() - totalDupes, points.end() );
        values.erase( values.end() - totalDupes, values.end() );
        printf( "\n%d point(s) with coinciding locations are averaged out.\n", totalDupes );
    }
}