#include "computeGrid.h"
#include "triangularReconstruction.h"
#include "topologicalReconstruction.h"
#include "interpolationMethods.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    IDW idw( 2.0 );
    biharmonic gdatav4( 1.0 );

    triangularReconstruction linearSuperRes;
    topologicalReconstruction topologyBasedSuperRes;

    std::vector<Point2f> pixels;
    std::vector<float> pixelValues;
    std::vector<Point2f> hiResGrid;

    vector<Mat> frames;
    vector<cv::String> fn;

    glob( "../../lowResolutionFrames/plate_3x_LR/*.tif", fn, false );
    for( int i = 0; i < fn.size(); i++ )
        frames.push_back( imread(fn[i], CV_8UC1) );

    vector<vector<float>> delta = {
            {+0.0000, -0.6667, -0.6667, +0.3333, +0.6667, -0.6667, +0.6667, +0.3333, +0.3333},
            {+0.0000, +0.6667, +0.0000, +0.6667, +0.6667, +0.0000, -0.6667, +0.6667, +0.0000}
    };
    vector<float> theta = {+0.0000, -1.5796, -3.0695, -1.2258, +0.7295, -3.6170, +3.3553, -2.3098, +1.2052};

    computeCoord( frames, delta, theta, 3, pixels, pixelValues, hiResGrid );

    Mat linear( boundingRect(hiResGrid).size(), CV_8UC1 );
    Mat topological( boundingRect(hiResGrid).size(), CV_8UC1 );

    linearSuperRes.reconstruct( pixels, pixelValues, hiResGrid, linear );
    topologyBasedSuperRes.reconstruct( pixels, pixelValues, idw, hiResGrid, topological );

    imshow( "Linear Interpolation", linear );
    imshow( "Topological Interpolation", topological );
    waitKey(0);
    return 0;
}