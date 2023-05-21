# Topological Interpolation
### Nonuniform Interpolation based on Image Topology for Super-Resolution Reconstruction

This program implements the topology based interpolation technique for utilization in high resolution pixel estimation stage of super-resolution image reconstruction process as described in 
B. Dedes, "Inverse distance weighted interpolation based on image topology for super-resolution," MSc Thesis, Dept. of Electrical and Electronics Eng., Anadolu Univ., Eskisehir, 2021. In the 
work, Degree of connectedness map computation algorithms in the literature are revised to work on nonuniform data. Then, connectedness map properties in brightness and darkness topology are 
exploited in order to improve the results of inverse distance weighting interpolation along the edges in super-resolution imaging.

In the thesis, connectivity maps in each interpolation window are computed using an initial point inserted into triangulation. The value of this initial point is computed using linear 
triangular interpolation. However, Subdiv2D class of OpenCV does not have a point deletion method, hence, any initial points inserted are left in the triangulation. To avoid this, connectivity 
maps are computed using the nearest sample to the query point. It is observed that this modificaiton does not significantly alter the results.

This program does not cover the image registration stage of super-resolution reconstruction process. Hence, pre-computed registration parameters shall be provided by the user. Using known
registration parameters, a super-resolution grid is produced by the computeCoord() method. For N low-resolution frames, format of the translation matrix is:
```math
          \begin{bmatrix}0 & \Delta x_1 & \Delta x_2 & \dots & \Delta x_{N-1} \\  0 & \Delta y_1 & \Delta y_2 & \dots & \Delta y_{N-1}\end{bmatrix}
```
and the format of the rotation vector is:
```math
          \begin{bmatrix}0 & \theta_1 & \theta_2 & \dots & \theta_{N-1} \end{bmatrix}
```
where each variable represents the shift and rotation angle with respect to the first LR frame. Unit of translational shifts is fraction of a 1 unit length (corresponding to a 1 pixel length) 
and unit of rotation is degrees. These parameters are suplied to the computeCoord() method as in the example below. Then, using the reconstruct() method of topologicalReconstruction class, low 
resolution pixel values are interpolated, interpolated values close to an edge are pulled towards to either lower or upper bound computed using brighness and darkness topologies. Interpolation 
function is provided as a functor argument to the reconstruct() method. In the following example, a set of 9 LR frames with known dislocation parameters are reconstructed using inverse distance 
weighting interpolation.


```
#include "computeGrid.h"
#include "topologicalReconstruction.h"
#include "interpolationMethods.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    IDW idw( 2.0 );
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
    Mat topological( boundingRect(hiResGrid).size(), CV_8UC1 );
    topologyBasedSuperRes.reconstruct( pixels, pixelValues, idw, hiResGrid, topological );

    imshow( "Topological Interpolation", topological );
    waitKey(0);
    return 0;
}
```

Since real super-resolution datasets are hard to come by, a MATLAB routine, generateFrames.m, for synthetic low-resolution image generation is provided to demonstrate the possibilities of the 
method. Implementation of biharmonic spline interpolation, another global interpolation method, is also available in biharmonic class to experiment with topology based technique. In addition, 
linear triangular interpolation is provided in triangularReconstruction class for reference. All algorithms are implemented for grayscale images but can be extended to color by computing 
connectedness information using grayscale intensity values and then interpolating each color channel.

The major disadvantage of the technique is its slow execution time due to high computational requirements of degree of connectedness map calculations. However, it can be executed in paralled 
in order to speed up the process since connectedness computation at each sample in the interpolation window do not require the results from the others. One of the eaiser ways of adding 
parallelism to the code is to use a performance library such as Intel's oneAPI Threading Building Blocks (TBB). When USE_TBB macro is defined, instead of the regular for loop, oneTBB's 
parallel for loop is utilized in connectedness computations and a performance improvement is achieved depending on the number of processor cores available.

The program is tested on a Linux machine with gcc 11, opencv version 4.5 and libtbb version 2020.1-2. This project is licensed under the terms of the GNU General Public License v3.0.
