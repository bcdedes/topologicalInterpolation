#ifndef TOPOLOGICALRECONSTRUCTION_H
#define TOPOLOGICALRECONSTRUCTION_H

#include "interpolationMethods.h"
#include <opencv2/imgproc.hpp>
#ifdef USE_TBB
#include <tbb/parallel_for_each.h>
#endif


/*! \mainpage Nonuniform Interpolation based on Image Topology for Super-Resolution Reconstruction

This program implements the topology based interpolation technique for utilization in high resolution pixel estimation stage of super-resolution image reconstruction process as described in 
B. Dedes, "Inverse distance weighted interpolation based on image topology for super-resolution," MSc Thesis, Dept. of Electrical and Electronics Eng., Anadolu Univ., Eskisehir, 2021. In the 
work, Degree of connectedness map computation algorithms in the literature are revised to work on nonuniform data. Then, connectedness map properties in brightness and darkness topology are 
exploited in order to improve the results of inverse distance weighting interpolation along the edges in super-resolution imaging.

In the thesis, connectivity maps in each interpolation window are computed using an initial point inserted into triangulation. The value of this initial point is computed using linear 
triangular interpolation. However, Subdiv2D class of OpenCV does not have a point deletion method, hence, any initial points inserted are left in the triangulation. To avoid this, connectivity 
maps are computed using the nearest sample to the query point. It is observed that this modificaiton does not significantly alter the results.

This program does not cover the image registration stage of super-resolution reconstruction process. Hence, pre-computed registration parameters shall be provided by the user. Using known
registration parameters, a super-resolution grid is produced by the \c computeCoord() method. For N low-resolution frames, format of the translation matrix is:
                                        \f[ \left[ {\begin{array}{ccccc}
                                            0 & \Delta x_1 & \Delta x_2 & \dots & \Delta x_{N-1}\\
                                            0 & \Delta y_1 & \Delta y_2 & \dots & \Delta y_{N-1}\\
                                        \end{array} } \right] \f]
and the format of the rotation vector is:
                                        \f[ \left[ {\begin{array}{ccccc} 
                                            0 & \theta_1 & \theta_2 & \dots & \theta_{N-1} 
                                        \end{array} } \right] \f]
where each variable represents the shift and rotation angle with respect to the first LR frame. Unit of translational shifts is fraction of a 1 unit length (corresponding to a 1 pixel length) 
and unit of rotation is degrees. These parameters are suplied to the \c computeCoord() method as in the example below. Then, using the \ref topologicalReconstruction::reconstruct "reconstruct()" 
method of \c topologicalReconstruction class, low resolution pixel values are interpolated, interpolated values close to an edge are pulled towards to either lower or upper bound computed using 
brighness and darkness topologies. Interpolation function is provided as a functor argument to the \ref topologicalReconstruction::reconstruct "reconstruct()" method. In the following example, 
a set of 9 LR frames with known dislocation parameters are reconstructed using inverse distance weighting interpolation.


<b>Example</b>
 \code{.cpp}
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
 \endcode

Since real super-resolution datasets are hard to come by, a MATLAB routine, \c generateFrames.m, for synthetic low-resolution image generation is provided to demonstrate the possibilities 
of the method. Implementation of biharmonic spline interpolation, another global interpolation method, is also available in \ref biharmonic class to experiment with topology based technique. 
In addition, linear triangular interpolation is provided in \c triangularReconstruction class for reference. All algorithms are implemented for grayscale images but can be extended to color 
by computing connectedness information using grayscale intensity values and then interpolating each color channel.

The major disadvantage of the technique is its slow execution time due to high computational requirements of degree of connectedness map calculations. However, it can be executed in paralled 
in order to speed up the process since connectedness computation at each sample in the interpolation window do not require the results from the others. One of the eaiser ways of adding 
parallelism to the code is to use a performance library such as Intel's oneAPI Threading Building Blocks (TBB). When \c USE_TBB macro is defined, instead of the regular for loop, oneTBB's 
parallel for loop is utilized in connectedness computations and a performance improvement is achieved depending on the number of processor cores available.

The program is tested on a Linux machine with gcc 11, opencv version 4.5 and libtbb version 2020.1-2.
*/



/*! \brief This class contains methods to compute degree of connectedness map for nonuniform data and performs super resolution reconstruction by combining interpolation results calculated using 
brightness and darkness topologies.

The class has one public method with the signiture:

 \code{.cpp}
void reconstruct( std::vector<cv::Point2f>& points, std::vector<float>& values, Func interp, std::vector<cv::Point2f>& grid, cv::Mat& dst )
 \endcode

 \c reconstruct() method takes nonuniform pixel locations, pixel values, interpolation function and uniform high resolution image grid as inputs and places the reconstructed image in \p dst.
 (Nonuniform) low resolution image pixel locations and (uniform) high resolution image pixel locations should be pre-computed and their scales should match. That is, if a zoom factor of k
 is desired, distances between each LR sample locations should be enlarged such that a signle LR frame covers the entire HR grid. This scaling is performed for translational and rotational 
 shifts in computeCoord() method.

*/
class topologicalReconstruction
{
public:
    /*! @brief Topology based super-resolution reconstruction object constructor

    The default constructor assigns the member variables \p windowLength and \p updatePower the values of 7.0 and 4, respectively.

    @param length Input (optional) interpolation window length.
    @param power Input (optional) sharpness adjustment parameter.
    */
    topologicalReconstruction( float length = 7.0, int power = 4 );


private:
    int updatePower; //!< Sharpness adjustment parameter. It is a power parameter greater than 1. As it increases, edges on resulting high resolution image are sharpened. The default value is 4.
    float windowLength;  //!< Interpolation window length. Length of the local window on which the connectedness information is computed and on which the interpolated value is calculated. The default value is 7.


    /*! @brief Finds indexes of points those fall in a given operation window

    @param points Input point set containing sample locations in rotated and shifted LR frames.
    @param q Input operation window center point.
    @param len Input operation window length.
    @returns a vector of integers containing indexes of input point set elements that are inside the operation window.
    */
    std::vector<int> getWindow( const std::vector<cv::Point2f>& points, cv::Point2f q, float len );


    /*! @brief Finds indexes of points that are connected by an edge to a given triangulation vertex

    @param subdiv Input Delaunay triangulation of a point set.
    @param E Input set of integers containing indexes of triangulation points that are inside the operation window \p E.
    @param qIdx Input point index to find the natural neighbors for.
    @returns a vector of integers containing indexes of triangulation points that are natural neighbors, within the window \p E, to the vertex at index \p qIdx 
    */
    std::vector<int> getNaturalNeighbors( const cv::Subdiv2D& subdiv, const std::vector<int>& E, int qIdx );


    /*! @brief Computes connectedness information of a center element to others in a binary set.

    @param subdiv Input Delaunay triangulation of a point set.
    @param E Input set of integers containing indexes of triangulation points that are inside the operation window \p E.
    @param B_t Input binary set of the same size as set \p E (obtained by thresholding triangulation samples at some intensity value).
    @param qIdx Input center point index used to compute the binary connectedness map.
    @returns a binary connectedness map containing the value of \p true if an element at corresponding index in \p B_t is connected to \p qIdx
    */
    std::vector<bool> binaryConnectedness( const cv::Subdiv2D& subdiv, const std::vector<int>& E, const std::vector<bool>& B_t, int qIdx );


    /*! @brief Computes degree of connectedness map between a center point and others within the operation window.

    @param subdiv Input Delaunay triangulation of a point set.
    @param values Input pixel values at corresponding point locations in triangulation.
    @param E Input set of integers containing indexes of triangulation points that are inside the operation window \p E.
    @param qIdx Input center point index used to compute the degree of connectedness map.
    @returns a degree of connectedness map that specify α-connectedness information between the center pixel and others within the window \p E
    */
    std::vector<float> computeDOCM( const cv::Subdiv2D& subdiv, const std::vector<float>& values, const std::vector<int>& E, int qIdx );


public:
    /** @brief Reconstructs a single high-resolution (HR) image from several low-resolution (LR) frames.

    @param points Input point set containing sample locations in rotated and shifted LR frames.
    @param values Input sample values of LR frame pixels.
    @param interp Input functor of interpolation algorithm to be used.
    @param grid Input uniform grid of HR image sample locations (interpolation values are computed for these points)
    @param dst Output reconstructed high resolution image.
    */
    template<typename Func> 
    void reconstruct( std::vector<cv::Point2f>& points, std::vector<float>& values, Func interp, std::vector<cv::Point2f>& grid, cv::Mat& dst );
};

#endif