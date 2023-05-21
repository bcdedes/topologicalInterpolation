#ifndef INTERPOLATIONMETHODS_H
#define INTERPOLATIONMETHODS_H

#include <vector>
#include <opencv2/imgproc.hpp>


/*! @brief Inverse distance weighting interpolation function object.

This functor implements the inverse distance weighting interpolation described in Shepard, D. "A Two-Dimensional Interpolation Function for Irregularly-Spaced Data," 
Proceedings of the 1968 ACM National Conference, New York, 27-29 August 1968, pp. 517-524. The object operator takes sample locations, sample values, brightness and 
darkness topology connectedness values as input and computes an interpolation result for each value set. The algorithm has one tuneable parameter called power parameter. 
When greater than 2, samples close to the interpolated point have more weight on the result. As power parameter gets smaller than 2, points farther away tend to have 
more emphasis on the output. Power parameter shall be grater than 0 and its default value is 2.
*/

class IDW
{
public:
    //! Constructor with power parameter value
    IDW( float _pp = 2.0 );

    /*! @brief Interpolation operator

    @param points Input point set containing sample locations in rotated and shifted LR frames.
    @param values Input sample values of LR frame pixels.
    @param DOCMb Input degree of connectedness map for bright pixels inside the interpolation window.
    @param DOCMd Input degree of connectedness map for dark pixels inside the interpolation window.
    @param E Input set of integers containing indexes of triangulation points that are inside the interpolation window.
    @param q Input query point.
    @returns a 3 element OpenCV vector that contains interpolation results computed using actual interpolation window, \p DOCMb and \p DOCMd values: [z, z_b, z_d] 
    */
    cv::Vec3f operator() ( const std::vector<cv::Point2f>& points, const std::vector<float>& values, const std::vector<float>& DOCMb,
                           const std::vector<float>& DOCMd, const std::vector<int>& E, cv::Point2f q ); 
private:
    float pp; //!<  Power parameter determines the amount of influence assigned to interpolating samples. The default value is 2.
};


/*! @brief Biharmonic spline interpolation function object.

This functor implements the biharmonic spline interpolation described in David T. Sandwell, "Biharmonic spline interpolation of GEOS-3 and SEASAT altimeter data," 
Geophysical Research Letters 14, 1987, pp. 139-142. The object operator takes sample locations, sample values, brightness and darkness topology connectedness values 
as input and computes an interpolation result for each value set. Interpolation value computation requires solution of linear systems of equations involving the 
inverse of Green's functions and sample values, which can produce overshoots and wrong results when the samples are noisy. To aviod this, the matrix representing 
Green's functions is diagonally loaded. As diagonal loading gets larger, output image becomes smoother. When diagonal loading parameter is close to 0 and input samples 
are noisy, solution to linear system of equations produces wrong results for some interpolation points. The default diagonal loading is 1.

*/
class biharmonic
{
public:
    //! Constructor with diagonal loading value
    biharmonic( float _dl = 1.0 );

    /*! @brief Interpolation operator

    @param points Input point set containing sample locations in rotated and shifted LR frames.
    @param values Input sample values of LR frame pixels.
    @param DOCMb Input degree of connectedness map for bright pixels inside the interpolation window.
    @param DOCMd Input degree of connectedness map for dark pixels inside the interpolation window.
    @param E Input set of integers containing indexes of triangulation points that are inside the interpolation window.
    @param q Input query point.
    @returns a 3 element OpenCV vector that contains interpolation results computed using actual interpolation window, \p DOCMb and \p DOCMd values: [z, z_b, z_d] 
    */
    cv::Vec3f operator() ( const std::vector<cv::Point2f>& points, const std::vector<float>& values, const std::vector<float>& DOCMb,
                           const std::vector<float>& DOCMd, const std::vector<int>& E, cv::Point2f q );
private:

    float dl; //!< Diagonal loading parameter controls the stability of the solution and smoothnes of resulting image. The default value is 1.
};

#endif