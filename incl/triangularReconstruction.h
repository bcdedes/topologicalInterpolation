#ifndef TRIANGULARRECONSTRUCTION_H
#define TRIANGULARRECONSTRUCTION_H

#include <opencv2/imgproc.hpp>


/*! \brief This class contains methods to perform super resolution reconstruction using linear triangular interpolation based on Delaunay triangularion of data.

The class has one public method with the signiture:

 \code{.cpp}
void reconstruct( std::vector<cv::Point2f>& points, std::vector<float>& values, std::vector<cv::Point2f>& grid, cv::Mat& dst )
 \endcode

 \c reconstruct() method takes nonuniform pixel locations, pixel values and uniform high resolution image grid as inputs and places the reconstructed image in \p dst.
 (Nonuniform) low resolution image pixel locations and (uniform) high resolution image pixel locations should be pre-computed and their scales should match. That is, 
 if a zoom factor of k is desired, distances between each LR sample locations should be enlarged such that a signle LR frame covers the entire HR grid. This scaling 
 is performed for translational and rotational shifts in computeCoord() method.

*/
class triangularReconstruction
{
private:
    /*! @brief Returns triangle vertices enclosing a point.

    @param subdiv Delaunay triangulation of a point set.
    @param values Input pixel values at corresponding point locations in triangulation
    @param q Input query point.
    @param vertices Output vertices of the triangle enclosing query point.
    @param vtxValues Output pixel values at the vertices of the triangle enclosing query point.
    @returns an integer which specify one of the following three cases for query point location:
        - -1 when query point \p q is not inside some triangle (out of triangulation)
        -  0 when query point \p q is inside some triangle. Then, \p vertices will contain the triangle vertices and \p vtxValues will contain the corresponding pixel values
        - > 0 when query point \p q coincides with one of the triangle vertices. Then, returned integer is the conciding vertex ID
    */
    int triangleEnclosingPoint( cv::Subdiv2D& subdiv, const std::vector<float>& values, cv::Point2f q, 
                                cv::Point2f (&vertices)[3], float (&vtxValues)[3] );



    /*! @brief Computes linear triangular interpolation value of a point using barycentric coordinates.

    @param vertices Input vertices of the triangle enclosing query point.
    @param vtxValues Input pixel values at the vertices of the triangle enclosing query point.
    @param pt Input query point.
    @returns interpolation value
    */
    float linearTriangularInterp( const cv::Point2f (&vertices)[3], const float (&values)[3], cv::Point2f pt );


public:
    /** @brief Reconstructs a single high-resolution (HR) image from several low-resolution (LR) frames.

    @param points Input point set containing sample locations in rotated and shifted LR frames.
    @param values Input sample values of LR frame pixels.
    @param grid Input uniform grid of HR image sample locations (interpolation values are computed for these points)
    @param dst Output reconstructed high resolution image.
    */
    void reconstruct( const std::vector<cv::Point2f>& points, const std::vector<float>& values, 
                      const std::vector<cv::Point2f>& grid, cv::Mat& dst );

};

#endif