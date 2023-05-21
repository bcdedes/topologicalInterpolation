#ifndef COMPUTEGRID_H
#define COMPUTEGRID_H

#include <opencv2/imgproc.hpp>

/**
 * @file computeGrid.h
 * computeCoord() method in this file composes super-resolution grid based on translational and rotational shifts and a zoom factor. It also provides a vector of 
 * locations of pixels on the high-resolution grid. mergePoints2D() is used by the computeCoord() to eliminate duplicate pixels in LR frames.
 */


/*! @brief Computes coordinates of rotated and shifted low-resolution (LR) frame pixel locations on super-resolution grid and high-resolution (HR) image pixel locations

 @param src Input set of LR images
 @param delta Input horizontal and vertical translation amounts between first LR frame and the others. For a set of N frames, the shift values are in the following format:<BR>
                \f[ \left[ {\begin{array}{ccccc}
                    0 & \Delta x_1 & \Delta x_2 & \dots & \Delta x_N\\
                    0 & \Delta y_1 & \Delta y_2 & \dots & \Delta y_N\\
                \end{array} } \right] \f]
 @param theta Input rotation angles (degrees) between first LR frame and the others. For a set of N frames, the rotation values are in the following format:<BR>
                \f[ \left[ {\begin{array}{ccccc} 
                    0 & \theta_1 & \theta_2 & \dots & \theta_N 
                \end{array} } \right] \f]
 @param points Output point set containing sample locations in rotated and shifted LR frames (sample locations in super-resolution grid).
 @param values Outout values of pixels in all LR frames (sample values in super-resolution grid).
 @param mesh Output HR image pixel locations (uniform HR image grid).
*/
void computeCoord( const std::vector<cv::Mat> &src, const std::vector<std::vector<float>> &delta, const std::vector<float> &theta,
                   int scale, std::vector<cv::Point2f> &points, std::vector<float> &values, std::vector<cv::Point2f> &mesh );


/*! @brief Looks for sample points that are coincident (within a tolerance) and averages out the values for these points

 @param points Input and Output point set containing sample locations in rotated and shifted LR frames.
 @param values Input and Output sample values of LR frame pixels.
*/
void mergePoints2D( std::vector<cv::Point2f> &points, std::vector<float> &values );

#endif
