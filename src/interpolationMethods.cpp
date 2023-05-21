#include "interpolationMethods.h"

using namespace std;
using namespace cv;

// Inverse Distance Weighted Interpolation
IDW::IDW( float _pp ) : pp( _pp ) {}
Vec3f IDW::operator() ( const vector<Point2f>& points, const vector<float>& values, const vector<float>& DOCMb,
                        const vector<float>& DOCMd, const vector<int>& E, Point2f q )
{
    Vec3f z; // output = [z, z_b, z_d]
    vector<float> d( E.size() ); // distance vector
    Point2f p;
    int idx;
    float w; // weight
    float w_sum = 0.0; // sum of weights

    for( int i = 0; i < E.size(); i++ )
    {
        p = points.at( E[i] ) - q;
        d[i] = sqrt( p.x*p.x + p.y*p.y );
    }

    auto it = find_if( d.begin(), d.end(), [](float v){return v == 0;} );
    if( it != d.end() )
    {
        idx = distance( d.begin(), it );
        z[0] = values.at( E[idx] );
        z[1] = DOCMb.at( idx );
        z[2] = DOCMd.at( idx );
        return z;
    }
    else
    {
        for( int i = 0; i < E.size(); i++ )
        {
            w = pow( d[i], -1*pp );
            z[0] += w * values.at( E[i] );
            z[1] += w * DOCMb.at( i );
            z[2] += w * DOCMd.at( i );

            w_sum += w;
        }
        return z / w_sum;
    }
}

// Biharmonic spline interpolation
biharmonic::biharmonic( float _dl ) : dl( _dl ) {}
Vec3f biharmonic::operator() ( const vector<Point2f>& points, const vector<float>& values, const vector<float>& DOCMb,
                               const vector<float>& DOCMd, const vector<int>& E, Point2f q )
{
    Vec3f z; // output = [z, z_b, z_d]
    Mat g( E.size(), E.size(), CV_32FC1, Scalar_<float>(dl) ); // Green's function
    Mat alpha_c( E.size(), 1, CV_32FC1 ); // weights
    Mat alpha_b( E.size(), 1, CV_32FC1 ); 
    Mat alpha_d( E.size(), 1, CV_32FC1 );
    Mat w_c( E.size(), 1, CV_32FC1 ); 
    Mat w_b( E.size(), 1, CV_32FC1 );
    Mat w_d( E.size(), 1, CV_32FC1 );

    float d;
    for( int i = 0; i < E.size(); i++ )
    {
        w_c.at<float>(i) = values.at( E[i] );
        w_b.at<float>(i) = DOCMb[i];
        w_d.at<float>(i) = DOCMd[i];
        for( int j = i + 1; j < E.size(); j++ )
        {
            d = norm( points.at( E[i] ) - points.at( E[j] ) );
            d = pow(d, 2) * (log(d) - 1);
            g.at<float>(i, j) = d;
            g.at<float>(j, i) = d;
        }
    }

    // alpha = g^-1 * w
    solve( g, w_c, alpha_c, DECOMP_LU | DECOMP_NORMAL ); 
    solve( g, w_b, alpha_b, DECOMP_LU | DECOMP_NORMAL );
    solve( g, w_d, alpha_d, DECOMP_LU | DECOMP_NORMAL );

    g = Mat( E.size(), 1, CV_32FC1, Scalar_<float>(0.0) );
    for( int i = 0; i < E.size(); i++ )
    {
        d = norm( points.at( E[i] ) - q );
        if( d > 0 )
            g.at<float>(i) = pow(d, 2) * (log(d) - 1);
    }

    z[0] = g.dot(alpha_c);
    z[1] = g.dot(alpha_b);
    z[2] = g.dot(alpha_d);
    return z;
}