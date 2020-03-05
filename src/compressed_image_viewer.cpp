#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

void imageCb(const sensor_msgs::CompressedImageConstPtr& msg)
{
    try
    {
        cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_UNCHANGED);
        cv::imshow("compressed", image);
        cv::waitKey(10);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}


int main(int argc, char** argv)
{
    ros::init(argc, argv, "compressed_image_viewer");
    ros::NodeHandle n;
    ros::Subscriber depth_image_sub = n.subscribe("/camera/rgb/image_rect_color/compressed", 30, imageCb);

    ros::spin();
    return 0;
}