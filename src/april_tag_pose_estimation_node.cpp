// ros
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include <Eigen/Dense>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}


typedef Eigen::Matrix<double, 3, 3, Eigen::RowMajor> Mat3;

rcl_interfaces::msg::ParameterDescriptor
desc(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor desc;

    desc.description = description;
    desc.read_only = read_only;

    return desc;
}

void getTagPose(const std::array<double, 9UL>& H,
             const Mat3& Pinv,
             geometry_msgs::msg::Transform& t,
             const double size)
{
    // compute extrinsic camera parameter
    // https://dsp.stackexchange.com/a/2737/31703
    // H = K * T  =>  T = K^(-1) * H
    const Mat3 T = Pinv * Eigen::Map<const Mat3>(H.data());
    Mat3 R;
    R.col(0) = T.col(0).normalized();
    R.col(1) = T.col(1).normalized();
    R.col(2) = R.col(0).cross(R.col(1));

    // rotate by half rotation about x-axis to have z-axis
    // point upwards orthogonal to the tag plane
    R.col(1) *= -1;
    R.col(2) *= -1;

    // the corner coordinates of the tag in the canonical frame are (+/-1, +/-1)
    // hence the scale is half of the edge size
    const Eigen::Vector3d tt = T.rightCols<1>() / ((T.col(0).norm() + T.col(0).norm()) / 2.0) * (size / 2.0);

    const Eigen::Quaterniond q(R);

    t.translation.x = tt.x();
    t.translation.y = tt.y();
    t.translation.z = tt.z();
    t.rotation.w = q.w();
    t.rotation.x = q.x();
    t.rotation.y = q.y();
    t.rotation.z = q.z();
}


class AprilTabPoseEstimationNode : public rclcpp::Node {
public:
    AprilTabPoseEstimationNode(const rclcpp::NodeOptions& options);

    ~AprilTabPoseEstimationNode() override;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<bool> profile;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

    const image_transport::CameraSubscriber sub_cam;
    const rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_ci_;
    const rclcpp::Subscription<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr sub_detections_;
    tf2_ros::TransformBroadcaster tf_broadcaster_;

    void onDetections(const apriltag_msgs::msg::AprilTagDetectionArray::ConstSharedPtr& msg_detection);
    void onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTabPoseEstimationNode)


AprilTabPoseEstimationNode::AprilTabPoseEstimationNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTabPoseEstimationNode::onParameter, this, std::placeholders::_1))),
    // topics
    sub_ci_(create_subscription<sensor_msgs::msg::CameraInfo>("camera_info", rclcpp::QoS(1), std::bind(&AprilTabPoseEstimationNode::onCameraInfo, this, std::placeholders::_1))),
    sub_detections_(create_subscription<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1), std::bind(&AprilTabPoseEstimationNode::onDetections, this, std::placeholders::_1))),
    tf_broadcaster_(this)
{
    // read-only parameters
    tag_edge_size = declare_parameter("size", 1.0, desc("default tag size", true));

    // get tag names, IDs and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, desc("tag ids", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, desc("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, desc("tag sizes per id", true));

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
    }

}

AprilTabPoseEstimationNode::~AprilTabPoseEstimationNode()
{
}

void AprilTabPoseEstimationNode::onDetections(const apriltag_msgs::msg::AprilTagDetectionArray::ConstSharedPtr& msg_detection)
{
    // precompute inverse projection matrix
    const Mat3 Pinv = Eigen::Map<const Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(camera_info_->p.data()).leftCols<3>().inverse();

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;

    for(const auto& det : msg_detections.detections) {
        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det.id)) { continue; }

        // 3D orientation and position
        geometry_msgs::msg::TransformStamped tf;
        tf.header = msg_detection->header;
        // set child frame name by generic tag name or configured tag name
        tf.child_frame_id = tag_frames.count(det.id) ? tag_frames.at(det.id) : std::string(det.family) + ":" + std::to_string(det.id);
        getTagPose(det.homography, Pinv, tf.transform, tag_sizes.count(det.id) ? tag_sizes.at(det.id) : tag_edge_size);

        tfs.push_back(tf);
    }

    tf_broadcaster_.sendTransform(tfs);
}

void AprilTabPoseEstimationNode::onCameraInfo(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    camera_info_ = msg_ci;
}

rcl_interfaces::msg::SetParametersResult
AprilTabPoseEstimationNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

    }

    mutex.unlock();

    result.successful = true;

    return result;
}
