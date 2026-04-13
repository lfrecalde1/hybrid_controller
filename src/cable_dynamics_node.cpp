#include <chrono>
#include <limits>
#include <memory>
#include <string>

#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

#include "hybrid_controller/cable_model.hpp"

namespace hybrid_controller {

class CableDynamicsNode : public rclcpp::Node {
public:
  CableDynamicsNode() : Node("hybrid_cable_dynamics_node"), plant_(params_) {
    state_.x1 = Vec3(0.0, 0.0, 0.0);
    state_.x2 = Vec3(0.0, 0.0, -0.3);
    state_.v1 = Vec3::Zero();
    state_.v2 = Vec3::Zero();

    force_sub_ = create_subscription<geometry_msgs::msg::Vector3>(
        "/carrier/force_cmd", 10,
        std::bind(&CableDynamicsNode::force_callback, this,
                  std::placeholders::_1));

    carrier_pub_ =
        create_publisher<nav_msgs::msg::Odometry>("/carrier/odom", 10);
    payload_pub_ =
        create_publisher<nav_msgs::msg::Odometry>("/payload/odom", 10);
    tension_pub_ =
        create_publisher<std_msgs::msg::Float64>("/cable/tension", 10);
    solve_time_pub_ = create_publisher<std_msgs::msg::Float64>(
        "/hybrid_dynamics/solve_time_ms", 10);
    residual_pub_ = create_publisher<std_msgs::msg::Float64>(
        "/hybrid_dynamics/residual_inf", 10);
    phi_pub_ =
        create_publisher<std_msgs::msg::Float64>("/hybrid_dynamics/phi", 10);

    timer_ = create_wall_timer(std::chrono::duration<double>(params_.dt),
                               std::bind(&CableDynamicsNode::step, this));

    RCLCPP_INFO(get_logger(), "Hybrid cable dynamics node started (mu=%.3e)",
                params_.mu);
  }

private:
  void force_callback(const geometry_msgs::msg::Vector3::SharedPtr msg) {
    last_force_ = Vec3(msg->x, msg->y, msg->z);
  }

  void step() {
    auto t0 = std::chrono::steady_clock::now();
    bool converged = false;
    double residual_norm = 0.0;
    state_ =
        plant_.step(state_, last_force_, &z_guess_, &converged, &residual_norm);
    const double solve_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

    const double gamma = plant_.gamma_from_w(z_guess_(6));
    const double phi = cable_gap(state_, params_);
    const auto stamp = get_clock()->now();

    carrier_pub_->publish(
        make_odom(stamp, "world", "carrier", state_.x1, state_.v1));
    payload_pub_->publish(
        make_odom(stamp, "world", "payload", state_.x2, state_.v2));

    std_msgs::msg::Float64 msg;
    msg.data = gamma;
    tension_pub_->publish(msg);
    msg.data = solve_ms;
    solve_time_pub_->publish(msg);
    msg.data = residual_norm;
    residual_pub_->publish(msg);
    msg.data = phi;
    phi_pub_->publish(msg);

    RCLCPP_INFO(get_logger(),
                "t=%.2f solve_ms=%.3f residual=%.3e phi=%.3e gamma=%.3e "
                "converged=%s u=[%.3f %.3f %.3f]",
                sim_time_, solve_ms, residual_norm, phi, gamma,
                converged ? "true" : "false", last_force_.x(), last_force_.y(),
                last_force_.z());

    sim_time_ += params_.dt;
  }

  nav_msgs::msg::Odometry make_odom(const rclcpp::Time &stamp,
                                    const std::string &frame_id,
                                    const std::string &child_frame_id,
                                    const Vec3 &position,
                                    const Vec3 &velocity) const {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = frame_id;
    msg.child_frame_id = child_frame_id;
    msg.pose.pose.position.x = position.x();
    msg.pose.pose.position.y = position.y();
    msg.pose.pose.position.z = position.z();
    msg.pose.pose.orientation.w = 1.0;
    msg.twist.twist.linear.x = velocity.x();
    msg.twist.twist.linear.y = velocity.y();
    msg.twist.twist.linear.z = velocity.z();
    return msg;
  }

  Params params_{[] {
    Params p;
    p.mu = 1.0e-8;
    return p;
  }()};
  SoftplusCablePlant plant_;
  State state_{};
  Vec3 last_force_{Vec3::Zero()};
  Vec7 z_guess_{Vec7::Constant(std::numeric_limits<double>::quiet_NaN())};
  double sim_time_{0.0};

  rclcpp::Subscription<geometry_msgs::msg::Vector3>::SharedPtr force_sub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr carrier_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr payload_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr tension_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr solve_time_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr residual_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr phi_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace hybrid_controller

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hybrid_controller::CableDynamicsNode>());
  rclcpp::shutdown();
  return 0;
}
