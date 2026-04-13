#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

#include "hybrid_controller/cable_model.hpp"

namespace hybrid_controller {

struct ILQRConfig {
  int horizon{100};
  int max_iters{10};
  double reg_min{1.0e-6};
  double reg_max{1.0e6};
  double reg_init{1.0};
  std::array<double, 6> alpha_list{{1.0, 0.5, 0.25, 0.1, 0.05, 0.01}};
  Vec3 u_min{Vec3(-20.0, -20.0, -20.0)};
  Vec3 u_max{Vec3(20.0, 20.0, 20.0)};
};

struct Weights {
  Eigen::Matrix3d q_payload{Eigen::Vector3d(40.0, 40.0, 80.0).asDiagonal()};
  Eigen::Matrix3d q_carrier_vel{Eigen::Vector3d(5.0, 5.0, 5.0).asDiagonal()};
  Eigen::Matrix3d q_payload_vel{Eigen::Vector3d(5.0, 5.0, 5.0).asDiagonal()};
  // Eigen::Matrix3d r_u{Eigen::Vector3d(9.0e-1, 9.0e-1, 9.0e-1).asDiagonal()};
  Eigen::Matrix3d r_u{Eigen::Vector3d(9.0e-3, 9.0e-3, 9.0e-3).asDiagonal()};
  Eigen::Matrix3d qf_payload{Eigen::Vector3d(120.0, 120.0, 180.0).asDiagonal()};
};

struct RolloutResult {
  std::vector<State> x_seq;
  std::vector<Vec3> u_seq;
  std::vector<Vec7> z_seq;
  std::vector<MatA> a_seq;
  std::vector<MatB> b_seq;
  double cost{0.0};
};

class ILQRCost {
public:
  ILQRCost(const Weights &weights, const Vec3 &payload_target)
      : weights_(weights), payload_target_(payload_target) {}

  double stage_cost(const State &state, const Vec3 &u) const {
    const Vec3 ep = state.x2 - payload_target_;
    return (ep.transpose() * weights_.q_payload * ep)(0, 0) +
           (state.v1.transpose() * weights_.q_carrier_vel * state.v1)(0, 0) +
           (state.v2.transpose() * weights_.q_payload_vel * state.v2)(0, 0) +
           (u.transpose() * weights_.r_u * u)(0, 0);
  }

  double terminal_cost(const State &state) const {
    const Vec3 ep = state.x2 - payload_target_;
    return (ep.transpose() * weights_.qf_payload * ep)(0, 0);
  }

  double total_cost(const std::vector<State> &x_seq,
                    const std::vector<Vec3> &u_seq) const {
    double cost = 0.0;
    for (std::size_t k = 0; k < u_seq.size(); ++k) {
      cost += stage_cost(x_seq[k], u_seq[k]);
    }
    return cost + terminal_cost(x_seq.back());
  }

  void stage_derivatives(const State &state, const Vec3 &u,
                         Eigen::Matrix<double, 12, 1> *lx, Eigen::Vector3d *lu,
                         Eigen::Matrix<double, 12, 12> *lxx,
                         Eigen::Matrix3d *luu,
                         Eigen::Matrix<double, 3, 12> *lux) const {
    lx->setZero();
    lu->setZero();
    lxx->setZero();
    luu->setZero();
    lux->setZero();

    const Vec3 ep = state.x2 - payload_target_;
    lx->segment<3>(3) = 2.0 * weights_.q_payload * ep;
    lx->segment<3>(6) = 2.0 * weights_.q_carrier_vel * state.v1;
    lx->segment<3>(9) = 2.0 * weights_.q_payload_vel * state.v2;
    *lu = 2.0 * weights_.r_u * u;

    lxx->block<3, 3>(3, 3) = 2.0 * weights_.q_payload;
    lxx->block<3, 3>(6, 6) = 2.0 * weights_.q_carrier_vel;
    lxx->block<3, 3>(9, 9) = 2.0 * weights_.q_payload_vel;
    *luu = 2.0 * weights_.r_u;
  }

  void terminal_derivatives(const State &state,
                            Eigen::Matrix<double, 12, 1> *vx,
                            Eigen::Matrix<double, 12, 12> *vxx) const {
    vx->setZero();
    vxx->setZero();

    const Vec3 ep = state.x2 - payload_target_;
    vx->segment<3>(3) = 2.0 * weights_.qf_payload * ep;
    vxx->block<3, 3>(3, 3) = 2.0 * weights_.qf_payload;
  }

private:
  Weights weights_;
  Vec3 payload_target_;
};

class IFTiLQRController {
public:
  IFTiLQRController(const SoftplusCablePlant &dynamics_plant,
                    const SoftplusCablePlant &linearization_plant,
                    const ILQRConfig &cfg, const ILQRCost &cost)
      : dynamics_plant_(dynamics_plant),
        linearization_plant_(linearization_plant), cfg_(cfg), cost_(cost),
        u_seq_(static_cast<std::size_t>(cfg.horizon), Vec3::Zero()),
        reg_(cfg.reg_init) {}

  struct Solution {
    std::vector<State> x_seq;
    std::vector<Vec3> u_seq;
    double cost{0.0};
  };

  void shift_warmstart() {
    if (u_seq_.empty()) {
      return;
    }
    for (std::size_t i = 0; i + 1U < u_seq_.size(); ++i) {
      u_seq_[i] = u_seq_[i + 1U];
    }
    u_seq_.back() = u_seq_[u_seq_.size() - 2U];
  }

  Solution solve(const State &x0) {
    std::vector<Vec3> u = u_seq_;
    RolloutResult rollout = rollout_trajectory(x0, u);

    for (int iter = 0; iter < cfg_.max_iters; ++iter) {
      std::vector<Eigen::Matrix<double, 3, 12>> k_seq(
          static_cast<std::size_t>(cfg_.horizon));
      std::vector<Vec3> d_seq(static_cast<std::size_t>(cfg_.horizon),
                              Vec3::Zero());

      if (!backward_pass(rollout, &k_seq, &d_seq)) {
        reg_ = std::min(reg_ * 10.0, cfg_.reg_max);
        continue;
      }

      bool accepted = false;
      RolloutResult best_rollout = rollout;
      std::vector<Vec3> best_u = u;

      for (double alpha : cfg_.alpha_list) {
        auto trial = forward_pass(x0, rollout.x_seq, u, k_seq, d_seq, alpha);
        if (trial.second.cost < best_rollout.cost) {
          best_rollout = std::move(trial.second);
          best_u = std::move(trial.first);
          accepted = true;
          break;
        }
      }

      if (accepted) {
        u = std::move(best_u);
        rollout = std::move(best_rollout);
        reg_ = std::max(reg_ / 5.0, cfg_.reg_min);
      } else {
        reg_ = std::min(reg_ * 10.0, cfg_.reg_max);
      }

      if (reg_ >= cfg_.reg_max) {
        break;
      }
    }

    u_seq_ = u;
    return Solution{rollout.x_seq, u, rollout.cost};
  }

private:
  static Vec3 clamp_control(const Vec3 &u, const Vec3 &u_min,
                            const Vec3 &u_max) {
    Vec3 out = u;
    for (int i = 0; i < 3; ++i) {
      out(i) = std::clamp(out(i), u_min(i), u_max(i));
    }
    return out;
  }

  RolloutResult rollout_trajectory(const State &x0,
                                   const std::vector<Vec3> &u_seq) const {
    RolloutResult result;
    result.x_seq.reserve(static_cast<std::size_t>(cfg_.horizon) + 1U);
    result.u_seq.reserve(static_cast<std::size_t>(cfg_.horizon));
    result.z_seq.reserve(static_cast<std::size_t>(cfg_.horizon));
    result.a_seq.reserve(static_cast<std::size_t>(cfg_.horizon));
    result.b_seq.reserve(static_cast<std::size_t>(cfg_.horizon));

    result.x_seq.push_back(x0);
    Vec7 z_guess = Vec7::Constant(std::numeric_limits<double>::quiet_NaN());

    for (int k = 0; k < cfg_.horizon; ++k) {
      const Vec3 uk = clamp_control(u_seq[static_cast<std::size_t>(k)],
                                    cfg_.u_min, cfg_.u_max);
      bool converged = false;
      double residual_norm = 0.0;
      MatA a = MatA::Zero();
      MatB b = MatB::Zero();
      const bool linearized = linearization_plant_.linearize(
          result.x_seq.back(), uk, &a, &b, &z_guess, &residual_norm);
      const State x_next = dynamics_plant_.step(
          result.x_seq.back(), uk, &z_guess, &converged, &residual_norm);

      result.cost += cost_.stage_cost(result.x_seq.back(), uk);
      if (!linearized || !converged) {
        result.cost += 1.0e5 + 1.0e4 * residual_norm;
      }

      result.x_seq.push_back(x_next);
      result.u_seq.push_back(uk);
      result.z_seq.push_back(z_guess);
      result.a_seq.push_back(a);
      result.b_seq.push_back(b);
    }

    result.cost += cost_.terminal_cost(result.x_seq.back());
    return result;
  }

  bool backward_pass(const RolloutResult &rollout,
                     std::vector<Eigen::Matrix<double, 3, 12>> *k_seq,
                     std::vector<Vec3> *d_seq) const {
    Eigen::Matrix<double, 12, 1> vx = Eigen::Matrix<double, 12, 1>::Zero();
    Eigen::Matrix<double, 12, 12> vxx = Eigen::Matrix<double, 12, 12>::Zero();
    cost_.terminal_derivatives(rollout.x_seq.back(), &vx, &vxx);

    const Eigen::Matrix3d reg_i = reg_ * Eigen::Matrix3d::Identity();

    for (int k = cfg_.horizon - 1; k >= 0; --k) {
      Eigen::Matrix<double, 12, 1> lx = Eigen::Matrix<double, 12, 1>::Zero();
      Eigen::Vector3d lu = Eigen::Vector3d::Zero();
      Eigen::Matrix<double, 12, 12> lxx = Eigen::Matrix<double, 12, 12>::Zero();
      Eigen::Matrix3d luu = Eigen::Matrix3d::Zero();
      Eigen::Matrix<double, 3, 12> lux = Eigen::Matrix<double, 3, 12>::Zero();

      cost_.stage_derivatives(rollout.x_seq[static_cast<std::size_t>(k)],
                              rollout.u_seq[static_cast<std::size_t>(k)], &lx,
                              &lu, &lxx, &luu, &lux);

      const MatA &a = rollout.a_seq[static_cast<std::size_t>(k)];
      const MatB &b = rollout.b_seq[static_cast<std::size_t>(k)];

      const Eigen::Matrix<double, 12, 1> qx = lx + a.transpose() * vx;
      const Eigen::Vector3d qu = lu + b.transpose() * vx;
      const Eigen::Matrix<double, 12, 12> qxx = lxx + a.transpose() * vxx * a;
      const Eigen::Matrix3d quu = luu + b.transpose() * vxx * b;
      const Eigen::Matrix<double, 3, 12> qux = lux + b.transpose() * vxx * a;

      const Eigen::Matrix3d quu_reg = 0.5 * (quu + quu.transpose()) + reg_i;
      Eigen::LLT<Eigen::Matrix3d> llt(quu_reg);
      if (llt.info() != Eigen::Success) {
        return false;
      }

      const Vec3 dk = llt.solve(-qu);
      const Eigen::Matrix<double, 3, 12> kk = llt.solve(-qux);

      (*d_seq)[static_cast<std::size_t>(k)] = dk;
      (*k_seq)[static_cast<std::size_t>(k)] = kk;

      vx = qx + kk.transpose() * quu * dk + kk.transpose() * qu +
           qux.transpose() * dk;
      vxx = qxx + kk.transpose() * quu * kk + kk.transpose() * qux +
            qux.transpose() * kk;
      vxx = 0.5 * (vxx + vxx.transpose());
    }

    return true;
  }

  std::pair<std::vector<Vec3>, RolloutResult>
  forward_pass(const State &x0, const std::vector<State> &x_nom,
               const std::vector<Vec3> &u_nom,
               const std::vector<Eigen::Matrix<double, 3, 12>> &k_seq,
               const std::vector<Vec3> &d_seq, double alpha) const {
    std::vector<Vec3> u_new;
    u_new.reserve(static_cast<std::size_t>(cfg_.horizon));

    RolloutResult rollout;
    rollout.x_seq.reserve(static_cast<std::size_t>(cfg_.horizon) + 1U);
    rollout.u_seq.reserve(static_cast<std::size_t>(cfg_.horizon));
    rollout.z_seq.reserve(static_cast<std::size_t>(cfg_.horizon));
    rollout.a_seq.reserve(static_cast<std::size_t>(cfg_.horizon));
    rollout.b_seq.reserve(static_cast<std::size_t>(cfg_.horizon));

    rollout.x_seq.push_back(x0);
    Vec7 z_guess = Vec7::Constant(std::numeric_limits<double>::quiet_NaN());

    for (int k = 0; k < cfg_.horizon; ++k) {
      Eigen::Matrix<double, 12, 1> dx = Eigen::Matrix<double, 12, 1>::Zero();
      const State &current = rollout.x_seq.back();
      const State &nominal = x_nom[static_cast<std::size_t>(k)];

      dx.segment<3>(0) = current.x1 - nominal.x1;
      dx.segment<3>(3) = current.x2 - nominal.x2;
      dx.segment<3>(6) = current.v1 - nominal.v1;
      dx.segment<3>(9) = current.v2 - nominal.v2;

      Vec3 uk = u_nom[static_cast<std::size_t>(k)] +
                alpha * d_seq[static_cast<std::size_t>(k)] +
                k_seq[static_cast<std::size_t>(k)] * dx;
      uk = clamp_control(uk, cfg_.u_min, cfg_.u_max);
      u_new.push_back(uk);

      bool converged = false;
      double residual_norm = 0.0;
      MatA a = MatA::Zero();
      MatB b = MatB::Zero();
      const bool linearized = linearization_plant_.linearize(
          current, uk, &a, &b, &z_guess, &residual_norm);
      const State x_next = dynamics_plant_.step(current, uk, &z_guess,
                                                &converged, &residual_norm);

      rollout.cost += cost_.stage_cost(current, uk);
      if (!linearized || !converged) {
        rollout.cost += 1.0e5 + 1.0e4 * residual_norm;
      }

      rollout.x_seq.push_back(x_next);
      rollout.u_seq.push_back(uk);
      rollout.z_seq.push_back(z_guess);
      rollout.a_seq.push_back(a);
      rollout.b_seq.push_back(b);
    }

    rollout.cost += cost_.terminal_cost(rollout.x_seq.back());
    return {u_new, rollout};
  }

  const SoftplusCablePlant &dynamics_plant_;
  const SoftplusCablePlant &linearization_plant_;
  ILQRConfig cfg_;
  ILQRCost cost_;
  std::vector<Vec3> u_seq_;
  double reg_;
};

class InverseDynamicsControllerNode : public rclcpp::Node {
public:
  InverseDynamicsControllerNode()
      : Node("hybrid_inverse_dynamics_controller_node"),
        dynamics_plant_(dynamics_params_),
        linearization_plant_(linearization_params_),
        cost_(weights_, payload_target_),
        controller_(dynamics_plant_, linearization_plant_, cfg_, cost_) {
    carrier_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/carrier/odom", 10,
        std::bind(&InverseDynamicsControllerNode::carrier_callback, this,
                  std::placeholders::_1));
    payload_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/payload/odom", 10,
        std::bind(&InverseDynamicsControllerNode::payload_callback, this,
                  std::placeholders::_1));

    force_pub_ =
        create_publisher<geometry_msgs::msg::Vector3>("/carrier/force_cmd", 10);
    cost_pub_ =
        create_publisher<std_msgs::msg::Float64>("/hybrid_controller/cost", 10);
    solve_time_pub_ = create_publisher<std_msgs::msg::Float64>(
        "/hybrid_controller/solve_time_ms", 10);
    target_error_pub_ = create_publisher<std_msgs::msg::Float64>(
        "/hybrid_controller/payload_error_norm", 10);
    carrier_path_pub_ = create_publisher<nav_msgs::msg::Path>(
        "/hybrid_controller/carrier_horizon_path", 10);
    payload_path_pub_ = create_publisher<nav_msgs::msg::Path>(
        "/hybrid_controller/payload_horizon_path", 10);

    timer_ = create_wall_timer(
        std::chrono::duration<double>(dynamics_params_.dt_nmpc),
        std::bind(&InverseDynamicsControllerNode::step, this));

    RCLCPP_INFO(get_logger(),
                "Hybrid IFT-iLQR controller node started "
                "(mu_dynamics=%.3e, mu_ilqr=%.3e)",
                dynamics_params_.mu, linearization_params_.mu);
  }

private:
  void carrier_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    state_.x1 = Vec3(msg->pose.pose.position.x, msg->pose.pose.position.y,
                     msg->pose.pose.position.z);
    state_.v1 = Vec3(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                     msg->twist.twist.linear.z);
    have_carrier_ = true;
  }

  void payload_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    state_.x2 = Vec3(msg->pose.pose.position.x, msg->pose.pose.position.y,
                     msg->pose.pose.position.z);
    state_.v2 = Vec3(msg->twist.twist.linear.x, msg->twist.twist.linear.y,
                     msg->twist.twist.linear.z);
    have_payload_ = true;
  }

  void step() {
    if (!have_carrier_ || !have_payload_) {
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto sol = controller_.solve(state_);
    const double solve_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - t0)
                                .count();

    const Vec3 u0 = sol.u_seq.front();
    controller_.shift_warmstart();

    geometry_msgs::msg::Vector3 msg;
    msg.x = u0.x();
    msg.y = u0.y();
    msg.z = u0.z();
    force_pub_->publish(msg);

    const Vec3 ep = state_.x2 - payload_target_;
    const auto stamp = get_clock()->now();

    carrier_path_pub_->publish(make_path(stamp, "world", sol.x_seq, true));
    payload_path_pub_->publish(make_path(stamp, "world", sol.x_seq, false));

    std_msgs::msg::Float64 scalar;
    scalar.data = sol.cost;
    cost_pub_->publish(scalar);
    scalar.data = solve_ms;
    solve_time_pub_->publish(scalar);
    scalar.data = ep.norm();
    target_error_pub_->publish(scalar);

    RCLCPP_INFO(
        get_logger(),
        "cost=%.3f solve_ms=%.3f payload=[%.3f %.3f %.3f] u0=[%.3f %.3f %.3f]",
        sol.cost, solve_ms, state_.x2.x(), state_.x2.y(), state_.x2.z(), u0.x(),
        u0.y(), u0.z());
  }

  nav_msgs::msg::Path make_path(const rclcpp::Time &stamp,
                                const std::string &frame_id,
                                const std::vector<State> &x_seq,
                                bool carrier) const {
    nav_msgs::msg::Path path;
    path.header.stamp = stamp;
    path.header.frame_id = frame_id;
    path.poses.reserve(x_seq.size());

    for (const auto &state : x_seq) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.stamp = stamp;
      pose.header.frame_id = frame_id;
      const Vec3 &pos = carrier ? state.x1 : state.x2;
      pose.pose.position.x = pos.x();
      pose.pose.position.y = pos.y();
      pose.pose.position.z = pos.z();
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }

    return path;
  }

  Params dynamics_params_{[] {
    Params p;
    p.mu = 1.0e-3;
    return p;
  }()};
  Params linearization_params_{[] {
    Params p;
    p.mu = 1.0e-3;
    return p;
  }()};
  ILQRConfig cfg_{};
  Weights weights_{};
  Vec3 payload_target_{Vec3(2.5, 2.5, 2.5)};
  SoftplusCablePlant dynamics_plant_;
  SoftplusCablePlant linearization_plant_;
  ILQRCost cost_;
  IFTiLQRController controller_;
  State state_{};
  bool have_carrier_{false};
  bool have_payload_{false};

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr carrier_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr payload_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr force_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr cost_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr solve_time_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr target_error_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr carrier_path_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr payload_path_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

} // namespace hybrid_controller

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(
      std::make_shared<hybrid_controller::InverseDynamicsControllerNode>());
  rclcpp::shutdown();
  return 0;
}
