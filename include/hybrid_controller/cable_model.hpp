#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Dense>

namespace hybrid_controller {

using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Vec7 = Eigen::Matrix<double, 7, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat7 = Eigen::Matrix<double, 7, 7>;
using MatA = Eigen::Matrix<double, 12, 12>;
using MatB = Eigen::Matrix<double, 12, 3>;
using MatFx = Eigen::Matrix<double, 7, 12>;
using MatFu = Eigen::Matrix<double, 7, 3>;

struct Params {
  double m1{1.0};
  double m2{0.2};
  double g{9.81};
  double cable_length{1.0};
  double dt{0.02};
  double dt_nmpc{0.03};
  double mu{1.0e-6};
  double damping1{0.1};
  double damping2{0.05};
  int inner_iters{80};
  double residual_tol{1.0e-9};
  double linear_reg{1.0e-10};
};

struct State {
  Vec3 x1{Vec3::Zero()};
  Vec3 x2{Vec3::Zero()};
  Vec3 v1{Vec3::Zero()};
  Vec3 v2{Vec3::Zero()};
};

inline Vec3 nominal_control(const Params &params) {
  return Vec3(0.0, 0.0, (params.m1 + params.m2) * params.g);
}

inline double softplus_mu(double w, double mu) {
  return 0.5 * (w + std::sqrt(w * w + 4.0 * mu));
}

inline double d_softplus_mu(double w, double mu) {
  const double root = std::sqrt(w * w + 4.0 * mu);
  return 0.5 * (1.0 + w / root);
}

inline double softplus_mu_neg(double w, double mu) {
  return 0.5 * (-w + std::sqrt(w * w + 4.0 * mu));
}

inline double d_softplus_mu_neg(double w, double mu) {
  const double root = std::sqrt(w * w + 4.0 * mu);
  return 0.5 * (-1.0 + w / root);
}

inline double cable_gap(const State &state, const Params &params) {
  return (state.x2 - state.x1).norm() - params.cable_length;
}

class SoftplusCablePlant {
public:
  explicit SoftplusCablePlant(const Params &params) : params_(params) {
    mass_.setZero();
    mass_.diagonal() << params_.m1, params_.m1, params_.m1, params_.m2,
        params_.m2, params_.m2;

    mass_inv_.setZero();
    mass_inv_.diagonal() << 1.0 / params_.m1, 1.0 / params_.m1,
        1.0 / params_.m1, 1.0 / params_.m2, 1.0 / params_.m2, 1.0 / params_.m2;
  }

  Vec6 free_velocity(const State &state, const Vec3 &u) const {
    const Vec3 gvec(0.0, 0.0, -params_.g);
    Vec6 tau;
    tau.segment<3>(0) = params_.m1 * gvec - params_.damping1 * state.v1 +
                        nominal_control(params_) + u;
    tau.segment<3>(3) = params_.m2 * gvec - params_.damping2 * state.v2;

    Vec6 v_curr;
    v_curr.segment<3>(0) = state.v1;
    v_curr.segment<3>(3) = state.v2;
    return v_curr + params_.dt * (mass_inv_ * tau);
  }

  Vec7 initial_guess(const State &state, const Vec3 &u) const {
    const Vec6 v_free = free_velocity(state, u);
    Vec3 n = state.x2 - state.x1;
    double dist = n.norm();
    if (dist < 1.0e-9) {
      dist = 1.0e-9;
      n = Vec3(0.0, 0.0, -1.0);
    } else {
      n /= dist;
    }

    Eigen::Matrix<double, 1, 6> j_row;
    j_row << -n.transpose(), n.transpose();
    const double phi = dist - params_.cable_length;
    double s0 = -(phi + params_.dt * (j_row * v_free)(0, 0));
    s0 = std::max(1.0e-6, s0 + 1.0e-4);
    const double gamma0 = params_.mu / s0;
    const double w0 = gamma0 - s0;

    Vec7 z;
    z.segment<6>(0) = v_free;
    z(6) = w0;
    return z;
  }

  Vec7 residual(const Vec7 &z, const State &state, const Vec3 &u) const {
    Vec3 d = state.x2 - state.x1;
    double dist = d.norm();
    if (dist < 1.0e-9) {
      dist = 1.0e-9;
      d = Vec3(0.0, 0.0, -dist);
    }
    const Vec3 n = d / dist;
    const double phi = dist - params_.cable_length;

    Vec6 j_vec;
    j_vec.segment<3>(0) = -n;
    j_vec.segment<3>(3) = n;

    const Vec6 v_free = free_velocity(state, u);
    const double gamma = softplus_mu(z(6), params_.mu);
    const double s = softplus_mu_neg(z(6), params_.mu);

    Vec7 f = Vec7::Zero();
    f.segment<6>(0) =
        mass_ * (z.segment<6>(0) - v_free) + params_.dt * j_vec * gamma;
    f(6) = s + phi + params_.dt * j_vec.dot(z.segment<6>(0));
    return f;
  }

  Mat7 jacobian(const Vec7 &z, const State &state) const {
    Vec3 d = state.x2 - state.x1;
    double dist = d.norm();
    if (dist < 1.0e-9) {
      dist = 1.0e-9;
      d = Vec3(0.0, 0.0, -dist);
    }
    const Vec3 n = d / dist;

    Vec6 j_vec;
    j_vec.segment<3>(0) = -n;
    j_vec.segment<3>(3) = n;

    Mat7 jac = Mat7::Zero();
    jac.topLeftCorner<6, 6>() = mass_;
    jac.block<6, 1>(0, 6) =
        params_.dt * j_vec * d_softplus_mu(z(6), params_.mu);
    jac.block<1, 6>(6, 0) = params_.dt * j_vec.transpose();
    jac(6, 6) = d_softplus_mu_neg(z(6), params_.mu);
    return jac;
  }

  MatFx jacobian_x(const Vec7 &z, const State &state) const {
    MatFx fx = MatFx::Zero();

    Vec3 d = state.x2 - state.x1;
    double dist = d.norm();
    if (dist < 1.0e-9) {
      dist = 1.0e-9;
      d = Vec3(0.0, 0.0, -dist);
    }
    const Vec3 n = d / dist;
    const Eigen::Matrix3d proj =
        (Eigen::Matrix3d::Identity() - n * n.transpose()) / dist;
    const double gamma = softplus_mu(z(6), params_.mu);
    const Vec3 rel_v = z.segment<3>(3) - z.segment<3>(0);

    // dF1/dx through J(x) * gamma
    fx.block<3, 3>(0, 0) = params_.dt * gamma * proj;
    fx.block<3, 3>(0, 3) = -params_.dt * gamma * proj;
    fx.block<3, 3>(3, 0) = -params_.dt * gamma * proj;
    fx.block<3, 3>(3, 3) = params_.dt * gamma * proj;

    // dF1/dv through v_free(v, u)
    fx.block<3, 3>(0, 6) = (-params_.m1 + params_.dt * params_.damping1) *
                           Eigen::Matrix3d::Identity();
    fx.block<3, 3>(3, 9) = (-params_.m2 + params_.dt * params_.damping2) *
                           Eigen::Matrix3d::Identity();

    // dF2/dx
    fx.block<1, 3>(6, 0) = (-n - params_.dt * proj * rel_v).transpose();
    fx.block<1, 3>(6, 3) = (n + params_.dt * proj * rel_v).transpose();

    return fx;
  }

  MatFu jacobian_u() const {
    MatFu fu = MatFu::Zero();
    fu.block<3, 3>(0, 0) = -params_.dt * Eigen::Matrix3d::Identity();
    return fu;
  }

  bool solve(const State &state, const Vec3 &u, Vec7 *z,
             double *residual_norm = nullptr) const {
    Vec7 current = (z != nullptr) ? *z : initial_guess(state, u);
    if (current.hasNaN()) {
      current = initial_guess(state, u);
    }

    bool converged = false;
    double last_norm = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < params_.inner_iters; ++iter) {
      const Vec7 f = residual(current, state, u);
      last_norm = f.lpNorm<Eigen::Infinity>();
      if (last_norm < params_.residual_tol) {
        converged = true;
        break;
      }

      Mat7 j = jacobian(current, state);
      j.diagonal().array() += params_.linear_reg;
      Eigen::FullPivLU<Mat7> lu(j);
      if (!lu.isInvertible()) {
        break;
      }

      const Vec7 dz = lu.solve(-f);
      const double merit0 = 0.5 * f.squaredNorm();

      double alpha = 1.0;
      bool accepted = false;
      while (alpha > 1.0e-6) {
        const Vec7 trial = current + alpha * dz;
        const double merit_trial =
            0.5 * residual(trial, state, u).squaredNorm();
        if (merit_trial < merit0) {
          current = trial;
          accepted = true;
          break;
        }
        alpha *= 0.5;
      }

      if (!accepted) {
        break;
      }
    }

    if (residual_norm != nullptr) {
      *residual_norm = last_norm;
    }
    if (z != nullptr) {
      *z = current;
    }
    return converged;
  }

  State step(const State &state, const Vec3 &u, Vec7 *z,
             bool *converged = nullptr, double *residual_norm = nullptr) const {
    Vec7 current = (z != nullptr) ? *z : initial_guess(state, u);
    const bool ok = solve(state, u, &current, residual_norm);
    if (z != nullptr) {
      *z = current;
    }
    if (converged != nullptr) {
      *converged = ok;
    }

    State next = state;
    next.v1 = current.segment<3>(0);
    next.v2 = current.segment<3>(3);
    next.x1 = state.x1 + params_.dt * next.v1;
    next.x2 = state.x2 + params_.dt * next.v2;
    return next;
  }

  bool linearize(const State &state, const Vec3 &u, MatA *a, MatB *b, Vec7 *z,
                 double *residual_norm = nullptr) const {
    Vec7 current = (z != nullptr) ? *z : initial_guess(state, u);
    const bool ok = solve(state, u, &current, residual_norm);
    if (z != nullptr) {
      *z = current;
    }

    Mat7 fz = jacobian(current, state);
    fz.diagonal().array() += params_.linear_reg;

    Eigen::FullPivLU<Mat7> lu(fz);
    if (!lu.isInvertible()) {
      return false;
    }

    const MatFx fx = jacobian_x(current, state);
    const MatFu fu = jacobian_u();
    const Eigen::Matrix<double, 7, 12> dzdx = lu.solve(-fx);
    const Eigen::Matrix<double, 7, 3> dzdu = lu.solve(-fu);

    const Eigen::Matrix<double, 6, 12> dvdx = dzdx.topRows<6>();
    const Eigen::Matrix<double, 6, 3> dvdu = dzdu.topRows<6>();

    if (a != nullptr) {
      a->setZero();
      a->block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      a->block<3, 12>(0, 0) += params_.dt * dvdx.topRows<3>();
      a->block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
      a->block<3, 12>(3, 0) += params_.dt * dvdx.bottomRows<3>();
      a->block<6, 12>(6, 0) = dvdx;
    }

    if (b != nullptr) {
      b->setZero();
      b->block<3, 3>(0, 0) = params_.dt * dvdu.topRows<3>();
      b->block<3, 3>(3, 0) = params_.dt * dvdu.bottomRows<3>();
      b->block<6, 3>(6, 0) = dvdu;
    }

    return ok;
  }

  double gamma_from_w(double w) const { return softplus_mu(w, params_.mu); }

private:
  Params params_;
  Mat6 mass_;
  Mat6 mass_inv_;
};

} // namespace hybrid_controller
