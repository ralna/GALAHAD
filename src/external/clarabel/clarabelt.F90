! TO DO: translate to fortran

!!$#include <clarabel.hpp>
!!$#include <Eigen/Eigen>
!!$#include <cmath>
!!$#include <gtest/gtest.h>
!!$#include <iostream>
!!$#include <limits>
!!$#include <vector>
!!$
!!$using namespace std;
!!$using namespace clarabel;
!!$using namespace Eigen;
!!$
!!$class BasicQPTest : public ::testing::Test
!!${
!!$  protected:
!!$    SparseMatrix<double> P, A;
!!$    Vector<double, 2> c = { 1., 1. };
!!$    Vector<double, 6> b = { -1., 0., 0., 1., 0.7, 0.7 };
!!$    vector<SupportedConeT<double>> cones = {
!!$        NonnegativeConeT<double>(3),
!!$        NonnegativeConeT<double>(3)
!!$    };
!!$    DefaultSettings<double> settings = DefaultSettings<double>::default_settings();
!!$
!!$    BasicQPTest()
!!$    {
!!$        MatrixXd P_dense(2, 2);
!!$        P_dense << 4., 1.,
!!$            1., 2.;
!!$        P = P_dense.sparseView();
!!$        P.makeCompressed();
!!$
!!$        MatrixXd A_dense(6, 2);
!!$        A_dense <<
!!$            -1., -1.,
!!$            -1., 0.,
!!$            0., -1.,
!!$            1., 1.,
!!$            1., 0.,
!!$            0., 1.;
!!$        A = A_dense.sparseView();
!!$        A.makeCompressed();
!!$    }
!!$};
!!$
!!$TEST_F(BasicQPTest, Univariate)
!!${
!!$    P = MatrixXd::Identity(1, 1).sparseView();
!!$    P.makeCompressed();
!!$
!!$    Vector<double, 1> c1 { 0. };
!!$
!!$    A = MatrixXd::Identity(1, 1).sparseView();
!!$    A.makeCompressed();
!!$
!!$    Vector<double, 1> b1 { 1. };
!!$
!!$    vector<SupportedConeT<double>> cones = {
!!$        NonnegativeConeT<double>(1)
!!$    };
!!$
!!$    DefaultSolver<double> solver(P, c1, A, b1, cones, settings);
!!$    solver.solve();
!!$
!!$    DefaultSolution<double> solution = solver.solution();
!!$    ASSERT_EQ(solution.status, SolverStatus::Solved);
!!$
!!$    // Compare the solution to the reference solution
!!$    ASSERT_NEAR(solution.x[0], 0.0, 1e-6);
!!$    ASSERT_NEAR(solution.obj_val, 0.0, 1e-6);
!!$    ASSERT_NEAR(solution.obj_val_dual, 0.0, 1e-6);
!!$}
!!$
!!$TEST_F(BasicQPTest, Feasible)
!!${
!!$    DefaultSolver<double> solver(P, c, A, b, cones, settings);
!!$    solver.solve();
!!$
!!$    DefaultSolution<double> solution = solver.solution();
!!$    ASSERT_EQ(solution.status, SolverStatus::Solved);
!!$
!!$    // Check the solution
!!$    Vector2d ref_solution{ 0.3, 0.7 };
!!$    ASSERT_EQ(solution.x.size(), 2);
!!$    ASSERT_TRUE(solution.x.isApprox(ref_solution, 1e-6));
!!$
!!$    double ref_obj = 1.8800000298331538;
!!$    ASSERT_NEAR(solution.obj_val, ref_obj, 1e-6);
!!$    ASSERT_NEAR(solution.obj_val_dual, ref_obj, 1e-6);
!!$}
!!$
!!$TEST_F(BasicQPTest, PrimalInfeasible)
!!${
!!$    b[0] = -1.;
!!$    b[3] = -1.;
!!$
!!$    DefaultSolver<double> solver(P, c, A, b, cones, settings);
!!$    solver.solve();
!!$    DefaultSolution<double> solution = solver.solution();
!!$
!!$    ASSERT_EQ(solution.status, SolverStatus::PrimalInfeasible);
!!$    ASSERT_TRUE(std::isnan(solution.obj_val));
!!$    ASSERT_TRUE(std::isnan(solution.obj_val_dual));
!!$
!!$}
!!$
!!$class BasicQPDualInfeasibleTest : public ::testing::Test
!!${
!!$  protected:
!!$    SparseMatrix<double> P, A;
!!$    Vector<double, 2> c = { 1., -1. };
!!$    Vector<double, 2> b = { 1., 1. };
!!$    vector<SupportedConeT<double>> cones = {
!!$        NonnegativeConeT<double>(2)
!!$    };
!!$    DefaultSettings<double> settings = DefaultSettings<double>::default_settings();
!!$
!!$    BasicQPDualInfeasibleTest()
!!$    {
!!$        P = MatrixXd::Ones(2, 2).sparseView();
!!$        P.makeCompressed();
!!$
!!$        MatrixXd A_dense(2, 2);
!!$        A_dense <<
!!$            1., 1.,
!!$            1., 0.;
!!$        A = A_dense.sparseView();
!!$        A.makeCompressed();
!!$    }
!!$};
!!$
!!$TEST_F(BasicQPDualInfeasibleTest, DualInfeasible)
!!${
!!$    DefaultSolver<double> solver(P, c, A, b, cones, settings);
!!$    solver.solve();
!!$    DefaultSolution<double> solution = solver.solution();
!!$
!!$
!!$    ASSERT_EQ(solution.status, SolverStatus::DualInfeasible);
!!$    ASSERT_TRUE(std::isnan(solution.obj_val));
!!$    ASSERT_TRUE(std::isnan(solution.obj_val_dual));
!!$}
!!$
!!$TEST_F(BasicQPDualInfeasibleTest, DualInfeasibleIllConditioned)
!!${
!!$    MatrixXd A_dense(1, 2);
!!$    A_dense << 1., 1.;
!!$    A = A_dense.sparseView();
!!$    A.makeCompressed();
!!$
!!$    cones = {
!!$        NonnegativeConeT<double>(1)
!!$    };
!!$
!!$    Vector<double, 1> b1{ 1. };
!!$
!!$    DefaultSolver<double> solver(P, c, A, b1, cones, settings);
!!$    solver.solve();
!!$    DefaultSolution<double> solution = solver.solution();
!!$
!!$    ASSERT_EQ(solution.status, SolverStatus::DualInfeasible);
!!$    ASSERT_TRUE(std::isnan(solution.obj_val));
!!$    ASSERT_TRUE(std::isnan(solution.obj_val_dual));
!!$}
