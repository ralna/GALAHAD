/*
 * This file is a replacement for qpOASES_wrapper.h, a part of qpOASES,
 * version 3.2
 *
 * Nick Gould, 2026-07-22, for GALAHAD productions, with assistance from
 * the Google Gemini AI 
 *
 * Interface that enables a fortran-specific call to qpOASES via plain C
 */

#include <qpOASES/QProblem.hpp>

USING_NAMESPACE_QPOASES

extern "C" {

#include "qpoases_c_interface.hpp"

    // All in one

#ifdef __USE_SINGLE_PRECISION__
#ifdef __USE_LONG_INTEGERS__
    int_t qpOASES_c_solve_sgl_64 (
#else
    int_t qpOASES_c_solve_sgl (
#endif
#else
#ifdef __USE_LONG_INTEGERS__
    int_t qpOASES_c_solve_dbl_64 (
#else
    int_t qpOASES_c_solve_dbl (
#endif
#endif
        // Dimensions
        int_t nV, int_t nC,
        // Symmetric Hessian (H) Upper-Triangular CSC arrays
        int_t* H_row, int_t* H_ptr, real_t* H_val,
        // Gradient vector (g)
        const real_t* g,
        // General Constraint Matrix (A) CSC arrays
        int_t* A_row, int_t* A_ptr, real_t* A_val,
        // Bounds and Constraints vectors
        const real_t* lb, const real_t* ub, 
        const real_t* lbA, const real_t* ubA,
        // Solver Options (Inputs)
        const qpOASES::Options* userOptions,
        // Limits & Budget (Inputs/Outputs)
        int_t* nWSR, real_t* cputime,
        // Optimal Results arrays (Outputs)
        real_t* x_sol, real_t* y_sol, real_t* obj_val
    ) {
        // 1. instantiate the matrices using appropriate structures

        qpOASES::SymSparseMat matH(nV, nV, H_row, H_ptr, H_val);
        matH.createDiagInfo(); // Optimize symmetric diagonal tracking

        // qpOASES::SparseMatrix matH(nV, nV, H_row, H_ptr, H_val);

        qpOASES::SparseMatrix matA(nC, nV, A_row, A_ptr, A_val);

        // 2. initialise the solver object instance
        qpOASES::QProblem solver(nV, nC);

        // 3. feed options layout seamlessly if provided by Fortran
        if (userOptions != nullptr) {
            solver.setOptions(*userOptions);
        }

        // 4. run the sparse optimization problem solver
        int_t status = (int_t)solver.init(&matH, g, &matA, lb, ub, lbA, ubA, 
                                          *nWSR, cputime);

        // 5. gather optimal solution metrics if successful
        if (status == 0) {
            if (x_sol != nullptr) solver.getPrimalSolution(x_sol);
            if (y_sol != nullptr) solver.getDualSolution(y_sol);
            if (obj_val != nullptr) *obj_val = (real_t)solver.getObjVal();
        }

        // C++ structures automatically clean themselves up upon falling 
        // out of block scope
        return status;
    }
}
