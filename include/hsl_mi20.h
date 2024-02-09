//* \file hsl_mi20.h */

/*
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 28th November 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 * Modified by Nick Gould for GALAHAD use, 2022-01-15
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MI20 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MI20 for full terms
 * and conditions. STFC may be contacted via hsl(at)stfc.ac.uk.
 *
 * Modified BSD licence (this header file only):
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of STFC nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ""AS IS""
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL STFC BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef HSL_MI20_H
#define HSL_MI20_H

// precision
#include "galahad_precision.h"

/* Derived type to hold control parameters for hsl_mi20 */
struct mi20_control {
   ipc_ f_arrays;           /* Use 1-based indexing if true(!=0) else 0-based */
   ipc_ aggressive;         /* number of coarsening steps per coarse level */
   ipc_ c_fail;             /* conditions for coarsening failure */
   ipc_ max_levels;         /* size of mi20_data object */
   ipc_ max_points;         /* max number of points allowed on a coarse level */
   real_wp_ reduction; /* definition of stagnation */
   ipc_ st_method;          /* method to find strong transpose connections */
   real_wp_ st_parameter;  /* defines 'strong' connections */
   ipc_ testing;            /* test for validity? */
   real_wp_ trunc_parameter; /* interpolation truncation parameter */
   ipc_ coarse_solver;      /* coarse solver to use */
   ipc_ coarse_solver_its;  /* number of coarse solver itr (itr methods only) */
   real_wp_ damping; /* damping factor for Jacobi smoother */
   real_wp_ err_tol; /* error tolerance for preconditioner */
   ipc_ levels;             /* number of coarse levels used */
   ipc_ pre_smoothing;      /* number of pre-smoothing iterations */
   ipc_ smoother;           /* smoother type */
   ipc_ post_smoothing;     /* number of post-smoothing iterations */
   ipc_ v_iterations;       /* number of AMG iterations for preconditoner */
   ipc_ print_level;        /* levels of messages output */
   ipc_ print;              /* Fortran output stream for general messages */
   ipc_ error;              /* Fortran output stream for error messages */
   ipc_ one_pass_coarsen;   /* use one pass coarsening */
};

/* Derived type to hold control parameters for hsl_mi20_solve */
struct mi20_solve_control {
  real_wp_ abs_tol; /* absolute convergence tolerance */
  real_wp_ breakdown_tol; /* tolerance to determine breakdown in Krylov solver */
  ipc_ gmres_restart; /* number of iterations before restart*/
  _Bool init_guess; /* initial guess supplied? */
  ipc_ krylov_solver; /* choice of krylov solver to use */
  ipc_ max_its; /* max no of iterations allowed */
  ipc_ preconditioner_side; /* left (<0) or right (>=0) preconditioning? */
  real_wp_ rel_tol; /* relative tolerance? */
};

/* Communucates errors and information to the user. */
struct mi20_info {
   ipc_ flag;               /* error/warning information */
   ipc_ clevels;            /* number of levels actually generated */
   ipc_ cpoints;            /* number of points on the coarsest level */
   ipc_ cnnz;               /* number of non-zeros in coarsest matrix */
   ipc_ stat;               /* Fortran stat parameter */
   ipc_ getrf_info;         /* getrf return code */
  ipc_ iterations;          /* number of iterations */
  real_wp_ residual;      /* norm of the residual */
};
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
