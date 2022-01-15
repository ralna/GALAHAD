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
   int f_arrays;           /* Use 1-based indexing if true(!=0) else 0-based */
   int aggressive;         /* number of coarsening steps per coarse level */
   int c_fail;             /* conditions for coarsening failure */
   int max_levels;         /* size of mi20_data object */
   int max_points;         /* max number of points allowed on a coarse level */
   real_wp_ reduction; /* definition of stagnation */
   int st_method;          /* method to find strong transpose connections */
   real_wp_ st_parameter;  /* defines 'strong' connections */
   int testing;            /* test for validity? */
   real_wp_ trunc_parameter; /* interpolation truncation parameter */
   int coarse_solver;      /* coarse solver to use */
   int coarse_solver_its;  /* number of coarse solver itr (itr methods only) */
   real_wp_ damping; /* damping factor for Jacobi smoother */
   real_wp_ err_tol; /* error tolerance for preconditioner */
   int levels;             /* number of coarse levels used */
   int pre_smoothing;      /* number of pre-smoothing iterations */
   int smoother;           /* smoother type */
   int post_smoothing;     /* number of post-smoothing iterations */
   int v_iterations;       /* number of AMG iterations for preconditoner */
   int print_level;        /* levels of messages output */
   int print;              /* Fortran output stream for general messages */
   int error;              /* Fortran output stream for error messages */
   int one_pass_coarsen;   /* use one pass coarsening */
};

/* Derived type to hold control parameters for hsl_mi20_solve */
struct mi20_solve_control {
  real_wp_ abs_tol; /* absolute convergence tolerance */
  real_wp_ breakdown_tol; /* tolerance to determine breakdown in Krylov solver */
  int gmres_restart; /* number of iterations before restart*/
  _Bool init_guess; /* initial guess supplied? */
  int krylov_solver; /* choice of krylov solver to use */
  int max_its; /* max no of iterations allowed */
  int preconditioner_side; /* left (<0) or right (>=0) preconditioning? */
  real_wp_ rel_tol; /* relative tolerance? */
};

/* Communucates errors and information to the user. */
struct mi20_info {
   int flag;               /* error/warning information */
   int clevels;            /* number of levels actually generated */
   int cpoints;            /* number of points on the coarsest level */
   int cnnz;               /* number of non-zeros in coarsest matrix */
   int stat;               /* Fortran stat parameter */
   int getrf_info;         /* getrf return code */
  int iterations;          /* number of iterations */
  real_wp_ residual;      /* norm of the residual */
};
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
