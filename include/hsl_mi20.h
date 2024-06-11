/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-11 AT 09:50 GMT
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 28th November 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
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

#ifndef HSL_MI20D_H
#define HSL_MI20D_H

/* precision */
#include "hsl_precision.h"

#ifndef mi20_control
#ifdef REAL_32
#ifdef INTEGER_64
#define mi20_default_control mi20_default_control_s_64
#define mi20_control mi20_control_s_64
#define mi20_default_solve_control mi20_default_solve_control_s_64
#define mi20_solve_control mi20_solve_control_s_64
#define mi20_info mi20_info_s_64
#define mi20_setup mi20_setup_s_64
#define mi20_setup_csr mi20_setup_csr_s_64
#define mi20_setup_csc mi20_setup_csc_s_64
#define mi20_setup_coord mi20_setup_coord_s_64
#define mi20_finalize mi20_finalize_s_64
#define mi20_precondition mi20_precondition_s_64
#define mi20_solve mi20_solve_s_64
#else
#define mi20_default_control mi20_default_control_s
#define mi20_control mi20_control_s
#define mi20_default_solve_control mi20_default_solve_control_s
#define mi20_solve_control mi20_solve_control_s
#define mi20_info mi20_info_s
#define mi20_setup mi20_setup_s
#define mi20_setup_csr mi20_setup_csr_s
#define mi20_setup_csc mi20_setup_csc_s
#define mi20_setup_coord mi20_setup_coord_s
#define mi20_finalize mi20_finalize_s
#define mi20_precondition mi20_precondition_s
#define mi20_solve mi20_solve_s
#endif
#else
#ifdef INTEGER_64
#define mi20_default_control mi20_default_control_d_64
#define mi20_control mi20_control_d_64
#define mi20_default_solve_control mi20_default_solve_control_d_64
#define mi20_solve_control mi20_solve_control_d_64
#define mi20_info mi20_info_d_64
#define mi20_setup mi20_setup_d_64
#define mi20_setup_csr mi20_setup_csr_d_64
#define mi20_setup_csc mi20_setup_csc_d_64
#define mi20_setup_coord mi20_setup_coord_d_64
#define mi20_finalize mi20_finalize_d_64
#define mi20_precondition mi20_precondition_d_64
#define mi20_solve mi20_solve_d_64
#else
#define mi20_default_control mi20_default_control_d
#define mi20_control mi20_control_d
#define mi20_default_solve_control mi20_default_solve_control_d
#define mi20_solve_control mi20_solve_control_d
#define mi20_info mi20_info_d
#define mi20_setup mi20_setup_d
#define mi20_setup_csr mi20_setup_csr_d
#define mi20_setup_csc mi20_setup_csc_d
#define mi20_setup_coord mi20_setup_coord_d
#define mi20_finalize mi20_finalize_d
#define mi20_precondition mi20_precondition_d
#define mi20_solve mi20_solve_d
#endif
#endif
#endif

/* Derived type to hold control parameters for hsl_mi20 */
struct mi20_control {
   ipc_ f_arrays;           /* Use 1-based indexing if true(!=0) else 0-based */
   ipc_ aggressive;         /* number of coarsening steps per coarse level */
   ipc_ c_fail;             /* conditions for coarsening failure */
   ipc_ max_levels;         /* size of mi20_data object */
   ipc_ max_points;         /* max number of points allowed on a coarse level */
   rpc_ reduction;          /* definition of stagnation */
   ipc_ st_method;          /* method to find strong transpose connections */
   rpc_ st_parameter;       /* defines 'strong' connections */
   ipc_ testing;            /* test for validity? */
   rpc_ trunc_parameter;    /* interpolation truncation parameter */
   ipc_ coarse_solver;      /* coarse solver to use */
   ipc_ coarse_solver_its;  /* number of coarse solver itr (itr methods only) */
   rpc_ damping;            /* damping factor for Jacobi smoother */
   rpc_ err_tol;            /* error tolerance for preconditioner */
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
  rpc_ abs_tol; /* absolute convergence tolerance */
  rpc_ breakdown_tol; /* tolerance to determine breakdown in Krylov solver */
  ipc_ gmres_restart; /* number of iterations before restart*/
  _Bool init_guess; /* initial guess supplied? */
  ipc_ krylov_solver; /* choice of krylov solver to use */
  ipc_ max_its; /* max no of iterations allowed */
  ipc_ preconditioner_side; /* left (<0) or right (>=0) preconditioning? */
  rpc_ rel_tol; /* relative tolerance? */
};

/* Communucates errors and information to the user. */
struct mi20_info {
   ipc_ flag;               /* error/warning information */
   ipc_ clevels;            /* number of levels actually generated */
   ipc_ cpoints;            /* number of points on the coarsest level */
   ipc_ cnnz;               /* number of non-zeros in coarsest matrix */
   ipc_ stat;               /* Fortran stat parameter */
   ipc_ getrf_info;         /* getrf return code */
   ipc_ iterations;         /* number of iterations */
   rpc_ residual; /* norm of the residual */
};

/* Set default values of control */
void mi20_default_control(struct mi20_control *control);
/* Set default values of solve_control */
void mi20_default_solve_control(struct mi20_solve_control *solve_control);
/* Generate coarse level data, allocate keep and coarse_data */
void mi20_setup(int n, const ipc_ ptr[], const ipc_ col[],
      const rpc_ val[], void **keep,
      const struct mi20_control *control, struct mi20_info *info);
/* setup for csr matrices */
void mi20_setup_csr(const ipc_ n,
            const ipc_ ptr[], const ipc_ col[], 
            const rpc_ val[], 
            void **keep, const struct mi20_control *control, 
            struct mi20_info *info);
/* setup for csc matrices */
void mi20_setup_csc(const ipc_ n,
            const ipc_ ptr[], const ipc_ row[], 
            const rpc_ val[], 
            void **keep, const struct mi20_control *control, 
            struct mi20_info *info);
/* setup for coord matrices */
void mi20_setup_coord(const ipc_ n, const ipc_ ne, 
         const ipc_ row[], const ipc_ col[], 
         const rpc_ val[], 
         void **keep, const struct mi20_control *control, 
         struct mi20_info *info);
/* Free memory keep and coarseata */
void mi20_finalize(void **keep, const struct mi20_control *control,
      struct mi20_info *info);
/* Perform the preconditioning operation */
void mi20_precondition(const rpc_ rhs[], rpc_ solution[],
      void **keep, const struct mi20_control *control,
      struct mi20_info *info);
void mi20_solve(const rpc_ rhs[], rpc_ solution[],
      void **keep, const struct mi20_control *control,
      const struct mi20_solve_control *solve_control,
      struct mi20_info *info);

#endif
