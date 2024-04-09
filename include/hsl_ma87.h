/*
 * THIS VERSION: HSL Subset 1.0 - 2024-02-17 AT 15:40 GMT
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 25 Feburary 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA87 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA87 for full terms
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef HSL_MA87D_H
#define HSL_MA87D_H

/* precision */
#include "hsl_precision.h"

#ifndef ma87_default_control
#ifdef SINGLE
#ifdef INTEGER_64
#define ma87_control ma87_control_s_64
#define ma87_info ma87_info_s_64
#define ma87_default_control ma87_default_control_s_64
#define ma87_analyse ma87_analyse_s_64
#define ma87_factor ma87_factor_s_64
#define ma87_factor_solve ma87_factor_solve_s_64
#define ma87_solve ma87_solve_s_64
#define ma87_sparse_fwd_solve ma87_sparse_fwd_solve_s_64
#define ma87_finalise ma87_finalise_s_64
#else
#define ma87_control ma87_control_s
#define ma87_info ma87_info_s
#define ma87_default_control ma87_default_control_s
#define ma87_analyse ma87_analyse_s
#define ma87_factor ma87_factor_s
#define ma87_factor_solve ma87_factor_solve_s
#define ma87_solve ma87_solve_s
#define ma87_sparse_fwd_solve ma87_sparse_fwd_solve_s
#define ma87_finalise ma87_finalise_s
#endif
#else
#ifdef INTEGER_64
#define ma87_control ma87_control_d_64
#define ma87_info ma87_info_d_64
#define ma87_default_control ma87_default_control_d_64
#define ma87_analyse ma87_analyse_d_64
#define ma87_factor ma87_factor_d_64
#define ma87_factor_solve ma87_factor_solve_d_64
#define ma87_solve ma87_solve_d_64
#define ma87_sparse_fwd_solve ma87_sparse_fwd_solve_d_64
#define ma87_finalise ma87_finalise_d_64
#else
#define ma87_control ma87_control_d
#define ma87_info ma87_info_d
#define ma87_default_control ma87_default_control_d
#define ma87_analyse ma87_analyse_d
#define ma87_factor ma87_factor_d
#define ma87_factor_solve ma87_factor_solve_d
#define ma87_solve ma87_solve_d
#define ma87_sparse_fwd_solve ma87_sparse_fwd_solve_d
#define ma87_finalise ma87_finalise_d
#endif
#endif
#endif

/* Data type for user controls */
struct ma87_control {
   /* Note: 0 is false, non-zero is true */

   /* C/Fortran interface related controls */
   ipc_ f_arrays; /* Treat arrays as 1-based (Fortran) if true or 0-based (C) if
                    false. */

   /* Printing controls */
   ipc_ diagnostics_level; /* Controls diagnostic printing.*/
               /* Possible values are:
                   < 0: no printing.
                     0: error and warning messages only.
                     1: as 0 plus basic diagnostic printing.
                     2: as 1 plus some more detailed diagnostic messages.
                     3: as 2 plus all entries of user-supplied arrays.       */
   ipc_ unit_diagnostics;  /* unit for diagnostic messages
                              Printing is suppressed if unit_diagnostics < 0. */
   ipc_ unit_error;        /* unit for error messages
                              Printing is suppressed if unit_error  <  0.     */
   ipc_ unit_warning;      /* unit for warning messages
                              Printing is suppressed if unit_warning  <  0.   */

   /* Controls used by ma87_analyse */
   ipc_ nemin; /* Node amalgamation parameter. A child node is merged with its
                  parent if they both involve fewer than nemin eliminations.*/
   ipc_ nb;    /* Controls the size of the blocks used within each node (used to
                  set nb within node_type)*/

   /* Controls used by ma87_factor and ma87_factor_solve */
   ipc_ pool_size; /* Size of task pool arrays*/
   rpc_ diag_zero_minus; /* Semi-definite rank detection */
   rpc_ diag_zero_plus;  /* Semi-definite rank detection */

   char unused[40];
};

/***************************************************/

/* data type for returning information to user.*/
struct ma87_info {
   rpc_ detlog;         /* Holds logarithm of abs det A (or 0) */
   ipc_ flag;           /* Error return flag (0 on success) */
   ipc_ maxdepth;       /* Maximum depth of the tree. */
   int64_t num_factor;  /* Number of entries in the factor. */
   int64_t num_flops;   /* Number of flops for factor. */
   ipc_ num_nodes;      /* Number of nodes in factors */
   ipc_ pool_size;      /* Maximum size of task pool used */
   ipc_ stat;           /* STAT value on error return -1. */
   ipc_ num_zero;       /* Number of zero pivots. */

   char unused[40];
};

/* Initialise control with default values */
void ma87_default_control(struct ma87_control *control);
/* Analyse the sparsity pattern and prepare for factorization */
void ma87_analyse(const ipc_ n, const ipc_ ptr[], const ipc_ row[], 
      ipc_ order[], void **keep, const struct ma87_control *control,
      struct ma87_info *info);
/* To factorize the matrix */
void ma87_factor(const ipc_ n, const ipc_ ptr[], const ipc_ row[],
      const rpc_ val[], const ipc_ order[], void **keep,
      const struct ma87_control *control, struct ma87_info *info);
/* To factorize the matrix AND solve AX = B */
void ma87_factor_solve(const ipc_ n, const ipc_ ptr[], const ipc_ row[],
      const rpc_ val[], const ipc_ order[], void **keep,
      const struct ma87_control *control, struct ma87_info *info,
      const ipc_ nrhs, const ipc_ ldx, rpc_ x[]);
/* To solve AX = B using the computed factors */
void ma87_solve(const ipc_ job, const ipc_ nrhs, const ipc_ ldx,
      rpc_ *x, const ipc_ order[], void **keep,
      const struct ma87_control *control, struct ma87_info *info);
/* To solve Ax = b for sparse b using the computed factors */
void ma87_sparse_fwd_solve(const ipc_ nbi, ipc_ bindex[],
      const rpc_ b[], const ipc_ order[], const ipc_ invp[],
      ipc_ *nxi, ipc_ index[], rpc_ x[],  rpc_ *w, void **keep,
      const struct ma87_control *control, struct ma87_info *info);
/* To clean up memory in keep */
void ma87_finalise(void **keep, const struct ma87_control *control);

#endif
