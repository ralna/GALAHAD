/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-22 AT 08:35 GMT
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 25 Feburary 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA86 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA86 for full terms
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

#ifndef HSL_MA86D_H
#define HSL_MA86D_H

/* precision */
#include "hsl_precision.h"

#ifndef ma86_default_control
#ifdef REAL_32
#ifdef INTEGER_64
#define ma86_control ma86_control_s_64
#define ma86_info ma86_info_s_64
#define ma86_default_control ma86_default_control_s_64
#define ma86_analyse ma86_analyse_s_64
#define ma86_factor ma86_factor_s_64
#define ma86_factor_solve ma86_factor_solve_s_64
#define ma86_solve ma86_solve_s_64
#define ma86_finalise ma86_finalise_s_64
#else
#define ma86_control ma86_control_s
#define ma86_info ma86_info_s
#define ma86_default_control ma86_default_control_s
#define ma86_analyse ma86_analyse_s
#define ma86_factor ma86_factor_s
#define ma86_factor_solve ma86_factor_solve_s
#define ma86_solve ma86_solve_s
#define ma86_finalise ma86_finalise_s
#endif
#else
#ifdef INTEGER_64
#define ma86_control ma86_control_d_64
#define ma86_info ma86_info_d_64
#define ma86_default_control ma86_default_control_d_64
#define ma86_analyse ma86_analyse_d_64
#define ma86_factor ma86_factor_d_64
#define ma86_factor_solve ma86_factor_solve_d_64
#define ma86_solve ma86_solve_d_64
#define ma86_finalise ma86_finalise_d_64
#else
#define ma86_control ma86_control_d
#define ma86_info ma86_info_d
#define ma86_default_control ma86_default_control_d
#define ma86_analyse ma86_analyse_d
#define ma86_factor ma86_factor_d
#define ma86_factor_solve ma86_factor_solve_d
#define ma86_solve ma86_solve_d
#define ma86_finalise ma86_finalise_d
#endif
#endif
#endif

/* Data type for user controls */
struct ma86_control {
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

   /* Controls used by ma86_analyse */
   ipc_ nemin; /* Node amalgamation parameter. A child node is merged with its
                  parent if they both involve fewer than nemin eliminations.*/
   ipc_ nb;    /* Controls the size of the blocks used within each node (used to
                  set nb within node_type)*/

   /* Controls used by ma86_factor and ma86_factor_solve */
   ipc_ action; /* Keep going even if matrix is singular if true, or abort
                  if false */
   ipc_ nbi;    /* Inner block size for use with ma64*/
   ipc_ pool_size; /* Size of task pool arrays*/
   rpc_ small_; /* Pivots less than small are treated as zero*/
   rpc_ static_;/* Control static pivoting*/
   rpc_ u;      /* Pivot tolerance*/
   rpc_ umin;   /* Minimum pivot tolerance*/
   ipc_ scaling;            /* Scaling algorithm to use */
};

/***************************************************/

/* data type for returning information to user.*/
struct ma86_info {
   rpc_ detlog;           /* Holds logarithm of abs det A (or 0) */
   ipc_ detsign;          /* Holds sign of determinant (+/-1 or 0) */
   ipc_ flag;             /* Error return flag (0 on success) */
   ipc_ matrix_rank;      /* Rank of matrix */
   ipc_ maxdepth;         /* Maximum depth of the tree. */
   ipc_ num_delay;        /* Number of delayed pivots */
   hsl_longc_ num_factor; /* Number of entries in the factor. */
   hsl_longc_ num_flops;  /* Number of flops for factor. */
   ipc_ num_neg;          /* Number of negative pivots */
   ipc_ num_nodes;        /* Number of nodes */
   ipc_ num_nothresh;     /* Number of pivots not satisfying u */
   ipc_ num_perturbed;    /* Number of perturbed pivots */
   ipc_ num_two;          /* Number of 2x2 pivots */
   ipc_ pool_size;        /* Maximum size of task pool used */
   ipc_ stat;             /* STAT value on error return -1. */
   rpc_ usmall;           /* smallest threshold parameter used */
};

/* Initialise control with default values */
void ma86_default_control(struct ma86_control *control);
/* Analyse the sparsity pattern and prepare for factorization */
void ma86_analyse(const ipc_ n, const ipc_ ptr[], const ipc_ row[], ipc_ order[],
      void **keep, const struct ma86_control *control,
      struct ma86_info *info);
/* To factorize the matrix */
void ma86_factor(const ipc_ n, const ipc_ ptr[], const ipc_ row[],
      const rpc_ val[], const ipc_ order[], void **keep,
      const struct ma86_control *control, struct ma86_info *info,
      const rpc_ scale[]);
/* To factorize the matrix AND solve AX = B */
void ma86_factor_solve(const ipc_ n, const ipc_ ptr[], const ipc_ row[],
      const rpc_ val[], const ipc_ order[], void **keep,
      const struct ma86_control *control, struct ma86_info *info,
      const ipc_ nrhs, const ipc_ ldx, rpc_ x[],
      const rpc_ scale[]);
/* To solve AX = B using the computed factors */
void ma86_solve(const ipc_ job, const ipc_ nrhs, const ipc_ ldx,
      rpc_ *x, const ipc_ order[], void **keep,
      const struct ma86_control *control, struct ma86_info *info,
      const rpc_ scale[]);
/* To clean up memory in keep */
void ma86_finalise(void **keep, const struct ma86_control *control);

#endif
