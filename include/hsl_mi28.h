/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-22 AT 08:40 GMT
 * COPYRIGHT (c) 2023 Science and Technology Facilities Council (STFC)
 * Original date 27 March 2023
 * All rights reserved
 *
 * Written by: Niall Bootland
 *
 * THIS FILE ONLY may be redistributed under the modified BSD licence below.
 * All other files distributed as part of the HSL_MI28 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MI28 for full terms
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

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

#ifndef HSL_MI28D_H /* start include guard */
#define HSL_MI28D_H

/* precision */
#include "hsl_precision.h"

#ifndef mi28_control
#ifdef REAL_32
#ifdef INTEGER_64
#define mi28_control mi28_control_s_64
#define mi28_info mi28_info_s_64
#define mi28_default_control mi28_default_control_s_64
#define mi28_factorize mi28_factorize_s_64
#define mi28_precondition mi28_precondition_s_64
#define mi28_solve mi28_solve_s_64
#define mi28_finalise mi28_finalise_s_64
#else
#define mi28_control mi28_control_s
#define mi28_info mi28_info_s
#define mi28_default_control mi28_default_control_s
#define mi28_factorize mi28_factorize_s
#define mi28_precondition mi28_precondition_s
#define mi28_solve mi28_solve_s
#define mi28_finalise mi28_finalise_s
#endif
#else
#ifdef INTEGER_64
#define mi28_control mi28_control_d_64
#define mi28_info mi28_info_d_64
#define mi28_default_control mi28_default_control_d_64
#define mi28_factorize mi28_factorize_d_64
#define mi28_precondition mi28_precondition_d_64
#define mi28_solve mi28_solve_d_64
#define mi28_finalise mi28_finalise_d_64
#else
#define mi28_control mi28_control_d
#define mi28_info mi28_info_d
#define mi28_default_control mi28_default_control_d
#define mi28_factorize mi28_factorize_d
#define mi28_precondition mi28_precondition_d
#define mi28_solve mi28_solve_d
#define mi28_finalise mi28_finalise_d
#endif
#endif
#endif

/* Derived type to hold control parameters for hsl_mi28 */
struct mi28_control {
   ipc_ f_arrays;           /* use 1-based indexing if true(!=0) else 0-based */
   rpc_ alpha;   /* initial shift */
   bool check;             /* if set to true, user's data is checked.
        ! Otherwise, no checking and may fail in unexpected way if
        ! there are duplicates/out-of-range entries. */
   ipc_ iorder;             /* controls ordering of A. Options:
!       ! <=0  no ordering
!       !   1  RCM
!       !   2  AMD
!       !   3  user-supplied ordering
!       !   4  ascending degree
!       !   5  Metis
!       ! >=6  Sloan (MC61) */
   ipc_ iscale;             /* controls whether scaling is used.
        ! iscale = 1 is Lin and More scaling (l2 scaling)
        ! iscale = 2 is mc77 scaling
        ! iscale = 3 is mc64 scaling
        ! iscale = 4 is diagonal scaling
        ! iscale = 5 user-supplied scaling
        ! iscale <= 0, no scaling
        ! iscale >= 6, Lin and More */
   rpc_ lowalpha; /* Shift after first breakdown is
        ! max(shift_factor*alpha,lowalpha) */
   ipc_ maxshift;           /* During search for shift, we decrease
        ! the lower bound max(alpha,lowalpha) on the shift by
        ! shift_factor2 at most maxshift times (so this limits the
        ! number of refactorizations that are performed ... idea is
        ! reducing alpha as much as possible will give better preconditioner
        ! but reducing too far will lead to breakdown and then a refactorization
        ! is required (expensive so limit number of reductions)
        ! Note: Lin and More set this to 3. */
   bool rrt;               /* controls whether entries of RR^T that cause no
        ! additional fill are allowed. They are allowed if
        ! rrt = true and not otherwise. */
   rpc_ shift_factor;  /* if the current shift is found
        ! to be too small, it is increased by at least a factor of shift_factor.
        ! Values <= 1.0 are treated as default. */
   rpc_ shift_factor2; /* if factorization is successful
        ! with current (non zero) shift, the shift
        ! is reduced by a factor of shift_factor2.
        ! Values <= 1.0 are treated as default. */
   rpc_ small;   /* small value */
   rpc_ tau1;    /* used to select "small" entries that
        ! are dropped from L (but may be included in R).  */
   rpc_ tau2;    /* used to select "tiny" entries that are
        ! dropped from R.  Require
        ! tau2 < tau1 (otherwise, tau2 = 0.0 is used locally). */
   ipc_ unit_error;         /* unit number for error messages.
        ! Printing is suppressed if unit_error  <  0. */
   ipc_ unit_warning;       /* unit number for warning messages.
        ! Printing is suppressed if unit_warning  <  0. */
};

/* Communicates errors and information to the user. */
struct mi28_info {
  ipc_ band_after;     /* semibandwidth after MC61 */
  ipc_ band_before;    /* semibandwidth before MC61 */
  ipc_ dup;            /* number of duplicated entries found in row */
  ipc_ flag;           /* error flag */
  ipc_ flag61;         /* error flag from mc61 */
  ipc_ flag64;         /* error flag from hsl_mc64 */
  ipc_ flag68;         /* error flag from hsl_mc68 */
  ipc_ flag77;         /* error flag from mc77 */
  ipc_ nrestart;       /* number of restarts (after reducing the shift) */
  ipc_ nshift;         /* number of non-zero shifts used */
  ipc_ oor;            /* number of out-of-range entries found in row */
  rpc_ profile_before; /* semibandwidth before MC61 */
  rpc_ profile_after;  /* semibandwidth after MC61 */
  hsl_longc_ size_r;   /* size of arrays jr and ar that are used for r */
  ipc_ stat;           /* Fortran stat parameter */
  rpc_ alpha;          /* on successful exit, holds shift used */
};

/* Set default values of control */
void mi28_default_control(struct mi28_control *control);
/* Perform the factorize operation */
void mi28_factorize(const ipc_ n, ipc_ ptr[], ipc_ row[],
      rpc_ val[], const ipc_ lsize, const ipc_ rsize, void **keep,
      const struct mi28_control *control, struct mi28_info *info,
      const rpc_ scale[], const ipc_ perm[]);
/* Perform the preconditioning operation */
void mi28_precondition(const ipc_ n, void **keep, const rpc_ z[],
      rpc_ y[], struct mi28_info *info);
/* Perform the solve operation */
void mi28_solve(const bool trans, const ipc_ n, void **keep,
      const rpc_ z[], rpc_ y[], struct mi28_info *info);
/* Free memory */
void mi28_finalise(void **keep, struct mi28_info *info);

#endif /* end include guard */

#ifdef __cplusplus
} /* end extern "C" */
#endif
