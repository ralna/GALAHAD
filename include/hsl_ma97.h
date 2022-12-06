//* \file hsl_ma97.h */

/*
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 20 September 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 * Version 2.6.0
 * Modified by Nick Gould for GALAHAD use, 2022-01-15
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA97 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA97 for full terms
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
#endif

// include guard
#ifndef HSL_MA97_H
#define HSL_MA97_H

// precision
#include "galahad_precision.h"

struct ma97_control {
    int f_arrays;             /* Use C or Fortran numbering */
    int action;               /* Continue on singularity if !=0 (true),
                                 otherwise abort */
    int nemin;                /* Supernode amalgamation if parent and child
                                 have fewer than nemin eliminations */
    real_wp_ multiplier;      /* Amount of extra memory to allow for delays */
    int ordering;             /* Control scaling algorithm used:
                                 0 - user supplied order (order absent=identity)
                                 1 - AMD
                                 2 - MD (as in MA27)
                                 3 - METIS nested dissection
                                 4 - MA47
                                 5 - Automatic choice between 1 and 3 */
    int print_level;          /* <0 for no printing, 0 for basic, >1 for most */
    int scaling;              /* 0 user/none, 1 mc64, 2 mc77 */
    real_wp_ small;           /* Minimum value to count as non-zero */
    real_wp_ u;               /* Pivoting parameter */
    int unit_diagnostics;     /* Fortran unit for diagnostics (<0 disables) */
    int unit_error;           /* Fortran unit for error msgs (<0 disables) */
    int unit_warning;         /* Fortran unit for warning msgs (<0 disables) */
    long factor_min;          /* Min number of flops for parallel execution */
    int solve_blas3;          /* Use BLAS3 in solve in true, else BLAS2 */
    long solve_min;           /* Min number of entries for parallel exection */
    int solve_mf;             /* If true use m/f solve, else use s/n */
    real_wp_ consist_tol;     /* Consistent equation tolerance */

    /* Reserve space for future interface changes */
    int ispare[5]; real_wp_ rspare[10];
};

struct ma97_info {
    int flag;                 /* <0 on error */
    int flag68;
    int flag77;
    int matrix_dup;           /* number duplicate entries in A */
    int matrix_rank;          /* matrix rank */
    int matrix_outrange;      /* number of out of range entries in A */
    int matrix_missing_diag;  /* number of zero diagonal entries in A */
    int maxdepth;             /* height of assembly tree */
    int maxfront;             /* maximum dimension of frontal matrix */
    int num_delay;            /* number of times a pivot was delayed */
    long num_factor;          /* number of entries in L */
    long num_flops;           /* number of floating point operations */
    int num_neg;              /* number of negative pivots */
    int num_sup;              /* number of supernodes in assembly tree */
    int num_two;              /* number of 2x2 pivots */
    int ordering;             /* ordering used (as per control.ordering) */
    int stat;                 /* error code from failed memory allocation */

    /* Reserve space for future interface changes */
    int ispare[5]; real_wp_ rspare[10];
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
