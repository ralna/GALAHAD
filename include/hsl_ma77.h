//* \file hsl_ma77.h */

/*
 * COPYRIGHT (c) 2011 Science and Technology Facilities Council (STFC)
 * Original date 18 May 2011
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 * Modified by Nick Gould for GALAHAD use, 2022-01-15
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA77 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA77 for full terms
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
#ifndef HSL_MA77_H
#define HSL_MA77_H

// precision
#include "galahad_precision.h"

/* Data type for user controls */
struct ma77_control {
   /* Note: 0 is false, non-zero is true */

   /* C/Fortran interface related controls */
   int f_arrays; /* Treat arrays as 1-based (Fortran) if true or 0-based (C) if
                    false. */

   /* Printing controls */
   int print_level;
   int unit_diagnostics;   /* unit for diagnostic messages
                              Printing is suppressed if unit_diagnostics < 0. */
   int unit_error;         /* unit for error messages
                              Printing is suppressed if unit_error  <  0.     */
   int unit_warning;       /* unit for warning messages
                              Printing is suppressed if unit_warning  <  0.   */

   /* Controls used by MA77_open */
   int bits;
   int buffer_lpage[2];
   int buffer_npage[2];
   long int file_size;
   long int maxstore;
   long int storage[3];

   /* Controls used by MA77_analyse */
   int nemin;  /* Node amalgamation parameter. A child node is merged with its
                  parent if they both involve fewer than nemin eliminations.*/

   /* Controls used by MA77_scale */
   int maxit;
   int infnorm;
   real_wp_ thresh;

   /* Controls used by MA77_factor with posdef true */
   int nb54;

   /* Controls used by MA77_factor with posdef false */
   int action;    /* Keep going even if matrix is singular if true, or abort
                     if false */
   real_wp_ multiplier;
   int nb64;
   int nbi;
   real_wp_ small;
   real_wp_ static_;
   long int storage_indef;
   real_wp_ u;       /* Pivot tolerance*/
   real_wp_ umin;    /* Minimum pivot tolerance*/

   /* Controls used by ma77_solve_fredholm */
   real_wp_ consist_tol;   /* Tolerance for consistent singular system */

   /* Pad data structure to allow for future growth */
   int ispare[5]; long int lspare[5]; real_wp_ rspare[5];
};

/***************************************************/

/* data type for returning information to user.*/
struct ma77_info {
   real_wp_ detlog;
   int detsign;
   int flag;
   int iostat;
   int matrix_dup;
   int matrix_rank;
   int matrix_outrange;
   int maxdepth;
   int maxfront;
   long int minstore;
   int ndelay;
   long int nfactor;
   long int nflops;
   int niter;
   int nsup;
   int num_neg;
   int num_nothresh;
   int num_perturbed;
   int ntwo;
   int stat;
   int index[4];
   long int nio_read[2];
   long int nio_write[2];
   long int nwd_read[2];
   long int nwd_write[2];
   int num_file[4];
   long int storage[4];
   int tree_nodes;
   int unit_restart;
   int unused;
   real_wp_ usmall;

   /* Pad data structure to allow for future growth */
   int ispare[5]; long int lspare[5]; real_wp_ rspare[5];
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
