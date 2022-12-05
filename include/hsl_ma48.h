//* \file hsl_ma48.h */

/*
 * COPYRIGHT (c) 2012 Science and Technology Facilities Council (STFC)
 * Original date 4 January 2012
 * All rights reserved
 *
 * Written by: Jonathan Hogg
 * Modified by Nick Gould for GALAHAD use, 2022-01-15
 *
 * THIS FILE ONLY may be redistributed under the below modified BSD licence.
 * All other files distributed as part of the HSL_MA48 package
 * require a licence to be obtained from STFC and may NOT be redistributed
 * without permission. Please refer to your licence for HSL_MA48 for full terms
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
#ifndef HSL_MA48_H
#define HSL_MA48_H

// precision
#include "galahad_precision.h"

struct ma48_control {
   int f_arrays; /* If eval to true, use 1-based indexing, else 0-based */
   real_wp_ multiplier; /* If arrays are too small, increase by factor */
   real_wp_ u; /* Pivot threshold */
   real_wp_ switch_; /* Density for switch to full code */
   real_wp_ drop; /* Drop tolerance */
   real_wp_ tolerance; /* Anything less than this is considered zero */
   real_wp_ cgce; /* Ratio for required reduction using IR */
   int lp; /* Fortran unit for error messages */
   int wp; /* Fortran unit for warning messages */
   int mp; /* Fortran unit for monitor output */
   int ldiag; /* Controls level of diagnostic output */
   int btf; /* Minimum block size for BTF ... >=N to avoid */
   int struct_; /* Abort if eval to true and structurally singular */
   int maxit; /* Maximum number of iterations */
   int factor_blocking; /* Level 3 blocking in factorize */
   int solve_blas; /* Switch for using Level 1 or 2 BLAS in solve. */
   int pivoting; /* Number of columns searched in pivoting. Markowitz=0 */
   int diagonal_pivoting; /* Use diagonal pivoting if eval to true */
   int fill_in; /* Initially fill_in * ne space allocated for factors */
   int switch_mode; /* If eval to true, switch to slow if fast mode unstable */
};


struct ma48_ainfo {
   real_wp_ ops; /* Number of operations in elimination */
   int flag; /* Return code */
   int more; /* More information on failure */
   long lena_analyse; /* Size for analysis (main arrays) */
   long lenj_analyse; /* Size for analysis (integer aux array) */
   long lena_factorize; /* Size for factorize (real array) */
   long leni_factorize; /* Size for factorize (integer array) */
   int ncmpa; /* Number of compresses in analyse */
   int rank; /* Estimated rank */
   long drop; /* Number of entries dropped */
   int struc_rank; /* Structural rank of matrix */
   long oor; /* Number of indices out-of-range */
   long dup; /* Number of duplicates */
   int stat; /* Fortran STAT value after allocate failure */
   int lblock; /* Size largest non-triangular block */
   int sblock; /* Sum of orders of non-triangular blocks */
   long tblock; /* Total entries in all non-triangular blocks */
};


struct ma48_finfo {
   real_wp_ ops; /* Number of operations in elimination */
   int flag; /* Return code */
   int more; /* More information on failure */
   long size_factor; /* Number of words to hold factors */
   long lena_factorize; /* Size for factorize (real array) */
   long leni_factorize; /* Size for factorize (integer array) */
   long drop; /* Number of entries dropped */
   int rank; /* Estimated rank */
   int stat; /* Fortran STAT value after allocate failure */
};

struct ma48_sinfo {
   int flag; /* Return code */
   int more; /* More information on failure */
   int stat; /* Fortran STAT value after allocate failure */
};

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
