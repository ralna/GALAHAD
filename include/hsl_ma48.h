/*
 * THIS VERSION: HSL Subset 1.0 - 2024-06-22 AT 08:30 GMT
 * COPYRIGHT (c) 2012 Science and Technology Facilities Council (STFC)
 * Original date 4 January 2012
 * All rights reserved
 *
 * Written by: Jonathan Hogg
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
#ifndef HSL_MA48D_H
#define HSL_MA48D_H

/* precision */
#include "hsl_precision.h"

#ifndef ma48_control
#ifdef REAL_32
#ifdef INTEGER_64
#define ma48_control ma48_control_s_64
#define ma48_ainfo ma48_ainfo_s_64
#define ma48_finfo ma48_finfo_s_64
#define ma48_sinfo ma48_sinfo_s_64
#define ma48_initialize ma48_initialize_s_64
#define ma48_default_control ma48_default_control_s_64
#define ma48_analyse ma48_analyse_s_64
#define ma48_get_perm ma48_get_perm_s_64
#define ma48_factorize ma48_factorize_s_64
#define ma48_solve ma48_solve_s_64
#define ma48_finalize ma48_finalize_s_64
#define ma48_special_rows_and_cols ma48_special_rows_and_cols_s_64
#define ma48_determinant ma48_determinant_s_64
#else
#define ma48_control ma48_control_s
#define ma48_ainfo ma48_ainfo_s
#define ma48_finfo ma48_finfo_s
#define ma48_sinfo ma48_sinfo_s
#define ma48_initialize ma48_initialize_s
#define ma48_default_control ma48_default_control_s
#define ma48_analyse ma48_analyse_s
#define ma48_get_perm ma48_get_perm_s
#define ma48_factorize ma48_factorize_s
#define ma48_solve ma48_solve_s
#define ma48_finalize ma48_finalize_s
#define ma48_special_rows_and_cols ma48_special_rows_and_cols_s
#define ma48_determinant ma48_determinant_s
#endif
#else
#ifdef INTEGER_64
#define ma48_control ma48_control_d_64
#define ma48_ainfo ma48_ainfo_d_64
#define ma48_finfo ma48_finfo_d_64
#define ma48_sinfo ma48_sinfo_d_64
#define ma48_initialize ma48_initialize_d_64
#define ma48_default_control ma48_default_control_d_64
#define ma48_analyse ma48_analyse_d_64
#define ma48_get_perm ma48_get_perm_d_64
#define ma48_factorize ma48_factorize_d_64
#define ma48_solve ma48_solve_d_64
#define ma48_finalize ma48_finalize_d_64
#define ma48_special_rows_and_cols ma48_special_rows_and_cols_d_64
#define ma48_determinant ma48_determinant_d_64
#else
#define ma48_control ma48_control_d
#define ma48_ainfo ma48_ainfo_d
#define ma48_finfo ma48_finfo_d
#define ma48_sinfo ma48_sinfo_d
#define ma48_initialize ma48_initialize_d
#define ma48_default_control ma48_default_control_d
#define ma48_analyse ma48_analyse_d
#define ma48_get_perm ma48_get_perm_d
#define ma48_factorize ma48_factorize_d
#define ma48_solve ma48_solve_d
#define ma48_finalize ma48_finalize_d
#define ma48_special_rows_and_cols ma48_special_rows_and_cols_d
#define ma48_determinant ma48_determinant_d
#endif
#endif
#endif

struct ma48_control {
   ipc_ f_arrays; /* If eval to true, use 1-based indexing, else 0-based */
   rpc_ multiplier; /* If arrays are too small, increase by factor */
   rpc_ u; /* Pivot threshold */
   rpc_ switch_; /* Density for switch to full code */
   rpc_ drop; /* Drop tolerance */
   rpc_ tolerance; /* Anything less than this is considered zero */
   rpc_ cgce; /* Ratio for required reduction using IR */
   ipc_ lp; /* Fortran unit for error messages */
   ipc_ wp; /* Fortran unit for warning messages */
   ipc_ mp; /* Fortran unit for monitor output */
   ipc_ ldiag; /* Controls level of diagnostic output */
   ipc_ btf; /* Minimum block size for BTF ... >=N to avoid */
   ipc_ struct_; /* Abort if eval to true and structurally singular */
   ipc_ maxit; /* Maximum number of iterations */
   ipc_ factor_blocking; /* Level 3 blocking in factorize */
   ipc_ solve_blas; /* Switch for using Level 1 or 2 BLAS in solve. */
   ipc_ pivoting; /* Number of columns searched in pivoting. Markowitz=0 */
   ipc_ diagonal_pivoting; /* Use diagonal pivoting if eval to true */
   ipc_ fill_in; /* Initially fill_in * ne space allocated for factors */
   ipc_ switch_mode; /* If eval to true, switch to slow if fast mode unstable */
};


struct ma48_ainfo {
   rpc_ ops; /* Number of operations in elimination */
   ipc_ flag; /* Return code */
   ipc_ more; /* More information on failure */
   hsl_longc_ lena_analyse; /* Size for analysis (main arrays) */
   hsl_longc_ lenj_analyse; /* Size for analysis (integer aux array) */
   hsl_longc_ lena_factorize; /* Size for factorize (real array) */
   hsl_longc_ leni_factorize; /* Size for factorize (integer array) */
   ipc_ ncmpa; /* Number of compresses in analyse */
   ipc_ rank; /* Estimated rank */
   hsl_longc_ drop; /* Number of entries dropped */
   ipc_ struc_rank; /* Structural rank of matrix */
   hsl_longc_ oor; /* Number of indices out-of-range */
   hsl_longc_ dup; /* Number of duplicates */
   ipc_ stat; /* Fortran STAT value after allocate failure */
   ipc_ lblock; /* Size largest non-triangular block */
   ipc_ sblock; /* Sum of orders of non-triangular blocks */
   hsl_longc_ tblock; /* Total entries in all non-triangular blocks */
};


struct ma48_finfo {
   rpc_ ops; /* Number of operations in elimination */
   ipc_ flag; /* Return code */
   ipc_ more; /* More information on failure */
   hsl_longc_ size_factor; /* Number of words to hold factors */
   hsl_longc_ lena_factorize; /* Size for factorize (real array) */
   hsl_longc_ leni_factorize; /* Size for factorize (integer array) */
   hsl_longc_ drop; /* Number of entries dropped */
   ipc_ rank; /* Estimated rank */
   ipc_ stat; /* Fortran STAT value after allocate failure */
};


struct ma48_sinfo {
   ipc_ flag; /* Return code */
   ipc_ more; /* More information on failure */
   ipc_ stat; /* Fortran STAT value after allocate failure */
};

void ma48_default_control(struct ma48_control *control);

void ma48_initialize(void **factors);

void ma48_analyse(ipc_ m, ipc_ n, hsl_longc_ ne, const ipc_ row[],
      const ipc_ col[], const rpc_ val[], void *factors,
      const struct ma48_control *control, struct ma48_ainfo *ainfo,
      struct ma48_finfo *finfo, const ipc_ perm[], const ipc_ endcol[]);

void ma48_get_perm(ipc_ m, ipc_ n, const void *factors, ipc_ perm[], 
                     const struct ma48_control *control);

void ma48_factorize(ipc_ m, ipc_ n, hsl_longc_ ne, const ipc_ row[],
      const ipc_ col[], const rpc_ val[], void *factors,
      const struct ma48_control *control, struct ma48_finfo *finfo,
      ipc_ fast, ipc_ partial);

void ma48_solve(ipc_ m, ipc_ n, hsl_longc_ ne, const ipc_ row[],
      const ipc_ col[], const rpc_ val[], const void *factors,
      const rpc_ rhs[], rpc_ x[],
      const struct ma48_control *control, struct ma48_sinfo *sinfo,
      ipc_ trans, rpc_ resid[], rpc_ *error);

ipc_ ma48_finalize(void **factors, const struct ma48_control *control);

ipc_ ma48_determinant(const void *factors, ipc_ *sgndet,
      rpc_ *logdet, const struct ma48_control *control);

ipc_ ma48_special_rows_and_cols(const void *factors, ipc_ *rank,
      ipc_ rows[], ipc_ cols[], const struct ma48_control *control);

#endif
