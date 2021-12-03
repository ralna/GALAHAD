/*
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

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// include guard
#ifndef HSL_MA48D_H
#define HSL_MA48D_H

#ifndef ma48_control
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

typedef double ma48pkgtype_d_;

struct ma48_control_d {
   int f_arrays; /* If eval to true, use 1-based indexing, else 0-based */
   ma48pkgtype_d_ multiplier; /* If arrays are too small, increase by factor */
   ma48pkgtype_d_ u; /* Pivot threshold */
   ma48pkgtype_d_ switch_; /* Density for switch to full code */
   ma48pkgtype_d_ drop; /* Drop tolerance */
   ma48pkgtype_d_ tolerance; /* Anything less than this is considered zero */
   ma48pkgtype_d_ cgce; /* Ratio for required reduction using IR */
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


struct ma48_ainfo_d {
   ma48pkgtype_d_ ops; /* Number of operations in elimination */
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


struct ma48_finfo_d {
   ma48pkgtype_d_ ops; /* Number of operations in elimination */
   int flag; /* Return code */
   int more; /* More information on failure */
   long size_factor; /* Number of words to hold factors */
   long lena_factorize; /* Size for factorize (real array) */
   long leni_factorize; /* Size for factorize (integer array) */
   long drop; /* Number of entries dropped */
   int rank; /* Estimated rank */
   int stat; /* Fortran STAT value after allocate failure */
};


struct ma48_sinfo_d {
   int flag; /* Return code */
   int more; /* More information on failure */
   int stat; /* Fortran STAT value after allocate failure */
};

void ma48_default_control_d(struct ma48_control_d *control);

void ma48_initialize_d(void **factors);

void ma48_analyse_d(int m, int n, long ne, const int row[],
      const int col[], const ma48pkgtype_d_ val[], void *factors,
      const struct ma48_control_d *control, struct ma48_ainfo_d *ainfo,
      struct ma48_finfo_d *finfo, const int perm[], const int endcol[]);

void ma48_factorize_d(int m, int n, long ne, const int row[],
      const int col[], const ma48pkgtype_d_ val[], void *factors,
      const struct ma48_control_d *control, struct ma48_finfo_d *finfo,
      int fast, int partial);

void ma48_solve_d(int m, int n, long ne, const int row[],
      const int col[], const ma48pkgtype_d_ val[], const void *factors,
      const ma48pkgtype_d_ rhs[], ma48pkgtype_d_ x[],
      const struct ma48_control_d *control, struct ma48_sinfo_d *sinfo,
      int trans, ma48pkgtype_d_ resid[], ma48pkgtype_d_ *error);

int ma48_finalize_d(void **factors, const struct ma48_control_d *control);

int ma48_determinant_d(const void *factors, int *sgndet,
      ma48pkgtype_d_ *logdet, const struct ma48_control_d *control);

int ma48_special_rows_and_cols_d(const void *factors, int *rank,
      int rows[], int cols[], const struct ma48_control_d *control);

#endif

#ifdef __cplusplus
} /* extern "C" */
#endif
