//* \file spral_ssids.h */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef SPRAL_SSIDS_H
#define SPRAL_SSIDS_H

// precision
#include "galahad_precision.h"

/************************************
 * Derived types
 ************************************/

struct spral_ssids_options {
   int array_base; // Not in Fortran type
   int print_level;
   int unit_diagnostics;
   int unit_error;
   int unit_warning;
   int ordering;
   int nemin;
   bool ignore_numa;
   bool use_gpu;
   bool gpu_only;
   int64_t min_gpu_work;
   float max_load_inbalance;
   float gpu_perf_coeff;
   int scaling;
   int64_t small_subtree_threshold;
   int cpu_block_size;
   bool action;
   int pivot_method;
   real_wp_ small;
   real_wp_ u;
   int nstream;
   real_wp_ multiplier;
   float min_loadbalance;
   int failed_pivot_method;
   // char unused[80]; // Allow for future expansion
};

struct spral_ssids_inform {
   int flag;
   int matrix_dup;
   int matrix_missing_diag;
   int matrix_outrange;
   int matrix_rank;
   int maxdepth;
   int maxfront;
   int num_delay;
   int64_t num_factor;
   int64_t num_flops;
   int num_neg;
   int num_sup;
   int num_two;
   int stat;
   int cuda_error;
   int cublas_error;
   int maxsupernode;
   int not_first_pass;
   int not_second_pass;
   int nparts;
   int64_t cpu_flops;
   int64_t gpu_flops;
   // char unused[76]; // Allow for future expansion
};

/************************************
 * Basic subroutines
 ************************************/

/* Initialize options to defaults */
void spral_ssids_default_options(struct spral_ssids_options *options);
/* Perform analysis phase for CSC data */
void spral_ssids_analyse(bool check, int n, int *order, const int64_t *ptr,
      const int *row, const real_wp_ *val, void **akeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
void spral_ssids_analyse_ptr32(bool check, int n, int *order, const int *ptr,
      const int *row, const real_wp_ *val, void **akeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform analysis phase for coordinate data */
void spral_ssids_analyse_coord(int n, int *order, int64_t ne, const int *row,
      const int *col, const real_wp_ *val, void **akeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform numerical factorization */
void spral_ssids_factor(bool posdef, const int64_t *ptr, const int *row,
      const real_wp_ *val, real_wp_ *scale, void *akeep, void **fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
void spral_ssids_factor_ptr32(bool posdef, const int *ptr, const int *row,
      const real_wp_ *val, real_wp_ *scale, void *akeep, void **fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform triangular solve(s) for single rhs */
void spral_ssids_solve1(int job, real_wp_ *x1, void *akeep, void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform triangular solve(s) for one or more rhs */
void spral_ssids_solve(int job, int nrhs, real_wp_ *x, int ldx, void *akeep,
      void *fkeep, const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Free memory */
int spral_ssids_free_akeep(void **akeep);
int spral_ssids_free_fkeep(void **fkeep);
int spral_ssids_free(void **akeep, void **fkeep);

/************************************
 * Advanced subroutines
 ************************************/

/* Retrieve information on pivots (positive-definite case) */
void spral_ssids_enquire_posdef(const void *akeep, const void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform, real_wp_ *d);
/* Retrieve information on pivots (indefinite case) */
void spral_ssids_enquire_indef(const void *akeep, const void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform, int *piv_order, real_wp_ *d);
/* Alter pivots (indefinite case only) */
void spral_ssids_alter(const real_wp_ *d, const void *akeep, void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // SPRAL_SSIDS_H
