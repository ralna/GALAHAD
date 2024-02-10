//* \file spral_ssids.h */
/**  
 * \version   GALAHAD 4.3 - 2024-02-04 AT 10:10 GMT
 */

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
#include "ssids_rip.hxx"

/************************************
 * Derived types
 ************************************/

struct spral_ssids_options {
   ipc_ array_base; // Not in Fortran type
   ipc_ print_level;
   ipc_ unit_diagnostics;
   ipc_ unit_error;
   ipc_ unit_warning;
   ipc_ ordering;
   ipc_ nemin;
   bool ignore_numa;
   bool use_gpu;
   bool gpu_only;
   longc_ min_gpu_work;
   float max_load_inbalance;
   float gpu_perf_coeff;
   ipc_ scaling;
   longc_ small_subtree_threshold;
   ipc_ cpu_block_size;
   bool action;
   ipc_ pivot_method;
   rpc_ small;
   rpc_ u;
   ipc_ nstream;
   rpc_ multiplier;
   float min_loadbalance;
   ipc_ failed_pivot_method;
   // char unused[80]; // Allow for future expansion
};

struct spral_ssids_inform {
   ipc_ flag;
   ipc_ matrix_dup;
   ipc_ matrix_missing_diag;
   ipc_ matrix_outrange;
   ipc_ matrix_rank;
   ipc_ maxdepth;
   ipc_ maxfront;
   ipc_ maxsupernode;
   ipc_ num_delay;
   longc_ num_factor;
   longc_ num_flops;
   ipc_ num_neg;
   ipc_ num_sup;
   ipc_ num_two;
   ipc_ stat;
   ipc_ cuda_error;
   ipc_ cublas_error;
   ipc_ not_first_pass;
   ipc_ not_second_pass;
   ipc_ nparts;
   longc_ cpu_flops;
   longc_ gpu_flops;
   // char unused[76]; // Allow for future expansion
};

/************************************
 * Basic subroutines
 ************************************/

/* Initialize options to defaults */
void spral_ssids_default_options(struct spral_ssids_options *options);
/* Perform analysis phase for CSC data */
void spral_ssids_analyse(bool check, ipc_ n, ipc_ *order, const longc_ *ptr,
      const ipc_ *row, const rpc_ *val, void **akeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
void spral_ssids_analyse_ptr32(bool check, ipc_ n, ipc_ *order, const ipc_ *ptr,
      const ipc_ *row, const rpc_ *val, void **akeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform analysis phase for coordinate data */
void spral_ssids_analyse_coord(ipc_ n, ipc_ *order, longc_ ne, const ipc_ *row,
      const ipc_ *col, const rpc_ *val, void **akeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform numerical factorization */
void spral_ssids_factor(bool posdef, const longc_ *ptr, const ipc_ *row,
      const rpc_ *val, rpc_ *scale, void *akeep, void **fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
void spral_ssids_factor_ptr32(bool posdef, const ipc_ *ptr, const ipc_ *row,
      const rpc_ *val, rpc_ *scale, void *akeep, void **fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform triangular solve(s) for single rhs */
void spral_ssids_solve1(ipc_ job, rpc_ *x1, void *akeep, void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Perform triangular solve(s) for one or more rhs */
void spral_ssids_solve(ipc_ job, ipc_ nrhs, rpc_ *x, ipc_ ldx, void *akeep,
      void *fkeep, const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);
/* Free memory */
ipc_ spral_ssids_free_akeep(void **akeep);
ipc_ spral_ssids_free_fkeep(void **fkeep);
ipc_ spral_ssids_free(void **akeep, void **fkeep);

/************************************
 * Advanced subroutines
 ************************************/

/* Retrieve information on pivots (positive-definite case) */
void spral_ssids_enquire_posdef(const void *akeep, const void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform, rpc_ *d);
/* Retrieve information on pivots (indefinite case) */
void spral_ssids_enquire_indef(const void *akeep, const void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform, ipc_ *piv_order, rpc_ *d);
/* Alter pivots (indefinite case only) */
void spral_ssids_alter(const rpc_ *d, const void *akeep, void *fkeep,
      const struct spral_ssids_options *options,
      struct spral_ssids_inform *inform);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // SPRAL_SSIDS_H
