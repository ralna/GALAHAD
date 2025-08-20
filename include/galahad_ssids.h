//* \file galahad_ssids.h */
/**
 * \version   GALAHAD 5.3 - 2025-08-25 AT 14:10 GMT
 */

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif

// include guard
#ifndef GALAHAD_SSIDS_H
#define GALAHAD_SSIDS_H

// precision
#include "galahad_precision.h"
#include "galahad_cfunctions.h"

// required packages
#include "galahad_nodend.h"

/************************************
 * Derived types
 ************************************/

struct ssids_control_type {
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
   int64_t min_gpu_work;
   real_sp_ max_load_inbalance;
   real_sp_ gpu_perf_coeff;
   ipc_ scaling;
   int64_t small_subtree_threshold;
   ipc_ cpu_block_size;
   bool action;
   ipc_ pivot_method;
   rpc_ small;
   rpc_ u;
   struct nodend_control_type nodend_control;
   ipc_ nstream;
   rpc_ multiplier;
   real_sp_ min_loadbalance;
   ipc_ failed_pivot_method;
   // char unused[80]; // Allow for future expansion
};

struct ssids_inform_type {
   ipc_ flag;
   ipc_ matrix_dup;
   ipc_ matrix_missing_diag;
   ipc_ matrix_outrange;
   ipc_ matrix_rank;
   ipc_ maxdepth;
   ipc_ maxfront;
   ipc_ maxsupernode;
   ipc_ num_delay;
   int64_t num_factor;
   int64_t num_flops;
   ipc_ num_neg;
   ipc_ num_sup;
   ipc_ num_two;
   ipc_ stat;
   ipc_ cuda_error;
   ipc_ cublas_error;
   struct nodend_inform_type nodend_inform;
   ipc_ not_first_pass;
   ipc_ not_second_pass;
   ipc_ nparts;
   int64_t cpu_flops;
   int64_t gpu_flops;
   // char unused[76]; // Allow for future expansion
};

/************************************
 * Basic subroutines
 ************************************/

/* Initialize control to defaults */
void ssids_default_control(struct ssids_control_type *control);
/* Perform analysis phase for CSC data */
void ssids_analyse(bool check, ipc_ n, ipc_ *order, const int64_t *ptr,
      const ipc_ *row, const rpc_ *val, void **akeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
void ssids_analyse_ptr32(bool check, ipc_ n, ipc_ *order, const ipc_ *ptr,
      const ipc_ *row, const rpc_ *val, void **akeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
/* Perform analysis phase for coordinate data */
void ssids_analyse_coord(ipc_ n, ipc_ *order, int64_t ne, const ipc_ *row,
      const ipc_ *col, const rpc_ *val, void **akeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
/* Perform numerical factorization */
void ssids_factor(bool posdef, const int64_t *ptr, const ipc_ *row,
      const rpc_ *val, rpc_ *scale, void *akeep, void **fkeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
void ssids_factor_ptr32(bool posdef, const ipc_ *ptr, const ipc_ *row,
      const rpc_ *val, rpc_ *scale, void *akeep, void **fkeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
/* Perform triangular solve(s) for single rhs */
void ssids_solve1(ipc_ job, rpc_ *x1, void *akeep, void *fkeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
/* Perform triangular solve(s) for one or more rhs */
void ssids_solve(ipc_ job, ipc_ nrhs, rpc_ *x, ipc_ ldx, void *akeep,
      void *fkeep, const struct ssids_control_type *control,
      struct ssids_inform_type *inform);
/* Free memory */
ipc_ ssids_free_akeep(void **akeep);
ipc_ ssids_free_fkeep(void **fkeep);
ipc_ ssids_free(void **akeep, void **fkeep);

/************************************
 * Advanced subroutines
 ************************************/

/* Retrieve information on pivots (positive-definite case) */
void ssids_enquire_posdef(const void *akeep, const void *fkeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform, rpc_ *d);
/* Retrieve information on pivots (indefinite case) */
void ssids_enquire_indef(const void *akeep, const void *fkeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform, ipc_ *piv_order, rpc_ *d);
/* Alter pivots (indefinite case only) */
void ssids_alter(const rpc_ *d, const void *akeep, void *fkeep,
      const struct ssids_control_type *control,
      struct ssids_inform_type *inform);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif // GALAHAD_SSIDS_H
