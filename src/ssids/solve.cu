/* Copyright 2013 The Science and Technology Facilities Council (STFC)
 * Copyright 2013 NVIDIA (in collaboration with STFC)
 *
 * Authors:
 * Jonathan Hogg        STFC     jonathan.hogg@stfc.ac.uk
 * Jeremey Appleyard    NVIDIA
 *
 * This code has not yet been publically released under any licence.
 * This version: GALAHAD 5.1 - 2024-11-21 AT 09:50 GMT
 */

#include <cublas_v2.h>
#include "spral_cuda_cuda_check.h"
#include "ssids_rip.hxx"

#ifdef REAL_32
#define gather gather_single
#define gemv_transpose_lookup gemv_transpose_lookup_single
#define gemv_transpose_sps_rhs gemv_transpose_sps_rhs_single
#define reducing_d_solve_lookup reducing_d_solve_lookup_single
#define reducing_d_solve reducing_d_solve_single
#define d_solve d_solve_single
#define scatter_lookup scatter_lookup_single
#define scatter scatter_single
#define scatter_sum scatter_sum_single
#define lookups_gpu_bwd lookups_gpu_bwd_single
#define simple_gemv simple_gemv_single
#define gemv_notrans_lookup gemv_notrans_lookup_single
#define reduce_notrans_lookup reduce_notrans_lookup_single
#define gemv_reduce_lookup gemv_reduce_lookup_single
#define assemble_blk_type assemble_blk_type_single
#define assemble_lookup2 assemble_lookup2_single
#define wait_for_sync wait_for_sync_single
#define assemble_lvl assemble_lvl_single
#define grabx grabx_single
#define lookups_gpu_fwd lookups_gpu_fwd_single
#define lookup_contrib_fwd lookup_contrib_fwd_single
#define spral_ssids_run_bwd_solve_kernels spral_ssids_run_bwd_solve_kernels_single
#define spral_ssids_run_d_solve_kernel spral_ssids_run_d_solve_kernel_single
#define spral_ssids_run_fwd_solve_kernels spral_ssids_run_fwd_solve_kernels_single
#define spral_ssids_run_slv_contrib_fwd spral_ssids_run_slv_contrib_fwd_single
#elif REAL_128
#define gather gather_quadruple
#define gemv_transpose_lookup gemv_transpose_lookup_quadruple
#define gemv_transpose_sps_rhs gemv_transpose_sps_rhs_quadruple
#define reducing_d_solve_lookup reducing_d_solve_lookup_quadruple
#define reducing_d_solve reducing_d_solve_quadruple
#define d_solve d_solve_quadruple
#define scatter_lookup scatter_lookup_quadruple
#define scatter scatter_quadruple
#define scatter_sum scatter_sum_quadruple
#define lookups_gpu_bwd lookups_gpu_bwd_quadruple
#define simple_gemv simple_gemv_quadruple
#define gemv_notrans_lookup gemv_notrans_lookup_quadruple
#define reduce_notrans_lookup reduce_notrans_lookup_quadruple
#define gemv_reduce_lookup gemv_reduce_lookup_quadruple
#define assemble_blk_type assemble_blk_type_quadruple
#define assemble_lookup2 assemble_lookup2_quadruple
#define wait_for_sync wait_for_sync_quadruple
#define assemble_lvl assemble_lvl_quadruple
#define grabx grabx_quadruple
#define lookups_gpu_fwd lookups_gpu_fwd_quadruple
#define lookup_contrib_fwd lookup_contrib_fwd_quadruple
#define spral_ssids_run_bwd_solve_kernels spral_ssids_run_bwd_solve_kernels_quadruple
#define spral_ssids_run_d_solve_kernel spral_ssids_run_d_solve_kernel_quadruple
#define spral_ssids_run_fwd_solve_kernels spral_ssids_run_fwd_solve_kernels_quadruple
#define spral_ssids_run_slv_contrib_fwd spral_ssids_run_slv_contrib_fwd_quadruple
#else
#define gather gather_double
#define gemv_transpose_lookup gemv_transpose_lookup_double
#define gemv_transpose_sps_rhs gemv_transpose_sps_rhs_double
#define reducing_d_solve_lookup reducing_d_solve_lookup_double
#define reducing_d_solve reducing_d_solve_double
#define d_solve d_solve_double
#define scatter_lookup scatter_lookup_double
#define scatter scatter_double
#define scatter_sum scatter_sum_double
#define lookups_gpu_bwd lookups_gpu_bwd_double
#define simple_gemv simple_gemv_double
#define gemv_notrans_lookup gemv_notrans_lookup_double
#define reduce_notrans_lookup reduce_notrans_lookup_double
#define gemv_reduce_lookup gemv_reduce_lookup_double
#define assemble_blk_type assemble_blk_type_double
#define assemble_lookup2 assemble_lookup2_double
#define wait_for_sync wait_for_sync_double
#define assemble_lvl assemble_lvl_double
#define grabx grabx_double
#define lookups_gpu_fwd lookups_gpu_fwd_double
#define lookup_contrib_fwd lookup_contrib_fwd_double
#define spral_ssids_run_bwd_solve_kernels spral_ssids_run_bwd_solve_kernels_double
#define spral_ssids_run_d_solve_kernel spral_ssids_run_d_solve_kernel_double
#define spral_ssids_run_fwd_solve_kernels spral_ssids_run_fwd_solve_kernels_double
#define spral_ssids_run_slv_contrib_fwd spral_ssids_run_slv_contrib_fwd_double
#endif

//#define MIN(x,y) (((x)>(y))?(y):(x))
#define MAX(x,y) (((x)>(y))?(x):(y))

#include "ssids_gpu_kernels_dtrsv.h"

#define TRSM_TR_NBX 256
#define TRSM_TR_NBY 32
#define TRSM_TR_THREADSX 32
#define TRSM_TR_THREADSY 4
#define REDUCING_D_SOLVE_THREADS_PER_BLOCK 256
#define SCATTER_NB 256
#define GEMV_NX 32
#define GEMV_NY 32
#define GEMV_THREADSX 32
#define GEMV_THREADSY 4
#define ASSEMBLE_NB 128

using namespace spral::ssids::gpu;

namespace /* anon */ {

/* Perform the assignment xdense(:) = xsparse( idx(:) ) */
template <int threadsx, ipc_ threadsy>
void __device__ gather(const ipc_ n, const ipc_ *const idx, const rpc_ *const xsparse,
      volatile rpc_ *const xdense) {
   ipc_ tid = threadsx*threadIdx.y + threadIdx.x;
   for(ipc_ i=tid; i<n; i+=threadsx*threadsy)
      xdense[i] = xsparse[ idx[i] ];
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

struct gemv_transpose_lookup {
   ipc_ m; // number of rows of L (cols of L^T) for block
   ipc_ n; // number of cols of L (rows of L^T) for block
   const rpc_ *a;
   ipc_ lda; // leading dimension of a
   const ipc_ *rlist;
   ipc_ yoffset; // offset ipc_o y for answer
};

/* This subroutine performs a matrix-vector multiplication y = Ax where
 * x is a sparse vector indexed into by rlist.
 * The lookup[] array is indexed into by the block id and specifies which
 * part of the matrix we're working on.
 *
 * Requires max(maxm + maxn*threadsx) shared memory.
 * Requires threadsy to exactly divide maxn.
 */
template <int threadsx, ipc_ threadsy, ipc_ maxm, ipc_ maxn>
__launch_bounds__(threadsx*threadsy, 6)
void __global__ gemv_transpose_sps_rhs(struct gemv_transpose_lookup *lookup,
      rpc_ *x, rpc_ *y
      ) {

   // Reuse shmem for two different purposes
   __shared__ volatile rpc_ shmem[maxn*threadsx];
   volatile rpc_ *const partSum = shmem;
   volatile rpc_ *const xlocal = shmem;

   rpc_ partSumReg[maxn / threadsy]; // Assumes neat division


   lookup += blockIdx.x;
   ipc_ m = lookup->m;
   ipc_ n = lookup->n;
   const rpc_ *a = lookup->a;
   const ipc_ *rlist = lookup->rlist;
   ipc_ lda = lookup->lda;
   y += lookup->yoffset;

   /* Read x(rlist(:)) ipc_o xlocal(:) */
   gather <threadsx,threadsy> (m, rlist, x, xlocal);
   __syncthreads();

   /* Perform matrix-vector multiply with answer y in register that
      is then stored in partSum for later reduction. */
   if(m==maxm) {
      volatile rpc_ *const xl = xlocal + threadIdx.x;
#pragma unroll
      for(ipc_ iLoop=0; iLoop<maxn/threadsy; iLoop++) { // row
         ipc_ i = iLoop * threadsy + threadIdx.y;
         partSumReg[iLoop] = 0;
         if (i < n) {
            const rpc_ *arow = a+i*lda+threadIdx.x;
            for(ipc_ j=0; j<maxm; j+=threadsx)
               partSumReg[iLoop] += xl[j] * arow[j];
         }
      }
   } else {
#pragma unroll
      for(ipc_ iLoop=0; iLoop<maxn/threadsy; iLoop++) { // row
         ipc_ i = iLoop * threadsy + threadIdx.y;
         partSumReg[iLoop] = 0;
         if (i < n) {
            const rpc_ *arow = a+i*lda;
            for(ipc_ j=threadIdx.x; j<m; j+=threadsx)
               partSumReg[iLoop] += xlocal[j] * arow[j];
         }
      }
   }

   __syncthreads(); // Wait till done with xlocal=shmem before using partSum
#pragma unroll
   for(ipc_ iLoop=0; iLoop<maxn/threadsy; iLoop++) { // row
      ipc_ i = iLoop * threadsy + threadIdx.y;
      if (i < n) {
         partSum[i*threadsx+threadIdx.x] = partSumReg[iLoop];
      }
   }

   __syncthreads();

   /* Reduce partSum across threads to get y contribution from this block */
   if(threadIdx.y==0) {
      for(ipc_ i=threadIdx.x; i<n; i+=threadsx) {
         rpc_ val = 0;
         /* The offset avoids large bank conflicts. */
         for(ipc_ j=threadIdx.x; j<threadsx+threadIdx.x; j++) {
            ipc_ j2 = (j >= threadsx ? j - threadsx : j);
            val += partSum[i*threadsx+j2];
         }
         y[i] = val;
      }
   }

}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

struct reducing_d_solve_lookup {
   ipc_ first_idx; // Index of supernode for thread 0 of this block.
   ipc_ m; // Number of columns in upd to reduce.
   ipc_ n; // Number of rows THIS BLOCK is responisble for.
   ipc_ ldupd; // Leading dimension of upd.
   ipc_ updoffset; // Offset into upd for supernode.
   const rpc_ *d;
   const ipc_ *perm; // Offset into perm for supernode.
};

/* This subroutine performs two unrelated tasks and subtracts the result of the
 * first from the second.
 * Task 1: Sum along the rows of the m x n matrix upd. (This is reducing the
 *         result of a previous gemv operation).
 * Task 2: Peform the special matrix-vector multiplication D^-1 P x where
 *         D is a block diagonal matrix with 1x1 and 2x2 blocks, and
 *         P is a (partial) permutation matrix, given by the vector perm.
 * The result x_2-x_1 is returned replacing the first column of upd.
 */
template <int threadsx, bool DSOLVE>
void __global__ reducing_d_solve(struct reducing_d_solve_lookup *lookup,
      rpc_ *upd, const rpc_ *x
      ) {

   /* Read details from lookup */
   lookup += blockIdx.x;
   ipc_ idx = lookup->first_idx + threadIdx.x;
   ipc_ m = lookup->m;
   ipc_ n = lookup->n;
   ipc_ ldupd = lookup->ldupd;
   upd += lookup->updoffset;
   const rpc_ *d = lookup->d;
   const ipc_ *perm = lookup->perm;


   /* Don't do anything on threads past end of arrays */
   if(threadIdx.x>=m) return;

   /* Task 1: Sum upd and negate */
   rpc_ val = upd[idx];
   for(ipc_ j=1; j<n; j++)
      val += upd[j*ldupd+idx];
   val = -val;

   /* Task 2: D solve (note that D is actually stored as inverse already) */
   if(DSOLVE) {
      ipc_ rp = perm[idx];
      if(idx!=0 && d[2*idx-1] != 0) {
         /* second part of 2x2 */
         ipc_ rp2 = perm[idx-1];
         val += d[2*idx-1] * x[rp2] +
                d[2*idx]   * x[rp];
      } else if (d[2*idx+1] != 0) {
         /* first part of 2x2 */
         ipc_ rp2 = perm[idx+1];
         val += d[2*idx]   * x[rp] +
                d[2*idx+1] * x[rp2];
      } else {
         /* 1x1 */
         val += x[rp]*d[2*idx];
      }
   } else {
      ipc_ rp = perm[idx];
      val += x[rp];
   }

   /* Store result as first column of upd */
   upd[idx] = val;

}

/* This subroutine only performs the solve with D. For best performance, use
 * reducing_d_solve() instead.
 * Peform the special matrix-vector multiplication D^-1 P x where
 * D is a block diagonal matrix with 1x1 and 2x2 blocks, and
 * P is a (partial) permutation matrix, given by the vector perm.
 * The result is not returned in-place due to 2x2 pivots potentially
 * split between blocks.
 */
template <int threadsx>
void __global__ d_solve(struct reducing_d_solve_lookup *lookup,
      const rpc_ *x, rpc_ *y) {

   /* Read details from lookup */
   lookup += blockIdx.x;
   ipc_ idx = lookup->first_idx + threadIdx.x;
   ipc_ m = lookup->m;
   const rpc_ *d = lookup->d;
   const ipc_ *perm = lookup->perm;

   /* Don't do anything on threads past end of arrays */
   if(threadIdx.x>=m) return;

   /* D solve (note that D is actually stored as inverse already) */
   ipc_ rp = perm[idx];
   rpc_ val;
   if(idx!=0 && d[2*idx-1] != 0) {
      /* second part of 2x2 */
      ipc_ rp2 = perm[idx-1];
      val = d[2*idx-1] * x[rp2] +
            d[2*idx]   * x[rp];
   } else if (d[2*idx+1] != 0) {
      /* first part of 2x2 */
      ipc_ rp2 = perm[idx+1];
      val = d[2*idx]   * x[rp] +
            d[2*idx+1] * x[rp2];
   } else {
      /* 1x1 */
      val = x[rp]*d[2*idx];
   }

   /* Store result in y[] */
   y[rp] = val;
}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

struct scatter_lookup {
   ipc_ n;
   ipc_ src_offset;
   const ipc_ *index;
   ipc_ dest_offset;
};

/* This subroutine performs the scatter operation dest( index(:) ) = src(:)
 */
void __global__ scatter(struct scatter_lookup *lookup, const rpc_ *src,
      rpc_ *dest
      ) {

   lookup += blockIdx.x;
   if(threadIdx.x >= lookup->n) return; // Skip on out of range threads
   src += lookup->src_offset;
   const ipc_ *index = lookup->index;
   dest += lookup->dest_offset;


   ipc_ idx = index[threadIdx.x];
   dest[idx] = src[threadIdx.x];

}

/* This subroutine performs the scatter operation dest( index(:) ) += src(:)
 */
void __global__ scatter_sum(struct scatter_lookup *lookup, const rpc_ *src,
      rpc_ *dest
      ) {

   lookup += blockIdx.x;
   if(threadIdx.x >= lookup->n) return; // Skip on out of range threads
   src += lookup->src_offset;
   const ipc_ *index = lookup->index;
   dest += lookup->dest_offset;


   ipc_ idx = index[threadIdx.x];
   dest[idx] += src[threadIdx.x];

}

/***********************************************************************/
/***********************************************************************/
/***********************************************************************/

struct lookups_gpu_bwd {
   ipc_ ngemv;
   ipc_ nrds;
   ipc_ ntrsv;
   ipc_ nscatter;
   struct gemv_transpose_lookup *gemv;
   struct reducing_d_solve_lookup *rds;
   struct trsv_lookup *trsv;
   struct scatter_lookup *scatter;
};

/*
 * Perform y = Ax
 * Result y actually output as array with leading dimn m that must be summed
 * externally.
 */
template <int threadsx, ipc_ threadsy, ipc_ maxm, ipc_ maxn>
void __global__ simple_gemv(ipc_ m, ipc_ n, const rpc_ *a, ipc_ lda,
      const rpc_ *x, rpc_ *y) {
   a += blockIdx.x*maxm + (blockIdx.y*maxn)*lda;
   x += blockIdx.y*maxn;
   y += m*blockIdx.y + maxm*blockIdx.x;

   __shared__ volatile rpc_ partSum[maxm*threadsy];

   m = MIN(maxm, m-blockIdx.x*maxm);
   n = MIN(maxn, n-blockIdx.y*maxn);

   volatile rpc_ *const ps = partSum + maxm*threadIdx.y;
   for(ipc_ j=threadIdx.x; j<m; j+=threadsx) {
      ps[j] = 0;
   }
   for(ipc_ i=threadIdx.y; i<n; i+=threadsy) {
      rpc_ xv = x[i];
      for(ipc_ j=threadIdx.x; j<m; j+=threadsx) {
         ps[j] += a[i*lda+j]*xv;
      }
   }

   __syncthreads();
   if(threadIdx.y==0) {
      for(ipc_ j=threadIdx.x; j<m; j+=threadsx) {
         rpc_ val = ps[j];
         for(ipc_ i=1; i<threadsy; i++) {
            val += ps[j+i*maxm];
         }
         y[j] = val;
      }
   }
}

struct gemv_notrans_lookup {
   ipc_ m;
   ipc_ n;
   const rpc_ *a;
   ipc_ lda;
   ipc_ x_offset;
   ipc_ y_offset;
};

template <int threadsx, ipc_ threadsy, ipc_ maxm, ipc_ maxn>
void __global__ simple_gemv_lookup(const rpc_ *x, rpc_ *y,
      struct gemv_notrans_lookup *lookup) {
   lookup += blockIdx.x;
   ipc_ m = lookup->m;
   ipc_ n = lookup->n;
   rpc_ const* a = lookup->a;
   ipc_ lda = lookup->lda;
   x += lookup->x_offset;
   y += lookup->y_offset;

   __shared__ volatile rpc_ partSum[maxm*threadsy];

   volatile rpc_ *const ps = partSum + maxm*threadIdx.y;

   // Templated parameters for shortcut
   if (maxm <= threadsx) {
      ps[threadIdx.x] = 0;
   }
   else {
      for(ipc_ j=threadIdx.x; j<m; j+=threadsx) {
         ps[j] = 0;
      }
   }
   for(ipc_ i=threadIdx.y; i<n; i+=threadsy) {
      rpc_ xv = x[i];
      // Templated parameters for shortcut - this reads out of bounds so shouldn't be uncommented
      /*if (maxm <= threadsx) {
         ps[threadIdx.x] += a[i*lda+threadIdx.x]*xv;
      }
      else {*/
         for(ipc_ j=threadIdx.x; j<m; j+=threadsx) {
            ps[j] += a[i*lda+j]*xv;
         }
      //}
   }

   __syncthreads();
   if(threadIdx.y==0) {
      // Templated parameters for shortcut
      if (maxm <= threadsx) {
         if (threadIdx.x < m) {
            rpc_ val = ps[threadIdx.x];
            for(ipc_ i=1; i<threadsy; i++) {
               val += ps[threadIdx.x+i*maxm];
            }
            y[threadIdx.x] = val;
         }
      }
      else {
         for(ipc_ j=threadIdx.x; j<m; j+=threadsx) {
            rpc_ val = ps[j];
            for(ipc_ i=1; i<threadsy; i++) {
               val += ps[j+i*maxm];
            }
            y[j] = val;
         }
      }
   }
}

struct reduce_notrans_lookup {
   ipc_ m;
   ipc_ n;
   ipc_ src_offset;
   ipc_ ldsrc;
   ipc_ dest_idx;
   ipc_ dest_offset;
};

void __global__ gemv_reduce_lookup(const rpc_ *src, rpc_ **dest, ipc_ numLookups, struct reduce_notrans_lookup *lookup) {
   ipc_ offset = blockIdx.x * blockDim.y + threadIdx.y;
   if (offset >= numLookups) return;

   lookup += offset;
   ipc_ m = lookup->m;
   if(threadIdx.x>=m) return;
   ipc_ n = lookup->n;
   src += lookup->src_offset + threadIdx.x;
   ipc_ ldsrc = lookup->ldsrc;
   rpc_ *d = dest[lookup->dest_idx] + lookup->dest_offset;

   rpc_ val = 0;
   for(ipc_ i=0; i<n; i++)
      val += src[i*ldsrc];
   d[threadIdx.x] -= val;
}

// FIXME: move to common header?
struct assemble_blk_type {
   ipc_ cp;
   ipc_ blk;
};

struct assemble_lookup {
   ipc_ m;
   ipc_ xend;
   ipc_ const* list;
   ipc_ x_offset;
   ipc_ contrib_idx;
   ipc_ contrib_offset;
   ipc_ nchild;
   ipc_ const* clen;
   ipc_ * const* clists;
   ipc_ * const* clists_direct;
   ipc_ cvalues_offset;
   ipc_ first; // First index of node. Used to shortcut searching
};

struct assemble_lookup2 {
   ipc_ m;
   ipc_ nelim;
   ipc_ x_offset;
   ipc_ *const* list;
   ipc_ cvparent;
   ipc_ cvchild;
   ipc_ sync_offset;
   ipc_ sync_waitfor;
};

void __device__ wait_for_sync(const ipc_ tid, volatile ipc_ *const sync, const ipc_ target) {
   if(tid==0) {
      while(*sync < target) {}
   }
   __syncthreads();
}

void __global__ assemble_lvl(struct assemble_lookup2 *lookup, struct assemble_blk_type *blkdata, rpc_ *xlocal, ipc_ *next_blk, volatile ipc_ *sync, rpc_ * const* cvalues) {
   __shared__ volatile ipc_ thisblk;
   if(threadIdx.x==0)
      thisblk = atomicAdd(next_blk, 1);
   __syncthreads();

   blkdata += thisblk;
   lookup += blkdata->cp;

   ipc_ blk = blkdata->blk;
   ipc_ m = lookup->m;
   ipc_ nelim = lookup->nelim;
   rpc_ *xparent = cvalues[lookup->cvparent];
   volatile const rpc_ *xchild = cvalues[lookup->cvchild];
   const ipc_ * list = *(lookup->list);
   xlocal += lookup->x_offset;

   // Wait for previous children to complete
   wait_for_sync(threadIdx.x, &(sync[lookup->sync_offset]), lookup->sync_waitfor);

   // Add block increments
   m = MIN(ASSEMBLE_NB, m-blk*ASSEMBLE_NB);
   list += blk*ASSEMBLE_NB;
   xchild += blk*ASSEMBLE_NB;

   // Perform actual assembly
   for(ipc_ i=threadIdx.x; i<m; i+=blockDim.x) {
      ipc_ j = list[i];
      if(j < nelim) {
         xlocal[j] += xchild[i];
      } else {
         xparent[j-nelim] += xchild[i];
      }
   }

   // Wait for all threads to complete, then increment sync object
   __threadfence();
   __syncthreads();
   if(threadIdx.x==0) {
      atomicAdd((ipc_*)&(sync[lookup->sync_offset]), 1);
   }
}

void __global__ grabx(rpc_ *xlocal, rpc_ **xstack, const rpc_ *x,
      struct assemble_lookup *lookup) {

   lookup += blockIdx.x;
   if(threadIdx.x>=lookup->m) return;
   ipc_ xend = lookup->xend;
   rpc_ *contrib =
      (threadIdx.x>=xend) ?
         xstack[lookup->contrib_idx]+lookup->contrib_offset :
         NULL;
   xlocal += lookup->x_offset;

   ipc_ row = lookup->list[threadIdx.x];

   if(threadIdx.x<xend) xlocal[threadIdx.x] = x[row];
   else                 contrib[threadIdx.x] = 0.0;
}

struct lookups_gpu_fwd {
   ipc_ nassemble;
   ipc_ nasm_sync;
   ipc_ nassemble2;
   ipc_ nasmblk;
   ipc_ ntrsv;
   ipc_ ngemv;
   ipc_ nreduce;
   ipc_ nscatter;
   struct assemble_lookup *assemble;
   struct assemble_lookup2 *assemble2;
   struct assemble_blk_type *asmblk;
   struct trsv_lookup *trsv;
   struct gemv_notrans_lookup *gemv;
   struct reduce_notrans_lookup *reduce;
   struct scatter_lookup *scatter;
};

struct lookup_contrib_fwd {
   ipc_ nscatter;
   struct scatter_lookup *scatter;
};

} /* anon namespace */

/*******************************************************************************
 * Following routines are exported with C binding so can be called from Fortran
 ******************************************************************************/

extern "C" {

void spral_ssids_run_fwd_solve_kernels(bool posdef,
      struct lookups_gpu_fwd const* gpu, rpc_ *xlocal_gpu,
      rpc_ **xstack_gpu, rpc_ *x_gpu, rpc_ ** cvalues_gpu,
      rpc_ *work_gpu, ipc_ nsync, ipc_ *sync, ipc_ nasm_sync, ipc_ *asm_sync,
      const cudaStream_t *stream) {

   if(nsync>0) {
      for(ipc_ i=0; i<nsync; i+=65535)
         trsv_init <<<MIN(65535,nsync-i), 1, 0, *stream>>> (sync+2*i);
      CudaCheckError();
   }
   for(ipc_ i=0; i<gpu->nassemble; i+=65535)
      grabx
         <<<MIN(65535,gpu->nassemble-i), ASSEMBLE_NB, 0, *stream>>>
         (xlocal_gpu, xstack_gpu, x_gpu, gpu->assemble+i);
   cudaMemset(asm_sync, 0, (1+gpu->nasm_sync)*sizeof(ipc_));
   for(ipc_ i=0; i<gpu->nasmblk; i+=65535)
      assemble_lvl
         <<<MIN(65535,gpu->nasmblk-i), ASSEMBLE_NB, 0, *stream>>>
         (gpu->assemble2, gpu->asmblk, xlocal_gpu, &asm_sync[0], &asm_sync[1], cvalues_gpu);
   CudaCheckError();
   if(gpu->ntrsv>0) {
      if(posdef) {
         for(ipc_ i=0; i<gpu->ntrsv; i+=65535)
            trsv_ln_exec
               <rpc_,TRSV_NB_TASK,THREADSX_TASK,THREADSY_TASK,false>
               <<<MIN(65535,gpu->ntrsv-i), dim3(THREADSX_TASK,THREADSY_TASK), 0, *stream>>>
               (xlocal_gpu, sync, gpu->trsv+i);
      } else {
         for(ipc_ i=0; i<gpu->ntrsv; i+=65535)
            trsv_ln_exec
               <rpc_,TRSV_NB_TASK,THREADSX_TASK,THREADSY_TASK,true>
               <<<MIN(65535,gpu->ntrsv-i), dim3(THREADSX_TASK,THREADSY_TASK), 0, *stream>>>
               (xlocal_gpu, sync, gpu->trsv+i);
      }
      CudaCheckError();
   }
   if(gpu->ngemv>0) {
      for(ipc_ i=0; i<gpu->ngemv; i+=65535)
         simple_gemv_lookup
            <GEMV_THREADSX, GEMV_THREADSY, GEMV_NX, GEMV_NY>
            <<<MIN(65535,gpu->ngemv-i), dim3(GEMV_THREADSX,GEMV_THREADSY), 0, *stream>>>
            (xlocal_gpu, work_gpu, gpu->gemv+i);
      CudaCheckError();
   }
   if(gpu->nreduce>0) {
      if((gpu->nreduce + 4 - 1) / 4 > 65535)
         printf("Unhandled error! fwd solve gemv_reduce_lookup()\n");
      gemv_reduce_lookup
         <<<dim3((gpu->nreduce + 4 - 1) / 4), dim3(GEMV_NX, 4), 0, *stream>>>
         (work_gpu, cvalues_gpu, gpu->nreduce, gpu->reduce);
      CudaCheckError();
   }
   for(ipc_ i=0; i<gpu->nscatter; i+=65535)
      scatter
         <<<MIN(65535,gpu->nscatter-i), SCATTER_NB, 0, *stream>>>
         (gpu->scatter+i, xlocal_gpu, x_gpu);
   CudaCheckError();
}

void spral_ssids_run_d_solve_kernel(rpc_ *x_gpu,
      rpc_ *y_gpu, struct lookups_gpu_bwd *gpu,
     const cudaStream_t *stream) {

   if(gpu->nrds>0) {
      d_solve
         <REDUCING_D_SOLVE_THREADS_PER_BLOCK>
         <<<gpu->nrds, REDUCING_D_SOLVE_THREADS_PER_BLOCK, 0, *stream>>>
         (gpu->rds, x_gpu, y_gpu);
      CudaCheckError();
   }
}

void spral_ssids_run_bwd_solve_kernels(bool dsolve,
      bool unit_diagonal, rpc_ *x_gpu, rpc_ *work_gpu,
      ipc_ nsync, ipc_ *sync_gpu, struct lookups_gpu_bwd *gpu,
      const cudaStream_t *stream) {

   /* === Kernel Launches === */
   if(nsync>0) {
      for(ipc_ i=0; i<nsync; i+=65535)
         trsv_init <<<MIN(65535,nsync-i), 1, 0, *stream>>> (sync_gpu+2*i);
      CudaCheckError();
   }
   if(gpu->ngemv>0) {
      for(ipc_ i=0; i<gpu->ngemv; i+=65535)
         gemv_transpose_sps_rhs
            <TRSM_TR_THREADSX, TRSM_TR_THREADSY, TRSM_TR_NBX, TRSM_TR_NBY>
            <<<MIN(65535,gpu->ngemv-i), dim3(TRSM_TR_THREADSX,TRSM_TR_THREADSY), 0, *stream>>>
            (gpu->gemv+i, x_gpu, work_gpu);
      CudaCheckError();
   }

   if(gpu->nrds>0) {
      if(dsolve) {
         for(ipc_ i=0; i<gpu->nrds; i+=65535)
            reducing_d_solve
               <REDUCING_D_SOLVE_THREADS_PER_BLOCK, true>
               <<<MIN(65535,gpu->nrds-i), REDUCING_D_SOLVE_THREADS_PER_BLOCK, 0, *stream>>>
               (gpu->rds+i, work_gpu, x_gpu);
      } else {
         for(ipc_ i=0; i<gpu->nrds; i+=65535)
            reducing_d_solve
               <REDUCING_D_SOLVE_THREADS_PER_BLOCK, false>
               <<<MIN(65535,gpu->nrds-i), REDUCING_D_SOLVE_THREADS_PER_BLOCK, 0, *stream>>>
               (gpu->rds+i, work_gpu, x_gpu);
      }
      CudaCheckError();
   }

   if(gpu->ntrsv>0) {
      if(unit_diagonal) {
         for(ipc_ i=0; i<gpu->ntrsv; i+=65535)
            trsv_lt_exec
               <rpc_,TRSV_NB_TASK,THREADSX_TASK,THREADSY_TASK,true>
               <<<MIN(65535,gpu->ntrsv-i), dim3(THREADSX_TASK,THREADSY_TASK), 0, *stream>>>
               (gpu->trsv+i, work_gpu, sync_gpu);
      } else {
         for(ipc_ i=0; i<gpu->ntrsv; i+=65535)
            trsv_lt_exec
               <rpc_,TRSV_NB_TASK,THREADSX_TASK,THREADSY_TASK,false>
               <<<MIN(65535,gpu->ntrsv-i), dim3(THREADSX_TASK,THREADSY_TASK), 0, *stream>>>
               (gpu->trsv+i, work_gpu, sync_gpu);
      }
      CudaCheckError();
   }

   if(gpu->nscatter>0) {
      for(ipc_ i=0; i<gpu->nscatter; i+=65535)
         scatter
            <<<MIN(65535,gpu->nscatter-i), SCATTER_NB, 0, *stream>>>
            (gpu->scatter+i, work_gpu, x_gpu);
      CudaCheckError();
   }
}

void spral_ssids_run_slv_contrib_fwd(
      struct lookup_contrib_fwd const* gpu,
      rpc_* x_gpu, rpc_ const* xstack_gpu,
      const cudaStream_t *stream) {
   if(gpu->nscatter>0) {
      for(ipc_ i=0; i<gpu->nscatter; i+=65535)
         scatter_sum
            <<<MIN(65535,gpu->nscatter-i), SCATTER_NB, 0, *stream>>>
            (gpu->scatter+i, xstack_gpu, x_gpu);
      CudaCheckError();
   }
}

} // end extern "C"
