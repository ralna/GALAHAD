/* Copyright (c) 2013 Science and Technology Facilities Council (STFC)
 * Copyright (c) 2013 NVIDIA
 * Authors: Evgueni Ovtchinnikov (STFC)
 *          Jeremy Appleyard (NVIDIA)
 * This version: GALAHAD 5.0 - 2024-06-11 AT 09:40 GMT
 */

#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "ssids_rip.hxx"
#include "spral_cuda_cuda_check.h"
#include "ssids_gpu_kernels_datatypes.h"

#ifdef REAL_32
#define load_nodes_type load_nodes_type_single
#define assemble_cp_type assemble_cp_type_single
#define assemble_blk_type assemble_blk_type_single
#define assemble_delay_type assemble_delay_type_single
#define assemble assemble_single
#define add_delays add_delays_single
#define cu_load_nodes cu_load_nodes_single
#define cu_load_nodes_sc cu_load_nodes_sc_single
#define cu_max_abs cu_max_abs_single
#define spral_ssids_add_delays spral_ssids_add_delays_single
#define spral_ssids_assemble spral_ssids_assemble_single
#define spral_ssids_load_nodes spral_ssids_load_nodes_single
#define spral_ssids_load_nodes_sc spral_ssids_load_nodes_sc_single
#define spral_ssids_max_abs spral_ssids_max_abs_single
#elif REAL_128
#define load_nodes_type load_nodes_type_quadruple
#define assemble_cp_type assemble_cp_type_quadruple
#define assemble_blk_type assemble_blk_type_quadruple
#define assemble_delay_type assemble_delay_type_quadruple
#define assemble assemble_quadruple
#define add_delays add_delays_quadruple
#define cu_load_nodes cu_load_nodes_quadruple
#define cu_load_nodes_sc cu_load_nodes_sc_quadruple
#define cu_max_abs cu_max_abs_quadruple
#define spral_ssids_add_delays spral_ssids_add_delays_quadruple
#define spral_ssids_assemble spral_ssids_assemble_quadruple
#define spral_ssids_load_nodes spral_ssids_load_nodes_quadruple
#define spral_ssids_load_nodes_sc spral_ssids_load_nodes_sc_quadruple
#define spral_ssids_max_abs spral_ssids_max_abs_quadruple
#else
#define load_nodes_type load_nodes_type_single
#define assemble_cp_type assemble_cp_type_double
#define assemble_blk_type assemble_blk_type_double
#define assemble_delay_type assemble_delay_type_double
#define assemble assemble_double
#define add_delays add_delays_double
#define cu_load_nodes cu_load_nodes_double
#define cu_load_nodes_sc cu_load_nodes_sc_double
#define cu_max_abs cu_max_abs_double
#define spral_ssids_add_delays spral_ssids_add_delays_double
#define spral_ssids_assemble spral_ssids_assemble_double
#define spral_ssids_load_nodes spral_ssids_load_nodes_double
#define spral_ssids_load_nodes_sc spral_ssids_load_nodes_sc_double
#define spral_ssids_max_abs spral_ssids_max_abs_double
#endif

#define HOGG_ASSEMBLE_TX 128  // Block height
#define HOGG_ASSEMBLE_TY 8    // Block width
#define HOGG_ASSEMBLE_NTX 32  // Number of threads x
#define HOGG_ASSEMBLE_NTY 4   // Number of threads y
#define ADD_DELAYS_TX 32
#define ADD_DELAYS_TY 4

namespace /* anon */ {

struct load_nodes_type {
  longc_ nnz;    // Number of entries to map
  ipc_ lda;    // Leading dimension of A
  ipc_ ldl;    // Leading dimension of L
  rpc_ *lcol; // Pointer to non-delay part of L
  longc_ offn;   // Offset into nlist
  longc_ offr;  // Offset into rlist
};

/*
 * Perform assembly according to nlist:
 * lval( nlist(2,i) ) = val( nlist(1,i) )    (in Fortran)
 *
 * Each block handles one node (regardless of size!!!)
 * Note: modified value lval is passed in via pointer in lndata, not as argument
 */
__global__ void
cu_load_nodes(
    const struct load_nodes_type *lndata,
    const longc_ *nlist,
    const rpc_ *aval
) {
   lndata += blockIdx.x;
   const longc_ nnz = lndata->nnz;
   const ipc_ lda = lndata->lda;
   const ipc_ ldl = lndata->ldl;

   nlist += 2*lndata->offn;
   rpc_ *const lval = lndata->lcol;

   for (ipc_ i = threadIdx.x; i < nnz; i += blockDim.x) {
     // Note: nlist is 1-indexed, not 0 indexed, so we have to adjust
     const ipc_ r = (nlist[2*i+1] - 1) % lda; // row index
     const ipc_ c = (nlist[2*i+1] - 1) / lda; // col index
     const longc_ sidx = nlist[2*i+0] - 1; // source index
     lval[r + c*ldl] = aval[sidx];
   }
}

/*
 * Perform assembly according to nlist:
 * lval( nlist(2,i) ) = val( nlist(1,i) )    (in Fortran)
 * with the added twist of needing to perform a scaling at the same time
 *
 * Each block handles one node (regardless of size!!!)
 * Note: modified value lval is passed in via pointer in lndata, not as argument
 */
__global__ void
cu_load_nodes_sc(
    const struct load_nodes_type *lndata,
    const longc_ *nlist,
    const ipc_ *rlist,
    const rpc_ *scale,
    const rpc_ *aval
) {
   lndata += blockIdx.x;
   const ipc_ nnz = lndata->nnz;
   const ipc_ lda = lndata->lda;
   const ipc_ ldl = lndata->ldl;

   nlist += 2*lndata->offn;
   rpc_ *const lval = lndata->lcol;
   rlist += lndata->offr;

   for (ipc_ i = threadIdx.x; i < nnz; i += blockDim.x) {
      // Note: nlist and rlist are 1-indexed, not 0 indexed, so we adjust
      const ipc_ r = (nlist[2*i+1] - 1) % lda; // row index
      const ipc_ c = (nlist[2*i+1] - 1) / lda; // col index
      const longc_ sidx = nlist[2*i+0] - 1; // source index
      const rpc_ rs = scale[rlist[r] - 1]; // row scaling
      const rpc_ cs = scale[rlist[c] - 1]; // col scaling
      lval[r + c*ldl] = rs * aval[sidx] * cs;
   }
}

// BLOCK_SIZE = blockDim.x
// maxabs must be initialized to zeros
template< typename ELEMENT_TYPE, uipc_ BLOCK_SIZE >
__global__ void
cu_max_abs( const longc_ n, const ELEMENT_TYPE *const u, ELEMENT_TYPE *const maxabs )
{
  __shared__ volatile ELEMENT_TYPE tmax[BLOCK_SIZE];

  tmax[threadIdx.x] = 0.0;
  for ( longc_ i = threadIdx.x + blockDim.x*blockIdx.x; i < n;
        i += blockDim.x*gridDim.x ) {
    const ELEMENT_TYPE v = fabs(u[i]);
    if ( v > tmax[threadIdx.x] )
      tmax[threadIdx.x] = v;
  }
  __syncthreads();

  for ( ipc_ inc = 1; inc < BLOCK_SIZE; inc *= 2 ) {
    if ( 2*inc*threadIdx.x + inc < BLOCK_SIZE
        && tmax[2*inc*threadIdx.x + inc] > tmax[2*inc*threadIdx.x] )
      tmax[2*inc*threadIdx.x] = tmax[2*inc*threadIdx.x + inc];
    __syncthreads();
  }
  if ( threadIdx.x == 0 && tmax[0] > 0.0 )
    maxabs[blockIdx.x] = tmax[0];
}


/* Following data type describes a single child-parent assembly */
struct assemble_cp_type {
  // Parent data
  ipc_ pvoffset; // Offset to start of parent node values
  rpc_ *pval; // Pointer to non-delay part of parent L
  ipc_ ldp; // Leading dimension of parent

  // Child data
  ipc_ cm; // Number of rows in child
  ipc_ cn; // Number of columns in child
  ipc_ ldc; // Leading dimension of child
  longc_ cvoffset; // Offset to start of child node values
  rpc_ *cv; // Pointer to start of child node values

  // Alignment data
  ipc_ *rlist_direct; // Pointer to start of child's rlist
  ipc_ *ind; // Pointer to start of child's contribution index

  // Sync data
  ipc_ sync_offset; // we watch sync[sync_offset]
  ipc_ sync_wait_for; // and wait for it to have value >= sync_wait_for
};

/* Following data type describes actions of single CUDA block */
struct assemble_blk_type {
  ipc_ cp; // node we're assembling into
  ipc_ blk; // block number of that node
};

/* Used to force volatile load of a declared non-volatile variable */
template <typename T_ELEM>
__inline__ __device__ T_ELEM loadVolatile(volatile T_ELEM *const vptr) {
  return *vptr;
}

/* Performs sparse assembly of a m x n child into a parent as dictated by
 * rlist_direct (supplied as part of cpdata).
 *
 * A lookup is performed in blkdata to determine which child-parent assembly
 * is to be performed next, and which block of that assembly this is.
 *
 * next_blk is used to ensure all blocks run in exact desired order.
 * sync[] is used to ensure dependencies are completed in the correct order.
 */
template <uipc_ blk_sz_x, uipc_ blk_sz_y, uipc_ ntx, unsigned nty>
void __global__ assemble(
    const struct assemble_blk_type *blkdata, // block mapping
    const struct assemble_cp_type *cpdata, // child-parent data
    const rpc_ *const children, // pointer to array containing children
    rpc_ *const parents, // pointer to array containing parents
    uipc_ *const next_blk, // gmem location used to determine next block
    volatile uipc_ *const sync // sync[cp] is #blocks completed so far for cp
) {
   // Get block number
   __shared__ volatile uipc_ mynext_blk;
   if(threadIdx.x==0 && threadIdx.y==0)
      mynext_blk = atomicAdd(next_blk, 1);
   __syncthreads();

   // Determine global information
   blkdata += mynext_blk;
   cpdata += blkdata->cp;
   ipc_ blk = blkdata->blk;
   ipc_ nx = (cpdata->cm-1) / blk_sz_x + 1; // number of blocks high child is
   ipc_ bx = blk % nx; // coordinate of block in x direction
   ipc_ by = blk / nx; // coordinate of block in y direction
   ipc_ ldc = cpdata->ldc;
   ipc_ ldp = cpdata->ldp;

   // Initialize local information
   ipc_ m = min(blk_sz_x, cpdata->cm - bx*blk_sz_x);
   ipc_ n = min(blk_sz_y, cpdata->cn - by*blk_sz_y);
   const rpc_ *src =
      cpdata->cv + ldc*by*blk_sz_y + bx*blk_sz_x;
   rpc_ *dest = cpdata->pval;
   ipc_ *rows = cpdata->rlist_direct + bx*blk_sz_x;
   ipc_ *cols = cpdata->rlist_direct + by*blk_sz_y;

   // Wait for previous child of this parent to complete
   if(threadIdx.x==0 && threadIdx.y==0) {
      while(sync[cpdata->sync_offset] < cpdata->sync_wait_for) /**/;
   }
   __syncthreads();

   // Perform assembly
   for(ipc_ j=0; j<blk_sz_y/nty; j++) {
      if( threadIdx.y+j*nty < n ) {
         ipc_ col = cols[threadIdx.y+j*nty]-1;
         for(ipc_ i=0; i<blk_sz_x/ntx; i++) {
            if( threadIdx.x+i*ntx < m ) {
               ipc_ row = rows[threadIdx.x+i*ntx]-1;
               dest[row + col*ldp] +=
                  src[threadIdx.x+i*ntx + (threadIdx.y+j*nty)*ldc];
            }
         }
      }
   }

   // Record that we're done
   __syncthreads();
   if(threadIdx.x==0 && threadIdx.y==0) {
      atomicAdd((ipc_*)&(sync[blkdata->cp]), 1);
   }
}

struct assemble_delay_type {
  ipc_ dskip; // Number of rows to skip for delays from later children
  ipc_ m; // Number of rows in child to copy
  ipc_ n; // Number of cols in child to copy
  ipc_ ldd; // Leading dimension of dest (parent)
  ipc_ lds; // Leading dimension of src (child)
  rpc_ *dval; // Pointer to dest (parent)
  rpc_ *sval; // Pointer to src (child)
  longc_ roffset; // Offset to rlist_direct
};

/* Copies delays from child to parent using one block per parent
 * Note: src and dest pointers both contained in dinfo
 */
void __global__ add_delays(
    struct assemble_delay_type *dinfo, // information on each block
    const ipc_ *rlist_direct // children's rows indices in parents
) {
   dinfo += blockIdx.x;
   const ipc_ dskip = dinfo->dskip; // number of delays
   const ipc_ m = dinfo->m; // number of rows
   const ipc_ n = dinfo->n; // number of cols
   const ipc_ ldd = dinfo->ldd; // leading dimension of dest
   const ipc_ lds = dinfo->lds; // leading dimension of src

   rpc_ *const dest = dinfo->dval;
   const rpc_ *const src = dinfo->sval;
   rlist_direct += dinfo->roffset;

   for ( ipc_ y = threadIdx.y; y < n; y += blockDim.y ) {
      for ( ipc_ x = threadIdx.x; x < m; x += blockDim.x ) {
         if ( x < n ) {
            dest[x + y*ldd] = src[x + y*lds];
         }
         else {
            ipc_ xt = dskip + rlist_direct[x - n] - 1;
            dest[xt + y*ldd] = src[x + y*lds];
         }
      }
   }
}

} /* anon namespace */

/*******************************************************************************
 * Following routines are exported with C binding so can be called from Fortran
 ******************************************************************************/

extern "C" {

/* Invokes the add_delays<<<>>>() kernel */
void spral_ssids_add_delays( const cudaStream_t *stream, ipc_ ndblk,
      struct assemble_delay_type *gpu_dinfo, ipc_ *rlist_direct ) {
   if ( ndblk == 0 ) return; // Nothing to see here
   dim3 threads(ADD_DELAYS_TX, ADD_DELAYS_TY);
   for ( ipc_ i = 0; i < ndblk; i += MAX_CUDA_BLOCKS ) {
      ipc_ nb = min(MAX_CUDA_BLOCKS, ndblk - i);
      add_delays
         <<< nb, threads, 0, *stream >>>
         ( gpu_dinfo + i, rlist_direct );
      CudaCheckError();
   }
}

/* Runs the kernel assemble<<<>>>() after setting up memory correctly. */
/* Requires gpu_next_sync[] to be of size >= (1+ncp)*sizeof(uipc_) */
void spral_ssids_assemble(const cudaStream_t *stream, ipc_ nblk, ipc_ blkoffset,
      struct assemble_blk_type *blkdata, ipc_ ncp,
      struct assemble_cp_type *cpdata, rpc_ *children,
      rpc_ *parents, uipc_ *gpu_next_sync) {
   /* Create and initialize synchronization objects using a single call:
      next_blk[1]
      sync[ncp]
    */
   CudaSafeCall(
         cudaMemsetAsync(gpu_next_sync,0,(1+ncp)*sizeof(uipc_),*stream)
         );
   /* Note, that we can only have at most 65535 blocks per dimn.
    * For some problems, nblk can exceed this, so we use more than one launch.
    * As the next block we look at is specified by next_blk this works fine.
    */
   dim3 threads(HOGG_ASSEMBLE_NTX, HOGG_ASSEMBLE_NTY);
   for(ipc_ i=0; i<nblk; i+=MAX_CUDA_BLOCKS) {
      ipc_ blocks = min(MAX_CUDA_BLOCKS, nblk-i);
      assemble
         <HOGG_ASSEMBLE_TX, HOGG_ASSEMBLE_TY,
          HOGG_ASSEMBLE_NTX, HOGG_ASSEMBLE_NTY>
         <<<blocks, threads, 0, *stream>>>
         (&blkdata[blkoffset], cpdata, children, parents, &gpu_next_sync[0],
          &gpu_next_sync[1]);
      CudaCheckError();
   }
}

// Note: modified value lval is passed in via pointer in lndata, not as argument
void spral_ssids_load_nodes( const cudaStream_t *stream, ipc_ nblocks,
      const struct load_nodes_type *lndata, const longc_* list,
      const rpc_* mval ) {
  for ( ipc_ i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    ipc_ nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    cu_load_nodes <<< nb, 128, 0, *stream >>> ( lndata + i, list, mval );
    CudaCheckError();
  }
}

// Note: modified value lval is passed in via pointer in lndata, not as argument
void spral_ssids_load_nodes_sc( const cudaStream_t *stream, ipc_ nblocks,
      const struct load_nodes_type *lndata, const longc_* list, const ipc_* rlist,
      const rpc_* scale, const rpc_* mval ) {
  for ( ipc_ i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    ipc_ nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    cu_load_nodes_sc <<< nb, 128, 0, *stream >>> ( lndata + i, list, rlist, scale, mval );
    CudaCheckError();
  }
}

void spral_ssids_max_abs( const cudaStream_t *stream,
      ipc_ nb, longc_ n, rpc_* u, rpc_* buff, rpc_* maxabs )
{
  cudaMemsetAsync(buff, 0, nb*sizeof(rpc_), *stream);
  cudaStreamSynchronize(*stream);
  if ( n > 1024*nb )
    cu_max_abs< rpc_, 256 ><<< nb, 256, 0, *stream >>>( n, u, buff );
  else
    cu_max_abs< rpc_, 32 ><<< nb, 32, 0, *stream >>>( n, u, buff );
  CudaCheckError();
  cu_max_abs< rpc_, 1024 ><<< 1, 1024, 0, *stream >>>( nb, buff, maxabs );
  CudaCheckError();
}


} // end extern "C"
