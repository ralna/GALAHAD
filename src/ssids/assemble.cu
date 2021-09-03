#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "cuda/cuda_check.h"
#include "ssids/gpu/kernels/datatypes.h"

#define HOGG_ASSEMBLE_TX 128  // Block height
#define HOGG_ASSEMBLE_TY 8    // Block width
#define HOGG_ASSEMBLE_NTX 32  // Number of threads x
#define HOGG_ASSEMBLE_NTY 4   // Number of threads y
#define ADD_DELAYS_TX 32
#define ADD_DELAYS_TY 4

namespace /* anon */ {

struct load_nodes_type {
  long nnz;    // Number of entries to map
  int lda;    // Leading dimension of A
  int ldl;    // Leading dimension of L
  double *lcol; // Pointer to non-delay part of L
  long offn;   // Offset into nlist
  long offr;  // Offset into rlist
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
    const long *nlist,
    const double *aval
) {
   lndata += blockIdx.x;
   const long nnz = lndata->nnz;
   const int lda = lndata->lda;
   const int ldl = lndata->ldl;

   nlist += 2*lndata->offn;
   double *const lval = lndata->lcol;
  
   for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
     // Note: nlist is 1-indexed, not 0 indexed, so we have to adjust
     const int r = (nlist[2*i+1] - 1) % lda; // row index
     const int c = (nlist[2*i+1] - 1) / lda; // col index
     const long sidx = nlist[2*i+0] - 1; // source index
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
    const long *nlist,
    const int *rlist,
    const double *scale,
    const double *aval
) {
   lndata += blockIdx.x;
   const int nnz = lndata->nnz;
   const int lda = lndata->lda;
   const int ldl = lndata->ldl;

   nlist += 2*lndata->offn;
   double *const lval = lndata->lcol;
   rlist += lndata->offr;
  
   for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
      // Note: nlist and rlist are 1-indexed, not 0 indexed, so we adjust
      const int r = (nlist[2*i+1] - 1) % lda; // row index
      const int c = (nlist[2*i+1] - 1) / lda; // col index
      const long sidx = nlist[2*i+0] - 1; // source index
      const double rs = scale[rlist[r] - 1]; // row scaling
      const double cs = scale[rlist[c] - 1]; // col scaling
      lval[r + c*ldl] = rs * aval[sidx] * cs;
   }
}

// BLOCK_SIZE = blockDim.x
// maxabs must be initialized to zeros
template< typename ELEMENT_TYPE, unsigned int BLOCK_SIZE >
__global__ void
cu_max_abs( const long n, const ELEMENT_TYPE *const u, ELEMENT_TYPE *const maxabs )
{
  __shared__ volatile ELEMENT_TYPE tmax[BLOCK_SIZE];
  
  tmax[threadIdx.x] = 0.0;
  for ( long i = threadIdx.x + blockDim.x*blockIdx.x; i < n; 
        i += blockDim.x*gridDim.x ) {
    const ELEMENT_TYPE v = fabs(u[i]);
    if ( v > tmax[threadIdx.x] )
      tmax[threadIdx.x] = v;
  }
  __syncthreads();
  
  for ( int inc = 1; inc < BLOCK_SIZE; inc *= 2 ) {
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
  int pvoffset; // Offset to start of parent node values
  double *pval; // Pointer to non-delay part of parent L
  int ldp; // Leading dimension of parent

  // Child data
  int cm; // Number of rows in child
  int cn; // Number of columns in child
  int ldc; // Leading dimension of child
  long cvoffset; // Offset to start of child node values
  double *cv; // Pointer to start of child node values

  // Alignment data
  int *rlist_direct; // Pointer to start of child's rlist
  int *ind; // Pointer to start of child's contribution index

  // Sync data
  int sync_offset; // we watch sync[sync_offset]
  int sync_wait_for; // and wait for it to have value >= sync_wait_for
};

/* Following data type describes actions of single CUDA block */
struct assemble_blk_type {
  int cp; // node we're assembling into
  int blk; // block number of that node
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
template <unsigned int blk_sz_x, unsigned int blk_sz_y,
          unsigned int ntx, unsigned nty>
void __global__ assemble(
    const struct assemble_blk_type *blkdata, // block mapping
    const struct assemble_cp_type *cpdata, // child-parent data
    const double *const children, // pointer to array containing children
    double *const parents, // pointer to array containing parents
    unsigned int *const next_blk, // gmem location used to determine next block
    volatile unsigned int *const sync // sync[cp] is #blocks completed so far for cp
) {
   // Get block number
   __shared__ volatile unsigned int mynext_blk;
   if(threadIdx.x==0 && threadIdx.y==0)
      mynext_blk = atomicAdd(next_blk, 1);
   __syncthreads();

   // Determine global information
   blkdata += mynext_blk;
   cpdata += blkdata->cp;
   int blk = blkdata->blk;
   int nx = (cpdata->cm-1) / blk_sz_x + 1; // number of blocks high child is
   int bx = blk % nx; // coordinate of block in x direction
   int by = blk / nx; // coordinate of block in y direction
   int ldc = cpdata->ldc;
   int ldp = cpdata->ldp;

   // Initialize local information
   int m = min(blk_sz_x, cpdata->cm - bx*blk_sz_x);
   int n = min(blk_sz_y, cpdata->cn - by*blk_sz_y);
   const double *src = 
      cpdata->cv + ldc*by*blk_sz_y + bx*blk_sz_x;
   double *dest = cpdata->pval;
   int *rows = cpdata->rlist_direct + bx*blk_sz_x;
   int *cols = cpdata->rlist_direct + by*blk_sz_y;

   // Wait for previous child of this parent to complete
   if(threadIdx.x==0 && threadIdx.y==0) {
      while(sync[cpdata->sync_offset] < cpdata->sync_wait_for) /**/;
   }
   __syncthreads();

   // Perform assembly
   for(int j=0; j<blk_sz_y/nty; j++) {
      if( threadIdx.y+j*nty < n ) {
         int col = cols[threadIdx.y+j*nty]-1;
         for(int i=0; i<blk_sz_x/ntx; i++) {
            if( threadIdx.x+i*ntx < m ) {
               int row = rows[threadIdx.x+i*ntx]-1;
               dest[row + col*ldp] += 
                  src[threadIdx.x+i*ntx + (threadIdx.y+j*nty)*ldc];
            }
         }
      }
   }

   // Record that we're done
   __syncthreads();
   if(threadIdx.x==0 && threadIdx.y==0) {
      atomicAdd((int*)&(sync[blkdata->cp]), 1);
   }
}

struct assemble_delay_type {
  int dskip; // Number of rows to skip for delays from later children
  int m; // Number of rows in child to copy
  int n; // Number of cols in child to copy
  int ldd; // Leading dimension of dest (parent)
  int lds; // Leading dimension of src (child)
  double *dval; // Pointer to dest (parent)
  double *sval; // Pointer to src (child)
  long roffset; // Offset to rlist_direct
};

/* Copies delays from child to parent using one block per parent 
 * Note: src and dest pointers both contained in dinfo
 */
void __global__ add_delays(
    struct assemble_delay_type *dinfo, // information on each block
    const int *rlist_direct // children's rows indices in parents
) {
   dinfo += blockIdx.x;
   const int dskip = dinfo->dskip; // number of delays
   const int m = dinfo->m; // number of rows
   const int n = dinfo->n; // number of cols
   const int ldd = dinfo->ldd; // leading dimension of dest
   const int lds = dinfo->lds; // leading dimension of src

   double *const dest = dinfo->dval;
   const double *const src = dinfo->sval;
   rlist_direct += dinfo->roffset;

   for ( int y = threadIdx.y; y < n; y += blockDim.y ) {
      for ( int x = threadIdx.x; x < m; x += blockDim.x ) {
         if ( x < n ) {
            dest[x + y*ldd] = src[x + y*lds];
         }
         else {
            int xt = dskip + rlist_direct[x - n] - 1;
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
void spral_ssids_add_delays( const cudaStream_t *stream, int ndblk,
      struct assemble_delay_type *gpu_dinfo, int *rlist_direct ) {
   if ( ndblk == 0 ) return; // Nothing to see here
   dim3 threads(ADD_DELAYS_TX, ADD_DELAYS_TY);
   for ( int i = 0; i < ndblk; i += MAX_CUDA_BLOCKS ) {
      int nb = min(MAX_CUDA_BLOCKS, ndblk - i);
      add_delays
         <<< nb, threads, 0, *stream >>>
         ( gpu_dinfo + i, rlist_direct );
      CudaCheckError();
   }
}

/* Runs the kernel assemble<<<>>>() after setting up memory correctly. */
/* Requires gpu_next_sync[] to be of size >= (1+ncp)*sizeof(unsigned int) */
void spral_ssids_assemble(const cudaStream_t *stream, int nblk, int blkoffset,
      struct assemble_blk_type *blkdata, int ncp,
      struct assemble_cp_type *cpdata, double *children,
      double *parents, unsigned int *gpu_next_sync) {
   /* Create and initialize synchronization objects using a single call:
      next_blk[1]
      sync[ncp]
    */
   CudaSafeCall(
         cudaMemsetAsync(gpu_next_sync,0,(1+ncp)*sizeof(unsigned int),*stream)
         );
   /* Note, that we can only have at most 65535 blocks per dimn.
    * For some problems, nblk can exceed this, so we use more than one launch.
    * As the next block we look at is specified by next_blk this works fine.
    */
   dim3 threads(HOGG_ASSEMBLE_NTX, HOGG_ASSEMBLE_NTY);
   for(int i=0; i<nblk; i+=MAX_CUDA_BLOCKS) {
      int blocks = min(MAX_CUDA_BLOCKS, nblk-i);
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
void spral_ssids_load_nodes( const cudaStream_t *stream, int nblocks,
      const struct load_nodes_type *lndata, const long* list,
      const double* mval ) {
  for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    cu_load_nodes <<< nb, 128, 0, *stream >>> ( lndata + i, list, mval );
    CudaCheckError();
  }
}

// Note: modified value lval is passed in via pointer in lndata, not as argument
void spral_ssids_load_nodes_sc( const cudaStream_t *stream, int nblocks,
      const struct load_nodes_type *lndata, const long* list, const int* rlist,
      const double* scale, const double* mval ) {
  for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    cu_load_nodes_sc <<< nb, 128, 0, *stream >>> ( lndata + i, list, rlist, scale, mval );
    CudaCheckError();
  }
}

void spral_ssids_max_abs( const cudaStream_t *stream, 
      int nb, long n, double* u, double* buff, double* maxabs )
{
  cudaMemsetAsync(buff, 0, nb*sizeof(double), *stream);
  cudaStreamSynchronize(*stream);
  if ( n > 1024*nb )
    cu_max_abs< double, 256 ><<< nb, 256, 0, *stream >>>( n, u, buff );
  else
    cu_max_abs< double, 32 ><<< nb, 32, 0, *stream >>>( n, u, buff );
  CudaCheckError();
  cu_max_abs< double, 1024 ><<< 1, 1024, 0, *stream >>>( nb, buff, maxabs );
  CudaCheckError();
}


} // end extern "C"
