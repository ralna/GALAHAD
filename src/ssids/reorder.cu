/* Copyright (c) 2013 Science and Technology Facilities Council (STFC)
 * Licence: BSD licence, see LICENCE file for details
 * Author: Jonathan Hogg
 * This version: GALAHAD 5.0 - 2024-06-11 AT 09:50 GMT
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
#include "ssids_gpu_kernels_datatypes.h"
#include "spral_cuda_cuda_check.h"

#ifdef REAL_32
#define multiswap_type multiswap_type_single
#define multireorder_data multireorder_data_single
#define multisymm_type multisymm_type_single
#define cu_copy_mc cu_copy_mc_single
#define cu_copy_ic cu_copy_ic_single
#define cu_swap_ni2D_ic cu_swap_ni2D_ic_single
#define cu_swap_ni2D_ir cu_swap_ni2D_ir_single
#define cu_multiswap_ni2D_c cu_multiswap_ni2D_c_single
#define cu_multiswap_ni2D_r cu_multiswap_ni2D_r_single
#define cu_reorder_rows cu_reorder_rows_single
#define cu_reorder_cols2 cu_reorder_cols2_single
#define cu_reorder_rows2 cu_reorder_rows2_single
#define copy_L_LD_no_perm copy_L_LD_no_perm_single
#define shuffle_perm_shmem shuffle_perm_shmem_single
#define copy_L_LD_perm_shmem copy_L_LD_perm_shmem_single
#define copy_L_LD_perm_noshmem copy_L_LD_perm_noshmem_single
#define cu_multireorder cu_multireorder_single
#define cu_multicopy cu_multicopy_single
#define cu_multisymm cu_multisymm_single
#define spral_ssids_copy_ic spral_ssids_copy_ic_single
#define spral_ssids_copy_mc spral_ssids_copy_mc_single
#define spral_ssids_multisymm spral_ssids_multisymm_single
#define spral_ssids_multicopy spral_ssids_multicopy_single
#define spral_ssids_multireorder spral_ssids_multireorder_single
#define spral_ssids_reorder_cols2 spral_ssids_reorder_cols2_single
#define spral_ssids_reorder_rows spral_ssids_reorder_rows_single
#define spral_ssids_reorder_rows2 spral_ssids_reorder_rows2_single
#define spral_ssids_swap_ni2Dm spral_ssids_swap_ni2Dm_single
#define spral_ssids_swap_ni2D_ic spral_ssids_swap_ni2D_ic_single
#define spral_ssids_swap_ni2D_ir spral_ssids_swap_ni2D_ir_single
#elif REAL_128
#define multiswap_type multiswap_type_quadruple
#define multireorder_data multireorder_data_quadruple
#define multisymm_type multisymm_type_quadruple
#define cu_copy_mc cu_copy_mc_quadruple
#define cu_copy_ic cu_copy_ic_quadruple
#define cu_swap_ni2D_ic cu_swap_ni2D_ic_quadruple
#define cu_swap_ni2D_ir cu_swap_ni2D_ir_quadruple
#define cu_multiswap_ni2D_c cu_multiswap_ni2D_c_quadruple
#define cu_multiswap_ni2D_r cu_multiswap_ni2D_r_quadruple
#define cu_reorder_rows cu_reorder_rows_quadruple
#define cu_reorder_cols2 cu_reorder_cols2_quadruple
#define cu_reorder_rows2 cu_reorder_rows2_quadruple
#define copy_L_LD_no_perm copy_L_LD_no_perm_quadruple
#define shuffle_perm_shmem shuffle_perm_shmem_quadruple
#define copy_L_LD_perm_shmem copy_L_LD_perm_shmem_quadruple
#define copy_L_LD_perm_noshmem copy_L_LD_perm_noshmem_quadruple
#define cu_multireorder cu_multireorder_quadruple
#define cu_multicopy cu_multicopy_quadruple
#define cu_multisymm cu_multisymm_quadruple
#define spral_ssids_copy_ic spral_ssids_copy_ic_quadruple
#define spral_ssids_copy_mc spral_ssids_copy_mc_quadruple
#define spral_ssids_multisymm spral_ssids_multisymm_quadruple
#define spral_ssids_multicopy spral_ssids_multicopy_quadruple
#define spral_ssids_multireorder spral_ssids_multireorder_quadruple
#define spral_ssids_reorder_cols2 spral_ssids_reorder_cols2_quadruple
#define spral_ssids_reorder_rows spral_ssids_reorder_rows_quadruple
#define spral_ssids_reorder_rows2 spral_ssids_reorder_rows2_quadruple
#define spral_ssids_swap_ni2Dm spral_ssids_swap_ni2Dm_quadruple
#define spral_ssids_swap_ni2D_ic spral_ssids_swap_ni2D_ic_quadruple
#define spral_ssids_swap_ni2D_ir spral_ssids_swap_ni2D_ir_quadruple
#else
#define multiswap_type multiswap_type_double
#define multireorder_data multireorder_data_double
#define multisymm_type multisymm_type_double
#define cu_copy_mc cu_copy_mc_double
#define cu_copy_ic cu_copy_ic_double
#define cu_swap_ni2D_ic cu_swap_ni2D_ic_double
#define cu_swap_ni2D_ir cu_swap_ni2D_ir_double
#define cu_multiswap_ni2D_c cu_multiswap_ni2D_c_double
#define cu_multiswap_ni2D_r cu_multiswap_ni2D_r_double
#define cu_reorder_rows cu_reorder_rows_double
#define cu_reorder_cols2 cu_reorder_cols2_double
#define cu_reorder_rows2 cu_reorder_rows2_double
#define copy_L_LD_no_perm copy_L_LD_no_perm_double
#define shuffle_perm_shmem shuffle_perm_shmem_double
#define copy_L_LD_perm_shmem copy_L_LD_perm_shmem_double
#define copy_L_LD_perm_noshmem copy_L_LD_perm_noshmem_double
#define cu_multireorder cu_multireorder_double
#define cu_multicopy cu_multicopy_double
#define cu_multisymm cu_multisymm_double
#define spral_ssids_copy_ic spral_ssids_copy_ic_double
#define spral_ssids_copy_mc spral_ssids_copy_mc_double
#define spral_ssids_multisymm spral_ssids_multisymm_double
#define spral_ssids_multicopy spral_ssids_multicopy_double
#define spral_ssids_multireorder spral_ssids_multireorder_double
#define spral_ssids_reorder_cols2 spral_ssids_reorder_cols2_double
#define spral_ssids_reorder_rows spral_ssids_reorder_rows_double
#define spral_ssids_reorder_rows2 spral_ssids_reorder_rows2_double
#define spral_ssids_swap_ni2Dm spral_ssids_swap_ni2Dm_double
#define spral_ssids_swap_ni2D_ic spral_ssids_swap_ni2D_ic_double
#define spral_ssids_swap_ni2D_ir spral_ssids_swap_ni2D_ir_double
#endif

#define min(x,y) ((x) < (y) ? (x) : (y))

#define BLOCK_SIZE 8
#define MAX_CUDA_BLOCKS 65535

//#define SM_3X (__CUDA_ARCH__ == 300 || __CUDA_ARCH__ == 350 || __CUDA_ARCH__ == 370)
//FIXME: Verify if the code for Keplers (sm_3x) is still correct for the later GPUs.
#define SM_3X (__CUDA_ARCH__ >= 300)

using namespace spral::ssids::gpu;

namespace /* anon */ {

template< typename ELEMENT_TYPE >
__global__ void
cu_copy_mc( ipc_ nrows, ipc_ ncols,
            ELEMENT_TYPE* a, ipc_ lda,
            ELEMENT_TYPE* b, ipc_ ldb,
            ipc_* mask )
{
  ipc_ i = threadIdx.x + blockDim.x*blockIdx.x;
  ipc_ j = threadIdx.y + blockDim.y*blockIdx.y;
  if ( i < nrows && j < ncols && mask[j] > 0 )
    b[i + ldb*j] = a[i + lda*j];
}

template< typename ELEMENT_TYPE >
__global__ void
cu_copy_ic( ipc_ nrows, ipc_ ncols,
            ELEMENT_TYPE* a, ipc_ lda,
            ELEMENT_TYPE* b, ipc_ ldb,
            ipc_* ind )
{
  ipc_ i = threadIdx.x + blockDim.x*blockIdx.x;
  ipc_ j = threadIdx.y + blockDim.y*blockIdx.y;
  if ( i < nrows && j < ncols && ind[j] > 0 )
    b[i + ldb*(ind[j] - 1)] = a[i + lda*j];
}

template< typename ELEMENT_TYPE >
__global__ void
cu_swap_ni2D_ic( ipc_ nrows, ipc_ ncols,
                 ELEMENT_TYPE* a, ipc_ lda,
                 ELEMENT_TYPE* b, ipc_ ldb,
                 ipc_* index )
// swaps columns of non-intersecting 2D arrays a(1:n,index(1:m)) and b(1:n,1:m)
// index is one-based
{
  ipc_ i = threadIdx.x + blockDim.x*blockIdx.x;
  ipc_ j = threadIdx.y + blockDim.y*blockIdx.y;
  ipc_ k;
  rpc_ s;

  if ( i < nrows && j < ncols && (k = index[j] - 1) > -1 ) {
    s = a[i + lda*k];
    a[i + lda*k] = b[i + ldb*j];
    b[i + ldb*j] = s;
  }
}

template< typename ELEMENT_TYPE >
__global__ void
cu_swap_ni2D_ir( ipc_ nrows, ipc_ ncols,
                 ELEMENT_TYPE* a, ipc_ lda,
                 ELEMENT_TYPE* b, ipc_ ldb,
                 ipc_* index )
// swaps rows of non-intersecting 2D arrays a(index(1:n),1:m) and b(1:n,1:m)
// index is one-based
{
  ipc_ i = threadIdx.x + blockDim.x*blockIdx.x;
  ipc_ j = threadIdx.y + blockDim.y*blockIdx.y;
  ipc_ k;
  rpc_ s;

  if ( i < nrows && j < ncols && (k = index[i] - 1) > -1 ) {
    s = a[k + lda*j];
    a[k + lda*j] = b[i + ldb*j];
    b[i + ldb*j] = s;
  }
}

struct multiswap_type {
   ipc_ nrows;
   ipc_ ncols;
   ipc_ k;
   rpc_ *lcol;
   ipc_ lda;
   ipc_ off;
};

template< typename ELEMENT_TYPE >
__global__ void
cu_multiswap_ni2D_c( struct multiswap_type *swapdata )
// swaps non-intersecting rows or cols of a 2D multiarray a
{
  swapdata += blockIdx.x;
  ipc_ nrows = swapdata->nrows;
  if ( blockIdx.y*blockDim.x >= nrows )
    return;

  ipc_ k     = swapdata->k;
  ELEMENT_TYPE *a = swapdata->lcol;
  ipc_ lda   = swapdata->lda;
  ipc_ off  = lda*swapdata->off;
  ELEMENT_TYPE s;

  for ( ipc_ i = threadIdx.x + blockIdx.y*blockDim.x; i < nrows;
        i += blockDim.x*gridDim.y )
    for ( ipc_ j = threadIdx.y; j < k; j += blockDim.y ) {
      s = a[i + lda*j];
      a[i + lda*j] = a[off + i + lda*j];
      a[off + i + lda*j] = s;
    }
}

template< typename ELEMENT_TYPE >
__global__ void
cu_multiswap_ni2D_r( struct multiswap_type *swapdata )
// swaps non-intersecting rows or cols of a 2D multiarray a
{
  swapdata += blockIdx.x;
  ipc_ ncols = swapdata->ncols;
  if ( blockIdx.y*blockDim.y >= ncols )
    return;

  ipc_ k     = swapdata->k;
  ELEMENT_TYPE *a = swapdata->lcol;
  ipc_ lda   = swapdata->lda;
  ipc_ off  = swapdata->off;
  ELEMENT_TYPE s;

  for ( ipc_ i = threadIdx.x; i < k; i += blockDim.x )
    for ( ipc_ j = threadIdx.y + blockIdx.y*blockDim.y; j < ncols;
          j += blockDim.y*gridDim.y ) {
      s = a[i + lda*j];
      a[i + lda*j] = a[off + i + lda*j];
      a[off + i + lda*j] = s;
    }
}

template< typename ELEMENT_TYPE >
__global__ void
cu_reorder_rows(
                ipc_ nrows, ipc_ ncols,
                ELEMENT_TYPE* a, ipc_ lda,
                ELEMENT_TYPE* b, ipc_ ldb,
                ipc_* index
               )
{
  ipc_ x;
  ipc_ y = threadIdx.y + blockIdx.y*blockDim.y;

  for ( x = threadIdx.x; x < nrows; x += blockDim.x )
    if ( y < ncols )
      b[index[x] - 1 + ldb*y] = a[x + lda*y];
  __syncthreads();
  for ( x = threadIdx.x; x < nrows; x += blockDim.x )
    if ( y < ncols )
      a[x + lda*y] = b[x + ldb*y];
}

template< typename ELEMENT_TYPE, uipc_ SIZE_X, uipc_ SIZE_Y >
__global__ void
cu_reorder_cols2( ipc_ nrows, ipc_ ncols,
                  ELEMENT_TYPE* a, ipc_ lda,
                  ELEMENT_TYPE* b, ipc_ ldb,
                  ipc_* index, ipc_ mode )
{
  ipc_ ix = threadIdx.x + blockIdx.x*blockDim.x;

  __shared__ volatile ELEMENT_TYPE work[SIZE_X*SIZE_Y];

  if ( blockIdx.y  ) {
    if ( mode > 0 ) {
      if ( ix < nrows && threadIdx.y < ncols )
        work[threadIdx.x + (index[threadIdx.y] - 1)*SIZE_X]
          = a[ix + lda*threadIdx.y];
    }
    else {
      if ( ix < nrows && threadIdx.y < ncols )
        work[threadIdx.x + threadIdx.y*SIZE_X]
          = a[ix + lda*(index[threadIdx.y] - 1)];
    }
    __syncthreads();
    if ( ix < nrows && threadIdx.y < ncols )
      a[ix + lda*threadIdx.y] = work[threadIdx.x + threadIdx.y*SIZE_X];
  }
  else {
    if ( mode > 0 ) {
      if ( ix < nrows && threadIdx.y < ncols )
        work[threadIdx.x + (index[threadIdx.y] - 1)*SIZE_X]
          = b[ix + ldb*threadIdx.y];
    }
    else {
      if ( ix < nrows && threadIdx.y < ncols )
        work[threadIdx.x + threadIdx.y*SIZE_X]
          = b[ix + ldb*(index[threadIdx.y] - 1)];
    }
    __syncthreads();
    if ( ix < nrows && threadIdx.y < ncols )
      b[ix + ldb*threadIdx.y] = work[threadIdx.x + threadIdx.y*SIZE_X];
  }
}

template< typename ELEMENT_TYPE, uipc_ SIZE_X, uipc_ SIZE_Y >
__global__ void
cu_reorder_rows2( ipc_ nrows, ipc_ ncols,
                  ELEMENT_TYPE* a, ipc_ lda,
                  ELEMENT_TYPE* b, ipc_ ldb,
                  ipc_* index, ipc_ mode )
{
  ipc_ iy = threadIdx.y + blockIdx.x*blockDim.y;

  __shared__ volatile ELEMENT_TYPE work[SIZE_X*SIZE_Y];

  if ( blockIdx.y ) {
    if ( mode > 0 ) {
      if ( threadIdx.x < nrows && iy < ncols )
        work[index[threadIdx.x] - 1 + threadIdx.y*SIZE_X]
          = a[threadIdx.x + lda*iy];
    }
    else {
      if ( threadIdx.x < nrows && iy < ncols )
        work[threadIdx.x + threadIdx.y*SIZE_X]
          = a[index[threadIdx.x] - 1 + lda*iy];
    }
    __syncthreads();
    if ( threadIdx.x < nrows && iy < ncols )
      a[threadIdx.x + lda*iy] = work[threadIdx.x + threadIdx.y*SIZE_X];
  }
  else {
    if ( mode > 0 ) {
      if ( threadIdx.x < nrows && iy < ncols )
        work[index[threadIdx.x] - 1 + threadIdx.y*SIZE_X]
          = b[threadIdx.x + ldb*iy];
    }
    else {
      if ( threadIdx.x < nrows && iy < ncols )
        work[threadIdx.x + threadIdx.y*SIZE_X]
          = b[index[threadIdx.x] - 1 + ldb*iy];
    }
    __syncthreads();
    if ( threadIdx.x < nrows && iy < ncols )
      b[threadIdx.x + ldb*iy] = work[threadIdx.x + threadIdx.y*SIZE_X];
  }
}

/*
 * Copies new L factors back to A array without any permutation
 */
template< typename ELEMENT_TYPE, ipc_ NTX >
__device__ void
__forceinline__ // Required to avoid errors about reg counts compiling with -G
copy_L_LD_no_perm(
      ipc_ nblk, ipc_ bidx, ipc_ tid,
      ipc_ nrows, ipc_ ncols,
      ELEMENT_TYPE *dest, ipc_ ldd,
      const ELEMENT_TYPE *src, ipc_ lds
) {
   ipc_ tx = tid % NTX;
   ipc_ ty = tid / NTX;
   src += NTX*bidx;
   dest += NTX*bidx;
   nrows -= NTX*bidx;
   if ( ty < ncols ) {
      for ( ipc_ x = tx; x < nrows; x += NTX*nblk )
         dest[x + ldd*ty] = src[x + lds*ty];
   }
}

/* Shuffles the permutation vector using shared memory
   [in case it overlaps itself] */
template < ipc_ SIZE_X >
__device__ void
shuffle_perm_shmem( ipc_ n, volatile const ipc_ *const indr, ipc_ *perm ) {
   // Update permutation
   __shared__ volatile ipc_ iwork[SIZE_X];
   if ( threadIdx.x < n && threadIdx.y == 0 )
      iwork[indr[threadIdx.x] - 1] = perm[threadIdx.x];
   __syncthreads();
   if ( threadIdx.x < n && threadIdx.y == 0 )
      perm[threadIdx.x] = iwork[threadIdx.x];
}

/*
 * Copies new L factors back to A array and applies permutation to rows and cols
 * This version uses shared memory and is designed for the case when the new
 * and old location of columns and rows overlap.
 */
template< typename ELEMENT_TYPE, uipc_ SIZE_X, uipc_ SIZE_Y >
__device__ void
__forceinline__ // Required to avoid errors about reg counts compiling with -G
copy_L_LD_perm_shmem(
      ipc_ block, ipc_ nblocks,
      ipc_ done, ipc_ pivoted, ipc_ delayed,
      ipc_ nrows, ipc_ ncols,
      ipc_ ib, ipc_ jb,
      ipc_ offc, ipc_ offp,
      ipc_ ld,
      volatile ipc_ *const indr,
      rpc_ *a, rpc_ *b, const rpc_ *c,
      ipc_ *perm
) {
   __shared__ volatile ELEMENT_TYPE work1[SIZE_X*SIZE_Y];
   __shared__ volatile ELEMENT_TYPE work2[SIZE_X*SIZE_Y];
#if (!SM_3X)
   __shared__ volatile ELEMENT_TYPE work3[SIZE_X*SIZE_Y];
   __shared__ volatile ELEMENT_TYPE work4[SIZE_X*SIZE_Y];
#endif

   // Extend permutation array to cover non-pivoted columns
   if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
      ipc_ i = 0;
      ipc_ j = pivoted;
      for ( ; i < delayed; i++ )
         indr[i] = ++j;
      for ( ; i < delayed + jb - ib + 1; i++ )
         if ( !indr[i] )
            indr[i] = ++j;
   }

   ipc_ off = done*ld;

   // We handle the (done-jb) x (done-jb) block that requires both
   // row and column permutations seperately using the first block.
   // All remaining rows and columns are handlded by the remaining blocks.
   // Note that while we do not need to perumute "above" the pivoted columns,
   // we do need to permute to the "left" of the pivoted rows!
   if ( block ) {
      // Swap columns of A and copy in L, but avoiding rows that need
      // permuted
      // Also, swap cols of LD but avoiding rows that need permuted
      ipc_ baseStep = blockDim.x*(nblocks - 1);
#if (SM_3X)
      for ( ipc_ i = jb + blockDim.x*(block - 1); i < nrows;
            i += baseStep ) {
#else
      for ( ipc_ i = jb + blockDim.x*(block - 1); i < nrows + baseStep;
            i += baseStep * 2 ) {
#endif
         ipc_ ix = i + threadIdx.x;
#if (!SM_3X)
         ipc_ ix2 = ix + baseStep;
#endif
         __syncthreads();

         if (threadIdx.y < jb - done) {
#if (!SM_3X)
           if ( ix2 < nrows ) {
             if ( indr[threadIdx.y] > pivoted ) {
               work1[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                 = a[off + ix + ld*threadIdx.y];
               work3[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                 = a[off + ix2 + ld*threadIdx.y];
             }
             else {
               work1[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                 = c[offc + ix + ld*(threadIdx.y - delayed)];
               work3[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                 = c[offc + ix2 + ld*(threadIdx.y - delayed)];
             }
             work2[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
               = b[off + ix + ld*threadIdx.y];
             work4[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
               = b[off + ix2 + ld*threadIdx.y];
           }
           else
#endif
             if ( ix < nrows ) {
               if ( indr[threadIdx.y] > pivoted )
                 work1[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                   = a[off + ix + ld*threadIdx.y];
               else
                 work1[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                   = c[offc + ix + ld*(threadIdx.y - delayed)];

               work2[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X]
                 = b[off + ix + ld*threadIdx.y];
            }
         }

         __syncthreads();

         if (threadIdx.y < jb - done) {
#if (!SM_3X)
           if ( ix2 < nrows) {
             a[off + ix + ld*threadIdx.y] = work1[threadIdx.x + threadIdx.y*SIZE_X];
             a[off + ix2 + ld*threadIdx.y] = work3[threadIdx.x + threadIdx.y*SIZE_X];
             b[off + ix + ld*threadIdx.y] = work2[threadIdx.x + threadIdx.y*SIZE_X];
             b[off + ix2 + ld*threadIdx.y] = work4[threadIdx.x + threadIdx.y*SIZE_X];
           }
           else
#endif
             if ( ix < nrows) {
               a[off + ix + ld*threadIdx.y] = work1[threadIdx.x + threadIdx.y*SIZE_X];
               b[off + ix + ld*threadIdx.y] = work2[threadIdx.x + threadIdx.y*SIZE_X];
             }
         }
      }

      if ( (block - 1)*blockDim.y >= ncols )
         return; // Block not needed for y direction (Note that n <= m always)

      off -= done*ld;
      off += done;

      // Swap rows of A
      baseStep = blockDim.y*(nblocks - 1);
#if (SM_3X)
      for ( ipc_ i = blockDim.y*(block - 1); i < ncols;
            i += baseStep  ) {
#else
      for ( ipc_ i = blockDim.y*(block - 1); i < ncols + baseStep;
            i += baseStep * 2 ) {
#endif
         ipc_ iy = i + threadIdx.y;
#if (!SM_3X)
         ipc_ iy2 = iy + baseStep;
#endif
         __syncthreads();

         if ( !(iy >= done && iy < jb) &&
                iy < ncols && threadIdx.x < jb - done ) {
            work1[indr[threadIdx.x] - 1 + threadIdx.y*SIZE_X] =
                a[off + threadIdx.x + ld*iy];
            work2[indr[threadIdx.x] - 1 + threadIdx.y*SIZE_X] =
                b[off + threadIdx.x + ld*iy];
         }
#if (!SM_3X)
         if ( !(iy2 >= done && iy2 < jb) &&
                iy2 < ncols && threadIdx.x < jb - done ) {
            work3[indr[threadIdx.x] - 1 + threadIdx.y*SIZE_X] =
                a[off + threadIdx.x + ld*iy2];
            work4[indr[threadIdx.x] - 1 + threadIdx.y*SIZE_X] =
                b[off + threadIdx.x + ld*iy2];
         }
#endif
         __syncthreads();

         if ( !(iy >= done && iy < jb) &&
               iy < ncols && threadIdx.x < jb - done ) {
            a[off + threadIdx.x + ld*iy] =
               work1[threadIdx.x + threadIdx.y*SIZE_X];
            b[off + threadIdx.x + ld*iy] =
            work2[threadIdx.x + threadIdx.y*SIZE_X];
         }
#if (!SM_3X)
         if ( !(iy2 >= done && iy2 < jb) &&
               iy2 < ncols && threadIdx.x < jb - done ) {
            a[off + threadIdx.x + ld*iy2] =
               work3[threadIdx.x + threadIdx.y*SIZE_X];
            b[off + threadIdx.x + ld*iy2] =
            work4[threadIdx.x + threadIdx.y*SIZE_X];
         }
#endif
      }
   } else {
      // Handle (jb-done) x (jb-done) block that needs both
      // row /and/ column permutations.
      shuffle_perm_shmem< SIZE_X > ( delayed + jb - ib + 1, indr, &perm[offp + done] );

      ipc_ pass = threadIdx.x < jb - done && threadIdx.y < jb - done;

      // Handle L and LD
      if ( pass ) {
        // Column permtuations + copy from c[]
        if ( indr[threadIdx.y] > pivoted )
          work1[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X] =
            a[off + done + threadIdx.x + ld*threadIdx.y];
        else
          work1[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X] =
            c[offc + done + threadIdx.x + ld*(threadIdx.y - delayed)];
        work2[threadIdx.x + (indr[threadIdx.y] - 1)*SIZE_X] =
          b[off + done + threadIdx.x + ld*threadIdx.y];
      }

      __syncthreads();

      // Row permutations
      if ( pass ) {
        a[off + done + threadIdx.x + ld*threadIdx.y] =
          work1[threadIdx.x + threadIdx.y*SIZE_X];
        b[off + done + threadIdx.x + ld*threadIdx.y] =
          work2[threadIdx.x + threadIdx.y*SIZE_X];

         off -= done*nrows;
         off += done;
      }

      __syncthreads();

      if ( pass ) {
        work1[indr[threadIdx.x] - 1 + threadIdx.y*SIZE_X] =
          a[off + threadIdx.x + ld*(done + threadIdx.y)];
        work2[indr[threadIdx.x] - 1 + threadIdx.y*SIZE_X] =
          b[off + threadIdx.x + ld*(done + threadIdx.y)];
      }

      __syncthreads();

      if ( pass ) {
        a[off + threadIdx.x + ld*(done + threadIdx.y)] =
          work1[threadIdx.x + threadIdx.y*SIZE_X];
        b[off + threadIdx.x + ld*(done + threadIdx.y)] =
          work2[threadIdx.x + threadIdx.y*SIZE_X];
      }
   }
}

/*
 * Copies new L factors back to A array and applies permutation to rows and cols
 * This version does this directly in global memory and is designed for the case
 * when the new and old location of columns and rows DO NOT overlap.
 */
template< typename ELEMENT_TYPE, uipc_ SIZE_X, uipc_ SIZE_Y >
__device__ void
__forceinline__ // Required to avoid errors about reg counts compiling with -G
copy_L_LD_perm_noshmem(
      ipc_ node,
      ipc_ block, ipc_ nblocks,
      ipc_ done, ipc_ pivoted, ipc_ delayed,
      ipc_ nrows, ipc_ ncols,
      ipc_ ib, ipc_ jb,
      ipc_ offc, ipc_ offp,
      ipc_ ld,
      const ipc_ *ind,
      const volatile ipc_ *const indf,
      rpc_ *a, rpc_ *b, const rpc_ *c,
      ipc_ *perm
) {

   ipc_ off1 = done;
   ipc_ off2 = ib - 1;
   ipc_ offi = node*SIZE_Y/2;

   // We handle the two pivoted x pivoted blocks where row and columns cross
   // over seperately using the first block.
   // The other blocks just exclude these rows/cols as appropriate
   // All remaining rows and columns are handlded by the remaining blocks.
   if ( block ) {
      // Handle parts of matrix that require EITHER row OR col shuffle
      ipc_ tx = (threadIdx.y < SIZE_Y/2) ? threadIdx.x : threadIdx.x + blockDim.x;
      ipc_ ty = (threadIdx.y < SIZE_Y/2) ? threadIdx.y : threadIdx.y - SIZE_Y/2;
      // Swap a[:,done:done+pivoted] and a[:,ib:jb] pulling in c[] as we go
      for ( ipc_ x = tx + 2*blockDim.x*(block - 1);
            x < nrows && ty < jb - ib + 1;
            x += 2*blockDim.x*(nblocks - 1) ) {
         ipc_ y = ind[offi + ty] - 1;
         if ( (x >= done   && x < done + jb - ib + 1)
               || (x >= ib - 1 && x < jb) || y < 0 )
            continue; // handled separately
         a[x + ld*(off2 + ty)] = a[x + ld*(off1 + y)];
         a[x + ld*(off1 + y)] = c[offc + x + ld*ty];
      }
      // Swap b[:,done:done+pivoted] and b[:,ib:jb]
      for ( ipc_ x = tx + 2*blockDim.x*(block - 1);
            x < nrows && ty < jb - ib + 1;
            x += 2*blockDim.x*(nblocks - 1) ) {
         ipc_ y = ind[offi + ty] - 1;
         if ( ( x >= done && x < done + jb - ib + 1 )
               || ( x >= ib - 1 && x < jb ) || y < 0)
            continue; // handled separately
         ELEMENT_TYPE s = b[x + ld*(off1 + y)];
         b[x + ld*(off1 + y)] =
         b[x + ld*(off2 + ty)];
         b[x + ld*(off2 + ty)] = s;
      }
      if ( (block - 1)*blockDim.y >= ncols )
         return;
      // swap a[done:done+pivoted,:] and a[ib:jb,:]
      for ( ipc_ y = threadIdx.y + blockDim.y*(block - 1);
            y < ncols && threadIdx.x < jb - ib + 1;
            y += blockDim.y*(nblocks - 1) ) {
         ipc_ x = ind[offi + threadIdx.x] - 1;
         if ( (y >= done && y < done + jb - ib + 1)
               || (y >= ib - 1 && y < jb) || x < 0 )
            continue; // handled separately
         ELEMENT_TYPE s = a[off1 + x + ld*y];
         a[off1 + x + ld*y] = a[off2 + threadIdx.x + ld*y];
         a[off2 + threadIdx.x + ld*y] = s;
      }
      // swap b[done:done+pivoted,:] and b[ib:jb,:]
      for ( ipc_ y = threadIdx.y + blockDim.y*(block - 1);
            y < ncols && threadIdx.x < jb - ib + 1;
            y += blockDim.y*(nblocks - 1) ) {
         ipc_ x = ind[offi + threadIdx.x] - 1;
         if ( (y >= done   && y < done + jb - ib + 1)
               || (y >= ib - 1 && y < jb) || x < 0)
            continue; // handled separately
         ELEMENT_TYPE s = b[off1 + x + ld*y];
         b[off1 + x + ld*y] = b[off2 + threadIdx.x + ld*y];
         b[off2 + threadIdx.x + ld*y] = s;
      }
   }
   else {
      // Handle part of matrix that requires BOTH row AND col shuffle
      if ( threadIdx.x < jb - ib + 1 && threadIdx.y == 0 ) {
         // Update permutation
         ipc_ i = indf[threadIdx.x] - 1;
         if ( i >= 0 ) {
            ipc_ s = perm[offp + ib - 1 + threadIdx.x];
            perm[offp + ib - 1 + threadIdx.x] = perm[offp + done + i];
            perm[offp + done + i] = s;
         }
      }

      // Swaps with L
      // FIXME: This might be sped up by doing 1.5 swaps instead of 3.5.
      // Swap a[done:done+pivoted,done:done+pivoted] and
      // a[done:done+pivoted,ib:jb]
      // pulling in new cols from c[] as we go.
      ipc_ x = done + threadIdx.x;
      ipc_ y = ind[offi + threadIdx.y] - 1;
      if ( x < done + jb - ib + 1 && threadIdx.y < jb - ib + 1 && y >= 0 ) {
         a[x + ld*(off2 + threadIdx.y)] = a[x + ld*(off1 + y)];
         a[x + ld*(off1 + y)] = c[offc + x + ld*threadIdx.y];
      }
      // Swap a[ib:jb,done:done+pivoted] and a[ib:jb,ib:jb]
      // pulling in new cols from c[] as we go.
      x = ib - 1 + threadIdx.x;
      y = ind[offi + threadIdx.y] - 1;
      if ( x < jb && threadIdx.y < jb - ib + 1
            && y >= 0 ) {
         a[x + ld*(off2 + threadIdx.y)] = a[x + ld*(off1 + y)];
         a[x + ld*(off1 + y)] = c[offc + x + ld*threadIdx.y];
      }
      __syncthreads(); // wait for a[] to be correct
      // Swap a[done:done+pivoted,done:done+pivoted] and
      // a[ib:jb,done:done+pivoted]
      x = ind[offi + threadIdx.x] - 1;
      y = done + threadIdx.y;
      if ( threadIdx.x < jb - ib + 1 && y < done + jb - ib + 1 && x >= 0 ) {
         ELEMENT_TYPE s = a[off1 + x + ld*y];
         a[off1 + x + ld*y] = a[off2 + threadIdx.x + ld*y];
         a[off2 + threadIdx.x + ld*y] = s;
      }
      // Swap a[done:done+pivoted,ib:jb] and a[ib:jb,ib:jb]
      x = ind[offi + threadIdx.x] - 1;
      y = ib - 1 + threadIdx.y;
      if ( threadIdx.x < jb - ib + 1 && y < jb && x >= 0 ) {
         ELEMENT_TYPE s = a[off1 + x + ld*y];
         a[off1 + x + ld*y] = a[off2 + threadIdx.x + ld*y];
         a[off2 + threadIdx.x + ld*y] = s;
      }
      // Swaps with LD
      // Swap a[done:done+pivoted,done:done+pivoted] and
      // a[done:done+pivoted,ib:jb]
      x = done + threadIdx.x;
      y = ind[offi + threadIdx.y] - 1;
      if ( x < done + jb - ib + 1 && threadIdx.y < jb - ib + 1
            && y >= 0 ) {
         ELEMENT_TYPE s = b[x + ld*(off1 + y)];
         b[x + ld*(off1 + y)] =
            b[x + ld*(off2 + threadIdx.y)];
         b[x + ld*(off2 + threadIdx.y)] = s;
      }
      // Swap a[ib:jb,done:done+pivoted] and a[ib:jb,ib:jb]
      x = ib - 1 + threadIdx.x;
      y = ind[offi + threadIdx.y] - 1;
      if ( x < jb && threadIdx.y < jb - ib + 1
            && y >= 0 ) {
         ELEMENT_TYPE s = b[x + ld*(off1 + y)];
         b[x + ld*(off1 + y)] =
            b[x + ld*(off2 + threadIdx.y)];
         b[x + ld*(off2 + threadIdx.y)] = s;
      }
      __syncthreads();
      // Swap a[done:done+pivoted,done:done+pivoted] and
      // a[ib:jb,done:done+pivoted]
      x = ind[offi + threadIdx.x] - 1;
      y = done + threadIdx.y;
      if ( threadIdx.x < jb - ib + 1 && y < done + jb - ib + 1 && x >= 0 ) {
         ELEMENT_TYPE s = b[off1 + x + ld*y];
         b[off1 + x + ld*y] = b[off2 + threadIdx.x + ld*y];
         b[off2 + threadIdx.x + ld*y] = s;
      }
      // Swap a[done:done+pivoted,ib:jb] and a[ib:jb,ib:jb]
      x = ind[offi + threadIdx.x] - 1;
      y = ib - 1 + threadIdx.y;
      if ( threadIdx.x < jb - ib + 1 && y < jb && x >= 0 ) {
         ELEMENT_TYPE s = b[off1 + x + ld*y];
         b[off1 + x + ld*y] = b[off2 + threadIdx.x + ld*y];
         b[off2 + threadIdx.x + ld*y] = s;
      }
   }
}

struct multireorder_data {
   ipc_ node;
   ipc_ block;
   ipc_ nblocks;
};

template< typename ELEMENT_TYPE, uipc_ SIZE_X, uipc_ SIZE_Y >
#if (SM_3X)
__launch_bounds__(256, 8)
#else
__launch_bounds__(256, 4)
#endif
__global__ void
cu_multireorder(
                const struct multinode_fact_type *ndata,
                const struct multireorder_data* rdata,
                const ELEMENT_TYPE* c,
                const ipc_* stat,
                const ipc_* ind,
                ipc_* perm,
                ipc_* ncb) {
   __shared__ volatile ipc_ indf[SIZE_X]; // index from node_fact
   __shared__ volatile ipc_ indr[SIZE_X]; // reorder index
   __shared__ volatile ipc_ simple;

   // Reset ncb ready for next call of muliblock_fact_setup()
   if ( blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
      ncb[0] = 0;
      ncb[1] = 0;
   }

   // Load data on block
   rdata += blockIdx.x;
   ipc_ node = rdata->node;
   ndata += node;
   ipc_ ib   = ndata->ib;
   ipc_ jb   = ndata->jb;
   if ( jb < ib )
      return;
   ipc_ pivoted = stat[node];
   if ( pivoted < 1 )
      return;
   ipc_ nrows = ndata->nrows;
   ipc_ bidx = rdata->block;
   if ( bidx > 1 && (bidx - 1)*blockDim.x >= nrows )
      return;

   ipc_ done = ndata->done;

   ipc_ ld = nrows;
   ipc_ delayed = ib - done - 1; // Number delayed before most recent factor

   if ( threadIdx.x == 0 && threadIdx.y == 0 )
      simple = (delayed == 0); // true if we don't need to offset
   __syncthreads();
   ipc_ next;
   if ( threadIdx.x < jb - ib + 1 && threadIdx.y == 0 ) {
      next = ind[node*SIZE_Y/2 + threadIdx.x]; // SIZE_Y=2*BLOCK_SIZE
      indf[threadIdx.x] = next;
      if ( jb - ib + 1 > delayed )
         indr[delayed + threadIdx.x] = next;
      if ( indf[threadIdx.x] != threadIdx.x + 1 )
         atomicMin((ipc_*)&simple, 0);
   }
   __syncthreads();

   ELEMENT_TYPE *a = ndata->lval;
   ELEMENT_TYPE *b = ndata->ldval;
   ipc_ offc = ndata->lbuf;
   ipc_ nblk = rdata->nblocks;
   if ( simple ) {
      // Copy successful columns from workspace c to factors a without an
      // offset or permutation.
      copy_L_LD_no_perm< ELEMENT_TYPE, SIZE_X*2 >
         ( nblk, bidx, threadIdx.x + blockDim.x*threadIdx.y, nrows, pivoted,
           &a[ld*done], ld, &c[offc], ld );
   }
   else {
      // We need a permutation
      ipc_ ncols = ndata->ncols;
      ipc_ offp = ndata->offp;
      if ( jb - ib + 1 > delayed ) {
         // Can't just shuffle along, as pivoted columns overlap with where they
         // need to be. However, we know that pivoted+delayed < 2*BLOCK_SIZE, so
         // we can do a shuffle via shmem.
         copy_L_LD_perm_shmem< ELEMENT_TYPE, SIZE_X, SIZE_Y >
            ( bidx, nblk, done, pivoted, delayed, nrows, ncols, ib, jb,
              offc, offp, ld, indr, a, b, c, perm );
      }
      else {
         // Pivoted columns don't overlap where they need to be, so can just
         // shuffle in global memory a[] and b[].
         copy_L_LD_perm_noshmem< ELEMENT_TYPE, SIZE_X, SIZE_Y >
            ( node, bidx, nblk, done, pivoted, delayed, nrows, ncols, ib, jb,
              offc, offp, ld, ind, indf, a, b, c, perm );

      }
   }
}

template< typename ELEMENT_TYPE, uipc_ SIZE_X, uipc_ SIZE_Y >
__global__ void
cu_multicopy(
              const struct multinode_fact_type *ndata,
              const struct multireorder_data* rdata,
              ELEMENT_TYPE* b,
              ipc_* stat,
              ipc_* ncb
            )
{

   if ( blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
      ncb[0] = 0;
      ncb[1] = 0;
   }

   rdata += blockIdx.x;
   ipc_ node = rdata->node;
   ndata += node;
   ipc_ ib   = ndata->ib;
   ipc_ jb   = ndata->jb;
   if ( jb < ib )
      return;
   ipc_ pivoted = stat[node];
   if ( pivoted < 1 )
      return;
   ipc_ nrows = ndata->nrows;
   ipc_ block = rdata->block;
   ipc_ nblocks = rdata->nblocks;
   if ( block > 1 && (block - 1)*blockDim.x >= nrows )
      return;

   ipc_ done = ndata->done;
   ELEMENT_TYPE *a = ndata->lval;
   ipc_ offb = ndata->lbuf;
   for ( ipc_ x = threadIdx.x + blockDim.x*block;
        x < nrows && threadIdx.y < pivoted;
        x += blockDim.x*nblocks ) {
     a[x + nrows*(done + threadIdx.y)] = b[offb + x + nrows*threadIdx.y];
   }
}

struct multisymm_type {
   rpc_ *lcol;
   ipc_ ncols;
   ipc_ nrows;
};

/*
 * Symmetrically fills the upper triangles of the upper square blocks of
 * matrices continuously packed in a
 * Note: modifed data is pointed to by component of *msdata
 */
template< typename ELEMENT_TYPE >
__global__ void
cu_multisymm( const struct multisymm_type* msdata )
{
  msdata += blockIdx.x;
  ELEMENT_TYPE *a = msdata->lcol;
  ipc_ ncols = msdata->ncols;
  ipc_ nrows = msdata->nrows;
  for ( ipc_ i = threadIdx.x + blockDim.x*blockIdx.y; i < ncols;
        i += blockDim.x*gridDim.y )
    for ( ipc_ j = threadIdx.y + blockDim.y*blockIdx.z; j < i;
          j += blockDim.y*gridDim.z )
        a[j + i*nrows] = a[i + j*nrows];
}

} /* anon namespace */

/*******************************************************************************
 * Following routines are exported with C binding so can be called from Fortran
 ******************************************************************************/

extern "C" {

void spral_ssids_copy_ic(cudaStream_t *stream, ipc_ nrows, ipc_ ncols,
    rpc_* a, ipc_ lda, rpc_* b, ipc_ ldb, ipc_* ind) {
  ipc_ rb = (nrows - 1)/BLOCK_SIZE + 1;
  ipc_ cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_copy_ic< rpc_ >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, ind );
}

void spral_ssids_copy_mc(cudaStream_t *stream, ipc_ nrows, ipc_ ncols, rpc_* a,
      ipc_ lda, rpc_* b, ipc_ ldb, ipc_* mask) {
  ipc_ rb = (nrows - 1)/BLOCK_SIZE + 1;
  ipc_ cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_copy_mc< rpc_ >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, mask );
}

void spral_ssids_multisymm(cudaStream_t *stream, ipc_ nblocks,
      const struct multisymm_type* msdata) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  for ( ipc_ i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    ipc_ nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    dim3 grid(nb,4,4);
    cu_multisymm< rpc_ ><<< grid, threads, 0, *stream >>>( msdata + i );
  }
}

void spral_ssids_multicopy(cudaStream_t *stream, ipc_ nblocks,
      const struct multinode_fact_type *ndata,
      const struct multireorder_data *rdata,
      rpc_* a, rpc_* b, ipc_* stat, ipc_* ncb) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  for ( ipc_ i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    ipc_ nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    cu_multicopy< rpc_, BLOCK_SIZE, BLOCK_SIZE >
      <<< nb, threads, 0, *stream >>>
      ( ndata, rdata + i, b, stat, ncb );
  }
}

void spral_ssids_multireorder(cudaStream_t *stream, ipc_ nblocks,
      const struct multinode_fact_type *ndata,
      const struct multireorder_data *rdata,
      rpc_* c, ipc_* stat, ipc_* ind, ipc_* index, ipc_* ncb) {
  dim3 threads(2*BLOCK_SIZE, 2*BLOCK_SIZE);
  for ( ipc_ i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    ipc_ nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    dim3 grid(nb,1);
    cu_multireorder< rpc_, 2*BLOCK_SIZE, 2*BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( ndata, rdata + i, c, stat, ind, index, ncb );
  }
}

// ncols <= 2*BLOCK_SIZE
void spral_ssids_reorder_cols2(cudaStream_t *stream, ipc_ nrows, ipc_ ncols,
    rpc_* a, ipc_ lda, rpc_* b, ipc_ ldb, ipc_* index, ipc_ mode ) {
  ipc_ rb = (nrows - 1)/BLOCK_SIZE + 1;
  dim3 grid(rb, 2);

  if ( ncols <= BLOCK_SIZE ) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    cu_reorder_cols2< rpc_, BLOCK_SIZE, BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
  else if ( ncols <= 2*BLOCK_SIZE ) {
    dim3 threads(BLOCK_SIZE, 2*BLOCK_SIZE);
    cu_reorder_cols2< rpc_, BLOCK_SIZE, 2*BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
}

void spral_ssids_reorder_rows(cudaStream_t *stream, ipc_ nrows, ipc_ ncols,
      rpc_* a, ipc_ lda, rpc_* b, ipc_ ldb, ipc_* index) {
  ipc_ cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 grid(1, cb);
  ipc_ tx = min(nrows, 1024/BLOCK_SIZE);
  dim3 threads(tx, BLOCK_SIZE);
  cu_reorder_rows< rpc_ >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, index );
}

// nrows <= 2*BLOCK_SIZE
void spral_ssids_reorder_rows2(cudaStream_t *stream, ipc_ nrows, ipc_ ncols,
    rpc_* a, ipc_ lda, rpc_* b, ipc_ ldb, ipc_* index, ipc_ mode ) {
  ipc_ cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 grid(cb, 2);

  if ( nrows <= BLOCK_SIZE ) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    cu_reorder_rows2< rpc_, BLOCK_SIZE, BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
  else if ( nrows <= 2*BLOCK_SIZE ) {
    dim3 threads(2*BLOCK_SIZE, BLOCK_SIZE);
    cu_reorder_rows2< rpc_, 2*BLOCK_SIZE, BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
}

void spral_ssids_swap_ni2Dm(cudaStream_t *stream, ipc_ nblocks,
      struct multiswap_type *swapdata) {
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   for ( ipc_ i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
      ipc_ nb = min(MAX_CUDA_BLOCKS, nblocks - i);
      dim3 grid(nb,8);
      cu_multiswap_ni2D_c
         < rpc_ >
         <<< grid, threads, 0, *stream >>>
         ( swapdata + i );
      cu_multiswap_ni2D_r
         < rpc_ >
         <<< grid, threads, 0, *stream >>>
         ( swapdata + i );
   }
}

void spral_ssids_swap_ni2D_ic(cudaStream_t *stream, ipc_ nrows, ipc_ ncols,
      rpc_* a, ipc_ lda, rpc_* b, ipc_ ldb, ipc_* index) {
  ipc_ rb = (nrows - 1)/BLOCK_SIZE + 1;
  ipc_ cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_swap_ni2D_ic< rpc_ >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, index );
}

void spral_ssids_swap_ni2D_ir(cudaStream_t *stream, ipc_ nrows, ipc_ ncols,
    rpc_* a, ipc_ lda, rpc_* b, ipc_ ldb, ipc_* index) {
  ipc_ rb = (nrows - 1)/BLOCK_SIZE + 1;
  ipc_ cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_swap_ni2D_ir< rpc_ >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, index );
}

} // end extern "C"
