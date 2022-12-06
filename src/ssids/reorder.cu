#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "ssids/gpu/kernels/datatypes.h"
#include "cuda/cuda_check.h"

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
cu_copy_mc( int nrows, int ncols,
            ELEMENT_TYPE* a, int lda,
            ELEMENT_TYPE* b, int ldb,
            int* mask )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  if ( i < nrows && j < ncols && mask[j] > 0 )
    b[i + ldb*j] = a[i + lda*j];
}

template< typename ELEMENT_TYPE >
__global__ void
cu_copy_ic( int nrows, int ncols,
            ELEMENT_TYPE* a, int lda,
            ELEMENT_TYPE* b, int ldb,
            int* ind )
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  if ( i < nrows && j < ncols && ind[j] > 0 )
    b[i + ldb*(ind[j] - 1)] = a[i + lda*j];
}

template< typename ELEMENT_TYPE >
__global__ void
cu_swap_ni2D_ic( int nrows, int ncols,
                 ELEMENT_TYPE* a, int lda,
                 ELEMENT_TYPE* b, int ldb,
                 int* index )
// swaps columns of non-intersecting 2D arrays a(1:n,index(1:m)) and b(1:n,1:m)
// index is one-based
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k;
  double s;

  if ( i < nrows && j < ncols && (k = index[j] - 1) > -1 ) {
    s = a[i + lda*k];
    a[i + lda*k] = b[i + ldb*j];
    b[i + ldb*j] = s;
  }
}

template< typename ELEMENT_TYPE >
__global__ void
cu_swap_ni2D_ir( int nrows, int ncols,
                 ELEMENT_TYPE* a, int lda,
                 ELEMENT_TYPE* b, int ldb,
                 int* index )
// swaps rows of non-intersecting 2D arrays a(index(1:n),1:m) and b(1:n,1:m)
// index is one-based
{
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  int j = threadIdx.y + blockDim.y*blockIdx.y;
  int k;
  double s;

  if ( i < nrows && j < ncols && (k = index[i] - 1) > -1 ) {
    s = a[k + lda*j];
    a[k + lda*j] = b[i + ldb*j];
    b[i + ldb*j] = s;
  }
}

struct multiswap_type {
   int nrows;
   int ncols;
   int k;
   double *lcol;
   int lda;
   int off;
};

template< typename ELEMENT_TYPE >
__global__ void
cu_multiswap_ni2D_c( struct multiswap_type *swapdata )
// swaps non-intersecting rows or cols of a 2D multiarray a
{
  swapdata += blockIdx.x;
  int nrows = swapdata->nrows;
  if ( blockIdx.y*blockDim.x >= nrows )
    return;

  int k     = swapdata->k;
  ELEMENT_TYPE *a = swapdata->lcol;
  int lda   = swapdata->lda;
  int off  = lda*swapdata->off;
  ELEMENT_TYPE s;

  for ( int i = threadIdx.x + blockIdx.y*blockDim.x; i < nrows;
        i += blockDim.x*gridDim.y )
    for ( int j = threadIdx.y; j < k; j += blockDim.y ) {
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
  int ncols = swapdata->ncols;
  if ( blockIdx.y*blockDim.y >= ncols )
    return;

  int k     = swapdata->k;
  ELEMENT_TYPE *a = swapdata->lcol;
  int lda   = swapdata->lda;
  int off  = swapdata->off;
  ELEMENT_TYPE s;

  for ( int i = threadIdx.x; i < k; i += blockDim.x )
    for ( int j = threadIdx.y + blockIdx.y*blockDim.y; j < ncols;
          j += blockDim.y*gridDim.y ) {
      s = a[i + lda*j];
      a[i + lda*j] = a[off + i + lda*j];
      a[off + i + lda*j] = s;
    }
}

template< typename ELEMENT_TYPE >
__global__ void
cu_reorder_rows(
                int nrows, int ncols,
                ELEMENT_TYPE* a, int lda,
                ELEMENT_TYPE* b, int ldb,
                int* index
               )
{
  int x;
  int y = threadIdx.y + blockIdx.y*blockDim.y;

  for ( x = threadIdx.x; x < nrows; x += blockDim.x )
    if ( y < ncols )
      b[index[x] - 1 + ldb*y] = a[x + lda*y];
  __syncthreads();
  for ( x = threadIdx.x; x < nrows; x += blockDim.x )
    if ( y < ncols )
      a[x + lda*y] = b[x + ldb*y];
}

template< typename ELEMENT_TYPE, unsigned int SIZE_X, unsigned int SIZE_Y >
__global__ void
cu_reorder_cols2( int nrows, int ncols,
                  ELEMENT_TYPE* a, int lda,
                  ELEMENT_TYPE* b, int ldb,
                  int* index, int mode )
{
  int ix = threadIdx.x + blockIdx.x*blockDim.x;

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

template< typename ELEMENT_TYPE, unsigned int SIZE_X, unsigned int SIZE_Y >
__global__ void
cu_reorder_rows2( int nrows, int ncols,
                  ELEMENT_TYPE* a, int lda,
                  ELEMENT_TYPE* b, int ldb,
                  int* index, int mode )
{
  int iy = threadIdx.y + blockIdx.x*blockDim.y;

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
template< typename ELEMENT_TYPE, int NTX >
__device__ void
__forceinline__ // Required to avoid errors about reg counts compiling with -G
copy_L_LD_no_perm(
      int nblk, int bidx, int tid,
      int nrows, int ncols,
      ELEMENT_TYPE *dest, int ldd,
      const ELEMENT_TYPE *src, int lds
) {
   int tx = tid % NTX;
   int ty = tid / NTX;
   src += NTX*bidx;
   dest += NTX*bidx;
   nrows -= NTX*bidx;
   if ( ty < ncols ) {
      for ( int x = tx; x < nrows; x += NTX*nblk )
         dest[x + ldd*ty] = src[x + lds*ty];
   }
}

/* Shuffles the permutation vector using shared memory
   [in case it overlaps itself] */
template < int SIZE_X >
__device__ void
shuffle_perm_shmem( int n, volatile const int *const indr, int *perm ) {
   // Update permutation
   __shared__ volatile int iwork[SIZE_X];
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
template< typename ELEMENT_TYPE, unsigned int SIZE_X, unsigned int SIZE_Y >
__device__ void
__forceinline__ // Required to avoid errors about reg counts compiling with -G
copy_L_LD_perm_shmem(
      int block, int nblocks,
      int done, int pivoted, int delayed,
      int nrows, int ncols,
      int ib, int jb,
      int offc, int offp,
      int ld,
      volatile int *const indr,
      double *a, double *b, const double *c,
      int *perm
) {
   __shared__ volatile ELEMENT_TYPE work1[SIZE_X*SIZE_Y];
   __shared__ volatile ELEMENT_TYPE work2[SIZE_X*SIZE_Y];
#if (!SM_3X)
   __shared__ volatile ELEMENT_TYPE work3[SIZE_X*SIZE_Y];
   __shared__ volatile ELEMENT_TYPE work4[SIZE_X*SIZE_Y];
#endif

   // Extend permutation array to cover non-pivoted columns
   if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
      int i = 0;
      int j = pivoted;
      for ( ; i < delayed; i++ )
         indr[i] = ++j;
      for ( ; i < delayed + jb - ib + 1; i++ )
         if ( !indr[i] )
            indr[i] = ++j;
   }

   int off = done*ld;

   // We handle the (done-jb) x (done-jb) block that requires both
   // row and column permutations seperately using the first block.
   // All remaining rows and columns are handlded by the remaining blocks.
   // Note that while we do not need to perumute "above" the pivoted columns,
   // we do need to permute to the "left" of the pivoted rows!
   if ( block ) {
      // Swap columns of A and copy in L, but avoiding rows that need
      // permuted
      // Also, swap cols of LD but avoiding rows that need permuted
      int baseStep = blockDim.x*(nblocks - 1);
#if (SM_3X)
      for ( int i = jb + blockDim.x*(block - 1); i < nrows;
            i += baseStep ) {
#else
      for ( int i = jb + blockDim.x*(block - 1); i < nrows + baseStep;
            i += baseStep * 2 ) {
#endif
         int ix = i + threadIdx.x;
#if (!SM_3X)
         int ix2 = ix + baseStep;
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
      for ( int i = blockDim.y*(block - 1); i < ncols;
            i += baseStep  ) {
#else
      for ( int i = blockDim.y*(block - 1); i < ncols + baseStep;
            i += baseStep * 2 ) {
#endif
         int iy = i + threadIdx.y;
#if (!SM_3X)
         int iy2 = iy + baseStep;
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

      int pass = threadIdx.x < jb - done && threadIdx.y < jb - done;

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
template< typename ELEMENT_TYPE, unsigned int SIZE_X, unsigned int SIZE_Y >
__device__ void
__forceinline__ // Required to avoid errors about reg counts compiling with -G
copy_L_LD_perm_noshmem(
      int node,
      int block, int nblocks,
      int done, int pivoted, int delayed,
      int nrows, int ncols,
      int ib, int jb,
      int offc, int offp,
      int ld,
      const int *ind,
      const volatile int *const indf,
      double *a, double *b, const double *c,
      int *perm
) {

   int off1 = done;
   int off2 = ib - 1;
   int offi = node*SIZE_Y/2;

   // We handle the two pivoted x pivoted blocks where row and columns cross
   // over seperately using the first block.
   // The other blocks just exclude these rows/cols as appropriate
   // All remaining rows and columns are handlded by the remaining blocks.
   if ( block ) {
      // Handle parts of matrix that require EITHER row OR col shuffle
      int tx = (threadIdx.y < SIZE_Y/2) ? threadIdx.x : threadIdx.x + blockDim.x;
      int ty = (threadIdx.y < SIZE_Y/2) ? threadIdx.y : threadIdx.y - SIZE_Y/2;
      // Swap a[:,done:done+pivoted] and a[:,ib:jb] pulling in c[] as we go
      for ( int x = tx + 2*blockDim.x*(block - 1);
            x < nrows && ty < jb - ib + 1;
            x += 2*blockDim.x*(nblocks - 1) ) {
         int y = ind[offi + ty] - 1;
         if ( (x >= done   && x < done + jb - ib + 1)
               || (x >= ib - 1 && x < jb) || y < 0 )
            continue; // handled separately
         a[x + ld*(off2 + ty)] = a[x + ld*(off1 + y)];
         a[x + ld*(off1 + y)] = c[offc + x + ld*ty];
      }
      // Swap b[:,done:done+pivoted] and b[:,ib:jb]
      for ( int x = tx + 2*blockDim.x*(block - 1);
            x < nrows && ty < jb - ib + 1;
            x += 2*blockDim.x*(nblocks - 1) ) {
         int y = ind[offi + ty] - 1;
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
      for ( int y = threadIdx.y + blockDim.y*(block - 1);
            y < ncols && threadIdx.x < jb - ib + 1;
            y += blockDim.y*(nblocks - 1) ) {
         int x = ind[offi + threadIdx.x] - 1;
         if ( (y >= done && y < done + jb - ib + 1)
               || (y >= ib - 1 && y < jb) || x < 0 )
            continue; // handled separately
         ELEMENT_TYPE s = a[off1 + x + ld*y];
         a[off1 + x + ld*y] = a[off2 + threadIdx.x + ld*y];
         a[off2 + threadIdx.x + ld*y] = s;
      }
      // swap b[done:done+pivoted,:] and b[ib:jb,:]
      for ( int y = threadIdx.y + blockDim.y*(block - 1);
            y < ncols && threadIdx.x < jb - ib + 1;
            y += blockDim.y*(nblocks - 1) ) {
         int x = ind[offi + threadIdx.x] - 1;
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
         int i = indf[threadIdx.x] - 1;
         if ( i >= 0 ) {
            int s = perm[offp + ib - 1 + threadIdx.x];
            perm[offp + ib - 1 + threadIdx.x] = perm[offp + done + i];
            perm[offp + done + i] = s;
         }
      }

      // Swaps with L
      // FIXME: This might be sped up by doing 1.5 swaps instead of 3.5.
      // Swap a[done:done+pivoted,done:done+pivoted] and
      // a[done:done+pivoted,ib:jb]
      // pulling in new cols from c[] as we go.
      int x = done + threadIdx.x;
      int y = ind[offi + threadIdx.y] - 1;
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
   int node;
   int block;
   int nblocks;
};

template< typename ELEMENT_TYPE, unsigned int SIZE_X, unsigned int SIZE_Y >
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
                const int* stat,
                const int* ind,
                int* perm,
                int* ncb) {
   __shared__ volatile int indf[SIZE_X]; // index from node_fact
   __shared__ volatile int indr[SIZE_X]; // reorder index
   __shared__ volatile int simple;

   // Reset ncb ready for next call of muliblock_fact_setup()
   if ( blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
      ncb[0] = 0;
      ncb[1] = 0;
   }

   // Load data on block
   rdata += blockIdx.x;
   int node = rdata->node;
   ndata += node;
   int ib   = ndata->ib;
   int jb   = ndata->jb;
   if ( jb < ib )
      return;
   int pivoted = stat[node];
   if ( pivoted < 1 )
      return;
   int nrows = ndata->nrows;
   int bidx = rdata->block;
   if ( bidx > 1 && (bidx - 1)*blockDim.x >= nrows )
      return;

   int done = ndata->done;

   int ld = nrows;
   int delayed = ib - done - 1; // Number delayed before most recent factor

   if ( threadIdx.x == 0 && threadIdx.y == 0 )
      simple = (delayed == 0); // true if we don't need to offset
   __syncthreads();
   int next;
   if ( threadIdx.x < jb - ib + 1 && threadIdx.y == 0 ) {
      next = ind[node*SIZE_Y/2 + threadIdx.x]; // SIZE_Y=2*BLOCK_SIZE
      indf[threadIdx.x] = next;
      if ( jb - ib + 1 > delayed )
         indr[delayed + threadIdx.x] = next;
      if ( indf[threadIdx.x] != threadIdx.x + 1 )
         atomicMin((int*)&simple, 0);
   }
   __syncthreads();

   ELEMENT_TYPE *a = ndata->lval;
   ELEMENT_TYPE *b = ndata->ldval;
   int offc = ndata->lbuf;
   int nblk = rdata->nblocks;
   if ( simple ) {
      // Copy successful columns from workspace c to factors a without an
      // offset or permutation.
      copy_L_LD_no_perm< ELEMENT_TYPE, SIZE_X*2 >
         ( nblk, bidx, threadIdx.x + blockDim.x*threadIdx.y, nrows, pivoted,
           &a[ld*done], ld, &c[offc], ld );
   }
   else {
      // We need a permutation
      int ncols = ndata->ncols;
      int offp = ndata->offp;
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

template< typename ELEMENT_TYPE, unsigned int SIZE_X, unsigned int SIZE_Y >
__global__ void
cu_multicopy(
              const struct multinode_fact_type *ndata,
              const struct multireorder_data* rdata,
              ELEMENT_TYPE* b,
              int* stat,
              int* ncb
            )
{

   if ( blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 ) {
      ncb[0] = 0;
      ncb[1] = 0;
   }

   rdata += blockIdx.x;
   int node = rdata->node;
   ndata += node;
   int ib   = ndata->ib;
   int jb   = ndata->jb;
   if ( jb < ib )
      return;
   int pivoted = stat[node];
   if ( pivoted < 1 )
      return;
   int nrows = ndata->nrows;
   int block = rdata->block;
   int nblocks = rdata->nblocks;
   if ( block > 1 && (block - 1)*blockDim.x >= nrows )
      return;

   int done = ndata->done;
   ELEMENT_TYPE *a = ndata->lval;
   int offb = ndata->lbuf;
   for ( int x = threadIdx.x + blockDim.x*block;
        x < nrows && threadIdx.y < pivoted;
        x += blockDim.x*nblocks ) {
     a[x + nrows*(done + threadIdx.y)] = b[offb + x + nrows*threadIdx.y];
   }
}

struct multisymm_type {
   double *lcol;
   int ncols;
   int nrows;
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
  int ncols = msdata->ncols;
  int nrows = msdata->nrows;
  for ( int i = threadIdx.x + blockDim.x*blockIdx.y; i < ncols;
        i += blockDim.x*gridDim.y )
    for ( int j = threadIdx.y + blockDim.y*blockIdx.z; j < i;
          j += blockDim.y*gridDim.z )
        a[j + i*nrows] = a[i + j*nrows];
}

} /* anon namespace */

/*******************************************************************************
 * Following routines are exported with C binding so can be called from Fortran
 ******************************************************************************/

extern "C" {

void spral_ssids_copy_ic(cudaStream_t *stream, int nrows, int ncols,
    double* a, int lda, double* b, int ldb, int* ind) {
  int rb = (nrows - 1)/BLOCK_SIZE + 1;
  int cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_copy_ic< double >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, ind );
}

void spral_ssids_copy_mc(cudaStream_t *stream, int nrows, int ncols, double* a,
      int lda, double* b, int ldb, int* mask) {
  int rb = (nrows - 1)/BLOCK_SIZE + 1;
  int cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_copy_mc< double >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, mask );
}

void spral_ssids_multisymm(cudaStream_t *stream, int nblocks,
      const struct multisymm_type* msdata) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    dim3 grid(nb,4,4);
    cu_multisymm< double ><<< grid, threads, 0, *stream >>>( msdata + i );
  }
}

void spral_ssids_multicopy(cudaStream_t *stream, int nblocks,
      const struct multinode_fact_type *ndata,
      const struct multireorder_data *rdata,
      double* a, double* b, int* stat, int* ncb) {
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    cu_multicopy< double, BLOCK_SIZE, BLOCK_SIZE >
      <<< nb, threads, 0, *stream >>>
      ( ndata, rdata + i, b, stat, ncb );
  }
}

void spral_ssids_multireorder(cudaStream_t *stream, int nblocks,
      const struct multinode_fact_type *ndata,
      const struct multireorder_data *rdata,
      double* c, int* stat, int* ind, int* index, int* ncb) {
  dim3 threads(2*BLOCK_SIZE, 2*BLOCK_SIZE);
  for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
    int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
    dim3 grid(nb,1);
    cu_multireorder< double, 2*BLOCK_SIZE, 2*BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( ndata, rdata + i, c, stat, ind, index, ncb );
  }
}

// ncols <= 2*BLOCK_SIZE
void spral_ssids_reorder_cols2(cudaStream_t *stream, int nrows, int ncols,
    double* a, int lda, double* b, int ldb, int* index, int mode ) {
  int rb = (nrows - 1)/BLOCK_SIZE + 1;
  dim3 grid(rb, 2);

  if ( ncols <= BLOCK_SIZE ) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    cu_reorder_cols2< double, BLOCK_SIZE, BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
  else if ( ncols <= 2*BLOCK_SIZE ) {
    dim3 threads(BLOCK_SIZE, 2*BLOCK_SIZE);
    cu_reorder_cols2< double, BLOCK_SIZE, 2*BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
}

void spral_ssids_reorder_rows(cudaStream_t *stream, int nrows, int ncols,
      double* a, int lda, double* b, int ldb, int* index) {
  int cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 grid(1, cb);
  int tx = min(nrows, 1024/BLOCK_SIZE);
  dim3 threads(tx, BLOCK_SIZE);
  cu_reorder_rows< double >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, index );
}

// nrows <= 2*BLOCK_SIZE
void spral_ssids_reorder_rows2(cudaStream_t *stream, int nrows, int ncols,
    double* a, int lda, double* b, int ldb, int* index, int mode ) {
  int cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 grid(cb, 2);

  if ( nrows <= BLOCK_SIZE ) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    cu_reorder_rows2< double, BLOCK_SIZE, BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
  else if ( nrows <= 2*BLOCK_SIZE ) {
    dim3 threads(2*BLOCK_SIZE, BLOCK_SIZE);
    cu_reorder_rows2< double, 2*BLOCK_SIZE, BLOCK_SIZE >
      <<< grid, threads, 0, *stream >>>
      ( nrows, ncols, a, lda, b, ldb, index, mode );
  }
}

void spral_ssids_swap_ni2Dm(cudaStream_t *stream, int nblocks,
      struct multiswap_type *swapdata) {
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
      int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
      dim3 grid(nb,8);
      cu_multiswap_ni2D_c
         < double >
         <<< grid, threads, 0, *stream >>>
         ( swapdata + i );
      cu_multiswap_ni2D_r
         < double >
         <<< grid, threads, 0, *stream >>>
         ( swapdata + i );
   }
}

void spral_ssids_swap_ni2D_ic(cudaStream_t *stream, int nrows, int ncols,
      double* a, int lda, double* b, int ldb, int* index) {
  int rb = (nrows - 1)/BLOCK_SIZE + 1;
  int cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_swap_ni2D_ic< double >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, index );
}

void spral_ssids_swap_ni2D_ir(cudaStream_t *stream, int nrows, int ncols,
    double* a, int lda, double* b, int ldb, int* index) {
  int rb = (nrows - 1)/BLOCK_SIZE + 1;
  int cb = (ncols - 1)/BLOCK_SIZE + 1;
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(rb, cb);
  cu_swap_ni2D_ir< double >
    <<< grid, threads, 0, *stream >>>
    ( nrows, ncols, a, lda, b, ldb, index );
}

} // end extern "C"
