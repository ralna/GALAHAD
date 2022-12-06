/*
Copyright (c) 2012-13, The Science and Technology Facilities Council (STFC)
Copyright (c) 2012, NVIDIA
Principal Author: Jonathan Hogg (STFC)
Other Contributors:
   Christopher Munro (STFC)
   Philippe Vandermersch (NVIDIA)
All rights reserved.

This file is a modified version of the ASEArch blas version. It has had a
lookup capability added to allow execution on multiple small matrices
simulateously.
*/

namespace spral { namespace ssids { namespace gpu {

/** \brief Return value at address vptr using volatile load. */
template<typename T_ELEM>
__inline__ __device__ T_ELEM loadVolatile(const volatile T_ELEM *const vptr)
{
  return *vptr;
}

#include "cuda/cuda_check.h"
#include <cuComplex.h>

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))

/* If INV_AFTER is defined, then for the global memory variant, an explicit
   backwards stable inverse is calculated for the diagonal blocks of row
   INV_AFTER and all subsequent rows */
#define INV_AFTER 4

/*
 * Global memory variant parameters:
 * TRSV_NB_TASK is rows per thread block = numThreads.x
 * THREADSY_TASK is numThreads.y
 */
#define TRSV_NB_TASK 32 // Strongly recommend = warpSize
#define THREADSX_TASK TRSV_NB_TASK // This MUST == NB_TASK <= warpSize
#ifndef THREADSY_TASK
#define THREADSY_TASK 4
#endif

/** \brief Return physical SM id as per special register %smid. */
unsigned int __inline__ __device__ getSM(void) {
  volatile unsigned int output;
  asm volatile("mov.u32 %0,%smid;" : "=r"(output) : );
  return output;
}

/**
 * \brief Perform non-transpose solve with diagonal block.
 * \warning blkSize MUST be <= warpSize.
 * \tparam T_ELEM Underlying data type, e.g. double.
 * \tparam blkSize Block size.
 * \tparam ISUNIT True if diagonal is 1.0, false otherwise.
 * \param minus_a Diagonal block
 * \param lda Leading dimension of a.
 * \param val This thread's element of x.
 * \sa dblkSolve_trans()
 */
template <typename T_ELEM, int blkSize, bool ISUNIT>
void __device__ dblkSolve(const volatile T_ELEM *const minus_a, const int lda, T_ELEM &val)
{
   volatile T_ELEM __shared__ xs;

#pragma unroll 16
   for (int i=0; i<blkSize; ++i) {
     if (threadIdx.x==i) {
       if (!ISUNIT) val *= minus_a[i*lda+i];
       xs = val;
     }
     if (threadIdx.x > i)
       val += minus_a[i*lda+threadIdx.x] * xs;
   }
}

/**
 * \brief Perform transpose solve with diagonal block.
 * \tparam T_ELEM Underlying data type, e.g. double.
 * \tparam blkSize Block size.
 * \tparam ISUNIT True if diagonal is 1.0, false otherwise.
 * \param minus_a Diagonal block
 * \param lda Leading dimension of a.
 * \param val This thread's element of x.
 * \sa dblkSolve()
 */
template <typename T_ELEM, int blkSize, bool ISUNIT>
void __device__ dblkSolve_trans(const volatile T_ELEM *const minus_a, const int lda, T_ELEM &val)
{
   volatile T_ELEM __shared__ xs;

#pragma unroll 16
   for (int i=blkSize-1; i>=0; --i) {
     if (threadIdx.x==i) {
       if (!ISUNIT) val *= minus_a[i*lda+i];
       xs = val;
     }
     if (threadIdx.x < i)
       val += minus_a[i*lda+threadIdx.x] * xs;
   }
}

/**
 * \brief Copies an nbi x nbi block of a to provided cache.
 * \details Copies -a and only the half triangle
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam nbi Inner block size.
 * \tparam ntid Number of participating threads.
 * \tparam TRANS Perform transpose during copy if true.
 * \tparam ISUNIT true if diagonal is 1.0, false otherwise.
 * \param tid The thread id.
 * \param a Matrix values.
 * \param lda Leading dimension of a.
 * \param cache Location to copy to, leading dimension nbi.
 * \sa tocache_small()
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid, bool TRANS, bool ISUNIT>
void __device__ tocache(const unsigned int tid, const volatile T_ELEM *const a, const int lda, volatile T_ELEM *const cache)
{
  const int x = tid % nbi;
  const int y = tid / nbi;
  const int ty = ntid/nbi;

  if (!TRANS) {
    for (int i=0; i<nbi; i+=ty) {
      if (x>(i+y)) cache[(i+y)*nbi+x] = -a[(i+y)*lda+x];
      else if ((i+y)<nbi) cache[(i+y)*nbi+x] = 0.0;
      if ((!ISUNIT) && (x==(i+y))) cache[(i+y)*nbi+x] = 1.0 / a[(i+y)*lda+x];
    }
  } else {
    for (int i=0; i<nbi; i+=ty) {
      if (x>(i+y)) cache[(i+y)+nbi*x] = -a[(i+y)*lda+x];
      else if ((i+y)<nbi) cache[(i+y)+nbi*x] = 0.0;
      if ((!ISUNIT) && (x==(i+y))) cache[(i+y)+nbi*x] = 1.0 / a[(i+y)*lda+x];
    }
  }
}

/**
 * \brief Copy an n x n block of a to provided cache, provided n<nbi.
 * \details If diag is true, then only copy lower triangle. ???
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam nbi Inner block size.
 * \tparam ntid Number of participating threads.
 * \tparam TRANS Perform transpose during copy if true.
 * \tparam ISUNIT true if diagonal is 1.0, false otherwise.
 * \param n Size of matrix
 * \param tid The thread id.
 * \param a Matrix values.
 * \param lda Leading dimension of a.
 * \param cache Location to copy to, leading dimension nbi.
 * \sa tocache()
 */
template <typename T_ELEM, unsigned int nbi, unsigned int ntid, bool TRANS, bool ISUNIT>
void __device__ tocache_small(const int n, const unsigned int tid, const volatile T_ELEM *const a, int lda, volatile T_ELEM *const cache)
{
  const int x = tid % nbi;
  const int y = tid / nbi;
  const int ty = ntid/nbi;

  if (!TRANS) {
    for (int i=0; i<n; i+=ty) {
      if (i+y>=nbi) continue; // past end of cache array
      if ((i+y)<n && (x>(i+y) && x<n))  cache[(i+y)*nbi+x] = -a[(i+y)*lda+x];
      else                              cache[(i+y)*nbi+x] = 0.0;
      if ((!ISUNIT) && x==(i+y) && x<n) cache[(i+y)*nbi+x] = 1.0 / a[(i+y)*lda+x];
    }
  } else {
    for (int i=0; i<nbi; i+=ty) {
      if (i+y>=nbi) continue; // past end of cache array
      if ((i+y)<n && x>(i+y) && x<n)    cache[(i+y)+nbi*x] = -a[(i+y)*lda+x];
      else                              cache[(i+y)+nbi*x] = 0.0;
      if ((!ISUNIT) && x==(i+y) && x<n) cache[(i+y)+nbi*x] = 1.0 / a[(i+y)*lda+x];
    }
  }
}

/**
 * \brief Wait until *sync >= col_to_wait.
 * \details Needs to be a separate function to force volatile onto *sync.
 * \warning Call __syncthreads().
 * \param tid This thread's index.
 * \param sync memory location to wait on.
 * \param col_to_wait Target number to wait for.
 * \param col_done Value of *sync on return.
 */
void __device__ wait_until_ge(const int tid, const volatile int *const sync, const int col_to_wait, int *const col_done)
{
  if (tid == 0) {
    /* Only read global memory when necessary */
    if (*col_done < col_to_wait) {
      while (*sync < col_to_wait) {}
      *col_done = *sync;
    }
  }
  __syncthreads();
}

/**
 * \brief Atomically increment next row counter and return old value to all
 *        threads.
 * \warning Contains a call to __syncthreads()
 * \param address pointer to global address used to store next row.
 * \returns next row index
 */
int __device__ nextRow(int *const address)
{
  volatile int __shared__ old;
  if (threadIdx.x==0 && threadIdx.y==0)
    old = atomicAdd(address, 1);
  __syncthreads();
  return old;
}

/**
 * \brief Solves the system \f$ L_{22} X_{21} = - L_{21} X_{11} \f$
 *        for \f$ X_{21} \f$.
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam n size of matrix.
 * \tparam lda leading dimension of x11, a21 and l22.
 * \tparam threadsx compile-time version of blockDim.x.
 * \tparam threadsy compile-time version of blockDim.y.
 * \tparam ISUNIT true if diagonal is 1.0, false otherwise.
 * \param x11 The matrix \f$ X_{11} \f$.
 * \param a21 On input, the matrix \f$ L_{21} \f$, on return the solution
 *        \f$ X_{21} \f$.
 * \param l22 The matrix \f$ L_{21} \f$.
 * \param xsarray workspace???
*/
template <typename T_ELEM, int n, int lda, int threadsx, int threadsy, bool ISUNIT>
void __device__ slv21(const volatile T_ELEM *const x11, volatile T_ELEM *const a21, const volatile T_ELEM *const l22, volatile T_ELEM *const xsarray)
{
  const int tid = threadsx*threadIdx.y+threadIdx.x;
  const int ntid = threadsx*threadsy;
  const int x = (n>0) ? tid % n : 0;
  const int y = (n>0) ? tid / n : 0;
  const int ty = (n>0) ? ntid/n : 1;

   /* Note: as different threads within a warp can work on different
      columns, we need different xs variables (one per col being worked on) */
  volatile T_ELEM *const xs = &(xsarray[y]);

  if (y>n) return;

#pragma unroll
  for (int j=0; j<n; j+=ty) {
    if ((j+y)>=n) continue;

    /* construct col (j+y) of -L_21 X_11 */
    T_ELEM val = 0;
    for (int k=j; k<n; ++k) {
      if (k+y<n) val += a21[(k+y)*lda+x] * x11[(j+y)*lda+k+y];
    }
    val = -val;

    /* solve L_22 X_21(col j) = a21(col j) in place */
#pragma unroll 2
    for (int k=0; k<n; ++k) { // Column of l22, must be done in order
      if (x==k) {
        if (!ISUNIT) val *= l22[k*lda+k];
        xs[0] = val;
      }
      if (x>k)
        val += l22[k*lda+x]*xs[0];
    }
    a21[(j+y)*lda+x] = -val;
  }
}

/**
 * \brief Take transpose of a matrix in shared memory
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam threadsy compile-time version of blockDim.y.
 * \tparam lda leading dimension of a and at.
 * \param n Matrix dimension.
 * \param a Matrix to transpose.
 * \param at Space to output transpose of a.
 */
template <typename T_ELEM, int threadsy, int lda>
void __device__ transpose(const int n, const volatile T_ELEM *const a, volatile T_ELEM *const at)
{
  if (threadIdx.y==0 && threadIdx.x<n) {
    for (int j=0; j<n; ++j)
      at[j*lda+threadIdx.x] = a[threadIdx.x*lda+j];
  }
}

/**
 * \brief Invert a lower triangular matrix
 * \details Works recursively using the formula:
 * \f[\left(\begin{array}{cc}
 *       L_{11} &        \\
 *       L_{21} & L_{22}
 *    \end{array}\right)^{-1} = \left(\begin{array}{cc}
 *        L_{11}^{-1} &      \\
 *       -L_{22}^{-1} L_{21} L_{11}^{-1} & L_{22}^{-1}
 *    \end{array}\right)
 * \f]
 *
 * Note: Expects \f$ -L \f$ to be passed in, and factorises to \f$ +L \f$
 *
 * (This method is recommended as componentwise backwards stable for
 * divide an conquer computation of triangular matrix in version in:
 * Stability of parallel triangular system solvers, Higham, 1995)
 * \tparam T_ELEM Underlying datatype, e.g. double.
 * \tparam n matrix size
 * \tparam lda leading dimension of a
 * \tparam threadsx compile-time version of blockDim.x.
 * \tparam threadsy compile-time version of blockDim.y.
 * \tparam ISUNIT true if diagonal is 1.0, false otherwise.
 * \tparam TRANS true if we actually want to invert L^T, false otherwise.
 * \param a matrix to invert
 * \param xsarray workspace ???
 */
template <typename T_ELEM, int n, int lda, int threadsx, int threadsy, bool ISUNIT, bool TRANS>
void __device__ invert(volatile T_ELEM *const a, volatile T_ELEM /*__shared__*/ *const xsarray)
{
  if (n==2) {
    if (threadIdx.x==0 && threadIdx.y==0) {
      if(ISUNIT) {
        a[0] = 1;
        a[lda+1] = 1;
        a[1] = a[1];
      } else {
        a[0] = a[0];
        a[lda+1] = a[lda+1];
        a[1] = a[1]*(a[0]*a[lda+1]);
      }
      if (TRANS) a[lda] = a[1];
    }
  } else {
    invert<T_ELEM, n/2, lda, threadsx, threadsy, ISUNIT, TRANS>(a, xsarray); // A_11
    __syncthreads();
    slv21<T_ELEM, n/2, lda, threadsx, threadsy, ISUNIT>(a, &a[n/2], &a[(lda+1)*n/2], xsarray); // A_21
    if (TRANS) {
      __syncthreads();
      transpose<T_ELEM, threadsy, lda> (n/2, &a[n/2], &a[(n/2)*lda]);
    }
    __syncthreads();
    invert<T_ELEM, n/2, lda, threadsx, threadsy, ISUNIT, TRANS>(&a[(lda+1)*n/2], xsarray); // A_22
  }
}

/**
 * \brief Performs a solve through a precalulated matrix inverse
 * \details (so actually a triangular matrix-vector multiply)
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam n matrix size.
 * \tparam threadsy compile-time version of blockDim.y.
 * \param a precalculated matrix inverse
 * \param xshared workspace???
 * \param val this thread's component of rhs vector in input, replaced with
 *        solution on output
 * \param partSum workspace???
 */
template<typename T_ELEM, int n, int threadsy>
void __device__ slvinv(const volatile T_ELEM *a, volatile T_ELEM *xshared, T_ELEM &val, volatile T_ELEM *const partSum)
{
  a += threadIdx.y*n+threadIdx.x;
  xshared += threadIdx.y;

  if (threadIdx.y==0) {
    xshared[threadIdx.x] = val;
  }
  __syncthreads();

  /* matrix-vector multiply for solution */
  if (threadIdx.y<threadsy && threadIdx.x<n) {
    val=0;
    for (int j=0; j<n; j+=threadsy) {
      val += a[j*n] * xshared[j];
    }
    partSum[threadIdx.y*n+threadIdx.x] = val;
  }
  __syncthreads();
  if (threadIdx.y==0) {
    for(int i=1; i<threadsy; ++i)
      val += partSum[i*n+threadIdx.x];
  }
}

/**
 * \brief Performs a solve through a transpose precalulated matrix inverse
 * \details (so actually a transpose triangular matrix-vector multiply)
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam n matrix size.
 * \tparam threadsy compile-time version of blockDim.y.
 * \param a precalculated matrix inverse
 * \param xshared workspace???
 * \param val this thread's component of rhs vector in input, replaced with
 *        solution on output
 * \param partSum workspace???
 * \param row FIXME: unused???
 */
template<typename T_ELEM, int n, int threadsy>
void __device__ slvinv_trans(const volatile T_ELEM *a, volatile T_ELEM *xshared, T_ELEM &val, volatile T_ELEM *const partSum, const int row)
{
  a += threadIdx.y*n+threadIdx.x;
  xshared += threadIdx.y;

  if (threadIdx.y==0) {
    xshared[threadIdx.x] = val;
  }
  __syncthreads();

  /* matrix-vector multiply for solution */
  val=0;
  if (threadIdx.x<n) {
    for (int j=0; j<n; j+=threadsy) {
      if (threadIdx.x <= j+threadIdx.y) {
        val += a[j*n] * xshared[j];
      }
    }
  }
  partSum[threadIdx.y*n+threadIdx.x] = val;
  __syncthreads();
  if (threadIdx.y==0) {
    for (int i=1; i<threadsy; ++i)
      val += partSum[i*n+threadIdx.x];
  }
}

/** \brief Sets sync values correctly prior to call to trsv_ln_exec */
void __global__ trsv_init(volatile int *sync)
{
  sync += 2*blockIdx.x;
  sync[0] = -1; // Last ready column
  sync[1] = 0; // Next row to assign
}

/** \brief Represents a single block for batched trsv call. */
struct trsv_lookup {
  int n; ///< Size of matrix this block works on.
  const double *a; ///< Data for matrix this block works on.
  int lda; ///< Leading dimension of a.
  int x_offset; ///< Offset into x vector we're solving for
  int sync_offset; ///< Offset into sync vector for this matrix
};

#ifdef TIMING
struct trsv_times {
  unsigned int sm;
  unsigned int sa;
  unsigned int en;
};
#endif

/**
 * \brief Performs batched trsv for Transposed Lower-triangular matrices
 * \details Requires trsv_init() to be called first to initialize sync[].
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam nb block size.
 * \tparam threadsx compile-time version of blockDim.x.
 * \tparam threadsy compile-time version of blockDim.y.
 * \tparam ISUNIT true if diagonal is 1.0, false otherwise.
 * \param lookup batch lookup array.
 * \param xglobal x array to offset into.
 * \param sync sync array to offset into.
 */
template <typename T_ELEM, unsigned int nb, unsigned int threadsx, unsigned int threadsy, bool ISUNIT>
#ifndef DOXYGEN_SHOULD_SKIP_THIS
__launch_bounds__(threadsx*threadsy, 4)
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
void __global__ trsv_lt_exec(const struct trsv_lookup *lookup, T_ELEM *xglobal, int *sync
#ifdef TIMING
      , struct trsv_times *times
#endif
) {
   lookup += blockIdx.x;
   const int n = lookup->n;
   const T_ELEM *const a = lookup->a;
   const int lda = lookup->lda;
   xglobal += lookup->x_offset;
   sync += lookup->sync_offset;

#ifdef TIMING
   const unsigned int sa = clock();
#endif

   const int nblk = (n + (nb-1)) / nb;
   const int tid = threadsx*threadIdx.y + threadIdx.x;

   /* sync components:
    *    sync[0] => nblk - Last ready column [init to -1]
    *    sync[1] => nblk - Next row to assign [init to 0]
    */

   volatile T_ELEM __shared__ partSum[threadsy*threadsx];
   volatile T_ELEM __shared__ cache[nb*nb];
   T_ELEM regcache[nb/threadsy];
   T_ELEM ps[nb/threadsy];

   /* Get row handled by this block */
   const int row = nblk-1 - nextRow(&sync[1]);

   const bool short_row = ((n-1)/nb==row && n%nb!=0); /* requires special handling */

   if (row!=nblk-1) {
     const T_ELEM *const aval = &a[(row*nb+threadIdx.x)*lda+(row+1)*nb+threadIdx.y];
#pragma unroll
     for (int j=0; j<nb; j+=threadsy)
       regcache[j/threadsy] = aval[j];
   }

   /* Copy diagonal block to shared memory */
   if (!short_row) {
     /* on a block row of full size */
#ifdef INV_AFTER
     if (nblk-1-row >= INV_AFTER) {
       tocache <T_ELEM,nb,threadsx*threadsy,false,ISUNIT> (tid, &a[row*nb*lda+row*nb], lda, cache);
     } else {
       tocache <T_ELEM,nb,threadsx*threadsy,true,ISUNIT> (tid, &a[row*nb*lda+row*nb], lda, cache);
     }
#else /* INV_AFTER */
     tocache <T_ELEM,nb,threadsx*threadsy,true,ISUNIT> (tid, &a[row*nb*lda+row*nb], lda, cache);
#endif /* INV_AFTER */
   } else {
     /* on last row, smaller than full blkSize */
#ifdef INV_AFTER
     if (nblk-1-row>=INV_AFTER) {
       tocache_small <T_ELEM,nb,threadsx*threadsy,false,ISUNIT> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
     } else {
       tocache_small <T_ELEM,nb,threadsx*threadsy,true,ISUNIT> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
     }
#else /* INV_AFTER */
     tocache_small <T_ELEM,nb,threadsx*threadsy,true,ISUNIT> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
#endif /* INV_AFTER */
   }
   __syncthreads();

#ifdef INV_AFTER
   if (nblk-1-row>=INV_AFTER)
     invert<T_ELEM, nb, nb, threadsx, threadsy, ISUNIT, true>(cache, partSum);
#endif /* INV_AFTER */

   /* Loop over blocks as they become available */
   volatile T_ELEM __shared__ soln[nb];
   if (threadIdx.y==0) {
     if (!short_row) {
       soln[threadIdx.x] = xglobal[int(row*nb+threadIdx.x)];
     } else {
       if (threadIdx.x<n%nb)
         soln[threadIdx.x] = xglobal[int(row*nb+threadIdx.x)];
       else
         soln[threadIdx.x] = 0;
     }
   }
#pragma unroll
   for(int j=0; j<nb/threadsy; ++j) ps[j] = 0;
   int col_done = -1;
   for (int col=nblk-1; col>row+1; --col) {
     /* apply update from block (row, col) */
     const T_ELEM *const aval = &a[(row*nb+threadIdx.y)*lda + col*nb+threadIdx.x];
     T_ELEM *const xg = &(xglobal[int(col*nb)]);
     wait_until_ge(tid, &sync[0], nblk-1-col, &col_done); // Wait for diagonal block to be done
     T_ELEM xl;
     if (col<nblk-1) {
       xl = *(const volatile T_ELEM*)&(xg[int(threadIdx.x)]);
     } else {
       if (threadIdx.x<(n-1)%nb+1) xl = *(const volatile T_ELEM*)&(xg[int(threadIdx.x)]);
       else                        xl = 0;
     }
     if (nb % threadsy == 0) {
       if (col!=nblk-1 || n%nb==0) {
#pragma unroll
         for (int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
           ps[j/threadsy] += aval[j*lda] * xl;
       } else {
         for (int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
           if (threadIdx.x<n%nb) ps[j/threadsy] += aval[j*lda] * xl;
       }
     } else {
#pragma unroll
       for (int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
         if (j+threadIdx.y<nb)
           ps[j/threadsy] += aval[j*lda] * xl;
     }
   }
   T_ELEM val = 0;
#pragma unroll
   for (int i=0; i<nb; i+=threadsy) {
     partSum[threadIdx.x*threadsy+threadIdx.y] = ps[i/threadsy];
     __syncthreads();
     if (threadIdx.y==0 && threadIdx.x>=i && threadIdx.x<i+threadsy) {
       for (int j=0; j<nb; ++j)
         val += partSum[(threadIdx.x-i)+threadsy*j];
     }
     __syncthreads();
   }
   if (row!=nblk-1) {
     /* apply update from block (row, col) */
     const int col = row+1;
     T_ELEM *const xg = &(xglobal[int(col*nb)]);
     wait_until_ge(tid, &sync[0], nblk-1-col, &col_done); // Wait for diagonal block to be done
     T_ELEM __shared__ xlocal[nb];
     T_ELEM *const xl = xlocal+threadIdx.y;
     if (col<nblk-1) {
       if (tid<nb) xlocal[tid] = *(const volatile T_ELEM*)&(xg[int(tid)]);
       __syncthreads();
#pragma unroll
       for (int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
         val += regcache[j/threadsy] * xl[j];
     } else {
       if (tid<(n-1)%nb+1) xlocal[tid] = *(const volatile T_ELEM*)&(xg[int(tid)]);
       __syncthreads();
#pragma unroll
       for (int j=0; j<(n-1)%nb+1; j+=threadsy) // do j=0,nb-1,threadsy
         if (j+threadIdx.y<(n-1)%nb+1) val += regcache[j/threadsy] * xl[j];
     }
   }
   partSum[threadIdx.y*threadsx+threadIdx.x] = val;
   __syncthreads();
   if (threadIdx.y==0) {
     for (int i=1; i<threadsy; ++i)
       val += partSum[i*threadsx+threadIdx.x];
     val = soln[threadIdx.x]-val;
   }

   /* Apply update from diagonal block (row, row) */
#ifdef INV_AFTER
   if (nblk-1-row>=INV_AFTER) {
     T_ELEM __shared__ xshared[nb];
     slvinv_trans<T_ELEM, nb, threadsy>(cache, xshared, val, partSum, row);
     if (!short_row || threadIdx.x<n%nb) {
       if (threadIdx.y==0) {
         xglobal[int(row*nb+tid)] = val;
       }
     }
   } else {
     if (threadIdx.y==0) {
       dblkSolve_trans<T_ELEM,nb,ISUNIT>(cache, nb, val);
       if (!short_row || threadIdx.x<n%nb) {
         xglobal[int(row*nb+tid)] = val;
       }
     }
   }
#else /* INV_AFTER */
   if (threadIdx.y==0) {
     dblkSolve_trans<T_ELEM,nb,ISUNIT>(cache, nb, val);
     if (!short_row || threadIdx.x<n%nb) {
       xglobal[int(row*nb+tid)] = val;
     }
   }
#endif /* INV_AFTER */
   /* Notify other blocks that soln is ready for this row */
   __threadfence_system(); // Wait for xglobal to be visible to other blocks
   if (tid==0) atomicAdd(&sync[0],1); // Use atomicAdd to bypass L1 miss
   __threadfence_system(); // Flush sync[0] asap

#ifdef TIMING
   const unsigned int en = clock();
   if (threadIdx.x==0 && threadIdx.y==0) {
     times += blockIdx.x;
     times->sm = getSM();
     times->sa = sa;
     times->en = en;
   }
#endif
}


/**
 * \brief Performs batched trsv for Non-transposed Lower-triangular matrices
 * \details Requires trsv_init() to be called first to initialize sync[].
 * \tparam T_ELEM underlying data type, e.g. double.
 * \tparam nb block size.
 * \tparam threadsx compile-time version of blockDim.x.
 * \tparam threadsy compile-time version of blockDim.y.
 * \tparam ISUNIT true if diagonal is 1.0, false otherwise.
 * \param xglobal x array to offset into.
 * \param sync sync array to offset into.
 * \param lookup batch lookup array.
 */
template <typename T_ELEM, unsigned int nb, unsigned int threadsx, unsigned int threadsy, bool ISUNIT>
#ifndef DOXYGEN_SHOULD_SKIP_THIS
__launch_bounds__(threadsx*threadsy, 4)
#endif /* DOXYGEN_SHOULD_SKIP_THIS */
/* Note: setting above occupany to 5 causes random errors on large problems:
   suspect compiler bug */
void __global__ trsv_ln_exec(T_ELEM *__restrict__ xglobal, int *__restrict__ sync, struct trsv_lookup *lookup)
{
   lookup += blockIdx.x;
   const int n = lookup->n;
   const T_ELEM *const a = lookup->a;
   const int lda = lookup->lda;
   xglobal += lookup->x_offset;
   sync += lookup->sync_offset;
   const int incx=1;

   const int tid = threadsx*threadIdx.y + threadIdx.x;

   /* sync components:
    *    sync[0] => Last ready column [init to -1]
    *    sync[1] => Next row to assign [init to 0]
    */

   T_ELEM __shared__ partSum[threadsy*threadsx];
   T_ELEM __shared__ cache[nb*nb];
   T_ELEM __shared__ xlocal[nb];
   T_ELEM regcache[nb/threadsy];

   if (incx<0) xglobal+=(1-n)*incx;

   /* Get row handled by this block */
   const int row = nextRow(&sync[1]);

   const bool short_row = ((n-1)/nb==row && n%nb!=0); /* requires special handling */

   if (row!=0) {
     const T_ELEM *const aval = &a[((row-1)*nb+threadIdx.y)*lda+row*nb+threadIdx.x];
#pragma unroll
     for (int j=0; j<nb; j+=threadsy)
       regcache[j/threadsy] = aval[j*lda];
   }

   /* Copy diagonal block to shared memory */
   if (!short_row) {
     /* on a block row of full size */
     tocache <T_ELEM,nb,threadsx*threadsy,false,ISUNIT> (tid, &a[row*nb*lda+row*nb], lda, cache);
   } else {
     /* on last row, smaller than full blkSize */
     tocache_small <T_ELEM,nb,threadsx*threadsy,false,ISUNIT> (n%nb, tid, &a[row*nb*lda+row*nb], lda, cache);
   }
   __syncthreads();

#ifdef INV_AFTER
   if (row>=INV_AFTER)
     invert<T_ELEM, nb, nb, threadsx, threadsy, ISUNIT, false>(cache, partSum);
#endif /* INV_AFTER */

   /* Loop over blocks as they become available */
   T_ELEM val = 0;
   if (threadIdx.y==0) {
     if (!short_row) {
       val = -xglobal[int(row*nb+threadIdx.x)*incx];
     } else {
       if (threadIdx.x<n%nb) val = -xglobal[int(row*nb+threadIdx.x)*incx];
     }
   }
   int col_done = -1;
   for (int col=0; col<row-1; ++col) {
     /* apply update from block (row, col) */
     const T_ELEM *const aval = &a[(col*nb+threadIdx.y)*lda + row*nb+threadIdx.x];
     T_ELEM *const xg = &(xglobal[int(col*nb)*incx]);
     wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
     T_ELEM *const xl = xlocal+threadIdx.y;
     if (tid<nb) xlocal[tid] = *(const volatile T_ELEM*)&(xg[int(tid)*incx]);
     __syncthreads();
     if (nb % threadsy == 0) {
#pragma unroll
       for (int j=0; j<nb; j+=threadsy)
         val += aval[j*lda] * xl[j];
     } else {
#pragma unroll
       for (int j=0; j<nb; j+=threadsy)
         if(j+threadIdx.y<nb) val += aval[j*lda] * xl[j];
     }
   }
   if (row!=0) {
     const int col = row-1;
     /* apply update from block (row, col) */
     T_ELEM *const xg = &(xglobal[int(col*nb)*incx]);
     wait_until_ge(tid, &sync[0], col, &col_done); // Wait for diagonal block to be done
     T_ELEM *const xl = xlocal+threadIdx.y;
     if (tid<nb) xlocal[tid] = *(const volatile T_ELEM*)&(xg[int(tid)*incx]);
     __syncthreads();
#pragma unroll
     for (int j=0; j<nb; j+=threadsy) // do j=0,nb-1,threadsy
       val += regcache[j/threadsy] * xl[j];
   }
   partSum[threadIdx.y*threadsx+threadIdx.x] = val;
   __syncthreads();
   if (threadIdx.y==0) {
     for (int i=1; i<threadsy; ++i)
       val += partSum[i*threadsx+threadIdx.x];
     val = -val;
     if (short_row && threadIdx.x>=n%nb) val = 0.0;
   }

   /* Apply update from diagonal block (row, row) */
#ifdef INV_AFTER
   if (row>=INV_AFTER) {
     slvinv<T_ELEM, nb, threadsy>(cache, xlocal, val, partSum);
     if (!short_row || threadIdx.x<n%nb) {
       if (threadIdx.y==0) {
         xglobal[int(row*nb+tid)*incx] = val;
       }
     }
   } else {
     if (threadIdx.y==0) {
       dblkSolve<T_ELEM,nb,ISUNIT>(cache, nb, val);
       if (!short_row || threadIdx.x<n%nb) {
         xglobal[int(row*nb+tid)*incx] = val;
       }
     }
   }
#else /* INV_AFTER */
   if (threadIdx.y==0) {
     dblkSolve<T_ELEM,nb,ISUNIT>(cache, nb, val);
     if (!short_row || threadIdx.x<n%nb) {
       xglobal[int(row*nb+tid)*incx] = val;
     }
   }
#endif /* INV_AFTER */
   /* Notify other blocks that soln is ready for this row */
   __threadfence(); // Wait for xglobal to be visible to other blocks
   if (tid==0) atomicAdd(&sync[0],1); // Use atomicAdd to bypass L1 miss
   __threadfence(); // Flush sync[0] asap
}

}}} /* namespace spral::ssids::gpu */
