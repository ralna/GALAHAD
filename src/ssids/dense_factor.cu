/* Copyright (c) 2013 Science and Technology Facilities Council (STFC)
 * Authors: Evgueni Ovtchinnikov and Jonathan Hogg
 *
 * This file contains CUDA kernels for partial LL^T and LDL^T factorization
 * of dense submatrices.
 */

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

#define FAVOUR2x2 100
#define CBLOCKS 3
#define MCBLOCKS 8
#define BLOCKS 7
#define MBLOCKS 11
#define BLOCK_SIZE 8

#define MAX_CUDA_BLOCKS 65535

using namespace spral::ssids::gpu;

namespace /* anon */ {

extern __shared__ volatile double SharedMemory[];

__global__ void
cu_block_ldlt_init(
    const int ncols,
    int *const stat,
    int *const ind
) {
  if (threadIdx.x == 0) {
    stat[0] = ncols; // successful pivots
    stat[1] = 0;
  }
  if (threadIdx.x < ncols) ind[threadIdx.x] = ncols + 1;
}

template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_init_chol_fact(
    const unsigned int block,
    const int nrows, // number of rows of the factorized matrix
    const int ncols, // number of columns thereof
    const ELEMENT_TYPE *const a, // array of elements of A
    const int lda, // leading dimension of a
    volatile ELEMENT_TYPE *const fs // initial L factor (shared mem)
) {
  const int SIZE_X = TILES*TILE_SIZE;

  int x; // row index

  for ( int tile = 0; tile < TILES; tile++ ) {
    if ( tile ) { // load A's offdiagonal tiles into shared memory
      x = ncols + threadIdx.x + (tile - 1)*TILE_SIZE +
        (TILES - 1)*TILE_SIZE*block; // offdiagonal row index in A
      fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y] =
        ( x < nrows && threadIdx.y < ncols ) ?
          a[x + lda*threadIdx.y] : 0.0;
    }
    else { // load the diagonal (pivot) tile
      fs[threadIdx.x + SIZE_X*threadIdx.y] =
        ( threadIdx.x < ncols && threadIdx.y < ncols ) ?
          a[threadIdx.x + lda*threadIdx.y] : 0.0;
    }
  }

}

template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_save_chol_fact(
    const unsigned int block,
    const int nrows, // number of rows of the factorized matrix
    const int ncols, // number of columns thereof
    const volatile ELEMENT_TYPE *const fs, // initial L factor (shared mem)
    ELEMENT_TYPE *const f, // array of elements of L
    const int ldf // leading dimension of f
) {
  const int SIZE_X = TILES*TILE_SIZE;

  int x; // row index

  for ( int tile = 0; tile < TILES; tile++ ) {
    if ( tile ) { // upload the relevant elements of fs to f
      x = ncols + threadIdx.x + (tile - 1)*TILE_SIZE +
        (TILES - 1)*TILE_SIZE*block;
      if ((x < nrows) && (threadIdx.y < ncols))
        f[x + ldf*threadIdx.y] =
          fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y];
    }
    else if ( block == 0 ) {
      // upload to f and fd
      if ( threadIdx.x < ncols && threadIdx.y < ncols )
        f[threadIdx.x + ldf*threadIdx.y] =
          fs[threadIdx.x + SIZE_X*threadIdx.y];
    }
  } // loop through tiles ends here
}

template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_block_chol(
    const int block,
    const int nrows,
    const int ncols,
    const ELEMENT_TYPE *const a,
    const int lda,
    ELEMENT_TYPE *const f,
    const int ldf,
    int *const stat
) {
  const int SIZE_X = TILES * TILE_SIZE;

  int ip;
  ELEMENT_TYPE v;

  volatile ELEMENT_TYPE *const work = (volatile ELEMENT_TYPE*)SharedMemory;

  // load A into shared memory
  dev_init_chol_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, a, lda, work );
  __syncthreads();

  for (ip = 0; ip < ncols; ip++) {

    v = work[ip + SIZE_X*ip];
    if ( v <= 0.0 ) {
      if ((block == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
        stat[0] = ip;
      return;
    }

    v = sqrt(v);
    __syncthreads();

    if (threadIdx.y < TILES)
      work[threadIdx.x + TILE_SIZE*threadIdx.y + SIZE_X*ip] /= v;
    __syncthreads();

    if ((threadIdx.y > ip) && (threadIdx.y < ncols)) {
      for (int x = threadIdx.x + TILE_SIZE; x < SIZE_X; x += TILE_SIZE)
        work[x + SIZE_X*threadIdx.y] -=
          work[threadIdx.y + SIZE_X*ip] * work[x + SIZE_X*ip];
      if (threadIdx.x > ip)
        work[threadIdx.x + SIZE_X*threadIdx.y] -=
          work[threadIdx.y + SIZE_X*ip] * work[threadIdx.x + SIZE_X*ip];
    }
    __syncthreads();

  }
  if ((block == 0) && (threadIdx.x == 0) && (threadIdx.y == 0))
    stat[0] = ncols;

  // save the L factor
  dev_save_chol_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, work, f, ldf );
}

template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__global__ void
cu_block_chol(
    const int nrows,
    const int ncols,
    const ELEMENT_TYPE *const a,
    const int lda,
    ELEMENT_TYPE *const f,
    const int ldf,
    int *const stat
) {
  dev_block_chol< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( blockIdx.x, nrows, ncols, a, lda, f, ldf, stat );
}

struct multinode_chol_type {
   int nrows;
   int ncols;
   double *lcol;
};

// input data type for multiblock_fact and multiblock_chol
// each CUDA block gets a copy
struct multiblock_fact_type {
   int nrows; // no node's rows
   int ncols; // no node's cols
   int ld;    // node's leading dimension
   int p;     // no rows above the pivot block
   double *aptr; // pointer to this node's A matrix
   double *ldptr; // pointer to this node's LD matrix
   int offf;  // this node's L offset in the array of all Ls
   double *dptr; // pointer to this node's D in array of all Ds
   int node;  // node index
   int offb;  // the idx of the first CUDA block processing this node
};

__global__ void
cu_multiblock_fact_setup(
    struct multinode_fact_type *ndata,
    struct multiblock_fact_type *const mbfdata,
    const int step,
    const int block_size,
    const int blocks,
    const int offb,
    int *const stat,
    int *const ind,
    int *const nl
) {
  ndata += blockIdx.x;
  const int ncols = ndata->ncols;
  const int nrows = ndata->nrows;
  double *const lval  = ndata->lval;
  double *const ldval = ndata->ldval;
  double *const dval  = ndata->dval;
  int ib    = ndata->ib;
  int jb    = ndata->jb;
  int done  = ndata->done;
  int rght  = ndata->rght;
  const int lbuf  = ndata->lbuf;

  if (jb < ib)
    return;

  const int pivoted = stat[blockIdx.x];

  if (pivoted > 0) {
    done += pivoted;
    if (jb == rght)
      jb = done;
  }

  if (jb <= ncols)
    ib = jb + 1;

  __syncthreads();
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    ndata->ib = ib;
    ndata->jb = jb;
    ndata->done = done;
  }

  if (ib > ncols)
    return;

  if (ib > rght) {
    rght += step;
    if (rght > ncols)
      rght = ncols;
    if ((threadIdx.x == 0) && (threadIdx.y == 0))
      ndata->rght = rght;
  }

  const int rb = nrows - done;
  int cb = rght - ib + 1;

  if (cb > block_size)
    cb = block_size;

  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    ndata->jb = jb + cb;
    stat[blockIdx.x] = cb; // successful pivots
  }
  if (ind && (threadIdx.x < cb) && (threadIdx.y == 0))
    ind[blockIdx.x*block_size + threadIdx.x] = cb + 1;

  int k = (rb - cb - 1)/(block_size*(blocks - 1)) + 1;

  __shared__ volatile int ncb;
  if ((threadIdx.x == 0) && (threadIdx.y == 0))
    ncb = atomicAdd(&nl[0], k);

  __shared__ volatile int iwork[9];
  __shared__ double *volatile lptr, *volatile ldptr, *volatile dptr;
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    iwork[0] = cb;
    iwork[1] = rb;
    iwork[2] = nrows;
    iwork[3] = ib - done - 1;
    lptr = lval + done + (ib - 1)*nrows;
    ldptr = ldval + done + (ib - 1)*nrows;
    iwork[5] = lbuf + done;
    dptr = dval + 2*done;
    iwork[7] = offb + blockIdx.x;
    iwork[8] = ncb;
  }
  __syncthreads();

  for (int i = threadIdx.y; i < k; i += blockDim.y) {
    switch(threadIdx.x) {
    case 0: mbfdata[ncb+i].ncols = iwork[0]; break;
    case 1: mbfdata[ncb+i].nrows = iwork[1]; break;
    case 2: mbfdata[ncb+i].ld    = iwork[2]; break;
    case 3: mbfdata[ncb+i].p     = iwork[3]; break;
    case 4: mbfdata[ncb+i].aptr = lptr;
            mbfdata[ncb+i].ldptr = ldptr;    break;
    case 5: mbfdata[ncb+i].offf  = iwork[5]; break;
    case 6: mbfdata[ncb+i].dptr = dptr;      break;
    case 7: mbfdata[ncb+i].node  = iwork[7]; break;
    case 8: mbfdata[ncb+i].offb  = i;        break;
    }
  }

}

////////////////////////////////////////////////////////////////////////////

/*

Functions below participate in the LDLT factorization

         |    A_u P|   |L_u|
Q A P  = |P^T A_d P| = |L_d| * D * (L_d)^T = L * D * (L_d)^T        (LDLT)
         |    A_l P|   |L_l|

where A is nrows x ncols, P is a ncols x ncols permutation matrix,

    |I_u        |
Q = |    P^T    |, where I_u and I_l are identities,
    |        I_l|

L_d is a ncols x ncols lower triangular matrix with unit main diagonal
and D is a ncols x ncols block diagonal matrix with 1x1 and 2x2 blocks
on the main diagonal.

Common variable names:

nrow        number of rows in A/L
ncols       numbre of columns in A/L
offp        number of rows in A_u

*/

////////////////////////////////////////////////////////////////////////////

/*

The next function initializes L and the main diagonal and subdiagonal
of D**(-1).

L and L*D are stored in two shared memory arrays fs and fds, each
arranged into TILES square tiles of size TILE_SIZE. The kernel for
factorizing just one node uses TILES = 7, and the one for simultaneous
factorization of several nodes uses TILES = 11.

Each CUDA block uses dev_init_fact to load A_d into the first tile of fs
and up to (TILES - 1)*TILE_SIZE rows of A_u and A_l into the remaining
TILES - 1 tiles.

The two diagonals of D**(-1) are stored in a shared memory array
of size 2*TILE_SIZE, initialized to 0 by this kernel.

*/
template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_init_fact(
    const unsigned int block, // relative CUDA block number
    const int nrows,
    const int ncols,
    const int offp,
    const ELEMENT_TYPE *const a, // array of elements of A
    const int lda, // leading dimension of a
    volatile ELEMENT_TYPE *const fs, // initial L factor (shared mem)
    volatile ELEMENT_TYPE *const ds // initial D**(-1) (shared mem)
) {
  const int SIZE_X = TILES * TILE_SIZE;

  int x, y; // position indices

  y = threadIdx.y % TILE_SIZE; // fs & fds column processed by this thread

  if ( threadIdx.y < TILE_SIZE ) {
    for ( int tile = 0; tile < TILES; tile += 2 ) {
      if ( tile ) { // load A_u and A_l's even tiles into shared memory
        x = threadIdx.x + (tile - 1)*TILE_SIZE +
            (TILES - 1)*TILE_SIZE*block; // offdiagonal row index in A
        if ( x >= offp )
          x += ncols; // skip A_d
        fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y] =
          ( x < nrows && threadIdx.y < ncols ) ?
            a[x + lda*threadIdx.y] : 0.0;
      }
      else { // load A_d
        fs[threadIdx.x + SIZE_X*threadIdx.y] =
          ( threadIdx.x < ncols && threadIdx.y < ncols ) ?
            a[offp + threadIdx.x + lda*threadIdx.y] : 0.0;
      }
    }
  }
  else {
    // load A_u and A_l's odd tiles into shared memory
    for (int tile = 1; tile < TILES; tile += 2) {
      x = threadIdx.x + (tile - 1)*TILE_SIZE +
        (TILES - 1)*TILE_SIZE*block;
      if (x >= offp)
        x += ncols;
      fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*y] =
        ((x < nrows) && (y < ncols)) ? a[x + lda*y] : 0.0;
    }
  }
  // main diagonal and subdiagonal of D**(-1) set to 0
  if (threadIdx.y < 2)
    ds[2*threadIdx.x + threadIdx.y] = 0.0;

}

/* The next function uploads L, L*D and D to global memory */

template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_save_fact(
    const unsigned int block,
    const int nrows,
    const int ncols,
    const int offp,
    const int my, // save only if my is non-zero
    const volatile ELEMENT_TYPE *const fs, // L (shared mem)
    const volatile ELEMENT_TYPE *const fds, // L*D (shared mem)
    const volatile ELEMENT_TYPE *const ds, // 2 diags of D**(-1) (shared mem)
    ELEMENT_TYPE *const f, // L (global mem)
    const int ldf, // leading dimension of f
    ELEMENT_TYPE *const fd, // L*D (global mem)
    const int ldfd, // leading dimension of fd
    ELEMENT_TYPE *const d // 2 diags of D**(-1) (global mem)
) {
  const int SIZE_X = TILES * TILE_SIZE;

  int x, y; // position indices

  y = threadIdx.y % TILE_SIZE; // fs & fds column processed by this thread

  if ( threadIdx.y < TILE_SIZE ) { // warps 0, 1
    for ( int tile = 0; tile < TILES; tile += 2 ) {
      if ( tile ) { // upload L_u, L_l, L_u*D and L_l*D's even tiles
        x = threadIdx.x + (tile - 1)*TILE_SIZE +
            (TILES - 1)*TILE_SIZE*block;
        if ( x >= offp ) // skip L_d
          x += ncols;
        if ( x < nrows && threadIdx.y < ncols && my ) {
          f[x + ldf*threadIdx.y] =
            fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y];
          fd[x + ldfd*threadIdx.y] =
            fds[threadIdx.x + tile*TILE_SIZE + SIZE_X*threadIdx.y];
        }
      }
      else if ( block == 0 ) {
        // upload L_d and L_d*D
        if ( threadIdx.x < ncols && threadIdx.y < ncols && my ) {
          f[offp + threadIdx.x + ldf*threadIdx.y] =
            fs[threadIdx.x + SIZE_X*threadIdx.y];
          fd[offp + threadIdx.x + ldfd*threadIdx.y] =
            fds[threadIdx.x + SIZE_X*threadIdx.y];
        }
        // upload D**(-1)
        if ( threadIdx.x < 2 && threadIdx.y < ncols )
          d[threadIdx.x + 2*threadIdx.y] = ds[threadIdx.x + 2*threadIdx.y];
      }
    } // loop through even tiles ends here
  }
  else { // upload L_u, L_l, L_u*D and L_l*D's odd tiles (warps 2, 3)
    for (int tile = 1; tile < TILES; tile += 2) {
      x = threadIdx.x + (tile - 1)*TILE_SIZE +
        (TILES - 1)*TILE_SIZE*block;
      if (x >= offp) // skip L_d
        x += ncols;
      if ((x < nrows) && (y < ncols) && my) {
        f[x + ldf*y] = fs[threadIdx.x + tile*TILE_SIZE + SIZE_X*y];
        fd[x + ldfd*y] = fds[threadIdx.x + tile*TILE_SIZE + SIZE_X*y];
      }
    }
  }
}

/* The next function finds the largest element of the first row of A_d */

template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__device__ void
dev_init_max(
    const int ncols,
    const volatile ELEMENT_TYPE *const fs,
    const int mx, // this thread mask
    volatile int *const mask, // pivot index/mask
    volatile bool *const not_max, // "not largest" flag
    volatile int &jps, // the index of the largest element
    volatile int &quit // pivoting failure flag
) {
  const int SIZE_X = TILES*TILE_SIZE;

  if (threadIdx.y == 0) {
    mask[threadIdx.x] = mx; // initialize the pivot index
    not_max[threadIdx.x] = mx; // initialize the "not largest" flag
  }
  if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
    jps = TILE_SIZE; // initialize pivot col jp: cf the case of a tie below
    quit = 0; // initialize failure flag
  }
  __syncthreads();

  // check if the element in the column threadIdx.x
  // of the first row is (one of) the largest one(s)
  if ((threadIdx.x < ncols) && (threadIdx.y < ncols) &&
      (threadIdx.x != threadIdx.y) &&
      (fabs(fs[SIZE_X*threadIdx.x]) < fabs(fs[SIZE_X*threadIdx.y])))
    not_max[threadIdx.x] = 1; // no good: a larger value exists elsewhere
  __syncthreads();

  // select the leftmost among the largest elements of the row
  if ((threadIdx.y == 0) && (not_max[threadIdx.x] == 0))
    atomicMin((int*)&jps, threadIdx.x); // in case of a tie, choose the leftmost
  __syncthreads();
}

/*

The next function selects pivot based on the pending row number ip
and the column number for the largest element in this row.

Three options are considered:

(1) use 1x1 pivot a11 = fs[ip + ld*ip],

(2) use 1x1 pivot a22 = fs[jp + ld*jp],

(3) use 2x2 pivot

 | a_11 a_12 |
 | a_12 a_22 |,

where a12 = fs[ip + ld*jp].

The pivot that has the smallest inverse is selected.

*/

template< typename ELEMENT_TYPE >
__device__ void
dev_select_pivots_at_root(
    const ELEMENT_TYPE *const fs,
    const int ld, // leading dimension of fs
    int &ip,
    int &jp,
    ELEMENT_TYPE &a11,
    ELEMENT_TYPE &a12,
    ELEMENT_TYPE &a22,
    ELEMENT_TYPE &det
) {
  // select the pivot based on the row's largest element index
  if (ip != jp) { // choose between 1x1 and 2x2 pivots
    a11 = fs[ip + ld*ip];
    a12 = fs[ip + ld*jp];
    a22 = fs[jp + ld*jp];
    det = a11*a22 - a12*a12; // determinant of 2x2 pivot stored in det
    if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a11) > fabs(det) ) {
      if (fabs(a11) > fabs(a22) ) { // choose the best 1x1 alternative
        jp = ip; // select a11
        det = a11; // pivot value stored in det
      }
      else {
        ip = jp; // select a22
        det = a22; // pivot value stored in det
      }
    }
    else if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a22) > fabs(det) ) {
      ip = jp; // select a22
      det = a22; // pivot value stored in det
    }
  }
  else
    det = fs[ip + ld*ip]; // pivot value stored in det
}

template< typename ELEMENT_TYPE >
__device__ void
dev_select_pivots(
    const volatile ELEMENT_TYPE *const fs,
    const int ld, // leading dimension of fs
    int &ip,
    int &jp,
    ELEMENT_TYPE &a11,
    ELEMENT_TYPE &a12,
    ELEMENT_TYPE &a22,
    ELEMENT_TYPE &det
) {
  // select the pivot based on the row's largest element index
  if (ip != jp) { // choose between 1x1 and 2x2 pivots
    a11 = fs[ip + ld*ip];
    a12 = fs[ip + ld*jp];
    a22 = fs[jp + ld*jp];
    det = a11*a22 - a12*a12; // determinant of 2x2 pivot stored in det
    if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a11) > FAVOUR2x2*fabs(det) ) {
      if ( fabs(a11) > fabs(a22) ) { // choose the best 1x1 alternative
        jp = ip; // select a11
        det = a11; // pivot value stored in det
      }
      else {
        ip = jp; // select a22
        det = a22; // pivot value stored in det
      }
    }
    else if ( (fabs(a12) + fabs(a11) + fabs(a22))*fabs(a22) > FAVOUR2x2*fabs(det) ) {
      ip = jp; // select a22
      det = a22; // pivot value stored in det
    }
  }
  else
    det = fs[ip + ld*ip]; // pivot value stored in det
}

/* The next function tries to apply 1x1 pivot. */

template< typename ELEMENT_TYPE >
__device__ bool
dev_1x1_pivot_fails(
    const int x,
    const int ip,
    volatile ELEMENT_TYPE *const fs,
    volatile ELEMENT_TYPE *const fds,
    const int ld,
    const ELEMENT_TYPE det,
    const ELEMENT_TYPE delta,
    const ELEMENT_TYPE eps
) {
  // the column of fds is that of fs before the division by pivot
  const ELEMENT_TYPE u = fds[x + ld*ip] = fs[x + ld*ip];
  if ( fabs(det) <= eps ) { // the pivot is considered to be zero
    if ( fabs(u) <= eps ) { // the off-diagonal is considered to be zero
      if ( x == ip )
        fs[x + ld*ip] = 1.0;
      else
        fs[x + ld*ip] = 0.0;
    }
    else {      // non-zero off-diagonal element found ->
      return 1; // this column to be delayed
    }
  }
  else if ( fabs(det) <= delta*fabs(u) ) // pivot too small ->
    return 1; // this column to be delayed
  else
    fs[x + ld*ip] = u/det; // ok to divide
  return 0;
}

/* The next function tries to apply 1x1 pivot. */

template< typename ELEMENT_TYPE >
__device__ bool
dev_2x2_pivot_fails(
    const int x,
    const int ip,
    const int jp,
    volatile ELEMENT_TYPE *const fs,
    volatile ELEMENT_TYPE *const fds,
    const int ld,
    const ELEMENT_TYPE a11,
    const ELEMENT_TYPE a12,
    const ELEMENT_TYPE a22,
    const ELEMENT_TYPE det,
    const ELEMENT_TYPE delta,
    const ELEMENT_TYPE eps
) {
  // the columns of fds is those of fd before division by pivot
  const ELEMENT_TYPE u = fds[x + ld*ip] = fs[x + ld*ip];
  const ELEMENT_TYPE v = fds[x + ld*jp] = fs[x + ld*jp];
  if ( fabs(det) <= fabs(a11)*fabs(a22)*1.0e-15 ||
       // the determinant is smaller than round-off errors ->
       // the pivot is considered to be zero
       fabs(det) <= eps*(fabs(a11) + fabs(a22) + fabs(a12))
       // the inverse of the pivot is of the order 1/eps ->
       // the pivot is considered to be zero
    ) {
    if ( max(fabs(u), fabs(v)) <= eps ) { // the off-diagonal is "zero"
      if ( x == ip ) {
        fs[x + ld*ip] = 1.0;
        fs[x + ld*jp] = 0.0;
      }
      else if ( x == jp ) {
        fs[x + ld*ip] = 0.0;
        fs[x + ld*jp] = 1.0;
      }
      else {
        fs[x + ld*ip] = 0.0;
        fs[x + ld*jp] = 0.0;
      }
    }
    else // non-zero off-diagonal element found ->
      return 1; // this column to be delayed
  }
  else if ( fabs(det) <=
             delta*max(fabs(a22*u - a12*v), fabs(a11*v - a12*u)) )
             // pivot too small ->
    return 1; // this column to be delayed
  else { // ok to divide
    fs[x + ld*ip] = (a22*u - a12*v)/det;
    fs[x + ld*jp] = (a11*v - a12*u)/det;
  }
  return 0;
}

/* The next function eliminates the pivoted column from non-pivoted */

template <
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES // = 7 for a single node and = 11 for many nodes
>
__device__ void
dev_eliminate_1x1(
    int &x, // row for this thread
    const int y, // column for this thread
    const int ip, // pivoted column
    volatile ELEMENT_TYPE *const fs,
    const int ld,
    const ELEMENT_TYPE p // pivot value
) {
  if ( x != ip )
    fs[x + ld*y] -= p * fs[x + ld*ip];
  x += 2*TILE_SIZE; // move to the next tile pair
  fs[x + ld*y] -= p * fs[x + ld*ip];
  if ( TILES == 11 ) { // several nodes case
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= p * fs[x + ld*ip];
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= p * fs[x + ld*ip];
  }
}

/* The next function eliminates the two pivoted columns from non-pivoted */

template< typename ELEMENT_TYPE,
unsigned int TILE_SIZE, unsigned int TILES >
__device__ void
dev_eliminate_2x2(
    int &x,
    const int y,
    const int ip,
    const int jp,
    volatile ELEMENT_TYPE *const fs,
    const int ld,
    const ELEMENT_TYPE pi,
    const ELEMENT_TYPE pj
) {
  if ( x != ip && x != jp )
    fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
  x += 2*TILE_SIZE; // move to the next tile pair
  fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
  if ( TILES == 11 ) { // several nodes case
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
    x += 2*TILE_SIZE; // move to the next tile pair
    fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
  }
}

/* The next function performs elimination in one tile only */

template< typename ELEMENT_TYPE, unsigned int TILE_SIZE >
inline __device__ void
dev_eliminate(
    int &x,
    const int y,
    const int ip,
    const int jp,
    volatile ELEMENT_TYPE *const fs,
    const int ld,
    const ELEMENT_TYPE pi,
    const ELEMENT_TYPE pj
) {
  x += TILE_SIZE;
  if ( ip == jp )
      fs[x + ld*y] -= pi * fs[x + ld*ip];
  else
      fs[x + ld*y] -= pi * fs[x + ld*ip] + pj * fs[x + ld*jp];
}

/*

Performs the factorization (LDLT).

The outline of the factorization algorithm is as follows.
1. L = A
2. A diagonal block of L of size 1 or 2 is selected
3. A division of the corresponding (one or two) columns of L
   by the selected block (pivoting) is considered and
   is accepted only if the elements of the resulting
   columns are not going to be greater than the inverse
   of the "pivoting threshold" delta; otherwise kernel
   terminates.
4. If not all columns are pivoted, go to 2.

Called by cu_block_ldlt and cu_multiblock_ldlt factorization kernels.

*/
template< typename ELEMENT_TYPE,
unsigned int TILE_SIZE, unsigned int TILES >
__device__ void
dev_block_ldlt(
    const unsigned int block,
    const int nrows, // number of rows of the factorized matrix
    const int ncols, // number of columns thereof
    const int offp, // number of rows above the pivot block
    ELEMENT_TYPE *const a, // array of elements of A
    const int lda, // leading dimension of a
    ELEMENT_TYPE *const f, // array of elements of the L factor
    const int ldf, // leading dimension of f
    ELEMENT_TYPE *const fd, // array of elements of L*D
    const int ldfd, // leading dimension of fd
    ELEMENT_TYPE *const d, // array for main diagonal and subdiagonal of D
    const ELEMENT_TYPE delta, // pivoting threashold
    const ELEMENT_TYPE eps, // zero pivot threashold
    int *const index, // pivot order index
    int *const stat  // number of successful pivots
) {
  const int SIZE_X = TILES*TILE_SIZE;

  int ip, jp; // pivot row and col indices
  int x, y; // position indices
  int mx, my; // masks
  ELEMENT_TYPE a11, a12, a22, det; // 2x2 pivot data

  __shared__ volatile ELEMENT_TYPE fs[SIZE_X*TILE_SIZE]; // work array for f
  __shared__ volatile ELEMENT_TYPE fds[SIZE_X*TILE_SIZE]; // work array for fd
  __shared__ volatile ELEMENT_TYPE ds[2*TILE_SIZE]; // work array for d
  __shared__ volatile int mask[TILE_SIZE]; // pivot mask/index
  __shared__ volatile bool not_max[TILE_SIZE]; // flag for finding the largest row elm

  __shared__ volatile int quit; // failure flag
  __shared__ volatile int jps; // pivot column index

  y = threadIdx.y % TILE_SIZE; // fs & fds column processed by this thread

  // load the diagonal and off-diagonal tiles into shared memory
  dev_init_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, offp, a, lda, fs, ds );

  mx = (threadIdx.x < ncols ? 0 : ncols + 1); // initial pivot index

  // find the largest element in the first row
  dev_init_max< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( ncols, fs, mx, mask, not_max, jps, quit );

  for ( int row = 0, pivoted = 0; row < ncols; ) {

    // select the pivot based on the row's largest element index jps
    ip = row;
    jp = jps;
    dev_select_pivots< ELEMENT_TYPE >
      ( fs, SIZE_X, ip, jp, a11, a12, a22, det );
    __syncthreads();

    if ( threadIdx.y < TILE_SIZE + 4 ) { // the first 3 warps try to pivot

      x = threadIdx.x + TILE_SIZE*threadIdx.y; // fs/fds row to process
      if (  x < SIZE_X && (threadIdx.y || mx == 0 || mx > ncols) ) {
                       // elements of the pivot block that should have been
                       // zeroed by elimination are ignored
        if ( ip == jp ) { // 1x1 pivot
          if ( dev_1x1_pivot_fails< ELEMENT_TYPE >
            ( x, ip, fs, fds, SIZE_X, det, delta, eps ) )
            quit = 1;
        }
        else { // 2x2 pivot
          if ( dev_2x2_pivot_fails< ELEMENT_TYPE >
            ( x, ip, jp, fs, fds, SIZE_X, a11, a12, a22, det, delta, eps ) )
            quit = 1;
        }
      }

    }
    else { // meanwhile, one thread of the fourth warp is inverting the pivot

      if ( threadIdx.x == 0 && threadIdx.y == TILE_SIZE + 4 ) {
        mask[ip] = pivoted + 1; // assume pivot is ok for now
        if ( ip == jp ) {
          if ( fabs(det) > eps )
            ds[2*pivoted] = 1.0/det; // ok to invert
        }
        else {
          mask[jp] = pivoted + 2; // assume pivot is ok for now
          if ( fabs(det) > fabs(a11)*fabs(a22)*1.0e-15 &&
               fabs(det) > eps*(fabs(a11) + fabs(a22) + fabs(a12)) ) {
            ds[2*pivoted    ] = a22/det;
            ds[2*pivoted + 1] = -a12/det;
            ds[2*pivoted + 2] = a11/det;
          }
        }
        if ( atomicMin(&stat[0], ncols) <= pivoted )
          quit = 1; // some other CUDA block failed to pivot this column
      }

    } // warp fork ends here

    __syncthreads();
    if ( quit ) {
      if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
        atomicMin(&stat[0], pivoted); // record the failure in stat[0]
        // column(s) should not be saved - mark as non-processed
        mask[ip] = 0;
        if ( ip != jp )
          mask[jp] = 0;
      }
      __syncthreads();
      break; // done
    }

    // update successful pivots count
    if ( ip == jp )
      pivoted++;
    else
      pivoted += 2;

    // find next pivot row to process
    if ( ip == row )
      row++; // move forward only if this row participated in pivoting

    while ( row < ncols && mask[row] )
      row++; // skip processed rows (parts of previous 2x2 pivots)

    // eliminate the recently pivoted column(s) from the rest

    // first row to be processed by this thread
    x = threadIdx.x + (threadIdx.y/TILE_SIZE)*TILE_SIZE;

    mx = mask[threadIdx.x];
    my = mask[y];

    // process the first (TILES - 3) tiles right away;
    // the even tiles are processed by the first two warps,
    // the odd by the other two
    if ( ip == jp ) {
      a11 = fs[ip + SIZE_X*y];
      if ( my == 0 )
        dev_eliminate_1x1< ELEMENT_TYPE, TILE_SIZE, TILES >
          ( x, y, ip, fs, SIZE_X, a11 );
    }
    else {
      a11 = fs[ip + SIZE_X*y];
      a12 = fs[jp + SIZE_X*y];
      if ( my == 0 )
        dev_eliminate_2x2< ELEMENT_TYPE, TILE_SIZE, TILES >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }

    // from here on, the first two warps deal with finding the largest element
    // in the next pivot row, while the other two continue elimination
    // in the remaining three tiles

    if ( threadIdx.y < TILE_SIZE ) {
      if ( row < ncols && threadIdx.y == 0 ) {
        not_max[threadIdx.x] = mx; // mask away processed elements
        if ( threadIdx.x == 0 )
          jps = TILE_SIZE; // initialise the largest element column index
      }
    }
    else { // do elimination in the (TILES - 2)-th tile
      if ( my == 0 )
        dev_eliminate< ELEMENT_TYPE, TILE_SIZE >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    __syncthreads();

    if ( threadIdx.y < TILE_SIZE ) {
      // mark elements in the pending row that cannot be largest
      if ( row < ncols ) {
        // check the element in column threadIdx.x
        if ( threadIdx.x != threadIdx.y && mx == 0 && my == 0 &&
             fabs(fs[row + SIZE_X*threadIdx.x]) <
             fabs(fs[row + SIZE_X*threadIdx.y]) )
          not_max[threadIdx.x] = 1; // no good: a larger value exists elsewhere
      }
    }
    else { // do elimination in the (TILES - 1)-th tile
      if ( my == 0 )
        dev_eliminate< ELEMENT_TYPE, TILE_SIZE >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    __syncthreads();

    if ( threadIdx.y < TILE_SIZE ) {
      // select leftmost largest element in the row
      if ( row < ncols ) {
        if ( threadIdx.y == 0 && not_max[threadIdx.x] == 0 )
          atomicMin((int*)&jps, threadIdx.x); // in case of a tie, choose the leftmost
      }
    }
    else { // do elimination in the (TILES)-th tile
      if ( my == 0 )
        dev_eliminate< ELEMENT_TYPE, TILE_SIZE >
          ( x, y, ip, jp, fs, SIZE_X, a11, a12 );
    }
    __syncthreads();

  } // for loop through pivot rows ends here

  my = mask[y];

  // update successful pivot ordering in index;
  // if this CUDA block failed to pivot the part of column threadIdx.y of A
  // delegated to it, then possible successful pivoting of its other parts
  // by other blocks is canceled by zeroing index[threadIdx.y];
  // if some other part of this column is unsuccessful, index[threadIdx.y]
  // remains zero
  if ( threadIdx.x == 0 && threadIdx.y < ncols )
    atomicMin(&index[threadIdx.y], my);

  // save L and D factors and LD
  dev_save_fact< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, offp, my, fs, fds, ds, f, ldf, fd, ldfd, d );
}

template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__global__ void
cu_block_ldlt(
    const int nrows, // n.o. rows in A
    const int ncols, // n.o. cols in A (<= TILE_SIZE)
    const int offp,  // n.o. rows in A_u
    ELEMENT_TYPE *const a, // array of A's elements
    const int lda, // leading dimension of a
    ELEMENT_TYPE *const f, // array of L's elements
    const int ldf, // leading dimension of f
    ELEMENT_TYPE *const fd, // array of (L*D)'s elements
    const int ldfd, // leading dimension of fd
    ELEMENT_TYPE *const d, // array of D**(-1)'s diagonal and subdiagonal elements
    const ELEMENT_TYPE delta, // pivoting threshold
    const ELEMENT_TYPE eps, // zero column threshold:
    // the column is zeroed if all elements are <= eps
    int *const index, // pivot index (cf. permutation matrix P)
    int *const stat // n.o. successful pivots
) {
   dev_block_ldlt< ELEMENT_TYPE, TILE_SIZE, TILES >
      ( blockIdx.x, nrows, ncols, offp, a, lda, f, ldf,
         fd, ldfd, d, delta, eps, index, stat );
   return;
}

// Same as cu_block_fact but for several A's of different size simultaneously
//
// Called by multinode_ldlt factorization subroutine.
//
template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__global__ void
cu_multiblock_ldlt(
    struct multiblock_fact_type *mbfdata, // factorization data
    ELEMENT_TYPE *f, // same for L
    const ELEMENT_TYPE delta, // same as in cu_block_fact
    const ELEMENT_TYPE eps, // same as in cu_block_fact
    int *const index, // array of all pivot indices
    int *const stat // array of successful pivots' numbers
) {
   /*
    * Read information on what to do from global memory
    */
   mbfdata += blockIdx.x; // shift to the data for this CUDA block
   int ncols = mbfdata->ncols; // n.o. cols in A processed by this CUDA block
   if ( ncols < 1 )
      return;
   int nrows = mbfdata->nrows; // n.o. rows in A
   int lda   = mbfdata->ld; // leading dimension of A
   int p     = mbfdata->p; // n.o. rows in A_u
   int node  = mbfdata->node; // A's number
   int block  = mbfdata->offb; // relative CUDA block index

   f += mbfdata->offf; // shift to the array of this L elements
   double *fd = mbfdata->ldptr;
   double *a = mbfdata->aptr; // pointer to A
   double *d = mbfdata->dptr; // pointer to D**(-1)

   dev_block_ldlt < double, TILE_SIZE, TILES >
     ( block, nrows, ncols, p, a, lda, f, lda,
       fd, lda, d, delta, eps, &index[node*TILE_SIZE], &stat[node]);
}

/*
 LDLT factorization kernel for the root delays block.

 The columns which the above factorization kernels failed to pivot
 are delayed, ie left unchanged, until some other columns in the
 same node are successfully pivoted, after which pivoting of
 delayed columns is attempted again. When a factorization
 subroutine terminates, generally there still may be delayed
 columns which this subroutine cannot possibly pivot, and they
 are passed on to the parent node in the elimination tree.
 At the root node, however, this is not possible, and a special
 kernel given below is applied to delayed columns, which
 together with the respective rows now form a square block at the
 lower left corner of the root node matrix.

 The main difference between the factorization kernel below and
 those above is that the pivot is sought in the whole matrix
 because, in the above notation, blocks A_u and A_l are no
 longer present. Since this matrix may be too large to fit into
 shared memory, the kernel below works mostly in the global memory
 (shared memory is only used for finding the largest element of
 a column).
*/
template< typename ELEMENT_TYPE >
__global__ void
cu_square_ldlt(
    const int n,
    ELEMENT_TYPE *const a, // A on input, L on output
    ELEMENT_TYPE *const f, // L
    ELEMENT_TYPE *const w, // L*D
    ELEMENT_TYPE *const d, // main diag and subdiag of the inverse of D
    const int ld, // leading dimension of a, f, w
    const ELEMENT_TYPE delta, // same as above
    const ELEMENT_TYPE eps, // same as above
    int *const ind, // same as in cu_block_fact
    int *const stat // same as in cu_block_fact
) {
  int x, y;
  int col;
  int ip, jp;
  int pivoted, recent;
  ELEMENT_TYPE a11, a12, a22, det;

  volatile ELEMENT_TYPE *work = (volatile ELEMENT_TYPE*)SharedMemory; // work array
  volatile int *const iwork = (volatile int*)&(work[blockDim.x]); // integer work array
  volatile int *const iw = (volatile int*)&(iwork[blockDim.x]); // iw[0]: failure flag,
                                       // iw[1]: largest col. elem. index

  for ( x = threadIdx.x; x < n; x += blockDim.x ) {
    ind[x] = 0; // initialize pivot index/processed columns mask
    for ( y = 0; y < n; y++ )
      f[x + ld*y] = a[x + ld*y]; // copy A to L
  }
  for ( x = threadIdx.x; x < 2*n; x += blockDim.x )
    d[x] = 0.0; // initialize D
  __syncthreads();

  pivoted = 0; // n.o. pivoted cols

  for ( int pass = 0; ; pass++ ) { // failed cols are skipped until next pass

    recent = 0; // n.o. cols pivoted during this pass

    for ( col = 0; col < n; ) {

      if ( ind[col] ) {
        col++; // already pivoted, move on
        continue;
      }

      if ( threadIdx.x == 0 )
        iw[0] = 0; // initialize failure flag
      __syncthreads();

      // find the largest element in the pending column
      //
      // first, each thread finds its candidate for the largest one
      a11 = -1.0;
      y = -1;
      for ( x = threadIdx.x; x < n; x += blockDim.x ) {
        if ( ind[x] == 0 ) {
          a12 = fabs(f[x + ld*col]);
          if ( a12 >= a11 ) {
            a11 = a12;
            y = x;
          }
        }
      }
      work[threadIdx.x] = a11; // the largest one for this thread
      iwork[threadIdx.x] = y; // its index
      __syncthreads();

      // now first 8 threads reduce the number of candidates to 8
      if ( threadIdx.x < 8 ) {
        for ( x = threadIdx.x + 8; x < blockDim.x; x += 8 )
          if ( iwork[x] >= 0 && work[x] > work[threadIdx.x] ) {
            work[threadIdx.x] = work[x];
            iwork[threadIdx.x] = iwork[x];
          }
      }
      __syncthreads();
      // the first thread finds the largest element and its index
      if ( threadIdx.x == 0 ) {
        y = 0;
        for ( x = 1; x < 8 && x < blockDim.x; x++ )
          if ( iwork[x] >= 0 && (iwork[y] < 0 || work[x] > work[y]) )
            y = x;
        iw[1] = iwork[y]; // the largest element index
      }
      __syncthreads();

      // select the pivot based on the largest element index
      ip = col;
      jp = iw[1];

      dev_select_pivots_at_root< ELEMENT_TYPE >
        ( f, ld, ip, jp, a11, a12, a22, det );

      // try to pivot
      if ( ip == jp ) { // 1x1 pivot
        for ( x = threadIdx.x; x < n; x += blockDim.x )
          if ( ind[x] == 0 )
            if ( dev_1x1_pivot_fails< ELEMENT_TYPE >
              ( x, ip, f, w, ld, det, delta, eps ) )
                iw[0] = 1;
      }
      else { // 2x2 pivot
        for ( x = threadIdx.x; x < n; x += blockDim.x )
          if ( ind[x] == 0 )
            if ( dev_2x2_pivot_fails< ELEMENT_TYPE >
              ( x, ip, jp, f, w, ld, a11, a12, a22, det, delta, eps ) )
                iw[0] = 1;
      }
      __syncthreads();
      if ( iw[0] ) { // pivot failed, restore the failed column(s)
        for ( x = threadIdx.x; x < n; x += blockDim.x ) {
          if ( ind[x] )
            continue;
          f[x + ld*ip] = w[x + ld*ip];
          if ( ip != jp )
            f[x + ld*jp] = w[x + ld*jp];
        }
        __syncthreads();
        col++; // move on
        continue;
      }

      if ( threadIdx.x == 0 ) {
        // mark pivoted columns and invert the pivot if possible
        ind[ip] = pivoted + 1;
        if ( ip == jp ) {
          if ( fabs(det) > eps ) // ok to invert
            d[2*pivoted] = 1.0/det;
        }
        else {
          ind[jp] = pivoted + 2;
          if ( fabs(det) > fabs(a11)*fabs(a22)*1.0e-15 &&
               fabs(det) > eps*(fabs(a11) + fabs(a22) + fabs(a12)) ) {
            // ok to invert
            d[2*pivoted    ] = a22/det;
            d[2*pivoted + 1] = -a12/det;
            d[2*pivoted + 2] = a11/det;
          }
        }
      }
      __syncthreads();

      // update pivot counters
      if ( ip == jp ) {
        pivoted++;
        recent++;
      }
      else {
        pivoted += 2;
        recent += 2;
      }

      // eliminate pivoted columns from non-processed
      if ( ip == jp ) {
        for ( x = threadIdx.x; x < n; x += blockDim.x )
          for ( y = 0; y < n; y++ )
            if ( x != ip && ind[y] == 0 )
              f[x + ld*y] -= f[x + ld*ip] * f[ip + ld*y];
      }
      else {
        for ( x = threadIdx.x; x < n; x += blockDim.x ) {
          for ( y = 0; y < n; y++ ) {
            if ( x != ip && x != jp && ind[y] == 0 ) {
              f[x + ld*y] -= f[x + ld*ip] * f[ip + ld*y] +
                             f[x + ld*jp] * f[jp + ld*y];
            }
          }
        }
      }
      __syncthreads();

      if ( ip == col ) // this column is pivoted, move on
        col++;

    } // loop across columns

    if ( pivoted == n // all done
            ||
         recent == 0 ) // no pivotable columns left
      break;
  } // pass

  if ( threadIdx.x == 0 )
    stat[0] = pivoted;

  if ( pivoted < n ) // factorization failed
    return;

  // copy L to A
  for ( x = threadIdx.x; x < n; x += blockDim.x )
    for ( y = 0; y < n; y++ )
      a[ind[x] - 1 + ld*(ind[y] - 1)] = f[x + ld*y];

}

template
<
typename ELEMENT_TYPE,
unsigned int TILE_SIZE,
unsigned int TILES
>
__global__ void
cu_multiblock_chol(
    struct multiblock_fact_type *mbfdata,
    ELEMENT_TYPE *f, // array of L nodes
    int *stat // execution status
) {
  /*
   * Read information on what to do from global memory
   */
  mbfdata += blockIdx.x;
  int ncols = mbfdata->ncols;
  if ( ncols < 1 )
    return;
  int nrows = mbfdata->nrows;
  int ld    = mbfdata->ld;
  int node  = mbfdata->node;
  int block  = mbfdata->offb;

  ELEMENT_TYPE *const a = mbfdata->aptr;
  f += mbfdata->offf;
  stat += node;
  dev_block_chol< ELEMENT_TYPE, TILE_SIZE, TILES >
    ( block, nrows, ncols, a, ld, f, ld, stat );
}

struct cstat_data_type {
  int nelim;
  double *dval;
};

__global__ void
cu_collect_stats(
    const struct cstat_data_type *csdata,
    struct cuda_stats *const stats
) {
   // Designed to be run with a single thread
   csdata += blockIdx.x;
   double *const d = csdata->dval;
   const int nelim = csdata->nelim;

   int num_zero = 0;
   int num_neg = 0;
   int num_two = 0;

   for (int i = 0; i < nelim; ) {
      const double a11 = d[2*i];
      const double a21 = d[2*i + 1];
      if ( a21 == 0.0 ) {
         // 1x1 pivot (can be a zero pivot)
         if ( a11 == 0 )
            num_zero++;
         if ( a11 < 0 )
            num_neg++;
         i++;
      }
      else {
         // 2x2 pivot (can't be a zero pivot)
         const double a22 = d[2*(i + 1)];
         num_two++;
         // To check for negative eigenvalues, we exploit
         // det   = product of evals
         // trace = sum of evals
         // if det is negative, exactly one eval is negative;
         // otherwise, both have same sign, equal to sign of trace
         const double det = a11*a22 - a21*a21;
         const double trace = a11 + a22;
         if ( det < 0 )
            num_neg++;
         else if ( trace < 0 )
            num_neg += 2;
         i += 2;
      }
   }

   if ( num_neg > 0 )
      atomicAdd(&(stats->num_neg), num_neg);
   if ( num_zero > 0 )
      atomicAdd(&(stats->num_zero), num_zero);
   if ( num_two > 0 )
      atomicAdd(&(stats->num_two), num_two);
}

} /* anon namespace */

/*******************************************************************************
 * Following routines are exported with C binding so can be called from Fortran
 ******************************************************************************/

extern "C" {

void spral_ssids_block_ldlt(
      cudaStream_t *stream, int nrows, int ncols, int p,
      double* a, int lda,
      double* f, int ldf,
      double* fd, int ldfd,
      double* d,
      double delta, double eps,
      int* index, int* stat
      ) {

   int nblocks = (nrows - ncols - 1)/(BLOCK_SIZE*(BLOCKS - 1)) + 1;
   cu_block_ldlt_init<<< 1, BLOCK_SIZE, 0, *stream >>>( ncols, stat, index );

   dim3 threads(BLOCK_SIZE, 2*BLOCK_SIZE);
   cu_block_ldlt
      < double, BLOCK_SIZE, BLOCKS >
      <<< nblocks, threads, 0, *stream >>>
      ( nrows, ncols, p, a, lda, f, ldf, fd, ldfd, d, delta, eps, index, stat );
}

void spral_ssids_block_llt( cudaStream_t *stream, int nrows, int ncols,
      double* a, int lda, double* f, int ldf, int* stat ) {
   int smsize = CBLOCKS*BLOCK_SIZE*BLOCK_SIZE*sizeof(double);
   int nblocks = (nrows - ncols - 1)/(BLOCK_SIZE*(CBLOCKS - 1)) + 1;
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   cu_block_chol
      < double, BLOCK_SIZE, CBLOCKS >
      <<< nblocks, threads, smsize, *stream >>>
      ( nrows, ncols, a, lda, f, ldf, stat );
}

void spral_ssids_collect_stats(cudaStream_t *stream, int nblk,
      const struct cstat_data_type *csdata, struct cuda_stats *stats) {
   for(int i=0; i<nblk; i+=MAX_CUDA_BLOCKS) {
      int nb = min(MAX_CUDA_BLOCKS, nblk-i);
      cu_collect_stats <<<nb, 1, 0, *stream>>> (csdata+i, stats);
      CudaCheckError();
   }
}

void spral_ssids_multiblock_ldlt( cudaStream_t *stream, int nblocks,
      struct multiblock_fact_type *mbfdata, double* f, double delta,
      double eps, int* index, int* stat ) {
   dim3 threads(BLOCK_SIZE, 2*BLOCK_SIZE);
   for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
      int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
      cu_multiblock_ldlt
         < double, BLOCK_SIZE, MBLOCKS >
         <<< nb, threads, 0, *stream >>>
         ( mbfdata + i, f, delta, eps, index, stat );
   }
}

void spral_ssids_multiblock_ldlt_setup( cudaStream_t *stream, int nblocks,
      struct multinode_fact_type *ndata, struct multiblock_fact_type *mbfdata,
      int step, int block_size, int blocks, int* stat, int* ind, int* ncb ) {
   dim3 threads(10,8);
   for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
      int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
      cu_multiblock_fact_setup
         <<< nb, threads, 0, *stream >>>
         ( ndata + i, mbfdata, step, block_size, blocks,
         i, stat + i, ind + block_size*i, ncb );
   }
}

void spral_ssids_multiblock_llt( cudaStream_t *stream, int nblocks,
      struct multiblock_fact_type *mbfdata, double* f, int* stat ) {
   if ( nblocks < 1 )
      return;

   int smsize = MCBLOCKS*BLOCK_SIZE*BLOCK_SIZE*sizeof(double);
   dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
   for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
      int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
      cu_multiblock_chol
         < double, BLOCK_SIZE, MCBLOCKS >
         <<< nb, threads, smsize, *stream >>>
         ( mbfdata + i, f, stat );
   }
}

void spral_ssids_multiblock_llt_setup( cudaStream_t *stream, int nblocks,
      struct multinode_fact_type *ndata, struct multiblock_fact_type *mbfdata,
      int step, int block_size, int blocks, int* stat, int* ncb ) {
   dim3 threads(16,8);
   for ( int i = 0; i < nblocks; i += MAX_CUDA_BLOCKS ) {
      int nb = min(MAX_CUDA_BLOCKS, nblocks - i);
      cu_multiblock_fact_setup
         <<< nb, threads, 0, *stream >>>
         ( ndata + i, mbfdata, step, block_size, blocks, i, stat + i, 0, ncb );
   }
}

void spral_ssids_square_ldlt(
            cudaStream_t *stream,
            int n,
            double* a,
            double* f,
            double* w,
            double* d,
            int ld,
            double delta, double eps,
            int* index,
            int* stat
           )
{
  int nt = min(n, 256);
  int sm = nt*sizeof(double) + (nt + 2)*sizeof(int);
  cu_square_ldlt< double ><<< 1, nt, sm, *stream >>>
    ( n, a, f, w, d, ld, delta, eps, index, stat );
}

} // end extern "C"
