/* Copyright (c) 2013 Science and Technology Facilities Council (STFC)
 * Copyright (c) 2013 NVIDIA
 * Authors: Evgueni Ovtchinnikov (STFC)
 *          Jeremy Appleyard (NVIDIA)
 * This version: GALAHAD 5.1 - 2024-11-21 AT 09:40 GMT
 */

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "ssids_rip.hxx"
#include "ssids_gpu_kernels_datatypes.h"
#include "spral_cuda_cuda_check.h"

#ifdef REAL_32
#define loadDevToSmem_generic loadDevToSmem_generic_single
#define multisyrk_type multisyrk_type_single
#define multielm_data multielm_data_single
#define cu_multisyrk_lc_r4x4 cu_multisyrk_lc_r4x4_single
#define cu_multisyrk_r4x4 cu_multisyrk_r4x4_single
#define cu_syrk_r4x4 cu_syrk_r4x4_single
#define spral_ssids_dsyrk spral_ssids_dsyrk_single
#define spral_ssids_multidsyrk spral_ssids_multidsyrk_single
#define spral_ssids_multidsyrk_low_col spral_ssids_multidsyrk_low_col_single
#elif REAL_128
#define loadDevToSmem_generic loadDevToSmem_generic_quadruple
#define multisyrk_type multisyrk_type_quadruple
#define multielm_data multielm_data_quadruple
#define cu_multisyrk_lc_r4x4 cu_multisyrk_lc_r4x4_quadruple
#define cu_multisyrk_r4x4 cu_multisyrk_r4x4_quadruple
#define cu_syrk_r4x4 cu_syrk_r4x4_quadruple
#define spral_ssids_dsyrk spral_ssids_dsyrk_quadruple
#define spral_ssids_multidsyrk spral_ssids_multidsyrk_quadruple
#define spral_ssids_multidsyrk_low_col spral_ssids_multidsyrk_low_col_quadruple
#else
#define loadDevToSmem_generic loadDevToSmem_generic_double
#define multisyrk_type multisyrk_type_double
#define multielm_data multielm_data_double
#define cu_multisyrk_lc_r4x4 cu_multisyrk_lc_r4x4_double
#define cu_multisyrk_r4x4 cu_multisyrk_r4x4_double
#define cu_syrk_r4x4 cu_syrk_r4x4_double
#define spral_ssids_dsyrk spral_ssids_dsyrk_double
#define spral_ssids_multidsyrk spral_ssids_multidsyrk_double
#define spral_ssids_multidsyrk_low_col spral_ssids_multidsyrk_low_col_double
#endif

#define min(x,y) ((x) < (y) ? (x) : (y))
#define max(x,y) ((x) > (y) ? (x) : (y))

#define MAX_CUDA_BLOCKS 65535

//#define SM_3X (__CUDA_ARCH__ == 300 || __CUDA_ARCH__ == 350 || __CUDA_ARCH__ == 370)
//FIXME: Verify if the code for Keplers (sm_3x) is still correct for the later GPUs.
#define SM_3X (__CUDA_ARCH__ >= 300)

using namespace spral::ssids::gpu;

namespace /* anon */ {



template< ipc_ WIDTH >
inline __device__ void
loadDevToSmem_generic( volatile rpc_ *const __restrict__ as, volatile rpc_ *const __restrict__ bs,
               const rpc_* __restrict__ a, const rpc_* __restrict__ b,
               ipc_ bx, ipc_ by, ipc_ offa, ipc_ lda, ipc_ ldb,
               ipc_ n, ipc_ i, ipc_ k)
{
  switch (WIDTH) {
    case 4:
    if ( i + 3 < k ) {
      if ( threadIdx.y < 4 ) {
        ipc_ x = threadIdx.x + (threadIdx.y + bx*4)*8;
        if ( x < n ) {
          as[threadIdx.x + threadIdx.y*8     ] = a[offa + x + i*lda];
          as[threadIdx.x + threadIdx.y*8 + 32] = a[offa + x + (i + 1)*lda];
          as[threadIdx.x + threadIdx.y*8 + 64] = a[offa + x + (i + 2)*lda];
          as[threadIdx.x + threadIdx.y*8 + 96] = a[offa + x + (i + 3)*lda];
        }
      }
      else {
        ipc_ x = threadIdx.x + (threadIdx.y - 4 + by*4)*8;
        if ( x < n ) {
          bs[threadIdx.x + (threadIdx.y - 4)*8     ] = b[offa + x + i*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 32] = b[offa + x + (i + 1)*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 64] = b[offa + x + (i + 2)*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 96] = b[offa + x + (i + 3)*ldb];
        }
      }
    }
    else if ( i + 2 < k ) {
      if ( threadIdx.y < 4 ) {
        ipc_ x = threadIdx.x + (threadIdx.y + bx*4)*8;
        if ( x < n ) {
          as[threadIdx.x + threadIdx.y*8     ] = a[offa + x + i*lda];
          as[threadIdx.x + threadIdx.y*8 + 32] = a[offa + x + (i + 1)*lda];
          as[threadIdx.x + threadIdx.y*8 + 64] = a[offa + x + (i + 2)*lda];
          as[threadIdx.x + threadIdx.y*8 + 96] = 0.0;
        }
      }
      else {
        ipc_ x = threadIdx.x + (threadIdx.y - 4 + by*4)*8;
        if ( x < n ) {
          bs[threadIdx.x + (threadIdx.y - 4)*8     ] = b[offa + x + i*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 32] = b[offa + x + (i + 1)*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 64] = b[offa + x + (i + 2)*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 96] = 0.0;
        }
      }
    }
    else if ( i + 1 < k ) {
      if ( threadIdx.y < 4 ) {
        ipc_ x = threadIdx.x + (threadIdx.y + bx*4)*8;
        if ( x < n ) {
          as[threadIdx.x + threadIdx.y*8     ] = a[offa + x + i*lda];
          as[threadIdx.x + threadIdx.y*8 + 32] = a[offa + x + (i + 1)*lda];
          as[threadIdx.x + threadIdx.y*8 + 64] = 0.0;
          as[threadIdx.x + threadIdx.y*8 + 96] = 0.0;
        }
      }
      else {
        ipc_ x = threadIdx.x + (threadIdx.y - 4 + by*4)*8;
        if ( x < n ) {
          bs[threadIdx.x + (threadIdx.y - 4)*8     ] = b[offa + x + i*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 32] = b[offa + x + (i + 1)*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 64] = 0.0;
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 96] = 0.0;
        }
      }
    }
    else {
      if ( threadIdx.y < 4 ) {
        ipc_ x = threadIdx.x + (threadIdx.y + bx*4)*8;
        if ( x < n ) {
          as[threadIdx.x + threadIdx.y*8     ] = a[offa + x + i*lda];
          as[threadIdx.x + threadIdx.y*8 + 32] = 0.0;
          as[threadIdx.x + threadIdx.y*8 + 64] = 0.0;
          as[threadIdx.x + threadIdx.y*8 + 96] = 0.0;
        }
      }
      else {
        ipc_ x = threadIdx.x + (threadIdx.y - 4 + by*4)*8;
        if ( x < n ) {
          bs[threadIdx.x + (threadIdx.y - 4)*8     ] = b[offa + x + i*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 32] = 0.0;
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 64] = 0.0;
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 96] = 0.0;
        }
      }
    }
    break;

case 2:
    if ( i + 1 < k ) {
      if ( threadIdx.y < 4 ) {
        ipc_ x = threadIdx.x + (threadIdx.y + bx*4)*8;
        if ( x < n ) {
          as[threadIdx.x + threadIdx.y*8     ] = a[offa + x + i*lda];
          as[threadIdx.x + threadIdx.y*8 + 32] = a[offa + x + (i + 1)*lda];
        }
      }
      else {
        ipc_ x = threadIdx.x + (threadIdx.y - 4 + by*4)*8;
        if ( x < n ) {
          bs[threadIdx.x + (threadIdx.y - 4)*8     ] = b[offa + x + i*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 32] = b[offa + x + (i + 1)*ldb];
        }
      }
    }
    else {
      if ( threadIdx.y < 4 ) {
        ipc_ x = threadIdx.x + (threadIdx.y + bx*4)*8;
        if ( x < n ) {
          as[threadIdx.x + threadIdx.y*8     ] = a[offa + x + i*lda];
          as[threadIdx.x + threadIdx.y*8 + 32] = 0.0;
        }
      }
      else {
        ipc_ x = threadIdx.x + (threadIdx.y - 4 + by*4)*8;
        if ( x < n ) {
          bs[threadIdx.x + (threadIdx.y - 4)*8     ] = b[offa + x + i*ldb];
          bs[threadIdx.x + (threadIdx.y - 4)*8 + 32] = 0.0;
        }
      }
    }
    break;

    default:
      printf("Invalid SYRK width\n");
  }
}

struct multisyrk_type {
  ipc_ first;
  rpc_ *lval;
  rpc_ *ldval;
  long offc;
  ipc_ n;
  ipc_ k;
  ipc_ lda;
  ipc_ ldb;
};

// multisyrk kernels below compute the low trangular part of a*b^T
// (stored columnwise) using 8x8 cuda blocks

template< typename ELEMENT_TYPE >
#if SM_3X
__launch_bounds__(64, 14)
#endif
__global__ void
cu_multisyrk_lc_r4x4(
  const struct multisyrk_type* msdata, ipc_ off, ELEMENT_TYPE* c
){

// The number of elements we want in each shared memory buffer depends on 
/  the shared memory:register ratio SM 3.0+ has precision_ the number of 
// registers per shared memory, so need half the shared memory here.
#if SM_3X
  #define SYRK_WIDTH 4
  #define DOUBLE_BUFFERED 0
  #define USE_DOUBLE2 1
#else
  #define SYRK_WIDTH 4
  #define DOUBLE_BUFFERED 1
  #define USE_DOUBLE2 0
#endif

#if (USE_DOUBLE2)
  __shared__ volatile double2 as[32 * SYRK_WIDTH];
  __shared__ volatile ELEMENT_TYPE bs[32 * SYRK_WIDTH];
#if (DOUBLE_BUFFERED)
  __shared__ volatile double2 as2[32 * SYRK_WIDTH];
  __shared__ volatile ELEMENT_TYPE bs2[32 * SYRK_WIDTH];
#endif

#else
  __shared__ volatile ELEMENT_TYPE as[32 * SYRK_WIDTH], bs[32 * SYRK_WIDTH];
#if (DOUBLE_BUFFERED)
  __shared__ volatile ELEMENT_TYPE as2[32 * SYRK_WIDTH], bs2[32 * SYRK_WIDTH];
#endif
#endif

  msdata += blockIdx.x;
  ipc_ first = msdata->first;
  const ELEMENT_TYPE * __restrict__ a = msdata->lval;
  const ELEMENT_TYPE * __restrict__ b = msdata->ldval;
  ipc_ offc  = msdata->offc;
  ipc_ n     = msdata->n;
  ipc_ k     = msdata->k;
  ipc_ lda   = msdata->lda;
  ipc_ ldb   = msdata->ldb;

  if ( n < 1 )
    return;


  ipc_ bx, by;
  {
    ipc_ nb = (n - 1)/32 + 1;
    for ( bx = 0, by = 0; by < nb; by++ ) {
      if ( off + blockIdx.x - first - bx < nb - by ) {
        bx = off + blockIdx.x - first - bx + by;
        break;
      }
      bx += nb - by;
    }
  }

#if (USE_DOUBLE2)
  double2 s[8];
  for ( ipc_ i = 0; i < 8; i++ ) {
    s[i].x = 0.0;
    s[i].y = 0.0;
  }
#else
  ELEMENT_TYPE s[16];
  for ( ipc_ i = 0; i < 16; i++ )
    s[i] = 0.0;
#endif


#if (SYRK_WIDTH <= 2 && DOUBLE_BUFFERED)
  loadDevToSmem_generic<SYRK_WIDTH>( (volatile rpc_*)as, bs, a, b,
    bx, by, 0, lda, ldb, n, 0, k );
#endif

  for ( ipc_ i = 0; i < k; i += SYRK_WIDTH ) {



    // We want to get these in flight as early as possible so we can hide their
    // latency. We would also want to get the other set of loads in flight in a
    // similar manner, but this degrades performance (and makes the code more
    // complicated). I suspect it adds register pressure as it was quite a
    // challenge to get it working without spilling.
#if (DOUBLE_BUFFERED)
    if ( i + SYRK_WIDTH < k ) {
       loadDevToSmem_generic<SYRK_WIDTH>( (volatile rpc_*)as2, bs2,
         a, b, bx, by, 0, lda, ldb, n, i + SYRK_WIDTH, k );
    }
#endif // (DOUBLE_BUFFERED)

#if (SYRK_WIDTH > 2 || DOUBLE_BUFFERED)
    loadDevToSmem_generic<SYRK_WIDTH>( (volatile rpc_*)as, bs, a, b,
      bx, by, 0, lda, ldb, n, i, k );
#endif
    __syncthreads();


    #pragma unroll
    for ( ipc_ ix = 0; ix < SYRK_WIDTH; ix++) {
      for ( ipc_ iy = 0; iy < 4; iy++ ) {
#if (USE_DOUBLE2)
        s[iy*2    ].x += as[threadIdx.x + ix * 16    ].x*bs[threadIdx.y + 8*iy + ix * 32];
        s[iy*2    ].y += as[threadIdx.x + ix * 16    ].y*bs[threadIdx.y + 8*iy + ix * 32];
        s[iy*2 + 1].x += as[threadIdx.x + ix * 16 + 8].x*bs[threadIdx.y + 8*iy + ix * 32];
        s[iy*2 + 1].y += as[threadIdx.x + ix * 16 + 8].y*bs[threadIdx.y + 8*iy + ix * 32];
#else
        s[iy*4]     += as[threadIdx.x + 32 * ix     ]*bs[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 1] += as[threadIdx.x + 32 * ix + 8 ]*bs[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 2] += as[threadIdx.x + 32 * ix + 16]*bs[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 3] += as[threadIdx.x + 32 * ix + 24]*bs[threadIdx.y + 8*iy + 32 * ix];
#endif
      }
    }


#if (DOUBLE_BUFFERED)
    i += SYRK_WIDTH;

    if ( i >= k ) break;

    __syncthreads();
    if ( i + SYRK_WIDTH < k ) {
#if (SYRK_WIDTH <= 2)
       loadDevToSmem_generic<SYRK_WIDTH>( (volatile rpc_*)as, bs, a, b, bx, by, 0, lda, ldb, n, i + SYRK_WIDTH, k );
#endif
    }

    #pragma unroll
    for ( ipc_ ix = 0; ix < SYRK_WIDTH; ix++) {
      for ( ipc_ iy = 0; iy < 4; iy++ ) {
#if (USE_DOUBLE2)
        s[iy*2    ].x += as2[threadIdx.x + ix * 16    ].x*bs2[threadIdx.y + 8*iy + ix * 32];
        s[iy*2    ].y += as2[threadIdx.x + ix * 16    ].y*bs2[threadIdx.y + 8*iy + ix * 32];
        s[iy*2 + 1].x += as2[threadIdx.x + ix * 16 + 8].x*bs2[threadIdx.y + 8*iy + ix * 32];
        s[iy*2 + 1].y += as2[threadIdx.x + ix * 16 + 8].y*bs2[threadIdx.y + 8*iy + ix * 32];
#else
        s[iy*4]     += as2[threadIdx.x + 32 * ix     ]*bs2[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 1] += as2[threadIdx.x + 32 * ix + 8 ]*bs2[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 2] += as2[threadIdx.x + 32 * ix + 16]*bs2[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 3] += as2[threadIdx.x + 32 * ix + 24]*bs2[threadIdx.y + 8*iy + 32 * ix];
#endif
      }
    }

#endif // DOUBLE_BUFFERED
    __syncthreads();

  }

#if (USE_DOUBLE2)
  for ( ipc_ iy = 0; iy < 4; iy++ ) {
    for ( ipc_ ix = 0; ix < 2; ix++ ) {
      ipc_ x = threadIdx.x * 2 + ix*16 + bx*32;
      ipc_ y = threadIdx.y + iy*8 + by*32;
      if ( x < n && y < n && y <= x ) {
        c[offc + x + y*n] = -s[ix + iy*2].x;
      }

      x += 1;
      if ( x < n && y < n && y <= x ) {
        c[offc + x + y*n] = -s[ix + iy*2].y;
      }
    }
  }
#else
  ipc_ xMaxBase = (3 + bx*4)*8;
  ipc_ yMaxBase = (3 + by*4)*8;

  ipc_ XNPass = xMaxBase + 8 < n;
  ipc_ YNPass = yMaxBase + 8 < n;
  ipc_ YXPass = yMaxBase + 8 <= xMaxBase;

  // This is only a small improvement (~1%)
  if (XNPass && YNPass && YXPass) {
    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      for ( ipc_ ix = 0; ix < 4; ix++ ) {
        ipc_ x = threadIdx.x + (ix + bx*4)*8;
        ipc_ y = threadIdx.y + (iy + by*4)*8;
        c[offc + x + y*n] = -s[ix + iy*4];
      }
    }
  }
  else if (XNPass && YNPass) {
    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      for ( ipc_ ix = 0; ix < 4; ix++ ) {
        ipc_ x = threadIdx.x + (ix + bx*4)*8;
        ipc_ y = threadIdx.y + (iy + by*4)*8;
        if ( y <= x )
          c[offc + x + y*n] = -s[ix + iy*4];
      }
    }
  }
  else {
    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      for ( ipc_ ix = 0; ix < 4; ix++ ) {
        ipc_ x = threadIdx.x + (ix + bx*4)*8;
        ipc_ y = threadIdx.y + (iy + by*4)*8;
        if ( x < n && y < n && y <= x )
          c[offc + x + y*n] = -s[ix + iy*4];
      }
    }
  }
#endif

// Release function-scope #defines
#undef SYRK_WIDTH
#undef DOUBLE_BUFFERED
#undef USE_DOUBLE2
}

struct multielm_data {
  ipc_ node;
  ipc_ offb;
};

template< typename ELEMENT_TYPE >
//#if SM_3X
//__launch_bounds__(64, 14)
//#endif
__global__ void
cu_multisyrk_r4x4(
    bool posdef,
    ipc_* stat,
    multielm_data* mdata,
    ipc_ off,
    struct multinode_fact_type *ndatat
){
  ipc_ bx, by;
  ipc_ n, m, k;
  ipc_ offa, offc;
  ipc_ lda, ldb;
  ipc_ nb;
  ELEMENT_TYPE s[16];
#if SM_3X
  #define SYRK_WIDTH 2
  #define DOUBLE_BUFFERED 0
#else
  #define SYRK_WIDTH 4
  #define DOUBLE_BUFFERED 0
#endif

  __shared__ volatile ELEMENT_TYPE as[32 * SYRK_WIDTH];
  __shared__ volatile ELEMENT_TYPE bs[32 * SYRK_WIDTH];

#if (DOUBLE_BUFFERED)
  __shared__ volatile ELEMENT_TYPE as2[32 * SYRK_WIDTH];
  __shared__ volatile ELEMENT_TYPE bs2[32 * SYRK_WIDTH];
#endif

  mdata += blockIdx.x;
  bx = mdata->node;
  ndatat += bx;
  k = stat[bx];
  if ( k < 1 )
    return;

  if ( ndatat->ib > ndatat->jb )
    return;

  n   = ndatat->nrows;
  lda = ndatat->done;
  m   = ndatat->rght;
  by = lda + k;
  if ( by >= n || by >= m )
    return;

  const rpc_ * __restrict__ a = ndatat->lval;
  const rpc_ * __restrict__ b = posdef ? ndatat->lval : ndatat->ldval;
  rpc_ * __restrict__ c = ndatat->lval;

  offa = by + lda*n;
  offc = by + by*n;
  lda = n;
  ldb = n;
  m -= by;
  n -= by;

  by = off + blockIdx.x - mdata->offb;
  if ( by > ((n - 1)/32 + 1)*((m - 1)/32 + 1) )
    return;

  nb = (n - 1)/32 + 1;
  bx = by%nb;
  by = by/nb;

  for ( ipc_ i = 0; i < 16; i++ ) {
    s[i] = 0.0;
  }

#if (DOUBLE_BUFFERED)
  loadDevToSmem_generic<SYRK_WIDTH>( (volatile rpc_*)as, bs, a, b, bx, by, offa, lda, ldb, n, 0, k );
#endif

  for ( ipc_ i = 0; i < k; i += SYRK_WIDTH ) {
#if (!DOUBLE_BUFFERED)
    loadDevToSmem_generic<SYRK_WIDTH>( (volatile rpc_*)as, bs, a, b, bx, by, offa, lda, ldb, n, i, k );
#endif

    __syncthreads();

#if (DOUBLE_BUFFERED)
    if (i + SYRK_WIDTH < k) {
      loadDevToSmem_generic<SYRK_WIDTH>( as2, bs2, a, b, bx, by, offa, lda, ldb, n, i + SYRK_WIDTH, k );
    }
#endif

    #pragma unroll
    for ( ipc_ ix = 0; ix < SYRK_WIDTH; ix++) {
      for ( ipc_ iy = 0; iy < 4; iy++ ) {
        s[iy*4]     += as[threadIdx.x + 32 * ix     ]*bs[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 1] += as[threadIdx.x + 32 * ix + 8 ]*bs[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 2] += as[threadIdx.x + 32 * ix + 16]*bs[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 3] += as[threadIdx.x + 32 * ix + 24]*bs[threadIdx.y + 8*iy + 32 * ix];
      }
    }

    __syncthreads();

#if (DOUBLE_BUFFERED)
    i += SYRK_WIDTH;

    if (i >= k) break;

    if (i + SYRK_WIDTH < k) {
      loadDevToSmem_generic<SYRK_WIDTH>( as, bs, a, b, bx, by, offa, lda, ldb, n, i + SYRK_WIDTH, k );
    }

    #pragma unroll
    for ( ipc_ ix = 0; ix < SYRK_WIDTH; ix++) {
      for ( ipc_ iy = 0; iy < 4; iy++ ) {
        s[iy*4]     += as2[threadIdx.x + 32 * ix     ]*bs2[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 1] += as2[threadIdx.x + 32 * ix + 8 ]*bs2[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 2] += as2[threadIdx.x + 32 * ix + 16]*bs2[threadIdx.y + 8*iy + 32 * ix];
        s[iy*4 + 3] += as2[threadIdx.x + 32 * ix + 24]*bs2[threadIdx.y + 8*iy + 32 * ix];
      }
    }
#endif
  }

  for ( ipc_ iy = 0; iy < 4; iy++ )
    for ( ipc_ ix = 0; ix < 4; ix++ ) {
      ipc_ x = threadIdx.x + (ix + bx*4)*8;
      ipc_ y = threadIdx.y + (iy + by*4)*8;
      if ( x < n && y < m )
        c[offc + x + y*lda] = c[offc + x + y*lda] - s[ix + iy*4];
    }
}

template< typename ELEMENT_TYPE >
__global__ void
cu_syrk_r4x4(
  ipc_ n, ipc_ m, ipc_ k,
  rpc_ alpha, const rpc_* a, ipc_ lda, const rpc_* b, ipc_ ldb,
  rpc_ beta, rpc_* c, ipc_ ldc
){
  ELEMENT_TYPE s[16];

  __shared__ volatile ELEMENT_TYPE as[128], bs[128];

  for ( ipc_ i = 0; i < 16; i++ )
    s[i] = 0;

  for ( ipc_ i = 0; i < k; i += 4 ) {

    loadDevToSmem_generic< 4 >( as, bs, a, b, blockIdx.x, blockIdx.y, 0, lda, ldb,
                              n, i, k );
    __syncthreads();

    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      s[iy*4]     += as[threadIdx.x     ]*bs[threadIdx.y + 8*iy];
      s[iy*4 + 1] += as[threadIdx.x + 8 ]*bs[threadIdx.y + 8*iy];
      s[iy*4 + 2] += as[threadIdx.x + 16]*bs[threadIdx.y + 8*iy];
      s[iy*4 + 3] += as[threadIdx.x + 24]*bs[threadIdx.y + 8*iy];
    }

    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      s[iy*4]     += as[threadIdx.x + 32]*bs[threadIdx.y + 8*iy + 32];
      s[iy*4 + 1] += as[threadIdx.x + 40]*bs[threadIdx.y + 8*iy + 32];
      s[iy*4 + 2] += as[threadIdx.x + 48]*bs[threadIdx.y + 8*iy + 32];
      s[iy*4 + 3] += as[threadIdx.x + 56]*bs[threadIdx.y + 8*iy + 32];
    }

    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      s[iy*4]     += as[threadIdx.x + 64]*bs[threadIdx.y + 8*iy + 64];
      s[iy*4 + 1] += as[threadIdx.x + 72]*bs[threadIdx.y + 8*iy + 64];
      s[iy*4 + 2] += as[threadIdx.x + 80]*bs[threadIdx.y + 8*iy + 64];
      s[iy*4 + 3] += as[threadIdx.x + 88]*bs[threadIdx.y + 8*iy + 64];
    }

    for ( ipc_ iy = 0; iy < 4; iy++ ) {
      s[iy*4]     += as[threadIdx.x + 96 ]*bs[threadIdx.y + 8*iy + 96];
      s[iy*4 + 1] += as[threadIdx.x + 104]*bs[threadIdx.y + 8*iy + 96];
      s[iy*4 + 2] += as[threadIdx.x + 112]*bs[threadIdx.y + 8*iy + 96];
      s[iy*4 + 3] += as[threadIdx.x + 120]*bs[threadIdx.y + 8*iy + 96];
    }

    __syncthreads();
  }

  if ( beta ) {
    for ( ipc_ iy = 0; iy < 4; iy++ )
      for ( ipc_ ix = 0; ix < 4; ix++ ) {
        ipc_ x = threadIdx.x + (ix + blockIdx.x*4)*8;
        ipc_ y = threadIdx.y + (iy + blockIdx.y*4)*8;
        if ( x < n && y < m )
          c[x + y*ldc] = beta*c[x + y*ldc] + alpha*s[ix + iy*4];
      }
  }
  else {
    for ( ipc_ iy = 0; iy < 4; iy++ )
      for ( ipc_ ix = 0; ix < 4; ix++ ) {
        ipc_ x = threadIdx.x + (ix + blockIdx.x*4)*8;
        ipc_ y = threadIdx.y + (iy + blockIdx.y*4)*8;
        if ( x < n && y < m )
          c[x + y*ldc] = alpha*s[ix + iy*4];
      }
  }
}


} /* anon namespace */

/*******************************************************************************
 * Following routines are exported with C binding so can be called from Fortran
 ******************************************************************************/

extern "C" {

void spral_ssids_dsyrk(cudaStream_t *stream, ipc_ n, ipc_ m, ipc_ k,
      rpc_ alpha, const rpc_* a, ipc_ lda, const rpc_* b,
      ipc_ ldb, rpc_ beta, rpc_* c, ipc_ ldc) {
  ipc_ nx, ny;
  nx = (n - 1)/32 + 1;
  ny = (m - 1)/32 + 1;
  dim3 threads(8,8);
  dim3 grid(nx,ny);
  cu_syrk_r4x4< rpc_ > <<< grid, threads, 0, *stream >>>
    ( n, m, k, alpha, a, lda, b, ldb, beta, c, ldc );
}

void spral_ssids_multidsyrk(cudaStream_t *stream, bool posdef, ipc_ nb,
      ipc_* stat, struct multielm_data* mdata,
      struct multinode_fact_type *ndata) {
  dim3 threads(8,8);
  for ( ipc_ i = 0; i < nb; i += MAX_CUDA_BLOCKS ) {
    ipc_ blocks = min(MAX_CUDA_BLOCKS, nb - i);
    cu_multisyrk_r4x4< rpc_ >
      <<< blocks, threads, 0, *stream >>>
      ( posdef, stat, mdata + i, i, ndata );
  }
}

void spral_ssids_multidsyrk_low_col(cudaStream_t *stream, ipc_ nb,
      struct multisyrk_type* msdata, rpc_* c) {
  dim3 threads(8,8);
  for ( ipc_ i = 0; i < nb; i += MAX_CUDA_BLOCKS ) {
    ipc_ blocks = min(MAX_CUDA_BLOCKS, nb - i);
    cu_multisyrk_lc_r4x4< rpc_ >
      <<< blocks, threads, 0, *stream >>>( msdata + i, i, c );
  }
}
} // end extern "C"
