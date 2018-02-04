/* Copyright (c) 2013 Science and Technology Facilities Council (STFC)
 * Authors: Evgueni Ovtchinnikov and Jonathan Hogg
 *
 * This file provides wrappers around functions that are non-trivial to
 * otherwise provide a Fortran interface to using iso_c_binding.
*/

#include <stdio.h>
#include <ctype.h>
#include <cublas_v2.h>

// Following wrappers needed as cudaStream_t not interoperable
// FIXME: According to driver_types.h, cudaStream_t is a typedef for a C pointer type, which should be interoperable...
// extern "C" {
//   cudaError_t spral_cudaStreamCreate(cudaStream_t **const pStream)
//   {
//     *pStream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
//     return cudaStreamCreate(*pStream);
//   }
//   cudaError_t spral_cudaStreamDestroy(cudaStream_t *const stream)
//   {
//     const cudaError_t ret = cudaStreamDestroy(*stream);
//     free(stream);
//     return ret;
//   }
//   cudaError_t spral_cudaMemcpyAsync(void *const dst, const void *const src,
//       const size_t count, const enum cudaMemcpyKind kind,
//       cudaStream_t *const stream)
//   {
//     return cudaMemcpyAsync(dst, src, count, kind, *stream);
//   }
//   cudaError_t spral_cudaMemcpy2DAsync(void *const dst, const size_t dpitch,
//       const void *const src, const size_t spitch, const size_t width,
//       const size_t height, const enum cudaMemcpyKind kind,
//       cudaStream_t *const stream)
//   {
//     return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, 
//       *stream);
//   }
//   cudaError_t spral_cudaMemsetAsync(void *const devPtr, const int value,
//       const size_t count, cudaStream_t *const stream)
//   {
//     return cudaMemsetAsync(devPtr, value, count, *stream);
//   }
//   cudaError_t spral_cudaStreamSynchronize(cudaStream_t *const stream)
//   {
//     return cudaStreamSynchronize(*stream);
//   }
// }

// Following wrappers needed as cudaEvent_t and cudaStream_t not interoperable
// FIXME: According to driver_types.h, cudaEvent_t is a typedef for a C pointer type, which should be interoperable...
// extern "C" {
//   cudaError_t spral_cudaEventCreateWithFlags(cudaEvent_t **const event, const int flags)
//   {
//     *event = (cudaEvent_t*)malloc(sizeof(cudaEvent_t));
//     const unsigned int uflags = (unsigned int)flags;
//     return cudaEventCreateWithFlags(*event, uflags);
//   }
//   cudaError_t spral_cudaEventDestroy(cudaEvent_t *const event)
//   {
//     const cudaError_t ret = cudaEventDestroy(*event);
//     free(event);
//     return ret;
//   }
//   cudaError_t spral_cudaEventRecord(cudaEvent_t *const event, cudaStream_t *const stream)
//   {
//     return cudaEventRecord(*event, *stream);
//   }
//   cudaError_t spral_cudaEventSynchronize(cudaEvent_t *const event)
//   {
//     return cudaEventSynchronize(*event);
//   }
// }

// Following wrappers needed as cublasHandle_t not interoperable
// FIXME: According to driver_types.h, cudaEvent_t is a typedef for a C pointer type, which should be interoperable...
extern "C" {
  // cublasStatus_t spral_cublasCreate(cublasHandle_t **const handle)
  // {
  //   *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
  //   return cublasCreate(*handle);
  // }
  // cublasStatus_t spral_cublasDestroy(cublasHandle_t *const handle)
  // {
  //   const cublasStatus_t error = cublasDestroy(*handle);
  //   free(handle);
  //   return error;
  // }
  cublasStatus_t spral_cublasDgemm(cublasHandle_t *const handle, const char *const transa,
      const char *const transb, const int *const m, const int *const n, const int *const k,
      const double *const alpha, const double *const devPtrA, const int *const lda,
      const double *const devPtrB, const int *const ldb, const double *const beta,
      double *const devPtrC, const int *const ldc)
  {
      cublasOperation_t tA, tB;
      if (toupper(*transa) == 'N') 
        tA = CUBLAS_OP_N;
      else
        tA = CUBLAS_OP_T;
      if (toupper(*transb) == 'N') 
        tB = CUBLAS_OP_N;
      else
        tB = CUBLAS_OP_T;
      return cublasDgemm(*handle, tA, tB, *m, *n, *k, alpha, devPtrA, *lda,
          devPtrB, *ldb, beta, devPtrC, *ldc);
  }
  // cublasStatus_t spral_cublasSetStream(cublasHandle_t *const handle,
  //     cudaStream_t *const streamId)
  // {
  //   return cublasSetStream(*handle, *streamId);
  // }
}

/*
 * Exceptionally useful non-CUDA API functions
 */

// Used to provide pointer arithmetic in Fortran
extern "C"
void *spral_c_ptr_plus(void *const base, const size_t sz)
{
  return (void*)(((char*)base) + sz);
}

// Allow pretty printing of a C pointer in Fortran
extern "C"
void spral_c_print_ptr(void *const ptr)
{
  (void)printf("ptr = %p\n", ptr);
}
