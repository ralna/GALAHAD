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
extern "C" {
   cudaError_t spral_cudaStreamCreate(cudaStream_t **pStream) {
      *pStream = (cudaStream_t *) malloc(sizeof(cudaStream_t));
      return cudaStreamCreate(*pStream);
   }
   cudaError_t spral_cudaStreamDestroy(cudaStream_t *stream) {
      cudaError_t ret = cudaStreamDestroy(*stream);
      free(stream);
      return ret;
   }
   cudaError_t spral_cudaMemcpyAsync(void *dst, const void *src, size_t count,
         enum cudaMemcpyKind kind, cudaStream_t *stream) {
      return cudaMemcpyAsync(dst, src, count, kind, *stream);
   }
   cudaError_t spral_cudaMemcpy2DAsync(void *dst, size_t dpitch,
         const void *src, size_t spitch, size_t width, size_t height,
         enum cudaMemcpyKind kind, cudaStream_t *stream) {
      return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind,
         *stream);
   }
   cudaError_t spral_cudaMemsetAsync(void *devPtr, int value, size_t count,
         cudaStream_t *stream) {
      return cudaMemsetAsync(devPtr, value, count, *stream);
   }
   cudaError_t spral_cudaStreamSynchronize(cudaStream_t *stream) {
      return cudaStreamSynchronize(*stream);
   }
}

// Following wrappers needed as cudaEvent_t and cudaStream_t not interoperable
extern "C" {
   cudaError_t spral_cudaEventCreateWithFlags(cudaEvent_t **event, int flags) {
      *event = (cudaEvent_t *) malloc(sizeof(cudaEvent_t));
      unsigned int uflags = (unsigned int) flags;
      return cudaEventCreateWithFlags(*event, uflags);
   }
   cudaError_t spral_cudaEventDestroy(cudaEvent_t *event) {
      cudaError_t ret = cudaEventDestroy(*event);
      free(event);
      return ret;
   }
   cudaError_t spral_cudaEventRecord(cudaEvent_t *event, cudaStream_t *stream) {
      return cudaEventRecord(*event, *stream);
   }
   cudaError_t spral_cudaEventSynchronize(cudaEvent_t *event) {
      return cudaEventSynchronize(*event);
   }
}

// Following wrappers needed as cublasHandle_T not interoperable
extern "C" {
   cublasStatus_t spral_cublasCreate(cublasHandle_t **const handle)
   {
      *handle = (cublasHandle_t*)malloc(sizeof(cublasHandle_t));
      return cublasCreate(*handle);
   }
   cublasStatus_t spral_cublasDestroy(cublasHandle_t *const handle)
   {
      const cublasStatus_t error = cublasDestroy(*handle);
      free(handle);
      return error;
   }
   cublasStatus_t spral_cublasDgemm(cublasHandle_t *const handle,
                                    const char *const transa,
                                    const char *const transb,
                                    const int *const m,
                                    const int *const n,
                                    const int *const k,
                                    const double *const alpha,
                                    const double *const devPtrA,
                                    const int *const lda,
                                    const double *const devPtrB,
                                    const int *const ldb,
                                    const double *const beta,
                                    double *const devPtrC,
                                    const int *const ldc)
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
   cublasStatus_t spral_cublasSgemm(cublasHandle_t *const handle,
                                    const char *const transa,
                                    const char *const transb,
                                    const int *const m,
                                    const int *const n,
                                    const int *const k,
                                    const float *const alpha,
                                    const float *const devPtrA,
                                    const int *const lda,
                                    const float *const devPtrB,
                                    const int *const ldb,
                                    const float *const beta,
                                    float *const devPtrC,
                                    const int *const ldc)
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
      return cublasSgemm(*handle, tA, tB, *m, *n, *k, alpha, devPtrA, *lda,
                         devPtrB, *ldb, beta, devPtrC, *ldc);
   }
   cublasStatus_t spral_cublasSetStream(cublasHandle_t *handle,
         cudaStream_t *streamId) {
      return cublasSetStream(*handle, *streamId);
   }
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
