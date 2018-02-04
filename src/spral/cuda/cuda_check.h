// To enable error checking use -DCUDA_CHECK_ERROR

#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#define CudaSafeCall( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()        __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
  do {
    if ( cudaSuccess != err ) {
      fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#endif  // CUDA_CHECK_ERROR
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
  do {
    cudaError_t err = cudaGetLastError();
    if ( cudaSuccess != err ) {
      fprintf( stderr, "cudaCheckError() failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment if not needed.
    err = cudaThreadSynchronize();
    if ( cudaSuccess != err ) {
      fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s.\n",
               file, line, cudaGetErrorString( err ) );
      exit( -1 );
    }
  } while ( 0 );
#endif // CUDA_CHECK_ERROR
  return;
}

#endif // !CUDA_CHECK_H
