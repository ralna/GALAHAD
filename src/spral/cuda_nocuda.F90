! THIS VERSION: GALAHAD 5.3 - 2025-08-18 AT 13:00 GMT

#include "spral_procedures.h"

!  provides limited interface definitions for CUDA functions in the case
!  we are not compiled against CUDA libraries

MODULE SPRAL_CUDA_precision
  USE GALAHAD_KINDS_precision
  USE, INTRINSIC :: iso_c_binding
  IMPLICIT none

  PRIVATE
  PUBLIC :: cudaGetErrorString
  PUBLIC :: detect_gpu

CONTAINS

  CHARACTER( LEN = 200 ) FUNCTION cudaGetErrorString(error)

!  convert a CUDA error code to a Fortran character string

    IMPLICIT none
    INTEGER( C_IP_ ), INTENT( IN ) :: error

    WRITE( cudaGetErrorString, "( 'Not compiled with CUDA support ', I3 )" )   &
      error
  END FUNCTION cudaGetErrorString

  LOGICAL FUNCTION detect_gpu( )

!  return true if a GPU is present and code is compiled with CUDA support

    IMPLICIT none

    detect_gpu = .FALSE.
  END FUNCTION detect_gpu
END MODULE SPRAL_CUDA_precision
