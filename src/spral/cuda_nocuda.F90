! THIS VERSION: GALAHAD 4.1 - 2023-01-25 AT 09:10 GMT.

#include "spral_procedures.h"

! Provides limited interface definitions for CUDA functions in the case
! we are not compiled against CUDA libraries
module spral_cuda_precision
  use spral_kinds_precision
  use, intrinsic :: iso_c_binding
  implicit none

  private
  public :: cudaGetErrorString
  public :: detect_gpu

contains
  ! Convert a CUDA error code to a Fortran character string
  character(len=200) function cudaGetErrorString(error)
    implicit none
    integer(C_IP_), intent(in) :: error

    write(cudaGetErrorString, "(a,i3)") "Not compiled with CUDA support ", error
  end function cudaGetErrorString

  ! Return true if a GPU is present and code is compiled with CUDA support
  logical function detect_gpu()
    implicit none
    detect_gpu = .false.
  end function detect_gpu
end module spral_cuda_precision
