! THIS VERSION: GALAHAD 4.1 - 2023-01-25 AT 09:10 GMT.

#include "spral_procedures.h"

! Provides interface definitions for CUDA functions
module spral_cuda_precision
  use spral_kinds_precision
  use, intrinsic :: iso_c_binding
  implicit none

  private
  ! enum values for cudaMemcpy
  public :: cudaMemcpyHostToHost, cudaMemcpyHostToDevice, &
       cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice
  ! enum values for cudaDeviceGetSharedMemConfig
  public :: cudaSharedMemBankSizeDefault, cudaSharedMemBankSizeFourByte, &
       cudaSharedMemBankSizeEightByte
  ! #define values for cudaEventCreateWithFlags
  public :: cudaEventDefault, cudaEventBlockingSync, cudaEventDisableTiming
  ! enum values for cudaError
  public :: cudaSuccess, cudaErrorInsufficientDriver, cudaErrorNoDevice
  ! Literal interfaces to C functions in CUDA API
  public :: cudaDeviceEnablePeerAccess, cudaDeviceSynchronize, cudaFree, &
       cudaGetDeviceCount, cudaGetLastError, cudaMalloc, cudaMemset, &
       cudaMemcpy, cudaMemcpy2D, cudaSetDevice, &
       cudaDeviceGetSharedMemConfig, cudaDeviceSetSharedMemConfig
  ! Wrapper interfaces to C functions provided by CUDA API
  public :: cudaEventCreateWithFlags, cudaEventDestroy, cudaEventRecord, &
       cudaEventSynchronize, cudaMemcpyAsync, cudaMemcpy2DAsync, &
       cudaMemsetAsync, cudaStreamCreate, cudaStreamDestroy, &
       cudaStreamSynchronize
  ! Wrapper interfaces to C function provided by cuBLAS API
  public :: cublasCreate, cublasDestroy, cublasgemm, cublasSetStream
  ! Helper functions for dealing with type(C_PTR)
  public :: c_ptr_plus, c_print_ptr, c_ptr_plus_aligned, aligned_size
  ! A Fortran version of cudaGetErrorString
  public :: cudaGetErrorString
  ! Syntactically nicer ways of calling cudaMemcpy
  public :: cudaMemcpy_h2d, cudaMemcpy_d2h, cudaMemcpy_d2d, &
       cudaMemcpyAsync_h2d, cudaMemcpyAsync_d2h, cudaMemcpyAsync_d2d
  ! Utility functions
  public :: detect_gpu

  ! Based on enum in driver_types.h
  integer(C_IP_), parameter :: cudaMemcpyHostToHost     = 0_C_IP_
  integer(C_IP_), parameter :: cudaMemcpyHostToDevice   = 1_C_IP_
  integer(C_IP_), parameter :: cudaMemcpyDeviceToHost   = 2_C_IP_
  integer(C_IP_), parameter :: cudaMemcpyDeviceToDevice = 3_C_IP_
  integer(C_IP_), parameter :: cudaMemcpyDefault        = 4_C_IP_

  ! Based on enum in driver_types.h
  integer(C_IP_), parameter :: cudaSharedMemBankSizeDefault   = 0_C_IP_
  integer(C_IP_), parameter :: cudaSharedMemBankSizeFourByte  = 1_C_IP_
  integer(C_IP_), parameter :: cudaSharedMemBankSizeEightByte = 2_C_IP_

  ! Based on #define in driver_types.h
  integer(C_IP_), parameter :: cudaEventDefault       = 0_C_IP_
  integer(C_IP_), parameter :: cudaEventBlockingSync  = 1_C_IP_
  integer(C_IP_), parameter :: cudaEventDisableTiming = 2_C_IP_
  integer(C_IP_), parameter :: cudaEventInterprocess  = 4_C_IP_

  ! Based on enum in driver_types.h
  integer(C_IP_), parameter :: cudaSuccess                 =  0_C_IP_
  integer(C_IP_), parameter :: cudaErrorInsufficientDriver = 35_C_IP_
  integer(C_IP_), parameter :: cudaErrorNoDevice           = 38_C_IP_

  ! CUDA C provided functions (listed alphabetically)
  interface
     integer(C_IP_) function cudaDeviceEnablePeerAccess(peerDevice, flags) &
          bind(C, name="cudaDeviceEnablePeerAccess")
       use spral_kinds
       integer(C_IP_), value :: peerDevice
       integer(C_IP_), value :: flags ! must be 0, actually unsigned int
     end function cudaDeviceEnablePeerAccess
     integer(C_IP_) function cudaDeviceGetSharedMemConfig(pConfig) &
          bind(C, name="cudaDeviceGetSharedMemConfig")
       use spral_kinds
       integer(C_IP_) :: pConfig
     end function cudaDeviceGetSharedMemConfig
     integer(C_IP_) function cudaDeviceSetSharedMemConfig(config) &
          bind(C, name="cudaDeviceSetSharedMemConfig")
       use spral_kinds
       integer(C_IP_), value :: config
     end function cudaDeviceSetSharedMemConfig
     integer(C_IP_) function cudaDeviceSynchronize() &
          bind(C, name="cudaDeviceSynchronize")
       use spral_kinds
     end function cudaDeviceSynchronize
     integer(C_IP_) function cudaFree(dev_ptr) &
          bind(C, name="cudaFree")
       use spral_kinds
       type(C_PTR), value :: dev_ptr
     end function cudaFree
     integer(C_IP_) function cudaGetDeviceCount(cnt) &
          bind(C, name="cudaGetDeviceCount")
       use spral_kinds
       integer(C_IP_), intent(out) :: cnt
     end function cudaGetDeviceCount
     integer(C_IP_) function cudaGetLastError() &
          bind(C, name="cudaGetLastError")
       use spral_kinds
     end function cudaGetLastError
     integer(C_IP_) function cudaMalloc(dev_ptr, bytes) &
          bind(C, name="cudaMalloc")
       use spral_kinds
       type(C_PTR), intent(out) :: dev_ptr
       integer(C_SIZE_T), intent(in), value :: bytes
     end function cudaMalloc
     integer(C_IP_) function cudaMemset(devPtr, val, cnt) &
          bind(C, name="cudaMemset")
       use spral_kinds
       type(C_PTR), value :: devPtr
       integer(C_IP_), value :: val
       integer(C_SIZE_T), value :: cnt
     end function cudaMemset
     integer(C_IP_) function cudaMemcpy(dst, src, cnt, knd) &
          bind(C, name="cudaMemcpy")
       use spral_kinds
       type(C_PTR), value :: dst
       type(C_PTR), value :: src
       integer(C_SIZE_T), value :: cnt
       integer(C_IP_), value :: knd
     end function cudaMemcpy
     integer(C_IP_) function cudaMemcpy2D(dst, dpitch, src, spitch, width, &
          height, kind) bind(C, name="cudaMemcpy2D")
       use spral_kinds
       type(C_PTR), value :: dst
       integer(C_SIZE_T), value :: dpitch
       type(C_PTR), value :: src
       integer(C_SIZE_T), value :: spitch
       integer(C_SIZE_T), value :: width
       integer(C_SIZE_T), value :: height
       integer(C_IP_), value :: kind
     end function cudaMemcpy2D
     integer(C_IP_) function cudaMemGetInfo(free, total) &
          bind(C, name="cudaMemGetInfo")
       use spral_kinds
       integer(C_SIZE_T), intent(out) :: free
       integer(C_SIZE_T), intent(out) :: total
     end function cudaMemGetInfo
     integer(C_IP_) function cudaSetDevice(device) &
          bind(C, name="cudaSetDevice")
       use spral_kinds
       integer(C_IP_), value :: device
     end function cudaSetDevice
  end interface

  ! Stream functions - all wrapped as cudaStream_t not interoperable
  interface
     integer(C_IP_) function cudaStreamCreate(pStream) &
          bind(C, name="spral_cuda_precisionStreamCreate")
       use spral_kinds
       type(C_PTR), intent(out) :: pStream
     end function cudaStreamCreate
     integer(C_IP_) function cudaStreamDestroy(stream) &
          bind(C, name="spral_cuda_precisionStreamDestroy")
       use spral_kinds
       type(C_PTR), value :: stream
     end function cudaStreamDestroy
     integer(C_IP_) function cudaMemsetAsync(devPtr, value, count, stream) &
          bind(C, name="spral_cuda_precisionMemsetAsync")
       use spral_kinds
       type(C_PTR), value :: devPtr
       integer(C_IP_), value :: value
       integer(C_SIZE_T), value :: count
       type(C_PTR), value :: stream
     end function cudaMemsetAsync
     integer(C_IP_) function cudaMemcpyAsync(dst, src, count, kind, &
          stream) bind(C, name="spral_cuda_precisionMemcpyAsync")
       use spral_kinds
       type(C_PTR), value :: dst
       type(C_PTR), value :: src
       integer(C_SIZE_T), value :: count
       integer(C_IP_), value :: kind
       type(C_PTR), value :: stream
     end function cudaMemcpyAsync
     integer(C_IP_) function cudaMemcpy2DAsync(dst, dpitch, src, spitch, &
          width, height, kind, stream) bind(C, &
           name="spral_cuda_precisionMemcpy2DAsync")
       use spral_kinds
       type(C_PTR), value :: dst
       integer(C_SIZE_T), value :: dpitch
       type(C_PTR), value :: src
       integer(C_SIZE_T), value :: spitch
       integer(C_SIZE_T), value :: width
       integer(C_SIZE_T), value :: height
       integer(C_IP_), value :: kind
       type(C_PTR), value :: stream
     end function cudaMemcpy2DAsync
     integer(C_IP_) function cudaStreamSynchronize(stream) &
          bind(C, name="spral_cuda_precisionStreamSynchronize")
       use spral_kinds
       type(C_PTR), value :: stream
     end function cudaStreamSynchronize
  end interface

  ! Event functions - all wrapped as cudaEvent_t and cudaStream_t don't interop
  interface
     integer(C_IP_) function cudaEventCreateWithFlags(event, flags) &
          bind(C, name="spral_cuda_precisionEventCreateWithFlags")
       use spral_kinds
       type(C_PTR) :: event
       integer(C_IP_), value :: flags
     end function cudaEventCreateWithFlags
     integer(C_IP_) function cudaEventDestroy(event) &
          bind(C, name="spral_cuda_precisionEventDestroy")
       use spral_kinds
       type(C_PTR), value :: event
     end function cudaEventDestroy
     integer(C_IP_) function cudaEventRecord(event, stream) &
          bind(C, name="spral_cuda_precisionEventRecord")
       use spral_kinds
       type(C_PTR), value :: event
       type(C_PTR), value :: stream
     end function cudaEventRecord
     integer(C_IP_) function cudaEventSynchronize(event) &
          bind(C, name="spral_cuda_precisionEventSynchronize")
       use spral_kinds
       type(C_PTR), value :: event
     end function cudaEventSynchronize
  end interface

  ! CUBLAS functions - all wrapped as cublasHandle_t not interoperable
  interface
     integer(C_IP_) function cublasCreate(handle) &
          bind(C, name="spral_cublasCreate")
       use spral_kinds
       type(C_PTR), intent(out) :: handle
     end function cublasCreate
     integer(C_IP_) function cublasDestroy(handle) &
          bind(C, name="spral_cublasDestroy")
       use spral_kinds
       type(C_PTR), value :: handle
     end function cublasDestroy
     integer(C_IP_) function cublasgemm(handle, transa, transb, &
          m, n, k, alpha, devPtrA, lda, devPtrB, ldb, beta, devPtrC, ldc) &
          bind(C, name="spral_cublasgemm")
       use spral_kinds_precision
       type(C_PTR), value :: handle
       character(C_CHAR), intent(in) :: transa
       character(C_CHAR), intent(in) :: transb
       integer(C_IP_), intent(in) :: m
       integer(C_IP_), intent(in) :: n
       integer(C_IP_), intent(in) :: k
       real(C_RP_), intent(in) :: alpha
       real(C_RP_), intent(in) :: beta
       type(C_PTR), value :: devPtrA
       type(C_PTR), value :: devPtrB
       type(C_PTR), value :: devPtrC
       integer(C_IP_), intent(in) :: lda
       integer(C_IP_), intent(in) :: ldb
       integer(C_IP_), intent(in) :: ldc
     end function cublasgemm
     integer(C_IP_) function cublasSetStream(handle, streamId) &
          bind(C, name="spral_cublasSetStream")
       use spral_kinds
       type(C_PTR), value :: handle
       type(C_PTR), value :: streamId
     end function cublasSetStream
  end interface

  ! Additional functions that are hard to use CUDA without
  interface
     type(C_PTR) function c_ptr_plus(base, offset) &
          bind(C, name="spral_c_ptr_plus")
       use spral_kinds
       implicit none
       type(C_PTR), value :: base
       integer(C_SIZE_T), value :: offset
     end function c_ptr_plus
     subroutine c_print_ptr(ptr) bind(C, name="spral_c_print_ptr")
       use spral_kinds
       implicit none
       type(C_PTR), value :: ptr
     end subroutine c_print_ptr
  end interface

  ! Generic helper functions
  interface cudaMemcpy_h2d
     module procedure cudaMemcpy_h2d_ptr, cudaMemcpy_h2d_int, &
          cudaMemcpy_h2d_precision
  end interface cudaMemcpy_h2d

contains

  !
  ! Functions for creating aligned pointers
  !

  ! This function adds a size on to a pointer plus up to 256 bytes more to ensure
  ! that the returned pointer is correctly aligned for GPU usage
  type(C_PTR) function c_ptr_plus_aligned(base, sz)
    implicit none
    type(C_PTR), intent(in) :: base
    integer(C_SIZE_T), intent(in) :: sz

    c_ptr_plus_aligned = c_ptr_plus(base, aligned_size(sz))
  end function c_ptr_plus_aligned

  integer(C_SIZE_T) pure function aligned_size(sz)
    implicit none
    integer(C_SIZE_T), intent(in) :: sz

    integer(C_SIZE_T), parameter :: alignon = 256

    aligned_size = (sz + (alignon - 1)) / alignon
    aligned_size = aligned_size * alignon
  end function aligned_size

  !
  ! Implement our own cudaGetErrorString that returns a Fortran string
  ! as opposed to a character array
  !
  character(len=200) function cudaGetErrorString(error)
    implicit none
    integer(C_IP_) :: error

    integer(ip_) :: i
    type(C_PTR) :: cstr
    character(kind=C_CHAR), dimension(:), pointer, contiguous :: fstr

    interface
       type(C_PTR) function c_cudaGetErrorString(error) &
            bind(C, name="cudaGetErrorString")
         use spral_kinds
         integer(C_IP_), value :: error
       end function c_cudaGetErrorString
       integer(C_SIZE_T) function strlen(s) bind(C)
         use spral_kinds
         type(C_PTR), value :: s
      end function strlen
   end interface

   cstr = c_cudaGetErrorString(error)
   call C_F_POINTER(cstr, fstr, shape=(/strlen(cstr)/))
   cudaGetErrorString = ""
   do i = 1, min(size(fstr), 200)
      cudaGetErrorString(i:i) = fstr(i)
   end do
end function cudaGetErrorString

!
! Convieniece functions to avoid longwinded parameter passing in code
!
integer(C_IP_) function cudaMemcpy_h2d_ptr(dest, src, bytes)
  type(C_PTR), value :: dest
  type(C_PTR), value :: src
  integer(C_SIZE_T), intent(in) :: bytes

  cudaMemcpy_h2d_ptr = cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice)
end function cudaMemcpy_h2d_ptr
integer(C_IP_) function cudaMemcpy_h2d_int(dest, n, src)
  type(C_PTR), value :: dest
  integer(ip_), intent(in) :: n
  integer(C_IP_), dimension(n), target, intent(in) :: src

  cudaMemcpy_h2d_int = cudaMemcpy(dest, C_LOC(src), C_SIZEOF(src), &
       cudaMemcpyHostToDevice)
end function cudaMemcpy_h2d_int
integer(C_IP_) function cudaMemcpy_h2d_precision(dest, n, src)
  type(C_PTR), value :: dest
  integer(ip_), intent(in) :: n
  real(C_RP_), dimension(n), target, intent(in) :: src

  cudaMemcpy_h2d_precision = cudaMemcpy(dest, C_LOC(src), C_SIZEOF(src), &
       cudaMemcpyHostToDevice)
end function cudaMemcpy_h2d_precision
integer(C_IP_) function cudaMemcpy_d2h(dest, src, bytes)
  use spral_kinds
  type(C_PTR), value :: dest
  type(C_PTR), value :: src
  integer(C_SIZE_T), intent(in) :: bytes

  cudaMemcpy_d2h = cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToHost)
end function cudaMemcpy_d2h
integer(C_IP_) function cudaMemcpy_d2d(dest, src, bytes)
  use spral_kinds
  type(C_PTR), value :: dest
  type(C_PTR), value :: src
  integer(C_SIZE_T), intent(in) :: bytes

  cudaMemcpy_d2d = cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice)
end function cudaMemcpy_d2d
integer(C_IP_) function cudaMemcpyAsync_H2D(dst, src, count, stream)
  use spral_kinds
  type(C_PTR), value :: dst
  type(C_PTR), value :: src
  integer(C_SIZE_T), value :: count
  type(C_PTR), value :: stream

  cudaMemcpyAsync_H2D = cudaMemcpyAsync(dst, src, count, &
       cudaMemcpyHostToDevice, stream)
end function cudaMemcpyAsync_H2D
integer(C_IP_) function cudaMemcpyAsync_D2H(dst, src, count, stream)
  use spral_kinds
  type(C_PTR), value :: dst
  type(C_PTR), value :: src
  integer(C_SIZE_T), value :: count
  type(C_PTR), value :: stream

  cudaMemcpyAsync_D2H = cudaMemcpyAsync(dst, src, count, &
       cudaMemcpyDeviceToHost, stream)
end function cudaMemcpyAsync_D2H
integer(C_IP_) function cudaMemcpyAsync_D2D(dst, src, count, stream)
  use spral_kinds
  type(C_PTR), value :: dst
  type(C_PTR), value :: src
  integer(C_SIZE_T), value :: count
  type(C_PTR), value :: stream

  cudaMemcpyAsync_D2D = cudaMemcpyAsync(dst, src, count, &
       cudaMemcpyDeviceToDevice, stream)
end function cudaMemcpyAsync_D2D

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! Return true if a GPU is present and code is compiled with CUDA support
! FIXME: actually detect gpus
logical function detect_gpu()
   detect_gpu = .true.
end function detect_gpu

end module spral_cuda_precision
