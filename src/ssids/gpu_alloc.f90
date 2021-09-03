module spral_ssids_gpu_alloc
  use, intrinsic :: iso_c_binding
  use spral_cuda
  implicit none

  private
  ! CUDA stack allocator
  public :: cuda_stack_alloc_type, & ! Data type
            custack_init,          & ! Initialize stack allocator
            custack_alloc,         & ! Allocate from top of stack
            custack_free,          & ! Free from top of stack
            custack_finalize         ! Free memory associated with stack

  type cuda_stack_alloc_type
     private
     type(C_PTR) :: stack = C_NULL_PTR ! GPU pointer to memory
     integer(C_SIZE_T) :: stack_sz = 0 ! Size of stack
     integer(C_SIZE_T) :: top = 0 ! Current top of stack
  end type cuda_stack_alloc_type

contains

  subroutine custack_init(stack, bytes, cuda_error)
    implicit none
    type(cuda_stack_alloc_type), intent(inout) :: stack
    integer(C_SIZE_T), intent(in) :: bytes
    integer, intent(out) :: cuda_error

    ! integer(C_SIZE_T) :: free, total

    cuda_error = 0 ! All is good

    ! Check stack not still in use
    if (stack%top .ne. 0) then
       ! We should never reach this point
       print *, "Attempting to resize non-empty stack!"
       stop
    end if

    ! If stack is already large enough, do nothing
    if (bytes .le. stack%stack_sz) return

    ! Free any preexisting stack
    if (C_ASSOCIATED(stack%stack)) then
       cuda_error = cudaFree(stack%stack)
       if (cuda_error .ne. 0) return
    end if

    ! Always align!
    stack%stack_sz = aligned_size(bytes)

    ! cuda_error = cudaMemGetInfo(free, total)
    ! print *, "[custack_init] Mem free (MB) = ", free/(1024.0*1024.0), ", total (MB) = ", total/(1024.0*1024.0)
    ! print *, "[custack_init] stack_sz (MB) = ", stack%stack_sz/(1024.0*1024.0)

    ! Allocate stack to new size
    cuda_error = cudaMalloc(stack%stack, stack%stack_sz)
    if (cuda_error .ne. 0) return
  end subroutine custack_init

  subroutine custack_finalize(stack, cuda_error)
    implicit none
    type(cuda_stack_alloc_type), intent(inout) :: stack
    integer, intent(out) :: cuda_error

    cuda_error = 0 ! All is good

    ! Reset information
    stack%top = 0
    stack%stack_sz = 0

    ! Don't bother trying to free memory if none allocated
    if (.not. C_ASSOCIATED(stack%stack)) return ! Not initialized

    ! Free memory
    cuda_error = cudaFree(stack%stack)
    if (cuda_error .ne. 0) return

    ! Nullify pointer
    stack%stack = C_NULL_PTR
  end subroutine custack_finalize

  type(C_PTR) function custack_alloc(stack, bytes)
    implicit none
    type(cuda_stack_alloc_type), intent(inout) :: stack
    integer(C_SIZE_T), intent(in) :: bytes

    integer(C_SIZE_T) :: bytes_aligned ! Alloc size has to round up to nearest
      ! multiple of 256 bytes

    bytes_aligned = aligned_size(bytes)
    if ((stack%top + bytes_aligned) .gt. stack%stack_sz) then
       ! Insufficient space
       custack_alloc = C_NULL_PTR
       return
    end if

    custack_alloc = c_ptr_plus(stack%stack, stack%top)
    stack%top = stack%top + bytes_aligned
  end function custack_alloc

  subroutine custack_free(stack, bytes)
    implicit none
    type(cuda_stack_alloc_type), intent(inout) :: stack
    integer(C_SIZE_T), intent(in) :: bytes

    integer(C_SIZE_T) :: bytes_aligned ! Alloc size has to round up to nearest
      ! multiple of 256 bytes

    bytes_aligned = aligned_size(bytes)
    stack%top = stack%top - bytes_aligned
  end subroutine custack_free

end module spral_ssids_gpu_alloc
