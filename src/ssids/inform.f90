!> \file
!> \copyright 2016 The Science and Technology Facilities Council (STFC)
!> \licence   BSD licence, see LICENCE file for details
!> \author    Jonathan Hogg
module spral_ssids_inform
  use spral_cuda, only : cudaGetErrorString
  use spral_scaling, only : auction_inform
  use spral_ssids_datatypes
  implicit none

  private
  public :: ssids_inform

  !
  ! Data type for information returned by code
  !
  type ssids_inform
     integer :: flag = SSIDS_SUCCESS ! Takes one of the enumerated flag values:
         ! SSIDS_SUCCESS
         ! SSIDS_ERROR_XXX
         ! SSIDS_WARNING_XXX
     integer :: matrix_dup = 0 ! Number of duplicated entries.
     integer :: matrix_missing_diag = 0 ! Number of missing diag. entries
     integer :: matrix_outrange = 0 ! Number of out-of-range entries.
     integer :: matrix_rank = 0 ! Rank of matrix (anal=structral, fact=actual)
     integer :: maxdepth = 0 ! Maximum depth of tree
     integer :: maxfront = 0 ! Maximum front size
     integer :: num_delay = 0 ! Number of delayed variables
     integer(long) :: num_factor = 0_long ! Number of entries in factors
     integer(long) :: num_flops = 0_long ! Number of floating point operations
     integer :: num_neg = 0 ! Number of negative pivots
     integer :: num_sup = 0 ! Number of supernodes
     integer :: num_two = 0 ! Number of 2x2 pivots used by factorization
     integer :: stat = 0 ! stat parameter
     type(auction_inform) :: auction
     integer :: cuda_error = 0
     integer :: cublas_error = 0

     ! Undocumented FIXME: should we document them?
     integer :: not_first_pass = 0
     integer :: not_second_pass = 0
     integer :: nparts = 0
     integer(long) :: cpu_flops = 0
     integer(long) :: gpu_flops = 0
   contains
     procedure :: flag_to_character
     procedure :: print_flag
     procedure :: reduce
  end type ssids_inform

contains

!
! Returns a string representation
! Member function inform%flagToCharacter
!
  function flag_to_character(this) result(msg)
    implicit none
    class(ssids_inform), intent(in) :: this
    character(len=200) :: msg ! return value

    select case(this%flag)
       !
       ! Success
       !
    case(SSIDS_SUCCESS)
       msg = 'Success'
       !
       ! Errors
       !
    case(SSIDS_ERROR_CALL_SEQUENCE)
       msg = 'Error in sequence of calls.'
    case(SSIDS_ERROR_A_N_OOR)
       msg = 'n or ne is out of range (or has changed)'
    case(SSIDS_ERROR_A_PTR)
       msg = 'Error in ptr'
    case(SSIDS_ERROR_A_ALL_OOR)
       msg = 'All entries in a column out-of-range (ssids_analyse) &
            &or all entries out-of-range (ssids_analyse_coord)'
    case(SSIDS_ERROR_SINGULAR)
       msg = 'Matrix found to be singular'
    case(SSIDS_ERROR_NOT_POS_DEF)
       msg = 'Matrix is not positive-definite'
    case(SSIDS_ERROR_PTR_ROW)
       msg = 'ptr and row should be present'
    case(SSIDS_ERROR_ORDER)
       msg = 'Either control%ordering out of range or error in user-supplied  &
            &elimination order'
    case(SSIDS_ERROR_X_SIZE)
       msg = 'Error in size of x or nrhs'
    case(SSIDS_ERROR_JOB_OOR)
       msg = 'job out of range'
    case(SSIDS_ERROR_NOT_LLT)
       msg = 'Not a LL^T factorization of a positive-definite matrix'
    case(SSIDS_ERROR_NOT_LDLT)
       msg = 'Not a LDL^T factorization of an indefinite matrix'
    case(SSIDS_ERROR_ALLOCATION)
       write (msg,'(a,i6)') 'Allocation error. stat parameter = ', this%stat
    case(SSIDS_ERROR_VAL)
       msg = 'Optional argument val not present when expected'
    case(SSIDS_ERROR_NO_SAVED_SCALING)
       msg = 'Requested use of scaling from matching-based &
            &ordering but matching-based ordering not used'
    case(SSIDS_ERROR_UNIMPLEMENTED)
       msg = 'Functionality not yet implemented'
    case(SSIDS_ERROR_CUDA_UNKNOWN)
       write(msg,'(2a)') ' Unhandled CUDA error: ', &
            trim(cudaGetErrorString(this%cuda_error))
    case(SSIDS_ERROR_CUBLAS_UNKNOWN)
       msg = 'Unhandled CUBLAS error:'
!$  case(SSIDS_ERROR_OMP_CANCELLATION)
!$     msg = 'SSIDS CPU code requires OMP cancellation to be enabled'

       !
       ! Warnings
       !
    case(SSIDS_WARNING_IDX_OOR)
       msg = 'out-of-range indices detected'
    case(SSIDS_WARNING_DUP_IDX)
       msg = 'duplicate entries detected'
    case(SSIDS_WARNING_DUP_AND_OOR)
       msg = 'out-of-range indices detected and duplicate entries detected'
    case(SSIDS_WARNING_MISSING_DIAGONAL)
       msg = 'one or more diagonal entries is missing'
    case(SSIDS_WARNING_MISS_DIAG_OORDUP)
       msg = 'one or more diagonal entries is missing and out-of-range and/or &
            &duplicate entries detected'
    case(SSIDS_WARNING_ANAL_SINGULAR)
       msg = 'Matrix found to be structually singular'
    case(SSIDS_WARNING_FACT_SINGULAR)
       msg = 'Matrix found to be singular'
    case(SSIDS_WARNING_MATCH_ORD_NO_SCALE)
       msg = 'Matching-based ordering used but associated scaling ignored'
!$  case(SSIDS_WARNING_OMP_PROC_BIND)
!$     msg = 'OMP_PROC_BIND=false, this may reduce performance'
    case default
       msg = 'SSIDS Internal Error'
    end select
  end function flag_to_character

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!> @brief Print out warning or error if flag is non-zero
!> @param this Instance variable.
!> @param options Options to be used for printing
!> @param context Name of routine to report error from
  subroutine print_flag(this, options, context)
    implicit none
    class(ssids_inform), intent(in) :: this
    type(ssids_options), intent(in) :: options
    character (len=*), intent(in) :: context

    character(len=200) :: msg

    if (this%flag .eq. SSIDS_SUCCESS) return ! Nothing to print
    if (options%print_level .lt. 0) return ! No printing
    if (this%flag .gt. SSIDS_SUCCESS) then
       ! Warning
       if (options%unit_warning .lt. 0) return ! printing supressed
       write (options%unit_warning,'(/3a,i3)') ' Warning from ', &
            trim(context), '. Warning flag = ', this%flag
       msg = this%flag_to_character()
       write (options%unit_warning, '(a)') msg
    else
       if (options%unit_error .lt. 0) return ! printing supressed
       write (options%unit_error,'(/3a,i3)') ' Error return from ', &
            trim(context), '. Error flag = ', this%flag
       msg = this%flag_to_character()
       write (options%unit_error, '(a)') msg
    end if
  end subroutine print_flag

!> @brief Combine other's values into this object.
!>
!> Primarily intended for reducing inform objects after parallel execution.
!> @param this Instance object.
!> @param other Object to reduce values from
  subroutine reduce(this, other)
    implicit none
    class(ssids_inform), intent(inout) :: this
    class(ssids_inform), intent(in) :: other

    if ((this%flag .lt. 0) .or. (other%flag .lt. 0)) then
       ! An error is present
       this%flag = min(this%flag, other%flag)
    else
       ! Otherwise only success if both are zero
       this%flag = max(this%flag, other%flag)
    end if
    this%matrix_dup = this%matrix_dup + other%matrix_dup
    this%matrix_missing_diag = this%matrix_missing_diag + &
         other%matrix_missing_diag
    this%matrix_outrange = this%matrix_outrange + other%matrix_outrange
    this%matrix_rank = this%matrix_rank + other%matrix_rank
    this%maxdepth = max(this%maxdepth, other%maxdepth)
    this%maxfront = max(this%maxfront, other%maxfront)
    this%num_delay = this%num_delay + other%num_delay
    this%num_factor = this%num_factor + other%num_factor
    this%num_flops = this%num_flops + other%num_flops
    this%num_neg = this%num_neg + other%num_neg
    this%num_sup = this%num_sup + other%num_sup
    this%num_two = this%num_two + other%num_two
    if (other%stat .ne. 0) this%stat = other%stat
    ! FIXME: %auction ???
    if (other%cuda_error .ne. 0) this%cuda_error = other%cuda_error
    if (other%cublas_error .ne. 0) this%cublas_error = other%cublas_error
    this%not_first_pass = this%not_first_pass + other%not_first_pass
    this%not_second_pass = this%not_second_pass + other%not_second_pass
    this%nparts = this%nparts + other%nparts
    this%cpu_flops = this%cpu_flops + other%cpu_flops
    this%gpu_flops = this%gpu_flops + other%gpu_flops
  end subroutine reduce
end module spral_ssids_inform
