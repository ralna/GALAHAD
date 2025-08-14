! THIS VERSION: GALAHAD 5.3 - 2025-08-13 AT 13:40 GMT

#include "spral_procedures.h"

!  copyright 2016 The Science and Technology Facilities Council (STFC)
!  licence   BSD licence, see LICENCE file for details
!  author    Jonathan Hogg

MODULE GALAHAD_SSIDS_INFORM_precision
  USE SPRAL_KINDS_precision
  USE SPRAL_CUDA_precision, ONLY: cudaGetErrorString
  USE SPRAL_SCALING_precision, ONLY: auction_inform
  USE GALAHAD_SSIDS_TYPES_precision
  USE GALAHAD_NODEND_precision, ONLY: NODEND_inform_type
  IMPLICIT NONE

  PRIVATE
  PUBLIC :: SSIDS_inform_type

!  Data type for information returned by code

  TYPE SSIDS_inform_type
     INTEGER( ip_ ) :: flag = SSIDS_SUCCESS ! Takes one of the enumerated
         ! flag values:
         !  SSIDS_SUCCESS
         !  SSIDS_ERROR_XXX
         !  SSIDS_WARNING_XXX
     INTEGER( ip_ ) :: matrix_dup = 0 ! # duplicated entries.
     INTEGER( ip_ ) :: matrix_missing_diag = 0 ! # missing diagonal entries
     INTEGER( ip_ ) :: matrix_outrange = 0 ! # out-of-range entries.
     INTEGER( ip_ ) :: matrix_rank = 0 ! Rank of matrix (anal=structral,
                                     ! fact=actual)
     INTEGER( ip_ ) :: maxdepth = 0 ! Maximum depth of tree
     INTEGER( ip_ ) :: maxfront = 0 ! Maximum front size
     INTEGER( ip_ ) :: maxsupernode = 0 ! Maximum supernode size
     INTEGER( ip_ ) :: num_delay = 0 ! # delayed variables
     INTEGER( long_ ) :: num_factor = 0_long_ ! # entries in factors
     INTEGER( long_ ) :: num_flops = 0_long_ ! # floating point operations
     INTEGER( ip_ ) :: num_neg = 0 ! # negative pivots
     INTEGER( ip_ ) :: num_sup = 0 ! # supernodes
     INTEGER( ip_ ) :: num_two = 0 ! # 2x2 pivots used by factorization
     INTEGER( ip_ ) :: stat = 0 ! stat parameter
     TYPE( auction_inform ) :: auction
     INTEGER( ip_ ) :: cuda_error = 0
     INTEGER( ip_ ) :: cublas_error = 0
     TYPE( NODEND_inform_type ) :: nodend_inform

     ! Undocumented FIXME: should we document them?
     INTEGER( ip_ ) :: not_first_pass = 0
     INTEGER( ip_ ) :: not_second_pass = 0
     INTEGER( ip_ ) :: nparts = 0
     INTEGER( long_ ) :: cpu_flops = 0
     INTEGER( long_ ) :: gpu_flops = 0
     ! character(C_CHAR) :: unused(76)
   CONTAINS
     PROCEDURE :: flag_to_character
     PROCEDURE :: print_flag
     PROCEDURE :: reduce
  END TYPE SSIDS_inform_type

CONTAINS

  FUNCTION flag_to_character(this) result(msg)

!  returns a string representation
!  member function inform%flagToCharacter

    IMPLICIT NONE
    CLASS( SSIDS_inform_type ), INTENT( IN ) :: this
    CHARACTER( len=200 ) :: msg ! return value

    SELECT CASE(this%flag)
       !
       ! Success
       !
    CASE(SSIDS_SUCCESS)
       msg = 'Success'
       !
       ! Errors
       !
    CASE(SSIDS_ERROR_CALL_SEQUENCE)
       msg = 'Error in sequence of calls.'
    CASE(SSIDS_ERROR_A_N_OOR)
       msg = 'n or ne is out of range (or has changed)'
    CASE(SSIDS_ERROR_A_PTR)
       msg = 'Error in ptr'
    CASE(SSIDS_ERROR_A_ALL_OOR)
       msg = 'All entries in a column out-of-range (ssids_analyse) &
            &or all entries out-of-range (ssids_analyse_coord)'
    CASE(SSIDS_ERROR_SINGULAR)
       msg = 'Matrix found to be singular'
    CASE(SSIDS_ERROR_NOT_POS_DEF)
       msg = 'Matrix is not positive-definite'
    CASE(SSIDS_ERROR_PTR_ROW)
       msg = 'ptr and row should be present'
    CASE(SSIDS_ERROR_ORDER)
       msg = 'Either control%ordering out of range or error in user-supplied  &
            &elimination order'
    CASE(SSIDS_ERROR_X_SIZE)
       msg = 'Error in size of x or nrhs'
    CASE(SSIDS_ERROR_JOB_OOR)
       msg = 'job out of range'
    CASE(SSIDS_ERROR_NOT_LLT)
       msg = 'Not a LL^T factorization of a positive-definite matrix'
    CASE(SSIDS_ERROR_NOT_LDLT)
       msg = 'Not a LDL^T factorization of an indefinite matrix'
    CASE(SSIDS_ERROR_ALLOCATION)
       write (msg,'(a,i6)') 'Allocation error. stat parameter = ', this%stat
    CASE(SSIDS_ERROR_VAL)
       msg = 'Optional argument val not present when expected'
    CASE(SSIDS_ERROR_NO_SAVED_SCALING)
       msg = 'Requested use of scaling from matching-based &
            &ordering but matching-based ordering not used'
    CASE(SSIDS_ERROR_UNIMPLEMENTED)
       msg = 'Functionality not yet implemented'
    CASE(SSIDS_ERROR_CUDA_UNKNOWN)
       write(msg,'(2a)') ' Unhandled CUDA error: ', &
            trim(cudaGetErrorString(this%cuda_error))
    CASE(SSIDS_ERROR_CUBLAS_UNKNOWN)
       msg = 'Unhandled CUBLAS error:'
!$  CASE(SSIDS_ERROR_OMP_CANCELLATION)
!$     msg = 'SSIDS CPU code requires OMP cancellation to be enabled'
    CASE(SSIDS_ERROR_NO_METIS)
       msg = 'MeTiS is not available'

!  warnings

    CASE(SSIDS_WARNING_IDX_OOR)
       msg = 'out-of-range indices detected'
    CASE(SSIDS_WARNING_DUP_IDX)
       msg = 'duplicate entries detected'
    CASE(SSIDS_WARNING_DUP_AND_OOR)
       msg = 'out-of-range indices detected and duplicate entries detected'
    CASE(SSIDS_WARNING_MISSING_DIAGONAL)
       msg = 'one or more diagonal entries is missing'
    CASE(SSIDS_WARNING_MISS_DIAG_OORDUP)
       msg = 'one or more diagonal entries is missing and out-of-range and/or &
            &duplicate entries detected'
    CASE(SSIDS_WARNING_ANAL_SINGULAR)
       msg = 'Matrix found to be structually singular'
    CASE(SSIDS_WARNING_FACT_SINGULAR)
       msg = 'Matrix found to be singular'
    CASE(SSIDS_WARNING_MATCH_ORD_NO_SCALE)
       msg = 'Matching-based ordering used but associated scaling ignored'
!$  CASE(SSIDS_WARNING_OMP_PROC_BIND)
!$     msg = 'OMP_PROC_BIND=false, this may reduce performance'
    CASE DEFAULT
       msg = 'SSIDS Internal Error'
    END SELECT
  END FUNCTION flag_to_character

  SUBROUTINE print_flag(this, options, context)

!  print out warning or error if flag is non-zero
!   this Instance variable.
!   options Options to be used for printing
!   context Name of routine to report error from

    IMPLICIT none
    CLASS( SSIDS_inform_type ), INTENT( IN ) :: this
    TYPE( SSIDS_options_type ), INTENT( IN ) :: options
    CHARACTER( len = * ), INTENT( IN ) :: context

    CHARACTER( len=200 ) :: msg

    IF ( this%flag == SSIDS_SUCCESS ) RETURN ! Nothing to print
    IF ( options%print_level < 0) RETURN ! No printing

!  warning

    IF ( this%flag > SSIDS_SUCCESS ) THEN
      IF ( options%unit_warning < 0 ) RETURN ! printing supressed
      WRITE( options%unit_warning,'(/3a,i0)') ' Warning from ',                &
           TRIM( context ), '. Warning flag = ', this%flag
      msg = this%flag_to_character( )
      WRITE (options%unit_warning, '(a)') msg
    ELSE
      IF ( options%unit_error < 0 ) RETURN ! printing supressed
      WRITE( options%unit_error,'(/3a,i0)') ' Error return from ',             &
           TRIM( context ), '. Error flag = ', this%flag
      msg = this%flag_to_character( )
      WRITE( options%unit_error, '(a)') msg
    END IF
  END SUBROUTINE print_flag

  SUBROUTINE reduce( this, other )

!  combine other's values into this object.
!
!  primarily intended for reducing inform objects after parallel execution.
!   this  instance object
!   other object to reduce values from

    IMPLICIT none
    CLASS( SSIDS_inform_type ), INTENT( INOUT ) :: this
    CLASS( SSIDS_inform_type ), INTENT( IN ) :: other

    IF ( this%flag < 0 .OR. other%flag < 0 ) THEN
!  an error is present
      this%flag = MIN( this%flag, other%flag )
    ELSE
!  otherwise only success if both are zero
      this%flag = MAX( this%flag, other%flag )
    END IF
    this%matrix_dup = this%matrix_dup + other%matrix_dup
    this%matrix_missing_diag = this%matrix_missing_diag +                      &
         other%matrix_missing_diag
    this%matrix_outrange = this%matrix_outrange + other%matrix_outrange
    this%matrix_rank = this%matrix_rank + other%matrix_rank
    this%maxdepth = max(this%maxdepth, other%maxdepth)
    this%maxfront = max(this%maxfront, other%maxfront)
    this%maxsupernode = max(this%maxsupernode, other%maxsupernode)
    this%num_delay = this%num_delay + other%num_delay
    this%num_factor = this%num_factor + other%num_factor
    this%num_flops = this%num_flops + other%num_flops
    this%num_neg = this%num_neg + other%num_neg
    this%num_sup = this%num_sup + other%num_sup
    this%num_two = this%num_two + other%num_two
    IF ( other%stat /= 0 ) this%stat = other%stat
    ! FIXME: %auction ???
    IF ( other%cuda_error /= 0 ) this%cuda_error = other%cuda_error
    IF ( other%cublas_error /= 0 ) this%cublas_error = other%cublas_error
    this%not_first_pass = this%not_first_pass + other%not_first_pass
    this%not_second_pass = this%not_second_pass + other%not_second_pass
    this%nparts = this%nparts + other%nparts
    this%cpu_flops = this%cpu_flops + other%cpu_flops
    this%gpu_flops = this%gpu_flops + other%gpu_flops
  END SUBROUTINE reduce
END MODULE GALAHAD_SSIDS_INFORM_precision
