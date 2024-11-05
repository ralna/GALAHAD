! THIS VERSION: GALAHAD 5.1 - 2023-11-05 AT 14:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  S H A    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SHA_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SHA_precision, ONLY:                                           &
        f_sha_control_type   => SHA_control_type,                              &
        f_sha_inform_type    => SHA_inform_type,                               &
        f_sha_full_data_type => SHA_full_data_type,                            &
        f_sha_initialize     => SHA_initialize,                                &
        f_sha_read_specfile  => SHA_read_specfile,                             &
        f_sha_reset_control  => SHA_reset_control,                             &
        f_sha_analyse_matrix => SHA_analyse_matrix,                            &
        f_sha_recover_matrix => SHA_recover_matrix,                            &
        f_sha_information    => SHA_information,                               &
        f_sha_terminate      => SHA_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: sha_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: approximation_algorithm
      INTEGER ( KIND = ipc_ ) :: dense_linear_solver
      INTEGER ( KIND = ipc_ ) :: extra_differences
      INTEGER ( KIND = ipc_ ) :: sparse_row
      INTEGER ( KIND = ipc_ ) :: recursion_max
      INTEGER ( KIND = ipc_ ) :: recursion_entries_required
      LOGICAL ( KIND = C_BOOL ) :: average_off_diagonals
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE sha_control_type

    TYPE, BIND( C ) :: sha_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: max_degree
      INTEGER ( KIND = ipc_ ) :: differences_needed
      INTEGER ( KIND = ipc_ ) :: max_reduced_degree
      INTEGER ( KIND = ipc_ ) :: approximation_algorithm_used
      INTEGER ( KIND = ipc_ ) :: bad_row
      REAL ( KIND = rpc_ ) :: max_off_diagonal_difference
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
    END TYPE sha_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( sha_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_sha_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%approximation_algorithm = ccontrol%approximation_algorithm
    fcontrol%dense_linear_solver = ccontrol%dense_linear_solver
    fcontrol%extra_differences = ccontrol%extra_differences
    fcontrol%sparse_row = ccontrol%sparse_row
    fcontrol%recursion_max = ccontrol%recursion_max
    fcontrol%recursion_entries_required = ccontrol%recursion_entries_required

    ! Logicals
    fcontrol%average_off_diagonals = ccontrol%average_off_diagonals
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Strings
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_sha_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( sha_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%approximation_algorithm = fcontrol%approximation_algorithm
    ccontrol%dense_linear_solver = fcontrol%dense_linear_solver
    ccontrol%extra_differences = fcontrol%extra_differences
    ccontrol%sparse_row = fcontrol%sparse_row
    ccontrol%recursion_max = fcontrol%recursion_max
    ccontrol%recursion_entries_required = fcontrol%recursion_entries_required

    ! Logicals
    ccontrol%average_off_diagonals = fcontrol%average_off_diagonals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Strings
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( sha_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_sha_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%max_degree = cinform%max_degree
    finform%differences_needed = cinform%differences_needed
    finform%max_reduced_degree = cinform%max_reduced_degree
    finform%approximation_algorithm_used = cinform%approximation_algorithm_used
    finform%bad_row = cinform%bad_row

    ! Reals
    finform%max_off_diagonal_difference = cinform%max_off_diagonal_difference

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_sha_inform_type ), INTENT( IN ) :: finform
    TYPE ( sha_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%max_degree = finform%max_degree
    cinform%differences_needed = finform%differences_needed
    cinform%max_reduced_degree = finform%max_reduced_degree
    cinform%approximation_algorithm_used = finform%approximation_algorithm_used
    cinform%bad_row = finform%bad_row

    ! Reals
    cinform%max_off_diagonal_difference = finform%max_off_diagonal_difference

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SHA_precision_ciface

!  -------------------------------------
!  C interface to fortran sha_initialize
!  -------------------------------------

  SUBROUTINE sha_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( sha_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_sha_full_data_type ), POINTER :: fdata
  TYPE ( f_sha_control_type ) :: fcontrol
  TYPE ( f_sha_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_sha_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sha_initialize

!  ----------------------------------------
!  C interface to fortran sha_read_specfile
!  ----------------------------------------

  SUBROUTINE sha_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( sha_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_sha_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_sha_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sha_read_specfile

!  ----------------------------------------
!  C interface to fortran sha_reset_control
!  ----------------------------------------

  SUBROUTINE sha_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sha_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sha_control_type ) :: fcontrol
  TYPE ( f_sha_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_sha_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE sha_reset_control

!  -----------------------------------------
!  C interface to fortran sha_analyse_matrix
!  -----------------------------------------

  SUBROUTINE sha_analyse_matrix( ccontrol, cdata, status, n, ne,               &
                                 row, col, m ) BIND( C )
!                                row, col ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sha_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ) :: row
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ) :: col
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: m
! INTEGER ( KIND = ipc_ ) :: m

!  local variables

  TYPE ( f_sha_control_type ) :: fcontrol
  TYPE ( f_sha_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  analyse_matrix the problem data into the required SHA structure

  CALL f_sha_analyse_matrix( fcontrol, fdata, status, n, ne, row, col, m )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sha_analyse_matrix

!  -----------------------------------------
!  C interface to fortran sha_recover_matrix
!  -----------------------------------------

  SUBROUTINE sha_recover_matrix( cdata, status, ne, m, ls1, ls2, s,         &
                                 ly1, ly2, y, val, order ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: ne, m, ls1, ls2, ly1, ly2
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ls2, ls1 ) :: s ! reverse order
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ly2, ly1 ) :: y ! for C !!
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( ne ) :: val
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m ), OPTIONAL :: order
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sha_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_sha_recover_matrix( fdata, status, m, s, y, val,                      &
                             order = order )
  RETURN

  END SUBROUTINE sha_recover_matrix

!  --------------------------------------
!  C interface to fortran sha_information
!  --------------------------------------

  SUBROUTINE sha_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sha_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_sha_full_data_type ), POINTER :: fdata
  TYPE ( f_sha_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain SHA solution information

  CALL f_sha_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE sha_information

!  ------------------------------------
!  C interface to fortran sha_terminate
!  ------------------------------------

  SUBROUTINE sha_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SHA_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sha_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( sha_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_sha_full_data_type ), pointer :: fdata
  TYPE ( f_sha_control_type ) :: fcontrol
  TYPE ( f_sha_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_sha_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE sha_terminate
