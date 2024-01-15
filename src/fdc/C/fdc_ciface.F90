! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  F D C    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 13th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_FDC_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_FDC_precision, ONLY:                                           &
        f_fdc_control_type        => FDC_control_type,                         &
        f_fdc_time_type           => FDC_time_type,                            &
        f_fdc_inform_type         => FDC_inform_type,                          &
        f_fdc_full_data_type      => FDC_full_data_type,                       &
        f_fdc_initialize          => FDC_initialize,                           &
        f_fdc_read_specfile       => FDC_read_specfile,                        &
        f_fdc_find_dependent_rows => FDC_find_dependent_rows,                  &
        f_fdc_terminate           => FDC_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in   => copy_inform_in,                                &
        copy_sls_inform_out  => copy_inform_out,                               &
        copy_sls_control_in  => copy_control_in,                               &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_ULS_precision_ciface, ONLY:                                    &
        uls_inform_type,                                                       &
        uls_control_type,                                                      &
        copy_uls_inform_in   => copy_inform_in,                                &
        copy_uls_inform_out  => copy_inform_out,                               &
        copy_uls_control_in  => copy_control_in,                               &
        copy_uls_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: fdc_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: indmin
      INTEGER ( KIND = ipc_ ) :: valmin
      REAL ( KIND = rpc_ ) :: pivot_tol
      REAL ( KIND = rpc_ ) :: zero_pivot
      REAL ( KIND = rpc_ ) :: max_infeas
      LOGICAL ( KIND = C_BOOL ) :: use_sls
      LOGICAL ( KIND = C_BOOL ) :: scale
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: unsymmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( uls_control_type ) :: uls_control
    END TYPE fdc_control_type

    TYPE, BIND( C ) :: fdc_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
    END TYPE fdc_time_type

    TYPE, BIND( C ) :: fdc_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      REAL ( KIND = rpc_ ) :: non_negligible_pivot
      TYPE ( fdc_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( uls_inform_type ) :: uls_inform
    END TYPE fdc_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( fdc_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_fdc_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%indmin = ccontrol%indmin
    fcontrol%valmin = ccontrol%valmin

    ! Reals
    fcontrol%pivot_tol = ccontrol%pivot_tol
    fcontrol%zero_pivot = ccontrol%zero_pivot
    fcontrol%max_infeas = ccontrol%max_infeas

    ! Logicals
    fcontrol%use_sls = ccontrol%use_sls
    fcontrol%scale = ccontrol%scale
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_uls_control_in( ccontrol%uls_control, fcontrol%uls_control )

    ! Strings
    DO i = 1, LEN( fcontrol%symmetric_linear_solver )
      IF ( ccontrol%symmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%symmetric_linear_solver( i : i )                                &
        = ccontrol%symmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%unsymmetric_linear_solver )
      IF ( ccontrol%unsymmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%unsymmetric_linear_solver( i : i )                              &
        = ccontrol%unsymmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_fdc_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( fdc_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%indmin = fcontrol%indmin
    ccontrol%valmin = fcontrol%valmin

    ! Reals
    ccontrol%pivot_tol = fcontrol%pivot_tol
    ccontrol%zero_pivot = fcontrol%zero_pivot
    ccontrol%max_infeas = fcontrol%max_infeas

    ! Logicals
    ccontrol%use_sls = fcontrol%use_sls
    ccontrol%scale = fcontrol%scale
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_uls_control_out( fcontrol%uls_control, ccontrol%uls_control )

    ! Strings
    l = LEN( fcontrol%symmetric_linear_solver )
    DO i = 1, l
      ccontrol%symmetric_linear_solver( i )                                    &
        = fcontrol%symmetric_linear_solver( i : i )
    END DO
    ccontrol%symmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%unsymmetric_linear_solver )
    DO i = 1, l
      ccontrol%unsymmetric_linear_solver( i )                                  &
        = fcontrol%unsymmetric_linear_solver( i : i )
    END DO
    ccontrol%unsymmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( fdc_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_fdc_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%clock_total = ctime%clock_total
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_fdc_time_type ), INTENT( IN ) :: ftime
    TYPE ( fdc_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%clock_total = ftime%clock_total
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( fdc_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_fdc_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%factorization_status = cinform%factorization_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real

    ! Reals
    finform%non_negligible_pivot = cinform%non_negligible_pivot

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_uls_inform_in( cinform%uls_inform, finform%uls_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_fdc_inform_type ), INTENT( IN ) :: finform
    TYPE ( fdc_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%factorization_status = finform%factorization_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real

    ! Reals
    cinform%non_negligible_pivot = finform%non_negligible_pivot

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_uls_inform_out( finform%uls_inform, cinform%uls_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_FDC_precision_ciface

!  -------------------------------------
!  C interface to fortran fdc_initialize
!  -------------------------------------

  SUBROUTINE fdc_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_FDC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( fdc_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_fdc_full_data_type ), POINTER :: fdata
  TYPE ( f_fdc_control_type ) :: fcontrol
  TYPE ( f_fdc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_fdc_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE fdc_initialize

!  ----------------------------------------
!  C interface to fortran fdc_read_specfile
!  ----------------------------------------

  SUBROUTINE fdc_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_FDC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( fdc_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_fdc_control_type ) :: fcontrol
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

  CALL f_fdc_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE fdc_read_specfile

!  ----------------------------------------------
!  C interface to fortran fdc_find_dependent_rows
!  ----------------------------------------------

  SUBROUTINE fdc_find_dependent_rows( ccontrol, cdata, cinform, status,        &
                                      m, n, ane, acol, aptr, aval, b,          &
                                      n_depen, depen ) BIND( C )

  USE GALAHAD_FDC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( fdc_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( fdc_inform_type ), INTENT( OUT ) :: cinform
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, n, ane
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: n_depen
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ) :: aptr
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ) :: acol
! REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( aptr( m + 1 ) - 1 ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION(  m ) :: b
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION(  m ) :: depen

!  local variables

  TYPE ( f_fdc_control_type ) :: fcontrol
  TYPE ( f_fdc_full_data_type ), POINTER :: fdata
  TYPE ( f_fdc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  CALL f_FDC_find_dependent_rows( fcontrol, fdata, finform, status,            &
                                  m, n, acol, aptr, aval, b, n_depen, depen )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE fdc_find_dependent_rows

!  ------------------------------------
!  C interface to fortran fdc_terminate
!  ------------------------------------

  SUBROUTINE fdc_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_FDC_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( fdc_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( fdc_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_fdc_full_data_type ), pointer :: fdata
  TYPE ( f_fdc_control_type ) :: fcontrol
  TYPE ( f_fdc_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_fdc_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE fdc_terminate
