! THIS VERSION: GALAHAD 5.3 - 2025-07-23 AT 13:00 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  S B L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 5.3. July 23rd 2025

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SSLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SSLS_precision, ONLY:                                          &
        f_ssls_control_type     => SSLS_control_type,                          &
        f_ssls_time_type        => SSLS_time_type,                             &
        f_ssls_inform_type      => SSLS_inform_type,                           &
        f_ssls_full_data_type   => SSLS_full_data_type,                        &
        f_ssls_initialize       => SSLS_initialize,                            &
        f_ssls_read_specfile    => SSLS_read_specfile,                         &
        f_ssls_import           => SSLS_import,                                &
        f_ssls_reset_control    => SSLS_reset_control,                         &
        f_ssls_factorize_matrix => SSLS_factorize_matrix,                      &
        f_ssls_solve_system     => SSLS_solve_system,                          &
        f_ssls_information      => SSLS_information,                           &
        f_ssls_terminate        => SSLS_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in   => copy_inform_in,                                &
        copy_sls_inform_out  => copy_inform_out,                               &
        copy_sls_control_in  => copy_control_in,                               &
        copy_sls_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: ssls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
    END TYPE ssls_control_type

    TYPE, BIND( C ) :: ssls_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE ssls_time_type

    TYPE, BIND( C ) :: ssls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      INTEGER ( KIND = ipc_ ) :: rank
      LOGICAL ( KIND = C_BOOL ) :: rank_def
      TYPE ( ssls_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
    END TYPE ssls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( ssls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_ssls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level

    ! Logicals
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )

    ! Strings
    DO i = 1, LEN( fcontrol%symmetric_linear_solver )
      IF ( ccontrol%symmetric_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%symmetric_linear_solver( i : i )                                &
        = ccontrol%symmetric_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_ssls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( ssls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level

    ! Logicals
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )

    ! Strings
    l = LEN( fcontrol%symmetric_linear_solver )
    DO i = 1, l
      ccontrol%symmetric_linear_solver( i )                                    &
        = fcontrol%symmetric_linear_solver( i : i )
    END DO
    ccontrol%symmetric_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( ssls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_ssls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%clock_total = ctime%clock_total
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_ssls_time_type ), INTENT( IN ) :: ftime
    TYPE ( ssls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%clock_total = ftime%clock_total
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( ssls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_ssls_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%rank = cinform%rank

    ! Logicals
    finform%rank_def = cinform%rank_def

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_ssls_inform_type ), INTENT( IN ) :: finform
    TYPE ( ssls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%rank = finform%rank

    ! Logicals
    cinform%rank_def = finform%rank_def

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_SSLS_precision_ciface

!  -------------------------------------
!  C interface to fortran ssls_initialize
!  -------------------------------------

  SUBROUTINE ssls_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( ssls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_ssls_full_data_type ), POINTER :: fdata
  TYPE ( f_ssls_control_type ) :: fcontrol
  TYPE ( f_ssls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_ssls_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE ssls_initialize

!  ----------------------------------------
!  C interface to fortran ssls_read_specfile
!  ----------------------------------------

  SUBROUTINE ssls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( ssls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_ssls_control_type ) :: fcontrol
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

  CALL f_ssls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE ssls_read_specfile

!  ----------------------------------
!  C interface to fortran ssls_import
!  ----------------------------------

  SUBROUTINE ssls_import( ccontrol, cdata, status, n, m,                       &
                          chtype, hne, hrow, hcol, hptr,                       &
                          catype, ane, arow, acol, aptr,                       &
                          cctype, cne, crow, ccol, cptr ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( ssls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, ane, cne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( cne ), OPTIONAL :: crow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( cne ), OPTIONAL :: ccol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: cptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cctype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( cctype ) ) :: fctype
  TYPE ( f_ssls_control_type ) :: fcontrol
  TYPE ( f_ssls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )
  fatype = cstr_to_fchar( catype )
  fctype = cstr_to_fchar( cctype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required SSLS structure

  CALL f_ssls_import( fcontrol, fdata, status, n, m,                           &
                      fhtype, hne, hrow, hcol, hptr,                           &
                      fatype, ane, arow, acol, aptr,                           &
                      fctype, cne, crow, ccol, cptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE ssls_import

!  ----------------------------------------
!  C interface to fortran ssls_reset_control
!  -----------------------------------------

  SUBROUTINE ssls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( ssls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_ssls_control_type ) :: fcontrol
  TYPE ( f_ssls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_SSLS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE ssls_reset_control

!  --------------------------------------------
!  C interface to fortran ssls_factorize_matrix
!  --------------------------------------------

  SUBROUTINE ssls_factorize_matrix( cdata, status, hne, hval,               &
                                    ane, aval, cne, cval ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: ane, hne, cne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( cne ) :: cval
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_ssls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_ssls_factorize_matrix( fdata, status, hval, aval, cval )
  RETURN

  END SUBROUTINE ssls_factorize_matrix

!  ----------------------------------------
!  C interface to fortran ssls_solve_system
!  ----------------------------------------

  SUBROUTINE ssls_solve_system( cdata, status, n, m, sol ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n + m ) :: sol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_ssls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_ssls_solve_system( fdata, status, sol )
  RETURN

  END SUBROUTINE ssls_solve_system

!  ---------------------------------------
!  C interface to fortran ssls_information
!  ---------------------------------------

  SUBROUTINE ssls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( ssls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_ssls_full_data_type ), pointer :: fdata
  TYPE ( f_ssls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain SSLS solution information

  CALL f_ssls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE ssls_information

!  ------------------------------------
!  C interface to fortran ssls_terminate
!  ------------------------------------

  SUBROUTINE ssls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SSLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( ssls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( ssls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_ssls_full_data_type ), pointer :: fdata
  TYPE ( f_ssls_control_type ) :: fcontrol
  TYPE ( f_ssls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_ssls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE ssls_terminate
