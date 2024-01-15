! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  C R O    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 28th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_CRO_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_CRO_precision, ONLY:                                           &
        f_cro_control_type       => CRO_control_type,                          &
        f_cro_time_type          => CRO_time_type,                             &
        f_cro_inform_type        => CRO_inform_type,                           &
        f_cro_full_data_type     => CRO_full_data_type,                        &
        f_cro_initialize         => CRO_initialize,                            &
        f_cro_read_specfile      => CRO_read_specfile,                         &
        f_cro_crossover_solution => CRO_crossover_solution,                    &
        f_cro_terminate          => CRO_terminate

    USE GALAHAD_SLS_precision_ciface, ONLY:                                    &
        sls_inform_type,                                                       &
        sls_control_type,                                                      &
        copy_sls_inform_in   => copy_inform_in,                                &
        copy_sls_inform_out  => copy_inform_out,                               &
        copy_sls_control_in  => copy_control_in,                               &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_SBLS_precision_ciface, ONLY:                                   &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in   => copy_inform_in,                               &
        copy_sbls_inform_out  => copy_inform_out,                              &
        copy_sbls_control_in  => copy_control_in,                              &
        copy_sbls_control_out => copy_control_out

    USE GALAHAD_ULS_precision_ciface, ONLY:                                    &
        uls_inform_type,                                                       &
        uls_control_type,                                                      &
        copy_uls_inform_in   => copy_inform_in,                                &
        copy_uls_inform_out  => copy_inform_out,                               &
        copy_uls_control_in  => copy_control_in,                               &
        copy_uls_control_out => copy_control_out

    USE GALAHAD_IR_precision_ciface, ONLY:                                     &
        ir_inform_type,                                                        &
        ir_control_type,                                                       &
        copy_ir_inform_in   => copy_inform_in,                                 &
        copy_ir_inform_out  => copy_inform_out,                                &
        copy_ir_control_in  => copy_control_in,                                &
        copy_ir_control_out => copy_control_out

    USE GALAHAD_SCU_precision_ciface, ONLY:                                    &
        scu_inform_type,                                                       &
        scu_control_type,                                                      &
!       copy_scu_control_in  => copy_control_in,                               &
!       copy_scu_control_out => copy_control_out,                              &
        copy_scu_inform_in   => copy_inform_in,                                &
        copy_scu_inform_out  => copy_inform_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: cro_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: max_schur_complement
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: feasibility_tolerance
      LOGICAL ( KIND = C_BOOL ) :: check_io
      LOGICAL ( KIND = C_BOOL ) :: refine_solution
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: unsymmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( sbls_control_type ) :: sbls_control
      TYPE ( uls_control_type ) :: uls_control
      TYPE ( ir_control_type ) :: ir_control
    END TYPE cro_control_type

    TYPE, BIND( C ) :: cro_time_type
      REAL ( KIND = spc_ ) :: total
      REAL ( KIND = spc_ ) :: analyse
      REAL ( KIND = spc_ ) :: factorize
      REAL ( KIND = spc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE cro_time_type

    TYPE, BIND( C ) :: cro_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: dependent
      TYPE ( cro_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( sbls_inform_type ) :: sbls_inform
      TYPE ( uls_inform_type ) :: uls_inform
      INTEGER ( KIND = ipc_ ) :: scu_status
      TYPE ( scu_inform_type ) :: scu_inform
      TYPE ( ir_inform_type ) :: ir_inform
    END TYPE cro_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( cro_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_cro_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%max_schur_complement = ccontrol%max_schur_complement

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%feasibility_tolerance = ccontrol%feasibility_tolerance

    ! Logicals
    fcontrol%check_io = ccontrol%check_io
    fcontrol%refine_solution = ccontrol%refine_solution
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
    CALL copy_uls_control_in( ccontrol%uls_control, fcontrol%uls_control )
    CALL copy_ir_control_in( ccontrol%ir_control, fcontrol%ir_control )

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
    TYPE ( f_cro_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( cro_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%max_schur_complement = fcontrol%max_schur_complement

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%feasibility_tolerance = fcontrol%feasibility_tolerance

    ! Logicals
    ccontrol%check_io = fcontrol%check_io
    ccontrol%refine_solution = fcontrol%refine_solution
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
    CALL copy_uls_control_out( fcontrol%uls_control, ccontrol%uls_control )
    CALL copy_ir_control_out( fcontrol%ir_control, ccontrol%ir_control )

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
    TYPE ( cro_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_cro_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%clock_total = ctime%clock_total
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_cro_time_type ), INTENT( IN ) :: ftime
    TYPE ( cro_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%clock_total = ftime%clock_total
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( cro_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_cro_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%dependent = cinform%dependent
    finform%scu_status = cinform%scu_status

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )
    CALL copy_uls_inform_in( cinform%uls_inform, finform%uls_inform )
    CALL copy_scu_inform_in( cinform%scu_inform, finform%scu_inform )
    CALL copy_ir_inform_in( cinform%ir_inform, finform%ir_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_cro_inform_type ), INTENT( IN ) :: finform
    TYPE ( cro_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%dependent = finform%dependent
    cinform%scu_status = finform%scu_status

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )
    CALL copy_uls_inform_out( finform%uls_inform, cinform%uls_inform )
    CALL copy_scu_inform_out( finform%scu_inform, cinform%scu_inform )
    CALL copy_ir_inform_out( finform%ir_inform, cinform%ir_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_CRO_precision_ciface

!  -------------------------------------
!  C interface to fortran cro_initialize
!  -------------------------------------

  SUBROUTINE cro_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_CRO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( cro_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_cro_full_data_type ), POINTER :: fdata
  TYPE ( f_cro_control_type ) :: fcontrol
  TYPE ( f_cro_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_cro_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE cro_initialize

!  ----------------------------------------
!  C interface to fortran cro_read_specfile
!  ----------------------------------------

  SUBROUTINE cro_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_CRO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( cro_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_cro_control_type ) :: fcontrol
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

  CALL f_cro_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE cro_read_specfile

  SUBROUTINE cro_crossover_solution( cdata, ccontrol, cinform,                 &
                                     n, m, mequal, hne, hval, hcol, hptr,      &
                                     ane, aval, acol, aptr, g, cl, cu, xl,     &
                                     xu, x, c, y, z, xstat, cstat ) BIND( C )
  USE GALAHAD_CRO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, mequal, hne, ane
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ), DIMENSION( n + 1 ) :: hptr
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ), DIMENSION( hne ) :: hcol
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ), DIMENSION( m + 1 ) :: aptr
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ), DIMENSION( ane ) :: acol
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( m ) :: c, y
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( cro_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( cro_inform_type ), INTENT( OUT ) :: cinform

!  local variables

  TYPE ( f_cro_control_type ) :: fcontrol
  TYPE ( f_cro_inform_type ) :: finform
  TYPE ( f_cro_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

! CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required CRO structure

  CALL f_cro_crossover_solution( n, m, mequal, hval, hcol, hptr, aval,         &
                                 acol, aptr, g, cl, cu, xl, xu, x, c, y, z,    &
                                 xstat, cstat, fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE cro_crossover_solution

!  ------------------------------------
!  C interface to fortran cro_terminate
!  ------------------------------------

  SUBROUTINE cro_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_CRO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( cro_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( cro_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_cro_full_data_type ), pointer :: fdata
  TYPE ( f_cro_control_type ) :: fcontrol
  TYPE ( f_cro_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_cro_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE cro_terminate
