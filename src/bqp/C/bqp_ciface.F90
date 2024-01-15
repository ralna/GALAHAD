! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  B Q P    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. February 21st 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_BQP_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_BQP_precision, ONLY:                                           &
        f_bqp_control_type         => BQP_control_type,                        &
        f_bqp_time_type            => BQP_time_type,                           &
        f_bqp_inform_type          => BQP_inform_type,                         &
        f_bqp_full_data_type       => BQP_full_data_type,                      &
        f_bqp_initialize           => BQP_initialize,                          &
        f_bqp_read_specfile        => BQP_read_specfile,                       &
        f_bqp_import               => BQP_import,                              &
        f_bqp_import_without_h     => BQP_import_without_h,                    &
        f_bqp_reset_control        => BQP_reset_control,                       &
        f_bqp_solve_given_h        => BQP_solve_given_h,                       &
        f_bqp_solve_reverse_h_prod => BQP_solve_reverse_h_prod,                &
        f_bqp_information          => BQP_information,                         &
        f_bqp_terminate            => BQP_terminate

    USE GALAHAD_SBLS_precision_ciface, ONLY:                                   &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in   => copy_inform_in,                               &
        copy_sbls_inform_out  => copy_inform_out,                              &
        copy_sbls_control_in  => copy_control_in,                              &
        copy_sbls_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: bqp_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: cold_start
      INTEGER ( KIND = ipc_ ) :: ratio_cg_vs_sd
      INTEGER ( KIND = ipc_ ) :: change_max
      INTEGER ( KIND = ipc_ ) :: cg_maxit
      INTEGER ( KIND = ipc_ ) :: sif_file_device
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: stop_p
      REAL ( KIND = rpc_ ) :: stop_d
      REAL ( KIND = rpc_ ) :: stop_c
      REAL ( KIND = rpc_ ) :: identical_bounds_tol
      REAL ( KIND = rpc_ ) :: stop_cg_relative
      REAL ( KIND = rpc_ ) :: stop_cg_absolute
      REAL ( KIND = rpc_ ) :: zero_curvature
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      LOGICAL ( KIND = C_BOOL ) :: exact_arcsearch
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sbls_control_type ) :: sbls_control
    END TYPE bqp_control_type

    TYPE, BIND( C ) :: bqp_time_type
      REAL ( KIND = spc_ ) :: total
      REAL ( KIND = spc_ ) :: analyse
      REAL ( KIND = spc_ ) :: factorize
      REAL ( KIND = spc_ ) :: solve
    END TYPE bqp_time_type

    TYPE, BIND( C ) :: bqp_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: cg_iter
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: norm_pg
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( bqp_time_type ) :: time
      TYPE ( sbls_inform_type ) :: sbls_inform
    END TYPE bqp_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( bqp_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_bqp_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%maxit = ccontrol%maxit
    fcontrol%cold_start = ccontrol%cold_start
    fcontrol%ratio_cg_vs_sd = ccontrol%ratio_cg_vs_sd
    fcontrol%change_max = ccontrol%change_max
    fcontrol%cg_maxit = ccontrol%cg_maxit
    fcontrol%sif_file_device = ccontrol%sif_file_device

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%stop_p = ccontrol%stop_p
    fcontrol%stop_d = ccontrol%stop_d
    fcontrol%stop_c = ccontrol%stop_c
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%stop_cg_relative = ccontrol%stop_cg_relative
    fcontrol%stop_cg_absolute = ccontrol%stop_cg_absolute
    fcontrol%zero_curvature = ccontrol%zero_curvature
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit

    ! Logicals
    fcontrol%exact_arcsearch = ccontrol%exact_arcsearch
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file

    ! Derived types
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )

    ! Strings
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_bqp_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( bqp_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%maxit = fcontrol%maxit
    ccontrol%cold_start = fcontrol%cold_start
    ccontrol%ratio_cg_vs_sd = fcontrol%ratio_cg_vs_sd
    ccontrol%change_max = fcontrol%change_max
    ccontrol%cg_maxit = fcontrol%cg_maxit
    ccontrol%sif_file_device = fcontrol%sif_file_device

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%stop_p = fcontrol%stop_p
    ccontrol%stop_d = fcontrol%stop_d
    ccontrol%stop_c = fcontrol%stop_c
    ccontrol%identical_bounds_tol = fcontrol%identical_bounds_tol
    ccontrol%stop_cg_relative = fcontrol%stop_cg_relative
    ccontrol%stop_cg_absolute = fcontrol%stop_cg_absolute
    ccontrol%zero_curvature = fcontrol%zero_curvature
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit

    ! Logicals
    ccontrol%exact_arcsearch = fcontrol%exact_arcsearch
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file

    ! Derived types
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )

    ! Strings
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( bqp_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_bqp_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_bqp_time_type ), INTENT( IN ) :: ftime
    TYPE ( bqp_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( bqp_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_bqp_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%factorization_status = cinform%factorization_status
    finform%iter = cinform%iter
    finform%cg_iter = cinform%cg_iter

    ! Reals
    finform%obj = cinform%obj
    finform%norm_pg = cinform%norm_pg

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sbls_inform_in( cinform%sbls_inform, finform%sbls_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_bqp_inform_type ), INTENT( IN ) :: finform
    TYPE ( bqp_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%factorization_status = finform%factorization_status
    cinform%iter = finform%iter
    cinform%cg_iter = finform%cg_iter

    ! Reals
    cinform%obj = finform%obj
    cinform%norm_pg = finform%norm_pg

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sbls_inform_out( finform%sbls_inform, cinform%sbls_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_BQP_precision_ciface

!  -------------------------------------
!  C interface to fortran bqp_initialize
!  -------------------------------------

  SUBROUTINE bqp_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( bqp_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_bqp_full_data_type ), POINTER :: fdata
  TYPE ( f_bqp_control_type ) :: fcontrol
  TYPE ( f_bqp_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_bqp_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE bqp_initialize

!  ----------------------------------------
!  C interface to fortran bqp_read_specfile
!  ----------------------------------------

  SUBROUTINE bqp_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( bqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_bqp_control_type ) :: fcontrol
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

  CALL f_bqp_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE bqp_read_specfile

!  ---------------------------------
!  C interface to fortran bqp_inport
!  ---------------------------------

  SUBROUTINE bqp_import( ccontrol, cdata, status, n,                           &
                         chtype, hne, hrow, hcol, hptr ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( bqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  TYPE ( f_bqp_control_type ) :: fcontrol
  TYPE ( f_bqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required BQP structure

  CALL f_bqp_import( fcontrol, fdata, status, n, fhtype, hne, hrow, hcol, hptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE bqp_import

!  -------------------------------------------
!  C interface to fortran bqp_inport_without_h
!  -------------------------------------------

  SUBROUTINE bqp_import_without_h( ccontrol, cdata, status, n ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( bqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n

!  local variables

  TYPE ( f_bqp_control_type ) :: fcontrol
  TYPE ( f_bqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required BQP structure

  CALL f_bqp_import_without_h( fcontrol, fdata, status, n )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE bqp_import_without_h

!  ---------------------------------------
!  C interface to fortran bqp_reset_control
!  ----------------------------------------

  SUBROUTINE bqp_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( bqp_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_bqp_control_type ) :: fcontrol
  TYPE ( f_bqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_BQP_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE bqp_reset_control

!  ----------------------------------------
!  C interface to fortran bqp_solve_given_h
!  ----------------------------------------

  SUBROUTINE bqp_solve_given_h( cdata, status, n, hne, hval, g, f, xl, xu,     &
                                x, z, xstat ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, hne
  REAL ( KIND = rpc_ ), DIMENSION( hne ), INTENT( IN ) :: hval
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: xl, xu
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( INOUT ) :: x, z
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat

!  local variables

  TYPE ( f_bqp_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the bound-constrained quadratic program

  CALL f_bqp_solve_given_h( fdata, status, hval, g, f, xl, xu, x, z, xstat )

  RETURN

  END SUBROUTINE bqp_solve_given_h

!  -----------------------------------------------
!  C interface to fortran bqp_solve_reverse_h_prod
!  -----------------------------------------------

  SUBROUTINE bqp_solve_reverse_h_prod( cdata, status, n, g, f,                 &
                                       xl, xu, x, z, xstat, v, prod,           &
                                       nz_v, nz_v_start, nz_v_end,             &
                                       nz_prod, nz_prod_end ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: xl, xu
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( INOUT ) :: x, z
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: nz_prod_end
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: nz_v_start, nz_v_end
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n ) :: nz_prod
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: nz_v
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: prod
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: v

!  local variables

  TYPE ( f_bqp_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  solve the bound-constrained least-squares problem by reverse communication

  IF ( f_indexing ) THEN
    CALL f_bqp_solve_reverse_h_prod( fdata, status, g, f,                      &
                                      xl, xu, x, z, xstat, v, prod,            &
                                      nz_v, nz_v_start, nz_v_end,              &
                                      nz_prod, nz_prod_end )
  ELSE
    CALL f_bqp_solve_reverse_h_prod( fdata, status, g, f,                      &
                                      xl, xu, x, z, xstat, v, prod,            &
                                      nz_v, nz_v_start, nz_v_end,              &
                                      nz_prod + 1, nz_prod_end )
    IF ( status == 3 .OR. status == 4 )                                        &
      nz_v( nz_v_start : nz_v_end ) = nz_v( nz_v_start : nz_v_end ) - 1
  END IF

  RETURN

  END SUBROUTINE bqp_solve_reverse_h_prod

!  --------------------------------------
!  C interface to fortran bqp_information
!  --------------------------------------

  SUBROUTINE bqp_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( bqp_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_bqp_full_data_type ), pointer :: fdata
  TYPE ( f_bqp_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain BQP solution information

  CALL f_bqp_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE bqp_information

!  ------------------------------------
!  C interface to fortran bqp_terminate
!  ------------------------------------

  SUBROUTINE bqp_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_BQP_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( bqp_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( bqp_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_bqp_full_data_type ), pointer :: fdata
  TYPE ( f_bqp_control_type ) :: fcontrol
  TYPE ( f_bqp_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_bqp_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE bqp_terminate
