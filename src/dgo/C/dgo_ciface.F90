! THIS VERSION: GALAHAD 5.0 - 2024-06-09 AT 10:00 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ D G O   C   I N T E R F A C E  -*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. August 3rd 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_DGO_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_DGO_precision, ONLY:                                           &
        f_dgo_time_type                 => DGO_time_type,                      &
        f_dgo_inform_type               => DGO_inform_type,                    &
        f_dgo_control_type              => DGO_control_type,                   &
        f_dgo_full_data_type            => DGO_full_data_type,                 &
        f_dgo_initialize                => DGO_initialize,                     &
        f_dgo_read_specfile             => DGO_read_specfile,                  &
        f_dgo_import                    => DGO_import,                         &
        f_dgo_reset_control             => DGO_reset_control,                  &
        f_dgo_solve_with_mat            => DGO_solve_with_mat,                 &
        f_dgo_solve_without_mat         => DGO_solve_without_mat,              &
        f_dgo_solve_reverse_with_mat    => DGO_solve_reverse_with_mat,         &
        f_dgo_solve_reverse_without_mat => DGO_solve_reverse_without_mat,      &
        f_dgo_information               => DGO_information,                    &
        f_dgo_terminate                 => DGO_terminate

    USE GALAHAD_USERDATA_precision, ONLY:                                      &
        f_galahad_userdata_type => GALAHAD_userdata_type

    USE GALAHAD_TRB_precision_ciface, ONLY:                                    &
        trb_inform_type,                                                       &
        trb_control_type,                                                      &
        copy_trb_inform_in   => copy_inform_in,                                &
        copy_trb_inform_out  => copy_inform_out,                               &
        copy_trb_control_in  => copy_control_in,                               &
        copy_trb_control_out => copy_control_out

    USE GALAHAD_UGO_precision_ciface, ONLY:                                    &
        ugo_inform_type,                                                       &
        ugo_control_type,                                                      &
        copy_ugo_inform_in   => copy_inform_in,                                &
        copy_ugo_inform_out  => copy_inform_out,                               &
        copy_ugo_control_in  => copy_control_in,                               &
        copy_ugo_control_out => copy_control_out

    USE GALAHAD_HASH_ciface, ONLY:                                             &
        hash_inform_type,                                                      &
        hash_control_type,                                                     &
        copy_hash_inform_in   => copy_inform_in,                               &
        copy_hash_inform_out  => copy_inform_out,                              &
        copy_hash_control_in  => copy_control_in,                              &
        copy_hash_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: dgo_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: max_evals
      INTEGER ( KIND = ipc_ ) :: dictionary_size
      INTEGER ( KIND = ipc_ ) :: alive_unit
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: alive_file
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: lipschitz_lower_bound
      REAL ( KIND = rpc_ ) :: lipschitz_reliability
      REAL ( KIND = rpc_ ) :: lipschitz_control
      REAL ( KIND = rpc_ ) :: stop_length
      REAL ( KIND = rpc_ ) :: stop_f
      REAL ( KIND = rpc_ ) :: obj_unbounded
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      REAL ( KIND = rpc_ ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: hessian_available
      LOGICAL ( KIND = C_BOOL ) :: prune
      LOGICAL ( KIND = C_BOOL ) :: perform_local_optimization
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( hash_control_type ) :: hash_control
      TYPE ( ugo_control_type ) :: ugo_control
      TYPE ( trb_control_type ) :: trb_control
    END TYPE dgo_control_type

    TYPE, BIND( C ) :: dgo_time_type
      REAL ( KIND = spc_ ) :: total
      REAL ( KIND = spc_ ) :: univariate_global
      REAL ( KIND = spc_ ) :: multivariate_local
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_univariate_global
      REAL ( KIND = rpc_ ) :: clock_multivariate_local
    END TYPE dgo_time_type

    TYPE, BIND( C ) :: dgo_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: f_eval
      INTEGER ( KIND = ipc_ ) :: g_eval
      INTEGER ( KIND = ipc_ ) :: h_eval
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: norm_pg
      REAL ( KIND = rpc_ ) :: length_ratio
      REAL ( KIND = rpc_ ) :: f_gap
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 2 ) :: why_stop
      TYPE ( dgo_time_type ) :: time
      TYPE ( hash_inform_type ) :: hash_inform
      TYPE ( ugo_inform_type ) :: ugo_inform
      TYPE ( trb_inform_type ) :: trb_inform
    END TYPE dgo_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_f( n, x, f, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), value :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), INTENT( OUT ) :: f
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_f
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_g( n, x, g, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: g
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_g
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_h( n, ne, x, hval, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: ne
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( ne ), INTENT( OUT ) :: hval
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_h
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_hprod( n, x, u, v, got_h, userdata ) RESULT( status )      &
                                                         BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( INOUT ) :: u
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: v
        LOGICAL ( KIND = C_BOOL ), INTENT( IN ), VALUE :: got_h
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_hprod
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_shprod( n, x, nnz_v, index_nz_v, v, nnz_u, index_nz_u,     &
                            u, got_h, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: nnz_v
        INTEGER ( KIND = ipc_ ), DIMENSION( n ), INTENT( IN ) :: index_nz_v
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: v
        INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: nnz_u
        INTEGER ( KIND = ipc_ ), DIMENSION( n ), INTENT( OUT ) :: index_nz_u
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: u
        LOGICAL( KIND = C_BOOL ), INTENT( IN ), VALUE :: got_h
        TYPE (C_PTR), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_shprod
    END INTERFACE

    ABSTRACT INTERFACE
      FUNCTION eval_prec( n, x, u, v, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: x
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: u
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: v
        TYPE ( C_PTR ), INTENT( IN ), VALUE :: userdata
        INTEGER ( KIND = ipc_ ) :: status
      END FUNCTION eval_prec
    END INTERFACE

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( dgo_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_dgo_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) )  f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%print_gap = ccontrol%print_gap
    fcontrol%maxit = ccontrol%maxit
    fcontrol%max_evals = ccontrol%max_evals
    fcontrol%dictionary_size = ccontrol%dictionary_size
    fcontrol%alive_unit = ccontrol%alive_unit

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%lipschitz_lower_bound = ccontrol%lipschitz_lower_bound
    fcontrol%lipschitz_reliability = ccontrol%lipschitz_reliability
    fcontrol%lipschitz_control = ccontrol%lipschitz_control
    fcontrol%stop_length = ccontrol%stop_length
    fcontrol%stop_f = ccontrol%stop_f
    fcontrol%obj_unbounded = ccontrol%obj_unbounded
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%hessian_available = ccontrol%hessian_available
    fcontrol%prune = ccontrol%prune
    fcontrol%perform_local_optimization = ccontrol%perform_local_optimization
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_hash_control_in( ccontrol%hash_control, fcontrol%hash_control )
    CALL copy_ugo_control_in( ccontrol%ugo_control, fcontrol%ugo_control )
    CALL copy_trb_control_in( ccontrol%trb_control, fcontrol%trb_control )

    ! Strings
    DO i = 1, LEN( fcontrol%alive_file )
      IF ( ccontrol%alive_file( i ) == C_NULL_CHAR ) EXIT
      fcontrol%alive_file( i : i ) = ccontrol%alive_file( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_dgo_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( dgo_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) )  ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%print_gap = fcontrol%print_gap
    ccontrol%maxit = fcontrol%maxit
    ccontrol%max_evals = fcontrol%max_evals
    ccontrol%dictionary_size = fcontrol%dictionary_size
    ccontrol%alive_unit = fcontrol%alive_unit

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%lipschitz_lower_bound = fcontrol%lipschitz_lower_bound
    ccontrol%lipschitz_reliability = fcontrol%lipschitz_reliability
    ccontrol%lipschitz_control = fcontrol%lipschitz_control
    ccontrol%stop_length = fcontrol%stop_length
    ccontrol%stop_f = fcontrol%stop_f
    ccontrol%obj_unbounded = fcontrol%obj_unbounded
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%hessian_available = fcontrol%hessian_available
    ccontrol%prune = fcontrol%prune
    ccontrol%perform_local_optimization = fcontrol%perform_local_optimization
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_hash_control_out( fcontrol%hash_control, ccontrol%hash_control )
    CALL copy_ugo_control_out( fcontrol%ugo_control, ccontrol%ugo_control )
    CALL copy_trb_control_out( fcontrol%trb_control, ccontrol%trb_control )

    ! Strings
    l = LEN( fcontrol%alive_file )
    DO i = 1, l
      ccontrol%alive_file( i ) = fcontrol%alive_file( i : i )
    END DO
    ccontrol%alive_file( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime )
    TYPE ( dgo_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_dgo_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%clock_total = ctime%clock_total
    ftime%clock_univariate_global = ctime%clock_univariate_global
    ftime%clock_multivariate_local = ctime%clock_multivariate_local
    ftime%total = ctime%total
    ftime%univariate_global = ctime%univariate_global
    ftime%multivariate_local = ctime%multivariate_local
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_dgo_time_type ), INTENT( IN ) :: ftime
    TYPE ( dgo_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%clock_total = ftime%clock_total
    ctime%clock_univariate_global = ftime%clock_univariate_global
    ctime%clock_multivariate_local = ftime%clock_multivariate_local
    ctime%total = ftime%total
    ctime%univariate_global = ftime%univariate_global
    ctime%multivariate_local = ftime%multivariate_local
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( dgo_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_dgo_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%f_eval = cinform%f_eval
    finform%g_eval = cinform%g_eval
    finform%h_eval = cinform%h_eval

    ! Reals
    finform%obj = cinform%obj
    finform%norm_pg = cinform%norm_pg
    finform%length_ratio = cinform%length_ratio
    finform%f_gap = cinform%f_gap

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_hash_inform_in( cinform%hash_inform, finform%hash_inform )
    CALL copy_ugo_inform_in( cinform%ugo_inform, finform%ugo_inform )
    CALL copy_trb_inform_in( cinform%trb_inform, finform%trb_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    DO i = 1, LEN( finform%why_stop )
      IF ( cinform%why_stop( i ) == C_NULL_CHAR ) EXIT
      finform%why_stop( i : i ) = cinform%why_stop( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_dgo_inform_type ), INTENT( IN ) :: finform
    TYPE ( dgo_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%f_eval = finform%f_eval
    cinform%g_eval = finform%g_eval
    cinform%h_eval = finform%h_eval

    ! Reals
    cinform%obj = finform%obj
    cinform%norm_pg = finform%norm_pg
    cinform%length_ratio = finform%length_ratio
    cinform%f_gap = finform%f_gap

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_hash_inform_out( finform%hash_inform, cinform%hash_inform )
    CALL copy_ugo_inform_out( finform%ugo_inform, cinform%ugo_inform )
    CALL copy_trb_inform_out( finform%trb_inform, cinform%trb_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    l = LEN( finform%why_stop )
    DO i = 1, l
      cinform%why_stop( i ) = finform%why_stop( i : i )
    END DO
    cinform%why_stop( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_DGO_precision_ciface

!  -------------------------------------
!  C interface to fortran dgo_initialize
!  -------------------------------------

  SUBROUTINE dgo_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( dgo_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_dgo_full_data_type ), POINTER :: fdata
  TYPE ( f_dgo_control_type ) :: fcontrol
  TYPE ( f_dgo_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_dgo_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dgo_initialize

!  ----------------------------------------
!  C interface to fortran dgo_read_specfile
!  ----------------------------------------

  SUBROUTINE dgo_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( dgo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_dgo_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = ipc_ ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  open specfile for reading

  open( UNIT = device, FILE = fspecfile )

!  read control parameters from the specfile

  CALL f_dgo_read_specfile( fcontrol, device )

!  close specfile

  close( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dgo_read_specfile

!  ---------------------------------
!  C interface to fortran dgo_inport
!  ---------------------------------

  SUBROUTINE dgo_import( ccontrol, cdata, status, n, xl, xu, ctype,            &
                         ne, row, col, ptr ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ), optional :: row, col
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), optional :: ptr
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: ctype
  TYPE ( dgo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( ctype ) ) :: ftype
  TYPE ( f_dgo_control_type ) :: fcontrol
  TYPE ( f_dgo_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  ftype = cstr_to_fchar( ctype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required DGO structure

  CALL f_dgo_import( fcontrol, fdata, status, n, xl, xu, ftype, ne,            &
                     row, col, ptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE dgo_import

!  ----------------------------------------
!  C interface to fortran dgo_reset_control
!  ----------------------------------------

  SUBROUTINE dgo_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( dgo_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dgo_control_type ) :: fcontrol
  TYPE ( f_dgo_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_dgo_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE dgo_reset_control

!  -----------------------------------------
!  C interface to fortran dgo_solve_with_mat
!  -----------------------------------------

  SUBROUTINE dgo_solve_with_mat( cdata, cuserdata, status, n, x, g, ne,        &
                                 ceval_f, ceval_g, ceval_h, ceval_hprod,       &
                                 ceval_prec ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_f, ceval_g
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_h, ceval_hprod, ceval_prec

!  local variables

  TYPE ( f_dgo_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_f ), POINTER :: feval_f
  PROCEDURE( eval_g ), POINTER :: feval_g
  PROCEDURE( eval_h ), POINTER :: feval_h
  PROCEDURE( eval_hprod ), POINTER :: feval_hprod
  PROCEDURE( eval_prec ), POINTER :: feval_prec

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_galahad_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_f, feval_f )
  CALL C_F_PROCPOINTER( ceval_g, feval_g )
  CALL C_F_PROCPOINTER( ceval_h, feval_h )
  IF ( C_ASSOCIATED( ceval_hprod ) ) THEN
    CALL C_F_PROCPOINTER( ceval_hprod, feval_hprod )
  ELSE
    NULLIFY( feval_hprod )
  END IF
  IF ( C_ASSOCIATED( ceval_prec ) ) THEN
    CALL C_F_PROCPOINTER( ceval_prec, feval_prec )
  ELSE
    NULLIFY( feval_prec )
  END IF

!  solve the problem when the Hessian is explicitly available

  IF ( ASSOCIATED( feval_prec ) .AND. ASSOCIATED( feval_hprod ) ) THEN
    CALL f_dgo_solve_with_mat( fdata, fuserdata, status, x, g,                 &
                               wrap_eval_f, wrap_eval_g, wrap_eval_h,          &
                               eval_HPROD = wrap_eval_hprod,                   &
                               eval_PREC = wrap_eval_prec )
  ELSE IF ( ASSOCIATED( feval_prec ) ) THEN
    CALL f_dgo_solve_with_mat( fdata, fuserdata, status, x, g,                 &
                               wrap_eval_f, wrap_eval_g, wrap_eval_h,          &
                               eval_PREC = wrap_eval_prec )
  ELSE IF ( ASSOCIATED(feval_hprod ) ) THEN
    CALL f_dgo_solve_with_mat( fdata, fuserdata, status, x, g,                 &
                               wrap_eval_f, wrap_eval_g, wrap_eval_h,          &
                               eval_HPROD = wrap_eval_hprod )
  ELSE
    CALL f_dgo_solve_with_mat( fdata, fuserdata, status, x, g,                 &
                               wrap_eval_f, wrap_eval_g, wrap_eval_h )
  END IF

  RETURN

!  wrappers

  CONTAINS

!  eval_F wrapper

    SUBROUTINE wrap_eval_f( status, x, userdata, f )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), INTENT( OUT ) :: f

!  call C interoperable eval_f

    status = feval_f( n, x, f, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_f

!  eval_G wrapper

    SUBROUTINE wrap_eval_g( status, x, userdata, g )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: g

!  Call C interoperable eval_g
    status = feval_g( n, x, g, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_g

!  eval_H wrapper

    SUBROUTINE wrap_eval_h( status, x, userdata, hval )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: hval

!  Call C interoperable eval_h
    status = feval_h( n, ne, x, hval, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_h

! eval_HPROD wrapper

    SUBROUTINE wrap_eval_hprod( status, x, userdata, u, v, fgot_h )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( INOUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

! Call C interoperable eval_hprod
    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .FALSE.
    END IF
    status = feval_hprod( n, x, u, v, cgot_h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hprod

!  eval_PREC wrapper

    SUBROUTINE wrap_eval_prec( status, x, userdata, u, v )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v

!  call C interoperable eval_prec

    status = feval_prec( n, x, u, v, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_prec

  END SUBROUTINE dgo_solve_with_mat

!  --------------------------------------------
!  C interface to fortran dgo_solve_without_mat
!  --------------------------------------------

  SUBROUTINE dgo_solve_without_mat( cdata, cuserdata, status, n, x, g,         &
                                    ceval_f, ceval_g, ceval_hprod,             &
                                    ceval_shprod, ceval_prec ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cuserdata
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_f, ceval_g
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_hprod, ceval_shprod
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_prec

!  local variables

  TYPE ( f_dgo_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_f ), POINTER :: feval_f
  PROCEDURE( eval_g ), POINTER :: feval_g
  PROCEDURE( eval_hprod ), POINTER :: feval_hprod
  PROCEDURE( eval_shprod ), POINTER :: feval_shprod
  PROCEDURE( eval_prec ), POINTER :: feval_prec
  LOGICAL :: f_indexing

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_galahad_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  associate procedure pointers

  CALL C_F_PROCPOINTER( ceval_f, feval_f )
  CALL C_F_PROCPOINTER( ceval_g, feval_g )
  CALL C_F_PROCPOINTER( ceval_hprod, feval_hprod )
  CALL C_F_PROCPOINTER( ceval_shprod, feval_shprod )
  IF ( C_ASSOCIATED( ceval_prec ) ) THEN
    CALL C_F_PROCPOINTER( ceval_prec, feval_prec )
  ELSE
    NULLIFY( feval_prec )
  END IF

!  solve the problem when the Hessian is only available via products

  IF ( ASSOCIATED( feval_prec ) ) THEN
    CALL f_dgo_solve_without_mat( fdata, fuserdata, status, x, g,              &
                                  wrap_eval_f, wrap_eval_g, wrap_eval_hprod,   &
                                  wrap_eval_shprod, eval_PREC = wrap_eval_prec )
  ELSE
    CALL f_dgo_solve_without_mat( fdata, fuserdata, status, x, g,              &
                                  wrap_eval_f, wrap_eval_g, wrap_eval_hprod,   &
                                  wrap_eval_shprod )
  END IF

  RETURN

!  wrappers

  CONTAINS

!  eval_F wrapper

    SUBROUTINE wrap_eval_f( status, x, userdata, f )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), INTENT( OUT ) :: f

!  call C interoperable eval_f

    status = feval_f( n, x, f, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_f

!  eval_G wrapper

    SUBROUTINE wrap_eval_g( status, x, userdata, g )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: g

!  call C interoperable eval_g

    status = feval_g( n, x, g, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_g

!  eval_HPROD wrapper

    SUBROUTINE wrap_eval_hprod( status, x, userdata, u, v, fgot_h )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( INOUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

!  call C interoperable eval_hprod

    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .false.
    END IF
    status = feval_hprod( n, x, u, v, cgot_h, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_hprod

!  eval_SHPROD wrapper

    SUBROUTINE wrap_eval_shprod( status, x, userdata, nnz_v, index_nz_v, v,    &
                                 nnz_u, index_nz_u, u, fgot_h )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    INTEGER ( KIND = ipc_ ), INTENT( IN ) :: nnz_v
    INTEGER ( KIND = ipc_ ), DIMENSION(:), INTENT( IN ) :: index_nz_v
    REAL ( KIND = rpc_ ), dimension( : ), INTENT( IN ) :: v
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: nnz_u
    INTEGER ( KIND = ipc_ ), DIMENSION( : ), INTENT( OUT ) :: index_nz_u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: u
    LOGICAL, OPTIONAL, INTENT( IN ) :: fgot_h
    LOGICAL ( KIND = C_BOOL ) :: cgot_h

!  call C interoperable eval_shprod

    IF ( PRESENT( fgot_h ) ) THEN
      cgot_h = fgot_h
    ELSE
      cgot_h = .false.
    END IF

    IF ( f_indexing ) then
      status = feval_shprod( n, x, nnz_v, index_nz_v, v, nnz_u, index_nz_u,    &
                             u, cgot_h, cuserdata )
    ELSE ! handle C sparse matrix indexing
      status = feval_shprod(n, x, nnz_v, index_nz_v - 1, v, nnz_u, index_nz_u, &
                             u, cgot_h, cuserdata )
      index_nz_u = index_nz_u + 1
    END IF
    RETURN

    END SUBROUTINE wrap_eval_shprod

!  eval_PREC wrapper

    SUBROUTINE wrap_eval_prec( status, x, userdata, u, v )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: x
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: u
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v

!  call C interoperable eval_prec

    status = feval_prec( n, x, u, v, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_prec

  END SUBROUTINE dgo_solve_without_mat

!  -------------------------------------------------
!  C interface to fortran dgo_solve_reverse_with_mat
!  -------------------------------------------------

  SUBROUTINE dgo_solve_reverse_with_mat( cdata, status, eval_status,           &
                                         n, x, f, g, ne, val, u, v ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g, u, v
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( ne ) :: val
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dgo_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the problem when the Hessian is available by reverse communication

  CALL f_dgo_solve_reverse_with_mat( fdata, status, eval_status, x, f, g, val, &
                                      u, v )
  RETURN

  END SUBROUTINE dgo_solve_reverse_with_mat

!  ----------------------------------------------------
!  C interface to fortran dgo_solve_reverse_without_mat
!  ----------------------------------------------------

  SUBROUTINE dgo_solve_reverse_without_mat( cdata, status, eval_status, n,     &
                                            x, f, g, u, v, index_nz_v, nnz_v,  &
                                            index_nz_u, nnz_u ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, nnz_u
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: nnz_v
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n ) :: x, g, u, v
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n ) :: index_nz_u
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: index_nz_v
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_dgo_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  solve the problem when Hessian products are available by reverse
!  communication

  IF ( f_indexing ) THEN
    CALL f_dgo_solve_reverse_without_mat( fdata, status, eval_status, x, f, g, &
                                          u, v, index_nz_v, nnz_v,             &
                                          index_nz_u, nnz_u )
  ELSE
    CALL f_dgo_solve_reverse_without_mat( fdata, status, eval_status, x, f, g, &
                                          u, v, index_nz_v, nnz_v,             &
                                          index_nz_u + 1, nnz_u )

!  convert to C indexing if required

     IF ( status == 7 ) index_nz_v( : nnz_v ) = index_nz_v( : nnz_v ) - 1
  END IF

  RETURN

  END SUBROUTINE dgo_solve_reverse_without_mat

!  --------------------------------------
!  C interface to fortran dgo_information
!  --------------------------------------

  SUBROUTINE dgo_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( dgo_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_dgo_full_data_type ), pointer :: fdata
  TYPE ( f_dgo_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain DGO solution information

  CALL f_dgo_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE dgo_information

!  ------------------------------------
!  C interface to fortran dgo_terminate
!  ------------------------------------

  SUBROUTINE dgo_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_DGO_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( dgo_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( dgo_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_dgo_full_data_type ), pointer :: fdata
  TYPE ( f_dgo_control_type ) :: fcontrol
  TYPE ( f_dgo_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_dgo_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE dgo_terminate

