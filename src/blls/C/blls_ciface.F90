! THIS VERSION: GALAHAD 4.3 - 2024-02-02 AT 07:50 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  B L L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. February 21st 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_BLLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_BLLS_precision, ONLY:                                          &
        f_blls_control_type         => BLLS_control_type,                      &
        f_blls_time_type            => BLLS_time_type,                         &
        f_blls_inform_type          => BLLS_inform_type,                       &
        f_blls_full_data_type       => BLLS_full_data_type,                    &
        f_blls_initialize           => BLLS_initialize,                        &
        f_blls_read_specfile        => BLLS_read_specfile,                     &
        f_blls_import               => BLLS_import,                            &
        f_blls_import_without_a     => BLLS_import_without_a,                  &
        f_blls_reset_control        => BLLS_reset_control,                     &
        f_blls_solve_given_a        => BLLS_solve_given_a,                     &
        f_blls_solve_reverse_a_prod => BLLS_solve_reverse_a_prod,              &
        f_blls_information          => BLLS_information,                       &
        f_blls_terminate            => BLLS_terminate

    USE GALAHAD_USERDATA_precision, ONLY:                                      &
        f_galahad_userdata_type => GALAHAD_userdata_type

    USE GALAHAD_SBLS_precision_ciface, ONLY:                                   &
        sbls_inform_type,                                                      &
        sbls_control_type,                                                     &
        copy_sbls_inform_in   => copy_inform_in,                               &
        copy_sbls_inform_out  => copy_inform_out,                              &
        copy_sbls_control_in  => copy_control_in,                              &
        copy_sbls_control_out => copy_control_out

    USE GALAHAD_CONVRT_precision_ciface, ONLY:                                 &
        convert_inform_type,                                                   &
        convert_control_type,                                                  &
        copy_convert_inform_in   => copy_inform_in,                            &
        copy_convert_inform_out  => copy_inform_out,                           &
        copy_convert_control_in  => copy_control_in,                           &
        copy_convert_control_out => copy_control_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: blls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: start_print
      INTEGER ( KIND = ipc_ ) :: stop_print
      INTEGER ( KIND = ipc_ ) :: print_gap
      INTEGER ( KIND = ipc_ ) :: maxit
      INTEGER ( KIND = ipc_ ) :: cold_start
      INTEGER ( KIND = ipc_ ) :: preconditioner
      INTEGER ( KIND = ipc_ ) :: ratio_cg_vs_sd
      INTEGER ( KIND = ipc_ ) :: change_max
      INTEGER ( KIND = ipc_ ) :: cg_maxit
      INTEGER ( KIND = ipc_ ) :: arcsearch_max_steps
      INTEGER ( KIND = ipc_ ) :: sif_file_device
      REAL ( KIND = rpc_ ) :: weight
      REAL ( KIND = rpc_ ) :: infinity
      REAL ( KIND = rpc_ ) :: stop_d
      REAL ( KIND = rpc_ ) :: identical_bounds_tol
      REAL ( KIND = rpc_ ) :: stop_cg_relative
      REAL ( KIND = rpc_ ) :: stop_cg_absolute
      REAL ( KIND = rpc_ ) :: alpha_max
      REAL ( KIND = rpc_ ) :: alpha_initial
      REAL ( KIND = rpc_ ) :: alpha_reduction
      REAL ( KIND = rpc_ ) :: arcsearch_acceptance_tol
      REAL ( KIND = rpc_ ) :: stabilisation_weight
      REAL ( KIND = rpc_ ) :: cpu_time_limit
      LOGICAL ( KIND = C_BOOL ) :: direct_subproblem_solve
      LOGICAL ( KIND = C_BOOL ) :: exact_arc_search
      LOGICAL ( KIND = C_BOOL ) :: advance
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sbls_control_type ) :: sbls_control
      TYPE ( convert_control_type ) :: convert_control
    END TYPE blls_control_type

    TYPE, BIND( C ) :: blls_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: analyse
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: solve
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_analyse
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_solve
    END TYPE blls_time_type

    TYPE, BIND( C ) :: blls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      INTEGER ( KIND = ipc_ ) :: factorization_status
      INTEGER ( KIND = ipc_ ) :: iter
      INTEGER ( KIND = ipc_ ) :: cg_iter
      REAL ( KIND = rpc_ ) :: obj
      REAL ( KIND = rpc_ ) :: norm_pg
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      TYPE ( blls_time_type ) :: time
      TYPE ( sbls_inform_type ) :: sbls_inform
      TYPE ( convert_inform_type ) :: convert_inform
    END TYPE blls_inform_type

!----------------------
!   I n t e r f a c e s
!----------------------

    ABSTRACT INTERFACE
      FUNCTION eval_prec( n, v, p, userdata ) RESULT( status ) BIND( C )
        USE GALAHAD_KINDS_precision
        INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: v
        REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: p
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
    TYPE ( blls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_blls_control_type ), INTENT( OUT ) :: fcontrol
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
    fcontrol%preconditioner = ccontrol%preconditioner
    fcontrol%ratio_cg_vs_sd = ccontrol%ratio_cg_vs_sd
    fcontrol%change_max = ccontrol%change_max
    fcontrol%cg_maxit = ccontrol%cg_maxit
    fcontrol%arcsearch_max_steps = ccontrol%arcsearch_max_steps
    fcontrol%sif_file_device = ccontrol%sif_file_device

    ! Reals
    fcontrol%weight = ccontrol%weight
    fcontrol%infinity = ccontrol%infinity
    fcontrol%stop_d = ccontrol%stop_d
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%stop_cg_relative = ccontrol%stop_cg_relative
    fcontrol%stop_cg_absolute = ccontrol%stop_cg_absolute
    fcontrol%alpha_max = ccontrol%alpha_max
    fcontrol%alpha_initial = ccontrol%alpha_initial
    fcontrol%alpha_reduction = ccontrol%alpha_reduction
    fcontrol%arcsearch_acceptance_tol = ccontrol%arcsearch_acceptance_tol
    fcontrol%stabilisation_weight = ccontrol%stabilisation_weight
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit

    ! Logicals
    fcontrol%direct_subproblem_solve = ccontrol%direct_subproblem_solve
    fcontrol%exact_arc_search = ccontrol%exact_arc_search
    fcontrol%advance = ccontrol%advance
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file

    ! Derived types
    CALL copy_sbls_control_in( ccontrol%sbls_control, fcontrol%sbls_control )
    CALL copy_convert_control_in( ccontrol%convert_control,                    &
                                  fcontrol%convert_control )

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
    TYPE ( f_blls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( blls_control_type ), INTENT( OUT ) :: ccontrol
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
    ccontrol%preconditioner = fcontrol%preconditioner
    ccontrol%ratio_cg_vs_sd = fcontrol%ratio_cg_vs_sd
    ccontrol%change_max = fcontrol%change_max
    ccontrol%cg_maxit = fcontrol%cg_maxit
    ccontrol%arcsearch_max_steps = fcontrol%arcsearch_max_steps
    ccontrol%sif_file_device = fcontrol%sif_file_device

    ! Reals
    ccontrol%weight = fcontrol%weight
    ccontrol%infinity = fcontrol%infinity
    ccontrol%stop_d = fcontrol%stop_d
    ccontrol%identical_bounds_tol = fcontrol%identical_bounds_tol
    ccontrol%stop_cg_relative = fcontrol%stop_cg_relative
    ccontrol%stop_cg_absolute = fcontrol%stop_cg_absolute
    ccontrol%alpha_max = fcontrol%alpha_max
    ccontrol%alpha_initial = fcontrol%alpha_initial
    ccontrol%alpha_reduction = fcontrol%alpha_reduction
    ccontrol%arcsearch_acceptance_tol = fcontrol%arcsearch_acceptance_tol
    ccontrol%stabilisation_weight = fcontrol%stabilisation_weight
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit

    ! Logicals
    ccontrol%direct_subproblem_solve = fcontrol%direct_subproblem_solve
    ccontrol%exact_arc_search = fcontrol%exact_arc_search
    ccontrol%advance = fcontrol%advance
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file

    ! Derived types
    CALL copy_sbls_control_out( fcontrol%sbls_control, ccontrol%sbls_control )
    CALL copy_convert_control_out( fcontrol%convert_control,                   &
                                   ccontrol%convert_control )

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
    TYPE ( blls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_blls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_blls_time_type ), INTENT( IN ) :: ftime
    TYPE ( blls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( blls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_blls_inform_type ), INTENT( OUT ) :: finform
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
    CALL copy_convert_inform_in( cinform%convert_inform,                       &
                                 finform%convert_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_blls_inform_type ), INTENT( IN ) :: finform
    TYPE ( blls_inform_type ), INTENT( OUT ) :: cinform
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
    CALL copy_convert_inform_out( finform%convert_inform,                      &
                                  cinform%convert_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_BLLS_precision_ciface

!  --------------------------------------
!  C interface to fortran blls_initialize
!  --------------------------------------

  SUBROUTINE blls_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( blls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_blls_full_data_type ), POINTER :: fdata
  TYPE ( f_blls_control_type ) :: fcontrol
  TYPE ( f_blls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_blls_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE blls_initialize

!  -----------------------------------------
!  C interface to fortran blls_read_specfile
!  -----------------------------------------

  SUBROUTINE blls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( blls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_blls_control_type ) :: fcontrol
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

  CALL f_blls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE blls_read_specfile

!  ----------------------------------
!  C interface to fortran blls_inport
!  ----------------------------------

  SUBROUTINE blls_import( ccontrol, cdata, status, n, o, caotype, aone,        &
                          aorow, aocol, aoptrne, aoptr ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( blls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, o
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: aone, aoptrne
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aone ), OPTIONAL :: aorow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aone ), OPTIONAL :: aocol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( aoptrne ), OPTIONAL :: aoptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: caotype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( caotype ) ) :: faotype
  TYPE ( f_blls_control_type ) :: fcontrol
  TYPE ( f_blls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

   faotype = cstr_to_fchar( caotype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required BLLS structure

  CALL f_blls_import( fcontrol, fdata, status, n, o,                           &
                      faotype, aone, aorow, aocol, aoptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE blls_import

!  --------------------------------------------
!  C interface to fortran blls_inport_without_a
!  --------------------------------------------

  SUBROUTINE blls_import_without_a( ccontrol, cdata, status, n, o ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( blls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, o

!  local variables

  TYPE ( f_blls_control_type ) :: fcontrol
  TYPE ( f_blls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

  CALL f_blls_import_without_a( fcontrol, fdata, status, n, o )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE blls_import_without_a

!  -----------------------------------------
!  C interface to fortran blls_reset_control
!  -----------------------------------------

  SUBROUTINE blls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( blls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_blls_control_type ) :: fcontrol
  TYPE ( f_blls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_BLLS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE blls_reset_control

!  -----------------------------------------
!  C interface to fortran blls_solve_given_a
!  -----------------------------------------

  SUBROUTINE blls_solve_given_a( cdata, cuserdata, status, n, o, aone, aoval,  &
                                 b, xl, xu, x, z, r, g, xstat, w,              &
                                 ceval_prec ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( C_PTR ), INTENT( INOUT ) :: cuserdata
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, o, aone
  REAL ( KIND = rpc_ ), DIMENSION( aone ), INTENT( IN ) :: aoval
  REAL ( KIND = rpc_ ), DIMENSION( o ), INTENT( IN ) :: b
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: xl, xu
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( INOUT ) :: x, z
  REAL ( KIND = rpc_ ), DIMENSION( o ), INTENT( OUT ) :: r
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: g
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  REAL ( KIND = rpc_ ), DIMENSION( o ), INTENT( IN ), OPTIONAL :: w
  TYPE ( C_FUNPTR ), INTENT( IN ), VALUE :: ceval_prec

!  local variables

  TYPE ( f_blls_full_data_type ), POINTER :: fdata
  PROCEDURE( eval_prec ), POINTER :: feval_prec

!  ignore Fortran userdata type (not interoperable)

! TYPE ( f_galahad_userdata_type ), POINTER :: fuserdata => NULL( )
  TYPE ( f_galahad_userdata_type ) :: fuserdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  associate procedure pointers

  IF ( C_ASSOCIATED( ceval_prec ) ) THEN
    CALL C_F_PROCPOINTER( ceval_prec, feval_prec )
  ELSE
    NULLIFY( feval_prec )
  END IF

!  solve the bound-constrained least-squares problem

  IF ( PRESENT( w ) ) THEN
    CALL f_blls_solve_given_a( fdata, fuserdata, status, aoval, b, xl, xu,     &
                               x, z, r, g, xstat, W = w,                       &
                               eval_PREC = wrap_eval_prec )
   ELSE
    CALL f_blls_solve_given_a( fdata, fuserdata, status, aoval, b, xl, xu,     &
                               x, z, r, g, xstat,                              &
                               eval_PREC = wrap_eval_prec )
   END IF

  RETURN

!  wrappers

  CONTAINS

!  eval_PREC wrapper

    SUBROUTINE wrap_eval_prec( status, userdata, v, p )
    INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
    TYPE ( f_galahad_userdata_type ), INTENT( INOUT ) :: userdata
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( IN ) :: v
    REAL ( KIND = rpc_ ), DIMENSION( : ), INTENT( OUT ) :: p

!  call C interoperable eval_prec

    status = feval_prec( n, v, p, cuserdata )
    RETURN

    END SUBROUTINE wrap_eval_prec

  END SUBROUTINE blls_solve_given_a

!  ------------------------------------------------
!  C interface to fortran blls_solve_reverse_a_prod
!  ------------------------------------------------

  SUBROUTINE blls_solve_reverse_a_prod( cdata, status, eval_status, n, o, b,   &
                                        xl, xu, x, z, r, g, xstat, v, p,       &
                                        nz_v, nz_v_start, nz_v_end,            &
                                        nz_p, nz_p_end, w ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status, eval_status
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, o
  REAL ( KIND = rpc_ ), DIMENSION( o ), INTENT( IN ) :: b
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( IN ) :: xl, xu
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( INOUT ) :: x, z
  REAL ( KIND = rpc_ ), DIMENSION( o ), INTENT( OUT ) :: r
  REAL ( KIND = rpc_ ), DIMENSION( n ), INTENT( OUT ) :: g
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: nz_p_end
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: nz_v_start, nz_v_end
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( o ) :: nz_p
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( MAX( n, o ) ) :: nz_v
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( MAX( n, o ) ) :: p
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( MAX( n, o ) ) :: v
  REAL ( KIND = rpc_ ), DIMENSION( o ), INTENT( IN ), OPTIONAL :: w

!  local variables

  TYPE ( f_blls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  f_indexing = fdata%f_indexing

!  solve the bound-constrained least-squares problem by reverse communication

  IF ( f_indexing ) THEN
    CALL f_blls_solve_reverse_a_prod( fdata, status, eval_status, b, xl, xu,   &
                                      x, z, r, g, xstat, v, p,                 &
                                      nz_v, nz_v_start, nz_v_end,              &
                                      nz_p, nz_p_end, W = w )
  ELSE
    CALL f_blls_solve_reverse_a_prod( fdata, status, eval_status, b, xl, xu,   &
                                      x, z, r, g, xstat, v, p,                 &
                                      nz_v, nz_v_start, nz_v_end,              &
                                      nz_p( : nz_p_end ) + 1, nz_p_end, W = w )
    IF ( status == 4 .OR. status == 5 .OR. status == 6 ) then
      nz_v( nz_v_start : nz_v_end ) = nz_v( nz_v_start : nz_v_end ) - 1
    END IF
  END IF

  RETURN

  END SUBROUTINE blls_solve_reverse_a_prod

!  ---------------------------------------
!  C interface to fortran blls_information
!  ---------------------------------------

  SUBROUTINE blls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( blls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_blls_full_data_type ), pointer :: fdata
  TYPE ( f_blls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain BLLS solution information

  CALL f_blls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE blls_information

!  -------------------------------------
!  C interface to fortran blls_terminate
!  -------------------------------------

  SUBROUTINE blls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_BLLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( blls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( blls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_blls_full_data_type ), pointer :: fdata
  TYPE ( f_blls_control_type ) :: fcontrol
  TYPE ( f_blls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_blls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE blls_terminate
