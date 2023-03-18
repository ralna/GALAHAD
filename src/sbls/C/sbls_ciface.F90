! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  S B L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. November 24th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_SBLS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_SBLS_precision, ONLY:                                          &
        f_sbls_control_type     => SBLS_control_type,                          &
        f_sbls_time_type        => SBLS_time_type,                             &
        f_sbls_inform_type      => SBLS_inform_type,                           &
        f_sbls_full_data_type   => SBLS_full_data_type,                        &
        f_sbls_initialize       => SBLS_initialize,                            &
        f_sbls_read_specfile    => SBLS_read_specfile,                         &
        f_sbls_import           => SBLS_import,                                &
        f_sbls_reset_control    => SBLS_reset_control,                         &
        f_sbls_factorize_matrix => SBLS_factorize_matrix,                      &
        f_sbls_solve_system     => SBLS_solve_system,                          &
        f_sbls_information      => SBLS_information,                           &
        f_sbls_terminate        => SBLS_terminate

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

    TYPE, BIND( C ) :: sbls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: indmin
      INTEGER ( KIND = ipc_ ) :: valmin
      INTEGER ( KIND = ipc_ ) :: len_ulsmin
      INTEGER ( KIND = ipc_ ) :: itref_max
      INTEGER ( KIND = ipc_ ) :: maxit_pcg
      INTEGER ( KIND = ipc_ ) :: new_a
      INTEGER ( KIND = ipc_ ) :: new_h
      INTEGER ( KIND = ipc_ ) :: new_c
      INTEGER ( KIND = ipc_ ) :: preconditioner
      INTEGER ( KIND = ipc_ ) :: semi_bandwidth
      INTEGER ( KIND = ipc_ ) :: factorization
      INTEGER ( KIND = ipc_ ) :: max_col
      INTEGER ( KIND = ipc_ ) :: scaling
      INTEGER ( KIND = ipc_ ) :: ordering
      REAL ( KIND = rpc_ ) :: pivot_tol
      REAL ( KIND = rpc_ ) :: pivot_tol_for_basis
      REAL ( KIND = rpc_ ) :: zero_pivot
      REAL ( KIND = rpc_ ) :: static_tolerance
      REAL ( KIND = rpc_ ) :: static_level
      REAL ( KIND = rpc_ ) :: min_diagonal
      REAL ( KIND = rpc_ ) :: stop_absolute
      REAL ( KIND = rpc_ ) :: stop_relative
      LOGICAL ( KIND = C_BOOL ) :: remove_dependencies
      LOGICAL ( KIND = C_BOOL ) :: find_basis_by_transpose
      LOGICAL ( KIND = C_BOOL ) :: affine
      LOGICAL ( KIND = C_BOOL ) :: allow_singular
      LOGICAL ( KIND = C_BOOL ) :: perturb_to_make_definite
      LOGICAL ( KIND = C_BOOL ) :: get_norm_residual
      LOGICAL ( KIND = C_BOOL ) :: check_basis
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: symmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: definite_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: unsymmetric_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( uls_control_type ) :: uls_control
    END TYPE sbls_control_type

    TYPE, BIND( C ) :: sbls_time_type
      REAL ( KIND = rpc_ ) :: total
      REAL ( KIND = rpc_ ) :: form
      REAL ( KIND = rpc_ ) :: factorize
      REAL ( KIND = rpc_ ) :: apply
      REAL ( KIND = rpc_ ) :: clock_total
      REAL ( KIND = rpc_ ) :: clock_form
      REAL ( KIND = rpc_ ) :: clock_factorize
      REAL ( KIND = rpc_ ) :: clock_apply
    END TYPE sbls_time_type

    TYPE, BIND( C ) :: sbls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: sort_status
      INTEGER ( KIND = longc_ ) :: factorization_integer
      INTEGER ( KIND = longc_ ) :: factorization_real
      INTEGER ( KIND = ipc_ ) :: preconditioner
      INTEGER ( KIND = ipc_ ) :: factorization
      INTEGER ( KIND = ipc_ ) :: d_plus
      INTEGER ( KIND = ipc_ ) :: rank
      LOGICAL ( KIND = C_BOOL ) :: rank_def
      LOGICAL ( KIND = C_BOOL ) :: perturbed
      INTEGER ( KIND = ipc_ ) :: iter_pcg
      REAL ( KIND = rpc_ ) :: norm_residual
      LOGICAL ( KIND = C_BOOL ) :: alternative
      TYPE ( sbls_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( uls_inform_type ) :: uls_inform
    END TYPE sbls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( sbls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_sbls_control_type ), INTENT( OUT ) :: fcontrol
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
    fcontrol%len_ulsmin = ccontrol%len_ulsmin
    fcontrol%itref_max = ccontrol%itref_max
    fcontrol%maxit_pcg = ccontrol%maxit_pcg
    fcontrol%new_a = ccontrol%new_a
    fcontrol%new_h = ccontrol%new_h
    fcontrol%new_c = ccontrol%new_c
    fcontrol%preconditioner = ccontrol%preconditioner
    fcontrol%semi_bandwidth = ccontrol%semi_bandwidth
    fcontrol%factorization = ccontrol%factorization
    fcontrol%max_col = ccontrol%max_col
    fcontrol%scaling = ccontrol%scaling
    fcontrol%ordering = ccontrol%ordering

    ! Reals
    fcontrol%pivot_tol = ccontrol%pivot_tol
    fcontrol%pivot_tol_for_basis = ccontrol%pivot_tol_for_basis
    fcontrol%zero_pivot = ccontrol%zero_pivot
    fcontrol%static_tolerance = ccontrol%static_tolerance
    fcontrol%static_level = ccontrol%static_level
    fcontrol%min_diagonal = ccontrol%min_diagonal
    fcontrol%stop_absolute = ccontrol%stop_absolute
    fcontrol%stop_relative = ccontrol%stop_relative

    ! Logicals
    fcontrol%remove_dependencies = ccontrol%remove_dependencies
    fcontrol%find_basis_by_transpose = ccontrol%find_basis_by_transpose
    fcontrol%affine = ccontrol%affine
    fcontrol%allow_singular = ccontrol%allow_singular
    fcontrol%perturb_to_make_definite = ccontrol%perturb_to_make_definite
    fcontrol%get_norm_residual = ccontrol%get_norm_residual
    fcontrol%check_basis = ccontrol%check_basis
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
    DO i = 1, LEN( fcontrol%definite_linear_solver )
      IF ( ccontrol%definite_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%definite_linear_solver( i : i )                                 &
        = ccontrol%definite_linear_solver( i )
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
    TYPE ( f_sbls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( sbls_control_type ), INTENT( OUT ) :: ccontrol
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
    ccontrol%len_ulsmin = fcontrol%len_ulsmin
    ccontrol%itref_max = fcontrol%itref_max
    ccontrol%maxit_pcg = fcontrol%maxit_pcg
    ccontrol%new_a = fcontrol%new_a
    ccontrol%new_h = fcontrol%new_h
    ccontrol%new_c = fcontrol%new_c
    ccontrol%preconditioner = fcontrol%preconditioner
    ccontrol%semi_bandwidth = fcontrol%semi_bandwidth
    ccontrol%factorization = fcontrol%factorization
    ccontrol%max_col = fcontrol%max_col
    ccontrol%scaling = fcontrol%scaling
    ccontrol%ordering = fcontrol%ordering

    ! Reals
    ccontrol%pivot_tol = fcontrol%pivot_tol
    ccontrol%pivot_tol_for_basis = fcontrol%pivot_tol_for_basis
    ccontrol%zero_pivot = fcontrol%zero_pivot
    ccontrol%static_tolerance = fcontrol%static_tolerance
    ccontrol%static_level = fcontrol%static_level
    ccontrol%min_diagonal = fcontrol%min_diagonal
    ccontrol%stop_absolute = fcontrol%stop_absolute
    ccontrol%stop_relative = fcontrol%stop_relative

    ! Logicals
    ccontrol%remove_dependencies = fcontrol%remove_dependencies
    ccontrol%find_basis_by_transpose = fcontrol%find_basis_by_transpose
    ccontrol%affine = fcontrol%affine
    ccontrol%allow_singular = fcontrol%allow_singular
    ccontrol%perturb_to_make_definite = fcontrol%perturb_to_make_definite
    ccontrol%get_norm_residual = fcontrol%get_norm_residual
    ccontrol%check_basis = fcontrol%check_basis
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
    l = LEN( fcontrol%definite_linear_solver )
    DO i = 1, l
      ccontrol%definite_linear_solver( i )                                     &
        = fcontrol%definite_linear_solver( i : i )
    END DO
    ccontrol%definite_linear_solver( l + 1 ) = C_NULL_CHAR
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
    TYPE ( sbls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_sbls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%form = ctime%form
    ftime%factorize = ctime%factorize
    ftime%apply = ctime%apply
    ftime%clock_total = ctime%clock_total
    ftime%clock_form = ctime%clock_form
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_apply = ctime%clock_apply
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime )
    TYPE ( f_sbls_time_type ), INTENT( IN ) :: ftime
    TYPE ( sbls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%form = ftime%form
    ctime%factorize = ftime%factorize
    ctime%apply = ftime%apply
    ctime%clock_total = ftime%clock_total
    ctime%clock_form = ftime%clock_form
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_apply = ftime%clock_apply
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( sbls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_sbls_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%sort_status = cinform%sort_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%preconditioner = cinform%preconditioner
    finform%factorization = cinform%factorization
    finform%d_plus = cinform%d_plus
    finform%rank = cinform%rank
    finform%iter_pcg = cinform%iter_pcg

    ! Reals
    finform%norm_residual = cinform%norm_residual

    ! Logicals
    finform%rank_def = cinform%rank_def
    finform%perturbed = cinform%perturbed
    finform%alternative = cinform%alternative

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
    TYPE ( f_sbls_inform_type ), INTENT( IN ) :: finform
    TYPE ( sbls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%sort_status = finform%sort_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%preconditioner = finform%preconditioner
    cinform%factorization = finform%factorization
    cinform%d_plus = finform%d_plus
    cinform%rank = finform%rank
    cinform%iter_pcg = finform%iter_pcg

    ! Reals
    cinform%norm_residual = finform%norm_residual

    ! Logicals
    cinform%rank_def = finform%rank_def
    cinform%perturbed = finform%perturbed
    cinform%alternative = finform%alternative

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

  END MODULE GALAHAD_SBLS_precision_ciface

!  -------------------------------------
!  C interface to fortran sbls_initialize
!  -------------------------------------

  SUBROUTINE sbls_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( sbls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_sbls_full_data_type ), POINTER :: fdata
  TYPE ( f_sbls_control_type ) :: fcontrol
  TYPE ( f_sbls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_sbls_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )

  RETURN

  END SUBROUTINE sbls_initialize

!  ----------------------------------------
!  C interface to fortran sbls_read_specfile
!  ----------------------------------------

  SUBROUTINE sbls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( sbls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_sbls_control_type ) :: fcontrol
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

  CALL f_sbls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sbls_read_specfile

!  ----------------------------------
!  C interface to fortran sbls_import
!  ----------------------------------

  SUBROUTINE sbls_import( ccontrol, cdata, status, n, m,                       &
                          chtype, hne, hrow, hcol, hptr,                       &
                          catype, ane, arow, acol, aptr,                       &
                          cctype, cne, crow, ccol, cptr ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sbls_control_type ), INTENT( INOUT ) :: ccontrol
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
  TYPE ( f_sbls_control_type ) :: fcontrol
  TYPE ( f_sbls_full_data_type ), POINTER :: fdata
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

!  import the problem data into the required SBLS structure

  CALL f_sbls_import( fcontrol, fdata, status, n, m,                           &
                      fhtype, hne, hrow, hcol, hptr,                           &
                      fatype, ane, arow, acol, aptr,                           &
                      fctype, cne, crow, ccol, cptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE sbls_import

!  ----------------------------------------
!  C interface to fortran sbls_reset_control
!  -----------------------------------------

  SUBROUTINE sbls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( sbls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sbls_control_type ) :: fcontrol
  TYPE ( f_sbls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_SBLS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE sbls_reset_control

!  --------------------------------------------
!  C interface to fortran sbls_factorize_matrix
!  --------------------------------------------

  SUBROUTINE sbls_factorize_matrix( cdata, status, n, hne, hval,               &
                                    ane, aval, cne, cval, d ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, ane, hne, cne
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( cne ) :: cval
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ), OPTIONAL :: d
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sbls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_sbls_factorize_matrix( fdata, status, hval, aval, cval, D = d )
  RETURN

  END SUBROUTINE sbls_factorize_matrix

!  ----------------------------------------
!  C interface to fortran sbls_solve_system
!  ----------------------------------------

  SUBROUTINE sbls_solve_system( cdata, status, n, m, sol ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( n + m ) :: sol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_sbls_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  CALL f_sbls_solve_system( fdata, status, sol )
  RETURN

  END SUBROUTINE sbls_solve_system

!  ---------------------------------------
!  C interface to fortran sbls_information
!  ---------------------------------------

  SUBROUTINE sbls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sbls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_sbls_full_data_type ), pointer :: fdata
  TYPE ( f_sbls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain SBLS solution information

  CALL f_sbls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE sbls_information

!  ------------------------------------
!  C interface to fortran sbls_terminate
!  ------------------------------------

  SUBROUTINE sbls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_SBLS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( sbls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( sbls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_sbls_full_data_type ), pointer :: fdata
  TYPE ( f_sbls_control_type ) :: fcontrol
  TYPE ( f_sbls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_sbls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE sbls_terminate
