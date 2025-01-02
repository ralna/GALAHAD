! THIS VERSION: GALAHAD 4.1 - 2023-01-24 AT 09:30 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-*-*-*-  G A L A H A D _  U L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. November 30th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_ULS_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_ULS_precision, ONLY:                                           &
        f_uls_control_type => ULS_control_type,                                &
        f_uls_inform_type => ULS_inform_type,                                  &
        f_uls_full_data_type => ULS_full_data_type,                            &
        f_uls_initialize => ULS_initialize,                                    &
        f_uls_read_specfile => ULS_read_specfile,                              &
        f_uls_factorize_matrix => ULS_factorize_matrix,                        &
        f_uls_solve_system => ULS_solve_system,                                &
        f_uls_reset_control => ULS_reset_control,                              &
        f_uls_information => ULS_information,                                  &
        f_uls_terminate => ULS_terminate

    USE GALAHAD_GLS_precision_ciface, ONLY:                                    &
        gls_control_type,                                                      &
        gls_ainfo_type,                                                        &
        gls_finfo_type,                                                        &
        gls_sinfo_type,                                                        &
        copy_gls_ainfo_in => copy_ainfo_in,                                    &
        copy_gls_finfo_in => copy_finfo_in,                                    &
        copy_gls_sinfo_in => copy_sinfo_in,                                    &
        copy_gls_ainfo_out => copy_ainfo_out,                                  &
        copy_gls_finfo_out => copy_finfo_out,                                  &
        copy_gls_sinfo_out => copy_sinfo_out

    USE hsl_ma48_precision_ciface, ONLY:                                       &
        ma48_control,                                                          &
        ma48_ainfo,                                                            &
        ma48_finfo,                                                            &
        ma48_sinfo,                                                            &
        copy_ma48_ainfo_out => copy_ainfo_out,                                 &
        copy_ma48_finfo_out => copy_finfo_out,                                 &
        copy_ma48_sinfo_out => copy_sinfo_out

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: uls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: error
      INTEGER ( KIND = ipc_ ) :: warning
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: print_level
      INTEGER ( KIND = ipc_ ) :: print_level_solver
      INTEGER ( KIND = ipc_ ) :: initial_fill_in_factor
      INTEGER ( KIND = ipc_ ) :: min_real_factor_size
      INTEGER ( KIND = ipc_ ) :: min_integer_factor_size
      INTEGER ( KIND = longc_ ) :: max_factor_size
      INTEGER ( KIND = ipc_ ) :: blas_block_size_factorize
      INTEGER ( KIND = ipc_ ) :: blas_block_size_solve
      INTEGER ( KIND = ipc_ ) :: pivot_control
      INTEGER ( KIND = ipc_ ) :: pivot_search_limit
      INTEGER ( KIND = ipc_ ) :: minimum_size_for_btf
      INTEGER ( KIND = ipc_ ) :: max_iterative_refinements
      LOGICAL ( KIND = C_BOOL ) :: stop_if_singular
      REAL ( KIND = rpc_ ) :: array_increase_factor
      REAL ( KIND = rpc_ ) :: switch_to_full_code_density
      REAL ( KIND = rpc_ ) :: array_decrease_factor
      REAL ( KIND = rpc_ ) :: relative_pivot_tolerance
      REAL ( KIND = rpc_ ) :: absolute_pivot_tolerance
      REAL ( KIND = rpc_ ) :: zero_tolerance
      REAL ( KIND = rpc_ ) :: acceptable_residual_relative
      REAL ( KIND = rpc_ ) :: acceptable_residual_absolute
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE uls_control_type

    TYPE, BIND( C ) :: uls_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = ipc_ ) :: more_info
      INTEGER ( KIND = longc_ ) :: out_of_range
      INTEGER ( KIND = longc_ ) :: duplicates
      INTEGER ( KIND = longc_ ) :: entries_dropped
      INTEGER ( KIND = longc_ ) :: workspace_factors
      INTEGER ( KIND = ipc_ ) :: compresses
      INTEGER ( KIND = longc_ ) :: entries_in_factors
      INTEGER ( KIND = ipc_ ) :: rank
      INTEGER ( KIND = ipc_ ) :: structural_rank
      INTEGER ( KIND = ipc_ ) :: pivot_control
      INTEGER ( KIND = ipc_ ) :: iterative_refinements
      LOGICAL ( KIND = C_BOOL ) :: alternative
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 21 ) :: solver
      TYPE ( gls_ainfo_type ) :: gls_ainfo
      TYPE ( gls_finfo_type ) :: gls_finfo
      TYPE ( gls_sinfo_type ) :: gls_sinfo
      TYPE ( ma48_ainfo ) :: ma48_ainfo
      TYPE ( ma48_finfo ) :: ma48_finfo
      TYPE ( ma48_sinfo ) :: ma48_sinfo
      INTEGER ( KIND = ipc_ ) :: lapack_error
    END TYPE uls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( uls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_uls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%warning = ccontrol%warning
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%print_level_solver = ccontrol%print_level_solver
    fcontrol%initial_fill_in_factor = ccontrol%initial_fill_in_factor
    fcontrol%min_real_factor_size = ccontrol%min_real_factor_size
    fcontrol%min_integer_factor_size = ccontrol%min_integer_factor_size
    fcontrol%max_factor_size = ccontrol%max_factor_size
    fcontrol%blas_block_size_factorize = ccontrol%blas_block_size_factorize
    fcontrol%blas_block_size_solve = ccontrol%blas_block_size_solve
    fcontrol%pivot_control = ccontrol%pivot_control
    fcontrol%pivot_search_limit = ccontrol%pivot_search_limit
    fcontrol%minimum_size_for_btf = ccontrol%minimum_size_for_btf
    fcontrol%max_iterative_refinements = ccontrol%max_iterative_refinements

    ! Reals
    fcontrol%array_increase_factor = ccontrol%array_increase_factor
    fcontrol%switch_to_full_code_density = ccontrol%switch_to_full_code_density
    fcontrol%array_decrease_factor = ccontrol%array_decrease_factor
    fcontrol%relative_pivot_tolerance = ccontrol%relative_pivot_tolerance
    fcontrol%absolute_pivot_tolerance = ccontrol%absolute_pivot_tolerance
    fcontrol%zero_tolerance = ccontrol%zero_tolerance
    fcontrol%acceptable_residual_relative                                      &
      = ccontrol%acceptable_residual_relative
    fcontrol%acceptable_residual_absolute                                      &
      = ccontrol%acceptable_residual_absolute

    ! Logicals
    fcontrol%stop_if_singular = ccontrol%stop_if_singular

    ! Strings
    DO i = 1, 31
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_uls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( uls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%warning = fcontrol%warning
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%print_level_solver = fcontrol%print_level_solver
    ccontrol%initial_fill_in_factor = fcontrol%initial_fill_in_factor
    ccontrol%min_real_factor_size = fcontrol%min_real_factor_size
    ccontrol%min_integer_factor_size = fcontrol%min_integer_factor_size
    ccontrol%max_factor_size = fcontrol%max_factor_size
    ccontrol%blas_block_size_factorize = fcontrol%blas_block_size_factorize
    ccontrol%blas_block_size_solve = fcontrol%blas_block_size_solve
    ccontrol%pivot_control = fcontrol%pivot_control
    ccontrol%pivot_search_limit = fcontrol%pivot_search_limit
    ccontrol%minimum_size_for_btf = fcontrol%minimum_size_for_btf
    ccontrol%max_iterative_refinements = fcontrol%max_iterative_refinements

    ! Reals
    ccontrol%array_increase_factor = fcontrol%array_increase_factor
    ccontrol%switch_to_full_code_density = fcontrol%switch_to_full_code_density
    ccontrol%array_decrease_factor = fcontrol%array_decrease_factor
    ccontrol%relative_pivot_tolerance = fcontrol%relative_pivot_tolerance
    ccontrol%absolute_pivot_tolerance = fcontrol%absolute_pivot_tolerance
    ccontrol%zero_tolerance = fcontrol%zero_tolerance
    ccontrol%acceptable_residual_relative                                      &
      = fcontrol%acceptable_residual_relative
    ccontrol%acceptable_residual_absolute                                      &
      = fcontrol%acceptable_residual_absolute

    ! Logicals
    ccontrol%stop_if_singular = fcontrol%stop_if_singular

    ! Strings
    DO i = 1, LEN( fcontrol%prefix )
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( LEN( fcontrol%prefix ) + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( uls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_uls_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%more_info = cinform%more_info
    finform%out_of_range = cinform%out_of_range
    finform%duplicates = cinform%duplicates
    finform%entries_dropped = cinform%entries_dropped
    finform%workspace_factors = cinform%workspace_factors
    finform%compresses = cinform%compresses
    finform%entries_in_factors = cinform%entries_in_factors
    finform%rank = cinform%rank
    finform%structural_rank = cinform%structural_rank
    finform%pivot_control = cinform%pivot_control
    finform%iterative_refinements = cinform%iterative_refinements
    finform%lapack_error = cinform%lapack_error

    ! Logicals
    finform%alternative = cinform%alternative

    ! Derived types
    CALL copy_gls_ainfo_in( cinform%gls_ainfo, finform%gls_ainfo )
    CALL copy_gls_finfo_in( cinform%gls_finfo, finform%gls_finfo )
    CALL copy_gls_sinfo_in( cinform%gls_sinfo, finform%gls_sinfo )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    DO i = 1, LEN( finform%solver )
      IF ( cinform%solver( i ) == C_NULL_CHAR ) EXIT
      finform%solver( i : i ) = cinform%solver( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_uls_inform_type ), INTENT( IN ) :: finform
    TYPE ( uls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%more_info = finform%more_info
    cinform%out_of_range = finform%out_of_range
    cinform%duplicates = finform%duplicates
    cinform%entries_dropped = finform%entries_dropped
    cinform%workspace_factors = finform%workspace_factors
    cinform%compresses = finform%compresses
    cinform%entries_in_factors = finform%entries_in_factors
    cinform%rank = finform%rank
    cinform%structural_rank = finform%structural_rank
    cinform%pivot_control = finform%pivot_control
    cinform%lapack_error = finform%lapack_error

    ! Logicals
    cinform%alternative = finform%alternative

    ! Derived types
    CALL copy_gls_ainfo_out( finform%gls_ainfo, cinform%gls_ainfo )
    CALL copy_gls_finfo_out( finform%gls_finfo, cinform%gls_finfo )
    CALL copy_gls_sinfo_out( finform%gls_sinfo, cinform%gls_sinfo )
    CALL copy_ma48_ainfo_out( finform%ma48_ainfo, cinform%ma48_ainfo )
    CALL copy_ma48_finfo_out( finform%ma48_finfo, cinform%ma48_finfo )
    CALL copy_ma48_sinfo_out( finform%ma48_sinfo, cinform%ma48_sinfo )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    l = LEN( finform%solver )
    DO i = 1, l
      cinform%solver( i ) = finform%solver( i : i )
    END DO
    cinform%solver( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_ULS_precision_ciface

!  -------------------------------------
!  C interface to fortran uls_initialize
!  -------------------------------------

  SUBROUTINE uls_initialize( csolver, cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: csolver
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( uls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_uls_full_data_type ), POINTER :: fdata
  TYPE ( f_uls_control_type ) :: fcontrol
  TYPE ( f_uls_inform_type ) :: finform
  LOGICAL :: f_indexing
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( csolver ) ) :: fsolver

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  convert C string to Fortran string

  fsolver = cstr_to_fchar( csolver )

!  initialize required fortran types

  CALL f_uls_initialize( fsolver, fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE uls_initialize

!  ----------------------------------------
!  C interface to fortran uls_read_specfile
!  ----------------------------------------

  SUBROUTINE uls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( uls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_uls_control_type ) :: fcontrol
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

  CALL f_uls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE uls_read_specfile

!  -----------------------------------------
!  C interface to fortran uls_factorize_matrix
!  -----------------------------------------

  SUBROUTINE uls_factorize_matrix( ccontrol, cdata, status, m, n, ctype, ne,   &
                                   val, row, col, ptr  ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( uls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, n, ne
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: ctype
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ne ) :: val
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ), OPTIONAL :: row
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ne ), OPTIONAL :: col
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: ptr

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( ctype ) ) :: ftype
  TYPE ( f_uls_control_type ) :: fcontrol
  TYPE ( f_uls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  ftype = cstr_to_fchar( ctype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  factorize_matrix the problem data into the required ULS structure

  CALL f_uls_factorize_matrix( fcontrol, fdata, status, m, n, ftype, ne,       &
                               val, row, col, ptr )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE uls_factorize_matrix

!  ----------------------------------------
!  C interface to fortran uls_reset_control
!  -----------------------------------------

  SUBROUTINE uls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( uls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_uls_control_type ) :: fcontrol
  TYPE ( f_uls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_ULS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE uls_reset_control

!  ----------------------------------------
!  C interface to fortran uls_solve_system
!  ----------------------------------------

  SUBROUTINE uls_solve_system( cdata, status, m, n, sol, ctrans ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: m, n
  LOGICAL ( KIND = C_BOOL ), INTENT( IN ), VALUE :: ctrans
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( INOUT ), DIMENSION( MAX( m, n ) ) :: sol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_uls_full_data_type ), POINTER :: fdata
  LOGICAL :: ftrans

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!   form and factorize the block matrix

  ftrans = ctrans
  CALL f_uls_solve_system( fdata, status, sol, ftrans )
  RETURN

  END SUBROUTINE uls_solve_system

!  --------------------------------------
!  C interface to fortran uls_information
!  --------------------------------------

  SUBROUTINE uls_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( uls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_uls_full_data_type ), pointer :: fdata
  TYPE ( f_uls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain ULS solution information

  CALL f_uls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE uls_information

!  ------------------------------------
!  C interface to fortran uls_terminate
!  ------------------------------------

  SUBROUTINE uls_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_ULS_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( uls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( uls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_uls_full_data_type ), pointer :: fdata
  TYPE ( f_uls_control_type ) :: fcontrol
  TYPE ( f_uls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_uls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE uls_terminate
