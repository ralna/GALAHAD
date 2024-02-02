! THIS VERSION: GALAHAD 4.3 - 2023-02-02 AT 07:50 GMT.

#include "galahad_modules.h"
#include "galahad_cfunctions.h"

!-*-*-*-*-  G A L A H A D _  P R E S O L V E    C   I N T E R F A C E  -*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. March 27th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_PRESOL_precision_ciface
    USE GALAHAD_KINDS_precision
    USE GALAHAD_common_ciface
    USE GALAHAD_PRESOLVE_precision, ONLY:                                      &
        f_presolve_control_type => PRESOLVE_control_type,                      &
        f_presolve_inform_type => PRESOLVE_inform_type,                        &
        f_presolve_full_data_type => PRESOLVE_full_data_type,                  &
        f_presolve_initialize => PRESOLVE_initialize,                          &
        f_presolve_read_specfile => PRESOLVE_read_specfile,                    &
        f_presolve_import_problem => PRESOLVE_import_problem,                  &
        f_presolve_transform_problem => PRESOLVE_transform_problem,            &
        f_presolve_restore_solution => PRESOLVE_restore_solution,              &
        f_presolve_information => PRESOLVE_information,                        &
        f_presolve_terminate => PRESOLVE_terminate

    IMPLICIT NONE

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: presolve_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = ipc_ ) :: termination
      INTEGER ( KIND = ipc_ ) :: max_nbr_transforms
      INTEGER ( KIND = ipc_ ) :: max_nbr_passes
      REAL ( KIND = rpc_ ) :: c_accuracy
      REAL ( KIND = rpc_ ) :: z_accuracy
      REAL ( KIND = rpc_ ) :: infinity
      INTEGER ( KIND = ipc_ ) :: out
      INTEGER ( KIND = ipc_ ) :: errout
      INTEGER ( KIND = ipc_ ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: dual_transformations
      LOGICAL ( KIND = C_BOOL ) :: redundant_xc
      INTEGER ( KIND = ipc_ ) :: primal_constraints_freq
      INTEGER ( KIND = ipc_ ) :: dual_constraints_freq
      INTEGER ( KIND = ipc_ ) :: singleton_columns_freq
      INTEGER ( KIND = ipc_ ) :: doubleton_columns_freq
      INTEGER ( KIND = ipc_ ) :: unc_variables_freq
      INTEGER ( KIND = ipc_ ) :: dependent_variables_freq
      INTEGER ( KIND = ipc_ ) :: sparsify_rows_freq
      INTEGER ( KIND = ipc_ ) :: max_fill
      INTEGER ( KIND = ipc_ ) :: transf_file_nbr
      INTEGER ( KIND = ipc_ ) :: transf_buffer_size
      INTEGER ( KIND = ipc_ ) :: transf_file_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: transf_file_name
      INTEGER ( KIND = ipc_ ) :: y_sign
      INTEGER ( KIND = ipc_ ) :: inactive_y
      INTEGER ( KIND = ipc_ ) :: z_sign
      INTEGER ( KIND = ipc_ ) :: inactive_z
      INTEGER ( KIND = ipc_ ) :: final_x_bounds
      INTEGER ( KIND = ipc_ ) :: final_z_bounds
      INTEGER ( KIND = ipc_ ) :: final_c_bounds
      INTEGER ( KIND = ipc_ ) :: final_y_bounds
      INTEGER ( KIND = ipc_ ) :: check_primal_feasibility
      INTEGER ( KIND = ipc_ ) :: check_dual_feasibility
!     LOGICAL ( KIND = C_BOOL ) :: get_q
!     LOGICAL ( KIND = C_BOOL ) :: get_f
!     LOGICAL ( KIND = C_BOOL ) :: get_g
!     LOGICAL ( KIND = C_BOOL ) :: get_H
!     LOGICAL ( KIND = C_BOOL ) :: get_A
!     LOGICAL ( KIND = C_BOOL ) :: get_x
!     LOGICAL ( KIND = C_BOOL ) :: get_x_bounds
!     LOGICAL ( KIND = C_BOOL ) :: get_z
!     LOGICAL ( KIND = C_BOOL ) :: get_z_bounds
!     LOGICAL ( KIND = C_BOOL ) :: get_c
!     LOGICAL ( KIND = C_BOOL ) :: get_c_bounds
!     LOGICAL ( KIND = C_BOOL ) :: get_y
!     LOGICAL ( KIND = C_BOOL ) :: get_y_bounds
      REAL ( KIND = rpc_ ) :: pivot_tol
      REAL ( KIND = rpc_ ) :: min_rel_improve
      REAL ( KIND = rpc_ ) :: max_growth_factor
    END TYPE presolve_control_type

    TYPE, BIND( C ) :: presolve_inform_type
      INTEGER ( KIND = ipc_ ) :: status
      INTEGER ( KIND = ipc_ ) :: status_continue   ! ignore - doxygen fix
      INTEGER ( KIND = ipc_ ) :: status_continued  ! ignore - doxygen fix
      INTEGER ( KIND = ipc_ ) :: nbr_transforms
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 3, 81 ) :: message
    END TYPE presolve_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing )
    TYPE ( presolve_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_presolve_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, OPTIONAL, INTENT( OUT ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%termination = ccontrol%termination
    fcontrol%max_nbr_transforms = ccontrol%max_nbr_transforms
    fcontrol%max_nbr_passes = ccontrol%max_nbr_passes
    fcontrol%out = ccontrol%out
    fcontrol%errout = ccontrol%errout
    fcontrol%print_level = ccontrol%print_level
    fcontrol%primal_constraints_freq = ccontrol%primal_constraints_freq
    fcontrol%dual_constraints_freq = ccontrol%dual_constraints_freq
    fcontrol%singleton_columns_freq = ccontrol%singleton_columns_freq
    fcontrol%doubleton_columns_freq = ccontrol%doubleton_columns_freq
    fcontrol%unc_variables_freq = ccontrol%unc_variables_freq
    fcontrol%dependent_variables_freq = ccontrol%dependent_variables_freq
    fcontrol%sparsify_rows_freq = ccontrol%sparsify_rows_freq
    fcontrol%max_fill = ccontrol%max_fill
    fcontrol%transf_file_nbr = ccontrol%transf_file_nbr
    fcontrol%transf_buffer_size = ccontrol%transf_buffer_size
    fcontrol%transf_file_status = ccontrol%transf_file_status
    fcontrol%y_sign = ccontrol%y_sign
    fcontrol%inactive_y = ccontrol%inactive_y
    fcontrol%z_sign = ccontrol%z_sign
    fcontrol%inactive_z = ccontrol%inactive_z
    fcontrol%final_x_bounds = ccontrol%final_x_bounds
    fcontrol%final_z_bounds = ccontrol%final_z_bounds
    fcontrol%final_c_bounds = ccontrol%final_c_bounds
    fcontrol%final_y_bounds = ccontrol%final_y_bounds
    fcontrol%check_primal_feasibility = ccontrol%check_primal_feasibility
    fcontrol%check_dual_feasibility = ccontrol%check_dual_feasibility

    ! Reals
    fcontrol%c_accuracy = ccontrol%c_accuracy
    fcontrol%z_accuracy = ccontrol%z_accuracy
    fcontrol%infinity = ccontrol%infinity
    fcontrol%pivot_tol = ccontrol%pivot_tol
    fcontrol%min_rel_improve = ccontrol%min_rel_improve
    fcontrol%max_growth_factor = ccontrol%max_growth_factor

    ! Logicals
    fcontrol%dual_transformations = ccontrol%dual_transformations
    fcontrol%redundant_xc = ccontrol%redundant_xc
!   fcontrol%get_q = ccontrol%get_q
!   fcontrol%get_f = ccontrol%get_f
!   fcontrol%get_g = ccontrol%get_g
!   fcontrol%get_H = ccontrol%get_H
!   fcontrol%get_A = ccontrol%get_A
!   fcontrol%get_x = ccontrol%get_x
!   fcontrol%get_x_bounds = ccontrol%get_x_bounds
!   fcontrol%get_z = ccontrol%get_z
!   fcontrol%get_z_bounds = ccontrol%get_z_bounds
!   fcontrol%get_c = ccontrol%get_c
!   fcontrol%get_c_bounds = ccontrol%get_c_bounds
!   fcontrol%get_y = ccontrol%get_y
!   fcontrol%get_y_bounds = ccontrol%get_y_bounds

    ! Strings
    DO i = 1, LEN( fcontrol%transf_file_name )
      IF ( ccontrol%transf_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%transf_file_name( i : i ) = ccontrol%transf_file_name( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing )
    TYPE ( f_presolve_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( presolve_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER ( KIND = ip_ ) :: i, l

    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%termination = fcontrol%termination
    ccontrol%max_nbr_transforms = fcontrol%max_nbr_transforms
    ccontrol%max_nbr_passes = fcontrol%max_nbr_passes
    ccontrol%out = fcontrol%out
    ccontrol%errout = fcontrol%errout
    ccontrol%print_level = fcontrol%print_level
    ccontrol%primal_constraints_freq = fcontrol%primal_constraints_freq
    ccontrol%dual_constraints_freq = fcontrol%dual_constraints_freq
    ccontrol%singleton_columns_freq = fcontrol%singleton_columns_freq
    ccontrol%doubleton_columns_freq = fcontrol%doubleton_columns_freq
    ccontrol%unc_variables_freq = fcontrol%unc_variables_freq
    ccontrol%dependent_variables_freq = fcontrol%dependent_variables_freq
    ccontrol%sparsify_rows_freq = fcontrol%sparsify_rows_freq
    ccontrol%max_fill = fcontrol%max_fill
    ccontrol%transf_file_nbr = fcontrol%transf_file_nbr
    ccontrol%transf_buffer_size = fcontrol%transf_buffer_size
    ccontrol%transf_file_status = fcontrol%transf_file_status
    ccontrol%y_sign = fcontrol%y_sign
    ccontrol%inactive_y = fcontrol%inactive_y
    ccontrol%z_sign = fcontrol%z_sign
    ccontrol%inactive_z = fcontrol%inactive_z
    ccontrol%final_x_bounds = fcontrol%final_x_bounds
    ccontrol%final_z_bounds = fcontrol%final_z_bounds
    ccontrol%final_c_bounds = fcontrol%final_c_bounds
    ccontrol%final_y_bounds = fcontrol%final_y_bounds
    ccontrol%check_primal_feasibility = fcontrol%check_primal_feasibility
    ccontrol%check_dual_feasibility = fcontrol%check_dual_feasibility

    ! Reals
    ccontrol%c_accuracy = fcontrol%c_accuracy
    ccontrol%z_accuracy = fcontrol%z_accuracy
    ccontrol%infinity = fcontrol%infinity
    ccontrol%pivot_tol = fcontrol%pivot_tol
    ccontrol%min_rel_improve = fcontrol%min_rel_improve
    ccontrol%max_growth_factor = fcontrol%max_growth_factor

    ! Logicals
    ccontrol%dual_transformations = fcontrol%dual_transformations
    ccontrol%redundant_xc = fcontrol%redundant_xc
!   ccontrol%get_q = fcontrol%get_q
!   ccontrol%get_f = fcontrol%get_f
!   ccontrol%get_g = fcontrol%get_g
!   ccontrol%get_H = fcontrol%get_H
!   ccontrol%get_A = fcontrol%get_A
!   ccontrol%get_x = fcontrol%get_x
!   ccontrol%get_x_bounds = fcontrol%get_x_bounds
!   ccontrol%get_z = fcontrol%get_z
!   ccontrol%get_z_bounds = fcontrol%get_z_bounds
!   ccontrol%get_c = fcontrol%get_c
!   ccontrol%get_c_bounds = fcontrol%get_c_bounds
!   ccontrol%get_y = fcontrol%get_y
!   ccontrol%get_y_bounds = fcontrol%get_y_bounds

    ! Strings
    l = LEN( fcontrol%transf_file_name )
    DO i = 1, l
      ccontrol%transf_file_name( i ) = fcontrol%transf_file_name( i : i )
    END DO
    ccontrol%transf_file_name( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform )
    TYPE ( presolve_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_presolve_inform_type ), INTENT( OUT ) :: finform
    INTEGER ( KIND = ip_ ) :: i, j

    ! Integers
    finform%status = cinform%status
    finform%nbr_transforms = cinform%nbr_transforms

    ! Strings
    DO j = 1, 3
      DO i = 1, LEN( finform%message( j ) )
        IF ( cinform%message( j, i ) == C_NULL_CHAR ) EXIT
        finform%message( j )( i : i ) = cinform%message( j, i )
      END DO
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform )
    TYPE ( f_presolve_inform_type ), INTENT( IN ) :: finform
    TYPE ( presolve_inform_type ), INTENT( OUT ) :: cinform
    INTEGER ( KIND = ip_ ) :: i, j, l

    ! Integers
    cinform%status = finform%status
    cinform%nbr_transforms = finform%nbr_transforms

    ! Strings
    DO j = 1, 3
      l = LEN( finform%message( j ) )
      DO i = 1, l
        cinform%message( j, i ) = finform%message( j )( i : i )
      END DO
      cinform%message( j, l + 1 ) = C_NULL_CHAR
    END DO
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_PRESOL_precision_ciface

!  -------------------------------------
!  C interface to fortran presolve_initialize
!  -------------------------------------

  SUBROUTINE presolve_initialize( cdata, ccontrol, status ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( presolve_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_presolve_full_data_type ), POINTER :: fdata
  TYPE ( f_presolve_control_type ) :: fcontrol
  TYPE ( f_presolve_inform_type ) :: finform
  LOGICAL :: f_indexing

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_presolve_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE presolve_initialize

!  ----------------------------------------
!  C interface to fortran presolve_read_specfile
!  ----------------------------------------

  SUBROUTINE presolve_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( presolve_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_presolve_control_type ) :: fcontrol
  TYPE ( f_presolve_inform_type ) :: finform
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

  CALL f_presolve_read_specfile( device, fcontrol, finform )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE presolve_read_specfile

!  ----------------------------------------------
!  C interface to fortran presolve_inport_problem
!  ----------------------------------------------

  SUBROUTINE presolve_import_problem( ccontrol, cdata, status, n, m,           &
                                      chtype, hne, hrow, hcol, hptr, hval, g,  &
                                      f, catype, ane, arow, acol, aptr, aval,  &
                                      cl, cu, xl, xu, n_out, m_out,            &
                                      hne_out, ane_out ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( presolve_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, ane
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: n_out, m_out, hne_out, ane_out
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hrow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( hne ) :: hval
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: chtype
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( IN ), VALUE :: f
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( ane ) :: aval
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: xl, xu


!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( chtype ) ) :: fhtype
  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_presolve_control_type ) :: fcontrol
  TYPE ( f_presolve_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fhtype = cstr_to_fchar( chtype )
  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required PRESOLVE structure

  CALL f_presolve_import_problem( fcontrol, fdata, status, n, m,               &
                                  fhtype, hne, hrow, hcol, hptr, hval, g, f,   &
                                  fatype, ane, arow, acol, aptr, aval,         &
                                  cl, cu, xl, xu,                              &
                                  n_out, m_out, hne_out, ane_out )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE presolve_import_problem

!  -------------------------------------------------
!  C interface to fortran presolve_transform_problem
!  -------------------------------------------------

  SUBROUTINE presolve_transform_problem( cdata, status, n, m,                  &
                                         hne, hcol, hptr, hval, g, f,          &
                                         ane, acol, aptr, aval, cl, cu,        &
                                         xl, xu, yl, yu, zl, zu ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n, m, hne, ane
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( hne ), OPTIONAL :: hcol
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( n + 1 ), OPTIONAL :: hptr
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( hne ) :: hval
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: g
  REAL ( KIND = rpc_ ), INTENT( OUT ) :: f
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = ipc_ ), INTENT( OUT ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( ane ) :: aval
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: yl, yu
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: zl, zu

!  local variables

  TYPE ( f_presolve_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  check if sufficient space has been provided to record the transformation

   IF ( n /= fdata%n_trans .OR. m /= fdata%m_trans .OR.                        &
        hne /= fdata%h_ne_trans .OR. ane /= fdata%a_ne_trans ) THEN
     status = - 3
     RETURN
   END IF

!  transform the problem data

  CALL f_presolve_transform_problem( fdata, status, hcol, hptr,                &
                                     hval, g, f, acol, aptr, aval,             &
                                     cl, cu, xl, xu, yl, yu, zl, zu )
  RETURN

  END SUBROUTINE presolve_transform_problem

!  ----------------------------------------------
!  C interface to fortran presolve_solve_presolve
!  ----------------------------------------------

  SUBROUTINE presolve_restore_solution( cdata, status,                         &
                                        n_in, m_in, x_in, c_in, y_in, z_in,    &
                                        n, m, x, c, y, z ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = ipc_ ), INTENT( IN ), VALUE :: n_in, m_in, n, m
  INTEGER ( KIND = ipc_ ), INTENT( INOUT ) :: status
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( m ) :: x_in, z_in
  REAL ( KIND = rpc_ ), INTENT( IN ), DIMENSION( n ) :: c_in, y_in
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = rpc_ ), INTENT( OUT ), DIMENSION( m ) :: c, y
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_presolve_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  check if sufficient space has been provided to recover the solution

   IF ( n_in /= fdata%n_trans .OR. m_in /= fdata%m_trans .OR.                  &
        n /= fdata%n_orig .OR. m /= fdata%m_orig ) THEN
     status = - 3
     RETURN
   END IF

!  solve the qp

  CALL f_presolve_restore_solution( fdata, status, x_in, c_in, y_in, z_in,     &
                                    x, c, y, z )
  RETURN

  END SUBROUTINE presolve_restore_solution

!  -------------------------------------------
!  C interface to fortran presolve_information
!  -------------------------------------------

  SUBROUTINE presolve_information( cdata, cinform, status ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( presolve_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = ipc_ ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_presolve_full_data_type ), pointer :: fdata
  TYPE ( f_presolve_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain PRESOLVE solution information

  CALL f_presolve_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE presolve_information

!  ------------------------------------
!  C interface to fortran presolve_terminate
!  ------------------------------------

  SUBROUTINE presolve_terminate( cdata, ccontrol, cinform ) BIND( C )
  USE GALAHAD_PRESOL_precision_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( presolve_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( presolve_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_presolve_full_data_type ), pointer :: fdata
  TYPE ( f_presolve_control_type ) :: fcontrol
  TYPE ( f_presolve_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_presolve_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR
  RETURN

  END SUBROUTINE presolve_terminate
