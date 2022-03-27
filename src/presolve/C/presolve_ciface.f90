! THIS VERSION: GALAHAD 4.0 - 2022-03-27 AT 11:46 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  P R E S O L V E    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. March 27th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_PRESOLVE_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_PRESOLVE_double, ONLY: &
        f_presolve_control_type => PRESOLVE_control_type, &
        f_presolve_inform_type => PRESOLVE_inform_type, &
        f_presolve_full_data_type => PRESOLVE_full_data_type, &
        f_presolve_initialize => PRESOLVE_initialize, &
        f_presolve_read_specfile => PRESOLVE_read_specfile, &
        f_presolve_import => PRESOLVE_import, &
        f_presolve_reset_control => PRESOLVE_reset_control, &
        f_presolve_information => PRESOLVE_information, &
        f_presolve_terminate => PRESOLVE_terminate

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: presolve_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: termination
      INTEGER ( KIND = C_INT ) :: max_nbr_transforms
      INTEGER ( KIND = C_INT ) :: max_nbr_passes
      REAL ( KIND = wp ) :: c_accuracy
      REAL ( KIND = wp ) :: z_accuracy
      REAL ( KIND = wp ) :: infinity
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: errout
      INTEGER ( KIND = C_INT ) :: print_level
      LOGICAL ( KIND = C_BOOL ) :: dual_transformations
      LOGICAL ( KIND = C_BOOL ) :: redundant_xc
      INTEGER ( KIND = C_INT ) :: primal_constraints_freq
      INTEGER ( KIND = C_INT ) :: dual_constraints_freq
      INTEGER ( KIND = C_INT ) :: singleton_columns_freq
      INTEGER ( KIND = C_INT ) :: doubleton_columns_freq
      INTEGER ( KIND = C_INT ) :: unc_variables_freq
      INTEGER ( KIND = C_INT ) :: dependent_variables_freq
      INTEGER ( KIND = C_INT ) :: sparsify_rows_freq
      INTEGER ( KIND = C_INT ) :: max_fill
      INTEGER ( KIND = C_INT ) :: transf_file_nbr
      INTEGER ( KIND = C_INT ) :: transf_buffer_size
      INTEGER ( KIND = C_INT ) :: transf_file_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: transf_file_name
      INTEGER ( KIND = C_INT ) :: y_sign
      INTEGER ( KIND = C_INT ) :: inactive_y
      INTEGER ( KIND = C_INT ) :: z_sign
      INTEGER ( KIND = C_INT ) :: inactive_z
      INTEGER ( KIND = C_INT ) :: final_x_bounds
      INTEGER ( KIND = C_INT ) :: final_z_bounds
      INTEGER ( KIND = C_INT ) :: final_c_bounds
      INTEGER ( KIND = C_INT ) :: final_y_bounds
      INTEGER ( KIND = C_INT ) :: check_primal_feasibility
      INTEGER ( KIND = C_INT ) :: check_dual_feasibility
      LOGICAL ( KIND = C_BOOL ) :: get_q
      LOGICAL ( KIND = C_BOOL ) :: get_f
      LOGICAL ( KIND = C_BOOL ) :: get_g
      LOGICAL ( KIND = C_BOOL ) :: get_H
      LOGICAL ( KIND = C_BOOL ) :: get_A
      LOGICAL ( KIND = C_BOOL ) :: get_x
      LOGICAL ( KIND = C_BOOL ) :: get_x_bounds
      LOGICAL ( KIND = C_BOOL ) :: get_z
      LOGICAL ( KIND = C_BOOL ) :: get_z_bounds
      LOGICAL ( KIND = C_BOOL ) :: get_c
      LOGICAL ( KIND = C_BOOL ) :: get_c_bounds=
      LOGICAL ( KIND = C_BOOL ) :: get_y
      LOGICAL ( KIND = C_BOOL ) :: get_y_bounds
      REAL ( KIND = wp ) :: pivot_tol
      REAL ( KIND = wp ) :: min_rel_improve
      REAL ( KIND = wp ) :: max_growth_factor
    END TYPE presolve_control_type

    TYPE, BIND( C ) :: presolve_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: nbr_transforms
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) ::
    END TYPE presolve_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( presolve_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_presolve_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
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
    fcontrol%get_q = ccontrol%get_q
    fcontrol%get_f = ccontrol%get_f
    fcontrol%get_g = ccontrol%get_g
    fcontrol%get_H = ccontrol%get_H
    fcontrol%get_A = ccontrol%get_A
    fcontrol%get_x = ccontrol%get_x
    fcontrol%get_x_bounds = ccontrol%get_x_bounds
    fcontrol%get_z = ccontrol%get_z
    fcontrol%get_z_bounds = ccontrol%get_z_bounds
    fcontrol%get_c = ccontrol%get_c
    fcontrol%get_c_bounds= = ccontrol%get_c_bounds=
    fcontrol%get_y = ccontrol%get_y
    fcontrol%get_y_bounds = ccontrol%get_y_bounds

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
    INTEGER :: i, l
    
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
    ccontrol%get_q = fcontrol%get_q
    ccontrol%get_f = fcontrol%get_f
    ccontrol%get_g = fcontrol%get_g
    ccontrol%get_H = fcontrol%get_H
    ccontrol%get_A = fcontrol%get_A
    ccontrol%get_x = fcontrol%get_x
    ccontrol%get_x_bounds = fcontrol%get_x_bounds
    ccontrol%get_z = fcontrol%get_z
    ccontrol%get_z_bounds = fcontrol%get_z_bounds
    ccontrol%get_c = fcontrol%get_c
    ccontrol%get_c_bounds= = fcontrol%get_c_bounds=
    ccontrol%get_y = fcontrol%get_y
    ccontrol%get_y_bounds = fcontrol%get_y_bounds

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
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%nbr_transforms = cinform%nbr_transforms

    ! Strings
    DO i = 1, LEN( finform% )
      IF ( cinform%( i ) == C_NULL_CHAR ) EXIT
      finform%( i : i ) = cinform%( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_presolve_inform_type ), INTENT( IN ) :: finform
    TYPE ( presolve_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%nbr_transforms = finform%nbr_transforms

    ! Strings
    l = LEN( finform% )
    DO i = 1, l
      cinform%( i ) = finform%( i : i )
    END DO
    cinform%( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_PRESOLVE_double_ciface

!  -------------------------------------
!  C interface to fortran presolve_initialize
!  -------------------------------------

  SUBROUTINE presolve_initialize( cdata, ccontrol, status ) BIND( C ) 
  USE GALAHAD_PRESOLVE_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
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
  USE GALAHAD_PRESOLVE_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( presolve_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_presolve_control_type ) :: fcontrol
  CHARACTER ( KIND = C_CHAR, LEN = strlen( cspecfile ) ) :: fspecfile
  LOGICAL :: f_indexing

!  device unit number for specfile

  INTEGER ( KIND = C_INT ), PARAMETER :: device = 10

!  convert C string to Fortran string

  fspecfile = cstr_to_fchar( cspecfile )

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )
  
!  open specfile for reading

  OPEN( UNIT = device, FILE = fspecfile )
  
!  read control parameters from the specfile

  CALL f_presolve_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE presolve_read_specfile

!  ---------------------------------
!  C interface to fortran presolve_inport
!  ---------------------------------

  SUBROUTINE presolve_import( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_PRESOLVE_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( presolve_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_presolve_control_type ) :: fcontrol
  TYPE ( f_presolve_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN

!  import the problem data into the required PRESOLVE structure

    CALL f_presolve_import( fcontrol, fdata, status )
  ELSE
    CALL f_presolve_import( fcontrol, fdata, status )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE presolve_import

!  ---------------------------------------
!  C interface to fortran presolve_reset_control
!  ----------------------------------------

  SUBROUTINE presolve_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_PRESOLVE_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( presolve_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_presolve_control_type ) :: fcontrol
  TYPE ( f_presolve_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_PRESOLVE_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE presolve_reset_control

!  --------------------------------------
!  C interface to fortran presolve_information
!  --------------------------------------

  SUBROUTINE presolve_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_PRESOLVE_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( presolve_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

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
  USE GALAHAD_PRESOLVE_double_ciface
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
