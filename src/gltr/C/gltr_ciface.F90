! THIS VERSION: GALAHAD 4.0 - 2022-01-06 AT 09:00 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  G L T R    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 3.3. December 16th 2021

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_GLTR_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_GLTR_double, ONLY:                                             &
        f_gltr_control_type => GLTR_control_type,                              &
        f_gltr_inform_type => GLTR_inform_type,                                &
        f_gltr_full_data_type => GLTR_full_data_type,                          &
        f_gltr_initialize => GLTR_initialize,                                  &
        f_gltr_read_specfile => GLTR_read_specfile,                            &
        f_gltr_import_control => GLTR_import_control,                          &
        f_gltr_solve_problem => GLTR_solve_problem,                            &
        f_gltr_information => GLTR_information,                                &
        f_gltr_terminate => GLTR_terminate

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: gltr_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: itmax
      INTEGER ( KIND = C_INT ) :: Lanczos_itmax
      INTEGER ( KIND = C_INT ) :: extra_vectors
      INTEGER ( KIND = C_INT ) :: ritz_printout_device
      REAL ( KIND = wp ) :: stop_relative
      REAL ( KIND = wp ) :: stop_absolute
      REAL ( KIND = wp ) :: fraction_opt
      REAL ( KIND = wp ) :: f_min
      REAL ( KIND = wp ) :: rminvr_zero
      REAL ( KIND = wp ) :: f_0
      LOGICAL ( KIND = C_BOOL ) :: unitm
      LOGICAL ( KIND = C_BOOL ) :: steihaug_toint
      LOGICAL ( KIND = C_BOOL ) :: boundary
      LOGICAL ( KIND = C_BOOL ) :: equality_problem
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: print_ritz_values
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: ritz_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE gltr_control_type

    TYPE, BIND( C ) :: gltr_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = C_INT ) :: iter
      INTEGER ( KIND = C_INT ) :: iter_pass2
      REAL ( KIND = wp ) :: obj
      REAL ( KIND = wp ) :: multiplier
      REAL ( KIND = wp ) :: mnormx
      REAL ( KIND = wp ) :: piv
      REAL ( KIND = wp ) :: curv
      REAL ( KIND = wp ) :: rayleigh
      REAL ( KIND = wp ) :: leftmost
      LOGICAL ( KIND = C_BOOL ) :: negative_curvature
      LOGICAL ( KIND = C_BOOL ) :: hard_case
    END TYPE gltr_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( gltr_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_gltr_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%itmax = ccontrol%itmax
    fcontrol%Lanczos_itmax = ccontrol%Lanczos_itmax
    fcontrol%extra_vectors = ccontrol%extra_vectors
    fcontrol%ritz_printout_device = ccontrol%ritz_printout_device

    ! Reals
    fcontrol%stop_relative = ccontrol%stop_relative
    fcontrol%stop_absolute = ccontrol%stop_absolute
    fcontrol%fraction_opt = ccontrol%fraction_opt
    fcontrol%f_min = ccontrol%f_min
    fcontrol%rminvr_zero = ccontrol%rminvr_zero
    fcontrol%f_0 = ccontrol%f_0

    ! Logicals
    fcontrol%unitm = ccontrol%unitm
    fcontrol%steihaug_toint = ccontrol%steihaug_toint
    fcontrol%boundary = ccontrol%boundary
    fcontrol%equality_problem = ccontrol%equality_problem
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%print_ritz_values = ccontrol%print_ritz_values

    ! Strings
    DO i = 1, LEN( fcontrol%ritz_file_name )
      IF ( ccontrol%ritz_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%ritz_file_name( i : i ) = ccontrol%ritz_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_gltr_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( gltr_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i, l
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%itmax = fcontrol%itmax
    ccontrol%Lanczos_itmax = fcontrol%Lanczos_itmax
    ccontrol%extra_vectors = fcontrol%extra_vectors
    ccontrol%ritz_printout_device = fcontrol%ritz_printout_device

    ! Reals
    ccontrol%stop_relative = fcontrol%stop_relative
    ccontrol%stop_absolute = fcontrol%stop_absolute
    ccontrol%fraction_opt = fcontrol%fraction_opt
    ccontrol%f_min = fcontrol%f_min
    ccontrol%rminvr_zero = fcontrol%rminvr_zero
    ccontrol%f_0 = fcontrol%f_0

    ! Logicals
    ccontrol%unitm = fcontrol%unitm
    ccontrol%steihaug_toint = fcontrol%steihaug_toint
    ccontrol%boundary = fcontrol%boundary
    ccontrol%equality_problem = fcontrol%equality_problem
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%print_ritz_values = fcontrol%print_ritz_values

    ! Strings
    l = LEN( fcontrol%ritz_file_name )
    DO i = 1, l
      ccontrol%ritz_file_name( i ) = fcontrol%ritz_file_name( i : i )
    END DO
    ccontrol%ritz_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( gltr_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_gltr_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%iter_pass2 = cinform%iter_pass2

    ! Reals
    finform%obj = cinform%obj
    finform%multiplier = cinform%multiplier
    finform%mnormx = cinform%mnormx
    finform%piv = cinform%piv
    finform%curv = cinform%curv
    finform%rayleigh = cinform%rayleigh
    finform%leftmost = cinform%leftmost

    ! Logicals
    finform%negative_curvature = cinform%negative_curvature
    finform%hard_case = cinform%hard_case

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_gltr_inform_type ), INTENT( IN ) :: finform
    TYPE ( gltr_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%iter_pass2 = finform%iter_pass2

    ! Reals
    cinform%obj = finform%obj
    cinform%multiplier = finform%multiplier
    cinform%mnormx = finform%mnormx
    cinform%piv = finform%piv
    cinform%curv = finform%curv
    cinform%rayleigh = finform%rayleigh
    cinform%leftmost = finform%leftmost

    ! Logicals
    cinform%negative_curvature = finform%negative_curvature
    cinform%hard_case = finform%hard_case

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_GLTR_double_ciface

!  -------------------------------------
!  C interface to fortran gltr_initialize
!  -------------------------------------

  SUBROUTINE gltr_initialize( cdata, ccontrol, status ) BIND( C ) 
  USE GALAHAD_GLTR_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( gltr_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_gltr_full_data_type ), POINTER :: fdata
  TYPE ( f_gltr_control_type ) :: fcontrol
  TYPE ( f_gltr_inform_type ) :: finform
  LOGICAL :: f_indexing 

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_gltr_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE gltr_initialize

!  ----------------------------------------
!  C interface to fortran gltr_read_specfile
!  ----------------------------------------

  SUBROUTINE gltr_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_GLTR_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( gltr_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_gltr_control_type ) :: fcontrol
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

  CALL f_gltr_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE gltr_read_specfile

!  ------------------------------------------
!  C interface to fortran gltr_import_control
!  ------------------------------------------

  SUBROUTINE gltr_import_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_GLTR_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( gltr_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_gltr_control_type ) :: fcontrol
  TYPE ( f_gltr_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the problem data into the required GLTR structure

   CALL f_gltr_import_control( fcontrol, fdata, status )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE gltr_import_control

!  -----------------------------------------
!  C interface to fortran gltr_solve_problem
!  -----------------------------------------

  SUBROUTINE gltr_solve_problem( cdata, status, n, radius, x, r,               &
                                 vector ) BIND( C )
  USE GALAHAD_GLTR_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  REAL ( KIND = wp ), INTENT( IN ), VALUE :: radius
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: r
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: vector
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_gltr_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_gltr_solve_problem( fdata, status, n, radius, x, r, vector )
  RETURN

  END SUBROUTINE gltr_solve_problem

!  ---------------------------------------
!  C interface to fortran gltr_information
!  ---------------------------------------

  SUBROUTINE gltr_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_GLTR_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( gltr_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_gltr_full_data_type ), pointer :: fdata
  TYPE ( f_gltr_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain GLTR solution information

  CALL f_gltr_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE gltr_information

!  ------------------------------------
!  C interface to fortran gltr_terminate
!  ------------------------------------

  SUBROUTINE gltr_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_GLTR_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( gltr_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( gltr_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_gltr_full_data_type ), pointer :: fdata
  TYPE ( f_gltr_control_type ) :: fcontrol
  TYPE ( f_gltr_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_gltr_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE gltr_terminate
