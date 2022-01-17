! THIS VERSION: GALAHAD 4.0 - 2022-01-14 AT 14:24 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  P S L S    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 14th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_PSLS_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_PSLS_double, ONLY: &
        f_psls_control_type => PSLS_control_type, &
        f_psls_time_type => PSLS_time_type, &
        f_psls_inform_type => PSLS_inform_type, &
        f_psls_full_data_type => PSLS_full_data_type, &
        f_psls_initialize => PSLS_initialize, &
        f_psls_read_specfile => PSLS_read_specfile, &
        f_psls_import => PSLS_import, &
        f_psls_reset_control => PSLS_reset_control, &
        f_psls_information => PSLS_information, &
        f_psls_terminate => PSLS_terminate

    USE GALAHAD_SLS_double_ciface, ONLY: &
        sls_inform_type, &
        sls_control_type, &
        copy_sls_inform_in => copy_inform_in, &
        copy_sls_inform_out => copy_inform_out, &
        copy_sls_control_in => copy_control_in, &
        copy_sls_control_out => copy_control_out

    USE GALAHAD_MI28_double_ciface, ONLY: &
        mi28_inform_type, &
        mi28_control_type, &
        copy_mi28_inform_in => copy_inform_in, &
        copy_mi28_inform_out => copy_inform_out, &
        copy_mi28_control_in => copy_control_in, &
        copy_mi28_control_out => copy_control_out

    USE GALAHAD_MI28_double_ciface, ONLY: &
        mi28_inform_type, &
        mi28_control_type, &
        copy_mi28_inform_in => copy_inform_in, &
        copy_mi28_inform_out => copy_inform_out, &
        copy_mi28_control_in => copy_control_in, &
        copy_mi28_control_out => copy_control_out

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: psls_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: preconditioner
      INTEGER ( KIND = C_INT ) :: semi_bandwidth
      INTEGER ( KIND = C_INT ) :: scaling
      INTEGER ( KIND = C_INT ) :: ordering
      INTEGER ( KIND = C_INT ) :: max_col
      INTEGER ( KIND = C_INT ) :: icfs_vectors
      INTEGER ( KIND = C_INT ) :: mi28_lsize
      INTEGER ( KIND = C_INT ) :: mi28_rsize
      REAL ( KIND = wp ) :: min_diagonal
      LOGICAL ( KIND = C_BOOL ) :: new_structure
      LOGICAL ( KIND = C_BOOL ) :: get_semi_bandwidth
      LOGICAL ( KIND = C_BOOL ) :: get_norm_residual
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: definite_linear_solver
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
      TYPE ( sls_control_type ) :: sls_control
      TYPE ( mi28_control_type ) :: mi28_control
    END TYPE psls_control_type

    TYPE, BIND( C ) :: psls_time_type
      REAL ( KIND = sp ) :: total
      REAL ( KIND = sp ) :: assemble
      REAL ( KIND = sp ) :: analyse
      REAL ( KIND = sp ) :: factorize
      REAL ( KIND = sp ) :: solve
      REAL ( KIND = sp ) :: update
      REAL ( KIND = wp ) :: clock_total
      REAL ( KIND = wp ) :: clock_assemble
      REAL ( KIND = wp ) :: clock_analyse
      REAL ( KIND = wp ) :: clock_factorize
      REAL ( KIND = wp ) :: clock_solve
      REAL ( KIND = wp ) :: clock_update
    END TYPE psls_time_type

    TYPE, BIND( C ) :: psls_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      INTEGER ( KIND = C_INT ) :: analyse_status
      INTEGER ( KIND = C_INT ) :: factorize_status
      INTEGER ( KIND = C_INT ) :: solve_status
      INTEGER ( KIND = C_INT ) :: factorization_integer
      INTEGER ( KIND = C_INT ) :: factorization_real
      INTEGER ( KIND = C_INT ) :: preconditioner
      INTEGER ( KIND = C_INT ) :: semi_bandwidth
      INTEGER ( KIND = C_INT ) :: reordered_semi_bandwidth
      INTEGER ( KIND = C_INT ) :: out_of_range
      INTEGER ( KIND = C_INT ) :: duplicates
      INTEGER ( KIND = C_INT ) :: upper
      INTEGER ( KIND = C_INT ) :: missing_diagonals
      INTEGER ( KIND = C_INT ) :: semi_bandwidth_used
      INTEGER ( KIND = C_INT ) :: neg1
      INTEGER ( KIND = C_INT ) :: neg2
      LOGICAL ( KIND = C_BOOL ) :: perturbed
      REAL ( KIND = wp ) :: fill_in_ratio
      REAL ( KIND = wp ) :: norm_residual
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = C_INT ) :: mc61_info
      REAL ( KIND = wp ) :: mc61_rinfo
      TYPE ( psls_time_type ) :: time
      TYPE ( sls_inform_type ) :: sls_inform
      TYPE ( mi28_inform_type ) :: mi28_inform
    END TYPE psls_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( psls_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_psls_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%preconditioner = ccontrol%preconditioner
    fcontrol%semi_bandwidth = ccontrol%semi_bandwidth
    fcontrol%scaling = ccontrol%scaling
    fcontrol%ordering = ccontrol%ordering
    fcontrol%max_col = ccontrol%max_col
    fcontrol%icfs_vectors = ccontrol%icfs_vectors
    fcontrol%mi28_lsize = ccontrol%mi28_lsize
    fcontrol%mi28_rsize = ccontrol%mi28_rsize

    ! Reals
    fcontrol%min_diagonal = ccontrol%min_diagonal

    ! Logicals
    fcontrol%new_structure = ccontrol%new_structure
    fcontrol%get_semi_bandwidth = ccontrol%get_semi_bandwidth
    fcontrol%get_norm_residual = ccontrol%get_norm_residual
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_in( ccontrol%sls_control, fcontrol%sls_control )
    CALL copy_mi28_control_in( ccontrol%mi28_control, fcontrol%mi28_control )

    ! Strings
    DO i = 1, LEN( fcontrol%definite_linear_solver )
      IF ( ccontrol%definite_linear_solver( i ) == C_NULL_CHAR ) EXIT
      fcontrol%definite_linear_solver( i : i ) = ccontrol%definite_linear_solver( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_psls_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( psls_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i, l
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%preconditioner = fcontrol%preconditioner
    ccontrol%semi_bandwidth = fcontrol%semi_bandwidth
    ccontrol%scaling = fcontrol%scaling
    ccontrol%ordering = fcontrol%ordering
    ccontrol%max_col = fcontrol%max_col
    ccontrol%icfs_vectors = fcontrol%icfs_vectors
    ccontrol%mi28_lsize = fcontrol%mi28_lsize
    ccontrol%mi28_rsize = fcontrol%mi28_rsize

    ! Reals
    ccontrol%min_diagonal = fcontrol%min_diagonal

    ! Logicals
    ccontrol%new_structure = fcontrol%new_structure
    ccontrol%get_semi_bandwidth = fcontrol%get_semi_bandwidth
    ccontrol%get_norm_residual = fcontrol%get_norm_residual
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal

    ! Derived types
    CALL copy_sls_control_out( fcontrol%sls_control, ccontrol%sls_control )
    CALL copy_mi28_control_out( fcontrol%mi28_control, ccontrol%mi28_control )

    ! Strings
    l = LEN( fcontrol%definite_linear_solver )
    DO i = 1, l
      ccontrol%definite_linear_solver( i ) = fcontrol%definite_linear_solver( i : i )
    END DO
    ccontrol%definite_linear_solver( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime ) 
    TYPE ( psls_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_psls_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%clock_total = ctime%clock_total
    ftime%clock_assemble = ctime%clock_assemble
    ftime%clock_analyse = ctime%clock_analyse
    ftime%clock_factorize = ctime%clock_factorize
    ftime%clock_solve = ctime%clock_solve
    ftime%clock_update = ctime%clock_update
    ftime%total = ctime%total
    ftime%assemble = ctime%assemble
    ftime%analyse = ctime%analyse
    ftime%factorize = ctime%factorize
    ftime%solve = ctime%solve
    ftime%update = ctime%update
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime ) 
    TYPE ( f_psls_time_type ), INTENT( IN ) :: ftime
    TYPE ( psls_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%clock_total = ftime%clock_total
    ctime%clock_assemble = ftime%clock_assemble
    ctime%clock_analyse = ftime%clock_analyse
    ctime%clock_factorize = ftime%clock_factorize
    ctime%clock_solve = ftime%clock_solve
    ctime%clock_update = ftime%clock_update
    ctime%total = ftime%total
    ctime%assemble = ftime%assemble
    ctime%analyse = ftime%analyse
    ctime%factorize = ftime%factorize
    ctime%solve = ftime%solve
    ctime%update = ftime%update
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( psls_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_psls_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%analyse_status = cinform%analyse_status
    finform%factorize_status = cinform%factorize_status
    finform%solve_status = cinform%solve_status
    finform%factorization_integer = cinform%factorization_integer
    finform%factorization_real = cinform%factorization_real
    finform%preconditioner = cinform%preconditioner
    finform%semi_bandwidth = cinform%semi_bandwidth
    finform%reordered_semi_bandwidth = cinform%reordered_semi_bandwidth
    finform%out_of_range = cinform%out_of_range
    finform%duplicates = cinform%duplicates
    finform%upper = cinform%upper
    finform%missing_diagonals = cinform%missing_diagonals
    finform%semi_bandwidth_used = cinform%semi_bandwidth_used
    finform%neg1 = cinform%neg1
    finform%neg2 = cinform%neg2
    finform%mc61_info = cinform%mc61_info

    ! Reals
    finform%fill_in_ratio = cinform%fill_in_ratio
    finform%norm_residual = cinform%norm_residual
    finform%mc61_rinfo = cinform%mc61_rinfo

    ! Logicals
    finform%perturbed = cinform%perturbed

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
    CALL copy_sls_inform_in( cinform%sls_inform, finform%sls_inform )
    CALL copy_mi28_inform_in( cinform%mi28_inform, finform%mi28_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_psls_inform_type ), INTENT( IN ) :: finform
    TYPE ( psls_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%analyse_status = finform%analyse_status
    cinform%factorize_status = finform%factorize_status
    cinform%solve_status = finform%solve_status
    cinform%factorization_integer = finform%factorization_integer
    cinform%factorization_real = finform%factorization_real
    cinform%preconditioner = finform%preconditioner
    cinform%semi_bandwidth = finform%semi_bandwidth
    cinform%reordered_semi_bandwidth = finform%reordered_semi_bandwidth
    cinform%out_of_range = finform%out_of_range
    cinform%duplicates = finform%duplicates
    cinform%upper = finform%upper
    cinform%missing_diagonals = finform%missing_diagonals
    cinform%semi_bandwidth_used = finform%semi_bandwidth_used
    cinform%neg1 = finform%neg1
    cinform%neg2 = finform%neg2
    cinform%mc61_info = finform%mc61_info

    ! Reals
    cinform%fill_in_ratio = finform%fill_in_ratio
    cinform%norm_residual = finform%norm_residual
    cinform%mc61_rinfo = finform%mc61_rinfo

    ! Logicals
    cinform%perturbed = finform%perturbed

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
    CALL copy_sls_inform_out( finform%sls_inform, cinform%sls_inform )
    CALL copy_mi28_inform_out( finform%mi28_inform, cinform%mi28_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_PSLS_double_ciface

!  -------------------------------------
!  C interface to fortran psls_initialize
!  -------------------------------------

  SUBROUTINE psls_initialize( cdata, ccontrol, status ) BIND( C ) 
  USE GALAHAD_PSLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( psls_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_psls_full_data_type ), POINTER :: fdata
  TYPE ( f_psls_control_type ) :: fcontrol
  TYPE ( f_psls_inform_type ) :: finform
  LOGICAL :: f_indexing 

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_psls_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE psls_initialize

!  ----------------------------------------
!  C interface to fortran psls_read_specfile
!  ----------------------------------------

  SUBROUTINE psls_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_PSLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( psls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_psls_control_type ) :: fcontrol
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

  CALL f_psls_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE psls_read_specfile

!  ---------------------------------
!  C interface to fortran psls_inport
!  ---------------------------------

  SUBROUTINE psls_import( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_PSLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( psls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_psls_control_type ) :: fcontrol
  TYPE ( f_psls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN

!  import the problem data into the required PSLS structure

    CALL f_psls_import( fcontrol, fdata, status )
  ELSE
    CALL f_psls_import( fcontrol, fdata, status )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE psls_import

!  ---------------------------------------
!  C interface to fortran psls_reset_control
!  ----------------------------------------

  SUBROUTINE psls_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_PSLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( psls_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_psls_control_type ) :: fcontrol
  TYPE ( f_psls_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_PSLS_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE psls_reset_control

!  --------------------------------------
!  C interface to fortran psls_information
!  --------------------------------------

  SUBROUTINE psls_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_PSLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( psls_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_psls_full_data_type ), pointer :: fdata
  TYPE ( f_psls_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain PSLS solution information

  CALL f_psls_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE psls_information

!  ------------------------------------
!  C interface to fortran psls_terminate
!  ------------------------------------

  SUBROUTINE psls_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_PSLS_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( psls_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( psls_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_psls_full_data_type ), pointer :: fdata
  TYPE ( f_psls_control_type ) :: fcontrol
  TYPE ( f_psls_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_psls_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE psls_terminate
