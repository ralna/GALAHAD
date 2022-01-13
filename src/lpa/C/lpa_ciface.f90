! THIS VERSION: GALAHAD 4.0 - 2022-01-07 AT 16:19 GMT.

!-*-*-*-*-*-*-*-  G A L A H A D _  L P A    C   I N T E R F A C E  -*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Jaroslav Fowkes & Nick Gould

!  History -
!    originally released GALAHAD Version 4.0. January 7th 2022

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_LPA_double_ciface
    USE iso_c_binding
    USE GALAHAD_common_ciface
    USE GALAHAD_LPA_double, ONLY:                                              &
        f_lpa_control_type => LPA_control_type,                                &
        f_lpa_time_type => LPA_time_type,                                      &
        f_lpa_inform_type => LPA_inform_type,                                  &
        f_lpa_full_data_type => LPA_full_data_type,                            &
        f_lpa_initialize => LPA_initialize,                                    &
        f_lpa_read_specfile => LPA_read_specfile,                              &
        f_lpa_import => LPA_import,                                            &
        f_lpa_solve_lp => LPA_solve_lp,                                        &
        f_lpa_reset_control => LPA_reset_control,                              &
        f_lpa_information => LPA_information,                                  &
        f_lpa_terminate => LPA_terminate

!   USE GALAHAD_RPD_double_ciface, ONLY:                                       &
!       rpd_inform_type,                                                       &
!       rpd_control_type,                                                      &
!       copy_rpd_inform_in => copy_inform_in,                                  &
!       copy_rpd_inform_out => copy_inform_out,                                &
!       copy_rpd_control_in => copy_control_in,                                &
!       copy_rpd_control_out => copy_control_out

    IMPLICIT NONE

!--------------------
!   P r e c i s i o n
!--------------------

    INTEGER, PARAMETER :: wp = C_DOUBLE ! double precision
    INTEGER, PARAMETER :: sp = C_FLOAT  ! single precision

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

    TYPE, BIND( C ) :: lpa_control_type
      LOGICAL ( KIND = C_BOOL ) :: f_indexing
      INTEGER ( KIND = C_INT ) :: error
      INTEGER ( KIND = C_INT ) :: out
      INTEGER ( KIND = C_INT ) :: print_level
      INTEGER ( KIND = C_INT ) :: start_print
      INTEGER ( KIND = C_INT ) :: stop_print
      INTEGER ( KIND = C_INT ) :: maxit
      INTEGER ( KIND = C_INT ) :: max_iterative_refinements
      INTEGER ( KIND = C_INT ) :: min_real_factor_size
      INTEGER ( KIND = C_INT ) :: min_integer_factor_size
      INTEGER ( KIND = C_INT ) :: random_number_seed
      INTEGER ( KIND = C_INT ) :: sif_file_device
      INTEGER ( KIND = C_INT ) :: qplib_file_device
      REAL ( KIND = wp ) :: infinity
      REAL ( KIND = wp ) :: tol_data
      REAL ( KIND = wp ) :: feas_tol
      REAL ( KIND = wp ) :: relative_pivot_tolerance
      REAL ( KIND = wp ) :: growth_limit
      REAL ( KIND = wp ) :: zero_tolerance
      REAL ( KIND = wp ) :: change_tolerance
      REAL ( KIND = wp ) :: identical_bounds_tol
      REAL ( KIND = wp ) :: cpu_time_limit
      REAL ( KIND = wp ) :: clock_time_limit
      LOGICAL ( KIND = C_BOOL ) :: scale
      LOGICAL ( KIND = C_BOOL ) :: dual
      LOGICAL ( KIND = C_BOOL ) :: warm_start
      LOGICAL ( KIND = C_BOOL ) :: steepest_edge
      LOGICAL ( KIND = C_BOOL ) :: space_critical
      LOGICAL ( KIND = C_BOOL ) :: deallocate_error_fatal
      LOGICAL ( KIND = C_BOOL ) :: generate_sif_file
      LOGICAL ( KIND = C_BOOL ) :: generate_qplib_file
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: sif_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: qplib_file_name
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 31 ) :: prefix
    END TYPE lpa_control_type

    TYPE, BIND( C ) :: lpa_time_type
      REAL ( KIND = wp ) :: total
      REAL ( KIND = wp ) :: preprocess
      REAL ( KIND = wp ) :: clock_total
      REAL ( KIND = wp ) :: clock_preprocess
    END TYPE lpa_time_type

    TYPE, BIND( C ) :: lpa_inform_type
      INTEGER ( KIND = C_INT ) :: status
      INTEGER ( KIND = C_INT ) :: alloc_status
      CHARACTER ( KIND = C_CHAR ), DIMENSION( 81 ) :: bad_alloc
      INTEGER ( KIND = C_INT ) :: iter
      INTEGER ( KIND = C_INT ) :: la04_job
      INTEGER ( KIND = C_INT ) :: la04_job_info
      REAL ( KIND = wp ) :: obj
      REAL ( KIND = wp ) :: primal_infeasibility
      LOGICAL ( KIND = C_BOOL ) :: feasible
      REAL ( KIND = wp ), DIMENSION( 40 ) :: RINFO
      TYPE ( lpa_time_type ) :: time
!     TYPE ( rpd_inform_type ) :: rpd_inform
    END TYPE lpa_inform_type

!----------------------
!   P r o c e d u r e s
!----------------------

  CONTAINS

!  copy C control parameters to fortran

    SUBROUTINE copy_control_in( ccontrol, fcontrol, f_indexing ) 
    TYPE ( lpa_control_type ), INTENT( IN ) :: ccontrol
    TYPE ( f_lpa_control_type ), INTENT( OUT ) :: fcontrol
    LOGICAL, optional, INTENT( OUT ) :: f_indexing
    INTEGER :: i
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) f_indexing = ccontrol%f_indexing

    ! Integers
    fcontrol%error = ccontrol%error
    fcontrol%out = ccontrol%out
    fcontrol%print_level = ccontrol%print_level
    fcontrol%start_print = ccontrol%start_print
    fcontrol%stop_print = ccontrol%stop_print
    fcontrol%maxit = ccontrol%maxit
    fcontrol%max_iterative_refinements = ccontrol%max_iterative_refinements
    fcontrol%min_real_factor_size = ccontrol%min_real_factor_size
    fcontrol%min_integer_factor_size = ccontrol%min_integer_factor_size
    fcontrol%random_number_seed = ccontrol%random_number_seed
    fcontrol%sif_file_device = ccontrol%sif_file_device
    fcontrol%qplib_file_device = ccontrol%qplib_file_device

    ! Reals
    fcontrol%infinity = ccontrol%infinity
    fcontrol%tol_data = ccontrol%tol_data
    fcontrol%feas_tol = ccontrol%feas_tol
    fcontrol%relative_pivot_tolerance = ccontrol%relative_pivot_tolerance
    fcontrol%growth_limit = ccontrol%growth_limit
    fcontrol%zero_tolerance = ccontrol%zero_tolerance
    fcontrol%change_tolerance = ccontrol%change_tolerance
    fcontrol%identical_bounds_tol = ccontrol%identical_bounds_tol
    fcontrol%cpu_time_limit = ccontrol%cpu_time_limit
    fcontrol%clock_time_limit = ccontrol%clock_time_limit

    ! Logicals
    fcontrol%scale = ccontrol%scale
    fcontrol%dual = ccontrol%dual
    fcontrol%warm_start = ccontrol%warm_start
    fcontrol%steepest_edge = ccontrol%steepest_edge
    fcontrol%space_critical = ccontrol%space_critical
    fcontrol%deallocate_error_fatal = ccontrol%deallocate_error_fatal
    fcontrol%generate_sif_file = ccontrol%generate_sif_file
    fcontrol%generate_qplib_file = ccontrol%generate_qplib_file

    ! Strings
    DO i = 1, LEN( fcontrol%sif_file_name )
      IF ( ccontrol%sif_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%sif_file_name( i : i ) = ccontrol%sif_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%qplib_file_name )
      IF ( ccontrol%qplib_file_name( i ) == C_NULL_CHAR ) EXIT
      fcontrol%qplib_file_name( i : i ) = ccontrol%qplib_file_name( i )
    END DO
    DO i = 1, LEN( fcontrol%prefix )
      IF ( ccontrol%prefix( i ) == C_NULL_CHAR ) EXIT
      fcontrol%prefix( i : i ) = ccontrol%prefix( i )
    END DO
    RETURN

    END SUBROUTINE copy_control_in

!  copy fortran control parameters to C

    SUBROUTINE copy_control_out( fcontrol, ccontrol, f_indexing ) 
    TYPE ( f_lpa_control_type ), INTENT( IN ) :: fcontrol
    TYPE ( lpa_control_type ), INTENT( OUT ) :: ccontrol
    LOGICAL, OPTIONAL, INTENT( IN ) :: f_indexing
    INTEGER :: i, l
    
    ! C or Fortran sparse matrix indexing
    IF ( PRESENT( f_indexing ) ) ccontrol%f_indexing = f_indexing

    ! Integers
    ccontrol%error = fcontrol%error
    ccontrol%out = fcontrol%out
    ccontrol%print_level = fcontrol%print_level
    ccontrol%start_print = fcontrol%start_print
    ccontrol%stop_print = fcontrol%stop_print
    ccontrol%maxit = fcontrol%maxit
    ccontrol%max_iterative_refinements = fcontrol%max_iterative_refinements
    ccontrol%min_real_factor_size = fcontrol%min_real_factor_size
    ccontrol%min_integer_factor_size = fcontrol%min_integer_factor_size
    ccontrol%random_number_seed = fcontrol%random_number_seed
    ccontrol%sif_file_device = fcontrol%sif_file_device
    ccontrol%qplib_file_device = fcontrol%qplib_file_device

    ! Reals
    ccontrol%infinity = fcontrol%infinity
    ccontrol%tol_data = fcontrol%tol_data
    ccontrol%feas_tol = fcontrol%feas_tol
    ccontrol%relative_pivot_tolerance = fcontrol%relative_pivot_tolerance
    ccontrol%growth_limit = fcontrol%growth_limit
    ccontrol%zero_tolerance = fcontrol%zero_tolerance
    ccontrol%change_tolerance = fcontrol%change_tolerance
    ccontrol%identical_bounds_tol = fcontrol%identical_bounds_tol
    ccontrol%cpu_time_limit = fcontrol%cpu_time_limit
    ccontrol%clock_time_limit = fcontrol%clock_time_limit

    ! Logicals
    ccontrol%scale = fcontrol%scale
    ccontrol%dual = fcontrol%dual
    ccontrol%warm_start = fcontrol%warm_start
    ccontrol%steepest_edge = fcontrol%steepest_edge
    ccontrol%space_critical = fcontrol%space_critical
    ccontrol%deallocate_error_fatal = fcontrol%deallocate_error_fatal
    ccontrol%generate_sif_file = fcontrol%generate_sif_file
    ccontrol%generate_qplib_file = fcontrol%generate_qplib_file

    ! Strings
    l = LEN( fcontrol%sif_file_name )
    DO i = 1, l
      ccontrol%sif_file_name( i ) = fcontrol%sif_file_name( i : i )
    END DO
    ccontrol%sif_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%qplib_file_name )
    DO i = 1, l
      ccontrol%qplib_file_name( i ) = fcontrol%qplib_file_name( i : i )
    END DO
    ccontrol%qplib_file_name( l + 1 ) = C_NULL_CHAR
    l = LEN( fcontrol%prefix )
    DO i = 1, l
      ccontrol%prefix( i ) = fcontrol%prefix( i : i )
    END DO
    ccontrol%prefix( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_control_out

!  copy C time parameters to fortran

    SUBROUTINE copy_time_in( ctime, ftime ) 
    TYPE ( lpa_time_type ), INTENT( IN ) :: ctime
    TYPE ( f_lpa_time_type ), INTENT( OUT ) :: ftime

    ! Reals
    ftime%total = ctime%total
    ftime%preprocess = ctime%preprocess
    ftime%clock_total = ctime%clock_total
    ftime%clock_preprocess = ctime%clock_preprocess
    RETURN

    END SUBROUTINE copy_time_in

!  copy fortran time parameters to C

    SUBROUTINE copy_time_out( ftime, ctime ) 
    TYPE ( f_lpa_time_type ), INTENT( IN ) :: ftime
    TYPE ( lpa_time_type ), INTENT( OUT ) :: ctime

    ! Reals
    ctime%total = ftime%total
    ctime%preprocess = ftime%preprocess
    ctime%clock_total = ftime%clock_total
    ctime%clock_preprocess = ftime%clock_preprocess
    RETURN

    END SUBROUTINE copy_time_out

!  copy C inform parameters to fortran

    SUBROUTINE copy_inform_in( cinform, finform ) 
    TYPE ( lpa_inform_type ), INTENT( IN ) :: cinform
    TYPE ( f_lpa_inform_type ), INTENT( OUT ) :: finform
    INTEGER :: i

    ! Integers
    finform%status = cinform%status
    finform%alloc_status = cinform%alloc_status
    finform%iter = cinform%iter
    finform%la04_job = cinform%la04_job
    finform%la04_job_info = cinform%la04_job_info

    ! Reals
    finform%obj = cinform%obj
    finform%primal_infeasibility = cinform%primal_infeasibility
    finform%RINFO = cinform%RINFO

    ! Logicals
    finform%feasible = cinform%feasible

    ! Derived types
    CALL copy_time_in( cinform%time, finform%time )
!   CALL copy_rpd_inform_in( cinform%rpd_inform, finform%rpd_inform )

    ! Strings
    DO i = 1, LEN( finform%bad_alloc )
      IF ( cinform%bad_alloc( i ) == C_NULL_CHAR ) EXIT
      finform%bad_alloc( i : i ) = cinform%bad_alloc( i )
    END DO
    RETURN

    END SUBROUTINE copy_inform_in

!  copy fortran inform parameters to C

    SUBROUTINE copy_inform_out( finform, cinform ) 
    TYPE ( f_lpa_inform_type ), INTENT( IN ) :: finform
    TYPE ( lpa_inform_type ), INTENT( OUT ) :: cinform
    INTEGER :: i, l

    ! Integers
    cinform%status = finform%status
    cinform%alloc_status = finform%alloc_status
    cinform%iter = finform%iter
    cinform%la04_job = finform%la04_job
    cinform%la04_job_info = finform%la04_job_info

    ! Reals
    cinform%obj = finform%obj
    cinform%primal_infeasibility = finform%primal_infeasibility
    cinform%RINFO = finform%RINFO

    ! Logicals
    cinform%feasible = finform%feasible

    ! Derived types
    CALL copy_time_out( finform%time, cinform%time )
!   CALL copy_rpd_inform_out( finform%rpd_inform, cinform%rpd_inform )

    ! Strings
    l = LEN( finform%bad_alloc )
    DO i = 1, l
      cinform%bad_alloc( i ) = finform%bad_alloc( i : i )
    END DO
    cinform%bad_alloc( l + 1 ) = C_NULL_CHAR
    RETURN

    END SUBROUTINE copy_inform_out

  END MODULE GALAHAD_LPA_double_ciface

!  -------------------------------------
!  C interface to fortran lpa_initialize
!  -------------------------------------

  SUBROUTINE lpa_initialize( cdata, ccontrol, status ) BIND( C ) 
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( C_PTR ), INTENT( OUT ) :: cdata ! data is a black-box
  TYPE ( lpa_control_type ), INTENT( OUT ) :: ccontrol

!  local variables

  TYPE ( f_lpa_full_data_type ), POINTER :: fdata
  TYPE ( f_lpa_control_type ) :: fcontrol
  TYPE ( f_lpa_inform_type ) :: finform
  LOGICAL :: f_indexing 

!  allocate fdata

  ALLOCATE( fdata ); cdata = C_LOC( fdata )

!  initialize required fortran types

  CALL f_lpa_initialize( fdata, fcontrol, finform )
  status = finform%status

!  C sparse matrix indexing by default

  f_indexing = .FALSE.
  fdata%f_indexing = f_indexing

!  copy control out 

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE lpa_initialize

!  ----------------------------------------
!  C interface to fortran lpa_read_specfile
!  ----------------------------------------

  SUBROUTINE lpa_read_specfile( ccontrol, cspecfile ) BIND( C )
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( lpa_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: cspecfile

!  local variables

  TYPE ( f_lpa_control_type ) :: fcontrol
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

  CALL f_lpa_read_specfile( fcontrol, device )

!  close specfile

  CLOSE( device )

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE lpa_read_specfile

!  ---------------------------------
!  C interface to fortran lpa_inport
!  ---------------------------------

  SUBROUTINE lpa_import( ccontrol, cdata, status, n, m,                        &
                         catype, ane, arow, acol, aptr ) BIND( C )
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( lpa_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, ane
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: arow
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( ane ), OPTIONAL :: acol
  INTEGER ( KIND = C_INT ), INTENT( IN ), DIMENSION( m + 1 ), OPTIONAL :: aptr
  TYPE ( C_PTR ), INTENT( IN ), VALUE :: catype

!  local variables

  CHARACTER ( KIND = C_CHAR, LEN = opt_strlen( catype ) ) :: fatype
  TYPE ( f_lpa_control_type ) :: fcontrol
  TYPE ( f_lpa_full_data_type ), POINTER :: fdata
  INTEGER, DIMENSION( : ), ALLOCATABLE :: arow_find, acol_find, aptr_find
  LOGICAL :: f_indexing

!  copy control and inform in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  convert C string to Fortran string

  fatype = cstr_to_fchar( catype )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  handle C sparse matrix indexing

  IF ( .NOT. f_indexing ) THEN
    IF ( PRESENT( arow ) ) THEN
      ALLOCATE( arow_find( ane ) )
      arow_find = arow + 1
    END IF
    IF ( PRESENT( acol ) ) THEN
      ALLOCATE( acol_find( ane ) )
      acol_find = acol + 1
    END IF
    IF ( PRESENT( aptr ) ) THEN
      ALLOCATE( aptr_find( m + 1 ) )
      aptr_find = aptr + 1
    END IF

!  import the problem data into the required LPA structure

    CALL f_lpa_import( fcontrol, fdata, status, n, m,                          &
                       fatype, ane, arow_find, acol_find, aptr_find )

    IF ( ALLOCATED( arow_find ) ) DEALLOCATE( arow_find )
    IF ( ALLOCATED( acol_find ) ) DEALLOCATE( acol_find )
    IF ( ALLOCATED( aptr_find ) ) DEALLOCATE( aptr_find )
  ELSE
    CALL f_lpa_import( fcontrol, fdata, status, n, m,                          &
                       fatype, ane, arow, acol, aptr )
  END IF

!  copy control out

  CALL copy_control_out( fcontrol, ccontrol, f_indexing )
  RETURN

  END SUBROUTINE lpa_import

!  ----------------------------------------
!  C interface to fortran lpa_reset_control
!  ----------------------------------------

  SUBROUTINE lpa_reset_control( ccontrol, cdata, status ) BIND( C )
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status
  TYPE ( lpa_control_type ), INTENT( INOUT ) :: ccontrol
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_lpa_control_type ) :: fcontrol
  TYPE ( f_lpa_full_data_type ), POINTER :: fdata
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  is fortran-style 1-based indexing used?

  fdata%f_indexing = f_indexing

!  import the control parameters into the required structure

  CALL f_lpa_reset_control( fcontrol, fdata, status )
  RETURN

  END SUBROUTINE lpa_reset_control

!  ------------------------------------
!  C interface to fortran lpa_solve_lpa
!  ------------------------------------

  SUBROUTINE lpa_solve_lp( cdata, status, n, m, g, f, ane, aval,               &
                           cl, cu, xl, xu, x, c, y, z, xstat, cstat ) BIND( C )
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  INTEGER ( KIND = C_INT ), INTENT( IN ), VALUE :: n, m, ane
  INTEGER ( KIND = C_INT ), INTENT( INOUT ) :: status
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ane ) :: aval
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: g
  REAL ( KIND = wp ), INTENT( IN ), VALUE :: f
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: cl, cu
  REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: xl, xu
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: x, z
  REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: y
  REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( m ) :: c
  INTEGER ( KIND = C_INT ), INTENT( OUT ), DIMENSION( n ) :: xstat
  INTEGER ( KIND = C_INT ), INTENT( OUT ), DIMENSION( m ) :: cstat
  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata

!  local variables

  TYPE ( f_lpa_full_data_type ), POINTER :: fdata

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  solve the qp

  CALL f_lpa_solve_lp( fdata, status, g, f, aval, cl, cu, xl, xu, x, c, y, z,  &
                       xstat, cstat )
  RETURN

  END SUBROUTINE lpa_solve_lp

!  --------------------------------------
!  C interface to fortran lpa_information
!  --------------------------------------

  SUBROUTINE lpa_information( cdata, cinform, status ) BIND( C ) 
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( lpa_inform_type ), INTENT( INOUT ) :: cinform
  INTEGER ( KIND = C_INT ), INTENT( OUT ) :: status

!  local variables

  TYPE ( f_lpa_full_data_type ), pointer :: fdata
  TYPE ( f_lpa_inform_type ) :: finform

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  obtain LPA solution information

  CALL f_lpa_information( fdata, finform, status )

!  copy inform out

  CALL copy_inform_out( finform, cinform )
  RETURN

  END SUBROUTINE lpa_information

!  ------------------------------------
!  C interface to fortran lpa_terminate
!  ------------------------------------

  SUBROUTINE lpa_terminate( cdata, ccontrol, cinform ) BIND( C ) 
  USE GALAHAD_LPA_double_ciface
  IMPLICIT NONE

!  dummy arguments

  TYPE ( C_PTR ), INTENT( INOUT ) :: cdata
  TYPE ( lpa_control_type ), INTENT( IN ) :: ccontrol
  TYPE ( lpa_inform_type ), INTENT( INOUT ) :: cinform

!  local variables

  TYPE ( f_lpa_full_data_type ), pointer :: fdata
  TYPE ( f_lpa_control_type ) :: fcontrol
  TYPE ( f_lpa_inform_type ) :: finform
  LOGICAL :: f_indexing

!  copy control in

  CALL copy_control_in( ccontrol, fcontrol, f_indexing )

!  copy inform in

  CALL copy_inform_in( cinform, finform )

!  associate data pointer

  CALL C_F_POINTER( cdata, fdata )

!  deallocate workspace

  CALL f_lpa_terminate( fdata, fcontrol, finform )

!  copy inform out

  CALL copy_inform_out( finform, cinform )

!  deallocate data

  DEALLOCATE( fdata ); cdata = C_NULL_PTR 
  RETURN

  END SUBROUTINE lpa_terminate
