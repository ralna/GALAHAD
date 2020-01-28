! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ F D C    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.0.August 14th 2006

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_FDC_double

!     -----------------------------------------------------
!     |                                                   |
!     | Check if A x = c is consistent and if so which    |
!     | if any of the constraints are linearly dependeent |
!     |                                                   |
!     -----------------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_SLS_double
      USE GALAHAD_ULS_double
      USE GALAHAD_ROOTS_double
      USE GALAHAD_SPECFILE_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: FDC_initialize, FDC_read_specfile, FDC_find_dependent,         &
                FDC_terminate, SMT_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: FDC_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  initial estimate of integer workspace for sls (obsolete)

        INTEGER :: indmin = 1000

!  initial estimate of real workspace for sls (obsolete)

        INTEGER :: valmin = 1000

!  the relative pivot tolerance (obsolete)

        REAL ( KIND = wp ) :: pivot_tol = half

!  the absolute pivot tolerance used (obsolete)

!       REAL ( KIND = wp ) :: zero_pivot = epsmch ** 0.75_wp
        REAL ( KIND = wp ) :: zero_pivot = epsmch

!  the largest permitted residual

!    REAL ( KIND = wp ) :: max_infeas = epsmch ** 0.33_wp
     REAL ( KIND = wp ) :: max_infeas = epsmch

!  chose whether SLS or ULS is used to determine dependencies

        LOGICAL :: use_sls = .FALSE.

!   should the rows of A be scaled to have unit infinity norm or
!   should no scaling be applied

        LOGICAL :: scale = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  symmetric (indefinite) linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "sils" // REPEAT( ' ', 26 )

!  unsymmetric linear equation solver

        CHARACTER ( LEN = 30 ) :: unsymmetric_linear_solver =                  &
           "gls" // REPEAT( ' ', 27 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for ULS

        TYPE ( ULS_control_type ) :: ULS_control
      END TYPE

      TYPE, PUBLIC :: FDC_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ) :: factorize = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0
      END TYPE

      TYPE, PUBLIC :: FDC_inform_type

!  return status. See FDC_find_dependent for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_real = - 1

!  the smallest pivot which was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  timings (see above)

        TYPE ( FDC_time_type ) :: time

!  SLS inform type

        TYPE ( SLS_inform_type ) :: SLS_inform

!  ULS inform type

        TYPE ( ULS_inform_type ) :: ULS_inform
      END TYPE

      TYPE, PUBLIC :: FDC_data_type
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: P
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEP
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCALE
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
        TYPE ( SMT_type ) :: A, K
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( ULS_data_type ) :: ULS_data
        TYPE ( FDC_control_type ) :: control
      END TYPE

   CONTAINS

!-*-*-*-*-*-   F D C _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE FDC_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for FDC. This routine should be called before
!  FDC_find_dependent
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  see preamble
!  iform    see preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( FDC_data_type ), INTENT( INOUT ) :: data
      TYPE ( FDC_control_type ), INTENT( INOUT ) :: control
      TYPE ( FDC_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  initalize SLS components

      CALL SLS_INITIALIZE( control%symmetric_linear_solver,                    &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform )
      control%SLS_control%prefix = '" - SLS:"                    '

!  initalize ULS components

      CALL ULS_INITIALIZE( control%unsymmetric_linear_solver,                  &
                           data%ULS_data, control%ULS_control,                 &
                           inform%ULS_inform )
      control%ULS_control%prefix = '" - ULS:"                    '

!  Set outstanding control parameters

!  integer parameter

      control%SLS_control%pivot_control = 1

!  real parameters

      control%max_infeas = epsmch ** 0.33
      control%zero_pivot = epsmch ** 0.75
      control%SLS_control%relative_pivot_tolerance = half

      RETURN

!  End of FDC_initialize

      END SUBROUTINE FDC_initialize

!-*-*-*-*-   F D C _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE FDC_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by FDC_initialize could (roughly)
!  have been set as:

! BEGIN FDC SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  pivot-tolerance-used-for-dependencies             0.5
!  maximum-permitted-infeasibility                   1.0D-5
!  use-sls                                           T
!  scale-A                                           F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  sils
!  unsymmetric-linear-equation-solver                gls
!  output-line-prefix                                ""
! END FDC SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( FDC_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: indmin = print_level + 1                ! obsolete
      INTEGER, PARAMETER :: valmin = indmin + 1                     ! obsolete
      INTEGER, PARAMETER :: max_infeas = valmin + 1
      INTEGER, PARAMETER :: pivot_tol = max_infeas + 1              ! obsolete
      INTEGER, PARAMETER :: zero_pivot = pivot_tol + 1              ! obsolete
      INTEGER, PARAMETER :: use_sls = zero_pivot + 1
      INTEGER, PARAMETER :: scale = use_sls + 1
      INTEGER, PARAMETER :: space_critical = scale + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: symmetric_linear_solver = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: unsymmetric_linear_solver =                        &
                              symmetric_linear_solver + 1
      INTEGER, PARAMETER :: prefix = unsymmetric_linear_solver + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'FDC'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec( : )%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( indmin )%keyword = 'initial-integer-workspace'
      spec( valmin )%keyword = 'initial-real-workspace'

!  Real key-words

      spec( pivot_tol )%keyword = 'pivot-tolerance-used-for-dependencies'
      spec( zero_pivot )%keyword = 'zero-pivot-tolerance'
      spec( max_infeas )%keyword = 'maximum-permitted-infeasibility'

!  Logical key-words

      spec( use_sls )%keyword = 'use-sls'
      spec( scale )%keyword = 'scale-A'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
      spec( unsymmetric_linear_solver )%keyword =                              &
       'unsymmetric-linear-equation-solver'
      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( error ),                               &
                                  control%error,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ),                                 &
                                  control%out,                                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( indmin ),                              &
                                  control%indmin,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( valmin ),                              &
                                  control%valmin,                              &
                                  control%error )

!  Set real values


      CALL SPECFILE_assign_value( spec( max_infeas ),                          &
                                  control%max_infeas,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( pivot_tol ),                           &
                                  control%pivot_tol,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( zero_pivot ),                          &
                                  control%zero_pivot,                          &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( use_sls ),                             &
                                  control%use_sls,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( scale ),                               &
                                  control%scale,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),             &
                                  control%symmetric_linear_solver,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( unsymmetric_linear_solver ),           &
                                  control%unsymmetric_linear_solver,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read the specfile for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

!  Read the specfile for ULS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL ULS_read_specfile( control%ULS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-ULS' )
      ELSE
        CALL ULS_read_specfile( control%ULS_control, device )
      END IF

      RETURN

      END SUBROUTINE FDC_read_specfile

!-*-*-*-   F D C _ F I N D _ D E P E N D E N T  S U B R O U T I N E   -*-*-*-

      SUBROUTINE FDC_find_dependent( n, m, A_val, A_col, A_ptr, C,             &
                                     n_depen, C_depen,                         &
                                     data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Determine which, if any, of the equality constraints A x = c is dependent
!  by forming LBL^T factors of ( I A^T )
!                              ( A  0  )
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!   n is an INTEGER variable, which must be set by the user to the
!    number of columns of A. RESTRICTION: n >= 1
!
!   m is an INTEGER variable, which must be set by the user to the
!    number of rows of A. RESTRICTION: m >= 0
!
!   A_val is a REAL array of length A_ptr( m + 1 ) - 1 that must be set by
!    the user to the values of the components of A, stored by row, that is
!    entries for row i must directly preceed those in row i+1 (the order
!    within each row is unimportant).
!
!   A_col is an INTEGER array of length A_ptr( m + 1 ) - 1 that must be set by
!    the user to the column indices of the components of A corresponding
!    to the values in A_val
!
!   A_ptr is an INTEGER array of length that must be set by the user to point
!    to the position in A_val and A_col for the start of each row, as well
!    as one position past the last entry.
!
!   C is a REAL array of length m, which is used to store the values of c
!
!   n_depen is an INTEGER variable that gives the number of rows of A that
!    are deemed to be linearly dependent
!
!   C_depen is a INTEGER pointer array that will have been allocated to
!    be of length n_depen, and will contain the indices of the rows of A
!    that are deemed to be linearly dependent
!
!   data is a structure of type FDC_data_type that need not be set by the user.
!
!   control is a structure of type FDC_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to FDC_initialize. See FDC_initialize
!   for details
!
!   inform is a structure of type FDC_inform_type that provides
!    information on exit from FDC_find_dependent. The component status
!    has possible values:
!
!     0 Normal termination with a prediction of how many (and which)
!       constraints are dependent
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 One of the restrictions n >= 0 and m >= 0 has been violated
!
!    -5 The constraints appear to be inconsistent
!
!    -9 The ordering failed. The return status from the factorization
!      package is given in inform%factorization_status
!
!   -10 The factorization failed. The return status from the factorization
!      package is given in inform%factorization_status
!
!  On exit from FDC_find_dependent, other components of inform give the
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation
!     factorization_status = The return status from the factorization
!     factorization_integer = The total integer workspace required by the
!              factorization.
!     factorization_real = The total real workspace required by the
!              factorization.
!     nfacts = The total number of factorizations performed.
!     factorization_status = the return status from the matrix factorization
!              package.
!     non_negligible_pivot = the smallest pivot which was not judged to be
!       zero when detecting linearly dependent constraints
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package.
!     time%analyse = the time spent analysing the required matrices prior to
!                  factorization.
!     time%factorize = the time spent factorizing the required matrices.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: n_depen
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C

      INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_depen
      TYPE ( FDC_data_type ), INTENT( INOUT ) :: data
      TYPE ( FDC_control_type ), INTENT( IN ) :: control
      TYPE ( FDC_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now
      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering FDC_find_dependent ' )" ) prefix

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )
      IF ( control%use_sls ) THEN
        CALL FDC_find_dependent_sls( n, m, A_val, A_col, A_ptr, C,             &
                                     n_depen, C_depen, data, control, inform )
      ELSE
        CALL FDC_find_dependent_uls( n, m, A_val, A_col, A_ptr, C,             &
                                     n_depen, C_depen, data, control, inform )
      END IF
      CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving FDC_find_dependent ' )" ) prefix
      RETURN

!  End of FDC_find_dependent

      END SUBROUTINE FDC_find_dependent

!-*-*   F D C _ F I N D _ D E P E N D E N T _ S L S  S U B R O U T I N E   *-*-

      SUBROUTINE FDC_find_dependent_sls( n, m, A_val, A_col, A_ptr, C,         &
                                         n_depen, C_depen,                     &
                                         data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Determine which, if any, of the equality constraints A x = c is dependent
!  by forming LBL^T factors of ( I A^T )
!                              ( A  0  )
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments: see FDC_find_dependent
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: n_depen
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C

      INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_depen
      TYPE ( FDC_data_type ), INTENT( INOUT ) :: data
      TYPE ( FDC_control_type ), INTENT( IN ) :: control
      TYPE ( FDC_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: A_ne, i, ii, l, nroots, out, pmax, pmin
      REAL :: time_now, time_record
      REAL ( KIND = wp ) :: clock_now, clock_record
      REAL ( KIND = wp ) :: root1, root2, dmax, dmin, dmax_allowed
      REAL ( KIND = wp ) :: big, res, res_max, rmax, rmin

      LOGICAL ::  twobytwo
      CHARACTER ( LEN = 80 ) :: array_name

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

! initialize solver-specific data

      data%control = control
      CALL SLS_initialize_solver( control%symmetric_linear_solver,             &
                                  data%SLS_data, inform%SLS_inform )
      data%control%SLS_control%relative_pivot_tolerance = control%pivot_tol

      out = data%control%out
      IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,           &
       "( /, A, 5( ' -' ), ' SLS test for rank defficiency', 5( ' - ' ) )" )   &
           prefix

!  Check that the problem makes sense

      IF ( n < 0 .OR. m < 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; RETURN ; END IF

!  Check the case where n is zero

      IF ( n == 0 ) THEN
        IF ( MAXVAL( C ) <= data%control%max_infeas ) THEN
          DO i = 1, m - 1
            C_depen( i ) = i
          END DO
          n_depen = m - 1
          inform%status = GALAHAD_ok
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        RETURN
      END IF

!  Analyse the sparsity pattern. Set the dimensions of K

      A_ne = A_ptr( m + 1 ) - 1
      data%K%n = n + m ; data%K%ne = A_ne + n
      CALL STRING_put( data%K%type, 'COORDINATE', inform%status )

!  Allocate the arrays for the analysis phase

      array_name = 'fdc: data%K%row'
      CALL SPACE_resize_array( data%K%ne, data%K%row, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%K%col'
      CALL SPACE_resize_array( data%K%ne, data%K%col, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%K%val'
      CALL SPACE_resize_array( data%K%ne, data%K%val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%SCALE'
      CALL SPACE_resize_array( m, data%SCALE, inform%status,                   &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  Put A into K

      DO i = 1, m
        ii = n + i
        IF ( data%control%scale ) THEN
          IF ( A_ptr( i + 1 ) - 1 >= A_ptr( i ) ) THEN
            data%SCALE( i ) = MAX( one,                                        &
              MAXVAL( ABS( A_val( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) ) ) )
          ELSE
            data%SCALE( i ) = one
          END IF
        ELSE
          data%SCALE( i ) = one
        END IF
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          data%K%row( l ) = ii ; data%K%col( l ) = A_col( l )
          data%K%val( l ) = A_val( l ) / data%SCALE( i )
        END DO
      END DO
!     dmax = MAXVAL( ABS( A_val( : A_ne ) ) )
!     dmax = MAX( dmax, one )
!     dmax = one
      dmax = ten ** ( - 2 )

!  Put the diagonal into K

      DO i = 1, n
        data%K%row( A_ne + i ) = i ; data%K%col( A_ne + i ) = i
      END DO
      data%K%val( A_ne + 1 : A_ne + n ) = dmax

!     write(78,*) data%K%n, data%K%ne
!     DO i = 1,  data%K%ne
!       write(78,"( 2I8, ES22.14 )" ) data%K%row( i ), data%K%col( i ),        &
!        data%K%val( i )
!     END DO

!  Analyse the sparsity pattern of the matrix

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SLS_analyse( data%K, data%SLS_data, data%control%SLS_control,       &
                        inform%SLS_inform )
      CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now )
      inform%time%analyse = inform%time%analyse + time_now - time_record
      inform%time%clock_analyse =                                              &
        inform%time%clock_analyse + time_now - time_record

!  Record the storage requested

      inform%factorization_integer = inform%SLS_inform%integer_size_desirable
      inform%factorization_real = inform%SLS_inform%real_size_desirable

!  Check for error returns

      inform%factorization_status = inform%SLS_inform%status
      IF ( inform%SLS_inform%status < 0 ) THEN
        IF ( data%control%error > 0 .AND. data%control%print_level >= 1 )      &
           WRITE( data%control%error,                                          &
           "( A, '   **  Error return ', I6, ' from SLS_analyse' )")           &
          prefix, inform%SLS_inform%status
        inform%status = GALAHAD_error_analysis ; RETURN
      END IF

      IF ( out > 0 .AND. data%control%print_level >= 2 ) WRITE( out,           &
          "( A, ' real/integer space required for factors ', 2I10 )" )         &
          prefix, inform%SLS_inform%real_size_necessary,                       &
          inform%SLS_inform%integer_size_necessary

      IF ( out > 0 .AND. data%control%print_level >= 2 ) WRITE( out,           &
     &  "( A, ' ** analysis time = ', F10.2 ) " ) prefix, inform%time%analyse

!  Factorize the matrix

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SLS_factorize( data%K, data%SLS_data, data%control%SLS_control,     &
                          inform%SLS_inform )
      CALL CPU_TIME( time_now ); CALL CLOCK_time( clock_now )
      inform%time%factorize = inform%time%factorize + time_now - time_record
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + time_now - time_record

      IF ( out > 0 .AND. data%control%print_level >= 2 ) WRITE( out,           &
     &  "( A, ' ** factorize time = ', F10.2 ) " ) prefix, inform%time%factorize

!  Record the storage required

      inform%factorization_integer = inform%SLS_inform%integer_size_desirable
      inform%factorization_real = inform%SLS_inform%real_size_desirable

!  Test that the factorization succeeded

      inform%factorization_status = inform%SLS_inform%status
      IF ( inform%SLS_inform%status < 0 ) THEN
        IF ( data%control%error > 0 .AND. data%control%print_level >= 1 )      &
           WRITE( data%control%error,                                          &
            "( A, '   **  Error return ', I6, ' from SLS_factorize' )")        &
              prefix, inform%SLS_inform%status
        inform%status = GALAHAD_error_factorization
        RETURN
      END IF

!  Record warning conditions

      IF ( out > 0 .AND. data%control%print_level >= 1 .AND.                   &
           inform%SLS_inform%rank < data%K%n ) WRITE( out,                     &
            "( /, A, ' ** Warning - matrix has ', I0, ' zero eigenvalues' )" ) &
            prefix, data%K%n - inform%SLS_inform%rank

!  Allocate the arrays for the rank detetmination phase

      array_name = 'fdc: data%P'
      CALL SPACE_resize_array( data%K%n, data%P, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%SOL'
      CALL SPACE_resize_array( data%K%n, data%SOL, inform%status,              &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%D'
      CALL SPACE_resize_array( 2, data%K%n, data%D, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  Determine the block diagonal part of the factors and the pivot order

      CALL SLS_enquire( data%SLS_data, inform%SLS_inform, PIVOTS = data%P,     &
                        D = data%D )

!  Compute the smallest and largest eigenvalues of the block diagonal factor

      n_depen = 0 ; twobytwo = .FALSE.
      dmax = zero ; dmin = HUGE( one )
      dmax_allowed = zero
      big = one / MAX( data%control%zero_pivot, epsmch )

!  Loop over the diagonal blocks

      DO i = 1, inform%SLS_inform%rank
        IF ( twobytwo ) THEN
          twobytwo = .FALSE.
          CYCLE
        END IF
        IF ( i < inform%SLS_inform%rank ) THEN

!  A 2x2 block

          IF ( data%P( i ) < 0 ) THEN
            twobytwo = .TRUE.

            CALL ROOTS_quadratic( data%D( 1, i ) * data%D( 1, i + 1 )          &
                                  - data%D( 2, i ) ** 2,                       &
                                  - data%D( 1, i ) - data%D( 1, i + 1 ),       &
                                  one, epsmch, nroots, root1, root2, .FALSE. )
            rmax = MAX( ABS( root1 ), ABS( root2 ) )
            rmin = MIN( ABS( root1 ), ABS( root2 ) )
            dmax = MAX( rmax, dmax ) ; dmin = MIN( rmin, dmin )

            pmax = MAX( ABS( data%P( i ) ), ABS( data%P( i + 1 ) ) )
            pmin = MIN( ABS( data%P( i ) ), ABS( data%P( i + 1 ) ) )

            IF ( rmax >= big ) THEN
              n_depen = n_depen + 1
              IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
                WRITE(  out, "( A, ' 2x2 block ', 2i7, ' eval = ', ES12.4 )" ) &
                  prefix, pmax - n, pmin - n, one / rmax
              END IF
            ELSE IF ( rmin == zero ) THEN
              n_depen = n_depen + 1
              IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
                WRITE( out, "( A, ' 2x2 block ', 2i7, ' eval = infinity' )" )  &
                  prefix, pmax - n, pmin - n
              END IF
            END IF

            IF ( rmin >= big ) THEN
              n_depen = n_depen + 1
              IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
                WRITE( out, "( A, ' 2x2 block ', 2i7, ' eval = ', ES12.4 )" )  &
                   prefix, pmin - n, pmax - n, one / rmin
              END IF
            ELSE IF ( rmax == zero ) THEN
              n_depen = n_depen + 1
              IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
                WRITE( out, "( A, ' 2x2 block ', 2i7, ' eval = infinity' )" )  &
                   prefix, pmin - n, pmax - n
              END IF
            END IF
            IF ( rmax < big .AND. rmin > zero )                                &
              dmax_allowed = MAX( rmax, dmax_allowed )

!  A 1x1 block

          ELSE
            rmax = ABS( data%D( 1, i ) ) ; rmin = rmax
            dmax = MAX( rmax, dmax ) ; dmin = MIN( rmin, dmin )
            IF ( rmax >= big ) THEN
              n_depen = n_depen + 1
              IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
                WRITE( out, "( A, ' 1x1 block ', i7, 8x, 'eval = ', ES12.4 )" )&
                  prefix, data%P( i ) - n,  one / rmax
              END IF
            ELSE IF ( rmax == zero ) THEN
              n_depen = n_depen + 1
              IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
                WRITE( out, "( A, ' 1x1 block ', i7, 8x, 'eval = infinity' )" )&
                  prefix, data%P( i ) - n
              END IF
            END IF
            IF ( rmax < big .AND. rmin > zero )                                &
              dmax_allowed = MAX( rmax, dmax_allowed )
          END IF
        ELSE

!  The final 1x1 block

          rmax = ABS( data%D( 1, i ) ) ; rmin = rmax
          dmax = MAX( rmax, dmax ) ; dmin = MIN( rmin, dmin )
          IF ( rmax >= big ) THEN
            n_depen = n_depen + 1
            IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
              WRITE( out, "( A, ' 1x1 block ', i7, 7x, ' eval = ', ES12.4 )" ) &
                prefix, data%P( i ) - n,  one / rmax
            END IF
          ELSE IF ( rmax == zero ) THEN
            n_depen = n_depen + 1
            IF ( out > 0 .AND. data%control%print_level >= 3 ) THEN
              WRITE( out, "( A, ' 1x1 block ', i7, 7x, ' eval = infinity ' )" )&
                prefix, data%P( i ) - n
            END IF
          END IF
          IF ( rmax < big .AND. rmin > zero )                                  &
            dmax_allowed = MAX( rmax, dmax_allowed )
        END IF
      END DO

!  Any null blocks

      IF ( data%K%n > inform%SLS_inform%rank ) THEN
         n_depen = n_depen + data%K%n - inform%SLS_inform%rank
         dmax = HUGE( one )
      END IF

      DO i = inform%SLS_inform%rank + 1, data%K%n
        IF ( out > 0 .AND. data%control%print_level >= 3 )                     &
         WRITE( out, "( A, ' 1x1 block ', i7, 7x, ' eval = ', ES12.4 )" )      &
            prefix, data%P( i ) - n, zero
      END DO

      IF ( out > 0 .AND. data%control%print_level >= 1 ) THEN
        IF ( dmin == zero .OR. dmax == zero ) THEN
          WRITE( out, "( A, ' 1/ smallest,largest block eigenvalues =',        &
         &               2ES12.4)" ) prefix, dmin, dmax
        ELSE
          WRITE( out, "( A, ' smallest,largest block eigenvalues =',           &
         &               2ES12.4 )" ) prefix, one / dmax, one / dmin
          WRITE( out, "( A, ' smallest non-negligible eigenvalue =',           &
         &               ES12.4 )" ) prefix, one / dmax_allowed
        END IF
        WRITE( out, "( A, I7, ' constraint', A, ' appear', A, ' to be ',       &
       &   'dependent ' )" ) prefix, n_depen,                                  &
          TRIM( STRING_pleural( n_depen ) ),                                   &
          TRIM( STRING_verb_pleural( n_depen ) )
      END IF

      IF ( dmax_allowed > zero )                                               &
        inform%non_negligible_pivot = one / dmax_allowed

!  Mark any dependent constraints for removal

      IF ( n_depen > 0 ) THEN

!  Allocate arrays to indicate which constraints have been freed

        array_name = 'fdc: data%C_depen'
        CALL SPACE_resize_array( n_depen, C_depen, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = data%control%deallocate_error_fatal,   &
               exact_size = data%control%space_critical,                       &
               bad_alloc = inform%bad_alloc, out = data%control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  A second loop over the diagonal blocks

        n_depen = 0 ; twobytwo = .FALSE.
        DO i = 1, inform%SLS_inform%rank
          IF ( twobytwo ) THEN
            twobytwo = .FALSE.
            CYCLE
          END IF
          IF ( i < inform%SLS_inform%rank ) THEN

!  A 2x2 block

            IF ( data%P( i ) < 0 ) THEN
              twobytwo = .TRUE.

              CALL ROOTS_quadratic( data%D( 1, i ) * data%D( 1, i + 1 )        &
                                    - data%D( 2, i ) ** 2,                     &
                                    - data%D( 1, i ) - data%D( 1, i + 1 ),     &
                                    one, epsmch, nroots, root1, root2, .FALSE. )

              IF ( ABS( root2 ) >= big .OR.                                    &
                   root1 == zero .OR. root2 == zero ) THEN
                IF ( ABS( root1 ) >= big .OR.                                  &
                     ( root1 == zero .AND. root2 == zero ) ) THEN
                  n_depen = n_depen + 1
                  C_depen( n_depen )                                           &
                    = MIN( ABS( data%P( i ) ), ABS( data%P( i + 1 ) ) ) - n
                  data%D( 1, i ) = zero ;  data%D( 2, i ) = zero
                  data%D( 1, i + 1 ) = zero
                END IF
                n_depen = n_depen + 1
                C_depen( n_depen ) = MAX( ABS( data%P( i ) ),                  &
                                          ABS( data%P( i + 1 ) ) ) - n
              END IF

            ELSE

!  A 1x1 block

              IF ( ABS( data%D( 1, i ) ) >= big .OR.                           &
                        data%D( 1, i ) == zero ) THEN
                n_depen = n_depen + 1
                C_depen( n_depen ) = data%P( i ) - n
                data%D( 1, i ) = zero
              END IF
            END IF
          ELSE

!  The final 1x1 block

            IF ( ABS( data%D( 1, i ) ) >= big .OR. data%D( 1, i ) == zero ) THEN
              n_depen = n_depen + 1
              C_depen( n_depen ) = data%P( i ) - n
              data%D( 1, i ) = zero
            END IF
          END IF
        END DO

!  Any null blocks

        DO i = inform%SLS_inform%rank + 1, data%K%n
          n_depen = n_depen + 1
          C_depen( n_depen ) = data%P( i ) - n
        END DO

!  Reset "small" pivots to zero

        CALL SLS_alter_d( data%SLS_data, data%D, inform%SLS_inform )

!  Check to see if the constraints are consistent

        data%SOL( : n ) = zero
        data%SOL( n + 1 : data%K%n ) = C( : m ) / data%SCALE( : m )
        CALL SLS_solve( data%K, data%SOL, data%SLS_data,                       &
                        data%control%SLS_control, inform%SLS_inform )

        res_max = zero
        DO i = 1, m
          res = - C( i ) / data%SCALE( i )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            res = res + A_val( l ) * data%SOL( A_col( l ) ) / data%SCALE( i )
          END DO
          res_max = MAX( res_max, ABS( res ) )
        END DO

        IF ( res_max <= data%control%max_infeas ) THEN
          IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,       &
            "( A, ' constraints are consistent: maximum infeasibility = ',     &
          &    ES12.4 )" ) prefix, res_max
        ELSE
          IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,       &
            "( A, ' constraints are inconsistent: maximum infeasibility = ',   &
          &    ES12.4, /, A, 31X, ' is larger than control%max_infeas ' )" )   &
            prefix, res_max, prefix
          inform%status = GALAHAD_error_primal_infeasible ; GO TO 800
        END IF

      END IF

!  Successful call

      inform%status = GALAHAD_ok

  800 CONTINUE
      IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,           &
       "( A, 4( ' -' ), ' end of SLS test for rank defficiency', 4( ' - ' ) )")&
           prefix

      RETURN

!  End of FDC_find_dependent_sls

      END SUBROUTINE FDC_find_dependent_sls

!-*-*   F D C _ F I N D _ D E P E N D E N T _ U L S  S U B R O U T I N E   *-*-

      SUBROUTINE FDC_find_dependent_uls( n, m, A_val, A_col, A_ptr, C,         &
                                         n_depen, C_depen,                     &
                                         data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Determine which, if any, of the equality constraints A x = c is dependent
!  by forming LU factors of A
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments: see FDC_find_dependent
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      INTEGER, INTENT( OUT ) :: n_depen
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_col
      REAL ( KIND = wp ), INTENT( IN ),                                        &
                          DIMENSION( A_ptr( m + 1 ) - 1 ) :: A_val
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C

      INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_depen
      TYPE ( FDC_data_type ), INTENT( INOUT ) :: data
      TYPE ( FDC_control_type ), INTENT( IN ) :: control
      TYPE ( FDC_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: A_ne, i, l, out
      REAL :: time_now, time_record
      REAL ( KIND = wp ) :: clock_now, clock_record
      REAL ( KIND = wp ) :: res, res_max
      CHARACTER ( LEN = 80 ) :: array_name

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

! initialize solver-specific data; ensure that block-triangular form is NOT
! used as this may underestimate the rank

      data%control = control
      CALL ULS_initialize_solver( control%unsymmetric_linear_solver,           &
                                  data%ULS_data, inform%ULS_inform )
      data%control%ULS_control%relative_pivot_tolerance = control%pivot_tol
      data%control%ULS_control%minimum_size_for_btf = MAX( n, m )

      inform%non_negligible_pivot = control%pivot_tol

      out = data%control%out
      IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,           &
         "( /, A, 5( ' -' ), ' ULS test for rank defficiency', 5( ' - ' ) )" ) &
           prefix

!  Check that the problem makes sense

      IF ( n < 0 .OR. m < 0 ) THEN
        inform%status = GALAHAD_error_restrictions ; RETURN ; END IF

!  Check the case where n is zero

      IF ( n == 0 ) THEN
        IF ( MAXVAL( C ) <= data%control%max_infeas ) THEN
          DO i = 1, m - 1
            C_depen( i ) = i
          END DO
          n_depen = m - 1
          inform%status = GALAHAD_ok
        ELSE
          inform%status = GALAHAD_error_primal_infeasible
        END IF
        RETURN
      END IF

!  Check the case where m is zero

      IF ( m == 0 ) THEN
        n_depen = 0
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  Analyse the sparsity pattern. Set the dimensions of K

      A_ne = A_ptr( m + 1 ) - 1
      data%A%n = n ; data%A%m = m ; data%A%ne = A_ne
      CALL STRING_put( data%A%type, 'COORDINATE', inform%status )

!  Allocate the arrays for the analysis phase

      array_name = 'fdc: data%A%row'
      CALL SPACE_resize_array( data%A%ne, data%A%row, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%A%col'
      CALL SPACE_resize_array( data%A%ne, data%A%col, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%A%val'
      CALL SPACE_resize_array( data%A%ne, data%A%val, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%SCALE'
      CALL SPACE_resize_array( m, data%SCALE, inform%status,                   &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = data%control%deallocate_error_fatal,     &
             exact_size = data%control%space_critical,                         &
             bad_alloc = inform%bad_alloc, out = data%control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  insert the scaled data

      DO i = 1, m
        IF ( data%control%scale ) THEN
          IF ( A_ptr( i + 1 ) - 1 >= A_ptr( i ) ) THEN
            data%SCALE( i ) = MAX( one,                                        &
              MAXVAL( ABS( A_val( A_ptr( i ) : A_ptr( i + 1 ) - 1 ) ) ) )
          ELSE
            data%SCALE( i ) = one
          END IF
        ELSE
          data%SCALE( i ) = one
        END IF
        DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
          data%A%row( l ) = i ; data%A%col( l ) = A_col( l )
          data%A%val( l ) = A_val( l ) / data%SCALE( i )
        END DO
      END DO

!     write(78,*) data%A%m, data%A%n, data%A%ne
!     DO i = 1,  data%A%ne
!       write(78,"( 2I8, ES22.14 )" ) data%A%row( i ), data%A%col( i ),        &
!        data%A%val( i )
!     END DO

!  Factorize the matrix

      CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
      CALL ULS_factorize( data%A, data%ULS_data, data%control%ULS_control,     &
                          inform%ULS_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%factorize = inform%time%factorize + time_now - time_record
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + time_now - time_record

      IF ( out > 0 .AND. data%control%print_level >= 2 ) WRITE( out,           &
     &  "( A, ' ** factorize time = ', F0.2 ) " ) prefix, inform%time%factorize

!  Record the storage required

      inform%factorization_integer = inform%ULS_inform%workspace_factors
      inform%factorization_real = inform%ULS_inform%workspace_factors

!  Test that the factorization succeeded

      inform%factorization_status = inform%ULS_inform%status
      IF ( inform%ULS_inform%status < 0 ) THEN
        IF ( data%control%error > 0 .AND. data%control%print_level >= 1 )      &
           WRITE( data%control%error,                                          &
            "( A, '   **  Error return ', I6, ' from ULS_factorize' )")        &
              prefix, inform%ULS_inform%status
        inform%status = GALAHAD_error_factorization
        RETURN
      END IF

!  record how many dependencies there are

      n_depen = data%A%m - inform%ULS_inform%rank
      IF ( n_depen > 0 ) THEN

!  Record warning conditions

        IF ( out > 0 .AND. data%control%print_level >= 1 )                     &
          WRITE( out, "( A, I7, ' constraint', A, ' appear', A, ' to be ',     &
         &   'dependent ' )" ) prefix, n_depen,                                &
            TRIM( STRING_pleural( n_depen ) ),                                 &
            TRIM( STRING_verb_pleural( n_depen ) )

!  Allocate the arrays for the rank detetmination phase

        array_name = 'fdc: data%P'
        CALL SPACE_resize_array( data%A%m, data%P, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = data%control%deallocate_error_fatal,   &
               exact_size = data%control%space_critical,                       &
               bad_alloc = inform%bad_alloc, out = data%control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%INDEP'
        CALL SPACE_resize_array( MAX( data%A%m, data%A%n ), data%INDEP,        &
               inform%status,                                                  &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = data%control%deallocate_error_fatal,   &
               exact_size = data%control%space_critical,                       &
               bad_alloc = inform%bad_alloc, out = data%control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%C_depen'
        CALL SPACE_resize_array( n_depen, C_depen, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = data%control%deallocate_error_fatal,   &
               exact_size = data%control%space_critical,                       &
               bad_alloc = inform%bad_alloc, out = data%control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%SOL'
        CALL SPACE_resize_array( n, data%SOL, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = data%control%deallocate_error_fatal,   &
               exact_size = data%control%space_critical,                       &
               bad_alloc = inform%bad_alloc, out = data%control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN


!  Record the independent rows

        CALL ULS_enquire( data%ULS_data, inform%ULS_inform, data%P, data%INDEP )
        data%INDEP = 0
        data%INDEP( data%P( : inform%ULS_inform%rank ) ) = 1

!  Now record the dependent ones

        l = 0
        DO i = 1, data%A%m
          IF ( data%INDEP( i ) == 0 ) THEN
            l = l + 1
            C_depen( l ) = i
          END IF
        END DO

!  Check to see if the constraints are consistent

        CALL ULS_solve( data%A, C( : m ) / data%SCALE( : m ), data%SOL,        &
                        data%ULS_data, data%control%ULS_control,               &
                        inform%ULS_inform, .FALSE. )

        res_max = zero
        DO i = 1, m
          res = - C( i ) / data%SCALE( i )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            res = res + A_val( l ) * data%SOL( A_col( l ) ) / data%SCALE( i )
          END DO
          res_max = MAX( res_max, ABS( res ) )
        END DO

        IF ( res_max <= data%control%max_infeas ) THEN
          IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,       &
            "( A, ' constraints are consistent: maximum infeasibility = ',     &
          &    ES12.4 )" ) prefix, res_max
        ELSE
          IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,       &
            "( A, ' constraints are inconsistent: maximum infeasibility = ',   &
          &    ES12.4, /, A, 31X, ' is larger than control%max_infeas ' )" )   &
            prefix, res_max, prefix
          inform%status = GALAHAD_error_primal_infeasible ; GO TO 800
        END IF

      END IF

!  Successful call

      inform%status = GALAHAD_ok

  800 CONTINUE
      IF ( out > 0 .AND. data%control%print_level >= 1 ) WRITE( out,           &
       "( A, 4( ' -' ), ' end of ULS test for rank defficiency', 4( ' - ' ) )")&
           prefix

      RETURN

!  End of FDC_find_dependent_uls

      END SUBROUTINE FDC_find_dependent_uls

!-*-*-*-*-*-*-   F D C _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE FDC_terminate( data, control, inform, C_depen )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   C_depen  see Subroutine FDC_find_dependent
!   data     see Subroutine FDC_find_dependent
!   control  see preamble
!   inform   see preamble

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( FDC_data_type ), INTENT( INOUT ) :: data
      TYPE ( FDC_control_type ), INTENT( IN ) :: control
      TYPE ( FDC_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, ALLOCATABLE, OPTIONAL, DIMENSION( : ) :: C_depen

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated within SLS or ULS as appropriate

      IF ( control%use_sls ) THEN
        CALL SLS_terminate( data%SLS_data, control%SLS_control,                &
                            inform%SLS_inform )
        IF ( inform%SLS_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%bad_alloc = 'data%SLS_data'
          IF ( control%deallocate_error_fatal ) RETURN
        END IF
      ELSE
        CALL ULS_terminate( data%ULS_data, control%ULS_control,                &
                            inform%ULS_inform )
        IF ( inform%ULS_inform%status /= GALAHAD_ok ) THEN
          inform%status = GALAHAD_error_deallocate
          inform%bad_alloc = 'data%ULS_data'
          IF ( control%deallocate_error_fatal ) RETURN
        END IF
      END IF

!  Deallocate all remaining allocated arrays

      IF ( control%use_sls ) THEN
        array_name = 'fdc: data%K%row'
        CALL SPACE_dealloc_array( data%K%row,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%K%col'
        CALL SPACE_dealloc_array( data%K%col,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%K%val'
        CALL SPACE_dealloc_array( data%K%val,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%D'
        CALL SPACE_dealloc_array( data%D,                                      &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN
      ELSE
        array_name = 'fdc: data%A%row'
        CALL SPACE_dealloc_array( data%A%row,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%A%col'
        CALL SPACE_dealloc_array( data%A%col,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%A%val'
        CALL SPACE_dealloc_array( data%A%val,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN

        array_name = 'fdc: data%INDEP'
        CALL SPACE_dealloc_array( data%INDEP,                                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN
      END IF

      array_name = 'fdc: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%SCALE'
      CALL SPACE_dealloc_array( data%SCALE,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'fdc: data%SOL'
      CALL SPACE_dealloc_array( data%SOL,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      IF ( PRESENT( C_depen ) ) THEN
        array_name = 'fdc: C_depen'
        CALL SPACE_dealloc_array( C_depen,                                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND.                              &
             inform%status /= GALAHAD_ok ) RETURN
      END IF

      RETURN

!  End of subroutine FDC_terminate

      END SUBROUTINE FDC_terminate

!  End of module FDC

   END MODULE GALAHAD_FDC_double
