! THIS VERSION: GALAHAD 2.6 - 2/8/2013 AT 17:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ L L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started October 20th 2007
!   originally released GALAHAD Version 2.1. October 20th 2007

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

  MODULE GALAHAD_LLS_double

!      -----------------------------------------
!     |                                         |
!     | Solve the linear least-squares problem  |
!     |                                         |
!     |   minimize     || W ( A x + c ) ||_2    |
!     |   subject to || S^{1} x ||_2 <= Delta   |
!     |                                         |
!     | using a preconditined CG method         |
!     |                                         |
!      -----------------------------------------

      USE GALAHAD_SYMBOLS
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SPACE_double
!     USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SBLS_double
      USE GALAHAD_GLTR_double
      USE GALAHAD_SPECFILE_double
   
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LLS_initialize, LLS_read_specfile, LLS_solve, LLS_terminate,   &
                LLS_solve_main, QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: LLS_control_type

!   error and warning diagnostics occur on stream error 
   
        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0
 
!   preconditioner. The preconditioner to be used for the CG is defined by 
!    preconditioner. Possible values are
!
!    variable:
!
!      0  no preconditioner
!
!    explicit factorization:
!
!      1  G = I
!
!    implicit factorization:
!
!      -1  G_11 = 0, G_21 = 0, G_22 = I
!
        INTEGER :: preconditioner = 0

!   radius. An upper bound on the permitted step

        REAL ( KIND = wp ) :: radius = HUGE( one )

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SBLS

        TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for GLTR

        TYPE ( GLTR_control_type ) :: GLTR_control
      END TYPE

      TYPE, PUBLIC :: LLS_data_type
        LOGICAL :: new_h, new_c
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ATc
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Ax
        TYPE ( SBLS_data_type ) :: SBLS_data
        TYPE ( SMT_type ) :: H, C
        TYPE ( GLTR_data_type ) :: GLTR_data
      END TYPE

      TYPE, PUBLIC :: LLS_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ):: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = wp ) :: clock_solve = 0.0
      END TYPE

      TYPE, PUBLIC :: LLS_inform_type

!  return status. See LLS_solve for details

        INTEGER :: status = GALAHAD_ok

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of CG iterations required

        INTEGER :: cg_iter = - 1

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_real = - 1

!  the value of the objective function at the best estimate of the solution 
!   determined by LLS_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the norm ||S^{-1} x||_2 of the estimated solution x

        REAL ( KIND = wp ) :: norm_x  = HUGE( one )

!  timings (see above)

        TYPE ( LLS_time_type ) :: time

!  inform parameters for SBLS

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  inform parameters for GLTR

        TYPE ( GLTR_inform_type ) :: GLTR_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LLS. This routine should be called before
!  LLS_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( LLS_data_type ), INTENT( OUT ) :: data
      TYPE ( LLS_control_type ), INTENT( OUT ) :: control        
      TYPE ( LLS_inform_type ), INTENT( OUT ) :: inform 

      inform%status = GALAHAD_ok

!  Set control parameters

!  Real parameters

      control%radius = SQRT( point1 * HUGE( one ) )

!  Character parameters

      control%prefix = '""                            '

!  Ensure that the private data arrays have the correct initial status

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      CALL GLTR_initialize( data%GLTR_data, control%GLTR_control,              &
                            inform%GLTR_inform )

!  Reset GLTR and SBLS data for this package

      control%GLTR_control%unitm = .FALSE.
      control%GLTR_control%boundary = .FALSE.
      control%GLTR_control%prefix = '" - GLTR:"                    '

      control%SBLS_control%preconditioner = control%preconditioner
      control%SBLS_control%prefix = '" - SBLS:"                    '

      data%new_h = .TRUE.
      data%new_c = .TRUE.
      RETURN  

!  End of LLS_initialize

      END SUBROUTINE LLS_initialize

!-*-*-*-*-   L L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE LLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LLS_initialize could (roughly) 
!  have been set as:

!  BEGIN LLS SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   preconditioner-used                             0
!   truat-region-radius                             1.0D+19
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END LLS SPECIFICATIONS

!  Dummy arguments

      TYPE ( LLS_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: alive_unit = out + 1
      INTEGER, PARAMETER :: print_level = alive_unit + 1
      INTEGER, PARAMETER :: preconditioner = print_level + 1
      INTEGER, PARAMETER :: radius = preconditioner + 1
      INTEGER, PARAMETER :: space_critical = radius + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'LLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level' 
      spec( preconditioner )%keyword = 'preconditioner-used'

!  Real key-words

      spec( radius )%keyword = 'trust-region-radius'

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_integer( spec( error ), control%error,              &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( out ), control%out,                  &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( print_level ), control%print_level,  &
                                    control%error )
      CALL SPECFILE_assign_integer( spec( preconditioner ),                    &
                                    control%preconditioner,                    &
                                    control%error )

!  Set real values

      CALL SPECFILE_assign_real( spec( radius ), control%radius,               &
                                 control%error )

!  Set logical values

      CALL SPECFILE_assign_logical( spec( space_critical ),                    &
                                    control%space_critical,                    &
                                    control%error )
      CALL SPECFILE_assign_logical( spec( deallocate_error_fatal ),            &
                                    control%deallocate_error_fatal,            &
                                    control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read the specfile for SBLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
      ELSE
        CALL SBLS_read_specfile( control%SBLS_control, device )
      END IF

!  Read the specfile for GLTR

      IF ( PRESENT( alt_specname ) ) THEN
        CALL GLTR_read_specfile( control%GLTR_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-GLTR')
      ELSE
        CALL GLTR_read_specfile( control%GLTR_control, device )
      END IF

!  Reset GLTR and SBLS data for this package

      RETURN

      END SUBROUTINE LLS_read_specfile

!-*-*-*-*-*-*-*-*-   L L S _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE LLS_solve( prob, data, control, inform, W, S )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 || W ( A x + c ) ||_2^2
!
!     subject to    || S^{-1} x ||_2 <= Delta
!
!  where x is a vector of n components ( x_1, .... , x_n ), 
!  A is an m by n matrix, P and S are non-singular m x m and n x n 
!  diagonal matries, c is an m-vector and Delta is a constant, using 
!  a preconditioned conjugate-gradient method.
!  The subroutine is particularly appropriate when A is sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold 
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, that must be set to 
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to LLS_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, that must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!                 
!   %m is an INTEGER variable, that must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!                 
!   %A is a structure of type SMT_type used to hold the matrix A. 
!    Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%val( : )   the values of the components of A, stored row by row
!       A%col( : )   the column indices of the components of A
!       A%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 5 ) = TRANSFER( 'DENSE', A%type )
!       A%val( : )   the values of the components of A, stored row by row,
!                    with each the entries in each row in order of 
!                    increasing column indicies.
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output A%row will contain
!    the row numbers corresponding to the values in A%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).   
! 
!   %X is a REAL array of length %n, that must be set by the user
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
!
!   %C is a REAL array of length %m, that must be set by the user
!    to the values of the array c of constant terms in || Ax + c ||
!   
!  data is a structure of type LLS_data_type that holds private internal data
!
!  control is a structure of type LLS_control_type that controls the 
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LLS_initialize. See LLS_initialize 
!   for details
!
!  inform is a structure of type LLS_inform_type that provides 
!    information on exit from LLS_solve. The component status 
!    has possible values:
!  
!     0 Normal termination with a locally optimal solution.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!   - 3 one of the restrictions 
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 an error has occured in SILS_analyse; the status as returned by 
!       AINFO%FLAG is given in the component sils_analyse_status
!
!    -5 an error has occured in SILS_factorize; the status as returned by 
!       FINFO%FLAG is given in the component sils_factorize_status
!
!    -6 an error has occured in SILS_solve; the status as returned by 
!       SINFO%FLAG is given in the component sils_solve_status
!
!    -7 an error has occured in GLS_analyse; the status as returned by 
!       AINFO%FLAG is given in the component gls_analyse_status
!
!    -8 an error has occured in GLS_solve; the status as returned by 
!       SINFO%FLAG is given in the component gls_solve_status
!
!    -9 the computed precondition is insufficient. Try another
!
!   -11 the residuals are large; the factorization may be unsatisfactory
!
!  On exit from LLS_solve, other components of inform give the 
!  following:
!
!     alloc_status = the status of the last attempted allocation/deallocation 
!     bad_alloc = the name of the last array for which (de)allocation failed
!     cg_iter = the total number of conjugate gradient iterations required.
!     factorization_integer = the total integer workspace required for the 
!       factorization.
!     factorization_real = the total real workspace required for the 
!       factorization.
!     obj = the value of the objective function 1/2||Ax+c||^2 at the best 
!       estimate of the solution determined by LLS_solve.
!     time%total = the total time spent in the package.
!     time%factorize = the time spent factorizing the required matrices.
!     time%solve = the time spent computing the search direction.
!     SBLS_inform = inform components from SBLS
!     GLTR_inform = inform components from GLTR
!
!   W is an optional REAL array of length prob%m, that if present must be set 
!    by the user to the (nonzero) values of the diagonal scaling matrix W. 
!    If W is absent, scaling with the identity is assumed.
!   
!   S is an optional REAL array of length prob%n, that if present must be set 
!    by the user to the (nonzero) values of the diagonal trust-region scaling 
!    matrix S. If S is absent, scaling with the identity is assumed.
!   
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLS_control_type ), INTENT( INOUT ) :: control
      TYPE ( LLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( prob%m ) :: W
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( prob%m ) :: S

!  Local variables

      INTEGER :: i, j
      REAL ( KIND = wp ) :: time_end, clock_end

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Set initial values for inform 

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%cg_iter = - 1
      inform%obj = zero
      CALL CPU_TIME( inform%time%total )
      CALL CLOCK_time( inform%time%clock_total )
      IF ( prob%n <= 0 .OR. prob%m < 0 .OR.                                    &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
            "( ' ', /, A, ' **  Error return ', I0,' from LLS ' )" )           &
            prefix, inform%status 
        RETURN
      END IF 

!  Return the value zero if there are no constraints

    IF ( prob%m == 0 ) THEN
      prob%X = zero      
      CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
      inform%time%total = time_end - inform%time%total
      inform%time%clock_total = clock_end - inform%time%clock_total
      RETURN
    END IF

!  If required, write out problem 

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', 2I8 )" ) prob%n, prob%m
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' A (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ),                 &
            i = 1, prob%A%ne )
        END IF
        WRITE( control%out, "( ' C = ', /, ( 5ES12.4 ) )" )                    &
          prob%C( : prob%m )
      END IF

!  Call the solver

      CALL LLS_solve_main( prob%n, prob%m, prob%A, prob%C, prob%q, prob%X,     &
                           data, control, inform, W, S )

      CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
      inform%time%total = time_end - inform%time%total
      inform%time%clock_total = clock_end - inform%time%clock_total
      RETURN

!  End of LLS_solve

      END SUBROUTINE LLS_solve

!-*-*-*-*-   L L S _ S O L V E _ M A I N   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LLS_solve_main( n, m, A, C, q, X, data, control, inform, W, S )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 || W( A x + c ) ||_2^2
!
!     subject to    || S^{-1} x ||_2 <= Delta
!
!  where x is a vector of n components ( x_1, .... , x_n ), 
!  A is an m by n matrix, P and S are non-singular m x m and n x n 
!  diagonal matries, c is an m-vector and Delta is a constant, using 
!  a preconditioned conjugate-gradient method.
!  The subroutine is particularly appropriate when A is sparse
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( OUT ) :: q
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: C
      TYPE ( SMT_type ), INTENT( IN ) :: A
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      TYPE ( LLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLS_control_type ), INTENT( INOUT ) :: control
      TYPE ( LLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: W
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( m ) :: S

!  Local variables

      INTEGER :: out, i, j, l
      LOGICAL :: printt, printw, w_ne_id
      REAL ( KIND = wp ) :: time_end, clock_end
      REAL ( KIND = wp ) :: radius
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Return the value zero if there are no constraints or variables

    IF ( n == 0 ) THEN
      q = zero
      inform%status = GALAHAD_ok
      RETURN
    END IF

    IF ( m == 0 ) THEN
      X = zero      
      q = zero
      inform%status = GALAHAD_ok
      RETURN
    END IF

!  ===========================
!  Control the output printing
!  ===========================

      out = control%out

!  Single line of output per iteration
!  but with additional timings for various operations

      printt = out > 0 .AND. control%print_level >= 2 

!  As per printt, but with checking of residuals, etc, and also with an 
!  indication of where in the code we are

      printw = out > 0 .AND. control%print_level >= 4

      inform%GLTR_inform%status = 1 
      inform%GLTR_inform%negative_curvature = .TRUE.

      w_ne_id = PRESENT( W )

      IF ( control%preconditioner /= 0 ) THEN

!  Set C appropriately

        IF ( data%new_c ) THEN
          data%new_c = .FALSE.
          control%SBLS_control%new_c = 2
          data%C%ne = 0

          array_name = 'lls: data%C%row'
          CALL SPACE_resize_array( data%C%ne, data%C%row, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%C%col'
          CALL SPACE_resize_array( data%C%ne, data%C%col, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%C%val'
          CALL SPACE_resize_array( data%C%ne, data%C%val, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%C%type'
          CALL SPACE_dealloc_array( data%C%type, inform%status,                &
             inform%alloc_status, array_name = array_name, out = control%error )
          CALL SMT_put( data%C%type, 'COORDINATE', inform%alloc_status )
        ELSE
          control%SBLS_control%new_c = 0
        END IF

!  Set C appropriately

        IF ( data%new_h ) THEN
          data%new_c = .FALSE.
          data%H%ne = n
          array_name = 'lls: data%H%val'
          CALL SPACE_resize_array( data%H%ne, data%H%val, inform%status,       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) RETURN

          array_name = 'lls: data%H%type'
          CALL SPACE_dealloc_array( data%H%type, inform%status,                &
             inform%alloc_status, array_name = array_name, out = control%error )
          CALL SMT_put( data%H%type, 'DIAGONAL', inform%alloc_status )
        ELSE
          control%SBLS_control%new_h = 0
        END IF

        IF ( w_ne_id ) THEN
          data%H%val( : n ) = ( one / W( : n ) ** 2 )
        ELSE
          data%H%val = one
        END IF

        control%SBLS_control%new_h = 2

!  --------------------
!   Allocate workspace
!  --------------------

        array_name = 'lls: data%VECTOR'
        CALL SPACE_resize_array( n + m, data%VECTOR, inform%status,            &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN
     ELSE
        array_name = 'lls: data%VECTOR'
        CALL SPACE_resize_array( n, data%VECTOR, inform%status,                &
           inform%alloc_status, array_name = array_name,                       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) RETURN
      END IF

      array_name = 'lls: data%Ax'
      CALL SPACE_resize_array( m, data%Ax, inform%status,                      &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'lls: data%ATc'
      CALL SPACE_resize_array( n, data%ATc, inform%status,                     &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'lls: data%R'
      CALL SPACE_resize_array( n + m, data%R, inform%status,                   &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

      array_name = 'lls: data%S'
      CALL SPACE_resize_array( n, data%S, inform%status,                       &
         inform%alloc_status, array_name = array_name,                         &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) RETURN

!  ----------------------------------------------------------------
!   Use GLTR to minimize the objective
!  ----------------------------------------------------------------

      CALL CPU_TIME( inform%time%solve )
      CALL CLOCK_time( inform%time%clock_solve )

!  Compute the gradient A^T W^2 c

      data%ATc( : n ) = zero
      IF ( w_ne_id ) THEN
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            data%ATc( i ) = data%ATc( i )                                     &
              + DOT_PRODUCT( A%val( l + 1 : l + m ), C * W ** 2 )
            l = l + m
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              data%ATc( j ) = data%ATc( j ) + A%val( l ) * C( i ) * W( i ) ** 2
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l )
            data%ATc( j ) = data%ATc( j ) + A%val( l ) * C( i ) * W( i ) ** 2
          END DO
        END SELECT
      ELSE
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, n
            data%ATc( i )                                                      &
              = data%ATc( i ) + DOT_PRODUCT( A%val( l + 1 : l + m ), C )
            l = l + m
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              data%ATc( j ) = data%ATc( j ) + A%val( l ) * C( i )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            j = A%col( l )
            data%ATc( j ) = data%ATc( j ) + A%val( l ) * C( A%row( l ) )
          END DO
        END SELECT
      END IF

!  Set initial data
     
      IF ( control%radius > zero ) THEN
        radius = control%radius
      ELSE
        radius = SQRT( point1 * HUGE( one ) )
      END IF 

      control%GLTR_control%f_0 = half * DOT_PRODUCT( C, C )

      data%R( : n ) = data%ATc( : n )
      inform%GLTR_inform%status = 1
      inform%cg_iter = 0

      control%GLTR_control%rminvr_zero = two * epsmch ** 2
!     control%GLTR_control%rminvr_zero = hundred * epsmch ** 2
      control%GLTR_control%unitm = control%preconditioner == 0
      control%GLTR_control%boundary = .FALSE.
      control%GLTR_control%prefix = '" - GLTR:"                    '

      DO
        CALL GLTR_solve( n, radius, q, X( : n ), data%R( : n ),               &
                         data%VECTOR( : n ), data%GLTR_data,                  &
                         control%GLTR_control, inform%GLTR_inform )

!  Check for error returns

!       WRITE(6,"( ' case ', i3  )" ) inform%GLTR_inform%status
        SELECT CASE( inform%GLTR_inform%status )

!  Successful return

        CASE ( GALAHAD_ok )
          EXIT

!  Warnings

        CASE ( GALAHAD_warning_on_boundary, GALAHAD_error_max_iterations )
          IF ( printt ) WRITE( out, "( /,                                      &
         &  A, ' Warning return from GLTR, status = ', I6 )" ) prefix,         &
              inform%GLTR_inform%status
          EXIT
          
!  Allocation errors

         CASE ( GALAHAD_error_allocate )
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = inform%gltr_inform%alloc_status
           inform%bad_alloc = inform%gltr_inform%bad_alloc
           GO TO 900

!  Deallocation errors

         CASE ( GALAHAD_error_deallocate  )
           inform%status = GALAHAD_error_deallocate
           inform%alloc_status = inform%gltr_inform%alloc_status
           inform%bad_alloc = inform%gltr_inform%bad_alloc
           GO TO 900

!  Error return

        CASE DEFAULT
          inform%status = inform%gltr_inform%status
          IF ( printt ) WRITE( out, "( /,                                      &
         &  A, ' Error return from GLTR, status = ', I6 )" ) prefix,           &
              inform%GLTR_inform%status
          EXIT

!  Find the preconditioned gradient

        CASE ( 2, 6 )
          IF ( printw ) WRITE( out,                                            &
             "( A, ' ............... precondition  ............... ' )" ) prefix

          data%VECTOR( : n ) = data%VECTOR( : n )  * ( S( : n ) ) ** 2 

!!         control%SBLS_control%out = 6
!!         control%SBLS_control%print_level = 2
!          control%SBLS_control%affine = .TRUE.
!          CALL SBLS_solve( n, m, A, data%C, data%SBLS_data,                   &
!             control%SBLS_control, inform%SBLS_inform, data%VECTOR )

!          IF ( inform%SBLS_inform%status < 0 ) THEN
!            inform%status = inform%SBLS_inform%status
!            GO TO 900
!          END IF

!  Form the product of VECTOR with A^T W^2 A

        CASE ( 3, 7 )

          IF ( printw ) WRITE( out,                                            &
            "( A, ' ............ matrix-vector product ..........' )" ) prefix

          data%Ax( : m ) = zero
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            l = 0
            DO i = 1, m
              data%Ax( i ) = data%Ax( i )                                      &
                + DOT_PRODUCT( A%val( l + 1 : l + n ), data%VECTOR )
              l = l + n
            END DO
            IF ( w_ne_id ) data%Ax( : m ) = data%Ax( : m ) * W( : m ) ** 2
            l = 0
            data%VECTOR = zero
            DO i = 1, n
              data%VECTOR( i ) = data%VECTOR( i )                              &
                + DOT_PRODUCT( A%val( l + 1 : l + m ), data%Ax )
              l = l + m
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                data%Ax( i ) = data%Ax( i )                                    &
                  + A%val( l ) * data%VECTOR( A%col( l ) )
              END DO
            END DO
            IF ( w_ne_id ) data%Ax( : m ) = data%Ax( : m ) * W( : m ) ** 2
            data%VECTOR = zero
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                data%VECTOR( j ) = data%VECTOR( j )                            &
                  + A%val( l ) * data%Ax( A%col( i ) )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l )
              data%Ax( i ) = data%Ax( i )                                      &
                + A%val( l ) * data%VECTOR( A%col( l ) )
            END DO
            IF ( w_ne_id ) data%Ax( : m ) = data%Ax( : m ) * W( : m ) ** 2
            data%VECTOR = zero
            DO l = 1, A%ne
              j = A%col( l )
              data%VECTOR( j ) = data%VECTOR( j )                              &
                + A%val( l ) * data%Ax( A%row( l ) )
            END DO
          END SELECT

!  Reform the initial residual

        CASE ( 5 )
          
          IF ( printw ) WRITE( out,                                            &
            "( A, ' ................. restarting ................ ' )" ) prefix

          data%R( : n ) = data%ATc

        END SELECT

      END DO

      inform%obj = ABS( q )
      inform%cg_iter = inform%GLTR_inform%iter
      inform%norm_x = inform%GLTR_inform%mnormx
!     write(6,*) inform%GLTR_inform%iter_pass2

      IF ( printw ) THEN
        data%Ax( : m ) = C( : m )
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            data%Ax( i )                                                      &
              = data%Ax( i ) + DOT_PRODUCT( A%val( l + 1 : l + n ), X )
            l = l + n
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              data%Ax( i ) = data%Ax( i ) + A%val( l ) * X( A%col( l ) )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l )
            data%Ax( i ) = data%Ax( i ) + A%val( l ) * X( A%col( l ) )
          END DO
        END SELECT
        IF ( w_ne_id ) data%Ax( : m ) = data%Ax( : m ) * W( : m )
        WRITE( out, "(  A, ' computed objective ', ES12.4 )" )                 &
          half * DOT_PRODUCT( data%Ax( : m ), data%Ax( : m ) )
      END IF

      CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
      inform%time%solve = time_end - inform%time%solve
      inform%time%clock_solve = clock_end - inform%time%clock_solve

      IF ( printt ) WRITE( out,                                                &
         "(  A, ' on exit from GLTR: status = ', I0, ', CG iterations = ', I0, &
        &   ', time = ', F0.2 )" ) prefix,                                     &
            inform%GLTR_inform%status, inform%cg_iter, inform%time%solve

      IF ( printt ) THEN
        SELECT CASE( control%preconditioner )
        CASE( 0 )
          WRITE( out, "( A, ' No preconditioner' )" ) prefix
        CASE( 1 )
          WRITE( out, "( A, ' Preconditioner G = I' )" ) prefix
        CASE( - 1 )
          WRITE( out, "( A, ' Preconditioner G_22 = I' )" ) prefix
        END SELECT
      END IF

      RETURN
 
!  Error returns

  900 CONTINUE
      CALL CPU_TIME( time_end ) ; CALL CLOCK_time( clock_end )
      inform%time%solve = time_end - inform%time%solve
      inform%time%clock_solve = clock_end - inform%time%clock_solve

      RETURN

!  End of LLS_solve_main

      END SUBROUTINE LLS_solve_main

!-*-*-*-*-*-*-   L L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE LLS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine LLS_initialize
!   control see Subroutine LLS_initialize
!   inform  see Subroutine LLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLS_control_type ), INTENT( IN ) :: control        
      TYPE ( LLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by SBLS and GLTR

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                           inform%SBLS_inform )
      CALL GLTR_terminate( data%GLTR_data, control%GLTR_control,               &
                           inform%GLTR_inform )

!  Deallocate all remaining allocated arrays

      array_name = 'lls: data%C%row'
      CALL SPACE_dealloc_array( data%C%row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%C%col'
      CALL SPACE_dealloc_array( data%C%col,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%C%val'
      CALL SPACE_dealloc_array( data%C%val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%C%type'
      CALL SPACE_dealloc_array( data%C%type,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%G_f'
      CALL SPACE_dealloc_array( data%ATc,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'lls: data%Ax'
      CALL SPACE_dealloc_array( data%Ax,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  End of subroutine LLS_terminate

      END SUBROUTINE LLS_terminate

!  End of module LLS

   END MODULE GALAHAD_LLS_double
