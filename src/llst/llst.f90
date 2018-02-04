! THIS VERSION: GALAHAD 2.6 - 21/02/2014 AT 08:00 GMT.

!-*-*-*-*-*-*-  G A L A H A D _ L L S T  d o u b l e  M O D U L E  *-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 2.5. February 21st, 2011

!  For full documentation, see 
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LLST_double

!       -----------------------------------------------
!      |                                               |
!      | Solve the linear least-squares subproblem     |
!      | with trust-region regularization              |
!      |                                               |
!      |    minimize     1/2 || A x - b ||^2           |
!      |    subject to    ||x||_S <= radius            |
!      |                                               |
!      | where ||x||_S^2 = <x, Sx> and S is diagonally |
!      | dominant, using a sparse matrix factorization |
!      |                                               |
!       -----------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_RAND_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_ROOTS_double, ONLY: ROOTS_quadratic, ROOTS_cubic
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_SBLS_double
      USE GALAHAD_SLS_double
      USE GALAHAD_IR_double
      USE GALAHAD_MOP_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LLST_initialize, LLST_read_specfile, LLST_solve,               &
                LLST_terminate, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: history_max = 100
      INTEGER, PARAMETER :: max_degree = 3
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: point4 = 0.4_wp
      REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: six = 6.0_wp
      REAL ( KIND = wp ), PARAMETER :: sixth = one / six
      REAL ( KIND = wp ), PARAMETER :: twothirds = two /three
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: twentyfour = 24.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = half * HUGE( one ) 
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one ) 
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch

      REAL ( KIND = wp ), PARAMETER :: theta_ii = one
      REAL ( KIND = wp ), PARAMETER :: theta_eps = point01
      REAL ( KIND = wp ), PARAMETER :: theta_g = half
      REAL ( KIND = wp ), PARAMETER :: theta_n = half
      REAL ( KIND = wp ), PARAMETER :: theta_n_small = ten ** ( - 1 )
      REAL ( KIND = wp ), PARAMETER :: theta_n_tiny = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: gamma_eps = half
      REAL ( KIND = wp ), PARAMETER :: gamma = one
      REAL ( KIND = wp ), PARAMETER :: roots_tol = teneps
      LOGICAL :: roots_debug = .FALSE.

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: LLST_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  how much of A has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER :: new_a = 2

!  how much of S has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER :: new_s = 2

!  the maximum number of factorizations (=iterations) allowed. -ve => no limit

        INTEGER :: max_factorizations = - 1

!  maximum degree of Taylor approximant allowed (<= 3)

        INTEGER :: taylor_max_degree = 3

!  initial estimate of the Lagrange multipler

        REAL ( KIND = wp ) :: initial_multiplier = zero

!  lower and upper bounds on the multiplier, if known

        REAL ( KIND = wp ) :: lower = - half * HUGE( one )
        REAL ( KIND = wp ) :: upper =  HUGE( one )

!  stop when | ||x|| - (multiplier/sigma)^(1/(p-2)) | <=
!              stop_normal * max( ||x||, (multiplier/sigma)^(1/(p-2)) )

        REAL ( KIND = wp ) :: stop_normal = epsmch

!  is the solution is REQUIRED to lie on the boundary (i.e., is the constraint 
!  an equality)?

        LOGICAL :: equality_problem= .FALSE.

!  ignore initial_multiplier?

        LOGICAL :: use_initial_multiplier = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  definite linear equation solver

        CHARACTER ( LEN = 30 ) :: definite_linear_solver =                     &
           "sils" // REPEAT( ' ', 26 )

!  all output lines will be prefixed by 
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes, 
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix  = '""                            '

!  control parameters for the symmetric factorization and related linear solves

        TYPE ( SBLS_control_type ) :: SBLS_control

!  control parameters for the factorization of S and related linear solves

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for iterative refinement for definite system solves

        TYPE ( IR_control_type ) :: IR_control

      END TYPE

!  - - - - - - - - - -
!   data derived type 
!  - - - - - - - - - -

      TYPE, PUBLIC :: LLST_data_type
        PRIVATE
        INTEGER :: m, npm, s_ne, a_ne, m_end
        TYPE ( RAND_seed ) :: seed
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C, U, Y, Z
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_diag, S_offd
        LOGICAL :: accurate = .TRUE.
        TYPE ( SMT_type ) :: H_sbls, C_sbls
        TYPE ( IR_data_type ) :: IR_data
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( SBLS_data_type ) :: SBLS_data
        TYPE ( SLS_control_type ) :: SLS_control
        TYPE ( LLST_control_type ) :: control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LLST_time_type

!  total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  CPU time assembling K(lambda) = ( lambda * S   A^T )
!                                  (      A       - I )

        REAL ( KIND = wp ) :: assemble = 0.0

!  CPU time spent analysing K(lambda)

        REAL ( KIND = wp ) :: analyse = 0.0

!  CPU time spent factorizing K(lambda)

        REAL ( KIND = wp ) :: factorize = 0.0

!  CPU time spent solving linear systems inolving K(lambda)

        REAL ( KIND = wp ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time assembling K(lambda)

        REAL ( KIND = wp ) :: clock_assemble = 0.0

!  clock time spent analysing K(lambda)

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  clock time spent factorizing K(lambda)

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving K(lambda)

        REAL ( KIND = wp ) :: clock_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - -
!   history derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LLST_history_type

!  value of lambda

        REAL ( KIND = wp ) :: lambda = zero

!  corresponding value of ||x(lambda)||_S

        REAL ( KIND = wp ) :: x_norm = zero

!  corresponding value of ||A x(lambda) - b||_2

!       REAL ( KIND = wp ) :: r_norm = zero
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - 
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - 

      TYPE, PUBLIC :: LLST_inform_type

!   reported return status:
!      0 the solution has been found
!     -1 an array allocation has failed
!     -2 an array deallocation has failed
!     -3 n and/or Delta is not positive
!    -10 the factorization of K(lambda) failed
!    -15 M does not appear to be strictly diagonally dominant
!    -16 ill-conditioning has prevented furthr progress

        INTEGER :: status = 0

!  STAT value after allocate failure

        INTEGER :: alloc_status = 0

!   the number of factorizations performed

        INTEGER :: factorizations = 0

!  the number of (||x||_S,lambda) pairs in the history

        INTEGER :: len_history = 0

!  corresponding value of the two-norm of the residual, ||A x(lambda) - b||_2

        REAL ( KIND = wp ) :: r_norm = zero

!  the S-norm of x, ||x||_S

        REAL ( KIND = wp ) :: x_norm = zero

!  the Lagrange multiplier corresponding to the trust-region constraint

        REAL ( KIND = wp ) :: multiplier = zero

!  name of array which provoked an allocate failure

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  time information

        TYPE ( LLST_time_type ) :: time

!  history information

        TYPE ( LLST_history_type ), DIMENSION( history_max ) :: history

!  information from the symmetric factorization and related linear solves

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  information from the factorization of S and related linear solves

        TYPE ( SLS_inform_type ) :: SLS_inform

!  information from the iterative refinement for definite system solves

        TYPE ( IR_inform_type ) :: IR_inform
      END TYPE

    CONTAINS

!-*-*-*-*-*-  L L S T _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LLST_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!
!  .  Set initial values for the LLST control parameters  .
!   
!  Arguments:
!  =========
!
!   data     private internal data
!   control  a structure containing control information. See LLST_control_type
!   data     private internal data
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( LLST_DATA_TYPE ), INTENT( INOUT ) :: data
      TYPE ( LLST_CONTROL_TYPE ), INTENT( OUT ) :: control        
      TYPE ( LLST_inform_type ), INTENT( OUT ) :: inform    

      inform%status = GALAHAD_ok

!  Initalize random number seed

      CALL RAND_initialize( data%seed )

!  Set initial control parameter values

      control%stop_normal = epsmch ** 0.75

!  initalize IR components

      CALL IR_initialize( data%IR_data, control%IR_control,                    &
                          inform%IR_inform )
      control%IR_control%prefix = '" - IR:"'

!  initalize SLS components

      CALL SLS_initialize( control%definite_linear_solver,                     &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform )
      control%SLS_control%prefix = '" - SLS:"'

!  initalize SBLS components

      CALL SBLS_initialize( data%SBLS_data, control%SBLS_control,              &
                            inform%SBLS_inform )
      control%SBLS_control%prefix = '" - SBLS:"'

      RETURN

!  End of subroutine LLST_initialize

      END SUBROUTINE LLST_initialize

!-*-*-*-*   L L S T _ R E A D _ S P E C F I L E  S U B R O U T I N E   *-*-*-*-

      SUBROUTINE LLST_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LLST_initialize could (roughly) 
!  have been set as:

!  BEGIN LLST SPECIFICATIONS (DEFAULT)
!   error-printout-device                          6
!   printout-device                                6
!   print-level                                    0
!   has-a-changed                                  2
!   has-s-changed                                  2
!   factorization-limit                            -1
!   max-degree-taylor-approximant                  3
!   initial-multiplier                             0.0
!   lower-bound-on-multiplier                      0.0
!   upper-bound-on-multiplier                      1.0D+300
!   stop-normal-case                               1.0D-12
!   equality-problem                               F
!   use-initial-multiplier                         F
!   space-critical                                 F
!   deallocate-error-fatal                         F
!   definite-linear-equation-solver                sils
!   output-line-prefix                             ""
!  END LLST SPECIFICATIONS

!  Dummy arguments

      TYPE ( LLST_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: new_a = print_level + 1
      INTEGER, PARAMETER :: new_s = new_a + 1
      INTEGER, PARAMETER :: max_factorizations = new_s + 1
      INTEGER, PARAMETER :: taylor_max_degree = max_factorizations + 1
      INTEGER, PARAMETER :: initial_multiplier = taylor_max_degree + 1
      INTEGER, PARAMETER :: lower = initial_multiplier + 1
      INTEGER, PARAMETER :: upper = lower + 1
      INTEGER, PARAMETER :: stop_normal = upper + 1
      INTEGER, PARAMETER :: equality_problem = stop_normal + 1
      INTEGER, PARAMETER :: use_initial_multiplier = equality_problem + 1
      INTEGER, PARAMETER :: space_critical = use_initial_multiplier + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: definite_linear_solver = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: prefix = definite_linear_solver + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LLST'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level' 
      spec( new_a )%keyword = 'has-a-changed'
      spec( new_s )%keyword = 'has-s-changed'
      spec( max_factorizations )%keyword = 'factorization-limit'
      spec( taylor_max_degree )%keyword = 'max-degree-taylor-approximant'

!  Real key-words

      spec( initial_multiplier )%keyword = 'initial-multiplier'
      spec( lower )%keyword = 'lower-bound-on-multiplier'
      spec( upper )%keyword = 'upper-bound-on-multiplier'
      spec( stop_normal )%keyword = 'stop-normal-case'

!  Logical key-words

      spec( equality_problem )%keyword = 'equality-problem'
      spec( use_initial_multiplier )%keyword = 'use-initial-multiplier'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal  )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( definite_linear_solver )%keyword = 'definite-linear-equation-solver'
      spec( prefix )%keyword = 'output-line-prefix'

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( error ), control%error,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( out ), control%out,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ), control%print_level,    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_a ), control%new_a,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_s ), control%new_s,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_factorizations ),                  &
                                  control%max_factorizations,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( taylor_max_degree ),                   &
                                  control%taylor_max_degree,                   &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( initial_multiplier ),                  &
                                  control%initial_multiplier,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( lower ), control%lower,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( upper ), control%upper,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_normal ), control%stop_normal,    &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( equality_problem ),                    &
                                  control%equality_problem,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( use_initial_multiplier ),              &
                                  control%use_initial_multiplier,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal  ),             &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( definite_linear_solver ),              &
                                  control%definite_linear_solver,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read specfile data for IR

      IF ( PRESENT( alt_specname ) ) THEN
        CALL IR_read_specfile( control%IR_control, device,                     &
                                 alt_specname = TRIM( alt_specname ) // '-IR')
      ELSE
        CALL IR_read_specfile( control%IR_control, device )
      END IF

!  Read specfile data for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                 alt_specname = TRIM( alt_specname ) // '-SLS')
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

!  Read specfile data for SBLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SBLS_read_specfile( control%SBLS_control, device,                 &
                                 alt_specname = TRIM( alt_specname ) // '-SBLS')
      ELSE
        CALL SBLS_read_specfile( control%SBLS_control, device )
      END IF

      RETURN

      END SUBROUTINE LLST_read_specfile

!-*-*-*-*-*-*-*-*-*-*  L L S T _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE LLST_solve( m, n, radius, A, B, X, data, control, inform, S )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!    m - the number of rows of A and entries in b
!
!    n - the number of colunds of A and entries in x
!
!   radius - the trust-region radius
!
!   A -  a structure of type SMT_type used to hold the matrix A.
!    Three storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%m          the number of rows of A
!       A%ne         the number of nonzeros used to store A
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       A%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', A%type )
!       A%m          the number of rows of A
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
!       A%m          the number of rows of A
!       A%val( : )   the values of the components of A, stored row by row,
!                    with the entries in each row in order of increasing 
!                    column indicies.
!
!   B - a vector of values b
!
!   X - the required solution vector x
!
!   data - private internal data
!
!   control - a structure containing control information. See LLST_control_type
!
!   inform - a structure containing information. See LLST_inform_type
!
!   S - an optional structure of type SMT_type used to hold the LOWER TRIANGULAR
!    part of the symmetric, DIAGONALLY DOMINANT matrix S. Four storage formats 
!    are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       S%type( 1 : 10 ) = TRANSFER( 'COORDINATE', S%type )
!       S%ne         the number of nonzeros used to store 
!                    the LOWER TRIANGULAR part of S
!       S%val( : )   the values of the components of S
!       S%row( : )   the row indices of the components of S
!       S%col( : )   the column indices of the components of S
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       S%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', S%type )
!       S%val( : )   the values of the components of S, stored row by row
!       S%col( : )   the column indices of the components of S
!       S%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       S%type( 1 : 5 ) = TRANSFER( 'DENSE', S%type )
!       S%val( : )   the values of the components of S, stored row by row,
!                    with the entries in each row in order of increasing 
!                    column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       S%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', S%type )
!       S%val( : )   the values of the diagonals of S, stored in order
!
!    If the argument S is absent, S will be assumed to be the identity matrix,
!    and thus ||x||_S = ||x||_2
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: m, n
      REAL ( KIND = wp ), INTENT( IN ) :: radius
      TYPE ( SMT_type ), INTENT( IN ) :: A
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m ) :: B
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( LLST_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLST_control_type ), INTENT( IN ) :: control        
      TYPE ( LLST_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SMT_type ), OPTIONAL, INTENT( INOUT ) :: S

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, l, it, out, nroots, print_level
      INTEGER :: max_order, n_lambda, in_n
      REAL :: time_start, time_now, time_record
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_record, lambda_n
      REAL ( KIND = wp ) :: lambda, lambda_l, lambda_u, delta_lambda
      REAL ( KIND = wp ) :: c_norm, c_norm_over_radius, v_norm2, w_norm2, val
      REAL ( KIND = wp ) :: beta, z_norm2, root1, root2, root3, lambda_pert
      REAL ( KIND = wp ) :: width, lambda_plus, a_one, a_inf, lambda_sinv
      REAL ( KIND = wp ), DIMENSION( 3 ) :: lambda_new
      REAL ( KIND = wp ), DIMENSION( 0 : max_degree ) :: x_norm2, pi_beta
      LOGICAL :: printi, printt, printd, printh, psdef, try_zero, unit_s
      LOGICAL :: phase_1
      CHARACTER ( LEN = 1 ) :: region
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  set initial values

      inform%IR_inform%status = 0 
      inform%IR_inform%alloc_status = 0 
      inform%IR_inform%bad_alloc = ''

      unit_s = .NOT. PRESENT( S )
      phase_1 = .TRUE.
      X = zero ; inform%x_norm = zero ; inform%r_norm = TWO_NORM( B )
      inform%factorizations = 0 ; inform%len_history = 0
      inform%multiplier = zero
      data%control = control
      out = control%out

!  record desired output level

      print_level = control%print_level
      printi = out > 0 .AND. print_level > 0
      printt = out > 0 .AND. print_level > 1
      printd = out > 0 .AND. print_level > 2
      printh = printi

!  check for obvious errors

      IF ( n <= 0 ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
           "( A, ' n = ', I0, ' is too small ' )" ) prefix, n
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF

      IF ( radius <= zero ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
          "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, radius
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF

!  choose initial values for the control parameters for the factorization

      IF ( control%max_factorizations > 0 ) THEN
        data%control%max_factorizations = control%max_factorizations
      ELSE
        data%control%max_factorizations = HUGE( 0 )
      END IF

!  -----------------------------------------------------------------------
!  Set up data structure for the matrix K(lambda) = ( lambda * S     A^T )
!                                                   (      A         - I )
!  -----------------------------------------------------------------------

!  compute the space required to hold the matrix lambda S  ...

      IF ( .NOT. unit_s ) THEN
        S%n = n ; S%m = n
        IF ( data%control%new_s >= 2 ) THEN
          SELECT CASE ( SMT_get( S%type ) )
          CASE ( 'IDENTITY', 'SCALED_IDENTITY' ) 
            data%s_ne = 1
          CASE ( 'DIAGONAL' ) 
            data%s_ne = n
          CASE ( 'DENSE' ) 
            data%s_ne = ( n * ( n + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            data%s_ne = S%ptr( n + 1 ) - 1
          CASE ( 'COORDINATE' )
            data%s_ne = S%ne
          CASE DEFAULT
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error,                                            &
              "( A, ' Unknown storage format ', A, ' for S' )" ) prefix,       &
                 TRIM( SMT_get( S%type ) )
            inform%status = GALAHAD_error_restrictions
            GO TO 910
          END SELECT
        END IF        
      ELSE
        data%s_ne = 1
      END IF

!  ... and to hold A

      IF ( data%control%new_a >= 2 ) THEN
        data%npm = n + m
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          data%a_ne = m * n
        CASE ( 'SPARSE_BY_ROWS' )
          data%a_ne = A%ptr( m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%a_ne = A%ne
        CASE DEFAULT
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error,                                              &
            "( A, ' Unknown storage format ', A, ' for A' )" ) prefix,         &
               TRIM( SMT_get( A%type ) )
          inform%status = GALAHAD_error_restrictions
          GO TO 910
        END SELECT
      END IF        

!  make space for H = lambda * S

      IF ( data%control%new_a >= 2 .OR. data%control%new_s >= 2 ) THEN
        IF ( .NOT. unit_s ) THEN
          data%H_sbls%m = n ; data%H_sbls%n = n
          SELECT CASE ( SMT_get( S%type ) )
          CASE ( 'IDENTITY', 'SCALED_IDENTITY' )
            CALL SMT_put( data%H_sbls%type, 'SCALED_IDENTITY',                 &
                          inform%alloc_status ) 
          CASE ( 'DIAGONAL' ) 
            CALL SMT_put( data%H_sbls%type, 'DIAGONAL',                        &
                          inform%alloc_status ) 
          CASE ( 'DENSE' ) 
            CALL SMT_put( data%H_sbls%type, 'DENSE',                           &
                          inform%alloc_status ) 
          CASE ( 'SPARSE_BY_ROWS' )
            CALL SMT_put( data%H_sbls%type, 'SPARSE_BY_ROWS',                  &
                          inform%alloc_status )
          CASE ( 'COORDINATE' )
            data%H_sbls%ne = data%s_ne
            CALL SMT_put( data%H_sbls%type, 'COORDINATE',                      &
                          inform%alloc_status )
          END SELECT

          IF ( SMT_get( S%type ) == 'COORDINATE' ) THEN
            array_name = 'llst: H_sbls%row'
            CALL SPACE_resize_array( data%s_ne, data%H_sbls%row,               &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910
            data%H_sbls%row( : data%s_ne ) = S%row( : data%s_ne )
          END IF

          IF ( SMT_get( S%type ) == 'COORDINATE' .OR.                          &
               SMT_get( S%type ) == 'SPARSE_BY_ROWS' ) THEN
            array_name = 'llst: H_sbls%col'
            CALL SPACE_resize_array( data%s_ne, data%H_sbls%col,               &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910
            data%H_sbls%col( : data%s_ne ) = S%col( : data%s_ne )
          END IF

          IF ( SMT_get( S%type ) == 'SPARSE_BY_ROWS' ) THEN
            array_name = 'llst: H_sbls%ptr'
            CALL SPACE_resize_array( n + 1, data%H_sbls%ptr,                   &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910
            data%H_sbls%ptr( : n + 1 ) = S%ptr( : n + 1 )
          END IF

          IF ( SMT_get( S%type ) == 'COORDINATE' .OR.                          &
               SMT_get( S%type ) == 'SPARSE_BY_ROWS' .OR.                      &
               SMT_get( S%type ) == 'DENSE' .OR.                               &
               SMT_get( S%type ) == 'DIAGONAL' .OR.                            &
               SMT_get( S%type ) == 'SCALED_IDENTITY' .OR.                     &
               SMT_get( S%type ) == 'IDENTITY' ) THEN
            array_name = 'llst: H_sbls%val'
            CALL SPACE_resize_array( data%s_ne, data%H_sbls%val,               &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910
          END IF

!  special case for H = lambda I when S is not provided

        ELSE
          array_name = 'llst: H_sbls%val'
          CALL SPACE_resize_array( data%s_ne, data%H_sbls%val,                 &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910
          CALL SMT_put( data%H_sbls%type, 'SCALED_IDENTITY',                   &
                        inform%alloc_status ) 
        END IF

!  make space for C = - I

        data%C_sbls%m = m ; data%C_sbls%n = m
        CALL SMT_put( data%C_sbls%type, 'IDENTITY', inform%alloc_status ) 
      END IF

      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%assemble = inform%time%assemble + time_now - time_start
      inform%time%clock_assemble =                                             &
        inform%time%clock_assemble + clock_now - clock_start
      IF ( printt ) WRITE( out, "( A,  ' time( assembly ) = ', F0.2 )" )       &
        prefix, clock_now - clock_start

!  =====================
!  Array (re)allocations
!  =====================

!  allocate C, U, V, Y and Z

      array_name = 'llst: C'
      CALL SPACE_resize_array( n, data%C,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'llst: U'
      CALL SPACE_resize_array( data%npm, data%U,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'llst: Y'
      CALL SPACE_resize_array( data%npm, data%Y,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'llst: Z'
      CALL SPACE_resize_array( data%npm, data%Z,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

!  set c = A^T b

      CALL mop_AX( one, A, b, zero, data%C( : n ), transpose = .TRUE. )

!  the real line is partitioned into disjoint sets
!     N = { lambda: lambda <= max(0, -lambda_1(H))}
!     L = { lambda: max(0, -lambda_1(H)) < lambda <= lambda_optimal } and
!     G = { lambda: lambda > lambda_optimal };
!  for the equality problem, 0 plays no role in N and L.
!  The aim is to find a lambda in L, as generally then Newton's method 
!  will converge both globally and ultimately quadratically. We also let
!     F = L union G

!  construct values lambda_l and lambda_u for which lambda_l <= lambda_optimal 
!   <= lambda_u, and ensure that all iterates satisfy lambda_l <= lambda 
!   <= lambda_u

!  bounds on lambda currently used -

!  * lambda >=0 (for the inequality problem)
!  * ||A^Tb||_S^-1 / radius - lambda_n <= lambda <= ||A^Tb||_S^-1 / radius
!  where lambda_n^2 <= ||A S^{-1/2}||_1 . ||A S^{-1/2}||_inf
!                   <= ||A||_1 . ||A||_inf . ||S^{-1}||_2

!  ** unit trust-region scaling

      IF ( unit_s ) THEN

!  record || c || / radius 

        c_norm = TWO_NORM( data%C( : n ) )
        c_norm_over_radius = c_norm / radius

!  compute the sum of the absolute values of each row of A in U(1:m) and 
!  column of A in U(m+1:m+n)

        data%U( : data%npm ) = zero
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1 ; val = ABS( A%val( l ) )
              data%U( i ) =  data%U( i ) + val
              data%U( m + j ) =  data%U( m + j ) + val
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l ) ; val = ABS( A%val( l ) )
              data%U( i ) =  data%U( i ) + val
              data%U( m + j ) =  data%U( m + j ) + val
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l ) ; val = ABS( A%val( l ) )
            data%U( i ) =  data%U( i ) + val
            data%U( m + j ) =  data%U( m + j ) + val
          END DO
        END SELECT

!  compute ||A||_1 and ||A||_inf and the bound on lambda_n

        a_one = MAXVAL( data%U( 1 : m ) )
        a_inf = MAXVAL( data%U( m + 1 : data%npm ) )
!       lambda_n = SQRT( a_one * a_inf )
        lambda_n = a_one * a_inf

!  set known lower and upper bounds on lambda

        IF ( data%control%equality_problem ) THEN
          lambda_l = c_norm_over_radius - lambda_n
        ELSE
          lambda_l = MAX( zero, c_norm_over_radius - lambda_n )
        END IF
        lambda_u = c_norm_over_radius

!  ** diagonal trust-region scaling

      ELSE IF ( SMT_get( S%type ) == 'DIAGONAL' ) THEN

!  record || c ||_S^-1/ radius 

        c_norm = SQRT( DOT_PRODUCT( data%C( : n ),                             &
                                    data%C( : n )  / S%val( : n ) ) )
        c_norm_over_radius = c_norm / radius

!  compute the sum of the absolute values of each row of A S^{-1/2} in U(1:m)
!  and column of A S^{-1/2} in U(m+1:m+n)

        data%U( : data%npm ) = zero
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1 ; val = ABS( A%val( l ) / SQRT( S%val( j ) ) )
              data%U( i ) =  data%U( i ) + val
              data%U( m + j ) =  data%U( m + j ) + val
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l ) ; val = ABS( A%val( l ) / SQRT( S%val( j ) ) )
              data%U( i ) =  data%U( i ) + val
              data%U( m + j ) =  data%U( m + j ) + val
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l )
            val = ABS( A%val( l ) / SQRT( S%val( j ) ) )
            data%U( i ) =  data%U( i ) + val
            data%U( m + j ) =  data%U( m + j ) + val
          END DO
        END SELECT

!  compute ||A S^{-1/2}||_1 and ||A S^{-1/2}||_inf and the bound on lambda_n

        a_one = MAXVAL( data%U( 1 : m ) )
        a_inf = MAXVAL( data%U( m + 1 : data%npm ) )
!       lambda_n = SQRT( a_one * a_inf )
        lambda_n = a_one * a_inf

!  set known lower and upper bounds on lambda

        IF ( data%control%equality_problem ) THEN
          lambda_l = c_norm_over_radius - lambda_n
        ELSE
          lambda_l = MAX( zero, c_norm_over_radius - lambda_n )
        END IF
        lambda_u = c_norm_over_radius

!  ** general trust-region scaling

      ELSE

!  perform an analysis of the spasity pattern of S to identify a good
!  ordering for sparse factorization

        IF ( data%control%new_s >= 2 ) THEN
          data%control%SLS_control%pivot_control = 2
          CALL SLS_initialize_solver( control%definite_linear_solver,          &
                                      data%SLS_data, inform%SLS_inform )
          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SLS_analyse( S, data%SLS_data,                                  &
                            data%control%SLS_control, inform%SLS_inform )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%analyse = inform%time%analyse + time_now - time_record
          inform%time%clock_analyse =                                          &
            inform%time%clock_analyse + clock_now - clock_record
          IF ( printt ) WRITE( out, 2000 ) prefix, clock_now - clock_record
           
!  test that the analysis succeeded

          IF ( inform%SLS_inform%status < 0 ) THEN
            IF ( printi ) WRITE( out, "( A, ' error return from SLS',          &
           &  '_analyse: status = ', I0 )" ) prefix, inform%SLS_inform%status
            inform%status = GALAHAD_error_analysis ;  GO TO 910 ; END IF
        END IF

!  attempt an L B L^T factorization of S

        IF ( data%control%new_s >= 1 ) THEN
          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
          CALL SLS_factorize( S, data%SLS_data,                                &
                              data%control%SLS_control, inform%SLS_inform )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%factorize = inform%time%factorize + time_now - time_record
          inform%time%clock_factorize =                                        &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( printt ) WRITE( out, 2010 ) prefix, clock_now - clock_record
          inform%factorizations = inform%factorizations + 1

!  test that the factorization succeeded

          IF ( inform%SLS_inform%status == 0 ) THEN
            psdef = .TRUE.
          ELSE IF ( inform%SLS_inform%status == GALAHAD_error_inertia ) THEN
            GO TO 930
!           psdef = .FALSE.
          ELSE
            GO TO 920
          END IF
        END IF

!  compute S^{-1} c in V
      
        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        data%U( : n ) = data%C( : n )
        CALL IR_solve( S, data%U( : n ), data%IR_data, data%SLS_data,          &
                       data%control%IR_control, data%control%SLS_control,      &
                       inform%IR_inform, inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_record
        inform%time%clock_solve =                                              &
          inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  compute the S^-1 norm of c, while checking that S is positive definite

        c_norm = DOT_PRODUCT( data%U( : n ), data%C( : n ) )
        IF ( c_norm < zero ) GO TO 930
        c_norm = SQRT( c_norm )
        c_norm_over_radius = c_norm / radius

!  compute the sums of the absolute values of off-diagonal terms of S (in Y),
!  and its diagonal terms (in Z). Then record the Gershgorin bound on the 
!  smallest eigenvalue, which gives the reciprocal of the laregst eigenvalue
!  of S^-1

        data%Y( : n ) = zero ; data%Z( : n ) = zero
        SELECT CASE ( SMT_get( S%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              val = S%val( l )
              IF ( i == j ) THEN
                data%Z( i ) = val
              ELSE
                data%Y( i ) = data%Y( i ) + ABS( val )
                data%Y( j ) = data%Y( j ) + ABS( val )
              END IF
            END DO
          END DO
          lambda = MINVAL( data%Z( : n ) - data%Y( : n ) )
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = S%ptr( i ), S%ptr( i + 1 ) - 1
              j = S%col( l ) ; val = S%val( l )
              IF ( i == j ) THEN
                data%Z( i ) = val
              ELSE
                data%Y( i ) = data%Y( i ) + ABS( val )
                data%Y( j ) = data%Y( j ) + ABS( val )
              END IF
            END DO
          END DO
          lambda = MINVAL( data%Z( : n ) - data%Y( : n ) )
        CASE ( 'COORDINATE' )
          DO l = 1, S%ne
            i = S%row( l ) ; j = S%col( l )
            val = S%val( l )
            IF ( i == j ) THEN
              data%Z( i ) = val
            ELSE
              data%Y( i ) = data%Y( i ) + ABS( val )
              data%Y( j ) = data%Y( j ) + ABS( val )
            END IF
          END DO
          lambda = MINVAL( data%Z( : n ) - data%Y( : n ) )
        CASE( 'DIAGONAL' )
          lambda = MINVAL( S%val( : n ) )
        CASE( 'SCALED_IDENTITY' )
          lambda = S%val( 1 )
        CASE( 'IDENTITY' )
          lambda = one
        END SELECT
        IF ( lambda > zero ) THEN
          lambda_sinv = one / lambda
        ELSE
          GO TO 930
        END IF

!  compute the sum of the absolute values of each row of A in U(1:m) and 
!  column of A in U(m+1:m+n)

        data%U( : data%npm ) = zero
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' ) 
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1 ; val = ABS( A%val( l ) )
              data%U( i ) =  data%U( i ) + val
              data%U( m + j ) =  data%U( m + j ) + val
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l ) ; val = ABS( A%val( l ) )
              data%U( i ) =  data%U( i ) + val
              data%U( m + j ) =  data%U( m + j ) + val
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l ) ; val = ABS( A%val( l ) )
            data%U( i ) =  data%U( i ) + val
            data%U( m + j ) =  data%U( m + j ) + val
          END DO
        END SELECT

!  compute ||A||_1 and ||A||_inf and the bound on lambda_n

        a_one = MAXVAL( data%U( 1 : m ) )
        a_inf = MAXVAL( data%U( m + 1 : data%npm ) )
!       lambda_n = SQRT( a_one * a_inf ) * lambda_sinv
        lambda_n = a_one * a_inf * lambda_sinv

!  set known lower and upper bounds on lambda

        IF ( data%control%equality_problem ) THEN
          lambda_l = c_norm_over_radius - lambda_n
        ELSE
          lambda_l = MAX( zero, c_norm_over_radius - lambda_n )
        END IF
        lambda_u = c_norm_over_radius
      END IF

!  assign the initial lambda

      IF ( data%control%use_initial_multiplier ) THEN
        IF ( data%control%initial_multiplier >= lambda_l .AND.                 &
             data%control%initial_multiplier <= lambda_u ) THEN
          lambda =  data%control%initial_multiplier
        ELSE
          lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                   &
                        lambda_l + theta_eps * ( lambda_u - lambda_l ) )
        END IF
      ELSE
        IF ( data%control%equality_problem ) THEN
          lambda = lambda_l + theta_eps * ( lambda_u - lambda_l )
        ELSE
          IF ( lambda_l == zero ) THEN
            try_zero = .FALSE.
            lambda = zero
          ELSE
            lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                 &
                          lambda_l + theta_eps * ( lambda_u - lambda_l ) )
          END IF
        END IF
      END IF

      lambda_pert = MAX( lambda, epsmch )
      try_zero = lambda > zero .AND. lambda_l == zero
      region = ' '
      max_order = MAX( 1, MIN( max_degree, control%taylor_max_degree ) )

      IF ( printt )                                                            &
        WRITE( out, "( A, 4X, 28( '-' ), ' phase one ', 28( '-' ) )" ) prefix

!  start the main loop

      data%control%SBLS_control%preconditioner = 2
      data%control%SBLS_control%new_h = MAX( control%new_s, 1 )
      data%control%SBLS_control%new_a = control%new_a
      data%control%SBLS_control%new_c = 0
      IF ( data%control%equality_problem )                                     &
        data%control%SBLS_control%factorization = 2

      it = 0 ; in_n = 0
      DO
        it = it + 1
!if(it>10)stop

!  introduce lambda * S to form K(lambda)

        IF ( unit_s ) THEN
          data%H_sbls%val( : data%s_ne ) = lambda
        ELSE
          IF ( SMT_get( S%type ) == 'COORDINATE' .OR.                          &
               SMT_get( S%type ) == 'SPARSE_BY_ROWS' .OR.                      &
               SMT_get( S%type ) == 'DENSE' .OR.                               &
               SMT_get( S%type ) == 'DIAGONAL' .OR.                            &
               SMT_get( S%type ) == 'SCALED_IDENTITY' ) THEN
            data%H_sbls%val( : data%s_ne ) = lambda_pert * S%val( : data%s_ne )
          ELSE
            data%H_sbls%val( : data%s_ne ) = lambda_pert
          END IF
        END IF

!  attempt an L B L^T factorization of K(lambda)

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SBLS_form_and_factorize( n, m, data%H_sbls, A,                    &
                                      data%C_sbls, data%SBLS_data,             &
                                      data%control%SBLS_control,               &
                                      inform%SBLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, clock_now - clock_record
        inform%factorizations = inform%factorizations + 1

!  test that the factorization succeeded

        IF ( inform%SBLS_inform%status == 0 ) THEN
          IF ( inform%SBLS_inform%factorization == 1 ) THEN
            psdef = inform%SBLS_inform%SLS_inform%negative_eigenvalues == 0
          ELSE
            psdef = inform%SBLS_inform%SLS_inform%negative_eigenvalues == m
          END IF
        ELSE IF ( inform%SBLS_inform%status == GALAHAD_error_inertia ) THEN
write(6,*) ' lambda = ', lambda_pert, inform%SBLS_inform%status
          psdef = .FALSE.
        ELSE
write(6,*) ' lambda = ', lambda_pert, inform%SBLS_inform%status
          GO TO 920
        END IF
        data%control%SBLS_control%new_h = 1
        data%control%SBLS_control%new_a = 0

!  if K(lambda) is positive definite, solve  
!   ( lambda * S     A^T ) ( x ) = ( c )
!   (      A         - I ) ( y )   ( 0 )

        IF ( psdef ) THEN 
          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
          data%U( : n ) = data%C( : n ) ; data%U( n + 1 : data%npm ) = zero
          CALL SBLS_solve( n, m, A, data%C_sbls, data%SBLS_data,               &
                           data%control%SBLS_control, inform%SBLS_inform,      &
                           data%U( : data%npm ) )
          X = data%U( : n )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  compute the S-norm of x, ||x||_S

          IF ( unit_s ) THEN
            inform%x_norm = TWO_NORM( X ) 
            x_norm2( 0 ) = inform%x_norm ** 2
          ELSE
            CALL mop_AX( one, S, X, zero, data%Y( : n ), 0,                    &
                         symmetric = .TRUE. )
            x_norm2( 0 ) = DOT_PRODUCT( X, data%Y( : n ) )
            IF ( x_norm2( 0 ) < zero ) GO TO 930
            inform%x_norm = SQRT( x_norm2( 0 ) )
          END IF

!  compute the two-norm of the residual, ||A x - b||_2 = ||y - b||_2

          inform%r_norm = TWO_NORM( B - data%U( n + 1 : data%npm ) )

!  reset the interval bounds (the secular equation is irrelevant)

          IF ( inform%x_norm <= epsmch ) THEN
            lambda_l = MAX( data%control%lower, lambda_l )
            lambda_u = MIN( data%control%upper, lambda_u )

            IF ( lambda_l == zero .AND. lambda_u == zero ) THEN
              lambda = zero
              IF ( printi ) THEN
                WRITE( out, 2020 ) prefix
                WRITE( out, "( A, A2, I4, 3ES22.15 )" )                        &
                  prefix, ' ', 0, lambda_l, lambda, lambda_u
                WRITE( out, "( A,                                              &
               &    ' Stopping criteria satisfied.',                           &
               &    ' Interval width =', ES11.4 )" ) prefix, lambda_u - lambda_l
              END IF
              inform%status = 0
              GO TO 900
            END IF

            region = 'G'
            lambda_u = MIN( lambda_u, lambda )
            IF ( .NOT. phase_1 ) THEN
              phase_1 = .TRUE.
              IF ( printi ) WRITE( out, 2020 ) prefix
            ELSE
              IF ( printh ) WRITE( out, 2020 ) prefix
            END IF
            IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" )              &
              prefix, region, it, lambda_l, lambda, lambda_u
            n_lambda = 0
            GO TO 200
          END IF

!  debug printing

          IF ( printd ) THEN
            WRITE( out, "( A, 8X, 'lambda', 13X, 'x_norm', 15X, 'radius' )" )  &
              prefix
            WRITE( out, "( A, 3ES20.12 )") prefix, lambda, inform%x_norm, radius
            IF ( phase_1 ) THEN
              WRITE( out, "( A, ' interval width =', ES21.14 )")               &
                prefix, lambda_u - lambda_l
            ELSE
              WRITE( out, 2020 ) prefix
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )                          &
                prefix, '  ', it, lambda_l, lambda, lambda_u
            END IF
          END IF

!  ---------------------------------------------------------------------------
!  The interval is small and the current estimate lies within the trust-region
!  ---------------------------------------------------------------------------

          IF ( lambda_u - lambda_l <= epsmch .AND. inform%x_norm - radius <=   &
                 data%control%stop_normal * MAX( one, radius ) ) THEN
            IF ( ( phase_1 .AND. printi ) .OR. printh )                        &
              WRITE( out, 2030 ) prefix
            IF ( printi ) THEN
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )  prefix, region,         &
              it, inform%x_norm - radius, lambda, ABS( delta_lambda )
              WRITE( out, "( A,                                                &
           &    ' Normal stopping criteria satisfied' )" ) prefix
            END IF
            inform%status = 0
            EXIT
          END IF

!  --------------------------------------------------------------------
!  The current estimate gives a good approximation to the required root
!  --------------------------------------------------------------------

          IF ( ABS( inform%x_norm - radius ) <=                                &
                 data%control%stop_normal * MAX( one, radius ) ) THEN
            IF ( inform%x_norm > radius ) THEN
              region = 'L'
              lambda_l = MAX( lambda_l, lambda )
            ELSE
              region = 'G'
              lambda_u = MIN( lambda_u, lambda )
            END IF
            IF ( ( phase_1 .AND. printi ) .OR. printh )                        &
              WRITE( out, 2030 ) prefix
            IF ( printi ) THEN
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )  prefix, region,         &
              it, ABS( inform%x_norm - radius ), lambda, ABS( delta_lambda )
              WRITE( out, "( A,                                                &
           &    ' Normal stopping criteria satisfied' )" ) prefix
            END IF
            inform%status = 0
            EXIT
          END IF

!  check to see if the factorization limit has been exceeded

          IF ( inform%factorizations > data%control%max_factorizations ) THEN
            inform%multiplier = lambda
            inform%status = GALAHAD_error_max_iterations ; GO TO 910
          END IF
!         write(6,*) ' ||x||-radius = ', inform%x_norm - radius

!  determine which region the current lambda lies in

!  ----------------------------
!  The current lambda lies in L
!  ----------------------------
            
          IF ( inform%x_norm > radius ) THEN
            region = 'L'
            lambda_l = MAX( lambda_l, lambda )

!  record that we are now in phase 2

            IF ( phase_1 ) THEN
              phase_1 = .FALSE.
              delta_lambda = zero
              IF ( printd ) THEN
                WRITE( out, 2020 ) prefix
                WRITE( out, "( A, A2, I4, 3ES22.15 )" )                        &
                  prefix, region, it, lambda_l, lambda, lambda_u
              END IF
              IF ( printt ) THEN
                WRITE( out, "( A, 4X, 28( '-' ), ' phase two ', 28( '-' ) )" ) &
                  prefix
              END IF
              IF ( printi ) WRITE( out, 2030 ) prefix
            ELSE
              IF ( printh ) WRITE( out, 2030 ) prefix
            END IF

!  a lambda in L has been found. It is now simply a matter of applying 
!  a variety of Taylor-series-based methods starting from this lambda

            IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" ) prefix,      &
              region, it, ABS( inform%x_norm - radius ), lambda,               &
              ABS( delta_lambda )

!  precaution against rounding producing lambda outside L

            IF ( lambda > lambda_u ) THEN
              inform%status = GALAHAD_error_ill_conditioned
              EXIT
            END IF

!  ----------------------------
!  The current lambda lies in G
!  ----------------------------

          ELSE
            region = 'G'
            lambda_u = MIN( lambda_u, lambda )
            IF ( .NOT. phase_1 ) THEN
              phase_1 = .TRUE.
              IF ( printi ) WRITE( out, 2020 ) prefix
            ELSE
              IF ( printh ) WRITE( out, 2020 ) prefix
            END IF
            IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" )              &
              prefix, region, it, lambda_l, lambda, lambda_u

!  -----------------------------------------------------
!  The solution lies in the interior of the trust-region 
!  -----------------------------------------------------

            IF ( lambda == zero .AND. .NOT. data%control%equality_problem ) THEN
              IF ( printi ) WRITE( out, "( A,                                  &
             &    ' Interior stopping criteria satisfied' )" ) prefix
              inform%status = 0
              GO TO 900
            END IF

!  record, for the future, values of lambda which give small ||x||

            IF ( inform%len_history < history_max ) THEN
              inform%len_history = inform%len_history + 1
              inform%history( inform%len_history )%lambda = lambda
              inform%history( inform%len_history )%x_norm = inform%x_norm 
            END IF
          END IF

!  compute first derivatives of x^T S x

          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = S x

          IF ( data%accurate ) THEN
            IF ( unit_s ) data%Y( : n ) = X
            data%Z( : n ) = data%Y( : n ) ; data%Z( n + 1 : data%npm ) = zero
            CALL SBLS_solve( n, m, A, data%C_sbls, data%SBLS_data,             &
                             data%control%SBLS_control, inform%SBLS_inform,    &
                             data%Z( : data%npm ) )

!  find y so that L y = S x

          ELSE
            IF ( unit_s ) data%Y( : n ) = X
            data%Y( n + 1 : data%npm ) = zero
            CALL SBLS_part_solve( 'L', n, m, A, data%SBLS_data,                &
                                  data%control%SBLS_control,                   &
                                  inform%SBLS_inform, data%Y( : data%npm ) )

!  find z so that D L z = D y = S x

            data%Z( : n ) = data%Y( : n ) ; data%Z( n + 1 : data%npm ) = zero
            CALL SBLS_part_solve( 'D', n, m, A, data%SBLS_data,                &
                                  data%control%SBLS_control,                   &
                                  inform%SBLS_inform, data%Z( : data%npm ) )
          END IF
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  form ||w||^2 = y^T z = x^T L^-T D^-1 L^-1 x = x^T H^-1(lambda) x

          w_norm2 = DOT_PRODUCT( data%Y( : n ), data%Z( : n ) )

!  compute the first derivative of x_norm2 = x^T S x

          x_norm2( 1 ) = - two * w_norm2

!  compute pi_beta = ||x||^beta and its first derivative when beta = - 1

          beta = - one
          CALL LLST_pi_derivs( 1, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = - 1)

          delta_lambda = - ( pi_beta( 0 ) - ( radius ) ** beta ) / pi_beta( 1 )
          n_lambda = 1
          lambda_new( n_lambda ) = lambda + delta_lambda

!  compute x' where H(lambda) x' = - S x ; - S x' is in data%z

          IF ( ( max_order >= 3 .AND. region == 'L' ) .OR.                     &
               ( max_order >= 2 .AND. region == 'G' ) ) THEN
            IF ( .NOT. data%accurate ) THEN
              CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
              CALL SBLS_part_solve( 'U', n, m, A, data%SBLS_data,              &
                                    data%control%SBLS_control,                 &
                                    inform%SBLS_inform, data%Z( : data%npm ) )
              CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
              inform%time%solve = inform%time%solve + time_now - time_record
              inform%time%clock_solve =                                        &
                inform%time%clock_factorize + clock_now - clock_record
              IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record
            END IF

!  form z^T S z

            IF ( unit_s ) THEN
              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Z( : n ) )
            ELSE
              CALL mop_AX( one, S, data%Z( : n ), zero, data%Y( : n ), 0,      &
                           symmetric = .TRUE. )
              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )
            END IF

!  compute the second derivative of x_norm2 = x^T S x

            x_norm2( 2 ) = six * z_norm2

            IF ( max_order >= 3 ) THEN
              CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = S x

              IF ( data%accurate ) THEN
                IF ( unit_s ) THEN
                  data%Y( : n ) = data%Z( : n )
                ELSE
                  data%Z( : n ) = data%Y( : n )
                END IF
                data%Z( n + 1 : data%npm ) = zero
                CALL SBLS_solve( n, m, A, data%C_sbls, data%SBLS_data,         &
                                 data%control%SBLS_control, inform%SBLS_inform,&
                                 data%Z( : data%npm ) )
!  find z so that L z = x'

              ELSE
                IF ( unit_s ) THEN
                  data%Z( n + 1 : data%npm ) = zero
                  CALL SBLS_part_solve( 'L', n, m, A, data%SBLS_data,          &
                                        data%control%SBLS_control,             &
                                        inform%SBLS_inform,                    &
                                        data%Z( : data%npm ) )

!  find y so that D L y = D z = x'

                  data%Y( : n ) = data%Z( : n )
                  data%Y( n + 1 : data%npm ) = zero
                  CALL SBLS_part_solve( 'D', n, m, A, data%SBLS_data,          &
                                        data%control%SBLS_control,             &
                                        inform%SBLS_inform,                    &
                                        data%Y( : data%npm ) )

!  find z so that L y = S x'

                ELSE
                  data%Y( n + 1 : data%npm ) = zero
                  CALL SBLS_part_solve( 'L', n, m, A, data%SBLS_data,          &
                                        data%control%SBLS_control,             &
                                        inform%SBLS_inform,                    &
                                        data%Y( : data%npm ) )

!  find y so that D L z = D y = x'

                  data%Z( : n ) = data%Y( : n )
                  data%Z( n + 1 : data%npm ) = zero
                  CALL SBLS_part_solve( 'D', n, m, A, data%SBLS_data,          &
                                        data%control%SBLS_control,             &
                                        inform%SBLS_inform,                    &
                                        data%Z( : data%npm ) )
                END IF
              END IF
              CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
              inform%time%solve = inform%time%solve + time_now - time_record
              inform%time%clock_solve =                                        &
                inform%time%clock_factorize + clock_now - clock_record
              IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  form ||v||^2 = z^T y = x'^T L^-T D^-1 L^-1 x' = x'^T H^-1(lambda) x'

              v_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )

!  compute the third derivatives of x_norm2 = x^T S x

              x_norm2( 3 ) = - twentyfour * v_norm2
            END IF
          END IF

!  compute pi_beta = ||x||^beta and its derivatives for various beta
!  and the resulting Taylor series approximants

 200      CONTINUE

!  ----------------------------
!  The current lambda lies in L
!  ----------------------------

          IF ( inform%x_norm > radius ) THEN
            IF ( max_order >= 3 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = 2

              beta = two
              CALL LLST_pi_derivs( max_order, beta, x_norm2( : max_order ),    &
                                  pi_beta( : max_order ) )

!  compute the "cubic Taylor approximaton" step (beta = 2)

              CALL ROOTS_cubic( pi_beta( 0 ) - ( radius ) ** beta,             &
                                pi_beta( 1 ), half * pi_beta( 2 ),             &
                                sixth * pi_beta( 3 ), roots_tol,               &
                                nroots, root1, root2, root3, roots_debug )
              n_lambda = n_lambda + 1
              IF ( nroots == 3 ) THEN
                lambda_new( n_lambda ) = lambda + root3
              ELSE
                lambda_new( n_lambda ) = lambda + root1
              END IF

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.4

              beta = - point4
              CALL LLST_pi_derivs( max_order, beta, x_norm2( : max_order ),    &
                                  pi_beta( : max_order ) )

!  compute the "cubic Taylor approximaton" step (beta = - 0.4)

              CALL ROOTS_cubic( pi_beta( 0 ) - ( radius ) ** beta,             &
                                pi_beta( 1 ), half * pi_beta( 2 ),             &
                                sixth * pi_beta( 3 ), roots_tol,               &
                                nroots, root1, root2, root3, roots_debug )
              n_lambda = n_lambda + 1
              IF ( nroots == 3 ) THEN
                lambda_new( n_lambda ) = lambda + root3
              ELSE
                lambda_new( n_lambda ) = lambda + root1
              END IF
            END IF

!  compute the new estimate of lambda

            IF ( printd ) WRITE( out, "( A, ' lambda_t', 3ES21.13 )" )         &
               prefix, lambda_new( : n_lambda )

!  compute the best Taylor improvement

            lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
            delta_lambda = lambda_plus - lambda
            lambda = lambda_plus
            lambda_pert = lambda

!  improve the lower bound if possible

            lambda_l = MAX( lambda_l, lambda_plus )
 
!  check that the best Taylor improvement is significant

            IF ( ABS( delta_lambda ) < epsmch * MAX( one, ABS( lambda ) ) ) THEN
              inform%status = 0
              EXIT
            END IF

!  ----------------------------
!  The current lambda lies in G
!  ----------------------------

          ELSE
            IF ( max_order >= 2 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.666

              beta = - twothirds
              CALL LLST_pi_derivs( 2, beta, x_norm2( : 2 ), pi_beta( : 2 ) )

!  compute the "quadratic Taylor approximaton" step (beta = - 0.666)

              CALL ROOTS_quadratic( pi_beta( 0 ) - ( radius ) ** beta,         &
                                    pi_beta( 1 ), half * pi_beta( 2 ),         &
                                    roots_tol, nroots, root1, root2,           &
                                    roots_debug )
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda + root1


              IF ( max_order >= 3 ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = - 0.4

                beta = - point4
                CALL LLST_pi_derivs( max_order, beta, x_norm2( : max_order ),  &
                                     pi_beta( : max_order ) )

!  Compute the "cubic Taylor approximaton" step (beta = - 0.4)

                CALL ROOTS_cubic( pi_beta( 0 ) - ( radius ) ** beta,           &
                     pi_beta( 1 ), half * pi_beta( 2 ), sixth * pi_beta( 3 ),  &
                     roots_tol,  nroots, root1, root2, root3, roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda + root1
              END IF
            END IF

!  record all of the estimates of the optimal lambda

            IF ( printd ) WRITE( out, "( A, ' lambda_t', 3ES21.13 )" )         &
              prefix, lambda_new( : n_lambda )

!  compute the best Taylor improvement

            lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
            delta_lambda = lambda_plus - lambda

!  improve the lower bound if possible

            lambda_l = MAX( lambda_l, lambda_plus )

!  if lambda = 0 hasn't yet been tried, do so

            IF ( try_zero ) THEN
              try_zero = .FALSE.
              IF ( MAXVAL( lambda_new( : n_lambda ) ) < zero ) THEN
                lambda = zero
                lambda_pert = epsmch
!               lambda_pert = ten * epsmch
                CYCLE
              END IF
            END IF

!  check that the best Taylor improvement is significant

            IF ( ABS( delta_lambda ) < epsmch * MAX( one, ABS( lambda ) ) ) THEN
              inform%status = 0
              EXIT
            END IF

            IF ( lambda_plus >= lambda_l ) THEN
              lambda = lambda_plus
            ELSE
              IF ( data%control%equality_problem ) THEN
                lambda = lambda_l + theta_eps * ( lambda_u - lambda_l )
              ELSE
                lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),             &
                              lambda_l + theta_eps * ( lambda_u - lambda_l ) )
              END IF
            END IF
            lambda_pert = lambda
          END IF

!  ----------------------------
!  The current lambda lies in N
!  ----------------------------

        ELSE
          IF ( printh ) WRITE( out, 2020 ) prefix
          region = 'N'
          IF ( printi ) WRITE( out, "( A, A2, I4, 3ES22.15 )" )                &
            prefix, region, it, lambda_l, lambda, lambda_u
          try_zero = .FALSE.
          lambda_l = lambda

!  compute the next lambda

          width = ABS( lambda_u - lambda_l )
          IF ( data%control%equality_problem ) THEN
            lambda = lambda_l + theta_n * width
          ELSE
            in_n = in_n + 1
            IF ( MOD( in_n, 2 ) == 1 ) THEN
              lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),               &
                            lambda_l + theta_n * width )
            ELSE
              lambda = lambda_l + theta_n_small * width
            END IF
          END IF
          lambda_pert = lambda
        END IF

        printh = printt .OR. printi .AND.                                     &
                   ( control%SBLS_control%print_level > 0 .OR.                &
                     control%SLS_control%print_level > 0 .OR.                 &
                     control%IR_control%print_level > 0 )

!  End of main iteration loop

      END DO

!  ----
!  Exit
!  ----

 900  CONTINUE
      inform%multiplier = lambda
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  -------------
!  General error
!  -------------

  910 CONTINUE
      IF ( printi ) WRITE( out, "( A, '   **  Error return ', I0,              &
     & ' from LLST ' )" ) prefix, inform%status 
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  ---------------------
!  Factorization failure
!  ---------------------

  920 CONTINUE
      IF ( printi ) WRITE( out, "( A, ' error return from ',                   &
     &   'SBLS_factorize: status = ', I0 )" ) prefix, inform%SBLS_inform%status
      inform%status = GALAHAD_error_factorization 
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  -----------------
!  Indefinite S-norm
!  -----------------

  930 CONTINUE
      IF ( printi ) WRITE( out,                                                &
         "( A, ' The matrix S provided for LLST appears not to be strictly ',  &
        &      ' diagonally dominant'  )" ) prefix
      inform%status = GALAHAD_error_preconditioner
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

! Non-executable statements

 2000 FORMAT( A, ' time( SBLS_analyse ) = ', F0.2 )
 2010 FORMAT( A, ' time( SBLS_factorize ) = ', F0.2 )
 2020 FORMAT( A, '    it        lambda_l                lambda ',              &
                 '               lambda_u' )
 2030 FORMAT( A, '    it     ||x||_S-radius             lambda ',              &
                 '               d_lambda' )
 2050 FORMAT( A, ' time( SBLS_solve ) = ', F0.2 )

!  End of subroutine LLST_solve

      END SUBROUTINE LLST_solve

!-*-*-*-*-*-  L L S T _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LLST_terminate( data, control, inform )

!  ...........................................
!  .                                         .
!  .  Deallocate arrays at end of LLST_solve  .
!  .                                         .
!  ...........................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine LLST_initialize
!   inform    see Subroutine LLST_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LLST_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLST_control_type ), INTENT( IN ) :: control        
      TYPE ( LLST_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'llst: S_diag'
      CALL SPACE_dealloc_array( data%S_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: S_offd'
      CALL SPACE_dealloc_array( data%S_offd,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: Z'
      CALL SPACE_dealloc_array( data%Z,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: H_sbls%row'
      CALL SPACE_dealloc_array( data%H_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: H_sbls%col'
      CALL SPACE_dealloc_array( data%H_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: H_sbls%ptr'
      CALL SPACE_dealloc_array( data%H_sbls%ptr,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llst: H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within IR

      CALL IR_terminate( data%IR_data, control%IR_control,                     &
                          inform%IR_inform )
      IF ( inform%IR_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'llst: IR_data'
      END IF

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'llst: SLS_data'
      END IF

!  Deallocate all arrays allocated within SBLS

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                          inform%SBLS_inform )
      IF ( inform%SBLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'llst: SBLS_data'
      END IF

      RETURN

!  End of subroutine LLST_terminate

      END SUBROUTINE LLST_terminate

!-*-*-*-*-*-*-  L L S T _ P I _ D E R I V S   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LLST_pi_derivs( max_order, beta, x_norm2, pi_beta )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute pi_beta = ||x||^beta and its derivatives
!
!  Arguments:
!  =========
!
!  Input - 
!   max_order - maximum order of derivative
!   beta - power 
!   x_norm2 - (0) value of ||x||^2, 
!             (i) ith derivative of ||x||^2, i = 1, max_order
!  Output - 
!   pi_beta - (0) value of ||x||^beta, 
!             (i) ith derivative of ||x||^beta, i = 1, max_order
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: max_order
      REAL ( KIND = wp ), INTENT( IN ) :: beta, x_norm2( 0 : max_order )
      REAL ( KIND = wp ), INTENT( OUT ) :: pi_beta( 0 : max_order )

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      REAL ( KIND = wp ) :: hbeta

      hbeta = half * beta
      pi_beta( 0 ) = x_norm2( 0 ) ** hbeta
      pi_beta( 1 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - one ) ) * x_norm2( 1 )
      IF ( max_order == 1 ) RETURN
      pi_beta( 2 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - two ) ) *             &
        ( ( hbeta - one ) * x_norm2( 1 ) ** 2 + x_norm2( 0 ) * x_norm2( 2 ) )
      IF ( max_order == 2 ) RETURN
      pi_beta( 3 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - three ) ) *           &
        ( x_norm2( 3 ) * x_norm2( 0 ) ** 2 + ( hbeta - one ) *                 &
          ( three * x_norm2( 0 ) * x_norm2( 1 ) * x_norm2( 2 ) +               &
            ( hbeta - two ) * x_norm2( 1 ) ** 3 ) ) 

      RETURN

!  End of subroutine LLST_pi_derivs

      END SUBROUTINE LLST_pi_derivs

!-*-*-*-*-  End of G A L A H A D _ L L S T  d o u b l e  M O D U L E  -*-*-*-*-

    END MODULE GALAHAD_LLST_double
