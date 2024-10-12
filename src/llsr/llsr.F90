! THIS VERSION: GALAHAD 5.1 - 2024-10-04 AT 14:00 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-  G A L A H A D _ L L S R  d o u b l e  M O D U L E  *-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released GALAHAD Version 4.1. June 5th, 2023

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LLSR_precision

!       ------------------------------------------------------------
!      |                                                            |
!      | Solve the regularized linear least-squares subproblem      |
!      |                                                            |
!      |   minimize    1/2 || A x - b ||^2  + sigma / p ||x||_S^p,  |
!      |                                                            |
!      | where ||x||_S^2 = <x, Sx> and S is diagonally dominant,    |
!      | using a sparse matrix factorization                        |
!      |                                                            |
!       ------------------------------------------------------------

      USE GALAHAD_KINDS_precision
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_precision
      USE GALAHAD_RAND_precision
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_ROOTS_precision, ONLY: ROOTS_quadratic, ROOTS_cubic
      USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
      USE GALAHAD_SBLS_precision
      USE GALAHAD_SLS_precision
      USE GALAHAD_IR_precision
      USE GALAHAD_MOP_precision

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LLSR_initialize, LLSR_read_specfile, LLSR_solve,               &
                LLSR_terminate,  LLSR_full_initialize, LLSR_full_terminate,    &
                LLSR_import, LLSR_import_scaling, LLSR_reset_control,          &
                LLSR_solve_problem, LLSR_information,                          &
                SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

      INTERFACE LLSR_initialize
        MODULE PROCEDURE LLSR_initialize, LLSR_full_initialize
      END INTERFACE LLSR_initialize

      INTERFACE LLSR_terminate
        MODULE PROCEDURE LLSR_terminate, LLSR_full_terminate
      END INTERFACE LLSR_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER ( KIND = ip_ ), PARAMETER :: history_max = 100
      INTEGER ( KIND = ip_ ), PARAMETER :: max_degree = 3
      INTEGER ( KIND = ip_ ), PARAMETER :: it_stalled = 100
      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: point1 = 0.1_rp_
      REAL ( KIND = rp_ ), PARAMETER :: point01 = 0.01_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: point4 = 0.4_rp_
      REAL ( KIND = rp_ ), PARAMETER :: point9 = 0.9_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: three = 3.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: six = 6.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: sixth = one / six
      REAL ( KIND = rp_ ), PARAMETER :: twothirds = two /three
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: twentyfour = 24.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = half * HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = rp_ ), PARAMETER :: teneps = ten * epsmch

      REAL ( KIND = rp_ ), PARAMETER :: lambda_pert = epsmch ** 0.75
      REAL ( KIND = rp_ ), PARAMETER :: theta_ii = one
      REAL ( KIND = rp_ ), PARAMETER :: theta_eps = point01
      REAL ( KIND = rp_ ), PARAMETER :: theta_eps5 = point1
      REAL ( KIND = rp_ ), PARAMETER :: theta_g = half
      REAL ( KIND = rp_ ), PARAMETER :: theta_n = half
      REAL ( KIND = rp_ ), PARAMETER :: theta_n_small = ten ** ( - 1 )
      REAL ( KIND = rp_ ), PARAMETER :: theta_n_tiny = ten ** ( - 4 )
      REAL ( KIND = rp_ ), PARAMETER :: gamma_eps = half
      REAL ( KIND = rp_ ), PARAMETER :: gamma = one
      REAL ( KIND = rp_ ), PARAMETER :: roots_tol = teneps
      LOGICAL :: roots_debug = .FALSE.

!--------------------------
!  Derived type definitions
!--------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LLSR_control_type

!  unit for error messages

        INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

        INTEGER ( KIND = ip_ ) :: out = 6

!  controls level of diagnostic output

        INTEGER ( KIND = ip_ ) :: print_level = 0

!  how much of A has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER ( KIND = ip_ ) :: new_a = 2

!  how much of S has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER ( KIND = ip_ ) :: new_s = 2

!  the maximum number of factorizations (=iterations) allowed. -ve => no limit

        INTEGER ( KIND = ip_ ) :: max_factorizations = - 1

!  maximum degree of Taylor approximant allowed (<= 3)

        INTEGER ( KIND = ip_ ) :: taylor_max_degree = 3

!  initial estimate of the Lagrange multipler

        REAL ( KIND = rp_ ) :: initial_multiplier = zero

!  lower and upper bounds on the multiplier, if known

        REAL ( KIND = rp_ ) :: lower = - half * HUGE( one )
        REAL ( KIND = rp_ ) :: upper =  HUGE( one )

!  stop when | ||x|| - (multiplier/sigma)^(1/(p-2)) | <=
!              stop_normal * max( 1, ||x|| )

        REAL ( KIND = rp_ ) :: stop_normal = epsmch

!  ignore initial_multiplier?

        LOGICAL :: use_initial_multiplier = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal = .FALSE.

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

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LLSR_time_type

!  total CPU time spent in the package

        REAL ( KIND = rp_ ) :: total = 0.0

!  CPU time assembling K(lambda) = ( lambda * S   A^T )
!                                  (      A       - I )

        REAL ( KIND = rp_ ) :: assemble = 0.0

!  CPU time spent analysing K(lambda)

        REAL ( KIND = rp_ ) :: analyse = 0.0

!  CPU time spent factorizing K(lambda)

        REAL ( KIND = rp_ ) :: factorize = 0.0

!  CPU time spent solving linear systems inolving K(lambda)

        REAL ( KIND = rp_ ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  clock time assembling K(lambda)

        REAL ( KIND = rp_ ) :: clock_assemble = 0.0

!  clock time spent analysing K(lambda)

        REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  clock time spent factorizing K(lambda)

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving K(lambda)

        REAL ( KIND = rp_ ) :: clock_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - - -
!   history derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LLSR_history_type

!  value of lambda

        REAL ( KIND = rp_ ) :: lambda = zero

!  corresponding value of ||x(lambda)||_S

        REAL ( KIND = rp_ ) :: x_norm = zero

!  corresponding value of ||A x(lambda) - b||_2

        REAL ( KIND = rp_ ) :: r_norm = zero
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LLSR_inform_type

!   reported return status:
!      0 the solution has been found
!     -1 an array allocation has failed
!     -2 an array deallocation has failed
!     -3 n and/or Delta is not positive
!    -10 the factorization of K(lambda) failed
!    -15 S does not appear to be strictly diagonally dominant
!    -16 ill-conditioning has prevented furthr progress

        INTEGER ( KIND = ip_ ) :: status = 0

!  STAT value after allocate failure

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!   the number of factorizations performed

        INTEGER ( KIND = ip_ ) :: factorizations = 0

!  the number of (||x||_S,lambda) pairs in the history

        INTEGER ( KIND = ip_ ) :: len_history = 0

!  corresponding value of the two-norm of the residual, ||A x(lambda) - b||_2

        REAL ( KIND = rp_ ) :: r_norm = zero

!  the S-norm of x, ||x||_S

        REAL ( KIND = rp_ ) :: x_norm = zero

!  the Lagrange multiplier corresponding to the regularization

        REAL ( KIND = rp_ ) :: multiplier = zero

!  name of array which provoked an allocate failure

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  time information

        TYPE ( LLSR_time_type ) :: time

!  history information

        TYPE ( LLSR_history_type ), DIMENSION( history_max ) :: history

!  information from the symmetric factorization and related linear solves

        TYPE ( SBLS_inform_type ) :: SBLS_inform

!  information from the factorization of S and related linear solves

        TYPE ( SLS_inform_type ) :: SLS_inform

!  information from the iterative refinement for definite system solves

        TYPE ( IR_inform_type ) :: IR_inform
      END TYPE

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: LLSR_data_type
!       PRIVATE
        INTEGER ( KIND = ip_ ) :: m, npm, s_ne, a_ne, m_end
        TYPE ( RAND_seed ) :: seed
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: D
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C, U, Y, Z
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: S_diag, S_offd
        TYPE ( SMT_type ) :: H_sbls, C_sbls
        TYPE ( IR_data_type ) :: IR_data
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( SBLS_data_type ) :: SBLS_data
        TYPE ( SLS_control_type ) :: SLS_control
        TYPE ( LLSR_control_type ) :: control
      END TYPE

      TYPE, PUBLIC :: LLSR_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( LLSR_data_type ) :: LLSR_data
        TYPE ( LLSR_control_type ) :: LLSR_control
        TYPE ( LLSR_inform_type ) :: LLSR_inform
        TYPE ( SMT_type ) :: A, S
        LOGICAL :: use_s
      END TYPE LLSR_full_data_type

    CONTAINS

!-*-*-*-*-*-  L L S R _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LLSR_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
!
!  .  Set initial values for the LLSR control parameters  .
!
!  Arguments:
!  =========
!
!   data     private internal data
!   control  a structure containing control information. See LLSR_control_type
!   data     private internal data
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( LLSR_DATA_TYPE ), INTENT( INOUT ) :: data
      TYPE ( LLSR_CONTROL_TYPE ), INTENT( OUT ) :: control
      TYPE ( LLSR_inform_type ), INTENT( OUT ) :: inform

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

!  End of subroutine LLSR_initialize

      END SUBROUTINE LLSR_initialize

! G A L A H A D - L L S R _ F U L L _ I N I T I A L I Z E   S U B R O U T I N E

      SUBROUTINE LLSR_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for LLSR controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLSR_control_type ), INTENT( OUT ) :: control
      TYPE ( LLSR_inform_type ), INTENT( OUT ) :: inform

      CALL LLSR_initialize( data%llsr_data, control, inform )

      RETURN

!  End of subroutine LLSR_full_initialize

      END SUBROUTINE LLSR_full_initialize

!-*-*-*-*   L L S R _ R E A D _ S P E C F I L E  S U B R O U T I N E   *-*-*-*-

      SUBROUTINE LLSR_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LLSR_initialize could (roughly)
!  have been set as:

!  BEGIN LLSR SPECIFICATIONS (DEFAULT)
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
!   use-initial-multiplier                         F
!   space-critical                                 F
!   deallocate-error-fatal                         F
!   definite-linear-equation-solver                sils
!   output-line-prefix                             ""
!  END LLSR SPECIFICATIONS

!  Dummy arguments

      TYPE ( LLSR_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: new_a = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: new_s = new_a + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: max_factorizations = new_s + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: taylor_max_degree                   &
                                            = max_factorizations + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: initial_multiplier                  &
                                            = taylor_max_degree + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lower = initial_multiplier + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: upper = lower + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_normal = upper + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: use_initial_multiplier              &
                                            = stop_normal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical                      &
                                            = use_initial_multiplier + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                            = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: definite_linear_solver              &
                                            = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = definite_linear_solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'LLSR'
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

      END SUBROUTINE LLSR_read_specfile

!-*-*-*-*-*-*-*-*-*-*  L L S R _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE LLSR_solve( m, n, power, weight, A, B, X, data,               &
                             control, inform, S )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!  =========
!
!   m - the number of rows of A and entries in b
!
!   n - the number of colunds of A and entries in x
!
!   power - power of the regularization, p
!   weight - the regularization weight, sigma
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
!   control - a structure containing control information. See LLSR_control_type
!
!   inform - a structure containing information. See LLSR_inform_type
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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n
      REAL ( KIND = rp_ ), INTENT( IN ) :: power, weight
      TYPE ( SMT_type ), INTENT( IN ) :: A
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( m ) :: B
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( LLSR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLSR_control_type ), INTENT( IN ) :: control
      TYPE ( LLSR_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SMT_type ), OPTIONAL, INTENT( INOUT ) :: S

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: i, j, l, it, itt, out, nroots, print_level
      INTEGER ( KIND = ip_ ) :: max_order, n_lambda, in_n
      REAL :: time_start, time_now, time_record
      REAL ( KIND = rp_ ) :: clock_start, clock_now, clock_record, lambda_n
      REAL ( KIND = rp_ ) :: lambda, lambda_l, lambda_u, delta_lambda
      REAL ( KIND = rp_ ) :: c_norm, sigma_c_norm_pm2, v_norm2, w_norm2, val
      REAL ( KIND = rp_ ) :: beta, z_norm2
      REAL ( KIND = rp_ ) :: width, lambda_plus, a_one, a_inf, lambda_sinv
      REAL ( KIND = rp_ ) :: p, sigma, oos, oos2, pm2, oopm2, target
      REAL ( KIND = rp_ ) :: a_0, a_1, a_2, a_3, a_max
      REAL ( KIND = rp_ ), DIMENSION( 3 ) :: roots
      REAL ( KIND = rp_ ), DIMENSION( 9 ) :: lambda_new
      REAL ( KIND = rp_ ), DIMENSION( 0 : max_degree ) :: x_norm2
      REAL ( KIND = rp_ ), DIMENSION( 0 : max_degree ) :: pi_beta, theta_beta
      LOGICAL :: printi, printt, printd, printh, psdef, try_zero, unit_s
      LOGICAL :: phase_1
      CHARACTER ( LEN = 1 ) :: region
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

      p = power ; sigma = weight

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

!  reccord useful constants

      oos = one / sigma ; oos2 = oos * oos

!  check for obvious errors

      IF ( n <= 0 ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
           "( A, ' n = ', I0, ' is too small ' )" ) prefix, n
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF

      IF ( power < two ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( A, ' The power ', ES12.4,                   &
         &  ' is smaller than two ' )" ) prefix, power
        inform%status = GALAHAD_error_restrictions
        GO TO 910
      END IF

      IF ( weight <= zero ) THEN
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
          "( A, ' The weight ', ES12.4, ' is not positive ' )" ) prefix, weight
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
            array_name = 'llsr: H_sbls%row'
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
            array_name = 'llsr: H_sbls%col'
            CALL SPACE_resize_array( data%s_ne, data%H_sbls%col,               &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910
            data%H_sbls%col( : data%s_ne ) = S%col( : data%s_ne )
          END IF

          IF ( SMT_get( S%type ) == 'SPARSE_BY_ROWS' ) THEN
            array_name = 'llsr: H_sbls%ptr'
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
            array_name = 'llsr: H_sbls%val'
            CALL SPACE_resize_array( data%s_ne, data%H_sbls%val,               &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 910
          END IF

!  special case for H = lambda I when S is not provided

        ELSE
          array_name = 'llsr: H_sbls%val'
          CALL SPACE_resize_array( data%s_ne, data%H_sbls%val,                 &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= 0 ) GO TO 910
          CALL SMT_put( data%H_sbls%type, 'SCALED_IDENTITY',                   &
                        inform%alloc_status )
        END IF

!  make space for C = I

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

!  allocate C and Y

      array_name = 'llsr: C'
      CALL SPACE_resize_array( n, data%C,                                      &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'llsr: Y'
      CALL SPACE_resize_array( data%npm, data%Y,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

!  allocate U and Z if necessary

      array_name = 'llsr: U'
      CALL SPACE_resize_array( data%npm, data%U,                             &
        inform%status, inform%alloc_status, array_name = array_name,         &
        deallocate_error_fatal = control%deallocate_error_fatal,             &
        exact_size = control%space_critical,                                 &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

!  set c = A^T b

      CALL mop_AX( one, A, b, zero, data%C( : n ), transpose = .TRUE. )

      IF ( p > two ) THEN
        pm2 = p - two ; oopm2 = one / pm2

        array_name = 'llsr: Z'
        CALL SPACE_resize_array( data%npm, data%Z,                             &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

!  the real line is partitioned into disjoint sets
!     N = { lambda: lambda <= max(0, -lambda_1(H))}
!     L = { lambda: max(0, -lambda_1(H)) < lambda <= lambda_optimal } and
!     G = { lambda: lambda > lambda_optimal }
!  The aim is to find a lambda in L, as generally then Newton's method
!  will converge both globally and ultimately quadratically. We also let
!     F = L union G

!  construct values lambda_l and lambda_u for which lambda_l <= lambda_optimal
!   <= lambda_u, and ensure that all iterates satisfy lambda_l <= lambda
!   <= lambda_u

!  bounds on lambda currently used -

!  * lambda >=0
!  * lambda_L <= lambda <= [sigma ||A^Tb||_S^(p-2)]^1/(p-1)
!    where lambda_L is the rightmost real root of the equation
!      (lambda + lambda_n) lambda^1/(p-2) = sigma^1/(p-2) ||A^Tb||_S
!    and lambda_n^2 <= ||A S^{-1/2}||_1 . ||A S^{-1/2}||_inf
!                   <= ||A||_1 . ||A||_inf . ||S^{-1}||_2

!  ** unit regularization-norm scaling

        IF ( unit_s ) THEN

!  record sigma || c ||**(p-2)

          c_norm = TWO_NORM( data%C( : n ) )

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
          lambda_n = a_one * a_inf

!  ** diagonal regularization-norm scaling

        ELSE IF ( SMT_get( S%type ) == 'DIAGONAL' ) THEN

!  record sigma || c ||**(p-2)

          c_norm = SQRT( DOT_PRODUCT( data%C( : n ),                           &
                                      data%C( : n )  / S%val( : n ) ) )

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
          lambda_n = a_one * a_inf

!  ** general regularization-norm scaling

        ELSE

!  perform an analysis of the spasity pattern of S to identify a good
!  ordering for sparse factorization

          IF ( data%control%new_s >= 2 ) THEN
            data%control%SLS_control%pivot_control = 2
            CALL SLS_initialize_solver( control%definite_linear_solver,        &
                                        data%SLS_data,                         &
                                        data%control%SLS_control%error,        &
                                        inform%SLS_inform )
            IF ( inform%SLS_inform%status < 0 ) THEN
              inform%status = inform%SLS_inform%status ; GO TO 910 ; END IF
            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SLS_analyse( S, data%SLS_data,                                &
                              data%control%SLS_control, inform%SLS_inform )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%analyse = inform%time%analyse + time_now - time_record
            inform%time%clock_analyse =                                        &
              inform%time%clock_analyse + clock_now - clock_record
            IF ( printt ) WRITE( out, 2000 ) prefix, clock_now - clock_record

!  test that the analysis succeeded

            IF ( inform%SLS_inform%status < 0 ) THEN
              IF ( printi ) WRITE( out, "( A, ' error return from SLS',        &
             &  '_analyse: status = ', I0 )" ) prefix, inform%SLS_inform%status
              inform%status = GALAHAD_error_analysis ;  GO TO 910 ; END IF
          END IF

!  attempt an L B L^T factorization of S

          IF ( data%control%new_s >= 1 ) THEN
            CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
            CALL SLS_factorize( S, data%SLS_data,                              &
                                data%control%SLS_control, inform%SLS_inform )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%factorize =                                            &
              inform%time%factorize + time_now - time_record
            inform%time%clock_factorize =                                      &
              inform%time%clock_factorize + clock_now - clock_record
            IF ( printt ) WRITE( out, 2010 ) prefix, clock_now - clock_record
            inform%factorizations = inform%factorizations + 1

!  test that the factorization succeeded

            IF ( inform%SLS_inform%status == 0 ) THEN
              psdef = .TRUE.
            ELSE IF ( inform%SLS_inform%status == GALAHAD_error_inertia ) THEN
              GO TO 930
!             psdef = .FALSE.
            ELSE
              GO TO 920
            END IF
          END IF

!  compute S^{-1} c in V

          CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
          data%U( : n ) = data%C( : n )
          CALL IR_solve( S, data%U( : n ), data%IR_data, data%SLS_data,        &
                         data%control%IR_control, data%control%SLS_control,    &
                         inform%IR_inform, inform%SLS_inform )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!  compute the S^-1 norm of c, while checking that S is positive definite

          c_norm = DOT_PRODUCT( data%U( : n ), data%C( : n ) )
          IF ( c_norm < zero ) GO TO 930
          c_norm = SQRT( c_norm )

!  compute the sums of the absolute values of off-diagonal terms of S (in Y),
!  and its diagonal terms (in Z). Then record the Gershgorin bound on the
!  smallest eigenvalue, which gives the reciprocal of the largest eigenvalue
!  of S^-1

          data%Y( : n ) = zero ; data%Z( : n ) = zero
          SELECT CASE ( SMT_get( S%type ) )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
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
            DO i = 1, n
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
          lambda_n = a_one * a_inf * lambda_sinv

        END IF

        IF ( p == three ) THEN
          sigma_c_norm_pm2 = sigma * c_norm
        ELSE
          sigma_c_norm_pm2 = sigma * c_norm ** pm2
        END IF

!  set known lower and upper bounds on lambda

        IF ( p == three ) THEN
          CALL ROOTS_quadratic( sigma_c_norm_pm2, lambda_n, one, roots_tol,    &
                                nroots, roots( 1 ), roots( 2 ), roots_debug )
          lambda_l = MAX( zero, roots( 2 ) )
          lambda_u = SQRT( sigma_c_norm_pm2 )
        ELSE
          lambda_l =  MAX( zero, LLSR_lambda_root( lambda_n,                   &
                           sigma_c_norm_pm2 ** oopm2, oopm2 ) )
          lambda_u = ( sigma_c_norm_pm2 ) ** ( one / ( p - one ) )
        END IF

!  assign the initial lambda

        IF ( data%control%use_initial_multiplier ) THEN
          IF ( data%control%initial_multiplier >= lambda_l .AND.               &
               data%control%initial_multiplier <= lambda_u ) THEN
            lambda =  data%control%initial_multiplier
          ELSE
            lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                 &
                          lambda_l + theta_eps * ( lambda_u - lambda_l ) )
          END IF
        ELSE
          IF ( lambda_l == zero ) THEN
            try_zero = .FALSE.
            lambda = zero
          ELSE
            lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                 &
                          lambda_l + theta_eps * ( lambda_u - lambda_l ) )
          END IF
        END IF
        lambda = lambda + lambda_pert
      ELSE ! p == 2
        lambda = sigma
        lambda_l = sigma
        lambda_u = sigma
      END IF

      try_zero = lambda > zero .AND. lambda_l == zero
      region = ' '
      target = zero
      max_order = MAX( 1, MIN( max_degree, control%taylor_max_degree ) )

      IF ( printt )                                                            &
        WRITE( out, "( A, 4X, 28( '-' ), ' phase one ', 28( '-' ) )" ) prefix

!  start the main loop

      data%control%SBLS_control%preconditioner = 2
      data%control%SBLS_control%new_h = MAX( control%new_s, 1 )
      data%control%SBLS_control%new_a = control%new_a
      data%control%SBLS_control%new_c = 0

      it = 0 ; in_n = 0
      DO
        it = it + 1

!  exit if the iteration has stalled

        IF ( it > it_stalled ) THEN
          inform%status = GALAHAD_error_ill_conditioned
          RETURN
        END IF

        itt = 0
 100    CONTINUE

!  precaution to stop infinite loop

        itt = itt + 1
        IF ( itt > 100 ) THEN
          inform%status = GALAHAD_error_ill_conditioned
          RETURN
        END IF

!  introduce lambda * S to form K(lambda)

        IF ( unit_s ) THEN
          data%H_sbls%val( : data%s_ne ) = lambda
        ELSE
          IF ( SMT_get( S%type ) == 'COORDINATE' .OR.                          &
               SMT_get( S%type ) == 'SPARSE_BY_ROWS' .OR.                      &
               SMT_get( S%type ) == 'DENSE' .OR.                               &
               SMT_get( S%type ) == 'DIAGONAL' .OR.                            &
               SMT_get( S%type ) == 'SCALED_IDENTITY' ) THEN
            data%H_sbls%val( : data%s_ne ) = lambda * S%val( : data%s_ne )
          ELSE
            data%H_sbls%val( : data%s_ne ) = lambda
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
          psdef = .FALSE.
        ELSE
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
            CALL mop_AX( one, S, X, zero, data%Y( : n ), 0_ip_,                &
                         symmetric = .TRUE. )
            x_norm2( 0 ) = DOT_PRODUCT( X, data%Y( : n ) )
            IF ( x_norm2( 0 ) < zero ) GO TO 930
            inform%x_norm = SQRT( x_norm2( 0 ) )
          END IF

!  compute the two-norm of the residual, ||A x - b||_2 = ||y - b||_2

          inform%r_norm = TWO_NORM( B - data%U( n + 1 : data%npm ) )

!  special p = 2 case

          IF ( p == two ) THEN
            inform%status = GALAHAD_ok
            GO TO 900
          END IF

!  compute the target value ( lambda / sigma )^(1/(p-2))

          target = ( lambda / weight ) ** oopm2

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
            WRITE( out, "( A, 8X, 'lambda', 13X, 'x_norm', 15X, 'target' )" )  &
              prefix
            WRITE( out, "( A, 3ES20.12 )") prefix, lambda, inform%x_norm, target
            IF ( phase_1 ) THEN
              WRITE( out, "( A, ' interval width =', ES21.14 )")               &
                prefix, lambda_u - lambda_l
            ELSE
              WRITE( out, 2020 ) prefix
              WRITE( out, "( A, A2, I4, 3ES22.15 )" )                          &
                prefix, '  ', it, lambda_l, lambda, lambda_u
            END IF
          END IF

!  --------------------------------------------------------------------
!  The current estimate gives a good approximation to the required root
!  --------------------------------------------------------------------

          IF ( ABS( inform%x_norm - target ) <= data%control%stop_normal *     &
               MAX( one, inform%x_norm, target ) ) THEN
            IF ( inform%x_norm > target ) THEN
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
              it, ABS( inform%x_norm - target ), lambda, ABS( delta_lambda )
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
!         write(6,*) ' ||x||-target = ', inform%x_norm - target

!  determine which region the current lambda lies in

!  ----------------------------
!  The current lambda lies in L
!  ----------------------------

          IF ( inform%x_norm > target ) THEN
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
              region, it, ABS( inform%x_norm - target ), lambda,               &
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

          IF ( unit_s ) data%Y( : n ) = X
          data%Z( : n ) = data%Y( : n ) ; data%Z( n + 1 : data%npm ) = zero
          CALL SBLS_solve( n, m, A, data%C_sbls, data%SBLS_data,               &
                           data%control%SBLS_control, inform%SBLS_inform,      &
                           data%Z( : data%npm ) )

!  check that the solution succeeded. If not, increase lambda and try again

          IF ( inform%IR_inform%status == GALAHAD_error_solve ) THEN
            IF ( printd ) WRITE( out,                                          &
           &    "( ' **** WARNING 3 *** iterative refinement diverged,',       &
           &    ' increasing lambda marginally' ) ")
            lambda_l = lambda
            lambda = lambda_l + theta_eps * ( lambda_u - lambda_l )
            it = it + 1
            GO TO 100
          END IF

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%solve = inform%time%solve + time_now - time_record
          inform%time%clock_solve =                                            &
            inform%time%clock_factorize + clock_now - clock_record
          IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!         IF ( inform%IR_inform%norm_final_residual >                          &
!              inform%IR_inform%norm_initial_residual ) THEN
! write(6, "( ' *********** WARNING 4 - initial and final residuals are ',     &
!& 2ES12.4 )" ) inform%IR_inform%norm_initial_residual,                        &
!              inform%IR_inform%norm_final_residual
!         END IF

!  form ||w||^2 = y^T z = x^T L^-T D^-1 L^-1 x = x^T H^-1(lambda) x

          w_norm2 = DOT_PRODUCT( data%Y( : n ), data%Z( : n ) )

!  compute the first derivative of x_norm2 = x^T S x

          x_norm2( 1 ) = - two * w_norm2

!  count the number of corrections computed

          n_lambda = 0

!  compute Taylor approximants of degree one; special (but frequent) case
!  when p = 3

          IF ( p == three ) THEN

!  compute pi_beta = ||x||^beta and its first derivative when beta = 2

            beta = two
            CALL LLSR_pi_derivs( 1_ip_, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = 2)

            a_0 = pi_beta( 0 ) - target ** 2
            a_1 = pi_beta( 1 ) - two * lambda * oos2
            a_2 = - oos2
            a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
            IF ( a_max > zero ) THEN
              a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
            END IF
            CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,            &
                                  roots( 1 ), roots( 2 ), roots_debug )
            lambda_plus = lambda + roots( 2 )
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  compute pi_beta = ||x||^beta and its first derivative when beta = 1

            beta = one
            CALL LLSR_pi_derivs( 1_ip_, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = 1)

            delta_lambda = - ( pi_beta( 0 ) - target ) / ( pi_beta( 1 ) - oos )
            lambda_plus = lambda + delta_lambda
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  compute pi_beta = ||x||^beta and its first derivative when beta = - 1

            beta = - one
            CALL LLSR_pi_derivs( 1_ip_, beta, x_norm2( : 1 ), pi_beta( : 1 ) )

!  compute the Newton correction (for beta = -1)

            a_0 = pi_beta( 0 ) * lambda - sigma
            a_1 = pi_beta( 0 ) + lambda * pi_beta( 1 )
            a_2 = pi_beta( 1 )
            a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
            IF ( a_max > zero ) THEN
              a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
            END IF
            CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,            &
                                  roots( 1 ), roots( 2 ), roots_debug )
            lambda_plus = lambda + roots( 2 )
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  more general p

          ELSE

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their first derivatives when beta = p-2

            beta = pm2
            CALL LLSR_pi_derivs( 1_ip_, beta, x_norm2( : 1 ), pi_beta( : 1 ) )
            CALL LLSR_theta_derivs( 1_ip_, beta / pm2, lambda, sigma,          &
                                    theta_beta( : 1 )  )

!  compute the "linear Taylor approximation" correction (for beta = p-2)

            delta_lambda = - ( pi_beta( 0 ) - theta_beta( 0 ) ) /              &
                             ( pi_beta( 1 ) - theta_beta( 1 ) )
            lambda_plus = lambda + delta_lambda
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their first derivatives when beta = (p-2)/2

            beta = pm2 / two
            CALL LLSR_pi_derivs( 1_ip_, beta, x_norm2( : 1 ), pi_beta( : 1 ) )
            CALL LLSR_theta_derivs( 1_ip_, beta / pm2, lambda, sigma,          &
                                    theta_beta( : 1 )  )

!  compute the "linear Taylor approximation" correction (for beta = (p-2)/2)

            delta_lambda = - ( pi_beta( 0 ) - theta_beta( 0 ) ) /              &
                             ( pi_beta( 1 ) - theta_beta( 1 ) )
            lambda_plus = lambda + delta_lambda
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their first derivatives when beta = max(2-p,-1)

            beta = max( - pm2, - one )
            CALL LLSR_pi_derivs( 1_ip_, beta, x_norm2( : 1 ), pi_beta( : 1 ) )
            CALL LLSR_theta_derivs( 1_ip_, beta / pm2, lambda, sigma,          &
                                    theta_beta( : 1 )  )

!  compute the "linear Taylor approximation" correction (for beta = max(2-p,-1))

            delta_lambda = - ( pi_beta( 0 ) - theta_beta( 0 ) ) /              &
                             ( pi_beta( 1 ) - theta_beta( 1 ) )
            lambda_plus = lambda + delta_lambda
            IF (  lambda_plus < lambda ) THEN
              n_lambda = n_lambda + 1
              lambda_new( n_lambda ) = lambda_plus
            END IF
          END IF

!         WRITE( out, "( ' lambda, delta ', 2ES22.15 )" ) lambda, delta_lambda

          IF ( ( max_order >= 3 .AND. region == 'L' ) .OR.                     &
               ( max_order >= 2 .AND. region == 'G' ) ) THEN

!  form z^T S z

            IF ( unit_s ) THEN
              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Z( : n ) )
            ELSE
              CALL mop_AX( one, S, data%Z( : n ), zero, data%Y( : n ), 0_ip_,  &
                           symmetric = .TRUE., m_matrix = n, n_matrix = n )
              z_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )
            END IF

!  compute the second derivative of x_norm2 = x^T S x

            x_norm2( 2 ) = six * z_norm2

            IF ( max_order >= 3 ) THEN
              CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )

!  solve  H(lambda) z = S x

              IF ( unit_s ) THEN
                data%Y( : n ) = data%Z( : n )
              ELSE
                data%Z( : n ) = data%Y( : n )
              END IF
              data%Z( n + 1 : data%npm ) = zero
              CALL SBLS_solve( n, m, A, data%C_sbls, data%SBLS_data,           &
                               data%control%SBLS_control, inform%SBLS_inform,  &
                               data%Z( : data%npm ) )

!  check that the solution succeeded. If not, increase lambda and try again

              IF ( inform%IR_inform%status == GALAHAD_error_solve ) THEN
                IF ( printd ) WRITE( out,                                      &
               &    "( ' **** WARNING 5 *** iterative refinement diverged,',   &
               &    ' increasing lambda marginally' ) ")
                lambda_l = lambda
                lambda = lambda_l + theta_eps5 * ( lambda_u - lambda_l )
                it = it + 1
                GO TO 100
              END IF

              CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
              inform%time%solve = inform%time%solve + time_now - time_record
              inform%time%clock_solve =                                        &
                inform%time%clock_factorize + clock_now - clock_record
              IF ( printt ) WRITE( out, 2050 ) prefix, clock_now - clock_record

!             IF ( inform%IR_inform%norm_final_residual >                      &
!                  inform%IR_inform%norm_initial_residual ) THEN
! write(6, "( ' *********** WARNING 6 - initial and final residuals are ',     &
!& 2ES12.4 )" ) inform%IR_inform%norm_initial_residual,                        &
!              inform%IR_inform%norm_final_residual
!             END IF

!  form ||v||^2 = z^T y = x'^T L^-T D^-1 L^-1 x' = x'^T H^-1(lambda) x'

              v_norm2 = DOT_PRODUCT( data%Z( : n ), data%Y( : n ) )

!  compute the third derivatives of x_norm2 = x^T S x

              x_norm2( 3 ) = - twentyfour * v_norm2
            END IF
          END IF

!  reset lower or upper bound appropriately

          IF ( region == 'L' ) lambda_l = MAX( lambda_l, lambda )
          IF ( region == 'G' ) lambda_u = MIN( lambda_u, lambda )

!  compute pi_beta = ||x||^beta and its derivatives for various beta
!  and the resulting Taylor series approximants

 200      CONTINUE

!  ----------------------------
!  The current lambda lies in L
!  ----------------------------

          IF ( inform%x_norm > target ) THEN

!  for Taylor approximants of degree larger than one

            IF ( max_order >= 3 ) THEN

!  special (but frequent) case when p = 3

              IF ( p == three ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = 2

                beta = two
                CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),              &
                                     pi_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = 2)

                a_0 = pi_beta( 0 ) - target ** 2
                a_1 = pi_beta( 1 ) - two * lambda * oos2
                a_2 = half * pi_beta( 2 ) - oos2
                a_3 = sixth * pi_beta( 3 )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                  a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                END IF
                CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,       &
                                  roots( 1 ), roots( 2 ), roots( 3 ),          &
                                  roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda + roots( 1 )

!  compute pi_beta = ||x||^beta and its derivatives when beta = 1

                beta = one
                CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),              &
                                     pi_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = 1)

                a_0 = pi_beta( 0 ) - target
                a_1 = pi_beta( 1 ) - oos
                a_2 = half * pi_beta( 2 )
                a_3 = sixth * pi_beta( 3 )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                  a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                END IF
                CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,       &
                                  roots( 1 ), roots( 2 ), roots( 3 ),          &
                                  roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda + roots( 1 )

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^beta and
!  their derivatives when beta = - 0.4

                IF ( lambda > 0 ) THEN
                  beta = - point4
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )
                  CALL LLSR_theta_derivs( 3_ip_, beta, lambda, sigma,          &
                                          theta_beta( : 3 )  )

!  compute the "cubic Taylor approximaton" step (beta = - 0.4)

                  a_0 = pi_beta( 0 ) - theta_beta( 0 )
                  a_1 = pi_beta( 1 ) - theta_beta( 1 )
                  a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                  a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3 ) )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda + roots( 1 )
                END IF

!  more general p

              ELSE

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = p-2

                beta = pm2
                CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),              &
                                     pi_beta( : 3 ) )
                CALL LLSR_theta_derivs( 3_ip_, beta / pm2, lambda, sigma,      &
                                       theta_beta( : 3 ) )

!  compute the "cubic Taylor approximation" correction (for beta = p-2)

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                  a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                END IF
                CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,       &
                                  roots( 1 ), roots( 2 ), roots( 3 ),          &
                                  roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda +                              &
                  LLSR_required_root( .TRUE., nroots, roots( : 3 ) )

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = (p-2)/2

                beta = pm2 / two
                CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),              &
                                     pi_beta( : 3 ) )
                CALL LLSR_theta_derivs( 3_ip_, beta / pm2, lambda, sigma,      &
                                       theta_beta( : 3 ) )

!  compute the "cubic Taylor approximation" correction (for beta = (p-2)/2)

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                  a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                END IF
                CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,       &
                                  roots( 1 ), roots( 2 ), roots( 3 ),          &
                                  roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda +                              &
                  LLSR_required_root( .TRUE., nroots, roots( : 3 ) )

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = max(2-p,-0.4)

                beta = max( - pm2, - point4 )
                CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),              &
                                     pi_beta( : 3 ) )
                CALL LLSR_theta_derivs( 3_ip_, beta / pm2, lambda, sigma,      &
                                       theta_beta( : 3 ) )

!  compute the "cubic Taylor approximation" correction (for beta=max(2-p,-0.4))

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                  a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                END IF
                CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,       &
                                  roots( 1 ), roots( 2 ), roots( 3 ),          &
                                  roots_debug )
                n_lambda = n_lambda + 1
                lambda_new( n_lambda ) = lambda +                              &
                  LLSR_required_root( .TRUE., nroots, roots( : 3 ) )
              END IF
            END IF

!  record all of the estimates of the optimal lambda

            IF ( printd ) THEN
              WRITE( out, "( A, ' lambda_t (', I1, ')', 3ES20.12 )" )          &
                prefix, MAXLOC( lambda_new( : n_lambda ) ),                    &
                lambda_new( : MIN( 3, n_lambda ) )
              IF ( n_lambda > 3 ) WRITE( out, "( A, 13X, 3ES20.12 )" )         &
                prefix, lambda_new( 4 : MIN( 6, n_lambda ) )
            END IF

!  compute the best Taylor improvement

            lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
            delta_lambda = lambda_plus - lambda
            lambda = lambda_plus

!  improve the lower bound if possible

            lambda_l = MAX( lambda_l, lambda_plus )

!  check that the best Taylor improvement is significant

            IF ( ABS( delta_lambda ) < epsmch * MAX( one, ABS( lambda ) ) ) THEN
              inform%status = GALAHAD_ok
              IF ( printi ) WRITE( out, "( A, ' normal exit with no ',         &
             &                     'significant Taylor improvement' )" ) prefix
              EXIT
            END IF
          ELSE

!  ----------------------------
!  The current lambda lies in G
!  ----------------------------

!  compute Taylor approximants of degree two

            IF ( max_order >= 2 ) THEN

!  special (but frequent) case when p = 3

              IF ( p == three ) THEN

!  compute pi_beta = ||x||^beta and its first derivative when beta = 2

                beta = two
                CALL LLSR_pi_derivs( 2_ip_, beta, x_norm2( : 2 ),              &
                                     pi_beta( : 2 ) )

!  compute the "quadratic Taylor approximaton" step (beta = 2)

                a_0 = pi_beta( 0 ) - target ** 2
                a_1 = pi_beta( 1 ) - two * lambda * oos2
                a_2 = half * pi_beta( 2 ) - oos2
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      roots( 1 ), roots( 2 ), roots_debug )
                lambda_plus = lambda +                                         &
                   LLSR_required_root( .FALSE., nroots, roots( : 2 ) )
                IF (  lambda_plus < lambda ) THEN
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda_plus
                END IF

!  compute pi_beta = ||x||^beta and its first derivative when beta = 1

                beta = one
                CALL LLSR_pi_derivs( 2_ip_, beta, x_norm2( : 2 ),              &
                                     pi_beta( : 2 ) )

!  compute the "quadratic Taylor approximaton" step (beta = 1)

                a_0 = pi_beta( 0 ) - target
                a_1 = pi_beta( 1 ) - oos
                a_2 = half * pi_beta( 2 )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      roots( 1 ), roots( 2 ), roots_debug )
                lambda_plus = lambda +                                         &
                   LLSR_required_root( .FALSE., nroots, roots( : 2 ) )
                IF (  lambda_plus < lambda ) THEN
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda_plus
                END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^beta and
!  their derivatives when beta = - 0.666

                beta = - twothirds
                CALL LLSR_pi_derivs( 2_ip_, beta, x_norm2( : 2 ),              &
                                     pi_beta( : 2 ) )
                CALL LLSR_theta_derivs( 2_ip_, beta, lambda, sigma,            &
                                       theta_beta( : 2 ) )

!  compute the "quadratic Taylor approximaton" step (beta = - 0.666)

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      roots( 1 ), roots( 2 ), roots_debug )
                lambda_plus = lambda +                                         &
                   LLSR_required_root( .FALSE., nroots, roots( : 2 ) )
                IF (  lambda_plus < lambda ) THEN
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda_plus
                END IF

!  more general p

              ELSE

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = p-2

                beta = pm2
                CALL LLSR_pi_derivs( 2_ip_, beta, x_norm2( : 2 ),              &
                                     pi_beta( : 2 ) )
                CALL LLSR_theta_derivs( 2_ip_, beta / pm2, lambda, sigma,      &
                                       theta_beta( : 2 )  )

!  compute the "quadratic Taylor approximation" correction (for beta = p-2)

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      roots( 1 ), roots( 2 ), roots_debug )
                lambda_plus = lambda +                                         &
                   LLSR_required_root( .FALSE., nroots, roots( : 2 ) )
                IF (  lambda_plus < lambda ) THEN
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda_plus
                END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = (p-2)/2

                beta = pm2 / two
                CALL LLSR_pi_derivs( 2_ip_, beta, x_norm2( : 2 ),              &
                                     pi_beta( : 2 ) )
                CALL LLSR_theta_derivs( 2_ip_, beta / pm2, lambda, sigma,      &
                                       theta_beta( : 2 )  )

!  compute the "quadratic Taylor approximation" correction (for beta = (p-2)/2)

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      roots( 1 ), roots( 2 ), roots_debug )
                lambda_plus = lambda +                                         &
                   LLSR_required_root( .FALSE., nroots, roots( : 2 ) )
                IF (  lambda_plus < lambda ) THEN
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda_plus
                END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = max(2-p,-0.666)

                beta = max( - pm2, - twothirds )
                CALL LLSR_pi_derivs( 2_ip_, beta, x_norm2( : 2 ),              &
                                     pi_beta( : 2 ) )
                CALL LLSR_theta_derivs( 2_ip_, beta / pm2, lambda, sigma,      &
                                       theta_beta( : 2 )  )

!  compute the "quadratic Taylor approximation" correction
!  (for beta = max(2-p,-0.666))

                a_0 = pi_beta( 0 ) - theta_beta( 0 )
                a_1 = pi_beta( 1 ) - theta_beta( 1 )
                a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ) )
                IF ( a_max > zero ) THEN
                  a_0 = a_0 / a_max ; a_1 = a_1 / a_max ; a_2 = a_2 / a_max
                END IF
                CALL ROOTS_quadratic( a_0, a_1, a_2, roots_tol, nroots,        &
                                      roots( 1 ), roots( 2 ), roots_debug )
                lambda_plus = lambda +                                         &
                   LLSR_required_root( .FALSE., nroots, roots( : 2 ) )
                IF (  lambda_plus < lambda ) THEN
                  n_lambda = n_lambda + 1
                  lambda_new( n_lambda ) = lambda_plus
                END IF
              END IF

!  compute Taylor approximants of degree three or larger

              IF ( max_order >= 3 ) THEN

!  special (but frequent) case when p = 3

                IF ( p == three ) THEN

!  compute pi_beta = ||x||^beta and its derivatives when beta = 2

                  beta = two
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = 2)

                  a_0 = pi_beta( 0 ) - target ** 2
                  a_1 = pi_beta( 1 ) - two * lambda * oos2
                  a_2 = half * pi_beta( 2 ) - oos2
                  a_3 = sixth * pi_beta( 3 )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  lambda_plus = lambda + roots( 1 )
                  IF (  lambda_plus < lambda ) THEN
                    n_lambda = n_lambda + 1
                    lambda_new( n_lambda ) = lambda_plus
                  END IF

!  compute pi_beta = ||x||^beta and its derivatives when beta = 1

                  beta = one
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = 1)

                  a_0 = pi_beta( 0 ) - target
                  a_1 = pi_beta( 1 ) - oos
                  a_2 = half * pi_beta( 2 )
                  a_3 = sixth * pi_beta( 3 )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  lambda_plus = lambda + roots( 1 )
                  IF (  lambda_plus < lambda ) THEN
                    n_lambda = n_lambda + 1
                    lambda_new( n_lambda ) = lambda_plus
                  END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^beta and
!  their derivatives when beta = - 0.4

                  beta = - point4
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )
                  CALL LLSR_theta_derivs( 3_ip_, beta, lambda, sigma,          &
                                         theta_beta( : 3 ) )

!  compute the "cubic Taylor approximaton" step (beta = - 0.4)

                  a_0 = pi_beta( 0 ) - theta_beta( 0 )
                  a_1 = pi_beta( 1 ) - theta_beta( 1 )
                  a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                  a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3) )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  lambda_plus = lambda + roots( 1 )
                  IF (  lambda_plus < lambda ) THEN
                    n_lambda = n_lambda + 1
                    lambda_new( n_lambda ) = lambda_plus
                  END IF

!  more general p

                ELSE

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = p-2

                  beta = pm2
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )
                  CALL LLSR_theta_derivs( 3_ip_, beta / pm2, lambda, sigma,    &
                                          theta_beta( : 3 ) )

!  compute the "cubic Taylor approximation" correction (for beta = p-2)

                  a_0 = pi_beta( 0 ) - theta_beta( 0 )
                  a_1 = pi_beta( 1 ) - theta_beta( 1 )
                  a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                  a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3) )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  lambda_plus = lambda +                                       &
                    LLSR_required_root( .FALSE., nroots, roots( : 3 ) )
                  IF (  lambda_plus < lambda ) THEN
                    n_lambda = n_lambda + 1
                    lambda_new( n_lambda ) = lambda_plus
                  END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = (p-2)/2

                  beta = pm2 / two
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )
                  CALL LLSR_theta_derivs( 3_ip_, beta / pm2, lambda, sigma,    &
                                          theta_beta( : 3 )  )

!  compute the "cubic Taylor approximation" correction (for beta = (p-2)/2)

                  a_0 = pi_beta( 0 ) - theta_beta( 0 )
                  a_1 = pi_beta( 1 ) - theta_beta( 1 )
                  a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                  a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3) )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  lambda_plus = lambda +                                       &
                    LLSR_required_root( .FALSE., nroots, roots( : 3 ) )
                  IF (  lambda_plus < lambda ) THEN
                    n_lambda = n_lambda + 1
                    lambda_new( n_lambda ) = lambda_plus
                  END IF

!  compute pi_beta = ||x||^beta and theta_beta = (lambda/sigma)^(beta/(p-2)) and
!  their derivatives when beta = max(2-p,-0.4)

                  beta = max( - pm2, - point4 )
                  CALL LLSR_pi_derivs( 3_ip_, beta, x_norm2( : 3 ),            &
                                       pi_beta( : 3 ) )
                  CALL LLSR_theta_derivs( 3_ip_, beta / pm2, lambda, sigma,    &
                                         theta_beta( : 3 )  )

!  compute the "cubic Taylor approximation" correction (for beta=max(2-p,-0.4))

                  a_0 = pi_beta( 0 ) - theta_beta( 0 )
                  a_1 = pi_beta( 1 ) - theta_beta( 1 )
                  a_2 = half * ( pi_beta( 2 ) - theta_beta( 2 ) )
                  a_3 = sixth * ( pi_beta( 3 ) - theta_beta( 3) )
                  a_max = MAX( ABS( a_0 ), ABS( a_1 ), ABS( a_2 ), ABS( a_3 ) )
                  IF ( a_max > zero ) THEN
                    a_0 = a_0 / a_max ; a_1 = a_1 / a_max
                    a_2 = a_2 / a_max ; a_3 = a_3 / a_max
                  END IF
                  CALL ROOTS_cubic( a_0, a_1, a_2, a_3, roots_tol, nroots,     &
                                    roots( 1 ), roots( 2 ), roots( 3 ),        &
                                    roots_debug )
                  lambda_plus = lambda +                                       &
                    LLSR_required_root( .FALSE., nroots, roots( : 3 ) )
                  IF (  lambda_plus < lambda ) THEN
                    n_lambda = n_lambda + 1
                    lambda_new( n_lambda ) = lambda_plus
                  END IF
                END IF
              END IF
            END IF

!  record all of the estimates of the optimal lambda

            IF ( printd ) THEN
              WRITE( out, "( A, ' lambda_t (', I1, ')', 3ES20.13 )" )          &
                prefix, MAXLOC( lambda_new( : n_lambda ) ),                    &
                lambda_new( : MIN( 3, n_lambda ) )
              IF ( n_lambda > 3 ) WRITE( out, "( A, 13X, 3ES20.13 )" )         &
                prefix, lambda_new( 4 : MIN( 6, n_lambda ) )
              IF ( n_lambda > 6 ) WRITE( out, "( A, 13X, 3ES20.13 )" )         &
                prefix, lambda_new( 7 : MIN( 9, n_lambda ) )
            END IF

!  compute the best Taylor improvement

            lambda_plus = MAXVAL( lambda_new( : n_lambda ) )
            delta_lambda = lambda_plus - lambda

!  improve the lower bound if possible

            lambda_l = MAX( lambda_l, lambda_plus )

!  if lambda = 0 hasn't yet been tried, do so

            IF ( try_zero ) THEN
              try_zero = .FALSE.
              IF ( lambda_l < zero ) THEN
                lambda = zero
                CYCLE
              END IF
            END IF

!  check that the best Taylor improvement is significant

            IF ( ABS( delta_lambda ) < epsmch * MAX( one, ABS( lambda ) ) ) THEN
              inform%status = GALAHAD_ok
              IF ( printi ) WRITE( out, "( A, ' normal exit with no ',         &
             &                   'significant Taylor improvement' )" ) prefix
              EXIT
            END IF
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
          in_n = in_n + 1
          IF ( MOD( in_n, 2 ) == 1 ) THEN
            lambda = MAX( gamma * SQRT( lambda_l * lambda_u ),                 &
                          lambda_l + theta_n * width )
          ELSE
            lambda = lambda_l + theta_n_small * width
          END IF
        END IF

        printh = printt .OR. printi .AND.                                      &
                   ( control%SBLS_control%print_level > 0 .OR.                 &
                     control%SLS_control%print_level > 0 .OR.                  &
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
     & ' from LLSR ' )" ) prefix, inform%status
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
         "( A, ' The matrix S provided for LLSR appears not to be strictly',   &
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
 2030 FORMAT( A, '    it    ||x||-target              lambda ',                &
                 '              d_lambda' )
 2050 FORMAT( A, ' time( SBLS_solve ) = ', F0.2 )

!  End of subroutine LLSR_solve

      END SUBROUTINE LLSR_solve

!-*-*-*-*-*-  L L S R _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LLSR_terminate( data, control, inform )

!  ...........................................
!  .                                         .
!  .  Deallocate arrays at end of LLSR_solve  .
!  .                                         .
!  ...........................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see Subroutine LLSR_initialize
!   inform    see Subroutine LLSR_solve

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LLSR_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLSR_control_type ), INTENT( IN ) :: control
      TYPE ( LLSR_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all internal arrays

      array_name = 'llsr: S_diag'
      CALL SPACE_dealloc_array( data%S_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: S_offd'
      CALL SPACE_dealloc_array( data%S_offd,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: U'
      CALL SPACE_dealloc_array( data%U,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: Y'
      CALL SPACE_dealloc_array( data%Y,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: Z'
      CALL SPACE_dealloc_array( data%Z,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: H_sbls%row'
      CALL SPACE_dealloc_array( data%H_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: H_sbls%col'
      CALL SPACE_dealloc_array( data%H_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: H_sbls%ptr'
      CALL SPACE_dealloc_array( data%H_sbls%ptr,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'llsr: H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  Deallocate all arrays allocated within IR

      CALL IR_terminate( data%IR_data, control%IR_control,                     &
                          inform%IR_inform )
      IF ( inform%IR_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'llsr: IR_data'
      END IF

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%SLS_control,                  &
                          inform%SLS_inform )
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'llsr: SLS_data'
      END IF

!  Deallocate all arrays allocated within SBLS

      CALL SBLS_terminate( data%SBLS_data, control%SBLS_control,               &
                          inform%SBLS_inform )
      IF ( inform%SBLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'llsr: SBLS_data'
      END IF

      RETURN

!  End of subroutine LLSR_terminate

      END SUBROUTINE LLSR_terminate

!   G A L A H A D -  L L S R _ f u l l _ t e r m i n a t e  S U B R O U T I N E

      SUBROUTINE LLSR_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLSR_control_type ), INTENT( IN ) :: control
      TYPE ( LLSR_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

      CALL LLSR_terminate( data%llsr_data, control, inform )
      RETURN

!  End of subroutine LLSR_full_terminate

      END SUBROUTINE LLSR_full_terminate

!-*-*-*-*-*-*-  L L S R _ P I _ D E R I V S   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE LLSR_pi_derivs( max_order, beta, x_norm2, pi_beta )

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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: max_order
      REAL ( KIND = rp_ ), INTENT( IN ) :: beta, x_norm2( 0 : max_order )
      REAL ( KIND = rp_ ), INTENT( OUT ) :: pi_beta( 0 : max_order )

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      REAL ( KIND = rp_ ) :: hbeta

      hbeta = half * beta
      pi_beta( 0 ) = x_norm2( 0 ) ** hbeta
      IF ( hbeta == one ) THEN
        pi_beta( 1 ) = x_norm2( 1 )
        IF ( max_order == 1 ) RETURN
        pi_beta( 2 ) = x_norm2( 2 )
        IF ( max_order == 2 ) RETURN
        pi_beta( 3 ) = x_norm2( 3 )
      ELSE IF ( hbeta == two ) THEN
        pi_beta( 1 ) = two * x_norm2( 0 ) * x_norm2( 1 )
        IF ( max_order == 1 ) RETURN
        pi_beta( 2 ) = two * ( x_norm2( 1 ) ** 2 + x_norm2( 0 ) * x_norm2( 2 ) )
        IF ( max_order == 2 ) RETURN
        pi_beta( 3 ) = two *                                                   &
          ( x_norm2( 0 ) * x_norm2( 3 ) + three * x_norm2( 1 ) * x_norm2( 2 ) )
      ELSE
        pi_beta( 1 )                                                           &
          = hbeta * ( x_norm2( 0 ) ** ( hbeta - one ) ) * x_norm2( 1 )
        IF ( max_order == 1 ) RETURN
        pi_beta( 2 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - two ) ) *           &
          ( ( hbeta - one ) * x_norm2( 1 ) ** 2 + x_norm2( 0 ) * x_norm2( 2 ) )
        IF ( max_order == 2 ) RETURN
        pi_beta( 3 ) = hbeta * ( x_norm2( 0 ) ** ( hbeta - three ) ) *         &
          ( x_norm2( 3 ) * x_norm2( 0 ) ** 2 + ( hbeta - one ) *               &
            ( three * x_norm2( 0 ) * x_norm2( 1 ) * x_norm2( 2 ) +             &
              ( hbeta - two ) * x_norm2( 1 ) ** 3 ) )
      END IF

      RETURN

!  End of subroutine LLSR_pi_derivs

      END SUBROUTINE LLSR_pi_derivs

!-*-*-*-*-*  L L S R _ T H E T A _ D E R I V S   S U B R O U T I N E   *-*-*-*-

      SUBROUTINE LLSR_theta_derivs( max_order, beta, lambda, sigma, theta_beta )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute theta_beta = (lambda/sigma)^beta and its derivatives
!
!  Arguments:
!  =========
!
!  Input -
!   max_order - maximum order of derivative
!   beta - power
!   lambda, sigma - lambda and sigma
!  Output -
!   theta_beta - (0) value of (lambda/sigma)^beta,
!                (i) ith derivative of (lambda/sigma)^beta, i = 1, max_order
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: max_order
      REAL ( KIND = rp_ ), INTENT( IN ) :: beta, lambda, sigma
      REAL ( KIND = rp_ ), INTENT( OUT ) :: theta_beta( 0 : max_order )

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      REAL ( KIND = rp_ ) :: los, oos

      los = lambda / sigma
      oos = one / sigma

      theta_beta( 0 ) = los ** beta
      IF ( beta == one ) THEN
        theta_beta( 1 ) = oos
        IF ( max_order == 1 ) RETURN
        theta_beta( 2 ) = zero
        IF ( max_order == 2 ) RETURN
        theta_beta( 3 ) = zero
      ELSE IF ( beta == two ) THEN
        theta_beta( 1 ) = two * los * oos
        IF ( max_order == 1 ) RETURN
        theta_beta( 2 ) = oos ** 2
        IF ( max_order == 2 ) RETURN
        theta_beta( 3 ) = zero
      ELSE
        theta_beta( 1 ) = beta * ( los ** ( beta - one ) ) * oos
        IF ( max_order == 1 ) RETURN
        theta_beta( 2 ) = beta * ( los ** ( beta - two ) ) *                   &
                          ( beta - one ) * oos ** 2
        IF ( max_order == 2 ) RETURN
        theta_beta( 3 ) = beta * ( los ** ( beta - three ) ) *                 &
                          ( beta - one ) * ( beta - two ) * oos ** 3
      END IF
      RETURN

!  End of subroutine LLSR_theta_derivs

      END SUBROUTINE LLSR_theta_derivs

!-*-*-*-*-*-  L L S R _ R E Q U I R E D _ R O O T  F U C T I O N   -*-*-*-*-*-

      FUNCTION LLSR_required_root( positive, nroots, roots )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Determine the required root of the three roots of the secular equation.
!  This is either the most positive root (positive=.TRUE.) or the least
!  negative one (positive=.FALSE.)
!
!  Arguments:
!  =========
!
!  Input -
!   positive - .TRUE. if the largest positive root is required,
!               .FALSE. if the least negative one
!   nroots - number of roots
!   roots - roots in increasing order
!  Output -
!   LLSR_required root - the required root
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      REAL ( KIND = rp_ ) :: LLSR_required_root
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: nroots
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: roots
      LOGICAL, INTENT( IN ) :: positive

      IF ( positive ) THEN
        IF ( SIZE( roots ) == 3 ) THEN
          IF ( nroots == 3 ) THEN
            LLSR_required_root = roots( 3 )
          ELSE IF ( nroots == 2 ) THEN
            LLSR_required_root = roots( 2 )
          ELSE
            LLSR_required_root = roots( 1 )
          END IF
        ELSE
          IF ( nroots == 2 ) THEN
            LLSR_required_root = roots( 2 )
          ELSE
            LLSR_required_root = roots( 1 )
          END IF
        END IF
      ELSE
        IF ( SIZE( roots ) == 3 ) THEN
          IF ( nroots == 3 ) THEN
            IF ( roots( 3 ) > zero ) THEN
              IF ( roots( 2 ) > zero ) THEN
                LLSR_required_root = roots( 1 )
              ELSE
                LLSR_required_root = roots( 2 )
              END IF
            ELSE
              LLSR_required_root = roots( 3 )
            END IF
          ELSE IF ( nroots == 2 ) THEN
            IF ( roots( 2 ) > zero ) THEN
              LLSR_required_root = roots( 1 )
            ELSE
              LLSR_required_root = roots( 2 )
            END IF
          ELSE
            LLSR_required_root = roots( 1 )
          END IF
        ELSE
          IF ( nroots == 2 ) THEN
            IF ( roots( 2 ) > zero ) THEN
              LLSR_required_root = roots( 1 )
            ELSE
              LLSR_required_root = roots( 2 )
            END IF
          ELSE
            LLSR_required_root = roots( 1 )
          END IF
        END IF
      END IF
      RETURN

!  End of function LLSR_required_root

      END FUNCTION LLSR_required_root

!-*-*-*-*-*-  L L S R _ L A M B D A  _ R O O T  F U C T I O N   -*-*-*-*-*-

      FUNCTION LLSR_lambda_root( a, b, power )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find the positive root of lambda + a = b/lambda^power
!
!  Arguments:
!  =========
!
!  Input -
!   a, b, power - data for the above problem (with b, power > 0)
!  Output -
!   LLSR_lambda root - the required root
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      REAL ( KIND = rp_ ) :: LLSR_lambda_root
      REAL ( KIND = rp_ ), INTENT( IN ) :: a, b, power

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: nroots, it
      INTEGER ( KIND = ip_ ), PARAMETER :: newton_max = 10
      REAL ( KIND = rp_ ) :: lambda, phi, phip, d_lambda, other, power_plus_1

!  special case: a = 0 = b

      IF ( a == zero .AND. b == zero ) THEN
        LLSR_lambda_root = zero ; RETURN
      END IF

!  compute as initial lower bound on the root

      IF ( power == one ) THEN
        CALL ROOTS_quadratic( - b , a, one, roots_tol, nroots, other,          &
                              lambda, roots_debug )
      ELSE
        power_plus_1 = power + one

!  when power > 1, 1/lambda <= 1/lambda^p for lambda in (0,1]

        IF ( power > one ) THEN
          CALL ROOTS_quadratic( - b , a, one, roots_tol, nroots, other,        &
                                lambda, roots_debug )
          lambda = MIN( one, lambda )
        ELSE
          lambda = epsmch
        END IF

!  check if lambda = 1 is acceptable

        IF ( one + a <= b ) lambda = MAX( lambda, one )

!  when a > 0, find where the tangent to b/lambda^power at
!  lambda = b^(1/power+1) intersects lambda + a

        IF ( a >= zero ) THEN
          lambda = MAX( lambda, b ** ( one / power_plus_1 ) - a / power_plus_1 )

!  when a < 0, both the lambda-intercept of lambda + a and the interection
!  of lambda with beta / lambda^(1/power+1) give lower bounds on the root

        ELSE
          lambda = MAX( lambda, - a, b ** ( one / power_plus_1 ) )
        END IF

!  perform Newton's method to refine the root

        DO it = 1, newton_max
          phi = lambda + a - b / ( lambda ** power )
          IF ( ABS( phi ) <= ten * epsmch *                                   &
                 MAX(  lambda + a, b / ( lambda ** power ) ) ) EXIT
          phip = one + b * power / ( lambda ** ( power + one ) )
          d_lambda = - phi / phip
          IF ( ABS( d_lambda ) <= epsmch * MAX( one, lambda ) ) EXIT
          lambda = lambda + d_lambda
        END DO
      END IF
      LLSR_lambda_root = lambda

      RETURN

!  End of function LLSR_lambda_root

      END FUNCTION LLSR_lambda_root

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-  G A L A H A D -  L L S R _ i m p o r t _ A _ S U B R O U T I N E -*-*-

     SUBROUTINE LLSR_import( control, data, status, m, n,                      &
                             A_type, A_ne, A_row, A_col, A_ptr )

!  import fixed problem data for the problem Jacobian A into internal
!  storage prior to solution. Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to LLSR_solve
!
!  data is a scalar variable of type LLSR_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, m >= 0 or requirement that the types contain
!       a relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'IDENTITY' has been violated.
!
!  m is a scalar variable of type default integer, that holds the number of
!   constraints
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  A_type is a character string that specifies the Jacobian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows' or 'dense'
!   if m > 0; lower or upper case variants are allowed
!
!  A_ne is a scalar variable of type default integer, that holds the number of
!   entries in J in the sparse co-ordinate storage scheme. It need not be set
!  for any of the other schemes.
!
!  A_row is a rank-one array of type default integer, that holds the row
!   indices J in the sparse co-ordinate storage scheme. It need not be set
!   for any of the other schemes, and in this case can be of length 0
!
!  A_col is a rank-one array of type default integer, that holds the column
!   indices of J in either the sparse co-ordinate, or the sparse row-wise
!   storage scheme. It need not be set when the dense scheme is used, and
!   in this case can be of length 0
!
!  A_ptr is a rank-one array of dimension n+1 and type default integer,
!   that holds the starting position of each row of J, as well as the total
!   number of entries plus one, in the sparse row-wise storage scheme.
!   It need not be set when the other schemes are used, and in this case
!   can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LLSR_control_type ), INTENT( INOUT ) :: control
     TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n
     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: A_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: A_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: A_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

     WRITE( data%llsr_control%out, "( '' )", ADVANCE = 'no') !prevents ifort bug
     data%llsr_control = control

     error = data%llsr_control%error
     space_critical = data%llsr_control%space_critical
     deallocate_error_fatal = data%llsr_control%space_critical

!  flag that S is not currently used

     data%use_s = .FALSE.

!  record the dimensions

     data%A%n = n ; data%A%m = m

!  set A appropriately in its storage type

     SELECT CASE ( A_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( A_ne ) .AND. PRESENT( A_row ) .AND.              &
                    PRESENT( A_col ) ) ) THEN
         data%llsr_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%A%type, 'COORDINATE', data%llsr_inform%alloc_status )
       data%A%ne = A_ne

       array_name = 'llsr: data%A%row'
       CALL SPACE_resize_array( data%A%ne, data%A%row,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%A%col'
       CALL SPACE_resize_array( data%A%ne, data%A%col,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%A%val'
       CALL SPACE_resize_array( data%A%ne, data%A%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%A%row( : data%A%ne ) = A_row( : data%A%ne )
         data%A%col( : data%A%ne ) = A_col( : data%A%ne )
       ELSE
         data%A%row( : data%A%ne ) = A_row( : data%A%ne ) + 1
         data%A%col( : data%A%ne ) = A_col( : data%A%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( A_col ) .AND. PRESENT( A_ptr ) ) ) THEN
         data%llsr_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%A%type, 'SPARSE_BY_ROWS',                            &
                     data%llsr_inform%alloc_status )
       IF ( data%f_indexing ) THEN
         data%A%ne = A_ptr( m + 1 ) - 1
       ELSE
         data%A%ne = A_ptr( m + 1 )
       END IF

       array_name = 'llsr: data%A%ptr'
       CALL SPACE_resize_array( m + 1, data%A%ptr,                             &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%A%col'
       CALL SPACE_resize_array( data%A%ne, data%A%col,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%A%val'
       CALL SPACE_resize_array( data%A%ne, data%A%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%A%ptr( : m + 1 ) = A_ptr( : m + 1 )
         data%A%col( : data%A%ne ) = A_col( : data%A%ne )
       ELSE
         data%A%ptr( : m + 1 ) = A_ptr( : m + 1 ) + 1
         data%A%col( : data%A%ne ) = A_col( : data%A%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%A%type, 'DENSE', data%llsr_inform%alloc_status )
       data%A%ne = m * n

       array_name = 'llsr: data%A%val'
       CALL SPACE_resize_array( data%A%ne, data%A%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%llsr_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%llsr_inform%status
     RETURN

!  End of subroutine LLSR_import

     END SUBROUTINE LLSR_import

!-  G A L A H A D - L L T R _ i m p o r t _ s c a l i n g  S U B R O U T I N E -

     SUBROUTINE LLSR_import_scaling( data, status, S_type, S_ne, S_row,        &
                                     S_col, S_ptr )

!  import fixed problem data for the scaling matrix S into internal
!  storage prior to solution. Arguments are as follows:

!  data is a scalar variable of type LLSR_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, m >= 0 or requirement that the types contain
!       a relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'IDENTITY' has been violated.
!
!  S_type is a character string that specifies the scaling matrix storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense',
!   'diagonal' or 'identity'. Lower or upper case variants are allowed.
!
!  S_ne is a scalar variable of type default integer, that holds the number of
!   entries in the lower triangular part of M in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other schemes.
!
!  S_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of M in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  S_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of M in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  S_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the lower
!   triangular part of M, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( IN ) :: S_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: S_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: n, error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     IF ( data%llsr_control%out > 0 ) WRITE( data%llsr_control%out,            &
            "( '' )", ADVANCE = 'no') !prevents ifort bug
     error = data%llsr_control%error
     space_critical = data%llsr_control%space_critical
     deallocate_error_fatal = data%llsr_control%space_critical

!  recover the dimension

     n = data%A%n

!  set M appropriately in its storage type

     SELECT CASE ( S_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( S_ne ) .AND. PRESENT( S_row ) .AND.               &
                    PRESENT( S_col ) ) ) THEN
         data%llsr_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%S%type, 'COORDINATE', data%llsr_inform%alloc_status )
       data%S%n = n ; data%S%ne = S_ne

       array_name = 'llsr: data%S%row'
       CALL SPACE_resize_array( data%S%ne, data%S%row,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%S%col'
       CALL SPACE_resize_array( data%S%ne, data%S%col,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%S%row( : data%S%ne ) = S_row( : data%S%ne )
         data%S%col( : data%S%ne ) = S_col( : data%S%ne )
       ELSE
         data%S%row( : data%S%ne ) = S_row( : data%S%ne ) + 1
         data%S%col( : data%S%ne ) = S_col( : data%S%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( S_col ) .AND. PRESENT( S_ptr ) ) ) THEN
         data%llsr_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%S%type, 'SPARSE_BY_ROWS',                            &
                     data%llsr_inform%alloc_status )
       data%S%n = n
       IF ( data%f_indexing ) THEN
         data%S%ne = S_ptr( n + 1 ) - 1
       ELSE
         data%S%ne = S_ptr( n + 1 )
       END IF

       array_name = 'llsr: data%S%ptr'
       CALL SPACE_resize_array( n + 1, data%S%ptr,                             &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%S%col'
       CALL SPACE_resize_array( data%S%ne, data%S%col,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       array_name = 'llsr: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%S%ptr( : n + 1 ) = S_ptr( : n + 1 )
         data%S%col( : data%S%ne ) = S_col( : data%S%ne )
       ELSE
         data%S%ptr( : n + 1 ) = S_ptr( : n + 1 ) + 1
         data%S%col( : data%S%ne ) = S_col( : data%S%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%S%type, 'DENSE', data%llsr_inform%alloc_status )
       data%S%n = n ; data%S%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'llsr: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%S%type, 'DIAGONAL', data%llsr_inform%alloc_status )
       data%S%n = n ; data%S%ne = n

       array_name = 'llsr: data%S%val'
       CALL SPACE_resize_array( data%S%ne, data%S%val,                         &
              data%llsr_inform%status, data%llsr_inform%alloc_status,          &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%llsr_inform%bad_alloc, out = error )
       IF ( data%llsr_inform%status /= 0 ) GO TO 900

     CASE ( 'identity', 'IDENTITY' )
       CALL SMT_put( data%S%type, 'IDENTITY', data%llsr_inform%alloc_status )
       data%S%n = n ; data%S%ne = 0

     CASE DEFAULT
       data%llsr_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     data%use_s = .TRUE.
     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%llsr_inform%status
     RETURN

!  End of subroutine LLSR_import_scaling

     END SUBROUTINE LLSR_import_scaling

! - G A L A H A D -  L L S R _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE LLSR_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See LLSR_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( LLSR_control_type ), INTENT( IN ) :: control
     TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%llsr_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine LLSR_reset_control

     END SUBROUTINE LLSR_reset_control

!-  G A L A H A D -  L L S R _ s o l v e _ p r o b l e m  S U B R O U T I N E  -

     SUBROUTINE LLSR_solve_problem( data, status, power, weight, A_val,        &
                                    B, X, S_val )

!  solve the regularized least-squares problem whose structure was previously
!  imported. See LLSR_solve for a description of the required arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type LLSR_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. Possible values are:
!
!    1. The import was succesful, and the package is ready for the solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, power, weight > 0, m >= 0 or requirement that the
!       types contain a relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' or 'IDENTITY' has been violated.
!
!  weight is a scalar of type default real, that holds the positive value
!   of the regularization weight
!
!  weight is a scalar of type default real, that holds the positive value
!   of the regularization weight
!
!  A_val is a one-dimensional array of size a_ne and type default real
!   that holds the values of the entries of the objective Jacobian A.
!
!  B is a rank-one array of dimension m and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal variables, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
!  S_val is an optional one-dimensional array of size s_ne and type default
!   real that holds the values of the entries of the lower triangular part of
!   the scaling matrix M in the storage scheme specified in llsr_import. This
!   need not be given if M is the identity matrix
!

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( IN ) :: power, weight
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: B
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: A_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: S_val

!  local variables

     INTEGER ( KIND = ip_ ) :: n, m

!  recover the dimension

     n = data%A%n ; m = data%A%m

!  save the Jacobian entries

     IF ( data%A%ne > 0 ) data%A%val( : data%A%ne ) = A_val( : data%A%ne )

!  call the solver

     IF ( .NOT. data%use_s ) THEN
       CALL LLSR_solve( m, n, power, weight, data%A, B, X, data%llsr_data,     &
                        data%llsr_control, data%llsr_inform )

     ELSE
       IF ( .NOT. PRESENT( S_val ) ) THEN
         data%llsr_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       IF ( data%S%ne > 0 ) data%S%val( : data%S%ne ) = S_val( : data%S%ne )
       CALL LLSR_solve( m, n, power, weight, data%A, B, X, data%llsr_data,     &
                        data%llsr_control, data%llsr_inform, S = data%S )
     END IF

     status = data%llsr_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%llsr_inform%status
     RETURN

!  End of subroutine LLSR_solve_problem

     END SUBROUTINE LLSR_solve_problem

!-  G A L A H A D -  L L S R _ i n f o r m a t i o n   S U B R O U T I N E  -

      SUBROUTINE LLSR_information( data, inform, status )

!  return solver information during or after solution by LLSR
!  See LLSR_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( LLSR_full_data_type ), INTENT( INOUT ) :: data
      TYPE ( LLSR_inform_type ), INTENT( OUT ) :: inform
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

      inform = data%llsr_inform

!  flag a successful call

      status = GALAHAD_ok
      RETURN

!  end of subroutine LLSR_information

      END SUBROUTINE LLSR_information

!-*-*-*-*-  End of G A L A H A D _ L L S R  d o u b l e  M O D U L E  -*-*-*-*-

    END MODULE GALAHAD_LLSR_precision
