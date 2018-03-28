! THIS VERSION: GALAHAD 3.0 - 11/03/2018 AT 11:00 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ D P S   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally from HSL VF06 June 30th 1997
!   modified for GALAHAD Version 2.5. April 12th 2011
!   extended for the regularization subproblem, March 11th 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DPS_DOUBLE

!      --------------------------------------------
!      | Solve the trust-region subproblem        |
!      |                                          |
!      |    minimize    1/2 <x, H x> + <c, x> + f |
!      |    subject to   <x, M x> <= delta^2      |
!      |                                          |
!      | or the regularized quadratic subproblem  |
!      |                                          |
!      |    minimize    1/2 <x, H x> + <c, x> + f |
!      |                   + (sigma/p) ||x||_M^p  |
!      |                                          |
!      | in a variety of diagonalising norms, M   |
!      --------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_NORMS_double, ONLY: TWO_NORM
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_SLS_double
      USE GALAHAD_TRS_double, ONLY: TRS_solve_diagonal, TRS_control_type,      &
                                    TRS_inform_type
      USE GALAHAD_RQS_double, ONLY: RQS_solve_diagonal, RQS_control_type,      &
                                    RQS_inform_type
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DPS_initialize, DPS_read_specfile, DPS_solve, DPS_resolve,     &
                DPS_terminate, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DPS_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  unit to write problem data into file problem_file

        INTEGER :: problem = 0

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  how much of H has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER :: new_h = 2

!  maximum degree of Taylor approximant allowed

        INTEGER :: taylor_max_degree = 3

!  smallest allowable value of an eigenvalue of the block diagonal factor of H

        REAL ( KIND = wp ) :: eigen_min = SQRT( epsmch )

!  lower and upper bounds on the multiplier, if known

        REAL ( KIND = wp ) :: lower = - half * HUGE( one )
        REAL ( KIND = wp ) :: upper =  HUGE( one )

!  stop trust-region solution when | ||x||_M - delta | <=
!     max( stop_normal * delta, stop_absolute_normal )

        REAL ( KIND = wp ) :: stop_normal = epsmch ** 0.75
        REAL ( KIND = wp ) :: stop_absolute_normal = epsmch ** 0.75

!  use the Goldfarb variant of the trust-region/regularization norm rather
!  than the modified absolute-value version

        LOGICAL :: goldfarb = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  name of file into which to write problem data

        CHARACTER ( LEN = 30 ) :: problem_file =                               &
          'dps_problem.data' // REPEAT( ' ', 14 )

!  symmetric (indefinite) linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "sils" // REPEAT( ' ', 26 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix  = '""                            '

!  control parameters for the Cholesky factorization and solution

        TYPE ( SLS_control_type ) :: SLS_control

     END TYPE DPS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DPS_time_type

!  total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  CPU time spent reordering H prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  CPU time spent factorizing H

        REAL ( KIND = wp ) :: factorize = 0.0

!  CPU time spent solving the diagonal model system

        REAL ( KIND = wp ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time spent reordering H prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  clock time spent factorizing H

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  clock time spent solving the diagonal model system

        REAL ( KIND = wp ) :: clock_solve = 0.0

      END TYPE DPS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DPS_inform_type

!  return status. See DPS_solve for details

        INTEGER :: status = 0

!  STAT value after allocate failure

        INTEGER :: alloc_status = 0

!  the number of 1 by 1 blocks from the factorization of H that were modified
!   when constructing M

        INTEGER :: mod_1by1 = 0

!  the number of 2 by 2 blocks from the factorization of H that were modified
!   when constructing M

        INTEGER :: mod_2by2 = 0

!  the value of the quadratic function

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the regularized quadratic function

        REAL ( KIND = wp ) :: obj_regularized = HUGE( one )

!  the M-norm of the solution

        REAL ( KIND = wp ) :: x_norm = - one

!  the Lagrange multiplier associated with the constraint/regularization

        REAL ( KIND = wp ) :: multiplier = - one

!  a lower bound max(0,-lambda_1), where lambda_1 is the left-most
!  eigenvalue of (H,M)

        REAL ( KIND = wp ) :: pole = zero

!  has the hard case occurred?

        LOGICAL :: hard_case = .FALSE.

!  name of array that provoked an allocate failure

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  time information

        TYPE ( DPS_time_type ) :: time

!  information from SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

      END TYPE DPS_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: DPS_data_type
        PRIVATE
        REAL ( KIND = wp ) :: p, f, old_f, old_obj, old_multiplier
        REAL ( KIND = wp ) :: delta, old_delta, sigma, old_sigma
        INTEGER :: val, ind, nsteps, maxfrt, latop
        LOGICAL :: old_on_boundary, old_convex, trs
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: PERM
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: EVAL, MOD_EVAL
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: EVECT, CS, D
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: B
        TYPE ( DPS_control_type ) :: control
        TYPE ( SLS_control_type ) :: SLS_control
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( TRS_control_type ) :: TRS_control
        TYPE ( TRS_inform_type ) :: TRS_inform
        TYPE ( RQS_control_type ) :: RQS_control
        TYPE ( RQS_inform_type ) :: RQS_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-  D P S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE DPS_initialize( data, control, inform )

!      .....................................
!      .                                   .
!      .  Set initial values for the DPS   .
!      .  control parameters               .
!      .                                   .
!      .....................................

!  Arguments:
!  =========

!   data     private internal data
!   control  a structure containing control information. Components are -
!            error       error message output unit
!            out         information message output unit
!            print_level print level. > 0 for output

!-----------------------------------------------
!   D u m m y   A r g u m e n t
!-----------------------------------------------

      TYPE ( DPS_data_type ), INTENT( OUT ) :: data
      TYPE ( DPS_control_type ), INTENT( OUT ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform

!  initalize SLS components

      CALL SLS_initialize( control%symmetric_linear_solver,                    &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform )
      data%SLS_control%scaling = 0

!  ensure that the initial value of the "old" delta is small

      data%old_delta = zero ; data%old_sigma = zero
      data%old_on_boundary = .FALSE. ; data%old_convex = .FALSE.

      RETURN

!  End of subroutine DPS_initialize

      END SUBROUTINE DPS_initialize

!-*-*-*-*-   D P S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE DPS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by DPS_initialize could (roughly)
!  have been set as:

!  BEGIN DPS SPECIFICATIONS (DEFAULT)
!   error-printout-device                          6
!   printout-device                                6
!   problem-device                                 0
!   print-level                                    0
!   has-h-changed                                  2
!   max-degree-taylor-approximant                  3
!   smallest-eigenvalue-allowed                    1.0D-8
!   lower-bound-on-multiplier                      0.0
!   upper-bound-on-multiplier                      1.0D+300
!   stop-normal-case                               1.0D-12
!   stop-hard-case                                 1.0D-12
!   build-goldfarb-preconditioner                  F
!   space-critical                                 F
!   deallocate-error-fatal                         F
!   symmetric-linear-equation-solver               sils
!   problem-file                                   dps_problem.data
!   output-line-prefix                             ""
!  END DPS SPECIFICATIONS

!  Dummy arguments

      TYPE ( DPS_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: problem = out + 1
      INTEGER, PARAMETER :: print_level = problem + 1
      INTEGER, PARAMETER :: new_h = print_level + 1
      INTEGER, PARAMETER :: taylor_max_degree = new_h
      INTEGER, PARAMETER :: eigen_min = taylor_max_degree + 1
      INTEGER, PARAMETER :: lower = eigen_min + 1
      INTEGER, PARAMETER :: upper = lower + 1
      INTEGER, PARAMETER :: stop_normal = upper + 1
      INTEGER, PARAMETER :: stop_absolute_normal = stop_normal + 1
      INTEGER, PARAMETER :: goldfarb = stop_absolute_normal + 1
      INTEGER, PARAMETER :: space_critical = goldfarb + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: problem_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: symmetric_linear_solver = problem_file + 1
      INTEGER, PARAMETER :: prefix = symmetric_linear_solver + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'DPS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( problem )%keyword = 'problem-device'
      spec( print_level )%keyword = 'print-level'
      spec( new_h )%keyword = 'has-h-changed'
      spec( taylor_max_degree )%keyword = 'max-degree-taylor-approximant'

!  Real key-words

      spec( eigen_min )%keyword = 'smallest-eigenvalue-allowed'
      spec( lower )%keyword = 'lower-bound-on-multiplier'
      spec( upper )%keyword = 'upper-bound-on-multiplier'
      spec( stop_normal )%keyword = 'stop-normal-case'
      spec( stop_absolute_normal )%keyword = 'stop-absolute-normal-case'

!  Logical key-words

      spec( goldfarb )%keyword = 'build-goldfarb-preconditioner'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( problem_file )%keyword = 'problem-file'
      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
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
      CALL SPECFILE_assign_value( spec( problem ),                             &
                                  control%problem,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_h ),                               &
                                  control%new_h,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( taylor_max_degree ),                   &
                                  control%taylor_max_degree,                   &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( eigen_min ),                           &
                                  control%eigen_min,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( lower ),                               &
                                  control%lower,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( upper ),                               &
                                  control%upper,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_normal ),                         &
                                  control%stop_normal,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_absolute_normal ),                &
                                  control%stop_absolute_normal,                &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( goldfarb ),                            &
                                  control%goldfarb,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( problem_file ),                        &
                                  control%problem_file,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),             &
                                  control%symmetric_linear_solver,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read specfile data for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

      RETURN

!  End of subroutine DPS_read_specfile

      END SUBROUTINE DPS_read_specfile

!-*-*-*-*-*-*-*-*-*-*  D P S _ S O L V E   S U B R O U T I N E  -*-*-*-*-*-*-*

      SUBROUTINE DPS_solve( n, H, C, f, X, data, control, inform,             &
                            delta, sigma, p )

!      .............................................
!      .                                           .
!      .  Obtain a diagonalising preconditioner    .
!      .  M, and solve the trust-region problem    .
!      .                                           .
!      .    minimize    1/2 <x, H x> + <c, x> + f  .
!      .    subject to  <x, M x> <= delta^2        .
!      .                                           .
!      . or the regularized quadratic subproblem   .
!      .                                           .
!      .    minimize    1/2 <x, H x> + <c, x> + f  .
!      .                  + (sigma/p) ||x||_M^p    .
!      .                                           .
!      .............................................

!  Arguments:
!  =========
!
!   n        number of unknowns
!   H        the matrix H
!   C        the vector c
!   f        the scalar f
!   X        the required solution vector. Need not be set on entry.
!            On exit, the optimal value
!   data     private internal data
!   control  a structure containing control information. See DPS_control
!   inform   a structure containing information. See DPS_inform
!   delta    trust-region radius (optional)
!   sigma    regularization weight (optional)
!   p        regularization order (optional)
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( IN ) :: f
      TYPE ( SMT_type ), INTENT( INOUT ) :: H
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: C
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: delta
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: sigma
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: p

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
           prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  check that input data is correct

      IF ( n <= 0 ) THEN
         IF ( control%error > 0 ) WRITE( control%error,                        &
           "( A, ' n = ', I6, ' is not positive ' )" ) prefix, n
        inform%status = GALAHAD_error_restrictions ; RETURN
      END IF

      IF ( PRESENT( delta ) ) THEN
        IF ( delta <= zero ) THEN
          IF ( control%error > 0 ) WRITE( control%error,                       &
           "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, delta
          inform%status = GALAHAD_error_restrictions ; RETURN
        END IF
        data%delta = delta ; data%trs = .TRUE.
      ELSE IF ( PRESENT( sigma ) ) THEN
        IF ( sigma <= zero ) THEN
          IF ( control%error > 0 ) WRITE( control%error,                       &
           "( A, ' The regularization weight ', ES12.4 ,                       &
          &  ' is not positive' )" ) prefix, sigma
          inform%status = GALAHAD_error_restrictions ; RETURN
        END IF
        IF ( PRESENT( p ) ) THEN
          IF ( p <= two ) THEN
            IF ( control%error > 0 ) WRITE( control%error,                     &
             "( A, ' The regularization power ', ES12.4 ,                      &
            &  ' is smaller than two' )" ) prefix, p
            inform%status = GALAHAD_error_restrictions ; RETURN
          END IF
          data%p = p
        ELSE
          data%p = three
        END IF
        data%sigma = sigma ; data%trs = .FALSE.
      ELSE
        IF ( control%error > 0 ) WRITE( control%error, "( A, ' Neither of ',   &
       &  'the optional arguments delta nor sigma is present')" ) prefix
        inform%status = GALAHAD_unsuitable_option ; RETURN
      END IF
      H%n = n

!  set control parameters for the main sub-problem iteration

      data%TRS_control%prefix = control%prefix
      data%TRS_control%problem = control%problem
      data%TRS_control%problem_file = control%problem_file
      data%TRS_control%out = control%out
      data%TRS_control%print_level = control%print_level
      data%TRS_control%equality_problem = .FALSE.
      data%TRS_control%lower = control%lower
      data%TRS_control%upper = control%upper
      data%TRS_control%taylor_max_degree = control%taylor_max_degree
      data%TRS_control%stop_normal = control%stop_normal
      data%TRS_control%stop_absolute_normal= control%stop_absolute_normal

      data%RQS_control%prefix = control%prefix
      data%RQS_control%problem = control%problem
      data%RQS_control%problem_file = control%problem_file
      data%RQS_control%out = control%out
      data%RQS_control%print_level = control%print_level
      data%RQS_control%taylor_max_degree = control%taylor_max_degree
      data%RQS_control%stop_normal = control%stop_normal

!  obtain the trust-region norm

      CALL DPS_build_preconditioner( n, H, data, control, inform )
      IF ( inform%status < 0 ) RETURN
      data%old_convex = inform%mod_1by1 == 0 .AND. inform%mod_2by2 == 0
      data%f = f

!  solve the TR problem

      CALL DPS_resolve( n, X, data, control, inform, C = C, f = f,             &
                        delta = delta, sigma = sigma )
      RETURN

!  End of subroutine DPS_solve

      END SUBROUTINE DPS_solve

!-*-*-*-*-*-*-*-  D P S _ R E S O L V E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE DPS_resolve( n, X, data, control, inform,                     &
                              C, f, delta, sigma, p )

!      .................................................
!      .                                               .
!      .  solve the trust-region problem               .
!      .                                               .
!      .    minimize     1/2 <x, H x> + <c, x> + f     .
!      .    subject to   <x, M x> <= delta^2           .
!      .                                               .
!      . or the regularized quadratic subproblem       .
!      .                                               .
!      .    minimize     1/2 <x, H x> + <c, x> + f     .
!      .                   + (sigma/p) ||x||_M^p       .
!      .                                               .
!      .  where M is a diagonilising norm for H        .
!      .                                               .
!      .................................................

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: f
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: C
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: delta
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: sigma
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: p

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER :: i
      REAL ( KIND = wp ) :: temp
      LOGICAL :: oneby1, printd, printe, debug
      CHARACTER ( LEN = 8 ) :: bad_alloc
      CHARACTER ( LEN = 80 ) :: array_name
!     REAL ( KIND = wp ), DIMENSION( n ) :: P

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Step 1: Initialize
!  ======

      printe = control%error > 0 .AND. control%print_level >= 1
      printd = control%out > 0 .AND. control%print_level >= 3
      debug = control%out > 0 .AND. control%print_level >= 10
      inform%status = 0

      IF ( PRESENT( delta ) ) THEN
        IF ( delta <= zero ) THEN
          IF ( control%error > 0 ) WRITE( control%error,                       &
           "( A, ' The radius ', ES12.4 , ' is not positive ' )" ) prefix, delta
          inform%status = GALAHAD_error_restrictions ; RETURN
        END IF
        data%delta = delta
      ELSE IF ( PRESENT( sigma ) ) THEN
        IF ( sigma <= zero ) THEN
          IF ( control%error > 0 ) WRITE( control%error,                       &
           "( A, ' The regularization weight ', ES12.4 ,                       &
          &  ' is not positive' )" ) prefix, sigma
          inform%status = GALAHAD_error_restrictions ; RETURN
        END IF
        IF ( PRESENT( p ) ) THEN
          IF ( p <= two ) THEN
            IF ( control%error > 0 ) WRITE( control%error,                     &
             "( A, ' The regularization power ', ES12.4 ,                      &
            &  ' is smaller than two' )" ) prefix, p
            inform%status = GALAHAD_error_restrictions ; RETURN
          END IF
          data%p = p
        ELSE
          data%p = three
        END IF
        data%sigma = sigma
      END IF

!  update the constant term in the objective function if requested

      IF ( PRESENT( f ) ) THEN
        data%old_f = data%f
        data%f = f
      END IF

!  allocate the arrays for the solution phase

      IF ( PRESENT( C ) ) THEN
        array_name = 'dps: data$CS'
        CALL SPACE_resize_array( n, data%CS,                                   &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= 0 ) GO TO 910

        IF ( debug ) WRITE( control%out, "( A, 1X, A, /, ( 10I8 ) )" )         &
          prefix,  'PERM', data%PERM( : n )
        data%CS = C
        IF ( debug ) WRITE( control%out, 2000 )                                &
          prefix, 'data%CS', data%CS( : n )

!  Step 2: Solve P L c_b = c
!  ======

        CALL SLS_part_solve( 'L', data%CS, data%SLS_data, data%SLS_control,    &
                             inform%SLS_inform )
        IF ( debug ) WRITE( control%out, 2000 )                                &
          prefix, 'data%CS', data%CS( : n )

!  Step 3: Obtain c_s = Gamma(-1/2) Q^T P^T c_b
!  ======

        data%CS = data%CS( ABS( data%PERM ) )
        IF ( debug ) WRITE( control%out, 2000 )                                &
          prefix, 'data%CS', data%CS( : n )

        oneby1 = .TRUE.
        DO i = 1, n
          IF ( oneby1 ) THEN
            IF ( i < n ) THEN
              oneby1 = data%PERM( i ) > 0
            ELSE
              oneby1 = .TRUE.
            END IF

!  2x2 pivot

            IF ( .NOT. oneby1 ) THEN
              temp = data%EVECT( i + 1 ) * data%CS( i ) -                      &
                     data%EVECT( i ) * data%CS( i + 1 )
              data%CS( i ) = data%EVECT( i ) * data%CS( i ) +                  &
                             data%EVECT( i + 1 ) * data%CS( i + 1 )

              data%CS( i + 1 ) = temp / SQRT( data%MOD_EVAL( i + 1 ) )
            END IF
            data%CS( i ) = data%CS( i ) / SQRT( data%MOD_EVAL( i ) )
          ELSE
            oneby1 = .TRUE.
          END IF
        END DO

!  special case: the previous problem was convex, and the solution lay on
!  the trust region boundary. The new solution is then simply a rescaled
!  version of the former solution

      ELSE
        IF ( data%trs ) THEN
          IF ( data%old_convex .AND. data%old_on_boundary .AND.                &
               data%delta <= data%old_delta ) THEN
            temp = data%delta / data%old_delta
            inform%obj = data%f + temp * ( data%old_obj - data%old_f )         &
                           + half * data%delta * ( data%delta - data%old_delta )
            X = temp * X
            inform%multiplier = ( one + data%old_multiplier ) / temp - one
            inform%x_norm = data%delta
            GO TO 900
          END IF
        END IF
      END IF

!  Step 4: Find x_s by solving the diagonal problem
!  ======

      IF ( debug ) WRITE( control%out, 2000 ) prefix, 'data%CS', data%CS( : n )

!  record the diagonal Hessian

      array_name = 'dps: data%D'
      CALL SPACE_resize_array( n, data%D,                                      &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      data%D( : n ) = data%EVAL( : n ) / data%MOD_EVAL( : n )

!  trust-region subproblem

      IF ( data%trs ) THEN

!  special case: D = I

        IF ( COUNT( data%D /= one ) == 0 ) THEN
          inform%x_norm = TWO_NORM( data%CS )
          IF ( inform%x_norm <= data%delta ) THEN
            X = - data%CS
            inform%multiplier = zero
          ELSE
            X = - ( data%delta / inform%x_norm ) * data%CS
            inform%multiplier = inform%x_norm / data%delta - one
            inform%x_norm = data%delta
          END IF
          inform%obj = data%f + half * ( DOT_PRODUCT( data%CS, X )             &
                         - inform%multiplier * data%delta * data%delta )
          inform%hard_case = .FALSE.
          inform%pole = zero
          IF ( printd ) WRITE( control%out,                                    &
            "( ' multiplier, step, delta, error', 4ES12.4 )" )                 &
              inform%multiplier, inform%x_norm, data%delta,                    &
              inform%x_norm - data%delta

!  usual case

        ELSE
          CALL TRS_solve_diagonal( n, data%delta, data%f, data%CS, data%D,     &
                                   X, data%TRS_control, data%TRS_inform )

!  record exit information and check for error returns

          inform%status = data%TRS_inform%status
          IF ( inform%status /= GALAHAD_ok ) RETURN
          inform%x_norm = data%TRS_inform%x_norm
          inform%obj = data%TRS_inform%obj
          inform%multiplier = data%TRS_inform%multiplier
          inform%hard_case = data%TRS_inform%hard_case
          inform%pole = data%TRS_inform%pole
          inform%time%solve = data%TRS_inform%time%total
          inform%time%clock_solve = data%TRS_inform%time%clock_total
        END IF

!  regularization subproblem

      ELSE
        CALL RQS_solve_diagonal( n, data%p, data%sigma, data%f, data%CS,       &
                                 data%D, X, data%RQS_control,                  &
                                 data%RQS_inform )

!  record exit information and check for error returns

        inform%status = data%RQS_inform%status
        IF ( inform%status /= GALAHAD_ok ) RETURN
        inform%x_norm = data%RQS_inform%x_norm
        inform%obj = data%RQS_inform%obj
        inform%obj_regularized = data%RQS_inform%obj_regularized
        inform%multiplier = data%RQS_inform%multiplier
        inform%hard_case = data%RQS_inform%hard_case
        inform%pole = data%RQS_inform%pole
        inform%time%solve = data%RQS_inform%time%total
        inform%time%clock_solve = data%RQS_inform%time%clock_total
      END IF

      IF ( debug ) WRITE( control%out, 2000 ) prefix, 'X', X( : n )

!  Step 5 (alternative): Recover x_r = P Q Gamma(-1/2) x_s
!  ======

      oneby1 = .TRUE.
      DO i = 1, n
        IF ( oneby1 ) THEN
          IF ( i < n ) THEN
            oneby1 = data%PERM( i ) > 0
          ELSE
            oneby1 = .TRUE.
          END IF
          X( i ) = X( i ) / SQRT( data%MOD_EVAL( i ) )

!  2x2 pivot

          IF ( .NOT. oneby1 ) THEN
            temp = X( i + 1 ) / SQRT( data%MOD_EVAL( i + 1 ) )
            X( i + 1 ) = data%EVECT( i + 1 ) * X( i ) -                        &
                         data%EVECT( i ) * temp
            X( i ) = data%EVECT( i ) * X( i ) + data%EVECT( i + 1 ) * temp
          END IF
        ELSE
          oneby1 = .TRUE.
        END IF
      END DO
      X( ABS( data%PERM ) ) = X

!  Step 6: (Alternative) Solve P L(trans) P(trans) x = x_r
!  ======

      CALL SLS_part_solve( 'U', X, data%SLS_data, data%SLS_control,            &
                           inform%SLS_inform )
      IF ( debug ) WRITE( control%out, 2000 ) prefix, 'X', X( : n )

!  successful return

  900 CONTINUE
      IF ( data%trs ) THEN
        data%old_delta = data%delta
      ELSE
        data%old_sigma = data%sigma
      END IF
      data%old_multiplier = inform%multiplier
      data%old_obj = inform%obj ; data%old_f = data%f
      data%old_on_boundary = inform%multiplier > zero
      RETURN

!  unsuccessful returns

  910 CONTINUE
      IF ( control%error > 0 .AND. control%print_level >= 1 )                  &
           WRITE( control%error, "( A, ' Message from DPS_resolve', /,         &
     &            ' Allocation error, for ', A, ', status = ', I0 )" )         &
        prefix, bad_alloc, inform%SLS_inform%alloc_status
      RETURN

!  Non-executable statement

 2000 FORMAT( A, 1X, A, /, ( 6ES12.4 ) )

!  End of subroutine DPS_resolve

      END SUBROUTINE DPS_resolve

!-*-  D P S _ B U I L D _ P R E C O N D I T I O N E R  S U B R O U T I N E   -*-

      SUBROUTINE DPS_build_preconditioner( n, H, data, control, inform )

!      ..........................................
!      .                                        .
!      .  Obtain a diagonalising preconditioner .
!      .  of the sparse symmetric matrix H      .
!      .                                        .
!      ..........................................

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n
      TYPE ( SMT_type ), INTENT( INOUT ) :: H
      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: zeig, rank, out
      REAL :: time_start, time_now, time_record
      REAL ( KIND = wp ) :: clock_start, clock_now, clock_record
      LOGICAL :: printi, printt
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
           prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  set initial values

      CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  set output levels

      out = control%out

!  record desired output level

      printi = out > 0 .AND. control%print_level > 0
      printt = out > 0 .AND. control%print_level > 1

!  Analyse the sparsity pattern of H
! ::::::::::::::::::::::::::::::::::

      IF ( control%new_H == 2 ) THEN

!  set up linear equation solver-dependent data

        CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_initialize_solver( control%symmetric_linear_solver,           &
                                    data%SLS_data, inform%SLS_inform )

!  perform the analysis

        CALL SLS_analyse( H, data%SLS_data, data%SLS_control,                  &
                          inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse =                                            &
          inform%time%clock_analyse + clock_now - clock_record
        IF ( printt ) WRITE( out, "( A, ' time( SLS_analyse ) = ', F0.2 )" )   &
          prefix, clock_now - clock_record

!  test that the analysis succeeded

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printi ) WRITE( out, "( A, ' error return from ',               &
         &  'SLS_analyse: status = ', I0 )" ) prefix, inform%SLS_inform%status
          inform%status = GALAHAD_error_analysis ;  GO TO 910 ; END IF
      END IF

!  Factorize H
! ::::::::::::

      CALL CPU_time( time_record ) ; CALL CLOCK_time( clock_record )
      CALL SLS_FACTORIZE( H, data%SLS_data, data%SLS_control,                  &
                          inform%SLS_inform )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%factorize = inform%time%factorize + time_now - time_record
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + clock_now - clock_record
      IF ( printt ) WRITE( out, "( A, ' time( SLS_factorize ) = ', F0.2 )" )   &
        prefix, clock_now - clock_record

!  test that the factorization succeeded

      IF ( inform%SLS_inform%status < 0 ) GO TO 920
      rank = inform%SLS_inform%rank
      IF ( rank /= n .AND. printt ) WRITE( control%out,                        &
        "( A, 1X, I0, ' zero eigenvalues ' )" ) prefix, n - rank
      zeig = n - rank

!  allocate further arrays

      array_name = 'dps: data%PERM'
      CALL SPACE_resize_array( n, data%PERM,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'dps: data%EVAL'
      CALL SPACE_resize_array( n, data%EVAL,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'dps: data%MOD_EVAL'
      CALL SPACE_resize_array( n, data%MOD_EVAL,                               &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'dps: data%EVECT'
      CALL SPACE_resize_array( n, data%EVECT,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

      array_name = 'dps: data%B'
      CALL SPACE_resize_array( 2, n, data%B,                                   &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= 0 ) GO TO 910

!  Modify the factorization to produce the preconditioner
!  ::::::::::::::::::::::::::::::::::::::::::::::::::::::

      CALL DPS_modify_factors( n, rank, control%goldfarb, control%eigen_min,   &
                               inform%mod_1by1, inform%mod_2by2,               &
                               data%SLS_data, inform%SLS_inform,               &
                               data%PERM( : n ), data%EVAL( : n ),             &
                               data%MOD_EVAL( : n ), data%EVECT( : n ),        &
                               data%B( : 2, : n ) )

!  test that the modification succeeded

      IF ( inform%SLS_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_alter_diagonal
      ELSE
        inform%status = GALAHAD_ok
      END IF
      RETURN

!  general error

  910 CONTINUE
      IF ( control%out > 0 .AND. control%print_level > 0 )                     &
        WRITE( control%out, "( A, '   **  Error return ', I0,                  &
        & ' from TRS ' )" ) prefix, inform%status
      RETURN

!  factorization failure

  920 CONTINUE
      IF ( printi ) WRITE( out, "( A, ' error return from ',                   &
     &   'SLS_factorize: status = ', I0 )" ) prefix, inform%SLS_inform%status
      inform%status = GALAHAD_error_factorization
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  End of subroutine DPS_build_preconditioner

      END SUBROUTINE DPS_build_preconditioner

!-*-*-*-*-  D P S _ M O D I F Y _ F A C T O R S   S U B R O U T I N E   -*-*-*

      SUBROUTINE DPS_modify_factors( n, rank, goldfarb, eigen_min,             &
                                     mod1, mod2, data, inform,                 &
                                     PERM, EVAL, MOD_EVAL, EVECT, B )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!   modify the factors of
!
!       H = P L Q D Q^T L^T P^T,
!
!   where L is lower triangular, P a permutation, Q orthogonal and D diagonal,
!   to obtain the modified absolute-value preconditioner (goldfarb = .FALSE.)
!
!       M = P L Q |D| Q^T L^T P^T
!
!   or the Goldfarb variant (goldfarb = .TRUE.)
!
!       M = P L L^T P^T
!
!   Based on the Gill-Murray-Ponceleon-Saunders code for modifying the negative
!   eigen-components obtained when factorizing a symmetric indefinite
!   matrix using the GALAHAD package SLS. (See SOL 90-8, P.19-21)
!
!   Also extracts eigenvalues and eigenvectors, and the modified eigenvalues
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      INTEGER, INTENT( IN ) :: n, rank
      INTEGER, INTENT( OUT ) :: mod1, mod2
      LOGICAL, INTENT( IN ) :: goldfarb
      REAL ( KIND = wp ), INTENT( IN ) :: eigen_min
      INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: EVAL, MOD_EVAL
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: EVECT
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: B

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER :: i
      REAL ( KIND = wp ) :: alpha, beta, gamma, tau
      REAL ( KIND = wp ) :: t, c , s, e1, e2, eigen
      LOGICAL :: oneby1

      CALL SLS_ENQUIRE( data, inform, PIVOTS = PERM, D = B )
      IF ( inform%status /= GALAHAD_ok ) RETURN
      B( 1, rank + 1 : n ) = zero

!  ----------------------
!  Absolute-value variant
!  ----------------------

      IF ( .NOT. goldfarb ) THEN

!  mod1 and mod2 are the number of negative eigenvalues which arise
!  from small or negative 1x1 and 2x2 block pivots

        mod1 = 0 ; mod2 = 0

!  loop over all the block pivots

        oneby1 = .TRUE.
        DO i = 1, n

!  decide if the current block is a 1x1 or 2x2 pivot block

          IF ( oneby1 ) THEN
            IF ( i < n ) THEN
              oneby1 = PERM( i ) > 0
            ELSE
              oneby1 = .TRUE.
            END IF
            alpha = B( 1, i )

!  =========
!  1x1 block
!  =========

            IF ( oneby1 ) THEN

!  record the eigenvalue

              IF ( alpha /= zero ) THEN
                eigen = one / alpha
              ELSE
                eigen = zero
              END IF
              EVAL( i ) = eigen

!  negative 1x1 block
!  ------------------

              IF ( eigen < - eigen_min ) THEN
                mod1 = mod1 + 1
                B( 1, i ) = - alpha

!  record the modification

                MOD_EVAL( i ) = - eigen

!  small 1x1 block
!  ---------------

              ELSE IF ( eigen < eigen_min ) THEN
                mod1 = mod1 + 1
                B( 1, i ) = one / eigen_min

!  record the modification

                MOD_EVAL( i ) = eigen_min

!  positive 1x1 block
!  ------------------

              ELSE

!  record the modification

                MOD_EVAL( i ) = eigen
              END IF

!  record the eigenvector

              EVECT( i ) = one

!  =========
!  2x2 block
!  =========

            ELSE
              beta = B( 2, i )
              gamma = B( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

              IF ( alpha * gamma < beta ** 2 ) THEN
                tau = ( gamma - alpha ) / ( two * beta )
                t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
                IF ( tau < zero ) t = - t
                c = one / SQRT( one + t ** 2 ) ; s = t * c
                e1 = alpha + beta * t ; e2 = gamma - beta * t

!  record the first eigenvalue

                eigen = one / e1
                EVAL( i ) = eigen

!  change e1 and e2 to their modified values and then multiply the
!  three 2 * 2 matrices to get the modified alpha, beta and gamma

                IF ( eigen < - eigen_min ) THEN

!  negative first eigenvalue
!  -------------------------

                  mod2 = mod2 + 1

!  record the modification

                  MOD_EVAL( i ) = - eigen
                  e1 = - e1

!  small first eigenvalue
!  ----------------------

                ELSE IF ( eigen < eigen_min ) THEN
                  mod2 = mod2 + 1

!  record the modification

                  MOD_EVAL( i ) = eigen_min
                  e1 = one / eigen_min

!  positive first eigenvalue
!  -------------------------

                ELSE

!  record the modification

                  MOD_EVAL( i ) = eigen
                END IF

!  record the second eigenvalue

                eigen = one / e2
                EVAL( i + 1 ) = eigen


!  negative second eigenvalue
!  --------------------------

                IF ( eigen < - eigen_min ) THEN
                  mod2 = mod2 + 1

!  record the modification

                  MOD_EVAL( i + 1 ) = - eigen
                  e2 = - e2

!  small second eigenvalue
!  -----------------------

                ELSE IF ( eigen < eigen_min ) THEN
                  mod2 = mod2 + 1

!  record the modification

                  MOD_EVAL( i + 1 ) = eigen_min
                  e2 = one / eigen_min

!  positive second eigenvalue
!  --------------------------

                ELSE

!  record its modification

                  MOD_EVAL( i + 1 ) = eigen
                END IF

!  record the modified block

                B( 1, i ) = c ** 2 * e1 + s ** 2 * e2
                B( 2, i ) = c * s * ( e1 - e2 )
                B( 1, i + 1 ) = s ** 2 * e1 + c ** 2 * e2

!  positive 2 by 2 block
!  ---------------------

              ELSE
                IF ( beta /= zero ) THEN
                  tau = ( gamma - alpha ) / ( two * beta )
                  t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
                  IF ( tau < zero ) t = - t
                  c = one / SQRT( one + t ** 2 ) ;  s = t * c
                  e1 = alpha + beta * t ; e2 = gamma - beta * t
                ELSE
                  c = one ; s = zero
                  e1 = alpha ; e2 = gamma
                END IF

!  record the eigenvalue and its modification

                EVAL( i ) = one / e1
                MOD_EVAL( i ) = EVAL( i )
                EVAL( i + 1 ) = one / e2
                MOD_EVAL( i + 1 ) = one / EVAL( i + 1 )
              END IF

!  record the eigenvector

              EVECT( i ) = c
              EVECT( i + 1 ) = s
            END IF
          ELSE
            oneby1 = .TRUE.
          END IF
        END DO

!  ----------------
!  Goldfarb variant
!  ----------------

      ELSE

!  mod1 and mod2 are the number of negative eigenvalues which arise
!  from small or negative 1x1 and 2x2 block pivots

        mod1 = 0 ; mod2 = 0

!  loop over all the block pivots

        oneby1 = .TRUE.
        DO i = 1, n

!  decide if the current block is a 1x1 or 2x2 pivot block

          IF ( oneby1 ) THEN
            IF ( i < n ) THEN
              oneby1 = PERM( i ) > 0
            ELSE
              oneby1 = .TRUE.
            END IF
            alpha = B( 1, i )

!  =========
!  1x1 block
!  =========

            IF ( oneby1 ) THEN

!  record the eigenvalue

              IF ( alpha /= zero ) THEN
                EVAL( i ) = one / alpha
              ELSE
                EVAL( i ) = zero
              END IF

!  record the modification

              mod1 = mod1 + 1
              MOD_EVAL( i ) = one
              B( 1, i ) = one

!  record the eigenvector

              EVECT( i ) = one

!  =========
!  2x2 block
!  =========

            ELSE
              beta = B( 2, i )
              gamma = B( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

              IF ( alpha * gamma < beta ** 2 ) THEN
                tau = ( gamma - alpha ) / ( two * beta )
                t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
                IF ( tau < zero ) t = - t
                c = one / SQRT( one + t ** 2 ) ; s = t * c
                e1 = alpha + beta * t ; e2 = gamma - beta * t
              ELSE
                IF ( beta /= zero ) THEN
                  tau = ( gamma - alpha ) / ( two * beta )
                  t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
                  IF ( tau < zero ) t = - t
                  c = one / SQRT( one + t ** 2 ) ;  s = t * c
                  e1 = alpha + beta * t ; e2 = gamma - beta * t
                ELSE
                  c = one ; s = zero
                  e1 = alpha ; e2 = gamma
                END IF
              END IF

!  record the eigenvalues

              EVAL( i ) = one / e1
              EVAL( i + 1 ) = one / e2

!  record the modifications

              mod2 = mod2 + 2
              MOD_EVAL( i ) = one
              MOD_EVAL( i + 1 ) = one

!  record the modified block

              B( 1, i ) = one
              B( 2, i ) = zero
              B( 1, i + 1 ) = one

!  record the eigenvector

              EVECT( i ) = c
              EVECT( i + 1 ) = s
            END IF
          ELSE
            oneby1 = .TRUE.
          END IF
        END DO
      END IF

!  register the (possibly modified) diagonal blocks

      CALL SLS_alter_D( data, B, inform )

      RETURN

!  End of subroutine DPS_modify_factors

      END SUBROUTINE DPS_modify_factors

!-*-*-*-*-   H S L _ D P S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*

      SUBROUTINE DPS_terminate( data, control, inform )

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    private internal data
!   control see preface
!   inform  see preface

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      CHARACTER ( LEN = 80 ) :: array_name

      inform%status = 0

!  deallocate all internal arrays

      array_name = 'trs: data%PERM'
      CALL SPACE_dealloc_array( data%PERM,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: data%D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: data%EVAL'
      CALL SPACE_dealloc_array( data%EVAL,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: data%MOD_EVAL'
      CALL SPACE_dealloc_array( data%MOD_EVAL,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: data%EVECT'
      CALL SPACE_dealloc_array( data%EVECT,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'trs: data%CS'
      CALL SPACE_dealloc_array( data%CS,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      CALL SLS_terminate( data%SLS_data, data%SLS_control, inform%SLS_inform )
      IF ( inform%SLS_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'trs: data%SLS_data'
      END IF

      RETURN

!  End of subroutine DPS_terminate

      END SUBROUTINE DPS_terminate

!  End of module GALAHAD_DPS

    END MODULE GALAHAD_DPS_DOUBLE
