! THIS VERSION: GALAHAD 5.1 - 2024-10-04 AT 14:10 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ D P S   M O D U L E  *-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally from HSL VF06 June 30th 1997
!   modified for GALAHAD Version 2.5. April 12th 2011
!   extended for the regularization subproblem, March 11th 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_DPS_precision

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

      USE GALAHAD_KINDS_precision
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_precision
      USE GALAHAD_NORMS_precision, ONLY: TWO_NORM
      USE GALAHAD_SPECFILE_precision
      USE GALAHAD_SMT_precision
      USE GALAHAD_SLS_precision
      USE GALAHAD_TRS_precision, ONLY: TRS_solve_diagonal, TRS_control_type,   &
                                       TRS_inform_type
      USE GALAHAD_RQS_precision, ONLY: RQS_solve_diagonal, RQS_control_type,   &
                                       RQS_inform_type
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: DPS_initialize, DPS_read_specfile, DPS_solve, DPS_resolve,     &
                DPS_terminate, DPS_full_initialize, DPS_full_terminate,        &
                DPS_import, DPS_solve_tr_problem, DPS_solve_rq_problem,        &
                DPS_resolve_tr_problem, DPS_resolve_rq_problem,                &
                DPS_reset_control, DPS_information, SMT_type, SMT_put, SMT_get

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE DPS_initialize
       MODULE PROCEDURE DPS_initialize, DPS_full_initialize
     END INTERFACE DPS_initialize

     INTERFACE DPS_terminate
       MODULE PROCEDURE DPS_terminate, DPS_full_terminate
     END INTERFACE DPS_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: half = 0.5_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: two = 2.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: three = 3.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DPS_control_type

!  unit for error messages

        INTEGER ( KIND = ip_ ) :: error = 6

!  unit for monitor output

        INTEGER ( KIND = ip_ ) :: out = 6

!  unit to write problem data into file problem_file

        INTEGER ( KIND = ip_ ) :: problem = 0

!  controls level of diagnostic output

        INTEGER ( KIND = ip_ ) :: print_level = 0

!  how much of H has changed since the previous call. Possible values are
!   0  unchanged
!   1  values but not indices have changed
!   2  values and indices have changed

        INTEGER ( KIND = ip_ ) :: new_h = 2

!  maximum degree of Taylor approximant allowed

        INTEGER ( KIND = ip_ ) :: taylor_max_degree = 3

!  smallest allowable value of an eigenvalue of the block diagonal factor of H

        REAL ( KIND = rp_ ) :: eigen_min = SQRT( epsmch )

!  lower and upper bounds on the multiplier, if known

        REAL ( KIND = rp_ ) :: lower = - half * HUGE( one )
        REAL ( KIND = rp_ ) :: upper =  HUGE( one )

!  stop trust-region solution when | ||x||_M - delta | <=
!     max( stop_normal * delta, stop_absolute_normal )

!       REAL ( KIND = rp_ ) :: stop_normal = epsmch ** 0.75
!       REAL ( KIND = rp_ ) :: stop_absolute_normal = epsmch ** 0.75
        REAL ( KIND = rp_ ) :: stop_normal = ten ** ( - 12 )
        REAL ( KIND = rp_ ) :: stop_absolute_normal = ten ** ( - 12 )

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

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver = "ssids" //         &
                                                             REPEAT( ' ', 25 )

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

        REAL ( KIND = rp_ ) :: total = 0.0

!  CPU time spent reordering H prior to factorization

        REAL ( KIND = rp_ ) :: analyse = 0.0

!  CPU time spent factorizing H

        REAL ( KIND = rp_ ) :: factorize = 0.0

!  CPU time spent solving the diagonal model system

        REAL ( KIND = rp_ ) :: solve = 0.0

!  total clock time spent in the package

        REAL ( KIND = rp_ ) :: clock_total = 0.0

!  clock time spent reordering H prior to factorization

        REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  clock time spent factorizing H

        REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  clock time spent solving the diagonal model system

        REAL ( KIND = rp_ ) :: clock_solve = 0.0

      END TYPE DPS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: DPS_inform_type

!  return status. See DPS_solve for details

        INTEGER ( KIND = ip_ ) :: status = 0

!  STAT value after allocate failure

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the number of 1 by 1 blocks from the factorization of H that were modified
!   when constructing M

        INTEGER ( KIND = ip_ ) :: mod_1by1 = 0

!  the number of 2 by 2 blocks from the factorization of H that were modified
!   when constructing M

        INTEGER ( KIND = ip_ ) :: mod_2by2 = 0

!  the value of the quadratic function

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the value of the regularized quadratic function

        REAL ( KIND = rp_ ) :: obj_regularized = HUGE( one )

!  the M-norm of the solution

        REAL ( KIND = rp_ ) :: x_norm = - one

!  the Lagrange multiplier associated with the constraint/regularization

        REAL ( KIND = rp_ ) :: multiplier = - one

!  a lower bound max(0,-lambda_1), where lambda_1 is the left-most
!  eigenvalue of (H,M)

        REAL ( KIND = rp_ ) :: pole = zero

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
        REAL ( KIND = rp_ ) :: p, f, old_f, old_obj, old_multiplier
        REAL ( KIND = rp_ ) :: delta, old_delta, sigma, old_sigma
        INTEGER ( KIND = ip_ ) :: val, ind, nsteps, maxfrt, latop
        LOGICAL :: old_on_boundary, old_convex, trs
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PERM
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: EVAL, MOD_EVAL
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: EVECT, CS, D
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: B
        TYPE ( DPS_control_type ) :: control
        TYPE ( SLS_control_type ) :: SLS_control
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( TRS_control_type ) :: TRS_control
        TYPE ( TRS_inform_type ) :: TRS_inform
        TYPE ( RQS_control_type ) :: RQS_control
        TYPE ( RQS_inform_type ) :: RQS_inform
      END TYPE

!  - - - - - - - - - - - -
!   full data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: DPS_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( DPS_data_type ) :: DPS_data
        TYPE ( DPS_control_type ) :: DPS_control
        TYPE ( DPS_inform_type ) :: DPS_inform
        TYPE ( SMT_type ) :: H
      END TYPE DPS_full_data_type

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

!  revise control parameters (not all compilers currently support fortran 2013)

      control%stop_normal = epsmch ** 0.75
      control%stop_absolute_normal = epsmch ** 0.75

!  set initial values for factorization controls and data

      control%SLS_control%ordering = 0

!  initalize SLS components

      CALL SLS_initialize( control%symmetric_linear_solver,                    &
                           data%SLS_data, control%SLS_control,                 &
                           inform%SLS_inform, check = .TRUE. )
      control%symmetric_linear_solver = inform%SLS_inform%solver
      data%SLS_control%scaling = 0
      control%SLS_control%prefix = '" - SLS:"                     '

!  ensure that the initial value of the "old" delta is small

      data%old_delta = zero ; data%old_sigma = zero
      data%old_on_boundary = .FALSE. ; data%old_convex = .FALSE.

      RETURN

!  End of subroutine DPS_initialize

      END SUBROUTINE DPS_initialize

!- G A L A H A D -  D P S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE DPS_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for DPS controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( DPS_control_type ), INTENT( OUT ) :: control
     TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform

     CALL DPS_initialize( data%dps_data, control, inform )

     RETURN

!  End of subroutine DPS_full_initialize

     END SUBROUTINE DPS_full_initialize

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
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: problem = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = problem + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: new_h = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: taylor_max_degree = new_h
      INTEGER ( KIND = ip_ ), PARAMETER :: eigen_min = taylor_max_degree + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lower = eigen_min + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: upper = lower + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_normal = upper + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stop_absolute_normal                &
                                            = stop_normal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: goldfarb = stop_absolute_normal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = goldfarb + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                            = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: problem_file                        &
                                            = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: symmetric_linear_solver             &
                                            = problem_file + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = symmetric_linear_solver + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
      REAL ( KIND = rp_ ), INTENT( IN ) :: f
      TYPE ( SMT_type ), INTENT( INOUT ) :: H
      REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: C
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: X
      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: delta
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: sigma
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: p

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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
      REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X
      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: f
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: C
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: delta
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: sigma
      REAL ( KIND = rp_ ), OPTIONAL, INTENT( IN ) :: p

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: i
      REAL ( KIND = rp_ ) :: temp
      LOGICAL :: oneby1, printd, printe, debug
      CHARACTER ( LEN = 8 ) :: bad_alloc
      CHARACTER ( LEN = 80 ) :: array_name
!     REAL ( KIND = rp_ ), DIMENSION( n ) :: P

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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n
      TYPE ( SMT_type ), INTENT( INOUT ) :: H
      TYPE ( DPS_data_type ), INTENT( INOUT ) :: data
      TYPE ( DPS_control_type ), INTENT( IN ) :: control
      TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: zeig, rank, out
      REAL :: time_start, time_now, time_record
      REAL ( KIND = rp_ ) :: clock_start, clock_now, clock_record
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
                                    data%SLS_data, data%SLS_control%error,     &
                                    inform%SLS_inform, check = .TRUE. )
        IF ( inform%SLS_inform%status < 0 ) THEN
          inform%status = inform%SLS_inform%status ; GO TO 910 ; END IF

!  perform the analysis

        data%SLS_control = control%SLS_control
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
      CALL SLS_factorize( H, data%SLS_data, data%SLS_control,                  &
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
      CALL SPACE_resize_array( 2_ip_, n, data%B,                               &
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

      INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, rank
      INTEGER ( KIND = ip_ ), INTENT( OUT ) :: mod1, mod2
      LOGICAL, INTENT( IN ) :: goldfarb
      REAL ( KIND = rp_ ), INTENT( IN ) :: eigen_min
      INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n ) :: PERM
      TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: EVAL, MOD_EVAL
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( n ) :: EVECT
      REAL ( KIND = rp_ ), INTENT( OUT ), DIMENSION( 2, n ) :: B

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

      INTEGER ( KIND = ip_ ) :: i
      REAL ( KIND = rp_ ) :: alpha, beta, gamma, tau
      REAL ( KIND = rp_ ) :: t, c , s, e1, e2, eigen
      LOGICAL :: oneby1

      CALL SLS_ENQUIRE( data, inform, PIVOTS = PERM, D = B )
      IF ( inform%status /= GALAHAD_ok ) RETURN
      B( 1, rank + 1 : n ) = zero
!write(6,"( ' P ', /, ( 20I4 ) )" ) PERM( : n )
!write(6,"( ' dia ', 5ES12.4 )" ) B( 1, 1 : n )
!write(6,"( ' off ', 5ES12.4 )" ) B( 2, 1 : n-1 )

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
                EVAL( i + 1 ) = one / e2
                IF ( e1 > zero ) THEN
                  MOD_EVAL( i ) = EVAL( i )
                  MOD_EVAL( i + 1 ) = EVAL( i + 1 )
!                 MOD_EVAL( i + 1 ) = one / EVAL( i + 1 )
                ELSE
                  MOD_EVAL( i ) = - EVAL( i )
                  MOD_EVAL( i + 1 ) = - EVAL( i + 1 )
!                 MOD_EVAL( i + 1 ) = - one / EVAL( i + 1 )
                  B( 1, i ) = - B( 1, i )
                  B( 2, i ) = - B( 2, i )
                  B( 1, i + 1 ) = - B( 1, i + 1 )
                END IF
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
!write(6,"( ' dia ', 5ES12.4 )" ) B( 1, 1 : n )
!write(6,"( ' off ', 5ES12.4 )" ) B( 2, 1 : n-1 )

!      CALL SLS_ENQUIRE( data, inform, D = B )
!      IF ( inform%status /= GALAHAD_ok ) RETURN
!write(6,"( ' dia ', 5ES12.4 )" ) B( 1, 1 : n )
!write(6,"( ' off ', 5ES12.4 )" ) B( 2, 1 : n-1 )

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

! -  G A L A H A D -  D P S _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE DPS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( DPS_control_type ), INTENT( IN ) :: control
     TYPE ( DPS_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL DPS_terminate( data%dps_data, control, inform )

     array_name = 'dps: data%H%ptr'
     CALL SPACE_dealloc_array( data%H%ptr,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dps: data%H%row'
     CALL SPACE_dealloc_array( data%H%row,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dps: data%H%col'
     CALL SPACE_dealloc_array( data%H%col,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'dps: data%H%val'
     CALL SPACE_dealloc_array( data%H%val,                                     &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine DPS_full_terminate

     END SUBROUTINE DPS_full_terminate

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-*-*-*-*-  G A L A H A D -  D P S _ i m p o r t _ S U B R O U T I N E -*-*-*-

     SUBROUTINE DPS_import( control, data, status, n,                          &
                            H_type, H_ne, H_row, H_col, H_ptr )

!  import fixed problem data into internal storage prior to solution.
!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to DPS_solve
!
!  data is a scalar variable of type DPS_full_data_type used for internal data
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
!   -3. The restriction n > 0, or requirement that H_type contains
!       its relevant string 'DENSE', 'COORDINATE', 'SPARSE_BY_ROWS',
!       'DIAGONAL' 'SCALED_IDENTITY', 'IDENTITY', 'ZERO' or 'NONE'
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   variables
!
!  H_type is a character string that specifies the Hessian storage scheme
!   used. It should be one of 'coordinate', 'sparse_by_rows', 'dense'
!   'diagonal' 'scaled_identity', 'identity', 'zero', or 'none'.
!   Lower or upper case variants are allowed.
!
!  H_ne is a scalar variable of type default integer, that holds the number of
!   entries in the lower triangular part of H in the sparse co-ordinate
!   storage scheme. It need not be set for any of the other schemes.
!
!  H_row is a rank-one array of type default integer, that holds
!   the row indices of the lower triangular part of H in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  H_col is a rank-one array of type default integer,
!   that holds the column indices of the lower triangular part of H in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  H_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the lower
!   triangular part of H, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DPS_control_type ), INTENT( INOUT ) :: control
     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, H_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: H_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL, INTENT( IN ) :: H_ptr

!  local variables

     INTEGER ( KIND = ip_ ) :: error
     LOGICAL :: deallocate_error_fatal, space_critical
     CHARACTER ( LEN = 80 ) :: array_name

!  copy control to data

     WRITE( control%out, "( '' )", ADVANCE = 'no') ! prevents ifort bug
     data%dps_control = control

     error = data%dps_control%error
     space_critical = data%dps_control%space_critical
     deallocate_error_fatal = data%dps_control%space_critical

!  set H appropriately in the smt storage type

     SELECT CASE ( H_type )
     CASE ( 'coordinate', 'COORDINATE' )
       IF ( .NOT. ( PRESENT( H_row ) .AND. PRESENT( H_col ) ) ) THEN
         data%dps_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%H%type, 'COORDINATE',                           &
                     data%dps_inform%alloc_status )
       data%H%n = n
       data%H%ne = H_ne

       array_name = 'dps: data%H%row'
       CALL SPACE_resize_array( data%H%ne, data%H%row,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

       array_name = 'dps: data%H%col'
       CALL SPACE_resize_array( data%H%ne, data%H%col,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

       array_name = 'dps: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%H%row( : data%H%ne ) = H_row( : data%H%ne )
         data%H%col( : data%H%ne ) = H_col( : data%H%ne )
       ELSE
         data%H%row( : data%H%ne ) = H_row( : data%H%ne ) + 1
         data%H%col( : data%H%ne ) = H_col( : data%H%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
       IF ( .NOT. ( PRESENT( H_ptr ) .AND. PRESENT( H_col ) ) ) THEN
         data%dps_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%H%type, 'SPARSE_BY_ROWS',                            &
                     data%dps_inform%alloc_status )
       data%H%n = n
       IF ( data%f_indexing ) THEN
         data%H%ne = H_ptr( n + 1 ) - 1
       ELSE
         data%H%ne = H_ptr( n + 1 )
       END IF

       array_name = 'dps: data%H%ptr'
       CALL SPACE_resize_array( n + 1, data%H%ptr,                             &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

       array_name = 'dps: data%H%col'
       CALL SPACE_resize_array( data%H%ne, data%H%col,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

       array_name = 'dps: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%H%ptr( : n + 1 ) = H_ptr( : n + 1 )
         data%H%col( : data%H%ne ) = H_col( : data%H%ne )
       ELSE
         data%H%ptr( : n + 1 ) = H_ptr( : n + 1 ) + 1
         data%H%col( : data%H%ne ) = H_col( : data%H%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%H%type, 'DENSE',                                     &
                     data%dps_inform%alloc_status )
       data%H%n = n
       data%H%ne = ( n * ( n + 1 ) ) / 2

       array_name = 'dps: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%H%type, 'DIAGONAL',                                  &
                     data%dps_inform%alloc_status )
       data%H%n = n
       data%H%ne = n

       array_name = 'dps: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

     CASE ( 'scaled_identity', 'SCALED_IDENTITY' )
       CALL SMT_put( data%H%type, 'SCALED_IDENTITY',                           &
                     data%dps_inform%alloc_status )
       data%H%n = n
       data%H%ne = 1

       array_name = 'dps: data%H%val'
       CALL SPACE_resize_array( data%H%ne, data%H%val,                         &
              data%dps_inform%status, data%dps_inform%alloc_status,            &
              array_name = array_name,                                         &
              deallocate_error_fatal = deallocate_error_fatal,                 &
              exact_size = space_critical,                                     &
              bad_alloc = data%dps_inform%bad_alloc, out = error )
       IF ( data%dps_inform%status /= 0 ) GO TO 900

     CASE ( 'identity', 'IDENTITY' )
       CALL SMT_put( data%H%type, 'IDENTITY',                                  &
                     data%dps_inform%alloc_status )
       data%H%n = n
       data%H%ne = 0

     CASE ( 'zero', 'ZERO', 'none', 'NONE' )
       CALL SMT_put( data%H%type, 'ZERO',                                      &
                     data%dps_inform%alloc_status )
       data%H%n = n
       data%H%ne = 0
     CASE DEFAULT
       data%dps_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

     status = GALAHAD_ok
     RETURN

!  error returns

 900 CONTINUE
     status = data%dps_inform%status
     RETURN

!  End of subroutine DPS_import

     END SUBROUTINE DPS_import

!-  G A L A H A D -  D P S _ r e s e t _ c o n t r o l   S U B R O U T I N E

     SUBROUTINE DPS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See DPS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DPS_control_type ), INTENT( IN ) :: control
     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%dps_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine DPS_reset_control

     END SUBROUTINE DPS_reset_control

! G A L A H A D -  D P S _ s o l v e _ t r _ p r o b l e m   S U B R O U T I N E

     SUBROUTINE DPS_solve_tr_problem( data, status, H_val, C, f, radius, X )

!  solve the trust-region problem whose structure was previously
!  imported. See DPS_solve for a description of the required arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type DPS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, dps_solve above.
!
!  H_val is a rank-one array of type default real, that holds the values
!   of the lower triangular part of the Hessian H in the storage scheme
!   specified in dps_import.
!
!  C is a rank-one array of dimension n and type default
!   real, that holds the vector of linear terms of the objective, c.
!   The j-th component of C, j = 1, ... , n, contains (c)_j.
!
!  f is a scalar of type default real, that holds the constant term, f,
!   of the objective.
!
!  radius is a scalar of type default  real, that holds the trust-region
!   radius, delta > 0
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal solution, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: H_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: C
     REAL ( KIND = rp_ ), INTENT( IN ) :: f, radius
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: X

!  local variables

     INTEGER ( KIND = ip_ ) :: n

!  recover the dimensions

     n = data%H%n

!  save the Hessian entries

     IF ( data%H%ne > 0 ) data%H%val( : data%H%ne ) = H_val( : data%H%ne )

!  call the solver

     CALL DPS_solve( n, data%H, C, f, X, data%dps_data, data%dps_control,      &
                     data%dps_inform, delta = radius )

     status = data%dps_inform%status
     RETURN

!  End of subroutine DPS_solve_tr_problem

     END SUBROUTINE DPS_solve_tr_problem

! G A L A H A D -  D P S _ s o l v e _ r q _ p r o b l e m   S U B R O U T I N E

     SUBROUTINE DPS_solve_rq_problem( data, status, H_val, C, f,               &
                                      weight, power, X )

!  solve the regularized-quadratic problem whose structure was previously
!  imported. See DPS_solve for a description of the required arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type DPS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, dps_solve above.
!
!  H_val is a rank-one array of type default real, that holds the values
!   of the lower triangular part of the Hessian H in the storage scheme
!   specified in dps_import.
!
!  C is a rank-one array of dimension n and type default
!   real, that holds the vector of linear terms of the objective, c.
!   The j-th component of C, j = 1, ... , n, contains (c)_j.
!
!  f is a scalar of type default real, that holds the constant term, f,
!   of the objective.
!
!  weight is a scalar of type default  real, that holds the regularization
!   weight, sigma > 0
!
!  power is a scalar of type default  real, that holds the regularization
!   power, p >= 2
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal solution, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: H_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: C
     REAL ( KIND = rp_ ), INTENT( IN ) :: f, weight, power
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: X

!  local variables

     INTEGER ( KIND = ip_ ) :: n

!  recover the dimensions

     n = data%H%n

!  save the Hessian entries

     IF ( data%H%ne > 0 ) data%H%val( : data%H%ne ) = H_val( : data%H%ne )

!  call the solver

     CALL DPS_solve( n, data%H, C, f, X, data%dps_data, data%dps_control,      &
                     data%dps_inform, sigma = weight, p = power )

     status = data%dps_inform%status
     RETURN

!  End of subroutine DPS_solve_rq_problem

     END SUBROUTINE DPS_solve_rq_problem

! G A L A H A D - D P S _ r e s o l v e _ t r _ p r o b l e m  S U B R O U T INE

     SUBROUTINE DPS_resolve_tr_problem( data, status, C, f, radius, X )

!  resolve the trust-region problem whose structure was previously imported
!  and Hessian unaltered. See DPS_solve for a description of the
!  required arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type DPS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, dps_solve above.
!
!  C is an optional rank-one array of dimension n and type default
!   real, that holds the vector of linear terms of the objective, c.
!   The j-th component of C, j = 1, ... , n, contains (c)_j.
!
!  f is an optional scalar of type default real, that holds the constant term,
!   f, of the objective.
!
!  radius is an optional scalar of type default  real, that holds the
!   trust-region radius, delta > 0.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal solution, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.


     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: C
     REAL ( KIND = rp_ ), INTENT( IN ) :: f, radius
     REAL ( KIND = rp_ ), INTENT( INOUT ) , DIMENSION( : ) :: X

!  local variables

     INTEGER ( KIND = ip_ ) :: n

!  recover the dimensions

     n = data%H%n

!  call the solver

     CALL DPS_resolve( n, X, data%dps_data, data%dps_control, data%dps_inform, &
                       C = C, f = f, delta = radius )

     status = data%dps_inform%status
     RETURN

!  End of subroutine DPS_resolve_tr_problem

     END SUBROUTINE DPS_resolve_tr_problem

! G A L A H A D - D P S _ r e s o l v e _ r q _ p r o b l e m  S U B R O U T INE

     SUBROUTINE DPS_resolve_rq_problem( data, status, C, f, weight, power, X )

!  resolve the regularized-quadratic problem whose structure was previously
!  imported and Hessian unaltered. See DPS_solve for a description of the
!  required arguments.

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

!  data is a scalar variable of type DPS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import. If status = 0, the solve was succesful.
!   For other values see, dps_solve above.
!
!  C is an optional rank-one array of dimension n and type default
!   real, that holds the vector of linear terms of the objective, c.
!   The j-th component of C, j = 1, ... , n, contains (c)_j.
!
!  f is an optional scalar of type default real, that holds the constant term,
!   f, of the objective.
!
!  weight is an optional scalar of type default  real, that holds the
!   regularization weight, sigma > 0.
!
!  power is an optional scalar of type default  real, that holds the
!   regularization power, p >= 2.
!
!  X is a rank-one array of dimension n and type default
!   real, that holds the vector of the primal solution, x.
!   The j-th component of X, j = 1, ... , n, contains (x)_j.
!
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( : ) :: C
     REAL ( KIND = rp_ ), INTENT( IN ) :: f, weight, power
     REAL ( KIND = rp_ ), INTENT( INOUT ) , DIMENSION( : ) :: X

!  local variables

     INTEGER ( KIND = ip_ ) :: n

!  recover the dimensions

     n = data%H%n

!  call the solver

     CALL DPS_resolve( n, X, data%dps_data, data%dps_control, data%dps_inform, &
                       C = C, f = f, sigma = weight, p = power )

     status = data%dps_inform%status
     RETURN

!  End of subroutine DPS_resolve_rq_problem

     END SUBROUTINE DPS_resolve_rq_problem

! -*-  G A L A H A D -  D P S _ i n f o r m a t i o n   S U B R O U T I N E -*-

     SUBROUTINE DPS_information( data, inform, status )

!  return solver information during or after solution by DPS
!  See DPS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( DPS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( DPS_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%dps_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine DPS_information

     END SUBROUTINE DPS_information

!  End of module GALAHAD_DPS

    END MODULE GALAHAD_DPS_precision
