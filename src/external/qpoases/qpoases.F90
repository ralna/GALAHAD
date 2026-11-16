! THIS VERSION: GALAHAD 5.6 - 2026-11-14 AT 11:50 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P O A S E S   M O D U L E  -*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 5.6. October 6th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_QPOASES_precision

!     ------------------------------------------------
!     |                                              |
!     | Minimize the quadratic objective function    |
!     |                                              |
!     |         1/2 x^T H x + g^T x + f              |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |          c_l <= A x <= c_u                   |
!     |          x_l <=  x <= x_u                    |
!     |                                              |
!     | for some positive definite Hessian using     |
!     | the QPOASES package (which must be provided  |
!     | externally). See README.qpoases for details  |
!     | of how this is possible                      |
!     |                                              |
!     ------------------------------------------------

      USE GALAHAD_KINDS_precision, ONLY: i4_, ip_, rp_
      USE GALAHAD_CLOCK, ONLY: CLOCK_time
      USE GALAHAD_SYMBOLS, ONLY: GALAHAD_ok,                                   &
                                 GALAHAD_error_restrictions,                   &
                                 GALAHAD_error_primal_infeasible,              &
                                 GALAHAD_error_dual_infeasible,                &
                                 GALAHAD_error_max_iterations,                 &
                                 GALAHAD_error_inertia,                        &
                                 GALAHAD_unavailable_solver,                   &
                                 GALAHAD_error_unbounded,                      &
                                 GALAHAD_error_factorization,                  &
                                 GALAHAD_error_io,                             &
                                 GALAHAD_error_qpoases
      USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
      USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read, &
                                            SPECFILE_assign_value
      USE GALAHAD_SMT_precision, ONLY: SMT_type, SMT_get, SMT_put
      USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A,        &
                                       QPT_keyword_H, QPT_summarize_problem
      USE GALAHAD_QPD_precision, ONLY: QPD_data_type, QPD_SIF
      USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,               &
                                          CONVERT_inform_type,                 &
                                          CONVERT_to_sparse_column_format
      USE GALAHAD_QPOASES_F_precision, ONLY: qpOASES_options_type,             &
                                     qpOASES_solve_internal => qpOASES_solve

      IMPLICIT NONE ( TYPE, EXTERNAL )

      PRIVATE
      PUBLIC :: QPOASES_initialize, QPOASES_read_specfile, QPOASES_solve,      &
                QPOASES_terminate, QPT_problem_type, SMT_type, SMT_get, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE QPOASES_initialize
       MODULE PROCEDURE QPOASES_initialize, QPOASES_full_initialize
     END INTERFACE QPOASES_initialize

     INTERFACE QPOASES_terminate
       MODULE PROCEDURE QPOASES_terminate, QPOASES_full_terminate
     END INTERFACE QPOASES_terminate

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = rp_ ), PARAMETER :: zero = 0.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: one = 1.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: ten = 10.0_rp_
      REAL ( KIND = rp_ ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!   extended control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPOASES_control_type

!  QPOASES_options derived type

         TYPE ( QPOASES_options_type ) :: options

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = rp_ ) :: infinity = ten ** 19

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

     END TYPE QPOASES_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   extended inform derived type with defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPOASES_inform_type

!  return status

        INTEGER ( KIND = ip_ ) :: status = 0

!  return run status from qpoases

        INTEGER ( KIND = ip_ ) :: qpoases_status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by QPOASES_solve

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

!  the total CPU time spent in the package

        REAL ( KIND = rp_ ) :: cpu_time = 0.0

      END TYPE QPOASES_inform_type

!  - - - - - - - - - - - - - -
!   data derived type for QPOASES
!  - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPOASES_data_type
        INTEGER ( KIND = ip_ ) :: n, m, h_ne, a_ne, a_ne_qp
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_ptr
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: H_row, H_ptr
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: A_val, H_val
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B_l, B_u, Y
        LOGICAL :: original_a, original_h
        LOGICAL :: new_structure = .TRUE.
      END TYPE QPOASES_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: QPOASES_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( QPOASES_data_type ) :: QPOASES_data
        TYPE ( QPOASES_control_type ) :: QPOASES_control
        TYPE ( QPOASES_inform_type ) :: QPOASES_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE QPOASES_full_data_type

   CONTAINS

!-*-*-*-*-   Q P O A S E S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-

      SUBROUTINE QPOASES_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for QPOASES. This routine should be called before
!  QPOASES_solve
!
!  ---------------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( QPOASES_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPOASES_control_type ), INTENT( OUT ) :: control
      TYPE ( QPOASES_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of QPOASES_initialize

      END SUBROUTINE QPOASES_initialize

!- G A L A H A D -  Q P O A S E S _ F U L L _ I N I T I A L I Z E  S U B ROUTINE

     SUBROUTINE QPOASES_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for QPOASES controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( QPOASES_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( QPOASES_control_type ), INTENT( OUT ) :: control
     TYPE ( QPOASES_inform_type ), INTENT( OUT ) :: inform

     CALL QPOASES_initialize( data%QPOASES_data, control, inform )

     RETURN

!  End of subroutine QPOASES_full_initialize

     END SUBROUTINE QPOASES_full_initialize

!-*-*-   Q P O A S E S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-

      SUBROUTINE QPOASES_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by QPOASES_initialize could (roughly)
!  have been set as:

! BEGIN QPOASES SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  print-level                                       0
!  enable-ramping                                    1
!  enable-far-bounds                                 1
!  enable-flipping-bounds                            1
!  enable-regularisation                             0
!  enable-full-li-tests                              0
!  enable-nzc-tests                                  1
!  enable-drift-correction                           1
!  enable-cholesky-refactorisation                   0
!  enable-equalities                                 0
!  initial-status-bounds                             -1
!  num-regularisation-steps                          0
!  num-refinement-steps                              1
!  enable-inertia-correction                         1
!  enable-dropin-feasibles                           0
!  drop-bound-priority                               1
!  drop-eqcon-priority                               1
!  drop-ineq-con-priority                            1
!  print-residuals                                   0
!  infinity-value                                    1.0D+19
!  termination-tolerance                             1.0E-14
!  bound-tolerance                                   1.0E-10
!  bound-relaxation                                  10000.0
!  eps-num                                           1.0E-14
!  eps-den                                           1.0E-13
!  max-primal-jump                                   1.0E+8
!  max-dual-jump                                     1.0E+8
!  initial-ramping                                   0.5
!  final-ramping                                     1.0
!  initial-farbounds                                 1.0E+6
!  grow-far-bounds                                   1000.0
!  eps-flipping                                      1.0E-13
!  eps-regularisation                                1.0E-12
!  eps-iter-ref                                      1.0E-14
!  eps-li-tests                                      1.0E-11
!  eps-nzc-tests                                     6.0E-12
!  rcond-smin                                         1.0E-14
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END QPOASES SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( QPOASES_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: printLevel = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableRamping = printLevel + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableFarBounds = enableRamping + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableFlippingBounds                &
                                             = enableFarBounds + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableRegularisation                &
                                             = enableFlippingBounds + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableFullLITests                   &
                                             = enableRegularisation + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableNZCTests                      &
                                             = enableFullLITests + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableDriftCorrection               &
                                             = enableNZCTests + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableCholeskyRefactorisation       &
                                             = enableDriftCorrection + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableEqualities                    &
                                             = enableCholeskyRefactorisation + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: initialStatusBounds                 &
                                             = enableEqualities + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: numRegularisationSteps              &
                                             = initialStatusBounds + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: numRefinementSteps                  &
                                             = numRegularisationSteps + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableInertiaCorrection             &
                                             = numRefinementSteps + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: enableDropInfeasibles               &
                                             = enableInertiaCorrection + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dropBoundPriority                   &
                                             = enableDropInfeasibles + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dropEqConPriority                   &
                                             = dropBoundPriority + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dropIneqConPriority                 &
                                             = dropEqConPriority + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: printResiduals                      &
                                             = dropIneqConPriority + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinity = printResiduals + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: terminationTolerance  = infinity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: boundTolerance                      &
                                             = terminationTolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: boundRelaxation = boundTolerance + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsNum = boundRelaxation + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsDen = epsNum + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxPrimalJump = epsDen + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxDualJump = maxPrimalJump + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: initialRamping = maxDualJump + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: finalRamping = initialRamping + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: initialFarBounds = finalRamping + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: growFarBounds = initialFarBounds + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsFlipping = growFarBounds + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsRegularisation = epsFlipping + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsIterRef = epsRegularisation + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsLITests = epsIterRef + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: epsNZCTests = epsLITests + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: rcondSMin = epsNZCTests + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = rcondSMin + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 7 ), PARAMETER :: specname = 'QPOASES'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( printLevel )%keyword = 'print-level'
      spec( enableRamping )%keyword = 'enable-ramping'
      spec( enableFarBounds )%keyword = 'enable-far-bounds'
      spec( enableFlippingBounds )%keyword = 'enable-flipping-bounds'
      spec( enableRegularisation )%keyword = 'enable-regularisation'
      spec( enableFullLITests )%keyword = 'enable-full-li-tests'
      spec( enableNZCTests )%keyword = 'enable-nzc-tests'
      spec( enableDriftCorrection )%keyword = 'enable-drift-correction'
      spec( enableCholeskyRefactorisation )%keyword                            &
          = 'enable-cholesky-refactorisation'
      spec( enableEqualities )%keyword = 'enable-equalities'
      spec( initialStatusBounds )%keyword = 'initial-status-bounds'
      spec( numRegularisationSteps )%keyword = 'num-regularisation-steps'
      spec( numRefinementSteps )%keyword = 'num-refinement-steps'
      spec( enableInertiaCorrection )%keyword = 'enable-inertia-correction'
      spec( enableDropInfeasibles )%keyword = 'enable-dropin-feasibles'
      spec( dropBoundPriority )%keyword = 'drop-bound-priority'
      spec( dropEqConPriority )%keyword = 'drop-eqcon-priority'
      spec( dropIneqConPriority )%keyword = 'drop-ineq-con-priority'
      spec( printResiduals )%keyword = 'print-residuals'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( terminationTolerance )%keyword = 'termination-tolerance'
      spec( boundTolerance )%keyword = 'bound-tolerance'
      spec( boundRelaxation )%keyword = 'bound-relaxation'
      spec( epsNum )%keyword = 'eps-num'
      spec( epsDen )%keyword = 'eps-den'
      spec( maxPrimalJump )%keyword = 'max-primal-jump'
      spec( maxDualJump )%keyword = 'max-dual-jump'
      spec( initialRamping )%keyword = 'initial-ramping'
      spec( finalRamping )%keyword = 'final-ramping'
      spec( initialFarBounds )%keyword = 'initial-farbounds'
      spec( growFarBounds )%keyword = 'grow-far-bounds'
      spec( epsFlipping )%keyword = 'eps-flipping'
      spec( epsRegularisation )%keyword = 'eps-regularisation'
      spec( epsIterRef )%keyword = 'eps-iter-ref'
      spec( epsLITests )%keyword = 'eps-li-tests'
      spec( epsNZCTests )%keyword = 'eps-nzc-tests'
      spec( rcondSMin )%keyword = 'rcond-smin'

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' qpoases: ', alt_specname

!  Read the specfile

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%error )
      END IF

!  Interpret the result

!  Set integer values

     CALL SPECFILE_assign_value( spec( error ),                                &
                                 control%error,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( printLevel ),                           &
                                 control%options%printLevel,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableRamping ),                        &
                                 control%options%enableRamping,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableFarBounds ),                      &
                                 control%options%enableFarBounds,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableFlippingBounds ),                 &
                                 control%options%enableFlippingBounds,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableRegularisation ),                 &
                                 control%options%enableRegularisation,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableFullLITests ),                    &
                                 control%options%enableFullLITests,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableNZCTests ),                       &
                                 control%options%enableNZCTests,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableDriftCorrection ),                &
                                 control%options%enableDriftCorrection,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableCholeskyRefactorisation ),        &
                                 control%options%enableCholeskyRefactorisation,&
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableEqualities ),                     &
                                 control%options%enableEqualities,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initialStatusBounds ),                  &
                                 control%options%initialStatusBounds,          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( numRegularisationSteps ),               &
                                 control%options%numRegularisationSteps,       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( numRefinementSteps ),                   &
                                 control%options%numRefinementSteps,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableInertiaCorrection ),              &
                                 control%options%enableInertiaCorrection,      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( enableDropInfeasibles ),                &
                                 control%options%enableDropInfeasibles,        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dropBoundPriority ),                    &
                                 control%options%dropBoundPriority,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dropEqConPriority ),                    &
                                 control%options%dropEqConPriority,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dropIneqConPriority ),                  &
                                 control%options%dropIneqConPriority,          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( printResiduals ),                       &
                                 control%options%printResiduals,               &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( terminationTolerance ),                 &
                                 control%options%terminationTolerance,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( boundTolerance ),                       &
                                 control%options%boundTolerance,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( boundRelaxation ),                      &
                                 control%options%boundRelaxation,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsNum ),                               &
                                 control%options%epsNum,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsDen ),                               &
                                 control%options%epsDen,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxPrimalJump ),                        &
                                 control%options%maxPrimalJump,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxDualJump ),                          &
                                 control%options%maxDualJump,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initialRamping ),                       &
                                 control%options%initialRamping,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( finalRamping ),                         &
                                 control%options%finalRamping,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initialFarBounds ),                     &
                                 control%options%initialFarBounds,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( growFarBounds ),                        &
                                 control%options%growFarBounds,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsFlipping ),                          &
                                 control%options%epsFlipping,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsRegularisation ),                    &
                                 control%options%epsRegularisation,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsIterRef ),                           &
                                 control%options%epsIterRef,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsLITests ),                           &
                                 control%options%epsLITests,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( epsNZCTests ),                          &
                                 control%options%epsNZCTests,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( rcondSMin ),                            &
                                 control%options%rcondSMin,                    &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )
      RETURN

      END SUBROUTINE QPOASES_read_specfile

!-*-*-*-*-*-*-*-   Q P O A S E S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE QPOASES_solve( prob, data, control, inform )

!  solve the quadratic program using the QPOASES package

!  A - by columns
!  H - both triangles by columns

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( QPOASES_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPOASES_control_type ), INTENT( IN ) :: control
      TYPE ( QPOASES_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: m, n, a_m, a_n
!     INTEGER ( KIND = ip_ ) :: a_ne, h_ne
      TYPE ( CONVERT_control_type ) :: control_convert
      TYPE ( CONVERT_inform_type ) :: inform_convert
      CHARACTER ( LEN = 80 ) :: array_name

!  transfer the data into QPOASES's QP format. First, record the number
!  of rows and columns

      n = prob%n ; m = prob%m

!!$!  find the number of nonzeros in A
!!$
!!$      SELECT CASE ( SMT_get( prob%A%type ) )
!!$      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
!!$        data%a_ne = m * n
!!$      CASE ( 'SPARSE_BY_ROWS' )
!!$        data%a_ne = prob%A%ptr( m + 1 ) - 1
!!$      CASE ( 'SPARSE_BY_COLUMNS' )
!!$        data%a_ne = prob%A%ptr( n + 1 ) - 1
!!$      CASE ( 'COORDINATE' )
!!$        data%a_ne = prob%A%ne
!!$      CASE DEFAULT
!!$        data%a_ne = 0
!!$      END SELECT
!!$
!!$!  find the number of nonzeros in one triangle of H, and the total in both
!!$
!!$      SELECT CASE ( SMT_get( prob%H%type ) )
!!$      CASE ( 'IDENTITY', 'SCALED_IDENTITY', 'DIAGONAL' )
!!$        data%h_ne = n ; data%h_ne_qp = data%h_ne
!!$      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
!!$        data%h_ne = ( n * ( n + 1 ) ) / 2 ; data%h_ne_qp = n * n
!!$      CASE ( 'SPARSE_BY_ROWS' )
!!$        data%h_ne = prob%H%ptr( n + 1 ) - 1 ; data%h_ne_qp = data%h_ne
!!$        DO i = 1, n
!!$          DO j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
!!$            IF ( prob%H%col( j ) /= i ) data%h_ne_qp = data%h_ne_qp + 1
!!$          END DO
!!$        END DO
!!$      CASE ( 'SPARSE_BY_COLUMNS' )
!!$        data%h_ne = prob%H%ptr( n + 1 ) - 1 ;  ; data%h_ne_qp = data%h_ne
!!$        DO i = 1, n
!!$          DO j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
!!$            IF ( prob%H%row( j ) /= i ) data%h_ne_qp = data%h_ne_qp + 1
!!$          END DO
!!$        END DO
!!$      CASE ( 'COORDINATE' )
!!$        data%h_ne = prob%H%ne ; data%h_ne_qp = data%h_ne
!!$        DO l = 1, data%h_ne
!!$          IF ( prob%H%row( l ) /=  prob%H%col( l ) )                        &
!!$            data%h_ne_qp = data%h_ne_qp + 1
!!$          END DO
!!$        END DO
!!$      CASE DEFAULT
!!$        data%h_ne = 0 ; data%h_ne_qp = data%h_ne
!!$      END SELECT

!  if necessary, convert the input A into by sparse-column format

      data%original_a = SMT_get( prob%A%type ) == 'SPARSE_BY_COLUMNS'
      IF ( .NOT. data%original_a ) THEN
        prob%A%m = m ; prob%A%n = n
        CALL CONVERT_to_sparse_column_format( prob%A, a_m, a_n, data%a_ne,     &
                                              data%A_val, data%A_row,          &
                                              data%A_ptr, control_convert,     &
                                              inform_convert )
!write(6,*) ' a_ptr ', data%A_ptr
!write(6,*) ' a_row ', data%A_row
!write(6,*) ' a_val ', data%A_val

      END IF

!  convert the input H into by sparse-row format (both triangles)

      prob%H%m = n ; prob%H%n = n
      CALL CONVERT_from_symmetric_to_CSC_main( prob%H, n, data%h_ne,           &
                                               data%H_val, data%H_row,         &
                                               data%H_ptr, control_convert,    &
                                               inform_convert )

!  provide workspace for the dual variables

      IF ( data%new_structure ) THEN
        array_name = 'qpoases: data%Y'
        CALL SPACE_resize_array( n + m, data%Y, inform%status,                 &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF 

!  call qpoases to solve the problem

      IF ( data%original_a ) THEN
        CALL qpoases_solve_internal( n, m, data%H_row, data%H_ptr,             &
                                     data%H_val, prob%G, prob%A%row,           &
                                     prob%A%ptr, prob%A%val,                   &
                                     prob%X_l, prob%X_u, prob%C_l, prob%C_u,   &
                                     control%options, inform%iter,             &
                                     inform%cpu_time, prob%X, data%Y,          &
                                     inform%obj, inform%qpoases_status )
      ELSE
        CALL qpoases_solve_internal( n, m, data%H_row, data%H_ptr,             &
                                     data%H_val, prob%G, data%A_row,           &
                                     data%A_ptr, data%A_val,                   &
                                     prob%X_l, prob%X_u, prob%C_l, prob%C_u,   &
                                     control%options, inform%iter,             &
                                     inform%cpu_time, prob%X, data%Y,          &
                                     inform%obj, inform%qpoases_status )
      END IF

!  record the exit status

      SELECT CASE ( inform%qpoases_status )

!  success states

      CASE ( 0, 40, 49, 59, 143 )
        inform%status = GALAHAD_ok

!  argument, dimension, or bound restriction errors

      CASE ( 2, 3, 4, 6, 12 : 21, 29 )
        inform%status = GALAHAD_error_restrictions

!  primal infeasibility (Simple status -2 or raw infeasible codes)

      CASE ( - 2, 47, 61, 73, 145 )
        inform%status = GALAHAD_error_primal_infeasible

!  unboundedness (Simple status -3 or raw unbounded codes)

      CASE ( - 3, 46, 62, 74, 146 )
        inform%status = GALAHAD_error_unbounded

!  iteration limit reached (Simple status 1 or RET_MAX_NWSR_REACHED)

      CASE ( 1, 64, 142 )
        inform%status = GALAHAD_error_max_iterations

!  factorization & regularization errors

      CASE ( 30, 31, 39, 101, 102, 118 )
        inform%status = GALAHAD_error_factorization

!  file I/O errors

      CASE ( 127 : 130 )
        inform%status = GALAHAD_error_io

!  all other internal, homotopy, or working-set errors (including 
!  simple status -1)

      CASE DEFAULT
        inform%status = GALAHAD_error_qpoases
      END SELECT

!  recover the solution

      IF ( inform%status >= 0 ) THEN
        IF ( data%new_structure ) THEN
          array_name = 'qpoases: prob%Y'
          CALL SPACE_resize_array( m, prob%Y, inform%status,                   &
                 inform%alloc_status, array_name = array_name,                 &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'qpoases: prob%Z'
          CALL SPACE_resize_array( n, prob%Z, inform%status,                   &
                 inform%alloc_status, array_name = array_name,                 &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc )
          IF ( inform%status /= GALAHAD_ok ) RETURN
        END IF 

!  record the Lagrange multiplers, dual variables and optimal objective value

        prob%Y( : m ) = data%Y( n + 1 : n + m )
        prob%Z( : n ) = data%Y( : n )
        prob%q = inform%obj

!  print details, if required

        IF ( control%print_level > 0 ) THEN
          WRITE( control%out, "( /, ' QPOASES - Fortran interface' )" )
          WRITE( control%out, "( ' objective function:', ES16.8 )" ) prob%q
!         WRITE( control%out, "( ' primal & dual residuals:', 2ES16.8 )" )     &
!           inform%info%pri_res_norm, inform%info%dua_res_norm
          WRITE( control%out, "( ' x:', ( 5ES16.8 ) )" ) prob%X
          WRITE( control%out, "( ' y:', ( 5ES16.8 ) )" ) prob%Y
          WRITE( control%out, "( ' z:', ( 5ES16.8 ) )" ) prob%Z
          WRITE( control%out, "( 1X, I0, ' iterations' ) ") inform%iter
          WRITE( control%out, "( ' status = ', I0 )" )  inform%qpoases_status
        END IF
      ELSE
        IF ( control%print_level > 0 )                                         &
          WRITE( control%out, "( ' Error. Problem not solved to optimality' )" )
      END IF

!  ensure that the existing structure remains for subsequent calls until
!  QPOASES_terminate removes it

      data%new_structure = .FALSE.

      RETURN

!  End of QPOASES_solve

      END SUBROUTINE QPOASES_solve

! C O N V E R T _ f r o m _ s y m m e t r i c _ T O _ C S C _ M A I N SUBROUTINE

      SUBROUTINE CONVERT_from_symmetric_to_CSC_main( A, A_out_n, A_out_ne,     &
                                                     A_out_val, A_out_row,     &
                                                     A_out_ptr, control,       &
                                                     inform )
!                                                    inform, MAP )

!  convert the input symmetric matrix (one triangle) to sparse-column
!  (both triangles) format
!
!  ** NB this ultimately belongs in GALAHAD's convert package **
!  ** The optional MAP argument is missimg **
!
!  Arguments:
!
!   A is a structure of type SMT_type used to hold the input matrix A.
!   (see CONVERT_to_CSC_main above)
!
!   A_out_n   the number of rows/columns of the output A
!   A_out_ne  the number of nonzros in both triangles of the output A
!   A_out_val( : )   the values of the components of A, stored
!                        column by columns
!   A_out_row( : )   the row indices of the components of A
!   A_out_ptr( : )   pointers to the start of each column, and past
!                        the end of the last column
!
!   control, inform and the optional arguments as above

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: A_out_n, A_out_ne
      INTEGER ( KIND = ip_ ), INTENT( INOUT ), ALLOCATABLE,                    &
                                DIMENSION( : ) :: A_out_row, A_out_ptr
      REAL ( KIND = rp_ ), INTENT( INOUT ), ALLOCATABLE,                       &
                                DIMENSION( : ) :: A_out_val
      TYPE ( CONVERT_control_type ), INTENT( IN ) :: control
      TYPE ( CONVERT_inform_type ), INTENT( INOUT ) :: inform
!     INTEGER ( KIND = ip_ ), OPTIONAL, INTENT( INOUT ),                       &
!                                       DIMENSION( SIZE( A%val ) ) :: MAP

!  Local variables

      INTEGER ( KIND = ip_ ) :: i, j, k, l, n, ne, ne_out
      REAL ( KIND = rp_ ) :: val, time_start, time_now, clock_start, clock_now
      LOGICAL :: sum_duplicates
!     LOGICAL :: map_vals
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  ensure that input parameters are within allowed ranges

      IF ( A%n < 1 ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%n must be +ve' )" )       &
            prefix
        RETURN
      END IF

!  ensure that all values are recorded if MAP is present

!     map_vals = PRESENT( MAP )

!  find ne, the number of nonzeros in one triangle of A, and ne_out,
!  the number in both triangles

      sum_duplicates = control%sum_duplicates
      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'IDENTITY', 'SCALED_IDENTITY', 'DIAGONAL' )
        ne = n ; ne_out = ne
        sum_duplicates = .FALSE.
      CASE ( 'DENSE', 'DENSE_BY_ROWS', 'DENSE_BY_COLUMNS' )
        ne = ( n * ( n + 1 ) ) / 2 ; ne_out = n * n
      CASE ( 'SPARSE_BY_ROWS' )
        ne = A%ptr( n + 1 ) - 1 ; ne_out = ne
        DO i = 1, n
          DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
            IF ( A%col( j ) /= i ) ne_out = ne_out + 1
          END DO
        END DO
      CASE ( 'SPARSE_BY_COLUMNS' )
        ne = A%ptr( n + 1 ) - 1 ;  ; ne_out = ne
        DO i = 1, n
          DO j = A%ptr( i ), A%ptr( i + 1 ) - 1
            IF ( A%row( j ) /= i ) ne_out = ne_out + 1
          END DO
        END DO
      CASE ( 'COORDINATE' )
        ne = A%ne ; ne_out = ne
        DO l = 1, ne
          IF ( A%row( l ) /=  A%col( l ) ) ne_out = ne_out + 1
        END DO
      CASE ( 'ZERO', 'NONE' )
        ne = 0 ; ne_out = ne
        sum_duplicates = .FALSE.

!  type of A unknown

      CASE DEFAULT
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, "( ' ', /, A, ' ** A%type unknown' )" ) prefix
        GO TO 900
      END SELECT

!  set initial values

      n = A%n

!  store A column-wise in A_out, perhaps with the row entries within
!  each column in increasing order (the latter is not yet implemented)

      A_out_n = n ; A_out_ne = ne_out

      array_name = 'CONVERT: A_out_ptr'
      CALL SPACE_resize_array( A_out_n + 1, A_out_ptr,                         &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out_row'
      CALL SPACE_resize_array( ne_out, A_out_row,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'CONVERT: A_out_val'
      CALL SPACE_resize_array( ne_out, A_out_val,                              &
        inform%status, inform%alloc_status, array_name = array_name,           &
        deallocate_error_fatal = control%deallocate_error_fatal,               &
        exact_size = control%space_critical,                                   &
        bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  copy the data to A. Consider the input storage scheme

      SELECT CASE( SMT_get( A%type ) )

!  A is a dense matrix (stored by rows)

      CASE ( 'DENSE', 'DENSE_BY_ROWS' )

!  count the number of nonzeros in each column

        k = 0
        A_out_ptr( 1 : n ) = 0
        DO i = 1, n
          DO j = 1, i
            k = k + 1
            IF ( A%val( k ) /= zero ) THEN
              A_out_ptr( j ) = A_out_ptr( j ) + 1
              IF ( i /= j ) THEN
                A_out_ptr( i ) = A_out_ptr( i ) + 1
              END IF
            END IF
          END DO
        END DO

!  find the starting address for each column in the output scheme

        l = 1
        DO j = 1, n
          k = l + A_out_ptr( j )
          A_out_ptr( j ) = l
          l = k
        END DO

!  now allocate the row indices and values in the output scheme

        k = 0
        DO i = 1, n
          DO j = 1, i
            k = k + 1
            val = A%val( k )
            IF ( val /= zero ) THEN
              l = A_out_ptr( j )
              A_out_row( l ) = i ; A_out_val( l ) = val
              A_out_ptr( j ) = l + 1
              IF ( i /= j ) THEN
                l = A_out_ptr( i )
                A_out_row( l ) = j ; A_out_val( l ) = val
                A_out_ptr( i ) = l + 1
              END IF
            END IF
          END DO
        END DO

!  finally, restore the starting addresses

        DO i = n, 1, - 1
          A_out_ptr( i + 1 ) = A_out_ptr( i )
        END DO
        A_out_ptr( 1 ) = 1

!  A is a dense matrix stored by columns

      CASE ( 'DENSE_BY_COLUMNS' )

!  count the number of nonzeros in each column

        k = 0
        A_out_ptr( 1 : n ) = 0
        DO j = 1, n
          DO i = j, n
            k = k + 1
            IF ( A%val( k ) /= zero ) THEN
              A_out_ptr( j ) = A_out_ptr( j ) + 1
              IF ( i /= j ) THEN
                A_out_ptr( i ) = A_out_ptr( i ) + 1
              END IF
            END IF
          END DO
        END DO

!  find the starting address for each column in the output scheme

        l = 1
        DO j = 1, n
          k = l + A_out_ptr( j )
          A_out_ptr( j ) = l
          l = k
        END DO

!  now allocate the row indices and values in the output scheme

        k = 0
        DO j = 1, n
          DO i = j, n
            k = k + 1
            val = A%val( k )
            IF ( val /= zero ) THEN
              l = A_out_ptr( j )
              A_out_row( l ) = i ; A_out_val( l ) = val
              A_out_ptr( j ) = l + 1
              IF ( i /= j ) THEN
                l = A_out_ptr( i )
                A_out_row( l ) = j ; A_out_val( l ) = val
                A_out_ptr( i ) = l + 1
              END IF
            END IF
          END DO
        END DO

!  finally, restore the starting addresses

        DO i = n, 1, - 1
          A_out_ptr( i + 1 ) = A_out_ptr( i )
        END DO
        A_out_ptr( 1 ) = 1

!  A is a sparse matrix stored by rows

      CASE ( 'SPARSE_BY_ROWS' )

!  count the number of nonzeros in each column

        A_out_ptr( 1 : n ) = 0
        DO i = 1, n
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            IF ( A%val( l ) /= zero ) THEN
              j = A%col( l )
              A_out_ptr( j ) = A_out_ptr( j ) + 1
              IF ( i /= j ) THEN
                A_out_ptr( i ) = A_out_ptr( i ) + 1
              END IF
            END IF
          END DO
        END DO

!  find the starting address for each column in the output scheme

        l = 1
        DO j = 1, n
          k = l + A_out_ptr( j )
          A_out_ptr( j ) = l
          l = k
        END DO

!  now allocate the row indices and values in the output scheme

        DO i = 1, n
          DO k = A%ptr( i ), A%ptr( i + 1 ) - 1
            val = A%val( k )
            IF ( val /= zero ) THEN
              j = A%col( k )
              l = A_out_ptr( j )
              A_out_row( l ) = i ; A_out_val( l ) = val
              A_out_ptr( j ) = l + 1
              IF ( i /= j ) THEN
                l = A_out_ptr( i )
                A_out_row( l ) = j ; A_out_val( l ) = val
                A_out_ptr( i ) = l + 1
              END IF
            END IF
          END DO
        END DO

!  finally, restore the starting addresses

        DO i = n, 1, - 1
          A_out_ptr( i + 1 ) = A_out_ptr( i )
        END DO
        A_out_ptr( 1 ) = 1

!  A is a sparse matrix stored by columns

      CASE ( 'SPARSE_BY_COLUMNS' )

!  count the number of nonzeros in each column

        A_out_ptr( 1 : n ) = 0
        DO j = 1, n
          DO k = A%ptr( j ), A%ptr( j + 1 ) - 1
            IF ( A%val( k ) /= zero ) THEN
              i = A%row( k )
              A_out_ptr( j ) = A_out_ptr( j ) + 1
              IF ( i /= j ) THEN
                A_out_ptr( i ) = A_out_ptr( i ) + 1
              END IF
            END IF
          END DO
        END DO

!  find the starting address for each column in the output scheme

        l = 1
        DO j = 1, n
          k = l + A_out_ptr( j )
          A_out_ptr( j ) = l
          l = k
        END DO

!  now allocate the row indices and values in the output scheme

        DO j = 1, n
          DO k = A%ptr( j ), A%ptr( j + 1 ) - 1
            val = A%val( k )
            IF ( val /= zero ) THEN
              i = A%row( k )
              l = A_out_ptr( j )
              A_out_row( l ) = i ; A_out_val( l ) = val
              A_out_ptr( j ) = l + 1
              IF ( i /= j ) THEN
                l = A_out_ptr( i )
                A_out_row( l ) = j ; A_out_val( l ) = val
                A_out_ptr( i ) = l + 1
              END IF
            END IF
          END DO
        END DO

!  finally, restore the starting addresses

        DO i = n, 1, - 1
          A_out_ptr( i + 1 ) = A_out_ptr( i )
        END DO
        A_out_ptr( 1 ) = 1

!  A is a sparse matrix stored by its co-ordinates

      CASE ( 'COORDINATE' )

!  count the number of nonzeros in each column

        A_out_ptr( 1 : n ) = 0
        DO k = 1, A%ne
          IF ( A%val( k ) /= zero ) THEN
            i = A%row( k ) ; j = A%col( k )
            A_out_ptr( j ) = A_out_ptr( j ) + 1
            IF ( i /= j ) THEN
              A_out_ptr( i ) = A_out_ptr( i ) + 1
            END IF
          END IF
        END DO

!  find the starting address for each column in the output scheme

        l = 1
        DO j = 1, n
          k = l + A_out_ptr( j )
          A_out_ptr( j ) = l
          l = k
        END DO

!  now allocate the row indices and values in the output scheme

        DO k = 1, A%ne
          val = A%val( k )
          IF ( val /= zero ) THEN
            i = A%row( k ) ; j = A%col( k )
            l = A_out_ptr( j )
            A_out_row( l ) = i ; A_out_val( l ) = val
            A_out_ptr( j ) = l + 1
            IF ( i /= j ) THEN
              l = A_out_ptr( i )
              A_out_row( l ) = j ; A_out_val( l ) = val
              A_out_ptr( i ) = l + 1
            END IF
          END IF
        END DO

!  finally, restore the starting addresses

        DO i = n, 1, - 1
          A_out_ptr( i + 1 ) = A_out_ptr( i )
        END DO
        A_out_ptr( 1 ) = 1

!  A is a diagonal matrix

      CASE ( 'DIAGONAL' )
        A_out_ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out_row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out_val( 1 : n ) = A%val( 1 : n )
      CASE ( 'SCALED-IDENTITY' )
        A_out_ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out_row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out_val( 1 : n ) = A%val( 1 )
      CASE ( 'IDENTITY' )
        A_out_ptr( 1 : n + 1 ) = [ ( i, i = 1, n + 1 ) ]
        A_out_row( 1 : n ) = [ ( i, i = 1, n ) ]
        A_out_val( 1 : n ) = 1.0_rp_

!  A is the zero matrix

      CASE ( 'ZERO', 'NONE' )
        A_out_ptr( 1 : n + 1 ) = 1
      END SELECT

!  if the mapping array is present, make sure that it is adjusted if
!  the matrix is to be stored so that the row indices in each column appear
!  in increasing order or if the matrix is to be squeezed to sum duplicate
!  entries

!     IF ( map_vals ) THEN
!  to do ... perhaps
!     END IF

!  record the total time taken

!890  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      RETURN

!  error returns

 900  CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start

      IF ( control%error > 0 .AND. control%print_level > 0 )                   &
        WRITE( control%error, "( ' ', /, A, '    ** Error return ', I0,        &
       &  ' from CONVERT ' )" ) prefix, inform%status
      RETURN

!  end of subroutine CONVERT_from_symmetric_to_CSC_main

      END SUBROUTINE CONVERT_from_symmetric_to_CSC_main

!-*-*-*-*-   Q P O A S E S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPOASES_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine QPOASES_initialize
!   control see Subroutine QPOASES_initialize
!   inform  see Subroutine QPOASES_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPOASES_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPOASES_control_type ), INTENT( IN ) :: control
      TYPE ( QPOASES_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate arrays

      array_name = 'qpoases: data%A_val'
      CALL SPACE_dealloc_array( data%A_val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%A_row'
      CALL SPACE_dealloc_array( data%A_row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%A_ptr'
      CALL SPACE_dealloc_array( data%A_ptr, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%B_l'
      CALL SPACE_dealloc_array( data%B_l, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%B_u'
      CALL SPACE_dealloc_array( data%B_u, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%Y'
      CALL SPACE_dealloc_array( data%Y, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      IF ( .NOT. data%original_a ) THEN
        array_name = 'qpoases: data%A_ptr'
        CALL SPACE_dealloc_array( data%A_ptr, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'qpoases: data%A_row'
        CALL SPACE_dealloc_array( data%A_row, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

        array_name = 'qpoases: data%A_val'
        CALL SPACE_dealloc_array( data%A_val, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN
      END IF

      array_name = 'qpoases: data%H_ptr'
      CALL SPACE_dealloc_array( data%H_ptr, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%H_row'
      CALL SPACE_dealloc_array( data%H_row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

      array_name = 'qpoases: data%H_val'
      CALL SPACE_dealloc_array( data%H_val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

!  ensure that the structure will be re-initialised on any subsequent call

      data%new_structure = .TRUE.
      RETURN

!  End of subroutine QPOASES_terminate

      END SUBROUTINE QPOASES_terminate

! - G A L A H A D -  Q P O A S E S _ f u l l _ t e r m i n a t e  S U B R OUTINE

     SUBROUTINE QPOASES_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( QPOASES_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( QPOASES_control_type ), INTENT( IN ) :: control
     TYPE ( QPOASES_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL QPOASES_terminate( data%qpoases_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'qpoases: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%C'
     CALL SPACE_dealloc_array( data%prob%C,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%H%type'
     CALL SPACE_dealloc_array( data%prob%H%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%H%ptr'
     CALL SPACE_dealloc_array( data%prob%H%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%H%row'
     CALL SPACE_dealloc_array( data%prob%H%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%H%col'
     CALL SPACE_dealloc_array( data%prob%H%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%H%val'
     CALL SPACE_dealloc_array( data%prob%H%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%A%type'
     CALL SPACE_dealloc_array( data%prob%A%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%A%ptr'
     CALL SPACE_dealloc_array( data%prob%A%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%A%row'
     CALL SPACE_dealloc_array( data%prob%A%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%A%col'
     CALL SPACE_dealloc_array( data%prob%A%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'qpoases: data%prob%A%val'
     CALL SPACE_dealloc_array( data%prob%A%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine QPOASES_full_terminate

     END SUBROUTINE QPOASES_full_terminate

!  End of module QPOASES

   END MODULE GALAHAD_QPOASES_precision
