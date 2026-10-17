! THIS VERSION: GALAHAD 5.6 - 2026-10-17 AT 15:50 GMT.

#include "galahad_modules.h"

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ B P M P D   M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 5.6. October 6th 2026

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_BPMPD_precision

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
!     | the BPMPD package (which must be provided    |
!     | externally). See README.bpmpd for details    |
!     | of how this is possible                      |
!     |                                              |
!     ------------------------------------------------

      USE GALAHAD_KINDS_precision, ONLY: i4_, ip_, rp_
      USE GALAHAD_CLOCK, ONLY: CLOCK_time
      USE GALAHAD_SYMBOLS, ONLY: GALAHAD_ok,                                   &
                                 GALAHAD_error_allocate,                       &
                                 GALAHAD_error_primal_infeasible,              &
                                 GALAHAD_error_dual_infeasible,                &
                                 GALAHAD_no_progress,                          &
                                 GALAHAD_error_max_iterations,                 &
                                 GALAHAD_error_inertia,                        &
                                 GALAHAD_unavailable_solver,                   &
                                 GALAHAD_error_bpmpd
      USE GALAHAD_SPACE_precision, ONLY: SPACE_resize_array, SPACE_dealloc_array
      USE GALAHAD_SPECFILE_precision, ONLY: SPECFILE_item_type, SPECFILE_read, &
                                            SPECFILE_assign_value
      USE GALAHAD_SMT_precision, ONLY: SMT_type, SMT_get, SMT_put
      USE GALAHAD_QPT_precision, ONLY: QPT_problem_type, QPT_keyword_A,        &
                                       QPT_keyword_H, QPT_summarize_problem
      USE GALAHAD_QPD_precision, ONLY: QPD_data_type, QPD_SIF
      USE GALAHAD_CONVERT_precision, ONLY: CONVERT_control_type,               &
                                       CONVERT_inform_type,                    &
                                       CONVERT_to_sparse_column_format,        &
                                       CONVERT_to_sparse_symmetric_column_format

      IMPLICIT NONE ( TYPE, EXTERNAL )

      PRIVATE
      PUBLIC :: BPMPD_initialize, BPMPD_read_specfile, BPMPD_solve,            &
                BPMPD_terminate, QPT_problem_type, SMT_type, SMT_get, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE BPMPD_initialize
       MODULE PROCEDURE BPMPD_initialize, BPMPD_full_initialize
     END INTERFACE BPMPD_initialize

     INTERFACE BPMPD_terminate
       MODULE PROCEDURE BPMPD_terminate, BPMPD_full_terminate
     END INTERFACE BPMPD_terminate

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

      TYPE, PUBLIC :: BPMPD_control_type

!   error and warning diagnostics occur on stream error

        INTEGER ( KIND = ip_ ) :: error = 6

!   general output occurs on stream out

        INTEGER ( KIND = ip_ ) :: out = 6

!   the level of output required is specified by print_level

        INTEGER ( KIND = ip_ ) :: print_level = 0

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = rp_ ) :: infinity = ten ** 19

!  - - - - - - - - - - - - - - - - - - - - - - - -
!  options from the BPMPD package (see bpmpd.par)
!  - - - - - - - - - - - - - - - - - - - - - - - -

!  density handling and factorization type parameters:
!  ---------------------------------------------------

!  maximal dense columns rate

        REAL ( KIND = rp_ ) :: maxdense = 0.2_rp_

!  density gap parameter

        REAL ( KIND = rp_ ) :: densgap = 3.5_rp_

!  value for the possibily dense columns

        INTEGER ( KIND = ip_ ) :: denslen = 10

!  'super' dense column length

        INTEGER ( KIND = ip_ ) :: supdens = 350

!   0 : density gap heuristic, ( AAT )
!  -1 : use LAM, ( AAT )
!   1 : density gap heuristic, factorize

        INTEGER ( KIND = ip_ ) :: setlam = 0

!  supernode parameters:
!  ---------------------

!  primer supernode length

        INTEGER ( KIND = ip_ ) :: psupnode = 4

!  secunder supernode length

        INTEGER ( KIND = ip_ ) :: ssupnode = 4

!  maximum nonzeros in one supernode

        INTEGER ( KIND = ip_ ) :: maxsnz = 99999999

!  pivot and factorization parameters:
!  -----------------------------------

!  first threshold pivot tolerance

        REAL ( KIND = rp_ ) :: tpiv1 = ten ** ( - 3 )

!  second threshold pivot tolerance

        REAL ( KIND = rp_ ) :: tpiv2 = ten ** ( - 8 )

!  absolute pivot tolerance for the first factorization

        REAL ( KIND = rp_ ) :: tabs = ten ** ( - 12 )

!  absolute pivot tolerance during the algorithm

        REAL ( KIND = rp_ ) :: trabs = ten ** ( - 15 )

!  minimum value of lambda

        REAL ( KIND = rp_ ) :: lam = ten ** ( - 5 )

!  pivot search loop count

        INTEGER ( KIND = ip_ ) :: tfind = 25

!   1.0 : Minimum degree
!   2.0 : Minimum fill-in
!   3.0 : Minimum fill-in for augmented system
!  other: No ordering

        REAL ( KIND = rp_ ) :: ordering = 2.0_rp_

!  stopping criterion parameters:
!  ------------------------------

!  relative duality gap tolerance

        REAL ( KIND = rp_ ) :: topt1 = ten ** ( - 8 )

!  average complementary gap tolerance

        REAL ( KIND = rp_ ) :: topt2 = ten ** ( - 20 )

!  relative primal feasibility tolerance

        REAL ( KIND = rp_ ) :: tfeas1 = 0.00000019_rp_

!  relative dual feasibility tolerance

        REAL ( KIND = rp_ ) :: tfeas2 = 0.00000019_rp_

!  infeasibility check tolerance

        REAL ( KIND = rp_ ) :: inftol = ten ** 4

!  search direction maximum norm tolerance

        REAL ( KIND = rp_ ) :: tsdir = ten ** ( -16 )

!  iteration limit

        INTEGER ( KIND = ip_ ) :: maxiter = 99

!  numerical tolerances:
!  ---------------------

!  relative addition tolerance

        REAL ( KIND = rp_ ) :: tplus = ten ** ( -10 )

!  relative zero tolerance

        REAL ( KIND = rp_ ) :: tzer = ten ** ( - 35 )

!  iterative refinement tolerances:
!  --------------------------------

!  acceptable residual in the primal space

        REAL ( KIND = rp_ ) :: tresx = ten ** ( - 10 )

!  acceptable residual in the dual space

        REAL ( KIND = rp_ ) :: tresy = ten ** ( - 10 )

!  maximal number of refinement

        INTEGER ( KIND = ip_ ) :: maxref = 5

!  refinement method: 1: Iterative improvement
!                     2: Preconditioned QMR
!                     4: Preconditioned CGM

        INTEGER ( KIND = ip_ ) :: refmet = 1

!  scaling parameters:
!  -------------------

!  scaling the objective to this max. norm

        REAL ( KIND = rp_ ) :: objnorm = ten ** 2

!  scaling the rhs to this max. norm

        REAL ( KIND = rp_ ) :: rhsnorm = zero

!  ignore values during scaling under this parameter before aggregator

        REAL ( KIND = rp_ ) :: signore = ten ** ( - 12 )

!  maximum number of passes ( <128 ) before aggregator

        INTEGER ( KIND = ip_ ) :: spasses1 = 5

!  scaling method before aggregator
!   0. : No scaling
!   1. : Simple scaling to rowmax=1, colmax=1
!   2. : Geometric mean scaling + simple scaling
!   3. : Curtis-Reid's algorithm + simple scaling
!   4. : Geometric mean scaling only
!   5. : Curtis-Reid's algorithm only

        INTEGER ( KIND = ip_ ) :: smethod1 = 2

!  complementary gap ballancing:
!  -----------------------------

!  minimum stepsize

        REAL ( KIND = rp_ ) :: minstep = 0.5_rp_

!  min/Max complementarity component threshold

        REAL ( KIND = rp_ ) :: complmin = ten ** ( - 3 )

!  predictor-corrector and barrier parameters:
!  -------------------------------------------

!  correction stop parameter

        REAL ( KIND = rp_ ) :: stopcor = ten ** ( - 3 )

!  barrier set-up limit

        REAL ( KIND = rp_ ) :: barset = 0.2_rp_

!  barrier grow controll

        REAL ( KIND = rp_ ) :: bargrow = ten ** 3

!  minimum barrier threshold

        REAL ( KIND = rp_ ) :: barmin = ten ** ( -15 )

!  number of the minimum corrections

        INTEGER ( KIND = ip_ ) :: mincorr = - 1

!  number of the maximum corrections

        INTEGER ( KIND = ip_ ) :: maxcorr = 1

!  use initial barrier parameter

        REAL ( KIND = rp_ ) :: inibarr = zero

!  centrality corrections parameters:
!  ----------------------------------

!  trial steplength improvement

        REAL ( KIND = rp_ ) :: target = 0.09_rp_

!  small complementarity bound

        REAL ( KIND = rp_ ) :: tsmall = 0.2_rp_

!  large compelmentarity bound

        REAL ( KIND = rp_ ) :: tlarge = 20.0_rp_

!  centrality force

        REAL ( KIND = rp_ ) :: center = 5.00D+00

!  correction stop parameter

        REAL ( KIND = rp_ ) :: corstop = 1.01_rp_

!  number of the minimum corrections

        INTEGER ( KIND = ip_ ) :: minccorr = 0

!  number of the maximum corrections

        INTEGER ( KIND = ip_ ) :: maxccorr = 9

!  steplenth parameters:
!  ---------------------

!  maximal primal steplength

        REAL ( KIND = rp_ ) :: prdare = 0.999_rp_

!  maximal dual   steplength

        REAL ( KIND = rp_ ) :: dudare = 0.999_rp_

!  variable fixing tolerances:
!  ---------------------------

!  variable reset parameter

        REAL ( KIND = rp_ ) :: tfixvar = ten ** ( - 20 )

!  slack reset parameter

        REAL ( KIND = rp_ ) :: tfixslack = ten ** ( - 20 )

!  dual slack variable limit

        REAL ( KIND = rp_ ) :: dslacklim = ten ** ( - 20 )

!  starting point paramerers:
!  --------------------------

!  minimum initial variable value

        REAL ( KIND = rp_ ) :: prmin = 150.0_rp_

!  maximum initial variable value

        REAL ( KIND = rp_ ) :: upmax = 50000.0_rp_

!  minimum initial slack value

        REAL ( KIND = rp_ ) :: dumin = 150.0_rp_

!  lsq weight over quadratic objective

        REAL ( KIND = rp_ ) :: lsqweight = ten

!  starting method  (1. or 2.)

        INTEGER ( KIND = ip_ ) :: smethod = 2

!  safe method  (1. 2. or 3.)

        INTEGER ( KIND = ip_ ) :: safemet = - 3

!  regularization of equality constraints as free variables
!  -1. : never regularize
!   0. : regularize the augmented system
!   1. : regularize always

        INTEGER ( KIND = ip_ ) :: regularize = 0

!  presolve parameters:
!  --------------------

!  method (sum of the following):
!     1 : singleton row check
!     2 : singleton column check
!     4 : min/max row value check
!     8 : cheap dual tests
!    16 : dual check
!    32 : primal bound check and relaxation
!    64 : search identical variables
!   128 : setting vars free in row doubletons
!   256 : elimination of free variables
!   512 : linear dependency check
!  1024 : sparser
!  2048 : extended dual tests

        INTEGER ( KIND = ip_ ) :: presolv = 2047

!  maximum loops in bound check

        INTEGER ( KIND = ip_ ) :: bndloop = 5

!  maximum loops in dual check

        INTEGER ( KIND = ip_ ) :: duloop = 10

!  maximal upper bound allowed during bndcheck

        REAL ( KIND = rp_ ) :: primalbnd = 1000.0_rp_

!  maximal upper bound allowed during ducheck

        REAL ( KIND = rp_ ) :: dualbnd = 10000.0_rp_

!  feasibility tolerance during presolve

        REAL ( KIND = rp_ ) :: presfeas = ten ** ( - 8 )

!  relative pivot tolerance in aggragation

        REAL ( KIND = rp_ ) :: pivrtol = 0.01_rp_

!  absolute pivot tolerance in aggregation

        REAL ( KIND = rp_ ) :: pivatol = ten ** ( - 4 )

!  absolute pivot tolerance in linear dep. check

        REAL ( KIND = rp_ ) :: pivxtol = ten ** ( - 8 )

!  fill-in controll during aggregation

        INTEGER ( KIND = ip_ ) :: filltol = 4

!  regularization parameters:
!  --------------------------

!  soft regularization parameter

        REAL ( KIND = rp_ ) :: softreg = ten ** ( - 12 )

!  hard regularization parameter

        REAL ( KIND = rp_ ) :: hardreg = ten ** ( - 16 )

!  regularization for free variables

        REAL ( KIND = rp_ ) :: scfree = ten ** ( - 6 )

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

     END TYPE BPMPD_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   extended inform derived type with defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BPMPD_inform_type

!  return status

        INTEGER ( KIND = ip_ ) :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the inform code from the fortran 77 BPMPD package

        INTEGER ( KIND = ip_ ) :: code = - 1

!  the total number of iterations required

        INTEGER ( KIND = ip_ ) :: iter = - 1

!  the value of the objective function at the best estimate of the solution
!   determined by BPMPD_solve

        REAL ( KIND = rp_ ) :: obj = HUGE( one )

      END TYPE BPMPD_inform_type

!  - - - - - - - - - - - - - -
!   data derived type for BPMPD
!  - - - - - - - - - - - - - -

      TYPE, PUBLIC :: BPMPD_data_type
        INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: STATUS
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RHS
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: LBOUND, UBOUND
        REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: PRIMAL, DUAL
        LOGICAL :: original_a, original_h
        LOGICAL :: new_structure = .TRUE.
        TYPE ( SMT_type ) :: A, H
      END TYPE BPMPD_data_type

!  - - - - - - - - - - - -
!   full_data derived type
!  - - - - - - - - - - - -

      TYPE, PUBLIC :: BPMPD_full_data_type
        LOGICAL :: f_indexing = .TRUE.
        TYPE ( BPMPD_data_type ) :: BPMPD_data
        TYPE ( BPMPD_control_type ) :: BPMPD_control
        TYPE ( BPMPD_inform_type ) :: BPMPD_inform
        TYPE ( QPT_problem_type ) :: prob
      END TYPE BPMPD_full_data_type

   CONTAINS

!-*-*-*-*-*-   B P M P D _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE BPMPD_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for BPMPD. This routine should be called before
!  BPMPD_solve
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

      TYPE ( BPMPD_data_type ), INTENT( INOUT ) :: data
      TYPE ( BPMPD_control_type ), INTENT( OUT ) :: control
      TYPE ( BPMPD_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

      RETURN

!  End of BPMPD_initialize

      END SUBROUTINE BPMPD_initialize

! G A L A H A D -  B P M P D _ F U L L _ I N I T I A L I Z E  S U B R O U T I NE

     SUBROUTINE BPMPD_full_initialize( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for BPMPD controls

!   Arguments:

!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BPMPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BPMPD_control_type ), INTENT( OUT ) :: control
     TYPE ( BPMPD_inform_type ), INTENT( OUT ) :: inform

     CALL BPMPD_initialize( data%BPMPD_data, control, inform )

     RETURN

!  End of subroutine BPMPD_full_initialize

     END SUBROUTINE BPMPD_full_initialize

!-*-*-*-   B P M P D _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE BPMPD_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by BPMPD_initialize could (roughly)
!  have been set as:

! BEGIN BPMPD SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  infinity-value                                    1.0D+19
!  maxdense                                          0.2
!  densgap                                           3.5
!  denslen                                           10
!  supdens                                           350
!  setlam                                            0
!  psupnode                                          4
!  ssupnode                                          4
!  maxsnz                                            99999999
!  tpiv1                                             1.0D-3
!  tpiv2                                             1.0D-8
!  tabs                                              1.0D-12
!  trabs                                             1.0D-15
!  lam                                               1.0D-5
!  tfind                                             25
!  ordering                                          2.0
!  topt1                                             1.0D-8
!  topt2                                             1.0D-20
!  tfeas1                                            0.00000019
!  tfeas2                                            0.00000019
!  inftol                                            1.0D4
!  tsdir                                             1.0D-16
!  maxiter                                           99
!  tplus                                             1.0D-10
!  tzer                                              1.0D-35
!  tresx                                             1.0D-10
!  tresy                                             1.0D-10
!  maxref                                            5
!  refmet                                            1
!  objnorm                                           100.0
!  rhsnorm                                           0.0
!  signore                                           1.0D-12
!  spasses1                                          5
!  smethod1                                          2
!  minstep                                           0.5
!  complmin                                          1.0D-3
!  stopcor                                           1.0D-3
!  barset                                            0.2
!  bargrow                                           1000.0
!  barmin                                            1.0D-15
!  mincor                                            -1
!  maxcorr                                           1
!  inibarr                                           0.0
!  target                                            0.09
!  tsmall                                            0.2
!  tlarge                                            20.0
!  center                                            5.0
!  corstop                                           1.01
!  minccorr                                          0
!  maxccorr                                          9
!  prdare                                            0.999
!  dudare                                            0.999
!  tfixvar                                           1.0D-20
!  tfixslack                                         1.0D-20
!  dslacklim                                         1.0D-20
!  prmin                                             150.0
!  upmax                                             50000.0
!  dumin                                             150.0
!  lsqweight                                         10.0
!  smethod                                           2
!  safemet                                           -3
!  regularize                                        0
!  presolv                                           2047
!  bndloop                                           5
!  duloop                                            10
!  primalbnd                                         1000.0
!  dualbnd                                           10000.0
!  presfeas                                          1.0D-8
!  pivrtol                                           0.01
!  pivatol                                           1.0D-4
!  pivxtol                                           1.0D-8
!  filltol                                           4
!  softreg                                           1.0D-12
!  hardreg                                           1.0D-16
!  scfree                                            1.0D-6
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  output-line-prefix                                ""
! END BPMPD SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( BPMPD_control_type ), INTENT( INOUT ) :: control
      INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
      INTEGER ( KIND = ip_ ), PARAMETER :: out = error + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: print_level = out + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: infinity = print_level + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: softreg  = infinity + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: hardreg = softreg + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tpiv1 = hardreg + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tpiv2 = tpiv1 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tabs = tpiv2 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: trabs = tabs + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lam = trabs + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tfind = lam + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: ordering = tfind + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: psupnode = ordering + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxsnz = psupnode + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: ssupnode = maxsnz + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: topt1 = ssupnode + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: topt2 = topt1 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tfeas1 = topt2 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tfeas2 = tfeas1 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: inftol = tfeas2 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: denslen = inftol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tplus = denslen + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tzer = tplus + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prdare = tzer + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tresx = prdare + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tresy = tresx + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxref = tresy + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: refmet = maxref + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dudare = refmet + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: smethod1 = dudare + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: objnorm = smethod1 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: rhsnorm = objnorm + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: signore = rhsnorm + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: spasses1 = signore + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: stopcor = spasses1 + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: barset = stopcor + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: bargrow = barset + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: barmin = bargrow + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: mincorr = barmin + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxcorr = mincorr + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: inibarr = maxcorr + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: target = inibarr + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tsmall = target + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tlarge = tsmall + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: center = tlarge + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: corstop = center + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: minccorr = corstop + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxccorr = minccorr + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: minstep = maxccorr + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: complmin = minstep + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tfixvar = complmin + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tfixslack = tfixvar + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dslacklim = tfixslack + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prmin = dslacklim + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: upmax = prmin + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dumin = upmax + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lsqweight = dumin + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: smethod = lsqweight + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: safemet = smethod + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: regularize = safemet + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: presolv = regularize + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxiter = presolv + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: maxdense = maxiter + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: densgap = maxdense + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: setlam = densgap + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: scfree = setlam + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: tsdir = scfree + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: supdens = tsdir + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: presfeas = supdens + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: duloop = presfeas + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: bndloop = duloop + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: primalbnd = bndloop + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: dualbnd = primalbnd + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: pivrtol = dualbnd + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: pivatol = pivrtol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: pivxtol = pivatol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: filltol = pivxtol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: space_critical = filltol + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: deallocate_error_fatal              &
                                             = space_critical + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: prefix = deallocate_error_fatal + 1
      INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
      CHARACTER( LEN = 5 ), PARAMETER :: specname = 'BPMPD'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( softreg )%keyword = 'softreg'
      spec( hardreg )%keyword = 'hardreg'
      spec( tpiv1 )%keyword = 'tpiv1'
      spec( tpiv2 )%keyword = 'tpiv2'
      spec( tabs )%keyword = 'tabs'
      spec( trabs )%keyword = 'trabs'
      spec( lam )%keyword = 'lam'
      spec( tfind )%keyword = 'tfind'
      spec( ordering )%keyword = 'ordering'
      spec( psupnode )%keyword = 'psupnode'
      spec( maxsnz )%keyword = 'maxsnz'
      spec( ssupnode )%keyword = 'ssupnode'
      spec( topt1 )%keyword = 'topt1'
      spec( topt2 )%keyword = 'topt2'
      spec( tfeas1 )%keyword = 'tfeas1'
      spec( tfeas2 )%keyword = 'tfeas2'
      spec( inftol )%keyword = 'inftol'
      spec( denslen )%keyword = 'denslen'
      spec( tplus )%keyword = 'tplus'
      spec( tzer )%keyword = 'tzer'
      spec( prdare )%keyword = 'prdare'
      spec( tresx )%keyword = 'tresx'
      spec( tresy )%keyword = 'tresy'
      spec( maxref )%keyword = 'maxref'
      spec( refmet )%keyword = 'refmet'
      spec( dudare )%keyword = 'dudare'
      spec( smethod1 )%keyword = 'smethod1'
      spec( objnorm )%keyword = 'objnorm'
      spec( rhsnorm )%keyword = 'rhsnorm'
      spec( signore )%keyword = 'signore'
      spec( spasses1 )%keyword = 'spasses1'
      spec( stopcor )%keyword = 'stopcor'
      spec( barset )%keyword = 'barset'
      spec( bargrow )%keyword = 'bargrow'
      spec( barmin )%keyword = 'barmin'
      spec( mincorr )%keyword = 'mincorr'
      spec( maxcorr )%keyword = 'maxcorr'
      spec( inibarr )%keyword = 'inibarr'
      spec( target )%keyword = 'target'
      spec( tsmall )%keyword = 'tsmall'
      spec( tlarge )%keyword = 'tlarge'
      spec( center )%keyword = 'center'
      spec( corstop )%keyword = 'corstop'
      spec( minccorr )%keyword = 'minccorr'
      spec( maxccorr )%keyword = 'maxccorr'
      spec( minstep )%keyword = 'minstep'
      spec( complmin )%keyword = 'complmin'
      spec( tfixvar )%keyword = 'tfixvar'
      spec( tfixslack )%keyword = 'tfixslack'
      spec( dslacklim )%keyword = 'dslacklim'
      spec( prmin )%keyword = 'prmin'
      spec( upmax )%keyword = 'upmax'
      spec( dumin )%keyword = 'dumin'
      spec( lsqweight )%keyword = 'lsqweight'
      spec( smethod )%keyword = 'smethod'
      spec( safemet )%keyword = 'safemet'
      spec( regularize )%keyword = 'regularize'
      spec( presolv )%keyword = 'presolv'
      spec( maxiter )%keyword = 'maxiter'
      spec( maxdense )%keyword = 'maxdense'
      spec( densgap )%keyword = 'densgap'
      spec( setlam )%keyword = 'setlam'
      spec( scfree )%keyword = 'scfree'
      spec( tsdir )%keyword = 'tsdir'
      spec( supdens )%keyword = 'supdens'
      spec( presfeas )%keyword = 'presfeas'
      spec( duloop )%keyword = 'duloop'
      spec( bndloop )%keyword = 'bndloop'
      spec( primalbnd )%keyword = 'primalbnd'
      spec( dualbnd )%keyword = 'dualbnd'
      spec( pivrtol )%keyword = 'pivrtol'
      spec( pivatol )%keyword = 'pivatol'
      spec( pivxtol )%keyword = 'pivxtol'
      spec( filltol )%keyword = 'filltol'

!  Logical key-words

      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' bpmpd: ', alt_specname

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

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( softreg ),                              &
                                 control%softreg,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( hardreg ),                              &
                                 control%hardreg,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tpiv1 ),                                &
                                 control%tpiv1,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tpiv2 ),                                &
                                 control%tpiv2,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tabs ),                                 &
                                 control%tabs,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( trabs ),                                &
                                 control%trabs,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lam ),                                  &
                                 control%lam,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tfind ),                                &
                                 control%tfind,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( ordering ),                             &
                                 control%ordering,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( psupnode ),                             &
                                 control%psupnode,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxsnz ),                               &
                                 control%maxsnz,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( ssupnode ),                             &
                                 control%ssupnode,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( topt1 ),                                &
                                 control%topt1,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( topt2 ),                                &
                                 control%topt2,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tfeas1 ),                               &
                                 control%tfeas1,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tfeas2 ),                               &
                                 control%tfeas2,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( inftol ),                               &
                                 control%inftol,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( denslen ),                              &
                                 control%denslen,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tplus ),                                &
                                 control%tplus,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tzer ),                                 &
                                 control%tzer,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prdare ),                               &
                                 control%prdare,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tresx ),                                &
                                 control%tresx,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tresy ),                                &
                                 control%tresy,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxref ),                               &
                                 control%maxref,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( refmet ),                               &
                                 control%refmet,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dudare ),                               &
                                 control%dudare,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( smethod1 ),                             &
                                 control%smethod1,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( objnorm ),                              &
                                 control%objnorm,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( rhsnorm ),                              &
                                 control%rhsnorm,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( signore ),                              &
                                 control%signore,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( spasses1 ),                             &
                                 control%spasses1,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stopcor ),                              &
                                 control%stopcor,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( barset ),                               &
                                 control%barset,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( bargrow ),                              &
                                 control%bargrow,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( barmin ),                               &
                                 control%barmin,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( mincorr ),                              &
                                 control%mincorr,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxcorr ),                              &
                                 control%maxcorr,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( inibarr ),                              &
                                 control%inibarr,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( target ),                               &
                                 control%target,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tsmall ),                               &
                                 control%tsmall,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tlarge ),                               &
                                 control%tlarge,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( center ),                               &
                                 control%center,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( corstop ),                              &
                                 control%corstop,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minccorr ),                             &
                                 control%minccorr,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxccorr ),                             &
                                 control%maxccorr,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minstep ),                              &
                                 control%minstep,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( complmin ),                             &
                                 control%complmin,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tfixvar ),                              &
                                 control%tfixvar,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tfixslack ),                            &
                                 control%tfixslack,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dslacklim ),                            &
                                 control%dslacklim,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prmin ),                                &
                                 control%prmin,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( upmax ),                                &
                                 control%upmax,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dumin ),                                &
                                 control%dumin,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( lsqweight ),                            &
                                 control%lsqweight,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( smethod ),                              &
                                 control%smethod,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( safemet ),                              &
                                 control%safemet,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( regularize ),                           &
                                 control%regularize,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( presolv ),                              &
                                 control%presolv,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxiter ),                              &
                                 control%maxiter,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxdense ),                             &
                                 control%maxdense,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( densgap ),                              &
                                 control%densgap,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( setlam ),                               &
                                 control%setlam,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scfree ),                               &
                                 control%scfree,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tsdir ),                                &
                                 control%tsdir,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( supdens ),                              &
                                 control%supdens,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( presfeas ),                             &
                                 control%presfeas,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( duloop ),                               &
                                 control%duloop,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( bndloop ),                              &
                                 control%bndloop,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( primalbnd ),                            &
                                 control%primalbnd,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dualbnd ),                              &
                                 control%dualbnd,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivrtol ),                              &
                                 control%pivrtol,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivatol ),                              &
                                 control%pivatol,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivxtol ),                              &
                                 control%pivxtol,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( filltol ),                              &
                                 control%filltol,                              &
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

      END SUBROUTINE BPMPD_read_specfile

!-*-*-*-*-*-*-*-*-   B P M P D _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE BPMPD_solve( prob, data, control, inform )

!  solve the quadratic program using the BPMPD package

!  A - by columns
!  H - lower triangle by columns

!  dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( BPMPD_data_type ), INTENT( INOUT ) :: data
      TYPE ( BPMPD_control_type ), INTENT( IN ) :: control
      TYPE ( BPMPD_inform_type ), INTENT( OUT ) :: inform

!  local variables

      INTEGER ( KIND = ip_ ) :: m, n, qn, nea, neh, np1, npm
      INTEGER ( KIND = ip_ ) :: i, msizi, msizr, loglevel
      TYPE ( CONVERT_control_type ) :: control_convert
      TYPE ( CONVERT_inform_type ) :: inform_convert
      CHARACTER ( LEN = 80 ) :: array_name

!  interface block for BPMPDX subroutine

      INTERFACE
        SUBROUTINE BPMPDX( m, n, nz, qn, qnz, ACOLCNT, ACOLIDX, ACOLNZS,       &
                           QCOLCNT, QCOLIDX, QCOLNZS, RHS, OBJ,                &
                           LBOUND, UBOUND, PRIMAL, DUAL, STATUS, big,          &
                           code, opt, msizi, msizr, f, loglevel, iter )
        IMPORT :: ip_, rp_
        INTEGER ( KIND = ip_ ), INTENT( IN ) :: m, n, nz, qn, qnz, loglevel
        INTEGER ( KIND = ip_ ), INTENT( OUT ) :: code, iter
        INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: msizi, msizr
        REAL ( KIND = rp_ ), INTENT( IN ) :: big, f
        REAL ( KIND = rp_ ), INTENT( OUT ) :: opt
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: ACOLCNT, QCOLCNT
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( nz ) :: ACOLIDX
        INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( qnz ) :: QCOLIDX
        INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( n + m ) :: STATUS
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n ) :: OBJ
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( m ) :: RHS
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( nz ) :: ACOLNZS
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( qnz ) :: QCOLNZS
        REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( n + m ) :: LBOUND, UBOUND
        REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n + m ) :: PRIMAL, DUAL
        END SUBROUTINE BPMPDX
      END INTERFACE

!  common blocks (aargh!!) from the BPMPD package

      INTEGER ( KIND = ip_ ) :: psupn, ssupn, maxsnz, maxref, refmet, scpass
      INTEGER ( KIND = ip_ ) :: sclmet, mincor, maxcor, mincc, maxcc, stamet
      INTEGER ( KIND = ip_ ) :: safmet, premet, matreg, dloop, bloop, setlam
      INTEGER ( KIND = ip_ ) :: denlen
      REAL ( KIND = rp_ ) :: tpiv1, tpiv2, tabs, trabs, lam, tfind, order
      REAL ( KIND = rp_ ) :: supdns, tsdir, topt1, topt2, tfeas1, tfeas2
      REAL ( KIND = rp_ ) :: inftol, maxits, tplus, tzer, tresx, tresy, objnor
      REAL ( KIND = rp_ ) :: rhsnor, signor, minstp, mingap, ccstop, barset
      REAL ( KIND = rp_ ) :: bargrw, barmin, inibar, target, tsmall, tlarge
      REAL ( KIND = rp_ ) :: center, corstp, palpha, dalpha, tfxvar, tfxslk
      REAL ( KIND = rp_ ) :: slklim, prmin, upmax, dumin, lsqwg, sftreg
      REAL ( KIND = rp_ ) :: hrdreg, scfree, prfeas, pbig, dbig, reltol
      REAL ( KIND = rp_ ) :: abstol, filtol, deptol, maxden, dengap
      COMMON / factor / tpiv1, tpiv2, tabs, trabs, lam, tfind, order, supdns
      COMMON / sprnod / psupn, ssupn, maxsnz
      COMMON / toler / tsdir, topt1, topt2, tfeas1, tfeas2, inftol, maxits
      COMMON / numer / tplus, tzer
      COMMON / itref / tresx, tresy, maxref, refmet
      COMMON / ascal / objnor, rhsnor, signor, scpass, sclmet
      COMMON / compl / minstp, mingap
      COMMON / predp / ccstop, barset, bargrw, barmin, inibar, mincor, maxcor
      COMMON / predc / target, tsmall, tlarge, center, corstp, mincc, maxcc
      COMMON / param / palpha, dalpha
      COMMON / drop /  tfxvar, tfxslk, slklim
      COMMON / initv / prmin, upmax, dumin, lsqwg, stamet, safmet, premet,     &
                       matreg
      COMMON / regul / sftreg, hrdreg, scfree
      COMMON / pretol / prfeas, pbig, dbig, reltol, abstol, filtol, deptol,    &
                        dloop, bloop
      COMMON / setden / maxden, dengap, setlam, denlen

write(6,*) ' in bpmpdx'

!  transfer control parameters to common blocks

      sftreg = control%softreg
      hrdreg = control%hardreg
      tpiv1 = control%tpiv1
      tpiv2 = control%tpiv2
      tabs = control%tabs
      trabs = control%trabs
      lam = control%lam
      tfind = control%tfind
      order = control%ordering
      psupn = control%psupnode
      maxsnz = control%maxsnz
      ssupn = control%ssupnode
      topt1 = control%topt1
      topt2 = control%topt2
      tfeas1 = control%tfeas1
      tfeas2 = control%tfeas2
      inftol = control%inftol
      denlen = control%denslen
      tplus = control%tplus
      tzer = control%tzer
      palpha = control%prdare
      tresx = control%tresx
      tresy = control%tresy
      maxref = control%maxref
      refmet = control%refmet
      dalpha = control%dudare
      sclmet = control%smethod1
      objnor = control%objnorm
      rhsnor = control%rhsnorm
      signor = control%signore
      scpass = control%spasses1
      ccstop = control%stopcor
      barset = control%barset
      bargrw = control%bargrow
      barmin = control%barmin
      mincor = control%mincorr
      maxcor = control%maxcorr
      inibar = control%inibarr
      target = control%target
      tsmall = control%tsmall
      tlarge = control%tlarge
      center = control%center
      corstp = control%corstop
      mincc = control%minccorr
      maxcc = control%maxccorr
      minstp = control%minstep
      mingap = control%complmin
      tfxvar = control%tfixvar
      tfxslk = control%tfixslack
      slklim = control%dslacklim
      prmin = control%prmin
      upmax = control%upmax
      dumin = control%dumin
      lsqwg = control%lsqweight
      stamet = control%smethod
      safmet = control%safemet
      matreg = control%regularize
      premet = control%presolv
      maxits = control%maxiter
      maxden = control%maxdense
      dengap = control%densgap
      setlam = control%setlam
      scfree = control%scfree
      tsdir = control%tsdir
      supdns = control%supdens
      prfeas = control%presfeas
      dloop = control%duloop
      bloop = control%bndloop
      pbig = control%primalbnd
      dbig = control%dualbnd
      reltol = control%pivrtol
      abstol = control%pivatol
      deptol = control%pivxtol
      filtol = control%filltol

!  transfer bound data into the format required by BPMPD

      n = prob%n ; m = prob%m ; np1 = n + 1 ; npm = n + m
      IF ( data%new_structure ) THEN
        array_name = 'bpmpd: data%LBOUND'
        CALL SPACE_resize_array( npm, data%LBOUND, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bpmpd: data%UBOUND'
        CALL SPACE_resize_array( npm, data%UBOUND, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF
      data%LBOUND( : n ) = prob%X_l( : n )
      data%LBOUND( np1 : npm ) = prob%C_l( : m )
      data%UBOUND( : n ) = prob%X_u( : n )
      data%UBOUND( np1 : npm ) = prob%C_u( : m )

!  convert A and H to column-wise format required by BPMPD

      prob%A%m = m ; prob%A%n = n ; prob%H%m = n ; prob%H%n = n
      CALL CONVERT_to_sparse_column_format( prob%A, data%A, control_convert,   &
                                            inform_convert )
      CALL CONVERT_to_sparse_symmetric_column_format( prob%H, data%H,          &
                                                      control_convert,         &
                                                      inform_convert )

!  recover the sizes of A and H, and the column counts for each

      nea = data%A%ptr( n + 1 ) - 1 ; neh = data%H%ptr( n + 1 ) - 1
      DO i = 1, n
        data%A%ptr( i ) = data%A%ptr( i + 1 ) - data%A%ptr( i )
        data%H%ptr( i ) = data%H%ptr( i + 1 ) - data%H%ptr( i )
      END DO

!  set space for the primal and dual variables, and start from zero

      IF ( data%new_structure ) THEN
        array_name = 'bpmpd: data%PRIMAL'
        CALL SPACE_resize_array( npm, data%PRIMAL, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bpmpd: data%DUAL'
        CALL SPACE_resize_array( npm, data%DUAL, inform%status,                &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

      data%PRIMAL = zero ; data%DUAL = zero

!  set the right hand side (as zero) and provide space for constraint statuses

      IF ( data%new_structure ) THEN
        array_name = 'bpmpd: data%STATUS'
        CALL SPACE_resize_array( npm, data%STATUS, inform%status,              &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bpmpd: data%RHS'
        CALL SPACE_resize_array( m, data%RHS, inform%status,                   &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

      data%RHS = zero

!  provide space for the solution

      IF ( data%new_structure ) THEN
        array_name = 'bpmpd: prob%X'
        CALL SPACE_resize_array( n, prob%X, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bpmpd: prob%Y'
        CALL SPACE_resize_array( m, prob%Y, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bpmpd: prob%Z'
        CALL SPACE_resize_array( n, prob%Z, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'bpmpd: prob%C'
        CALL SPACE_resize_array( m, prob%C, inform%status,                     &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

 ! make bpmpd choose appropriate sizes for workspace

      qn = n
      msizi = 0 ; msizr = 0

 ! turn off the output log file, and generate output if desired

      IF ( control%print_level <= 0 ) THEN
        loglevel = 0
      ELSE
        loglevel = 1
      END IF

!  solve the problem

      loglevel = 0 ! turn off the log file and output generation
      CALL BPMPDX( m, n, nea, qn, neh, data%A%ptr, data%A%row, data%A%val,     &
                   data%H%ptr, data%H%row, data%H%val, data%RHS, prob%G,       &
                   data%LBOUND, data%UBOUND, data%PRIMAL, data%DUAL,           &
                   data%STATUS, control%infinity, inform%code, inform%obj,     &
                   msizi, msizr, prob%f, loglevel, inform%iter )

!  record the exit status

      SELECT CASE ( inform%code )
      CASE ( - 199 )
        inform%status = GALAHAD_unavailable_solver
      CASE ( - 2, - 1 )
        inform%status = GALAHAD_error_allocate
      CASE ( 1 )
        inform%status = GALAHAD_no_progress
      CASE ( 2 )
        inform%status = GALAHAD_ok
      CASE ( 3 )
        inform%status = GALAHAD_error_dual_infeasible
      CASE ( 4 )
        inform%status = GALAHAD_error_primal_infeasible
      CASE ( 7 )
        inform%status = GALAHAD_error_max_iterations
      CASE ( 9 )
        inform%status = GALAHAD_error_inertia
      CASE DEFAULT
        inform%status = GALAHAD_error_bpmpd
      END SELECT

!  recover the solution

      IF ( inform%status >= 0 ) THEN

!  record the solution Lagrange multipliers, dual variables and residuals

        prob%X = data%PRIMAL( : n ) ; prob%Z = data%DUAL( : n )
        prob%C = data%PRIMAL( np1 : npm ) ; prob%Y = data%DUAL( np1 : npm )

!  print details, if required

        IF ( control%print_level > 0 ) THEN
          WRITE( control%out, "( /, ' BPMPD - Fortran interface' )" )
          WRITE( control%out, "( ' objective function:', ES16.8 )" ) inform%obj
!         WRITE( control%out, "( ' primal & dual residuals:', 2ES16.8 )" )     &
!           inform%info%prim_res, inform%info%dual_res
          WRITE( control%out, "( ' x:', ( 5ES16.8 ) )" ) prob%X
          WRITE( control%out, "( ' y:', ( 5ES16.8 ) )" ) prob%Y
          WRITE( control%out, "( ' z:', ( 5ES16.8 ) )" ) prob%Z
          WRITE( control%out, "( 1X, I0, ' iterations' ) ") inform%iter
          WRITE( control%out, "( ' status = ', I0, ', code = ', I0 )" )        &
            inform%status, inform%code
        END IF
      ELSE
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error,                                                &
                 "( ' Error. Problem not solved to optimality' )" )
      END IF

      RETURN

!  End of BPMPD_solve

      END SUBROUTINE BPMPD_solve

!-*-*-*-*-*-*-   B P M P D _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE BPMPD_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine BPMPD_initialize
!   control see Subroutine BPMPD_initialize
!   inform  see Subroutine BPMPD_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( BPMPD_data_type ), INTENT( INOUT ) :: data
      TYPE ( BPMPD_control_type ), INTENT( IN ) :: control
      TYPE ( BPMPD_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  deallocate arrays

      array_name = 'bpmpd: data%LBOUND'
      CALL SPACE_dealloc_array( data%LBOUND, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%UBOUND'
      CALL SPACE_dealloc_array( data%UBOUND, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%STATUS'
      CALL SPACE_dealloc_array( data%STATUS, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%RHS'
      CALL SPACE_dealloc_array( data%RHS, inform%status,                       &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%PRIMAL'
      CALL SPACE_dealloc_array( data%PRIMAL, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%DUAL'
      CALL SPACE_dealloc_array( data%DUAL, inform%status,                      &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%A%ptr'
      CALL SPACE_dealloc_array( data%A%ptr, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%A%row'
      CALL SPACE_dealloc_array( data%A%row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%A%val'
      CALL SPACE_dealloc_array( data%A%val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%A%type'
      CALL SPACE_dealloc_array( data%A%type, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%H%ptr'
      CALL SPACE_dealloc_array( data%H%ptr, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%H%row'
      CALL SPACE_dealloc_array( data%H%row, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%H%val'
      CALL SPACE_dealloc_array( data%H%val, inform%status,                     &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'bpmpd: data%H%type'
      CALL SPACE_dealloc_array( data%H%type, inform%status,                    &
             inform%alloc_status, array_name = array_name,                     &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  ensure that the structure will be re-initialised on any subsequent call

      data%new_structure = .TRUE.
      RETURN

!  End of subroutine BPMPD_terminate

      END SUBROUTINE BPMPD_terminate

!  G A L A H A D -  B P M P D _ f u l l _ t e r m i n a t e  S U B R O U T I N E

     SUBROUTINE BPMPD_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( BPMPD_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( BPMPD_control_type ), INTENT( IN ) :: control
     TYPE ( BPMPD_inform_type ), INTENT( INOUT ) :: inform

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     CHARACTER ( LEN = 80 ) :: array_name

!  deallocate workspace

     CALL BPMPD_terminate( data%bpmpd_data, control, inform )

!  deallocate any internal problem arrays

     array_name = 'cqp: data%prob%X'
     CALL SPACE_dealloc_array( data%prob%X,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%X_l'
     CALL SPACE_dealloc_array( data%prob%X_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%X_u'
     CALL SPACE_dealloc_array( data%prob%X_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%G'
     CALL SPACE_dealloc_array( data%prob%G,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%Y'
     CALL SPACE_dealloc_array( data%prob%Y,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%Z'
     CALL SPACE_dealloc_array( data%prob%Z,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%C'
     CALL SPACE_dealloc_array( data%prob%C,                                    &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%C_l'
     CALL SPACE_dealloc_array( data%prob%C_l,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%C_u'
     CALL SPACE_dealloc_array( data%prob%C_u,                                  &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%WEIGHT'
     CALL SPACE_dealloc_array( data%prob%WEIGHT,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%type'
     CALL SPACE_dealloc_array( data%prob%H%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%ptr'
     CALL SPACE_dealloc_array( data%prob%H%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%row'
     CALL SPACE_dealloc_array( data%prob%H%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%col'
     CALL SPACE_dealloc_array( data%prob%H%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%H%val'
     CALL SPACE_dealloc_array( data%prob%H%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%type'
     CALL SPACE_dealloc_array( data%prob%A%type,                               &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%ptr'
     CALL SPACE_dealloc_array( data%prob%A%ptr,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%row'
     CALL SPACE_dealloc_array( data%prob%A%row,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%col'
     CALL SPACE_dealloc_array( data%prob%A%col,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     array_name = 'cqp: data%prob%A%val'
     CALL SPACE_dealloc_array( data%prob%A%val,                                &
        inform%status, inform%alloc_status, array_name = array_name,           &
        bad_alloc = inform%bad_alloc, out = control%error )
     IF ( control%deallocate_error_fatal .AND. inform%status /= 0 ) RETURN

     RETURN

!  End of subroutine BPMPD_full_terminate

     END SUBROUTINE BPMPD_full_terminate

!  End of module BPMPD

   END MODULE GALAHAD_BPMPD_precision
