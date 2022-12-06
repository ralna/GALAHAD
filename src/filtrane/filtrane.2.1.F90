! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*    GALAHAD FILTRANE  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Philippe Toint
!
!  History -
!   originally released with GALAHAD Version 1.4. June  9th 2003
!   update released with GALAHAD Version 2.0. May 2nd 2006
!
!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html
!
!              **************************************************
!              *                                                *
!              *                   FILTRANE                     *
!              *                                                *
!              *     GALAHAD filter trust-region algorithm      *
!              *                                                *
!              *            for nonlinear equations             *
!              *                                                *
!              *          and nonlinear least-squares           *
!              *                                                *
!              **************************************************
!
!  Given a set of smooth equality and inequalitiy (possibly nonlinear or
!  nonconvex) constraints, FILTRANE attempts to find a point that solves
!  the nonlinear least-squares problem
!
!                       min 0.5 * || theta(x) ||^2,
!                        x
!
!  where theta(x) is the vector of constraints' violations at x and ||.|| the
!  Euclidean norm.
!
!  This module implements a combined filter and trust-region algorithm.
!  The filter is used as a tool for potentially accepting new iterates that
!  would otherwise be rejected by the more restrictive monotone trust-region
!  algorithm. The main original feature of the method is that it uses a
!  multi-dimensional sign unrestricted filter combined with prefiltering
!  techniques.
!
!  The method is described in
!
!  N. I. M. Gould, S. Leyffer and Ph. L. Toint,
!  "A Filter Algorithm for Nonlinear Equations and Nonlinear Least-Squares",
!  SIAM J. Optimization 15(1) 17-38 (2005)
!
!  and, in more details, in
!
!  N. I. M. Gould and Ph. L. Toint,
!  "FILTRANE, a Fortran 95 filter-trust-region package for solving nonlinear
!  feasibility problems",
!  Technical Report 03/17, Department of Mathematics, University of Namur,
!  Belgium.
!
!-------------------------------------------------------------------------------
!
!              T h e   F I L T R A N E   r o u t i n e s
!
!-------------------------------------------------------------------------------
!
!  The package consists in 4 routines:
!
!  FILTRANE_initialize :
!
!      assigns default values to the internal data type that controls the
!      algorithmic options of FILTRANE ( the control type described below)
!      and also initializes some global variables.
!
!  FILTRANE_read_specfile :
!
!      reads the specification file (see below), and possibly alters the
!      value of some algorithmic parameters.
!
!  FILTRANE_solve :
!
!      performs the effective numerical problem treatment by applying a
!      combined filter and trust-region method.
!
!  FILTRANE_terminate :
!
!      deallocates the FILTRANE internal data structures.
!
!  These routines must be called by the user in the following order:
!
!      ------------       ---------------       -------       -----------
!     | INITIALIZE | --> | READ_SPECFILE | --> | SOLVE | --> | TERMINATE |
!      ------------       ---------------       -------       -----------
!
!  the call to FILTRANE_read_specfile being optional.
!
!-------------------------------------------------------------------------------
!
!              T h e   s p e c i f i c a t i o n   f i l e
!
!-------------------------------------------------------------------------------
!
!  Like most algorithmic packages, FILTRANE features a number of "control
!  parameters", that is of parameters that condition the various algorithmic,
!  printing and other options of the package (They are documented in detail
!  with the FILTRANE_control type below). While the value of these parameters
!  may be changed (for overiding their default values) within the calling code
!  itself, it is often convenient to be able to alter those parameters without
!  recompiling the code.  This is achieved by specifying the desired values
!  for the control parameters in a "specification file" (specfile).
!
!  A specification file consists of a number of "specification commands",
!  each of these being decomposed into
!  - a "keyword", which is a string (in a close-to-natural language) that will
!    be used to identify a control parameter in the specfile, and
!  - an (optional) "value", which is the value to be attributed to the
!    said control parameter.
!
!  A specific algorithmic "control parameter" is associated to each such
!  keyword, and the effect of interpreting the specification file is to assign
!  the value associated with the keyword (in each specification command) to
!  the corresponding algorithmic parameter. The specification file starts with
!  a "BEGIN FILTRANE SPECIFICATIONS" command and ends with an
!  "END FILTRANE SPECIFICATIONS" command.  The syntax of the specfile is
!  defined as follows:
!
!      BEGIN FILTRANE SPECIFICATIONS
!         printout-device                            (integer)
!         error-printout-device                      (integer)
!         print-level                    SILENT|TRACE|ACTION|DETAILS|DEBUG|CRAZY
!         start-printing-at-iteration                (integer)
!         stop-printing-at-iteration                 (integer)
!         residual-accuracy                           (real)
!         gradient-accuracy                           (real)
!         stop-on-preconditioned-gradient-norm       (logical)
!         stop-on-maximum-gradient-norm              (logical)
!         maximum-number-of-iterations               (integer)
!         inequality-penalty-type                     2|3|4
!         model-type                     GAUSS_NEWTON|FULL_NEWTON|AUTOMATIC
!         automatic-model-inertia                    (integer)
!         automatic-model-criterion            BEST_FIT|BEST_REDUCTION
!         maximum-number-of-cg-iterations            (integer)
!         subproblem-accuracy                      ADAPTIVE|FULL
!         minimum-relative-subproblem-accuracy        (real)
!         relative-subproblem-accuracy-power          (real)
!         preconditioner-used                   NONE|BANDED|USER_DEFINED
!         semi-bandwidth-for-band-preconditioner     (integer)
!         external-Jacobian-products                 (logical)
!         equations-grouping                  NONE|AUTOMATIC|USER_DEFINED
!         number-of-groups                           (integer)
!         balance-initial-group-values               (logical)
!         use-filter                           NEVER|INITIAL|ALWAYS
!         filter-sign-restriction                    (logical)
!         maximum-filter-size                        (integer)
!         filter-size-increment                      (integer)
!         filter-margin-type                  CURRENT|FIXED|SMALLEST
!         filter-margin-factor                        (real)
!         remove-dominated-filter-entries            (logical)
!         minimum-weak-acceptance-factor              (real)
!         weak-acceptance-power                       (real)
!         initial-radius                              (real)
!         initial-TR-relaxation-factor                (real)
!         secondary-TR-relaxation-factor              (real)
!         minimum-rho-for-successful-iteration        (real)
!         minimum-rho-for-very-successful-iteration   (real)
!         radius-reduction-factor                     (real)
!         radius-increase-factor                      (real)
!         worst-case-radius-reduction-factor          (real)
!         save-best-point                            (logical)
!         checkpointing-frequency                    (integer)
!         checkpointing-device                       (integer)
!         checkpointing-file                         (string)
!         restart-from-checkpoint                    (logical)
!      END FILTRANE SPECIFICATIONS
!
!  where the | symbols means "or".  Thus print-level may take the values
!  SILENT or TRACE or ACTION or DETAILS or DEBUG or CRAZY, where the upper
!  case words are symbols that are recognized by FILTRANE. Empty values are
!  acceptable for logical switches, that is switches with ON|OFF|TRUE|FALSE|T|F
!  values, and are interpreted in the same manner as "ON", "TRUE" ot "T".
!  Note that the specification commands are case insensitive.
!
!  Furthermore, the specification command lines (between the BEGIN and
!  END delimiters) may be specified in any order. Blank lines and lines whose
!  first non-blank character is ! are ignored. The content of a line
!  after a ! or a * character is also ignored (as is the ! or * character
!  itself). This provides an easy manner to "comment off" some specification
!  commands or to add comments associated to specific values of certain control
!  parameters.
!
!  The specification file must be open for input when FILTRANE_read_specfile
!  is called. Note that the corresponding file is first REWINDed, which make
!  it possible to combine it with other specifications associated with other
!  algorithms used in conjunction with FILTRANE. For the same reason, the
!  specification file is not closed by FILTRANE.
!
!-------------------------------------------------------------------------------
!
!   T h e   i n t e r f a c e   w i t h   t h e   p r o b l e m ' s   d a t a
!
!-------------------------------------------------------------------------------
!
!-------------------------------------------------------------------------------
!                               INPUT
!-------------------------------------------------------------------------------
!
!  The algorithm used in FILTRANE_solve requires the user to provide
!  information about the problem using the NLPT_problem_type, as described in
!  the NLPT module.
!
!  IN ALL CASES,
!  -------------
!
!  the following components of this data structure should be set to
!  approriate values on first input in FILTRANE_solve:
!
!  - problem%n                       [INTEGER, INTENT( IN )]:
!         the number of variables in the problem,
!  - problem%m                       [INTEGER, INTENT( IN )]:
!         the number of constraints in the problem,
!  - problem%x( 1:problem%n )        [REAL( KIND = wp ), INTENT( INOUT )]:
!         the values of the variables a the starting point.
!
!  In addition, the vector
!
!  - problem%g( 1:problem%n )
!
!  should be allocated (it needs not being assigned a value). It is
!  used as workspace by the package.
!
! WHEN BOUNDS ON THE VARIABLES ARE PRESENT,
! -----------------------------------------
!
! the following components should be also set to approriate values on first
! input in FILTRANE_solve:
!
!  - problem%x_l( 1:problem%n)       [REAL( KIND = wp ), INTENT( IN )]:
!         the vector of lower bounds on the problem's variables,
!  - problem%x_u( 1:problem%n )      [REAL( KIND = wp ), INTENT( IN )]:
!         the vector of upper bounds on the problem's variables.
!
!  In addition, the vector
!
!  - problem%x_status( 1:problem%n )
!
!  should be allocated (it needs not being assigned a value). It is
!  used as workspace by the package.
!
! WHEN CONSTRAINTS ARE PRESENT (problem%m > 0),
! ---------------------------------------------
!
! the following components should be also set to approriate values on first
! input in FILTRANE_solve:
!
!  - problem%c_l( 1:problem%m )      [REAL( KIND = wp ), INTENT( IN )]:
!         the vector of lower bounds on the problem's constriants,
!  - problem%c_u( 1:problem%m )      [REAL( KIND = wp ), INTENT( IN )]:
!         the vector of upper bounds on the problem's constraints.
!
!  In addition, the vectors
!
!  - problem%c( 1:problem%m )
!  - problem%y( 1:problem%m )
!  - problem%equation( 1:problem%m )
!
!  should be allocated (they need not being assigned a value). These are
!  used as workspace by the package.
!
! *** NOTE *** THE PROBLEM ONLY MAKES SENSE IF BOUNDS OR CONSTRAINTS (OR BOTH)
!              ARE PRESENT IN THE PROBLEM!
!
! WHEN EXTERNAL JACOBIAN PRODUCTS ARE NOT REQUIRED (the default),
! ---------------------------------------------------------------
!
! the following components should be also set to approriate values on first
! input in FILTRANE_solve:
!
!  - problem%J_size                  [INTEGER, INTENT( IN )]:
!         the number of nonzero entries in the constraints' Jacobian,
!  - problem%J_type                  [INTEGER, INTENT( IN )]:
!         the type of matrix storage used for the Jacobian.  It must
!         be set to the value of the COORDINATE symbol (i.e. the integer 1).
!
!  In addition, the vectors
!
!  - problem%J_val( 1:problem%Jsize+problem%n )
!  - problem%J_col( 1:problem%Jsize+problem%n )
!  - problem%J_row( 1:problem%Jsize+problem%n )
!
!  should be allocated (they need not being assigned a value). These are
!  used as workspace by the package.
!
!-------------------------------------------------------------------------------
!                         DURING EXECUTION
!-------------------------------------------------------------------------------
!
!  The package uses a reverse communication interface.  This means that
!  control is passed back to the user, with an indication of the task
!  required and the values of the necessary arguments; the package should
!  then be called again after the desired computations have been performed.
!  These indications as well as the description of the arguments and results
!  are fully detailed in the description of the inform%status below.
!
!  In particular, FILTRANE_solve requires, at each iteration, the computation,
!  for a given x, of
!
!  - new constraint values,
!  - (possibly) Jacobian values, or product of the Jacobian (or its transpose)
!    with given vectors,
!  - (possibly) the product
!
!       H(x,y) * w     where    H(x,y) = SUM y_i \nabla_{xx} c_i(x)
!
!    (for given values of y) whenever the AUTOMATIC model choice is
!    selected (this is the default) or the NEWTON type model is selected,
!  - (when a user-defined preconditioner is requested) the multiplication of
!    a given vector with a preconditioner for the matrix J^T(x) J(x),
!    where J(x) is the Jacobian of the constraints at x.
!
!-------------------------------------------------------------------------------
!                               OUTPUT
!-------------------------------------------------------------------------------
!
!  On output of FILTRANE_solve (successful termination, i.e. inform%status = 0),
!  the following components of the problem structure are of interest (when
!  applicable):
!
!  - problem%x : the values of the variables at the solution found;
!  - problem%c : the values of the constraints at the solution found;
!  - problem%f : the value of the merit function at the solution found;
!  - problem%g : the gradient of the merit function at the solution found;
!  - problem%J_val : the values of the nonzero entries of the constraints'
!                Jacobian at the solution found,
!  - problem%J_row : the row indices of the nonzero entries of the
!                constraints' Jacobian at the solution found,
!  - problem%J_col : the column indices of the nonzero entries of the
!                constraints' Jacobian at the solution found,
!
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   MODULE GALAHAD_FILTRANE_double

!-------------------------------------------------------------------------------
!   U s e d   m o d u l e s   a n d   s y m b o l s
!-------------------------------------------------------------------------------

      USE LANCELOT_BAND_double    ! band factorization and solve

      USE GALAHAD_NLPT_double     ! the problem type

      USE GALAHAD_SPECFILE_double ! specfile manipulations

      USE GALAHAD_NORMS_double    ! norm functions

      USE GALAHAD_TOOLS           ! the GALAHAD toolbox

      USE GALAHAD_SORT_double     ! the sorting procedures

      USE GALAHAD_GLTR_double     ! the GLTR truncated CG procedure

      USE GALAHAD_COPYRIGHT       ! copyright statement

!     Print levels

      USE GALAHAD_SYMBOLS,                                                     &
          SILENT                      => GALAHAD_SILENT,                       &
          TRACE                       => GALAHAD_TRACE,                        &
          ACTION                      => GALAHAD_ACTION,                       &
          DETAILS                     => GALAHAD_DETAILS,                      &
          DEBUG                       => GALAHAD_DEBUG,                        &
          CRAZY                       => GALAHAD_CRAZY

!     Exit conditions

      USE GALAHAD_SYMBOLS,                                                     &
          OK                          => GALAHAD_SUCCESS,                      &
          MEMORY_FULL                 => GALAHAD_MEMORY_FULL,                  &
          FILE_NOT_OPENED             => GALAHAD_FILE_NOT_OPENED,              &
          COULD_NOT_WRITE             => GALAHAD_COULD_NOT_WRITE,              &
          MAX_ITERATIONS_REACHED      => GALAHAD_MAX_ITERATIONS_REACHED,       &
          PROGRESS_IMPOSSIBLE         => GALAHAD_PROGRESS_IMPOSSIBLE,          &
          NOT_INITIALIZED             => GALAHAD_NOT_INITIALIZED,              &
          WRONG_N                     => GALAHAD_WRONG_N,                      &
          WRONG_M                     => GALAHAD_WRONG_M

!     Misc

      USE GALAHAD_SYMBOLS,                                                     &
          AUTOMATIC                   => GALAHAD_AUTOMATIC,                    &
          USER_DEFINED                => GALAHAD_USER_DEFINED,                 &
          UNDEFINED                   => GALAHAD_UNDEFINED

!     Filter usage

      USE GALAHAD_SYMBOLS,                                                     &
          NEVER                       => GALAHAD_NEVER,                        &
          INITIAL                     => GALAHAD_INITIAL,                      &
          ALWAYS                      => GALAHAD_ALWAYS

!     Filter margin types

      USE GALAHAD_SYMBOLS,                                                     &
          CURRENT                     => GALAHAD_CURRENT,                      &
          SMALLEST                    => GALAHAD_SMALLEST

!     Preconditioners

      USE GALAHAD_SYMBOLS,                                                     &
          NONE                        => GALAHAD_NONE,                         &
          BANDED                      => GALAHAD_BANDED

!     Subproblem accuracy

      USE GALAHAD_SYMBOLS,                                                     &
          ADAPTIVE                    => GALAHAD_ADAPTIVE,                     &
          FULL                        => GALAHAD_FULL

!     Variable status

      USE GALAHAD_SYMBOLS,                                                     &
          FIXED                       => GALAHAD_FIXED,                        &
          FREE                        => GALAHAD_FREE,                         &
          LOWER                       => GALAHAD_LOWER,                        &
          UPPER                       => GALAHAD_UPPER,                        &
          RANGE                       => GALAHAD_RANGE

!     Model types

      USE GALAHAD_SYMBOLS,                                                     &
          NEWTON                      => GALAHAD_NEWTON,                       &
          GAUSS_NEWTON                => GALAHAD_GAUSS_NEWTON

!     Automatic model criteria

      USE GALAHAD_SYMBOLS,                                                     &
          BEST_FIT                    => GALAHAD_BEST_FIT,                     &
          BEST_REDUCTION              => GALAHAD_BEST_REDUCTION

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

!     Make everything private by default

      PRIVATE

!     Ensure the PRIVATE nature of the imported symbols.

      PRIVATE :: SILENT, TRACE, ACTION, DETAILS, DEBUG, CRAZY,                 &
                 OK, MEMORY_FULL, AUTOMATIC, USER_DEFINED, NONE, BANDED,       &
                 MAX_ITERATIONS_REACHED, PROGRESS_IMPOSSIBLE,                  &
                 NOT_INITIALIZED, NEVER, INITIAL, ALWAYS, NEWTON, GAUSS_NEWTON,&
                 ADAPTIVE, FULL, COULD_NOT_WRITE,                              &
                 WRONG_N, WRONG_M, LOWER, UPPER, FREE, RANGE, CURRENT,         &
                 SMALLEST, BEST_FIT, BEST_REDUCTION, FILE_NOT_OPENED


!     Make the FILTRANE calls public

      PUBLIC :: FILTRANE_initialize, FILTRANE_read_specfile, FILTRANE_solve,   &
                FILTRANE_terminate

!-------------------------------------------------------------------------------
!   P r e c i s i o n
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PRIVATE, PARAMETER :: wp = dp

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!                      PUBLIC TYPES AND DEFINITIONS
!
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!     NOTE:  None of these definitions should be altered by the user!!!
!
!-------------------------------------------------------------------------------
!              The structure that controls the FILTRANE execution
!-------------------------------------------------------------------------------

!     NOTE:  The default values for the components of this structure are set
!            by executing FILTRANE_initialize.  They must therefore
!            be altered to reflect user's need *only after* that execution.

      TYPE, PUBLIC :: FILTRANE_control_type

         REAL ( KIND = wp ) :: c_accuracy       ! INTENT( IN )

!                   Successful termination occurs if each constraint
!                   violation is below this parameter.
!                   Default: 10.**(-6) in double precision,
!                            10.**(-4) in single precision.

         REAL ( KIND = wp ) :: g_accuracy       ! INTENT( IN )

!                   Successful termination occurs if the (possibly
!                   preconditioned) Euclidean norm of the merit function
!                   is smaller that sqrt(n) times this parameter.
!                   Default: 10.**(-6) in double precision,
!                            10.**(-4) in single precision.

         LOGICAL :: stop_on_prec_g              ! INTENT( IN )

!                   .TRUE. iff the norm of the preconditioned gradient
!                   must be considered for stopping the iteration, .FALSE.
!                   if the Euclidean norm must be used.
!                   Default: .TRUE.

         LOGICAL :: stop_on_g_max               ! INTENT( IN )

!                   .TRUE. iff the norm of the maximum norm of the gradient
!                   must be considered for stopping the iteration, .FALSE.
!                   if the Euclidean norm must be used. Note that stopping
!                   on the max norm is only possible for the unpreconditioned
!                   and the diagonally preconditioned cases. In all other
!                   cases, this option is overridden (set to .FALSE.).
!                   Default: .FALSE.

         INTEGER :: max_iterations              ! INTENT( IN )

!                   The maximum number of iterations allowed before
!                   giving up. If negative, this stopping rule is inactive.
!                   Default: 1000

         INTEGER :: max_cg_iterations           ! INTENT( IN )

!                   The mutiple of problem%n iterations that are allowed at
!                   each call to the conjugate-gradients procedure.
!                   Default: 15

         INTEGER :: grouping          ! INTENT( IN )

!                   The type of equations/inequalities/bounds groups required
!                   for the FILTRANE algorithm. Valid values are:
!                   - NONE         : each equation is considered individually,
!                   - AUTOMATIC    : use the automatic grouping strategy
!                                    provided by the package
!                   - USER_DEFINED : use the groups defined by the user
!                                    and specified in control%group.
!                   Default: NONE

         INTEGER :: nbr_groups       ! INTENT( IN )

!                   If AUTOMATIC grouping is used:
!                      If positive, the smallest between this value and the
!                      problem's number of variables is used as the number of
!                      groups. If negative, this number of groups is set to
!                      the number of items divided by the absolute value of the
!                      parameter.
!                      Default: 10
!                   If USER_DEFINED grouping is used:
!                      The number of groups in the user-supplied grouping
!                      defined by the vector control%group.
!                      Default : none.

         INTEGER, POINTER, DIMENSION( : ) :: group   ! INTENT( IN )

!                   group( i ) contains the index of the group to which
!                   the i-th constrained item is assigned. A constrained
!                   item is either one constraint (equality or inequality)
!                   or a variable which is lower/upper/range bounded.
!                   The groups of the equalities and inequalities
!                   are specified in components 1 to problem%m of group.
!                   Components problem%m + 1 and above specify the groups
!                   of the bounded variables (variables are considered by
!                   increasing index and unbounded variables are skipped).
!                   This vector is only requested in USER_DEFINED grouping
!                   is used. control%group is nullified by FILTRANE if another
!                   grouping strategy is used.
!                   Default : no default value available.

         LOGICAL :: balance_group_values        ! INTENT( IN )

!                   .TRUE. if the constraints values (at the initial point)
!                   must be sorted before they are distributed into groups.
!                   This has the effect of approximately balancing the
!                   constraint violations (at the initial point) between
!                   the groups. It is only relevant if control%grouping
!                   is set to AUTOMATIC.
!                   Default: .FALSE.

         INTEGER :: prec_used                   ! INTENT( IN )

!                   The type of preconditioner used, if any. Recognized
!                   values are:
!                   - NONE       : no preconditioner (i.e. use the identity),
!                   - BANDED     : band preconditioner used, where the band is
!                                  extracted from the objective function's
!                                  Hessian.
!                   Default : NONE

         INTEGER :: semi_bandwidth              ! INTENT( IN )

!                   Semi-bandwidth of the band preconditioner, if used.
!                   Default : 5


         LOGICAL :: external_J_products         ! INTENT( IN )

!                   .TRUE. iff the products by the Jacobian and its transpose
!                   are supplied by the user via the reverse communication
!                   interface. In this case, the Jacobian is not stored at all
!                   in FILTRANE. On the other hand, the memory requirements
!                   for this option increase by MAX( problem%n, problem%m )
!                   reals.
!                   Note that this option also requires the preconditioner
!                   choice to be either NONE (no preconditioning) or
!                   USER_DEFINED, as the package does not know the values
!                   of the Jacobian and cannot build a banded preconditioner.
!                   Default: .FALSE.

         INTEGER :: out                         ! INTENT( IN )

!                   The unit number associated with the device used for
!                   printout.
!                   Default: 6

         INTEGER :: errout                      ! INTENT( IN )

!                   The unit number associated with the device used for
!                   error ouput.
!                   Default: 6

         INTEGER :: print_level                 ! INTENT( IN )

!                   The level of printout requested by the user. Can take
!                   the values:
!                   - SILENT  : no printout is produced,
!                   - TRACE   : only reports the major steps in the optimization
!                   - ACTION  : reports some detail,
!                   - DETAILS : reports more details (including vector values),
!                   - DEBUG   : reports LOTS of information,
!                   - CRAZY   : reports a completely silly amount of information
!                   Default: SILENT

         INTEGER :: start_print

!                   The first iteration at which printing is to occur.  If
!                   non-positive, printing starts immediately.
!                   Default: 0

         INTEGER :: stop_print

!                   The last iteration at which printing is to occur.  If
!                   negative, printing never stops once started.
!                   Default: -1

         INTEGER :: model_type                 ! INTENT( IN )

!                   The type of model to use. Possible values are
!                   AUTOMATIC, GAUSS_NEWTON and NEWTON. The first choice
!                   corresponds to choosing the model whose prediction was
!                   most often the most accurate over the past model_inertia
!                   iterations (see below).
!                   Default: AUTOMATIC

         INTEGER :: model_inertia

!                   The number of successive iterations during which the same
!                   model is being used in AUTOMATIC model type mode.
!                   Must be >= 1.
!                   Default: 5

         INTEGER :: model_criterion            ! INTENT( IN )

!                   The criterion that is used to choose an AUTOMATIC model,
!                   when this mode is enabled. Possible values are:
!                   BEST_FIT       : a model is prefered if it provides the best
!                                    fit of the objective function values;
!                   BEST_REDUCTION : a model is prefered if it provides the best
!                                    reduction in objective function values.
!                   Default: BEST_FIT

         INTEGER :: inequality_penalty_type

!                   Defines the type of penalty function to apply on the
!                   inequalities:
!                   2 : the L2 penalty function,
!                   3 : the L3 penalty function,
!                   4 : the L4 penalty function.
!                   Default: 2

         INTEGER :: subproblem_accuracy        ! INTENT( IN )

!                   The type of accuracy required in the solution of the
!                   linear subproblem.  Possibles values are:
!                   - ADAPTIVE : the residual norm at the trial point should be
!                                at most
!
!                                 MIN( control%min_gltr_accuracy,
!                                      ||r0||**gltr_accuracy_power   ),
!
!                                where ||r0|| is the norm of the residual at
!                                the beginning of the subproblem computation;
!                   - FULL     : the residual norm at the trial point should be
!                                at most one hundred times machine precision.
!                   Default: ADAPTIVE

         REAL ( KIND = wp ) :: min_gltr_accuracy

!                   The minimum relative gradient decrease that is used in
!                   the stopping rule for GLTR with adaptive accuracy. This
!                   parameter must be strictly between zero and one.
!                   Default: 0.01

         REAL ( KIND = wp ) :: gltr_accuracy_power

!                   The power at which the gradient norm must be raised in
!                   the stopping rule for GLTR with adaptive accuracy. It must
!                   be positive. Note that this power does not include the
!                   gradient norm used to make the test relative, so that
!                   choosing gltr_accuracy_power = 0 results in a requested
!                   decrease of the residual norm by min_gltr_accuracy.
!                   Default: 1.0

         REAL ( KIND = wp ) :: initial_radius   ! INTENT( IN )

!                   The initial trust-region radius. Note that this value is
!                   not used to limit the length of the first step if
!                   control%use_filter = .TRUE.
!                   Default: 1.0

         REAL( KIND = wp )  :: min_weak_accept_factor  ! INTENT( IN )

!                   The minimum value of the achieved objective reduction
!                   for the weak acceptance test, i.e. in the test
!                   that declares the trial point acceptable iff the decrease in
!                   constraint violation is larger that min_weak_accept_factor
!                   times the power weak_accept_power of this violation.
!                   This test is set to allow more iterates to be accepted,
!                   at the risk of slower convergence. It is only relevant
!                   if weak_accept_power >= 0, in which case it must be
!                   strictly positive.
!                   Default: 0.1

         REAL( KIND = wp )  :: weak_accept_power  ! INTENT( IN )

!                   The power at which the current fonction value is to be
!                   raised for the weak acceptance test, i.e. in the test
!                   that declares the trial point acceptable iff the decrease in
!                   constraint violation is larger that min_weak_accept_factor
!                   times the power weak_accept_power of this violation.
!                   This test is set to allow more iterates to be accepted,
!                   at the risk of slower convergence. If weak_accept_power < 0,
!                   then the trial point is always declared unacceptable
!                   for this test.
!                   Default: -1.0


         REAL( KIND = wp ) :: eta_1            ! INTENT( IN )

!                   The minimum ratio of achieved to predicted reduction
!                   for declaring the iteration successful.
!                   Default: 0.01

         REAL( KIND = wp ) :: eta_2            ! INTENT( IN )

!                   The minimum ratio of achieved to predicted reduction
!                   for declaring the iteration very successful.
!                   Default: 0.9

         REAL( KIND = wp ) :: gamma_0          ! INTENT( IN )

!                   The strongest factor by which the trust-region radius
!                   is decreased when the iteration is very unsuccessful.
!                   Default: 0.0625

         REAL( KIND = wp ) :: gamma_1          ! INTENT( IN )

!                   The factor by which the trust-region radius is decreased
!                   when the iteration is unsuccessful.
!                   Default: 0.25

         REAL( KIND = wp ) :: gamma_2          ! INTENT( IN )

!                   The factor by which the trust-region radius is increased
!                   when the iteration is very successful.
!                   Default: 2.0

         INTEGER :: use_filter                 ! INTENT( IN )

!                   Indicates at what stages of the algorithm the filter must
!                   be used to test acceptability of new iterates.  Possible
!                   values are:
!                   NEVER   : the filter is not used (pure trust-region method),
!                   INITIAL : the filter is used as long as it is successful,
!                             but never after it first fails,
!                   ALWAYS  : the filter is used at all iterations.
!                   Default: ALWAYS

         LOGICAL :: filter_sign_restriction    ! INTENT( IN )

!                   .TRUE. if the filter is constructed by looking at the
!                   absolute value of the constraint/bound violations,
!                   .FALSE. if the violation have to be considered with their
!                   sign. Note that each group containing more than a single
!                   constraint/inequality/bound is always considered in
!                   absolute value.
!                   Default: .FALSE.

         INTEGER :: maximal_filter_size        ! INTENT( IN )

!                   The maximum number of points that the filter can
!                   hold. Once this maximum is attained, no further point can
!                   be acceptable for the filter and the algorithm reduces
!                   to a pure trust-region scheme.  If negative, no upper
!                   limit is set on the number of filter entries.
!                   Default: -1

         INTEGER :: filter_size_increment      ! INTENT( IN )

!                   The number of theta values that is used as the size of
!                   the initial filter (if used), and also as an increment
!                   for the case where the filter capacity must be extended.
!                   Default: 50

         LOGICAL :: remove_dominated           ! INTENT( IN )

!                   .TRUE. iff the algorithm must remove the dominated
!                   filter entries.  IF .FALSE. the dominated entries
!                   are never removed, the acceptance test is a bit slower
!                   and the inclusion of a new filter point a bit faster.
!                   Default: .TRUE.

         INTEGER :: margin_type                ! INTENT( IN )

!                   This parameter specifies the quantity that is used to
!                   determine the width of the margin. Possible values are
!                   CURRENT  : the norm of the violations at the current trial
!                              point is used;
!                   FIXED    : the norm of the violation at the filter point
!                              itself is used;
!                   SMALLEST : the smaller of the two preceding values is
!                              used.
!                   Default: FIXED

         REAL ( KIND = wp ) :: gamma_f

!                   The filter margin is defined as the minimum between this
!                   constant and 1/(2 * sqrt( p ) ).
!                   Default: 0.001

         REAL ( KIND = wp ) :: itr_relax

!                   Initial Trust Region relaxation factor, i.e. the factor
!                   by which the trust region is enlarged on unrestricted
!                   steps, until a first restricted step occurs. It must
!                   be at least 1.0.
!                   Default: 10**20


         REAL ( KIND = wp ) :: str_relax

!                   Secondary Trust Region relaxation factor, i.e. the factor
!                   by which the trust region is enlarged on unrestricted
!                   steps, after a first restricted step has occured. It must
!                   be at least 1.0.
!                   Default: 1000.0

         LOGICAL :: save_best_point            ! INTENT( IN )

!                   .TRUE. if the best point found so far must be saved to be
!                   returned as the final iterate.  This may be useful when
!                   the filter is used (as this is a non-monotone method) and
!                   the problem very hard, but requires the storage of an
!                   additional vector of dimension problem%n.
!                   Default: .FALSE.

         INTEGER :: checkpoint_freq            ! INTENT( IN )

!                   The frequency (expressed in number of iterations) at which
!                   problem%x and problem%c are saved on a checkpointing
!                   file for a possible package restart. It must be
!                   non-negative.
!                   Default: 0 (no checkpointing)

         CHARACTER( LEN = 30 ) :: checkpoint_file ! INTENT( IN )

!                   The name of the file use for storing checkpointing
!                   information on disk.
!                   Default: FILTRANE.sav

         INTEGER :: checkpoint_dev            ! INTENT( IN )

!                   The device number to be used for opening the checkpointing
!                   file (on input or output).
!                   Default: 55

         LOGICAL :: restart_from_checkpoint   ! INTENT( IN )

!                   .TRUE. if the initial point and constraints values must
!                   be read from the checkpointing file control%checkpoint_file,
!                   overriding the value of problem%x specified on input.
!                   Default: .FALSE.

      END TYPE

!   Note that the default double precision settings correspond to the following
!   specfile, which is optionnaly read by FILTRANE.

!      BEGIN FILTRANE SPECIFICATIONS (DEFAULT)
!         printout-device                              6
!         error-printout-device                        6
!         print-level                                  SILENT
!         start-printing-at-iteration                  0
!         stop-printing-at-iteration                   -1
!         residual-accuracy                            1.0D-6
!         gradient-accuracy                            1.0D-6
!         stop-on-preconditioned-gradient-norm         YES
!         stop-on-maximum-gradient-norm                NO
!         maximum-number-of-iterations                 1000
!         inequality-penalty-type                      2
!         model-type                                   AUTOMATIC
!         automatic-model-inertia                      3
!         automatic-model-criterion                    BEST_FIT
!         maximum-number-of-cg-iterations              15
!         subproblem-accuracy                          ADAPTIVE
!         minimum-relative-subproblem-accuracy         0.01
!         relative-subproblem-accuracy-power           1.0
!         preconditioner-used                          NONE
!         semi-bandwidth-for-band-preconditioner       5
!         external-Jacobian-products                   NO
!         equations-grouping                           NONE
!         number-of-groups                             10
!         balance-initial-group-values                 NO
!         use-filter                                   ALWAYS
!         filter-sign-restriction                      NO
!         maximum-filter-size                          1000
!         filter-size-increment                        50
!         filter-margin-type                           FIXED
!         filter-margin-factor                         0.001
!         remove-dominated-filter-entries              YES
!         minimum-weak-acceptance-factor               0.1
!         weak-acceptance-power                        2.0
!         initial-radius                               1.0
!         minimum-rho-for-successful-iteration         0.01
!         minimum-rho-for-very-successful-iteration    0.9
!         radius-reduction-factor                      0.25
!         radius-increase-factor                       2.0
!         worst-case-radius-reduction-factor           0.0625
!         initial-TR-relaxation-factor                 10.0**20
!         secondary-TR-relaxation-factor               1000.0
!         save-best-point                              NO
!         checkpointing-frequency                      0
!         checkpointing-device                         55
!         checkpointing-file                           FILTRANE.chk
!         restart-from-checkpoint                      NO
!      END FILTRANE SPECIFICATIONS

!-------------------------------------------------------------------------------
!         The structure that returns information to the user
!-------------------------------------------------------------------------------

      TYPE, PUBLIC :: FILTRANE_inform_type

         INTEGER :: status                      ! INTENT( OUT )

!                   The FILTRANE exit condition.  It can take the following
!                   values:
!
!                    0 (OK)
!
!                        initial entry and successful exit;
!
!                    1 (GET_C_AND_J_0)
!
!                       FILTRANE_solve requires the user to compute
!                       the values, at the point problem%x, of
!                       - the constraint functions (in problem%c),
!                       - the nonzero entries of their Jacobian matrix
!                         (in problem%J_val),
!                       - the column indices of these entries
!                         (in problem%J_col),
!                       - the row indices of these entries
!                         (in problem%J_row);
!                       No other argument of FILTRANE may be modified
!                       before FILTRANE_solve is called again.
!                       This is only used at the initial (starting) point.
!
!                    2  (GET_C_AND_J_F)
!
!                       FILTRANE_solve requires the user to compute
!                       the values, at the point problem%x, of
!                       - the constraint functions (in problem%c),
!                       - the nonzero entries of their Jacobian matrix
!                         (in problem%J_val),
!                       No other argument of FILTRANE may be modified
!                       before FILTRANE_solve is called again.
!
!                    3, 4 and 5 (GET_C_0, GET_C and GET_C_F)
!
!                       FILTRANE_solve requires the user to compute the
!                       values of the constraints at the current point
!                       (problem%x). The result is to be set in the vector
!                       problem%c. No other argument of FILTRANE may be
!                       modified before FILTRANE_solve is called again.
!
!                    6 (GET_J_A)
!
!                       FILTRANE_solve requires the user to compute the
!                       values of the entries of the constraints Jacobian
!                       matrix at the current point (problem%x). The result
!                       is to be set in the vector problem%J_val
!                       No other argument of FILTRANE may be modified before
!                       FILTRANE_solve is called again. This is only used when
!                       extrenal Jacobian products are NOT resqueted (the
!                       default)
!
!                    7 (GET_JV)
!
!                       FILTRANE require the user to compute the product
!
!                          s%RC_Mv = J( problem%x) s%RC_v
!
!                       where J(problem%x) is the Jacobian of the constraints
!                       evaluated at the current point problem%x.
!                       This is only used if external Jacobian products are
!                       specifically requested (this is not the default).
!                       Note that only the vector targeted by s%RC_Mv may
!                       be modified by the user before  FILTRANE_solve is
!                       re-entered.
!
!                   8, 9, 10 and 11 (GET_JTC_0, GET_JTV, GET_JTC_A, GET_JTC_F)
!
!                       FILTRANE require the user to compute the product
!
!                          s%RC_Mv = J( problem%x)^T s%RC_v
!
!                       where J(problem%x) is the Jacobian of the constraints
!                       evaluated at the current point problem%x.
!                       This is only used if external Jacobian products are
!                       specifically requested (this is not the default).
!                       Note that only the vector targeted by s%RC_Mv may be
!                       modified by the user before FILTRANE_solve is
!                       re-entered.
!
!                   12, 13 and 14 (GET_PREC_G_0, GET_PREC and GET_PREC_G_A)
!
!                       FILTRANE requires the user to apply thus user-defined
!                       preconditioner for the model's Hessian matrix to the
!                       vector specified by s%RC_Pv and return the result in
!                       the same vector s%RC_Pv. Note that only the vector
!                       targeted by s%RC_Pv may be modified by the user before
!                       FILTRANE_solve is re-entered.
!                       This case only occurs if the USER_DEFINED preconditioner
!                       has been selected. It is most useful when there are no
!                       inequality/bound constraints and the GAUSS_NEWTON
!                       model is used, because the model's Hessian then
!                       reduces to J^TJ.
!
!                   15 and 16 (GET_MP1 and GET_MP2)
!
!                       FILTRANE_solve requires the user to compute the
!                       product
!
!                           s%RC_Mv = ( H( problem%x, problem%y ) ) * s%RC_v
!
!                       where, as above, the matrix H(x,y) is defined by
!
!                                  m
!                        H(x,y) = SUM problem%y_i \nabla_{xx} c_i( problem%x ).
!                                 i=1
!
!                       In addition, the logical variables s%RC_newx is
!                       has the value .TRUE. iff the current product request is
!                       the first that involves the Hessians of the constraints
!                       at problem%x. Note that only the vector targeted by
!                       s%RC_Mv may be modified by the user before
!                       FILTRANE_solve is re-entered.
!
!                   -1 (MEMORY_FULL)
!
!                        memory allocation failed;
!
!                   -2 (FILE_NOT_OPENED)      :
!
!                        the checkpointing file could not be opened;
!
!                   -3 (COULD_NOT_WRITE)      :
!
!                        an IO error occurred while saving variables and value
!                        on the checkpoiting file;
!
!                   -5 (PROGRESS_IMPOSSIBLE)
!
!                       the step is too short to allow for futher progress.
!                       This is often caused by ill-conditioning.
!
!                   -6 (MAX_ITERATIONS_REACHED)
!
!                       the maximum number of iterations has been reached,
!                       and computation terminated;
!
!                   -8 (WRONG_N)
!
!                       the number of variables is non positive;
!
!                   -9 (WRONG_M)
!
!                       the number of equations is non positive;
!
!                   -21 (CHECKPOINTING_ERROR)
!
!                       the information contained in the checkpointing file
!                       could not be read or does not correspond to the
!                       problem being solved;
!
!                   -22 (GLTR_ERROR)
!
!                       the step could not be computed by the GLTR procedure;
!
!                   -23 (UNASSOCIATED_INPUT)
!
!                       one of the vectors problem%x, problem%x_l, problem%x_u,
!                       problem%c, problem%c_l, problem%c_u, problem%y,
!                       problem%g, problem%J_val, problem%J_col, problem%J_row
!                       problem%equation is not allocated on input;
!
!                   -24 (SORT_TOO_LONG)        :
!
!                        the vectors are too long for the sorting routine;
!
!                   -25 (WRONG_P)
!
!                       the user-supplied number of groups is either negative
!                       or exceeds the number of constraints plus the number
!                       of bounded variables;
!
!                   -26 (USER_GROUPS_UNDEFINED)
!
!                       the vector control%group is not associated;
!
!                   -27 (WRONG_NUMBER_OF_USER_GROUPS)
!
!                       the dimension of the vector control%group is different
!                       from the total of the number of constraints plus the
!                       number of bounded variables;
!
!                   -28 (WRONG_USER_GROUP_INDEX)
!
!                       the user-supplied group index (for a constraint or a
!                       bound) is either negative, or exceeds
!                       control%nbr_groups;
!
!                   -29 (WRONG_STATUS)
!
!                       FILTRANE was re-entered (in the reverse communication
!                       protocol) with an invalid value for inform%status;
!
!                   -100 (INTERNAL_ERROR_100)
!
!                       this should not happen! (If it does anyway, please
!                       report (with problem data and specfile) to Ph. Toint.
!                       Thanks in advance.)

         INTEGER :: nbr_iterations                        ! INTENT( OUT )

!                   The number of iterations used by the minimization
!                   algorithm.

         INTEGER :: nbr_cg_iterations                     ! INTENT( OUT )

!                   The number of conjugate-gradients iterations used by
!                   the minimization algorithm.

         INTEGER :: nbr_c_evaluations                     ! INTENT( OUT )

!                   The number of evaluations of the residuals used by
!                   the minimization algorithm.

         INTEGER :: nbr_J_evaluations                     ! INTENT( OUT )

!                   The number of evaluations of the Jacobian used by
!                   the minimization algorithm.

         CHARACTER( LEN = 80 ), DIMENSION( 3 ) :: message  ! INTENT( OUT )

!                   A few lines containing a description of the exit condition
!                   on exit of FILTRANE, typically including more information
!                   than indicated in the description of control%status above.
!                   It is printed out on device errout at the end of execution
!                   if control%print_level >= TRACE.

      END TYPE

!-------------------------------------------------------------------------------
!         The structure that saves information between the various calls
!         to FILTRANE
!-------------------------------------------------------------------------------

!     NOTE:  This structure is used for purely internal purposes.  Thus the
!     ====   arguments (of type FILTRANE_data_type) should not be initialized
!            nor modified by the user before or in between calls to FILTRANE
!            (except for s%RC_Mv).

      TYPE, PUBLIC :: FILTRANE_data_type

         INTEGER :: p                   ! the number of equations groups

         INTEGER :: n_items             ! the combined number of equalities,
                                        ! inequalities and bounded variables

         INTEGER :: out                 ! the output device number

         INTEGER :: level               ! the effective iteration dependent
                                        ! printout level

         INTEGER :: print_level         ! the global printout level

         INTEGER :: exitc               ! the value of inform%status at the
                                        ! end of SOLVE.

         INTEGER :: model_used          ! the type of model currently in use

         INTEGER :: next_vote           ! the position of the next vote for a
                                        ! model in the vector s%vote

         INTEGER :: nsemib              ! the requested band preconditioner
                                        ! semi-bandwidth

         INTEGER :: bandw               ! the actual band preconditioner
                                        ! semi-bandwidth

         INTEGER :: filter_size         ! the number of theta values in the
                                        ! filter

         INTEGER :: filter_nbr_inactive ! the number of inactive values in the
                                        ! current filter

         INTEGER :: filter_capacity     ! the number of slices in the current
                                        ! filter

         INTEGER :: active_filter       ! which of the two possible hooks for
                                        ! the filter is active

         INTEGER :: cuter_J_size        ! the size of the Jacobian according
                                        ! to CUTEr (includes an additional
                                        ! proble%n locations for the gradient
                                        ! of the objective function)

         INTEGER :: stage               ! the current stage in the calculation
                                        ! Possible values: READY, ONGOING,
                                        ! DONE, VOID

         INTEGER :: filter_sign         ! indicates whether groups are
                                        ! restricted in sign or not :
                                        ! (RESTRICTED, UNRESTRICTED, MIXED )

         INTEGER :: step_accuracy       ! the subproblem accuracy currently
                                        ! used (ADAPTIVE, FULL)

         INTEGER :: filter_first        ! the position in the filter of the
                                        ! entry with smallest norm

         LOGICAL :: use_filter          ! .TRUE. iff the filter is currently
                                        ! in use

         LOGICAL :: gltr_initialized    ! as it says

         LOGICAL :: has_inequalities    ! .TRUE. when all problem constraints
                                        ! are not equalities

         LOGICAL :: has_fixed           ! .TRUE. iff there are fixed variables

         LOGICAL :: has_bounds          ! .TRUE. when some problem variables
                                        ! have bounds

         LOGICAL :: goth                ! .TRUE. if the Hessians of the
                                        ! individual constraints have already
                                        ! been calculated at the current iterate

         LOGICAL :: restrict            ! .TRUE. iff the next step must lie in
                                        ! the trust region

         LOGICAL :: unsuccess           ! .TRUE. iff the last iteration has
                                        ! not been successful

         LOGICAL :: first_TR_step       ! .TRUE. if no previous TR step has
                                        ! ever been taken

         LOGICAL :: filter_acceptable   ! .TRUE. if the trial point is
                                        ! acceptable for the filter

         LOGICAL :: weakly_acceptable   ! .TRUE. if the trial point is
                                        ! acceptable for the weak criterion

         LOGICAL :: tr_acceptable       ! .TRUE. if the trial point is
                                        ! acceptable for the trust region

         LOGICAL :: acceptable          ! .TRUE. if the trial point is
                                        ! acceptable for at least one criterion

         LOGICAL :: best_x_is_past      ! .TRUE. if the best point is not the
                                        ! current one

         LOGICAL :: bad_model           ! .TRUE. if the current model is not
                                        ! very satisfactory

         LOGICAL :: RC_newx             ! .TRUE. iff the value of x passed when
                                        ! requiring the product of the Hessian
                                        ! of the Lagrangian times a vector is
                                        ! different from that passed on the
                                        ! previous such situation

         LOGICAL :: u_allocated         ! .TRUE. iff the pointer us has been
                                        ! allocated (instead of pointed to
                                        ! to an existing target).

         REAL ( KIND = wp ) :: radius   ! the current trust-region radius

         REAL ( KIND = wp ) :: prev_radius   ! the trust-region radius at the
                                        ! previous iteration

         REAL ( KIND = wp ) :: f_max    ! the maximum accetable objective value

         REAL ( KIND = wp ) :: f_old    ! the previous objective value

         REAL ( KIND = wp ) :: f_plus   ! the objective value at the trial
                                        ! point

         REAL ( KIND = wp ) :: feps     ! the noisy function value

         REAL ( KIND = wp ) :: epsilon  ! the smoothing parameter for the l1
                                        ! penalty for inequalities

         REAL ( KIND = wp ) :: best_fx  ! the best objective value found so far

         REAL ( KIND = wp ) :: model_value ! the current model value

         REAL ( KIND = wp ) :: ared     ! the achieved reduction on obj value

         REAL ( KIND = wp ) :: rho      ! the ratio of achieved to predicted
                                        ! reduction for the current model

         REAL ( KIND = wp ) :: x_norm   ! the norm of the current iterate

         REAL ( KIND = wp ) :: s_norm   ! the norm of the current step

         REAL ( KIND = wp ) :: s_norm2  ! the Euclidean norm of the current step

         REAL ( KIND = wp ) :: g_norm   ! the norm of the current gradient

         REAL ( KIND = wp ) :: g_norminf! the max norm of the current gradient

         REAL ( KIND = wp ) :: g_norminf_u ! the max norm of the current
                                        ! unpreconditioned gradient

         REAL ( KIND = wp ) :: g_norm2  ! the Euclidean norm of the current
                                        ! gradient

         REAL ( KIND = wp ) :: extent   ! the current trust-region relaxation
                                        ! factor

         REAL ( KIND = wp ) :: gltr_radius ! the current radius used for
                                        ! computing the TR step with GLTR

         CHARACTER( LEN = 4 ) :: it_status ! a string reflecting the nature of
                                        ! the current iteration

!        ----------------------------------------------
!        The GLTR data
!        ----------------------------------------------

         TYPE ( GLTR_data_type )    :: GLTR_data
         TYPE ( GLTR_control_type ) :: GLTR_control
         TYPE ( GLTR_info_type )    :: GLTR_info

!        ----------------------------------------------
!        The value of the controls at the previous call
!        ----------------------------------------------

         TYPE ( FILTRANE_control_type ) :: prev_control ! the value of control
                                        ! for the previous execution of FILTRANE

!        ---------------------------------------------
!        Pointer arrays defined globally with FILTRANE
!        ---------------------------------------------

         INTEGER, POINTER, DIMENSION( : ) :: group ! group(i) is the index of
                                        ! the group to which c_i belongs.
                                        ! It points either s%aut_group (when
                                        ! AUTOMATIC grouping is used) or to
                                        ! control%group (when USER-DEFINED
                                        ! grouping is used).

         INTEGER, POINTER, DIMENSION( : ) :: aut_group ! the automatic avatar
                                        ! of s%group

         INTEGER, POINTER, DIMENSION( : ) :: g_status ! .TRUE. if the group
                                        ! is sign restricted.  Only allocated
                                        ! if s%filter_sign = MIXED

         INTEGER, POINTER, DIMENSION( : ) :: iw   ! integer worspace of size
                                        ! problem%m + 1

         INTEGER, POINTER, DIMENSION( : ) :: row  ! integer worspace of size
                                        ! problem%J_ne

         INTEGER, POINTER, DIMENSION( : ) :: perm ! integer worspace of size
                                        ! problem%J_ne

         INTEGER, POINTER, DIMENSION( : ) :: vote ! the most recent votes for
                                        ! a model

         INTEGER, POINTER, DIMENSION( : ) :: filter_next_1, filter_next_2
                                        ! the position in the
                                        ! filter of the entry which is next by
                                        ! order of increasing norms

         LOGICAL, POINTER, DIMENSION( : ) :: active_1, active_2
                                        ! tells if values in the filter are
                                        ! active

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: theta ! the vector of
                                        ! group residuals

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: step ! the trial step

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: r ! workspace of
                                        ! dimension problem%n

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: diag ! workspace of
                                        ! dimension problem%n

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: t ! workspace of
                                        ! dimension MAX( problem%n, problem%m )

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: u ! workspace of
                                        ! dimension problem%n

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: v ! workspace of
                                        ! dimension MAX( problem%n, problem%m )

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: w ! workspace of
                                        ! dimension MAX( problem%n, problem%m )

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: best_x ! the best point
                                        ! found so far

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: fnorm_1, fnorm_2
                                        ! the filter (two hooks)

         REAL( KIND = wp ), POINTER, DIMENSION( :, : ) :: filter_1, filter_2
                                        ! the filter (two hooks)

         REAL( KIND = wp ), POINTER, DIMENSION( :, : ) :: offdiag ! workspace
                                        ! of dimension control%semi_bandwith
                                        ! times problem%n

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: RC_v ! the
                                        ! value of the vector to be
                                        ! premultiplied by the Hessian of the
                                        ! Lagrangian.

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: RC_Mv ! the value of
                                        ! the result of premultiplying the
                                        ! vector RC_v by the Hessian of the
                                        ! Lagrangian.

         REAL( KIND = wp ), POINTER, DIMENSION( : ) :: RC_Pv ! the value of
                                        ! the result of preconditioning the
                                        ! vector RC_v.

!        ---------------------------------------------
!        Problem dimensions and characteristics
!        ---------------------------------------------

      END TYPE

!-------------------------------------------------------------------------------
!
!                           PRIVATE DEFINITIONS
!
!-------------------------------------------------------------------------------
!
!  NOTE: REALLY DON'T MODIFY WHAT FOLLOWS
!        (unless you are absolutely certain that you know what you are doing)!!
!
!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ZERO     = 0.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: TENTH    = 0.1_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: HALF     = 0.5_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ONE      = 1.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: TWO      = ONE + ONE
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: THREE    = TWO + ONE
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: TEN      = 10.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: HUNDRED  = TEN * TEN
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: THOUSAND = TEN * HUNDRED
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: INFINITY = 10.0_wp ** 20
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: OVERFLOW = HUGE( ONE )
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: EPSMACH  = EPSILON( ONE )
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ONE_DOT_ONE = ONE + TENTH

!  Local parameters

!     Stage flag values

      INTEGER, PRIVATE, PARAMETER :: VOID                         =  0
      INTEGER, PRIVATE, PARAMETER :: READY                        =  1
      INTEGER, PRIVATE, PARAMETER :: DONE                         =  2

!     Return codes

      INTEGER, PRIVATE, PARAMETER :: GET_C_AND_J_0                =   1
      INTEGER, PRIVATE, PARAMETER :: GET_C_AND_J_F                =   2
      INTEGER, PRIVATE, PARAMETER :: GET_C_0                      =   3
      INTEGER, PRIVATE, PARAMETER :: GET_C                        =   4
      INTEGER, PRIVATE, PARAMETER :: GET_C_F                      =   5
      INTEGER, PRIVATE, PARAMETER :: GET_J_A                      =   6
      INTEGER, PRIVATE, PARAMETER :: GET_JV                       =   7
      INTEGER, PRIVATE, PARAMETER :: GET_JTC_0                    =   8
      INTEGER, PRIVATE, PARAMETER :: GET_JTV                      =   9
      INTEGER, PRIVATE, PARAMETER :: GET_JTC_A                    =  10
      INTEGER, PRIVATE, PARAMETER :: GET_JTC_F                    =  11
      INTEGER, PRIVATE, PARAMETER :: GET_PREC_G_0                 =  12
      INTEGER, PRIVATE, PARAMETER :: GET_PREC                     =  13
      INTEGER, PRIVATE, PARAMETER :: GET_PREC_G_A                 =  14
      INTEGER, PRIVATE, PARAMETER :: GET_MP1                      =  15
      INTEGER, PRIVATE, PARAMETER :: GET_MP2                      =  16
      INTEGER, PRIVATE, PARAMETER :: GET_JTC                      =  20
      INTEGER, PRIVATE, PARAMETER :: CHECKPOINTING_ERROR          = -21
      INTEGER, PRIVATE, PARAMETER :: GLTR_ERROR                   = -22
      INTEGER, PRIVATE, PARAMETER :: UNASSOCIATED_INPUT           = -23
      INTEGER, PRIVATE, PARAMETER :: SORT_TOO_LONG                = -24
      INTEGER, PRIVATE, PARAMETER :: WRONG_P                      = -25
      INTEGER, PRIVATE, PARAMETER :: USER_GROUPS_UNDEFINED        = -26
      INTEGER, PRIVATE, PARAMETER :: WRONG_NUMBER_OF_USER_GROUPS  = -27
      INTEGER, PRIVATE, PARAMETER :: WRONG_USER_GROUP_INDEX       = -28
      INTEGER, PRIVATE, PARAMETER :: WRONG_STATUS                 = -29
      INTEGER, PRIVATE, PARAMETER :: INTERNAL_ERROR_100           = -100

!     Filter sign restrictions

      INTEGER, PRIVATE, PARAMETER :: RESTRICTED                   =  0
      INTEGER, PRIVATE, PARAMETER :: UNRESTRICTED                 =  1
      INTEGER, PRIVATE, PARAMETER :: MIXED                        =  2

!     Group status

      INTEGER, PRIVATE, PARAMETER :: SINGLE_UNRESTRICTED          =  1
      INTEGER, PRIVATE, PARAMETER :: SINGLE_RESTRICTED            =  2
      INTEGER, PRIVATE, PARAMETER :: MULTIPLE                     = -1

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

   CONTAINS ! The whole of the FILTRANE code

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================       I N I T I A L I Z E         =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

      SUBROUTINE FILTRANE_initialize( control, inform, s )

!     Checks the problem characteristics, verifies the user defined presolving
!     parameters and initialize the necessary data structures.

!     Arguments:

      TYPE ( FILTRANE_control_type ), INTENT( INOUT ) :: control

!              the FILTRANE control structure (see above)

      TYPE ( FILTRANE_inform_type ), INTENT( INOUT ) :: inform

!              the FILTRANE exit information structure (see above)

      TYPE ( FILTRANE_data_type ), INTENT( INOUT ) :: s

!              the FILTRANE saved information structure (see above)

!     Programming: Ph. Toint, November 2002

!==============================================================================

!-------------------------------------------------------------------------------
!
!                        Initial definitions
!
!-------------------------------------------------------------------------------

!     Printing
!     NOTE :  if the output device or amount of printing must be modified for
!             the execution of INITIALIZE, they must be reset in the following
!             two lines and the module recompiled.  (This is because the
!             SPECIFICATIONS file is only read after execution of INITIALIZE.)

      s%out   = 6                      !  the default printout device
      s%level = SILENT                 !  the default amount of printout

!     s%level = DEBUG

!     Print banner

      IF ( s%level >= TRACE ) THEN
         CALL FILTRANE_banner( s%out )
         IF ( s%level  >= DEBUG ) WRITE( s%out, 1001 )
      END IF

!     Initialize the exit status and the exit message to that corresponding
!     to a successful exit.

      inform%message( 1 ) = ''
      inform%message( 2 ) = ''
      inform%message( 3 ) = ''
      inform%status       = OK
      WRITE( inform%message( 1 ), 1002 )

!-------------------------------------------------------------------------------
!
!          Define the control parameters from their default values.
!
!-------------------------------------------------------------------------------

      IF ( s%level >= DETAILS ) WRITE( s%out, 1003 )

!     Printout device

      control%out           = s%out
      s%prev_control%out    = s%out
      control%errout        = s%out
      s%prev_control%errout = s%out

!     Print level

      control%print_level        = s%level
      s%prev_control%print_level = control%print_level
      IF ( s%level >= DEBUG ) THEN
         SELECT CASE ( s%level )
         CASE ( SILENT )
         CASE ( TRACE  )
            WRITE( s%out, 1004 )
         CASE ( ACTION )
            WRITE( s%out, 1005 )
         CASE ( DETAILS )
            WRITE( s%out, 1006 )
         CASE ( DEBUG )
            WRITE( s%out, 1007 )
         CASE ( CRAZY: )
            WRITE( s%out, 1008 )
         END SELECT
      END IF

!     First and last iteration for printing

      control%start_print =  0
      control%stop_print  = -1
      s%prev_control%start_print = control%start_print
      s%prev_control%stop_print  = control%stop_print
      IF ( s%level >= DEBUG ) WRITE( s%out, 1009 ) control%start_print,        &
                                                   control%stop_print

!     Residual accuracy

      IF ( wp == sp ) THEN
         control%c_accuracy = TEN ** ( -4 )
      ELSE
         control%c_accuracy = TEN ** ( -6 )
      END IF
      s%prev_control%c_accuracy = control%c_accuracy
      IF ( s%level >= DEBUG ) WRITE( s%out, 1010 ) control%c_accuracy

!     Gradient accuracy

      IF ( wp == sp ) THEN
         control%g_accuracy = TEN ** ( -4 )
      ELSE
         control%g_accuracy = TEN ** ( -6 )
      END IF
      s%prev_control%g_accuracy = control%g_accuracy
      IF ( s%level >= DEBUG ) WRITE( s%out, 1011 ) control%g_accuracy

!     Gradient norm for stopping (Note that stopping on preconditioned
!     maximum norm is impossible in general, unless the preconditioner
!     is diagonal, which is unknown at this stage)

      control%stop_on_prec_g = .TRUE.
      control%stop_on_g_max  = .FALSE.
      s%prev_control%stop_on_prec_g = control%stop_on_prec_g
      s%prev_control%stop_on_g_max  = control%stop_on_g_max
      IF ( s%level >= DEBUG ) THEN
         IF ( control%stop_on_g_max ) THEN
            IF ( control%stop_on_prec_g ) THEN
               WRITE( s%out, 1012 )
            ELSE
               WRITE( s%out, 1013 )
            END IF
          ELSE
            IF ( control%stop_on_prec_g ) THEN
               WRITE( s%out, 1069 )
            ELSE
               WRITE( s%out, 1070 )
            END IF
         END IF
      END IF

!     Maximum number of iterations

      control%max_iterations = 1000
      s%prev_control%max_iterations = control%max_iterations
      IF ( s%level >= DEBUG ) WRITE( s%out, 1014 ) control%max_iterations

!     Maximum number of conjugate gradient iterations

      control%max_cg_iterations = 15
      s%prev_control%max_cg_iterations = control%max_cg_iterations
      IF ( s%level >= DEBUG ) WRITE( s%out, 1015 )  control%max_cg_iterations

!     Inequality penalty type

      control%inequality_penalty_type = 2
      s%prev_control%inequality_penalty_type = control%inequality_penalty_type
      IF ( s%level >= DEBUG ) WRITE(s%out,1016) control%inequality_penalty_type

!     Model type

      control%model_type = AUTOMATIC
      s%prev_control%model_type = control%model_type
      IF ( s%level >= DEBUG ) THEN
         SELECT case ( control%model_type )
         CASE ( GAUSS_NEWTON )
             WRITE( s%out, 1017 )
         CASE ( NEWTON )
             WRITE( s%out, 1018 )
         CASE ( AUTOMATIC )
             WRITE( s%out, 1019 )
         END SELECT
      END IF

!     Automatic model criterion

      control%model_criterion = BEST_FIT
      s%prev_control%model_criterion = control%model_criterion
      IF ( s%level >= DEBUG .AND. control%model_type == AUTOMATIC ) THEN
         SELECT CASE ( control%model_criterion )
         CASE ( BEST_FIT )
            WRITE( s%out, 1020 )
         CASE ( BEST_REDUCTION )
            WRITE( s%out, 1021 )
         END SELECT
      END IF

!     Automatic model inertia ( must be >= 1)

      control%model_inertia = 5
      s%prev_control%model_inertia = control%model_inertia
      IF ( s%level >= DEBUG .AND. control%model_type == AUTOMATIC ) THEN
         WRITE( s%out, 1022 ) control%model_inertia
      END IF

!     Minimum GLTR accuracy in the adaptive mode (must be in (0,1))

      control%min_gltr_accuracy = 0.01_wp
      s%prev_control%min_gltr_accuracy = control%min_gltr_accuracy
      IF ( s%level >= DEBUG ) WRITE( s%out, 1023 ) control%min_gltr_accuracy

!     Power of the initial residual in the GLTR accuracy in the adaptive mode

      control%gltr_accuracy_power = 1.0_wp
      s%prev_control%gltr_accuracy_power = control%gltr_accuracy_power
      IF ( s%level >= DEBUG ) WRITE( s%out, 1067 ) control%gltr_accuracy_power

!     Weak acceptance test power

      control%weak_accept_power = -1.0_wp
      s%prev_control%weak_accept_power = control%weak_accept_power
      IF ( s%level >= DEBUG ) THEN
         IF ( control%weak_accept_power >= ZERO ) THEN
            WRITE( s%out, 1024 ) control%weak_accept_power
         ELSE
            WRITE( s%out, 1025 )
         END IF
      END IF

!     Minimum weak acceptance test factor (must be in (0,1))

      control%min_weak_accept_factor = 0.1_wp
      s%prev_control%min_weak_accept_factor = control%min_weak_accept_factor
      IF ( s%level >= DEBUG )WRITE( s%out, 1068 ) control%min_weak_accept_factor

!     Preconditioner used

      control%prec_used = NONE
      s%prev_control%prec_used = control%prec_used
      IF ( s%level >= DEBUG ) THEN
         SELECT case ( control%prec_used )
         CASE ( NONE )
             WRITE( s%out, 1026 )
         CASE ( BANDED )
             WRITE( s%out, 1027 )
         CASE ( USER_DEFINED )
             WRITE( s%out, 1028 )
         END SELECT
      END IF

!     Bandwith for the band preconditioner

      control%semi_bandwidth = 5
      s%prev_control%semi_bandwidth = control%semi_bandwidth
      IF ( s%level >= DEBUG .AND. control%prec_used /= NONE )                  &
         WRITE( s%out, 1029 ) control%semi_bandwidth

!     External Jacobian products

      control%external_J_products = .FALSE.
      s%prev_control%external_J_products = control%external_J_products
      IF ( s%level >= DEBUG ) THEN
         IF ( control%external_J_products ) THEN
            WRITE( s%out, 1030 )
         ELSE
            WRITE( s%out, 1031 )
         END IF
      END IF

!     Accuracy of the subproblem solution

      control%subproblem_accuracy = ADAPTIVE
      s%prev_control%subproblem_accuracy = control%subproblem_accuracy
      IF ( s%level >= DEBUG ) THEN
         SELECT case ( control%subproblem_accuracy )
         CASE ( ADAPTIVE )
             WRITE( s%out, 1032 )
         CASE ( FULL )
             WRITE( s%out, 1033 )
         END SELECT
      END IF

!     Equations grouping

      control%grouping = NONE
      s%prev_control%grouping = control%grouping
      IF ( s%level >= DEBUG ) THEN
         SELECT CASE ( control%grouping )
         CASE ( NONE )
             WRITE( s%out, 1034 )
         CASE ( AUTOMATIC )
             WRITE( s%out, 1035 )
         CASE ( USER_DEFINED )
             WRITE( s%out, 1036 )
         END SELECT
      END IF

!     Number of groups for the AUTOMATIC grouping strategy

      control%nbr_groups = 10
      s%prev_control%nbr_groups = control%nbr_groups
      IF ( s%level >= DEBUG .AND. control%grouping == AUTOMATIC ) THEN
         WRITE( s%out, 1037 ) control%nbr_groups
      END IF

!     Group balancing

      control%balance_group_values = .FALSE.
      s%prev_control%balance_group_values = control%balance_group_values
      IF ( s%level >= DEBUG .AND. control%grouping == AUTOMATIC ) THEN
         IF ( control%balance_group_values ) THEN
            WRITE( s%out, 1038 )
         ELSE
            WRITE( s%out, 1039 )
         END IF
      END IF

!     Trust-region parameter : initial radius

      control%initial_radius = ONE
      s%prev_control%initial_radius = control%initial_radius
      IF ( s%level >= DEBUG ) WRITE( s%out, 1040 ) control%initial_radius

!     Trust-region parameter : eta_1

      control%eta_1 = 0.01_wp
      s%prev_control%eta_1 = control%eta_1
      IF ( s%level >= DEBUG ) WRITE( s%out, 1041 ) control%eta_1

!     Trust-region parameter : eta_2

      control%eta_2 = 0.9_wp
      s%prev_control%eta_2 = control%eta_2
      IF ( s%level >= DEBUG ) WRITE( s%out, 1042 ) control%eta_2

!     Trust-region parameter : gamma_1

      control%gamma_0 = 0.0625_wp
      s%prev_control%gamma_0 = control%gamma_0
      IF ( s%level >= DEBUG ) WRITE( s%out, 1043 ) control%gamma_0

!     Trust-region parameter : gamma_1

      control%gamma_1 = 0.25_wp
      s%prev_control%gamma_1 = control%gamma_1
      IF ( s%level >= DEBUG ) WRITE( s%out, 1044 ) control%gamma_1

!     Trust-region parameter : gamma_2

      control%gamma_2 = 2.0_wp
      s%prev_control%gamma_2 = control%gamma_2
      IF ( s%level >= DEBUG ) WRITE( s%out, 1045 ) control%gamma_2

!     Initial trust-region relaxation factor (must be at least 1.0)

      control%itr_relax = INFINITY
      s%prev_control%itr_relax = control%itr_relax
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1046 ) control%itr_relax
      END IF

!     Secondary trust-region relaxation factor  (must be at least 1.0)

      control%str_relax = THOUSAND
      s%prev_control%str_relax = control%str_relax
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1047 ) control%str_relax
      END IF

!     Filter usage

      control%use_filter = ALWAYS
      s%prev_control%use_filter = control%use_filter
      IF ( s%level >= DEBUG ) THEN
         SELECT CASE ( control%use_filter )
         CASE ( NEVER )
            WRITE( s%out, 1048 )
         CASE ( INITIAL )
            WRITE( s%out, 1049 )
         CASE ( ALWAYS )
            WRITE( s%out, 1050 )
         END SELECT
      END IF

!     Filter sign restriction

      control%filter_sign_restriction = .FALSE.
      s%prev_control%filter_sign_restriction = control%filter_sign_restriction
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         IF ( control%filter_sign_restriction ) THEN
            WRITE( s%out, 1051 )
         ELSE
            WRITE( s%out, 1052 )
         ENDIF
      END IF

!     Filter margin type

      control%margin_type = FIXED
      s%prev_control%margin_type = control%margin_type
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         SELECT CASE ( control%margin_type )
         CASE ( FIXED )
            WRITE( s%out, 1053 )
         CASE ( CURRENT )
            WRITE( s%out, 1054 )
         CASE ( SMALLEST )
            WRITE( s%out, 1055 )
         END SELECT
      END IF

!     Maximal filter size

      control%maximal_filter_size = -1
      s%prev_control%maximal_filter_size = control%maximal_filter_size
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1056 ) control%maximal_filter_size
      END IF

!     Filter parameter : the maximum margin size

      control%gamma_f = 0.001_wp
      s%prev_control%gamma_f = control%gamma_f
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1057 ) control%gamma_f
      END IF

!     Filter parameter : filter initial size

      control%filter_size_increment = 50
      s%prev_control%filter_size_increment = control%filter_size_increment
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1058 ) control%filter_size_increment
      END IF

!     Removal of dominated filter entries

      control%remove_dominated = .TRUE.
      s%prev_control%remove_dominated = control%remove_dominated
      IF ( s%level >= DEBUG .AND. control%use_filter /= NEVER ) THEN
         IF ( control%remove_dominated ) THEN
            WRITE( s%out, 1059 )
         ELSE
            WRITE( s%out, 1060 )
         END IF
      END IF

!     Save best point

      control%save_best_point = .FALSE.
      s%prev_control%save_best_point = control%save_best_point
      IF ( s%level >= DEBUG ) THEN
         IF ( control%save_best_point ) THEN
            WRITE( s%out, 1061 )
         ELSE
            WRITE( s%out, 1062 )
         END IF
      END IF

!     Checkpointing frequency, file and device

      control%checkpoint_freq = 0
      control%checkpoint_file = 'FILTRANE.sav'
      control%checkpoint_dev  = 55
      s%prev_control%checkpoint_freq = control%checkpoint_freq
      s%prev_control%checkpoint_file = control%checkpoint_file
      s%prev_control%checkpoint_dev  = control%checkpoint_dev
      IF ( s%level >= DEBUG ) THEN
         IF ( control%checkpoint_freq == 0 ) THEN
            WRITE( s%out, 1063 )
         ELSE
            WRITE( s%out, 1064 ) control%checkpoint_freq,                      &
                                 control%checkpoint_file, control%checkpoint_dev
         END IF
      END IF

!     Restart from checkpoint

      control%restart_from_checkpoint = .FALSE.
      s%prev_control%restart_from_checkpoint = control%restart_from_checkpoint
      IF ( s%level >= DEBUG ) THEN
         IF ( control%restart_from_checkpoint ) THEN
            WRITE( s%out, 1065 )
         ELSE
            WRITE( s%out, 1066 )
         END IF
      END IF

!     Nullify the unassociated pointers in the FILTRANE_data_type structure.

      NULLIFY( s%theta, s%aut_group, s%iw, s%row, s%perm, s%r, s%u, s%v, s%w,  &
               s%step, s%filter_1, s%filter_2, s%active_1, s%active_2, s%diag, &
               s%vote, s%offdiag, s%best_x, s%group, s%fnorm_1, s%fnorm_2,     &
               s%filter_next_1, s%filter_next_2, s%RC_v, s%RC_Mv, s%RC_Pv,     &
               s%g_status, s%t )

!     Set the indicator to tell GLTR has not been initialized yet

      s%gltr_initialized = .FALSE.

!     Set the stage indicator

      s%stage = READY

!     Terminate

      CALL FILTRANE_say_goodbye( control, inform, s )

      RETURN

!     Formats

1001  FORMAT( 1x, 'initial verifications and workspace allocation' )
1002  FORMAT( 1x, 'FILTRANE: successful exit' )
1003  FORMAT( 3x, 'defining presolve default control parameters' )
1004  FORMAT( 4x, 'print level is TRACE' )
1005  FORMAT( 4x, 'print level is ACTION' )
1006  FORMAT( 4x, 'print level is DETAILS' )
1007  FORMAT( 4x, 'print level is DEBUG' )
1008  FORMAT( 4x, 'print level is CRAZY' )
1009  FORMAT( 4x, 'printing between iterations ',i6,' and ',i6 )
1010  FORMAT( 4x, 'residual accuracy set to ', 1pE11.3 )
1011  FORMAT( 4x, 'gradient accuracy set to ', 1pE11.3 )
1012  FORMAT( 4x, 'stopping on the preconditioned maximum gradient norm' )
1013  FORMAT( 4x, 'stopping on the unpreconditioned maximum gradient norm' )
1014  FORMAT( 4x, 'maximum number of iterations set to ', i6 )
1015  FORMAT( 4x, 'maximum number of CG iterations set to ', i6 )
1016  FORMAT( 4x, 'inequality penalty type set to ', i2 )
1017  FORMAT( 4x, 'GAUSS-NEWTON model used' )
1018  FORMAT( 4x, 'NEWTON model used' )
1019  FORMAT( 4x, 'AUTOMATIC model type used' )
1020  FORMAT( 4x, 'automatic model criterion is BEST_FIT' )
1021  FORMAT( 4x, 'automatic model criterion is BEST_REDUCTION' )
1022  FORMAT( 4x, 'automatic model inertia set to ', i3 )
1023  FORMAT( 4x, 'minimum relative subproblem accuracy set to ', 1pE11.3 )
1024  FORMAT( 4x, 'weak acceptance test enabled with power', 1pE11.3 )
1025  FORMAT( 4x, 'weak acceptance test disabled' )
1026  FORMAT( 4x, 'no preconditioner used' )
1027  FORMAT( 4x, 'band preconditioner used' )
1028  FORMAT( 4x, 'external user-defined preconditioner used' )
1029  FORMAT( 4x, 'semi-bandwidth for band preconditioner set to ', i6 )
1030  FORMAT( 4x, 'external Jacobian products' )
1031  FORMAT( 4x, 'internal Jacobian products' )
1032  FORMAT( 4x, 'subproblem accuracy level set to ADAPTIVE' )
1033  FORMAT( 4x, 'subproblem accuracy level set to FULL' )
1034  FORMAT( 4x, 'grouping mode set to NONE' )
1035  FORMAT( 4x, 'grouping mode set to AUTOMATIC' )
1036  FORMAT( 4x, 'grouping mode set to USER_DEFINED' )
1037  FORMAT( 4x, 'number of groups for automatic grouping set to ', i6 )
1038  FORMAT( 4x, 'attempt to balance initial group values' )
1039  FORMAT( 4x, 'no attempt to balance initial group values' )
1040  FORMAT( 4x, 'initial trust-region radius set to ', 1pE11.3 )
1041  FORMAT( 4x, 'minimum rho for successful iteration set to ', 1pE11.3 )
1042  FORMAT( 4x, 'minimum rho for very successful iteration set to ', 1pE11.3 )
1043  FORMAT( 4x, 'worst-case radius reduction factor set to ', 1pE11.3 )
1044  FORMAT( 4x, 'radius reduction factor set to ', 1pE11.3 )
1045  FORMAT( 4x, 'radius increase factor set to ', 1pE11.3 )
1046  FORMAT( 4x, 'initial TR relaxation factor set to ', 1pE11.3 )
1047  FORMAT( 4x, 'secondary TR relaxation factor set to ', 1pE11.3 )
1048  FORMAT( 4x, 'filter use set to NEVER' )
1049  FORMAT( 4x, 'filter use set to INITIAL' )
1050  FORMAT( 4x, 'filter use set to ALWAYS' )
1051  FORMAT( 4x, 'filter sign restriction is enabled' )
1052  FORMAT( 4x, 'filter sign restriction is disabled' )
1053  FORMAT( 4x, 'filter margin type set to FIXED' )
1054  FORMAT( 4x, 'filter margin type set to CURRENT' )
1055  FORMAT( 4x, 'filter margin type set to SMALLEST' )
1056  FORMAT( 4x, 'maximal filter size set to ', i6 )
1057  FORMAT( 4x, 'filter margin factor set to ', 1pE11.3 )
1058  FORMAT( 4x, 'filter size increment set to ', i6 )
1059  FORMAT( 4x, 'dominated filter entries are removed' )
1060  FORMAT( 4x, 'dominated filter entries are not removed' )
1061  FORMAT( 4x, 'best point saved' )
1062  FORMAT( 4x, 'best point not saved' )
1063  FORMAT( 4x, 'no checkpointing' )
1064  FORMAT( 4x, 'checkpointing every ', i6, ' iteration on file ', a30,      &
                  ' (device ', i2, ')')
1065  FORMAT( 4x, 'restarting from saved checkpoint' )
1066  FORMAT( 4x, 'starting from problem%x' )
1067  FORMAT( 4x, 'subproblem relative accuracy power is ', 1pE11.3 )
1068  FORMAT( 4x, 'minimum weak acceptance factor is ', 1pE11.3 )
1069  FORMAT( 4x, 'stopping on the preconditioned Euclidean gradient norm')
1070  FORMAT( 4x, 'stopping on the unpreconditioned Euclidean gradient norm')

      END SUBROUTINE FILTRANE_initialize

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================             R E A D               =========================
!===================                                   =========================
!===================         S P E C F I L E           =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

      SUBROUTINE FILTRANE_read_specfile( device, control, inform )

!     Reads the content of a specification files and performs the assignment
!     of values associated with given keywords to the corresponding control
!     parameters. See the documentation of the GALAHAD_specfile module for
!     further details.

!     Arguments:

      INTEGER, INTENT( IN ) :: device

!            The device number associated with the specification file. Note
!            that the file must be open for input.  The file is REWINDed
!            before use.

      TYPE ( FILTRANE_control_type ), INTENT( INOUT ) :: control

!            The FILTRANE control structure (see above)

      TYPE ( FILTRANE_inform_type ), INTENT( INOUT ) :: inform

!            The FILTRANE exit information structure (see above)

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER :: lspec, ios
      CHARACTER( LEN = 16 ), PARAMETER :: specname = 'FILTRANE        '
      TYPE ( SPECFILE_item_type ), DIMENSION( : ), ALLOCATABLE :: spec

!     Construct the specification list.

      lspec = 46        !  there are 46 possible specification commands

!     Allocate the spec item data structure.

      ALLOCATE( spec( lspec ), STAT = ios )
      IF ( ios /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), 1000 ) lspec
         RETURN
      END IF

!     Define the keywords.

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level'
      spec(  4 )%keyword = 'residual-accuracy'
      spec(  5 )%keyword = 'gradient-accuracy'
      spec(  6 )%keyword = 'maximum-number-of-iterations'
      spec(  7 )%keyword = 'maximum-number-of-cg-iterations'
      spec(  8 )%keyword = 'preconditioner-used'
      spec(  9 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 10 )%keyword = 'equations-grouping'
      spec( 11 )%keyword = 'minimum-rho-for-successful-iteration'
      spec( 12 )%keyword = 'minimum-rho-for-very-successful-iteration'
      spec( 13 )%keyword = 'worst-case-radius-reduction-factor'
      spec( 14 )%keyword = 'radius-reduction-factor'
      spec( 15 )%keyword = 'radius-increase-factor'
      spec( 16 )%keyword = 'initial-radius'
      spec( 17 )%keyword = 'use-filter'
      spec( 18 )%keyword = 'filter-size-increment'
      spec( 19 )%keyword = 'filter-margin-factor'
      spec( 20 )%keyword = 'number-of-groups'
      spec( 21 )%keyword = 'model-type'
      spec( 22 )%keyword = 'subproblem-accuracy'
      spec( 23 )%keyword = 'balance-initial-group-values'
      spec( 24 )%keyword = 'weak-acceptance-power'
      spec( 25 )%keyword = 'maximum-filter-size'
      spec( 26 )%keyword = 'automatic-model-inertia'
      spec( 27 )%keyword = 'inequality-penalty-type'
      spec( 28 )%keyword = 'minimum-relative-subproblem-accuracy'
      spec( 29 )%keyword = 'initial-TR-relaxation-factor'
      spec( 30 )%keyword = 'secondary-TR-relaxation-factor'
      spec( 31 )%keyword = 'filter-sign-restriction'
      spec( 32 )%keyword = 'filter-margin-type'
      spec( 33 )%keyword = 'automatic-model-criterion'
      spec( 34 )%keyword = 'save-best-point'
      spec( 35 )%keyword = 'stop-on-preconditioned-gradient-norm'
      spec( 36 )%keyword = 'remove-dominated-filter-entries'
      spec( 37 )%keyword = 'checkpointing-frequency'
      spec( 38 )%keyword = 'checkpointing-file'
      spec( 39 )%keyword = 'checkpointing-device'
      spec( 40 )%keyword = 'restart-from-checkpoint'
      spec( 41 )%keyword = 'start-printing-at-iteration'
      spec( 42 )%keyword = 'stop-printing-at-iteration'
      spec( 43 )%keyword = 'external-Jacobian-products'
      spec( 44 )%keyword = 'relative-subproblem-accuracy-power'
      spec( 45 )%keyword = 'minimum-weak-acceptance-factor'
      spec( 46 )%keyword = 'stop-on-maximum-gradient-norm'

!     Read the specfile.

      CALL SPECFILE_read( device, specname, spec, lspec, control%errout )

!     Interpret the result.

      CALL SPECFILE_assign_integer( spec( 1 ),                                 &
                                    control%errout,                            &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 2 ),                                 &
                                    control%out,                               &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 3 ),                                 &
                                    control%print_level,                       &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 4 ),                                 &
                                    control%c_accuracy,                        &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 5 ),                                 &
                                    control%g_accuracy,                        &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 6 ),                                 &
                                    control%max_iterations,                    &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 7 ),                                 &
                                    control%max_cg_iterations,                 &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 8 ),                                 &
                                    control%prec_used,                         &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 9 ),                                 &
                                    control%semi_bandwidth,                    &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 10 ),                                &
                                    control%grouping,                          &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 11 ),                                &
                                    control%eta_1,                             &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 12 ),                                &
                                    control%eta_2,                             &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 13 ),                                &
                                    control%gamma_0,                           &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 14 ),                                &
                                    control%gamma_1,                           &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 15 ),                                &
                                    control%gamma_2,                           &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 16 ),                                &
                                    control%initial_radius,                    &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 17 ),                                &
                                    control%use_filter,                        &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 18 ),                                &
                                    control%filter_size_increment,             &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 19 ),                                &
                                    control%gamma_f,                           &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 20 ),                                &
                                    control%nbr_groups,                        &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 21 ),                                &
                                    control%model_type,                        &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 22 ),                                &
                                    control%subproblem_accuracy,               &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 23 ),                                &
                                    control%balance_group_values,              &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 24 ),                                &
                                    control%weak_accept_power,                 &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 25 ),                                &
                                    control%maximal_filter_size,               &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 26 ),                                &
                                    control%model_inertia,                     &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 27 ),                                &
                                    control%inequality_penalty_type,           &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 28 ),                                &
                                    control%min_gltr_accuracy,                 &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 29 ),                                &
                                    control%itr_relax,                         &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 30 ),                                &
                                    control%str_relax,                         &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 31 ),                                &
                                    control%filter_sign_restriction,           &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 32 ),                                &
                                    control%margin_type,                       &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 33 ),                                &
                                    control%model_criterion,                   &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 34 ),                                &
                                    control%save_best_point,                   &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 35 ),                                &
                                    control%stop_on_prec_g,                    &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 36 ),                                &
                                    control%remove_dominated,                  &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 37 ),                                &
                                    control%checkpoint_freq,                   &
                                    control%errout )
      CALL SPECFILE_assign_string ( spec( 38 ),                                &
                                    control%checkpoint_file,                   &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 39 ),                                &
                                    control%checkpoint_dev,                    &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 40 ),                                &
                                    control%restart_from_checkpoint,           &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 41 ),                                &
                                    control%start_print,                       &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 42 ),                                &
                                    control%stop_print,                        &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 43 ),                                &
                                    control%external_J_products,               &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 44 ),                                &
                                    control%gltr_accuracy_power,               &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 45 ),                                &
                                    control%min_weak_accept_factor,            &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 46 ),                                &
                                    control%stop_on_g_max,                     &
                                    control%errout )
      DEALLOCATE( spec )

      RETURN

!     Formats

1000  FORMAT(1x,'FILTRANE ERROR: no memory left for allocating spec(',i2,')')

      END SUBROUTINE FILTRANE_read_specfile

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================            S O L V E              =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

      SUBROUTINE FILTRANE_solve( problem, control, inform, s )

!     Arguments

      TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: problem

!              the FILTRANE problem structure (see the CPT module)

      TYPE ( FILTRANE_control_type ), INTENT( INOUT ) :: control

!              the FILTRANE control structure (see above)

      TYPE ( FILTRANE_inform_type ), INTENT( INOUT ) :: inform

!              the FILTRANE exit information structure (see above)

      TYPE ( FILTRANE_data_type ), INTENT( INOUT ) :: s

!              the FILTRANE saved information structure (see above)

!     Programming: Ph. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: iostat, dim, i, k, SORT_exitcode, j, BAND_status,             &
                 n_Newton, n_votes, n_bounded, nxt, ig, n_free,                &
                 n_fixed, n_lower, n_upper, n_range

!     REAL( KIND = wp ) :: f
      REAL( KIND = wp ) :: prered, sqrtn, rp, gs, gam, Nprered, GNprered,      &
                 delta_model, rhoGN, rhoN, ci, cli, cui, t, d2phi, twoeps,     &
                 violation, xj, xlj, xuj, fwp
      CHARACTER( LEN =  1 ) mult
      CHARACTER( LEN =  6 ) rstring

!    Interface blocks for the single and double precision BLAS routines
!    giving the two-norm, the inner product and swapping two vectors.

!     INTERFACE TWO_NORM
!
!        FUNCTION SNRM2( n, x, incx )
!          REAL  :: SNRM2
!          INTEGER, INTENT( IN ) :: n, incx
!          REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
!        END FUNCTION SNRM2
!
!        FUNCTION DNRM2( n, x, incx )
!          DOUBLE PRECISION  :: DNRM2
!          INTEGER, INTENT( IN ) :: n, incx
!          DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
!        END FUNCTION DNRM2
!
!     END INTERFACE

!     INTERFACE INNER_PRODUCT
!
!        FUNCTION SDOT( n, x, incx, y, incy )
!          REAL :: SDOT
!          INTEGER, INTENT( IN ) :: n, incx, incy
!          REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
!          REAL, INTENT( IN ), DIMENSION( incy * ( n - 1 ) + 1 ) :: y
!        END FUNCTION SDOT
!
!        FUNCTION DDOT( n, x, incx, y, incy )
!          DOUBLE PRECISION :: DDOT
!          INTEGER, INTENT( IN ) :: n, incx, incy
!          DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
!          DOUBLE PRECISION, INTENT( IN ), DIMENSION( incy * ( n - 1 ) + 1 ) :: y
!        END FUNCTION DDOT
!
!     END INTERFACE

     INTERFACE SWAP

        SUBROUTINE SSWAP( n, x, incx, y, incy )
          INTEGER, INTENT( IN ) :: n, incx, incy
          REAL, INTENT( INOUT ), DIMENSION( incx * ( n - 1 ) + 1 ) :: x
          REAL, INTENT( INOUT ), DIMENSION( incy * ( n - 1 ) + 1 ) :: y
        END SUBROUTINE SSWAP

        SUBROUTINE DSWAP( n, x, incx, y, incy )
          INTEGER, INTENT( IN ) :: n, incx, incy
          DOUBLE PRECISION, INTENT( INOUT ),                                   &
                            DIMENSION( incx * ( n - 1 ) + 1 ) :: x
          DOUBLE PRECISION, INTENT( INOUT ),                                   &
                            DIMENSION( incy * ( n - 1 ) + 1 ) :: y
        END SUBROUTINE DSWAP

     END INTERFACE

!==============================================================================

!*******************************************************************************
!*******************************************************************************
!***************                                             *******************
!***************      The reverse communication interface    *******************
!***************                                             *******************
!*******************************************************************************
!*******************************************************************************

   SELECT CASE ( inform%status )
   CASE ( OK )
      IF ( control%print_level >= DEBUG ) THEN
         WRITE( s%out, 1003 )
         WRITE( s%out, 1004 )
         WRITE( s%out, 1003 )
      END IF
   CASE ( GET_C_AND_J_0 )
      GO TO 200
   CASE ( GET_C_0 )
      GO TO 200
   CASE ( GET_JTC_0 )
      GO TO 250
   CASE ( GET_PREC_G_0 )
      GO TO 270
   CASE ( GET_PREC )
      GO TO 300
   CASE ( GET_PREC_G_A )
      GO TO 680
   CASE ( GET_JV )
      GO TO 400
   CASE ( GET_JTV )
      GO TO 450
   CASE ( GET_MP1 )
      GO TO 500
   CASE ( GET_MP2 )
      GO TO 550
   CASE ( GET_C )
      GO TO 600
   CASE ( GET_J_A )
      GO TO 630
   CASE ( GET_JTC_A )
      GO TO 660
   CASE ( GET_C_AND_J_F )
      GO TO 700
   CASE ( GET_C_F )
      GO TO 700
   CASE ( GET_JTC_F )
      GO TO 710
   CASE DEFAULT
      WRITE( inform%message( 1 ), 1022 )
      WRITE( inform%message( 2 ), 1023 ) inform%status
      inform%status = WRONG_STATUS
      GO TO 999
   END SELECT

!*******************************************************************************
!*******************************************************************************
!***************                                             *******************
!***************             The sequential code             *******************
!***************                                             *******************
!*******************************************************************************
!*******************************************************************************

!==============================================================================
!==============================================================================

!                            INITIALIZATIONS

!==============================================================================
!==============================================================================

!------------------------------------------------------------------------------
!     Print the banner and licensing terms.
!------------------------------------------------------------------------------

      IF ( control%print_level >= TRACE ) THEN
         CALL FILTRANE_banner( s%out )
!!         CALL COPYRIGHT( s%out, '2003' )
      END IF

!------------------------------------------------------------------------------
!     Check the problem's dimensions.
!------------------------------------------------------------------------------

      IF ( problem%n <= 0 ) THEN
         inform%status = WRONG_N
         WRITE( inform%message( 1 ), 1024 ) problem%n
         GO TO 999
      END IF
      IF ( problem%m < 0 ) THEN
         inform%status = WRONG_M
         WRITE( inform%message( 1 ), 1025 ) problem%m
         GO TO 999
      END IF

!------------------------------------------------------------------------------
!     Review the control structure.
!------------------------------------------------------------------------------

      CALL FILTRANE_revise_control

!------------------------------------------------------------------------------
!     Verify that INITIALIZE has been called before.
!------------------------------------------------------------------------------

      IF ( s%stage /= READY .AND. s%stage /= DONE ) THEN
         inform%status = NOT_INITIALIZED
         if ( s%level >= DEBUG ) WRITE( s%out, 1026 ) s%stage
         WRITE( inform%message( 1 ), 1027 )
         s%stage = VOID
         GO TO 999
      ELSE
         s%stage = READY
      END IF

!------------------------------------------------------------------------------
!     Verify that the starting point has been allocated.
!------------------------------------------------------------------------------

      IF ( .NOT. ALLOCATED(  problem% x ) ) THEN
         inform%status = UNASSOCIATED_INPUT
         WRITE( inform%message( 1 ), 1028 )
         GO TO 999
      END IF

!------------------------------------------------------------------------------
!     Determine whether or not all variables are free.
!------------------------------------------------------------------------------

      s%has_bounds = ALLOCATED( problem%x_l ) .OR. ALLOCATED( problem%x_u )

      IF ( s%has_bounds ) THEN

         IF ( .NOT. ALLOCATED(  problem%x_l ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1029 )
            GO TO 999
         END IF

         IF ( .NOT. ALLOCATED(  problem%x_u ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1030 )
            GO TO 999
         END IF

         IF ( .NOT. ALLOCATED(  problem%x_status ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1031 )
            GO TO 999
         END IF

         n_free  = 0
         n_lower = 0
         n_upper = 0
         n_range = 0
         n_fixed = 0

         DO j = 1, problem%n
            IF ( problem%x_l( j ) > - problem%infinity ) THEN
               IF ( problem%x_u( j ) <= problem%x_l( j ) ) THEN
                  n_fixed = n_fixed + 1
                  problem%x_status( j ) = FIXED
                  problem%x( j )        = problem%x_l( j )
               ELSE
                  IF ( problem%x_u( j ) < problem%infinity ) THEN
                     n_range = n_range + 1
                     problem%x_status( j ) = RANGE
                  ELSE
                     n_lower = n_lower + 1
                     problem%x_status( j ) = LOWER
                  END IF
               END IF
            ELSE
               IF ( problem%x_u( j ) < problem%infinity ) THEN
                  n_upper = n_upper + 1
                  problem%x_status( j ) = UPPER
               ELSE
                  n_free  = n_free + 1
                  problem%x_status( j ) = FREE
               END IF
            END IF
         END DO
         s%has_fixed  = n_fixed > 0
         n_bounded    = n_lower + n_upper + n_range
         s%has_bounds = n_bounded > 0
      END IF

      IF ( .NOT. s%has_bounds .AND. s%level >= ACTION ) WRITE( s%out, 1032 )

!------------------------------------------------------------------------------
!     Terminate if there are no bounds or constraints, in which case
!     problem%x solves the problem.
!------------------------------------------------------------------------------

      IF ( .NOT. s%has_bounds .AND. problem%m == 0 ) GO TO 999

!------------------------------------------------------------------------------
!     Check that the other necessary problem arrays have been allocated.
!------------------------------------------------------------------------------

      IF ( problem%m > 0 ) THEN

         IF ( .NOT. ASSOCIATED(  problem%c ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1033 )
            GO TO 999
         END IF

         IF ( .NOT. ALLOCATED(  problem%c_l ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1034 )
            GO TO 999
         END IF

         IF ( .NOT. ALLOCATED(  problem%c_u ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1035 )
            GO TO 999
         END IF

         IF ( .NOT. ALLOCATED(  problem%equation ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1036 )
            GO TO 999
         END IF

         IF ( .NOT. ALLOCATED(  problem%y ) ) THEN
            inform%status = UNASSOCIATED_INPUT
            WRITE( inform%message( 1 ), 1037 )
            GO TO 999
         END IF

         IF ( .NOT. control%external_J_products ) THEN

            IF ( .NOT. ASSOCIATED(  problem%J_val ) ) THEN
               inform%status = UNASSOCIATED_INPUT
               WRITE( inform%message( 1 ), 1038 )
               GO TO 999
            END IF

            IF ( .NOT. ALLOCATED(  problem%J_col ) ) THEN
               inform%status = UNASSOCIATED_INPUT
               WRITE( inform%message( 1 ), 1039 )
               GO TO 999
            END IF

            IF ( .NOT. ALLOCATED(  problem%J_row ) ) THEN
               inform%status = UNASSOCIATED_INPUT
               WRITE( inform%message( 1 ), 1040 )
               GO TO 999
            END IF

         END IF

      END IF

      IF ( .NOT. ASSOCIATED(  problem%g ) ) THEN
         inform%status = UNASSOCIATED_INPUT
         WRITE( inform%message( 1 ), 1041 )
         GO TO 999
      END IF

!------------------------------------------------------------------------------
!     Determine the actual declared dimension of the Jacobian in CUTEr
!------------------------------------------------------------------------------

      s%cuter_J_size = problem%J_ne + problem%n

!------------------------------------------------------------------------------
!     Determine whether or not all equations are equalities.
!------------------------------------------------------------------------------

      s%has_inequalities = .FALSE.
      DO i = 1, problem%m
         IF ( problem%c_l( i ) == problem%c_u( i ) ) THEN
            problem%equation( i ) = .TRUE.
         ELSE
            problem%equation( i ) = .FALSE.
            s%has_inequalities    = .TRUE.
         END IF
      END DO
      IF ( s%level >= ACTION .AND. s%has_inequalities ) WRITE( s%out, 1042 )

!------------------------------------------------------------------------------
!     The model voting space
!------------------------------------------------------------------------------

      IF ( control%model_type == AUTOMATIC ) THEN
         ALLOCATE( s%vote( control%model_inertia ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1043 ) control%model_inertia
            GO TO 999
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, 1044 ) control%model_inertia
         s%next_vote  = 1
         s%vote       = UNDEFINED
      END IF

!------------------------------------------------------------------------------
!     Define the equations grouping mode and allocate the necessary space.
!------------------------------------------------------------------------------

      s%n_items = problem%m + n_bounded

      SELECT CASE ( control%grouping )

      CASE ( NONE )

         s%p = s%n_items
         NULLIFY( control%group )

      CASE ( AUTOMATIC )

!        The number of groups

         IF ( control%nbr_groups < 0 ) THEN
            s%p = s%n_items /MIN( ABS( control%nbr_groups ), s%n_items )
         ELSE
            s%p = MIN( s%n_items, control%nbr_groups )
         END IF
         IF ( s%level >= ACTION ) WRITE( s%out, 1045 ) s%p

!        Allocate the group index.

         IF ( ASSOCIATED( s%aut_group ) ) DEALLOCATE( s%aut_group )
         ALLOCATE( s%aut_group( s%n_items), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1046 ) s%n_items
            GO TO 999
         END IF
         s%group => s%aut_group
         NULLIFY( control%group )
         IF ( s%level >= DEBUG ) WRITE( s%out, 1047 ) s%n_items

      CASE ( USER_DEFINED )

!        The group vector is assumed to be given in the integer
!        vector control%group( 1:control%nbr_groups ).
!        First verify the user-supplied number of groups.

         IF ( control%nbr_groups <= 0 .OR. control%nbr_groups > s%n_items ) THEN
            inform%status = WRONG_P
            WRITE( inform%message( 1 ), 1048 ) control%nbr_groups
            GO TO 999
         END IF

!        Verify the user-supplied group information.

         IF ( .NOT. ASSOCIATED( control%group ) ) THEN
            inform%status = USER_GROUPS_UNDEFINED
            WRITE( inform%message( 1 ), 1049 )
            GO TO 999
         END IF

         DO i = 1, problem%m
            k = control%group( i )
            IF ( k <= 0 .OR. k > control%nbr_groups ) THEN
               inform%status = WRONG_USER_GROUP_INDEX
               WRITE( inform%message( 1 ), 1050 ) i
               WRITE( inform%message( 2 ), 1051 ) k
               GO TO 999
            END IF
         END DO
         nxt  = problem%m
         DO j = 1, problem%n
            SELECT CASE ( problem%x_status( j ) )
            CASE ( LOWER, UPPER, RANGE )
               nxt = nxt + 1
               k   = control%group( nxt )
               IF ( k <= 0 .OR. k > control%nbr_groups ) THEN
                  inform%status = WRONG_USER_GROUP_INDEX
                  WRITE( inform%message( 1 ), 1052 ) j
                  WRITE( inform%message( 2 ), 1051 ) k
                  GO TO 999
               END IF
            END SELECT
         END DO
         IF ( s%level >= DEBUG ) WRITE( s%out, 1053 ) control%nbr_groups

!        The user-supplied grouping information seems correct. Use it.

         s%p = control%nbr_groups
         s%group => control%group

      END SELECT

!------------------------------------------------------------------------------
!     Real workspace for the algorithm
!------------------------------------------------------------------------------

!     Allocate the vector theta, that defines the filter measure.

      IF ( ASSOCIATED( s%theta ) ) DEALLOCATE( s%theta )
      ALLOCATE( s%theta( s%p ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
        inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), 1054 ) s%p
         GO TO 999
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, 1055 ) s%p

!     Allocate the storage necessary to remember the best point found so far,
!     if that option is activated.

      IF ( control%save_best_point ) THEN
         IF ( ASSOCIATED( s%best_x ) ) DEALLOCATE( s%best_x )
         ALLOCATE( s%best_x( problem%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1056 ) problem%n
            GO TO 999
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, 1057 ) problem%n
      END IF

!     Allocate the real workspace vectors.

      IF ( ASSOCIATED( s%step ) ) DEALLOCATE( s%step )
      ALLOCATE( s%step( problem%n ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), 1058 ) problem%n
         GO TO 999
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, 1059 ) problem%n

      dim = MAX( problem%n, s%n_items )

      IF ( ASSOCIATED( s%v ) ) DEALLOCATE( s%v )
      ALLOCATE( s%v( dim ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), 1060 ) dim
         GO TO 999
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, 1061 ) dim

      IF ( control%external_J_products ) THEN

         IF ( s%has_fixed .OR. s%has_inequalities ) THEN

            IF ( ASSOCIATED( s%t ) ) DEALLOCATE( s%t )
            ALLOCATE( s%t( dim ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1062 ) dim
               GO TO 999
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, 1063 ) dim

         END IF

      END IF

      IF ( problem%m == 0 .OR. control%external_J_products ) THEN

         IF ( ASSOCIATED( s%u ) ) DEALLOCATE( s%u )
         ALLOCATE( s%u( problem%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1064 ) problem%n
            GO TO 999
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, 1065 ) problem%n
         s%u_allocated = .TRUE.

      ELSE

         s%u => problem%J_val( problem%J_ne+1:s%cuter_J_size )
         s%u_allocated = .FALSE.

      END IF

      IF ( ASSOCIATED( s%w ) ) DEALLOCATE( s%w )
      ALLOCATE( s%w( dim ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), 1066 ) dim
         GO TO 999
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, 1067 ) dim

      dim = MAX( problem%n, s%p )
      IF ( s%has_inequalities ) dim = MAX( dim, problem%m )
      IF ( ASSOCIATED( s%r ) ) DEALLOCATE( s%r )
      ALLOCATE( s%r( dim ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), 1068 ) dim
         GO TO 999
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, 1069 ) dim

!------------------------------------------------------------------------------
!     Integer workspace for the algorithm
!------------------------------------------------------------------------------

      dim = 0
      IF ( control%grouping  /= NONE   ) dim = s%n_items
      IF ( control%prec_used == BANDED ) dim = MAX( dim, problem%n + 1 )
      IF ( .NOT. control%filter_sign_restriction .AND. &
           ( s%has_bounds .OR. s%has_inequalities )    ) dim = MAX( dim, s%p )
      IF ( dim > 0 ) THEN
         IF ( ASSOCIATED( s%iw ) ) DEALLOCATE( s%iw )
         ALLOCATE( s%iw( dim ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1070 ) dim
            GO TO 999
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, 1071 ) dim
      END IF

!------------------------------------------------------------------------------
!
!     The initial point and must be obtained from saved checkpointing
!     information, overwriting the value of problem%x provided on input.
!
!------------------------------------------------------------------------------

      IF ( control%restart_from_checkpoint ) THEN

         OPEN( UNIT   = control%checkpoint_dev, FILE = control%checkpoint_file,&
               STATUS = 'OLD', IOSTAT = iostat )
         IF ( iostat > 0 ) THEN
            inform%status = FILE_NOT_OPENED
            WRITE( inform%message( 1 ), 1072 ) control%checkpoint_file
            GO TO 999
         END IF

!        Read the iteration number.

         READ( UNIT = control%checkpoint_dev, FMT = 1200, IOSTAT = iostat )    &
               inform%nbr_iterations
         IF ( iostat > 0 ) THEN
            inform%status = CHECKPOINTING_ERROR
            WRITE( inform%message( 1 ), 1073 ) control%checkpoint_file
            GO TO 999
         END IF

         IF ( s%level >= DEBUG ) WRITE( s%out, 1074 ) inform%nbr_iterations

!        Read the values of the variables.

         DO j = 1, problem%n
            READ( UNIT = control%checkpoint_dev, FMT = 1201, IOSTAT = iostat ) &
                  nxt, problem%x( j )
            IF ( iostat > 0 ) THEN
               inform%status = CHECKPOINTING_ERROR
               WRITE( inform%message( 1 ), 1073 ) control%checkpoint_file
               GO TO 999
            END IF
         END DO

!        Read the trust-region radius.

         READ( control%checkpoint_dev, 1202 ) rstring, control%initial_radius
         IF ( iostat > 0 .OR. rstring /= 'radius' ) THEN
            inform%status = CHECKPOINTING_ERROR
            WRITE( inform%message( 1 ), 1073 ) control%checkpoint_file
            GO TO 999
         END IF

         IF ( s%level >= DEBUG ) WRITE( s%out, 1075 ) control%initial_radius

         CLOSE( control%checkpoint_dev )

      ELSE
         inform%nbr_iterations = 0
      END IF

!------------------------------------------------------------------------------
!     Get the initial values of the residuals and the initial Jacobian.
!------------------------------------------------------------------------------

      inform%nbr_c_evaluations = 0
      inform%nbr_J_evaluations = 0
      IF ( problem%m > 0 ) THEN
         IF ( control%external_J_products ) THEN
            inform%status = GET_C_0
         ELSE
            inform%status = GET_C_AND_J_0
         END IF
         RETURN
      END IF

!*****************************************************
200  CONTINUE ! *** Reverse communication re-entry ***
!*****************************************************

      IF ( problem%m > 0 ) THEN
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, 1003 )
            SELECT CASE ( inform%status )
            CASE ( GET_C_AND_J_0 )
               WRITE( s%out, 1005 )
            CASE ( GET_C_0 )
               WRITE( s%out, 1006 )
            CASE DEFAULT
               WRITE( s%out, 1021 ) inform%status
            END SELECT
            WRITE( s%out, 1003 )
         END IF
         inform%status            = OK
         inform%nbr_c_evaluations = inform%nbr_c_evaluations + 1
         IF ( .NOT. control%external_J_products )                              &
            inform%nbr_J_evaluations = inform%nbr_J_evaluations + 1
         s%goth                   = .FALSE.
      END IF

!------------------------------------------------------------------------------
!     Balance the groups, if requested.
!------------------------------------------------------------------------------

      IF ( control%grouping == AUTOMATIC ) THEN

         IF ( control%balance_group_values ) THEN

            DO i = 1, s%n_items
               s%iw( i ) = i
            END DO

            DO i = 1, problem%m
               s%v( i ) = ABS( FILTRANE_c_violation( i ) )
            END DO
            IF ( s%has_bounds ) THEN
               nxt = problem%m
               DO j = 1, problem%n
                  SELECT CASE ( problem%x_status( j ) )
                  CASE ( LOWER, UPPER, RANGE )
                     nxt = nxt + 1
                     s%v( nxt ) = ABS( FILTRANE_x_violation( j ) )
                  END SELECT
               END DO
            END IF

!           Sort the violations in ascending order.

            CALL SORT_quicksort( s%n_items, s%v, SORT_exitcode, ix = s%iw )
            IF ( SORT_exitcode /= 0 ) THEN
               inform%status = SORT_TOO_LONG
               WRITE( inform%message( 1 ), 1076 )
               WRITE( inform%message( 2 ), 1077 )
               GO TO 999
            END IF
            CALL SORT_inplace_invert( s%n_items, s%iw )

!           Use the sorted values to define the groups. This attempts to
!           produce groups with approximately balanced constraints values.

            DO i = 1, problem%m
               nxt = nxt + 1
               k = MOD( i, s%p )
               IF ( k == 0 ) k = s%p
               s%group( s%iw( i ) ) = k
            END DO
            IF ( s%has_bounds ) THEN
               nxt = problem%m
               DO j = 1, problem%n
                  SELECT CASE ( problem%x_status( j ) )
                  CASE ( LOWER, UPPER, RANGE )
                     nxt = nxt + 1
                     k = MOD( nxt, s%p )
                     IF ( k == 0 ) k = s%p
                     s%group( s%iw( nxt ) ) =  k
                  END SELECT
               END DO
            END IF

!        Don't balance group values, just group them into s%p batches.

         ELSE
            DO i = 1, problem%m
               k = MOD( i, s%p )
               IF ( k == 0 ) k = s%p
               s%group( i ) = k
            END DO
            IF ( s%has_bounds ) THEN
               nxt = problem%m
               DO j = 1, problem%n
                  SELECT CASE ( problem%x_status( j ) )
                  CASE ( LOWER, UPPER, RANGE )
                     nxt = nxt + 1
                     k = MOD( nxt, s%p )
                     IF ( k == 0 ) k = s%p
                     s%group( nxt ) =  k
                  END SELECT
               END DO
            END IF
         END IF
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, 1078 )
            CALL TOOLS_output_vector( problem%m, s%group, s%out )
         END IF
      END IF

!------------------------------------------------------------------------------
!     Establish the group status, if necessary.
!------------------------------------------------------------------------------

!     First determine the type of filter sign restriction situation.
!     1) sign restriction for all items

      IF ( control%filter_sign_restriction ) THEN
         s%filter_sign = RESTRICTED

!     2) no automatic sign restriction

      ELSE

!        See if one has only one item per group

         IF ( control%grouping == NONE ) THEN
            s%filter_sign = UNRESTRICTED
         ELSE
            s%filter_sign = MIXED
         END IF
      END IF

      IF ( s%level >= DETAILS ) THEN
         SELECT CASE ( s%filter_sign )
         CASE ( RESTRICTED )
            WRITE( s%out, 1079 )
         CASE ( UNRESTRICTED )
            WRITE( s%out, 1080 )
         CASE ( MIXED )
            WRITE( s%out, 1081 )
         END SELECT
      END IF

!     Allocate the group status, if either we have mixed type groups, or if
!     there are multiple groups.

      IF ( s%filter_sign == MIXED .OR. control%grouping /= NONE ) THEN
         IF ( ASSOCIATED( s%g_status ) ) DEALLOCATE( s%g_status )
         ALLOCATE( s%g_status( s%p ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1082 ) s%p
            GO TO 999
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, 1083 ) s%p

!        Define the status of each group of constraints.

         s%iw( 1:s%p ) = 0
         DO i = 1, problem%m
            SELECT CASE ( control%grouping )
            CASE ( NONE )
               ig = i
            CASE ( AUTOMATIC, USER_DEFINED )
               ig = s%group( i )
            END SELECT
            IF ( s%iw( ig ) /= 0 ) THEN
               s%g_status( ig ) = MULTIPLE
            ELSE
               IF ( s%filter_sign == RESTRICTED ) THEN
                  s%g_status( ig ) = SINGLE_RESTRICTED
               ELSE
                  s%g_status( ig ) = SINGLE_UNRESTRICTED
               END IF
            END IF
            s%iw( ig ) = s%iw( ig ) + 1
         END DO

!        Define the status of each group of bounds.

         IF ( s%has_bounds ) THEN
            nxt = problem%m
            DO j = 1, problem%n
               SELECT CASE ( problem%x_status( j ) )
               CASE ( LOWER, UPPER, RANGE )
                  nxt = nxt + 1
                  SELECT CASE ( control%grouping )
                  CASE ( NONE )
                     ig = nxt
                  CASE ( AUTOMATIC )
                     ig = s%group( nxt )
                  CASE ( USER_DEFINED )
                  END SELECT
                  IF ( s%iw( ig ) /= 0 ) THEN
                     s%g_status( ig ) = MULTIPLE
                  ELSE
                     IF ( s%filter_sign == RESTRICTED ) THEN
                        s%g_status( ig ) = SINGLE_RESTRICTED
                     ELSE
                        s%g_status( ig ) = SINGLE_UNRESTRICTED
                     END IF
                  END IF
                  s%iw( ig ) = s%iw( ig ) + 1
               END SELECT
            END DO
         END IF

         IF ( s%level >= DETAILS ) THEN
            WRITE( s%out, 1084 ) MAXVAL(s%iw(1:s%p))
            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, 1085 )
               CALL TOOLS_output_vector( s%p, s%iw, s%out )
            END IF
         END IF

      END IF

!------------------------------------------------------------------------------
!     Compute the initial value of the objective function and its gradient.
!------------------------------------------------------------------------------

      CALL FILTRANE_compute_theta
      CALL FILTRANE_compute_f

!*****************************************************
250  CONTINUE ! *** Reverse communication re-entry ***
!*****************************************************

      IF ( inform%status == GET_JTC_0 ) THEN
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, 1003 )
            WRITE( s%out, 1007 )
            WRITE( s%out, 1003 )
         END IF
         inform%status = GET_JTC
      END IF
      CALL FILTRANE_compute_grad_f
      SELECT CASE ( inform%status )
      CASE ( :-1 )
         GO TO 999
      CASE ( GET_JTC )
         inform%status = GET_JTC_0
         RETURN
      END SELECT
      s%g_norm2     = NRM2( problem%n, problem%g, 1 )
      s%g_norminf_u = MAXVAL( ABS( problem%g( 1:problem%n ) ) )

!     Compute the maximum value that is acceptable for the objective function.

      s%f_max = MIN( THOUSAND * THOUSAND * problem%f, THOUSAND + problem%f )

!     Save the initial point as the best found iterate, if saving the best
!     point so far is requested.

      IF ( control%save_best_point ) THEN
         s%best_x    = problem%x
         s%best_fx   = problem%f
         IF ( s%level >= DEBUG ) WRITE( s%out, 1086 )
      END IF

!------------------------------------------------------------------------------
!     Allocate space for the band preconditioner.
!------------------------------------------------------------------------------

      IF ( control%prec_used == BANDED ) THEN

!        Verify that the Jacobian is known, and switch to no preconditioning
!        if this is not the case.

         IF ( control%external_J_products ) THEN
            control%prec_used = NONE
            IF ( s%level >= ACTION ) WRITE( s%out, 1087 )
         ELSE

!           Allocate the necessary workspace for extracting the band from J^TJ.

            IF ( ASSOCIATED( s%row ) ) DEALLOCATE( s%row )
            ALLOCATE( s%row( problem%J_ne ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1088 )  problem%J_ne
               GO TO 999
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, 1089 ) problem%J_ne

            IF ( ASSOCIATED( s%perm ) ) DEALLOCATE( s%perm )
            ALLOCATE( s%perm( problem%J_ne ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1090 ) problem%J_ne
               GO TO 999
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, 1091 ) problem%J_ne
         END IF
      END IF

!     Allocate the diagonal and off-diagonal parts of the preconditioner
!     itself.

      IF ( control%prec_used == BANDED ) THEN

!        Allocate the diagonal.

         IF ( ASSOCIATED( s%diag ) ) DEALLOCATE( s%diag )
         ALLOCATE( s%diag( problem%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1092 ) problem%n
            GO TO 999
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, 1093 ) problem%n

!        Compute the effective semi-bandwidth.

         s%nsemib = MIN( control%semi_bandwidth, problem%n - 1 )

!        Allocate the off-diagonal part.

         IF ( control%semi_bandwidth > 0 ) THEN
            IF ( ASSOCIATED( s%offdiag ) ) DEALLOCATE( s%offdiag )
            ALLOCATE( s%offdiag( s%nsemib, problem%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1094 ) s%nsemib, problem%n
               GO TO 999
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, 1095 ) s%nsemib, problem%n
         END IF
      END IF

!------------------------------------------------------------------------------
!     Compute the preconditioner.
!------------------------------------------------------------------------------

     SELECT CASE ( control%prec_used )

     CASE ( NONE )

     CASE ( BANDED )

!       Compute the structure of the Jacobian (in a sparse storage by rows,
!       with column indices in ascending order for each row).

        CALL FILTRANE_build_sparse_J
        IF ( inform%status /= OK ) THEN
           GO TO 999
        END IF

!       Extract the preconditioning matrix at the initial point.

        CALL FILTRANE_build_band_JTJ( s%bandw )

        IF ( s%level >= DEBUG ) THEN
           WRITE( s%out, 1096 ) s%nsemib, s%bandw
           DO i = 1, problem%n
              k = MIN( s%bandw, problem%n - i )
              WRITE( s%out, 1097 ) i, s%diag( i ),                             &
                                 ( s%offdiag( j, i ), j = 1, MIN( k, 5 ) )
              IF ( k > 5 ) WRITE( s%out, 1098 ) ( s%offdiag( j, i ), j = 6, k )
           END DO
        END IF

!       Compute its factors.

        CALL BAND_factor( problem%n, s%bandw, s%diag, s%offdiag, s%nsemib,     &
                          BAND_status )

        IF ( s%level >= DEBUG ) THEN
           WRITE( s%out, 1155 )
           DO i = 1, problem%n
              k = MIN( s%nsemib, problem%n - i )
              WRITE( s%out, 1097 ) i, s%diag( i ),                             &
                                ( s%offdiag( j, i ), j = 1, MIN( k, 5 ) )
              IF ( k > 5 ) WRITE( s%out, 1098 ) ( s%offdiag( j, i ), j = 6, k )
           END DO
        END IF

     END SELECT

!------------------------------------------------------------------------------
!    Compute the norm of the initial (preconditioned) gradient.
!------------------------------------------------------------------------------

     SELECT CASE ( control%prec_used )

     CASE ( NONE )

        s%g_norm    = s%g_norm2
        s%g_norminf = s%g_norminf_u

     CASE ( BANDED )

        s%v( 1:problem%n ) = problem%g
        CALL BAND_solve( problem%n, s%bandw, s%diag, s%offdiag,                &
                         s%nsemib, s%v, BAND_status )
        !s%g_norm = SQRT( INNER_PRODUCT( problem%n, problem%g, 1,               &
        !                                    s%v( 1:problem%n ), 1 ) )
        s%g_norm = SQRT( DOT_PRODUCT( problem%g, s%v( 1:problem%n ) ) )

        IF ( s%nsemib == 0 ) THEN
           s%g_norminf = ZERO
           DO j = 1, problem%n
              s%g_norminf = MAX( s%g_norminf, problem%g( j ) * s%v( j ) )
           END DO
           s%g_norminf = SQRT( s%g_norminf )
        END IF

     CASE ( USER_DEFINED )

        s%v( 1:problem%n ) = problem%g
        s%RC_Pv => s%v
        inform%status = GET_PREC_G_0
        RETURN

     END SELECT

!*****************************************************
270  CONTINUE ! *** Reverse communication re-entry ***
!*****************************************************

     IF ( control%prec_used == USER_DEFINED ) THEN

        IF ( s%level >= DEBUG ) THEN
           WRITE( s%out, 1003 )
           SELECT CASE ( inform%status )
           CASE ( GET_PREC_G_0 )
              WRITE( s%out, 1008 )
           CASE DEFAULT
               WRITE( s%out, 1021 ) inform%status
           END SELECT
           WRITE( s%out, 1003 )
        END IF
        inform%status = OK
        NULLIFY( s%RC_Pv )
        !s%g_norm = SQRT( INNER_PRODUCT( problem%n, problem%g, 1,               &
        !                                  s%v( 1:problem%n ), 1 ) )
        s%g_norm = SQRT( DOT_PRODUCT( problem%g, s%v( 1:problem%n ) ) )
     END IF

!------------------------------------------------------------------------------
!    Other algorithmic initializations
!------------------------------------------------------------------------------

!  Set the initial trust-region radius, for the case where
!  the filter is not used.

     s%radius      = control%initial_radius
     s%prev_radius = ZERO

!  Initialize the CG iteration counter.

     inform%nbr_cg_iterations = 0

!  Set the number of values in the filter and its capacity to zero.

     s%active_filter       = 0
     s%filter_size         = 0
     s%filter_nbr_inactive = 0
     s%filter_capacity     = 0

!------------------------------------------------------------------------------
!  Write information about the starting point and values.
!------------------------------------------------------------------------------

     IF ( s%level >= TRACE ) THEN
        WRITE( s%out, 1099 )
        WRITE( s%out, 1100 )
        WRITE( s%out, 1099 )
        WRITE( s%out, 1000 ) inform%nbr_iterations, problem%f, s%g_norm,       &
                             s%radius, inform%nbr_cg_iterations, s%filter_size
        IF ( s%level >= DETAILS ) THEN
           WRITE( s%out, 1099 )
           WRITE( s%out, 1101 )
           CALL TOOLS_output_vector( problem%n, problem%x, s%out )
           WRITE( s%out, 1102 )
           CALL TOOLS_output_vector( problem%n, problem%g, s%out )
           IF ( problem%m > 0 ) THEN
              WRITE( s%out, 1103 )
              CALL TOOLS_output_vector( problem%m, problem%c, s%out )
           END IF
           IF ( s%has_bounds .OR. s%has_inequalities .OR. &
                control%grouping /= NONE                  ) THEN
              WRITE( s%out, 1104 )
              CALL TOOLS_output_vector( s%p, s%theta, s%out )
           END IF
           IF ( s%level >= DEBUG .AND. problem%J_ne > 0 ) THEN
              WRITE( s%out, 1105 )
              CALL TOOLS_output_matrix_C( problem%J_ne, problem%J_val,&
                                          problem%J_row, problem%J_col, s%out )
           END IF
        END IF
     ENDIF

!------------------------------------------------------------------------------
!  Test for convergence.
!------------------------------------------------------------------------------

     sqrtn = problem%n
     sqrtn = SQRT( sqrtn )
     IF ( control%stop_on_prec_g ) THEN
        IF ( control%stop_on_g_max ) THEN
           IF ( s%g_norminf <= control%g_accuracy   ) THEN
              WRITE( inform%message( 1 ), 1159 )
              GO TO 999
           END IF
        ELSE
           IF ( s%g_norm <= sqrtn * control%g_accuracy   ) THEN
              WRITE( inform%message( 1 ), 1159 )
              GO TO 999
           END IF
        END IF
     ELSE
        IF ( control%stop_on_g_max ) THEN
           IF ( s%g_norminf_u <= control%g_accuracy  ) THEN
              WRITE( inform%message( 1 ), 1159 )
              GO TO 999
           END IF
        ELSE
           IF ( s%g_norm2 <= sqrtn * control%g_accuracy  ) THEN
              WRITE( inform%message( 1 ), 1159 )
              GO TO 999
           END IF
        END IF
     END IF
     IF ( ALL( ABS( s%theta ) <= control%c_accuracy ) ) THEN
        WRITE( inform%message( 1 ), 1160 )
        GO TO 999
     END IF

!------------------------------------------------------------------------------
!  Final initializations
!------------------------------------------------------------------------------

!  Ensure that the margin is small enough.

     rp = s%p
     control%gamma_f = MIN( control%gamma_f,  HALF / SQRT( rp ) )

!  Compute the norm of the current iterate.

     s%x_norm = NRM2( problem%n, problem%x, 1 )

!  Unrestrict the first step and indicate that no previous trust-region
!  step has been taken.

     s%use_filter    = control%use_filter /= NEVER
     IF ( control%weak_accept_power == -666.0_wp ) THEN
        s%restrict = .FALSE.
     ELSE
        s%restrict = .NOT. s%use_filter
     END IF
     s%extent        = control%itr_relax
     s%first_TR_step = .TRUE.
     s%unsuccess     = .FALSE.

!  Set the initial model.

     SELECT CASE ( control%model_type )
     CASE ( AUTOMATIC )
        s%model_used = GAUSS_NEWTON
     CASE ( GAUSS_NEWTON )
        s%model_used = GAUSS_NEWTON
     CASE ( NEWTON )
        s%model_used = NEWTON
     END SELECT

!==============================================================================
!==============================================================================

2000 CONTINUE  !               MAIN FILTRANE LOOP

!==============================================================================
!==============================================================================

!       Update the iteration counter.

        IF ( control%max_iterations >= 0 ) THEN
           IF ( inform%nbr_iterations >= control%max_iterations ) THEN
              inform%status = MAX_ITERATIONS_REACHED
              WRITE( inform%message( 1 ), 1106 )
              GO TO 999
           END IF
        END IF
        inform%nbr_iterations = inform%nbr_iterations + 1

!       -----------------------------------------------------------------------

!       Update the printing level

!       -----------------------------------------------------------------------

        IF ( inform%nbr_iterations >= control%start_print ) THEN
           IF ( control%stop_print < 0                      .OR. &
                inform%nbr_iterations <= control%stop_print      ) THEN
              s%level = s%print_level
           ELSE
              s%level = SILENT
           END IF
        ELSE
           s%level = SILENT
        END IF

!       -----------------------------------------------------------------------

!       Set the accuracy indicator to that requested by the user.

!       -----------------------------------------------------------------------

        s%step_accuracy = control%subproblem_accuracy

!       -----------------------------------------------------------------------

!       Compute the step

!       -----------------------------------------------------------------------

!       Loop on the possibility that negative curvature is found, which
!       requires the step to be restricted, or on the possibility that the
!       step might be negligible, in which case it must at least be computed
!       to full accuracy.

3000    CONTINUE

!          See if the step is restricted or unrestricted.

           IF ( s%restrict ) THEN
              s%gltr_radius = s%radius
              s%it_status = ' R  '
           ELSE
              s%gltr_radius = s%extent * s%radius
              s%it_status = ' U  '
           END IF
           IF ( s%level >= DEBUG ) THEN
              WRITE( s%out, 1124 ) s%gltr_radius
           END IF

!          Set the part of the iteration status telling which model
!          is being used.

           SELECT CASE ( s%model_used )
           CASE ( NEWTON )
              s%it_status( 1:1 ) = 'N'
           CASE ( GAUSS_NEWTON )
              s%it_status( 1:1 ) = 'G'
           END SELECT

!          See if this is a GLTR reentry with everything unchanged except
!          a smaller trust-region radius. If this the case, GLTR can produce
!          the subproblem solution in the largest Krylov space explored at the
!          previous iteration (if re-entered with info_status = 4).

           IF ( s%gltr_radius < s%prev_radius .AND. s%unsuccess ) THEN

              s%GLTR_info%status  = 4
              s%w( 1: problem%n ) = s%r( 1:problem%n )
              IF ( s%level >= DEBUG ) WRITE( s%out, 1107 ) s%prev_radius

!          If this is not the case, prepare a first call to GLTR.

           ELSE

!             Initialize GLTR, if not done already

              IF ( .NOT. s%gltr_initialized ) THEN
                 CALL GLTR_initialize( s%GLTR_data, s%GLTR_control )
                 s%GLTR_control%lanczos_itmax = 5
                 s%gltr_initialized = .TRUE.
                 IF ( s%level >= DEBUG ) WRITE( s%out, 1108 )
              END IF

!             Set the maximum number of GLTR iterations.

              s%GLTR_control%itmax = control%max_cg_iterations * problem%n

!             Set the GLTR preconditioning indicator.

              IF ( control%prec_used /= NONE ) s%GLTR_control%unitm = .FALSE.

!             Make sure the absolute requested precision on the subproblem is
!             not much higher than the final precision required for the problem.

              sqrtn = problem%n
              sqrtn = SQRT( sqrtn )
              s%GLTR_control%stop_absolute = SQRT( EPSMACH )
              IF ( control%stop_on_prec_g .OR. control%prec_used == NONE ) THEN
                 IF ( control%stop_on_g_max ) THEN
                    s%GLTR_control%stop_absolute                               &
                       = MAX( s%GLTR_control%stop_absolute,                    &
                              TENTH * control%g_accuracy )
                 ELSE
                    s%GLTR_control%stop_absolute                               &
                       = MAX( s%GLTR_control%stop_absolute,                    &
                              TENTH * sqrtn * control%g_accuracy )
                 END IF
              END IF

!             Adaptively increase the required relative precision on the
!             solution of the subproblem.

              SELECT CASE ( s%step_accuracy )
              CASE ( ADAPTIVE )
                 IF ( s%g_norm < ONE .AND. control%gltr_accuracy_power > ZERO )&
                    THEN
                    fwp = s%g_norm ** control%gltr_accuracy_power
                 ELSE
                    fwp = ONE
                 END IF
                 s%GLTR_control%stop_relative = MIN(control%min_gltr_accuracy, &
                                                    MAX( SQRT( EPSMACH), fwp ) )
              CASE ( FULL )
                 s%GLTR_control%stop_relative = SQRT( EPSMACH )
              END SELECT

              IF ( s%level >= ACTION ) THEN
                 WRITE( s%out, 1109 ) s%GLTR_control%stop_absolute,            &
                                      s%GLTR_control%stop_relative
              END IF

           END IF

!          Set the GLTR print level.

           IF ( s%level >= ACTION ) THEN
              IF ( s%level >= DETAILS ) THEN
                 WRITE ( s%out, 1110 )s%gltr_radius,s%GLTR_control%stop_relative
                 IF ( s%level >= DEBUG ) THEN
                    s%GLTR_control%print_level = 1
                    s%GLTR_control%out = s%out
                 END IF
              END IF
           END IF

!          Remember the current radius.

           s%prev_radius = s%gltr_radius

!          Set the gradient.

           s%u( 1:problem%n ) = problem%g

!          The GLTR iterations

4000       CONTINUE

              IF ( s%level >= CRAZY ) THEN
                  WRITE( s%out, 1111 )
                  WRITE( s%out, 1112 ) s%gltr_radius,                          &
                                       s%GLTR_control%stop_absolute,           &
                                       s%GLTR_control%stop_relative
                  WRITE( s%out, 1113 ) s%model_value
                  WRITE( s%out, 1114 )
                  CALL TOOLS_output_vector( problem%n, s%step, s%out )
                  WRITE( s%out, 1115 )
                  CALL TOOLS_output_vector( problem%n, s%u, s%out )
                  WRITE( s%out, 1116 )
                  CALL TOOLS_output_vector( problem%n, s%w, s%out )
                  WRITE( s%out, 1157 )
              END IF
              CALL GLTR_solve( problem%n, s%gltr_radius, s%model_value,s%step, &
                               s%u( 1:problem%n ), s%w( 1:problem%n ),         &
                               s%GLTR_data, s%GLTR_control, s%GLTR_info )
              IF ( s%level >= CRAZY ) THEN
                  WRITE( s%out, 1117 ) s%GLTR_info%status
              END IF

!             .................................
!             Restart with the initial gradient
!             .................................

              IF ( s%GLTR_info%status == 5 ) THEN

                 s%u = problem%g
                 IF ( s%level >= DEBUG ) WRITE( s%out, 1118 )
                 GO TO 4000

!             ......................
!             Successful termination
!             ......................

              ELSE IF ( s%GLTR_info%status >= -2 .AND. &
                        s%GLTR_info%status <=  0       ) THEN

                 s%s_norm = s%GLTR_info%mnormx

                 IF ( s%GLTR_info%iter >= s%GLTR_control%itmax ) THEN
                    s%it_status( 3:3 ) = 'M'
                 ELSE IF ( s%s_norm < 0.99_wp * s%radius ) THEN
                    s%it_status( 3:3 ) = 'I'
                 ELSE IF ( s%s_norm < 1.01_wp * s%radius ) THEN
                    s%it_status( 3:3 ) = 'B'
                 ELSE
                    s%it_status( 3:3 ) = 'E'
                 END IF
                 s%s_norm2 = NRM2( problem%n, s%step, 1 )
                 s%r( 1: problem%n ) = s%w( 1:problem%n )

                 GO TO 4001

!             ......................
!             Error return from GLTR
!             ......................

              ELSE IF ( s%GLTR_info%status <= -3 ) THEN

                 inform%status = GLTR_ERROR
                 WRITE( inform%message( 1 ), 1119 )
                 WRITE( inform%message( 2 ), 1120 ) s%GLTR_info%status
                 GO TO 999

!             ................................
!             Form the preconditioned gradient.
!             ................................

              ELSE IF ( s%GLTR_info%status == 2 .OR. &
                        s%GLTR_info%status == 6      ) THEN

                 SELECT CASE ( control%prec_used )

                 CASE ( BANDED )

                    CALL BAND_solve( problem%n, s%bandw, s%diag, s%offdiag,    &
                                     s%nsemib, s%w, BAND_status)
                    IF ( s%level >= CRAZY ) WRITE( s%out, 1121 )

                 CASE ( USER_DEFINED )

                    s%RC_Pv => s%w
                    inform%status = GET_PREC
                    RETURN

                 END SELECT

              END IF

!**************************************************************
300           CONTINUE ! *** Reverse communication re-entry ***
!**************************************************************

              IF ( s%GLTR_info%status == 2 .OR. &
                   s%GLTR_info%status == 6      ) THEN
                 IF ( control%prec_used == USER_DEFINED ) THEN
                    IF ( s%level >= DEBUG ) THEN
                       WRITE( s%out, 1003 )
                       SELECT CASE (inform%status )
                       CASE ( GET_PREC )
                          WRITE( s%out, 1009 )
                       CASE DEFAULT
                          WRITE( s%out, 1021 ) inform%status
                       END SELECT
                       WRITE( s%out, 1003 )
                    END IF
                    inform%status = OK
                    NULLIFY( s%RC_Pv )
                    IF ( s%level >= CRAZY ) WRITE( s%out, 1122 )
                 END IF
	         GO TO 4000

!             ............................................................
!             Last possibility: form the matrix-vector product
!                               w <-- Hessian . w
!             The Hessian is:
!             - for equalities:
!               > J^TJ                       for the  Gauss-Newton model,
!               > J^TJ + SUM c_i nabla^2 c_i for the full Newton model,
!
!             - for violated inequalities:
!               > J^T V**alpha J             for the  Gauss-Newton model,
!               > J^T V**alpha J  + gamma * SUM v**beta nabla^2 c_i
!                                     for the full Newton model
!             - for violated bounds:
!               > alpha * V ** beta
!             ............................................................

              ELSE IF (  s%GLTR_info%status == 3 .OR. &
                         s%GLTR_info%status == 7      ) THEN

!                Set the work vector s%r to zero.  This vector is used to
!                accumulate the components of the product that do not depend
!                on JTJ, i.e. the terms involving nabla^2 c_i and the
!                contribution of the scaled identity matrix for the
!                variables that violate their bounds.

                 IF ( s%has_bounds ) s%r = ZERO

!                Compute the terms in nabla^2 c_i if the full Newton model
!                is used.

                 IF ( problem%m > 0 ) THEN

                    IF ( s%model_used == NEWTON ) THEN

!                      Define the vector multiplying the individual
!                      Hessians.

                       IF ( .NOT. s%goth ) THEN
                          CALL FILTRANE_compute_Hmult( problem%c )
                       END IF

!                      Compute the product.

                       IF ( s%level >= CRAZY ) THEN
                          WRITE( s%out, 1123 )  s%goth
                       END IF

                       s%RC_v    => s%w( 1:problem%n )
                       s%RC_Mv   => s%r( 1:problem%n )
                       s%RC_newx = .NOT. s%goth
                       inform%status = GET_MP1
                       RETURN
                    END IF
                 END IF
              END IF

!**************************************************************
500           CONTINUE ! *** Reverse communication re-entry ***
!**************************************************************

              IF (  s%GLTR_info%status == 3 .OR. s%GLTR_info%status == 7 ) THEN
                 IF ( problem%m > 0 .AND. s%model_used == NEWTON ) THEN
                    IF ( s%level >= DEBUG ) THEN
                       WRITE( s%out, 1003 )
                       SELECT CASE ( inform%status )
                       CASE ( GET_MP1 )
                           WRITE( s%out, 1013 )
                       CASE DEFAULT
                           WRITE( s%out, 1021 ) inform%status
                       END SELECT
                       WRITE( s%out, 1003 )
                    END IF
                    inform%status = OK
                    NULLIFY( s%RC_v, s%RC_Mv )
                    s%goth = .TRUE.
                 END IF

!                Now consider the product of J times v.

                 IF ( problem%m > 0 ) THEN

                    IF ( control%external_J_products ) THEN

                       IF ( s%has_fixed ) THEN
                          DO j = 1, problem%n
                             IF ( problem%x_status( j ) == FIXED ) THEN
                                s%t( j ) = ZERO
                             ELSE
                                s%t( j ) = s%w( j )
                             END IF
                          END DO
                          s%RC_v => s%t( 1:problem%n )
                       ELSE
                          s%RC_v => s%w( 1:problem%n )
                       END IF
                       inform%status = GET_JV
                       s%RC_Mv => s%v(1:problem%m )
                       RETURN

                    ELSE

                       CALL FILTRANE_J_times_v( s%w( 1:problem%n ),            &
                                                s%v( 1:problem%m),             &
                                                .FALSE., s%has_inequalities )

                    END IF
                 END IF
              END IF

!**************************************************************
400           CONTINUE ! *** Reverse communication re-entry ***
!**************************************************************

              IF (  s%GLTR_info%status == 3 .OR. s%GLTR_info%status == 7 ) THEN
                 IF ( problem%m > 0 .AND. control%external_J_products ) THEN
                    IF ( s%level >= DEBUG ) THEN
                       WRITE( s%out, 1003 )
                       IF ( inform%status == GET_JV ) THEN
                          WRITE( s%out, 1011 )
                       ELSE
                          WRITE( s%out, 1021 ) inform%status
                       END IF
                       WRITE( s%out, 1003 )
                    END IF
                    inform%status = OK
                    NULLIFY( s%RC_v, s%RC_Mv )
                    IF ( s%has_inequalities ) THEN
                       DO i = 1, problem%m
                          IF ( FILTRANE_c_violation( i ) == ZERO )             &
                             s%v( i ) = ZERO
                       END DO
                    END IF
                 END IF

!                Add the contribution from the terms in JTV^aJ for the
!                inequality penalty functions (into s%v).

                 IF ( s%has_inequalities ) THEN

                    IF ( control%inequality_penalty_type /= 2 ) THEN
                       DO i = 1, problem%m
                          IF ( problem%equation( i ) ) CYCLE
                          SELECT CASE ( control%inequality_penalty_type )
                          CASE ( 1 )
                             ci  = problem%c( i )
                             cli = problem%c_l( i )
                             cui = problem%c_u( i )
                             twoeps = TWO * s%epsilon
                             IF ( ci > cli - s%epsilon .AND. &
                                  ci < cli + s%epsilon       ) THEN
                                t = ( cli + s%epsilon - ci ) / twoeps
                                d2phi = THREE * t * ( ONE - t ) / s%epsilon
                                s%v( i ) = d2phi * s%v( i )
                             END IF
                             IF ( ci > cui - s%epsilon .AND. &
                                  ci < cui + s%epsilon       ) THEN
                                t = ( ci - cui + s%epsilon ) / twoeps
                                d2phi = THREE * t * ( ONE - t ) / s%epsilon
                                s%v( i ) = d2phi * s%v( i )
                             END IF
                          CASE ( 3 )
                             s%v( i ) = s%v( i ) * 3.0_wp *                    &
                                      ABS(FILTRANE_c_violation( i ) )
                          CASE ( 4 )
                             s%v( i ) = s%v( i ) * 6.0_wp *                    &
                                      FILTRANE_c_violation( i ) ** 2
                          END SELECT
                       END DO
                    END IF

                 END IF

!                Add the contribution from the scaled identity for the
!                violated bounds (into s%r).

                 IF ( s%has_bounds ) THEN

                    DO j = 1, problem%n

                       SELECT CASE ( problem%x_status( j ) )
                       CASE ( LOWER, UPPER, RANGE )
                          violation = FILTRANE_x_violation( j )
                          SELECT CASE ( control%inequality_penalty_type )
                          CASE ( 1 )
                             xj  = problem%x( j )
                             xlj = problem%x_l( j )
                             xuj = problem%x_u( j )
                             twoeps = TWO * s%epsilon
                             IF ( xj > xlj - s%epsilon .AND. &
                                  xj < xlj + s%epsilon       ) THEN
                                t = ( s%epsilon - violation ) / twoeps
                                d2phi = THREE * t * ( ONE - t ) / s%epsilon
                                s%r( j ) = s%r( j ) + d2phi * s%w( j )
                             END IF
                             IF ( xj > xuj - s%epsilon .AND. &
                                  xj < xuj + s%epsilon       ) THEN
                                t = ( violation + s%epsilon ) / twoeps
                                d2phi = THREE * t * ( ONE - t ) / s%epsilon
                                s%r( j ) = s%r( j ) + d2phi * s%w( j )
                             END IF
                          CASE ( 2 )
                             IF ( violation /= ZERO )                          &
                                s%r( j ) = s%r( j ) + s%w( j )
                          CASE ( 3 )
                             IF ( violation /= ZERO )                          &
                                s%r( j ) = s%r( j ) + THREE * s%w( j ) *       &
                                           ABS( violation )
                          CASE ( 4 )
                             IF ( violation /= ZERO )                          &
                                s%r( j ) = s%r( j ) + 6.0_wp * s%w( j ) *      &
                                           violation ** 2
                          END SELECT
                       END SELECT

                    END DO

                 END IF

!                Post multiply by J^T, erasing the initial value of s%w.

                 IF ( problem%m > 0 ) THEN

                    IF ( control%external_J_products ) THEN

                       IF ( s%has_inequalities ) THEN
                          DO i = 1, problem%m
                             IF ( FILTRANE_c_violation( i ) == ZERO ) THEN
                                s%t( i ) = ZERO
                             ELSE
                                s%t( i ) = s%v( i )
                             END IF
                          END DO
                          s%RC_v => s%t( 1:problem%m )
                       ELSE
                          s%RC_v => s%v( 1:problem%m )
                       END IF
                       inform%status = GET_JTV
                       s%RC_Mv => s%w(1:problem%n )
                       RETURN

                    ELSE

                       CALL FILTRANE_J_times_v( s%v( 1:problem%m ),            &
                                                s%w( 1:problem%n),             &
                                                .TRUE., s%has_inequalities )

                    END IF
                 END IF
              END IF

!**************************************************************
450           CONTINUE ! *** Reverse communication re-entry ***
!**************************************************************

              IF (  s%GLTR_info%status == 3 .OR. s%GLTR_info%status == 7 ) THEN

                 IF ( control%external_J_products ) THEN
                    IF ( s%level >= DEBUG ) THEN
                       WRITE( s%out, 1003 )
                       IF ( inform%status == GET_JTV ) THEN
                          WRITE( s%out, 1012 )
                       ELSE
                          WRITE( s%out, 1021 ) inform%status
                       END IF
                       WRITE( s%out, 1003 )
                    END IF
                    inform%status = OK
                    NULLIFY( s%RC_v, s%RC_Mv )
                    IF ( s%has_fixed ) THEN
                         DO j = 1, problem%n
                            IF ( problem%x_status( j ) == FIXED )              &
                               s%w( j ) = ZERO
                         END DO
                    END IF
                 END IF

!                Add the terms collected in s%r, if any.

                 IF ( problem%m > 0 ) THEN
                    IF ( s%model_used == NEWTON .OR. s%has_bounds ) THEN
                       s%w( 1:problem%n ) = s%w( 1:problem%n )                 &
                                            + s%r( 1:problem%n )
                    END IF
                 ELSE
                    s%w( 1:problem%n ) = s%r
                 END IF

              END IF

           GO TO 4000 ! End of the GLTR iterations
4001       CONTINUE

!          --------------------------------------------------------------------

!          Keep track of the number of CG iterations.

!          --------------------------------------------------------------------

           inform%nbr_cg_iterations = inform%nbr_cg_iterations+ s%GLTR_info%iter

!          --------------------------------------------------------------------

!          See if negative curvature has been encountered.  If this is the
!          case and the step is unrestricted, then the step should be
!          recomputed with a restriction on its length.

!          --------------------------------------------------------------------

           IF ( s%GLTR_info%negative_curvature .AND. .NOT. s%restrict ) THEN
              IF ( control%weak_accept_power == -666.0_wp ) THEN
                 inform%status = -666
                 WRITE( inform%message( 1 ), 1158 )
                 GO TO 999
              ELSE
                 s%restrict   = .TRUE.
                 s%unsuccess  = .TRUE.
                 IF ( s%level >= ACTION ) WRITE ( s%out, 1125 )
                 GO TO 3000
              END IF
           END IF

!          --------------------------------------------------------------------

!          Print the norms of the residual and the computed step, if requested.

!          --------------------------------------------------------------------

           IF ( s%level >= ACTION ) THEN
              WRITE( s%out, 1156 ) s%GLTR_info%iter, NRM2( problem%n, s%u, 1 )
              IF ( control%prec_used == NONE ) THEN
                 WRITE( s%out, 1126 ) s%s_norm
              ELSE
                 WRITE( s%out, 1127 ) s%s_norm2
                 WRITE( s%out, 1128 ) s%s_norm
              END IF
           END IF

!          --------------------------------------------------------------------

!          Verify that the step is meaningful compared to the current values
!          of the variables. If it is too small, ensure that it is
!          computed with full accuracy before giving up.

!          --------------------------------------------------------------------

           IF ( s%s_norm2 < s%x_norm * EPSMACH ) THEN
              IF ( s%step_accuracy == FULL ) THEN
                 inform%status = PROGRESS_IMPOSSIBLE
                 WRITE( inform%message( 1 ), 1129 )
                 WRITE( inform%message( 2 ), 1130 ) s%x_norm, s%s_norm2
                 GO TO 999
              ELSE
                 IF ( s%level >= DEBUG ) WRITE( s%out, 1131 )
                 s%step_accuracy = FULL
                 GO TO 3000
              END IF
           END IF

!       END DO step loop

!       -----------------------------------------------------------------------

!       Compute the objective function value at the trial point.

!       -----------------------------------------------------------------------

!       Save the current values of f, c and x.

        s%v( 1:problem%n ) = problem%x
        s%f_old            = problem%f
        IF( problem%m > 0 ) s%w( 1:problem%m ) = problem%c

!       Compute the trial point.

        problem%x = problem%x + s%step

        IF ( s%level >= DETAILS ) THEN
           WRITE( s%out, 1099 )
           WRITE( s%out, 1132 )
           CALL TOOLS_output_vector( problem%n, problem%x, s%out )
        END IF

!       Compute the values of the constraints.

        IF ( problem%m > 0 ) THEN
           inform%status = GET_C
           RETURN
        END IF

!********************************************************
600     CONTINUE ! *** Reverse communication re-entry ***
!********************************************************

        IF ( problem%m > 0 ) THEN
           IF ( s%level >= DEBUG ) THEN
              WRITE( s%out, 1003 )
              IF ( inform%status == GET_C ) THEN
                 WRITE( s%out, 1015 )
              ELSE
                 WRITE( s%out, 1021 ) inform%status
              END IF
              WRITE( s%out, 1003 )
           END IF
           inform%status            = OK
           inform%nbr_c_evaluations = inform%nbr_c_evaluations + 1

           IF ( s%level >= DETAILS ) THEN
              WRITE( s%out, 1133 )
              CALL TOOLS_output_vector( problem%m, problem%c, s%out )
           END IF

!       Compute the value of the objective function, avoiding overflow
!       when possible.

           IF ( MAXVAL( problem%c ) >= SQRT( OVERFLOW ) ) THEN
              problem%f = OVERFLOW
              IF ( s%level >= ACTION ) WRITE( s%out, 1134 )
           ELSE
              CALL FILTRANE_compute_theta
              CALL FILTRANE_compute_f
              IF ( s%level >= ACTION ) WRITE( s%out, 1135 ) problem%f
           END IF

        ELSE

           CALL FILTRANE_compute_theta
           CALL FILTRANE_compute_f
           IF ( s%level >= ACTION ) WRITE( s%out, 1135 ) problem%f

        END IF

!       Save the current point and its associated objective value
!       if this is the best value found so far.

        IF ( control%save_best_point ) THEN
           IF ( problem%f < s%best_fx ) THEN
              s%best_x  = problem%x
              s%best_fx = problem%f
              IF ( s%level >= DETAILS ) WRITE( s%out, 1136 )
           END IF
        END IF

!       -----------------------------------------------------------------------

!       Evaluate rho, the ratio of achieved to predicted reductions,
!       ensuring that rounding errors do not dominate the computations.

!       -----------------------------------------------------------------------

        IF ( problem%f < OVERFLOW ) THEN
           s%ared = s%f_old - problem%f
           s%feps = MAX( ONE, ABS( s%f_old ) ) * EPSMACH
           prered = - s%model_value + s%feps
           s%rho  = ( s%ared + s%feps ) / prered

           IF ( s%level >= ACTION ) THEN
              WRITE( s%out, 1137 )  s%ared
              IF ( s%model_used == GAUSS_NEWTON ) THEN
                 WRITE( s%out, 1138 )  -s%model_value
              ELSE
                 WRITE( s%out, 1139 )  -s%model_value
              END IF
           END IF
        ELSE
           s%rho  = - INFINITY
           s%ared = - INFINITY
        END IF

!       -----------------------------------------------------------------------

!       Evaluate the alternative model, if necessary.

!       -----------------------------------------------------------------------

        IF ( control%model_type == AUTOMATIC ) THEN

!          See if the current model is satisfactory.

           SELECT CASE ( control%model_criterion )
           CASE ( BEST_FIT )
              s%bad_model = ABS( s%rho - ONE ) > 0.1_wp .AND. s%s_norm < 0.1_wp
           CASE ( BEST_REDUCTION )
              s%bad_model = s%rho < control%eta_1 .AND. s%s_norm < 0.1_wp
           END SELECT

           IF ( s%bad_model ) THEN

!             Evaluate the alternative model (at the old x)
!             and the associated ratio of achieved to predicted reductions.

              IF ( .NOT. s%goth ) CALL FILTRANE_compute_Hmult( s%w )

              CALL SWAP( problem%n, problem%x, 1, s%v, 1 )
              s%RC_v    => s%step( 1:problem%n )
              s%RC_Mv   => s%r( 1:problem%n )
              s%RC_newx = .NOT. s%goth
              inform%status = GET_MP2
              RETURN
           END IF
        END IF

!********************************************************
550     CONTINUE ! *** Reverse communication re-entry ***
!********************************************************

        IF ( control%model_type == AUTOMATIC ) THEN
           IF ( s%bad_model ) THEN
              IF ( s%level >= DEBUG ) THEN
                 WRITE( s%out, 1003 )
                 IF ( inform%status == GET_MP2 ) THEN
                    WRITE( s%out, 1014 )
                 ELSE
                    WRITE( s%out, 1021 ) inform%status
                 END IF
                 WRITE( s%out, 1003 )
              END IF
              inform%status  = OK
              NULLIFY( s%RC_v, s%RC_Mv )
              CALL SWAP( problem%n, problem%x, 1, s%v, 1 )
              s%goth      = .TRUE.
              !delta_model = HALF * INNER_PRODUCT( problem%n, s%step, 1, s%r, 1 )
              delta_model = HALF * DOT_PRODUCT( s%step, s%r )
              IF ( s%model_used == NEWTON ) THEN
                 Nprered  = - s%model_value
                 GNprered = Nprered + delta_model
                 IF ( s%level >= ACTION ) WRITE( s%out, 1138 ) GNprered
                 rhoN  = s%rho
                 rhoGN = ( s%ared + s%feps ) / ( GNprered + s%feps )
              ELSE
                 GNprered = - s%model_value
                 Nprered  = GNprered - delta_model
                 IF ( s%level >= ACTION ) WRITE( s%out, 1139 ) Nprered
                 rhoGN = s%rho
                 rhoN  = ( s%ared + s%feps ) / (  Nprered + s%feps )
              END IF

!             Vote for the best model at this iteration.

              SELECT CASE ( control%model_criterion )
              CASE ( BEST_FIT )
                 IF ( ABS( rhoN -ONE ) < ABS( rhoGN - ONE ) ) THEN
                    s%vote( s%next_vote ) = NEWTON
                 ELSE
                    s%vote( s%next_vote ) = GAUSS_NEWTON
                 END IF
              CASE ( BEST_REDUCTION )
                 IF ( rhoN > rhoGN ) THEN
                    s%vote( s%next_vote ) = NEWTON
                 ELSE
                    s%vote( s%next_vote ) = GAUSS_NEWTON
                 END IF
              END SELECT

           ELSE
             s%vote( s%next_vote ) = s%model_used
           END IF
           s%next_vote = s%next_vote + 1
           IF ( s%next_vote > control%model_inertia ) s%next_vote = 1

!          Define the next model to use by a majority rule on the
!          expressed votes.

           n_Newton = 0
           n_votes  = 0
           DO i = 1, control%model_inertia
              IF ( s%vote( i ) == UNDEFINED ) CYCLE
              n_votes = n_votes + 1
              IF ( s%vote( i ) == NEWTON ) n_Newton = n_Newton + 1
           END DO
           IF ( n_votes >= control%model_inertia ) THEN
              IF (  n_Newton > HALF * control%model_inertia ) THEN
                 IF ( s%level >= ACTION ) WRITE( s%out, 1140 ) n_Newton, n_votes
                 s%model_used = NEWTON
              ELSE
                 IF ( s%level >= ACTION )                                      &
                    WRITE( s%out, 1141 ) n_votes - n_Newton, n_votes
                 s%model_used = GAUSS_NEWTON
              END IF
              s%vote = UNDEFINED
           END IF

        END IF

!       -----------------------------------------------------------------------

!       Is the trial point acceptable ? If yes, compute the gradient.

!       -----------------------------------------------------------------------

!       Acceptable for the weak criterion ?  (weak acceptance enabled)

        IF ( control%weak_accept_power >= ZERO ) THEN
           IF ( control%weak_accept_power == ZERO .OR. problem%f >= ONE ) THEN
              fwp = ONE
           ELSE
              fwp = MIN( ONE, problem%f ** control%weak_accept_power )
           END IF
           s%weakly_acceptable = s%ared >= control%min_weak_accept_factor * fwp
        ELSE IF ( control%weak_accept_power == -666._wp ) THEN
           s%weakly_acceptable = .TRUE.
        ELSE
           s%weakly_acceptable = .FALSE.
        END IF

!       Acceptable for the filter ? Note that only test acceptablity for
!       the filter when the weak acceptance fails, in order to save the
!       work involved in filter comparisons when possible. this also prevents
!       including in the filter a point which is weakly acceptable.

        IF ( .NOT. s%weakly_acceptable ) THEN
           s%filter_acceptable = FILTRANE_is_acceptable( problem%f )

!       Acceptable for the trust region ?

           s%tr_acceptable =                                                   &
             s%rho >= control%eta_1 .AND. s%s_norm <= ONE_DOT_ONE * s%radius
	ELSE
           s%filter_acceptable = .FALSE.
           s%tr_acceptable     = .FALSE.
        END IF

!       Acceptable at all?

        s%acceptable = s%tr_acceptable .OR. s%weakly_acceptable .OR.           &
                       s%filter_acceptable

!       If acceptable, compute the Jacobian unless products are external,
!       in which case the gradient is requested from the user.

        IF ( s%acceptable ) THEN
           s%x_norm = NRM2( problem%n, problem%x, 1 )
           s%goth   = .FALSE.
           IF ( problem%m > 0 .AND. .NOT. control%external_J_products ) THEN
              inform%status = GET_J_A
              RETURN
           END IF
        END IF

!********************************************************
630     CONTINUE ! *** Reverse communication re-entry ***
!********************************************************

        IF ( inform%status == GET_J_A ) THEN
           IF ( s%level >= DEBUG ) THEN
              WRITE( s%out, 1003 )
              WRITE( s%out, 1016 )
              WRITE( s%out, 1003 )
           END IF
           inform%status = OK
           inform%nbr_J_evaluations = inform%nbr_J_evaluations + 1
        END IF

!********************************************************
660     CONTINUE ! *** Reverse communication re-entry ***
!********************************************************

        IF ( s%acceptable ) THEN
           IF ( inform%status == GET_JTC_A ) THEN
              IF ( s%level >= DEBUG ) THEN
                 WRITE( s%out, 1003 )
                 WRITE( s%out, 1017 )
                 WRITE( s%out, 1003 )
              END IF
              inform%status = GET_JTC
           END IF
           CALL FILTRANE_compute_grad_f
           SELECT CASE ( inform%status )
           CASE ( :-1 )
              GO TO 999
           CASE ( GET_JTC )
              inform%status = GET_JTC_A
              RETURN
           END SELECT
           s%g_norm2     = NRM2( problem%n, problem%g, 1 )
           s%g_norminf_u = MAXVAL( ABS( problem%g( 1:problem%n ) ) )
        END IF

!       -----------------------------------------------------------------------

!       Now take action depending on acceptability.

!       -----------------------------------------------------------------------

!       Acceptable for the objective decrease ? (weak acceptance enabled)

        IF ( s%weakly_acceptable ) THEN
           IF ( control%weak_accept_power >= ZERO ) THEN
              s%it_status( 4:4 ) = 'W'
           ELSE
              s%it_status( 4:4 ) = 'A'
           END IF
           s%unsuccess  = .FALSE.

!       Acceptable for the filter ?

        ELSE IF ( s%filter_acceptable ) THEN

           IF ( s%rho < control%eta_1             .OR. &
                s%s_norm > ONE_DOT_ONE * s%radius      ) THEN
              s%it_status( 4:4 ) = 'f'
              CALL FILTRANE_add_to_filter
              IF ( inform%status < 0 ) GO TO 999

              s%extent = MAX( ONE, HALF * s%extent )

!             If this is the last point that can be included in the filter,
!             return to pure trust-region from now on.

              IF ( control%maximal_filter_size > 0              .AND. &
                   s%filter_size >= control%maximal_filter_size       ) THEN
                 s%use_filter  = .FALSE.
                 s%restrict    = .TRUE.
              ELSE
                 s%restrict    = .FALSE.
              END IF
           ELSE
              s%it_status( 4:4 ) = 'F'
              IF ( .NOT. s%restrict .AND. s%extent < control%str_relax ) THEN
                 s%extent = MIN( control%str_relax, TWO * s%extent )
              END IF
              s%restrict = .FALSE.
           END IF
           s%unsuccess  = .FALSE.

!       Acceptable for the trust region ?

        ELSE IF ( s%tr_acceptable ) THEN

           IF ( s%rho >= control%eta_2 ) THEN
              IF ( s%level >= DETAILS ) WRITE( s%out, 1142 )
              s%it_status( 4:4 ) = 'S'
              IF ( .NOT. s%restrict ) THEN
                 s%extent = MIN( control%str_relax, TWO * s%extent )
              END IF
           ELSE
              IF ( s%level >= DETAILS ) WRITE( s%out, 1143 )
              s%it_status( 4:4 ) = 's'
           END IF
           IF ( s%use_filter ) s%restrict = .FALSE.
           s%unsuccess= .FALSE.

!       Unsuccessful cases

        ELSE

           problem%x   = s%v( 1:problem%n )
           IF ( problem%m > 0 ) problem%c   = s%w( 1:problem%m )
           s%f_plus    = problem%f
           problem%f   = s%f_old
           s%unsuccess = .TRUE.
           IF ( s%rho < ZERO ) THEN
              IF ( s%level >= DETAILS ) WRITE( s%out, 1144 )
              s%it_status( 4:4 ) = 'U'
           ELSE
              IF ( s%level >= DETAILS ) WRITE( s%out, 1145 )
              s%it_status( 4:4 ) = 'u'
           END IF
           IF ( .NOT. s%restrict ) THEN
              s%extent   = ONE
           END IF
           s%restrict = .TRUE.

        END IF

!       -----------------------------------------------------------------------

!       Compute the norm of the new (preconditioned) gradient.

!       -----------------------------------------------------------------------

        IF ( s%acceptable ) THEN

           SELECT CASE ( control%prec_used )

           CASE ( NONE )

              s%g_norm    = s%g_norm2
	      s%g_norminf = s%g_norminf_u

           CASE ( BANDED )

!             Obtain banded preconditioner from the JTJ, if not already done.

              CALL FILTRANE_build_band_JTJ( s%bandw )

              IF ( s%level >= DEBUG ) THEN
                 WRITE( s%out, 1096 ) s%nsemib, s%bandw
                 DO i = 1, problem%n
                    k = MIN( s%bandw, problem%n - i )
                    WRITE( s%out, 1097 ) i, s%diag( i ),                       &
                                      ( s%offdiag( j, i ), j = 1, MIN( k, 5 ) )
                    IF ( k > 5 ) WRITE( s%out, 1098 ) ( s%offdiag(j,i),j = 6,k )
                 END DO
              END IF

              CALL BAND_factor( problem%n, s%bandw, s%diag, s%offdiag,         &
                                s%nsemib, BAND_status )

              IF ( s%level >= DEBUG ) THEN
                 WRITE( s%out, 1155 )
                 DO i = 1, problem%n
                    k = MIN( s%bandw, problem%n - i )
                    WRITE( s%out, 1097 ) i, s%diag( i ),                       &
                                      ( s%offdiag( j, i ), j = 1, MIN( k, 5 ) )
                    IF ( k > 5 ) WRITE( s%out, 1098 ) ( s%offdiag(j,i),j=6,k )
                 END DO
              END IF

!             Compute the preconditioned gradient and its norm.

              s%w( 1:problem%n ) = problem%g
              CALL BAND_solve( problem%n, s%bandw, s%diag, s%offdiag,          &
                               s%nsemib, s%w, BAND_status )
              !s%g_norm = SQRT( INNER_PRODUCT( problem%n, problem%g, 1,         &
              !                                s%w( 1:problem%n ), 1 ) )
              s%g_norm = SQRT( DOT_PRODUCT( problem%g, s%w( 1:problem%n ) ) )
              IF ( s%nsemib == 0 ) THEN
                 s%g_norminf = ZERO
                 DO j = 1, problem%n
                    s%g_norminf = MAX( s%g_norminf, problem%g( j ) * s%v( j ) )
                 END DO
                 s%g_norminf = SQRT( s%g_norminf )
              END IF

           CASE ( USER_DEFINED )

              s%w( 1:problem%n ) = problem%g
              s%RC_pv => s%w
              inform%status = GET_PREC_G_A
              RETURN

           END SELECT

        END IF

!********************************************************
680     CONTINUE ! *** Reverse communication re-entry ***
!********************************************************

        IF ( s%acceptable ) THEN

           IF ( control%prec_used == USER_DEFINED ) THEN
              IF ( s%level >= DEBUG ) THEN
                 WRITE( s%out, 1003 )
                 IF ( inform%status == GET_PREC_G_A ) THEN
                    WRITE( s%out, 1010 )
                 ELSE
                    WRITE( s%out, 1021 ) inform%status
                 END IF
                 WRITE( s%out, 1003 )
              END IF
              inform%status = OK
              NULLIFY( s%RC_Pv )
              !s%g_norm = SQRT( INNER_PRODUCT( problem%n, problem%g, 1,         &
              !                                s%w( 1:problem%n ), 1 ) )
              s%g_norm = SQRT( DOT_PRODUCT( problem%g, s%w( 1:problem%n ) ) )
           END IF

           IF ( s%level >= DEBUG ) WRITE( s%out, 1146 ) s%g_norm

        END IF

!       -----------------------------------------------------------------------

!       Print the required information on the current iteration.

!       -----------------------------------------------------------------------

        IF ( s%level >= TRACE ) THEN
           IF ( s%level > TRACE ) THEN
              WRITE( s%out, 1099 )
              WRITE( s%out, 1147 )
              WRITE( s%out, 1099 )
           END IF
           IF ( inform%nbr_cg_iterations < 1000000 ) THEN
              WRITE( s%out, 1001 ) inform%nbr_iterations, problem%f,           &
!                                   s%g_norm, s%rho, s%s_norm, s%radius,       &
                                   s%g_norm, s%rho, s%s_norm2, s%radius,       &
                                   inform%nbr_cg_iterations, s%it_status,      &
                                   s%filter_size - s%filter_nbr_inactive
           ELSE
              ig = inform%nbr_cg_iterations / 1000
              IF ( ig < 100000 ) THEN
                 mult = 'K'
              ELSE
                 ig = ig / 1000
                 mult = 'M'
              END IF
              WRITE( s%out, 1002 ) inform%nbr_iterations, problem%f,           &
!                                   s%g_norm, s%rho, s%s_norm, s%radius,       &
                                   s%g_norm, s%rho, s%s_norm2, s%radius,       &
                                   ig, mult, s%it_status,                      &
                                   s%filter_size - s%filter_nbr_inactive
           END IF
           IF ( s%level > TRACE ) WRITE( s%out, 1099 )
           IF ( s%level >= DETAILS ) THEN
              WRITE( s%out, 1101 )
              CALL TOOLS_output_vector( problem%n, problem%x, s%out )
              WRITE( s%out, 1102 )
              CALL TOOLS_output_vector( problem%n, problem%g, s%out )
              IF ( problem%m > 0 ) THEN
                 WRITE( s%out, 1103 )
                 CALL TOOLS_output_vector( problem%m, problem%c, s%out )
              END IF
              IF ( s%has_bounds .OR. s%has_inequalities .OR. &
                   control%grouping /= NONE                  ) THEN
                  WRITE( s%out, 1104 )
                 CALL TOOLS_output_vector( s%p, s%theta, s%out )
              END IF
              WRITE( s%out, 1114 )
              CALL TOOLS_output_vector( problem%n, s%step, s%out )
              IF ( s%level >= DEBUG .AND. problem%J_ne > 0 ) THEN
                 WRITE( s%out, 1105 )
                 CALL TOOLS_output_matrix_C( problem%J_ne, problem%J_val,      &
                                           problem%J_row, problem%J_col, s%out )
              END IF
              WRITE( s%out, 1148 )
           END IF
        END IF

!       -----------------------------------------------------------------------

!       Test for convergence.

!       -----------------------------------------------------------------------

        sqrtn = problem%n
        sqrtn = SQRT( sqrtn )
        IF ( control%stop_on_prec_g ) THEN
           IF ( control%stop_on_g_max ) THEN
              IF ( s%g_norminf  <= control%g_accuracy ) THEN
                 WRITE( inform%message( 1 ), 1159 )
                 GO TO 999
              END IF
           ELSE
              IF ( s%g_norm  <= sqrtn * control%g_accuracy ) THEN
                 WRITE( inform%message( 1 ), 1159 )
                 GO TO 999
              END IF
           END IF
        ELSE
           IF ( control%stop_on_g_max ) THEN
              IF ( s%g_norminf_u <= control%g_accuracy ) THEN
                 WRITE( inform%message( 1 ), 1159 )
                 GO TO 999
              END IF
           ELSE
              IF ( s%g_norm2 <= sqrtn * control%g_accuracy ) THEN
                 WRITE( inform%message( 1 ), 1159 )
                 GO TO 999
              END IF
           END IF
        END IF
        IF ( ALL( ABS( s%theta( 1:s%p ) ) <= control%c_accuracy ) ) THEN
           WRITE( inform%message( 1 ), 1160 )
           GO TO 999
        END IF

!       -----------------------------------------------------------------------

!       Update the trust-region radius.

!       -----------------------------------------------------------------------

        IF ( s%s_norm <= ONE_DOT_ONE * s%radius ) THEN
           IF ( s%rho >= control%eta_1 ) THEN
              IF ( s%rho >= control%eta_2 ) THEN    ! very successful iteration
                 s%radius = MAX( s%radius, control%gamma_2 * s%s_norm )
              ELSE                                ! successful iteration
!                s%radius = SQRT( control%gamma_1 ) * s%radius
              END IF
           ELSE
              IF ( s%first_TR_step ) THEN
                 s%radius = s%s_norm
                 IF ( control%use_filter == INITIAL ) s%use_filter = .FALSE.
              END IF
              IF ( s%rho > ZERO ) THEN              ! unsuccessful iteration
                 s%radius = MAX( control%gamma_0 * s%radius,                   &
                                 control%gamma_1 * s%s_norm )
              ELSE                                ! very unsuccessful iteration
                 !gs  = INNER_PRODUCT( problem%n, problem%g, 1, s%step, 1 )
                 gs  = DOT_PRODUCT( problem%g, s%step )
                 gam = s%model_value - s%f_old - gs
                 gam = ( control%eta_2 - ONE ) * gs /                          &
                       ( s%f_plus - s%f_old - gs - control%eta_2 * gam )
                 s%radius = MIN( MAX( control%gamma_0, gam ),                  &
                                      control%gamma_1 ) * s%radius
              END IF
           END IF
        END IF

        IF ( s%level > DEBUG ) WRITE( s%out, 1149 ) s%radius

!       -----------------------------------------------------------------------

!       Save checkpointing information, if requested

!       -----------------------------------------------------------------------

        IF ( control%checkpoint_freq /= 0 ) THEN
           IF ( MOD(inform%nbr_iterations,ABS(control%checkpoint_freq))==0 )THEN

!             Open the checkpoint file

              OPEN( UNIT = control%checkpoint_dev,                             &
                    FILE = control%checkpoint_file,                            &
                    STATUS = 'UNKNOWN', IOSTAT = iostat )
              IF ( iostat > 0 ) THEN
                 inform%status = COULD_NOT_WRITE
                 WRITE( inform%message( 1 ), 1072 ) control%checkpoint_file
                 GO TO 999
              END IF

!             Write the checkpoint data

              WRITE( control%checkpoint_dev, 1150 ) inform%nbr_iterations

              IF ( ALLOCATED( problem%vnames ) ) THEN
                 DO j = 1, problem%n
                    WRITE( UNIT = control%checkpoint_dev, FMT = 1151,          &
                           IOSTAT = iostat ) j, problem%x(j), problem%vnames(j)
                    IF ( iostat > 0 ) THEN
                       inform%status = COULD_NOT_WRITE
                       WRITE( inform%message( 1 ), 1204 )control%checkpoint_file
                       GO TO 999
                    END IF
                 END DO
              ELSE
                 DO j = 1, problem%n
                    WRITE( control%checkpoint_dev, 1151 ) j, problem%x( j )
                 END DO
              END IF

              WRITE( UNIT= control%checkpoint_dev, FMT = 1152,                 &
                     IOSTAT = iostat ) s%radius
              IF ( iostat > 0 ) THEN
                 inform%status = COULD_NOT_WRITE
                 WRITE( inform%message( 1 ), 1204 ) control%checkpoint_file
                 GO TO 999
              END IF

              WRITE( UNIT= control%checkpoint_dev, FMT = 1203,                 &
                     IOSTAT = iostat ) problem%f
              IF ( iostat > 0 ) THEN
                 inform%status = COULD_NOT_WRITE
                 WRITE( inform%message( 1 ), 1204 ) control%checkpoint_file
                 GO TO 999
              END IF

!             Close the file

              CLOSE( control%checkpoint_dev )

              IF ( s%level >= DEBUG ) WRITE( s%out, 1153 )

           END IF
        END IF

!===============================================================================
!===============================================================================

     GO TO 2000 ! End of the main loop

!===============================================================================
!===============================================================================

!===============================================================================
!===============================================================================

!                              Conclude...

!===============================================================================
!===============================================================================

999  CONTINUE
     s%exitc = inform%status

!    Retrieve the best value, if necessary.

     IF ( control%save_best_point ) THEN
        s%best_x_is_past = s%best_fx < problem%f
        IF ( s%best_x_is_past ) THEN
           IF ( s%level >= ACTION ) WRITE( s%out, 1154 )
           problem%x = s%best_x
           IF ( control%external_J_products ) THEN
              inform%status = GET_C_F
           ELSE
              inform%status = GET_C_AND_J_F
           END IF
           RETURN
        END IF
     ELSE
        s%best_x_is_past = .FALSE.
     END IF

!*****************************************************
700  CONTINUE ! *** Reverse communication re-entry ***
!*****************************************************

     IF ( s%best_x_is_past) THEN
        IF ( s%level >= DEBUG ) THEN
           WRITE( s%out, 1003 )
           SELECT CASE ( inform%status )
           CASE ( GET_C_AND_J_F )
              WRITE( s%out, 1018 )
           CASE ( GET_C_F )
              WRITE( s%out, 1019 )
           CASE DEFAULT
              WRITE( s%out, 1021 ) inform%status
           END SELECT
           WRITE( s%out, 1003 )
        END IF
        inform%status            = OK
        inform%nbr_c_evaluations = inform%nbr_c_evaluations + 1
        IF ( .NOT. control%external_J_products )                               &
           inform%nbr_J_evaluations = inform%nbr_J_evaluations + 1
        CALL FILTRANE_compute_theta
        CALL FILTRANE_compute_f
     END IF

!*****************************************************
710  CONTINUE ! *** Reverse communication re-entry ***
!*****************************************************

     IF ( s%best_x_is_past ) THEN
        IF ( inform%status == GET_JTC_F ) THEN
           IF ( s%level >= DEBUG ) THEN
              WRITE( s%out, 1003 )
              WRITE( s%out, 1020 )
              WRITE( s%out, 1003 )
           END IF
           inform%status = GET_JTC
        END IF
        CALL FILTRANE_compute_grad_f
        IF ( inform%status == GET_JTC ) THEN
           inform%status = GET_JTC_F
           RETURN
        END IF
     END IF

!    Set the stage indicator.

     s%stage = DONE

!    Print the final messages.

     inform%status = s%exitc
     CALL FILTRANE_say_goodbye( control, inform, s )

     RETURN

!===============================================================================

!     Formats for printout

!===============================================================================

1000 FORMAT(1x,i4,1x,1pE11.3,1x,1pE10.3,22x,1E11.3,1x,i6,6x,i3)
1001 FORMAT(1x,i4,1x,1pE11.3,4(1x,1pE10.3),1x,i6,1x,a4,1x,i3 )
1002 FORMAT(1x,i4,1x,1pE11.3,4(1x,1pE10.3),1x,i5,a1,1x,a4,1x,i3 )
1003 FORMAT(4x,'============================================================')
1004 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = START)' )
1005 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_C_AND_J_0)')
1006 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_C_0)')
1007 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_JTC_0)')
1008 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_PREC_G_0)')
1009 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_PREC)')
1010 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_PREC_G_A)')
1011 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_PREC_JV)')
1012 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_JTV)')
1013 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_MP1)')
1014 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_MP2)')
1015 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_C)')
1016 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_J_A)')
1017 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_JTC_A)')
1018 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_C_AND_J_F)')
1019 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_C_F)')
1020 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status = GET_JTC_F)')
1021 FORMAT(4x,'REVERSE COMMUNICATION REENTRY (inform%status (unexpected) = ', &
            I3,')')
1022 FORMAT(1x,'FILTRANE ERROR: SOLVE  was entered with an erroneous status')
1023 FORMAT(1x,'                (inform%status = ',i2,')')
1024 FORMAT(1x,'FILTRANE ERROR: the problem has ', i6, ' variables!')
1025 FORMAT(1x,'FILTRANE ERROR: the problem has ', i6, ' constraints!')
1026 FORMAT(4x,'s%stage = ',i3)
1027 FORMAT(1x,'FILTRANE ERROR: attempt to SOLVE without a call to INITIALIZE')
1028 FORMAT(1x,'FILTRANE ERROR: problem%x has not been allocated!')
1029 FORMAT(1x,'FILTRANE ERROR: problem%x_l has not been allocated!')
1030 FORMAT(1x,'FILTRANE ERROR: problem%x_u has not been allocated!')
1031 FORMAT(1x,'FILTRANE ERROR: problem%x_status has not been allocated!')
1032 FORMAT(2x,'problem contains only free variables')
1033 FORMAT(1x,'FILTRANE ERROR: problem%c has not been allocated!')
1034 FORMAT(1x,'FILTRANE ERROR: problem%c_l has not been allocated!')
1035 FORMAT(1x,'FILTRANE ERROR: problem%c_u has not been allocated!')
1036 FORMAT(1x,'FILTRANE ERROR: problem%equation has not been allocated!')
1037 FORMAT(1x,'FILTRANE ERROR: problem%y has not been allocated!')
1038 FORMAT(1x,'FILTRANE ERROR: problem%J_val has not been allocated!')
1039 FORMAT(1x,'FILTRANE ERROR: problem%J_col has not been allocated!')
1040 FORMAT(1x,'FILTRANE ERROR: problem%J_row has not been allocated!')
1041 FORMAT(1x,'FILTRANE ERROR: problem%g has not been allocated!')
1042 FORMAT(2x,'the problem has inequality constraints')
1043 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating vote(',i6,')')
1044 FORMAT(4x,'vote(',i6,') allocated')
1045 FORMAT(2x,'number of groups = ',i6)
1046 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating group(',i6,')')
1047 FORMAT(4x,'group(',i6,') allocated')
1048 FORMAT(1x,'FILTRANE ERROR: ',                                             &
               'the user-defined number of groups (',i6,') is impossible.')
1049 FORMAT(1x,'FILTRANE ERROR: ',                                             &
               'the user-defined groups are incorrectly defined')
1050 FORMAT(1x,'FILTRANE ERROR: the group index of the ',i6,'-th constraint')
1051 FORMAT(1x,'                (value = ',i6,') is impossible.')
1052 FORMAT(1x,'FILTRANE ERROR: the group index of the ',i6,'-th variable')
1053 FORMAT(4x,'user-defined group information verified (',i6,' groups)')
1054 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating theta(',i6,')')
1055 FORMAT(4x,'theta(',i6,') allocated')
1056 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating best_x(',i6,')')
1057 FORMAT(4x,'best_x(',i6,') allocated')
1058 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating step(',i6,')')
1059 FORMAT(4x,'step(',i6,') allocated')
1060 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating v(',i6,')')
1061 FORMAT(4x,'v(',i6,') allocated')
1062 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating t(',i6,')')
1063 FORMAT(4x,'t(',i6,') allocated')
1064 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating u(',i6,')')
1065 FORMAT(4x,'u(',i6,') allocated')
1066 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating w(',i6,')')
1067 FORMAT(4x,'w(',i6,') allocated')
1068 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating r(',i6,')')
1069 FORMAT(4x,'r(',i6,') allocated')
1070 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating iw(',i6,')')
1071 FORMAT(4x,'iw(',i6,') allocated')
1072 FORMAT(1x,'FILTRANE ERROR: cannot open checkpointing file ',a)
1073 FORMAT(1x,'FILTRANE ERROR while reading checkpointing file ',a)
1074 FORMAT(4x,'Checkpointing information from iteration',i6)
1075 FORMAT(4x,'successfully read (radius =',1pE11.3,')')
1076 FORMAT(1x,'FILTRANE ERROR: sorting capacity too small.')
1077 FORMAT(1x,'                Increase log2s in SORT_quicksort.')
1078 FORMAT(4x,'GROUP :')
1079 FORMAT(3x,'filter sign mode is RESTRICTED')
1080 FORMAT(3x,'filter sign mode is UNRESTRICTED')
1081 FORMAT(3x,'filter sign mode is MIXED')
1082 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating g_status(',i6,')')
1083 FORMAT(4x,'g_status(',i6,') allocated')
1084 FORMAT(3x,'maximal group overlap = ',i6)
1085 FORMAT(/,4x,'GROUP OVERLAP :')
1086 FORMAT(4x,'best value/point initialized')
1087 FORMAT(2x,'FILTRANE WARNING: BANDED preconditioner is not possible',/,    &
             2x,'                  with external Jacobian products:',/,        &
             2x,'                  preconditioning abandonned.')
1088 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating col(',i6,')')
1089 FORMAT(4x,'col(',i6,') allocated')
1090 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating perm(',i6,')')
1091 FORMAT(4x,'perm(',i6,') allocated')
1092 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating diag(',i6,')')
1093 FORMAT(4x,'diag(',i6,') allocated')
1094 FORMAT(1x,'FILTRANE ERROR: no memory left for allocating offdiag(',i6,    &
            ',',i6,')')
1095 FORMAT(4x,'offdiag(',i6,',',i6,') allocated')
1096 FORMAT(4x,'PRECONDITIONER (nsemib = ',i6,', bandw =',i6,'):')
1097 FORMAT(1x,i4,2x,1pE11.3,1x,5(1x,1pE11.3))
1098 FORMAT(20x,5(1x,1pE11.3))
1099 FORMAT(' ')
1100 FORMAT(1x,'Iter     f(x)      ||g(x)||     rho',                          &
            '       ||s||      Delta   #CGits Type   F')
1101 FORMAT(4x,'X :')
1102 FORMAT(4x,'G :')
1103 FORMAT(4x,'C :')
1104 FORMAT(4x,'THETA :')
1105 FORMAT(4x,'JACOBIAN MATRIX at X :')
1106 FORMAT(1x,'FILTRANE ERROR: maximum number of iterations reached')
1107 FORMAT(4x,'---> re-entering GLTR (prev radius =',1pE11.3,')')
1108 FORMAT(4x,'GLTR initialized')
1109 FORMAT(2x,'GLTR accuracy: absolute = ',1pE11.3,' relative = ',1pE11.3)
1110 FORMAT(3x,'GLTR radius = ', 1pE11.3,' GLTR accuracy =',1pE11.3)
1111 FORMAT(5x,'re-entering GLTR (rc)')
1112 FORMAT(5x,'radius = ',1PE11.3,' accuracy: abs = ',1pE11.3,' rel =',1pE11.3)
1113 FORMAT(5x,'model value = ',1PE11.3)
1114 FORMAT(5x,'STEP :')
1115 FORMAT(5x,'U :')
1116 FORMAT(5x,'W :')
1117 FORMAT(5x,'---- exiting GLTR with status =',i2,' ----')
1118 FORMAT(4x,'reentering GLTR after resetting the gradient')
1119 FORMAT(1x,'FILTRANE ERROR: the step could not be computed by GLTR')
1120 FORMAT(1x,'                GLTR error code = ',i3)
1121 FORMAT(5x,'band preconditioner applied')
1122 FORMAT(5x,'user preconditioner applied')
1123 FORMAT(5x,'goth = ',l1)
1124 FORMAT(4x,'GLTR radius = ',1pE11.3)
1125 FORMAT(2x,'negative curvature detected on unrestricted step:',/,          &
            2x,'recomputing a restricted step')
1126 FORMAT(2x,'norm of the step =',1pE11.3)
1127 FORMAT(2x,'Euclidean norm of the step =',1pE11.3)
1128 FORMAT(2x,'preconditioned norm of the step =',1pE11.3)
1129 FORMAT(1x,'FILTRANE ERROR: no futher progress seems possible')
1130 FORMAT(1x,'                ( ||x|| = ',1pE11.3,' ||s|| =',1pE11.3,' )')
1131 FORMAT(4x,'step too small: attempting a full precision step computation')
1132 FORMAT(3x,'XPLUS :')
1133 FORMAT(3x,'CPLUS :')
1134 FORMAT(3x,'f(xplus) = INFINITY')
1135 FORMAT(3x,'f(xplus) = ',1pE24.16)
1136 FORMAT(3x,'best value/point updated')
1137 FORMAT(3x,'achieved reduction     = ',1pE24.16)
1138 FORMAT(3x,'Gauss-Newton reduction = ',1pE24.16)
1139 FORMAT(3x,'Newton reduction       = ',1pE24.16)
1140 FORMAT(2x,'majority of ',i3,'/',i3,' votes for Newton''s model')
1141 FORMAT(2x,'majority of ',i3,'/',i3,' votes for Gauss-Newton''s model')
1142 FORMAT(3x,'iteration very successful')
1143 FORMAT(3x,'iteration successful')
1144 FORMAT(3x,'iteration very unsuccessful')
1145 FORMAT(3x,'iteration unsuccessful')
1146 FORMAT(4x,'||g(xplus)|| = ',1pE11.3)
1147 FORMAT('**** Iteration summary :')
1148 FORMAT(4x,'--------------------------------------------',/)
1149 FORMAT(4x,'new radius = ',1pE11.3)
1150 FORMAT(1x,'FILTRANE checkpointing ( iteration', I10,')' )
1151 FORMAT(1x,'X(',I10, ') = ', 1E24.16, 3x, a)
1152 FORMAT(1x,'radius = ', 1E24.16)
1153 FORMAT(4x,'checkpointing successful')
1154 FORMAT(2x,'recomputing the values at the best point')
1155 FORMAT(4x,'PRECONDITIONER FACTORIZATION :')
1156 FORMAT(2x,'number of CG iterations = ',i7,' norm of the residual = ',     &
            1pE11.3)
1157 FORMAT(5X,'---- entering GLTR ----')
1158 FORMAT(1x,'FILTRANE ERROR: negative curvature found without globalization')
1159 FORMAT(1x,'Problem successfully solved: objective function is stationary.')
1160 FORMAT(1x,'Problem successfully solved: constraints violations are small.')
1200 FORMAT(35x,i10 )
1201 FORMAT(3x,i10,4x,E24.16)
1202 FORMAT(1x,a6,3x,E24.16)
1203 FORMAT(1x,'objective function value = ', 1E24.16)
1204 FORMAT(1x,'FILTRANE ERROR: error while writing to checkpointing file ',a)

!==============================================================================
!==============================================================================

   CONTAINS  ! The FILTRANE tools

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_revise_control

!     Verifies and activates the changes in the FILTRANE control parameters

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================


!     Print starter, if requested.

      IF ( s%level >= TRACE ) WRITE( control%out, 1000 )

!     Start iteration for printing

      IF ( control%start_print /= s%prev_control%start_print ) THEN
         IF ( control%print_level >= ACTION )                                  &
            WRITE( s%out, 1001 ) control%start_print
         s%prev_control%start_print = control%start_print
      ELSE IF ( control%print_level >= ACTION ) THEN
         WRITE( s%out, 1002 ) control%start_print
      END IF

!     Printout level

      IF ( control%print_level /= s%prev_control%print_level ) THEN
         IF ( control%print_level >= ACTION .AND. &
              control%start_print <= 0            ) THEN
            SELECT CASE ( control%print_level )
            CASE ( ACTION )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, 1003 ) 'SILENT to ACTION'
               CASE ( TRACE )
                  WRITE( s%out, 1003 ) 'TRACE to ACTION'
               CASE ( DETAILS )
                  WRITE( s%out, 1003 ) 'DETAILS to ACTION'
               CASE ( DEBUG )
                  WRITE( s%out, 1003 ) 'DEBUG to ACTION'
               CASE ( CRAZY )
                  WRITE( s%out, 1003 ) 'CRAZY to ACTION'
               END SELECT
            CASE ( DETAILS )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, 1003 ) 'SILENT to DETAILS'
               CASE ( TRACE )
                  WRITE( s%out, 1003 ) 'TRACE to DETAILS '
               CASE ( ACTION )
                  WRITE( s%out, 1003 ) 'ACTION to DETAILS'
               CASE ( DEBUG )
                  WRITE( s%out, 1003 ) 'DEBUG to DETAILS '
               CASE ( CRAZY )
                  WRITE( s%out, 1003 ) 'CRAZY to DETAILS'
               END SELECT
            CASE ( DEBUG )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, 1003 ) 'SILENT to DEBUG'
               CASE ( TRACE )
                  WRITE( s%out, 1003 ) 'TRACE to DEBUG'
               CASE ( ACTION )
                  WRITE( s%out, 1003 ) 'ACTION to DEBUG'
               CASE ( DETAILS )
                  WRITE( s%out, 1003 ) 'DETAILS to DEBUG'
               CASE ( CRAZY )
                  WRITE( s%out, 1003 ) 'CRAZY to DEBUG'
               END SELECT
            CASE ( CRAZY: )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, 1003 ) 'SILENT to CRAZY'
               CASE ( TRACE )
                  WRITE( s%out, 1003 ) 'TRACE  to CRAZY'
               CASE ( ACTION )
                  WRITE( s%out, 1003 ) 'ACTION to CRAZY'
               CASE ( DETAILS )
                  WRITE( s%out, 1003 ) 'DETAILS to CRAZY'
               CASE ( DEBUG )
                  WRITE( s%out, 1003 ) 'DEBUG to CRAZY'
               END SELECT
            END SELECT
         END IF
         s%prev_control%print_level = control%print_level
      ELSE IF ( control%print_level >= ACTION .AND. &
                control%start_print <= 0            ) THEN
         SELECT CASE ( control%print_level )
         CASE ( ACTION )
            WRITE( s%out, 1004 ) 'ACTION'
         CASE ( DETAILS )
            WRITE( s%out, 1004 ) 'DETAILS'
         CASE ( DEBUG )
            WRITE( s%out, 1004 ) 'DEBUG'
         CASE ( CRAZY: )
            WRITE( s%out, 1004 ) 'CRAZY'
         END SELECT
      END IF
      s%print_level = control%print_level

!     Define the effective printing level at iteration 0

      IF ( control%start_print <= 0 ) THEN
         s%level = control%print_level
      ELSE
         s%level = SILENT
      END IF

!     Printout device

      IF ( control%out /= s%prev_control%out ) THEN
         control%out = MAX( 1, control%out )
         IF ( s%level >= ACTION ) THEN
            WRITE( s%prev_control%out, 1005 ) s%prev_control%out, control%out
            WRITE( control%out, 1005 ) s%prev_control%out, control%out
         END IF
         s%prev_control%out = control%out
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1006 ) control%out
      END IF
      s%out = control%out

!     Error printout device

      IF ( control%errout /= s%prev_control%errout ) THEN
         control%errout = MAX( 1, control%errout )
         IF ( s%level >= ACTION ) THEN
            WRITE( s%prev_control%out, 1007)s%prev_control%errout,control%errout
            WRITE( control%out, 1007 ) s%prev_control%errout, control%errout
         END IF
         s%prev_control%errout = control%errout
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1008 ) control%errout
      END IF

!     Stop iteration for printing

      IF ( control%stop_print /= s%prev_control%stop_print ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1009 ) control%stop_print
         END IF
         s%prev_control%stop_print = control%stop_print
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1010 ) control%stop_print
      END IF

!     Accuracy for the residual

      IF ( control%c_accuracy /= s%prev_control%c_accuracy ) THEN
         control%c_accuracy = ABS( control%c_accuracy )
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1011 ) s%prev_control%c_accuracy, control%c_accuracy
         END IF
         s%prev_control%c_accuracy = control%c_accuracy
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1012 ) control%c_accuracy
      END IF

!     Accuracy for the gradient

      IF ( control%g_accuracy /= s%prev_control%g_accuracy ) THEN
         control%g_accuracy = ABS( control%g_accuracy )
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1013 ) s%prev_control%g_accuracy, control%g_accuracy
         END IF
         s%prev_control%g_accuracy = control%g_accuracy
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1014 ) control%g_accuracy
      END IF

!     Stopping norm for the gradient

      IF ( control%stop_on_prec_g .NEQV. s%prev_control%stop_on_prec_g )       &
         s%prev_control%stop_on_prec_g = control%stop_on_prec_g
      IF ( control%stop_on_g_max .NEQV. s%prev_control%stop_on_g_max ) THEN
         IF ( control%stop_on_g_max .AND.                   &
              ( control%prec_used == USER_DEFINED .OR.      &
                ( control%prec_used == BANDED .AND.         &
                  control%semi_bandwidth > 0        )   )   ) THEN
            control%stop_on_g_max = .FALSE.
            WRITE( s%out, 1117 )
         ELSE
            s%prev_control%stop_on_g_max = control%stop_on_g_max
         END IF
      END IF
      IF ( s%level >= ACTION ) THEN
         IF ( control%stop_on_g_max ) THEN
            IF ( control%stop_on_prec_g ) THEN
               WRITE( s%out, 1015)
            ELSE
               WRITE( s%out, 1016)
            END IF
         ELSE
            IF ( control%stop_on_prec_g ) THEN
               WRITE( s%out, 1017)
            ELSE
            END IF
            WRITE( s%out,1018)
         END IF
      END IF

!     Maximum number of iterations

      IF ( control%max_iterations /= s%prev_control%max_iterations ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1019 ) s%prev_control%max_iterations,                &
                                 control%max_iterations
         END IF
         s%prev_control%max_iterations = control%max_iterations
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1020 ) control%max_iterations
      END IF

!     Maximum number of CG iterations

      IF ( control%max_cg_iterations /= s%prev_control%max_cg_iterations ) THEN
         control%max_cg_iterations = MAX( 1, control%max_cg_iterations )
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1021 ) s%prev_control%max_cg_iterations,             &
                                 control%max_cg_iterations
         END IF
         s%prev_control%max_cg_iterations = control%max_cg_iterations
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1022 ) control%max_cg_iterations
      END IF

!     Inequality penalty type

      IF ( control%inequality_penalty_type /=                                  &
           s%prev_control%inequality_penalty_type ) THEN
         SELECT CASE ( control%inequality_penalty_type )
         CASE ( 1, 2, 3, 4 )       ! allow  L1
!        CASE ( 2, 3, 4 )          ! forbid L1
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1023 ) s%prev_control%inequality_penalty_type,    &
                                    control%inequality_penalty_type
            END IF
            s%prev_control%model_type = control%model_type
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1024 ) control%inequality_penalty_type,           &
                                    s%prev_control%inequality_penalty_type
            END IF
            control%model_type = s%prev_control%model_type
         END SELECT
      ELSE
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1025 ) control%inequality_penalty_type
         END IF
      END IF

!     Select the smoothing parameter for the L1 penalty.

      IF ( control%inequality_penalty_type == 1 ) s%epsilon = TENTH

!     Model type

      IF ( control%model_type /= s%prev_control%model_type ) THEN
         IF ( s%level >= ACTION ) THEN
            SELECT CASE ( control%model_type )
            CASE ( GAUSS_NEWTON )
               SELECT CASE ( s%prev_control%model_type )
               CASE ( NEWTON )
                  WRITE( s%out, 1026 ) 'NEWTON to GAUSS_NEWTON'
               CASE ( AUTOMATIC )
                  WRITE( s%out, 1026 ) 'AUTOMATIC to GAUSS_NEWTON'
               END SELECT
            CASE ( NEWTON )
               SELECT CASE ( s%prev_control%model_type )
               CASE ( GAUSS_NEWTON )
                  WRITE( s%out, 1026 ) 'GAUSS_NEWTON to NEWTON'
               CASE ( AUTOMATIC )
                  WRITE( s%out, 1026 ) 'AUTOMATIC to NEWTON'
               END SELECT
            CASE ( AUTOMATIC )
               SELECT CASE ( s%prev_control%model_type )
               CASE ( GAUSS_NEWTON )
                  WRITE( s%out, 1026 ) 'GAUSS_NEWTON to AUTOMATIC'
               CASE ( NEWTON )
                  WRITE( s%out, 1026 ) 'NEWTON to AUTOMATIC'
               END SELECT
            END SELECT
         END IF
         s%prev_control%model_type = control%model_type
      ELSE
         IF ( s%level >= ACTION ) THEN
            SELECT CASE ( control%model_type )
            CASE ( GAUSS_NEWTON )
               WRITE( s%out, 1027 ) 'GAUSS_NEWTON'
            CASE ( NEWTON )
               WRITE( s%out, 1027 ) 'NEWTON'
            CASE ( AUTOMATIC )
               WRITE( s%out, 1027 ) 'AUTOMATIC'
            END SELECT
         END IF
      END IF

!     Automatic model criterion

      IF ( control%model_type == AUTOMATIC ) THEN
         IF ( control%model_criterion /= s%prev_control%model_criterion ) THEN
            SELECT CASE ( control%model_criterion )
            CASE ( BEST_FIT )
               IF ( s%level >= ACTION )                                        &
                  WRITE( s%out, 1028 ) 'BEST_REDUCTION to BEST_FIT'
               s%prev_control%model_criterion = control%model_criterion
            CASE ( BEST_REDUCTION )
               IF ( s%level >= ACTION )                                        &
                  WRITE( s%out, 1028 ) 'BEST_FIT to BEST_REDUCTION'
               s%prev_control%model_criterion = control%model_criterion
            CASE DEFAULT
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, 1103 )
                  SELECT CASE ( control%model_criterion )
                  CASE ( BEST_FIT )
                     WRITE( s%out, 1104 ) 'BEST_FIT'
                  CASE ( BEST_REDUCTION )
                     WRITE( s%out, 1104 ) 'BEST_REDUCTION'
                  END SELECT
               END IF
               control%model_criterion = s%prev_control%model_criterion
            END SELECT
         ELSE
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( control%model_criterion )
               CASE ( BEST_FIT )
                  WRITE( s%out, 1029 ) 'BEST_FIT'
               CASE ( BEST_REDUCTION )
                  WRITE( s%out, 1029 ) 'BEST_REDUCTION'
               END SELECT
            END IF
         END IF

!     Automatic model inertia

         control%model_inertia = MAX( 1, ABS( control%model_inertia ) )
         IF ( control%model_inertia /= s%prev_control%model_inertia ) THEN
            IF ( s%stage == READY ) THEN
               IF ( s%level >= ACTION )                                        &
                  WRITE( s%out, 1094 ) s%prev_control%model_inertia,           &
                                       control%model_inertia
               s%prev_control%model_inertia = control%model_inertia
            ELSE
               IF ( s%level >= ACTION )                                        &
                  WRITE( s%out, 1095 ) s%prev_control%model_inertia
               control%model_inertia = s%prev_control%model_inertia
            END IF
         ELSE
            IF ( s%level >= ACTION ) WRITE( s%out, 1030 ) control%model_inertia
         END IF
      END IF

!     Minimum subproblem accuracy

      IF ( control%min_gltr_accuracy /= s%prev_control%min_gltr_accuracy ) THEN
         IF ( control%min_gltr_accuracy > ZERO .AND. &
              control%min_gltr_accuracy < ONE        ) THEN
            IF ( s%level >= ACTION )                                           &
               WRITE( s%out, 1031 ) s%prev_control%min_gltr_accuracy,          &
                                    control%min_gltr_accuracy
            s%prev_control%min_gltr_accuracy = control%min_gltr_accuracy
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1099 ) control%min_gltr_accuracy
               WRITE( s%out, 1108 ) s%prev_control%min_gltr_accuracy
            END IF
            control%min_gltr_accuracy = s%prev_control%min_gltr_accuracy
         END IF
      ELSE
         IF ( s%level >= ACTION ) WRITE( s%out, 1032 ) control%min_gltr_accuracy
      END IF

!     Subproblem accuracy power

      IF ( control%gltr_accuracy_power /= s%prev_control%gltr_accuracy_power ) &
         THEN
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, 1097 ) s%prev_control%gltr_accuracy_power,           &
                                 control%gltr_accuracy_power
         s%prev_control%gltr_accuracy_power = control%gltr_accuracy_power
      ELSE
         IF ( s%level >= ACTION )WRITE( s%out,1098 ) control%gltr_accuracy_power
      END IF

!     Preconditioner used

      IF ( control%prec_used /= s%prev_control%prec_used ) THEN
         SELECT CASE ( control%prec_used )
         CASE ( NONE )
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( s%prev_control%prec_used )
               CASE ( BANDED )
                  WRITE( s%out, 1033 ) 'BANDED to NONE'
               CASE ( USER_DEFINED )
                  WRITE( s%out, 1033 ) 'USER_DEFINED to NONE'
               END SELECT
            END IF
            s%prev_control%prec_used = control%prec_used
         CASE ( BANDED )
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( s%prev_control%prec_used )
               CASE ( NONE )
                  WRITE( s%out, 1033 ) 'NONE to BANDED'
               CASE ( USER_DEFINED )
                  WRITE( s%out, 1033 ) 'USER_DEFINED to BANDED'
               END SELECT
            END IF
            s%prev_control%prec_used = control%prec_used
         CASE ( USER_DEFINED )
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( s%prev_control%prec_used )
               CASE ( NONE )
                  WRITE( s%out, 1033 ) 'NONE to USER_DEFINED'
               CASE ( BANDED )
                  WRITE( s%out, 1033 ) 'BANDED to USER_DEFINED'
               END SELECT
            END IF
            s%prev_control%prec_used = control%prec_used
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1105 )
               SELECT CASE( s%prev_control%prec_used )
               CASE ( NONE )
                  WRITE( s%out, 1104 ) 'NONE'
               CASE ( BANDED )
                  WRITE( s%out, 1104 ) 'BANDED'
               CASE ( USER_DEFINED )
                  WRITE( s%out, 1104 ) 'USER_DEFINED'
               END SELECT
            END IF
            control%prec_used = s%prev_control%prec_used
         END SELECT
      ELSE
         IF ( s%level >= ACTION ) THEN
            SELECT CASE ( control%prec_used )
            CASE ( NONE )
               WRITE( s%out, 1034 ) 'NONE'
            CASE ( BANDED )
               WRITE( s%out, 1034 ) 'BANDED'
            CASE ( USER_DEFINED )
               WRITE( s%out, 1034 ) 'USER_DEFINED'
            END SELECT
         END IF
      END IF

!     Band preconditioner semi-bandwidth

      IF ( control%prec_used == BANDED ) THEN
         IF ( control%semi_bandwidth /= s%prev_control%semi_bandwidth ) THEN
            IF ( s%level >= ACTION .AND. control%prec_used /= NONE )           &
              WRITE( s%out, 1035 ) s%prev_control%semi_bandwidth,              &
                                   control%semi_bandwidth
            s%prev_control%semi_bandwidth = control%semi_bandwidth
            s%nsemib = MAX( 0, MIN( control%semi_bandwidth, problem%n - 1 ) )
         ELSE
            IF ( s%level >= ACTION .AND. control%prec_used /= NONE )           &
               WRITE( s%out, 1036 ) control%semi_bandwidth
         END IF
      END IF

!     External Jacobian products

      IF (control%external_J_products.NEQV.s%prev_control%external_J_products )&
         s%prev_control%external_J_products = control%external_J_products
      IF ( s%level >= ACTION ) THEN
         IF ( control%external_J_products ) THEN
            WRITE( s%out, 1037 )
         ELSE
            WRITE( s%out, 1038 )
         END IF
      END IF

!     Subproblem accuracy

      IF ( control%subproblem_accuracy /= s%prev_control%subproblem_accuracy ) &
         THEN
         SELECT CASE ( control%subproblem_accuracy )
         CASE ( ADAPTIVE )
            IF ( s%level >= ACTION ) WRITE( s%out, 1039 ) 'FULL to ADAPTIVE'
            s%prev_control%subproblem_accuracy = control%subproblem_accuracy
         CASE ( FULL )
            IF ( s%level >= ACTION ) WRITE( s%out, 1039 ) 'ADAPTIVE to FULL'
            s%prev_control%subproblem_accuracy = control%subproblem_accuracy
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1106 )
               SELECT CASE ( s%prev_control%subproblem_accuracy )
               CASE ( ADAPTIVE )
                  WRITE( s%out, 1104 ) 'ADAPTIVE'
               CASE ( FULL )
                  WRITE( s%out, 1104 ) 'FULL'
               END SELECT
            END IF
            control%subproblem_accuracy = s%prev_control%subproblem_accuracy
         END SELECT
      ELSE
         IF ( s%level >= ACTION ) THEN
            SELECT CASE ( control%subproblem_accuracy )
            CASE ( ADAPTIVE )
               WRITE( s%out, 1040 ) 'ADAPTIVE'
            CASE ( FULL )
               WRITE( s%out, 1040 ) 'FULL'
            END SELECT
         END IF
      END IF

!     Equations grouping

      IF ( control%grouping /= s%prev_control%grouping ) THEN
         IF ( s%stage == READY ) THEN
            SELECT CASE ( control%grouping )
            CASE ( NONE )
               IF ( s%level >= ACTION ) THEN
                  SELECT CASE ( s%prev_control%grouping )
                  CASE ( AUTOMATIC )
                     WRITE( s%out, 1041 ) 'AUTOMATIC to NONE'
                  CASE ( USER_DEFINED )
                     WRITE( s%out, 1041 ) 'USER_DEFINED to NONE'
                  END SELECT
               END IF
               s%prev_control%grouping = control%grouping
            CASE ( AUTOMATIC )
               IF ( s%level >= ACTION ) THEN
                  SELECT CASE ( s%prev_control%grouping )
                  CASE ( NONE )
                     WRITE( s%out, 1041 ) 'NONE to AUTOMATIC'
                  CASE ( USER_DEFINED )
                     WRITE( s%out, 1041 ) 'USER_DEFINED to AUTOMATIC'
                  END SELECT
               END IF
               s%prev_control%grouping = control%grouping
            CASE ( USER_DEFINED )
               IF ( s%level >= ACTION ) THEN
                  SELECT CASE ( s%prev_control%grouping )
                  CASE ( NONE )
                     WRITE( s%out, 1041 ) 'NONE to USER_DEFINED'
                  CASE ( AUTOMATIC )
                     WRITE( s%out, 1041 ) 'AUTOMATIC to USER_DEFINED'
                  END SELECT
               END IF
               s%prev_control%grouping = control%grouping
            CASE DEFAULT
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, 1107 )
                  SELECT CASE ( s%prev_control%grouping )
                  CASE ( NONE )
                     WRITE( s%out, 1104 ) 'NONE'
                  CASE ( AUTOMATIC )
                     WRITE( s%out, 1104 ) 'AUTOMATIC'
                  CASE ( USER_DEFINED )
                     WRITE( s%out, 1104 ) 'USER_DEFINED'
                  END SELECT
               END IF
               control%grouping = s%prev_control%grouping
            END SELECT
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1096 )
               SELECT CASE ( s%prev_control%grouping )
               CASE ( NONE )
                  WRITE( s%out, 1104 ) 'NONE'
               CASE ( AUTOMATIC )
                  WRITE( s%out, 1104 ) 'AUTOMATIC'
               CASE ( USER_DEFINED )
                  WRITE( s%out, 1104 ) 'USER_DEFINED'
               END SELECT
            END IF
            control%grouping = s%prev_control%grouping
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%grouping )
         CASE ( NONE )
            WRITE( s%out, 1042 ) 'NONE'
         CASE ( AUTOMATIC )
            WRITE( s%out, 1042 ) 'AUTOMATIC'
         CASE ( USER_DEFINED )
            WRITE( s%out, 1042 ) 'USER_DEFINED'
         END SELECT
      END IF

!     Number of groups for automatic grouping

      IF ( control%nbr_groups /= s%prev_control%nbr_groups ) THEN
         IF ( s%stage == READY ) THEN
            IF ( s%level >= ACTION .AND. control%grouping == AUTOMATIC ) THEN
               WRITE( s%out, 1043 ) s%prev_control%nbr_groups,                 &
                                    control%nbr_groups
            END IF
            s%prev_control%nbr_groups = control%nbr_groups
         ELSE
            IF ( s%level >= ACTION .AND. control%grouping == AUTOMATIC ) THEN
               WRITE( s%out, 1044 ) s%prev_control%nbr_groups
            END IF
            control%nbr_groups = s%prev_control%nbr_groups
         END IF
      ELSE IF ( s%level >= ACTION .AND. control%grouping == AUTOMATIC ) THEN
         WRITE( s%out, 1045 ) control%nbr_groups
      END IF

!     Group balancing

      IF ( control%grouping == AUTOMATIC ) THEN
         IF ( control%balance_group_values .NEQV.                              &
              s%prev_control%balance_group_values ) THEN
            IF ( s%stage == READY ) THEN
               IF ( s%level >= ACTION ) THEN
                  IF ( control%balance_group_values ) THEN
                     WRITE( s%out, 1046 )
                  ELSE
                     WRITE( s%out, 1047 )
                  END IF
               END IF
               s%prev_control%balance_group_values =control%balance_group_values
            ELSE
               IF ( s%level >= ACTION ) THEN
                  IF ( control%balance_group_values ) THEN
                     WRITE( s%out, 1048 )
                  ELSE
                     WRITE( s%out, 1049 )
                  END IF
               END IF
               control%balance_group_values =s%prev_control%balance_group_values
            END IF
         ELSE IF ( s%level >= ACTION ) THEN
            IF ( control%balance_group_values ) THEN
               WRITE( s%out, 1050 )
            ELSE
               WRITE( s%out, 1051 )
            END IF
         END IF
      END IF

!     Weak acceptance test

      IF ( control%weak_accept_power /= s%prev_control%weak_accept_power )     &
         s%prev_control%weak_accept_power = control%weak_accept_power
      IF ( s%level >= ACTION ) THEN
         IF ( control%weak_accept_power >= ZERO ) THEN
            WRITE( s%out, 1052 ) control%weak_accept_power
         ELSE
            WRITE( s%out, 1053 )
         END IF
      END IF

!     Minimum weak acceptance factor

      IF ( control%min_weak_accept_factor /=     &
           s%prev_control%min_weak_accept_factor ) THEN
         IF ( control%min_weak_accept_factor > ZERO .AND. &
              control%min_weak_accept_factor < ONE        ) THEN
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1100 ) s%prev_control%min_weak_accept_factor,     &
                                    control%min_weak_accept_factor
            END IF
            s%prev_control%min_weak_accept_factor=control%min_weak_accept_factor
         ELSE
            IF ( s%level >= ACTION )                                           &
               WRITE( s%out, 1101 ) control%min_weak_accept_factor,            &
                                    s%prev_control%min_weak_accept_factor
            control%min_weak_accept_factor=s%prev_control%min_weak_accept_factor
         END IF
      ELSE
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1102 ) control%min_weak_accept_factor
         END IF
      END IF

!     Trust-region parameter : initial_radius

      IF ( control%initial_radius /= s%prev_control%initial_radius ) THEN
         control%initial_radius = ABS( control%initial_radius )
         IF ( s%stage == READY ) THEN
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1054 ) s%prev_control%initial_radius,             &
                                    control%initial_radius
            END IF
            s%prev_control%initial_radius  = control%initial_radius
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1055 )
               WRITE( s%out, 1108 ) s%prev_control%initial_radius
            END IF
            control%initial_radius = s%prev_control%initial_radius
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1056 ) control%initial_radius
      END IF

!     Trust-region parameter : eta_1

      IF ( control%eta_1 /= s%prev_control%eta_1 ) THEN
         IF ( control%eta_1 > ZERO .AND. control%eta_1 < ONE ) THEN
            IF ( s%level >= ACTION )                                           &
               WRITE( s%out, 1057 ) s%prev_control%eta_1, control%eta_1
            s%prev_control%eta_1 = control%eta_1
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1109 )
               WRITE( s%out, 1108 ) s%prev_control%eta_1
            END IF
            control%eta_1 = s%prev_control%eta_1
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1058 ) control%eta_1
      END IF

!     Trust-region parameter : eta_2

      IF ( control%eta_2 /= s%prev_control%eta_2 ) THEN
         IF ( control%eta_2 > control%eta_1 .AND. control%eta_2 < ONE ) THEN
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1059 ) s%prev_control%eta_2, control%eta_2
            END IF
            s%prev_control%eta_2 = control%eta_2
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1110 )
               WRITE( s%out, 1108 ) s%prev_control%eta_2
            END IF
            control%eta_2 = s%prev_control%eta_2
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1060 ) control%eta_2
      END IF

!     Trust-region parameter : gamma_0

      IF ( control%gamma_0 /= s%prev_control%gamma_0 ) THEN
         IF ( control%gamma_0 > ZERO .AND. control%gamma_0 < ONE ) THEN
            IF ( s%level >= ACTION )                                           &
               WRITE( s%out, 1061 ) s%prev_control%gamma_0, control%gamma_0
            s%prev_control%gamma_0 = control%gamma_0
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1111 )
               WRITE( s%out, 1108 ) s%prev_control%gamma_0
            END IF
            control%gamma_0 = s%prev_control%gamma_0
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1062 ) control%gamma_0
      END IF

!     Trust-region parameter : gamma_1

      IF ( control%gamma_1 /= s%prev_control%gamma_1 ) THEN
         IF ( control%gamma_1 > control%gamma_0 .AND. &
              control%gamma_1 < ONE                   ) THEN
            IF ( s%level >= ACTION )                                           &
               WRITE( s%out, 1063 ) s%prev_control%gamma_1, control%gamma_1
            s%prev_control%gamma_1 = control%gamma_1
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1112 )
               WRITE( s%out, 1108 ) s%prev_control%gamma_1
            END IF
            control%gamma_1 = s%prev_control%gamma_1
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1064 ) control%gamma_1
      END IF

!     Trust-region parameter : gamma_2

      IF ( control%gamma_2 /= s%prev_control%gamma_2 ) THEN
         IF ( control%gamma_2 > ONE ) THEN
            IF ( s%level >= ACTION )                                           &
               WRITE( s%out, 1065 ) s%prev_control%gamma_2, control%gamma_2
            s%prev_control%gamma_2 = control%gamma_2
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1113 )
               WRITE( s%out, 1108 ) s%prev_control%gamma_2
            END IF
            control%gamma_2 = s%prev_control%gamma_2
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, 1066 ) control%gamma_2
      END IF

!     Filter usage

      IF ( control%use_filter /= s%prev_control%use_filter ) THEN
         SELECT CASE( control%use_filter )
         CASE ( NEVER )
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( s%prev_control%use_filter )
               CASE ( INITIAL )
                  WRITE( s%out, 1067 ) 'INITIAL to NEVER'
               CASE ( ALWAYS )
                  WRITE( s%out, 1067 ) 'ALWAYS to NEVER'
               END SELECT
            END IF
            s%prev_control%use_filter = control%use_filter
         CASE ( INITIAL )
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( s%prev_control%use_filter )
               CASE ( NEVER )
                  WRITE( s%out, 1067 ) 'NEVER to INITIAL'
               CASE ( ALWAYS )
                  WRITE( s%out, 1067 ) 'ALWAYS to INITIAL'
               END SELECT
            END IF
            s%prev_control%use_filter = control%use_filter
         CASE ( ALWAYS )
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( s%prev_control%use_filter )
               CASE ( NEVER )
                  WRITE( s%out, 1067 ) 'NEVER to ALWAYS'
               CASE ( INITIAL )
                  WRITE( s%out, 1067 ) 'INITIAL to ALWAYS'
               END SELECT
            END IF
            s%prev_control%use_filter = control%use_filter
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, 1114 )
               SELECT CASE ( s%prev_control%use_filter )
               CASE ( NEVER )
                  WRITE( s%out, 1104 ) 'NEVER'
               CASE ( INITIAL )
                  WRITE( s%out, 1104 ) 'INITIAL'
               CASE ( ALWAYS )
                  WRITE( s%out, 1104 ) 'ALWAYS'
               END SELECT
            END IF
            control%use_filter = s%prev_control%use_filter
         END SELECT
      ELSE
         IF ( s%level >= ACTION ) THEN
            SELECT CASE( control%use_filter )
            CASE ( NEVER )
               WRITE( s%out, 1068 ) 'NEVER'
            CASE ( INITIAL )
               WRITE( s%out, 1068 ) 'INITIAL'
            CASE ( ALWAYS )
               WRITE( s%out, 1068 ) 'ALWAYS'
            END SELECT
         END IF
      END IF

!     Filter sign restriction

      s%prev_control%filter_sign_restriction = control%filter_sign_restriction
      IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         IF ( control%filter_sign_restriction ) THEN
            WRITE( s%out, 1069 )
         ELSE
            WRITE( s%out, 1070 )
         END IF
      END IF

!     Filter margin type

      IF ( control%margin_type /= s%prev_control%margin_type ) THEN
         IF ( control%use_filter /= NEVER ) THEN
            IF ( s%stage == READY ) THEN
               SELECT CASE( control%margin_type )
               CASE ( FIXED )
                  IF ( s%level >= ACTION  ) THEN
                     SELECT CASE ( s%prev_control%margin_type )
                     CASE ( CURRENT )
                        WRITE( s%out, 1071 ) 'CURRENT to FIXED'
                     CASE ( SMALLEST )
                        WRITE( s%out, 1071 ) 'SMALLEST to FIXED'
                     END SELECT
                  END IF
                  s%prev_control%margin_type = control%margin_type
               CASE ( CURRENT )
                  IF ( s%level >= ACTION  ) THEN
                     SELECT CASE ( s%prev_control%use_filter )
                     CASE ( FIXED )
                        WRITE( s%out, 1071 ) 'FIXED to CURRENT'
                     CASE ( SMALLEST )
                        WRITE( s%out, 1071 ) 'SMALLEST to CURRENT'
                     END SELECT
                  END IF
                  s%prev_control%margin_type = control%margin_type
               CASE ( SMALLEST )
                  IF ( s%level >= ACTION  ) THEN
                     SELECT CASE ( s%prev_control%use_filter )
                     CASE ( FIXED )
                        WRITE( s%out, 1071 ) 'FIXED to SMALLEST'
                     CASE ( CURRENT )
                        WRITE( s%out, 1071 ) 'CURRENT to SMALLEST'
                     END SELECT
                  END IF
                  s%prev_control%margin_type = control%margin_type
               CASE DEFAULT
                  IF ( s%level >= ACTION ) THEN
                     WRITE( s%out, 1115 )
                     SELECT CASE ( s%prev_control%margin_type )
                     CASE ( FIXED )
                        WRITE( s%out, 1104 ) 'FIXED'
                     CASE ( CURRENT )
                        WRITE( s%out, 1104 ) 'CURRENT'
                     CASE ( SMALLEST )
                        WRITE( s%out, 1104 ) 'SMALLEST'
                     END SELECT
                  END IF
                  control%margin_type = s%prev_control%margin_type
               END SELECT
            ELSE
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, 1072 )
                  SELECT CASE ( s%prev_control%margin_type )
                  CASE ( FIXED )
                        WRITE( s%out, 1104 ) 'FIXED'
                  CASE ( CURRENT )
                        WRITE( s%out, 1104 ) 'CURRENT'
                  CASE ( SMALLEST )
                        WRITE( s%out, 1104 ) 'SMALLEST'
                  END SELECT
               END IF
               control%margin_type = s%prev_control%margin_type
            END IF
         END IF
      ELSE
         IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
            SELECT CASE( control%margin_type )
            CASE ( FIXED )
               WRITE( s%out, 1073 ) 'FIXED'
            CASE ( CURRENT )
               WRITE( s%out, 1073 ) 'CURRENT'
            CASE ( SMALLEST )
               WRITE( s%out, 1073 ) 'SMALLEST'
            END SELECT
         END IF
      END IF

!     Maximal filter size

      IF ( control%maximal_filter_size /= s%prev_control%maximal_filter_size ) &
         THEN
         control%maximal_filter_size = MAX( 0, control%maximal_filter_size )
         IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
            WRITE( s%out, 1074 ) s%prev_control%maximal_filter_size,           &
                                 control%maximal_filter_size
         END IF
         s%prev_control%maximal_filter_size = control%maximal_filter_size
      ELSE IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1075 ) control%maximal_filter_size
      END IF

!     Filter parameter : the maximum margin size

      IF ( control%gamma_f /= s%prev_control%gamma_f ) THEN
         IF ( control%gamma_f > ZERO .AND. control%gamma_f < ONE ) THEN
            IF ( s%level >= ACTION .AND. control%use_filter /= NEVER )         &
               WRITE( s%out, 1076 ) s%prev_control%gamma_f, control%gamma_f
            s%prev_control%gamma_f = control%gamma_f
         ELSE
            IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
               WRITE( s%out, 1116 )
               WRITE( s%out, 1108 ) s%prev_control%gamma_f
            END IF
            control%gamma_f = s%prev_control%gamma_f
         END IF
      ELSE IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1077 ) control%gamma_f
      END IF

!     Filter parameter : filter size increment

      IF (control%filter_size_increment/=s%prev_control%filter_size_increment) &
         THEN
         control%filter_size_increment = MAX( 1, control%filter_size_increment )
         IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
            WRITE( s%out, 1078 ) s%prev_control%filter_size_increment,         &
                                 control%filter_size_increment
         END IF
         s%prev_control%filter_size_increment = control%filter_size_increment
      ELSE IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1079 ) control%filter_size_increment
      END IF

!     Removal of dominated filter entries

      s%prev_control%remove_dominated = control%remove_dominated
      IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         IF ( control%remove_dominated ) THEN
            WRITE( s%out, 1080 )
         ELSE
            WRITE( s%out, 1081 )
         END IF
      END IF

!     Initial TR relaxation factor

      IF ( control%itr_relax /= s%prev_control%itr_relax ) THEN
         control%itr_relax = MAX( ONE, control%itr_relax )
         IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
            WRITE( s%out, 1082 ) s%prev_control%itr_relax, control%itr_relax
         END IF
         s%prev_control%itr_relax = control%itr_relax
      ELSE IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1083 ) control%itr_relax
      END IF

!     Secondary TR relaxation factor

      IF ( control%str_relax /= s%prev_control%str_relax ) THEN
         control%str_relax = MAX( ONE, control%str_relax )
         IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
            WRITE( s%out, 1084 ) s%prev_control%str_relax, control%str_relax
         END IF
         s%prev_control%str_relax = control%str_relax
      ELSE IF ( s%level >= ACTION .AND. control%use_filter /= NEVER ) THEN
         WRITE( s%out, 1085 ) control%str_relax
      END IF

!     Save best point

      IF ( control%save_best_point .NEQV. s%prev_control%save_best_point ) THEN
         s%prev_control%save_best_point = control%save_best_point
      END IF
      IF ( s%level >= ACTION ) THEN
         IF ( control%save_best_point ) THEN
            WRITE( s%out, 1086 )
         ELSE
            WRITE( s%out, 1087 )
         END IF
      END IF

!     Checkpointing frequency

      IF ( control%checkpoint_freq /= s%prev_control%checkpoint_freq ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, 1088 )                         &
            s%prev_control%checkpoint_freq, control%checkpoint_freq
         s%prev_control%checkpoint_freq = control%checkpoint_freq
      ELSE IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1089 ) control%checkpoint_freq
      END IF

!     Checkpointing device

      IF ( control%checkpoint_dev /= s%prev_control%checkpoint_dev ) THEN
         control%checkpoint_dev = MAX( 1, control%checkpoint_dev )
         IF ( s%level >= ACTION ) WRITE( s%out, 1090 )                         &
            s%prev_control%checkpoint_dev,  control%checkpoint_dev
         s%prev_control%checkpoint_dev = control%checkpoint_dev
      ELSE IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1091 ) control%checkpoint_dev
      END IF

!     Checkpointing file

      IF ( control%checkpoint_file /= s%prev_control%checkpoint_file ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, 1092 )                         &
            s%prev_control%checkpoint_file, control%checkpoint_file
         s%prev_control%checkpoint_file = control%checkpoint_file
      ELSE IF ( s%level >= ACTION ) THEN
            WRITE( s%out, 1093 ) control%checkpoint_file
      END IF

      RETURN

!     Formats

1000  FORMAT(1x,'verifying user-defined presolve control parameters')
1001  FORMAT(2x,'now starts printing at iteration',i6)
1002  FORMAT(2x,'starts printing at iteration',i6)
1003  FORMAT(2x,'print level changed from ',a17)
1004  FORMAT(2x,'print level is ',a7)
1005  FORMAT(2x,'print device changed from ',i2,' to ',i2)
1006  FORMAT(2x,'print device is ',i2)
1007  FORMAT(2x,'error output device changed from ',i2,' to ',i2)
1008  FORMAT(2x,'error output device is ',i2)
1009  FORMAT(2x,'now stops printing at iteration',i6)
1010  FORMAT(2x,'stops printing at iteration',i6)
1011  FORMAT(2x,'residual accuracy changed from ',1pE11.3,' to ',1pE11.3)
1012  FORMAT(2x,'residual accuracy is ',1pE11.3)
1013  FORMAT(2x,'gradient accuracy changed from ',1pE11.3,' to ',1pE11.3)
1014  FORMAT(2x,'gradient accuracy is ',1pE11.3)
1015  FORMAT(2x,'stopping the preconditioned maximum gradient norm')
1016  FORMAT(2x,'stopping the unpreconditioned maximum gradient norm')
1017  FORMAT(2x,'stopping the preconditioned Euclidean gradient norm')
1018  FORMAT(2x,'stopping the unpreconditioned Euclidean gradient norm')
1019  FORMAT(2x,'maximum number of iterations changed from ',i6,' to ',i6)
1020  FORMAT(2x,'maximum number of iterations is ',i6)
1021  FORMAT(2x,'maximum number of CG iterations changed from ',i6,            &
             ' * n to ',i6,' * n')
1022  FORMAT(2x,'maximum number of CG iterations is ',i6,' * n')
1023  FORMAT(2x,'inequality penalty type changed from ',i1,' to ',i1)
1024  FORMAT(2x,'FILTRANE WARNING: attempt to change the inequality penalty',/,&
             2x,'                  to unknown type.',i1,'!',/,                 &
             2x,'                  Keeping the value ',i1)
1025  FORMAT(2x,'inequality penalty type is ',i1)
1026  FORMAT(2x,'model type changed from ',a)
1027  FORMAT(2x,'model type is ',a)
1028  FORMAT(2x,'model criterion changed from ',a)
1029  FORMAT(2x,'model criterion is ',a)
1030  FORMAT(2x,'automatic model inertia is ',i3)
1031  FORMAT(2x,'minimum subproblem accuracy changed from ',1pE11.3,' to ',    &
             1pE11.3)
1032  FORMAT(2x,'minimum subproblem accuracy is ',1pE11.3)
1033  FORMAT(2x,'preconditioner changed from ',a)
1034  FORMAT(2x,'preconditioner is ',a)
1035  FORMAT(2x,'preconditioner semi-bandwidth changed from ',i6,' to ',i6)
1036  FORMAT(2x,'preconditioner semi-bandwidth is ',i6)
1037  FORMAT(2x,'external Jacobian products')
1038  FORMAT(2x,'internal Jacobian products')
1039  FORMAT(2x,'subproblem accuracy changed from ',a)
1040  FORMAT(2x,'subproblem accuracy is ',a)
1041  FORMAT(2x,'grouping mode changed from ',a)
1042  FORMAT(2x,'grouping mode is ',a)
1043  FORMAT(2x,'number of groups for automatic grouping changed from ',i6,    &
                ' to ',i6)
1044  FORMAT(2x,'FILTRANE WARNING: attempt to modify the number of automatic', &
                ' equation groups!',/,                                         &
             2x,'                  Keeping the value',i6)
1045  FORMAT(2x,'number of groups for automatic grouping is ',i6)
1046  FORMAT(2x,'group balancing activated')
1047  FORMAT(2x,'group balancing deactivated')
1048  FORMAT(2x,'FILTRANE WARNING: attempt to activate group balancing!',/,    &
             2x,'                  Keeping unbalanced groups.')
1049  FORMAT(2x,'FILTRANE WARNING: attempt to deactivate group balancing!',/,  &
             2x,'                  Keeping balanced groups.')
1050  FORMAT(2x,'attempting to balance group values')
1051  FORMAT(2x,'not attempting to balance group values')
1052  FORMAT(2x,'weak acceptance test enabled with power',1pE11.3)
1053  FORMAT(2x,'weak acceptance test disabled')
1054  FORMAT(2x,'initial trust-region radius changed from ',1pE11.3,' to ',    &
             1PE11.3)
1055  FORMAT(2x,'FILTRANE WARNING: attempt to modify the initial ',            &
                'trust-region radius!')
1056  FORMAT(2x,'initial trust-region radius is ',1pE11.3)
1057  FORMAT(2x,'minimum rho for successful iteration changed from',/,         &
             2x,'            ',1pE11.3,' to ',1pE11.3)
1058  FORMAT(2x,'minimum rho for successful iteration is',1pE11.3)
1059  FORMAT(2x,'minimum rho for very successful iteration changed from',/,    &
             2x,'            ',1pE11.3,' to ',1pE11.3)
1060  FORMAT(2x,'minimum rho for very successful iteration is',1pE11.3)
1061  FORMAT(2x,'worst-case radius reduction factor changed from',/,           &
             2x,'            ',1pE11.3,' to ',1pE11.3)
1062  FORMAT(2x,'worst-case radius reduction factor is',1pE11.3)
1063  FORMAT(2x,'radius reduction factor changed from',1pE11.3,' to ',1pE11.3)
1064  FORMAT(2x,'radius reduction factor is',1pE11.3)
1065  FORMAT(2x,'radius increase factor changed from',1pE11.3,' to ',1pE11.3)
1066  FORMAT(2x,'radius increase factor is',1pE11.3)
1067  FORMAT(2x,'filter usage changed from ',a)
1068  FORMAT(2x,'filter usage is ',a)
1069  FORMAT(2X,'filter sign restriction is enabled')
1070  FORMAT(2X,'filter sign restriction is disabled')
1071  FORMAT(2x,'filter margin type changed from ',a)
1072  FORMAT(2x,'FILTRANE WARNING: attempt to modify the margin type!')
1073  FORMAT(2x,'filter margin type is ',a)
1074  FORMAT(2x,'maximal filter size changed from ',i6,' to ',i6)
1075  FORMAT(2x,'maximal filter size is ',i6,' to ',i6)
1076  FORMAT(2x,'filter margin factor changed from ',1pE11.3,' to ',1pE11.3)
1077  FORMAT(2x,'filter margin factor is ',1pE11.3)
1078  FORMAT(2x,'filter size increment changed from ',i6,' to ',i6)
1079  FORMAT(2x,'filter size increment is ',i6,' to ',i6)
1080  FORMAT(2x,'dominated filter entries are removed')
1081  FORMAT(2x,'dominated filter entries are not removed')
1082  FORMAT(2x,'initial TR relaxation factor changed from ',1pE11.3,          &
                 ' to ',1pE11.3)
1083  FORMAT(2x,'initial TR relaxation factor is ',1pE11.3)
1084  FORMAT(2x,'secondary TR relaxation factor changed from ',1pE11.3,        &
                ' to ',1pE11.3)
1085  FORMAT(2x,'secondary TR relaxation factor is ',1pE11.3)
1086  FORMAT(2x,'saving best point found')
1087  FORMAT(2x,'not saving best point found')
1088  FORMAT(2x,'checkpointing frequency changed from ',i6,' to ',i6)
1089  FORMAT(2x,'checkpointing frequency is ',i6)
1090  FORMAT(2x,'checkpointing device changed from ',i2,' to ',i2)
1091  FORMAT(2x,'checkpointing device is ',i2)
1092  FORMAT(2x,'checkpointing file changed from ',a,/,' to ',a)
1093  FORMAT(2x,'checkpointing file is ',a)
1094  FORMAT(2x,'automatic model inertia changed from ',i3,' to ',i3)
1095  FORMAT(2x,'FILTRANE WARNING: attempt to attempt to modify' ,             &
                                   ' the automatic model inertia!',            &
             2x,'                  Keeping the value ',i3)
1096  FORMAT(2x,'FILTRANE WARNING: attempt to attempt to modify' ,             &
                                   ' the grouping strategy!')
1097  FORMAT(2x,'subproblem accuracy power changed from ',1pE11.3,' to ',      &
             1pE11.3)
1098  FORMAT(2x,'subproblem accuracy power is ',1pE11.3)
1099  FORMAT(2x,'FILTRANE WARNING: unacceptable value (',1pE11.3,') for ',     &
                                  'the minimum GLTR accuracy.')
1100  FORMAT(2x,'minimum weak acceptance factor changed from ',1pE11.3,' to ', &
             1pE11.3)
1101  FORMAT(2x,'FILTRANE WARNING: unacceptable value (',1pE11.3,') for ',     &
                                  'the minimum weak acceptance factor.',/,     &
             2x,'                  Keeping the value ',1pE11.3)
1102  FORMAT(2x,'minimum weak acceptance factor is ',1pE11.3)
1103  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the automatic model',&
                                  ' criterion.' )
1104  FORMAT(2x,'                  Keeping the value ', a)
1105  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the preconditioner', &
                                  ' choice.' )
1106  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the subproblem',     &
                                  ' accuracy choice.' )
1107  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the grouping',       &
                                  ' strategy.' )
1108  FORMAT(2x,'                  Keeping the value ',1pE11.3)
1109  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the trust-region ',  &
                                   'eta_1 parameter.')
1110  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the trust-region ',  &
                                   'eta_2 parameter.')
1111  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the trust-region ',  &
                                   'gamma_0 parameter.')
1112  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the trust-region ',  &
                                   'gamma_1 parameter.')
1113  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the trust-region ',  &
                                   'gamma_2 parameter.')
1114  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the filter usage')
1115  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the filter margin ', &
                                   'type.')
1116  FORMAT(2x,'FILTRANE WARNING: unacceptable value for the filter margin ', &
                                   'parameter gamma_f.')
1117  FORMAT(2x,'FILTRANE WARNING: stopping the maximum norm of the gradient',/&
       2x,'                  is not supported for nondiagonal preconditioning.')

      END SUBROUTINE FILTRANE_revise_control

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_add_to_filter

!     Adds the current theta value to the filter, if memory allows.

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

!     Local variables

      INTEGER :: iostat, new_capacity, k, pos, j, size, nxt, prv
      LOGICAL :: found
      REAL ( KIND = wp ) :: marginf, marginc, normpos, normtheta
      LOGICAL, DIMENSION( : ), POINTER :: active
      INTEGER, DIMENSION( : ), POINTER :: fnext
      REAL ( KIND = wp ), DIMENSION( : ),   POINTER :: fnorm
      REAL ( KIND = wp ), DIMENSION( :,: ), POINTER :: filter

      IF ( s%level >= DEBUG ) THEN
         WRITE( s%out, 1000 ) s%active_filter, s%filter_size, s%filter_capacity
      END IF

!------------------------------------------------------------------------------
!                          Filter capacity extension
!------------------------------------------------------------------------------

!     If the filter is not yet allocated, allocate a first slice

      IF ( s%active_filter == 0 .AND. control%maximal_filter_size /= 0 ) THEN

         size = control%filter_size_increment
         IF ( control%maximal_filter_size > 0 ) THEN
             size = MIN( size, control%maximal_filter_size )
         END IF

!        1) the filter itself

         IF ( ASSOCIATED( s%filter_1 ) ) DEALLOCATE( s%filter_1 )
         ALLOCATE( s%filter_1( size, s%p ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1001 ) 1, size, s%p
            RETURN
         END IF
         s%filter_capacity = control%filter_size_increment
         s%active_filter   = 1

!        2) the activity flags

         IF ( control%remove_dominated ) THEN
            IF ( ASSOCIATED( s%active_1 ) ) DEALLOCATE( s%active_1 )
            ALLOCATE( s%active_1( size ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1002 ) 1, size
               RETURN
            END IF
         END IF

!        3) the vector of filter points norms

         IF ( ASSOCIATED( s%fnorm_1 ) ) DEALLOCATE( s%fnorm_1 )
         ALLOCATE( s%fnorm_1( size ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1003 ) 1, size
            RETURN
         END IF

!        4) the ordered list of norms

         s%filter_first = -1
         IF ( ASSOCIATED( s%filter_next_1 ) ) DEALLOCATE( s%filter_next_1 )
         ALLOCATE( s%filter_next_1( size ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), 1004 ) 1, size
            RETURN
         END IF

         IF ( s%level >= DEBUG ) WRITE( s%out, 1005 ) 1, size, s%p

      END IF

!     The filter is completely full and can no longer be extended.

      IF ( control%maximal_filter_size > 0 ) THEN
         IF ( s%filter_size >= control%maximal_filter_size ) RETURN
      END IF

!     If the filter is fully active, extend it.

      IF ( s%filter_size         == s%filter_capacity .AND. &
           s%filter_nbr_inactive == 0                       ) THEN

         new_capacity = s%filter_capacity + control%filter_size_increment
         IF ( control%maximal_filter_size > 0 ) THEN
            new_capacity = MIN( new_capacity, control%maximal_filter_size )
         END IF

         SELECT CASE ( s%active_filter )

         CASE ( 1 )

!           1a) Allocate an alternative larger filter

            ALLOCATE( s%filter_2( new_capacity, s%p ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1001 ) 2, new_capacity, s%p
               RETURN
            END IF
            s%filter_2( 1:s%filter_capacity, 1:s%p ) = s%filter_1
            DEALLOCATE( s%filter_1 )

            IF ( control%remove_dominated ) THEN
               ALLOCATE( s%active_2( new_capacity ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), 1002 ) 2, new_capacity
                  RETURN
               END IF
               s%active_2( 1:s%filter_capacity ) = s%active_1
               DEALLOCATE( s%active_1 )
            END IF

            ALLOCATE( s%fnorm_2( new_capacity ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1003 ) 2, new_capacity
               RETURN
            END IF
            s%fnorm_2( 1:s%filter_capacity ) = s%fnorm_1
            DEALLOCATE( s%fnorm_1 )

            ALLOCATE( s%filter_next_2( new_capacity ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1004 ) 2, new_capacity
               RETURN
            END IF
            s%filter_next_2( 1:s%filter_capacity ) = s%filter_next_1
            DEALLOCATE( s%filter_next_1 )

!           1b) Make filter 2 active.

            s%filter_capacity = new_capacity
            s%active_filter   = 2

         CASE ( 2 )

!           1b) Allocate an alternative larger filter

            ALLOCATE( s%filter_1( new_capacity, s%p ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1001 ) 1, new_capacity, s%p
               RETURN
            END IF
            s%filter_1( 1:s%filter_capacity, 1:s%p ) = s%filter_2
            DEALLOCATE( s%filter_2 )

            IF ( control%remove_dominated ) THEN
               ALLOCATE( s%active_1( new_capacity ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), 1002 ) 1, new_capacity
                  RETURN
               END IF
               s%active_1( 1:s%filter_capacity ) = s%active_2
               DEALLOCATE( s%active_2 )
            END IF

            ALLOCATE( s%fnorm_1( new_capacity ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1003 ) 1, new_capacity
               RETURN
            END IF
            s%fnorm_1( 1:s%filter_capacity ) = s%fnorm_2
            DEALLOCATE( s%fnorm_2 )

            ALLOCATE( s%filter_next_1( new_capacity ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), 1004 ) 1, new_capacity
               RETURN
            END IF
            s%filter_next_1( 1:s%filter_capacity ) = s%filter_next_2
            DEALLOCATE( s%filter_next_2 )

!           2b) Make filter 1 active.

            s%filter_capacity = new_capacity
            s%active_filter   = 1

         END SELECT

!        Log this operation.

         IF ( s%level >= ACTION ) WRITE( s%out, 1006 )                         &
            s%filter_size, new_capacity, s%active_filter

      END IF

!------------------------------------------------------------------------------
!                      Insertion of the current theta value
!------------------------------------------------------------------------------

!     Compute the norm of the current theta.

      normtheta = NRM2( s%p, s%theta, 1 )

!     Determine the active filter.

      SELECT CASE ( s%active_filter )

      CASE ( 1 )

         filter => s%filter_1
         fnorm  => s%fnorm_1
         fnext  => s%filter_next_1
         IF ( control%remove_dominated ) active => s%active_1

      CASE ( 2 )

         filter => s%filter_2
         fnorm  => s%fnorm_2
         fnext  => s%filter_next_2
         IF ( control%remove_dominated ) active => s%active_2

      END SELECT

!     Determine the position of the new value.

      IF ( s%filter_nbr_inactive > 0 ) THEN
         DO pos = 1, s%filter_size
            IF ( .NOT. active( pos ) ) EXIT
         END DO
         s%filter_nbr_inactive =  s%filter_nbr_inactive - 1
      ELSE
         pos = s%filter_size + 1
         s%filter_size = pos
      END IF

!     Now insert the new value.

      IF ( control%margin_type == FIXED ) THEN
         marginc = control%gamma_f * normtheta
         SELECT CASE ( s%filter_sign )
         CASE ( RESTRICTED )
            filter( pos, 1:s%p ) = MAX( ABS( s%theta ) - marginc, ZERO )
         CASE ( UNRESTRICTED )
            filter( pos,1:s%p ) = ZERO
            WHERE ( s%theta < ZERO )                                           &
                  filter( pos,1:s%p ) = MIN( s%theta + marginc, ZERO )
            WHERE ( s%theta > ZERO )                                           &
                  filter( pos,1:s%p ) = MAX( s%theta - marginc, ZERO )
         CASE ( MIXED )
            DO j = 1, s%p
               SELECT CASE ( s%g_status( j ) )
               CASE ( SINGLE_RESTRICTED, MULTIPLE )
                   filter( pos, j ) = MAX( ABS( s%theta( j ) ) - marginc, ZERO )
               CASE ( SINGLE_UNRESTRICTED )
                   IF ( s%theta( j ) < ZERO ) THEN
                      filter( pos,j ) = MIN( s%theta( j ) + marginc, ZERO )
                   ELSE IF ( s%theta( j ) > ZERO ) THEN
                      filter( pos,j ) = MAX( s%theta( j ) - marginc, ZERO )
                   ELSE
                      filter( pos,j ) = ZERO
                   END IF
               END SELECT
            END DO
         END SELECT
      ELSE
         filter( pos, 1:s%p ) = s%theta
         marginc = ZERO
      END IF
      IF ( control%remove_dominated ) active( pos ) = .TRUE.

      IF ( s%level >= ACTION ) WRITE( s%out, 1007 ) pos

!     Compute the Euclidean norm of the new entry.

      normpos      = normtheta - marginc
      fnorm( pos ) = normpos

!     Update the normwise ordered list of filter entries.

      IF ( s%filter_first <= 0 ) THEN
         fnext( pos )   = s%filter_first
         s%filter_first = pos
      ELSE
         IF ( normpos < fnorm( s%filter_first ) ) THEN
            fnext( pos )   = s%filter_first
            s%filter_first = pos
         ELSE
            found = .FALSE.
            prv   = s%filter_first
            nxt   = fnext( prv )
            DO
               IF ( nxt < 0 ) EXIT
               IF ( normpos < fnorm( nxt ) ) THEN
                  fnext( pos ) = nxt
                  fnext( prv ) = pos
                  found = .TRUE.
                  EXIT
               ELSE
                  prv = nxt
                  nxt = fnext( nxt )
               END IF
            END DO
            IF ( .NOT. found ) THEN
               fnext( pos ) = -1
               fnext( prv ) = pos
            END IF
         END IF
      END IF

!     Make newly dominated values inactive, if requested.

      IF ( control%remove_dominated ) THEN

         SELECT CASE ( control%margin_type )

         CASE ( CURRENT, FIXED )

            DO k = 1, s%filter_size
               IF ( active( k ) .AND. k /= pos ) THEN
                  IF ( ALL( filter( k, 1:s%p) >= filter( pos, 1:s%p ) ) ) THEN

!                    Make entry k inactive

                     active( k ) = .FALSE.

!                    Decrease the filter size.

                     s%filter_nbr_inactive = s%filter_nbr_inactive + 1

!                    Remove entry k from the linked list of normwise increasing
!                    entries.

                     nxt = fnext( k )
                     prv = s%filter_first
                     pos = prv
                     DO
                        IF ( pos <= 0 ) THEN
                           inform%status = INTERNAL_ERROR_100
                           WRITE( inform%message( 1 ), 1008 ) k
                           WRITE( inform%message( 2 ), 1009 )
                           RETURN
                        END IF
                        IF ( pos == k ) THEN
                           IF ( pos == s%filter_first ) THEN
                              s%filter_first = nxt
                           ELSE
                              fnext( prv ) = nxt
                           END IF
                           EXIT
                        ELSE
                           prv = pos
                           pos = fnext( pos )
                        END IF
                     END DO
                     fnext( k ) = 0
                     IF ( s%level >= ACTION ) WRITE( s%out, 1010 ) k
                  END IF
               END IF
            END DO

         CASE ( SMALLEST )

            DO k = 1, s%filter_size
               IF ( active( k ) .AND. k /= pos ) THEN
                  marginf = fnorm( k )
                  IF ( ALL( filter(k,1:s%p) - marginf >= filter(pos,1:s%p)))THEN

!                    Make entry k inactive

                     active( k ) = .FALSE.

!                    Decrease the filter size.

                     s%filter_nbr_inactive = s%filter_nbr_inactive + 1

!                    Remove entry k from the linked list of normwise increasing
!                    entries.

                     nxt = fnext( k )
                     prv = s%filter_first
                     pos = prv
                     DO
                        IF ( pos <= 0 ) THEN
                           inform%status = INTERNAL_ERROR_100
                           WRITE( inform%message( 1 ), 1008 ) k
                           WRITE( inform%message( 2 ), 1009 )
                           RETURN
                        END IF
                        IF ( pos == k ) THEN
                           IF ( pos == s%filter_first ) THEN
                              s%filter_first = nxt
                           ELSE
                              fnext( prv ) = nxt
                           END IF
                           EXIT
                        ELSE
                           prv = pos
                           pos = fnext( pos )
                        END IF
                     END DO
                     fnext( k ) = 0
                     IF ( s%level >= ACTION ) WRITE( s%out, 1010 ) k
                  END IF
               END IF
            END DO

         END SELECT

      END IF

!     Write the new filter, if you really insist.

      IF ( s%level >= CRAZY ) THEN
         DO k = 1, s%filter_size
            IF ( control%remove_dominated ) THEN
               IF( active( k ) ) THEN
                  WRITE( s%out, 1011 ) k
               ELSE
                  WRITE( s%out, 1012 ) k
               END IF
            ELSE
               WRITE( s%out, 1013 ) k
            END IF
            CALL TOOLS_output_vector( s%p, filter( k, 1:s%p ), s%out )
         END DO
      END IF

      RETURN

!     Formats

1000  FORMAT(4x,'active filter is ',i1,' of size ',i6,' and capacity ',i6)
1001  FORMAT(1x,'FILTRANE ERROR: no memory left for allocating filter_',i1,'(',&
                i6, ',', i6, ')')
1002  FORMAT(1x,'FILTRANE ERROR: no memory left for allocating active_',i1,'(',&
                i6,')')
1003  FORMAT(1x,'FILTRANE ERROR: no memory left for allocating fnorm_',i1,'(', &
                i6,')')
1004  FORMAT(1x,'FILTRANE ERROR: no memory left for filter_next_',i1,'(',i6,')')
1005  FORMAT(4x,'initial filter_',i1,'(',i6,',',i6,')  allocated')
1006  FORMAT(2x,'filter size bumped from ',i6,' to ',i6,'(filter ',i1,         &
                ' now in use)')
1007  FORMAT(2x,'adding current theta to the filter in position ',i6)
1008  FORMAT(1x,'FILTRANE INTERNAL ERROR 100: k =', i10)
1009  FORMAT(1x,'         (Please report to Ph. Toint (with problem data',     &
                ' and specfile). Thanks.)')
1010  FORMAT(2x,'removing dominated filter entry in position ',i6)
1011  FORMAT(/,5x,'FILTER(',i6,') active :')
1012  FORMAT(/,5x,'FILTER(',i6,') inactive :')
1013  FORMAT(/,5x,'FILTER(',i6,') :')

      END SUBROUTINE FILTRANE_add_to_filter

!==============================================================================
!==============================================================================

      LOGICAL FUNCTION FILTRANE_is_acceptable( f_plus )

!     Checks if the current value of theta is acceptable for the filter.
!     The routine treats the various margin types and sign restrictions
!     separately in order to avoid logical tests in the inner loop as much
!     as possible.

!     NOTE: uses the workspace s%r

      REAL ( KIND = wp ), INTENT( IN ) :: f_plus

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, j, nxt
      REAL ( KIND = wp ) :: fval, cval, margin, marginc, cnorm, sqrtp

      LOGICAL, DIMENSION( : ), POINTER :: active
      INTEGER, DIMENSION( : ), POINTER :: fnext
      REAL ( KIND = wp ), DIMENSION( : ),   POINTER :: fnorm
      REAL ( KIND = wp ), DIMENSION( :,: ), POINTER :: filter

!     Inacceptable if the filter is not in use.

      IF ( .NOT. s%use_filter ) THEN
         FILTRANE_is_acceptable = .FALSE.
         RETURN
      END IF

!     Inacceptable if above the global upper bound for the objective.

      IF ( f_plus > s%f_max ) THEN
         FILTRANE_is_acceptable = .FALSE.
         RETURN
      END IF

!     The general case
!     Compute the margin, when it only depends on the current theta.

      cnorm = NRM2( s%p, s%theta( 1:s%p ), 1 )
      s%r( 1:s%p ) = ABS( s%theta( 1:s%p ) )
      IF ( control%margin_type /= FIXED ) THEN
         marginc = control%gamma_f * cnorm
         sqrtp = s%p
         sqrtp = SQRT( sqrtp )
      END IF

!     Select the active filter.

      SELECT CASE ( s%active_filter )

      CASE ( 0 )

         FILTRANE_is_acceptable = .TRUE.
         RETURN

      CASE ( 1 )

         filter => s%filter_1
         fnorm  => s%fnorm_1
         fnext  => s%filter_next_1
         IF ( control%remove_dominated ) active => s%active_1

      CASE ( 2 )

         filter => s%filter_2
         fnorm  => s%fnorm_2
         fnext  => s%filter_next_2
         IF ( control%remove_dominated ) active => s%active_2

      END SELECT

!     Verify acceptability for the current filter: the entry is a priori acceptable.

      FILTRANE_is_acceptable = .TRUE.

      SELECT CASE ( control%margin_type )

!     Consider first the case where the margin is FIXED.  In this case, the
!     filter entry already contains the margin, so the tests no longer involve
!     it.

      CASE( FIXED )

         SELECT CASE ( s%filter_sign )

!        The first case is when every group is sign restricted.

         CASE ( RESTRICTED )

            nxt = s%filter_first
            DO k = 1, s%filter_size
               IF ( control%remove_dominated ) THEN
                  IF ( active( nxt ) ) THEN
                     IF ( cnorm < fnorm( nxt ) ) THEN
                        RETURN
                     ELSE IF ( ANY ( s%r(1:s%p) < filter(nxt,1:s%p) ) ) THEN
                        nxt = fnext( nxt )
                        IF ( nxt > 0 ) CYCLE
                        RETURN
                     ELSE
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                  END IF
               ELSE
                  IF ( cnorm < fnorm( nxt ) ) THEN
                     RETURN
                  ELSE IF ( ANY( s%r( 1:s%p ) < filter( nxt,1:s%p ) ) ) THEN
                     nxt = fnext( nxt )
                     IF ( nxt > 0 ) CYCLE
                     RETURN
                  ELSE
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               END IF
            END DO


!        The second case is when every group is sign unrestricted.

         CASE ( UNRESTRICTED )

            nxt = s%filter_first
fpt7:       DO k = 1, s%filter_size
               IF ( control%remove_dominated ) THEN
                  IF ( active( nxt ) ) THEN
                     IF ( cnorm < fnorm( nxt ) ) THEN
                        RETURN
                     END IF
                     DO j = 1, s%p
                        cval = s%theta( j )
                        fval = filter( nxt, j )
                        IF ( cval < ZERO ) THEN
                           IF ( fval >= ZERO .OR. cval > fval ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt7
                              RETURN
                           END IF
                        ELSE IF ( cval == ZERO ) THEN
                           IF ( fval /= ZERO ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt7
                              RETURN
                           END IF
                        ELSE
                           IF ( fval <= ZERO .OR. cval < fval ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt7
                              RETURN
                           END IF
                        END IF
                     END DO
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               ELSE
                  IF ( cnorm < fnorm( nxt ) ) RETURN
                  DO j = 1, s%p
                     cval = s%theta( j )
                     fval = filter( nxt, j )
                     IF ( cval < ZERO ) THEN
                        IF ( fval >= ZERO .OR. cval > fval ) THEN
                           nxt = fnext( nxt )
                           IF ( nxt > 0 ) CYCLE fpt7
                           RETURN
                        END IF
                     ELSE IF ( cval == ZERO ) THEN
                        IF ( fval /= ZERO ) THEN
                           nxt = fnext( nxt )
                           IF ( nxt > 0 ) CYCLE fpt7
                           RETURN
                        END IF
                     ELSE
                        IF ( fval <= ZERO .OR. cval < fval ) THEN
                           nxt = fnext( nxt )
                           IF ( nxt > 0 ) CYCLE fpt7
                           RETURN
                        END IF
                     END IF
                  END DO
                  FILTRANE_is_acceptable = .FALSE.
                  RETURN
               END IF
            END DO fpt7

!        The third case is when there is a mixture of sign-restricted
!        an unrestricted groups.

         CASE ( MIXED )

            nxt = s%filter_first
fpt8:       DO k = 1, s%filter_size
               IF ( control%remove_dominated ) THEN
                  IF ( active( nxt ) ) THEN
                     IF ( cnorm < fnorm( nxt ) ) RETURN
                     DO j = 1, s%p
                        cval = s%theta( j )
                        fval = filter( nxt, j )
                        SELECT CASE ( s%g_status( j ) )
                        CASE ( SINGLE_RESTRICTED, MULTIPLE )
                           IF ( ABS( cval ) < fval ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt8
                              RETURN
                           END IF
                        CASE ( SINGLE_UNRESTRICTED )
                           IF ( cval < ZERO ) THEN
                              IF ( fval >= ZERO .OR. cval > fval ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt8
                                 RETURN
                              END IF
                           ELSE IF ( cval == ZERO ) THEN
                              IF ( fval /= ZERO ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt8
                                 RETURN
                              END IF
                           ELSE
                              IF ( fval <= ZERO .OR. cval < fval ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt8
                                 RETURN
                              END IF
                           END IF
                        END SELECT
                     END DO
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               ELSE
                  IF ( cnorm < fnorm( k ) ) RETURN
                  DO j = 1, s%p
                     cval = s%theta( j )
                     fval = filter( nxt, j )
                     SELECT CASE ( s%g_status( j ) )
                     CASE ( SINGLE_RESTRICTED, MULTIPLE )
                        IF ( ABS( cval ) < fval ) THEN
                           nxt = fnext( nxt )
                           IF ( nxt > 0 ) CYCLE fpt8
                           RETURN
                        END IF
                     CASE ( SINGLE_UNRESTRICTED )
                        IF ( cval < ZERO ) THEN
                           IF ( fval >= ZERO .OR. cval > fval ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt8
                              RETURN
                           END IF
                        ELSE IF ( cval == ZERO ) THEN
                           IF ( fval /= ZERO ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt8
                              RETURN
                           END IF
                        ELSE
                           IF ( fval <= ZERO .OR. cval < fval ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) THEN
                                 CYCLE fpt8
                              ELSE
                                 RETURN
                              END IF
                           END IF
                        END IF
                     END SELECT
                  END DO
                  FILTRANE_is_acceptable = .FALSE.
                  RETURN
               END IF
            END DO fpt8

         END SELECT

!     Consider now the CURRENT and SMALLEST margin type.  In these cases,
!     the margin is not included in the filter, and must therefore be
!     computed and used in the acceptability test.

      CASE( CURRENT, SMALLEST )

         SELECT CASE ( s%filter_sign )

!        The first case is when every group is sign restricted.

         CASE ( RESTRICTED )

            IF ( control%margin_type == SMALLEST ) THEN

               nxt = s%filter_first
               DO k = 1, s%filter_size
                  IF ( control%remove_dominated ) THEN
                     IF ( active( nxt ) ) THEN
                        margin = MIN( control%gamma_f * fnorm( nxt ), marginc )
                        IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) THEN
                           RETURN
                        ELSE IF (ANY(s%r(1:s%p)<filter(nxt,1:s%p)-margin ) )&
                           THEN
                           nxt = fnext( nxt )
                           IF ( nxt > 0 ) THEN
                              CYCLE
                           ELSE
                              RETURN
                           END IF
                        ELSE
                           FILTRANE_is_acceptable = .FALSE.
                           RETURN
                        END IF
                     END IF
                  ELSE
                     margin = MIN( control%gamma_f * fnorm( nxt ), marginc )
                     IF ( cnorm < fnorm( k ) - margin * sqrtp ) THEN
                        RETURN
                     ELSE IF ( ANY(s%r(1:s%p)<filter(nxt,1:s%p)-margin)) THEN
                        nxt = fnext( nxt )
                        IF ( nxt > 0 ) THEN
                           CYCLE
                        ELSE
                           RETURN
                        END IF
                     ELSE
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                  END IF
               END DO

            ELSE

               margin = marginc
               nxt = s%filter_first
               DO k = 1, s%filter_size
                  IF ( control%remove_dominated ) THEN
                     IF ( active ( nxt ) ) THEN
                        IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) THEN
                           RETURN
                        ELSE IF (ANY(s%r(1:s%p)<filter(nxt,1:s%p)-margin)) &
                           THEN
                           nxt = fnext( nxt )
                           IF ( nxt > 0 ) CYCLE
                           RETURN
                        ELSE
                           FILTRANE_is_acceptable = .FALSE.
                           RETURN
                        END IF
                     END IF
                  ELSE
                     IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) THEN
                        RETURN
                     ELSE IF (ANY(s%r(1:s%p)<filter(nxt,1:s%p)-margin)) THEN
                        nxt = fnext( nxt )
                        IF ( nxt > 0 ) CYCLE
                        RETURN
                     ELSE
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                  END IF
               END DO

            END IF

!        The second case is when every group is sign unrestricted.

         CASE ( UNRESTRICTED )

            IF ( control%margin_type == SMALLEST ) THEN

               nxt = s%filter_first
fpt9:          DO k = 1, s%filter_size
                  IF ( control%remove_dominated ) THEN
                     IF ( active( nxt ) ) THEN
                        margin = MIN( control%gamma_f * fnorm( nxt ), marginc )
                        IF ( cnorm < fnorm( nxt ) - margin *sqrtp ) RETURN
                        DO j = 1, s%p
                           cval = s%theta( j )
                           fval = filter( nxt, j )
                           IF ( cval < ZERO ) THEN
                              IF ( fval>=ZERO .OR. cval>fval+margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt9
                                 RETURN
                              END IF
                           ELSE IF ( cval == ZERO ) THEN
                              IF ( fval /= ZERO ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt9
                                 RETURN
                              END IF
                           ELSE
                              IF ( fval<=ZERO .OR. cval<fval-margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt9
                                 RETURN
                              END IF
                           END IF
                        END DO
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                  ELSE
                     margin = MIN( control%gamma_f * fnorm( nxt ), marginc )
                     IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) RETURN
                     DO j = 1, s%p
                        cval = s%theta( j )
                        fval = filter( nxt, j )
                        IF ( cval < ZERO ) THEN
                           IF ( fval >= ZERO .OR. cval > fval + margin ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt9
                              RETURN
                           END IF
                        ELSE IF ( cval == ZERO ) THEN
                           IF ( fval /= ZERO ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt9
                              RETURN
                           END IF
                        ELSE
                           IF ( fval <= ZERO .OR. cval < fval - margin ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt9
                              RETURN
                           END IF
                        END IF
                     END DO
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               END DO fpt9

            ELSE

               margin = marginc
               nxt    = s%filter_first
fpt0:          DO k   = 1, s%filter_size
                  IF ( control%remove_dominated ) THEN
                     IF ( active( nxt ) ) THEN
                        IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) RETURN
                        DO j = 1, s%p
                           cval = s%theta( j )
                           fval = filter( nxt, j )
                           IF ( cval < ZERO ) THEN
                              IF ( fval >= ZERO .OR. cval > fval+margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt0
                                 RETURN
                              END IF
                           ELSE IF ( cval == ZERO ) THEN
                              IF ( fval /= ZERO ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt0
                                 RETURN
                              END IF
                           ELSE
                              IF (fval <= ZERO .OR. cval < fval - margin) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt0
                                 RETURN
                              END IF
                           END IF
                        END DO
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                  ELSE
                     IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) RETURN
                     DO j = 1, s%p
                        cval = s%theta( j )
                        fval = filter( nxt, j )
                        IF ( cval < ZERO ) THEN
                           IF ( fval >= ZERO .OR. cval > fval + margin ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt0
                              RETURN
                           END IF
                        ELSE IF ( cval == ZERO ) THEN
                           IF ( fval /= ZERO ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt0
                              RETURN
                           END IF
                        ELSE
                           IF ( fval <= ZERO .OR. cval < fval - margin ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt0
                              RETURN
                           END IF
                        END IF
                     END DO
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               END DO fpt0

            END IF

!        The third case is when there is a mixture of sign-restricted
!        an unrestricted groups.

         CASE ( MIXED )

            IF ( control%margin_type == SMALLEST ) THEN

               nxt = s%filter_first
fpt10:         DO k = 1, s%filter_size
                  IF ( control%remove_dominated ) THEN
                     IF ( active( nxt ) ) THEN
                        margin = MIN( control%gamma_f * fnorm(nxt), marginc )
                        IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) RETURN
                        DO j = 1, s%p
                           cval = s%theta( j )
                           fval = filter( k, j )
                           SELECT CASE ( s%g_status( j ) )
                           CASE ( SINGLE_RESTRICTED, MULTIPLE )
                              IF ( ABS( cval ) < fval - margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt10
                                 RETURN
                              END IF
                           CASE ( SINGLE_UNRESTRICTED )
                              IF ( cval < ZERO ) THEN
                                 IF ( fval >= ZERO .OR. cval > fval+margin )THEN
                                    nxt = fnext( nxt )
                                    IF ( nxt > 0 ) CYCLE fpt10
                                    RETURN
                                 END IF
                              ELSE IF ( cval == ZERO ) THEN
                                 IF ( fval /= ZERO ) THEN
                                    nxt = fnext( nxt )
                                    IF ( nxt > 0 ) CYCLE fpt10
                                    RETURN
                                 END IF
                              ELSE
                                 IF ( fval <= ZERO .OR. cval < fval-margin )THEN
                                    nxt = fnext( nxt )
                                    IF ( nxt > 0 ) CYCLE fpt10
                                    RETURN
                                 END IF
                              END IF
                           END SELECT
                        END DO
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                 ELSE
                     margin = MIN( control%gamma_f * fnorm( k ), marginc )
                     IF ( cnorm < fnorm( nxt ) - margin *sqrtp ) RETURN
                     DO j = 1, s%p
                        cval = s%theta( j )
                        fval = filter( nxt, j )
                        SELECT CASE ( s%g_status( j ) )
                        CASE ( SINGLE_RESTRICTED, MULTIPLE )
                           IF ( ABS( cval ) < fval - margin ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt10
                              RETURN
                           END IF
                        CASE ( SINGLE_UNRESTRICTED )
                           IF ( cval < ZERO ) THEN
                              IF ( fval >= ZERO .OR. cval > fval + margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt10
                                 RETURN
                              END IF
                           ELSE IF ( cval == ZERO ) THEN
                              IF ( fval /= ZERO ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt10
                                 RETURN
                              END IF
                           ELSE
                              IF (fval <= ZERO .OR. cval < fval - margin) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt10
                                 RETURN
                              END IF
                           END IF
                        END SELECT
                     END DO
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               END DO fpt10

            ELSE

               margin= marginc
               nxt = s%filter_first
fpt11:         DO k = 1, s%filter_size
                  IF ( control%remove_dominated ) THEN
                     IF ( active( nxt ) ) THEN
                        IF ( cnorm < fnorm( k ) - margin * sqrtp ) RETURN
                        DO j = 1, s%p
                           cval = s%theta( j )
                           fval = filter( nxt, j )
                           SELECT CASE ( s%g_status( j ) )
                           CASE ( SINGLE_RESTRICTED, MULTIPLE )
                              IF ( ABS( cval ) <= fval - margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt11
                                 RETURN
                              END IF
                           CASE ( SINGLE_UNRESTRICTED )
                              IF ( cval < ZERO ) THEN
                                 IF( fval >= ZERO .OR. cval > fval+margin ) THEN
                                    nxt = fnext( nxt )
                                    IF ( nxt > 0 ) CYCLE fpt11
                                    RETURN
                                 END IF
                              ELSE IF ( cval == ZERO ) THEN
                                 IF ( fval /= ZERO ) THEN
                                    nxt = fnext( nxt )
                                    IF ( nxt > 0 ) CYCLE fpt11
                                    RETURN
                                 END IF
                              ELSE
                                 IF( fval <= ZERO .OR. cval < fval-margin ) THEN
                                    nxt = fnext( nxt )
                                    IF ( nxt > 0 ) CYCLE fpt11
                                    RETURN
                                 END IF
                              END IF
                           END SELECT
                        END DO
                        FILTRANE_is_acceptable = .FALSE.
                        RETURN
                     END IF
                  ELSE
                     IF ( cnorm < fnorm( nxt ) - margin * sqrtp ) RETURN
                     DO j = 1, s%p
                        cval = s%theta( j )
                        fval = filter( nxt, j )
                        SELECT CASE ( s%g_status( j ) )
                        CASE ( SINGLE_RESTRICTED, MULTIPLE )
                           IF ( ABS( cval ) <= fval - margin ) THEN
                              nxt = fnext( nxt )
                              IF ( nxt > 0 ) CYCLE fpt11
                              RETURN
                           END IF
                        CASE ( SINGLE_UNRESTRICTED )
                           IF ( cval < ZERO ) THEN
                              IF ( fval >= ZERO .OR. cval > fval+margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt11
                                 RETURN
                              END IF
                           ELSE IF ( cval == ZERO ) THEN
                              IF ( fval /= ZERO ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt11
                                 RETURN
                              END IF
                           ELSE
                              IF ( fval <= ZERO .OR. cval < fval-margin ) THEN
                                 nxt = fnext( nxt )
                                 IF ( nxt > 0 ) CYCLE fpt11
                                 RETURN
                              END IF
                           END IF
                        END SELECT
                     END DO
                     FILTRANE_is_acceptable = .FALSE.
                     RETURN
                  END IF
               END DO fpt11

            END IF

         END SELECT

      END SELECT

      FILTRANE_is_acceptable = .TRUE.

      RETURN

      END FUNCTION FILTRANE_is_acceptable

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_compute_theta

!     Computes the value of the thetas.

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

!     Local variables

      INTEGER :: i, j, nxt, ig
      LOGICAL :: has_multiple
      REAL ( KIND = wp ) :: v

      s%theta      = ZERO
      has_multiple = .FALSE.

!     The constraints

      DO i = 1, problem%m

         SELECT CASE ( control%grouping )
         CASE ( NONE )
            ig = i
         CASE ( AUTOMATIC, USER_DEFINED )
            ig = s%group( i )
         END SELECT

         v = FILTRANE_c_violation( i )
         SELECT CASE ( s%filter_sign )
         CASE ( RESTRICTED )
            s%theta( ig ) = s%theta( ig ) + ABS( v )
         CASE ( UNRESTRICTED )
            s%theta( ig ) = s%theta( ig ) + v
         CASE ( MIXED )
            SELECT CASE ( s%g_status( ig ) )
            CASE ( SINGLE_RESTRICTED )
               s%theta( ig ) = s%theta( ig ) + ABS( v )
            CASE ( SINGLE_UNRESTRICTED )
               s%theta( ig ) = s%theta( ig ) + v
            CASE ( MULTIPLE )
               s%theta( ig ) = s%theta( ig ) + v ** 2
               has_multiple = .TRUE.
            END SELECT
         END SELECT

      END DO

!     Add the contributions of the violated bounds, if any.

      IF ( s%has_bounds ) THEN
         nxt = problem%m
         DO j = 1, problem%n
            SELECT CASE ( problem%x_status( j ) )
            CASE ( LOWER, UPPER, RANGE )
               nxt = nxt + 1

               SELECT CASE ( control%grouping )
               CASE ( NONE )
                  ig = nxt
               CASE ( AUTOMATIC, USER_DEFINED )
                  ig = s%group( nxt )
               END SELECT

               v = FILTRANE_x_violation( j )
               SELECT CASE ( s%filter_sign )
               CASE ( RESTRICTED )
                  s%theta( ig ) = s%theta( ig ) + ABS( v )
               CASE ( UNRESTRICTED )
                  s%theta( ig ) = s%theta( ig ) + v
               CASE ( MIXED )
                  SELECT CASE ( s%g_status( ig ) )
                  CASE ( SINGLE_RESTRICTED )
                     s%theta( ig ) = s%theta( ig ) + ABS( v )
                  CASE ( MULTIPLE )
                     s%theta( ig ) = s%theta( ig ) + v ** 2
                     has_multiple = .TRUE.
                  END SELECT
               END SELECT
            END SELECT
         END DO
      END IF

!     Rescale the multiple groups.

      IF ( has_multiple ) THEN
         WHERE ( s%g_status == MULTIPLE ) s%theta = SQRT( s%theta )
      END IF

      RETURN

      END SUBROUTINE FILTRANE_compute_theta

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_compute_f

!     Compute the value of the objective function.

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

!     Local variables

      INTEGER :: i, j, nxt
      REAL ( KIND = wp ) :: t, phi_l, phi_u, cli, cui, xlj, xuj, ci, xj

      problem%f = ZERO

!     The constraints

      DO i = 1, problem%m

         IF ( problem%equation( i ) ) THEN
            problem%f = problem%f + problem%c( i ) ** 2
         ELSE
            SELECT CASE ( control%inequality_penalty_type )
            CASE ( 1 )
               ci  = problem%c( i )
               cli = problem%c_l( i )
               cui = problem%c_u( i )
               IF ( ci <= cli - s%epsilon ) THEN
                  phi_l = cli - ci
               ELSE IF ( ci < cli + s%epsilon ) THEN
                  t     = ( cli + s%epsilon - ci ) / ( TWO *s%epsilon )
                  phi_l = - s%epsilon * t ** 4 + TWO * s%epsilon * t ** 3
               ELSE
                  phi_l = ZERO
               END IF
               IF ( ci >= cui + s%epsilon ) THEN
                  phi_u = ci - cui
               ELSE IF ( ci > cui - s%epsilon ) THEN
                  t     = ( ci - cui + s%epsilon ) / ( TWO *s%epsilon )
                  phi_u = - s%epsilon * t ** 4 + TWO * s%epsilon * t ** 3
               ELSE
                  phi_u = ZERO
               END IF
               problem%f = problem%f + TWO * ( phi_l + phi_u )
            CASE ( 2 )
               problem%f = problem%f + FILTRANE_c_violation( i ) ** 2
            CASE ( 3 )
               problem%f = problem%f + ABS( FILTRANE_c_violation( i ) ) ** 3
            CASE ( 4 )
               problem%f = problem%f + FILTRANE_c_violation( i ) ** 4
            END SELECT
         END IF
      END DO

!     The bounded variables

      IF ( s%has_bounds ) THEN

         nxt = problem%m
         DO j = 1, problem%n
            SELECT CASE ( problem%x_status( j ) )
            CASE ( LOWER, UPPER, RANGE )
               nxt = nxt + 1

               SELECT CASE ( control%inequality_penalty_type )
               CASE ( 1 )
                  xj  = problem%x( j )
                  xlj = problem%x_l( j )
                  xuj = problem%x_u( j )
                  IF ( xj <= xlj - s%epsilon ) THEN
                     phi_l = xlj - xj
                  ELSE IF ( xj < xlj + s%epsilon ) THEN
                     t   = ( xlj + s%epsilon - xj ) / ( TWO *s%epsilon )
                     phi_l = - s%epsilon * t ** 4 + TWO * s%epsilon * t ** 3
                  ELSE
                     phi_l = ZERO
                  END IF
                  IF ( xj >= xuj + s%epsilon ) THEN
                     phi_u = xj - xuj
                  ELSE IF ( xj > xuj - s%epsilon ) THEN
                     t   = ( xj - xuj + s%epsilon ) / ( TWO *s%epsilon )
                     phi_u = - s%epsilon * t ** 4 + TWO * s%epsilon * t ** 3
                  ELSE
                     phi_u = ZERO
                  END IF
                  problem%f = problem%f + TWO * ( phi_l + phi_u )
               CASE ( 2 )
                  problem%f = problem%f + FILTRANE_x_violation( j ) ** 2
               CASE ( 3 )
                  problem%f = problem%f + ABS( FILTRANE_x_violation( j ) ) ** 3
               CASE ( 4 )
                  problem%f = problem%f + FILTRANE_x_violation( j ) ** 4
               END SELECT
            END SELECT
         END DO

      END IF

      problem%f = HALF * problem%f

      RETURN

      END SUBROUTINE FILTRANE_compute_f

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_compute_Hmult( c )

!     Compute the value of the multipliers of the individual Hessians in
!     the Newton model (in problem%y).

!     Argument :

      REAL ( KIND = wp ), DIMENSION( problem%m ), INTENT( IN ) :: c

!          the current value of the problems constraints from which to
!          compute the multipliers


!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

!     Local variables

      INTEGER :: i
      REAL( KIND = wp ) :: ci, cli, cui, t, v, twoeps

      DO i = 1, problem%m

         IF ( problem%equation( i ) ) THEN
            problem%y( i ) = c( i )
         ELSE
            ci  = c( i )
            cli = problem%c_l( i )
            cui = problem%c_u( i )
            IF ( ci < cli ) THEN
               v = ci - cli
            ELSE IF ( ci > cui ) THEN
               v = ci - cui
            ELSE
               v = ZERO
            END IF

            SELECT CASE ( control%inequality_penalty_type )
            CASE ( 1 )
               twoeps = TWO *s%epsilon
               IF ( ci <= cli - s%epsilon ) THEN
                  problem%y( i ) = - ONE
               ELSE IF ( ci < cli + s%epsilon ) THEN
                  t = ( s%epsilon - v ) / twoeps
                  problem%y( i ) =  TWO  * t ** 3 - THREE * t ** 2
               END IF
               IF ( ci >= cui + s%epsilon ) THEN
                  problem%y( i ) =  ONE
               ELSE IF ( ci > cui - s%epsilon ) THEN
                  t = ( v + s%epsilon ) / twoeps
                  problem%y( i ) = - TWO * t ** 3 + THREE * t ** 2
               END IF
            CASE ( 2 )
               problem%y( i ) = v
            CASE ( 3 )
               problem%y( i ) = 1.5_wp * v ** 2
            CASE ( 4 )
               problem%y( i ) = TWO * v ** 3
            END SELECT
         END IF

      END DO

      RETURN

      END SUBROUTINE FILTRANE_compute_Hmult

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_compute_grad_f

!     Compute the gradient of the objective function.

!     NOTE : uses the s%r workspace!

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

      INTEGER :: j
      REAL( KIND = wp ) :: violation, xj, xlj, xuj, dphi, twoeps, t,           &
                           cli, cui, ci

!     Reverse communication interface

      SELECT CASE ( inform%status )
      CASE ( OK )
      CASE ( GET_JTC )
          GO TO 100
      CASE DEFAULT
         WRITE( inform%message( 1 ), 1000 )
         WRITE( inform%message( 2 ), 1001 ) inform%status
         inform%status = WRONG_STATUS
         RETURN
      END SELECT

!     Compute the vector of violations to be multiplied by J^T.

      IF ( problem%m > 0 ) THEN
         IF ( s%has_inequalities ) THEN
            DO i = 1, problem%m
               violation = FILTRANE_c_violation( i )
               IF ( .NOT. problem%equation( i ) ) THEN
                  SELECT CASE ( control%inequality_penalty_type )
                  CASE ( 1 )
                     ci  = problem%c( i )
                     cli = problem%c_l( i )
                     cui = problem%c_u( i )
                     twoeps = TWO *s%epsilon
                     IF ( ci <= cli - s%epsilon ) THEN
                        s%r( i ) = - ONE
                     ELSE IF ( ci < cli + s%epsilon ) THEN
                        t    = ( s%epsilon - violation ) / twoeps
                        dphi = - TWO  * t ** 3 + THREE * t ** 2
                        s%r( i ) = -  dphi
                     END IF
                     IF ( ci >= cui + s%epsilon ) THEN
                        s%r( i ) = ONE
                     ELSE IF ( ci > cui - s%epsilon ) THEN
                        t    = ( violation + s%epsilon ) / twoeps
                        dphi = - TWO * t ** 3 + THREE * t ** 2
                        s%r( i ) = dphi
                     END IF
                  CASE ( 2 )
                     s%r( i ) = violation
                  CASE ( 3 )
                     IF( s%r( i ) > ZERO ) s%r( i ) =   1.5_wp * violation ** 2
                     IF( s%r( i ) < ZERO ) s%r( i ) = - 1.5_wp * violation ** 2
                  CASE ( 4 )
                     s%r( i ) = TWO * violation ** 3
                  END SELECT
               ELSE
                  s%r( i ) = violation
               END IF
            END DO
            IF ( control%external_J_products ) THEN
               inform%status = GET_JTC
               s%RC_v  => s%r( 1:problem%m )
               s%RC_Mv => problem%g( 1:problem%n )
               RETURN
            ELSE
               CALL FILTRANE_J_times_v( s%r( 1:problem%m ), problem%g,         &
                                                            .TRUE., .FALSE. )
            END IF
         ELSE
            IF ( control%external_J_products ) THEN
               inform%status = GET_JTC
               s%RC_v  => problem%c( 1:problem%m )
               s%RC_Mv => problem%g( 1:problem%n )
               RETURN
            ELSE
               CALL FILTRANE_J_times_v( problem%c, problem%g, .TRUE., .FALSE. )
            END IF
         END IF
      ELSE
         problem%g = ZERO
      END IF

!*****************************************************
100  CONTINUE ! *** Reverse communication re-entry ***
!*****************************************************

      IF ( problem%m > 0 .AND. control%external_J_products ) THEN
         inform%status = OK
         NULLIFY( s%RC_v, s%RC_Mv )
         IF ( s%has_fixed ) THEN
            DO j = 1, problem%n
               IF ( problem%x_status( j ) == FIXED ) problem%g( j ) = ZERO
            END DO
         END IF
      END IF

!     Add the contributions of the violated bounds, if any.

      IF ( s%has_bounds ) THEN

         DO j = 1, problem%n
            SELECT CASE ( problem%x_status( j ) )
            CASE ( LOWER, UPPER, RANGE )

               violation = FILTRANE_x_violation( j )

               SELECT CASE ( control%inequality_penalty_type )

               CASE ( 1 )

                  xj  = problem%x( j )
                  xlj = problem%x_l( j )
                  xuj = problem%x_u( j )
                  twoeps = TWO *s%epsilon
                  IF ( xj <= xlj - s%epsilon ) THEN
                     problem%g( j ) = problem%g( j ) - ONE
                  ELSE IF ( xj < xlj + s%epsilon ) THEN
                     t    = ( s%epsilon - violation ) / twoeps
                     dphi = - TWO  * t ** 3 + THREE * t ** 2
                     problem%g( j ) = problem%g( j ) -  dphi
                  END IF
                  IF ( xj >= xuj + s%epsilon ) THEN
                     problem%g( j ) = problem%g( j ) + ONE
                  ELSE IF ( xj > xuj - s%epsilon ) THEN
                     t    = ( violation + s%epsilon ) / twoeps
                     dphi = - TWO * t ** 3 + THREE * t ** 2
                     problem%g( j ) = problem%g( j ) + dphi
                  END IF

               CASE ( 2 )

                  problem%g( j ) = problem%g( j ) + violation

               CASE ( 3 )

                  IF ( violation >= ZERO ) THEN
                     problem%g( j ) = problem%g( j ) + 1.5_wp * violation **2
	 	     ELSE
                     problem%g( j ) = problem%g( j ) - 1.5_wp * violation **2
                  END IF

               CASE ( 4 )

                  problem%g( j ) = problem%g( j ) + TWO * violation ** 3

               END SELECT
            END SELECT
         END DO

      END IF

      RETURN

!     Formats

1000  FORMAT(1x,'FILTRANE ERROR: SOLVE  was entered with an erroneous status')
1001  FORMAT(1x,'                (inform%status = ',i2,')')

      END SUBROUTINE FILTRANE_compute_grad_f

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_J_times_v( v, Jv, trans, cond )

!     Computes the product of the Jacobian (or its transpose) times the
!     vector v and stores the result in Jv.

!     Arguments:

      REAL( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: v

!                the vector to be premultiplied by the Jacobian or its
!                transpose;

      REAL( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: Jv

!                the result of premultiplying the vector v by the Jacobian
!                (or its transpose);
!
      LOGICAL :: trans

!                .TRUE. if the product must involve the transpose of the
!                Jacobian, .FALSE. otherwise;

      LOGICAL :: cond

!                .TRUE. if the inequality constraints must be checked to avoid
!                those for which the violation is zero.

!     Programming: Ph. L. Toint, Fall 2002.

!==============================================================================

!     Local variables

      INTEGER :: k, i, j

      Jv = ZERO

      IF ( cond ) THEN
         IF ( s%has_fixed ) THEN
            IF ( trans ) THEN
               DO k = 1, problem%J_ne
                  j = problem%J_col( k )
                  IF ( problem%x_status( j ) == FIXED ) CYCLE
                  i = problem%J_row( k )
                  IF ( .NOT. problem%equation( i ) ) THEN
                     IF ( FILTRANE_c_violation( i ) == ZERO ) CYCLE
                  END IF
                  Jv( j ) = Jv(j) + problem%J_val(k) * v( i )
               END DO
            ELSE
               DO k = 1, problem%J_ne
                  j = problem%J_col( k )
                  IF ( problem%x_status( j ) == FIXED ) CYCLE
                  i = problem%J_row( k )
                  IF ( .NOT. problem%equation( i ) ) THEN
                     IF ( FILTRANE_c_violation( i ) == ZERO ) CYCLE
                  END IF
                  Jv( i ) = Jv(i) + problem%J_val( k ) * v( j )
               END DO
            END IF
         ELSE
            IF ( trans ) THEN
               DO k = 1, problem%J_ne
                  j = problem%J_col( k )
                  i = problem%J_row( k )
                  IF ( .NOT. problem%equation( i ) ) THEN
                     IF ( FILTRANE_c_violation( i ) == ZERO ) CYCLE
                  END IF
                  Jv( j ) = Jv(j) + problem%J_val(k) * v( i )
               END DO
            ELSE
               DO k = 1, problem%J_ne
                  i = problem%J_row( k )
                  IF ( .NOT. problem%equation( i ) ) THEN
                     IF ( FILTRANE_c_violation( i ) == ZERO ) CYCLE
                  END IF
                  Jv( i ) = Jv(i) + problem%J_val(k) * v( problem%J_col(k) )
               END DO
            END IF
         END IF
      ELSE
         IF ( s%has_fixed ) THEN
            IF ( trans ) THEN
               DO k = 1, problem%J_ne
                 j = problem%J_col( k )
                  IF ( problem%x_status( j ) == FIXED ) CYCLE
                  Jv( j ) = Jv(j) + problem%J_val(k) * v( problem%J_row(k) )
               END DO
            ELSE
               DO k = 1, problem%J_ne
                  j = problem%J_col( k )
                  IF ( problem%x_status( j ) == FIXED ) CYCLE
                  i = problem%J_row( k )
                  Jv( i ) = Jv(i) + problem%J_val( k ) * v( j )
               END DO
            END IF
         ELSE
            IF ( trans ) THEN
               DO k = 1, problem%J_ne
                  j = problem%J_col( k )
                  Jv( j ) = Jv(j) + problem%J_val(k) * v( problem%J_row(k) )
               END DO
            ELSE
               DO k = 1, problem%J_ne
                  i = problem%J_row( k )
                  Jv( i ) = Jv(i) + problem%J_val(k) * v( problem%J_col(k) )
               END DO
            END IF
         END IF
      END IF

      RETURN

      END SUBROUTINE FILTRANE_J_times_v

!===============================================================================
!===============================================================================

      SUBROUTINE FILTRANE_build_sparse_J

!     Builds the sparse by columns data structure for the Jacobian and
!     ensures each column has row indices in ascending order.

!     Programming: Ph. Toint, November 2002.

!==============================================================================

!     Local variables

      INTEGER :: i, rs, is, ie, ec

!     Get the permutation to sparse storage by columns.

      CALL NLPT_J_perm_from_C_to_Scol( problem, s%perm, s%row, s%iw )

!     Reorder each column by ascending row indices.

      DO i = 1, problem%n
         ie = s%iw( i + 1 )
         is = s%iw( i )
         rs = ie - is
         IF ( rs > 1 ) THEN
            ie = ie - 1
            CALL SORT_quicksort( rs, s%row(is:ie), ec, ix = s%perm(is:ie) )
            IF ( ec /= OK ) THEN
               inform%status = SORT_TOO_LONG
               WRITE( inform%message( 1 ), 1000 )
               WRITE( inform%message( 2 ), 1001 )
               RETURN
            END IF
         END IF
      END DO

      RETURN

!     Formats

1000  FORMAT(1x,'FILTRANE ERROR: sorting capacity too small.')
1001  FORMAT(1x,'                Increase log2s in SORT_quicksort.')

      END SUBROUTINE FILTRANE_build_sparse_J

!===============================================================================
!===============================================================================

      SUBROUTINE FILTRANE_build_band_JTJ( true_bandw )

!     Extract the band preconditioning matrix from the matrix JT * J and
!     compute its true semi-bandwidth.

!     Arguments:

      INTEGER, INTENT( OUT ) :: true_bandw

!     Programming: Ph. Toint, November 2002

!==============================================================================

      INTEGER :: kx, ky, sx, sy, nx, ny, i, j, jj
      LOGICAL :: xi_fixed
      REAL( KIND = wp ) :: val

!     Build the banded matrix.

      true_bandw = 0
      DO i = 1, problem%n

!        See if the i-th variable is fixed

         IF ( s%has_fixed ) THEN
            xi_fixed = problem%x_status( i ) == FIXED
         ELSE
            xi_fixed = .FALSE.
         END IF

!        Fixed variables

         IF ( xi_fixed ) THEN

            s%diag( i ) = ONE
            s%offdiag( 1:s%nsemib, i ) = ZERO

!        Free variables

         ELSE

!           Diagonal element
            val = ZERO
            DO kx = s%iw( i ), s%iw( i + 1 ) - 1
               val = val + problem%J_val( s%perm( kx ) ) ** 2
            END DO
            s%diag( i ) = val

!           Offdiagonal elements

            DO j = 1, MIN( s%nsemib, problem%n - i )

               jj  = i + j

               IF ( s%has_fixed ) THEN
                  xi_fixed = problem%x_status( jj ) == FIXED
               ELSE
                  xi_fixed = .FALSE.
               END IF

               val = ZERO
               IF ( .NOT. xi_fixed ) THEN
                  kx  = s%iw( i )
                  ky  = s%iw( jj )
                  nx  = s%iw( i + 1  )
                  ny  = s%iw( jj + 1 )
                  DO WHILE ( kx < nx .AND. ky < ny )
                     sx  = s%row( kx )
                     sy  = s%row( ky )
                     IF ( sx < sy ) THEN
                        kx = kx + 1
                     ELSE IF ( sx == sy ) THEN
                        val = val + problem%J_val( s%perm( kx ) ) *            &
                                    problem%J_val( s%perm( ky ) )
                        kx = kx + 1
                        ky = ky + 1
                     ELSE
                        ky = ky + 1
                     END IF
                  END DO
               END IF
               s%offdiag( j, i ) = val
               IF ( val /= ZERO ) true_bandw = MAX( j, true_bandw )
            END DO

         END IF

      END DO

      RETURN

      END SUBROUTINE FILTRANE_build_band_JTJ

!===============================================================================
!===============================================================================

      REAL( KIND = wp ) FUNCTION FILTRANE_x_violation( j )

!     Compute the violation of the j-th bound(s)

!     Argument:

      INTEGER, INTENT( IN ) :: j

!              the index of the variable whose violation must be computed.

!     Programming: Ph. Toint, November 2002.

!==============================================================================

      REAL ( KIND = wp ) :: xj, xlj, xuj

      xj  = problem%x( j )
      xlj = problem%x_l( j )
      xuj = problem%x_u( j )
      IF ( xj > xuj ) THEN
         FILTRANE_x_violation = xj - xuj
      ELSE IF ( xj < xlj ) THEN
         FILTRANE_x_violation = xj - xlj
      ELSE
         FILTRANE_x_violation = ZERO
      END IF

      RETURN

      END FUNCTION FILTRANE_x_violation

!===============================================================================
!===============================================================================

      REAL( KIND = wp ) FUNCTION FILTRANE_c_violation( i )

!     Compute the violation of the i-th (in)equality.

!     Argument

      INTEGER, INTENT( IN ) :: i

!              the index of the constraints whose violation must be computed.

!     Programming: Ph. Toint, November 2002

!==============================================================================

      REAL ( KIND = wp ) :: ci, cli, cui

      IF ( problem%equation( i ) ) THEN
         FILTRANE_c_violation = problem%c( i )
      ELSE
         ci  = problem%c( i )
         cli = problem%c_l( i )
         cui = problem%c_u( i )
         IF ( ci > cui ) THEN
            FILTRANE_c_violation = ci - cui
         ELSE IF ( ci < cli ) THEN
            FILTRANE_c_violation = ci - cli
         ELSE
            FILTRANE_c_violation = ZERO
         END IF
      END IF

      RETURN

      END FUNCTION FILTRANE_c_violation

!===============================================================================
!===============================================================================

      END SUBROUTINE FILTRANE_solve

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================         T E R M I N A T E         =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

      SUBROUTINE FILTRANE_terminate( control, inform, s )

!     Cleans up the workspace and the map space.

!     Arguments:

      TYPE ( FILTRANE_control_type ), INTENT( INOUT ) :: control

!              the FILTRANE control structure (see above)

      TYPE ( FILTRANE_inform_type ), INTENT( INOUT ) :: inform

!              the FILTRANE exit information structure (see above)

      TYPE ( FILTRANE_data_type ), INTENT( INOUT ) :: s

!              the FILTRANE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Allow for a revision of the printout level, which is the only
!     possible significant change in the control parameters.

      SELECT CASE ( control%print_level )
      CASE ( SILENT )
         s%level = SILENT
      CASE ( TRACE )
         s%level = TRACE
      CASE ( ACTION )
         s%level = ACTION
      CASE ( DEBUG )
         s%level = DEBUG
      CASE ( CRAZY )
         s%level = CRAZY
      END SELECt

!     Clean up the various workspace arrays.

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, 1000 )
         IF ( s%level >= DETAILS ) WRITE( s%out, 1001 )
      END IF

      IF ( ASSOCIATED( s%r             ) ) DEALLOCATE( s%r              )
      IF ( ASSOCIATED( s%t             ) ) DEALLOCATE( s%t              )
      IF ( ASSOCIATED( s%v             ) ) DEALLOCATE( s%v              )
      IF ( ASSOCIATED( s%w             ) ) DEALLOCATE( s%w              )
      IF ( ASSOCIATED( s%step          ) ) DEALLOCATE( s%step           )
      IF ( ASSOCIATED( s%theta         ) ) DEALLOCATE( s%theta          )
      IF ( ASSOCIATED( s%group         ) ) DEALLOCATE( s%group          )
      IF ( ASSOCIATED( s%aut_group     ) ) DEALLOCATE( s%aut_group      )
      IF ( ASSOCIATED( s%g_status      ) ) DEALLOCATE( s%g_status       )
      IF ( ASSOCIATED( s%iw            ) ) DEALLOCATE( s%iw             )
      IF ( ASSOCIATED( s%row           ) ) DEALLOCATE( s%row            )
      IF ( ASSOCIATED( s%perm          ) ) DEALLOCATE( s%perm           )
      IF ( ASSOCIATED( s%diag          ) ) DEALLOCATE( s%diag           )
      IF ( ASSOCIATED( s%offdiag       ) ) DEALLOCATE( s%offdiag        )
      IF ( ASSOCIATED( s%vote          ) ) DEALLOCATE( s%vote           )
      IF ( ASSOCIATED( s%best_x        ) ) DEALLOCATE( s%best_x         )
      IF ( ASSOCIATED( s%filter_1      ) ) DEALLOCATE( s%filter_1       )
      IF ( ASSOCIATED( s%filter_2      ) ) DEALLOCATE( s%filter_2       )
      IF ( ASSOCIATED( s%filter_next_1 ) ) DEALLOCATE( s%filter_next_1  )
      IF ( ASSOCIATED( s%filter_next_2 ) ) DEALLOCATE( s%filter_next_2  )
      IF ( ASSOCIATED( s%active_1      ) ) DEALLOCATE( s%active_1       )
      IF ( ASSOCIATED( s%active_2      ) ) DEALLOCATE( s%active_2       )
      IF ( ASSOCIATED( s%fnorm_1       ) ) DEALLOCATE( s%fnorm_1        )
      IF ( ASSOCIATED( s%fnorm_2       ) ) DEALLOCATE( s%fnorm_2        )
      IF ( s%u_allocated ) THEN
         IF ( ASSOCIATED( s%u          ) ) DEALLOCATE( s%u              )
      END IF

      IF ( s%level >= DETAILS ) WRITE( s%out, 1002 )

      s%stage = VOID

      CALL FILTRANE_say_goodbye( control, inform, s )

      RETURN

!     Formats

1000  FORMAT(/,1x,'FILTRANE workspace cleanup',/)
1001  FORMAT(3x,'cleaning up FILTRANE temporaries')
1002  FORMAT(3x,'temporaries cleanup successful')

      END SUBROUTINE FILTRANE_terminate

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================       I N D E P E N D E N T       =========================
!===================                                   =========================
!===================           R O U T I N E S         =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================


      SUBROUTINE FILTRANE_banner( out )

!     Prints the FILTRANE banner.

!     Arguments:

      INTEGER, INTENT( IN ) :: out

!              the output device.

!     Programming: Ph. L. Toint, Spring 2003.

!==============================================================================

      WRITE( out, 1000 )
      RETURN

!     Format

1000  FORMAT(/,14x,'**************************************************',/,     &
               14x,'*                                                *',/,     &
               14x,'*                   FILTRANE                     *',/,     &
               14x,'*                                                *',/,     &
               14x,'*     GALAHAD filter trust-region algorithm      *',/,     &
               14x,'*                                                *',/,     &
               14x,'*     for the nonlinear feasibility problem      *',/,     &
               14x,'*                                                *',/,     &
               14x,'**************************************************',/)

      END SUBROUTINE FILTRANE_banner

!==============================================================================
!==============================================================================

      SUBROUTINE FILTRANE_say_goodbye( control, inform, s )

!     Print diagnostic and say goodbye.

!     Arguments:

      TYPE ( FILTRANE_control_type ), INTENT( INOUT ) :: control

!              the FILTRANE control structure (see above)

      TYPE ( FILTRANE_inform_type ), INTENT( INOUT ) :: inform

!              the FILTRANE exit information structure (see above)

      TYPE ( FILTRANE_data_type ), INTENT( INOUT ) :: s

!              the FILTRANE saved information structure (see above)

!     Programming: Ph. L. Toint, Spring 2003.

!==============================================================================

!     Local variable

      INTEGER :: line

!     Terminate GLTR, if needed

      IF ( s%gltr_initialized ) THEN
         CALL GLTR_terminate( s%GLTR_data, s%GLTR_control, s%GLTR_info )
         s%gltr_initialized = .FALSE.
         IF ( s%level >= DEBUG ) WRITE( s%out, 1004 )
      END IF

!     Successful exit

      IF ( control%print_level >= TRACE ) THEN
         IF ( inform%status == OK ) THEN
            SELECT CASE ( s%stage )
            CASE ( READY )
               WRITE( s%out, 1000 )
            CASE ( DONE )
               WRITE( s%out, 1003 ) TRIM( inform%message( 1 ) )
            CASE ( VOID )
            END SELECT

!     Unsuccessful exit

         ELSE
            DO line = 1, 3
               IF ( LEN_TRIM( inform%message( line ) ) > 0 ) THEN
                  WRITE( control%errout, 1003 ) TRIM( inform%message( line ) )
               ELSE
                  EXIT
               END IF
            END DO
         END IF
         WRITE( s%out, 1002 )
      END IF

      RETURN

!     Formats

1000  FORMAT(/,1x,'Problem successfully set up.')
1002  FORMAT(/,14x,'*********************** Bye **********************',/)
1003  FORMAT(/,a)
1004  FORMAT(4x,'GLTR terminated')

      END SUBROUTINE FILTRANE_say_goodbye

!==============================================================================
!==============================================================================

!  End of module FILTRANE

   END MODULE GALAHAD_FILTRANE_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                       *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*   END GALAHAD FILTRANE  M O D U L E   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                       *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
