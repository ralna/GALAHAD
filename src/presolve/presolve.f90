! THIS VERSION: GALAHAD 2.4 - 07/02/2011 AT 11:30 GMT.

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*    GALAHAD PRESOLVE  M O D U L E    *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                     *-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!
!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Philippe Toint
!
!  History -
!   originally released pre GALAHAD Version 1.0. March 1st 2002
!   update released with GALAHAD Version 2.0. February 16th 2005
!
!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html
!
!             +-----------------------------------------------+
!             |                                               |
!             | Presolve the data for the quadratic program   |
!             |                                               |
!             |    minimize q( x ) = 1/2 x^T H x + g^T x + f  |
!             |                                               |
!             |    subject to        x_l <=   x   <= x_u      |
!             |    and               c_l <=  A x  <= c_u      |
!             |                                               |
!             | to make future manipulations more efficient.  |
!             |                                               |
!             +-----------------------------------------------+
!
!  Presolving consists in simplifying the formulation of the problem using
!  simple transformations, in order to produce a "reduced" problem in a
!  "standard form".  This reduced problem is then passed to a solver.
!  Once the reduced problem has been solved, it is then "restored" to obtain
!  its solution in the context of its original formulation.
!
!  The standard form of the reduced problem is as follows:
!
!     - The variables are ordered so that their bounds appear in the order
!
!                     free                      x
!                     non-negativity      0  <= x
!                     lower              x_l <= x
!                     range              x_l <= x <= x_u
!                     upper                     x <= x_u
!                     non-positivity            x <=  0
!
!       Fixed variables are removed. Within each category, the variables
!       are further ordered so that those with non-zero diagonal Hessian
!       entries occur before the remainder.
!
!     - The constraints are ordered so that their bounds appear in the order
!
!                     non-negativity      0  <= A x
!                     equality           c_l  = A x
!                     lower              c_l <= A x
!                     range              c_l <= A x <= c_u
!                     upper                     A x <= c_u
!                     non-positivity            A x <=  0
!
!      Free constraints are removed.
!
!  In addition:
!
!     - Constraints may be removed or bounds tightened, to reduce the size of
!       the feasible region or simplify the problem if this is possible.
!
!     - Bounds may be tightened on the dual variables and the multipliers
!       associated  with the problem.
!
!  The PRESOLVE module implements the presolving process by  providing
!  different subroutines  which specify the various actions that occur before
!  and after the solution of the reduced problem.  These routines are:
!
!  - PRESOLVE_initialize:
!         creates data structure and specifies default PRESOLVE control
!         parameters
!  - PRESOLVE_read_specfile:
!         optional reads a specification file (that allows changing some
!         control parameters without recompiling the code);
!  - PRESOLVE_apply:
!         reduces the problem and permutes the resulting reduced problem
!         to standard form;
!  - PRESOLVE_restore:
!         restores the (solved) reduced problem to the original definition
!         of variables and constraints;
!  - PRESOLVE_terminate:
!         deallocates the workspace that is specific to PRESOLVE.
!
!  The presolving process follows one of the following two sequences:
!
!
!                 +--------------------+         +---------------------+
!                 |    INITIALIZE      |         |     INITIALIZE      |
!                 +--------------------+         +---------------------+
!                          |                               |
!                 +--------------------+                   |
!                 |    READ_SPECFILE   |                   |
!                 +--------------------+                   |
!                          |                               |
!                 +--------------------+         +---------------------+
!                 |      APPLY         |         |       APPLY         |
!                 +--------------------+         +---------------------+
!                          |                               |
!                       (solve)                         (solve)
!                          |                               |
!                 +--------------------+         +---------------------+
!                 |     RESTORE        |         |      RESTORE        |
!                 +--------------------+         +---------------------+
!                          |                               |
!                 +--------------------+         +---------------------+
!                 |     TERMINATE      |         |      TERMINATE      |
!                 +--------------------+         +---------------------+
!
!  where (solve) indicates that the reduced problem is solved by a quadratic
!  or linear programming solver, thus ensuring sufficiently small primal-dual
!  feasibility and complementarity.
!
!  The module handles problems where the matrices A and H can be defined in
!  dense, sparse coordinate or sparse-by-rows formats.  However, the first two
!  of these formats are converted into sparse-by-rows for preprocessing
!  solution and post-processing.  Note that this implies that the solver
!  used in the (solve) stage must be able to handle sparse-by-rows format.
!
!  The process is controlled by the values assigned to the control parameters
!  (see the description of the PRESOLVE_control structure below).
!
!  Note that the values of the multipliers and dual variables (and thus of
!  their respective bounds) depend on the functional form assumed for the
!  Lagrangian function associated with the problem.  This form is given by
!
!        L( x, y, z ) = q( x ) - y_sign * y^T ( A x - c ) - z_sign * z,
!
!  (considering only active constraints A x = c), where the parameters
!  y_sign and z_sign are +1 or -1 and can be chosen by the user.  Thus,
!  if y_sign = +1, the multipliers associated to active constraints originally
!  posed as inequalities are non-negative if the inequality is a lower bound
!  and non-positive if it is an upper bound. Obvioulsy they are not
!  constrained in sign for constraints originally posed as equalities. These
!  sign conventions are reversed if y_sign = -1.
!  Similarly, if z_sign = +1, the dual variables associated to active bounds
!  are non-negative if the original bound is an lower bound, non-positive if
!  it is an upper bound, or unconstrained in sign if the variables is fixed;
!  and this convention is reversed in z_sign = -1.
!
!  The presolving techniques used in this module are
!
!     - the removal of empty and singleton rows,
!     - the removal of redundant and forcing primal constraints,
!     - the tightening of primal and dual bounds,
!     - the exploitation of linear singleton, linear doubleton and linearly
!       unconstrained columns,
!     - the merging dependent variables,
!     - row sparsification,
!     - split equalities.
!
!  They are documented in the paper
!
!     N. I. M. Gould and Ph. L. Toint, "Preprocessing for quadratic programming"
!     Report 01/10, Department of Mathematics, University of Namur, 2001.
!
!-------------------------------------------------------------------------------
!
!              T h e   s p e c i f i c a t i o n   f i l e
!
!-------------------------------------------------------------------------------
!
!  Like most algorithmic packages, PRESOLVE features a number of "control
!  parameters", that is of parameters that condition the various algorithmic,
!  printing and other options of the package (They are documented in detail
!  with the PRESOLVE_control type below). While the value of these parameters
!  may be changed (for overiding their default values) within the calling code
!  itself, it is often convenient to be able to alter those parameters without
!  recompiling the code.  This is achieved by specifying the desired values
!  for the control parameters in a "specification file" (specfile).

!  A specification file consists of a number of "specification commands",
!  each of these being decomposed into
!  - a "keyword", which is a string (in a close-to-natural language) that will
!    be used to identify a control parameter in the specfile, and
!  - an (optional) "value", which is the value to be attributed to the
!    said control parameter.

!  A specific algorithmic "control parameter" is associated to each such
!  keyword, and the effect of interpreting the specification file is to assign
!  the value associated with the keyword (in each specification command) to
!  the corresponding algorithmic parameter. The specification file starts with
!  a "BEGIN PRESOLVE SPECIFICATIONS" command and ends with an
!  "END PRESOLVE SPECIFICATIONS" command.  The syntax of the specfile is
!  defined as follows:

!      BEGIN PRESOLVE SPECIFICATIONS
!         printout-device                            (integer)
!         error-printout device                      (integer)
!         print-level                    SILENT|TRACE|ACTION|DETAILS|DEBUG|CRAZY
!         presolve-termination-strategy              REDUCED_SIZE|FULL_PRESOLVE
!         maximum-number-of-transformations          (integer)
!         maximum-number-of-passes                   (integer)
!         constraints-accuracy                       (real)
!         dual-variables-accuracy                    (real)
!         allow-dual-transformations                 ON|OFF|TRUE|FALSE|T|F|
!         remove-redundant-variables-constraints     ON|OFF|TRUE|FALSE|T|F|
!         primal-constraints-analysis-frequency      (integer)
!         dual-constraints-analysis-frequency        (integer)
!         singleton-columns-analysis-frequency       (integer)
!         doubleton-columns-analysis-frequency       (integer)
!         unconstrained-variables-analysis-frequency (integer)
!         dependent-variables-analysis-frequency     (integer)
!         row-sparsification-frequency               (integer)
!         maximum-percentage-row-fill                (integer)
!         transformations-buffer-size                (integer)
!         transformations-file-device                (integer)
!         transformations-file-status                KEEP|DELETE
!         transformations-file-name                  (filename)
!         primal-feasibility-check                   NONE|BASIC|SEVERE
!         dual-feasibility-check                     NONE|BASIC|SEVERE
!         active-multipliers-sign                    POSITIVE!NEGATIVE
!         inactive-multipliers-value                 LEAVE_AS_IS|FORCE_TO_ZERO
!         active-dual-variables-sign                 POSITIVE|NEGATIVE
!         inactive-dual-variables-value              LEAVE_AS_IS|FORCE_TO_ZERO
!         primal-variables-bound-status          TIGHTEST|NON_DEGENERATE|LOOSEST
!         dual-variables-bound-status            TIGHTEST|NON_DEGENERATE|LOOSEST
!         constraints-bound-status               TIGHTEST|NON_DEGENERATE|LOOSEST
!         multipliers-bound-status               TIGHTEST|NON_DEGENERATE|LOOSEST
!         infinity-value                             (real)
!         pivoting-threshold                         (real)
!         minimum-relative-bound-improvement         (real)
!         maximum-growth-factor                      (real)
!         compute-quadratic-value                    ON|OFF|TRUE|FALSE|T|F|
!         compute-objective-constant                 ON|OFF|TRUE|FALSE|T|F|
!         compute-gradient                           ON|OFF|TRUE|FALSE|T|F|
!         compute-Hessian                            ON|OFF|TRUE|FALSE|T|F|
!         compute-constraints-matrix                 ON|OFF|TRUE|FALSE|T|F|
!         compute-primal-variables-values            ON|OFF|TRUE|FALSE|T|F|
!         compute-primal-variables-bounds            ON|OFF|TRUE|FALSE|T|F|
!         compute-dual-variables-values              ON|OFF|TRUE|FALSE|T|F|
!         compute-dual-variables-bounds              ON|OFF|TRUE|FALSE|T|F|
!         compute-constraints-values                 ON|OFF|TRUE|FALSE|T|F|
!         compute-constraints-bounds                 ON|OFF|TRUE|FALSE|T|F|
!         compute-multipliers-values                 ON|OFF|TRUE|FALSE|T|F|
!         compute-multipliers-bounds                 ON|OFF|TRUE|FALSE|T|F|
!      END PRESOLVE SPECIFICATIONS
!
!  where the | symbols means "or".  Thus print-level may take the values
!  SILENT or TRACE or ACTION or DETAILS or DEBUG or CRAZY, where the upper
!  case words are symbols that are recognized by PRESOLVE. Empty values are
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
!  The specification file must be open for input when PRESOLVE_read_specfile
!  is called. Note that the corresponding file is first REWINDed, which make
!  it possible to combine it with other specifications associated with other
!  algorithms used in conjunction with PRESOLVE, such as the quadratic
!  programming routine. For the same reason, the file is not closed by PRESOLVE.
!
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   MODULE GALAHAD_PRESOLVE_double

!-------------------------------------------------------------------------------
!   U s e d   m o d u l e s   a n d   s y m b o l s
!-------------------------------------------------------------------------------

      USE GALAHAD_SMT_double      ! the matrix data type

      USE GALAHAD_QPT_double      ! the quadratic problem data type

      USE GALAHAD_SORT_double     ! sorting and permutation operations

      USE GALAHAD_SPECFILE_double ! specfile manipulation

!     Matrix storage schemes

      USE GALAHAD_SYMBOLS,                                                     &
          DIAGONAL              => GALAHAD_DIAGONAL,                           &
          DENSE                 => GALAHAD_DENSE,                              &
          SPARSE                => GALAHAD_SPARSE_BY_ROWS,                     &
          COORDINATE            => GALAHAD_COORDINATE,                         &
          ABSENT                => GALAHAD_ZERO

!     Variable and constraints status

      USE GALAHAD_SYMBOLS,                                                     &
          INACTIVE              => GALAHAD_INACTIVE,                           &
          ELIMINATED            => GALAHAD_ELIMINATED,                         &
          ACTIVE                => GALAHAD_ACTIVE,                             &
          RANGE                 => GALAHAD_RANGE,                              &
          UPPER                 => GALAHAD_UPPER,                              &
          LOWER                 => GALAHAD_LOWER,                              &
          FREE                  => GALAHAD_FREE

!     Sign conventions for the multipliers and dual variables

      USE GALAHAD_SYMBOLS,                                                     &
          POSITIVE              => GALAHAD_POSITIVE,                           &
          NEGATIVE              => GALAHAD_NEGATIVE

!     Special values

      USE GALAHAD_SYMBOLS,                                                     &
          ALL_ZEROS             => GALAHAD_ALL_ZEROS,                          &
          ALL_ONES              => GALAHAD_ALL_ONES

!     Print levels

      USE GALAHAD_SYMBOLS,                                                     &
          SILENT                => GALAHAD_SILENT,                             &
          TRACE                 => GALAHAD_TRACE,                              &
          ACTION                => GALAHAD_ACTION,                             &
          DETAILS               => GALAHAD_DETAILS,                            &
          DEBUG                 => GALAHAD_DEBUG,                              &
          CRAZY                 => GALAHAD_CRAZY

!     Termination strategies

      USE GALAHAD_SYMBOLS,                                                     &
          REDUCED_SIZE          => GALAHAD_REDUCED_SIZE,                       &
          FULL_PRESOLVE         => GALAHAD_FULL_PRESOLVE

!     Policy wrt dual variables and multipliers (inactive)

      USE GALAHAD_SYMBOLS,                                                     &
          FORCE_TO_ZERO         => GALAHAD_FORCE_TO_ZERO,                      &
          LEAVE_AS_IS           => GALAHAD_LEAVE_AS_IS

!     Final status of the bounds on the variables

      USE GALAHAD_SYMBOLS,                                                     &
          TIGHTEST              => GALAHAD_TIGHTEST,                           &
          NON_DEGENERATE        => GALAHAD_NON_DEGENERATE,                     &
          LOOSEST               => GALAHAD_LOOSEST

!     Level of feasibility checks

      USE GALAHAD_SYMBOLS,                                                     &
          NONE                  => GALAHAD_NONE,                               &
          BASIC                 => GALAHAD_BASIC,                              &
          SEVERE                => GALAHAD_SEVERE

!     Final status for files produced by the module

      USE GALAHAD_SYMBOLS,                                                     &
          KEEP                  => GALAHAD_KEEP,                               &
          DELETE                => GALAHAD_DELETE

!     Exit conditions

      USE GALAHAD_SYMBOLS,                                                     &
          OK                    => GALAHAD_SUCCESS,                            &
          MEMORY_FULL           => GALAHAD_MEMORY_FULL,                        &
          FILE_NOT_OPENED       => GALAHAD_FILE_NOT_OPENED,                    &
          COULD_NOT_WRITE       => GALAHAD_COULD_NOT_WRITE,                    &
          TOO_FEW_BITS_PER_BYTE => GALAHAD_TOO_FEW_BITS_PER_BYTE,              &
          NOT_DIAGONAL          => GALAHAD_NOT_DIAGONAL

!-------------------------------------------------------------------------------
!   A c c e s s
!-------------------------------------------------------------------------------

      IMPLICIT NONE

!     Make everything private by default

      PRIVATE

!     Ensure the PRIVATE nature of the imported symbols

      PRIVATE :: DENSE, SPARSE, COORDINATE, INACTIVE, ELIMINATED,              &
                 ACTIVE, RANGE, UPPER, LOWER, FREE, REDUCED_SIZE,FULL_PRESOLVE,&
                 SILENT, TRACE, ACTION, DETAILS, DEBUG, CRAZY, OK, MEMORY_FULL,&
                 FILE_NOT_OPENED, COULD_NOT_WRITE, TOO_FEW_BITS_PER_BYTE,      &
                 FORCE_TO_ZERO, LEAVE_AS_IS, TIGHTEST, NON_DEGENERATE, LOOSEST,&
                 NONE, BASIC, SEVERE, KEEP, DELETE

!     Make the PRESOLVE calls public

      PUBLIC :: PRESOLVE_initialize, PRESOLVE_read_specfile, PRESOLVE_apply,   &
                PRESOLVE_restore,    PRESOLVE_terminate, SMT_put, SMT_get

!-------------------------------------------------------------------------------
!   P r e c i s i o n
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: sp = KIND( 1.0 )
      INTEGER, PRIVATE, PARAMETER :: dp = KIND( 1.0D+0 )
      INTEGER, PRIVATE, PARAMETER :: wp = dp

!-------------------------------------------------------------------------------
!   C o n s t a n t s
!-------------------------------------------------------------------------------

      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ZERO    = 0.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: HALF    = 0.5_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: ONE     = 1.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: TWO     = 2.0_wp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: TEN = 10.0_wp
      REAL ( KIND = sp ), PRIVATE, PARAMETER :: TEN_sp  = 10.0_sp
      REAL ( KIND = dp ), PRIVATE, PARAMETER :: TEN_dp  = 10.0_dp
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: HUNDRED = TEN * TEN
      REAL ( KIND = wp ), PRIVATE, PARAMETER :: EPSMACH = EPSILON( ONE )

!-------------------------------------------------------------------------------
!   D e f a u l t   V a  l u e s
!-------------------------------------------------------------------------------

!     Default printout unit number

      INTEGER, PRIVATE, PARAMETER :: DEF_WRITE_UNIT            = 6

!     Default error output unit number

      INTEGER, PRIVATE, PARAMETER :: DEF_ERROR_UNIT            = 6

!     Default printout level

      INTEGER, PRIVATE, PARAMETER :: DEF_PRINT_LEVEL           = SILENT

!     Value beyond which a number is equal to plus infinity

      REAL( KIND = sp ), PRIVATE, PARAMETER :: DEF_INFINITY_sp = TEN_sp ** 19
      REAL( KIND = dp ), PRIVATE, PARAMETER :: DEF_INFINITY_dp = TEN_dp ** 19

!     Default maximum number of analysis passes for a single call to PRESOLVE

      INTEGER, PRIVATE, PARAMETER :: DEF_MAX_NBR_PASSES        = 25

!     Default maximum number of problem transformations

      INTEGER, PRIVATE, PARAMETER :: DEF_MAX_NBR_TRANSF        = 1000000

!     Default buffer size for the transformation history

      INTEGER, PRIVATE, PARAMETER :: DEF_MAX_T_BUFFER          = 50000

!     Default name for the file used to store the problem
!     transformations on disk

      CHARACTER( LEN = 30 ), PRIVATE, PARAMETER ::                             &
                                      DEF_TRANSF_FILE_NAME       = 'transf.sav'

!     Default unit number for these files

      INTEGER, PRIVATE, PARAMETER :: DEF_TRANSF_FILE_NBR       =  57

!     Default maximum percentage of row-wise fills in A

      INTEGER, PRIVATE, PARAMETER :: DEF_MAX_FILL              =  -1

!     Default relative tolerance for pivoting in A

      REAL ( KIND = sp ), PRIVATE, PARAMETER ::                                &
                                   DEF_PIVOT_TOL_sp = TEN_sp ** ( -6 )
      REAL ( KIND = dp ), PRIVATE, PARAMETER ::                                &
                                   DEF_PIVOT_TOL_dp = TEN_dp ** (-10 )

!     Default minimum relative bound improvement

      REAL ( KIND = sp ), PRIVATE, PARAMETER :: DEF_MRBI_sp = TEN_sp ** ( -6 )
      REAL ( KIND = dp ), PRIVATE, PARAMETER :: DEF_MRBI_dp = TEN_dp ** ( -10 )

!     Default maximum growth factor between original and reduced problems

      REAL ( KIND = sp ), PRIVATE, PARAMETER :: DEF_MAX_GROWTH_sp = TEN_sp ** 4
      REAL ( KIND = dp ), PRIVATE, PARAMETER :: DEF_MAX_GROWTH_dp = TEN_dp ** 8

!     Default relative accuracy for the linear constraints

      REAL ( KIND = sp ), PRIVATE, PARAMETER :: DEF_C_ACC_sp = TEN_sp ** ( -4 )
      REAL ( KIND = dp ), PRIVATE, PARAMETER :: DEF_C_ACC_dp = TEN_dp ** ( -6 )

!     Default relative accuracy for dual variables

      REAL ( KIND = sp ), PRIVATE, PARAMETER :: DEF_Z_ACC_sp = TEN_sp ** ( -4 )
      REAL ( KIND = dp ), PRIVATE, PARAMETER :: DEF_Z_ACC_dp = TEN_dp ** ( -6 )

!     Default frequency for primal constraint analysis

      INTEGER, PRIVATE, PARAMETER :: DEF_AN_PRIMAL_FREQ        =  1

!     Default frequency for dual constraint analysis

      INTEGER, PRIVATE, PARAMETER :: DEF_AN_DUAL_FREQ          =  1

!     Default frequency for analysis of singleton columns

      INTEGER, PRIVATE, PARAMETER :: DEF_AN_SING_FREQ          =  1

!     Default frequency for analysis of doubleton columns

      INTEGER, PRIVATE, PARAMETER :: DEF_AN_DOUB_FREQ          =  1

!     Default frequency for analysis of linearly unconstrained variables

      INTEGER, PRIVATE, PARAMETER :: DEF_UNC_VARS_FREQ         =  1

!     Default frequency for analysis of dependent variables

      INTEGER, PRIVATE, PARAMETER :: DEF_DEP_COLS_FREQ         =  1

!     Default frequency for row sparsification analysis

      INTEGER, PRIVATE, PARAMETER :: DEF_SPARSIFY_FREQ         =  1

!     Default frequency for bound checks

      INTEGER, PRIVATE, PARAMETER :: DEF_BD_CHECK_FREQ         =  1

!-------------------------------------------------------------------------------
!   N u m b e r   o f   h e u r i s t i c s
!-------------------------------------------------------------------------------

      INTEGER, PRIVATE, PARAMETER :: NBRH = 8

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!                      PUBLIC TYPES AND DEFINITIONS
!
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!     NOTE 1: None of these definitions should be altered by the user!!!
!
!-------------------------------------------------------------------------------
!              The structure that controls the PRESOLVE execution
!-------------------------------------------------------------------------------

!     NOTE:  The default values for the components of this structure are set
!            by executing PRESOLVE_initialize.  They must therefore
!            be altered to reflect user's need *only after* that execution.

      TYPE, PUBLIC :: PRESOLVE_control_type

         INTEGER :: termination = REDUCED_SIZE            ! INTENT( IN )

!                   Determines the strategy for terminating the presolve
!                   analysis.  Possible values are:
!                   - REDUCED_SIZE : presolving is continued as long as one of
!                         the sizes of the problem (n, m, a_ne, or h_ne) is
!                         being reduced;
!                   - FULL_PRESOLVE: presolving is continued as long as problem
!                         transformations remain possible.
!                   NOTE: the maximum number of analysis passes
!                         (control%max_nbr_passes)  and the maximum number of
!                         problem transformations (control%max_nbr_transforms)
!                         set an upper limit on the  presolving effort
!                         irrespective of the choice of control%termination.
!                         The only effect of this latter parameter is to allow
!                         for early termination.
!                   Default: REDUCED_SIZE

         INTEGER :: max_nbr_transforms = DEF_MAX_NBR_TRANSF ! INTENT( IN )

!                   The maximum number of problem transformations, cumulated
!                   over all calls to PRESOLVE.
!                   Default: 1000000

         INTEGER :: max_nbr_passes = DEF_MAX_NBR_PASSES ! INTENT( IN )

!                   The maximum number of analysis passes for problem analysis
!                   during a single call of PRESOLVE_apply.
!                   Default: 25

         REAL ( KIND = wp ) :: c_accuracy = TEN ** ( - 6 )  ! INTENT( IN )

!                   The relative accuracy at which the general linear
!                   constraints are satisfied at the exit of the solver.
!                   Note that this value is not used before the restoration
!                   of the problem.
!                   Default: 10.**(-6) in double precision,
!                            10.**(-4) in single precision.

         REAL ( KIND = wp ) :: z_accuracy = TEN ** ( - 6 )  ! INTENT( IN )

!                   The relative accuracy at which the dual feasibility
!                   constraints are satisfied at the exit of the solver.
!                   Note that this value is not used before the restoration
!                   of the problem.
!                   Default: 10.**(-6) in double precision,
!                            10.**(-4) in single precision.

         REAL ( KIND = wp ) :: infinity = ten ** 19  ! INTENT( IN )

!                   The value beyond which a number is deemed equal to
!                   plus infinity
!                   (minus infinity being defined as its opposite)
!                   Default: 10.**(19).

         INTEGER :: out = DEF_WRITE_UNIT            ! INTENT( IN )

!                   The unit number associated with the device used for
!                   printout.
!                   Default: 6

         INTEGER :: errout = DEF_WRITE_UNIT         ! INTENT( IN )

!                   The unit number associated with the device used for
!                   error ouput.
!                   Default: 6

         INTEGER :: print_level = DEF_PRINT_LEVEL  ! INTENT( IN )

!                   The level of printout requested by the user. Can take
!                   the values:
!                   - SILENT  : no printout is produced
!                   - TRACE   : only reports the major steps in the analysis
!                   - ACTION  : reports the identity of each problem
!                               transformation
!                   - DETAILS : reports more details
!                   - DEBUG   : reports LOTS of information.
!                   - CRAZY   : reports a completely silly amount of information
!                   Default: SILENT

         LOGICAL :: dual_transformations = .TRUE.     ! INTENT( IN )

!                   .TRUE. if dual transformations of the problem are allowed.
!                   Note that this implies that the reduced problem is solved
!                   accurately (for the dual feasibility condition to hold)
!                   as to be able to restore the problem to the original
!                   constraints and variables. .FALSE. prevents dual
!                   transformations to be applied, thus allowing for inexact
!                   solution of the reduced problem. The setting of this control
!                   parameter overides that of get_z, get_z_bounds, get_y,
!                   get_y_bounds, dual_constraints_freq, singleton_columns_freq,
!                   doubleton_columns_freq, z_accuracy, check_dual_feasibility.
!                   Default: .TRUE.

         LOGICAL :: redundant_xc = .TRUE.             ! INTENT( IN )

!                   .TRUE. if the redundant variables and constraints (i.e.
!                   variables that do not appear in the objective
!                   function and appear with a consistent sign in the
!                   constraints) are to be removed with their associated
!                   constraints before other transformations are attempted.
!                   Default: .TRUE.

         INTEGER :: primal_constraints_freq = DEF_AN_PRIMAL_FREQ ! INTENT( IN )

!                   The frequency of primal constraints analysis in terms of
!                   presolving passes.  A value of j = 2 indicates that primal
!                   constraints are analyzed every 2 presolving passes. A zero
!                   value indicates that they are never analyzed.
!                   Default: 1

         INTEGER :: dual_constraints_freq = DEF_AN_DUAL_FREQ     ! INTENT( IN )

!                   The frequency of dual constraints analysis in terms of
!                   presolving passes.  A value of j = 2 indicates that dual
!                   constraints are analyzed every 2 presolving passes.  A zero
!                   value indicates that they are never analyzed.
!                   Default: 1

         INTEGER :: singleton_columns_freq = DEF_AN_SING_FREQ    ! INTENT( IN )

!                   The frequency of singleton column analysis in terms of
!                   presolving passes.  A value of j = 2 indicates that
!                   singleton columns are analyzed every 2 presolving passes.
!                   A zero value indicates that they are never analyzed.
!                   Default: 1

         INTEGER :: doubleton_columns_freq = DEF_AN_DOUB_FREQ    ! INTENT( IN )

!                   The frequency of doubleton column analysis in terms of
!                   presolving passes.  A value of j indicates that doubleton
!                   columns are analyzed every 2 presolving passes.  A zero
!                   value indicates that they are never analyzed.
!                   Default: 1

         INTEGER :: unc_variables_freq  = DEF_UNC_VARS_FREQ     ! INTENT( IN )

!                   The frequency of the attempts to fix linearly unconstrained
!                   variables, expressed in terms of presolving passes.  A
!                   value of j = 2 indicates that attempts are made every 2
!                   presolving passes.  A zero value indicates that no attempt
!                   is ever made.
!                   Default: 1

         INTEGER :: dependent_variables_freq = DEF_DEP_COLS_FREQ ! INTENT( IN )

!                   The frequency of search for dependent variables in terms of
!                   presolving passes.  A value of j = 2 indicates that
!                   dependent variables are searched for every 2 presolving
!                   passes.  A zero value indicates that they are never
!                   searched for.
!                   Default: 1

         INTEGER :: sparsify_rows_freq   = DEF_SPARSIFY_FREQ     ! INTENT( IN )

!                   The frequency of the attempts to make A sparser in terms of
!                   presolving passes.  A value of j = 2 indicates that attempts
!                   are made every 2 presolving passes.  A zero value indicates
!                   that no attempt is ever made.
!                   Default: 1

         INTEGER :: max_fill = DEF_MAX_FILL     ! INTENT( IN )

!                   The maximum percentage of fill in each row of A. Note that
!                   this is a row-wise measure: globally fill never exceeds
!                   the storage initially used for A, no matter how large
!                   control%max_fill is chosen. If max_fill is negative,
!                   no limit is put on row fill.
!                   Default: -1 (no limit).

         INTEGER :: transf_file_nbr = DEF_TRANSF_FILE_NBR  ! INTENT( IN )

!                   The unit number to be associated with the file(s) used
!                   for saving problem transformations on a disk file.
!                   Default: 57

         INTEGER :: transf_buffer_size = DEF_MAX_T_BUFFER ! INTENT( IN )

!                   The number of transformations that can be kept in memory
!                   at once (that is without being saved on a disk file).
!                   Default: 50000

         INTEGER :: transf_file_status = KEEP    ! INTENT( IN )

!                   The exit status of the file where problem transformations
!                   are saved:
!                   KEEP   : the file is not deleted after program termination
!                   DELETE : the file is not deleted after program termination
!                   Default: KEEP

         CHARACTER( LEN = 30 ) :: transf_file_name  = DEF_transf_file_name
                                                ! INTENT( IN )

!                   The name of the file (to be) used for storing
!                   problem transformation on disk.
!                   Default: transf.sav
!                   NOTE: this parameter must be identical for all calls to
!                         PRESOLVE following PRESOLVE_read_specfile. It can
!                         then only be changed after calling PRESOLVE_terminate.

         INTEGER :: y_sign = POSITIVE          ! INTENT( IN )

!                   Determines the convention of sign used for the multipliers
!                   associated with the general linear constraints.
!                   - POSITIVE ( +1 ): All multipliers corresponding to active
!                                inequality constraints are non-negative for
!                                lower bound constraints and non-positive for
!                                upper bounds constraints.
!                   - NEGATIVE ( -1 ): All multipliers corresponding to active
!                                inequality constraints are non-positive for
!                                lower bound constraints and non-negative for
!                                upper bounds constraints.
!                   Default: POSITIVE.

         INTEGER :: inactive_y = LEAVE_AS_IS   ! INTENT( IN )

!                   Determines whether or not the multipliers corresponding
!                   to constraints that are inactive at the unreduced point
!                   corresponding to the reduced point on input of RESTORE
!                   must be set to zero. Possible values are:
!                   associated with the general linear constraints.
!                   - FORCE_TO_ZERO: All multipliers corresponding to inactive
!                                inequality constraints are forced to zero,
!                                possibly at the expense of deteriorating the
!                                dual feasibility condition.
!                                NOTE: this option is inactive unless
!                                      control%get_y = .TRUE.
!                                      control%get_c = .TRUE.
!                                      control%get_c_bounds = .TRUE.
!                   - LEAVE_AS_IS: Multipliers corresponding to inactive
!                                inequality constraints are left unaltered.
!                   Default: LEAVE_AS_IS

         INTEGER :: z_sign = POSITIVE            ! INTENT( IN )

!                   Determines the convention of sign used for the dual
!                   variables associated with the bound constraints.
!                   - POSITIVE ( +1 ): All dual variables corresponding to
!                                active lower bounds are non-negative, and
!                                non-positive for active upper bounds.
!                   - NEGATIVE ( -1 ): All dual variables corresponding to
!                                active lower bounds are non-positive, and
!                                non-negative for active upper bounds.
!                   Default: POSITIVE.

         INTEGER :: inactive_z = LEAVE_AS_IS     ! INTENT( IN )

!                   Determines whether or not the dual variables corresponding
!                   to bounds that are inactive at the unreduced point
!                   corresponding to the reduced point on input of RESTORE
!                   must be set to zero. Possible values are:
!                   associated with the general linear constraints.
!                   - FORCE_TO_ZERO: All dual variables corresponding to
!                                inactive bounds are forced to zero,
!                                possibly at the expense of deteriorating the
!                                dual feasibility condition.
!                                NOTE: this option is inactive unless
!                                      control%get_z = .TRUE.
!                                      control%get_x = .TRUE.
!                                      control%get_x_bounds = .TRUE.
!                   - LEAVE_AS_IS: Dual variables corresponding to inactive
!                                bounds are left unaltered.
!                   Default: LEAVE_AS_IS

         INTEGER :: final_x_bounds = TIGHTEST     ! INTENT( IN )

!                   The type of final bounds on the variables returned by the
!                   package.  This parameter can take the values:
!                   - TIGHTEST      : the final bounds are the tightest bounds
!                                     known on the variables (at the risk of
!                                     being redundant with other constraints,
!                                     which may cause degeneracy);
!                   - NON_DEGENERATE: the best known bounds that are known to
!                                     be non-degenerate. This option implies
!                                     that an additional real workspace of size
!                                     2 * prob%n must be allocated.
!                   - LOOSEST       : the loosest bounds that are known to
!                                     keep the problem equivalent to the
!                                     original problem. This option also
!                                     implies that an additional real
!                                     workspace of size 2 * prob%n must be
!                                     allocated.
!                   Default: TIGHTEST
!                   NOTE: this parameter must be identical for all calls to
!                         PRESOLVE (except INITIALIZE).

         INTEGER :: final_z_bounds = TIGHTEST       ! INTENT( IN )

!                   The type of final bounds on the dual variables returned by
!                   the package.  This parameter can take the values:
!                   - TIGHTEST      : the final bounds are the tightest bounds
!                                     known on the dual variables (at the risk
!                                     of being redundant with other constraints,
!                                     which may cause degeneracy);
!                   - NON_DEGENERATE: the best known bounds that are known to
!                                     be non-degenerate. This option implies
!                                     that an additional real workspace of size
!                                     2 * prob%n must be allocated.
!                   - LOOSEST       : the loosest bounds that are known to
!                                     keep the problem equivalent to the
!                                     original problem. This option also
!                                     implies that an additional real
!                                     workspace of size 2 * prob%n must be
!                                     allocated.
!                   Default: TIGHTEST
!                   NOTE: this parameter must be identical for all calls to
!                         PRESOLVE (except INITIALIZE).

         INTEGER :: final_c_bounds = TIGHTEST       ! INTENT( IN )

!                   The type of final bounds on the constraints returned by the
!                   package.  This parameter can take the values:
!                   - TIGHTEST      : the final bounds are the tightest bounds
!                                     known on the constraints (at the risk of
!                                     being redundant with other constraints,
!                                     which may cause degeneracy);
!                   - NON_DEGENERATE: the best known bounds that are known to
!                                     be non-degenerate. This option implies
!                                     that an additional real workspace of size
!                                     2 * prob%m must be allocated.
!                   - LOOSEST       : the loosest bounds that are known to
!                                     keep the problem equivalent to the
!                                     original problem. This option also
!                                     implies that an additional real
!                                     workspace of size 2 * prob%n must be
!                                     allocated.
!                   Default: TIGHTEST
!                   NOTES:
!                   1) This parameter must be identical for all calls to
!                      PRESOLVE (except INITIALIZE).
!                   2) If different from TIGHTEST, its value must be identical
!                      to that of control%final_x_bounds.

         INTEGER :: final_y_bounds = TIGHTEST       ! INTENT( IN )

!                   The type of final bounds on the multipliers returned by the
!                   package.  This parameter can take the values:
!                   - TIGHTEST      : the final bounds are the tightest bounds
!                                     known on the multipliers (at the risk of
!                                     being redundant with other constraints,
!                                     which may cause degeneracy);
!                   - NON_DEGENERATE: the best known bounds that are known to
!                                     be non-degenerate. This option implies
!                                     that an additional real workspace of size
!                                     2 * prob%m must be allocated.
!                   - LOOSEST       : the loosest bounds that are known to
!                                     keep the problem equivalent to the
!                                     original problem. This option also
!                                     implies that an additional real
!                                     workspace of size 2 * prob%n must be
!                                     allocated.
!                   Default: TIGHTEST
!                   NOTE: this parameter must be identical for all calls to
!                         PRESOLVE (except INITIALIZE).

         INTEGER :: check_primal_feasibility = NONE  ! INTENT( IN )

!                   The level of feasibility check (on the values of x) at
!                   the start of the restoration phase.  This parameter can
!                   take the values:
!                   - NONE  : no check at all;
!                   - BASIC : the primal constraints are recomputed at x
!                             and a message issued if the computed value
!                             does not match the input value, or if it is
!                             out of bounds (if control%print_level >= ACTION);
!                   - SEVERE: the same as for BASIC, but PRESOLVE is
!                             terminated if an incompatibilty is detected.
!                   Default: NONE

         INTEGER :: check_dual_feasibility = NONE   ! INTENT( IN )

!                   The level of dual feasibility check (on the values of x,
!                   y and z) at the start of the restoration phase.
!                   This parameter can take the values:
!                   - NONE  : no check at all;
!                   - BASIC : the dual feasibility condition is  recomputed
!                             at ( x, y, z ) and a message issued if the
!                             computed value does not match the input value
!                             (if control%print_level >= ACTION);
!                   - SEVERE: the same as for BASIC, but PRESOLVE is
!                             terminated if an incompatibilty is detected.
!                   The last two values imply the allocation of an additional
!                   real workspace vector of size equal to the number of
!                   variables in the reduced problem.
!                   Default: NONE

         LOGICAL :: get_q = .TRUE.                      ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the objective
!                   function must be reconstructed on RESTORE from the
!                   (possibly solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_f = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the objective
!                   function's independent term is to be be reconstructed
!                   on RESTORE from the (possibly solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_g = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the objective
!                   function's gradient must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_H = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the objective
!                   function's Hessian must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_A = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the constraints'
!                   coefficient matrix must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_x  = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the variables
!                   must be reconstructed on RESTORE from the (possibly
!                   solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_x_bounds = .TRUE.              ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the bounds on the
!                   problem variables must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   This parameter is only relevant in the RESTORE mode.
!                   Default: .TRUE.

         LOGICAL :: get_z = .TRUE.                      ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the dual variables
!                   must be reconstructed on RESTORE from the (possibly
!                   solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_z_bounds = .TRUE.              ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the bounds on the
!                   problem dual variables must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   If set to true, this may require to store specific
!                   additional information on the problem transformations,
!                   therefore increasing the storage needed for these
!                   transformations.
!                   Default: .TRUE.
!                   NOTE: this parameter must be identical for all calls to
!                         PRESOLVE (except INITIALIZE).

         LOGICAL :: get_c = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the constraints
!                   must be reconstructed on RESTORE from the (possibly
!                   solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_c_bounds= .TRUE.              ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the bounds on the
!                   problem constraints must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   This parameter is only relevant in the RESTORE mode.
!                   Default: .TRUE.

         LOGICAL :: get_y = .TRUE.                     ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the multipliers
!                   must be reconstructed on RESTORE from the (possibly
!                   solved) reduced problem.
!                   Default: .TRUE.

         LOGICAL :: get_y_bounds = .TRUE.              ! INTENT( IN )

!                   Must be set to .TRUE. if the value of the bounds on the
!                   problem multipliers must be reconstructed on RESTORE
!                   from the (possibly solved) reduced problem.
!                   If set to true, this may require to store specific
!                   additional information on the problem transformations,
!                   therefore increasing the storage needed for these
!                   transformations.
!                   Default: .FALSE.
!                   NOTE: this parameter must be identical for all calls to
!                         PRESOLVE (except INITIALIZE)

         REAL ( KIND = wp ) :: pivot_tol = TEN ** ( - 10 )  ! INTENT( IN )

!                   The relative pivot tolerance above which pivoting is
!                   considered as numerically stable in transforming the
!                   coefficient matrix A.  A zero value corresponds to a
!                   totally unsafeguarded pivoting strategy (potentially
!                   unstable).
!                   Default: 10.**(-10) in double precision,
!                            10.**(-6)  in single precision.

         REAL ( KIND = wp ) :: min_rel_improve = TEN ** ( - 10 ) ! INTENT( IN )

!                   The minimum relative improvement in the bounds on x, y
!                   and z for a tighter bound on these quantities to be
!                   accepted in the course of the analysis.  More formally,
!                   if lower is the current value of the lower bound on one
!                   of the x, y or z, and if new_lower is a tentative tighter
!                   lower bound on the same quantity, it is only accepted
!                   if
!
!                      new_lower >= lower + tol * MAX( 1, ABS( lower ) ),
!
!                   where
!
!                      tol = control%min_rel_improve.
!
!                   Similarly, a tentative tighter upper bound new_upper
!                   only replaces the current upper bound upper
!
!                      new_upper <= upper - tol * MAX( 1, ABS( upper ) ).
!
!                   Note that this parameter must exceed the machine
!                   precision significantly.
!                   Default: 10.**(-10) in double precision,
!                            10.**(-6)  in single precision.

         REAL ( KIND = wp ) :: max_growth_factor = TEN ** 8 ! INTENT( IN )

!                  The maximum growth factor (in absolute value) that is
!                  accepted between the maximum data item in the original
!                  problem  and any data item in the reduced problem.
!                  If a transformation results in this bound being exceeded,
!                  the transformation is skipped.
!                  Default : 10.**8 in double precision,
!                            10.**4 in single precision.

      END TYPE

!   Note that the default double precision settings correpond to the following
!   specfile, which is optionnaly read by PRESOLVE (see the description of
!   read_specfile below)

!      BEGIN PRESOLVE SPECIFICATIONS (DEFAULT)
!         printout-device                              6
!         error-printout device                        6
!         print-level                                  SILENT
!         presolve-termination-strategy                REDUCED_SIZE
!         maximum-number-of-transformations            1000000
!         maximum-number-of-passes                     25
!         constraints-accuracy                         1.0D-6
!         allow-dual-transformations
!         remove-redundant-variables-constraints
!         dual-variables-accuracy                      1.0D-6
!         primal-constraints-analysis-frequency        1
!         dual-constraints-analysis-frequency          1
!         singleton_columns-analysis-frequency         1
!         doubleton_columns-analysis-frequency         1
!         unconstrained-variables-analysis-frequency   1
!         dependent-variables-analysis-frequency       1
!         row-sparsification-frequency                 1
!         maximum-percentage-row-fill                  -1
!         transformations-buffer-size                  50000
!         transformations-file-device                  57
!         transformations-file-status                  KEEP
!         transformations-file-name                    transf.sav
!         primal-feasibility-check                     NONE
!         dual-feasibility-check                       NONE
!         active-multipliers-sign                      POSITIVE
!         inactive-multipliers-value                   LEAVE_AS_IS
!         active-dual-variables-sign                   POSITIVE
!         inactive-dual-variables-value                LEAVE_AS_IS
!         primal-variables-bound-status                TIGHTEST
!         dual-variables-bound-status                  TIGHTEST
!         constraints-bound-status                     TIGHTEST
!         multipliers-bound-status                     TIGHTEST
!         infinity-value                               1.0D19
!         pivoting-threshold                           1.10D-10
!         minimum-relative-bound-improvement           1.0D-10
!         maximum-growth-factor                        1.0D+8
!         compute-quadratic-value
!         compute-objective-constant
!         compute-gradient
!         compute-Hessian
!         compute-constraints-matrix
!         compute-primal-variables-values
!         compute-primal-variables-bounds
!         compute-dual-variables-values
!         compute-dual-variables-bounds
!         compute-constraints-values
!         compute-constraints-bounds
!         compute-multipliers-values
!         compute-multipliers-bounds
!      END PRESOLVE SPECIFICATIONS

!-------------------------------------------------------------------------------
!         The structure that return information on presolving to the user
!-------------------------------------------------------------------------------

      TYPE, PUBLIC :: PRESOLVE_inform_type

         INTEGER :: status = 0                 ! INTENT( OUT )

!                   The PRESOLVE exit condition.  It can take the following
!                   values:

!                    0 (OK)                    :
!
!                        successful exit;
!
!                    1 (MAX_NBR_TRANSF)        :
!
!                        the maximum number of problem transformation has been
!                        reached
!                        | NOTE:
!                        | this exit is not really an error, since the problem
!                        | can  nevertheless be permuted and  solved.  It merely
!                        | signals that further problem reduction could possibly
!                        | be obtained with a larger value of the parameter
!                        | control%max_nbr_transforms
!
!                   -21 (PRIMAL_INFEASIBLE)     :
!
!                        the problem is primal infeasible;
!
!                   -22 (DUAL_INFEASIBLE)       :
!
!                        the problem is dual infeasible;
!
!
!                   -23 (WRONG_G_DIMENSION)     :
!
!                        the dimension of the gradient is incompatible with
!                        the problem dimension;
!
!                   -24 (WRONG_HVAL_DIMENSION)  :
!
!                        the dimension of the vector containing the entries of
!                        the Hessian is erroneously specified;
!
!                   -25 (WRONG_HPTR_DIMENSION)  :
!
!                        the dimension of the vector containing the addresses
!                        of the first entry of each Hessian row is erroneously
!                        specified;
!
!                   -26 (WRONG_HCOL_DIMENSION)  :
!
!                        the dimension of the vector containing the column
!                        indices of the nonzero Hessian entries is erroneously
!                        specified;
!
!                   -27 (WRONG_HROW_DIMENSION)  :
!
!                        the dimension of the vector containing the row indices
!                        of the nonzero Hessian entries is erroneously
!                        specified;
!
!                   -28 (WRONG_AVAL_DIMENSION)  :
!
!                        the dimension of the vector containing the entries of
!                        the Jacobian is erroneously specified;
!
!                   -29 (WRONG_APTR_DIMENSION) :
!
!                        the dimension of the vector containing the addresses
!                        of the first entry of each Jacobian row is erroneously
!                        specified;
!
!                   -30 (WRONG_ACOL_DIMENSION) :
!
!                        the dimension of the vector containing the column
!                        indices of the nonzero Jacobian entries is erroneously
!                        specified;
!
!                   -31 (WRONG_AROW_DIMENSION) :
!
!                        the dimension of the vector containing the row indices
!                        of the nonzero Jacobian entries is erroneously
!                        specified;
!
!                   -32 (WRONG_X_DIMENSION)    :
!
!                        the dimension of the vector of variables is
!                        incompatible with the problem dimension;
!
!                   -33 (WRONG_XL_DIMENSION)   :
!
!                        the dimension of the vector of lower bounds on the
!                        variables is incompatible with the problem dimension;
!
!                   -34 (WRONG_XU_DIMENSION)   :
!
!                        the dimension of the vector of upper bounds on the
!                        variables is incompatible with the problem dimension;
!
!                   -35 (WRONG_Z_DIMENSION)    :
!
!                        the dimension of the vector of dual variables is
!                        incompatible with the problem dimension;
!
!                   -36 (WRONG_ZL_DIMENSION)   :
!
!                        the dimension of the vector of lower bounds on the dual
!                        variables is incompatible with the problem dimension;
!
!                   -37 (WRONG_ZU_DIMENSION)   :
!
!                        the dimension of the vector of upper bounds on the
!                        dual variables is incompatible with the problem
!                        dimension;
!
!                   -38 (WRONG_C_DIMENSION)    :
!
!                        the dimension of the vector of constraints values is
!                        incompatible with the problem dimension;
!
!                   -39 (WRONG_CL_DIMENSION)   :
!
!                        the dimension of the vector of lower bounds on the
!                        constraints is incompatible with the problem dimension;
!
!                   -40 (WRONG_CU_DIMENSION)   :
!
!                        the dimension of the vector of upper bounds on the
!                        constraints is incompatible with the problem dimension;
!
!                   -41 (WRONG_Y_DIMENSION)    :
!
!                        the dimension of the vector of multipliers values is
!                        incompatible with the problem dimension;
!
!                   -42 (WRONG_YL_DIMENSION)   :
!
!                        the dimension of the vector of lower bounds on the
!                        multipliers is incompatible with the problem dimension;
!
!                   -43 (WRONG_YU_DIMENSION)   :
!
!                        the dimension of the vector of upper bounds on the
!                        multipliers is incompatible with the problem dimension;
!
!                   -44 (STRUCTURE_NOT_SET)    :
!
!                        the problem structure has not been set or has been
!                        cleaned up before an attempt to analyze;
!
!                   -45 (PROBLEM_NOT_ANALYZED) :
!
!                        the problem has not been analyzed before an attempt to
!                        permute it;
!
!                   -46 (PROBLEM_NOT_PERMUTED) :
!
!                        the problem has not been permuted or fully reduced
!                        before an attempt to restore it
!
!                   -47 (H_MISSPECIFIED)       :
!
!                        the column indices of a row of the sparse Hessian are
!                        not in increasing order, in that they specify an entry
!                        above the diagonal;
!
!
!                   -48 (CORRUPTED_SAVE_FILE)  :
!
!                        one of the files containing saved problem
!                        transformations has been corrupted between  writing
!                        and reading;
!
!                   -49 (WRONG_XS_DIMENSION)
!
!                        the dimension of the vector of variables' status
!                        is incompatible with the problem dimension;
!
!                   -50 (WRONG_CS_DIMENSION)
!
!                        the dimension of the vector of constraints' status
!                        is incompatible with the problem dimension;
!
!                   -52 (WRONG_N)              :
!
!                        the problem does not contain any (active) variable;
!
!                   -53 (WRONG_M)              :
!
!                        the problem contains a negative number of constraints;
!
!                   -54 (SORT_TOO_LONG)        :
!
!                        the vectors are too long for the sorting routine;
!
!                   -55 (X_OUT_OF_BOUNDS)
!
!                        the value of a variable that is obtained by
!                        substitution from a constraint is incoherent with the
!                        variable's bounds.  This may be due to a relatively
!                        loose accuracy on the linear constraints. Try to
!                        increase control%c_accuracy.

!                   -56 (X_NOT_FEASIBLE)
!
!                        the value of a constraint that is obtained by
!                        recomputing its value on input of RESTORE from the
!                        current x is incompatible with its declared value
!                        or its bounds. This may caused the restored problem
!                        to be infeasible.

!                   -57 (Z_NOT_FEASIBLE)
!
!                        the value of a dual variable that is obtained by
!                        recomputing its value on input of RESTORE (assuming
!                        dual feasibility) from the current values of
!                        ( x, y, z ) is incompatible with its declared value.
!                        This may caused the restored problem to be infeasible
!                        or suboptimal.

!                   -58 (Z_CANNOT_BE_ZEROED)
!
!                        a dual variable whose value is nonzero because the
!                        corresponding primal is at an artificial bound cannot
!                        be zeroed while maintaining dual feasibility
!                        (at RESTORE). This can happen when ( x, y, z ) on
!                        input of RESTORE are not (sufficiently) optimal.

!                   -1 (MEMORY_FULL)           :
!
!                        memory allocation failed

!                   -2 (FILE_NOT_OPENED)      :
!
!                        a file intended for saving problem transformations
!                        could not be opened;

!                   -3 (COULD_NOT_WRITE)      :
!
!                        an IO error occurred while saving transformations on
!                        the relevant disk file;
!
!                   -4 (TOO_FEW_BITS_PER_BYTE)
!
!                        an integer contains less than NBRH + 1 bits.

!                   -60 (UNRECOGNIZED_KEYWORD)

!                       a keyword was not recognized in the analysis of the
!                       specification file

!                   -61 (UNRECOGNIZED_VALUE)

!                       a value was not recognized in the analysis of the
!                       specification file

!                   -63 (G_NOT_ALLOCATED)

!                       the vector prob%G has not been allocated although it
!                       has general values

!                   -64 (C_NOT_ALLOCATED)

!                       the vector prob%C has not been allocated although
!                       prob%m > 0

!                   -65 (AVAL_NOT_ALLOCATED)

!                       the vector prob%A%val has not been allocated although
!                       prob%m > 0

!                   -66 (APTR_NOT_ALLOCATED)

!                       the vector prob%A%ptr has not been allocated although
!                       prob%m > 0 and A is stored in row-wise sparse format

!                   -67 (ACOL_NOT_ALLOCATED)

!                       the vector prob%A%col has not been allocated although
!                       prob%m > 0 and A is stored in row-wise sparse format
!                       or sparse coordinate format

!                   -68 (AROW_NOT_ALLOCATED)

!                       the vector prob%A%row has not been allocated although
!                       prob%m > 0 and A is stored in sparse coordinate format

!                   -69 (HVAL_NOT_ALLOCATED)

!                       the vector prob%H%val has not been allocated although
!                       prob%H%ne > 0

!                   -70 (HPTR_NOT_ALLOCATED)

!                       the vector prob%H%ptr has not been allocated although
!                       prob%H%ne > 0 and H is stored in row-wise sparse format

!                   -71 (HCOL_NOT_ALLOCATED)

!                       the vector prob%H%col has not been allocated although
!                       prob%H%ne > 0 and H is stored in row-wise sparse format
!                       or sparse coordinate format

!                   -72 (HROW_NOT_ALLOCATED)

!                       the vector prob%H%row has not been allocated although
!                       prob%H%ne > 0 and A is stored in sparse coordinate
!                       format

!                   -73 (WRONG_ANE)

!                       incompatible value of prob%A_ne

!                   -74 (WRONG_HNE)

!                       incompatible value of prob%H_ne


         INTEGER :: nbr_transforms = 0               ! INTENT( OUT )

!                   The final number of problem transformations, as reported
!                   to the user at exit.

         CHARACTER( LEN = 80 ), DIMENSION( 3 ) ::                              &
                   message = REPEAT( ' ', 80 )       ! INTENT( OUT )

!                   A few lines containing a description of the exit condition
!                   on exit of PRESOLVE, typically including more information
!                   than indicated in the description of control%status above.
!                   It is printed out on device errout at the end of execution
!                   if control%print_level >= TRACE.

      END TYPE

!-------------------------------------------------------------------------------
!         The structure that saves information between the various calls
!         to presolve
!-------------------------------------------------------------------------------

!     NOTE:  This structure is used for purely internal purposes.  Thus the
!     ====   arguments (of type PRESOLVE_data_type) should not be initialized
!            nor modified by the user before or in between calls to PRESOLVE.

      TYPE, PUBLIC :: PRESOLVE_data_type

!        ---------------------------------------------
!        Problem dimensions and characteristics
!        ---------------------------------------------

         INTEGER :: m_original          ! the original number of constraints

         INTEGER :: n_original          ! the original number of variables

         INTEGER :: a_ne_original       ! the original number of elements in A

         INTEGER :: h_ne_original       ! the original number of elements in H

         INTEGER :: a_type              ! the type of A

         INTEGER :: h_type              ! the type of H

         INTEGER :: a_type_original     ! the original type of A

         INTEGER :: h_type_original     ! the original type of H

         INTEGER :: m_active            ! the number of currently active
                                        ! constraints

         INTEGER :: m_eq_active         ! the number of currently active
                                        ! equality constraints

         INTEGER :: n_active            ! the number of currently active
                                        ! variables

         INTEGER :: a_ne_active         ! the number of currently active
                                        ! elements in A

         INTEGER :: h_ne_active         ! the number of currently active
                                        ! elements in the lower triangular
                                        ! part of H

         INTEGER :: n_in_prob           ! the number of variables in the
                                        ! problem (as seen during the last
                                        ! call to PRESOLVE)

         INTEGER :: m_in_prob           ! the number of constraints in the
                                        !problem

!        ---------------------------------------------
!        Various presolving parameters
!        ---------------------------------------------

         INTEGER :: out                 ! the printing device number

         INTEGER :: level               ! the level of printout (globally)

         INTEGER :: lsc_f               ! the index of the 1rst linear
                                        ! singleton column

         INTEGER :: ldc_f               ! the index of the 1rst linear
                                        ! doubleton column

         INTEGER :: unc_f               ! the index if the first
                                        ! unconstrained variable

         INTEGER :: lfx_f               ! the index if the 1rst "last minute"
                                        ! fixed vars

         INTEGER :: recl                ! the record length for saving
                                        ! transformations

         INTEGER :: icheck1             ! the first integer checksum for
                                        ! transformation files

         INTEGER :: icheck2             ! the second integer checksum for
                                        ! transformation files

         INTEGER :: icheck3             ! the third integer checksum for
                                        ! transformation files

         INTEGER :: npass               ! the index of the current presolving
                                        ! pass

         INTEGER :: tm                  ! the number of transformations
                                        ! currently in memory

         INTEGER :: tt                  ! the total number of transformations
                                        ! so far

         INTEGER :: ts                  ! the number of saved transformations

         INTEGER :: max_tm              ! the maximum number of transformations
                                        ! that can be held in memory (the
                                        ! transf buffer size)

         INTEGER :: rts                 ! the number of reapplied saved
                                        ! transformations

         INTEGER :: rtm                 ! the number of reapplied
                                        ! transformations from memory

         INTEGER :: needs( 6, 10 )      ! the matrix of output dependence
                                        ! needs( i, j ) gives the index of the
                                        ! first transformation where the value
                                        ! of output j is used to define that
                                        ! of output i

         INTEGER :: stage               ! the current stage in the presolving
                                        ! process

         INTEGER :: loop                ! the index of the current presolving
                                        ! loop

         INTEGER :: hindex              ! the index of the current heuristic
                                        ! being applied in the presolving loop

         INTEGER :: nmods( NBRH )       ! the number of row and columns
                                        ! modified by other heuristics than
                                        ! the current one since the last pass
                                        ! in the current heuristic

         INTEGER :: maxmn               ! MAX( prob%n, prob%m )


         REAL ( KIND = wp ) :: a_max    ! the maximal element of A in
                                        ! absolute value

         REAL ( KIND = wp ) :: h_max    ! the maximal element of H in absolute
                                        ! value

         REAL ( KIND = wp ) :: x_max    ! the maximal bound on x in
                                        ! absolute value

         REAL ( KIND = wp ) :: z_max    ! the maximal bound on z in
                                        ! absolute value

         REAL ( KIND = wp ) :: c_max    ! the maximal bound on c in
                                        ! absolute value

         REAL ( KIND = wp ) :: y_max    ! the maximal bound on y in
                                        ! absolute value

         REAL ( KIND = wp ) :: g_max    ! the maximal element of g in
                                        ! absolute value

         REAL ( KIND = wp ) :: a_tol    ! the pivoting threshold for
                                        ! eliminations and  bounds in A

         REAL ( KIND = wp ) :: h_tol    ! the pivoting threshold for
                                        !bounds in H

         REAL ( KIND = wp ) :: rcheck   ! the real checksum for
                                        ! transformations files

         REAL ( KIND = wp ) :: max_fill_prop ! the maximal fill proportion
                                        ! for merging  operations in A

         REAL ( KIND = wp ) :: mrbi     ! the current minimum relative
                                        ! bound improvement threshold,

         REAL ( KIND = wp ) :: max_growth ! the maximal size for data in
                                        ! the reduced problem

         REAL ( KIND = wp ) :: ACCURACY ! the problem dependent precision
                                        ! value

         REAL ( KIND = wp ) :: INFINITY ! the value corresponding to plus
                                        ! infinity

         REAL ( KIND = wp ) :: P_INFINITY ! the threshold above which a value
                                        ! is deemed to be equal to plus infinity

         REAL ( KIND = wp ) :: M_INFINITY ! the threshold below which a value
                                        ! is deemed to be equal to minus
                                        ! infinity

!        ----------------------------------------------
!        The value of the controls at the previous call
!        ----------------------------------------------

         TYPE ( PRESOLVE_control_type ) ::  prev_control ! the value of control
                                        ! for the previous execution of PRESOLVE

!        ---------------------------------------------
!        Pointer arrays defined globally with PRESOLVE
!        ---------------------------------------------

         INTEGER, ALLOCATABLE, DIMENSION( : ) ::  A_col_f
                                        ! A_col_f( j ) is the position in A
                                        ! of the first element in column j of A

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_col_n
                                        ! A_col_n( k ) is the position in A of
                                        ! the next element in the same column
                                        ! as that in  position k

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row
                                        ! the row index of the element in
                                        ! position k in A

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_col_s
                                        ! the sizes of the columns of A

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row_s
                                        ! the sizes of the rows of H

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: conc
                                        ! conc( i ) is the index of the row
                                        ! of A that is to be concatenated
                                        ! to row i of A, if any

         INTEGER, ALLOCATABLE, DIMENSION( : ) ::  a_perm
                                        ! a work vector of dimension
                                        ! proportional to MAX( size of A, m, n )
                                        ! NOTE: this vector is used in order
                                        !       to hold the permutation of A,
                                        !       but also, during the analysis,
                                        !       to hold the flags (marks) for
                                        !       the different heuristics.

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_str
                                        ! the structure of the each row of H:
                                        ! > 0        : the row is diagonal
                                        !              with diagonal element
                                        !              in position H_str( j )
                                        !              in H
                                        ! 0 ( EMPTY ): the row is empty
                                        ! < 0        : the row has - H_str( j )
                                        !              nonzero off-diagonal
                                        !              elements

         INTEGER, ALLOCATABLE, DIMENSION( : ) ::  H_col_f
                                        ! H_col_f( j ) is the position in A of
                                        ! the first element in column j of H
                                        ! (below the diagonal)

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_col_n
                                        ! H_col_n( k ) is the position in A
                                        ! of the next element in the same
                                        ! column (below the diagonal) as that
                                        ! in position k

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_row
                                        ! the row index of element k in H

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: h_perm
                                        ! a work vector of dimension
                                        ! proportional to the size of H
                                        ! NOTE: this vector is used in order
                                        !       to hold the permutation of H,
                                        !       but also, during the analysis,
                                        !       to hold the linked lists for
                                        !       linear singleton, linear
                                        !       doubleton, unconstrained
                                        !       columns and variables
                                        !       fixed by forcing constraints

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: w_n
                                        ! a work vector of size n

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: w_m
                                        ! a work vector of size m + 1

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: w_mn
                                        ! a work vector of size max( m, n + 1 )

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: hist_type
                                        ! the types of the successive
                                        ! transformations

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: hist_i
                                        ! the first integer characteristic
                                        ! of the transformations

         INTEGER, ALLOCATABLE, DIMENSION( : ) :: hist_j
                                        ! the second integer characteristic
                                        ! of the transformations


         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: hist_r
                                        ! the first real characteristic
                                        ! of the transformations

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x_l2
                                        ! the vector of non-degenerate lower
                                        ! bounds on x (only used if
                                        ! final_x_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: x_u2
                                        ! the vector of non-degenerate upper
                                        !  bounds on x (only used if
                                        ! final_x_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: z_l2
                                        ! the vector of non-degenerate lower
                                        ! bounds on z (only used if
                                        ! final_z_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: z_u2
                                        ! the vector of non-degenerate upper
                                        ! bounds on z (only used if
                                        ! final_z_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: c_l2
                                        ! the vector of non-degenerate lower
                                        ! bounds on c (only used if
                                        ! final_c_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: c_u2
                                        ! the vector of non-degenerate upper
                                        ! bounds on c (only used if
                                        ! final_c_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: y_l2
                                        ! the vector of non-degenerate lower
                                        ! bounds on y (only used if
                                        ! final_y_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: y_u2
                                        ! the vector of non-degenerate upper
                                        ! bounds on y (only used if
                                        ! final_y_bounds /= TIGHTEST)

         REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ztmp
                                        ! a workspace vector of size equal
                                        ! to the dimension of the reduced
                                        ! problem (only used if
                                        ! control%check_dual_feasibility>=BASIC)

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

!  Local parameters
!
!     Presolving (old) modes

      INTEGER, PRIVATE, PARAMETER :: ANALYZE                   = 2
      INTEGER, PRIVATE, PARAMETER :: PERMUTE                   = 3
      INTEGER, PRIVATE, PARAMETER :: RESTORE                   = 4
      INTEGER, PRIVATE, PARAMETER :: TERMINATE                 = 5


!     Presolving heuristics identifiers
!     Note: they must be consecutive from 1 to NBRH.

      INTEGER, PRIVATE, PARAMETER :: EMPTY_AND_SINGLETON_ROWS  = 1
      INTEGER, PRIVATE, PARAMETER :: SPECIAL_LINEAR_COLUMNS    = 2
      INTEGER, PRIVATE, PARAMETER :: PRIMAL_CONSTRAINTS        = 3
      INTEGER, PRIVATE, PARAMETER :: DUAL_CONSTRAINTS          = 4
      INTEGER, PRIVATE, PARAMETER :: DEPENDENT_VARIABLES       = 5
      INTEGER, PRIVATE, PARAMETER :: ROW_SPARSIFICATION        = 6
      INTEGER, PRIVATE, PARAMETER :: REDUNDANT_VARIABLES       = 7
      INTEGER, PRIVATE, PARAMETER :: CHECK_BOUNDS_CONSISTENCY  = NBRH

!     Symbolic name for the 'cancelled entry' flag

      INTEGER, PRIVATE, PARAMETER :: CANCELLED                 = NBRH + 1

!     Possible map and problem stages

      INTEGER, PRIVATE, PARAMETER :: VOID                      =  -1
      INTEGER, PRIVATE, PARAMETER :: READY                     =   1
      INTEGER, PRIVATE, PARAMETER :: ANALYZED                  =   2
      INTEGER, PRIVATE, PARAMETER :: FULLY_REDUCED             =   3
      INTEGER, PRIVATE, PARAMETER :: PERMUTED                  =   4
      INTEGER, PRIVATE, PARAMETER :: RESTORED                  =   5

!     Problem transformation types

      INTEGER, PRIVATE, PARAMETER :: X_LOWER_UPDATED           =   1
      INTEGER, PRIVATE, PARAMETER :: X_UPPER_UPDATED           =   2
      INTEGER, PRIVATE, PARAMETER :: X_LOWER_UPDATED_P         =   3
      INTEGER, PRIVATE, PARAMETER :: X_UPPER_UPDATED_P         =   4
      INTEGER, PRIVATE, PARAMETER :: X_LOWER_UPDATED_D         =   5
      INTEGER, PRIVATE, PARAMETER :: X_UPPER_UPDATED_D         =   6
      INTEGER, PRIVATE, PARAMETER :: X_LOWER_UPDATED_S         =   7
      INTEGER, PRIVATE, PARAMETER :: X_UPPER_UPDATED_S         =   8
      INTEGER, PRIVATE, PARAMETER :: X_FIXED_DF                =   9
      INTEGER, PRIVATE, PARAMETER :: X_FIXED_SL                =  10
      INTEGER, PRIVATE, PARAMETER :: X_FIXED_SU                =  11
      INTEGER, PRIVATE, PARAMETER :: X_FIXED_ZV                =  12
      INTEGER, PRIVATE, PARAMETER :: X_MERGE                   =  13
      INTEGER, PRIVATE, PARAMETER :: X_IGNORED                 =  14
      INTEGER, PRIVATE, PARAMETER :: X_SUBSTITUTED             =  15
      INTEGER, PRIVATE, PARAMETER :: X_BOUNDS_TO_C             =  16
      INTEGER, PRIVATE, PARAMETER :: X_REDUCTION               =  36
      INTEGER, PRIVATE, PARAMETER :: Z_LOWER_UPDATED           =  17
      INTEGER, PRIVATE, PARAMETER :: Z_UPPER_UPDATED           =  18
      INTEGER, PRIVATE, PARAMETER :: Z_FIXED                   =  19
      INTEGER, PRIVATE, PARAMETER :: C_LOWER_UPDATED           =  20
      INTEGER, PRIVATE, PARAMETER :: C_UPPER_UPDATED           =  21
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_FL              =  22
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_FU              =  23
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_YV              =  24
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_YZ_LOW          =  25
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_YZ_UP           =  26
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_YZ_EQU          =  27
      INTEGER, PRIVATE, PARAMETER :: C_REMOVED_GY              =  28
      INTEGER, PRIVATE, PARAMETER :: Y_LOWER_UPDATED           =  29
      INTEGER, PRIVATE, PARAMETER :: Y_UPPER_UPDATED           =  30
      INTEGER, PRIVATE, PARAMETER :: Y_FIXED                   =  31
      INTEGER, PRIVATE, PARAMETER :: A_ROWS_COMBINED           =  32
      INTEGER, PRIVATE, PARAMETER :: A_ROWS_MERGED             =  33
      INTEGER, PRIVATE, PARAMETER :: A_ENTRY_REMOVED           =  34
      INTEGER, PRIVATE, PARAMETER :: H_ELIMINATION             =  35

!     Parameter names for logicals restoration links

      INTEGER, PRIVATE, PARAMETER :: X_VALS                    =  1
      INTEGER, PRIVATE, PARAMETER :: X_BNDS                    =  7
      INTEGER, PRIVATE, PARAMETER :: Z_VALS                    =  2
      INTEGER, PRIVATE, PARAMETER :: Z_BNDS                    =  8
      INTEGER, PRIVATE, PARAMETER :: C_BNDS                    =  3
      INTEGER, PRIVATE, PARAMETER :: Y_VALS                    =  4
      INTEGER, PRIVATE, PARAMETER :: F_VAL                     =  5
      INTEGER, PRIVATE, PARAMETER :: G_VALS                    =  6
      INTEGER, PRIVATE, PARAMETER :: H_VALS                    =  9
      INTEGER, PRIVATE, PARAMETER :: A_VALS                    = 10

!     Bound updating values

      INTEGER, PRIVATE, PARAMETER :: TIGHTEN                   =  2
      INTEGER, PRIVATE, PARAMETER :: SET                       =  4
      INTEGER, PRIVATE, PARAMETER :: UPDATE                    =  6

!     Methods to recover the dual variables and multipliers at RESTORE

      INTEGER, PRIVATE, PARAMETER :: Z_FROM_DUAL_FEAS          =  0
      INTEGER, PRIVATE, PARAMETER :: Z_FROM_YZ_LOW             =  1
      INTEGER, PRIVATE, PARAMETER :: Z_FROM_YZ_UP              =  2
      INTEGER, PRIVATE, PARAMETER :: Z_FROM_YZ_EQU             =  3
      INTEGER, PRIVATE, PARAMETER :: Z_GIVEN                   =  7
      INTEGER, PRIVATE, PARAMETER :: Y_FROM_Z_LOW              =  8
      INTEGER, PRIVATE, PARAMETER :: Y_FROM_Z_UP               =  9
      INTEGER, PRIVATE, PARAMETER :: Y_FROM_Z_BOTH             = 10
      INTEGER, PRIVATE, PARAMETER :: Y_FROM_GY                 = 11
      INTEGER, PRIVATE, PARAMETER :: Y_GIVEN                   = 12
      INTEGER, PRIVATE, PARAMETER :: Y_FROM_FORCING_LOW        = 13
      INTEGER, PRIVATE, PARAMETER :: Y_FROM_FORCING_UP         = 14

!     Exit conditions specific to presolve (outside the [-20,0] range)

      INTEGER, PRIVATE, PARAMETER :: MAX_NBR_TRANSF            =   1
      INTEGER, PRIVATE, PARAMETER :: PRIMAL_INFEASIBLE         = -21
      INTEGER, PRIVATE, PARAMETER :: DUAL_INFEASIBLE           = -22
      INTEGER, PRIVATE, PARAMETER :: WRONG_G_DIMENSION         = -23
      INTEGER, PRIVATE, PARAMETER :: WRONG_HVAL_DIMENSION      = -24
      INTEGER, PRIVATE, PARAMETER :: WRONG_HPTR_DIMENSION      = -25
      INTEGER, PRIVATE, PARAMETER :: WRONG_HCOL_DIMENSION      = -26
      INTEGER, PRIVATE, PARAMETER :: WRONG_HROW_DIMENSION      = -27
      INTEGER, PRIVATE, PARAMETER :: WRONG_AVAL_DIMENSION      = -28
      INTEGER, PRIVATE, PARAMETER :: WRONG_APTR_DIMENSION      = -29
      INTEGER, PRIVATE, PARAMETER :: WRONG_ACOL_DIMENSION      = -30
      INTEGER, PRIVATE, PARAMETER :: WRONG_AROW_DIMENSION      = -31
      INTEGER, PRIVATE, PARAMETER :: WRONG_X_DIMENSION         = -32
      INTEGER, PRIVATE, PARAMETER :: WRONG_XL_DIMENSION        = -33
      INTEGER, PRIVATE, PARAMETER :: WRONG_XU_DIMENSION        = -34
      INTEGER, PRIVATE, PARAMETER :: WRONG_Z_DIMENSION         = -35
      INTEGER, PRIVATE, PARAMETER :: WRONG_ZL_DIMENSION        = -36
      INTEGER, PRIVATE, PARAMETER :: WRONG_ZU_DIMENSION        = -37
      INTEGER, PRIVATE, PARAMETER :: WRONG_C_DIMENSION         = -38
      INTEGER, PRIVATE, PARAMETER :: WRONG_CL_DIMENSION        = -39
      INTEGER, PRIVATE, PARAMETER :: WRONG_CU_DIMENSION        = -40
      INTEGER, PRIVATE, PARAMETER :: WRONG_Y_DIMENSION         = -41
      INTEGER, PRIVATE, PARAMETER :: WRONG_YL_DIMENSION        = -42
      INTEGER, PRIVATE, PARAMETER :: WRONG_YU_DIMENSION        = -43
      INTEGER, PRIVATE, PARAMETER :: STRUCTURE_NOT_SET         = -44
      INTEGER, PRIVATE, PARAMETER :: PROBLEM_NOT_ANALYZED      = -45
      INTEGER, PRIVATE, PARAMETER :: PROBLEM_NOT_PERMUTED      = -46
      INTEGER, PRIVATE, PARAMETER :: H_MISSPECIFIED            = -47
      INTEGER, PRIVATE, PARAMETER :: CORRUPTED_SAVE_FILE       = -48
      INTEGER, PRIVATE, PARAMETER :: WRONG_XS_DIMENSION        = -49
      INTEGER, PRIVATE, PARAMETER :: WRONG_CS_DIMENSION        = -50
      INTEGER, PRIVATE, PARAMETER :: WRONG_GLOBAL_SETTINGS     = -52
      INTEGER, PRIVATE, PARAMETER :: WRONG_N                   = -53
      INTEGER, PRIVATE, PARAMETER :: WRONG_M                   = -54
      INTEGER, PRIVATE, PARAMETER :: SORT_TOO_LONG             = -55
      INTEGER, PRIVATE, PARAMETER :: X_OUT_OF_BOUNDS           = -56
      INTEGER, PRIVATE, PARAMETER :: X_NOT_FEASIBLE            = -57
      INTEGER, PRIVATE, PARAMETER :: Z_NOT_FEASIBLE            = -58
      INTEGER, PRIVATE, PARAMETER :: Z_CANNOT_BE_ZEROED        = -59
      INTEGER, PRIVATE, PARAMETER :: UNRECOGNIZED_KEYWORD      = -60
      INTEGER, PRIVATE, PARAMETER :: UNRECOGNIZED_VALUE        = -61
      INTEGER, PRIVATE, PARAMETER :: G_NOT_ALLOCATED           = -63
      INTEGER, PRIVATE, PARAMETER :: AVAL_NOT_ALLOCATED        = -65
      INTEGER, PRIVATE, PARAMETER :: APTR_NOT_ALLOCATED        = -66
      INTEGER, PRIVATE, PARAMETER :: ACOL_NOT_ALLOCATED        = -67
      INTEGER, PRIVATE, PARAMETER :: AROW_NOT_ALLOCATED        = -68
      INTEGER, PRIVATE, PARAMETER :: HVAL_NOT_ALLOCATED        = -69
      INTEGER, PRIVATE, PARAMETER :: HPTR_NOT_ALLOCATED        = -70
      INTEGER, PRIVATE, PARAMETER :: HCOL_NOT_ALLOCATED        = -71
      INTEGER, PRIVATE, PARAMETER :: HROW_NOT_ALLOCATED        = -72
      INTEGER, PRIVATE, PARAMETER :: WRONG_ANE                 = -73
      INTEGER, PRIVATE, PARAMETER :: WRONG_HNE                 = -74

!     Internal error indicators

      INTEGER, PRIVATE, PARAMETER :: NO_DOUBLETON_ENTRIES      = -1000
      INTEGER, PRIVATE, PARAMETER :: NO_SINGLETON_ENTRY        = -1001
      INTEGER, PRIVATE, PARAMETER :: ERRONEOUS_EMPTY_COL       = -1002
      INTEGER, PRIVATE, PARAMETER :: CORRUPTED_MAP             = -1003
      INTEGER, PRIVATE, PARAMETER :: WRONG_MAP                 = -1004
      INTEGER, PRIVATE, PARAMETER :: MAX_NBR_TRANSF_TMP        = -1005
      INTEGER, PRIVATE, PARAMETER :: NO_DOUBLETON_ROW          = -1006
      INTEGER, PRIVATE, PARAMETER :: WRONG_A_COUNT             = -1007
      INTEGER, PRIVATE, PARAMETER :: NO_SINGLE_OFFDIAGONAL     = -1008

!     H row structure indicator

      INTEGER, PRIVATE, PARAMETER :: EMPTY                     =  0

!     End of pointer lists

      INTEGER, PRIVATE, PARAMETER :: END_OF_LIST               = -1

!----------------------
!   G l o b a l s
!----------------------

      REAL( KIND = wp ) :: ACCURACY, MAX_GROWTH, M_INFINITY, P_INFINITY

!==============================================================================

!                        Define the presolving loop

!==============================================================================

!     The presolving loop is defined as a sequence of applications of
!     heuristics to the problem.  Each heuristic has its indentifier,
!     and the loop is defined by the vector SEQUENCE that contain the
!     list of the successive heuristics.

!     The number of heuristics applied within a single presolving loop

      INTEGER, PRIVATE, PARAMETER :: N_HEURISTICS = 8

!     The sequencing of the heuristics in the presolving loop

!     The constraints on the sequencing are as follows:
!     1) EMPTY_AND_SINGLETON_ROWS must occur immediately before
!        SPECIAL_LINEAR_COLUMNS, because of the possible occurence of
!        empty or singleton rows that modify the bounds.

      INTEGER, PRIVATE, DIMENSION( N_HEURISTICS ), PARAMETER ::                &
!
         PRESOLVING_SEQUENCE = (/                                              &
                                    EMPTY_AND_SINGLETON_ROWS,                  &
                                    SPECIAL_LINEAR_COLUMNS,                    &
                                    DUAL_CONSTRAINTS,                          &
                                    EMPTY_AND_SINGLETON_ROWS,                  &
                                    DEPENDENT_VARIABLES,                       &
                                    PRIMAL_CONSTRAINTS,                        &
                                    ROW_SPARSIFICATION,                        &
                                    CHECK_BOUNDS_CONSISTENCY                   &
                                                                /)

!     Note that N_HEURISTICS must be equal to the dimension of
!     PRESOLVING_SEQUENCE.

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

   CONTAINS ! The whole of the PRESOLVE code

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

      SUBROUTINE PRESOLVE_initialize( control, inform, s )

!     Checks the problem characteristics, verifies the user defined presolving
!     parameters and initialize the necessary data structures.

!     Arguments:

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER :: dim = 1
      INTEGER :: srecl
!     INTEGER * 4 :: srecl

!-------------------------------------------------------------------------------
!
!                        Initial definitions
!
!-------------------------------------------------------------------------------

!     Printing
!     NOTE :  if the output device or amount of printing must be modified for
!             the execution of INITIALIZE, they must be reset in the following
!             two lines and the module recompiled.

      s%out   = DEF_WRITE_UNIT         !  printout device
      s%level = DEF_PRINT_LEVEL        !  amount of printout

!     Print banner

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *         GALAHAD presolve for QPs         *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *         structure initialisation         *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' '
         IF ( s%level  >= DEBUG ) WRITE( s%out, * )                            &
              ' initial verifications and workspace allocation'
      END IF

!     Initialize the exit status and the exit message to that corresponding
!     to a successful exit, unless the dimensions of the problem are silly.

      inform%message( 1 ) = ''
      inform%message( 2 ) = ''
      inform%message( 3 ) = ''
      inform%nbr_transforms = 0
      inform%status = OK
      WRITE( inform%message( 1 ), * ) ' PRESOLVE: successful exit'

!     Verify that the computer supports at least NBRH + 1 flags packed into
!     a single integer.

      IF ( BIT_SIZE( dim ) < NBRH + 1 ) THEN
         inform%status = TOO_FEW_BITS_PER_BYTE
         WRITE( inform%message( 1 ), * )                                       &
             ' PRESOLVE ERROR: this &%#!!@* computer has less than',           &
             NBRH + 1, 'bits in an integer!!'
         RETURN
      END IF

!     Global accuracy.  Note that this value is recomputed by PRESOLVE_apply to
!     better reflect the problem's characteristics.

     s%ACCURACY = TEN * TEN * EPSMACH

!-------------------------------------------------------------------------------
!
!                       Define the control parameters
!
!-------------------------------------------------------------------------------

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   defining presolve default control parameters'

!     Printout device

      s%prev_control%out    = s%out
      s%prev_control%errout = s%out

!     Print level

      s%prev_control%print_level = control%print_level
      IF ( s%level >= DEBUG ) THEN
         SELECT CASE ( s%level )
         CASE ( SILENT )
         CASE ( TRACE  )
            WRITE( s%out, * ) '    print level is TRACE'
         CASE ( ACTION )
            WRITE( s%out, * ) '    print level is ACTION'
         CASE ( DETAILS )
            WRITE( s%out, * ) '    print level is DETAILS'
         CASE ( DEBUG )
            WRITE( s%out, * ) '    print level is DEBUG'
         CASE ( CRAZY: )
            WRITE( s%out, * ) '    print level is CRAZY'
         END SELECT
      END IF

!     Determine the record length for saving transformations on disk
!     files whenever necessary.

      IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    mode is INITIALIZE'
      INQUIRE ( IOLENGTH = srecl ) s%level, s%level, s%level, s%a_max, s%a_max
      s%recl = srecl

!    The value of INFINITY

      s%prev_control%infinity = control%infinity
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
            '    infinity value set to', control%infinity

!     Maximum number of analysis passes

      s%prev_control%max_nbr_passes = control%max_nbr_passes
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    maximum number of analysis passes set to', control%max_nbr_passes

!     Maximum number of problem transformations

      s%prev_control%max_nbr_transforms = control%max_nbr_transforms
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    maximum number of transformations set to',                       &
         control%max_nbr_transforms

!     Size of the transformation buffer in memory

      s%prev_control%transf_buffer_size = control%transf_buffer_size
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    size of transformation buffer set to',                           &
         control%transf_buffer_size

!     Name of the file used to store the problem transformations
!     on disk

      s%prev_control%transf_file_name = control%transf_file_name
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    file to save problem transformations set to ',                   &
              TRIM( control%transf_file_name )

!     Unit number for writing/reading the transformation file

      s%prev_control%transf_file_nbr = control%transf_file_nbr
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    unit number for transformations saving set to ',                 &
         control%transf_file_nbr

!     KEEP or DELETE status for the transformation files

      s%prev_control%transf_file_status = control%transf_file_status
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    final status for transformation files set to KEEP'

!     Maximum percentage of row-wise fill in A

      s%prev_control%max_fill = control%max_fill
      IF ( s%level >= DEBUG ) THEN
         IF ( control%max_fill >= 0 ) THEN
            WRITE( s%out, * )'    maximum percentage of fill in a row set to', &
                 control%max_fill, '%'
         ELSE
            WRITE( s%out, * )'    maximum percentage of fill in a row set to', &
                 ' infinite'
         END IF
      END IF

!     Tolerance for pivoting in A

      IF ( wp == sp ) control%pivot_tol = DEF_PIVOT_TOL_sp
      s%prev_control%pivot_tol = control%pivot_tol
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    relative tolerance for pivoting in A set to', control%pivot_tol

!     Minimal relative bound improvement

      IF ( wp == sp ) control%min_rel_improve = DEF_MRBI_sp
      s%prev_control%min_rel_improve = control%min_rel_improve
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    minimum relative bound improvement set to',                      &
         control%min_rel_improve

!     Maximum growth factor between original and reduce problems

      IF ( wp == sp ) control%max_growth_factor = DEF_MAX_GROWTH_sp
      s%prev_control%max_growth_factor = control%max_growth_factor
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    maximum growth factor set to', control%max_growth_factor

!     Relative accuracy on the linear constraints

      IF ( wp == sp ) control%c_accuracy = DEF_C_ACC_sp
      s%prev_control%c_accuracy = control%c_accuracy
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    relative accuracy for linear constraints set to',                &
         control%c_accuracy

!     Relative accuracy on the dual variables

      IF ( wp == sp ) control%z_accuracy = DEF_Z_ACC_sp
      s%prev_control%z_accuracy = control%z_accuracy
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    relative accuracy for dual variables set to', control%z_accuracy

!     Level of primal feasibility check at RESTORE

      s%prev_control%check_primal_feasibility = control%check_primal_feasibility
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    level of primal feasibility check at RESTORE set to NONE'

!     Level of dual feasibility check at RESTORE

      s%prev_control%check_dual_feasibility = control%check_dual_feasibility
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    level of dual feasibility check at RESTORE set to NONE'

!     The frequency of the various preprocessing actions

!     1) analysis of the primal constraints

      s%prev_control%primal_constraints_freq = control%primal_constraints_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of primal constraints analysis set to',                &
         control%primal_constraints_freq

!     2) analysis of the dual constraints

      s%prev_control%dual_constraints_freq = control%dual_constraints_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of dual constraints analysis set to',                  &
         control%dual_constraints_freq

!     3) analysis of the singleton columns

      s%prev_control%singleton_columns_freq = control%singleton_columns_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of singleton columns analysis set to',                 &
         control%singleton_columns_freq

!     4) analysis of the doubleton columns

      s%prev_control%doubleton_columns_freq = control%doubleton_columns_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of doubleton columns analysis set to',                 &
         control%doubleton_columns_freq

!     5) analysis of the linearly unconstrained variables

      s%prev_control%unc_variables_freq = control%unc_variables_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of unconstrained variables analysis set to',           &
         control%unc_variables_freq

!     6) analysis of the linearly dependent variables

      s%prev_control%dependent_variables_freq = control%dependent_variables_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of dependent variables analysis set to',               &
         control%dependent_variables_freq

!     7) row sparsification frequency

      s%prev_control%sparsify_rows_freq = control%sparsify_rows_freq
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    frequency of row sparsification analysis set to',                &
         control%sparsify_rows_freq

!     Strategy for presolve termination

      s%prev_control%termination = control%termination
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    strategy for presolve termination set to REDUCED_SIZE'

!     Sign convention for the multipliers

      s%prev_control%y_sign  = control%y_sign
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    sign convention for the multipliers set to POSITIVE'

!     Policy for the multipliers of inactive constraints

      s%prev_control%inactive_y = control%inactive_y
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    policy for multipliers of inactive constraints set to LEAVE_AS_IS'

!     Sign convention for the dual variables

      s%prev_control%z_sign  = control%z_sign
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    sign convention for the dual variables set to POSITIVE'

!    Policy for the multipliers of inactive bounds

      s%prev_control%inactive_z = control%inactive_z
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    policy for multipliers of inactive bounds set to LEAVE_AS_IS'

!     Final status of the bounds on the variables

      s%prev_control%final_x_bounds = control%final_x_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    final status of the bounds on the variables set to TIGHTEST'

!     Final status of the bounds on the dual variables

      s%prev_control%final_z_bounds = control%final_z_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    final status of the bounds on the dual variables set to TIGHTEST'

!     Final status of the bounds on the constraints

      s%prev_control%final_c_bounds = control%final_c_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    final status of the bounds on the constraints set to TIGHTEST'

!     Final status of the bounds on the multipliers

      s%prev_control%final_y_bounds = control%final_y_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    final status of the bounds on the multipliers set to TIGHTEST'

!     Request for dual transformations

      s%prev_control%dual_transformations = control%dual_transformations
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    dual transformations allowed'

!     Request for removal of redundant variables/constraints

      s%prev_control%redundant_xc = control%redundant_xc
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    attempt to remove redundant variables and constraints'

!     Obtention of various problem items on output
!     1) quadratic value

      s%prev_control%get_q = control%get_q
      IF ( s%level >= DEBUG )  WRITE( s%out, * ) '    quadratic value required'

!     2) quadratic constant

      s%prev_control%get_f = control%get_f
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    quadratic constant not required'

!     3) quadratic gradient

      s%prev_control%get_g = control%get_g
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    quadratic gradient not required'

!     4) quadratic Hessian

      s%prev_control%get_H = control%get_H
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    quadratic Hessian not required'

!     5) the matrix A

      s%prev_control%get_A = control%get_A
      IF ( s%level >= DEBUG )  WRITE( s%out, * ) '    matrix A not required'

!     6) the values of the variables

      s%prev_control%get_x = control%get_x
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the variables required'

!     7) the bounds on the variables

      s%prev_control%get_x_bounds = control%get_x_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    bounds on the variables not required'

!     8) the values of the dual variables

      s%prev_control%get_z = control%get_z
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the dual variables required'

!     9) the bounds on the dual variables

      s%prev_control%get_z_bounds = control%get_z_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the dual variables required'

!     10) the constraints values

      s%prev_control%get_c = control%get_c
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the constraints required'

!     11) the bounds on constraints

      s%prev_control%get_c_bounds = control%get_c_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the bounds on the constraints not required'

!     12) the multipliers

      s%prev_control%get_y = control%get_y
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the multipliers required'

!     13) the bounds on the multipliers

      s%prev_control%get_y_bounds = control%get_y_bounds
      IF ( s%level >= DEBUG )  WRITE( s%out, * )                               &
         '    values of the bounds on the multipliers not required'

!     Set the map indicator to indicate that INITIALIZE was successful.

      s%stage = READY

      CALL PRESOLVE_say_goodbye( control, inform, s )

      RETURN

      END SUBROUTINE PRESOLVE_initialize

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

      SUBROUTINE PRESOLVE_read_specfile( device, control, inform, alt_specname )

!     Reads the content of a specification files and performs the assignment
!     of values associated with given keywords to the corresponding control
!     parameters. See the documentation of the GALAHAD_specfile module for
!     further details.

!     Arguments:

      INTEGER, INTENT( IN ) :: device

!            The device number associated with the specification file. Note
!            that the file must be open for input.  The file is REWINDed
!            before use.

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!            The PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!            The PRESOLVE exit information structure (see above)

      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!            By default, the segment of the specfile relevant to
!            the PRESOLVE package will be enclosed by
!              BEGIN PRESOLVE
!              END PRESOLVE
!            An alternative to the string 'PRESOLVE' may be provided
!            using the optional string alt_specname

!     Programming: Ph. Toint, December 2001.
!
!===============================================================================

!     Local variables

      INTEGER :: lspec, ios
      CHARACTER( LEN = 16 ), PARAMETER :: specname = 'PRESOLVE        '
      TYPE ( SPECFILE_item_type ), DIMENSION( : ), ALLOCATABLE :: spec

!     Construct the specification list.

      lspec = 49         !  there are 49 possible specification commands

!     Allocate the spec item data structure.

      ALLOCATE( spec( lspec ), STAT = ios )
      IF ( ios /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: no memory left for allocating spec(',  lspec, ')'
         RETURN
      END IF

!     Define the keywords.

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level'
      spec(  4 )%keyword = 'infinity-value'
      spec(  5 )%keyword = 'maximum-number-of-passes'
      spec(  6 )%keyword = 'transformations-buffer-size'
      spec(  7 )%keyword = 'transformations-file-name'
      spec(  8 )%keyword = 'transformations-file-device'
      spec(  9 )%keyword = 'transformations-file-status'
      spec( 10 )%keyword = 'maximum-percentage-row-fill'
      spec( 11 )%keyword = 'pivoting-threshold'
      spec( 12 )%keyword = 'minimum-relative-bound-improvement'
      spec( 13 )%keyword = 'constraints-accuracy'
      spec( 14 )%keyword = 'dual-variables-accuracy'
      spec( 15 )%keyword = 'primal-feasibility-check'
      spec( 16 )%keyword = 'dual-feasibility-check'
      spec( 17 )%keyword = 'primal-constraints-analysis-frequency'
      spec( 18 )%keyword = 'dual-constraints-analysis-frequency'
      spec( 19 )%keyword = 'singleton-columns-analysis-frequency'
      spec( 20 )%keyword = 'doubleton-columns-analysis-frequency'
      spec( 21 )%keyword = 'unconstrained-variables-analysis-frequency'
      spec( 22 )%keyword = 'dependent-variables-analysis-frequency'
      spec( 23 )%keyword = 'row-sparsification-frequency'
      spec( 24 )%keyword = 'presolve-termination-strategy'
      spec( 25 )%keyword = 'active-multipliers-sign'
      spec( 26 )%keyword = 'inactive-multipliers-value'
      spec( 27 )%keyword = 'active-dual-variables-sign'
      spec( 28 )%keyword = 'inactive-dual-variables-value'
      spec( 29 )%keyword = 'primal-variables-bound-status'
      spec( 30 )%keyword = 'dual-variables-bound-status'
      spec( 31 )%keyword = 'constraints-bound-status'
      spec( 32 )%keyword = 'multipliers-bound-status'
      spec( 33 )%keyword = 'compute-quadratic-value'
      spec( 34 )%keyword = 'compute-objective-constant'
      spec( 35 )%keyword = 'compute-gradient'
      spec( 36 )%keyword = 'compute-Hessian'
      spec( 37 )%keyword = 'compute-constraints-matrix'
      spec( 38 )%keyword = 'compute-primal-variables-values'
      spec( 39 )%keyword = 'compute-primal-variables-bounds'
      spec( 40 )%keyword = 'compute-dual-variables-values'
      spec( 41 )%keyword = 'compute-dual-variables-bounds'
      spec( 42 )%keyword = 'compute-constraints-values'
      spec( 43 )%keyword = 'compute-constraints-bounds'
      spec( 44 )%keyword = 'compute-multipliers-values'
      spec( 45 )%keyword = 'compute-multipliers-bounds'
      spec( 46 )%keyword = 'maximum-number-of-transformations'
      spec( 47 )%keyword = 'allow-dual-transformations'
      spec( 48 )%keyword = 'remove-redundant-variables-constraints'
      spec( 49 )%keyword = 'maximum-growth-factor'

!     Read the specfile.

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SPECFILE_read( device, alt_specname, spec, lspec, control%errout )
      ELSE
        CALL SPECFILE_read( device, specname, spec, lspec, control%errout )
      END IF

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
                                    control%infinity,                          &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 5 ),                                 &
                                    control%max_nbr_passes,                    &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 6 ),                                 &
                                    control%transf_buffer_size,                &
                                    control%errout )
      CALL SPECFILE_assign_string ( spec( 7 ),                                 &
                                    control%transf_file_name,                  &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 8 ),                                 &
                                    control%transf_file_nbr,                   &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 9 ),                                 &
                                    control%transf_file_status,                &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 10 ),                                &
                                    control%max_fill,                          &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 11 ),                                &
                                    control%pivot_tol,                         &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 12 ),                                &
                                    control%min_rel_improve,                   &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 13 ),                                &
                                    control%c_accuracy,                        &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 14 ),                                &
                                    control%z_accuracy,                        &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 15 ),                                &
                                    control%check_primal_feasibility,          &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 16 ),                                &
                                    control%check_dual_feasibility,            &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 17 ),                                &
                                    control%primal_constraints_freq,           &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 18 ),                                &
                                    control%dual_constraints_freq,             &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 19 ),                                &
                                    control%singleton_columns_freq,            &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 20 ),                                &
                                    control%doubleton_columns_freq ,           &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 21 ),                                &
                                    control%unc_variables_freq,                &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 22 ),                                &
                                    control%dependent_variables_freq,          &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 23 ),                                &
                                    control%sparsify_rows_freq,                &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 24 ),                                &
                                    control%termination,                       &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 25 ),                                &
                                    control%y_sign,                            &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 26 ),                                &
                                    control%inactive_y,                        &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 27 ),                                &
                                    control%z_sign,                            &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 28 ),                                &
                                    control%inactive_z,                        &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 29 ),                                &
                                    control%final_x_bounds,                    &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 30 ),                                &
                                    control%final_z_bounds,                    &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 31 ),                                &
                                    control%final_c_bounds,                    &
                                    control%errout )
      CALL SPECFILE_assign_symbol ( spec( 32 ),                                &
                                    control%final_y_bounds,                    &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 33 ),                                &
                                    control%get_q,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 34 ),                                &
                                    control%get_f,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 35 ),                                &
                                    control%get_g,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 36 ),                                &
                                    control%get_H,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 37 ),                                &
                                    control%get_A,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 38 ),                                &
                                    control%get_x,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 39 ),                                &
                                    control%get_x_bounds,                      &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 40 ),                                &
                                    control%get_z,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 41 ),                                &
                                    control%get_z_bounds,                      &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 42 ),                                &
                                    control%get_c,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 43 ),                                &
                                    control%get_c_bounds,                      &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 44 ),                                &
                                    control%get_y,                             &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 45 ),                                &
                                    control%get_y_bounds,                      &
                                    control%errout )
      CALL SPECFILE_assign_integer( spec( 46 ),                                &
                                    control%max_nbr_transforms,                &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 47 ),                                &
                                    control%dual_transformations,              &
                                    control%errout )
      CALL SPECFILE_assign_logical( spec( 48 ),                                &
                                    control%redundant_xc,                      &
                                    control%errout )
      CALL SPECFILE_assign_real   ( spec( 49 ),                                &
                                    control%max_growth_factor,                 &
                                    control%errout )
      DEALLOCATE( spec )

      RETURN

      END SUBROUTINE PRESOLVE_read_specfile

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================            A P P L Y              =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_apply( prob, control, inform, s )

!     Analyzes the structure of the problem, performing problem
!     transformations to reduce it to a simpler form or to deduce tighter
!     bounds on its variables, dual variables or multipliers.

!     Arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Review the control structure.

      CALL PRESOLVE_revise_control( ANALYZE, prob, control, inform, s )

!     Check consistency of input and allocate problem dependent workspace
!     if not already done.

      IF ( prob%new_problem_structure ) CALL PRESOLVE_check_consistency

!     The work!

      IF ( inform%status == OK ) CALL PRESOLVE_analyze
      IF ( inform%status >= 0  ) CALL PRESOLVE_permute

!     Conclude...

      CALL PRESOLVE_say_goodbye( control, inform, s )

      RETURN

!==============================================================================
!==============================================================================

   CONTAINS  ! The analyze routines

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_check_consistency

!     Checks consistency of the problem data and initializes problem dependent
!     workspace. Additionally, performs the structure analysis for the
!     Jacobian and Hessian.

!     Programming: Ph. Toint, November 2000

!==============================================================================


!     Local variables

      INTEGER :: dim, i_space, r_space, iostat, a_ne, h_ne, j

!-------------------------------------------------------------------------------
!
!                           Consistency checks
!
!-------------------------------------------------------------------------------
!

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   checking arguments for consistency'

!     Check the problem dimensions.

      IF ( prob%n <= 0 ) THEN
         inform%status = WRONG_N
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: the problem has', prob%n, 'variables!'
         RETURN
      END IF
      IF ( prob%m < 0 ) THEN
         inform%status = WRONG_M
         WRITE( inform%message( 1 ), * )                                       &
                 ' PRESOLVE ERROR: the problem has', prob%m, 'constraints!'
         RETURN
      END IF

!-------------------------------------------------------------------------------
!
!                      Check the size of the vectors
!
!-------------------------------------------------------------------------------
!
!     1) size n vectors

!     G

      SELECT CASE ( prob%gradient_kind )

      CASE ( ALL_ZEROS )

         IF ( .NOT. ALLOCATED( prob%G ) ) THEN
            ALLOCATE( prob%X_l( prob%n ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating g(',         &
               prob%n, ')'
               RETURN
            END IF
         END IF
         prob%G = ZERO

      CASE ( ALL_ONES )

         IF ( .NOT. ALLOCATED( prob%G ) ) THEN
            ALLOCATE( prob%X_l( prob%n ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating g(',         &
               prob%n, ')'
               RETURN
            END IF
         END IF
         prob%G = ONE

      CASE DEFAULT

         IF ( .NOT. ALLOCATED( prob%G ) ) THEN
            inform%status = G_NOT_ALLOCATED
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: the vector x has not been allocated'
            RETURN
         END IF

      END SELECT
      IF ( SIZE( prob%G ) < prob%n ) THEN
         inform%status = WRONG_G_DIMENSION
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: the size of g (', SIZE( prob%G ), ')'
         WRITE( inform%message( 2 ), * )                                       &
              '    is incompatible with the problem dimension (', prob%n, ')'
         RETURN
      END IF

!     X_status

      IF ( ALLOCATED( prob%X_status ) ) THEN
         IF ( SIZE( prob%X_status ) < prob%n ) THEN
            inform%status = WRONG_XS_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of x_status (',                    &
                 SIZE( prob%X_status ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the problem dimension (', prob%n, ')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%X_status( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating x_status(',     &
               prob%n, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    prob%x_status(', prob%n, ') alloacted',                       &
            ' with all variables ACTIVE'
         prob%X_status = ACTIVE
      END IF

!     X_l

      IF ( ALLOCATED( prob%X_l ) ) THEN
         IF ( SIZE( prob%X_l ) < prob%n ) THEN
            inform%status = WRONG_XL_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of x_l (', SIZE( prob%X_l ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the problem dimension (', prob%n, ')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%X_l( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating x_l(',          &
               prob%n, ')'
            RETURN
         END IF
         prob%X_l = - control%infinity
      END IF

!     X_u

      IF ( ALLOCATED( prob%X_u ) ) THEN
         IF ( SIZE( prob%X_u ) < prob%n ) THEN
            inform%status = WRONG_XU_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of x_u (', SIZE( prob%X_u ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the problem dimension (', prob%n, ')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%X_u( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating x_u(',          &
               prob%n, ')'
            RETURN
         END IF
         prob%X_u = control%infinity
      END IF

!     X

      IF ( ALLOCATED( prob%X ) ) THEN
         IF ( SIZE( prob%X ) < prob%n ) THEN
            inform%status = WRONG_C_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of x (', SIZE( prob%X ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the number of variables (',         &
                 prob%n,')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%X( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating x(',            &
               prob%n, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    allocating prob%x(', prob%n, ')'
         DO j = 1, prob%n
            CALL PRESOLVE_guess_x( j, prob%X( j ), prob, s )
         END DO
      END IF

!     Z_l

      IF ( ALLOCATED( prob%Z_l ) ) THEN
         IF ( SIZE( prob%Z_l ) < prob%n ) THEN
            inform%status = WRONG_ZL_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of z_l (', SIZE( prob%Z_l ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the problem dimension (', prob%n, ')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%Z_l( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating z_l(',          &
               prob%n, ')'
            RETURN
         END IF
         prob%Z_l = - control%infinity
      END IF

!     Z_u

      IF ( ALLOCATED( prob%Z_u ) ) THEN
         IF ( SIZE( prob%Z_u ) < prob%n ) THEN
            inform%status = WRONG_ZU_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of z_u (', SIZE( prob%Z_u ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the problem dimension (', prob%n, ')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%Z_u( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating z_u(',          &
               prob%n, ')'
            RETURN
         END IF
         prob%Z_u = control%infinity
      END IF

!     Z

      IF ( ALLOCATED( prob%Z ) ) THEN
         IF ( SIZE( prob%Z ) < prob%n ) THEN
            inform%status = WRONG_Z_DIMENSION
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: the size of z (', SIZE( prob%Z ), ')'
            WRITE( inform%message( 2 ), * )                                    &
                 '    is incompatible with the problem dimension (', prob%n, ')'
            RETURN
         END IF
      ELSE
         ALLOCATE( prob%Z( prob%n ) , STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating z(',            &
               prob%n, ')'
            RETURN
         END IF
         prob%Z = ZERO
      END IF

!     2) size m vectors

      IF ( prob%m > 0 ) THEN

!        C

         IF ( ALLOCATED( prob%C ) ) THEN
            IF ( SIZE( prob%C ) < prob%m ) THEN
               inform%status = WRONG_C_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of c (', SIZE( prob%C ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m,')'
               RETURN
            END IF
         ELSE
            ALLOCATE( prob%C( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating c(',         &
                  prob%m, ')'
               RETURN
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    allocating prob%c(', prob%m, ')'
         END IF

!     C_status

         IF ( ALLOCATED( prob%C_status ) ) THEN
            IF ( SIZE( prob%C_status ) < prob%m ) THEN
               inform%status = WRONG_CS_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of c_status (',                 &
                    SIZE( prob%C_status ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m, ')'
               RETURN
             END IF
         ELSE
            ALLOCATE( prob%C_status( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating c_status(',  &
                  prob%m, ')'
               RETURN
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    prob%c_status(', prob%m, ') allocated',                    &
               ' with all constraints ACTIVE'
            prob%C_status = ACTIVE
         END IF

!        C_l

         IF ( ALLOCATED( prob%C_l ) ) THEN
            IF ( SIZE( prob%C_l ) < prob%m ) THEN
               inform%status = WRONG_CL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of c_l (', SIZE( prob%C_l ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m,')'
               RETURN
            END IF
         ELSE
            ALLOCATE( prob%C_l( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating c_l(',       &
                  prob%m, ')'
               RETURN
            END IF
            prob%C_l = - control%infinity
         END IF

!        C_u

         IF ( ALLOCATED( prob%C_u ) ) THEN
            IF ( SIZE( prob%C_u ) < prob%m ) THEN
               inform%status = WRONG_CU_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of c_u (', SIZE( prob%C_u ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m,')'
               RETURN
            END IF
         ELSE
            ALLOCATE( prob%C_l( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating c_u(',       &
                  prob%m, ')'
               RETURN
            END IF
            prob%C_u = control%infinity
         END IF

!        Y

         IF ( ALLOCATED( prob%Y ) ) THEN
            IF ( SIZE( prob%Y ) < prob%m ) THEN
               inform%status = WRONG_Y_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of y (', SIZE( prob%Y ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m,')'
               RETURN
            END IF
         ELSE
            ALLOCATE( prob%Y( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating y(',         &
                  prob%m, ')'
               RETURN
            END IF
            prob%Y = ZERO
         END IF

!        Y_l

         IF ( ALLOCATED( prob%Y_l ) ) THEN
            IF ( SIZE( prob%Y_l ) < prob%m ) THEN
               inform%status = WRONG_YL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of y_l (', SIZE( prob%Y_l ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m,')'
               RETURN
            END IF
         ELSE
            ALLOCATE( prob%Y_l( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating y_l(',       &
                  prob%m, ')'
               RETURN
            END IF
            prob%Y_l = - control%infinity
         END IF

!        Y_u

         IF ( ALLOCATED( prob%Y_u ) ) THEN
            IF ( SIZE( prob%Y_u ) < prob%m ) THEN
               inform%status = WRONG_YU_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: the size of y_u (', SIZE( prob%Y_u ), ')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the number of constraints (',    &
                    prob%m,')'
               RETURN
            END IF
         ELSE
            ALLOCATE( prob%Y_u( prob%m ) , STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating y_u(',       &
                  prob%m, ')'
               RETURN
            END IF
            prob%Y_u = control%infinity
         END IF
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    vector sizes verified'

!-------------------------------------------------------------------------------
!
!                   Check the size of the matrix arrays.
!
!-------------------------------------------------------------------------------
!
!     1 ) the matrix A

      IF ( prob%m > 0 ) THEN

         SELECT CASE ( SMT_get( prob%A%type ) )
         CASE ( 'DENSE' )
            s%a_type = DENSE
            a_ne     = prob%n * prob%m
         CASE ( 'SPARSE_BY_ROWS' )
            s%a_type = SPARSE
            a_ne     = prob%A%ptr( prob%m + 1 ) - 1
         CASE ( 'COORDINATE' )
            s%a_type = COORDINATE
            a_ne     = prob%A%ne
         CASE DEFAULT
            inform%status = WRONG_ANE
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: wrong value for the storage type A_ne'
            RETURN
         END SELECT

         IF ( .NOT. ALLOCATED( prob%A%val ) ) THEN
            inform%status = AVAL_NOT_ALLOCATED
             WRITE( inform%message( 1 ), * )                                   &
               ' PRESOLVE ERROR: the vector A_val has not been allocated'
            RETURN
         END IF

         SELECT CASE ( s%a_type )

         CASE ( DENSE )

            IF ( SIZE( prob%A%val ) < a_ne ) THEN
               inform%status = WRONG_AVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of a_val (', SIZE( prob%A%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with the problem dimensions ( n * m =', &
                  a_ne, ')'
               RETURN
            END IF

         CASE  ( SPARSE )

            IF ( .NOT. ALLOCATED( prob%A%ptr ) ) THEN
                inform%status = APTR_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector A_ptr has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%A%ptr ) < prob%m + 1 ) THEN
               inform%status = WRONG_AVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of A_ptr (', SIZE( prob%A%ptr ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with the number of constraints (',      &
                  prob%m, ')'
               RETURN
            END IF
            IF ( SIZE( prob%A%val ) < a_ne ) THEN
               inform%status = WRONG_APTR_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of a_val (', SIZE( prob%A%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', a_ne, ')'
               RETURN
            END IF
            IF ( .NOT. ALLOCATED( prob%A%col ) ) THEN
               inform%status = ACOL_NOT_ALLOCATED
               WRITE( inform%message( 1 ), * )                                &
                     ' PRESOLVE ERROR: the vector A_col has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%A%col ) < a_ne ) THEN
               inform%status = WRONG_ACOL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of a_col (', SIZE( prob%A%col ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', a_ne,')'
               RETURN
            END IF

         CASE  ( COORDINATE )

            IF ( SIZE( prob%A%val ) < a_ne ) THEN
               inform%status = WRONG_AVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of a_val (', SIZE( prob%A%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', a_ne,')'
               RETURN
            END IF
            IF ( .NOT. ALLOCATED( prob%A%col ) ) THEN
                inform%status = ACOL_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector A_col has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%A%col ) < a_ne ) THEN
               inform%status = WRONG_ACOL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of a_col (', SIZE( prob%A%col ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', a_ne,')'
               RETURN
            END IF
            IF ( .NOT. ALLOCATED( prob%A%row ) ) THEN
                inform%status = AROW_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector A_row has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%A%row ) < a_ne ) THEN
               inform%status = WRONG_AROW_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of a_row (', SIZE( prob%A%row ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', a_ne,')'
               RETURN
            END IF

         END SELECT
         IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    A sizes verified'

!     Set the Jacobian dimension to zero if there are no constraints.

      ELSE
         a_ne = 0
      END IF

!     2 ) the matrix H

      SELECT CASE ( SMT_get( prob%H%type ) )
      CASE ( 'DIAGONAL' )
         s%h_type = DIAGONAL
         h_ne     = prob%n
      CASE ( 'DENSE' )
         s%h_type = DENSE
         h_ne     = ( prob%n * ( prob%n + 1 ) ) / 2
      CASE ( 'SPARSE_BY_ROWS' )
         s%h_type = SPARSE
         h_ne     = prob%H%ptr( prob%n + 1 ) - 1
      CASE ( 'COORDINATE' )
         s%h_type = COORDINATE
         h_ne     = prob%H%ne
      CASE ( 'ZERO' )
         s%h_type = ABSENT
         h_ne     = 0
      CASE DEFAULT
         inform%status = WRONG_HNE
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: wrong value for the storage type H_ne'
         RETURN
      END SELECT

      IF ( h_ne > 0 ) THEN
         IF ( .NOT. ALLOCATED( prob%H%val ) ) THEN
             inform%status = HVAL_NOT_ALLOCATED
             WRITE( inform%message( 1 ), * )                                   &
               ' PRESOLVE ERROR: the vector H_val has not been allocated'
            RETURN
         END IF

         SELECT CASE( s%h_type )

         CASE ( DIAGONAL )

            IF ( SIZE( prob%H%val ) < h_ne  ) THEN
               inform%status = WRONG_HVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                   ' PRESOLVE ERROR: the size of h_val (',SIZE( prob%H%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the problem dimensions'
               WRITE( inform%message( 3 ), * ) ' ( n =', h_ne, ')'
               RETURN
            END IF

         CASE ( DENSE )

            IF ( SIZE( prob%H%val ) < h_ne  ) THEN
               inform%status = WRONG_HVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                   ' PRESOLVE ERROR: the size of h_val (',SIZE( prob%H%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                    '    is incompatible with the problem dimensions'
               WRITE( inform%message( 3 ), * ) ' ( n*(n+1)/2 =', h_ne, ')'
               RETURN
            END IF

         CASE  ( SPARSE )

            IF ( .NOT. ALLOCATED( prob%H%ptr ) ) THEN
                inform%status = HPTR_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector H_ptr has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%H%ptr ) < prob%n + 1  ) THEN
               inform%status = WRONG_HPTR_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of h_ptr (', SIZE( prob%H%ptr ),')'
               WRITE( inform%message( 2 ), * )                                 &
                 '    is incompatible with the number of variables (',prob%n,')'
               RETURN
            END IF
            IF ( SIZE( prob%H%val ) < h_ne ) THEN
               inform%status = WRONG_HVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of h_val (', SIZE( prob%H%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', h_ne,')'
               RETURN
            END IF
            IF ( .NOT. ALLOCATED( prob%H%col ) ) THEN
                inform%status = HCOL_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector H_col has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%H%col ) < h_ne ) THEN
               inform%status = WRONG_HCOL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of h_col (', SIZE( prob%H%col ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', h_ne,')'
               RETURN
            END IF

         CASE  ( COORDINATE )

            IF ( SIZE( prob%H%val ) < h_ne )THEN
               inform%status = WRONG_HVAL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of h_val (', SIZE( prob%H%val ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', h_ne,')'
               RETURN
            END IF
            IF ( .NOT. ALLOCATED( prob%H%col ) ) THEN
                inform%status = HCOL_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector H_col has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%H%col ) < h_ne ) THEN
               inform%status = WRONG_HCOL_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of h_col (', SIZE( prob%H%col ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', h_ne,')'
               RETURN
            END IF
            IF ( .NOT. ALLOCATED( prob%H%row ) ) THEN
                inform%status = HROW_NOT_ALLOCATED
                WRITE( inform%message( 1 ), * )                                &
                  ' PRESOLVE ERROR: the vector H_row has not been allocated'
               RETURN
            END IF
            IF ( SIZE( prob%H%row ) < h_ne ) THEN
               inform%status = WRONG_HROW_DIMENSION
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the size of h_row (', SIZE( prob%H%row ),')'
               WRITE( inform%message( 2 ), * )                                 &
                  '    is incompatible with its declared value (', h_ne,')'
               RETURN
            END IF

         END SELECT

         IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    H sizes verified'

      END IF

!-------------------------------------------------------------------------------
!
!                  Initialize the mapping structure
!
!-------------------------------------------------------------------------------

      IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   initializing map'

!     Space for storing the type of successive problem transformations

      i_space = 0
      r_space = 0
      IF ( control%transf_buffer_size > 0 ) THEN
         ALLOCATE( s%hist_type( control%transf_buffer_size ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating hist_type(',    &
               control%transf_buffer_size, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG )WRITE( s%out, * )                              &
            '    hist_type(', control%transf_buffer_size, ') allocated'
         i_space = i_space + control%transf_buffer_size

!        Space for storing the first integer associated with the problem
!        transformations

         ALLOCATE( s%hist_i( control%transf_buffer_size ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating hist_i(',       &
               control%transf_buffer_size, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    hist_i(', control%transf_buffer_size, ') allocated'
         i_space = i_space + control%transf_buffer_size

!        Space for storing the second integer associated with the problem
!        transformations

         ALLOCATE( s%hist_j( control%transf_buffer_size ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating hist_j(',       &
               control%transf_buffer_size, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    hist_j(', control%transf_buffer_size, ') allocated'
         i_space = i_space + control%transf_buffer_size

!        Space for storing the real associated with the problem
!        transformations, allowing restoration of the original problem.
!        Also used as temporary workstorage for the efficient computation of
!        the dual variables at the end of PRESOLVE_restore (hence the minimal
!        dimension prob%n).

         dim = MAX( control%transf_buffer_size, prob%n )
         ALLOCATE( s%hist_r( dim ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating hist_r(',       &
               dim, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    hist_r(', dim, ') allocated'
         r_space = r_space + dim

      ELSE
         inform%status = MAX_NBR_TRANSF
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: the maximum number of problem transformations'
         WRITE( inform%message( 2 ), * )                                       &
              '(', control%transf_buffer_size, ') has been reached'
         RETURN
      END IF

!     Space for the permutation of A and the various linked lists during the
!     analysis phase (linear singletons, linear doubletons, linearly
!     unconstrained fixed variables) and heuristic dependent marks.

      s%maxmn = MAX( prob%m, prob%n )
      dim     = MAX( a_ne, s%maxmn )
      ALLOCATE( s%a_perm( dim ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
      inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: no memory left for allocating a_perm(', dim, ')'
         RETURN
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    a_perm(', dim,') allocated'
      i_space = i_space + dim
      s%a_perm = 0

!     Space for the permutation of H

      dim = MAX( h_ne, prob%n )
      ALLOCATE( s%h_perm( dim ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: no memory left for allocating h_perm(', dim, ')'
         RETURN
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    h_perm(', dim,') allocated'
      i_space = i_space + dim

!     Initialize the lists by defining their first element and
!     setting h_perm to zero

      s%lsc_f  = END_OF_LIST
      s%ldc_f  = END_OF_LIST
      s%unc_f  = END_OF_LIST
      s%lfx_f  = END_OF_LIST
      s%h_perm = 0

!-------------------------------------------------------------------------------
!
!                         Additional workspace
!
!-------------------------------------------------------------------------------

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   initializing additional workspace'

!     A vector of size MAX( prob%m, prob%n + 1 )

      dim = MAX( prob%m, prob%n + 1 )
      ALLOCATE( s%w_mn( dim ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: no memory left for allocating w_mn(', dim, ')'
         RETURN
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    w_mn(', dim,') allocated'

!     A vector of size prob%n

      ALLOCATE( s%w_n( prob%n ), STAT = iostat )
      IF ( iostat /= 0 ) THEN
         inform%status = MEMORY_FULL
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: no memory left for allocating w_n(', prob%n,')'
         RETURN
      END IF
      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    w_n(', prob%n, ') allocated'
      i_space = i_space + prob%n

!     A vector of size prob%m + 1

      IF ( prob%m > 0 ) THEN
         dim = prob%m + 1
         ALLOCATE( s%w_m( dim ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating w_m(', dim, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    w_m(', dim, ') allocated'
         i_space = i_space + dim

!     A vector of size m (used to remember which rows are concatenated)

         ALLOCATE( s%conc( prob%m ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
            ' PRESOLVE ERROR: no memory left for allocating conc(',prob%m,')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    conc(', prob%m, ') allocated'
         i_space = i_space + prob%m
         s%conc = END_OF_LIST
      END IF

!     A_col_s: the number of nonzero entries in the active part of
!              an original column of A. Since it is also used as temporary
!              workspace in apply_permutation for the Hessian, its size is
!              increased by one and is also allocated even if there are no
!              constraints, but the Hessian is non empty.

      IF ( prob%m > 0 .OR. h_ne > 0 ) THEN
         dim = prob%n + 1
         ALLOCATE( s%A_col_s( dim ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating A_col_s(',      &
               dim, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    A_col_s(', dim, ') allocated'
         i_space   = i_space + dim
         s%A_col_s = 0
      END IF

!-------------------------------------------------------------------------------
!
!               Initial analysis of the structure of A and H
!
!-------------------------------------------------------------------------------

!     Allocate workspace:
!
!     1) the matrix A

      IF ( prob%m > 0 ) THEN

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   analyzing A'

!        A_row_s: the number of nonzero entries in the active part of
!                 an original row of A.

         ALLOCATE( s%A_row_s( prob%m ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating A_row_s(',      &
               prob%m, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    A_row_s(', prob%m, ') allocated'
         i_space = i_space + prob%m

!        Allocate and build the pointers giving its column structure.

!        A_col_f: the position of the first entry of each column
!
         ALLOCATE( s%A_col_f( prob%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating A_col_f(',      &
               prob%n, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    A_col_f(', prob%n, ') allocated'
         i_space = i_space + prob%n

!        A_col_n : the position of the next entry in the same column,
!                  or - the column index if none

         ALLOCATE( s%A_col_n( a_ne ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating A_col_n(',      &
               a_ne, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out,* )                              &
            '    A_col_n(', a_ne, ') allocated'
         i_space = i_space + a_ne

!        A_row   : the row index of the current entry
!        Note that this is only required if problem transformations are allowed.

         IF ( control%transf_buffer_size > 0 ) THEN
            ALLOCATE( s%A_row( a_ne ), STAT = iostat )
            IF ( iostat /= 0 ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating A_row(',     &
                  a_ne, ')'
               RETURN
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    A_row(', a_ne, ') allocated'
            i_space = i_space + a_ne
         END IF
      END IF

!     2) the matrix H

      IF ( h_ne > 0 ) THEN

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   analyzing H'

!        H_str : the structure of the each row of H

         ALLOCATE( s%H_str( prob%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating H_str(',        &
               prob%n, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    H_str(', prob%n, ') allocated'
         i_space = i_space + prob%n

!        Allocate and build the pointers that hold the structure of the
!        superdiagonal part of the Hessian rows.

!        H_col_f: the position of the first entry of each superdiagonal row

         ALLOCATE( s%H_col_f( prob%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating H_col_f(',      &
               prob%n, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    H_col_f(', prob%n, ') allocated'
         i_space = i_space + prob%n

!        H_col_n : the position of the next entry in the same superdiagonal
!                  row, or - the row index if none

         ALLOCATE( s%H_col_n( h_ne ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating H_col_n(',      &
               h_ne, ')'
            RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    H_col_n(', h_ne, ') allocated'
         i_space = i_space + h_ne

!        H_row   : the row index of the current entry

         ALLOCATE( s%H_row( h_ne ),  STAT = iostat )
         IF ( iostat /= 0 ) THEN
            inform%status = MEMORY_FULL
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: no memory left for allocating H_row(',        &
               h_ne, ')'
               RETURN
         END IF
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    H_row(', h_ne , ') allocated'
         i_space = i_space + h_ne

      END IF

!-------------------------------------------------------------------------------
!
!               Store and print the problem's dimensions
!
!-------------------------------------------------------------------------------

!     Initialize the counters for the different categories of transformations.

      s%tm    = 0          ! current number of transformations in memory
      s%tt    = 0          ! total number of transformations so far
      s%ts    = 0          ! number of transformations in saved file
      s%npass = 0          ! current index of the presolving pass
      s%needs = control%max_nbr_transforms + 1 ! no interdependence so far

!     Remember the dimensions of the original problem.

      s%a_type_original = s%a_type
      s%h_type_original = s%h_type
      s%n_original      = prob%n
      s%m_original      = prob%m
      s%a_ne_original   = a_ne
      s%h_ne_original   = h_ne

!-------------------------------------------------------------------------------

!     Define checksum values for the current problem.  These values are written
!     at the beginning of the transformation files and verified at reading
!     (implying that the transformations read are for the current problem).
!     This mechanism is intended to provide some safeguard against accidental
!     altaration of the history file between calls to PRESOLVE.

      s%icheck1 = prob%m
      s%icheck2 = prob%n
!     s%icheck3 = prob%A%ne + prob%H%ne
      s%icheck3 = a_ne + h_ne
      CALL RANDOM_NUMBER( s%rcheck )

      RETURN

      END SUBROUTINE PRESOLVE_check_consistency

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_analyze

!     Analyzes the structure of the problem, performing problem
!     transformations to reduce it to a simpler form or to deduce tighter
!     bounds on its variables, dual variables or multipliers.

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: l_current, maxloop, i, j, itmp, loop,              &
                            ttbef, npass_1, n_current, m_current,              &
                            a_ne_current, h_ne_current, heuristic
      LOGICAL            :: tupletons
      REAL ( KIND = wp ) :: tmp

! -----------------------------------------------------------------------------
!
!                        Initial definitions
!
! -----------------------------------------------------------------------------

!     Print banner

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *         GALAHAD presolve for QPs         *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *            problem analysis              *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' '
      END IF

!     Check the previous steps in the presolve process.

      SELECT CASE ( s%stage )
      CASE ( READY )
      CASE ( ANALYZED )
      CASE ( FULLY_REDUCED )
      CASE ( PERMUTED )
      CASE ( VOID )
         inform%status = STRUCTURE_NOT_SET
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: the problem structure has not been set'
         WRITE( inform%message( 2 ), * )                                       &
              '    or has been cleaned up before an attempt to analyze'
         RETURN
      CASE DEFAULT
         inform%status = CORRUPTED_MAP
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE INTERNAL ERROR: corrupted map'
         RETURN
      END SELECT

!     Initialize the exit condition.

      inform%status = OK

!-------------------------------------------------------------------------------
!
!     Compute the active dimensions, the number of active equality constraints
!
!-------------------------------------------------------------------------------

!     Count the active variables, while changing the sign of the dual
!     variables, if necessary. Use the loop to compute the maximal values
!     of the bounds and gradient, and to impose the standard convention
!     on the sign of the dual variables.

      s%x_max    = ZERO
      s%z_max    = ZERO
      s%g_max    = ZERO
      s%n_active = 0
      DO j = 1, prob%n

!        Compute the maximal bound on x.

         IF ( prob%X_u( j ) < s%P_INFINITY )                                   &
            s%x_max = MAX( s%x_max, ABS( prob%X_u( j ) ) )
         IF ( prob%X_l( j ) > s%M_INFINITY )                                   &
            s%x_max = MAX( s%x_max, ABS( prob%X_l( j ) ) )

!        Compute the maximal bound on z.

         IF ( prob%Z_u( j ) < s%P_INFINITY )                                   &
            s%z_max = MAX( s%z_max, ABS( prob%Z_u( j ) ) )
         IF ( prob%Z_l( j ) > s%M_INFINITY )                                   &
            s%z_max = MAX( s%z_max, ABS( prob%Z_l( j ) ) )

!        Compute the maximal gradient component.

         s%g_max = MAX( s%g_max, ABS( prob%G( j ) ) )

!        Impose the sign convention on the dual variables.

         IF ( control%z_sign /= POSITIVE ) THEN
            prob%Z( j )   = - prob%Z( j )
            tmp           = - prob%Z_l( j )
            prob%Z_l( j ) = - prob%Z_u( j )
            prob%Z_u( j ) = tmp
         END IF

!        Count the active variables.

         IF ( prob%X_status( j ) /= INACTIVE ) s%n_active = s%n_active + 1

      END DO

!     Exit if there are no active variables in the problem.

      IF ( s%n_active == 0 ) THEN
         s%stage = FULLY_REDUCED
         RETURN
      END IF

!     Set the number of variables present in the problem.

      s%n_in_prob = s%n_active

!     Count the active constraints, while changing the sign of the
!     multipliers, if necessary.

      s%m_active    = 0
      s%m_eq_active = 0
      s%c_max       = ZERO
      s%y_max       = ZERO
      DO i = 1, prob%m

!        Compute the maximal constraint bound.

         IF ( prob%C_u( i ) < s%P_INFINITY )                                   &
            s%c_max = MAX( s%c_max, ABS( prob%C_u( i ) ) )
         IF ( prob%C_l( i ) > s%M_INFINITY )                                   &
            s%c_max = MAX( s%c_max, ABS( prob%C_l( i ) ) )

!        Compute the maximal multiplier bound.

         IF ( prob%Y_u( i ) < s%P_INFINITY )                                   &
            s%y_max = MAX( s%y_max, ABS( prob%Y_u( i ) ) )
         IF ( prob%Y_l( i ) > s%M_INFINITY )                                   &
            s%y_max = MAX( s%y_max, ABS( prob%Y_l( i ) ) )

!        Impose the sign convention on the multipliers.

         IF ( control%y_sign /= POSITIVE ) THEN
            prob%Y( i )   = - prob%Y( i )
            tmp           = - prob%Y_l( i )
            prob%Y_l( i ) = - prob%Y_u( i )
            prob%Y_u( i ) = tmp
         END IF

!        Count the active and active-equality constraints.

         IF ( prob%C_status( i ) == ACTIVE ) THEN
            s%m_active = s%m_active + 1
            IF ( prob%C_l( i ) == prob%C_u( i ) )                              &
               s%m_eq_active = s%m_eq_active + 1
         END IF

      END DO
      m_current = s%m_active

!     Set the number of constraints present in the problem.

      s%m_in_prob = s%m_active

!     Print the active dimensions, if requested.

      IF ( s%level >= DEBUG ) THEN
         WRITE( s%out, * )    '    n_active =', s%n_active,                    &
              ' m_active =', s%m_active, 'm_eq_active =', s%m_eq_active
         WRITE( s%out, * )    '    n_in_prob =', s%n_in_prob,                  &
              '    m_in_prob =', s%m_in_prob
      END IF

!-------------------------------------------------------------------------------
!
!                    Analysis of the structure of A and H
!
!-------------------------------------------------------------------------------

!     1) the matrix A

      s%a_max       = ZERO
      s%a_ne_active = 0
      IF ( prob%m > 0 ) THEN

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   analyzing A'

!        Transform to sparse row-wise format if needed.

         SELECT CASE ( s%a_type )
         CASE ( COORDINATE )
            CALL QPT_A_from_C_to_S( prob, inform%status )
            IF ( inform%status /= OK ) RETURN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '    A processed from coordinate to sparse storage'
         CASE ( DENSE )
            CALL QPT_A_from_D_to_S( prob, inform%status )
            IF ( inform%status /= OK ) RETURN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '    A processed from dense to sparse storage'
         END SELECT
         prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1

!        Get the column structure of A, when it is sparse.

         CALL PRESOLVE_get_sparse_cols_A
         IF ( s%level >= DEBUG )  WRITE( s%out, * )                            &
            '    column structure of A determined'

!        Compute the initial row and column sizes of A.

         CALL PRESOLVE_get_sizes_A
         IF ( inform%status /= OK ) RETURN
         IF ( s%level >= DEBUG ) THEN
            DO i = 1, prob%m
               IF ( prob%C_status( i ) /= INACTIVE ) THEN
                  WRITE( s%out, * )    '    row   ', i, 'of A has',            &
                       s%A_row_s( i ), 'active nonzeros'
               ELSE
                  WRITE( s%out, * ) '    row   ', i,'of A is inactive'
               END IF
            END DO
            DO j = 1, prob%n
               IF ( prob%X_status( j ) /= INACTIVE ) THEN
                  WRITE( s%out, * )    '    column', j, 'of A has',            &
                       s%A_col_s( j ), 'active nonzeros'
               ELSE
                  WRITE( s%out, * ) '    column', j, 'of A is inactive'
               END IF
            END DO
            WRITE( s%out, * )                                                  &
                 '    maximal entry in A has absolute value =', s%a_max
         END IF

!     Set the Jacobian dimension to zero if there are no constraints.

      ELSE
         prob%A%ne = 0
      END IF

!     2) the matrix H

      s%h_max       = ZERO
      s%h_ne_active = 0

      SELECT CASE ( SMT_get( prob%H%type ) )
      CASE ( 'DIAGONAL' )
         j = prob%n
      CASE ( 'DENSE' )
         j = ( prob%n * ( prob%n + 1 ) ) / 2
      CASE ( 'SPARSE_BY_ROWS' )
         j = prob%H%ptr( prob%n + 1 ) - 1
      CASE DEFAULT
         j = prob%H%ne
      END SELECT
      IF ( j > 0 ) THEN

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   analyzing H'

!        Transform to sparse row-wise format if needed.

         SELECT CASE ( s%h_type )
         CASE ( COORDINATE )
            CALL QPT_H_from_C_to_S( prob, inform%status )
            IF ( inform%status /= OK ) RETURN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                   '  H processed from coordinate to sparse storage'
         CASE ( DENSE )
            CALL QPT_H_from_D_to_S( prob, inform%status )
            IF ( inform%status /= OK ) RETURN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                   '  H processed from dense to sparse storage'
         CASE ( DIAGONAL )
            CALL QPT_H_from_Di_to_S( prob, inform%status )
            IF ( inform%status /= OK ) RETURN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '    H processed from diagonal to sparse storage'
         END SELECT
         prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1

!        Get the structure of the column of the lower triangular part of H.

         CALL PRESOLVE_get_sparse_cols_H
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    superdiagonal row structure of H determined'

!        Obtain the structure of the rows/columns of the Hessian.

         CALL PRESOLVE_get_struct_H
         IF ( inform%status /= OK ) RETURN
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    row structure of H determined'
            IF ( s%level >= CRAZY ) THEN
               WRITE( s%out, * ) '     prob%H%ptr =', prob%H%ptr(:prob%n+1)
               WRITE( s%out, * ) '     prob%H%col =', prob%H%col(:prob%H%ne)
               WRITE( s%out, * ) '     H_col_f    =', s%H_col_f(:prob%n)
               WRITE( s%out, * ) '     H_col_n    =', s%H_col_n(:prob%H%ne)
               WRITE( s%out, * ) '     H_row      =', s%H_row(:prob%H%ne)
               WRITE( s%out, * ) '     H_str      =', s%H_str(:prob%n)
            END IF
         END IF
      ELSE
         prob%H%ne = 0
      END IF

!-------------------------------------------------------------------------------
!
!          Determine the problem dependent precision and tolerances
!          and prepare for analysis
!
!-------------------------------------------------------------------------------

!     Compute the accuracy level

      itmp = 500
      IF ( prob%A%ne > 0 ) THEN
         itmp = MAX( itmp, MAXVAL( s%A_row_s( :prob%m ) ) )
         itmp = MAX( itmp, MAXVAL( s%A_col_s( :prob%n ) ) )
         s%a_tol = control%pivot_tol * s%a_max
      END IF
      IF ( prob%H%ne > 0 ) THEN
         itmp = MAX( itmp, MAXVAL( ABS( s%H_str ) ) )
         s%h_tol = control%pivot_tol * s%h_max
      END IF
      tmp = MAX( s%a_max, s%h_max, s%x_max, s%z_max, s%g_max, s%c_max, s%y_max )
      s%max_growth = control%max_growth_factor * tmp
      s%ACCURACY   = tmp * itmp * EPSMACH

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * ) '   setting problem dependent parameters'
         IF ( s%level >= DEBUG ) THEN
            IF ( wp == sp ) THEN
               WRITE( s%out, * ) '    single precision option'
            ELSE
               WRITE( s%out, * ) '    double precision option'
            END IF
            WRITE( s%out, * )                                                  &
                 '    problem dependent accuracy set to', s%ACCURACY
            WRITE( s%out, * )                                                  &
                '    maximal acceptable value for reduced problem data set to',&
                 s%max_growth
            IF ( prob%A%ne > 0 ) WRITE( s%out, * )                             &
               '    tolerance for operations in A set to', s%a_tol
            IF ( prob%H%ne > 0 ) WRITE( s%out, * )                             &
               '    tolerance for operations in H set to', s%h_tol
         END IF
      END IF

!     Define the global variables.

      MAX_GROWTH = s%max_growth
      ACCURACY   = s%ACCURACY
      M_INFINITY = s%M_INFINITY
      P_INFINITY = s%P_INFINITY

!     Check the maximum number of analysis passes.

      maxloop = MAX( control%max_nbr_passes, 0 ) - s%npass

!     Initialize the markers of eliminated entries, modified variables and
!     constraints, and cancelled entries.

      s%a_perm( :s%maxmn ) = 0
      s%nmods(  :NBRH  )   = s%maxmn

!-------------------------------------------------------------------------------
!
!                   Indicate that analysis is starting.
!
!-------------------------------------------------------------------------------

!     Write the problem dimensions

      IF ( s%level >= TRACE ) WRITE( s%out, * )                                &
         ' problem dimensions : ', 'n =', prob%n   , 'm =', prob%m,            &
         'a_ne =', MAX( 0, prob%A%ne ), 'h_ne =', MAX( 0, prob%H%ne )

!     Return if the buffer is empty or if no problem transformation is allowed.

      IF ( s%max_tm == 0 .OR. control%max_nbr_transforms == 0 ) GO TO 1000

!     Otherwise, print message.

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * )                                                     &
              ' ============ starting problem analysis ============'
         WRITE( s%out, * ) ' '
      END IF

!-------------------------------------------------------------------------------
!
!              Additional elementary checks at first analysis
!
!-------------------------------------------------------------------------------

      IF ( s%tt == 0 ) THEN

!        Check the initial bounds on the problem quantities for inconsistency.

         IF ( s%level > TRACE ) THEN
            WRITE( s%out, * ) ' checking bounds on x, y, z, and c'
         END IF
         ttbef    = s%tt
         s%hindex = CHECK_BOUNDS_CONSISTENCY
         CALL PRESOLVE_check_bounds( .TRUE. )
         IF ( s%level >= TRACE ) WRITE( s%out, * )                             &
            ' checking bounds on x, y, z, and c:', s%tt - ttbef,               &
            'transformations'
         IF ( inform%status /= OK ) GO TO 1000
         IF ( s%level >= DEBUG ) THEN
            CALL PRESOLVE_write_full_prob( prob, control, s )
            CALL PRESOLVE_check_active_sizes
            IF ( inform%status /= OK ) GO TO 1000
         END IF

!        Reduce redundant variables and constraints.

         IF ( control%redundant_xc ) THEN
            IF ( s%level > TRACE ) THEN
               WRITE( s%out, * ) ' redundant variables and constraints'
            END IF
            ttbef    = s%tt
            s%hindex = REDUNDANT_VARIABLES
            CALL PRESOLVE_redundant_variables
            IF ( s%level >= TRACE ) WRITE( s%out, * )                          &
               ' redundant variables and constraints:', s%tt - ttbef,          &
               'transformations'
            IF ( inform%status /= OK ) GO TO 1000
            IF ( s%level >= DEBUG ) THEN
               CALL PRESOLVE_write_full_prob( prob, control, s )
               CALL PRESOLVE_check_active_sizes
               IF ( inform%status /= OK ) GO TO 1000
            END IF
         END IF
      END IF

!-------------------------------------------------------------------------------
!
!                       Main loop of problem analysis
!
!-------------------------------------------------------------------------------

!     Remember the initial dimensions of the problem and the initial number
!     of transformations so far.

      l_current    = s%tt
      n_current    = s%n_active
      n_current    = s%n_active
      m_current    = s%m_active
      a_ne_current = s%a_ne_active
      h_ne_current = s%h_ne_active

                             !-------------------------------------------------!
pre:  DO loop = 1, maxloop   !            the main presolving loop             !
                             !-------------------------------------------------!
         s%loop  = loop
         npass_1 = s%npass
         s%npass = s%npass + 1

!        Exit if the variables are already eliminated.

         IF ( s%n_active == 0 ) EXIT

!        print loop header

         IF ( s%level >= TRACE ) THEN
            WRITE( s%out, * ) ' '
            WRITE( s%out, * )                                                  &
                 ' ============= main processing loop', s%npass,               &
                 ' ============='
            WRITE( s%out, * ) '   ( n =', s%n_active,                          &
                               ', m =', s%m_active,                            &
                               ', a_ne =', s%a_ne_active,                      &
                               ', h_ne =', s%h_ne_active, ')'
            WRITE( s%out, * ) ' '
         END IF

!        Now consider the sequence of heuristics to apply.

         DO heuristic = 1, N_HEURISTICS       ! the heuristics loop

            ttbef  = s%tt
            s%hindex = PRESOLVING_SEQUENCE( heuristic )

            SELECT CASE ( s%hindex )

!           Build the linked lists of linear singleton, linear doubleton and
!           linear unconstrained quadratic variables, whenever needed

!           Handle special linear columns
!           (singletons, doubletons and unconstrained ).

            CASE ( SPECIAL_LINEAR_COLUMNS )

               tupletons = .FALSE.
               IF ( control%unc_variables_freq > 0 )                           &
                  tupletons = tupletons .OR.                                   &
                         MOD( npass_1, control%unc_variables_freq ) == 0
               IF ( control%singleton_columns_freq > 0 )                       &
                  tupletons = tupletons .OR.                                   &
                         MOD( npass_1, control%singleton_columns_freq ) == 0
               IF ( control%doubleton_columns_freq > 0 )                       &
                  tupletons = tupletons .OR.                                   &
                         MOD( npass_1, control%doubleton_columns_freq ) == 0
               IF ( tupletons ) THEN
                  IF ( s%level > TRACE ) WRITE( s%out, * )                     &
                        ' analyzing special linear columns'
                  CALL PRESOLVE_linear_tupletons
                  IF ( s%level >= TRACE ) WRITE( s%out, * )                    &
                     ' analyzing special linear columns:',                     &
                     s%tt - ttbef, 'transformations'
               END IF

!           Remove empty and singleton rows.

            CASE ( EMPTY_AND_SINGLETON_ROWS )

               IF ( control%primal_constraints_freq > 0 .AND. &
                    s%m_active                      > 0       )THEN
                  IF ( MOD( npass_1, control%primal_constraints_freq ) == 0)THEN
                     IF ( s%level > TRACE ) WRITE( s%out, * )                  &
                        ' removing empty and singleton rows'
                     CALL PRESOLVE_primal_constraints(EMPTY_AND_SINGLETON_ROWS )
                     IF ( s%level >= TRACE ) WRITE( s%out, * )                 &
                        ' removing empty and singleton rows:', s%tt - ttbef,   &
                       'transformations'
                  END IF
               END IF

!           Full analysis of primal constraints, including bounds
!           tightening and detection of forcing constraints

            CASE ( PRIMAL_CONSTRAINTS )

               IF ( control%primal_constraints_freq > 0 .AND. &
                    s%m_active                      > 0       ) THEN
                  IF ( MOD( npass_1, control%primal_constraints_freq ) == 0)THEN
                     IF ( s%level > TRACE ) WRITE( s%out, * )                  &
                        ' analyzing primal constraints'
                     CALL PRESOLVE_primal_constraints( PRIMAL_CONSTRAINTS )
                     IF ( s%level >= TRACE ) WRITE( s%out, * )                 &
                        ' analyzing primal constraints:', s%tt - ttbef,        &
                        'transformations'
                  END IF
               END IF

!           Analyze dual constraints.

            CASE ( DUAL_CONSTRAINTS )

               IF ( control%dual_constraints_freq > 0 .AND. &
                    s%m_active > 0                          ) THEN
                  IF ( MOD( npass_1, control%dual_constraints_freq ) == 0 ) THEN
                     IF ( s%level > TRACE ) WRITE( s%out, * )                  &
                        ' analyzing dual constraints'
                     CALL PRESOLVE_dual_constraints
                     IF ( s%level >= TRACE ) WRITE( s%out, * )                 &
                        ' analyzing dual constraints:', s%tt - ttbef,          &
                        'transformations'
                  END IF
               END IF

!           Check for the possible merging of dependent variables.

            CASE ( DEPENDENT_VARIABLES )

               IF ( control%dependent_variables_freq > 0 ) THEN
                  IF ( MOD( npass_1, control%dependent_variables_freq )==0 )THEN
                     IF ( s%level > TRACE ) WRITE( s%out, * )                  &
                        ' checking dependent variables'
                     CALL PRESOLVE_dependent_variables
                     IF ( s%level >= TRACE ) WRITE( s%out, * )                 &
                        ' checking dependent variables:', s%tt - ttbef,        &
                        ' transformations'
                  END IF
               END IF

!           Try to make A sparser by combining its rows.

            CASE ( ROW_SPARSIFICATION )

               IF ( control%sparsify_rows_freq > 0 .AND. &
                    s%m_active             > 1     .AND. &
                    s%m_eq_active          > 0           ) THEN
                  IF ( MOD( npass_1, control%sparsify_rows_freq ) == 0 ) THEN
                     IF ( s%level > TRACE ) WRITE( s%out, * )                  &
                        ' trying to make A sparser'
                     CALL PRESOLVE_sparsify_A
                     IF ( s%level >= TRACE ) WRITE( s%out, * )                 &
                        ' trying to make A sparser:', s%tt - ttbef,            &
                        'transformations'
                  END IF
               END IF

!           Check the consistency of the bounds on all problem quantities.

            CASE ( CHECK_BOUNDS_CONSISTENCY )

               IF ( s%level >  TRACE ) WRITE( s%out, * )                       &
                  ' checking bounds on x, y, z, and c'
               CALL PRESOLVE_check_bounds( .FALSE. )
               IF ( s%level >= TRACE ) WRITE( s%out, * )                       &
                  ' checking bounds on x, y, z, and c:', s%tt - ttbef, &
                  'transformations'

            END SELECT

            IF ( inform%status /= OK ) GO TO 1000
            IF ( s%level >= DEBUG ) THEN
               CALL PRESOLVE_write_full_prob( prob, control, s )
               CALL PRESOLVE_check_active_sizes
               IF ( inform%status /= OK ) GO TO 1000
            END IF

!           Exit of the presolving loop if the problem is fully reduced.

            IF ( s%n_active == 0 ) EXIT pre

         END DO  ! end of the heuristics loop

!        Check if something has happened by checking if the number of problem
!        transformations has increased during this main analysis loop. Also
!        verify the problem size.

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * )  '    new history length =', s%tt,               &
                 '    old history length =', l_current
            WRITE( s%out, * ) ' '
         END IF

!        See if one of the analysis stopping criteria is satisfied (before
!        the maximum number of loops is attained).

         SELECT CASE ( control%termination )
         CASE ( FULL_PRESOLVE )
            IF ( s%tt == l_current ) EXIT
         CASE ( REDUCED_SIZE )
            IF ( s%n_active    >= n_current    .AND. &
                 s%m_active    >= m_current    .AND. &
                 s%a_ne_active >= a_ne_current .AND. &
                 s%h_ne_active >= h_ne_current       ) EXIT
         END SELECT

!        Remember current parameters.

         l_current    = s%tt
         n_current    = s%n_active
         n_current    = s%n_active
         m_current    = s%m_active
         a_ne_current = s%a_ne_active
         h_ne_current = s%h_ne_active

!-------------------------------------------------------------------------------
!
      END DO pre   !       End of the main presolving loop
!
!-------------------------------------------------------------------------------

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * )                                                     &
              ' ======== end of the main processing loop ( loop =', s%npass,   &
              ') ========'
         WRITE( s%out, * ) ' '
      END IF

      IF ( s%npass >= control%max_nbr_passes ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '   maximum number of analysis passes reached'
      END IF

!-------------------------------------------------------------------------------

!     End of the analysis phase

!-------------------------------------------------------------------------------

1000  CONTINUE

!-------------------------------------------------------------------------------

!     Enforce the user's convention on the sign of the multipliers and dual
!     variables.

!-------------------------------------------------------------------------------

      IF ( control%z_sign /= POSITIVE ) THEN
         DO j = 1, prob%n
            prob%Z( j )   = - prob%Z( j )
            tmp           = - prob%Z_l( j )
            prob%Z_l( j ) = - prob%Z_u( j )
            prob%Z_u( j ) = tmp
         END DO
      END IF
      IF ( control%y_sign /= POSITIVE ) THEN
         DO i = 1, prob%m
            prob%Y( i )   = - prob%Y( i )
            tmp           = - prob%Y_l( i )
            prob%Y_l( i ) = - prob%Y_u( i )
            prob%Y_u( i ) = tmp
         END DO
      END IF

!-------------------------------------------------------------------------------

!     Update the transformation counts and save the transformations remaining in
!     memory, if previous transformations were already saved

!-------------------------------------------------------------------------------

!     If tt > tm, then some transformations have been saved.
!     The transformations currently in memory must then be saved too.

      IF ( s%tt > s%tm ) THEN
         s%ts = s%tt - s%tm
         IF ( s%ts > 0 ) THEN
            CALL PRESOLVE_save_transf
            CLOSE( control%transf_file_nbr, STATUS = 'KEEP' )
         END IF

!     Else, (tt == tm),  no transformation was ever saved,
!     and therefore ts = 0.

      ELSE
         s%ts = 0
      END IF

!-------------------------------------------------------------------------------

!     Finalize the information that is passed back to the user at exit

!-------------------------------------------------------------------------------

!     Pass the total number of transformations (for user information).

      inform%nbr_transforms = s%tt

!     Modify the error status if caused by reaching the maximum number of
!     transformations, because permutation can nevertheless be applied in
!     this case.

      IF ( inform%status == MAX_NBR_TRANSF_TMP ) inform%status = MAX_NBR_TRANSF

!     Define the PRESOLVE stage to indicate that analysis has been completed.

      IF ( s%n_active > 0 ) THEN
         s%stage = ANALYZED
      ELSE
         s%stage = FULLY_REDUCED
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_analyze

! =============================================================================
! =============================================================================

      SUBROUTINE PRESOLVE_permute

!     Guess values for the active variables and multipliers and permutes
!     the active subproblem resulting from the analysis (if the problem has
!     not been fully reduced).
!
!     Note: At the end of the routine, we have that
!           x_status( j ) = jj means that jj is the position of the original
!                           variable j in the permuted structure.
!           c_status( i ) = ii means that ii is the position of the original
!                           row i in the permuted structure.
!           a_perm and h_perm are the associated permutations for A and H
!                           but may also contain additional information on
!                           the row structure of the original matrix which is
!                           represented by negative shifts of the permutation
!                           indices.

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER :: i, j, smt_stat

!     Print banner

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *         GALAHAD PRESOLVE for QPs         *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *            problem permutation           *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' '
      END IF

!-------------------------------------------------------------------------------

!                        Initial verifications

!-------------------------------------------------------------------------------

!     Review the control structure.

!     CALL PRESOLVE_revise_control( PERMUTE, prob, control, inform, s )

!     Check if any analysis was done before.

      IF ( s%stage /= ANALYZED      .AND. &
           s%stage /= FULLY_REDUCED       ) THEN
         inform%status = PROBLEM_NOT_ANALYZED
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: the problem structure has not been analyzed'
         WRITE( inform%message( 2 ), * )                                       &
              '    before an attempt is made to permute it'
         RETURN
      END IF

!     Check that the problem is not fully reduced ( ie n_active = 0 ).

      IF ( s%stage /= FULLY_REDUCED ) THEN

!        Guess values for x, z, c, y, and q.

         IF ( s%level >= TRACE ) WRITE( s%out, * )                             &
            ' guessing values for x, y, z, c and q'
         DO j = 1, prob%n
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
            CALL PRESOLVE_guess_x( j, prob%X(j), prob, s )
            CALL PRESOLVE_guess_z( j, prob%Z(j), prob, s )
         END DO
         IF ( s%level >= DETAILS ) WRITE( s%out, * )                           &
              '   values assigned to x and z'
         CALL PRESOLVE_guess_y( prob, s )
         CALL PRESOLVE_compute_q( prob )
         CALL PRESOLVE_compute_c( .TRUE., prob, s )
         IF ( s%level >= DETAILS ) WRITE( s%out, * )                           &
              '   values assigned to c, y and q'
         IF ( s%level >= DEBUG )                                               &
            CALL PRESOLVE_write_full_prob( prob, control, s )

!-------------------------------------------------------------------------------

!        Switch to non-degenerate bounds or loosest, if requested and if
!        any transformation was performed.

!-------------------------------------------------------------------------------

         IF ( s%tt > 0 ) THEN

!           Bounds on x

            IF ( control%final_x_bounds /= TIGHTEST ) THEN
               IF ( s%level >= TRACE ) THEN
                  IF ( control%final_x_bounds == NON_DEGENERATE ) THEN
                     WRITE( s%out, * )                                         &
                          ' swapping tightest and non-degenerate bounds on x'
                  ELSE
                     WRITE( s%out, * )                                         &
                          ' swapping tightest and loosest bounds on x'
                  END IF
               END IF
               CALL PRESOLVE_swap( prob%n, s%x_l2, prob%X_l )
               CALL PRESOLVE_swap( prob%n, s%x_u2, prob%X_u )
            END IF

!           Bounds on z

            IF ( control%final_z_bounds /= TIGHTEST ) THEN
               IF ( s%level >= TRACE ) THEN
                  IF ( control%final_z_bounds == NON_DEGENERATE ) THEN
                     WRITE( s%out, * )                                         &
                          ' swapping tightest and non-degenerate bounds on z'
                  ELSE
                     WRITE( s%out, * )                                         &
                          ' swapping tightest and loosest bounds on z'
                  END IF
               END IF
               CALL PRESOLVE_swap( prob%n, s%z_l2, prob%Z_l )
               CALL PRESOLVE_swap( prob%n, s%z_u2, prob%Z_u )
            END IF

!           Bounds on c

            IF ( prob%m > 0 ) THEN
               IF ( control%final_c_bounds /= TIGHTEST  ) THEN
                  IF ( s%level >= TRACE ) THEN
                     IF ( control%final_c_bounds == NON_DEGENERATE ) THEN
                        WRITE( s%out, * )                                      &
                             ' swapping tightest and non-degenerate bounds on c'
                     ELSE
                        WRITE( s%out, * )                                      &
                            ' swapping tightest and loosest bounds on c'
                     END IF
                  END IF
                  CALL PRESOLVE_swap( prob%m, s%c_l2, prob%C_l )
                  CALL PRESOLVE_swap( prob%m, s%c_u2, prob%C_u )
               END IF

!           Bounds on y

               IF ( control%final_z_bounds /= TIGHTEST ) THEN
                  IF ( s%level >= TRACE ) THEN
                     IF ( control%final_y_bounds == NON_DEGENERATE ) THEN
                        WRITE( s%out, * )                                      &
                             ' swapping tightest and non-degenerate bounds on y'
                     ELSE
                        WRITE( s%out, * )                                      &
                             ' swapping tightest and loosest bounds on y'
                     END IF
                  END IF
                  CALL PRESOLVE_swap( prob%m, s%y_l2, prob%Y_l )
                  CALL PRESOLVE_swap( prob%m, s%y_u2, prob%Y_u )
               END IF
            END IF
         END IF

!-------------------------------------------------------------------------------

!              Now permute the reduced problem to standard form.

!-------------------------------------------------------------------------------

!        Find the new order for the variables.

         IF ( s%level >= TRACE ) WRITE( s%out, * ) ' reordering the variables'
         CALL PRESOLVE_reorder_variables
         IF (s%level >= DEBUG ) THEN
             WRITE( s%out, * ) '    x permutation '
             DO j = 1, prob%n
                WRITE( s%out, * )  '    ', j, ' --> ', prob%X_status( j )
             END DO
         END IF

!        Find the new order for the constraints.

         IF ( prob%m > 0 ) THEN
            IF ( s%level >= TRACE ) WRITE( s%out, * )                          &
               ' reordering the constraints'
            CALL PRESOLVE_reorder_constraints
            IF ( s%level >= DEBUG ) THEN
                WRITE( s%out, * ) '    c permutation '
                DO i = 1, prob%m
                   WRITE( s%out, * )  &
                        '    ', i, ' --> ', prob%C_status( i )
                END DO
            END IF
         END IF

!       Perform final resulting permutations.

         IF ( s%level >= TRACE ) WRITE( s%out, * )                             &
            ' building the reduced problem'
         CALL PRESOLVE_apply_permutations
         s%stage = PERMUTED

!        Mark the resulting matrices as stored in row-wise sparse format.

         IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
         CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
         IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
         CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )

!        Write the problem if requested.

         IF ( s%level >= DEBUG ) CALL QPT_write_problem( s%out, prob )

!     All variables have been eliminated (the problem is fully reduced)!
!     In this case, the resulting problem is empty (and its solution trivial).

      ELSE
         IF ( s%level >= TRACE ) WRITE( s%out, * )                             &
            ' all variables and constraints have been eliminated!'
         prob%n    = 0
         prob%m    = 0
         IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
         CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
         IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
         CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
         s%stage     = FULLY_REDUCED
         IF ( control%get_q ) prob%q = prob%f
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_permute

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_primal_constraints( analysis_level )

!     Processes primal constraints in search of empty, singleton, redundant
!     or forcing constraints.  Also updates the bounds on the variables
!     whenever possible using the values of the derived implied bounds.

!     Argument:

      INTEGER, INTENT( IN ) :: analysis_level

!            the level of analysis requested.  Possible values are:
!
!            EMPTY_AND_SINGLETON_ROWS: the analysis is restricted to the
!                                      elimination of empty and singleton rows;
!            PRIMAL_CONSTRAINTS      : the analysis also includes bounds
!                                      tightening and constraint fixing when
!                                      possible.

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: i, k, j, iu, il, iu_k, il_k, l, ic, xfx, xst, kk
      REAL ( KIND = wp ) :: v, xlj, xuj, cli, cui, imp_low, imp_up,            &
                            cili, ciui, aij, nlj, nuj, nlil, nuil,             &
                            nliu, nuiu
      LOGICAL            :: lower_active, upper_active

!     Check to see if there is any primal constraint left

      IF ( s%m_active == 0 ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    no constraint left'
         RETURN
      END IF

!     Loop on the active rows.

rows: DO i = 1, prob%m
         IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE

         cli = prob%C_l( i )
         cui = prob%C_u( i )

         SELECT CASE ( s%A_row_s( i ) )

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

!                              Empty row

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

         CASE ( 0 )

            IF ( s%level >= DETAILS ) WRITE( s%out,* ) '   empty row', i

!           1) If the left- and right-hand sides are not coherent, the problem
!              is infeasible.

            IF ( cui < -EPSMACH .OR. cli > EPSMACH ) THEN
               inform%status = PRIMAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE analysis stopped: the problem is primal infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                  '    because the bounds on c(', i, ') are incompatible'
               RETURN

!           2) If not,remove the constraint and fix the multiplier to zero.

            ELSE
               CALL PRESOLVE_remove_c( i, Y_GIVEN, yval = ZERO )
               IF ( inform%status /= OK ) RETURN
            END IF

!           Note that the value of the corresponding dual variables are
!           irrelevant and can be assumed to be zero. They should therefore
!           not be remembered.

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!                              Singleton row
!
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

         CASE ( 1 )

!           First verify that enough transformations are allowed to handle
!           a singleton row.

            IF ( s%tt + 4 <= control%max_nbr_transforms ) THEN

               IF ( s%level >= DETAILS ) THEN
                  WRITE( s%out,* ) '   singleton row ', i
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    c(', i, ') in [', cli, ',', cui, ']'
               END IF

!              ----------------------------------
!              Find the singleton row parameters.
!              ----------------------------------

!              Find the value of the unique non-zero entry in the row (aij)
!              its position in A (k) and the corresponding original
!              column index (j).

               ic = i
lic:           DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     j = prob%A%col( k )
                     IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
                     aij = prob%A%val( k )
                     kk  = k
                     IF ( aij /= ZERO ) EXIT lic
                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO lic

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    nonzero in column', j, ' and of value', aij

               IF ( ABS( aij ) < s%a_tol ) THEN
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    too small element in singleton row', i
                  CYCLE rows
               END IF

!              Remember the values of the bounds on x(j) before
!              their modification.

               xlj = prob%X_l( j )
               xuj = prob%X_u( j )

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    x(', j, ') in [', xlj, ',',  xuj, ']'

!              Compute the new bounds, according to the sign of the coefficient.

               IF ( aij >= s%a_tol ) THEN
                  IF ( cui < s%P_INFINITY ) THEN
                     nuj = cui / aij
                  ELSE
                     nuj = s%INFINITY
                  END IF
                  IF ( cli > s%M_INFINITY ) THEN
                     nlj = cli / aij
                  ELSE
                     nlj = - s%INFINITY
                  END IF
               ELSE IF ( aij <= - s%a_tol ) THEN
                  IF ( cli > s%M_INFINITY ) THEN
                     nuj = cli / aij
                  ELSE
                     nuj = s%INFINITY
                  END IF
                  IF ( cui < s%P_INFINITY ) THEN
                     nlj = cui / aij
                  ELSE
                     nlj = - s%INFINITY
                  END IF
               END IF

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                   '    => x(', j, ') in [', nlj, ',', nuj, ']'

!              -------------------------
!              Update the bounds on x(j)
!              -------------------------

!              1) Consider first the case where the lower and upper bounds are
!                 equal, in which case x(j) may be fixed (this happens, for
!                 instance, if row i is an equality constraint).
!                 Note that in this case the final z(j) must be zero (since we
!                 may ignore the effect of a previous bound on x(j) that would
!                 be exactly the same as nlj = nuj).  This simplifies the dual
!                 equation.  However, also note that z(j) for the reduced
!                 problem need not being zero, as it is translated, at
!                 restoration, into a value for y(i).

               IF ( nlj > s%M_INFINITY              .AND. &
                    nuj < s%P_INFINITY              .AND. &
                    PRESOLVE_is_zero( nlj - nuj )       ) THEN

                  v = HALF * ( nlj + nuj )
                  IF ( PRESOLVE_is_zero( v - xlj ) ) v = xlj
                  IF ( PRESOLVE_is_zero( xuj - v ) ) v = xuj

!                 Check primal feasibility.

                  IF ( PRESOLVE_is_pos( xlj - v ) .OR. &
                       PRESOLVE_is_pos( v - xuj )      )THEN
                     inform%status = PRIMAL_INFEASIBLE
                     WRITE( inform%message( 1 ), * )                           &
                         ' PRESOLVE analysis stopped:',                        &
                         ' the problem is primal infeasible'
                     WRITE( inform%message( 2 ), * )                           &
                          '    because the bounds on x(', j,                   &
                          ') are incompatible'
                     RETURN
                  END IF

!                 Deactivate the singleton row, as it will be replaced
!                 by bounds on the associated variable

                  CALL PRESOLVE_remove_c( i, Y_FROM_Z_BOTH, pos = kk )
                  IF ( inform%status /= OK ) RETURN

!                 Now fix the x(j)

                  CALL PRESOLVE_fix_x( j, v, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN

!              The upper and lower bounds are different.

               ELSE

!                 Determine which of the new bound(s) is(are) active, in that
!                 they are stronger than the previous bounds on x(j)

                  lower_active = nlj >= xlj .AND. nlj > s%M_INFINITY
                  upper_active = nuj <= xuj .AND. nuj < s%P_INFINITY

!                 Remember the status of x(j) before the action due to
!                 the singleton row has taken place.

                  xst = prob%X_status( j )

!                 2) Both the lower and upper bounds are active.

                  IF ( lower_active .AND. upper_active ) THEN

                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    both lower and upper bounds are active'

!                    Deactivate the row, as it will be replaced by bounds on the
!                    associated variable.

                     CALL PRESOLVE_remove_c( i, Y_FROM_Z_BOTH, pos = kk )
                     IF ( inform%status /= OK ) RETURN

!                    Tighten the lower bound.

                     IF ( PRESOLVE_is_zero( nlj - xuj ) ) THEN
                        CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
                        CYCLE rows
                     ELSE
                        CALL PRESOLVE_set_bound_x( j, LOWER, nlj )
                     END IF
                     IF ( inform%status /= OK ) RETURN

!                    Tighten the upper bound.

                     IF ( PRESOLVE_is_zero( nuj - xlj ) ) THEN
                        CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
                        CYCLE rows
                     ELSE
                        CALL PRESOLVE_set_bound_x( j, UPPER, nuj )
                     END IF
                     IF ( inform%status /= OK ) RETURN

!                 3) lower bound only

                  ELSE IF ( lower_active ) THEN

                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    the lower bound is active'

!                    Deactivate the singleton row, as it will be replaced
!                    by bounds on the associated variable.

                     CALL PRESOLVE_remove_c( i, Y_FROM_Z_LOW, pos = kk )
                     IF ( inform%status /= OK ) RETURN

!                    Tighten the lower bound to reflect the singleton
!                    constraint.

                     IF ( PRESOLVE_is_zero( nlj - xuj ) ) THEN
                        CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
                     ELSE
                        CALL PRESOLVE_set_bound_x( j, LOWER, nlj )
                     END IF
                     IF ( inform%status /= OK ) RETURN

!                 4) upper bound only

                  ELSE IF ( upper_active ) THEN

                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    the upper bound is active'

!                    Deactivate the singleton row, as it will be replaced
!                    by bounds on the associated variable.

                     CALL PRESOLVE_remove_c( i, Y_FROM_Z_UP, pos = kk )
                     IF ( inform%status /= OK ) RETURN

!                    Tighten the upper bound to reflect the singleton
!                    constraint.

                     IF ( PRESOLVE_is_zero( nuj - xlj ) ) THEN
                        CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
                     ELSE
                        CALL PRESOLVE_set_bound_x( j, UPPER, nuj )
                     END IF
                     IF ( inform%status /= OK ) RETURN

!                 5 ) Neither the lower nor the upper bounds are active:
!                     the constraint is therefore redundant and inactive.

                  ELSE

                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    no bound is active'

!                    Remove the redundant singleton constraint.

                     CALL PRESOLVE_remove_c( i, Y_GIVEN, yval = ZERO )
                     IF ( inform%status /= OK ) RETURN

                  END IF

!                 Readjust the value of z(j).

                  SELECT CASE ( prob%X_status( j ) )
                  CASE ( RANGE )
                     IF ( xst == LOWER .OR. xst == FREE ) THEN
                        CALL PRESOLVE_bound_z( j, LOWER, SET, - s%INFINITY )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     IF ( xst == UPPER .OR. xst == FREE ) THEN
                        CALL PRESOLVE_bound_z( j, UPPER, SET,   s%INFINITY )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                  CASE ( LOWER )
                     IF ( xst == FREE ) THEN
                        CALL PRESOLVE_bound_z( j, UPPER, SET,   s%INFINITY )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                  CASE ( UPPER )
                     IF ( xst == FREE ) THEN
                        CALL PRESOLVE_bound_z( j, LOWER, SET, - s%INFINITY )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                  END SELECT

               END IF
            END IF

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!
!                          General primal constraint
!
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

         CASE ( 2: )
            IF ( analysis_level == EMPTY_AND_SINGLETON_ROWS ) CYCLE

            IF ( s%level >= DETAILS ) WRITE( s%out, * )                        &
               '   row', i, 'of size', s%A_row_s( i ),                         &
               '(out of', s%m_active, 'active rows)'

!           ------------------------------------------------
!           Build the implied bounds for the i-th constraint
!           ------------------------------------------------

!           Initialize the accumulators for the implied bounds.
!           The variable iu is intended to be 0 if no infinite variable
!           upper bound was found for row i, or to be k if the variable
!           associated with the entry in position k is the only one to have
!           an infinite upper bound, or to be -1 in all other cases. The
!           variable il plays the same role for infinite lower bounds.

            imp_low = ZERO
            imp_up  = ZERO
            il      = 0
            iu      = 0

!           Now accumulate the implied bounds by looping on all variables
!           occurring in the i-th constraint.

            IF ( s%level >= DETAILS ) THEN
               WRITE( s%out, * )                                               &
                    '   accumulating implied bounds for c(', i, ')'
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '        j    A(i,j)    x_l(j)    x_u(j)    imp_low',        &
                  '    imp_up     il     iu'
            END IF

            ic = i
            DO
               DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  j   = prob%A%col( k )
                  IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
                  CALL PRESOLVE_implied_bounds( k, prob%A%val( k ),            &
                                             prob%X_l( j ), prob%X_u( j ),     &
                                             imp_low, imp_up, il, iu      )
                  IF ( s%level >= DEBUG ) WRITE( s%out, 1000 )                 &
                          j, prob%A%val( k), prob%X_l( j ), prob%X_u( j ),     &
                          imp_low, imp_up, il, iu
                  IF ( il == k ) il_k = j
                  IF ( iu == k ) iu_k = j
                  IF ( il < 0 .AND. iu < 0 ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    too many infinite bounds on y'
                     CYCLE rows
                  END IF
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO

!           Correct the implied bounds for the possible occurrence of
!           infinite bounds.

            IF ( s%level >= DEBUG .AND. ( il /= 0 .OR. iu /= 0 ) )             &
                  WRITE( s%out, * )                                            &
                  '    correcting implied bounds for infinite values ',        &
                  '( il =', il, 'iu =', iu, ')'

            IF ( il == 0 ) THEN
               cili = imp_low
            ELSE
               cili = - s%INFINITY
            END IF
            IF ( il <= 0 ) il_k = 0

            IF ( iu == 0 ) THEN
               ciui = imp_up
            ELSE
               ciui = s%INFINITY
            END IF
            IF ( iu <= 0 ) iu_k = 0

!           The implied bounds are now available for row i.

            IF ( s%level >= DETAILS ) THEN
               WRITE( s%out, 1001 ) cili, cli
               WRITE( s%out, 1002 ) ciui, cui
            END IF

!           ----------------------------------------------------------------
!           First eliminate the case where the problem is infeasible because
!           the implied bounds are not compatible with the original ones.
!           ----------------------------------------------------------------

            IF ( PRESOLVE_is_neg( cui - cili ) .OR. &
                 PRESOLVE_is_pos( cli - ciui )      ) THEN
               inform%status = PRIMAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE analysis stopped: the problem is primal infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                  '    because the bounds on c(', i,') are incompatible'
               RETURN
            END IF

!           ----------------------------------------------------------------
!           Now consider the cases where one of the sides of constraint i is
!           forcing because the lower/upper implied bound is equal to the
!           current upper/lower one.
!           ----------------------------------------------------------------

!           case 1: the implied lower bound is exactly equal to the
!           ------- actual upper bound and both bounds are finite.

            IF ( cui  >  s%M_INFINITY  .AND.  &
                 cili == cui           .AND.  &
                 cili <  s%P_INFINITY  .AND.  &
                 s%tt + 3 + s%A_row_s( i ) <= control%max_nbr_transforms ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    constraint', i, 'is forcing for its upper bound'

!              Remove the i-th constraint as it is always automatically
!              satisfied.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    removing c(', i, ')'
               CALL PRESOLVE_remove_c( i, Y_FROM_FORCING_UP, pos = l )
               IF ( inform%status /= OK ) RETURN

!              Note that the head of the list of fixed variable is still
!              unknown, and must be set later (when the first variable
!              is fixed).

!              Fix the variables of the row to their upper bound when
!              multiplied by a negative entry, and to their lower bound
!              when multiplied by a positive entry.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing the variables in c(', i, ')'

               ic = i
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     j  = prob%A%col( k )
                     IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
                     v = prob%A%val( k )
                     IF ( v > ZERO ) THEN
                        CALL PRESOLVE_fix_x( j, prob%X_l(j), Z_FROM_DUAL_FEAS )
                        CALL PRESOLVE_add_at_end( j, s%hist_j( l ), xfx )
                     ELSE IF ( v < ZERO ) THEN
                        CALL PRESOLVE_fix_x( j, prob%X_u(j), Z_FROM_DUAL_FEAS )
                        CALL PRESOLVE_add_at_end( j, s%hist_j( l ), xfx )
                     END IF
                     IF ( inform%status /= OK ) RETURN
                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO

               CYCLE

!           case 2: the implied upper bound is exactly equal to the
!           ------- actual lower bound and both bounds are finite.

            ELSE IF ( -s%INFINITY <  cli          .AND.    &
                             ciui == cli          .AND.    &
                             ciui <  s%P_INFINITY .AND.    &
                       s%tt+1+s%A_row_s(i) <= control%max_nbr_transforms ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c(', i, ') is forcing for its lower bound'

!              Remove the i-th constraint as it is always automatically
!              satisfied.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    removing c(', i, ')'
               CALL PRESOLVE_remove_c( i, Y_FROM_FORCING_LOW, pos = l  )
               IF ( inform%status /= OK ) RETURN

!              Fix the variables of the row to their upper bound when
!              multiplied by a positive entry, and to their lower bound
!              when multiplied by a negative entry.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing the variables in c(', i, ')'

               ic = i
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     j  = prob%A%col( k )
                     IF ( prob%X_status( j ) > ELIMINATED ) THEN
                        v = prob%A%val( k )
                        IF ( v < ZERO ) THEN
                           CALL PRESOLVE_fix_x( j, prob%X_l( j ),              &
                                                 Z_FROM_DUAL_FEAS )
                           CALL PRESOLVE_add_at_end( j, s%hist_j( l ), xfx )
                        ELSE IF ( v > ZERO ) THEN
                           CALL PRESOLVE_fix_x( j, prob%X_u( j ),              &
                                                 Z_FROM_DUAL_FEAS )
                           CALL PRESOLVE_add_at_end( j, s%hist_j( l ), xfx )
                        END IF
                        IF ( inform%status /= OK ) RETURN
                     END IF
                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO

               CYCLE

            END IF

!           ----------------------------------------------------------------
!           Now consider the cases where one side of constraint i may be
!           relaxed because the associated implied bound is tighter than
!           the current one.
!           ----------------------------------------------------------------

!           case 0: the constraint is redundant: it can be removed
!           ------  from the problem.

            IF ( s%M_INFINITY < cli .AND. cli < cili  .AND. &
                 ciui < cui .AND. cui < s%P_INFINITY        ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * ) &
                  '    c(', i, ') is redundant'
               CALL PRESOLVE_remove_c( i, Y_GIVEN, yval = ZERO )
               IF ( inform%status /= OK ) RETURN
               CYCLE
            END IF

!           case 1: the implied lower bound is larger than the actual one,
!           ------  which is finite: the lhs of the constraint can be set
!                   to -infinity.

            IF ( s%M_INFINITY < cli .AND. cli < cili ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c(', i, ') is dominated below'
               CALL PRESOLVE_bound_c( i, LOWER, SET, -s%INFINITY )
            END IF

!           case 2: the implied upper bound is larger than the actual one,
!           ------  which is finite: the rhs of the constraint can be set
!                   to + infinity.

            IF ( ciui < cui .AND. cui < s%P_INFINITY ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c(', i, ') is dominated above'
               CALL PRESOLVE_bound_c( i, UPPER, SET, s%INFINITY )
            END IF

!           ----------------------------------------------------------------
!           Now consider the case where the implied bounds are inconclusive
!           as far the complete constraint is concerned.  However, further
!           improvement of the variables' bounds are still possible.
!           ----------------------------------------------------------------

!           No improvement is possible if both implied bounds are infinite
!           and contain more that a single infinite term: cycle to consider
!           the next primal constraint.

            IF ( il < 0 .AND. iu < 0 ) CYCLE

!           Some improvement is not excluded for each variable, as at least
!           one of the implied bounds is finite: loop on the variables
!           of the i-th constraint after ensuring that the resulting
!           transformations can be stored.

            IF ( s%tt + 2 > control%max_nbr_transforms ) CYCLE

            IF ( il == 0 .OR. iu == 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using finite implied bounds for variables in c(', i, ')'

               ic = i
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     j = prob%A%col( k )
                     IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
                     aij = prob%A%val( k )
                     IF ( aij == ZERO ) CYCLE
                     xlj = prob%X_l( j )
                     xuj = prob%X_u( j )
                     nlj = - s%INFINITY
                     nuj =   s%INFINITY

!                    Check if the lower bound can be used.

                     IF ( cui < s%P_INFINITY ) THEN
                        IF ( il == 0 ) THEN
                          IF ( aij <= - s%a_tol ) THEN
                              nlj = MAX( nlj, xuj + ( cui - imp_low ) / aij )
                           ELSE IF ( aij >= s%a_tol ) THEN
                              nuj = MIN( nuj, xlj + ( cui - imp_low ) / aij )
                           END IF
                        ELSE IF ( il_k == j ) THEN
                           IF ( aij <= - s%a_tol ) THEN
                              nlj = MAX( nlj, ( cui - imp_low ) / aij )
                           ELSE IF ( aij >= s%a_tol ) THEN
                              nuj = MIN( nuj, ( cui - imp_low ) / aij )
                           END IF
                        END IF
                     END IF

!                    Check if the upper bound can be used.

                     IF ( cli > s%M_INFINITY ) THEN
                        IF ( iu == 0 ) THEN
                           IF ( aij <= - s%a_tol ) THEN
                              nuj = MIN( nuj, xlj + ( cli - imp_up  ) / aij )
                           ELSE IF ( aij >= s%a_tol ) THEN
                              nlj = MAX( nlj, xuj + ( cli - imp_up  ) / aij )
                           END IF
                        ELSE IF ( iu_k == j ) THEN
                           IF ( aij <= - s%a_tol ) THEN
                              nuj = MIN( nuj, ( cli - imp_up ) / aij )
                           ELSE IF ( aij >= s%a_tol ) THEN
                              nlj = MAX( nlj, ( cli - imp_up ) / aij )
                           END IF
                        END IF
                     END IF

!                    Update the bounds on x(j).

                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    x(', j, ') must lie in [', nlj, ',', nuj, ']'

                     CALL PRESOLVE_primal_bound_x( j, k, LOWER, nlj )
                     IF ( inform%status /= OK ) RETURN
                     CALL PRESOLVE_primal_bound_x( j, k, UPPER, nuj )
                     IF ( inform%status /= OK ) RETURN

                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO


!           Consider the case where one or both the implied bounds only
!           contain a single infinite contribution.

            ELSE

!              There is more than one infinite contribution in the implied
!              bounds.

               IF ( il > 0 ) THEN
                  IF ( s%level >= DETAILS ) WRITE( s%out, * )                  &
                     '   trying to use the implied lower bound for',           &
                     ' the single x(', il_k,') in c(', i, ')'
                  aij = prob%A%val( il )
                  IF ( aij >= s%a_tol ) THEN
                     nuil = ( cui - imp_low ) / aij
                     CALL PRESOLVE_primal_bound_x( il_k, il, UPPER, nuil )
                  ELSE IF ( aij <= - s%a_tol ) THEN
                     nlil = ( cui - imp_low ) / aij
                     CALL PRESOLVE_primal_bound_x( il_k, il, LOWER, nlil )
                  END IF
                  IF ( inform%status /= OK ) RETURN
               END IF

               IF ( iu > 0 ) THEN
                  IF ( s%level >= DETAILS ) WRITE( s%out, * )                  &
                     '   trying to use the implied upper bound for',           &
                     ' the single x(', iu_k,') in c(', i, ')'
                  aij = prob%A%val( iu )
                  IF ( aij >= s%a_tol ) THEN
                     nliu = ( cli - imp_up ) / aij
                     CALL PRESOLVE_primal_bound_x( iu_k, iu, LOWER, nliu )
                  ELSE IF ( aij <= - s%a_tol ) THEN
                     nuiu = ( cli - imp_up ) / aij
                     CALL PRESOLVE_primal_bound_x( iu_k, iu, UPPER, nuiu )
                  END IF
                  IF ( inform%status /= OK ) RETURN
               END IF

            END IF

!        End the selection on the type of row.

         END SELECT

!  End of the loop on the rows

      END DO rows

      RETURN

!     Debugging formats

1000  FORMAT( 4x, i6, 1x, 5ES10.2, 1x, i6, 1x, i6 )
1001  FORMAT( '    bounds: implied lower =',ES12.4,'  current lower =', ES12.4 )
1002  FORMAT( '    bounds: implied upper =',ES12.4,'  current upper =', ES12.4 )

      END SUBROUTINE PRESOLVE_primal_constraints

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_linear_tupletons

!     Processes the special linear columns: unconstrained, singletons and
!     doubletons. Handling an unconstrained column is given priority over
!     handling a singleton column, which is itself given priority to handling
!     a linear doubleton column.
!     For singletons or doubletons columns, freeing the considered variable
!     using a split equality is allowed in two cases:
!     - the index of the current presolving loop is at least equal to the
!       index of the first preprocessing loop at which freeing the
!       singleton/doubleton variable using a split equality is allowed,
!       irrespective of the possibility to eliminate other singleton/doubleton
!       columns
!     - the elimination of singleton/doubleton columns has been unsuccessful
!       without resorting to split equalities, in which case a second pass over
!       the special linear columns is performed, allowing split equalities for
!       the so far unsuccessful singleton/doubleton transformation(s). This
!       is only possible for presolving loops beyond the first one.

!     Tuning parameters:

!     1) the index of the first preprocessing loop at which freeing the
!        singleton variable using a split equality is allowed irrespective
!        of the possibility to eliminate other singleton columns

      INTEGER, PARAMETER :: first_singleton_split_loop = 3

!     2) the index of the first preprocessing loop at which freeing the
!        doubleton variable using a split equality is allowed irrespective
!        of the possibility to eliminate other doubleton columns

      INTEGER, PARAMETER :: first_doubleton_split_loop = 3

!     Programming: Ph. Toint, May 2001.
!
!===============================================================================

!     Local variables

      INTEGER :: j, pass, nred_s, nred_d, ttbef
      LOGICAL :: split_s, split_d, prev_split_s, prev_split_d, do_singletons,  &
                 do_doubletons, do_unconstrained

!     Determine which transformations may be applied at the current loop
!     as a function of the user specified frequency of their application
!     and the value of the index of the current presolving loop.

      pass = s%npass + 1
      do_unconstrained = .FALSE.
      IF ( control%unc_variables_freq > 0 )                                    &
         do_unconstrained = MOD( pass, control%unc_variables_freq  ) == 0
      do_singletons = .FALSE.
      IF ( control%singleton_columns_freq > 0 .AND. prob%m > 0 )               &
         do_singletons = MOD( pass, control%singleton_columns_freq ) == 0
      do_doubletons = .FALSE.
      IF ( control%doubleton_columns_freq > 0 .AND. prob%m > 0 )               &
         do_doubletons = MOD( pass, control%doubleton_columns_freq ) == 0

!     Determine if equality splitting is allowed at this stage.

      split_s = s%loop >= first_singleton_split_loop  ! for linear singletons
      split_d = s%loop >= first_doubleton_split_loop  ! for linear doubletons

!     Initialize the success counter for the first pass.

      nred_s = 0                                    ! for linear singletons
      nred_d = 0                                    ! for linear doubletons

!  -----------------------------------------------------------------------------
!  Loop on the linked lists of linear singleton, doubleton and unconstrained
!  columns
!  -----------------------------------------------------------------------------

!     There are at most two passes, the second potentially allowing split
!     equalities if they were not allowed during the first.

      DO pass = 1, 2

!        build the linked lists for the three categories of special linear
!        variables (linear singletons, linear doubletons, linearly
!        unconstrained).

         CALL PRESOLVE_get_x_lists( s )

!        Loop over the successive columns in the three lists of special
!        linear columns.

         DO

!          Obtain the next unconstrained variable.

            IF ( s%unc_f /= END_OF_LIST ) THEN

!              Remove the column from the list of linearly unconstrained
!              variables.

               j = s%unc_f
               s%unc_f = s%h_perm( j )
               s%h_perm( j ) = 0

!             Avoid inactive columns or columns that are no longer
!             unconstrained.

               IF ( prob%X_status(j) > ELIMINATED .AND. &
                    s%A_col_s(j) == 0             .AND. &
                    do_unconstrained                    )                      &
                  CALL PRESOLVE_unconstrained( j )

!           Obtain the next linear singleton.

            ELSE IF ( s%lsc_f /= END_OF_LIST ) THEN

!              Remove the column from the list of linear singletons.

               j = s%lsc_f
               s%lsc_f = s%h_perm( j )
               s%h_perm( j ) = 0

!              Avoid inactive columns or columns that are no longer singletons.

               IF ( prob%X_status(j) > ELIMINATED .AND. &
                    s%A_col_s(j) == 1             .AND. &
                    do_singletons                       ) THEN
                  ttbef = s%tt
                  CALL PRESOLVE_linear_singleton( j, split_s )
                  nred_s = nred_s + s%tt - ttbef
               END IF

!           Obtain the next linear doubleton.

            ELSE IF ( s%ldc_f /= END_OF_LIST ) THEN

!              Remove the column from the list of linear doubletons.

               j = s%ldc_f
               s%ldc_f = s%h_perm( j )
               s%h_perm( j ) = 0

!              Avoid inactive columns or columns that are no longer doubletons.

               IF ( prob%X_status(j) > ELIMINATED .AND. &
                    s%A_col_s(j) == 2             .AND. &
                    do_doubletons                       ) THEN
                  ttbef = s%tt
                  CALL PRESOLVE_linear_doubleton( j, split_d )
                  nred_d = nred_d + s%tt - ttbef

               END IF

!           All lists are now empty: this is the end of the current pass.

            ELSE

               EXIT

            END IF

         END DO

!        Check for errors.

         IF ( inform%status /= OK ) RETURN
         IF ( s%tt >= control%max_nbr_transforms ) THEN
            inform%status = MAX_NBR_TRANSF
            EXIT
         END IF

!        If this is the end of the second pass or if the current presolving
!        loop is the first, then stop considering the special linear columns.

         IF ( pass == 2 .OR. s%loop == 1 ) EXIT

!        If no reduction was obtained for either singletons or doubletons
!        try allowing split equalities in another pass provided this is
!        not the first presolving loop.

         prev_split_s = split_s
         split_s = s%loop >= first_singleton_split_loop .OR. nred_s == 0
         prev_split_d = split_d
         split_d = s%loop >= first_doubleton_split_loop .OR. nred_d == 0

!        Exit if split equalities were already allowed at the finished pass
!        or if they are still not allowed for the next.

         IF ( ( prev_split_s .AND. prev_split_d ) .OR. &
             .NOT. ( split_s .OR. split_d )            )  EXIT

      END DO

      RETURN

      END SUBROUTINE PRESOLVE_linear_tupletons

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_dual_constraints

!     Exploit the dual feasibility condition
!
!     g + H x - A^T y - z = 0    ie    A^T y - Hx = g - z
!
!     to simplify the dual constraints (forcing) or to update the bounds
!     on the variables and multipliers when possible (using implied bounds).
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: k, i, ii, j, a_il, a_iu, h_il, h_iu,               &
                            h_il_k, h_iu_k, a_il_k, a_iu_k, hsj, csj
      LOGICAL            :: uf, lf
      REAL ( KIND = wp ) :: gj, imp_low, imp_up, xlj, xuj,   v, hij,           &
                            aij, yli, yui, nli, nui, xui, xli, zlj, zuj

!     Loop on all active columns.

vars: DO j = 1, prob%n
         IF ( prob%X_status( j ) <= ELIMINATED  ) CYCLE

!        Get the associated gradient component, bounds and dimensions.

         gj  = prob%G( j )
         xlj = prob%X_l( j )
         xuj = prob%X_u( j )
         zlj = prob%Z_l( j )
         zuj = prob%Z_u( j )
         csj = s%A_col_s( j )
         IF ( prob%H%ne > 0 ) THEN
            hsj = s%H_str( j )
         ELSE
            hsj = EMPTY
         END IF

         IF ( s%level >= DETAILS ) WRITE( s%out, * )                           &
            '   dual constraint', j, '( hsj =', hsj, 'csj =', csj, ')'

!        -----------------------------------------------------------------------
!        Compute the implied lower and upper bounds
!        -----------------------------------------------------------------------

!        Initialize the accumulators for the implied bounds.

!        The variable a_iu is intended to be 0 if no infinite variable upper
!        bound was found for row i of A, or to be k if the variable associated
!        with the entry in position k  is the only one to have an infinite
!        upper bound, or to be -1 in all other cases. The variable a_il plays
!        the same role for infinite lower bounds. The variables h_il and h_iu
!        play the same role for H.

         imp_low = ZERO
         imp_up  = ZERO
         h_il    = 0
         h_iu    = 0
         a_il    = 0
         a_iu    = 0

!        Now accumulate the implied bounds.
!        1) Compute the terms from the Hessian.

         IF ( hsj /= EMPTY ) THEN

!           Loop over the subdiagonal elements in the j-th row

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * )                                               &
                    '    accumulating implied bounds on subdiagonal  ',        &
                    j, 'of H'
               WRITE( s%out, * ) '        i     H(i,j)   x_l(i)',              &
                    '    x_u(i)   imp_low     imp_up    h_il   h_iu'
            END IF

            DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
               i = prob%H%col( k )
               IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
               v = - prob%H%val( k )
               IF ( v /= ZERO ) THEN
                  CALL PRESOLVE_implied_bounds( k, v, prob%X_l( i ),           &
                                                prob%X_u( i ), imp_low, imp_up,&
                                                h_il, h_iu )
                  IF ( s%level >= DEBUG ) WRITE( s%out, 1000 )                 &
                     i,-v,prob%X_l(i),prob%X_u(i), imp_low,imp_up,h_il,h_iu
                  IF ( h_il == k ) h_il_k = i
                  IF ( h_iu == k ) h_iu_k = i
                  IF ( h_il < 0 .AND. h_iu < 0 ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    too many infinite bounds on x'
                     CYCLE vars
                  END IF
               END IF
            END DO

!           Loop over the superdiagonal elements in the j-th column.

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * )                                               &
                    '    accumulating implied bounds on superdiagonal',        &
                    j, 'of H'
               WRITE( s%out, * ) '        i     H(i,j)   x_l(i)',              &
                    '    x_u(i)   imp_low     imp_up    h_il   h_iu'
            END IF
            k = s%H_col_f( j )
            IF ( END_OF_LIST /= k ) THEN
               DO ii = j + 1, prob%n
                  i  = s%H_row( k )
                  IF ( prob%X_status( i ) > ELIMINATED ) THEN
                     v = - prob%H%val( k )
                     IF ( v /= ZERO ) THEN
                        CALL PRESOLVE_implied_bounds( k, v,  prob%X_l( i ),    &
                                                      prob%X_u( i ), imp_low,  &
                                                      imp_up, h_il, h_iu  )
                        IF ( s%level >= DEBUG ) WRITE( s%out, 1000 )           &
                           i, -v, prob%X_l( i ), prob%X_u( i ),                &
                           imp_low, imp_up, h_il, h_iu
                        IF ( h_il == k ) h_il_k = i
                        IF ( h_iu == k ) h_iu_k = i
                        IF ( h_il < 0 .AND. h_iu < 0 ) THEN
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    too many infinite bounds on x'
                           CYCLE vars
                        END IF
                     END IF
                  END IF
                  k = s%H_col_n( k )
                  IF ( k == END_OF_LIST ) EXIT
               END DO
            END IF

         END IF

!        2) Compute the terms from the Jacobian.

         IF ( csj > 0 ) THEN

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * )                                               &
                    '    accumulating implied bounds on column', j, 'of A'
               WRITE( s%out, * ) '        i     A(i,j)   y_l(i)',              &
                    '    y_u(i)    imp_low    imp_up   a_il   a_iu'
            END IF

           k = s%A_col_f( j )
           IF ( END_OF_LIST /= k ) THEN
               DO ii = 1, prob%m
                  i = s%A_row( k )
                  IF ( prob%C_status( i ) > ELIMINATED ) THEN
                     v = prob%A%val( k )
                     IF ( v /= ZERO )  THEN
                        CALL PRESOLVE_implied_bounds( k, v,  prob%Y_l( i ),    &
                                                      prob%Y_u( i ), imp_low,  &
                                                      imp_up, a_il, a_iu )
                        IF ( s%level >= DEBUG ) WRITE( s%out, 1000 )           &
                           i, v, prob%Y_l( i ), prob%Y_u( i ),                 &
                           imp_low, imp_up, a_il, a_iu
                        IF ( a_il == k ) a_il_k = i
                        IF ( a_iu == k ) a_iu_k = i
                        IF ( a_il < 0 .AND. a_iu < 0 ) THEN
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    too many infinite bounds on y'
                           CYCLE vars
                        END IF
                     END IF
                  END IF
                  k = s%A_col_n( k )
                  if ( k == END_OF_LIST ) EXIT
               END DO
            END IF

         END IF

!        Make sure that the row indices are positive only if a single
!        infinite bound has been met.

         IF ( a_il <= 0 ) a_il_k = 0
         IF ( a_iu <= 0 ) a_iu_k = 0
         IF ( h_il <= 0 ) h_il_k = 0
         IF ( h_iu <= 0 ) h_iu_k = 0

!        The implied bounds are now available for the j-th dual variable.

         IF ( s%level >= DETAILS ) THEN
            IF ( a_il == 0 .AND. h_il == 0 ) THEN
               IF ( h_iu == 0 .AND. a_iu == 0 ) THEN
                  WRITE( s%out, 1001 ) imp_low, gj, imp_up
               ELSE
                  WRITE( s%out, 1001 ) imp_low, gj, s%INFINITY
               END IF
            ELSE
               IF ( h_iu == 0 .AND. a_iu == 0 ) THEN
                  WRITE( s%out, 1001 ) - s%INFINITY, gj, imp_up
               ELSE
                  WRITE( s%out, 1001 ) - s%INFINITY, gj, s%INFINITY
               END IF
            END IF
         END IF

!        -----------------------------------------------------------------------
!        First update the bounds on z(j) from the relation
!
!                      imp_low <= g(j) - z(j) <= imp_up
!        yielding
!                     g(j) - imp_up <= z(j) <= g(j) - imp_low
!
!        -----------------------------------------------------------------------

         IF ( a_il == 0 .AND. h_il == 0 ) THEN
            IF ( h_iu == 0 .AND. a_iu == 0 ) THEN
               CALL PRESOLVE_bound_z( j, UPPER, TIGHTEN, gj - imp_low )
               IF ( inform%status /= OK ) RETURN
               CALL PRESOLVE_bound_z( j, LOWER, TIGHTEN, gj - imp_up  )
               IF ( inform%status /= OK ) RETURN
            ELSE
               CALL PRESOLVE_bound_z( j, UPPER, TIGHTEN, gj - imp_low )
               IF ( inform%status /= OK ) RETURN
            END IF
         ELSE IF ( h_iu == 0 .AND. a_iu == 0 ) THEN
            CALL PRESOLVE_bound_z( j, LOWER, TIGHTEN, gj - imp_up  )
            IF ( inform%status /= OK ) RETURN
         END IF

!        If the bound tightening resulted in fixing x(j), loop.

         IF ( prob%X_status( j ) == ELIMINATED ) CYCLE vars

!        -----------------------------------------------------------------------
!        Consider next the case where the dual constraint is forcing,
!        that is g(j) is equal to one of the implied bounds (augmented
!        with the bounds on z(j)).
!        -----------------------------------------------------------------------

         lf = ( a_il == 0 .AND. h_il == 0 ) .AND. (      &
              ( prob%X_status( j ) == FREE .AND.         &
                PRESOLVE_is_zero( gj - imp_low ) )       &
              .OR.                                       &
              ( zlj > s%M_INFINITY .AND.                 &
                PRESOLVE_is_zero( gj - zlj - imp_low ) ) )
         uf = ( a_iu == 0 .AND. h_iu == 0 ) .AND. (      &
              ( prob%X_status( j ) == FREE .AND.         &
                PRESOLVE_is_zero( gj - imp_up )  )       &
              .OR.                                       &
              ( zuj < s%P_INFINITY .AND.                 &
                PRESOLVE_is_zero( gj - zuj - imp_up  ) ) )

!        If the implied lower bound is forcing, set the x and the y
!        to their appropriate bound.

         IF ( lf ) THEN

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    lower implied bound on column', j,  'is forcing'

!           Set z(j) to its lower bound, unless already done.

            IF ( zlj > s%M_INFINITY                   .AND. &
                 .NOT. PRESOLVE_is_zero( zuj - zlj )      ) THEN
               CALL PRESOLVE_fix_z( j, zlj )
               IF ( inform%status /= OK ) RETURN
            END IF

            IF ( hsj /= EMPTY ) THEN

!              Loop over the subdiagonal elements in the j-th row.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing variables on subdiagonal  ', j, 'of H'

               DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
                  i = prob%H%col( k )
                  IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
                  v = prob%H%val( k )
                  IF ( v > ZERO ) THEN
                     CALL PRESOLVE_fix_x( i, prob%X_u(i), Z_FROM_DUAL_FEAS )
                  ELSE IF ( v < ZERO ) THEN
                     CALL PRESOLVE_fix_x( i, prob%X_l(i), Z_FROM_DUAL_FEAS )
                  END IF
                  IF ( inform%status /= OK ) RETURN
               END DO

!              Loop over the superdiagonal elements in the j-th column.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing variables on superdiagonal', j, 'of H'

               k = s%H_col_f( j )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = j + 1, prob%n
                     i  = s%H_row( k )
                     IF ( prob%X_status( i ) > ELIMINATED ) THEN
                        v = prob%H%val( k )
                        IF ( v > ZERO ) THEN
                           CALL PRESOLVE_fix_x( i, prob%X_u( i ),              &
                                                Z_FROM_DUAL_FEAS  )
                        ELSE IF ( v < ZERO ) THEN
                           CALL PRESOLVE_fix_x( i, prob%X_l( i ),              &
                                                Z_FROM_DUAL_FEAS  )
                        END IF
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     k = s%H_col_n( k )
                     IF ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

!           2) Compute the terms from the Jacobian.

            IF ( csj > 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing multipliers on column', j, 'of A'

               k = s%A_col_f( j )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = 1, prob%m
                     i = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN
                        v = prob%A%val( k )
                        IF( v > ZERO )  THEN
                           CALL PRESOLVE_fix_y( i, prob%Y_l( i ) )
                        ELSE IF ( v < ZERO ) THEN
                           CALL PRESOLVE_fix_y( i, prob%Y_u( i ) )
                        END IF
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     k = s%A_col_n( k )
                     if ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

            CYCLE vars

!        If the upper implied bound is forcing, fix the variables and
!        multipliers to their appropriate bound.

         ELSE IF ( uf ) THEN

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    implied upper bound on column', j,  'is forcing'

!           Set z(j) to its upper bound, unless already done.

            IF ( zuj < s%P_INFINITY                 .AND. &
                 .NOT. PRESOLVE_is_zero( zuj - zlj )      ) THEN
               CALL PRESOLVE_fix_z( j, zuj )
               IF ( inform%status /= OK ) RETURN
            END IF

            IF ( hsj /= EMPTY ) THEN

!              Loop over the subdiagonal elements in the j-th row.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing variables on subdiagonal  ', j, 'of H'

               DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
                  i = prob%H%col( k )
                  IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
                  v = prob%H%val( k )
                  IF ( v > ZERO ) THEN
                     CALL PRESOLVE_fix_x( i, prob%X_l( i ), Z_FROM_DUAL_FEAS )
                  ELSE IF ( v < ZERO ) THEN
                     CALL PRESOLVE_fix_x( i, prob%X_u( i ), Z_FROM_DUAL_FEAS )
                  END IF
                  IF ( inform%status /= OK ) RETURN
               END DO

!              Loop over the superdiagonal elements in the j-th column.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing variables on superdiagonal', j, 'of H'

               k = s%H_col_f( j )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = j + 1, prob%n
                     i  = s%H_row( k )
                     IF ( prob%X_status( i ) > ELIMINATED ) THEN
                        v = prob%H%val( k )
                        IF ( v > ZERO ) THEN
                           CALL PRESOLVE_fix_x( i, prob%X_l( i ),              &
                                                Z_FROM_DUAL_FEAS  )
                        ELSE IF ( v < ZERO ) THEN
                           CALL PRESOLVE_fix_x( i, prob%X_u( i ),              &
                                                Z_FROM_DUAL_FEAS  )
                        END IF
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     k = s%H_col_n( k )
                     IF ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

!           2) Compute the terms from the Jacobian.

            IF ( csj > 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    fixing multipliers on column', j, 'of A'

              k = s%A_col_f( j )
              IF ( END_OF_LIST /= k ) THEN
                  DO ii = 1, prob%m
                     i = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN
                        v = prob%A%val( k )
                        IF( v > ZERO )  THEN
                           CALL PRESOLVE_fix_y( i, prob%Y_u( i ) )
                        ELSE IF ( v < ZERO ) THEN
                           CALL PRESOLVE_fix_y( i, prob%Y_l( i ) )
                        END IF
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     k = s%A_col_n( k )
                     if ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

            CYCLE vars

         END IF

!        -----------------------------------------------------------------------
!        Consider next the case where the implied bounds are inconclusive as
!        far as they do not imply dual infeasibility nor that x(j) is
!        dominated.  In this case, possible improvements on the bounds for
!        the variables and multipliers may be possible.
!        -----------------------------------------------------------------------

!        First avoid the case where both implied bounds contain more than
!        an infinite contribution, in which case no bound can be deduced.

         IF ( ( a_il < 0 .OR. h_il < 0 .OR. ( h_il > 0 .AND. a_il > 0 ) )      &
              .AND.                                                            &
              ( a_iu < 0 .OR. h_iu < 0 .OR. ( h_iu > 0 .AND. a_iu > 0 ) )      &
              ) CYCLE

!        At least one implied bound is finite or contains a single infinite
!        contribution.  Consider first the case where at least one implied
!        bound is finite, in which case improvement is possible for the
!        bounds of all variables occuring in the j-th row of the Hessian
!        and all multipliers occuring in the j-th column of A.

         lf = ( a_il == 0 .AND. h_il == 0 )
         uf = ( a_iu == 0 .AND. h_iu == 0 )

         IF ( lf .OR. uf ) THEN

!           Compute the terms from the Hessian.

            IF ( hsj /= EMPTY ) THEN

!              Loop over the subdiagonal elements in the j-th row of H.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using finite implied bounds for variables ',            &
                  'on subdiagonal  ', j, 'of H'

               DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
                  i = prob%H%col( k )
                  IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
                  hij = prob%H%val( k )
                  xli = prob%X_l( i )
                  xui = prob%X_u( i )
                  nli = - s%INFINITY
                  nui =   s%INFINITY

!                 Check if the lower bound can be used.

                  IF ( xuj >= s%P_INFINITY .OR. zlj >= ZERO ) THEN
                     IF ( lf ) THEN
                        IF ( hij >= s%h_tol ) THEN
                           nli = MAX( nli, xui - ( gj - imp_low ) / hij )
                        ELSE IF ( hij <= - s%h_tol ) THEN
                           nui = MIN( nui, xli - ( gj - imp_low ) / hij )
                        END IF
                     ELSE IF ( a_il == 0 .AND. h_il_k == i ) THEN
                        IF ( hij >= s%h_tol ) THEN
                           nli = MAX( nli, - ( gj - imp_low ) / hij )
                        ELSE IF ( hij <= - s%h_tol ) THEN
                           nui = MIN( nui, - ( gj - imp_low ) / hij )
                        END IF
                     END IF
                  END IF

!                 Check if the upper bound can be used.

                  IF ( xlj <= s%M_INFINITY .OR. zuj <= ZERO ) THEN
                     IF ( uf ) THEN
                        IF ( hij >= s%h_tol ) THEN
                           nui = MIN( nui, xli - ( gj - imp_up  ) / hij )
                        ELSE IF ( hij <= - s%h_tol ) THEN
                           nli = MAX( nli, xui - ( gj - imp_up  ) / hij )
                        END IF
                     ELSE IF ( a_iu == 0 .AND. h_iu_k == i ) THEN
                        IF ( hij >= s%h_tol ) THEN
                           nui = MIN( nui, - ( gj - imp_up  ) / hij )
                        ELSE IF ( hij <= - s%h_tol ) THEN
                           nli = MAX( nli, - ( gj - imp_up  ) / hij )
                        END IF
                     END IF
                  END IF

!                 Update the bounds on x(i).

                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    x(', i, ') must lie in [', nli, ',', nui, ']'

                  IF ( PRESOLVE_is_zero( nui - nli ) ) THEN
                     CALL PRESOLVE_fix_x( i, HALF * ( nli + nui ) ,            &
                                                   Z_FROM_DUAL_FEAS )
                  ELSE
                     CALL PRESOLVE_dual_bound_x( i, LOWER, nli )
                     IF ( inform%status /= OK ) RETURN
                     CALL PRESOLVE_dual_bound_x( i, UPPER, nui )
                     IF ( inform%status /= OK ) RETURN
                  END IF
               END DO

!              Loop over the superdiagonal elements in the j-th column.

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using finite implied bounds for variables ',            &
                  'on superdiagonal', j, 'of H'

               k = s%H_col_f( j )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = j + 1, prob%n
                     i  = s%H_row( k )
                     IF ( prob%X_status( i ) > ELIMINATED ) THEN
                        hij = prob%H%val( k )
                        xli = prob%X_l( i )
                        xui = prob%X_u( i )
                        nli = - s%INFINITY
                        nui =   s%INFINITY

!                       Check if the lower bound can be used

                        IF ( xuj >= s%P_INFINITY .OR. zlj >= ZERO ) THEN
                           IF ( lf ) THEN
                              IF ( hij >= s%h_tol ) THEN
                                 nli = MAX( nli, xui - ( gj - imp_low ) / hij )
                              ELSE IF ( hij <= - s%h_tol ) THEN
                                 nui = MIN( nui, xli - ( gj - imp_low ) / hij )
                              END IF
                           ELSE IF ( a_il == 0 .AND. h_il_k == i ) THEN
                              IF ( hij >= s%h_tol ) THEN
                                 nli = MAX( nli, - ( gj - imp_low ) / hij )
                              ELSE IF ( hij <= - s%h_tol ) THEN
                                 nui = MIN( nui, - ( gj - imp_low ) / hij )
                              END IF
                           END IF
                        END IF

!                       Check if the upper bound can be used.

                        IF ( xlj <= s%M_INFINITY .OR. zuj <= ZERO ) THEN
                           IF ( uf ) THEN
                              IF ( hij >= s%h_tol ) THEN
                                 nui = MIN( nui, xli - ( gj - imp_up ) / hij )
                              ELSE IF ( hij <= - s%h_tol ) THEN
                                 nli = MAX( nli, xui - ( gj - imp_up ) / hij )
                              END IF
                           ELSE IF ( a_iu == 0 .AND. h_iu_k == i ) THEN
                              IF ( hij >= s%h_tol ) THEN
                                 nui = MIN( nui, - ( gj - imp_up  ) / hij )
                              ELSE IF ( hij <= - s%h_tol ) THEN
                                 nli = MAX( nli, - ( gj - imp_up  ) / hij )
                              END IF
                           END IF
                        END IF

!                       Update the bounds on x(i)

                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                        '    x(', i, ') must lie in [', nli, ',', nui, ']'

                        IF ( PRESOLVE_is_zero( nui - nli ) ) THEN
                           CALL PRESOLVE_fix_x( i, HALF * ( nli + nui ),       &
                                                        Z_FROM_DUAL_FEAS )
                        ELSE
                           CALL PRESOLVE_dual_bound_x( i, LOWER, nli )
                           IF ( inform%status /= OK ) RETURN
                           CALL PRESOLVE_dual_bound_x( i, UPPER, nui )
                           IF ( inform%status /= OK ) RETURN
                        END IF
                     END IF
                     k = s%H_col_n( k )
                     IF ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

!           Compute the terms from the Jacobian.

            IF ( csj > 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using finite implied bounds for multipliers in column', &
                  j, 'of A'

               k = s%A_col_f( j )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = 1, prob%m
                     i  = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN
                        aij = prob%A%val( k )
                        yli = prob%Y_l( i )
                        yui = prob%Y_u( i )
                        nli = - s%INFINITY
                        nui =   s%INFINITY

!                       Check if the lower bound can be used

                        IF ( xuj >= s%P_INFINITY .OR. zlj >= ZERO ) THEN
                           IF ( lf ) THEN
                              IF ( aij >= s%a_tol ) THEN
                                 nui = MIN( nui, yli + ( gj - imp_low ) / aij )
                              ELSE IF ( aij <= - s%a_tol ) THEN
                                 nli = MAX( nli, yui + ( gj - imp_low ) / aij )
                              END IF
                           ELSE IF ( a_il_k == i .AND. h_il == 0 ) THEN
                              IF ( aij >= s%a_tol ) THEN
                                 nui = MIN( nui, ( gj - imp_low ) / aij )
                              ELSE IF ( aij <= - s%a_tol ) THEN
                                 nli = MAX( nli, ( gj - imp_low ) / aij )
                              END IF
                           END IF
                        END IF

!                       Check if the upper bound can be used

                        IF ( xlj <= s%M_INFINITY .OR. zuj <= ZERO ) THEN
                           IF ( uf ) THEN
                              IF ( aij >= s%a_tol ) THEN
                                 nli = MAX( nli, yui + ( gj - imp_up ) / aij )
                              ELSE IF ( aij <= - s%a_tol ) THEN
                                 nui = MIN( nui, yli + ( gj - imp_up ) / aij )
                              END IF
                           ELSE IF ( a_iu_k == i .AND. h_iu == 0 ) THEN
                              IF ( aij >= s%a_tol ) THEN
                                 nli = MAX( nli, ( gj - imp_up  ) / aij )
                              ELSE IF ( aij <= - s%a_tol ) THEN
                                 nui = MIN( nui, ( gj - imp_up  ) / aij )
                              END IF
                           END IF
                        END IF

!                       Update the bounds on y(i).

                        IF ( s%level >= DEBUG ) WRITE( s%out, * )               &
                           '    y(', i, ') must lie in [', nli, ',', nui, ']'

                        IF ( PRESOLVE_is_zero( nui - nli ) ) THEN
                           CALL PRESOLVE_fix_y( i, HALF * ( nli + nui ) )
                        ELSE
                           CALL PRESOLVE_bound_y( i, LOWER, TIGHTEN, nli )
                           IF ( inform%status /= OK ) RETURN
                           CALL PRESOLVE_bound_y( i, UPPER, TIGHTEN, nui )
                        END IF
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     k = s%A_col_n( k )
                     if ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

!        -----------------------------------------------------------------------
!        Consider next the case where where one of the implied bounds
!        contains a single infinite term.
!        -----------------------------------------------------------------------

         ELSE

!           The lower bound contains a single infinite term that occurs
!           in the x components.

            IF ( xuj >= s%P_INFINITY .AND. h_il > 0 .AND. a_il == 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using the implied lower bound for a single',            &
                  ' variable on subdiagonal  ', j, 'of H'
               hij = prob%H%val( h_il )
               IF ( hij >= s%h_tol ) THEN
                  nli = - ( gj - imp_low ) / hij
                  CALL PRESOLVE_dual_bound_x( h_il_k, LOWER, nli )
                  IF ( inform%status /= OK ) RETURN
               ELSE IF ( hij <= - s%h_tol ) THEN
                  nui = - ( gj - imp_low ) / hij
                  CALL PRESOLVE_dual_bound_x( h_il_k, UPPER, nui )
                  IF ( inform%status /= OK ) RETURN
               END IF
            END IF

!           The upper bound contains a single infinite term that occurs
!           in the x components.

            IF ( xlj <= s%M_INFINITY .AND. h_iu > 0 .AND. a_iu == 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using the implied upper bound for a single',            &
                  ' variable on  subdiagonal  ', j, 'of H'
               hij = prob%H%val( h_iu )
               IF ( hij >= s%h_tol ) THEN
                  nui = - ( gj - imp_up ) / hij
                  CALL PRESOLVE_dual_bound_x( h_iu_k, UPPER, nui )
                  IF ( inform%status /= OK ) RETURN
               ELSE IF ( hij <= - s%h_tol ) THEN
                  nli = - ( gj - imp_up ) / hij
                  CALL PRESOLVE_dual_bound_x( h_iu_k, LOWER, nli )
                  IF ( inform%status /= OK ) RETURN
               END IF
               IF ( inform%status /= OK ) RETURN
            END IF

!           The lower bound contains a single infinite term that occurs
!           in the y components.

            IF ( xuj >= s%P_INFINITY .AND. h_il == 0 .AND. a_il > 0 ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using the implied lower bound for a single',            &
                  ' multiplier on c(', a_il_k, ')'
               aij = prob%A%val( a_il )
               IF ( aij >= s%a_tol ) THEN
                  CALL PRESOLVE_bound_y( a_il_k, UPPER, TIGHTEN,               &
                                         ( gj - imp_low ) / aij  )
               ELSE IF ( aij <= - s%a_tol ) THEN
                  CALL PRESOLVE_bound_y( a_il_k, LOWER, TIGHTEN,               &
                                         ( gj - imp_low ) / aij  )
               END IF
               IF ( inform%status /= OK ) RETURN
            END IF

!           The upper bound contains a single infinite term that occurs
!           in the y components.

            IF ( xlj <= s%M_INFINITY .AND. h_iu == 0 .AND. a_iu > 0 ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    using the implied upper bound for a single',            &
                  ' multiplier on c(', a_iu_k, ')'
               aij = prob%A%val( a_iu )
               IF ( aij >= s%a_tol ) THEN
                  CALL PRESOLVE_bound_y( a_iu_k, LOWER, TIGHTEN,               &
                                         ( gj - imp_up ) / aij   )
               ELSE IF ( aij <= - s%a_tol ) THEN
                  CALL PRESOLVE_bound_y( a_iu_k, UPPER, TIGHTEN,               &
                                         ( gj - imp_up ) / aij   )
               END IF
               IF ( inform%status /= OK ) RETURN
            END IF
         END IF  ! uf or lf

!     End of the loop on the variables

      END DO vars

      RETURN

!     Debugging formats

1000  FORMAT( 4x, i6, 1x, 5ES10.2, 1x, i6, 1x, i6 )
1001  FORMAT( '    bounds: implied lower =', ES12.4, '  gj =', ES12.4,         &
              '  implied upper =', ES12.4 )

      END SUBROUTINE PRESOLVE_dual_constraints

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_redundant_variables

!     If x(j) does not appear in the objective function, then one
!     of the following situations may occur.
!     1) x(j) is unbounded above, A(i,j), when non zero, is positive
!        for all constraints of >= type or negative for all constraints
!        of <= type.
!        In this case, the value of x(j) may be chosen sufficiently large
!        to ensure that all constraints involving x(j) as well as any lower
!        bound on x(j) are inactive. It can thus be removed from the
!        problems together with all constraints where it occurs.
!     2) x(j) is unbounded below, A(i,j), when non zero, is negative
!        for all constraints of >= type or positive for all constraints
!        of <= type.
!        In this case, the value of x(j) may be chosen sufficiently small
!        to ensure that all constraints involving x(j) as well as any upper
!        bound on x(j) are inactive.
!     3) x(j) is free and occurs in a single equality constraint.
!        In this case, its value may be chosen to satisfy this constraint
!        exactly.
!     In all these cases, x(j) may be removed from the problem as well as
!     all the constraints in which it occurs, and the associated constraint
!     multipliers and z(j) may be set to zero. On restoration, its value may
!     be chosen such that all the bounds and constraints that involve it are
!     inactive (and their associated duals/multipliers zero).
!
!     Programming: Ph. Toint, May 2002.
!
!===============================================================================

!     Local variables

      INTEGER            :: j, sj, k, i, ii, l, neq, nin
      REAL ( KIND = wp ) :: cli, cui, xlj, xuj, aij

!     Loop on all active columns

vars: DO j = 1, prob%n
         IF ( prob%X_status( j ) <= ELIMINATED  ) CYCLE

!        Cycle if x(j) appears in the objective function.

         IF ( prob%G( j ) /= ZERO  ) CYCLE
         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) /= EMPTY  ) CYCLE
         END IF

         IF ( s%level >= DETAILS ) WRITE( s%out, * )                           &
            '    x(', j, ') does not occur in the objective function'

!        Loop over the j-th column of A.

         IF ( s%A_col_s( j ) == 0 ) CYCLE

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    verifying constraint signs for x(', j, ')'

         sj  = 0
         nin = 0
         neq = 0
         xlj = prob%X_l( j )
         xuj = prob%X_u( j )
         k   = s%A_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO ii = 1, prob%m

!              Consider the next variable if an equality constraint has
!              already been found.

               IF ( neq > 0 ) CYCLE vars

!              Otherwise, loop over the j-th column of A.

               i  = s%A_row( k )
               IF ( prob%C_status( i ) > ELIMINATED ) THEN

!                 Avoid zero coefficient.

                  aij = prob%A%val( k )
                  IF ( PRESOLVE_is_zero( aij ) ) CYCLE vars

                  cli = prob%C_l( i )
                  cui = prob%C_u( i )
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    c(', i, '): aij =', aij, 'cli =', cli, 'cui =', cui

!                 Equality constraint. Note that only a single inequality
!                 constraint is acceptable if x(j) is free.

                  IF ( PRESOLVE_is_zero( cui - cli ) ) THEN
                     IF ( nin > 0 ) CYCLE vars
                     IF ( xlj > s%M_INFINITY .OR. xuj < s%P_INFINITY )CYCLE vars
                     neq = neq + 1

!                 Inequalities

                  ELSE
                     nin = nin + 1

!                    Verify the signs of the constraints coefficients
!                    and the corresponding bound.

                     IF ( aij > ZERO )  THEN
                        IF ( cli > s%M_INFINITY .AND. &
                             cui > s%P_INFINITY       ) THEN
                           IF ( xuj < s%P_INFINITY .OR. sj == NEGATIVE )       &
                              CYCLE vars
                           sj = POSITIVE
                        ELSE IF ( cli < s%M_INFINITY .AND. &
                                  cui < s%P_INFINITY       ) THEN
                           IF ( xlj > s%M_INFINITY .OR. sj == POSITIVE )       &
                              CYCLE vars
                           sj = NEGATIVE
                        ELSE
                           CYCLE vars
                        END IF
                     ELSE IF ( aij < ZERO ) THEN
                        IF ( cli > s%M_INFINITY .AND. &
                             cui > s%P_INFINITY       ) THEN
                           IF ( xlj > s%M_INFINITY .OR. sj == POSITIVE )       &
                              CYCLE vars
                           sj = NEGATIVE
                        ELSE IF ( cli < s%M_INFINITY .AND. &
                                  cui < s%P_INFINITY       ) THEN
                           IF ( xuj < s%P_INFINITY .OR. sj == NEGATIVE )       &
                              CYCLE vars
                           sj = POSITIVE
                        ELSE
                           CYCLE vars
                        END IF
                     END IF
                  END IF
               END IF
               k = s%A_col_n( k )
               if ( k == END_OF_LIST ) EXIT
            END DO
         END IF

!        The signs of the coefficients of x(j) in A are consistent.
!        It can therefore be eliminated together with all constraints
!        involving it.

!        Remove the redundant variable.

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = X_REDUCTION
         s%hist_j( l )    = j
         s%hist_i( l )    = sj
         s%hist_r( l )    = ZERO
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * ) '  [', s%tt, '] reducing x(', j, ')'
            IF ( s%level >= DEBUG ) THEN
               IF ( sj == POSITIVE )  THEN
                  WRITE( s%out, * ) '    sign is POSITIVE'
               ELSE IF ( sj == NEGATIVE ) THEN
                  WRITE( s%out, * ) '    sign is NEGATIVE'
               ELSE
                  WRITE( s%out, * ) '    single equality constraint'
               END IF
            END IF
         END IF
         CALL PRESOLVE_do_ignore_x( j )

!        Remember the dependencies.

         s%needs( X_VALS, A_VALS ) = MIN( s%tt, s%needs( X_VALS, A_VALS ) )
         s%needs( X_VALS, C_BNDS ) = MIN( s%tt, s%needs( X_VALS, C_BNDS ) )
         s%needs( X_VALS, X_BNDS ) = MIN( s%tt, s%needs( X_VALS, X_BNDS ) )

!        Remove the involved constraints.

         k = s%A_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO ii = 1, prob%m
               i  = s%A_row( k )
               IF ( prob%C_status( i ) > ELIMINATED )                          &
                  CALL PRESOLVE_remove_c( i, Y_GIVEN, yval = ZERO )
               k = s%A_col_n( k )
               if ( k == END_OF_LIST ) EXIT
            END DO
         END IF

!     End of the loop on the variables.

      END DO vars

      RETURN

      END SUBROUTINE PRESOLVE_redundant_variables

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_dependent_variables

!     Check if variables cannot be removed because they are linearly
!     dependent from another one.
!
!     Tuning parameter: the index of the first presolving pass where this
!     procedure is applied

      INTEGER, PARAMETER :: first_depvar_loop = 1

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: i, j, imin, k, ii, jmin, Acolj, kj, inz,          &
                            Asmin, ic, Hcolj, ij, jc, kmin
      LOGICAL            :: aset
      REAL ( KIND = wp ) :: alpha


!     Return if there is only one column, or if the list of potentially
!     dependent variables if empty (after first pass)

      IF ( s%n_active <= 1                         .OR. &
           s%loop < first_depvar_loop              .OR. &
           ( s%loop > first_depvar_loop           .AND. &
             s%nmods( DEPENDENT_VARIABLES ) == 0        )       ) THEN

!        reset the list of potentially dependent variables

         s%a_perm( :prob%n ) = IBCLR( s%a_perm( :prob%n ), DEPENDENT_VARIABLES )
         s%nmods( DEPENDENT_VARIABLES ) = 0
         RETURN
      END IF

!     Loop over all active columns.

lj:   DO j = 1, prob%n

!        If not at first pass, avoid columns that are not in the list of
!        potential dependent variables.

         IF ( s%loop > first_depvar_loop  .AND.                                &
              .NOT. BTEST( s%a_perm( j ), DEPENDENT_VARIABLES ) )CYCLE lj

!        Avoid inactive columns

         IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE lj

!        Also avoid columns with a single nonzero in H, as the latter cannot be
!        multiple of another column. Column with nonempty Hessian contribution
!        are also avoided in the Hessina is not active.

         Hcolj = PRESOLVE_H_row_s( j )
         IF ( prob%H%ne > 0 ) THEN
            IF ( Hcolj == 1 ) CYCLE lj
         END IF

!        Also avoid unconstrained linear variables.

         Acolj = s%A_col_s( j )
         IF ( Hcolj == 0 .AND. Acolj == 0 ) CYCLE lj

         IF ( s%level >= DETAILS ) THEN
            WRITE( s%out, * ) '   considering column', j,                      &
                 'for possible combination'
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
                 '    this column has', Hcolj, 'nonzeros in H and',            &
                 Acolj, 'nonzeros in A'
         END IF

!        Loop over the column's nonzero elements to find the shortest row
!        of A having a nonzero in column j, remembering the position in A
!        of A(i,j) in w_m(i).

!        The j-th column of the Jacobian is non empty.

         Asmin = prob%n + 1
         IF ( Acolj > 0 ) THEN
            s%w_m  = 0
            k      = s%A_col_f( j )
            DO ii = 1, prob%m
               i  = s%A_row( k )
               IF ( prob%A%val( k ) /= ZERO ) THEN
                  s%w_m( i ) = k
                  IF ( prob%C_status( i ) > ELIMINATED ) THEN
                     IF ( s%A_row_s( i ) < Asmin ) THEN
                        Asmin = s%A_row_s( i )
                        imin = i
                     END IF
                  END IF
               END IF
               k = s%A_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END DO

            IF ( Asmin > prob%n ) THEN
               inform%status = ERRONEOUS_EMPTY_COL
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE INTERNAL ERROR: unexpected empty column', j,    &
                    'found'
               WRITE( inform%message( 2 ), * )                                 &
                    '    while searching in A for dependent variables'
               IF ( s%level >= DEBUG )                                         &
                  CALL PRESOLVE_write_full_prob( prob, control, s )
               RETURN
            END IF

!           Asmin is the minimal number of nonzeros in a row of A
!           intersecting column j.
!           Consider the next column if this row has only one nonzero
!           (which must then be that in column j).

            IF ( s%A_row_s( imin ) == 1 ) CYCLE lj

!           Else remember the minimal row in ic, and choose jmin to be the
!           first column with a nonzero in the minimal row.

            ic   = imin
            kmin = prob%A%ptr( imin )
            jmin = prob%A%col( kmin )
         END IF

!        The j-th column of the Hessian is non empty
!         (since either Hcolj or Acolj must be > 0 ).

         inz = 0
         IF ( Hcolj > 0 ) THEN

!           Remember the position of the nonzero elements of column
!           j of the Hessian (in w_n).  Also remember in w_mn the sequence
!           of rows of H having a nonzero entry in column j.

            s%w_n = 0

!           (subdiagonal)

            DO kj = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
               IF ( prob%H%val( kj ) == ZERO ) CYCLE
               ij = prob%H%col( kj )
               IF ( prob%X_status( ij ) <= ELIMINATED ) CYCLE
               inz           = inz + 1
               s%w_mn( inz ) = ij
               s%w_n( ij )   = kj
            END DO

!           (superdiagonal)

            kj = s%H_col_f( j )
            IF ( kj /= END_OF_LIST ) THEN
               DO ii = j + 1, prob%n
                  IF ( prob%H%val( kj ) /= ZERO ) THEN
                     ij = s%H_row( kj )
                     IF ( prob%X_status( ij ) > ELIMINATED ) THEN
                        inz           = inz + 1
                        s%w_mn( inz ) = ij
                        s%w_n( ij )   = kj
                     END IF
                  END IF
                  kj = s%H_col_n( kj )
                  IF ( kj == END_OF_LIST ) EXIT
               END DO
            END IF

            IF ( inz <= 0 ) THEN
               inform%status = ERRONEOUS_EMPTY_COL
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE INTERNAL ERROR: unexpected empty column', j,    &
                    'found'
               WRITE( inform%message( 1 ), * )                                 &
                    '    while searching in H for dependent variables'
               IF ( s%level >= DEBUG )                                         &
                  CALL PRESOLVE_write_full_prob( prob, control, s )
               RETURN
            END IF

!           Choose jmin as a column having nonzeros in the same positions
!           as column j, if this column has fewer nonzeros than the minimal
!           row.

            IF ( Hcolj < Asmin ) THEN
               ic   = 1
               jmin = s%w_mn( ic )
            END IF
         END IF

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    minimal exploration for column', j,':'
            IF ( prob%A%ne > 0 ) WRITE( s%out, * ) '    =>', Asmin,            &
                                      'in A ( row', imin,')'
            IF ( prob%H%ne > 0 ) WRITE( s%out, * ) '    =>',  Hcolj, 'in H'
          END IF

!        Loop on the columns jmin that can possibly be multiple of column j.

ljc:     DO jc = 1, prob%n

!           Find the next possible column that can be a multiple of column j.

            IF ( jc > 1 ) THEN
               IF ( Asmin <= Hcolj .OR. Hcolj == 0 ) THEN
                  kmin = kmin + 1
                  IF ( kmin == prob%A%ptr( ic + 1 ) ) THEN
                     ic = s%conc( ic )
                     IF ( ic == END_OF_LIST ) EXIT ljc
                     kmin = prob%A%ptr( ic )
                  END IF
                  jmin = prob%A%col( kmin )
               ELSE
                  ic = ic + 1
                  IF ( ic > Hcolj ) EXIT ljc
                  jmin = s%w_mn( ic )
               END IF
            END IF

!           At first application, only keep active columns beyond column j
!           and having the same number of nonzeros as column j (in A and H).

            IF ( prob%X_status( jmin )    <= ELIMINATED .OR. &
                 s%A_col_s( jmin )        /= Acolj      .OR. &
                 jmin                     == j          .OR. &
                 PRESOLVE_H_row_s( jmin ) /= Hcolj           ) CYCLE ljc

            IF ( s%loop == first_depvar_loop .AND. jmin <= j   ) CYCLE ljc

!           Column jmin is worth trying (as far as we can see).

            IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    trying column', jmin

!           Check if column j and j min are multiple (giving up the
!           comparison of the columns as early as possible).

            IF ( Asmin <= Hcolj ) THEN
               aset = .FALSE.
               CALL PRESOLVE_Acols_mult( jmin, j, s%w_m, aset, alpha )
               IF ( .NOT. aset ) CYCLE ljc
               CALL PRESOLVE_Hcols_mult( jmin, j, s%w_n, aset, alpha )
               IF ( .NOT. aset ) CYCLE ljc
            ELSE
               IF ( Hcolj > 0 ) THEN
                  aset = .FALSE.
                  CALL PRESOLVE_Hcols_mult( jmin, j, s%w_n, aset, alpha )
                  IF ( .NOT. aset ) CYCLE ljc
               END IF
               IF ( Acolj > 0 ) THEN
                   aset = .FALSE.
                  CALL PRESOLVE_Acols_mult( jmin, j, s%w_m, aset, alpha )
                  IF ( .NOT. aset ) CYCLE ljc
               END IF
            END IF

!           We have that:  alpha * column jmin = column j , both in A and H,
!           which indicates that we may define the new variable jmin as
!           the old variable jmin + alpha times the old variable j.

            CALL PRESOLVE_merge_x( j, jmin, alpha )
            if ( inform%status /= OK ) RETURN
            CYCLE lj

         END DO ljc

      END DO lj

!     Reset the list of potentially dependent variables.

      s%a_perm( :prob%n ) = IBCLR( s%a_perm( :prob%n ), DEPENDENT_VARIABLES )
      s%nmods( DEPENDENT_VARIABLES ) = 0

      RETURN

      END SUBROUTINE PRESOLVE_dependent_variables

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_sparsify_A

!     Improves the sparsity pattern of A by eliminating entries in rows whose
!     sparsity pattern are a superset of that of an equality row.
!
!     Note: one verifies here that all active equality constraints contain
!           at least two nonzero entries. This is realistic if the primal
!           constraints have already been examined, as singleton equality
!           constraints have been removed by this procedure.
!
!     Tuning parameter: the index of the first presolving loop where this
!     procedure is applied

      INTEGER, PARAMETER :: first_sparsif_loop = 1

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: i, e, col, next, ie, iej, iek, ek, ej, ii, inew,   &
                            it, nze, j, l, k, rsie, rse, ic, kpiv, npsp,       &
                            nbr_canceled
      REAL ( KIND = wp ) :: alpha, piv, ae, aie, bound, ylie, yuie, yle, yue,  &
                            clie, cuie, maxaie, maxae, tmp, nclie, ncuie,      &
                            nyle, nyue

!     At the first presolving pass, find the indices of the equality type active
!     rows with at least two nonzeros and store then in a_perm, while storing
!     their sizes in w_m. At later passes, only consider the rows whose index
!     has been flagged in a_perm.

!     Not yet the first pass

      IF ( s%loop < first_sparsif_loop ) THEN
         GO TO 1000

!     This is the first pass: build the initial list.

      ELSE IF ( s%loop == first_sparsif_loop ) THEN
         npsp = 0
         DO i = 1, prob%m
            IF ( prob%C_status( i ) <= ELIMINATED .OR. &
                 s%A_row_s( i ) < 2               .OR. &
                 prob%C_l( i ) /= prob%C_u( i )        ) CYCLE
            npsp           = npsp + 1
            s%w_mn( npsp ) = i
            s%w_m( npsp )  = s%A_row_s( i )
         END DO

!     Beyond the first pass: use the flagged rows.

      ELSE
         IF ( s%nmods( ROW_SPARSIFICATION ) <= 0 ) GO TO 1000
         npsp = 0
         DO i = 1, prob%m
            IF ( prob%C_status( i ) <= ELIMINATED               .OR. &
                 .NOT. BTEST( s%a_perm( i ), ROW_SPARSIFICATION )      ) CYCLE
            npsp          = npsp + 1
            s%w_m( npsp ) = i
         END DO
         DO i = 1, npsp
            k = s%w_m( i )
            s%w_mn( i ) = k
            s%w_m( i )  = s%A_row_s( k )
         END DO
      END IF
      IF ( npsp == 0 ) RETURN

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * ) '   the equality constraints are:'
         DO i = 1, npsp
            WRITE( s%out, * ) '   constraint', s%w_mn( i ), ' with',           &
                 s%A_row_s( s%w_mn( i ) ), 'nonzeros'
         END DO
      END IF

!     Sort the retained equality rows by increasing row size.

      CALL SORT_quicksort( npsp, s%w_m, inform%status, ivector = s%w_mn)
      IF ( inform%status /= OK ) THEN
         inform%status = SORT_TOO_LONG
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: sorting capacity too small.'
         WRITE( inform%message( 2 ), * )                                       &
            ' Increase log2s in SORT_quicksort.'
      END IF

      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    equality constraints sorted by increasing number of nonzeros'

!     Transform the ordered list to a pointer list, in order to allow easy
!     reinsertion in the case where an equality row is sparsified.

      s%w_m = 0
      e     = s%w_mn( 1 )
      DO ii = 1, npsp - 1
         s%w_m( s%w_mn( ii ) ) = s%w_mn( ii + 1 )
      END DO
      s%w_m( s%w_mn( npsp ) ) = END_OF_LIST

!     Loop over the equality constraints by increasing size.

      DO

!        Exit the procedure if there is not enough history space left.

         IF ( s%tt + 6 > control%max_nbr_transforms ) GO TO 1000

!        Analyze the pivot row.

         rse = s%A_row_s( e )

         IF ( s%level >= DETAILS ) THEN
            WRITE( s%out, * ) '   considering row =', e, 'as pivot (',         &
                 rse, 'nonzeros )'
            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     remains --> w_m =', s%w_m( :prob%m )
         END IF

!        Find the column with fewest elements amongst those that have
!        nonzero elements in row e.

         inew  = prob%m + 1
         ic    = e
         maxae = ZERO
         DO
            DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
               j = prob%A%col( k )
               IF ( prob%X_status( j ) <= ELIMINATED .OR.  &
                    s%A_col_s( j )     <  2          .OR.  &
                    s%A_col_s( j )     >= inew             ) CYCLE
               ae  = prob%A%val( k )
               tmp = ABS( ae )
               IF ( tmp <= s%a_tol ) CYCLE
               maxae = MAX( maxae, tmp )
               inew  = s%A_col_s( j )
               col   = j
               piv   = ae
               kpiv  = k
            END DO
            ic = s%conc( ic )
            IF ( ic == END_OF_LIST ) EXIT
         END DO

!        If no column is found (which must occur because the pivoting
!        tolerance is too large) then consider the next equality constraint.

         IF ( inew == prob%m + 1 ) THEN

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    no pivot large enough in constraint', e

!           Find the next pivot row.

            inew = e
            e    = s%w_m( inew )
            IF ( e == END_OF_LIST ) EXIT

!           Mark the current pivot row as already analyzed.

            s%w_m( inew ) = 0

!           End of the loop on pivot rows.


            CYCLE
         END IF

!        col now contains the index of the pivot column
!        piv contains the pivotal element, that is the entry
!        in row e and column col.

!        In order to verify that row ie is a superset of row e,
!        first store the positions of the nonzeros of row e in w_n.

         s%w_n = 0
         ic    = e
         DO
            DO ek = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
               ej = prob%A%col( ek )
               IF ( prob%X_status( ej ) <= ELIMINATED .OR. &
                    prob%A%val( ek )    == ZERO            ) CYCLE
               s%w_n( ej ) = ek
            END DO
            ic = s%conc( ic )
            IF ( ic == END_OF_LIST ) EXIT
         END DO

!        Now see if row e can be used to sparsify another row
!        having a nonzero in column col.
!        Loop over the nonzero entries this column.

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    column', col, 'has fewest nonzeros, pivot value =', piv

         next  = s%A_col_f( col )
rlit:    DO it = 1, prob%m
            IF ( it > 1 ) THEN
               next = s%A_col_n( next )
               IF ( next == END_OF_LIST ) EXIT
            END IF

            aie  = prob%A%val( next )
            ie   = s%A_row( next )
            rsie = s%A_row_s( ie )

!           Find the row index corresponding to the current entry
!           and avoid row e, all inactive rows and all rows that contain
!           less nonzeros than row e.

            IF ( s%tt + 5 + rsie > control%max_nbr_transforms ) CYCLE rlit

            IF ( ie == e                            .OR. &
                 prob%C_status( ie ) <= ELIMINATED  .OR. &
                 aie == ZERO                        .OR. &
                 rse > rsie                         .OR. &
                 ( rse == rsie .AND. ie < e )            ) THEN
                IF ( s%level >= DEBUG ) WRITE( s%out, * )                      &
                   '    row', ie, 'unsuitable'
                CYCLE
            END IF

            IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    row', ie, 'of size',&
               rsie, 'has a nonzero in column', col

!           Row ie is a possible candidate for a superset row.

            IF ( s%level >= DETAILS ) WRITE( s%out, * )                        &
               '   trying to pivot c(', e, ') in c(', ie, ')'

!           Verify that row ie is a superset of row e
!           by verifying that each column index occuring in row e
!           also occurs in row ie. To do this, count the number of nonzeros
!           in row ie that correspond to nonzeros in row e.

            nze    = 0
            ic     = ie
            maxaie = ZERO
            DO
               DO iek = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  iej = prob%A%col( iek )
                  tmp = prob%A%val( iek )
                  IF ( prob%X_status( iej ) <= ELIMINATED .OR. &
                       tmp                  == ZERO            ) CYCLE
                  IF ( s%w_n( iej ) > 0 ) nze = nze + 1
                  maxaie = MAX( maxaie, ABS( tmp ) )
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO

!           If every nonzero in the pivot row is not matched by a nonzero
!           in row ie, row ie cannot be a superset of row e.

            IF ( nze /= rse ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    row', ie, 'not superset of row', e
               CYCLE
            END IF

!           Row ie is a superset of row e.

!           Compute the multiple alpha of row e that, if added
!           to row ie, would annihilate its col-th entry.

            alpha = - aie / piv

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    pivoting is possible on A(', ie, ',', col,                 &
               ') with alpha =', alpha

!           Avoid the operation if, in the worst case, the resulting
!           maximal value of the new row exceeds the maximal acceptable one
!           for the reduced problem.

            IF ( PRESOLVE_is_too_big ( maxaie + ABS( alpha ) * maxae ) ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    row sparsification prevented by inacceptable growth'
               CYCLE
            END IF

!           Anticipate the values  of the new constraints bounds for row ie.

            clie  = prob%C_l( ie )
            cuie  = prob%C_u( ie )
            bound = alpha * prob%C_l( e )
            IF ( clie > s%M_INFINITY ) THEN
               nclie = clie + bound
            ELSE
               nclie = ZERO         ! a convention which is never too big
            END IF
            IF ( cuie < s%P_INFINITY ) THEN
               ncuie = cuie + bound
            ELSE
               ncuie = ZERO         ! a convention which is never too big
            END IF

!           Avoid the transformation if any of these is too big.

            IF ( PRESOLVE_is_too_big( nclie ) .OR. &
                 PRESOLVE_is_too_big( ncuie )      ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    row sparsification prevented by inacceptable growth'
               CYCLE
            END IF

!           Anticipate the corresponding multipliers.

            ylie = prob%Y_l( ie )
            yuie = prob%Y_u( ie )
            yle  = prob%Y_l( e )
            yue  = prob%Y_u( e )
            IF ( alpha > ZERO ) THEN
               IF ( yle > s%M_INFINITY .AND. yuie < s%P_INFINITY ) THEN
                  nyle = yle - alpha * yuie
               ELSE
                  nyle = -s%INFINITY
               END IF
               IF ( yue < s%P_INFINITY .AND. ylie > s%M_INFINITY ) THEN
                  nyue = yue - alpha * ylie
               ELSE
                  nyue = s%INFINITY
               END IF
            ELSE
               IF ( yle > s%M_INFINITY .AND. ylie > s%M_INFINITY ) THEN
                  nyle = yle - alpha * ylie
               ELSE
                  nyle = -s%INFINITY
               END IF
               IF ( yue < s%P_INFINITY .AND. yuie < s%P_INFINITY ) THEN
                  nyue = yue - alpha * yuie
               ELSE
                  nyue = s%INFINITY
               END IF
            END IF

!           Avoid the transformation if any of these is too big.

            IF ( PRESOLVE_is_too_big( nyle ) .OR. &
                 PRESOLVE_is_too_big( nyue )      ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    row sparsification prevented by inacceptable growth'
               CYCLE
            END IF

!           Now perform and record the operation:
!
!                row ie  <--- row ie + alpha * row e

            IF ( s%tm >= s%max_tm ) THEN
               CALL PRESOLVE_save_transf
               IF ( inform%status /= OK ) RETURN
            END IF
            l    = s%tm + 1
            s%tt = s%tt + 1
            s%tm = l
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  [', s%tt, '] combining constraints', e, '(e) and', ie
            s%hist_type( l ) = A_ROWS_COMBINED
            s%hist_i( l )    = kpiv
            s%hist_j( l )    = ie
            s%hist_r( l )    = alpha

!           Replace, in row ie, the nonzero entries corresponding
!           to nonzero entries of row e (not in the pivot column)
!           by the suitable linear combination.
!           Also detect if any of these new values is zero, in which
!           case unexpected cancellation occurs.

            nbr_canceled = 0
            ic = ie
            DO
               DO iek = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  iej = prob%A%col( iek )
                  IF ( prob%X_status( iej ) <= ELIMINATED .OR. &
                       prob%A%val( iek )    == ZERO            ) CYCLE

!                 Find the position of the matching nonzero in row e,
!                 if it exists.

                  ek = s%w_n( iej )
                  IF ( ek == 0 ) CYCLE

!                 If it is in the pivot column, just remember its position.

                  IF ( iej == col ) THEN
                     CALL PRESOLVE_rm_A_entry( ie, col, iek )

!                 Otherwise perform the linear combination
!                 (+ cancellation detection)

                  ELSE
                     ae = prob%A%val( ek )
                     IF ( ae == ZERO ) CYCLE
                     ae = prob%A%val( iek ) + alpha * ae
                     prob%A%val( iek ) = ae
                     IF ( PRESOLVE_is_zero( ae ) ) THEN
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    cancellation in A(', ie, ',', iej, '), k =', iek
                        CALL PRESOLVE_rm_A_entry( ie, iej, iek )
                        nbr_canceled = nbr_canceled + 1
                     END IF
                  END IF
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO

!           If cancellation occured, see if row ie is itself an equality
!           row already considered as pivot.  Since its size has
!           decreased because of cancellation, it must be reintroduced
!           in the list of potential pivot rows.

            IF ( clie == cuie          .AND. &
                 s%w_m( ie ) == 0      .AND. &
                 s%A_row_s( ie ) <= rse      ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    reinserting row', ie, 'in the list of potential pivots'
               inew        = s%w_m( e )
               s%w_m( e )  = ie
               s%w_m( ie ) = inew
            END IF

!           Now update the lower and upper bounds on row ie.

            IF ( clie > s%M_INFINITY ) THEN
               CALL PRESOLVE_bound_c( ie, LOWER, SET, nclie )
               IF ( inform%status /= OK ) RETURN
            END IF
            IF ( cuie < s%P_INFINITY ) THEN
               CALL PRESOLVE_bound_c( ie, UPPER, SET, ncuie )
               IF ( inform%status /= OK ) RETURN
            END IF

!           Now update the corresponding multipliers.

            CALL PRESOLVE_bound_y( e, LOWER, SET, nyle )
            IF ( inform%status /= OK ) RETURN
            CALL PRESOLVE_bound_y( e, UPPER, SET, nyue )
            IF ( inform%status /= OK ) RETURN
            IF ( s%level >= DETAILS ) WRITE( s%out, * )                        &
               '   lhs, rhs and y bounds updated'

         END DO rlit

!        Find the next pivot row.

         inew = e
         e    = s%w_m( inew )
         IF ( e == END_OF_LIST ) EXIT

!        Mark the current pivot row as already analyzed.

         s%w_m( inew ) = 0

!        End of the loop on pivot rows.

      END DO

!     Reset the marks for row sparsification.

1000  CONTINUE
      s%a_perm( :prob%m ) = IBCLR( s%a_perm( :prob%m ), ROW_SPARSIFICATION )
      s%nmods( ROW_SPARSIFICATION ) = 0

      RETURN

      END SUBROUTINE PRESOLVE_sparsify_A

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_check_bounds( first_pass )

!     Verify the coherence of the initial bounds on the variables and
!     multipliers as well as lhs and rhs of the constraints.
!     If first_pass ==.TRUE., possibly fix some variables and initialize the
!     status of the bounds.

!     Argument:

      LOGICAL, INTENT( IN ) :: first_pass

!            .TRUE. iff variables may be fixed and the status of the variables
!            must be initialized

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: i, j
      REAL ( KIND = wp ) :: xl, xu, cl, cu, zl, zu, yl, yu

!     Loop on all active variables.

      DO j = 1, prob%n
         IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE

         xl = prob%X_l( j )
         xu = prob%X_u( j )
         zl = prob%Z_l( j )
         zu = prob%Z_u( j )

!        Check feasibility of the multiplier bounds.

         IF ( PRESOLVE_is_pos( zl - zu ) ) THEN
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because the bounds on z(', j, ') are incompatible'
            RETURN
         ELSE IF ( zl >= s%P_INFINITY .OR. zu <= s%M_INFINITY ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is primal infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because z(', j, ') has an infinite active bound'
            RETURN
         END IF

!        Check to presence of constraints on both sides.

         IF ( PRESOLVE_is_pos( zl ) ) THEN
            CALL PRESOLVE_fix_x( j, xl, Z_FROM_DUAL_FEAS )
            IF ( inform%status /= OK ) RETURN
            CYCLE
         END IF
         IF ( PRESOLVE_is_neg( zu ) ) THEN
            CALL PRESOLVE_fix_x( j, xu, Z_FROM_DUAL_FEAS )
            IF ( inform%status /= OK ) RETURN
            CYCLE
         END IF

!        Check the variables.

         IF ( xl >= s%P_INFINITY .OR. xu <= s%M_INFINITY ) THEN
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because x(', j, ') has an infinite active bound'
            RETURN
         END IF

         IF ( xl <= s%M_INFINITY ) THEN
            IF ( xu >= s%P_INFINITY ) THEN                       ! free variable
               IF ( first_pass ) THEN
                  CALL PRESOLVE_fix_z( j, ZERO )
                  prob%X_status( j ) = FREE
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    x(', j, ') is free'
               END IF
            ELSE                               ! non-positivity or upper bounded
               CALL PRESOLVE_bound_z( j, UPPER, TIGHTEN, ZERO )
               IF ( inform%status /= OK ) RETURN
               IF ( first_pass ) prob%X_status( j ) = UPPER
            END IF
         ELSE
            IF ( xu < s%P_INFINITY ) THEN                       ! fixed variable
               IF ( PRESOLVE_is_zero( xu - xl ) ) THEN
                  CALL PRESOLVE_fix_x( j, HALF * ( xl + xu ), Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE IF ( PRESOLVE_is_neg( xu - xl ) ) THEN      ! inconsistent
                  inform%status = PRIMAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                  ' PRESOLVE analysis stopped: the problem is primal infeasible'
                  WRITE( inform%message( 1 ), * )                              &
                  '    because the bounds on x(', j, ') are incompatible'
                  RETURN
               ELSE
                  IF ( first_pass ) prob%X_status( j ) = RANGE
               END IF
            ELSE                               ! non-negativity or lower bounded
               CALL PRESOLVE_bound_z( j, LOWER, TIGHTEN,  ZERO )
               IF ( inform%status /= OK ) RETURN
               IF ( first_pass ) prob%X_status( j ) = LOWER
            END IF
         END IF

      END DO
      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   bounds on the variables verified'

!     Loop on all active rows.

      DO i = 1, prob%m
         IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE

         cl = prob%C_l( i )
         cu = prob%C_u( i )
         yl = prob%Y_l( i )
         yu = prob%Y_u( i )

!        Check feasibility of the multiplier bounds

         IF ( PRESOLVE_is_pos( yl - yu ) ) THEN
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because the bounds on y(', i, ') are incompatible'
            RETURN
         END IF

         IF ( yl >= s%P_INFINITY .OR. yu <= s%M_INFINITY ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is primal infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because y(', i, ') has an infinite active bound'
            RETURN
         END IF

!        Check to presence of constraints on both sides.

         IF ( PRESOLVE_is_pos( yl ) ) THEN
            CALL PRESOLVE_bound_c( i, UPPER, SET, cl )
         END IF
         IF ( PRESOLVE_is_neg( yu ) ) THEN
            CALL PRESOLVE_bound_c( i, LOWER, SET, cu )
         END IF

!        Check the constraints.

         IF ( cl >= s%P_INFINITY .OR. cu <= s%M_INFINITY ) THEN
            IF ( .NOT. PRESOLVE_is_zero( cu - cl ) ) THEN
               inform%status = DUAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE analysis stopped: the problem is dual infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                  '    because c(', i, ') has an infinite active bound'
               RETURN
            ELSE
               IF ( s%level >= TRACE .AND. first_pass ) WRITE( s%out, * )      &
               '  === WARNING === c(', i, ') is an infinite equality constraint'
            END IF
         END IF

         IF ( cl <= s%M_INFINITY ) THEN
            IF ( cu >= s%P_INFINITY ) THEN                 ! free constraint
               CALL PRESOLVE_remove_c( i, Y_GIVEN, yval = ZERO )
               IF ( inform%status /= OK ) RETURN
            ELSE                                           ! upper bounded
               CALL PRESOLVE_bound_y( i, UPPER, TIGHTEN, ZERO )
            END IF
         ELSE
            IF ( cu < s%P_INFINITY ) THEN
               IF ( cu < cl ) THEN                         ! inconsistent
                  inform%status = PRIMAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                  ' PRESOLVE analysis stopped: the problem is primal infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                  '    because the bounds on c(', i, ') are incompatible'
                  RETURN
               END IF
            ELSE                                           ! lower bounded
               CALL PRESOLVE_bound_y( i, LOWER, TIGHTEN, ZERO )
            END IF
         END IF
         IF ( inform%status /= OK ) RETURN
      END DO
      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   bounds on the constraints verified'

      RETURN

      END SUBROUTINE PRESOLVE_check_bounds

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_get_x_lists( s )

!     Build the (disjoint) pointer lists of the linear singleton, linear
!     doubleton columns and active unconstrained variables.

!     Argument:

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Notes:
!
!     s%lsc_f    indicates the position in (1:n) of the first linear singleton
!              variable, or is equal to END_OF_LIST if there is none;
!
!     s%ldc_f    indicates the position in (1:n) of the first linear doubleton
!              variable, or is equal to END_OF_LIST if there is none;
!
!     s%unc_f    indicates the position in (1:n) of the first active
!              unconstrained variable, or is equal to END_OF_LIST if there
!              is none.
!
!     h_perm( i ) indicates the position in (1:n) of the next variable in the
!              list and is equal to END_OF_LIST if there is none.

!     Programming: Ph. Toint, November 2000

!===============================================================================

!     Local variables

      INTEGER :: j

!     Reset the lists

      IF ( s%lsc_f /= END_OF_LIST ) CALL PRESOLVE_reset( s%lsc_f )
      IF ( s%ldc_f /= END_OF_LIST ) CALL PRESOLVE_reset( s%ldc_f )
      IF ( s%unc_f /= END_OF_LIST ) CALL PRESOLVE_reset( s%unc_f )

      IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    lists reset'

!     Loop on the active columns

      DO j = prob%n, 1, -1
         IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE

!        --------------------------------------------------------
         IF ( prob%m > 0 .AND. prob%H%ne > 0 ) THEN
!        --------------------------------------------------------

            SELECT CASE ( s%A_col_s( j ) )
            CASE ( 0 )                           ! unconstrained variable
               CALL PRESOLVE_add_at_beg( j, s%unc_f )
            CASE ( 1 )                           ! linear singleton column
               IF ( s%H_str( j ) == EMPTY )                                    &
                  CALL PRESOLVE_add_at_beg( j, s%lsc_f )
            CASE ( 2 )                           ! linear doubleton
               IF ( s%H_str( j ) == EMPTY )                                    &
                  CALL PRESOLVE_add_at_beg( j, s%ldc_f )
            END SELECT

!        --------------------------------------------------------
         ELSE IF ( prob%H%ne > 0 ) THEN  !  no linear constraints
!        --------------------------------------------------------

            CALL PRESOLVE_add_at_beg( j, s%unc_f )

!        --------------------------------------------------------
         ELSE IF ( prob%m > 0 ) THEN     ! zero Hessian
!        --------------------------------------------------------

            SELECT CASE ( s%A_col_s( j ) )
            CASE ( 0 )                           ! unconstrained variable
               CALL PRESOLVE_add_at_beg( j, s%unc_f )
            CASE ( 1 )                           ! linear singleton column
               CALL PRESOLVE_add_at_beg( j, s%lsc_f )
            CASE ( 2 )                           ! linear doubleton
               CALL PRESOLVE_add_at_beg( j, s%ldc_f )
            END SELECT
         END IF
      END DO

      IF ( s%level >= CRAZY ) THEN
         WRITE( s%out, * ) '     lsc_f =', s%lsc_f, ' ldc_f =', s%ldc_f,       &
                                    ' unc_f =', s%unc_f
         WRITE( s%out, * ) '     h_perm =', s%h_perm( :prob%n )
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_get_x_lists

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_Acols_mult( k, j, w_m, multiple, alpha )

!     Check if columns j and k of the sparse A are multiple in the sense
!     that alpha * column k = column j.

!     Arguments:

      INTEGER, INTENT( IN  ) :: k

!            the index of the first column to examine

      INTEGER, INTENT( IN  ) :: j

!            the index of the second column to examine

      INTEGER, DIMENSION( prob%m ), INTENT( IN  ) :: w_m

!            for each nonzero A( i,jmin ), w_m( i ) is the position in A
!            of A( i, j ), if it is nonzero, or zero otherwise

      LOGICAL, INTENT( INOUT ) :: multiple

!             on input : .TRUE. iff the ratio alpha between columns j and k
!                        is already determined (and must therefore be confirmed
!                        or infirmed)
!             on output: .TRUE. iff alpha * column k of A is equal to column j

      REAL ( KIND = wp ), INTENT( OUT ) :: alpha

!            the ratio between the entries of column k and j

!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: kk, ii, ik, kj
      REAL ( KIND = wp ) :: ratio

!     Loop over column k, checking it is a multiple of column j in A
!     note that, if a column is finished, the other must be
!     finished too because they have the same number of nonzeros.

      kk    = s%A_col_f( k )
      DO ii = 1, prob%m
         IF ( .NOT. PRESOLVE_is_zero( prob%A%val( kk ) ) ) THEN
            ik = s%A_row( kk )
            IF ( prob%C_status( ik ) > ELIMINATED ) THEN

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    found element A(', ik, ',', k, ')'

!              See if there is a matching (same row) nonzero in
!              column j. If not the two columns cannot be
!              nonzero multiples: return.

               kj = w_m( ik )
               IF ( kj == 0 ) THEN
                   IF ( s%level >= DEBUG ) WRITE( s%out, * )                   &
                      '    columns', j, 'and' , k , 'of A not multiple'
                   multiple = .FALSE.
                   RETURN
               END IF
               ratio = prob%A%val( kj ) / prob%A%val( kk )

               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    found element A(', ik, ',', j,')  ratio =', ratio

!              Check if these nonzeros are multiples.

               IF ( multiple ) THEN
                  IF ( ratio /= alpha ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    columns', j, 'and' , k, ' of A not multiple'
                     multiple = .FALSE.
                     RETURN
                  END IF
               ELSE
                  multiple = .TRUE.
                  alpha    = ratio
               END IF
            END IF
         END IF

!        Find the positions of the next elements in column jmin in A.

         kk = s%A_col_n( kk )
         IF ( kk == END_OF_LIST ) EXIT
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_Acols_mult

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_Hcols_mult( k, j, w_n, multiple, alpha )

!     Check if columns j and k of the sparse H are multiple in the sense
!     that alpha * column k = column j

!     Arguments:


      INTEGER, INTENT( IN  ) :: k

!            the index of the first column to examine

      INTEGER, INTENT( IN  ) :: j

!            the index of the second column to examine

      INTEGER, DIMENSION( prob%n ), INTENT( IN  ) :: w_n

!            for each nonzero H( i, jmin ), w_n( i ) is the position in H
!            of H i, j ), if it is nonzero, or zero otherwise

      LOGICAL, INTENT( INOUT ) :: multiple

!             on input : .TRUE. iff the ratio alpha between columns j and k
!                        is already determined (and must therefore be confirmed
!                        or infirmed)
!             on output: .TRUE. iff alpha * column k of H is equal to column j

      REAL ( KIND = wp ), INTENT( INOUT ) :: alpha

!            the ratio between the entries of column k and j

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: kk, ii, ik, kj
      REAL ( KIND = wp ) :: ratio

!     Loop on column k of H to verify that it is a multiple of column j
!     (subdiagonal).

      DO kk = prob%H%ptr( k ), prob%H%ptr( k + 1 ) - 1
         IF ( prob%H%val( kk ) == ZERO ) CYCLE
         ik = prob%H%col( kk )
         IF ( prob%X_status( ik ) <= ELIMINATED ) CYCLE
         kj = w_n( ik )
         IF ( kj == 0 ) THEN
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    columns', j, 'and',  k , 'of H not multiple'
            multiple = .FALSE.
            RETURN
         END IF
         ratio = prob%H%val( kj ) / prob%H%val( kk )

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    found element H(', ik, ',', j, ')  ratio =', ratio

!        Check if these nonzeros are multiples.

         IF ( multiple ) THEN
            IF ( ratio /= alpha ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    columns', j, 'and' , k, ' of H not multiple'
               multiple = .FALSE.
               RETURN
            END IF
         ELSE
            multiple = .TRUE.
            alpha    = ratio
         END IF
      END DO

!     (superdiagonal)

      kk = s%H_col_f( k )
      IF ( kk /= END_OF_LIST ) THEN
         DO ii = k + 1, prob%n
            IF ( prob%H%val( kk ) /= ZERO ) THEN
               ik = s%H_row( kk )
               IF ( prob%X_status( ik ) > ELIMINATED ) THEN
                  kj = w_n( ik )
                  IF ( kj == 0 ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    columns', j, 'and' , k , 'of H not multiple'
                     multiple = .FALSE.
                     RETURN
                  END IF
                  ratio = prob%H%val( kj ) / prob%H%val( kk )

                  IF ( s%level >= DEBUG ) WRITE( s%out , * )                   &
                     '    found element H(', ik, ',', j, ')  ratio =', ratio

!                 Check if these nonzeros are multiples

                  IF ( multiple ) THEN
                     IF ( ratio /= alpha ) THEN
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    columns', j, 'and' , k, ' of H not multiple'
                        multiple = .FALSE.
                        RETURN
                     END IF
                  ELSE
                     multiple  = .TRUE.
                     alpha     = ratio
                  END IF
               END IF
            END IF

!           Find the positions of the next elements in column jmin in H.

            kk = s%H_col_n( kk )
            IF ( kk == END_OF_LIST ) EXIT
         END DO
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_Hcols_mult

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_mark( p, h )

!     Marks the p-th variable/constraint has being modified, and, if this is
!     the first time, increment the counter of modified variables/constraints

!     Argument

      INTEGER, INTENT( IN ) :: p

!            the index of the variable/constraint to mark

      INTEGER, INTENT( IN ) :: h

!            the heuristic for which to mark column/row p

!     Programming: Ph. Toint, May 2001.
!
!===============================================================================

!     Local variables

      INTEGER :: oldm

      IF ( h == s%hindex ) RETURN
      oldm = s%a_perm( p )
      IF ( .NOT. BTEST( oldm, h ) ) s%nmods( h ) = s%nmods( h ) + 1
      s%a_perm( p ) = IBSET( oldm, h )

      RETURN

      END SUBROUTINE PRESOLVE_mark

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_reorder_variables

!     The variables are ordered so that their bounds appear in the order
!
!     free                      x
!     non-negativity      0  <= x
!     lower              x_l <= x
!     range              x_l <= x <= x_u
!     upper                     x <= x_u
!     non-positivity            x <=  0
!
!     Fixed variables are removed. Within each category, the variables
!     are further ordered so that those with non-zero diagonal Hessian
!     entries occur before the remainder.
!
!     Programming: Ph. Toint, November 2000, on the basis of a routine
!     by N. Gould.
!
!===============================================================================

!     Local variables

      INTEGER            :: i, j, k, hjj
      INTEGER            :: isfree, nonneg, lowerb, inrange, upperb, nonpos
      INTEGER            :: h_free, h_nonneg, h_lowerb, h_inrange, h_upperb
      INTEGER            :: h_nonpos, d_free, o_free, d_nonneg, o_nonneg
      INTEGER            :: d_lowerb, o_lowerb, d_inrange, o_inrange
      INTEGER            :: d_upperb, o_upperb, d_nonpos, o_nonpos
      INTEGER            :: ignored, d_ignored, inact, d_inactive
      REAL ( KIND = wp ) :: xl, xu

!     Mark the variables whose diagonal element in H is nonzero by
!     setting the corresponding value of H_str to 1

      IF ( prob%H%ne > 0 ) THEN
         IF ( s%level >= DETAILS ) WRITE( s%out, * )                           &
            '   exploring the diagonal of H'

         s%H_str( :prob%n ) = 0
         DO i = 1, prob%n
            k = prob%H%ptr( i + 1 ) - 1
            IF ( k >= prob%H%ptr( i ) ) THEN
               IF ( i == prob%H%col( k ) .AND. prob%H%val( k ) /= ZERO )       &
                  s%H_str( i ) = 1
            END IF
         END DO
      END IF

!     Run through the bounds to see how many fall into each of the
!     categories:  free (isfree), non-negativity (nonneg), lower (lowerb),
!     range (inrange), upper(upperb), non-positivity (nonpos) and
!     ignored (ignored); of these, h_free, h_nonneg, h_lowerb, h_inrange,
!     h_upperb and h_nonpos  have diagonal Hessian entries (the distinction is
!     useless for ignored ones).

      isfree    = 0
      nonneg    = 0
      lowerb    = 0
      inrange   = 0
      upperb    = 0
      nonpos    = 0
      ignored   = 0
      inact     = 0
      h_free    = 0
      h_nonneg  = 0
      h_lowerb  = 0
      h_inrange = 0
      h_upperb  = 0
      h_nonpos  = 0

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   starting the first pass on the variables'

      DO j = 1, prob%n
         xl = prob%X_l( j )
         xu = prob%X_u( j )
         SELECT CASE ( prob%X_status( j ) )
         CASE ( ACTIVE:FREE )
            IF ( prob%H%ne > 0 ) THEN
               hjj = s%H_str( j )
            ELSE
               hjj = 0
            END IF
            IF ( xl <= s%M_INFINITY ) THEN

!  Free variable

               IF ( xu >= s%P_INFINITY ) THEN
                  isfree = isfree + 1
                  IF ( hjj > 0 ) h_free = h_free + 1
               ELSE

!  Non-positivity

                  IF ( xu == ZERO ) THEN
                     nonpos = nonpos + 1
                     IF ( hjj > 0 ) h_nonpos = h_nonpos + 1

!  Upper bounded variable

                  ELSE
                     upperb = upperb + 1
                     IF ( hjj > 0 ) h_upperb = h_upperb + 1
                  END IF
               END IF
            ELSE

!  Fixed variable
!  NOTE: this only happens when permutation is applied without problem
!        reduction, thus pssibly leaving fixed variables undetected.  However
!        these have to be removed for the permuted problem to be in the
!        standard form.  Hence we fix them here "at the last minute", outside
!        the set of ordinary transformations, and remember their list in
!        the linked list whose header is s%lfx_f.

               IF ( xu < s%P_INFINITY ) THEN
                  IF ( xu == xl ) THEN
                     CALL PRESOLVE_do_fix_x( j, xu )
                     CALL PRESOLVE_add_at_beg( j, s%lfx_f )
                     IF ( s%level >= DEBUG ) THEN
                        WRITE( s%out, * )                                      &
                             '   fixing x(', j, ') at the last minute to', xu
                        IF ( s%level >= CRAZY ) WRITE( s%out, * )              &
                           '     lsc_f = ', s%lsc_f, 'ldc_f =', s%ldc_f,       &
                           ' lfx_f =', s%lfx_f,                                &
                           '-->  h_perm =', s%h_perm(:prob%n)
                     END IF
                     ignored  = ignored + 1

!  Inconsistent bounds

                  ELSE IF ( xu < xl ) THEN
                     inform%status = PRIMAL_INFEASIBLE
                     WRITE( inform%message( 1 ), * )                           &
                          ' PRESOLVE analysis stopped:',                       &
                          ' the problem is primal infeasible'
                     WRITE( inform%message( 2 ), * )                           &
                          '    because the bounds on x(', j, 'are incompatible'
                     RETURN

!  Range bounded variable

                  ELSE
                     inrange = inrange + 1
                     IF ( hjj > 0 ) h_inrange = h_inrange + 1
                  END IF
               ELSE

!  Non-negativity

                  IF ( xl == ZERO ) THEN
                     nonneg = nonneg + 1
                     IF ( hjj > 0 ) h_nonneg = h_nonneg + 1

!  Lower bounded variable

                  ELSE
                     lowerb = lowerb + 1
                     IF ( hjj > 0 ) h_lowerb = h_lowerb + 1
                  END IF
               END IF
            END IF

!  Ignored

         CASE ( ELIMINATED )
            ignored = ignored + 1

!  Inactive

         CASE ( INACTIVE )
            inact = inact + 1
         END SELECT
      END DO

!     Now set starting addresses for each division of the variables.

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * ) '   end of the first pass on the variables'
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * )                                                  &
                 '    free      =', isfree,  'of which h_free   =', h_free
            WRITE( s%out, * )                                                  &
                 '    nonneg    =', nonneg,  'of which h_nonneg =', h_nonneg
            WRITE( s%out, * )                                                  &
                 '    lower     =', lowerb,  'of which h_lower  =', h_lowerb
            WRITE( s%out, * )                                                  &
                 '    range     =', inrange, 'of which h_range  =', h_inrange
            WRITE( s%out, * )                                                  &
                 '    upper     =', upperb,  'of which h_upper  =', h_upperb
            WRITE( s%out, * )                                                  &
                 '    nonpos    =', nonpos,  'of which h_nonpos =', h_nonpos
            WRITE( s%out, * ) '    ignored   =', ignored
            WRITE( s%out, * ) '    inactive  =', inact
         END IF
      END IF

      d_free     = 0
      o_free     = d_free    + h_free
      d_nonneg   = d_free    + isfree
      o_nonneg   = d_nonneg  + h_nonneg
      d_lowerb   = d_nonneg  + nonneg
      o_lowerb   = d_lowerb  + h_lowerb
      d_inrange  = d_lowerb  + lowerb
      o_inrange  = d_inrange + h_inrange
      d_upperb   = d_inrange + inrange
      o_upperb   = d_upperb  + h_upperb
      d_nonpos   = d_upperb  + upperb
      o_nonpos   = d_nonpos  + h_nonpos
      d_ignored  = d_nonpos  + nonpos
      d_inactive = d_ignored + ignored

      IF ( s%level >= DEBUG ) THEN
         IF ( d_ignored /= s%n_active ) THEN
            WRITE( s%out, * ) '    ERROR: found ', d_ignored,                  &
                 ' active variables instead of', s%n_active
            STOP
         END IF
      END IF

!     Also set the starting and ending addresses as required.

      prob%h_diag_end_free    = o_free
      prob%x_free             = d_nonneg
      prob%h_diag_end_nonneg  = o_nonneg
      prob%x_l_start          = d_lowerb + 1
      prob%h_diag_end_lower   = o_lowerb
      prob%x_u_start          = d_inrange + 1
      prob%h_diag_end_range   = o_inrange
      prob%x_l_end            = d_upperb
      prob%h_diag_end_upper   = o_upperb
      prob%x_u_end            = d_nonpos
      prob%h_diag_end_nonpos  = o_nonpos

!     Run through the bounds for a second time, this time building the mapping
!     array.

      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
              '   starting the second pass on the variables'
      DO j = 1, prob%n
         SELECT CASE ( prob%X_status( j ) )
         CASE ( ACTIVE:FREE )
            xl  = prob%X_l( j )
            xu  = prob%X_u( j )
            IF ( prob%H%ne > 0 ) THEN
               hjj = s%H_str( j )
            ELSE
               hjj = 0
            END IF
            IF ( xl <= s%M_INFINITY ) THEN

!  Free variable

               IF ( xu >= s%P_INFINITY ) THEN
                  IF ( hjj > 0 ) THEN
                     d_free             = d_free + 1
                     prob%X_status( j ) = d_free
                  ELSE
                     o_free             = o_free + 1
                     prob%X_status( j ) = o_free
                  END IF
               ELSE

!  Non-positivity

                  IF ( xu == ZERO ) THEN
                     IF ( hjj > 0 ) THEN
                        d_nonpos           = d_nonpos + 1
                        prob%X_status( j ) = d_nonpos
                     ELSE
                        o_nonpos           = o_nonpos + 1
                        prob%X_status( j ) = o_nonpos
                     END IF

!  Upper bounded variable

                  ELSE
                     IF ( hjj > 0 ) THEN
                        d_upperb           = d_upperb + 1
                        prob%X_status( j ) = d_upperb
                     ELSE
                        o_upperb           = o_upperb + 1
                        prob%X_status( j ) = o_upperb
                     END IF
                  END IF
               END IF
            ELSE

!  Fixed variable

               IF ( xu < s%P_INFINITY ) THEN
                  IF ( xu == xl ) THEN
                     d_ignored          = d_ignored + 1
                     prob%X_status( j ) = d_ignored
                  ELSE

!  Range bounded variable

                     IF ( hjj > 0 ) THEN
                        d_inrange          = d_inrange + 1
                        prob%X_status( j ) = d_inrange
                     ELSE
                        o_inrange          = o_inrange + 1
                        prob%X_status( j ) = o_inrange
                     END IF
                  END IF
               ELSE

!  Non-negativity

                  IF ( xl == ZERO ) THEN
                     IF ( hjj > 0 ) THEN
                        d_nonneg           = d_nonneg + 1
                        prob%X_status( j ) = d_nonneg
                     ELSE
                        o_nonneg           = o_nonneg + 1
                        prob%X_status( j ) = o_nonneg
                     END IF

!  Lower bounded variable

                  ELSE
                     IF ( hjj > 0 ) THEN
                        d_lowerb           = d_lowerb + 1
                        prob%X_status( j ) = d_lowerb
                     ELSE
                        o_lowerb           = o_lowerb + 1
                        prob%X_status( j ) = o_lowerb
                     END IF
                  END IF
               END IF
            END IF

!  Ignored

         CASE ( ELIMINATED )
            d_ignored          = d_ignored + 1
            prob%X_status( j ) = d_ignored

!  Inactive

         CASE ( INACTIVE )
            d_inactive         = d_inactive + 1
            prob%X_status( j ) = d_inactive
         END SELECT
      END DO
      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
              '   end of the second pass on the variables'

      RETURN

      END SUBROUTINE PRESOLVE_reorder_variables

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_reorder_constraints

!     The constraints are ordered so that their bounds appear in the order
!
!     non-negativity      0  <= A x
!     equality           c_l  = A x
!     lower              c_l <= A x
!     range              c_l <= A x <= c_u
!     upper                     A x <= c_u
!     non-positivity            A x <=  0
!
!     Programming: Ph. Toint, November 2000, on the basis of a routine
!     by N. Gould.
!
!===============================================================================

!     Local variables

      INTEGER            :: free, lower, range, upper, equality, ignored
      INTEGER            :: a_equality, a_lower, a_range, a_upper, a_free
      INTEGER            :: a_ignored, i, a_inactive, inact
      REAL ( KIND = wp ) :: cl, cu

!     Run through the constraint bounds to see how many fall into each of the
!     categories:  free, lower, range, upper, equality and  ignored (or
!     inactive).

      free     = 0
      lower    = 0
      range    = 0
      upper    = 0
      equality = 0
      ignored  = 0
      inact    = 0

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   starting the first pass on the constraints'

      DO i = 1, prob%m
         SELECT CASE ( prob%C_status( i ) )
         CASE ( ACTIVE )
            cl = prob%C_l( i )
            cu = prob%C_u( i )
            IF ( cl <= s%M_INFINITY ) THEN

!  Free constraint

               IF ( cu >= s%P_INFINITY ) THEN
                  free = free + 1
               ELSE

!  Upper bounded constraint

                  upper = upper + 1
               END IF
            ELSE

!  Equality constraint

               IF ( cu < s%P_INFINITY ) THEN
                  IF ( cu == cl ) THEN
                     equality = equality + 1

!  Inconsistent constraints

                  ELSE IF ( cu < cl ) THEN
                     inform%status = PRIMAL_INFEASIBLE
                     WRITE( inform%message( 1 ), * )                           &
                          ' PRESOLVE analysis stopped:',                       &
                          ' the problem is primal infeasible'
                     WRITE( inform%message( 2 ), * )                           &
                          '    because the bounds on c(', i, 'are incompatible'
                     RETURN

!  Range bounded constraint

                  ELSE
                     range = range + 1
                  END IF
               ELSE

!  Lower bounded constraint

                  lower = lower + 1
               END IF
            END IF

!  Inactive constraints

         CASE ( ELIMINATED )
            ignored = ignored + 1
         CASE ( INACTIVE )
            inact = inact + 1
         END SELECT
      END DO

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * )'   end of the first pass on the constraints'
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    equality   =', equality
            WRITE( s%out, * ) '    lower      =', lower
            WRITE( s%out, * ) '    range      =', range
            WRITE( s%out, * ) '    upper      =', upper
            WRITE( s%out, * ) '    free       =', free
            WRITE( s%out, * ) '    ignored    =', ignored
            WRITE( s%out, * ) '    inactive   =', inact
         END IF
      END IF

!     Now set starting addresses for each division of the constraints.

      a_equality = 0
      a_lower    = a_equality + equality
      a_range    = a_lower + lower
      a_upper    = a_range + range
      a_free     = a_upper + upper
      a_ignored  = a_free + free
      a_inactive = a_ignored + ignored

!     Also set the starting and ending addresses as required.

      prob%c_equality = a_equality
      prob%c_u_start  = a_range + 1
      prob%c_l_end    = a_upper
      prob%c_u_end    = a_free

!     Run through the bounds for a second time, this time building the mapping
!     array.

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   starting the second pass on the constraints'

      DO i = 1, prob%m
         SELECT CASE ( prob%C_status( i ) )
         CASE ( ACTIVE )
           cl = prob%C_l( i )
           cu = prob%C_u( i )
           IF ( cl <= s%M_INFINITY ) THEN

!  Free constraint

              IF ( cu >= s%P_INFINITY ) THEN
                 a_free             = a_free + 1
                 prob%C_status( i ) = a_free
              ELSE

!  Upper bounded constraint

                 a_upper            = a_upper + 1
                 prob%C_status( i ) = a_upper
              END IF
           ELSE

!  Equality constraint

              IF ( cu < s%P_INFINITY ) THEN
                 IF ( cu == cl ) THEN
                    a_equality         = a_equality + 1
                    prob%C_status( i ) = a_equality

!  Range bounded constraint

                 ELSE
                    a_range            = a_range + 1
                    prob%C_status( i ) = a_range
                 END IF
              ELSE

!  Lower bounded constraint

                 a_lower            = a_lower + 1
                 prob%C_status( i ) = a_lower
              END IF
           END IF

!  Inactive constraints

        CASE ( ELIMINATED )
           a_ignored          = a_ignored + 1
           prob%C_status( i ) = a_ignored
        CASE ( INACTIVE )
           a_inactive         = a_inactive + 1
           prob%C_status( i ) = a_inactive
        END SELECT
      END DO

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
              '   end of the second pass on the constraints'

      RETURN

      END SUBROUTINE PRESOLVE_reorder_constraints

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_apply_permutations

!     The computed permutations are effectively applied on the problem.
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: i, j, k, ii, jj, kk, nnz, ic, iis, iie

      IF ( s%level >= DEBUG )                                                  &
         WRITE( s%out, * ) '    n_active =', s%n_active,                       &
                             'm_active =', s%m_active,                         &
                             'm_eq_active =', s%m_eq_active

!-------------------------------

!  Permutations of the variables

!-------------------------------

      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%X   )
      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%X_l )
      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%X_u )
      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%Z   )
      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%Z_l )
      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%Z_u )
      CALL SORT_inplace_permute( prob%n, prob%X_status, x = prob%G   )

      IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   x-vectors permuted'

!-------------------------------

!     the matrix H

!-------------------------------

!     Save the information contained in h_perm (notably the lists of variables
!     in forcing constraints and the list of variables fixed at the last minute)
!     before overwriting it with the permutation of H.

      s%w_n( :prob%n ) = s%h_perm( :prob%n )

!     Now consider permuting the matrix H.

      IF ( prob%H%ne > 0 ) THEN

!        Save the starting positions of the rows in the unpermuted structure
!        in w_mn.

         s%w_mn( :prob%n + 1 ) = prob%H%ptr( :prob%n + 1 )

!        First determine the size of subdiagonal part of each row of H
!        after permutation.  This is achieved by looping on all elements
!        of H.

         prob%H%ptr( :prob%n ) = 0
         nnz = 0
         DO i  = 1, prob%n
            ii = prob%X_status( i )
            DO k  = s%w_mn( i ), s%w_mn( i + 1 ) - 1
               jj = prob%X_status( prob%H%col( k ) )
               IF ( prob%H%val( k ) /= ZERO .AND. &
                    ii <= s%n_active    .AND. &
                    jj <= s%n_active          ) THEN
                  nnz = nnz + 1
                  IF ( jj <= ii ) THEN
                     prob%H%ptr( ii ) = prob%H%ptr( ii ) + 1
                  ELSE
                     prob%H%ptr( jj ) = prob%H%ptr( jj ) + 1
                  END IF
               END IF
            END DO
         END DO

         IF ( s%level >= CRAZY ) WRITE( s%out, * ) '     H_ptr =',             &
            prob%H%ptr( :s%n_active )

!        Now convert the size of each permuted row to the starting address
!        for this row.

         ii = 1
         DO i = 1, s%n_active
            jj              = prob%H%ptr( i )
            prob%H%ptr( i ) = ii
            ii              = ii + jj
         END DO
         prob%H%ptr( s%n_active + 1 ) = ii

         IF ( s%level >= CRAZY ) WRITE( s%out, * )                             &
            '     H_ptr =', prob%H%ptr( :s%n_active )

!        Determine the position of each element of the permuted H
!        in its unpermuted version (the inverse permutation for H).

         IF ( s%level >= DEBUG .AND. prob%H%ne > 0 ) THEN
            WRITE( s%out, * ) '    Building the permutation of H'
            WRITE( s%out, * )                                                  &
                 '       k       i    j       status         ii   jj         kk'
         END IF

         DO i  = 1, prob%n
            ii = prob%X_status( i )
            DO k  = s%w_mn( i ), s%w_mn( i + 1 ) - 1
               j  = prob%H%col( k )
               jj = prob%X_status( j )
               IF ( prob%H%val( k ) /= ZERO .AND. &
                    ii <= s%n_active .AND. jj <= s%n_active ) THEN
                  IF ( jj <= ii ) THEN
                     kk               = prob%H%ptr( ii )
                     prob%H%ptr( ii ) = kk + 1
                     prob%H%col( k )  = jj
                     IF ( s%level >= DEBUG ) WRITE( s%out, 1001 )              &
                        k, i, j, 'active row', ii, jj, kk
                  ELSE
                     kk               = prob%H%ptr( jj )
                     prob%H%ptr( jj ) = kk + 1
                     prob%H%col( k )  = ii
                     IF ( s%level >= DEBUG ) WRITE( s%out, 1001 )              &
                        k, i, j, 'active column', ii, jj, kk
                  END IF
               ELSE
                  nnz             = nnz + 1
                  kk              = nnz
                  prob%H%col( k ) = jj
                     IF ( s%level >= DEBUG ) WRITE( s%out, 1001 )              &
                        k, i, j, 'inactive', ii, jj, kk
               END IF
               s%h_perm( kk ) = k
            END DO
         END DO

         IF ( s%level >= CRAZY ) THEN
            WRITE( s%out, * )'     H_perm_inv  =', s%h_perm( :prob%H%ne )
            WRITE( s%out, * )'     H_col       =', prob%H%col(:prob%H%ne )
            WRITE( s%out, * )'     H_ptr       =', prob%H%ptr( :prob%n )
         END IF

!        Build the starting addresses for the permuted structure in H_ptr

         DO i = s%n_active, 2, -1
            prob%H%ptr( i ) = prob%H%ptr( i - 1 )
         END DO
         prob%H%ptr( 1 ) = 1
         s%h_ne_active = prob%H%ptr( s%n_active + 1 ) - 1

         IF ( s%level >= CRAZY ) WRITE( s%out, * )                             &
            '     H_ptr     =', prob%H%ptr( :prob%n )

!        Permute the matrix entries and its column indices.
!        Note that these column indices are not sorted (in each row).

         CALL SORT_inverse_permute( prob%H%ne , s%h_perm,                      &
                                    x = prob%H%val, ix = prob%H%col  )

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * )'    entries and column indices permuted'
            WRITE( s%out, * )'    H_perm_inv =', s%h_perm(:prob%H%ne)
            WRITE( s%out, * )'    H_col      =', prob%H%col(:prob%H%ne)
         END IF

!        Now reorder the active rows by increasing column indices.
!        Also update the inverse permutation accordingly.

         DO ii = 1, s%n_active
            iis = prob%H%ptr( ii )
            iie = prob%H%ptr( ii + 1 ) - 1
            IF ( iie <= iis ) CYCLE

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * )                                               &
                 '    reordering row', ii, '( positions', iis, 'to', iie, ')'
               IF ( s%level >= CRAZY ) WRITE( s%out, * )                       &
                   '     columns before sorting =', prob%H%col( iis: iie )
            END IF

            CALL SORT_quicksort( iie - iis + 1, prob%H%col( iis:iie ),         &
                                 inform%status,                                &
                                 ivector = s%h_perm( iis:iie ),                &
                                 rvector = prob%H%val( iis:iie ) )
            IF ( inform%status /= OK ) THEN
               SELECT CASE ( inform%status )
               CASE ( 1 )
                  inform%status = -100
                  WRITE( inform%message( 1 ), * )                              &
                      ' PRESOLVE INTERNAL ERROR 100: Please report to Ph. Toint'
                  WRITE( inform%message( 2 ), * )                              &
                      ' with the data causing the error. Thank you.'
               CASE ( 2 )
                  inform%status = SORT_TOO_LONG
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: sorting capacity too small.'
                  WRITE( inform%message( 2 ), * )                              &
                       ' Increase log2s in SORT_quicksort.'
               END SELECT
               RETURN
            END IF
            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     columns after sorting  =', prob%H%col( iis:iie )

         END DO

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * )'    column indices reordered'
            IF ( s%level >= CRAZY ) THEN
               WRITE( s%out, * )'     H_perm_inv =', s%h_perm(:prob%H%ne)
               WRITE( s%out, * )'     H_col      =', prob%H%col(:prob%H%ne)
            END IF
         END IF

!        Obtain the direct permutation for H from its inverse.

         CALL SORT_inplace_invert( prob%H%ne, s%h_perm )

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    permutation inverted'
            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
                  '     H_perm =', s%h_perm( :prob%H%ne )
         END IF

!        Set the number of active elements in H.

         prob%H%ne = s%h_ne_active

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   H permuted'

      END IF

!---------------------------------

!  Permutations of the constraints

!---------------------------------

      IF ( prob%m > 0 ) THEN
         CALL SORT_inplace_permute( prob%m, prob%C_status, x = prob%C   )
         CALL SORT_inplace_permute( prob%m, prob%C_status, x = prob%C_l )
         CALL SORT_inplace_permute( prob%m, prob%C_status, x = prob%C_u )
         CALL SORT_inplace_permute( prob%m, prob%C_status, x = prob%Y   )
         CALL SORT_inplace_permute( prob%m, prob%C_status, x = prob%Y_l )
         CALL SORT_inplace_permute( prob%m, prob%C_status, x = prob%Y_u )

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   y-vectors permuted'

!---------------------------------

!  Permutations of A

!---------------------------------

!        Save the starting positions of the rows in the unpermuted structure
!        in w_m .

         s%w_m( :prob%m + 1 ) = prob%A%ptr( :prob%m + 1 )

!        Permute the row sizes.

         IF ( s%level >= CRAZY ) WRITE( s%out, * )                             &
            '     A_row_s =', s%A_row_s( :prob%m )

         prob%A%ptr( :prob%m ) = s%A_row_s
         CALL SORT_inplace_permute( prob%m, prob%C_status, ix = prob%A%ptr )

         IF ( s%level >= CRAZY ) WRITE( s%out, * )'     A_ptr =',              &
              prob%A%ptr( :prob%m )

!        Build the pointers to the beginning of the permuted rows in A_ptr
!        Also accumulate the total number of active nonzeros in A in nnz.

         nnz = 1
         DO i = 1, s%m_active
            jj              = prob%A%ptr( i )
            prob%A%ptr( i ) = nnz
            nnz             = nnz + jj
         END DO
         prob%A%ptr( s%m_active + 1 ) = nnz

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    number of active entries =', nnz - 1
            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     A_ptr =', prob%A%ptr( :s%m_active + 1 )

!        Build the inverse permutation for the active nonzero entries.
!        The zero entries are stored beyond the last active nonzero,
!        that is starting from position a_ne_active + 1 in A.

!        One first needs to mark all concatenated rows (which must be
!        inactive) in order to avoid reprocessing them when processing
!        the row to which they are concatenated (or vice-versa).  This is
!        done by negating their C_status.

            WRITE( s%out, * ) '    Row concatenation:'
         END IF

         DO i  = 1, prob%m
            ii = prob%C_status( i )

!           Avoid active rows.

            IF ( ii < 0 ) CYCLE

!           Get the concatenated rows if any, and mark them.

            ic = i
            DO
               kk = ic
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
               IF ( s%level >= DEBUG )  WRITE( s%out, * )                      &
                  '     row', ic, 'is concatenated to row', kk, 'in row', i
               k = prob%C_status( ic )
               IF ( k > 0 ) prob%C_status( ic ) = - k
            END DO
         END DO

!        Now build the permutation, while avoiding concatenated rows (and
!        resetting their C_status).

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    Building the permutation of A:'
            WRITE( s%out, * )                                                  &
                 '       k       i    j     status      ii   jj         kk'
         END IF

         DO i  = 1, prob%m
            ii = prob%C_status( i )

!           Avoid concatenated rows and reset their status.

            IF ( ii < 0 ) THEN
               prob%C_status( i ) = - ii
               CYCLE
            END IF

!           Loop on the remaining rows

            ic = i
            DO
               DO k  = s%w_m( ic ), s%w_m( ic + 1 ) - 1
                  j  = prob%A%col( k )
                  jj = prob%X_status( j )

!                 Inactive entry
!                 (inactive column, inactive row or zero value )

                  IF ( jj > s%n_active         .OR. &
                       ii > s%m_active         .OR. &
                       prob%A%val( k ) == ZERO      ) THEN
                     kk  = nnz
                     nnz = kk + 1
                     IF ( s%level >= DEBUG ) WRITE( s%out, 1000 )              &
                          k, i, j, 'inactive', ii, jj, kk

!                 Active entry

                  ELSE
                     kk               = prob%A%ptr( ii )
                     prob%A%ptr( ii ) = kk + 1
                     IF ( s%level >= DEBUG ) WRITE( s%out, 1000 )              &
                         k, i, j, 'active', ii, jj, kk
                  END IF
                  s%a_perm( kk )    = k
                  prob%A%col( k ) = jj
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO
         END DO

         IF ( s%level >= CRAZY ) THEN
            WRITE( s%out, * ) '     A_ptr      =', prob%A%ptr(:s%m_active+1)
            WRITE( s%out, * ) '     A_perm_inv =', s%a_perm( :prob%A%ne )
         END IF

!        Now recover the pointers to the beginning of the permuted rows.

         DO i = s%m_active, 2, -1
            prob%A%ptr( i ) = prob%A%ptr( i - 1 )
         END DO
         prob%A%ptr( 1 )    = 1
         s%a_ne_active = prob%A%ptr( s%m_active + 1 ) - 1

         IF ( s%level >= CRAZY ) WRITE( s%out, * )                             &
            '     A_ptr =', prob%A%ptr( :s%m_active + 1 )

!        Permute the matrix entries and its column indices.
!        Note that these column indices are not sorted (in each row).

         CALL SORT_inverse_permute( prob%A%ne, s%a_perm,                       &
                                    x = prob%A%val, ix = prob%A%col )

!        Now reorder the rows by inceasing column indices.
!        Also update the inverse permutation accordingly.

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    entries and column indices permuted'
            IF ( s%level >= CRAZY ) THEN
               WRITE( s%out, * ) '     A_perm_inv =', s%a_perm( :prob%A%ne )
               WRITE( s%out, * ) '     A_col      =',                          &
                                        prob%A%col( :prob%A%ne )
            END IF
         END IF

         DO ii = 1, s%m_active
            iis = prob%A%ptr( ii )
            iie = prob%A%ptr( ii + 1 ) - 1
            IF ( iie <= iis ) CYCLE

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * )                                               &
               '    reordering row', ii, '( positions', iis, 'to', iie, ')'
               IF ( s%level >= CRAZY ) WRITE( s%out, * )                       &
                  '     columns before sorting =', prob%A%col( iis:iie )
            END IF

            CALL SORT_quicksort( iie - iis + 1, prob%A%col( iis:iie ),         &
                                 inform%status,                                &
                                 ivector = s%a_perm( iis:iie ),                &
                                 rvector = prob%A%val ( iis:iie ) )
            IF ( inform%status /= OK ) THEN
               SELECT CASE ( inform%status )
               CASE ( 1 )
                  inform%status = -101
                  WRITE( inform%message( 1 ), * )                              &
                      ' PRESOLVE INTERNAL ERROR 101: Please report to Ph. Toint'
                  WRITE( inform%message( 2 ), * )                              &
                      ' with the data causing the error. Thank you.'
               CASE ( 2 )
                  inform%status = SORT_TOO_LONG
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: sorting capacity too small.'
                  WRITE( inform%message( 2 ), * )                              &
                       ' Increase log2s in SORT_quicksort.'
               END SELECT
               RETURN
            END IF

            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     columns after sorting  =',  prob%A%col( iis:iie )

         END DO

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    column indices reordered'
            IF ( s%level >= CRAZY ) THEN
               WRITE( s%out, * ) '     A_perm_inv =', s%a_perm( :prob%A%ne )
               WRITE( s%out, * ) '     A_col      =',                          &
                                           prob%A%col( :prob%A%ne )
            END IF
         END IF

!        Obtain the direct permutation from its inverse.

         CALL SORT_inplace_invert( prob%A%ne, s%a_perm )

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    permutation inverted'
            IF ( s%level >= CRAZY ) THEN
               WRITE( s%out, * ) '     A_perm =', s%a_perm( :prob%A%ne )
            END IF
         END IF

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   A permuted'

      END IF

!     Set the new number of active elements

      prob%A%ne = s%a_ne_active

!---------------------------------

!  Update the dimensions.

!---------------------------------

      prob%n = s%n_active
      prob%m = s%m_active

      RETURN

1000  FORMAT( 5x, i4,' = ', 2(1x, i4 ), 3x,  a8, 3x, 2( 1x, i4), '  -->  ', i4 )
1001  FORMAT( 5x, i4,' = ', 2(1x, i4 ), 3x, a13, 3x, 2( 1x, i4), '  -->  ', i4 )

      END SUBROUTINE PRESOLVE_apply_permutations

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_do_fix_x( j, xval )

!     Perform the operations of fixing variable j at the value xval, updating
!     the relevant other problem quantities (objective function' gradient and
!     independent term, constraints bounds) and the data structures for A and H
!     to take the elimination of variable j into account.
!
!     Arguments:

      INTEGER, INTENT( IN ) :: j

!            the index of the variable to be fixed

      REAL( KIND = wp ), INTENT( IN ) :: xval

!            the value at which x(j) is to be fixed

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: k, i, ii, hj, hsj
      REAL ( KIND = wp ) :: ccorr, f_add, aval, hval

!     Fix the variable.

      prob%X_status( j ) = ELIMINATED
      s%n_active         = s%n_active - 1
      prob%X( j )        = xval

!     Determine the Hessian row structure.

      IF ( prob%H%ne > 0 ) THEN
         hsj = s%H_str( j )
      ELSE
         hsj = EMPTY
      END IF

!     Update the lhs and rhs and the value of the constraints

      IF ( s%A_col_s( j ) > 0 ) THEN

!        Remember the dependencies.

         s%needs( C_BNDS, A_VALS ) = MIN( s%tt, s%needs( C_BNDS, A_VALS ) )
         s%needs( C_BNDS, X_VALS ) = MIN( s%tt, s%needs( C_BNDS, X_VALS ) )
         IF ( xval /= ZERO ) THEN
            s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
            s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )
         END IF

         k = s%A_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO ii = 1, prob%m
               i  = s%A_row( k )
               IF ( prob%C_status( i ) > ELIMINATED ) THEN
                  aval  = prob%A%val( k )
                  IF ( aval /= ZERO ) THEN

!                    Update the constraint values

                     IF ( xval /= ZERO ) THEN
                        ccorr         = xval * aval
                        prob%C_l( i ) = prob%C_l( i ) - ccorr
                        prob%C_u( i ) = prob%C_u( i ) - ccorr
                        IF ( control%final_c_bounds /= TIGHTEST ) THEN
                           s%c_l2( i ) = prob%C_l( i )
                           s%c_u2( i ) = prob%C_u( i )
                        END IF
                      END IF

!                    Update the constraint size.

                     CALL PRESOLVE_decr_A_row_size( i )

                  END IF
               END IF
               k = s%A_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END DO
         END IF

          s%a_ne_active = s%a_ne_active - s%A_col_s( j )
         IF ( s%level >= DETAILS ) WRITE( s%out,* )                            &
            '   constraint values updated'
      END IF

!     Update the objective function for the fixed term in the gradient.

      IF ( xval /= ZERO ) THEN
         prob%f = prob%f + xval * prob%G( j )
         s%needs( F_VAL, G_VALS ) = MIN( s%tt, s%needs( F_VAL, G_VALS ) )
         s%needs( F_VAL, X_VALS ) = MIN( s%tt, s%needs( F_VAL, X_VALS ) )
      END IF

!     Update the gradient and objective function for the fixed terms
!     in the Hessian. This is achieved by looping on the j-th row of
!     the Hessian column by column, and updating gradient components
!     corresponding to active column indices. At the same time, we
!     update the description (in H_str) of the Hessian's structure to
!     reflect that column and row j become inactive.

      IF ( hsj /= EMPTY ) THEN

!        Update the number of active Hessian entries.

         s%h_ne_active = s%h_ne_active - PRESOLVE_H_row_s( j )
         s%H_str( j )  = EMPTY

!        Remember the dependencies

         IF ( xval /= ZERO ) THEN
            s%needs( G_VALS, H_VALS ) = MIN( s%tt, s%needs( G_VALS, H_VALS ) )
            s%needs( G_VALS, X_VALS ) = MIN( s%tt, s%needs( G_VALS, X_VALS ) )
            s%needs( Z_VALS, H_VALS ) = MIN( s%tt, s%needs( Z_VALS, H_VALS ) )
            s%needs( Z_VALS, X_VALS ) = MIN( s%tt, s%needs( Z_VALS, X_VALS ) )
         END IF

!        Update the objective for the diagonal term.

         k = prob%H%ptr( j + 1 ) - 1
         IF ( k > 0 ) THEN
            IF ( prob%H%col( k ) == j ) THEN
               f_add  = prob%H%val( k )
               IF ( f_add /= ZERO ) THEN
                  prob%f = prob%f + HALF * f_add * xval**2
                  s%needs( F_VAL, H_VALS ) = MIN( s%tt, s%needs( F_VAL, H_VALS))
                  s%needs( F_VAL, X_VALS ) = MIN( s%tt, s%needs( F_VAL, X_VALS))
               END IF
            END IF
         END IF

!        Loop over the subdiagonal elements in the j-th row
!        Note that the loop avoids the diagonal entry, as column j is
!        already eliminated.

         IF ( hsj < 0 ) THEN
            DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
               hj = prob%H%col( k )
               IF ( prob%X_status( hj ) <= ELIMINATED ) CYCLE
               hval = prob%H%val( k )
               IF ( hval == ZERO ) CYCLE
               IF ( xval /= ZERO ) prob%G( hj ) = prob%G( hj ) + xval * hval
               CALL PRESOLVE_decr_H_row_size( hj )
            END DO

!           Loop over the superdiagonal elements in the j-th column.

            k = s%H_col_f( j )
            IF ( END_OF_LIST /= k ) THEN
               DO ii = j + 1, prob%n
                  hj = s%H_row( k )
                  IF ( prob%X_status( hj ) > ELIMINATED) THEN
                     hval = prob%H%val( k )
                     IF ( hval /= ZERO ) THEN
                        IF ( xval /= ZERO ) prob%G(hj) = prob%G(hj)+xval*hval
                        CALL PRESOLVE_decr_H_row_size( hj )
                     END IF
                  END IF
                  k = s%H_col_n( k )
                  IF ( k == END_OF_LIST ) EXIT
               END DO
            END IF
         END IF

      END IF
      IF ( s%level >= DETAILS ) WRITE( s%out,* )                               &
            '   active gradient and objective function updated'

!     Remove x(j) from the lists of unconstrained, linear singletons and
!     doubletons.

      IF ( prob%m > 0 ) THEN
         SELECT CASE ( s%A_col_s( j ) )
         CASE ( 0 )
            CALL PRESOLVE_rm_from_list( j, s%unc_f )
         CASE ( 1 )
            CALL PRESOLVE_rm_from_list( j, s%lsc_f )
         CASE ( 2 )
            CALL PRESOLVE_rm_from_list( j, s%ldc_f )
         END SELECT
      ELSE
         CALL PRESOLVE_rm_from_list( j, s%unc_f )
      END IF

!     Update the column size in A

      s%A_col_s( j ) = 0

!     Remember what is needed to reconstruct the corresponding dual variable.

      s%needs( Z_VALS, G_VALS ) = MIN( s%tt, s%needs( Z_VALS, G_VALS ) )

      RETURN

      END SUBROUTINE PRESOLVE_do_fix_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_fix_x( j, xval, z_type, pos, zval )

!     Fix variable j at value xval and update the relevant other
!     problem quantities (by calling PRESOLVE_do_fix_x).
!
!     Arguments:

      INTEGER, INTENT( IN ) :: j

!            the index of the variable to be fixed

      REAL( KIND = wp ), INTENT( IN ) :: xval

!            the value at which the variable must be fixed

      INTEGER, INTENT( IN ) :: z_type

!            the method that must be used to restore the z(j), the dual
!            variable of x(j).  Possible values are:
!            Z_GIVEN         : the value of z(j) is known and equal to zval;
!            Z_FROM_YZ_LOW   : z(j) must be zeroed (if positive ) with a
!                              corresponding adaptation of a single multiplier
!                              and a single other dual variable
!            Z_FROM_YZ_UP    : z(j) must be zeroed (if negative ) with a
!                              corresponding adaptation of a single multiplier
!                              and a single other dual variable
!            Z_FROM_DUAL_FEAS: z(j) must be determined from the j-th dual
!                              feasibility equation (the general case).

      INTEGER, OPTIONAL, INTENT( IN ) :: pos

!            the position in A of the element A(i,k) in the case where
!            variable j has its bounds defined by a shift with variable k
!            using constraint i (required when z_type = Z_FROM_YZ_LOW or
!            Z_FROM_YZ_UP).


      REAL( KIND = wp ), OPTIONAL, INTENT( IN ) :: zval

!            the value of z(j) (required if z_type = Z_GIVEN).

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER :: l

!     Exit if the value(s) exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( xval ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    fixing x(', j, ') prevented by unacceptable growth'
         RETURN
      END IF
      IF ( z_type == Z_GIVEN ) THEN
         IF ( PRESOLVE_is_too_big( zval ) ) THEN
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    fixing x(', j, ') prevented by unacceptable growth'
            RETURN
         END IF
      END IF

!     Do nothing if the variable is already eliminated.

      IF ( prob%X_status( j ) <= ELIMINATED ) RETURN

!     Deactivate the variable.

      IF ( s%tm >= s%max_tm ) THEN
         CALL PRESOLVE_save_transf
         IF ( inform%status /= OK ) RETURN
      END IF
      l             = s%tm + 1
      s%tt          = s%tt + 1
      s%tm          = l
      s%hist_i( l ) = j
      s%hist_r( l ) = xval
      SELECT CASE ( z_type )

      CASE ( Z_FROM_DUAL_FEAS )

         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  [', s%tt, '] fixing x(', j, ') to',xval,                        &
            '[z from dual feasibility]'
         s%hist_type( l ) = X_FIXED_DF
         s%hist_j( l )    = 0

!        Remember the dependencies.

         s%needs( Z_VALS, G_VALS ) = MIN( s%tt, s%needs( Z_VALS, G_VALS ) )
         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) /= EMPTY ) THEN
               s%needs( Z_VALS, H_VALS ) = MIN( s%tt, s%needs( Z_VALS, H_VALS ))
               s%needs( Z_VALS, X_VALS ) = MIN( s%tt, s%needs( Z_VALS, X_VALS ))
            END IF
         END IF
         IF ( prob%A%ne > 0 ) THEN
            s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
            s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )
         END IF

      CASE ( Z_FROM_YZ_LOW )

         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  [', s%tt, '] fixing x(', j, ') to', xval, '[z from shift low]'
         s%hist_type( l ) = X_FIXED_SL
         s%hist_j( l )    = pos

!        Remember the dependencies.

         s%needs( Z_VALS, G_VALS ) = MIN( s%tt, s%needs( Z_VALS, G_VALS ) )
         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) /= EMPTY ) THEN
               s%needs( Z_VALS, H_VALS ) = MIN( s%tt, s%needs( Z_VALS, H_VALS ))
               s%needs( Z_VALS, X_VALS ) = MIN( s%tt, s%needs( Z_VALS, X_VALS ))
            END IF
         END IF
         IF ( prob%A%ne > 0 ) THEN
            s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
            s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )
         END IF

      CASE ( Z_FROM_YZ_UP )

         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  [', s%tt, '] fixing x(', j, ') to', xval, '[z from shift up]'
         s%hist_type( l ) = X_FIXED_SU
         s%hist_j( l )    = pos

!        Remember the dependencies.

         s%needs( Z_VALS, G_VALS ) = MIN( s%tt, s%needs( Z_VALS, G_VALS ) )
         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) /= EMPTY ) THEN
               s%needs( Z_VALS, H_VALS ) = MIN( s%tt, s%needs( Z_VALS, H_VALS ))
               s%needs( Z_VALS, X_VALS ) = MIN( s%tt, s%needs( Z_VALS, X_VALS ))
            END IF
         END IF
         IF ( prob%A%ne > 0 ) THEN
            s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
            s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )
         END IF

      CASE ( Z_GIVEN )

         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  [', s%tt, '] fixing x(', j, ') to', xval, '[ z(', j, ') =',     &
            zval,']'
         s%hist_type( l ) = X_FIXED_ZV
         prob%Z( j )      = zval
         CALL PRESOLVE_fix_z( j, zval )

      END SELECT

!     Perform the necessary updating operations.

      CALL PRESOLVE_do_fix_x( j, xval )
      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_fix_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_fix_y( i, yi )

!     Fix y(i) to its optimal value yi (by setting its lower and upper bounds
!     equal to yi).

!     Note that if the user requests the bounds on y to be reconstructed at
!     restoration, then the previous values of the bounds must be stored: the
!     "fixing" transformation is then replaced by updates of the lower and
!     upper bounds (to the same value yi).

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the multiplier to fix

      REAL ( KIND = wp ), INTENT( IN ) :: yi

!            the value at which the multiplier must be fixed

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER :: l

      IF ( yi <= s%M_INFINITY .OR. yi >= s%P_INFINITY ) THEN
         inform%status = PRIMAL_INFEASIBLE
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE analysis stopped: the problem is primal infeasible'
         WRITE( inform%message( 2 ), * ) '    because y(', i, ') is infinite'
         RETURN
      END IF

!     Exit if the value exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( yi ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    fixing y(', i, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current multiplier.

      prob%Y( i ) = yi

!     The bounds on y must be reconstructed at restoration.

      IF ( control%get_y_bounds ) THEN

         CALL PRESOLVE_do_fix_y_r( i, yi )

!     The bounds on y must not be reconstructed at restoration.

      ELSE

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = Y_FIXED
         s%hist_i( l )    = i
         s%hist_j( l )    = 0
         s%hist_r( l )    = yi
         prob%Y_u( i )    = yi
         prob%Y_l( i )    = yi
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  [', s%tt, '] fixing y(', i, ') to', yi
         IF ( s%tt >= control%max_nbr_transforms )                             &
            inform%status = MAX_NBR_TRANSF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_fix_y

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_do_fix_y_r( i, yi )

!     Fix y(i) to its optimal value yi while remembering the previous values
!     of the bounds on y(i).

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the multiplier to fix

      REAL ( KIND = wp ), INTENT( IN ) :: yi

!            the value at which the multiplier must be fixed

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER :: l

!     Upper bound

      IF ( s%tm >= s%max_tm ) THEN
         CALL PRESOLVE_save_transf
         IF ( inform%status /= OK ) RETURN
      END IF

      l                = s%tm + 1
      s%tt             = s%tt + 1
      s%tm             = l
      s%hist_type( l ) = Y_UPPER_UPDATED
      s%hist_i( l )    = i
      s%hist_j( l )    = SET
      s%hist_r( l )    = prob%Y_u( i )
      prob%Y_u( i )    = yi
      IF (s%level >= ACTION ) WRITE( s%out, * )                   &
          '  [', s%tt, '] fixing upper bound for y(', i, ') to', yi
      IF ( s%tt >= control%max_nbr_transforms ) THEN
         inform%status = MAX_NBR_TRANSF
         RETURN
      END IF

!     Lower bound

      IF ( s%tm >= s%max_tm ) THEN
         CALL PRESOLVE_save_transf
         IF ( inform%status /= OK ) RETURN
      END IF

      l                = s%tm + 1
      s%tt             = s%tt + 1
      s%tm             = l
      s%hist_type( l ) = Y_LOWER_UPDATED
      s%hist_i( l )    = i
      s%hist_j( l )    = SET
      s%hist_r( l )    = prob%Y_l( i )
      prob%Y_l( i )    = yi
      IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
         '  [', s%tt, '] fixing lower bound for y(', i, ') to', yi
      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_do_fix_y_r

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_fix_z( j, zj )

!     Fix z(j) to its optimal value zj

!     Note that if the user requests the bounds on z to be reconstructed at
!     restoration, then the previous values of the bounds must be stored: the
!     "fixing" transformation is then replaced by updates of the lower and
!     upper bounds (to the same value zj).

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!            the index of the dual variable to be fixed

      REAL ( KIND = wp ), INTENT( IN ) :: zj

!            the value at which z(j) must be fixed

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER :: l

      IF ( zj <= s%M_INFINITY .OR. zj >= s%P_INFINITY ) THEN
         inform%status = PRIMAL_INFEASIBLE
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE analysis stopped: the problem is primal infeasible'
         WRITE( inform%message( 2 ), * ) '    because z(', j, ') is infinite'
         RETURN
      END IF

!     Exit if the value exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( zj ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    fixing z(', j, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current multiplier.

!     The bounds on z must be reconstructed at restoration.  This requires
!     that the values of both the lower and upper bounds must be remebered,
!     and thus stored in the transformation history.

      IF ( control%get_z_bounds ) THEN

!        The upper bound

         IF ( zj < prob%Z_u( j ) ) THEN
            IF ( s%tm >= s%max_tm ) THEN
               CALL PRESOLVE_save_transf
               IF ( inform%status /= OK ) RETURN
            END IF
            l                = s%tm + 1
            s%tt             = s%tt + 1
            s%tm             = l
            s%hist_type( l ) = Z_UPPER_UPDATED
            s%hist_i( l )    = j
            s%hist_j( l )    = SET
            s%hist_r( l )    = prob%Z_u( j )
            prob%Z_u( j )    = zj
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
               '  [', s%tt, '] fixing upper bound for z(', j, ') to', zj
            IF ( s%tt >= control%max_nbr_transforms ) THEN
               inform%status = MAX_NBR_TRANSF
               RETURN
            END IF
         END IF

!        The lower bound

         IF ( zj > prob%Z_l( j ) ) THEN
            IF ( s%tm >= s%max_tm ) THEN
               CALL PRESOLVE_save_transf
               IF ( inform%status /= OK ) RETURN
            END IF
            l                = s%tm + 1
            s%tt             = s%tt + 1
            s%tm             = l
            s%hist_type( l ) = Z_LOWER_UPDATED
            s%hist_i( l )    = j
            s%hist_j( l )    = SET
            s%hist_r( l )    = prob%Z_l( j )
            prob%Z_l( j )    = zj
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
               '  [', s%tt, '] fixing lower bound for z(', j, ') to', zj
         END IF

!     The bounds on z must not be reconstructed at restoration.  The previous
!     values of the bounds are thus irrelevant and it is enough to update
!     the current ones without remebering their previous values.

      ELSE

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = Z_FIXED
         s%hist_i( l )    = j
         s%hist_j( l )    = 0
         s%hist_r( l )    = zj
         prob%Z_u( j )    = zj
         prob%Z_l( j )    = zj
         IF ( s%level >= ACTION ) WRITE( s%out, * )                &
            '  [', s%tt, '] fixing z(', j, ') to', zj

      END IF

      prob%Z( j ) = zj

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_fix_z

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_do_ignore_x( j )

!     Updates the Jacobian and Hessian row structures to reflect the fact that
!     variable j is being deactivated.
!
!     Argument:

      INTEGER, INTENT( IN ) :: j

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER :: k, ii, hj, i

!     Deactivate the variable.

      prob%X_status( j ) = ELIMINATED
      s%n_active         = s%n_active - 1

!     Update the Hessian structure.

      IF ( prob%H%ne > 0 ) THEN

         s%h_ne_active = s%h_ne_active - PRESOLVE_H_row_s( j )
         s%H_str( j )  = EMPTY

!        Subdiagonal part of row j
!        Note that the diagonal element is avoided since column j
!        is already eliminated.

         DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
            IF ( prob%H%val( k ) == ZERO ) CYCLE
            hj = prob%H%col( k )
            IF ( prob%X_status( hj ) <= ELIMINATED ) CYCLE
            CALL PRESOLVE_decr_H_row_size( hj )
         END DO

!        Superdiagonal part of row j

         k = s%H_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO ii = j + 1, prob%n
               IF ( prob%H%val( k ) == ZERO ) CYCLE
               hj = s%H_row( k )
               IF ( prob%X_status( hj ) > ELIMINATED ) THEN
                  CALL PRESOLVE_decr_H_row_size( hj )
               END IF
               k = s%H_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END DO
         END IF

!        Remember the dependencies.

         s%needs( Z_VALS, H_VALS ) = MIN ( s%tt, s%needs( Z_VALS, H_VALS ) )
         s%needs( Z_VALS, X_VALS ) = MIN ( s%tt, s%needs( Z_VALS, X_VALS ) )

      END IF

!     Update the Jacobian structure.

      IF ( s%A_col_s( j ) > 0 ) THEN

         s%a_ne_active = s%a_ne_active - s%A_col_s( j )

         k = s%A_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO ii = 1, prob%m
               i  = s%A_row( k )
               IF ( prob%C_status( i ) >  ELIMINATED .AND. &
                    prob%A%val( k )    /= ZERO             )                   &
                  CALL PRESOLVE_decr_A_row_size( i )
               k = s%A_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END DO
         END IF

!        Remember the dependencies.

         s%needs( Z_VALS, A_VALS ) = MIN ( s%tt, s%needs( Z_VALS, A_VALS ) )
         s%needs( Z_VALS, Y_VALS ) = MIN ( s%tt, s%needs( Z_VALS, Y_VALS ) )

      END IF
      s%A_col_s( j ) = 0

      s%needs( Z_VALS, G_VALS ) = MIN ( s%tt, s%needs( Z_VALS, G_VALS ) )

      RETURN

      END SUBROUTINE PRESOLVE_do_ignore_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_ignore_x( j )

!     Ignore variable j.

!     Argument:

      INTEGER, INTENT( IN ) :: j

!            the index of the variable to ignore

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variable

      INTEGER :: l

!     Ignore the variable.

      IF ( s%tm >= s%max_tm ) THEN
         CALL PRESOLVE_save_transf
         IF ( inform%status /= OK ) RETURN
      END IF
      l    = s%tm + 1
      s%tt = s%tt + 1
      s%tm = l
      IF ( s%level >= ACTION ) WRITE( s%out, * )'  [',s%tt,'] ignoring x(',j,')'
      s%hist_type( l ) = X_IGNORED
      s%hist_i( l )    = j
      s%hist_j( l )    = 0
      CALL PRESOLVE_guess_x( j, prob%X( j ), prob, s )
      s%hist_r( l ) = prob%X( j )

!     Update the data structures for A and H.

      CALL PRESOLVE_do_ignore_x( j )
      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_ignore_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_set_bound_x( j, lowup, bound )

!     Sets a new bound for variable j.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!              the index of the considered variable

      INTEGER, INTENT( IN ) :: lowup

!              LOWER or UPPER, for lower or upper bound

      REAL ( KIND = wp ), INTENT( IN ) :: bound

!              the new lower or upper bound on x_j

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: b, xlj, xuj

!     Exit if the bound exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( bound ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    bounding x(', j, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current bounds.

      xlj = prob%X_l( j )
      xuj = prob%X_u( j )

!     Update the variable's bounds.

      SELECT CASE ( lowup )

      CASE ( UPPER )

!        Check the new bound.

!        Round the new bound to the existing one, if close enough.
!        In that case there is nothing to do.
!        Also round the values that are undistinguishable from zero .

         IF ( PRESOLVE_is_zero( bound ) ) THEN
            b = ZERO
         ELSE
            b = MIN( s%INFINITY, bound )
         END IF
         IF ( control%final_x_bounds /= TIGHTEST ) s%x_u2( j ) = b
         IF ( b /= xuj ) THEN

            IF ( b <= s%M_INFINITY ) THEN
               inform%status = DUAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                    '    because x(', j, ') has an infinite active upper bound'
               RETURN
            END IF

!           Record the transformation.

            IF ( s%tm >= s%max_tm ) THEN
               CALL PRESOLVE_save_transf
               IF ( inform%status /= OK ) RETURN
            END IF
            l                = s%tm + 1
            s%tt             = s%tt + 1
            s%tm             = l
            s%hist_type( l ) = X_UPPER_UPDATED
            s%hist_i( l )    = j
            s%hist_j( l )    = prob%X_status( j )
            s%hist_r( l )    = xuj
            prob%X_u( j )    = b
            prob%X( j )      = MIN( prob%X( j ), b )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                    '  [', s%tt, '] setting upper bound for x(', j, ') to', b
         ELSE
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
                    '    upper bound on x(', j, ') already equal to', b
         END IF

!        Reset the variable's implied status.

         IF ( b < s%P_INFINITY ) THEN
            SELECT CASE ( prob%X_status( j ) )
            CASE ( FREE )
               prob%X_status( j ) = UPPER
            CASE ( LOWER )
               prob%X_status( j ) = RANGE
            END SELECT
         ELSE
            SELECT CASE ( prob%X_status( j ) )
            CASE ( RANGE )
               prob%X_status( j ) = LOWER
            CASE ( UPPER )
               prob%X_status( j ) = FREE
            END SELECT
         END IF

         IF ( s%level >= DEBUG ) THEN
            SELECT CASE ( prob%X_status( j ) )
            CASE ( FREE )
               WRITE( s%out, * ) '    x(',j,') now has implied status FREE'
            CASE ( LOWER )
               WRITE( s%out, * ) '    x(',j,') now has implied status LOWER'
            CASE ( UPPER )
               WRITE( s%out, * ) '    x(',j,') now has implied status UPPER'
            CASE ( RANGE )
               WRITE( s%out, * ) '    x(',j,') now has implied status RANGE'
            END SELECT
         END IF

      CASE ( LOWER )

!        Check the new bound.

!        Round the new bound to the existing one, if close enough.
!        In that case there is nothing to do.
!        Also round the values that are undistinguishable from zero .

         IF ( PRESOLVE_is_zero( bound ) ) THEN
            b = ZERO
         ELSE
            b = MAX( -s%INFINITY, bound )
         END IF
         IF ( control%final_x_bounds /= TIGHTEST ) s%x_l2( j ) = b

         IF ( xlj /= b ) THEN

            IF ( b >= s%P_INFINITY ) THEN
               inform%status = DUAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE analysis stopped: the problem is dual infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                  '    because x(', j, ') has an infinite active lower bound'
               RETURN
            END IF

!           Record the transformation.

            IF ( s%tm >= s%max_tm ) THEN
               CALL PRESOLVE_save_transf
               IF ( inform%status /= OK ) RETURN
            END IF
            l                = s%tm + 1
            s%tt             = s%tt + 1
            s%tm             = l
            s%hist_type( l ) = X_LOWER_UPDATED
            s%hist_i( l )    = j
            s%hist_j( l )    = prob%X_status( j )
            s%hist_r( l )    = xlj
            prob%X_l( j )    = b
            prob%X( j )      = MAX( prob%X( j ), b )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  [', s%tt, '] setting lower bound for x(', j, ') to', b
         ELSE
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
                    '    lower bound on x(', j, ') already equal to', b
         END IF

!        Reset the variable's implied status.

         IF ( b > s%M_INFINITY ) THEN
            SELECT CASE ( prob%X_status( j ) )
            CASE ( FREE )
               prob%X_status( j ) = LOWER
            CASE ( UPPER )
               prob%X_status( j ) = RANGE
            END SELECT
         ELSE
            SELECT CASE ( prob%X_status( j ) )
            CASE ( RANGE )
               prob%X_status( j ) = UPPER
            CASE ( LOWER )
               prob%X_status( j ) = FREE
            END SELECT
         END IF

         IF ( s%level >= DEBUG ) THEN
            SELECT CASE ( prob%X_status( j ) )
            CASE ( FREE )
               WRITE( s%out, * ) '    x(',j,') now has implied status FREE'
            CASE ( LOWER )
               WRITE( s%out, * ) '    x(',j,') now has implied status LOWER'
            CASE ( UPPER )
               WRITE( s%out, * ) '    x(',j,') now has implied status UPPER'
            CASE ( RANGE )
               WRITE( s%out, * ) '    x(',j,') now has implied status RANGE'
            END SELECT
         END IF

      END SELECT

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_set_bound_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_primal_bound_x( j, k, lowup, bound )

!     Tighten the bound for variable j as a result of the analysis of
!     a primal constraint.  In doing so, verify that the new bound
!     preserves primal feasibility for the problem. Also maintain the
!     implied status of variable j.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!              the index of the considered variable

      INTEGER, INTENT( IN ) :: k

!              the position in A of A(i,j)

      INTEGER, INTENT( IN ) :: lowup

!              LOWER or UPPER, for lower or upper bound

      REAL ( KIND = wp ), INTENT( IN ) :: bound

!              the new lower or upper bound on x_j

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: b, xlj, xuj

!     Exit if the bound exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( bound ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    primal bounding x(', j, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current bounds.

      xlj = prob%X_l( j )
      xuj = prob%X_u( j )

!     Update the variable's bounds.

      SELECT CASE ( lowup )

      CASE ( UPPER )

!        Round the new bound to the existing one, if close enough.
!        In that case there is nothing to do.
!        Also round the values that are undistinguishable from zero .

         IF ( PRESOLVE_is_zero( xuj - bound ) ) RETURN
         IF ( PRESOLVE_is_zero( bound ) ) THEN
            b = ZERO
         ELSE
            b = MIN( s%INFINITY, bound )
         END IF

!        Verify that the improvement is large enough.

         IF ( b >= xuj - s%mrbi * MAX( ONE, ABS( xuj ) ) ) RETURN

          IF ( b <= s%M_INFINITY ) THEN
             inform%status = DUAL_INFEASIBLE
             WRITE( inform%message( 1 ), * )                                   &
                ' PRESOLVE analysis stopped: the problem is dual infeasible'
             WRITE( inform%message( 2 ), * )                                   &
                '    because x(', j, ') has an infinite active upper bound'
             RETURN
          END IF

!         Remember the best non-degenerate bound, if requested.

          IF ( control%final_x_bounds == NON_DEGENERATE ) s%x_u2( j ) = xuj

!         Update the variable's implied status.

          IF ( PRESOLVE_IS_ZERO( b - xlj ) ) THEN
             CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
             RETURN
          ELSE
             SELECT CASE ( prob%X_status( j ) )
             CASE ( RANGE )
                prob%X_status( j ) = LOWER
             CASE ( UPPER )
                prob%X_status( j ) = FREE
             END SELECT
          END IF

!         Record the transformation.

          IF ( s%tm >= s%max_tm ) THEN
             CALL PRESOLVE_save_transf
             IF ( inform%status /= OK ) RETURN
          END IF
          l                = s%tm + 1
          s%tt             = s%tt + 1
          s%tm             = l
          s%hist_type( l ) = X_UPPER_UPDATED_P
          s%hist_i( l )    = j
          s%hist_j( l )    = k
          s%hist_r( l )    = xuj
          prob%X_u( j )    = b
          prob%X( j )      = MIN( prob%X( j ), b )

!         Printout the requested information.

          IF ( s%level >= ACTION ) THEN
             WRITE( s%out, * ) '  [', s%tt, '] tightening upper bound for x(', &
                  j, ') to', b, '(p)'
             IF ( s%level >= DEBUG ) THEN
                SELECT CASE ( prob%X_status( j ) )
                CASE ( FREE )
                   WRITE( s%out, * )'    x(',j,') now has implied status FREE'
                CASE ( LOWER )
                   WRITE( s%out, * )'    x(',j,') now has implied status LOWER'
                CASE ( UPPER )
                   WRITE( s%out, * )'    x(',j,') now has implied status UPPER'
                CASE ( RANGE )
                   WRITE( s%out, * )'    x(',j,') now has implied status RANGE'
                END SELECT
             END IF
          END IF

      CASE ( LOWER )

!        Round the new bound to the existing one, if close enough.
!        In that case there is nothing to do.
!        Also round the values that are undistinguishable from zero .

         IF ( PRESOLVE_is_zero( xlj - bound ) ) RETURN
         IF ( PRESOLVE_is_zero( bound ) ) THEN
            b = ZERO
         ELSE
            b = MAX( -s%INFINITY, bound )
         END IF

!        Verify that the improvement is large enough.

         IF ( b <= xlj + s%mrbi * MAX( ONE, ABS( xlj ) ) ) RETURN

          IF ( b >= s%P_INFINITY ) THEN
             inform%status = DUAL_INFEASIBLE
             WRITE( inform%message( 1 ), * )                                   &
                ' PRESOLVE analysis stopped: the problem is dual infeasible'
             WRITE( inform%message( 2 ), * )                                   &
                '    because x(', j, ') has an infinite active lower bound'
             RETURN
          END IF

!         Remember the best non-degenerate bound, if requested.

          IF ( control%final_x_bounds == NON_DEGENERATE ) s%x_l2( j ) = xlj

!         Update the variable's implied status.

          IF ( PRESOLVE_is_zero( b - xuj ) ) THEN
             CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
             RETURN
          ELSE
             SELECT CASE ( prob%X_status( j ) )
             CASE ( RANGE )
                prob%X_status( j ) = UPPER
             CASE ( LOWER )
                prob%X_status( j ) = FREE
             END SELECT
          END IF

!         Record the transformation.

          IF ( s%tm >= s%max_tm ) THEN
             CALL PRESOLVE_save_transf
             IF ( inform%status /= OK ) RETURN
          END IF
          l                = s%tm + 1
          s%tt             = s%tt + 1
          s%tm             = l
          s%hist_type( l ) = X_LOWER_UPDATED_P
          s%hist_i( l )    = j
          s%hist_j( l )    = k
          s%hist_r( l )    = xlj
          prob%X_l( j )    = b
          prob%X( j )      = MAX( prob%X( j ), b )

!         Printout the requested information.

          IF ( s%level >= ACTION ) THEN
             WRITE( s%out, * ) '  [', s%tt, '] tightening lower bound for x(', &
                  j, ') to', b, '(p)'
             IF ( s%level >= DEBUG ) THEN
                SELECT CASE ( prob%X_status( j ) )
                CASE ( FREE )
                   WRITE( s%out, * )'    x(',j,') now has implied status FREE'
                CASE ( LOWER )
                   WRITE( s%out, * )'    x(',j,') now has implied status LOWER'
                CASE ( UPPER )
                   WRITE( s%out, * )'    x(',j,') now has implied status UPPER'
                CASE ( RANGE )
                   WRITE( s%out, * )'    x(',j,') now has implied status RANGE'
                END SELECT
             END IF
          END IF

      END SELECT

!     Ensure that the duals may be recovered at RESTORE.

      s%needs( Y_VALS, Z_VALS ) = MIN( s%tt, s%needs( Y_VALS, Z_VALS ) )
      s%needs( Y_VALS, A_VALS ) = MIN( s%tt, s%needs( Y_VALS, A_VALS ) )
      s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
      s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )

!     Check for primal infeasibility.

      IF ( prob%X_l( j ) > prob%X_u( j ) ) THEN
         inform%status = PRIMAL_INFEASIBLE
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE analysis stopped: the problem is primal infeasible'
         WRITE( inform%message( 2 ), * )                                       &
            '    because the bounds on x(', j, ') are incompatible'
      END IF

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_primal_bound_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_dual_bound_x( j, lowup, bound )

!     Tighten the  bound for variable j as a result of the analysis of
!     a dual constraint.  In doing so, verify that the new bound
!     preserves primal feasibility for the problem. Also maintain the
!     implied status of variable j.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!              the index of the considered variable

      INTEGER, INTENT( IN ) :: lowup

!              LOWER or UPPER, for lower or upper bound

      REAL ( KIND = wp ), INTENT( IN ) :: bound

!              the new lower or upper bound on x_j

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: b, xlj, xuj

!     Exit if the bound exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( bound ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    dual bounding x(', j, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current bounds.

      xlj = prob%X_l( j )
      xuj = prob%X_u( j )

!     Update the variable's bounds.

      SELECT CASE ( lowup )

      CASE ( UPPER )

!        Round the new bound to the existing one, if close enough.
!        In that case there is nothing to do.
!        Also round the values that are undistinguishable from zero .

         IF ( PRESOLVE_is_zero( xuj - bound ) ) RETURN
         IF ( PRESOLVE_is_zero( bound ) ) THEN
            b = ZERO
         ELSE
            b = MIN( s%INFINITY, bound )
         END IF
         IF ( b >= xuj - s%mrbi * MAX( ONE, ABS( xuj ) ) ) RETURN

          IF ( b <= s%M_INFINITY ) THEN
             inform%status = DUAL_INFEASIBLE
             WRITE( inform%message( 1 ), * )                                   &
                ' PRESOLVE analysis stopped: the problem is dual infeasible'
             WRITE( inform%message( 2 ), * )                                   &
                '    because x(', j, ') has an infinite active upper bound'
             RETURN
          END IF

!         Remember the best non-degenerate bound, if requested.

          IF ( control%final_x_bounds == NON_DEGENERATE ) s%x_u2( j ) = xuj

!         Update the variable's implied status.

          IF ( PRESOLVE_IS_ZERO( b - xlj ) ) THEN
             CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
             RETURN
          ELSE
             SELECT CASE ( prob%X_status( j ) )
             CASE ( RANGE )
                prob%X_status( j ) = LOWER
             CASE ( UPPER )
                prob%X_status( j ) = FREE
             END SELECT
          END IF

!         Record the transformation.

          IF ( s%tm >= s%max_tm ) THEN
             CALL PRESOLVE_save_transf
             IF ( inform%status /= OK ) RETURN
          END IF
          l                = s%tm + 1
          s%tt             = s%tt + 1
          s%tm             = l
          s%hist_type( l ) = X_UPPER_UPDATED_D
          s%hist_i( l )    = j
          s%hist_j( l )    = 0
          s%hist_r( l )    = xuj
          prob%X_u( j )    = b
          prob%X( j )      = MIN( prob%X( j ), b )

!         Printout the requested information.

          IF ( s%level >= ACTION ) THEN
             WRITE( s%out, * ) '  [', s%tt, '] tightening upper bound for x(', &
                  j, ') to', b, '(d)'
             IF ( s%level >= DEBUG ) THEN
                SELECT CASE ( prob%X_status( j ) )
                CASE ( FREE )
                   WRITE( s%out, * )'    x(',j,') now has implied status FREE'
                CASE ( LOWER )
                   WRITE( s%out, * )'    x(',j,') now has implied status LOWER'
                CASE ( UPPER )
                   WRITE( s%out, * )'    x(',j,') now has implied status UPPER'
                CASE ( RANGE )
                   WRITE( s%out, * )'    x(',j,') now has implied status RANGE'
                END SELECT
             END IF
          END IF

      CASE ( LOWER )

!        Round the new bound to the existing one, if close enough.
!        In that case there is nothing to do.
!        Also round the values that are undistinguishable from zero .

         IF ( PRESOLVE_is_zero( xlj - bound ) ) RETURN
         IF ( PRESOLVE_is_zero( bound ) ) THEN
            b = ZERO
         ELSE
            b = MAX( -s%INFINITY, bound )
         END IF
         IF ( b <= xlj + s%mrbi * MAX( ONE, ABS( xlj ) ) ) RETURN

          IF ( b >= s%P_INFINITY ) THEN
             inform%status = DUAL_INFEASIBLE
             WRITE( inform%message( 1 ), * )                                   &
                ' PRESOLVE analysis stopped: the problem is dual infeasible'
             WRITE( inform%message( 2 ), * )                                   &
                '    because x(', j, ') has an infinite active lower bound'
             RETURN
          END IF

!         Remember the best non-degenerate bound, if requested.

          IF ( control%final_x_bounds == NON_DEGENERATE ) s%x_l2( j ) = xlj

!         Update the variable's implied status

          IF ( PRESOLVE_is_zero( b - xuj ) ) THEN
             CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
             RETURN
          ELSE
             SELECT CASE ( prob%X_status( j ) )
             CASE ( RANGE )
                prob%X_status( j ) = UPPER
             CASE ( LOWER )
                prob%X_status( j ) = FREE
             END SELECT
          END IF

!         Record the transformation

          IF ( s%tm >= s%max_tm ) THEN
             CALL PRESOLVE_save_transf
             IF ( inform%status /= OK ) RETURN
          END IF
          l                = s%tm + 1
          s%tt             = s%tt + 1
          s%tm             = l
          s%hist_type( l ) = X_LOWER_UPDATED_D
          s%hist_i( l )    = j
          s%hist_j( l )    = 0
          s%hist_r( l )    = xlj
          prob%X_l( j )    = b
          prob%X( j )       = MAX( prob%X( j ), b )

!         Printout the requested information

          IF ( s%level >= ACTION ) THEN
             WRITE( s%out, * ) '  [', s%tt, '] tightening lower bound for x(', &
                  j, ') to', b, '(d)'
             IF ( s%level >= DEBUG ) THEN
                SELECT CASE ( prob%X_status( j ) )
                CASE ( FREE )
                   WRITE( s%out, * )'    x(',j,') now has implied status FREE'
                CASE ( LOWER )
                   WRITE( s%out, * )'    x(',j,') now has implied status LOWER'
                CASE ( UPPER )
                   WRITE( s%out, * )'    x(',j,') now has implied status UPPER'
                CASE ( RANGE )
                   WRITE( s%out, * )'    x(',j,') now has implied status RANGE'
                END SELECT
             END IF
          END IF

      END SELECT

!     Ensure that the duals may be recovered at RESTORE.

      s%needs( Y_VALS, Z_VALS ) = MIN( s%tt, s%needs( Y_VALS, Z_VALS ) )
      s%needs( Y_VALS, A_VALS ) = MIN( s%tt, s%needs( Y_VALS, A_VALS ) )
      s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
      s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )

!     Check for primal infeasibility.

      IF ( prob%X_l( j ) > prob%X_u( j ) ) THEN
         inform%status = PRIMAL_INFEASIBLE
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE analysis stopped: the problem is primal infeasible'
         WRITE( inform%message( 2 ), * )                                       &
            '    because the bounds on x(', j, ') are incompatible'
      END IF

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_dual_bound_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_bound_y( i, lowup, type, bound )

!     Set a new bound for multiplier i, or tighten the existing one.  In doing
!     so, verify that the new bound preserves dual feasibility for the
!     problem.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!              the index of the considered multiplier

      INTEGER, INTENT( IN ) :: lowup

!              LOWER or UPPER, for lower or upper bound

      INTEGER, INTENT( IN ) :: type

!              SET or TIGHTEN, depending whether the specified bound should be
!              tightened (that is replaced by a stronger bound), or set,
!              irrespective of its previous value

      REAL ( KIND = wp ), INTENT( IN ) :: bound

!              the new lower or upper bound on y_i

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!    Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: b, yli, yui, cbnd

!     Exit if the bound exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( bound ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    bounding y(', i, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current bounds.

      yli = prob%Y_l( i )
      yui = prob%Y_u( i )

      SELECT CASE ( lowup )

      CASE ( UPPER )

         b   = MIN( s%INFINITY, bound )
         IF ( PRESOLVE_is_zero( b - yli ) ) b = yli
         IF ( b >= yui - s%mrbi * MAX( ONE, ABS( yui ) ) .AND. &
               ( type == TIGHTEN .OR. b == yui )             ) RETURN
         IF ( b <= s%M_INFINITY ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is primal infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because y(', i, ') has an infinite active upper bound'
            RETURN
         END IF

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tm             = l
         s%tt             = s%tt + 1
         s%hist_type( l ) = Y_UPPER_UPDATED
         s%hist_i( l )    = i
         s%hist_j( l )    = type
         s%hist_r( l )    = yui
         prob%Y_u( i )    = b
         IF ( type == TIGHTEN ) THEN
            IF ( control%final_y_bounds == NON_DEGENERATE ) s%y_u2( i ) = yui
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                  '  [', s%tt, '] tightening upper bound for y(', i, ') to', b
         ELSE
            IF ( control%final_y_bounds /= TIGHTEST ) s%y_u2( i ) = b
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                  '  [', s%tt, '] setting upper bound for y(', i, ') to', b
         END IF

      CASE ( LOWER )

         b   = MAX( -s%INFINITY, bound )
         IF ( PRESOLVE_is_zero( b - yui ) ) b = yui
         IF ( b <= yli + s%mrbi * MAX( ONE, ABS( yli ) ) .AND. &
              ( type == TIGHTEN .OR. b == yli )              ) RETURN
         IF ( b >= s%P_INFINITY ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is primal infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because y(', i, ') has an infinite active lower bound'
            RETURN
         END IF

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tm             = l
         s%tt             = s%tt + 1
         s%hist_type( l ) = Y_LOWER_UPDATED
         s%hist_i( l )    = i
         s%hist_j( l )    = type
         s%hist_r( l )    = yli
         prob%Y_l( i )    = b
         IF ( type == TIGHTEN ) THEN
            IF ( control%final_y_bounds == NON_DEGENERATE ) s%y_l2( i ) = yli
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
                  '  [', s%tt, '] tightening lower bound for y(', i, ') to', b
         ELSE
            IF ( control%final_y_bounds /= TIGHTEST ) s%y_l2( i ) = b
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
                    '  [', s%tt, '] setting lower bound for y(', i, ') to', b
         END IF

      END SELECT

!     Check for dual infeasibility.

      IF ( type == TIGHTEN .AND. prob%Y_l( i ) > prob%Y_u( i ) ) THEN
         inform%status = DUAL_INFEASIBLE
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE analysis stopped: the problem is dual infeasible'
         WRITE( inform%message( 2 ), * )                                       &
            '    because the bounds on y(', i, ') are incompatible'
         RETURN
      END IF

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) THEN
         inform%status = MAX_NBR_TRANSF
         RETURN
      END IF

!     See if constraint i can be fixed to one of its bounds.

      IF ( PRESOLVE_is_neg( b ) .AND. lowup == UPPER ) THEN
         cbnd = prob%C_u( i )
         IF ( cbnd < s%P_INFINITY ) THEN
            CALL PRESOLVE_bound_c( i, LOWER, SET, cbnd )
         ELSE
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because c(', i, ') has an infinite active upper bound'
            RETURN
         END IF
      END IF
      IF ( PRESOLVE_is_pos( b ) .AND. lowup == LOWER ) THEN
         cbnd = prob%C_l( i )
         IF ( cbnd > s%M_INFINITY ) THEN
            CALL PRESOLVE_bound_c( i, UPPER, SET, cbnd )
         ELSE
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because c(', i, ') has an infinite active lower bound'
            RETURN
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_bound_y

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_bound_z( j, lowup, type, bound )

!     Set a new bound for the dual variable j, or tighten the existing one.
!     In doing so, verify that the new bound preserves dual feasibility for
!     the problem.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!              the index of the considered dual variable

      INTEGER, INTENT( IN ) :: lowup

!              LOWER or UPPER, for lower or upper bound

      INTEGER, INTENT( IN ) :: type

!              SET or TIGHTEN, depending whether the specified bound should be
!              tightened (that is replaced by a stronger bound), or set,
!              irrespective of its previous value

      REAL ( KIND = wp ), INTENT( IN ) :: bound

!              the new lower or upper bound on z_j

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: b, zlj, zuj, xbnd

!     Exit if the bound exceeds the maximum value acceptable in the
!     reduced problem.

      IF ( PRESOLVE_is_too_big( bound ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    bounding z(', j, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current bounds.

      zuj = prob%Z_u( j )
      zlj = prob%Z_l( j )

      SELECT CASE ( lowup )

      CASE ( UPPER )

         b   = MIN( s%INFINITY, bound )
         IF ( PRESOLVE_is_zero( b - zlj ) ) THEN
            b = zlj
         ELSE
            IF ( b >= zuj - s%mrbi * MAX( ONE, ABS( zuj ) ) .AND. &
                 ( type == TIGHTEN .OR. b == zuj )              ) RETURN
            IF ( b <= s%M_INFINITY ) THEN
               inform%status = PRIMAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE analysis stopped: the problem is primal infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                  '    because z(', j, ') has an infinite active upper bound'
               RETURN
            END IF
         END IF

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tm             = l
         s%tt             = s%tt + 1
         s%hist_type( l ) = Z_UPPER_UPDATED
         s%hist_i( l )    = j
         s%hist_j( l )    = type
         s%hist_r( l )    = zuj
         prob%Z_u( j )    = b
         IF ( type == TIGHTEN ) THEN
            IF ( control%final_z_bounds == NON_DEGENERATE ) s%z_u2( j ) = zuj
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                  '  [', s%tt, '] tightening upper bound for z(', j, ') to', b
         ELSE
            IF ( control%final_z_bounds /= TIGHTEST ) s%z_u2( j ) = b
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                    '  [', s%tt, '] setting upper bound for z(', j, ') to', b
         END IF

      CASE ( LOWER )

         b   = MAX( -s%INFINITY, bound )
         IF ( PRESOLVE_is_zero( b - zuj ) ) THEN
            b = zuj
         ELSE
            IF ( b <= zlj + s%mrbi * MAX( ONE, ABS( zlj ) ) .AND. &
                 ( type == TIGHTEN .OR. b == zlj )              ) RETURN
            IF ( b >= s%P_INFINITY ) THEN
               inform%status = PRIMAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE analysis stopped: the problem is primal infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                  '    because z(', j, ') has an infinite active lower bound'
               RETURN
            END IF
         END IF

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tm             = l
         s%tt             = s%tt + 1
         s%hist_type( l ) = Z_LOWER_UPDATED
         s%hist_i( l )    = j
         s%hist_j( l )    = type
         s%hist_r( l )    = zlj
         prob%Z_l( j )    = b
         IF ( type == TIGHTEN ) THEN
            IF ( control%final_z_bounds == NON_DEGENERATE ) s%z_l2( j ) = zlj
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                    '  [', s%tt, '] tightening lower bound for z(', j, ') to', b
         ELSE
            IF ( control%final_z_bounds /= TIGHTEST ) s%z_l2( j ) = b
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                    '  [', s%tt, '] setting lower bound for z(', j, ') to', b
         END IF

      END SELECT

!     Preserve the order of the bounds at precision level.

      IF ( prob%Z_l( j ) > prob%Z_u( j ) .AND. &
           PRESOLVE_is_zero( prob%Z_l( j ) - prob%Z_u( j )  ) ) THEN
         zlj = HALF * ( prob%Z_l( j ) + prob%Z_u( j )  )
         prob%Z_l( j ) = zlj
         prob%Z_u( j ) = zlj
      END IF

!     Check for dual infeasibility.

      IF ( type == TIGHTEN .AND. prob%Z_l( j ) > prob%Z_u( j ) ) THEN
         inform%status = DUAL_INFEASIBLE
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE analysis stopped: the problem is dual infeasible'
         WRITE( inform%message( 2 ), * )                                       &
            '    because the bounds on z(', j, ') are incompatible'
         RETURN
      END IF

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) THEN
         inform%status = MAX_NBR_TRANSF
         RETURN
      END IF

!     See if variable j can be fixed to one of its bounds.

      IF ( PRESOLVE_is_neg( b ) .AND. lowup == UPPER ) THEN
         xbnd = prob%X_u( j )
         IF ( xbnd < s%P_INFINITY ) THEN
            CALL PRESOLVE_fix_x( j, xbnd, Z_FROM_DUAL_FEAS  )
         ELSE
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because x(', j, ') has an infinite active upper bound'
            RETURN
         END IF
      END IF
      IF ( PRESOLVE_is_pos( b ) .AND. lowup == LOWER ) THEN
         xbnd = prob%X_l( j )
         IF ( xbnd > s%M_INFINITY ) THEN
            CALL PRESOLVE_fix_x( j, xbnd, Z_FROM_DUAL_FEAS  )
         ELSE
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because x(', j, ') has an infinite active upper bound'
            RETURN
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_bound_z

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_bound_c( i, lowup, type, bound )

!     Set a new bound for constraint i.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!              the index of the considered constraint

      INTEGER, INTENT( IN ) :: lowup

!              LOWER or UPPER, for lower or upper bound

      INTEGER, INTENT( IN ) :: type

!              SET or UPDATE, depending whether or not the specified bound
!              should be remembered in the loose bounds

      REAL ( KIND = wp ), INTENT( IN ) :: bound

!              the new lower or upper bound on c_i
!
!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: l
      LOGICAL            :: equality
      REAL ( KIND = wp ) :: b, cli, cui

!     Exit if the bound exceeds the maximum value acceptable in the
!     reduced problem

      IF ( PRESOLVE_is_too_big( bound ) ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    bounding c(', i, ') prevented by unacceptable growth'
         RETURN
      END IF

!     Remember the current bounds.

      cli = prob%C_l( i )
      cui = prob%C_u( i )
      equality = cui == cli

      SELECT CASE ( lowup )

      CASE ( UPPER )

         b = MIN(  s%INFINITY, bound )
         IF ( PRESOLVE_is_zero( b - cui ) ) RETURN

         IF ( b <= s%M_INFINITY ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because c(', i, ') has an infinite active upper bound'
            RETURN
         END IF
         IF ( PRESOLVE_is_zero( b - cli ) ) b = cli

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = C_UPPER_UPDATED
         s%hist_i( l )    = i
         s%hist_j( l )    = 0
         s%hist_r( l )    = cui
         prob%C_u( i )    = b
         IF ( type == UPDATE ) THEN
            IF ( control%final_c_bounds == NON_DEGENERATE ) s%c_u2( i ) = cui
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
               '  [', s%tt, '] updating upper bound for c(', i, ') to', b
         ELSE
            IF ( control%final_c_bounds /= TIGHTEST ) s%c_u2( i ) = b
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
               '  [', s%tt, '] setting upper bound for c(', i, ') to', b
         END IF

      CASE ( LOWER )

         b = MAX( -s%INFINITY, bound )
         IF ( PRESOLVE_is_zero( b - cli ) ) RETURN

         IF ( b >= s%P_INFINITY ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because c(', i, ') has an infinite active lower bound'
            RETURN
         END IF

         IF ( PRESOLVE_is_zero( b - cui ) ) b = cui

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = C_LOWER_UPDATED
         s%hist_i( l )    = i
         s%hist_j( l )    = 0
         s%hist_r( l )    = cli
         prob%C_l( i )    = b
         IF ( type == UPDATE ) THEN
            IF ( control%final_c_bounds == NON_DEGENERATE ) s%c_l2( i ) = cli
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
               '  [', s%tt, '] updating lower bound for c(', i, ') to', b
         ELSE
            IF ( control%final_c_bounds /= TIGHTEST ) s%c_l2( i ) = b
            IF ( s%level >= ACTION ) WRITE( s%out, * )             &
               '  [', s%tt, '] setting lower bound for c(', i, ') to', b
         END IF

      END SELECT

!     Check for an equality constraint.

      IF ( equality ) THEN
         IF ( .NOT. PRESOLVE_is_zero( prob%C_u( i ) -  prob%C_l( i ) ) )       &
            s%m_eq_active = s%m_eq_active - 1
      ELSE
         IF ( PRESOLVE_is_zero( prob%C_u( i ) - prob%C_l( i ) ) ) THEN
            s%m_eq_active = s%m_eq_active + 1
            b = HALF * ( prob%C_l( i ) + prob%C_u( i ) )
            prob%C_l( i ) = b
            prob%C_u( i ) = b
         END IF
      END IF

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_bound_c

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_remove_c( i, y_type, yval, pos )

!     Deactivates constraint i.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the constraint to ignore

      INTEGER, INTENT ( IN ) :: y_type

!            the method to use in order to recover the i-th multplier at
!            RESTORE.  Possible values are:
!
!            Y_GIVEN           : the value of y(i) is known and equal to yval;
!            Y_FROM_Z_LOW      : y(i) must be computed from z(j) if x(j) is
!                                at its lower bound;
!            Y_FROM_Z_UP       : y(i) must be computed from z(j) if x(j) is
!                                at its upper bound;
!            Y_FROM_Z_BOTH     : y(i) must be computed from z(j) if x(j) is
!                                at one of its bounds;
!            Y_FROM_GY         : y(i) must be computed from the column
!                                doubleton procedure (since the latter is
!                                more efficient);
!            Y_FROM_ATY        : y(i) must be computed from A^T y = 0;
!            Y_FROM_FORCING_LOW: y(i) is deduced from the list of variables
!                                for the forcing lower constraint;
!            Y_FROM_FORCING_UP : y(i) is deduced from the list of variables
!                                for the forcing lower constraint.

      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ) :: yval

!            the value of the i-th multplier, when y_type = Y_GIVEN

      INTEGER, OPTIONAL, INTENT( INOUT ) :: pos

!            the position in A of A(i,j) if y_type /= Y_GIVEN (input); or
!            the index of the current transformation if
!            y_type = Y_FROM_FORCING_LOW or Y_FROM_FORCING_UP (output);
!            or the index of the column from which the constraint is reduced
!            if y_type = Y_FROM_ATY.

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER :: l

!     Record the transformation.

      IF ( s%tm >= s%max_tm ) THEN
         CALL PRESOLVE_save_transf
         IF ( inform%status /= OK ) RETURN
      END IF
      l             = s%tm + 1
      s%tt          = s%tt + 1
      s%tm          = l
      s%hist_i( l ) = i

      SELECT CASE ( y_type )

      CASE ( Y_GIVEN )

         s%hist_type( l ) = C_REMOVED_YV
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y(', i, ') =', yval, ']'
         s%hist_j( l ) = 0
         s%hist_r( l ) = yval
         prob%Y( i )   = yval

      CASE ( Y_FROM_Z_LOW )

         s%hist_type( l ) = C_REMOVED_YZ_LOW
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y from z low]'
         s%hist_j( l )  = pos
         s%hist_r( l )  = prob%A%val( pos )

      CASE ( Y_FROM_Z_UP )

         s%hist_type( l ) = C_REMOVED_YZ_UP
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y from z up]'
         s%hist_j( l ) = pos
         s%hist_r( l ) = prob%A%val( pos )

      CASE ( Y_FROM_Z_BOTH )

         s%hist_type( l ) = C_REMOVED_YZ_EQU
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y from z equ]'
         s%hist_j( l ) = pos
         s%hist_r( l ) = prob%A%val( pos )

      CASE ( Y_FROM_GY )

         s%hist_type( l ) = C_REMOVED_GY
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y from doubleton]'
         s%hist_j( l ) = 0
         s%hist_r( l ) = ZERO

      CASE ( Y_FROM_FORCING_LOW )

         s%hist_type( l ) = C_REMOVED_FL
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y from forcing lower]'
         s%hist_j( l ) = END_OF_LIST
         s%hist_r( l ) = ZERO
         pos           = l
         s%needs( Y_VALS, Z_VALS ) = MIN( s%tt, s%needs( Y_VALS, Z_VALS ) )

      CASE ( Y_FROM_FORCING_UP )

         s%hist_type( l ) = C_REMOVED_FU
         IF ( s%level >= ACTION )                                              &
            WRITE( s%out, * ) '  [', s%tt, '] deactivating c(', i, ') =',      &
            prob%C( i ), '[ y from forcing upper]'
         s%hist_j( l ) = END_OF_LIST
         s%hist_r( l ) = ZERO
         pos = l
         s%needs( Y_VALS, Z_VALS ) = MIN( s%tt, s%needs( Y_VALS, Z_VALS ) )

      END SELECT

      CALL PRESOLVE_do_ignore_c( i )

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_remove_c

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_do_ignore_c( i  )

!     Effectively deactivates constraint i and update the relevant problem
!     quantities

!     Argument:

      INTEGER, INTENT( IN ) :: i

!         the (original) index of the row to deactivate

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER :: j, k, ic

!     Deactivate the constraint.

      prob%C_status( i ) = ELIMINATED
      s%m_active     = s%m_active - 1
      IF ( prob%C_l( i ) == prob%C_u( i ) )                                    &
         s%m_eq_active = s%m_eq_active - 1
      s%a_ne_active  = s%a_ne_active - s%A_row_s( i )
      s%A_row_s( i )   = 0

!     Update the column sizes.

      ic = i
      DO
         DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
            IF ( prob%A%val( k ) == ZERO ) CYCLE
            j = prob%A%col( k )
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
            CALL PRESOLVE_decr_A_col_size( j )
            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     lsc_f = ', s%lsc_f, 'ldc_f =', s%ldc_f, '-->  h_perm =',  &
               s%h_perm( :prob%n )
         END DO
         ic = s%conc( ic )
         IF ( ic == END_OF_LIST ) EXIT
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_do_ignore_c

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_rm_A_entry( i, j, k )

!     Removes the (i,j)-th entry of A, which is found in position k.
!     The column index of the entry is remembered as well as its position and
!     value, and the row and column sizes updated.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the row index of the entry of A which must be removed

      INTEGER, INTENT( IN ) :: j

!            the column index of the entry of A which must be removed

      INTEGER, INTENT( IN ) :: k

!            the position in A of the entry of A which must be removed

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: a

!     Remove the (i,j)-th entry.

      a = prob%A%val( k )
      IF ( s%tm >= s%max_tm ) THEN
         CALL PRESOLVE_save_transf
         IF ( inform%status /= OK ) RETURN
      END IF
      l                 = s%tm + 1
      s%tt              = s%tt + 1
      s%tm              = l
      s%hist_type( l )  = A_ENTRY_REMOVED
      s%hist_i( l )     = k
      s%hist_j( l )     = 0
      prob%A%val( k ) = ZERO
      s%hist_r( l )     = a

      IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
         '  [', s%tt, '] removing A(', i, ',', j, ') = ', a

      s%a_ne_active = s%a_ne_active - 1
      s%a_perm( k )   = IBSET( s%a_perm( k ), ELIMINATED )

!     Update the row and column sizes.

      CALL PRESOLVE_decr_A_row_size( i )
      CALL PRESOLVE_decr_A_col_size( j )

!     Check for maximum number of transformations.

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_rm_A_entry

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_implied_bounds( k, val, xl, xu,                      &
                                          imp_low, imp_up, il, iu )

!     Updates the values of the lower and upper implied bounds on a constraint
!     given the value of a matrix element occurring in that constraint and the
!     lower and upper bounds for the associated variable (or multiplier).

!     Arguments:

      INTEGER, INTENT( IN ) :: k

!              the index of the matrix entry being considered

      REAL ( KIND = wp ), INTENT( IN ) :: val

!              the value of the current matrix entry

      REAL ( KIND = wp ), INTENT( IN ) :: xu, xl

!              the lower (xl) and upper (xu) bounds on the variable
!              corresponding to the current matrix entry

      REAL ( KIND = wp ), INTENT( INOUT ) :: imp_low, imp_up

!              the lower (imp_low) and upper (imp_up) implied bounds on the
!              current constraint.  It is updated by the subroutine.

      INTEGER, INTENT( INOUT ) :: il, iu

!              the indices indicating whether infinite bounds have been met so
!              far.  Their meaning is as follows:
!              il = 0 : no infinite lower bound was met
!              il > 0 : a single infinite lower bound was met for variable
!                       corresponding the entry at position k in the matrix.
!              il = -1: more than an infinite lower bound was met
!              The meaning of iu is identical, but relates to the upper bound.
!              These values are updated by the subroutine.
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     The value of the coefficient is negative

      IF ( val < ZERO ) THEN

!        Infinite upper bound: update the infinite bound indicator.

         IF ( xu >= s%P_INFINITY ) THEN
            IF ( il == 0 ) THEN
                il = k
            ELSE IF ( il > 0 ) THEN
                il = -1
            END IF

!        Finite upper bound: update the implied lower bound.

         ELSE
            imp_low = imp_low + val * xu
         END IF

!        Infinite lower bound: update the infinite bound indicator.

         IF ( xl <= s%M_INFINITY ) THEN
            IF ( iu == 0 ) THEN
               iu = k
            ELSE IF ( iu > 0 ) THEN
               iu = -1
            END IF

!        Finite lower bound: update the implied upper bound.

         ELSE
            imp_up  = imp_up  + val * xl
         END IF

!     The value of the coefficient is positive.

      ELSE IF ( val > ZERO ) THEN

!        Infinite lower bound: update the infinite bound indicator.

         IF ( xl <= s%M_INFINITY ) THEN
            IF ( il == 0 ) THEN
               il = k
         ELSE IF ( il > 0 ) THEN
               il = -1
         END IF

!        Finite lower bound: update the implied lower bound.

      ELSE
         imp_low = imp_low + val * xl
      END IF

!        Infinite upper bound: update the infinite bound indicator.

         IF ( xu >= s%P_INFINITY ) THEN
            IF ( iu == 0 ) THEN
               iu = k
            ELSE IF ( iu > 0 ) THEN
               iu = -1
            END IF

!        Finite upper bound: update the implied upper bound.

         ELSE
            imp_up = imp_up + val * xu
         END IF
      END IF
      RETURN

      END SUBROUTINE PRESOLVE_implied_bounds

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_linear_singleton( j, split_ok )

!     Eliminate (if possible) the variable corresponding to a linear singleton
!     column.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!              the index of the linear singleton column

      LOGICAL, INTENT( IN ) :: split_ok

!              .TRUE. if splitting inequalities is allowed at the current
!              stage of the preprocessing

!     Programming: Ph. Toint, November 2000.

!===============================================================================

      INTEGER           :: kij, i, ii, l, k, jj, ic, itmp
      LOGICAL           :: split_equality
      REAL( KIND = wp ) :: aij, cli, cui, yi, nf, a_val
      REAL( KIND = wp ), DIMENSION ( 6 ) :: txdata

!     Get the value (aij) and the position (kij) of the single
!     nonzero element in the j-th column of A, as well as its row
!     index (i).

      kij   = s%A_col_f( j )
      DO ii = 1, prob%m
         i  = s%A_row( kij )
         IF ( prob%C_status( i ) > ELIMINATED ) THEN
            aij = prob%A%val( kij )
            IF ( aij /= ZERO ) EXIT
         END IF
         kij = s%A_col_n( kij )
         IF ( kij == END_OF_LIST ) EXIT
      END DO

!     Error return if aij is not found

      IF ( kij == END_OF_LIST ) THEN
         inform%status = NO_SINGLETON_ENTRY
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE INTERNAL ERROR: singleton entry in column',           &
              j, 'not found'
         IF ( s%level >= DEBUG )                                               &
            CALL PRESOLVE_write_full_prob( prob, control, s )
         RETURN
      END IF

      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    the single nonzero of column', j, 'is in row', i,  'and is', aij

!     If aij is too small, return

      IF ( ABS( aij ) < s%a_tol ) RETURN

      cli = prob%C_l( i )
      cui = prob%C_u( i )

!     -----------------------------------------------------------------
!     First see if x_j cannot be made implied free by exploiting
!     an equality doubleton, or by transfering its bounds to
!     a split equality.
!     -----------------------------------------------------------------

      split_equality = .FALSE.
      IF ( prob%X_status( j ) /= FREE .AND. cli == cui ) THEN
         IF ( s%A_row_s( i ) == 2                    .AND. &
              s%tt + 9 <= control%max_nbr_transforms       ) THEN
            CALL PRESOLVE_shift_x_bounds( i, j, kij )
            IF ( inform%status /= OK ) RETURN
         ELSE IF (  split_ok                                .AND. &
                   .NOT. BTEST( s%a_perm( i ), ROW_SPARSIFICATION ) ) THEN
            prob%X_status( j ) = FREE
            split_equality = .TRUE.
         END IF
      END IF

!     -----------------------------------------------------------------
!     The multiplier may now be computed from dual feasibility.
!     The sign of the multiplier yi then indicates which of
!     the <= or >= is active (if any).  Moreover, since column j is
!     a singleton column and variable j is purely linear, it can be
!     determined uniquely as a function of the other variables
!     occuring in constraint i. Thus it can be eliminated from the
!     problem, which  also causes constraint i to be eliminated.
!     (Unfortunately, this may create a lot of fill-in in H when x( j )
!     occurs nonlinearly in the objective function, which is why this
!     substitution is only performed for purely linear variables.)
!     -----------------------------------------------------------------

!     Compute the multiplier.

      yi = prob%G( j ) / aij

!     Avoid any transformation if it exceeds the maximum acceptable value
!     for the reduced problem data.

      IF ( PRESOLVE_is_too_big( yi ) ) THEN
         IF (  s%level >= DEBUG ) WRITE( s%out, * )                            &
            '    elimination of linear singleton column prevented by',         &
            ' unacceptable growth'
         RETURN
      END IF

!     -----------------------------------------------------------------
!     Consider the different cases for a linear singleton column
!
!     First consider the active columns with (possibly implied)
!     infinite bounds for the associated variable
!     (Also verify there is enough history space to store the
!     transformation)
!     -----------------------------------------------------------------

      IF ( prob%X_status( j ) == FREE           .AND. &
           s%tt + 2 <= control%max_nbr_transforms     ) THEN

         IF ( yi /= ZERO ) THEN

!           Anticipate the new value of the objective

            IF ( PRESOLVE_is_zero( cli - cui ) ) THEN
               nf = prob%f + cli * yi
            ELSE IF ( yi > ZERO ) THEN
               nf = prob%f + cli * yi
            ELSE
               nf = prob%f + cui * yi
            END IF
            IF ( PRESOLVE_is_too_big( nf ) ) THEN
               IF (  s%level >= DEBUG ) WRITE( s%out, * )                      &
                  '    elimination of linear singleton column prevented by',   &
                  ' unacceptable growth'
               RETURN
            END IF

!           Anticipate the new values for the gradient components corresponding
!           to the nonzeros of row i. Note that the j-th component may be
!           neglected as the j-th variable is going to be eliminated by
!           the transformation.

            ic = i
            DO
               DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  a_val = prob%A%val( k )
                  IF ( a_val /= ZERO ) THEN
                     jj = prob%A%col( k )
                     IF ( prob%X_status( jj ) <= ELIMINATED .OR. j == jj ) CYCLE
                     IF ( PRESOLVE_is_too_big( prob%G( jj ) - a_val * yi ) ) THEN
                        IF (  s%level >= DEBUG ) WRITE( s%out, * )             &
                            '    elimination of linear singleton column',      &
                            ' prevented by unacceptable growth'
                        RETURN
                     END IF
                  END IF
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO

         END IF

!        Determine what to do with the i-th constraint.

         IF ( split_equality ) THEN

!           If the i-th constraint cannot be removed, transform it to
!           reflect the neglected bound(s) on x(j) by splitting the equality.

            CALL PRESOLVE_transfer_x_bounds( i, j, kij, 0, txdata, yi )
            IF ( PRESOLVE_is_too_big( MAXVAL( ABS( txdata ) ) ) ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    elimination of linear singleton column prevented by',   &
                  ' unacceptable growth'
               RETURN
            END IF
            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * ) '    column', j,                              &
                    'is a (possibly implied) free linear singleton column'
               WRITE( s%out, * ) '    g(',j,') =', prob%G( j ),                &
                    'A(',i,',',j,') =', aij
            END IF

            CALL PRESOLVE_transfer_x_bounds( i, j, kij, 1, txdata, yi )

         ELSE

!           The constraint may be removed.

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * ) '    column', j,                              &
                    'is a (possibly implied) free linear singleton column'
               WRITE( s%out, * ) '    g(',j,') =', prob%G( j ),                &
                    'A(',i,',',j,') =', aij
            END IF

!           Deactivate constraint i (with multiplier yi)

            CALL PRESOLVE_remove_c( i, Y_GIVEN, yval = yi )

         END IF

!        Ensure that x_j is obtained by elimination.

         IF ( s%level >= DETAILS ) THEN
            WRITE( s%out, * ) '   x(', j, ') can be obtained from the other',  &
                 ' variables of c(', i, ')'
            IF ( s%level >= DEBUG )                                            &
               WRITE( s%out, * ) '    in [', prob%X_l(j),',', prob%X_u(j),']'
         END IF

!        Record the transformation.

!        Note that one remembers the index of the variable to substitute.
!        The index of the row (i) can be reconstructed from the position
!        of A(i,j) (ie kij), which is also needed to reconstruct the value
!        of x(j).
!        Also note that one uses cli and cui, that is the bounds on the
!        constraints before their possible modification in the transfer
!        of the bounds on x(j) to c(i).

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = X_SUBSTITUTED
         s%hist_i( l )    = j
         s%hist_j( l )    = kij
         IF ( PRESOLVE_is_zero( cli - cui ) ) THEN
            s%hist_r( l ) = cli
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  [', s%tt, '] computing x(', j, ') out from c(', i, ') =',    &
               cli, '(equality)'
         ELSE IF ( yi > ZERO ) THEN
            s%hist_r( l ) = cli
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  [', s%tt, '] computing x(', j, ') out from c(', i, ') =',    &
               cli, '(lower bound active)'
         ELSE IF ( yi < ZERO ) THEN
            s%hist_r( l ) = cui
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  [', s%tt, '] computing x(', j, ') out from c(', i, ') =',    &
               cui, '(upper bound active)'
         ELSE
            s%hist_r( l ) = MIN( MAX( ZERO, cli ), cui )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  [', s%tt, '] computing x(', j, ') out from c(', i,           &
               ') =', s%hist_r( l ), '(free feasible value)'
         END IF
         CALL PRESOLVE_do_ignore_x( j )

!        Remember the dependencies

         s%needs( X_VALS, A_VALS ) = MIN( s%tt, s%needs( X_VALS, A_VALS ) )
         s%needs( X_VALS, C_BNDS ) = MIN( s%tt, s%needs( X_VALS, C_BNDS ) )

!        If the case where constraint i is active, the objective function
!        and gradient are influenced by its value and must be updated to
!        reflect its elimination.

         IF ( yi /= ZERO ) THEN

!           Update the objective value.

            IF ( PRESOLVE_is_zero( cli - cui ) ) THEN
               prob%f = nf
            ELSE IF ( yi > ZERO ) THEN
               prob%f = nf
            ELSE
               prob%f = nf
            END IF

!           Remember the dependencies.

            s%needs( F_VAL, C_BNDS ) = MIN( s%tt, s%needs( F_VAL, C_BNDS ) )
            s%needs( F_VAL, G_VALS ) = MIN( s%tt, s%needs( F_VAL, G_VALS ) )
            s%needs( F_VAL, A_VALS ) = MIN( s%tt, s%needs( F_VAL, A_VALS ) )

!           Update the gradient.

            itmp = MIN( s%tt, s%needs( G_VALS, A_VALS ))
            ic   = i
            DO
               DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  a_val = prob%A%val( k )
                  IF ( a_val /= ZERO ) THEN
                     jj = prob%A%col( k )
                     IF ( prob%X_status( jj ) <= ELIMINATED ) CYCLE
                     prob%G( jj ) = prob%G( jj ) - a_val * yi
                     s%needs( G_VALS, A_VALS ) = itmp
                  END IF
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO

            IF ( s%level >= DETAILS ) WRITE( s%out, * )                        &
               '    gradient and objective function updated'

!           Remember the dependencies.

            IF ( s%needs( G_VALS, A_VALS ) == s%tt ) THEN
               s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ))
               s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ))
            END IF

         END IF

!        If a split equality was used, row i has been marked as a modified
!        equality row by do_ignore_x( j ), which is inadequate in this case
!        since it is no longer an equality. Thus make sure it is not selected
!        in the list of potential sparsification pivots.

         IF ( split_equality )                                                 &
            s%a_perm(i) = IBCLR( s%a_perm(i), ROW_SPARSIFICATION )

!     -----------------------------------------------------------------
!     Next consider the lower bounded active columns.
!     -----------------------------------------------------------------

      ELSE IF ( prob%X_l( j ) >  s%M_INFINITY .AND. &
                prob%X_u( j ) >= s%P_INFINITY       ) THEN

!        Bound the variable.

         IF ( aij > ZERO ) THEN
            CALL PRESOLVE_bound_y( i, UPPER, TIGHTEN, yi )
         ELSE
            CALL PRESOLVE_bound_y( i, LOWER, TIGHTEN, yi )
         END IF
         IF ( inform%status /= OK ) RETURN

!     -----------------------------------------------------------------
!     Finally consider the upper bounded active columns
!     -----------------------------------------------------------------

      ELSE IF ( prob%X_l( j ) <= s%M_INFINITY .AND. &
                prob%X_u( j ) <  s%P_INFINITY       ) THEN

!        Bound the variable.

         IF ( aij > ZERO ) THEN
            CALL PRESOLVE_bound_y( i, LOWER, TIGHTEN, yi )
         ELSE
            CALL PRESOLVE_bound_y( i, UPPER, TIGHTEN, yi )
         END IF
         IF ( inform%status /= OK ) RETURN

!     -----------------------------------------------------------------
!     No action has been taken, either because variable j cannot be
!     freed or because the action would require transformations beyond
!     their maximum number.
!     -----------------------------------------------------------------

      ELSE
         IF ( s%level >= DEBUG ) THEN
            IF ( prob%X_status( j ) /= FREE ) THEN
               WRITE( s%out, * ) '    variable', j, 'cannot be freed'
            ELSE
               WRITE( s%out, * ) '    not enough history space left'
            END IF
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_linear_singleton

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_linear_doubleton( j, split_ok )

!     Eliminate (if possible) the variable corresponding to a linear doubleton
!     column by substituting its value from one equality constraint into the
!     other constraint.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!              the index of the doubleton column

      LOGICAL, INTENT( IN ) :: split_ok

!              .TRUE. iff splitting equality constraints is allowed at the
!              the current stage of preprocessing

!     Programming: Ph. Toint, November 2000.

!===============================================================================

      INTEGER           :: ko, ke, k, i, io, ie, ii, icase, ic, last, jo, ns,  &
                           nfills, je, l, nbr_canceled, pjo, xsj, itmp
      LOGICAL           :: split_equality
      REAL( KIND = wp ) :: cie, pivot, gj, ao, r, rg, yle, yue, a, maxgie, &
                           maxaie, nclio, ncuio, nylio, nyuio
      REAL( KIND = wp ), DIMENSION( 6 ) :: txdata

!-------------------------------------------------------------------------------

!     Get the positions and row indices of the two nonzero elements in
!     the j-th column of A.  The positions are stored in ke and ko,
!     respectively, and the rows in ie and io, respectively.

!-------------------------------------------------------------------------------

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   examining column', j, 'for useful doubleton'

      ke = 0
      ko = 0

      k  = s%A_col_f( j )
      IF ( END_OF_LIST /= k ) THEN
         DO ii = 1, prob%m
            i  = s%A_row( k )
            IF ( prob%C_status( i ) > ELIMINATED ) THEN
               IF ( prob%A%val( k ) /= ZERO ) THEN
                  IF ( PRESOLVE_is_zero( prob%C_l(i) - prob%C_u(i) )           &
                       .AND. ke == 0 ) THEN
                     ke  = k
                     ie  = i
                  ELSE IF ( ko == 0 ) THEN
                     ko  = k
                     io  = i
                  END IF
                  IF ( ke > 0 .AND. ko > 0 ) EXIT
               END IF
            END IF
            k = s%A_col_n( k )
            IF ( k == END_OF_LIST ) EXIT
         END DO
      END IF

!     If no element was found, error.

      IF ( ko == 0 ) THEN
         inform%status = NO_DOUBLETON_ENTRIES
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE INTERNAL ERROR: doubleton entries in column', j,      &
              'not found (ANALYZE)'
         IF ( s%level >= DEBUG )                                               &
            CALL PRESOLVE_write_full_prob( prob, control, s )
         RETURN
      END IF

!-------------------------------------------------------------------------------

!     The doubleton is now identified

!-------------------------------------------------------------------------------

!     Assume now that the doubleton column contains at least an equality row.

      IF ( ke > 0 ) THEN

!        Loop over the two possibilities for pivot, in order to take the
!        fact that constraint io might also be an equality, in which case
!        it could also be used for pivoting if pivoting on constraint ie
!        fails.

         DO icase = 1, 2

            cie   = prob%C_l( ie )
            pivot = prob%A%val( ke )

!           If the pivot is too small, see if a switch between rows
!           ie and io is possible.

            IF ( ABS( pivot ) <= s%a_tol ) THEN

               IF ( icase == 1 ) THEN

!                 if not, examine the next linear doubleton.

                  IF ( .NOT. PRESOLVE_is_zero( prob%C_l(io) - prob%C_u(io) ) ) &
                      EXIT

!                 The switch is possible because constraint io is also
!                 an equality:
!                 1) exchange the row indices...

                  ii = ie
                  ie = io
                  io = ii

!                 2) ... and the positions of their nonzero in column j.

                  ii = ke
                  ke = ko
                  ko = ii

!                 3) loop

                  CYCLE
               ELSE
                  EXIT
               END IF
            END IF

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    doubleton', j, ': rows', ie, '(equality) and', io

!           See if the column cannot be made implied free by exploiting
!           an equality doubleton or a split equality.

            split_equality = .FALSE.
            IF ( prob%X_status( j ) /= FREE ) THEN
               IF ( s%A_row_s( ie ) == 2                   .AND. &
                    s%tt + 9 <= control%max_nbr_transforms       ) THEN
                  CALL PRESOLVE_shift_x_bounds( ie, j, ke )
                  IF ( inform%status /= OK ) RETURN
               ELSE IF ( split_ok                                    .AND. &
                         .NOT. BTEST( s%a_perm( ie ), ROW_SPARSIFICATION ) )THEN
                  CALL PRESOLVE_transfer_x_bounds( ie, j, ke, 0, txdata )
                  IF ( PRESOLVE_is_too_big( MAXVAL( ABS( txdata ) ) ) ) THEN
                      IF (  s%level >= DEBUG ) WRITE( s%out, * )               &
                         '    elimination of linear doubleton column',         &
                         ' prevented by unacceptable growth'
                      RETURN
                  END IF
                  xsj                = prob%X_status( j )
                  prob%X_status( j ) = FREE
                  split_equality     = .TRUE.
               END IF
            END IF

!-------------------------------------------------------------------------------

!           Assume now that x_j is unbounded: it can therefore be
!           eliminated from row ie  (if there remains enough history space
!           to remember the transformation).

!-------------------------------------------------------------------------------

            IF ( prob%X_status( j ) == FREE                     .AND. &
                 s%tt + 3 + s%A_row_s( ie ) <= control%max_nbr_transforms ) THEN

!              Compute the number of fills created by
!              the merging of rows ie and io, and verify it is not too
!              large, compared with the original size of row io.

!              Loop over the non-pivot row to remember the positions
!              of its nonzeros in w_n, as well as the index of the
!              last of its concatenated rows (in last).

               s%w_n  = 0
               ic     = io
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     jo = prob%A%col( k )
                     IF ( prob%X_status( jo ) <= ELIMINATED .OR. &
                          prob%A%val( k )     == ZERO            )CYCLE
                     s%w_n( jo ) = k
                  END DO
                  last = ic
                  ic   = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO

!              Now loop on the pivot row, accumulating the number of
!              fills in nfills.

               nfills = 0
               ic     = ie
               maxaie = ZERO
               maxgie = ZERO
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     je = prob%A%col( k )
                     a  = prob%A%val( k )
                     IF ( prob%X_status( je ) <= ELIMINATED .OR.&
                          a                   == ZERO       .OR.&
                          s%w_n( je )            > 0              ) CYCLE

                     maxaie = MAX( maxaie, ABS( a ) )
                     maxgie = MAX( maxgie, ABS( prob%G( je ) ) )

!                    If variable jo does not appear in the non-pivot
!                    row, then a fill is created in this row.

                     nfills = nfills + 1
                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO

!              Check if the number of fills does not exceed the
!              maximum proportion of the original size of row io.

               IF ( split_equality                 .OR. &
                    s%max_fill_prop < s%P_INFINITY      ) THEN

                  ns = prob%A%ptr( io + 1 ) - prob%A%ptr( io )
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                           '    pivoting leads to', nfills,                    &
                           'fills in a row of size', s%A_row_s( io ),          &
                           '( originally', ns, ')'

                  IF ( split_equality ) THEN

!                    If a split equality has been used, no fill is
!                    allowed to occur, as this would prevent keeping
!                    the shortened row io in the problem.
!                    remember to restore the status of x(j) before exiting

                     IF ( nfills > 0 ) THEN
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    too many fills in sparse A with split',        &
                           ' equality: skipping pivoting'
                        prob%X_status( j ) = xsj
                        EXIT
                     END IF

                  ELSE

                     IF ( s%A_row_s( io ) + nfills > s%max_fill_prop * ns ) THEN
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    too many fills in sparse A: skipping pivoting'
                        EXIT
                     END IF

                  END IF

               END IF

!              Pivoting is possible: compute the ratios for updating the
!              gradient and matrix elements.

               gj = prob%G( j )
               ao = prob%A%val( ko )
               r  = ao / pivot
               rg = gj / pivot

!              Avoid the entire operation if there is a risk that the reduced
!              problem data exceeds its maximum acceptable value.

               IF ( PRESOLVE_is_too_big( ABS(  r ) * maxaie ) .OR. &
                    PRESOLVE_is_too_big( ABS( rg ) * maxgie )      ) THEN
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    elimination of linear doubleton column',             &
                     ' prevented by unacceptable growth'
                  EXIT
               END IF

!              Anticipate that the multiplier y(io) will be given
!              by [ g(j) - pivot * y(ie) ] / a(io,j), exiting if this
!              is too big.

               yle = prob%Y_l( ie )
               yue = prob%Y_u( ie )
               IF ( ABS( ao ) >= s%a_tol ) THEN
                  IF ( r > 0 ) THEN
                     IF ( yue < s%P_INFINITY ) THEN
                        nylio = ( gj - pivot * yue ) / ao
                        IF ( PRESOLVE_is_too_big( nylio ) ) THEN
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    elimination of linear doubleton column',    &
                              ' prevented by unacceptable growth'
                           EXIT
                        END IF
                     END IF
                     IF ( yle > s%M_INFINITY ) THEN
                        nyuio = ( gj - pivot * yle ) / ao
                        IF ( PRESOLVE_is_too_big( nyuio ) ) THEN
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    elimination of linear doubleton column',    &
                              ' prevented by unacceptable growth'
                           EXIT
                        END IF
                     END IF
                  ELSE
                     IF ( yle > s%M_INFINITY ) THEN
                        nylio = ( gj - pivot * yle ) / ao
                        IF ( PRESOLVE_is_too_big( nylio ) ) THEN
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    elimination of linear doubleton column',    &
                              ' prevented by unacceptable growth'
                           EXIT
                        END IF
                     END IF
                     IF ( yue < s%P_INFINITY ) THEN
                        nyuio = ( gj - pivot * yue ) / ao
                        IF ( PRESOLVE_is_too_big( nyuio ) ) THEN
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    elimination of linear doubleton column',    &
                              ' prevented by unacceptable growth'
                           EXIT
                        END IF
                     END IF
                  END IF
               END IF

!              Anticipate the new bounds on the io-th constraint.

               a = r * cie
               IF ( prob%C_l( io ) > s%M_INFINITY ) THEN
                  nclio = prob%C_l( io ) - a
                  IF ( PRESOLVE_is_too_big( nclio ) ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    elimination of linear doubleton column',          &
                        ' prevented by unacceptable growth'
                     EXIT
                  END IF
               END IF
               IF ( prob%C_u( io ) <  s%P_INFINITY ) THEN
                  ncuio = prob%C_u( io ) - a
                  IF ( PRESOLVE_is_too_big( ncuio ) ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    elimination of linear doubleton column',          &
                        ' prevented by unacceptable growth'
                     EXIT
                  END IF
               END IF

!              Write that things are ok.

               IF ( s%level >= DETAILS ) THEN
                  WRITE( s%out, * )                                            &
                       '   unbounded linear doubleton found in column',        &
                       j, '( ie =', ie, ', io =', io,')'
                  IF ( s%level >= DEBUG ) THEN
                     WRITE( s%out, * ) '    g(j)  =', gj
                     WRITE( s%out, * ) '    a(ie) =', pivot, ' a(io) =', ao
                     WRITE( s%out, * ) '    r     =', r, ' rg    =', rg
                  END IF
               END IF

!              Remember that the value of x(j) is obtained from c(ie).

               IF ( s%tm >= s%max_tm ) THEN
                  CALL PRESOLVE_save_transf
                  IF ( inform%status /= OK ) RETURN
               END IF
               l    = s%tm + 1
               s%tt = s%tt + 1
               IF ( s%level >= ACTION ) THEN
                  IF ( s%level >= DETAILS ) WRITE( s%out, * ) '    x(', j,     &
                          ') can be substituted from c(', ie,  ') =', cie
                  WRITE( s%out, * ) '  [', s%tt, '] pivoting c(', ie,          &
                       ') and c(', io, ') from A(', ie, ',', j,')'
               END IF

               s%tm             = l
               s%hist_type( l ) = A_ROWS_MERGED
               s%hist_i( l )    = ke
               s%hist_j( l )    = ko
               s%hist_r( l )    = pivot
               CALL PRESOLVE_do_ignore_x( j )

!              Effectively update the multipliers.

               IF ( ABS( ao ) >= s%a_tol ) THEN
                  IF ( r > 0 ) THEN
                     IF ( yue < s%P_INFINITY ) THEN
                        CALL PRESOLVE_bound_y( io, LOWER, TIGHTEN, nylio )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     IF ( yle > s%M_INFINITY ) THEN
                        CALL PRESOLVE_bound_y( io, UPPER, TIGHTEN, nyuio )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                  ELSE
                     IF ( yle > s%M_INFINITY ) THEN
                        CALL PRESOLVE_bound_y( io, LOWER, TIGHTEN, nylio )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                     IF ( yue < s%P_INFINITY ) THEN
                        CALL PRESOLVE_bound_y( io, UPPER, TIGHTEN, nyuio )
                        IF ( inform%status /= OK ) RETURN
                     END IF
                  END IF
               END IF
               s%needs( Y_VALS, G_VALS ) = MIN( s%tt, s%needs( Y_VALS, G_VALS ))
               s%needs( Y_VALS, A_VALS ) = MIN( s%tt, s%needs( Y_VALS, A_VALS ))

!              Update the objective function value.

               prob%f = prob%f + rg * cie
               s%needs( F_VAL, G_VALS ) = MIN( s%tt, s%needs( F_VAL, G_VALS ) )
               s%needs( F_VAL, C_BNDS ) = MIN( s%tt, s%needs( F_VAL, C_BNDS ) )
               s%needs( F_VAL, A_VALS ) = MIN( s%tt, s%needs( F_VAL, A_VALS ) )

!              Update the bounds on the io-th constraint.

               IF ( prob%C_l( io ) > s%M_INFINITY ) THEN
                  prob%C_l( io ) = nclio
                  s%needs( C_BNDS,A_VALS ) = MIN( s%tt, s%needs( C_BNDS,A_VALS))
                  IF ( control%final_c_bounds /= TIGHTEST )                    &
                     s%c_l2( io ) = prob%C_l( io )
               END IF
               IF ( prob%C_u( io ) <  s%P_INFINITY ) THEN
                  prob%C_u( io ) = ncuio
                  s%needs( C_BNDS,A_VALS ) = MIN( s%tt, s%needs( C_BNDS,A_VALS))
                  IF ( control%final_c_bounds /= TIGHTEST )                    &
                     s%c_u2( io ) = prob%C_u( io )
               END IF

               IF ( s%level >= DETAILS ) WRITE( s%out, * ) '    f and c updated'

!              Perform the pivoting (elimination of x_j from its value in
!              constraint ie) on the pivot and non-pivot rows, as well as
!              on the gradient.

               nbr_canceled = 0

!              Loop on the pivot row, performing the pivoting in the
!              non-pivot row.

               itmp   = MIN( s%tt, s%needs( G_VALS, A_VALS ))
               nfills = 0
               ic     = ie
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     je = prob%A%col( k )
                     IF ( prob%X_status( je ) <= ELIMINATED ) CYCLE
                     a = prob%A%val( k )
                     IF ( a == ZERO ) CYCLE

!                    If variable jo also appears in the non-pivot row,
!                    its coefficient in the non-pivot row must be updated
!                    (detecting possible cancellation if it occurs).

                     pjo = s%w_n( je )
                     IF ( pjo > 0 ) THEN
                        ao = prob%A%val( pjo ) - r * a
                        IF ( PRESOLVE_is_zero( ao ) ) THEN
                           nbr_canceled = nbr_canceled + 1
                           s%w_mn( nbr_canceled ) = pjo
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    cancellation in A(', io, ',', je, ')'
                        END IF
                        prob%A%val( pjo ) = ao

!                    If it does not appear in the non-pivot row, then
!                    a fill is created in this row, which is kept in
!                    the pivot row considered as an extension of the
!                    non-pivot row.

                     ELSE
                        nfills = nfills + 1
                        prob%A%val( k ) = - r * a
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    fill in (', io, ',', je, ') =',                &
                           prob%A%val( k ), '( k =', k, ')'
                     END IF

!                    Finally update the gradient.

                     prob%G( je ) = prob%G( je ) - rg * a
                     s%needs( G_VALS, A_VALS ) = itmp

                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO

!              If there remain fills in the old pivot row, append it to
!              the non-pivot row and update its structure.

               IF ( nfills > 0 ) THEN

                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    the pivot row still contains', nfills, 'nonzero(s)'

!                 Concatenate the pivot row to the non-pivot row.

                  s%conc( last ) = ie

!                 Loop over the old pivot row, considering if a
!                 nonzero must be removed because it has moved to
!                 the non pivot row or if it has to stay, in which
!                 case the row and column structure must be updated.

                  ic = ie
                  DO ii = 1, prob%m
                     DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                        je = prob%A%col( k )
                        IF ( prob%X_status( je )  <= ELIMINATED .OR. &
                             prob%A%val( k )      == ZERO            ) CYCLE

!                       No fill created: zero element ( ie, je ).

                        IF ( s%w_n( je ) > 0 ) THEN

!                          element transfered to the non-pivot row:
!                          remove its occurrence from the pivot row

                           CALL PRESOLVE_rm_A_entry( ie, je, k )
                           IF ( inform%status /= OK ) RETURN

!                       New fill element ( ie, je ), now considered
!                       as element ( io, je ) because of concatenation

                        ELSE

!                          Update the column structure of both rows.

                           s%A_row( k ) = io

!                          Update the row and column sizes.

                           s%A_row_s( io ) = s%A_row_s( io ) + 1
                           CALL PRESOLVE_incr_A_col_size( je )
                           IF ( s%level >= CRAZY ) WRITE( s%out, * )           &
                              '     lsc_f = ', s%lsc_f, 'ldc_f =', s%ldc_f,    &
                              '-->  h_perm =',  s%h_perm( :prob%n )
                            s%a_ne_active = s%a_ne_active + 1

                         END IF
                      END DO
                      ic = s%conc( ic )
                      IF ( ic == END_OF_LIST ) EXIT
                   END DO

               END IF

!              If cancellation occurred, effectively zero the
!              negligible elements.

               DO ic = 1, nbr_canceled
                  k = s%w_mn( ic )
                  CALL PRESOLVE_rm_A_entry( s%A_row( k ), prob%A%col( k ), k )
               END DO

               IF ( s%level >= DETAILS ) WRITE( s%out, * ) '    g and A updated'

!              Now consider the future of row ie, depending whether
!              it has been split or not.

!              Split row: keep it around, but transfer the bounds.

               IF ( split_equality ) THEN

                  CALL PRESOLVE_transfer_x_bounds( ie, j, ke, 1, txdata )

!              No split: deactivate row ie.

               ELSE

                  CALL PRESOLVE_remove_c( ie, Y_FROM_GY )

               END IF
               IF ( inform%status /= OK ) RETURN

!              Exit from the loop on pivoting possibilities, as one has
!              been used.

               EXIT

            ELSE

!              The current doubleton cannot be exploited in this
!              configuration.

               IF ( prob%X_status( j ) /= FREE ) THEN
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    variable', j, 'cannot be made free'

!                 See if a switch between rows ie and io is possible.

!                 The switch is possible because constraint io is also
!                 an equality:

                  IF ( icase == 1 ) THEN
                     IF ( PRESOLVE_is_zero( prob%C_l(io)-prob%C_u(io) ) ) THEN

!                       1) exchange the row indices...

                        ii = ie
                        ie = io
                        io = ii

!                       2) ... and  the positions of their nonzero
!                          in column j.

                        ii = ke
                        ke = ko
                        ko = ii

!                    If not, examine the next linear doubleton.

                     ELSE
                         EXIT
                     END IF
                  ELSE
                     EXIT
                  END IF

               ELSE

                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    not enough history space left'

!                 Restore the status of x(j) before branching for the
!                 next doubleton.

                  IF ( split_equality ) prob%X_status( j ) = xsj
                  EXIT
               END IF
            END IF
         END DO

      ELSE
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                            &
            '    no equality constraint found'
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_linear_doubleton

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_unconstrained( j )

!     Consider the case of an unconstrained variable. It is eliminated
!     whenever enough information is available to obtain its value.

!     Argument:

      INTEGER, INTENT( IN ) :: j

!              the index of the unconstrained variable

!     Programming: Ph. Toint, November 2000.

!==============================================================================

      INTEGER            :: hsj, hsj2, k, jk, khpj, p, i, khjj, ii, khpp, l
      LOGICAL            :: helim
      REAL ( KIND = wp ) :: gj, xlj, xuj, hjj, b, flow, fup, hpj, hpp, r,     &
                            newh, newf, newg


      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
            '   checking unconstrained x(', j, ')'

!     Set the scene.

      gj  = prob%G( j )
      xlj = prob%X_l( j )
      xuj = prob%X_u( j )
      IF ( prob%H%ne > 0 ) THEN
         hsj = s%H_str( j )
      ELSE
         hsj = EMPTY
      END IF
      hsj2 = PRESOLVE_H_row_s( j )

!-------------------------------------------------------------------------------
!     If the linearly unconstrained variable j occurs linearly in the
!     objective function, then it can be fixed to one of its bounds.
!-------------------------------------------------------------------------------

      IF ( hsj == EMPTY ) THEN

!        Variable j is a pure linear variable:
!        it may be set to one of its bounds.

         IF ( gj == ZERO ) THEN
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
              '   x(', j, ') is purely linear with zero gradient'

!           The associated gradient component is zero: fix
!           the variable to its current value

            CALL PRESOLVE_ignore_x( j )
            IF ( inform%status /= OK ) RETURN
            CALL PRESOLVE_fix_z( j, ZERO )
            IF ( inform%status /= OK ) RETURN

         ELSE IF ( gj < ZERO ) THEN

!           Negative gradient: fix the variable to its upper bound.

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '   x(', j, ') is purely linear with negative gradient'
            IF ( xuj < s%P_INFINITY ) THEN
               CALL PRESOLVE_fix_x( j, xuj, Z_GIVEN, zval=  gj )
               IF ( inform%status /= OK ) RETURN
            ELSE
               inform%status = DUAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                 ' PRESOLVE analysis stopped: the problem is dual infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                 '    because x(', j, ') has an infinite active upper bound'
               RETURN
            END IF

         ELSE

!           Positive gradient: fix the variable to its lower bound.

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
            '   x(', j, ') is purely linear with positive gradient'

            IF ( xlj > s%M_INFINITY ) THEN
               CALL PRESOLVE_fix_x( j, xlj, Z_GIVEN, zval=  gj )
               IF ( inform%status /= OK ) RETURN
            ELSE
               inform%status = DUAL_INFEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                 ' PRESOLVE analysis stopped: the problem is dual infeasible'
               WRITE( inform%message( 2 ), * )                                 &
                 '    because x(', j, ') has an infinite active lower bound'
               RETURN
            END IF
         END IF

!-------------------------------------------------------------------------------
!     Variable j is a pure quadratic variable: it may be set to
!     one of its bounds or to its minimizer (if between the bounds).
!-------------------------------------------------------------------------------

      ELSE IF ( hsj > 0 ) THEN

!        1) Obtain the j-th diagonal element of H.

         hjj = prob%H%val( hsj )

!        2) If the quadratic is convex, compute the step to its
!           minimizer and fix the variable accordingly.

         IF ( hjj > ZERO ) THEN
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '   x(', j, ') is purely quadratic and convex'
            b = - gj / hjj
            IF ( b >= xlj ) THEN
               IF ( b <= xuj ) THEN
                  CALL PRESOLVE_fix_x( j, b, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  CALL PRESOLVE_fix_x( j, xuj, Z_GIVEN, zval = ZERO )
                  IF ( inform%status /= OK ) RETURN
               END IF
            ELSE
               CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
               IF ( inform%status /= OK ) RETURN
            END IF
!           z is given by g+Hx

!        3) The quadratic is concave: compute the its value
!            at both bounds and pick the lowest.

         ELSE
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '   x(', j, ') is purely quadratic and concave'
            flow  = HALF * hjj * xlj**2 + gj * xlj
            fup   = HALF * hjj * xuj**2 + gj * xuj
            IF ( flow < fup ) THEN
               CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
            ELSE
               CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
            END IF
            IF ( inform%status /= OK ) RETURN
         END IF

!-------------------------------------------------------------------------------
!     The j-th column of the Hessian contains a single nondiagonal element
!     and x(j) is free.  First find the position and value of that nonzero.
!-------------------------------------------------------------------------------

      ELSE IF ( prob%X_status( j ) == FREE .AND. hsj == -1 ) THEN

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    column', j, 'of H contains a single off-diagonal nonzero'

         khpj = 0

         khjj = prob%H%ptr( j + 1 ) - 1

!        Search the subdiagonal.

         DO k  = prob%H%ptr( j ), khjj - 1
            jk = prob%H%col( k )
            IF ( prob%X_status( jk ) <= ELIMINATED ) CYCLE
            b  = prob%H%val( k )
            IF ( b == ZERO ) CYCLE
            p    = jk
            khpj = k
            hpj  = b
            EXIT
         END DO

!        Search the superdiagonal.

         IF ( khpj == 0 ) THEN
            k = s%H_col_f( j )
            IF ( END_OF_LIST /= k ) THEN
               DO ii = j + 1, prob%n
                  i = s%H_row( k )
                  IF ( prob%X_status( i ) > ELIMINATED ) THEN
                     b  = prob%H%val( k )
                     IF ( b /= ZERO ) THEN
                        p    = i
                        khpj = k
                        hpj  = b
                        EXIT
                     END IF
                  END IF
                  k = s%H_col_n( k )
                  IF ( k == END_OF_LIST ) EXIT
               END DO
            END IF
            IF ( khpj == 0 ) THEN
               inform%status = NO_SINGLE_OFFDIAGONAL
               WRITE( inform%message( 1 ), * )                                 &
                   ' PRESOLVE INTERNAL ERROR: single off-diagonal entry',      &
                   ' in column', j, 'not found'
               IF ( s%level >= DEBUG )                                         &
                  CALL PRESOLVE_write_full_prob( prob, control, s )
               RETURN
            END IF
         END IF

!-------------------------------------------------------------------------------
!        The j-th column of the Hessian contains a diagonal element and a
!        single other element in row p.  In this case, variable p can be
!        eliminated from the dual feasibility condition provided H(p,p) is
!        nonzero.  This corresponds to a simple Gaussian elimination on the
!        2 X 2 principal submatrix of H spanned by j and p.
!-------------------------------------------------------------------------------

         IF (  hsj2 == 2 ) THEN

            IF ( s%level >= DEBUG ) THEN
               WRITE( s%out, * )'    considering elimination with x(', p,')'
               WRITE( s%out, * ) '    using H(', p, ',', j, ') =', hpj
            END IF

!           Verify the value of H(p,p) and H(j,j).

            hjj  = prob%H%val( khjj )
            helim = ABS( hjj ) >=  s%h_tol
            IF ( helim ) THEN
               khpp = prob%H%ptr( p + 1 ) - 1
               IF ( khpp >= prob%H%ptr( p ) ) THEN
                  hpp = prob%H%val( khpp )
                  helim = helim .AND. ABS( hpp ) >= s%h_tol
               ELSE
                  helim = .FALSE.
               END IF
            END IF

!           The diagonal elements are ok: perform the elimination.

            IF ( helim ) THEN

               IF ( s%level >= DEBUG ) THEN
                     WRITE( s%out, * ) '          H(', j, ',', j, ') =', hjj
                     WRITE( s%out, * ) '          H(', p, ',', p, ') =', hpp
               END IF

               r = gj / hjj

!              Anticipate the new Hessian, objective and gradient values, and
!              avoid the operation is they exceed the maximum acceptable value
!              for the reduced problem.

               newh = hpp - hpj * hpj / hjj
               newf = prob%f - r * prob%G( p )
               newg = prob%G( p ) - r * hpj
               IF ( PRESOLVE_is_too_big( newh ) .OR. &
                    PRESOLVE_is_too_big( newf ) .OR. &
                    PRESOLVE_is_too_big( newg )      ) THEN
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    eliminating x(', j, ') prevented by unacceptable',   &
                     ' growth'
                  RETURN
               END IF

!              Eliminate x(j).

               IF ( s%tm >= s%max_tm ) THEN
                  CALL PRESOLVE_save_transf
                  IF ( inform%status /= OK ) RETURN
               END IF
               l                = s%tm + 1
               s%tt             = s%tt + 1
               s%tm             = l
               s%hist_type( l ) = H_ELIMINATION
               s%hist_i( l )    = j
               s%hist_j( l )    = khpj
               s%hist_r( l )    = r
               IF ( s%level >= ACTION ) WRITE( s%out, * ) '  [', s%tt,         &
                  '] x(', j, ') can be obtained by elimination using x(', p, ')'
               CALL PRESOLVE_do_ignore_x( j )

!              Effectively update H, f and g.

               prob%H%val( khpp ) = newh
               prob%f             = newf
               prob%G( p )        = newg

!              Remember the dependencies.

               s%needs( X_VALS, H_VALS ) = MIN( s%tt, s%needs( X_VALS, H_VALS ))
               s%needs( X_VALS, G_VALS ) = MIN( s%tt, s%needs( X_VALS, G_VALS ))
               s%needs( G_VALS, H_VALS ) = MIN( s%tt, s%needs( G_VALS, H_VALS ))
               s%needs( F_VAL , G_VALS ) = MIN( s%tt, s%needs( F_VAL , G_VALS ))
               s%needs( F_VAL , H_VALS ) = MIN( s%tt, s%needs( F_VAL , H_VALS ))

            END IF

!-------------------------------------------------------------------------------
!        The j-th column of the Hessian contains a single nonzero element
!        in row p, which is not on the diagonal.  In this case, variable p
!        can be eliminated from the dual feasibility condition.
!-------------------------------------------------------------------------------

         ELSE IF ( hsj2 == 1 ) THEN

            CALL PRESOLVE_fix_x( p, - gj / hpj, Z_FROM_DUAL_FEAS )

         END IF

      END IF

      RETURN

      END SUBROUTINE PRESOLVE_unconstrained

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_merge_x( j, k, alpha )

!     We have that alpha * column k = column j (in A and H).
!     In this case, merge the variables j and k, such that
!     new variable k <--- old variable k + alpha * old variable j.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!            the index of the variable to be merged in variable k

      INTEGER, INTENT( IN ) :: k

!            the index of the variable into which variable j is to be merged

      REAL ( KIND = wp ), INTENT( IN ) :: alpha

!            the coefficient of the merging.

!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: nxl, nxu, nzl, nzu, xlj, xlk, xuj, xuk,            &
                            zlj, zuj, zlk, zuk

!     Get the involved variables' bounds.

      xlj = prob%X_l( j )
      xuj = prob%X_u( j )
      xlk = prob%X_l( k )
      xuk = prob%X_u( k )

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * )                                                     &
              '   trying: x(', k, ') <-- x(', k, ') +', alpha, '* x(', j, ')'
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * )                                                  &
                 '   with g(', k, ') =', prob%G( k ), 'and g(', j, ') =',      &
                 prob%G( j )
            WRITE( s%out, * ) '    x(', k, ') in [', xlk, ',', xuk, '] and '
            WRITE( s%out, * ) '    x(', j, ') in [', xlj, ',', xuj, ']'
         END IF
      END IF

!     See if we must define an aggregate variable, or of one variable
!     is dominated.

!-------------------------------------------------------------------------------

!     Consider first the case of "duplicate" variables, that is variables
!     playing exactly the same role in the objective function and constraints.
!     the k-th variable is replaced by  the linear combination
!     ( variable k + alpha * variable j ).

!-------------------------------------------------------------------------------

      IF ( PRESOLVE_is_zero( alpha * prob%G( k ) - prob%G( j ) ) ) THEN

         IF ( s%tt + 5  > control%max_nbr_transforms ) RETURN

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    duplicate variables (no cancellation)'

!        Compute the bounds for the new merged variable.

         IF ( alpha > 0 ) THEN
            IF ( xlj > s%M_INFINITY .AND. xlk > s%M_INFINITY ) THEN
               nxl = alpha * xlj + xlk
            ELSE
               nxl = - s%INFINITY
            END IF
            IF ( xuj < s%P_INFINITY .AND. xuk < s%P_INFINITY ) THEN
               nxu = alpha * xuj + xuk
            ELSE
               nxu = s%INFINITY
            END IF
         ELSE
            IF ( xuj < s%P_INFINITY .AND. xlk > s%M_INFINITY ) THEN
               nxl = alpha * xuj + xlk
            ELSE
               nxl = - s%INFINITY
            END IF
            IF ( xlj > s%M_INFINITY .AND. xuk < s%P_INFINITY ) THEN
               nxu = alpha * xlj + xuk
            ELSE
               nxu = s%INFINITY
            END IF
         END IF

!        Avoid the transformation if any of the finite bounds exceed the
!        maximum acceptable value for the reduced problem.

         IF ( PRESOLVE_is_too_big( nxl ) .OR. &
              PRESOLVE_is_too_big( nxu )      ) THEN
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '   merging x(', j, ') into x(', k, ') prevented by',           &
               ' unacceptable growth'
            RETURN
         END IF

!        Compute the bounds for the dual of the new merged variable.

         zlj = prob%Z_l( j )
         zuj = prob%Z_u( j )
         zlk = prob%Z_l( k )
         zuk = prob%Z_u( k )
         IF ( alpha > 0 ) THEN
            IF ( zlj > s%M_INFINITY .AND. zlk > s%M_INFINITY ) THEN
               nzl = alpha * zlj + zlk
            ELSE
               nzl = -s%INFINITY
            END IF
            IF ( zuj < s%P_INFINITY .AND. zuk < s%P_INFINITY ) THEN
               nzu = alpha * zuj + zuk
            ELSE
               nzu = s%INFINITY
            END IF
         ELSE
            IF ( zuj < s%P_INFINITY .AND. zlk > s%M_INFINITY ) THEN
               nzl = alpha * zuj + zlk
            ELSE
               nzl = -s%INFINITY
            END IF
            IF ( zlj > s%M_INFINITY .AND. zuk < s%P_INFINITY ) THEN
               nzu = alpha * zlj + zuk
            ELSE
               nzu = s%INFINITY
            END IF
         END IF

!        Avoid the transformation if any of the finite bounds exceed the
!        maximum acceptable value for the reduced problem.

         IF ( PRESOLVE_is_too_big( nzl ) .OR. &
              PRESOLVE_is_too_big( nzu )      ) THEN
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '   merging x(', j, ') into x(', k, ') prevented by',           &
               ' unacceptable growth'
            RETURN
         END IF

!        Record the transformation.

         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l                = s%tm + 1
         s%tt             = s%tt + 1
         s%tm             = l
         s%hist_type( l ) = X_MERGE
         s%hist_i( l )    = j
         s%hist_j( l )    = k
         s%hist_r( l )    = alpha
         IF ( s%level >= ACTION ) WRITE( s%out, * )                           &
            '  [', s%tt, '] merging x(', j, ') into x(', k, ') with', alpha

!        Remember the dependencies.

         s%needs( Z_VALS, X_VALS ) = MIN( s%tt, s%needs( Z_VALS, X_VALS ) )
         s%needs( X_VALS, X_BNDS ) = MIN( s%tt, s%needs( X_VALS, X_BNDS ) )

!        Deactivate the j-th variable.

         CALL PRESOLVE_do_ignore_x( j )

!        Now update the bounds on x(k).

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    => new x(', k, ')  in [', nxl, ',',         &
                 nxu, ']'
         END IF

         CALL PRESOLVE_set_bound_x( k, LOWER, nxl )
         IF ( inform%status /= OK ) RETURN
         CALL PRESOLVE_set_bound_x( k, UPPER, nxu )
         IF ( inform%status /= OK ) RETURN

!        Check primal feasibility.

         IF ( prob%X_l( k ) > prob%X_u( k ) ) THEN
            inform%status = PRIMAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is primal infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because the bounds on x(', k, ') are incompatible'
            RETURN
         END IF

!        Now update the bounds on z(k), using the SET attribute to impose
!        the new bounds as the previous ones are now irrelevant.

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    z(', k, ') in [', zlk, ',', zuk, '] and '
            WRITE( s%out, * ) '    z(', j, ') in [', zlj, ',', zuj, ']'
            WRITE( s%out, * ) '    => new z(', k, ')  in [', nzl, ',',        &
                 nzu, ']'
         END IF

         CALL PRESOLVE_bound_z( k, LOWER, SET, nzl )
         IF ( inform%status /= OK ) RETURN
         CALL PRESOLVE_bound_z( k, UPPER, SET, nzu )
         IF ( inform%status /= OK ) RETURN

!        Check dual feasibility (which couldn't be done by PRESOLVE_bound_z
!        because the SET attribute is used).

         IF ( prob%Z_l( k ) > prob%Z_u( k ) ) THEN
            inform%status = DUAL_INFEASIBLE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE analysis stopped: the problem is dual infeasible'
            WRITE( inform%message( 2 ), * )                                    &
               '    because the bounds on z(', k, ') are incompatible'
            RETURN
         END IF

!-------------------------------------------------------------------------------

!                   One of variables j or k is dominated

!-------------------------------------------------------------------------------

      ELSE IF ( alpha * prob%G( k ) > prob%G( j ) ) THEN

!        We have that alpha * z(k) > z(j).

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    x(', k, ') dominates x(', j, ')'

         IF ( alpha > ZERO ) THEN

!           From:  alpha * z(k) > z(j) and alpha > 0:
!           if z(j) >= 0, then z(k) > 0, and x(k) may be fixed to its
!           lower bound.

!           NOTE: Here and in the other symmetric cases below, one needs to
!                 consider the real bounds on the dominated variables (and
!                 not only its implied status).

            IF (  xuj >= s%P_INFINITY .OR. prob%Z_l( j ) >= ZERO ) THEN
               IF ( xlk > s%M_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( k, xlk, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                  ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                  '    because x(', k, ') has an active infinite lower bound'
                  RETURN
               END IF
            END IF

!           From:  alpha * z(k) > z(j) and alpha > 0:
!           if z(k) <= 0, then z(j) < 0, and x(j) may be fixed to its
!           upper bound.

            IF ( xlk <= s%M_INFINITY .OR. prob%Z_u( k ) <= ZERO ) THEN
               IF ( xuj < s%P_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                    '    because x(', j, ') has an active infinite upper bound'
                  RETURN
               END IF
            END IF

         ELSE ! i.e. if alpha * z(k) > z(j) and alpha < 0

!           From:  z(k) < z(j) / alpha  and  alpha < 0
!           if z(j) >= 0, then z(k) < 0, and x(k) may be fixed to its
!           upper bound.

            IF ( xuj >= s%P_INFINITY .OR. prob%Z_l( j ) >= ZERO ) THEN
               IF ( xuk < s%P_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( k, xuk, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                    '    because x(', k, ') has an active infinite upper bound'
                  RETURN
               END IF
            END IF

!           From:  alpha * z(k) > z(j)  and  alpha < 0:
!           if z(k) >= 0, then z(j) < 0, and x(j) may be fixed to its
!           upper bound.

            IF ( xuk >= s%P_INFINITY .OR. prob%Z_l( k ) >= ZERO ) THEN
               IF ( xuj < s%P_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( j, xuj, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 1 ), * )                              &
                    '    because x(', j, ') has an active infinite upper bound'
                  RETURN
               END IF
            END IF
         END IF

      ELSE ! i.e. when alpha * prob%G( k ) < prob%G( j )

!        We have that alpha * z(k) < z(j).

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    x(', j, ') dominates x(', k, ')'

         IF ( alpha > ZERO ) THEN

!           From:  alpha * z(k) < z(j) and alpha > 0
!           if z(j) <= 0, then z(k) < 0, and x(k) may be fixed to its
!           upper bound.

            IF ( xlj <= s%M_INFINITY .OR. prob%Z_u( j ) <= ZERO ) THEN
               IF ( xuk > s%M_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( k, xuk, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                    '    because x(', k, ') has an active infinite upper bound'
                  RETURN
               END IF
            END IF

!           From:  alpha * z(k) < z(j) and alpha > 0:
!           if z(k) >= 0, then z(j) > 0, and x(j) may be fixed to its
!           lower bound.

            IF ( xuk >= s%P_INFINITY .OR. prob%Z_l( k ) >= ZERO ) THEN
               IF ( xlj < s%P_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                    '    because x(', j, ') has an active infinite lower bound'
                  RETURN
               END IF
            END IF

         ELSE ! i.e. if alpha * z(k) < z(j) and alpha < 0

!           From:  z(k) > z(j) / alpha  and  alpha < 0
!           if z(j) <= 0, then z(k) > 0, and x(k) may be fixed to its
!           lower bound.

            IF (  xlj <= s%M_INFINITY .OR. prob%Z_u( j ) <= ZERO ) THEN
               IF ( xlk < s%P_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( k, xlk, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 2 ), * )                              &
                    '   because x(', k, ') has an active infinite lower bound'
                  RETURN
               END IF
            END IF

!           From:  alpha * z(k) < z(j)  and  alpha < 0:
!           if z(k) <= 0, then z(j) > 0, and x(j) may be fixed to its
!           lower bound.

            IF ( xlk <= s%M_INFINITY .OR. prob%Z_u( k ) <= ZERO ) THEN
               IF ( xlj < s%P_INFINITY ) THEN
                  CALL PRESOLVE_fix_x( j, xlj, Z_FROM_DUAL_FEAS )
                  IF ( inform%status /= OK ) RETURN
               ELSE
                  inform%status = DUAL_INFEASIBLE
                  WRITE( inform%message( 1 ), * )                              &
                    ' PRESOLVE analysis stopped: the problem is dual infeasible'
                  WRITE( inform%message( 1 ), * )                              &
                    '    because x(', j, ') has an active infinite lower bound'
                  RETURN
               END IF
            END IF
         END IF

      END IF

      IF ( s%tt >= control%max_nbr_transforms ) inform%status = MAX_NBR_TRANSF

      RETURN

      END SUBROUTINE PRESOLVE_merge_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_shift_x_bounds( i, j, kij )

!     Uses the doubleton equality row i to transfer the bounds on variable j
!     on the other variable occuring in row i.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the equality doubleton constraint

      INTEGER, INTENT( IN ) :: j

!            the index of the variable one wishes to free by removing its bounds

      INTEGER, INTENT( IN ) :: kij

!            the position in A of the element A(i,j)

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: k, jj, ic, l, kik
      LOGICAL            :: lower_active, upper_active
      REAL ( KIND = wp ) :: aij, aik, xlj, xuj, cli, xlk, xuk,                 &
                            nxlk, nxuk, zlk, zuk, yli, yui, nyli, nyui

!     Find the other active nonzero in row i.

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '    c(', i, ') is an equality doubleton'

      k  = 0
      ic = i
lic:  DO
         DO kik = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
            jj  = prob%A%col( kik )
            IF ( prob%X_status( jj ) <= ELIMINATED ) CYCLE
            aik = prob%A%val( kik )
            IF ( ABS( aik ) >= s%a_tol .AND. jj /= j ) THEN
               k = jj
               EXIT lic
            END IF
         END DO
         ic = s%conc( ic )
         IF ( ic == END_OF_LIST ) EXIT
      END DO lic

!     If the second element of the doubleton is too small exit.

      IF ( k == 0 ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    doubleton value too small'
         RETURN
      END IF

!     The doubleton has been found.

      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    doubleton element is in column', k

!     Do nothing if the other column in the doubleton row is itself a linear
!     singleton column which is already implied free.

      IF ( ( prob%X_status( k ) /= FREE .OR. s%A_col_s( k ) /= 1 ) .AND. &
           prob%X_status( k )   >  ELIMINATED                          ) THEN

!        Get the values that are relevant for the doubleton.

         aij = prob%A%val( kij )
         cli = prob%C_l( i )
         xlj = prob%X_l( j )
         xuj = prob%X_u( j )
         xlk = prob%X_l( k )
         xuk = prob%X_u( k )
         zlk = prob%Z_l( k )
         zuk = prob%Z_u( k )
         yli = prob%Y_l( i )
         yui = prob%Y_u( i )

!        Update the bounds on x(k).

         IF ( ABS( aik ) >= s%a_tol ) THEN
            IF ( aij * aik > 0 ) THEN
               IF ( xuj < s%P_INFINITY ) THEN
                  nxlk = ( cli - aij * xuj ) / aik
               ELSE
                  nxlk = - s%INFINITY
               END IF
               IF ( xlj > s%M_INFINITY ) THEN
                  nxuk = ( cli - aij * xlj ) / aik
               ELSE
                  nxuk = s%INFINITY
               END IF
            ELSE
               IF ( xlj > s%M_INFINITY ) THEN
                   nxlk = ( cli - aij * xlj ) / aik
               ELSE
                   nxlk = - s%INFINITY
               END IF
               IF ( xuj < s%P_INFINITY ) THEN
                  nxuk = ( cli - aij * xuj ) / aik
               ELSE
                  nxuk = s%INFINITY
               END IF
            END IF
         END IF

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    equality doubleton analysis details:',      &
                  '  c(', i, ') =', cli
            WRITE( s%out, * )                                                  &
                 '    A(', i,',', j , ') =', aij, 'A(', i,',', k , ') =', aik
            WRITE( s%out, * ) '    x(', j,') in [', xlj, ',', xuj, ']'
            WRITE( s%out, * ) '    => x(', k, ') in [', nxlk, ',', nxuk, ']'
         END IF

!        Determine which of the new bound is active in the sense that it
!        cuts away the original problem bound.

         lower_active = .NOT. PRESOLVE_is_neg( nxlk - xlk ) .AND. &
                        nxlk > s%M_INFINITY
         upper_active = .NOT. PRESOLVE_is_pos( nxuk - xuk ) .AND. &
                        nxuk < s%P_INFINITY

!        Record the shift in bound constraints, in order to allow restoration
!        of the proper values of the doubleton multiplier and of the dual
!        variables associated with x(j) and x(k).

!        Note that one remembers the index of the variable on which the new
!        bound(s) are imposed (ie k) and the position of A(i,k) in A (ie kij),
!        from which the values of i, k and A(i,k) can easily be reconstructed.

!        1) The new lower bound is active.

         IF ( lower_active ) THEN

!           If the new lower bound is equal to old upper bound, fix x(k)
!           but remember how to compute z(k).

            IF ( PRESOLVE_is_zero( nxlk - xuk ) ) THEN

!              fix x(k)

               CALL PRESOLVE_fix_x( k, xuk, Z_FROM_YZ_LOW, pos = kij )

!           No fixing is possible.

            ELSE

!              Update the lower bound on x(k).

               IF ( .NOT. PRESOLVE_is_zero( nxlk - xlk ) .AND. &
                    .NOT. PRESOLVE_is_too_big( nxlk )          ) THEN
                  IF ( s%tm >= s%max_tm ) THEN
                     CALL PRESOLVE_save_transf
                     IF ( inform%status /= OK ) RETURN
                  END IF
                  l    = s%tm + 1
                  s%tt = s%tt + 1
                  s%tm = l
                  s%hist_type( l ) = X_LOWER_UPDATED_S
                  s%hist_i( l ) = k
                  s%hist_j( l ) = kij
                  s%hist_r( l ) = xlk
                  prob%X_l( k ) = nxlk
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                      '  [', s%tt, '] tightening lower bound for x(', k,       &
                      ') to', nxlk, '(s)'
                  IF ( s%tt >= control%max_nbr_transforms ) THEN
                     inform%status = MAX_NBR_TRANSF
                     RETURN
                  END IF
               END IF

!              Update the status of x(k).

               SELECT CASE ( prob%X_status( k ) )
               CASE ( FREE )
                  prob%X_status( k ) = LOWER
                  CALL PRESOLVE_bound_z( k, UPPER, SET,   s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
               CASE ( UPPER )
                  prob%X_status( k ) = RANGE
                  CALL PRESOLVE_bound_z( k, UPPER, SET,   s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
               CASE ( RANGE )
                  CALL PRESOLVE_bound_z( k, LOWER, SET, - s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
                  CALL PRESOLVE_bound_z( k, UPPER, SET,   s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
               END SELECT
            END IF

!           Remember the dependencies.

            s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
            s%needs( Y_VALS, Z_VALS ) = MIN( s%tt, s%needs( Y_VALS, Z_VALS ) )
            s%needs( Y_VALS, A_VALS ) = MIN( s%tt, s%needs( Y_VALS, A_VALS ) )

         END IF

!        Update the loose lower bound.

         IF ( control%final_x_bounds /= TIGHTEST )                             &
             s%x_l2( k ) = MAX( nxlk, s%x_l2( k ) )

!        Update the value of the tight lower bound to reflect the fact that
!        it might have been reset just now.

         xlk = prob%X_l( k )

!        2) The new upper bound is active, but x(k) is not eliminated by
!           the possible thightening of the lower bound.

         IF ( upper_active .AND. prob%X_status( k ) > ELIMINATED ) THEN

!           If the new upper bound is equal to old lower bound, fix x(k)
!           but remember how to compute z(k).

            IF ( PRESOLVE_is_zero( nxuk - xlk ) ) THEN

!              Fix x(k).

               CALL PRESOLVE_fix_x( k, xlk, Z_FROM_YZ_UP, pos = kij )

!           No fixing is possible.

            ELSE

!              Update the upper bound on x(k).

               IF ( .NOT. PRESOLVE_is_zero( nxuk - xuk ) .AND. &
                    .NOT. PRESOLVE_is_too_big( nxuk )          ) THEN
                  IF ( s%tm >= s%max_tm ) THEN
                     CALL PRESOLVE_save_transf
                     IF ( inform%status /= OK ) RETURN
                  END IF
                  l    = s%tm + 1
                  s%tt = s%tt + 1
                  s%tm = l
                  s%hist_type( l ) = X_UPPER_UPDATED_S
                  s%hist_i( l ) = k
                  s%hist_j( l ) = kij
                  s%hist_r( l ) = xuk
                  prob%X_u( k ) = nxuk
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                      '  [', s%tt, '] tightening upper bound for x(', k,       &
                      ') to', nxuk, '(s)'
                  IF ( s%tt >= control%max_nbr_transforms ) THEN
                     inform%status = MAX_NBR_TRANSF
                     RETURN
                  END IF
               END IF

!              Update the status of x(k).

               SELECT CASE ( prob%X_status( k ) )
               CASE ( FREE )
                  prob%X_status( k )  = UPPER
                  CALL PRESOLVE_bound_z( k, LOWER, SET, - s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
               CASE ( LOWER )
                  prob%X_status( k )  = RANGE
                  CALL PRESOLVE_bound_z( k, LOWER, SET, - s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
               CASE ( RANGE )
                  CALL PRESOLVE_bound_z( k, LOWER, SET, - s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
                  CALL PRESOLVE_bound_z( k, UPPER, SET,   s%INFINITY )
                  IF ( inform%status /= OK ) RETURN
               END SELECT
            END IF

!           Remember the dependencies.

            s%needs( Z_VALS, A_VALS ) = MIN( s%tt, s%needs( Z_VALS, A_VALS ) )
            s%needs( Y_VALS, Z_VALS ) = MIN( s%tt, s%needs( Y_VALS, Z_VALS ) )
            s%needs( Y_VALS, A_VALS ) = MIN( s%tt, s%needs( Y_VALS, A_VALS ) )

         END IF

!        Update the loose upper bound.

         IF ( control%final_x_bounds /= TIGHTEST )                             &
            s%x_u2( k ) = MIN( nxuk, s%x_u2( k ) )

!        Print the status of x(k), if requested.

         IF ( s%level >= DEBUG ) THEN
            SELECT CASE ( prob%X_status( k ) )
            CASE ( FREE )
               WRITE( s%out, * ) '    x(',k,') now has implied status FREE'
            CASE ( LOWER )
               WRITE( s%out, * ) '    x(',k,') now has implied status LOWER'
            CASE ( UPPER )
               WRITE( s%out, * ) '    x(',k,') now has implied status UPPER'
            CASE ( RANGE )
               WRITE( s%out, * ) '    x(',k,') now has implied status RANGE'
            END SELECT
         END IF

!        Make column j implied free.

         prob%X_status( j ) = FREE
         CALL PRESOLVE_fix_z( j, ZERO )

!        Update the bounds on y(i).

         IF ( lower_active .OR. upper_active ) THEN
            IF ( aik >= s%a_tol ) THEN
               IF ( yli > s%M_INFINITY .AND. zlk > s%M_INFINITY ) THEN
                  nyli = yli + zlk / aik
               ELSE
                  nyli = - s%INFINITY
               END IF
               IF ( yui < s%P_INFINITY .AND. zuk < s%P_INFINITY ) THEN
                  nyui = yui + zuk / aik
               ELSE
                  nyui = s%INFINITY
               END IF
            ELSE IF ( aik <= - s%a_tol ) THEN
               IF ( yli > s%M_INFINITY .AND. zuk < s%P_INFINITY ) THEN
                  nyli = yli + zuk / aik
               ELSE
                  nyli = - s%INFINITY
               END IF
               IF ( yui < s%P_INFINITY .AND. zlk > s%M_INFINITY ) THEN
                  nyui = yui + zlk / aik
               ELSE
                  nyui = s%INFINITY
               END IF
            END IF
            CALL PRESOLVE_bound_y( i, LOWER, SET, nyli )
            CALL PRESOLVE_bound_y( i, UPPER, SET, nyui )
         END IF

      END IF

      RETURN

      END SUBROUTINE PRESOLVE_shift_x_bounds

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_transfer_x_bounds( i, j, kij, txstage, txdata, yi )

!     Transforms an equality constraint of the form

!                 A(i,j) x(j) + SUM A(i,p) x(p) = c

!     in which x(j) in [ x_l, x_u ] into a new constraint of the form

!                   c_l <= SUM A(i,p) x(p) <= c_u

!     assuming that variable j is then eliminated by substitution in the
!     original constraint. In effect, this frees x(j) by transfering its
!     bounds into the new inequalities.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the considered equality constraint;

      INTEGER, INTENT( IN ) :: j

!            the index of the variable whose bounds are transfered;

      INTEGER, INTENT( IN ) :: kij

!            the position of A(i,j) in A

      INTEGER, INTENT( IN ) :: txstage

!            if txstage = 0, then the values for the modified data items of the
!            reduced problem are computed and stored in txdata; if txstage =,
!            the transformation is applied from the data in txdata

      REAL( KIND = wp ) , DIMENSION( 6 ), INTENT( INOUT ) :: txdata

!            if txstage = 0, the following values are computed:
!            txdata( 1 ) : the new lower bound on ci after transformation,
!            txdata( 2 ) : the new upper bound on ci after transformation,
!            txdata( 3 ) : the new loose lower bound on ci after transformation,
!            txdata( 4 ) : the new loose upper bound on ci after transformation,
!            txdata( 5 ) : the new lower bound on yi after transformation,
!            txdata( 6 ) : the new upper bound on yi after transformation,
!            they are used in the transformation if txstage = 1

      REAL( KIND = wp ), OPTIONAL, INTENT( IN ) :: yi

!            the value of the multiplier associated with the constraint,
!            if known

!     Programming: Ph. Toint, May 2001.

!===============================================================================

!     Local variables

      INTEGER            :: l
      REAL ( KIND = wp ) :: xuj, xlj, zlj, zuj, aij, ci

!     Get the value of the pivotal element.

      aij = prob%A%val( kij )
      zuj = prob%Z_u( j )
      zlj = prob%Z_l( j )
      xlj = prob%X_l( j )
      xuj = prob%X_u( j )
      ci  = prob%C_l( i )

      SELECT CASE ( txstage )

      CASE ( 0 ) ! Compute the transformation values.

         txdata = ZERO

!        Anticipate the values for the lower and upper bounds of the i-th
!        constraint that reflect the bounds on x(j). (Note that at least one
!        of the new bounds must be finite, otherwise x(j) was already free.)

         IF ( aij > ZERO ) THEN
         IF ( xuj < s%P_INFINITY ) txdata( 1 ) = ci - aij * xuj
         IF ( xlj > s%M_INFINITY ) txdata( 2 ) = ci - aij * xlj
         ELSE
            IF ( xuj < s%P_INFINITY ) txdata( 2 ) = ci - aij * xuj
            IF ( xlj > s%M_INFINITY ) txdata( 1 ) = ci - aij * xlj
         END IF

!        Anticipate the values of the finite loose bounds in a similar manner.

         IF ( control%final_c_bounds /= TIGHTEST ) THEN
            xlj = s%x_l2( j )
            xuj = s%x_u2( j )
            IF ( aij > ZERO ) THEN
               IF ( xuj < s%P_INFINITY ) txdata( 3 ) = s%c_l2( i ) - aij * xuj
               IF ( xlj > s%M_INFINITY ) txdata( 4 ) = s%c_u2( i ) - aij * xlj
            ELSE
               IF ( xuj < s%P_INFINITY ) txdata( 4 ) = s%c_u2( i ) - aij * xuj
               IF ( xlj > s%M_INFINITY ) txdata( 3 ) = s%c_l2( i ) - aij * xlj
            END IF
         END IF

!        Anticipate the values of the finite bounds on y(i)

         IF ( aij > ZERO ) THEN
            IF ( zuj < s%P_INFINITY ) txdata( 5 ) = - zuj / aij
            IF ( zlj > s%M_INFINITY ) txdata( 6 ) = - zlj / aij
         ELSE
            IF ( zuj < s%P_INFINITY ) txdata( 6 ) = - zuj / aij
            IF ( zlj > s%M_INFINITY ) txdata( 5 ) = - zlj / aij
         END IF

      CASE ( 1 ) !  Apply the transformation.

!        Record the transformation.

!        Note that one remembers the index of the variable from which the
!        bound(s) are freed (ie j) and the position of A(i,j) in A (ie kij),
!        from which the value of i can easily be reconstructed.


         IF ( s%tm >= s%max_tm ) THEN
            CALL PRESOLVE_save_transf
            IF ( inform%status /= OK ) RETURN
         END IF
         l    = s%tm + 1
         s%tt = s%tt + 1
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  [', s%tt, '] transfering the bounds on x(', j, ') to c(', i, ')'
         s%tm             = l
         s%hist_type( l ) = X_BOUNDS_TO_C
         s%hist_i( l )    = kij

!        If the multiplier is known, save it and mark this fact by negating
!        the column index, otherwise save aij (unused).

         IF ( PRESENT( yi ) ) THEN
            s%hist_j( l ) = - j
            s%hist_r( l ) = yi
         ELSE
            s%hist_j( l ) =   j
            s%hist_r( l ) = aij
         END IF

!        Row i has been marked as a modified equality row by do_ignore_x( j ),
!        which is inadequate in this case since it is no longer an equality.
!        Thus make sure it is not selected in the list of potential
!        sparsification pivots.

         s%a_perm( i ) = IBCLR( s%a_perm( i ), ROW_SPARSIFICATION )

!        Effectively adapt the lower and upper bounds of the i-th constraint
!        to reflect the bounds on x(j).

         IF ( aij > ZERO ) THEN
            IF ( xuj < s%P_INFINITY ) THEN
               CALL PRESOLVE_bound_c( i, LOWER, UPDATE, txdata( 1 ) )
            ELSE
               CALL PRESOLVE_bound_c( i, LOWER, UPDATE, -s%INFINITY )
            END IF
            IF ( inform%status /= OK ) RETURN
            IF ( xlj > s%M_INFINITY ) THEN
               CALL PRESOLVE_bound_c( i, UPPER, UPDATE, txdata( 2 ) )
            ELSE
               CALL PRESOLVE_bound_c( i, UPPER, UPDATE,  s%INFINITY )
            END IF
         ELSE
            IF ( xuj < s%P_INFINITY ) THEN
               CALL PRESOLVE_bound_c( i, UPPER, UPDATE, txdata( 2 ) )
            ELSE
            CALL PRESOLVE_bound_c( i, UPPER, UPDATE,  s%INFINITY )
            END IF
            IF ( inform%status /= OK ) RETURN
            IF ( xlj > s%M_INFINITY ) THEN
               CALL PRESOLVE_bound_c( i, LOWER, UPDATE, txdata( 1 ) )
            ELSE
               CALL PRESOLVE_bound_c( i, LOWER, UPDATE, -s%INFINITY )
            END IF
         END IF
         IF ( inform%status /= OK ) RETURN

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    x(', j, ') in [', xlj, ',', xuj, ']'
            WRITE( s%out, * ) '    => shortened c(', i, ') in [',              &
                 prob%C_l( i ),  ',', prob%C_u( i ), ']'
            WRITE( s%out, * ) '    eliminated entry A(', i, ',', j, ') =', aij
         END IF

!        Effectively update the loose bounds.

         IF ( control%final_c_bounds /= TIGHTEST ) THEN
            xlj = s%x_l2( j )
            xuj = s%x_u2( j )
            IF ( aij > ZERO ) THEN
              IF ( xuj < s%P_INFINITY ) THEN
                  s%c_l2( i ) = txdata( 3 )
               ELSE
                  s%c_l2( i ) = - s%INFINITY
               END IF
               IF ( xlj > s%M_INFINITY ) THEN
                  s%c_u2( i ) = txdata( 4 )
               ELSE
                  s%c_u2( i ) = s%INFINITY
               END IF
            ELSE
               IF ( xuj < s%P_INFINITY ) THEN
                  s%c_u2( i ) = txdata( 4 )
               ELSE
                  s%c_u2( i ) = s%INFINITY
               END IF
               IF ( xlj > s%M_INFINITY ) THEN
                  s%c_l2( i ) = txdata( 3 )
               ELSE
                  s%c_l2( i ) = - s%INFINITY
               END IF
            END IF
         END IF

!        Compute the infinite bounds on the multipliers.

         IF ( aij > ZERO ) THEN
            IF ( zuj >= s%P_INFINITY ) txdata( 5 ) = - s%INFINITY
            IF ( zlj <= s%M_INFINITY ) txdata( 6 ) =   s%INFINITY
         ELSE
            IF ( zuj >= s%P_INFINITY ) txdata( 6 ) =   s%INFINITY
            IF ( zlj <= s%M_INFINITY ) txdata( 5 ) = - s%INFINITY
         END IF

!        Effectively reset the bounds.

         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    z(', j, ') in [', zlj, ',', zuj, ']'
            WRITE( s%out, * ) '    => new y(', i, ') in [', txdata( 5 ), ',',  &
                 txdata( 6 ), ']'
         END IF

         CALL PRESOLVE_bound_y( i, LOWER, SET, txdata( 5 ) )
         IF ( inform%status /= OK ) RETURN
         CALL PRESOLVE_bound_y( i, UPPER, SET, txdata( 6 ) )
         IF ( inform%status /= OK ) RETURN

!        Remember the dependencies.

         s%needs( C_BNDS, A_VALS ) = MIN( s%tt, s%needs( C_BNDS, A_VALS ) )
         s%needs( C_BNDS, X_BNDS ) = MIN( s%tt, s%needs( C_BNDS, X_BNDS ) )
         s%needs( Z_VALS, Y_VALS ) = MIN( s%tt, s%needs( Z_VALS, Y_VALS ) )

      END SELECT

      RETURN

      END SUBROUTINE PRESOLVE_transfer_x_bounds

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_save_h( last_written, first, last, filename )

!     Writes the transformations from positions first to last in the memory
!     buffer onto filename, starting after record last_written.

!     Arguments:

      INTEGER, INTENT( INOUT ) :: last_written

!              the (global) index of the last transformation written on
!              the file filename.  Updated by the subroutine.

      INTEGER, INTENT( IN ) :: first

!              the index (in memory) of the first transformation to write

      INTEGER, INTENT( IN ) ::  last

!              the index (in memory) of the last transformation to write

      CHARACTER( LEN = 30 ), INTENT( IN ) :: filename

!              the name of the file where the transformations have to be written
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: l, recnbr, iostat

!     Save the current content of the history from position first to last
!     (incrementing all record numbers by one to take the checksums record
!     into account).

      recnbr = last_written + 1
      DO l = first, last
         recnbr = recnbr + 1
         WRITE ( control%transf_file_nbr, REC = recnbr, IOSTAT = iostat )      &
               s%hist_type( l ), s%hist_i( l ), s%hist_j ( l ), s%hist_r( l )
         IF ( iostat /= 0 ) THEN
            inform%status = COULD_NOT_WRITE
            WRITE( inform%message( 1 ), * )                                    &
               ' PRESOLVE ERROR: an error occurred while writing on file'
            WRITE( inform%message( 2 ), * )                                    &
               '    ', filename, ' ( unit =', control%transf_file_nbr, ')'
            RETURN
         END IF
      END DO

      last_written = last_written + last - first + 1

      RETURN

      END SUBROUTINE PRESOLVE_save_h

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_save_transf

!     Saves the transformations currently in the memory buffer onto a file,
!     depending on the current presolving stage.

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Open the history file, if needed

      CALL PRESOLVE_open_h( 'REPLACE', control,inform, s )
      IF ( inform%status /= OK ) RETURN

!     Save the transformations in memory.

      IF ( s%level >= DEBUG ) WRITE( s%out,* ) '    saving transformations',   &
         s%ts + 1, 'to', s%ts + s%tm, 'to file ', control%transf_file_name

      CALL PRESOLVE_save_h( s%ts, 1, s%tm, control%transf_file_name )

!     Reset the number of transformations currently in memory and compute
!     how many can still fit, given the upper bound on their total number.

      s%tm     = 0
      s%max_tm = MIN( s%max_tm, control%max_nbr_transforms - s%tt )

      IF ( s%level >= DEBUG ) WRITE( s%out, * )'    ts =', s%ts,               &
                                               'max_tm =', s%max_tm

      IF ( s%max_tm == 0 ) THEN
         inform%status = MAX_NBR_TRANSF_TMP
         WRITE( inform%message( 1 ), * )                                       &
            ' PRESOLVE ERROR: the maximum number of problem transformations'
         WRITE( inform%message( 2 ), * )  '    has been reached'
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_save_transf

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_add_at_beg( j, first )

!     Adds the index j at the beginning of the linked list in h_perm starting
!     with first element first.

!     Arguments:

      INTEGER, INTENT( IN )    :: j

!            the index to add at the beginning of the linked list

      INTEGER, INTENT( INOUT ) :: first

!            the first element of the linked list, which also identifies
!            the list

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

      s%h_perm( j ) = first
      first         = j

      RETURN

      END SUBROUTINE PRESOLVE_add_at_beg

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_add_at_end( j, first, last )

!     Adds the index j at the end of the linked list in h_perm starting with
!     first element first and whose last element is in position last.

!     Arguments

      INTEGER, INTENT( IN )    :: j

!            the index to add at the end of the linked list

      INTEGER, INTENT( INOUT ) :: first

!            the first element of the linked list, which also identifies
!            the list

      INTEGER, INTENT( INOUT ) :: last

!            the last element of the linked list (to be replaced by j)

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

      IF ( first == END_OF_LIST ) THEN
         first = j
      ELSE
         s%h_perm( last ) = j
      END IF
      s%h_perm( j ) = END_OF_LIST
      last          = j

      RETURN

      END SUBROUTINE PRESOLVE_add_at_end

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_rm_from_list( j, first )

!     Removes the index j from the linked list (in h_perm) starting with first
!     element first.

!     Argument:

      INTEGER, INTENT( IN )    :: j

!            the index to remove from the linked list

      INTEGER, INTENT( INOUT ) :: first

!            the first element of the linked list, which also identifies
!            the list

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!  Local variables

      INTEGER :: p, k

      IF ( first == END_OF_LIST ) RETURN
      IF ( j == first ) THEN
         first         = s%h_perm( j )
         s%h_perm( j ) = 0
      ELSE
         p = first
         DO
           k = s%h_perm( p )
           IF ( k == j ) THEN
              s%h_perm( p ) = s%h_perm( j )
              s%h_perm( j ) = 0
              EXIT
           ELSE
              p = k
           END IF
           IF ( p == END_OF_LIST ) EXIT
         END DO
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_rm_from_list

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_reset( first )

!     Resets the linked list (in h_perm) to the empty list, starting with
!     the first element first.

!     Argument:

      INTEGER, INTENT( INOUT ) :: first

!            the first element of the linked list, which also identifies
!            the list

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER :: j, nxt

      j     = first
      first = END_OF_LIST
      DO
         nxt         = s%h_perm( j )
         s%h_perm( j ) = 0
         IF ( nxt  == END_OF_LIST ) EXIT
         j = nxt
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_reset

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_get_sizes_A

!     Obtain the number of nonzeros in the rows and columns of A, while, at
!     the same time, verifying that the column indices are in increasing order
!     within each row (and reordering the row if necessary). Also compute the
!     absolute pivoting threshold for A.

!     Programming: Ph. Toint, November 2000

!===============================================================================

!     Local variables

      INTEGER            :: i, k, j, nnzr, ic
      REAL ( KIND = wp ) :: a

!     Initialize the column sizes.

      s%A_col_s( :prob%n ) = 0

!     Initialize the maximal element.

      s%a_max = ZERO

!     Count the number of nonzeros in the row and columns

!     Loop on the active rows.

      DO i = 1, prob%m
         nnzr = 0
         IF ( prob%C_status( i ) > ELIMINATED ) THEN
            ic = i
            DO
               DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  j = prob%A%col( k )
                  IF (  prob%X_status( j ) <= ELIMINATED ) CYCLE
                  a = prob%A%val( k )
                  IF ( a == ZERO ) CYCLE
                  nnzr         = nnzr + 1
                  s%a_max        = MAX( s%a_max, ABS( a ) )
                  s%A_col_s( j ) = s%A_col_s( j ) + 1
               END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO
         END IF
         s%A_row_s( i ) = nnzr
         s%a_ne_active  = s%a_ne_active + nnzr
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_get_sizes_A

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_get_struct_H

!     Obtain the structure of nonzeros in the rows of H. This structure
!     is held in the vector H_str, whose j-th component means
!     > 0        : the j-th row is diagonal with diagonal element in
!                  position H_str( j ) in H,
!     0 ( EMPTY ): the j-th row is empty,
!     < 0        : the j-th row has - H_str( j ) nonzero off-diagonal elements
!                  (plus, possibly, a diagonal element)

!     Programming: Ph. Toint, November 2000

!===============================================================================

!     Local variables

      INTEGER                :: i, k, j, ii, is, ie
      LOGICAL                :: diagonal

!     Initialize the row structure indicators (to empty)

      s%H_str( :prob%n ) = EMPTY

!     Loop on the active rows.

      s%h_max = ZERO
      IF ( prob%H%ne > 0 ) THEN
         DO j = 1, prob%n
            IF ( prob%X_status( j ) == INACTIVE ) CYCLE

!           Subdiagonal part of the row

            diagonal = .FALSE.
            is   = prob%H%ptr( j )
            ie   = prob%H%ptr( j + 1 ) - 1
            DO k = is, ie
               i = prob%H%col( k )
               IF ( i > j ) THEN
                  inform%status = H_MISSPECIFIED
                  WRITE( inform%message( 1 ), * )                              &
                     ' PRESOLVE ERROR: upper diagonal entry found in row',     &
                     j, 'of H'
                  RETURN
               END IF
               IF ( prob%H%val( k ) == ZERO ) CYCLE
               s%h_max = MAX( s%h_max, ABS( prob%H%val ( k ) ) )
               s%h_ne_active = s%h_ne_active + 1
               IF ( i == j ) THEN
                  diagonal    = .TRUE.
               ELSE IF ( prob%X_status( i ) /= INACTIVE ) THEN
                  s%H_str( j )  = s%H_str( j ) - 1
               END IF
            END DO

!           Super-diagonal part of the row

            k = s%H_col_f( j )
            IF ( END_OF_LIST /= k ) THEN
               DO ii = 1, prob%n
                  i = s%H_row( k )
                  IF ( prob%X_status( i ) /= INACTIVE .AND. &
                       prob%H%val( k )    /= ZERO           ) THEN
                    s%H_str( j )  = s%H_str( j ) - 1
                  END IF
                  k = s%H_col_n( k )
                  IF ( k == END_OF_LIST ) EXIT
               END DO
            END IF

!          Check for the diagonal element.

            IF ( diagonal .AND. s%H_str( j ) == EMPTY ) THEN

!              see if the diagonal is nonzero

               IF ( prob%H%val( ie ) /= ZERO ) s%H_str( j ) = ie
            END IF

         END DO

      END IF

      IF ( s%level >= CRAZY ) WRITE( s%out, * )                               &
         '     H_str =', s%H_str( :prob%n )

      RETURN

      END SUBROUTINE PRESOLVE_get_struct_H

!===============================================================================
!===============================================================================

      INTEGER FUNCTION PRESOLVE_H_row_s( j )

!     Compute the size of row j of H, given H_str( j ).

!     Argument:

      INTEGER, INTENT( IN ) :: j

!            the index of the row of H whose size is requested

!     Programming: Ph. Toint, November 2000

!===============================================================================

!     Local variables

      INTEGER :: hsj, kd

!     The Hessian is non empty.

      IF ( prob%H%ne > 0 ) THEN

         hsj = s%H_str( j )

         SELECT CASE ( hsj )

         CASE ( EMPTY )               ! empty row
            PRESOLVE_H_row_s = 0
         CASE ( 1: )                  ! diagonal row
            PRESOLVE_H_row_s = 1
         CASE ( :-1 )                 ! non-diagonal row
            PRESOLVE_H_row_s = - hsj

!           See if the diagnal entry is nonzero

            kd = prob%H%ptr( j + 1 )
            IF ( kd > prob%H%ptr( j ) ) THEN
               kd = kd - 1
               IF ( prob%H%val( kd ) /= ZERO .AND. &
                    prob%H%col( kd ) == j          ) THEN
                  PRESOLVE_H_row_s = PRESOLVE_H_row_s + 1
               END IF
            END IF

         END SELECT

!     The Hessian is empty.

      ELSE
         PRESOLVE_H_row_s = 0
      END IF

      RETURN

      END FUNCTION PRESOLVE_H_row_s

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_get_sparse_cols_A

!     Obtains the column oriented structure of the matrix A
!     when it is stored in row-wise sparse format.
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: i, k, j

!     Initialize the pointers to the first column entries and to the next
!     entry in the current column.

      s%A_col_f( :prob%n  )   = END_OF_LIST
      s%A_col_n( :prob%A%ne ) = 0

!     Loop on the rows.

      DO i = prob%m, 1, -1

!        Loop on rows corresponding to nonzero entries.

         IF ( prob%C_status( i ) == INACTIVE ) CYCLE

         DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            IF ( prob%A%val( k ) == ZERO ) CYCLE

!           A nonzero is found, update the column linked list

            j = prob%A%col( k )
            IF ( prob%X_status( j ) == INACTIVE ) CYCLE
            IF ( s%A_col_f( j ) == END_OF_LIST ) THEN
               s%A_col_n( k ) = END_OF_LIST
            ELSE
               s%A_col_n( k ) = s%A_col_f( j )
            END IF
            s%A_col_f( j )  = k
            s%A_row( k )    = i
         END DO
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_get_sparse_cols_A

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_get_sparse_cols_H

!     Obtains the column oriented structure corresponding to superdiagonal
!     part of the rows of the Hessian H when it is stored in column-wise
!     sparse format, given that only its lower triangular part is stored.
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: i, k, j, is, ie
      REAL ( KIND = wp ) :: tmp

!     Initialize the pointers to the first row entries and to the next
!     entry in the current row.

      s%H_col_f( :prob%n  )   = END_OF_LIST
      s%H_col_n( :prob%H%ne ) = 0
      s%H_row  ( :prob%H%ne ) = 0

!     Loop on the active variables.

      DO i = prob%n, 1, -1
         is = prob%H%ptr( i )
         ie = prob%H%ptr( i + 1 ) - 1

!        Loop on active columns corresponding to nonzero entries to determine
!        the sparse structure. At the same time, ensure that the diagonal
!        element (if present) is the last element of the subdiagonal (that
!        is resides in position ie).

         IF ( prob%X_status( i ) == INACTIVE ) CYCLE

         DO k = is, ie

            IF ( prob%H%val( k ) == ZERO ) CYCLE

!           The diagonal is found, but is out of place: permute it at the end
!           of the subdiagonal.

            j = prob%H%col( k )
            IF ( j == i .AND. ie /= k  ) THEN
               j                = prob%H%col( ie )
               prob%H%col( k  ) = j
               prob%H%col( ie ) = i
               tmp              = prob%H%val( k  )
               prob%H%val( k  ) = prob%H%val( ie )
               prob%H%val( ie ) = tmp
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    row', i, ': diagonal element moved from position', k,   &
                  'to', ie
            END IF

!           An active nonzero is found, update the row linked list.

            IF ( prob%X_status( j ) == INACTIVE .OR. j == i ) CYCLE
            IF ( s%H_col_f( j ) == END_OF_LIST ) THEN
               s%H_col_n( k ) = END_OF_LIST
            ELSE
               s%H_col_n( k ) = s%H_col_f( j )
            END IF
            s%H_col_f( j ) = k
            s%H_row( k )   = i
         END DO
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_get_sparse_cols_H

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_decr_H_row_size( j )

!     Decreases by one the size of a non-diagonal Hessian row, while
!     maintaining the vector H_str.

!     Argument:

      INTEGER, INTENT( IN ) :: j

!            the index of the Hessian row whose size must be decreased

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER :: kd

!     Insert variable j in the list of potentially dependent variables.

      CALL PRESOLVE_mark( j, DEPENDENT_VARIABLES )

!     Update the Hessian structure

      s%H_str( j ) = s%H_str( j ) + 1
      IF ( s%H_str( j ) == 0 ) THEN
         kd = prob%H%ptr( j + 1 ) - 1
         IF ( kd >= prob%H%ptr( j ) ) THEN
            IF ( prob%H%val( kd ) /= ZERO .AND. &
                 prob%H%col( kd ) == j          ) s%H_str( j ) = kd
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_decr_H_row_size

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_decr_A_row_size( i )

!     Decreases the size of the i-th row of A by 1, also updating the list
!     of potential sparsification pivots.
!     Note that we have to guard against negative sizes because an element
!     might be removed from a concatenated row, in which case its size is
!     already set to 0.

!     Argument:

      INTEGER, INTENT( IN ) :: i

!            the index of the row of A whose size must be decreased

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Update the row size.

      s%A_row_s( i ) = MAX( s%A_row_s( i ) - 1, 0 )

!     Reinsert the i-th constraint in the list of potential
!     pivots for sparsification and the j-th variable in the list of
!     potentially dependent variables.

      IF ( s%A_row_s( i ) >= 2               .AND.   &
           prob%C_status( i ) > ELIMINATED .AND.   &
           prob%C_l( i ) == prob%C_u( i )        ) &
         CALL PRESOLVE_mark( i, ROW_SPARSIFICATION  )

      RETURN

      END SUBROUTINE PRESOLVE_decr_A_row_size

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_decr_A_col_size( j )

!     Decreases the size of the j-th column of A by one, while maintaining
!     the lists of linear singletons and doubletons. The list of potential
!     dependent variables is also updated.

!     Argument

      INTEGER, INTENT( IN ) :: j

!            the index of the column of A whose size must be decreased

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER :: col_size

!     Decrement the column size.

      col_size     = s%A_col_s( j )
      s%A_col_s( j ) = col_size - 1

!     Mark the variable at a potentially new dependent variable.

      CALL PRESOLVE_mark( j, DEPENDENT_VARIABLES )

!     Update the lists of special linear variables

      SELECT CASE ( col_size )

      CASE ( 1 )

!        Remove column j from the list, if it is in it.

         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) == EMPTY )                                       &
               CALL PRESOLVE_rm_from_list( j, s%lsc_f )
         ELSE
            CALL PRESOLVE_rm_from_list( j, s%lsc_f )
         END IF

      CASE ( 2 )

!        Add j to the list of linear singleton columns if it is linear.

         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) == EMPTY ) THEN
               CALL PRESOLVE_rm_from_list( j, s%ldc_f )
               CALL PRESOLVE_add_at_beg( j, s%lsc_f )
            END IF
         ELSE
            CALL PRESOLVE_rm_from_list( j, s%ldc_f )
            CALL PRESOLVE_add_at_beg( j, s%lsc_f )
         END IF

      CASE ( 3 )

!        Add j to the list of linear doubleton columns if it is linear.

         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) == EMPTY ) CALL PRESOLVE_add_at_beg( j, s%ldc_f )
         ELSE
            CALL PRESOLVE_add_at_beg( j, s%ldc_f )
         END IF

      END SELECT

      RETURN

      END SUBROUTINE PRESOLVE_decr_A_col_size

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_incr_A_col_size( j )

!     Increases the size of the j-th column of A by 1, while maintaining
!     the lists of linear singletons and doubletons.

!     Argument:

      INTEGER, INTENT( IN ) :: j

!            the index of the column of A whose size must be increased

!     Programming: Ph. Toint, November 2000.
!
!==============================================================================

!     Local variables

      INTEGER :: col_size

!     Increment the column size.

      col_size       = s%A_col_s( j )
      s%A_col_s( j ) = col_size + 1

!     Update the lists of special linear columns, if necessary.

      SELECT CASE ( col_size )

      CASE ( 0 )

!        Add j to the list of linear singleton column if it is linear

         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) == EMPTY ) CALL PRESOLVE_add_at_beg( j, s%lsc_f )
         ELSE
            CALL PRESOLVE_add_at_beg( j, s%lsc_f )
         END IF

      CASE ( 1 )

!        Add j to the list of linear doubleton columns if it is linear.

         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) == EMPTY ) THEN
               CALL PRESOLVE_rm_from_list( j, s%lsc_f )
               CALL PRESOLVE_add_at_beg( j, s%ldc_f )
            END IF
         ELSE
            CALL PRESOLVE_rm_from_list( j, s%lsc_f )
            CALL PRESOLVE_add_at_beg( j, s%ldc_f )
         END IF

      CASE ( 2 )

!        Remove j from the list of linear doubleton columns.

         IF ( prob%H%ne > 0 ) THEN
            IF ( s%H_str( j ) == EMPTY )                                       &
               CALL PRESOLVE_rm_from_list( j, s%ldc_f )
         ELSE
            CALL PRESOLVE_rm_from_list( j, s%ldc_f )
         END IF

      END SELECT

      RETURN

      END SUBROUTINE PRESOLVE_incr_A_col_size

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_check_active_sizes

!     Checks the sizes of A and H (only used in print modes at least DEBUG)

!     Programming: Ph. Toint, March 2001.
!
!===============================================================================

!     Local variables

      INTEGER :: i, ic, j, k, ie, nactx, nactc, nacta, nacte, nacth
      LOGICAL :: error

      error = .FALSE.

!     Accumulate the number of active variables and constraints.

      nactx = 0               ! number of active variables
      DO j = 1, prob%n
         IF ( prob%X_status( j ) > ELIMINATED ) nactx = nactx + 1
      END DO

      nactc = 0               ! number of active constraints
      DO i = 1, prob%m
         IF ( prob%C_status( i ) > ELIMINATED ) nactc = nactc + 1
      END DO

!     Verify those.

      IF ( nactx /= s%n_active ) THEN
         WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', nactx,              &
              'active columns found instead of', s%n_active
         error = .TRUE.
      END IF

      IF ( nactc /= s%m_active ) THEN
         WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', nactc,              &
              'active rows found instead of', s%m_active
         error = .TRUE.
      END IF


!     Accumulate the sizes of A.

      IF ( prob%m > 0 ) THEN

         s%w_n( :prob%n ) = 0   ! number of entries in each column
         s%w_m( :prob%m ) = 0   ! number of entries in each row
         nacta            = 0   ! number of active entries in A
         nacte            = 0   ! number of active equality constraints
         DO i = 1, prob%m
            IF ( prob%C_status( i ) > ELIMINATED ) THEN
               nactc = nactc + 1
               IF ( prob%C_l( i ) == prob%C_u( i ) ) nacte = nacte + 1
               ic = i
               DO
                  DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                     j = prob%A%col( k )
                     IF ( prob%X_status( j ) <= ELIMINATED .OR. &
                          prob%A%val( k )    == ZERO            ) CYCLE
                     nacta      = nacta    + 1
                     s%w_n( j ) = s%w_n( j ) + 1
                     s%w_m( i ) = s%w_m( i ) + 1
                  END DO
                  ic = s%conc( ic )
                  IF ( ic == END_OF_LIST ) EXIT
               END DO
            END IF
         END DO

!        Verify the row sizes.

         DO i  = 1, prob%m
            IF ( s%w_m( i ) /= s%A_row_s( i ) ) THEN
               WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', s%w_m ( i ),  &
                    'active elements found in row', i, 'of A instead of',      &
                    s%A_row_s( i )
               error = .TRUE.
            END IF
         END DO

!        Verify the column sizes.

         DO j = 1, prob%n
            IF ( s%w_n( j ) /= s%A_col_s( j ) ) THEN
               WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', s%w_n( j ) ,  &
                    ' active elements found in column', j, 'of A instead of',  &
                    s%A_col_s( j )
              error = .TRUE.
            END IF
         END DO

!        Verify the active dimensions.

         IF ( nacta /= s%a_ne_active ) THEN
            WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', nacta,           &
                 'active elements of A found instead of', s%a_ne_active
            error = .TRUE.
         END IF
         IF ( nacte /= s%m_eq_active ) THEN
            WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', nacte,           &
                 'active equality rows found instead of', s%m_eq_active
            error = .TRUE.
         END IF

      END IF

!     Accumulate the sizes of H.

      IF ( prob%H%ne > 0 ) THEN
         s%w_n( : prob%n ) = 0        ! number of nonzeros in each row
         s%w_mn( :prob%n ) = 0        ! diagonal nature of each row
         nacth             = 0        ! number of active entries in H
         DO j = 1, prob%n
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
            DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
               IF ( prob%H%val( k ) == ZERO ) CYCLE
               ic = prob%H%col( k )
               IF ( prob%X_status( ic ) <= ELIMINATED ) CYCLE
               nacth = nacth + 1
               s%w_n( j )  = s%w_n( j )  + 1
               IF ( j /= ic ) THEN
                  s%w_n( ic )  = s%w_n( ic ) + 1
                  s%w_mn( j )  = -1
                  s%w_mn( ic ) = -1
               END IF
            END DO

         END DO

!        Verify the row sizes.

         DO j = 1, prob%n
            k = PRESOLVE_H_row_s( j )
            IF ( s%w_n( j ) /= k ) THEN
               WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', s%w_n ( j ),  &
                    'active elements found in row', j, 'of H instead of', k
               error = .TRUE.
            END IF

!           Verify the diagonal row.

            IF ( s%w_mn( j ) >= 0 .AND. k == 1 ) THEN
               ie = prob%H%ptr( j + 1 ) - 1
               IF ( prob%H%val( ie ) == ZERO .OR. &
                   ie <  prob%H%ptr( j )     .OR. &
                   ie /= s%H_str( j )               ) THEN
                  WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:',            &
                    'row', j, 'of H is not diagonal ( H_str(', j, ') =',       &
                    s%H_str( j ), ')'
                  error = .TRUE.
               END IF
            END IF

         END DO

!        Verify the active dimensions.

         IF ( nacth /= s%h_ne_active ) THEN
            WRITE( s%out, * ) '    PRESOLVE INTERNAL ERROR:', nacth,           &
                 'active elements of H found instead of', s%h_ne_active
            error = .TRUE.
         END IF

      END IF

!     Stop if a wrong count was detected.

      IF ( error ) THEN
         inform%status = WRONG_A_COUNT
         WRITE( inform%message( 1 ) , * ) ' PRESOLVE INTERNAL error(s):',      &
              ' the dimensions of A or H have become inconsistent'
      ELSE
         WRITE( s%out, * ) '    A and H row and column sizes ok'
      END IF
      RETURN

      END SUBROUTINE PRESOLVE_check_active_sizes

!==============================================================================
!==============================================================================

      END SUBROUTINE PRESOLVE_apply

!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===================                                   =========================
!===================                                   =========================
!===================          R E S T O R E            =========================
!===================                                   =========================
!===================                                   =========================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_restore( prob, control, inform, s )

!     Restores the original problem bounds and the values of the variables,
!     multipliers, constraints and dual variables that correspond, in the
!     formulation of the original problem, to those obtained for the reduced
!     problem.
!
!     Note that restoring certain quantities implies the restoration of certain
!     others.  These dependencies have been tracked in the array
!
!        s%needs( desired, needed )
!
!     that contains the index of the first transformation where this dependence
!     is observed.
!
!     Besides the values returned from the solver, the restoration is based
!     on three sources of information:
!
!     - the original problem dimensions (n, m, a_ne, h_ne);
!     - the transformation history (a type, 2 integer and a real values per
!       transformation), possibly stored on disk;
!     - the content of the workspace saved from the analysis and permutation
!       phases.  In particular,
!
!       * a_perm( :prob%A%ne )       : the permutation of A,
!       * conc( :prob%m )            : the constraint concatenation indicators,
!       * A_col_f, A_col_n and A_row : the column structure of A,
!       * w_m( :prob%m + 1 )         : the starting positions of the rows of A
!                                      in the original structure,
!       * h_perm( :prob%H%ne )       : the permutation of H,
!       * H_col_f, A_col_n and H_row : the column structure of the lower
!                                      triangular part of H,
!       * w_mn( :prob%n + 1 )        : the starting positions of the rows of H
!                                      in the original structure,
!       * w_n( :prob%n )             : the linked lists of variables in forcing
!                                      constraints, and of variables fixed "at
!                                      the last minute",
!       * x_l2, x_u2, ..., y_u2      : the (saved) tightest bounds on the
!                                      reduced problem (in NON_DEGENERATE or
!                                      LOOSEST bounding schemes).
!
!     Note that not all information available in the analysis phase is not
!     reconstructed.  In particular, the row and columns sizes for A and H
!     are no longer available in the restore mode.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER :: l, ii, jj, k1, k2, i, j, k, kk, ic, io, ie, ko, jo, je, ke,   &
                 col, next, lfirst, llast, ll, e, iostat, khpj, khpp, smt_stat
      LOGICAL :: get( 10 ), primal_feasible, dual_feasible
      REAL ( KIND = wp ) :: rr, xval, aval, ccorr, alpha, aij, r, gp, yval,    &
                 rg, a, zj, cval, cl, cu, zval, err, hpj, hjj

!-------------------------------------------------------------------------------

!                            Initial checks

!-------------------------------------------------------------------------------

!    See if any value must be restored.

      IF ( .NOT. ( control%get_x .OR. control%get_x_bounds .OR.        &
                   control%get_z .OR. control%get_z_bounds .OR.        &
                   control%get_c .OR. control%get_c_bounds .OR.        &
                   control%get_y .OR. control%get_y_bounds .OR.        &
                   control%get_A .OR. control%get_H .OR.               &
                   control%get_f .OR. control%get_g .OR. control%get_q )) RETURN

!     Check if the problem has been processed before.

      IF ( s%stage /= PERMUTED .AND. &
           s%stage /= ANALYZED .AND. &
           s%stage /= FULLY_REDUCED  ) THEN
         inform%status = PROBLEM_NOT_PERMUTED
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: the problem has not been permuted or fully'
         WRITE( inform%message( 2 ), * )                                       &
              '    reduced before an attempt is made to restore it'
         RETURN
      END IF

!     Print banner.

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *         GALAHAD PRESOLVE for QPs         *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *            problem restoration           *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' '
      END IF

!     Reset the matrix dimensions.

      prob%A%ne = prob%A%ptr( prob%m + 1 ) - 1
      IF (  s%h_type /= ABSENT ) THEN
        prob%H%ne = prob%H%ptr( prob%n + 1 ) - 1
      ELSE
        prob%H%ne = 0
      END IF

!     Review the control structure.

      CALL PRESOLVE_revise_control( RESTORE, prob, control, inform, s )

!-------------------------------------------------------------------------------
!
!              Test the primal and dual feasibility of the input.
!
!-------------------------------------------------------------------------------

!     Primal feasibility

      IF ( control%check_primal_feasibility >= BASIC ) THEN

!        Accumulate the primal feasibility error on the linear constraints

         primal_feasible = .TRUE.
         err             = ZERO

         IF ( s%level >= DEBUG ) CALL PRESOLVE_compute_c( .FALSE., prob, s )

         DO i = 1, prob%m

!           Compute the error on the i-th primal constraint (alpha).

            cval  = prob%C( i )
            cl    = prob%C_l( i )
            cu    = prob%C_u( i )
            alpha = MAX( ZERO, cl - cval, cval - cu )

!           If the error is larger that the user specified maximum, set the
!           primal infeasibility flag and possibly print the current error.

            IF ( alpha > control%c_accuracy ) THEN
               err = MAX( err, alpha )
               IF ( primal_feasible ) THEN
                  primal_feasible = .FALSE.
                  IF ( s%level >= TRACE ) THEN
                     WRITE( s%out, * ) ' PRESOLVE WARNING:',                   &
                          ' the input value of x is not primal feasible!'
                     IF ( s%level >= ACTION ) THEN
                        WRITE( s%out, * ) '      i      C_l( i )       ',      &
                             'C( i )        C_u( i )   infeasibility'
                     END IF
                  END IF
               END IF
               IF ( s%level >= ACTION ) WRITE( s%out, 1007 )                   &
                                        i, cl, cval, cu, alpha
            END IF
         END DO

!        Print the result of the verification  and possibly stop if x is
!           not primal feasible and the level of verification is SEVERE.

         IF ( .NOT. primal_feasible ) THEN
            IF ( s%level >= TRACE ) WRITE( s%out, * )                          &
               '    ( maximum primal infeasibility =', err, ')'
            IF ( control%check_primal_feasibility >= SEVERE ) THEN
               inform%status = X_NOT_FEASIBLE
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: x is infeasible on input of RESTORE'
               RETURN
            END IF
         ELSE
           IF ( s%level >= TRACE ) WRITE( s%out, * )                           &
              ' the input value of x is primal feasible'
         END IF
      END IF

!     Dual feasibility

      IF ( control%check_dual_feasibility >= BASIC ) THEN

!        Compute the ideal value of the dual variables, given those of
!        the primal variables and of the multipliers.

!        Allocate the space needed to store the ideal dual variables

         ALLOCATE( s%ztmp( prob%n ), STAT = iostat )
         IF ( iostat /= 0 ) THEN
            WRITE( s%out, * )                                                  &
                 ' PRESOLVE WARNING: no memory left for allocating ztmp(',     &
                  prob%n, ')'
            WRITE( s%out, * )                                                  &
                 '                   Skipping dual feasibility check.'
         ELSE

!           Compute the vector of ideal dual variables (in the sense that they
!           satisfy the dual feasibility equation exactly).

            CALL PRESOLVE_compute_z( s%ztmp )

!           Accumulate the dual feasibility error on all variables.

            dual_feasible = .TRUE.
            err           = ZERO

            DO j = 1, prob%n

!              Compute the error of the j-th dual variable (alpha).

               zval  = s%ztmp( j )
               alpha = ABS( prob%Z( j ) - zval )

!              If the error is larger than the user specified maximum value,
!              set the dual infeasibility flag and possibly print the current
!              value of the error.

               IF ( alpha > control%z_accuracy ) THEN
                  err = MAX( err, alpha )
                  IF ( dual_feasible ) THEN
                     dual_feasible = .FALSE.
                     IF ( s%level >= TRACE ) THEN
                        WRITE( s%out, * ) ' PRESOLVE WARNING:',                &
                             ' the input value of z is not dual feasible!'
                        IF ( s%level >= ACTION ) THEN
                           WRITE( s%out, * )                                   &
                            '      j      Z( j )    optimal Z( j )   difference'
                        END IF
                     END IF
                  END IF
                  IF ( s%level >= ACTION ) WRITE( s%out, 1007 ) j, prob%Z( j ),&
                     zval, alpha
               END IF
            END DO

!           Print the result of the verification and possibly stop if z is
!           not dual feasible and the level of verification is SEVERE.

            IF ( .NOT. dual_feasible ) THEN
               IF ( s%level >= TRACE ) WRITE( s%out, * )                       &
                  '    ( maximum dual infeasibility =', err, ')'
               IF ( control%check_dual_feasibility >= SEVERE ) THEN
                  inform%status = Z_NOT_FEASIBLE
                  WRITE( inform%message( 2 ), * ) ' PRESOLVE ERROR:',          &
                       ' ( x, y, z ) are not dual feasible on input of RESTORE'
                  RETURN
               END IF
            ELSE
              IF ( s%level >= TRACE ) WRITE( s%out, * )                        &
              ' the input value of z(x,y) is dual feasible'
           END IF
         END IF
      END IF

!-------------------------------------------------------------------------------
!
!                  Restore the original problem dimensions.
!
!-------------------------------------------------------------------------------

      s%n_active = prob%n
      s%m_active = prob%m
      prob%n     = s%n_original
      prob%m     = s%m_original
      prob%A%ne  = s%a_ne_original
      prob%H%ne  = s%h_ne_original

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * ) '   dimensions restored'
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    -->', MAX( 0, prob%A%ne ), 'nonzeros in A and',               &
            MAX( 0, prob%H%ne ), 'in H'
      END IF

!-------------------------------------------------------------------------------
!
!           Impose logical relations between the desired restorations.
!
!      Note: since the value of the constraints is restored at the end
!            from the equation c = A*x, restoring them implies restoring
!            x and A. Similarly, the value of q is compute at the end
!            from q =f + g^T*x + (1/2)x^T*H*x and thus the restoration
!            of q implies that of f, g and H. Finally, the value of the
!            dual variables is computed from z = g + H*x - A^T y, and
!            the restoration of z therefore implies that of g, A, H, x,
!            and y.
!
!-------------------------------------------------------------------------------

      control%get_x        = control%get_x .OR. control%get_q .OR.             &
                             control%get_c .OR. control%get_z
      control%get_f        = control%get_f .OR. control%get_q
      control%get_g        = control%get_g .OR. control%get_q .OR. control%get_z
      control%get_y        = control%get_y .OR. control%get_c .OR. control%get_z
      control%get_H        = control%get_H .OR. control%get_q .OR. control%get_z
      control%get_A        = control%get_A .OR. control%get_c .OR. control%get_z

!     Avoid reconstructing empty values.

      control%get_c        = control%get_c        .AND. prob%m    > 0
      control%get_c_bounds = control%get_c_bounds .AND. prob%m    > 0
      control%get_y        = control%get_y        .AND. prob%m    > 0
      control%get_A        = control%get_A        .AND. prob%A%ne > 0
      control%get_H        = control%get_H        .AND. prob%H%ne > 0

!     Ensure mutual implications and check dependencies at the end of the
!     transformation history.
!     (note that there are no dependencies from C_VALS, Q_VAL or Y_BNDS)

      CALL PRESOLVE_solve_implications
      get = .TRUE.
      CALL PRESOLVE_check_dependencies( s%tt, next, get )

!     Print the table of dependencies.

      IF ( s%level >= DEBUG ) THEN
         WRITE( s%out, * ) '   needs  ',                                       &
         'X_VALS Z_VALS C_BNDS Y_VALS F_VAL  G_VALS X_BNDS Z_BNDS H_VALS A_VALS'
         IF ( get( X_VALS ) ) WRITE( s%out,'( 3X, A7, 10I7)')                  &
              ' X_VALS', ( s%needs( X_VALS, k ), k = 1, 10 )
         IF ( get( Z_VALS ) ) WRITE( s%out,'( 3X, A7, 10I7)')                  &
              ' Z_VALS', ( s%needs( Z_VALS, k ), k = 1, 10 )
         IF ( get( C_BNDS ) ) WRITE( s%out,'( 3X, A7, 10I7)')                  &
              ' C_BNDS', ( s%needs( C_BNDS, k ), k = 1, 10 )
         IF ( get( Y_VALS ) ) WRITE( s%out,'( 3X, A7, 10I7)')                  &
              ' Y_VALS', ( s%needs( Y_VALS, k ), k = 1, 10 )
         IF ( get( F_VAL  ) ) WRITE( s%out,'( 3X, A7, 10I7)')                  &
              ' F_VAL ', ( s%needs( F_VAL , k ), k = 1, 10 )
         IF ( get( G_VALS ) ) WRITE( s%out,'( 3X, A7, 10I7)')                  &
              ' G_VALS', ( s%needs( G_VALS, k ), k = 1, 10 )
      END IF

      IF ( s%level >= DETAILS ) THEN
         WRITE( s%out, * ) '   restoration logic revised'
         IF( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    get_x =', get( X_VALS ),                    &
                                    '    get_x_bounds =', get( X_BNDS ) ,      &
                                    '    get_z =', get( Z_VALS ),              &
                                    '    get_z_bounds =', get( Z_BNDS )
            WRITE( s%out, * ) '    get_c =', control%get_c,                    &
                                    '    get_c_bounds =', get( C_BNDS ),       &
                                    '    get_y =', get( Y_VALS ),              &
                                    '    get_y_bounds =', control%get_y_bounds
            WRITE( s%out, * ) '    get_f =', get( F_VAL ),                     &
                                    '    get_g        =', get( G_VALS ),       &
                                    '    get_H =', get( H_VALS ),              &
                                    '    get_A        =', get( A_VALS )
         END IF
      END IF

!-------------------------------------------------------------------------------
!
!     Enforce the default convention on the sign of the multipliers and dual
!     variables.
!
!-------------------------------------------------------------------------------

!     Dual variables

      IF ( control%z_sign /= POSITIVE .AND.            &
           ( control%get_z .OR. control%get_z_bounds ) ) THEN
         DO j = 1, prob%n
            IF ( control%get_z ) prob%Z( j ) = - prob%Z( j )
            IF ( control%get_z_bounds ) THEN
               zj            = - prob%Z_l( j )
               prob%Z_l( j ) = - prob%Z_u( j )
               prob%Z_u( j ) = zj
            END IF
         END DO
      END IF

!     Multipliers

      IF ( control%y_sign /= POSITIVE   .AND.          &
           ( control%get_y .OR. control%get_y_bounds ) ) THEN
         DO i = 1, prob%m
            IF ( control%get_y ) prob%Y( i ) = - prob%Y( i )
            IF ( control%get_y_bounds ) THEN
               zj            = - prob%Y_l( i )
               prob%Y_l( i ) = - prob%Y_u( i )
               prob%Y_u( i ) = zj
            END IF
         END DO
      END IF

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
!                     THE PROBLEM HAS BEEN PERMUTED
!     In this case, the vectors and matrices have been permuted and must
!     be unpermuted before undoing any other transformation.
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

      IF ( s%stage == PERMUTED ) THEN
         inform%status = OK

         IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    map reported set'

!-------------------------------------------------------------------------------
!
!                     Restore the vectors in the variable space.
!
!-------------------------------------------------------------------------------

         IF ( get( X_VALS ) ) THEN
            CALL SORT_inverse_permute( prob%n, prob%X_status, x = prob%X   )
         END IF
         IF ( get( X_BNDS ) ) THEN
            IF ( control%final_x_bounds /= TIGHTEST ) THEN
               CALL PRESOLVE_swap( prob%n, s%x_l2, prob%X_l )
               CALL PRESOLVE_swap( prob%n, s%x_u2, prob%X_u )
            ELSE
               CALL SORT_inverse_permute( prob%n, prob%X_status, x=prob%X_l)
               CALL SORT_inverse_permute( prob%n, prob%X_status, x=prob%X_u)
            END IF
         END IF
         IF ( get( Z_VALS ) ) THEN
            CALL SORT_inverse_permute( prob%n, prob%X_status, x = prob%Z   )
         END IF
         IF ( get( Z_BNDS ) ) THEN
           IF ( control%final_z_bounds /= TIGHTEST ) THEN
              CALL PRESOLVE_swap( prob%n, s%z_l2, prob%Z_l )
              CALL PRESOLVE_swap( prob%n, s%z_u2, prob%Z_u )
           ELSE
               CALL SORT_inverse_permute( prob%n, prob%X_status, x=prob%Z_l)
               CALL SORT_inverse_permute( prob%n, prob%X_status, x=prob%Z_u)
           END IF
         END IF
         IF ( get( G_VALS ) ) THEN
            CALL SORT_inverse_permute( prob%n, prob%X_status, x = prob%G   )
         END IF

         IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   x-vectors restored'

!-------------------------------------------------------------------------------
!
!                                Restore H.
!
!-------------------------------------------------------------------------------

         IF ( get( H_VALS ) ) THEN

!           Compute the inverse permutation for the variables.

            CALL SORT_inplace_invert( prob%n, prob%X_status )

!           Recompute the column indices, taking into account that the
!           permutation may have caused nonzero entries to move from
!           the lower triangular part of H to its upper triangular part.

            IF ( s%level >= DEBUG ) THEN
               IF ( s%level >= CRAZY ) THEN
                  WRITE( s%out, * ) '     H_perm =', s%h_perm( :prob%H%ne )
                  WRITE( s%out, * ) '     H_ptr  =', s%w_mn( :prob%n + 1 )
                  WRITE( s%out, * ) '     H_col  =', prob%H%col(:prob%H%ne)
                  WRITE( s%out, 1001 )   prob%H%val( :prob%H%ne )
               END IF
               WRITE( s%out, * ) '    unscrambling H columns'
               WRITE( s%out, * )                                               &
                    '       k     ii   jj                    i    j'
            END IF

            DO ii = 1, s%n_active
               i  = prob%X_status( ii )
               DO k  = prob%H%ptr( ii ), prob%H%ptr( ii + 1 ) - 1
                  jj = prob%H%col( k )
                  j  = prob%X_status( jj )
                  IF ( i < j ) THEN
                     prob%H%col( k ) = ii
                     IF ( s%level >= DEBUG ) WRITE( s%out, 1003 )              &
                        k, ii, jj, 'flipped', j, i
                  ELSE IF ( s%level >= DEBUG ) THEN
                     WRITE( s%out, 1003 ) k, ii, jj, ' direct', i, j
                  END IF
               END DO
            END DO

!           Remember the starting positions of the rows in the
!           original structure.

            prob%H%ptr( :prob%n + 1 ) = s%w_mn( :prob%n + 1 )

            IF ( s%level >= DEBUG ) THEN
               IF ( s%level >= CRAZY ) WRITE( s%out, * )                       &
                  '     H_col  =', prob%H%col(:prob%H%ne)
               WRITE( s%out, * ) '    unpermuting H'
            END IF

!           Unpermute H and its column indices.

            CALL SORT_inverse_permute( prob%H%ne, s%h_perm,                    &
                                       x = prob%H%val, ix = prob%H%col )
            IF ( s%level >= DEBUG ) THEN
               IF ( s%level >= CRAZY ) THEN
                  WRITE( s%out, * ) '     H_ptr  =', prob%H%ptr( :prob%n + 1 )
                  WRITE( s%out, * ) '     H_col  =', prob%H%col( :prob%H%ne )
                  WRITE( s%out, 1001 )   prob%H%val( :prob%H%ne )
               END IF
               WRITE( s%out, * ) '    unpermuting the column indices'
            END IF

!           Unpermute the values of the column indices.

            prob%H%col( :prob%H%ne ) = prob%X_status( prob%H%col( :prob%H%ne ) )
            CALL SORT_inplace_invert( prob%n, prob%X_status )
            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     H_col =', prob%H%col( :prob%H%ne )

            IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   H restored'
         END IF

!        Restore the information preserved in w_n into h_perm (in particular
!        the lists of fixed variables corresponding to each forcing constraint).

         s%h_perm( :prob%n ) = s%w_n( :prob%n )

!-------------------------------------------------------------------------------
!
!      Restore the constraints left- and right-hand sides, multipliers and
!                        their (tightest) bounds.
!
!-------------------------------------------------------------------------------

         IF ( prob%m > 0 ) THEN
            CALL SORT_inverse_permute( prob%m, prob%C_status, x=prob%C  )
            IF ( get( C_BNDS ) ) THEN
               IF ( control%final_c_bounds /= TIGHTEST ) THEN
                  CALL PRESOLVE_swap( prob%m, s%c_l2, prob%C_l )
                  CALL PRESOLVE_swap( prob%m, s%c_u2, prob%C_u )
               ELSE
                  CALL SORT_inverse_permute( prob%m, prob%C_status, x=prob%C_l )
                  CALL SORT_inverse_permute( prob%m, prob%C_status, x=prob%C_u )
               END IF
            END IF
            IF ( get( Y_VALS ) ) THEN
               CALL SORT_inverse_permute( prob%m, prob%C_status, x=prob%Y  )
            END IF
            IF ( control%get_y_bounds ) THEN
               IF ( control%final_y_bounds /= TIGHTEST ) THEN
                  CALL PRESOLVE_swap( prob%m, s%y_l2, prob%Y_l )
                  CALL PRESOLVE_swap( prob%m, s%y_u2, prob%Y_u )
               ELSE
                  CALL SORT_inverse_permute( prob%m, prob%C_status, x=prob%Y_l )
                  CALL SORT_inverse_permute( prob%m, prob%C_status, x=prob%Y_u )
               END IF
            END IF
            IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   c-vectors restored'
         END IF

!-------------------------------------------------------------------------------
!
!                               Restore A.
!
!-------------------------------------------------------------------------------

         IF ( get( A_VALS ) ) THEN

!           Recover the starting position of each original row.

            prob%A%ptr( :prob%m + 1 ) = s%w_m( :prob%m + 1 )

!           Unpermute A and its column indices.

            IF ( s%level >= DEBUG ) THEN
               IF ( s%level >= CRAZY ) THEN
                  WRITE( s%out, * ) '     A_perm =', s%a_perm( :prob%A%ne )
                  WRITE( s%out, * ) '     A_ptr  =', prob%A%ptr( :prob%m + 1 )
                  WRITE( s%out, * ) '     A_col  =', prob%A%col( :prob%A%ne )
                  WRITE( s%out, 1000 )   prob%A%val( :prob%A%ne )
               END IF
               WRITE( s%out, * ) '    unpermuting A'
            END IF

            CALL SORT_inverse_permute( prob%A%ne, s%a_perm,                    &
                                       x = prob%A%val,ix = prob%A%col )

!           Unpermute the values of the column indices.

            IF ( s%level >= DEBUG ) THEN
               IF ( s%level >= CRAZY ) THEN
                  WRITE( s%out, * ) '     A_ptr  =', prob%A%ptr( :prob%m + 1 )
                  WRITE( s%out, * ) '     A_col  =', prob%A%col( :prob%A%ne )
                  WRITE( s%out, 1000 )   prob%A%val( :prob%A%ne )
               END IF
               WRITE( s%out, * ) '    unpermuting the column indices'
            END IF

            CALL SORT_inplace_invert( prob%n, prob%X_status )
            DO k = 1, prob%A%ne
               prob%A%col( k ) = prob%X_status( prob%A%col( k ) )
            END DO
            CALL SORT_inplace_invert( prob%n, prob%X_status )

            IF ( s%level >= CRAZY ) WRITE( s%out, * )                          &
               '     A_col =', prob%A%col( :prob%A%ne )

            IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   A restored'

!-------------------------------------------------------------------------------
!
!         Reset permutations to recover variable and constraint status.
!
!             Note: the status vectors haven't been modified if the
!                   the problem is infeasible or empty.
!
!-------------------------------------------------------------------------------

!           Permutations on the variables

            DO j = 1, prob%n
               IF ( prob%X_status( j ) > s%n_in_prob ) THEN
                  prob%X_status( j ) = INACTIVE
               ELSE IF ( prob%X_status( j ) > s%n_active ) THEN
                  prob%X_status( j ) = ELIMINATED
               ELSE
                  prob%X_status( j ) = ACTIVE
               END IF
            END DO

!           Permutations on the constraints

            IF ( prob%m > 0 ) THEN
               DO i = 1, prob%m
                  IF ( prob%C_status( i ) > s%m_in_prob ) THEN
                     prob%C_status( j ) = INACTIVE
                  ELSE IF ( prob%C_status( i ) > s%m_active ) THEN
                     prob%C_status( i ) = ELIMINATED
                  ELSE
                     prob%C_status( i ) = ACTIVE
                  END IF
               END DO
            END IF

         END IF

!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------
      END IF ! End of the permutation unscrambling: the rest of the
             ! processing does not depend on the permutations being set
             ! or not, and therefore also applies to infeasible or empty
             ! problems.
!-------------------------------------------------------------------------------
!-------------------------------------------------------------------------------

      IF ( s%level >= DEBUG ) CALL PRESOLVE_write_full_prob( prob, control, s )

!-------------------------------------------------------------------------------
!
!     If variables have been fixed in the last minute, unfix them now.
!
!-------------------------------------------------------------------------------

!     Loop over the (linked) list of variables that were fixed at
!     the last minute.

      IF ( s%lfx_f /= END_OF_LIST ) THEN
         DO jj = 1, prob%n
            j = s%lfx_f

!           Unfix x( j)

            xval = prob%X( j )
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    unfixing (last minute) x(', j, ') =', xval
            CALL PRESOLVE_do_unfix_x( j, xval, get( X_VALS), get( G_VALS ),    &
                                      get( F_VAL ),get( Z_VALS ),get( C_BNDS ) )

!           Find the associated dual variable, if requested.

            IF ( get( Z_VALS ) ) THEN
               prob%Z( j ) = PRESOLVE_compute_zj( j )
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    setting z(', j, ') =', prob%Z( j )
            END IF

!           Find the next variable fixed at the last minute.

            s%lfx_f       = s%h_perm( j )
            s%h_perm( j ) = 0
            IF ( END_OF_LIST == s%lfx_f ) EXIT
         END DO
      END IF

!-------------------------------------------------------------------------------
!
!                       Loop on the history (backwards)
!
!-------------------------------------------------------------------------------

!     Initialize the counters of restored transformations for the
!     saved and memory categories, respectively.

      s%rts = 0
      s%rtm = 0

!     Initialize the index of the transformations in the complete sequence

      ll = s%tt + 1

!     Start the historical loop.

      IF ( s%level >= TRACE .AND. s%tt > 0 ) WRITE( s%out, * )                 &
         ' ===  starting historical loop'
                                               !-------------------------------!
      DO                                       !      The historical loop      !
                                               !-------------------------------!
!        Print the current transformation counts.

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
                 '    ts =', s%ts, 'rts =', s%rts, 'rtm =', s%rtm

!        Get the next set of transformations (of size at most max_tm),
!        either from memory or from the saved files.
!
!        If no transformations were saved but there are transformations
!        in memory , undo these first.

         IF ( s%ts == 0 .AND. s%tm > s%rtm ) THEN
            s%rtm = s%tm

!        If there remains transformations stored in a save file,
!        undo them by slices of max_tm, starting from the end.

         ELSE IF ( s%ts > s%rts ) THEN
            llast  = s%ts - s%rts
            lfirst = MAX( 1, llast - s%max_tm + 1 )
            s%tm   = llast - lfirst + 1

!           Open the history file, if needed.

            CALL PRESOLVE_open_h( 'OLD    ', control, inform, s )
            IF ( inform%status /= OK ) RETURN

!           Read the tm transformations.

            IF ( s%level >= DEBUG )                                            &
               WRITE( s%out,* ) '    reading transformations', llast, 'to',    &
                    lfirst, 'from file ', control%transf_file_name
            CALL PRESOLVE_read_h( lfirst, llast, control%transf_file_name )
            IF ( inform%status /= OK ) RETURN

            s%rts = s%rts + s%tm
            IF ( s%rts == s%ts ) THEN
               SELECT CASE ( control%transf_file_status )
               CASE ( KEEP )
                  CLOSE( control%transf_file_nbr, STATUS = 'KEEP' )
               CASE ( DELETE )
                  CLOSE( control%transf_file_nbr, STATUS = 'DELETE' )
               END SELECT
            END IF

!        No further transformation exists, exit from the historical loop.

         ELSE
            EXIT
         END IF

!        Now loop on the current slice of transformations to undo.

         DO l = s%tm, 1, -1

!           Keep track of the index of the transformation to undo in the
!           original complete sequence.

            ll = ll - 1

!           Recompute dependencies, if necessary

            IF ( ll == next ) CALL PRESOLVE_check_dependencies( ll, next, get )

!           Check that optimality is maintained, if debugging

            IF ( s%level >= DEBUG) THEN
               CALL PRESOLVE_compute_q( prob )
               WRITE( s%out, * ) '    objective function value     = ', prob%q
               IF ( s%level >= CRAZY ) THEN
                  CALL PRESOLVE_check_optimality( .TRUE. )
               ELSE
                  CALL PRESOLVE_check_optimality( .FALSE. )
               END IF
            END IF

!           Now consider the various transformation types, and undo the l-th
!           transformation according to its type.

            SELECT CASE ( s%hist_type( l ) )

!           ------------------------
!           Row transformations
!           ------------------------

!           -------------------------------------------------------------------
            CASE ( A_ROWS_COMBINED ) ! linear combination of rows of A
                                     ! to make it sparser
!           -------------------------------------------------------------------

               IF ( get( A_VALS ) ) THEN

                  ie = s%hist_j( l )   ! the index of the modified row
                  rr = s%hist_r( l )   ! the linear combination coefficient

!                 Restore the entries of the modified row.

!                 Find the index of the pivot row (e) and the column index
!                 of the pivot (col).

                  k = s%hist_i ( l ) ! the position of the pivot element
                  e = s%A_row( k )
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                    '  [', ll, '] uncombining c(', e, ') and c(', ie,          &
                    ') with', rr
                  col = prob%A%col( k )

!                 Loop over the pivot row to remember the
!                 positions of its nonzeros in w_n.

                  s%w_n = 0
                  ic    = e
                  DO
                     DO k1 = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                        j  = prob%A%col( k1 )
                        IF ( prob%X_status( j ) <= ELIMINATED .OR. &
                             prob%A%val( k1 )   == ZERO       .OR. &
                             j == col                              ) CYCLE
                        s%w_n( j ) = k1
                     END DO
                     ic = s%conc( ic )
                     IF ( ic == END_OF_LIST ) EXIT
                   END DO

!                 Now uncombine the rows

                  ic = ie
                  DO
                     DO k1 = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                        j  = prob%A%col( k1 )
                        IF ( prob%X_status( j ) <= ELIMINATED .OR. &
                             j == col                              ) CYCLE
                        k2 = s%w_n( j )
                        IF ( k2 == 0 ) CYCLE
                        a = prob%A%val( k2 )
                        IF ( a == ZERO ) CYCLE
                        prob%A%val( k1 ) = prob%A%val( k1 ) - rr * a
                     END DO
                     ic = s%conc( ic )
                     IF ( ic == END_OF_LIST ) EXIT
                  END DO

               END IF

!              Restore the values of the constraints.

               prob%C( ie ) = prob%C( ie ) - rr * prob%C( e )
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    updating c(', ie, ') to', prob%C( ie )

!              Restore the multipliers, if required

               IF ( control%get_y ) THEN
                  prob%Y( e) = prob%Y( e ) + rr * prob%Y( ie )
                 IF ( s%level >= DEBUG ) WRITE( s%out, * )                     &
                     '    updating y(', ie, ') to', prob%Y( ie )
               END IF

!           -------------------------------------------------------------------
            CASE ( A_ROWS_MERGED ) ! row merging from linear doubleton
!           -------------------------------------------------------------------

               ke  = s%hist_i( l ) ! the position of the pivot in A
               ko  = s%hist_j( l ) ! the position of the other doubleton element
               aij = s%hist_r( l ) ! the value of the pivot

               IF ( get( A_VALS ) .OR. get( G_VALS ) .OR. &
                    get( Y_VALS ) .OR. get( F_VAL )  .OR. get( X_VALS ) ) THEN

!                 Find the row and column indices of the doubleton entries
!                 from their positions.

                  io = s%A_row( ko )
                  ie = s%A_row( ke )
                  j  = prob%A%col( ke )

!                 Get the ratios for updating the matrix A and the gradient

                  prob%A%val( ke ) = aij
                  r  = prob%A%val( ko ) / aij
                  rg = prob%G( j )      / aij

!                 Unperform the pivoting (elimination of x_j from its value in
!                 constraint ie) on the pivot and non-pivot rows, as well as
!                 on the gradient.

                  IF ( get( A_VALS ) .OR. get( G_VALS ) ) THEN

!                    Loop over the non-pivot row to remember the
!                    positions of its nonzeros in w_n.  At the end
!                    of the non-pivot row, separate it from the pivot
!                    row (containing the fills).
!                    (Note that we do not exclude zero entries in the
!                    non-pivot row because they might be created by
!                    cancellation during pivoting.  But this creates the
!                    problem that zero entries corresponding to removed
!                    entries in concatenated rows are not exluded either,
!                    although they should.  This is solved by only
!                    retaining the first occurrence of a column in the
!                    non-pivot row, as such entries always occur in the
!                    concatenated part, which is always *after* the
!                    original part of the non-pivot row.).

                     s%w_n = 0
                     ic  = io
                     DO ii = 1, prob%m
                        DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                           jo = prob%A%col( k )
                           IF ( prob%X_status( jo )  <= ELIMINATED ) CYCLE

!                          Only consider the first occurence of column jo

                           IF ( s%w_n( jo ) == 0 ) s%w_n( jo ) = k

                        END DO
                        k1 = s%conc( ic )

!                       End of the non-pivot row: separate it from the fills.

                        IF ( k1 == ie ) THEN
                           s%conc( ic ) = END_OF_LIST
                           EXIT

!                       Not yet the end of the non-pivot row: continue.

                        ELSE
                           IF ( k1 == END_OF_LIST ) EXIT
                           ic = k1
                        END IF
                     END DO

!                    Now loop on the pivot row, undoing the pivoting.

                     ic = ie
                     DO
                        DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                           je = prob%A%col( k )
                           IF ( prob%X_status( je ) <= ELIMINATED ) CYCLE
                           a = prob%A%val( k )
                           IF ( a == ZERO ) CYCLE

!                          See if variable je also appears in the
!                          nonpivot row (in which case w_n( je ) contains
!                          the position of A(io,je)).

                           kk = s%w_n( je )

!                          If variable je also appears in the non-pivot row,
!                          only its coefficient in the non-pivot row must be
!                          updated, the coefficient in the pivot row being
!                          restored from its zeroing.

                           IF ( get( A_VALS ) ) THEN
                              IF ( kk > 0 ) THEN
                                 prob%A%val( kk ) = prob%A%val( kk ) + r * a

!                             If variable je does not appear in the non-pivot
!                             row, the entry is thus a fill: it must be
!                             rescaled and the column structure updated.

                              ELSE
                                 prob%A%val( k ) = - a / r
                                 s%A_row( k )    = ie
                              END IF
                           END IF

!                          Restore the gradient.

                           IF ( get( G_VALS ) )                                &
                               prob%G( je ) = prob%G( je ) + rg*prob%A%val(k)

                        END DO
                        ic = s%conc( ic )
                        IF ( ic == END_OF_LIST ) EXIT
                     END DO
                  END IF

!                 Update the objective function value.

                  cval = HALF * ( prob%C_l( ie ) + prob%C_u( ie ) )
                  IF ( get( F_VAL ) ) prob%f = prob%f - rg * cval

!                 Update the bounds on the io-th constraint.

                  a = r * cval
                  IF ( prob%C_l( io ) > s%M_INFINITY )                         &
                     prob%C_l( io ) = prob%C_l( io ) + a
                  IF ( prob%C_u( io ) < s%P_INFINITY )                         &
                     prob%C_u( io ) = prob%C_u( io ) + a

!                 Obtain the value of x_j from the ie-th row.

                  IF ( get( X_VALS ) ) THEN
                     IF ( s%level >= ACTION ) WRITE( s%out, * )                &
                        '  [', ll, '] substituting x(', j, ') from c(', ie,    &
                        ') =', cval, '[d]'
                     CALL PRESOLVE_substitute_x_from_c( j, ie, aij, cval )
                     IF ( inform%status /= OK ) RETURN
                  ELSE
                     prob%X( j ) = s%hist_r( l )
                     IF ( s%level >= ACTION ) WRITE( s%out, * )                &
                        '  [', ll, '] not substituting x(', j, ') from c(', ie,&
                        ') =', cval, '[d]'
                  END IF

               END IF

!              Reactivate x(j), while maintaining the problem dimensions.

               s%A_row_s( io )    = s%A_row_s( io ) + 1
               prob%X_status( j ) = ACTIVE
               s%n_active         = s%n_active + 1

!              Compute the multiplier associated with the pivot row.

               IF ( get( Y_VALS ) ) THEN
                  prob%Y( ie ) = rg - r * prob%Y( io )
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting y(', ie, ') =', prob%Y( ie )
               END IF

!              Zero z(j)

               IF ( get( Z_VALS ) ) THEN
                  prob%Z( j ) = ZERO
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting z(', j, ') =', ZERO
               END IF

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_FL ) ! forcing row with its lower bound
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the forcing row

!              Compute the duals if requested.

               IF ( get( Z_VALS ) .OR. get( Y_VALS ) ) THEN
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    computing y(', ii, ') from forcing list [lower]'

!                 Note that hist_j( l ) points to the first variable occuring
!                 in the list of variables that are fixed with row i.

                  CALL PRESOLVE_duals_from_forcing( ii, s%hist_j( l ), LOWER,  &
                                                   get( Y_VALS ), get( Z_VALS) )
                  IF ( inform%status /= OK ) RETURN
               END IF

!              Reactivate the constraint.

               prob%C_status( ii ) = ACTIVE
               prob%C( ii ) = prob%C_l( ii )
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') =', prob%C( ii )

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_FU ) ! forcing row with its upper bound
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the forcing row

!              Compute the duals if requested.

               IF ( get( Z_VALS ) .OR. get( Y_VALS ) ) THEN
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    computing y(', ii, ') from forcing list [upper]'

!                 Note that s%hist_j( l ) points to the first variable occuring
!                 in the list of variables that are fixed with row i.

                  CALL PRESOLVE_duals_from_forcing( ii, s%hist_j( l ), UPPER,  &
                                                  get( Y_VALS ), get( Z_VALS ) )
                  IF ( inform%status /= OK ) RETURN
               END IF

!              Reactivate the constraint.

               prob%C_status( ii ) = ACTIVE
               prob%C( ii ) = prob%C_u( ii )
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') =', prob%C( ii )

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_YV ) ! row deactivation with known multiplier
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the deactivated row
               rr = s%hist_r( l ) ! the value of the associated multiplier

!              Reactivate the constraint with a value that depends on the
!              associated multiplier.

               prob%C_status( ii ) = ACTIVE

!              Obtain the value of the constraint

               prob%C( ii ) = PRESOLVE_c_from_y( ii, rr )
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') =', prob%C( ii ),     &
                  '[y known]'

!              Assign the multiplier, if requested.

               IF ( get( Y_VALS ) ) THEN
                  prob%Y( ii ) = rr
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting y(', ii, ') =', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_YZ_LOW ) ! row deactivation (lower singleton row)
!           -------------------------------------------------------------------

               ii = s%hist_i( l )       ! the index of the singleton row
               kk = s%hist_j( l )       ! the position of the singleton element
               rr = s%hist_r( l )       ! the value of the singleton element

!              Reactivate the constraint.

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') [lower singleton row]'
               prob%C_status( ii ) = ACTIVE

!              Obtain the column index of the singleton element from its
!              position.

               jj = prob%A%col( kk )

!              Obtain the value of the dual quantites, if requested.

               IF ( prob%Z( jj ) >= ZERO ) THEN
                  rr = prob%Z( jj ) / rr
                  IF ( get( Y_VALS ) ) THEN
                     prob%Y( ii ) = rr
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    setting y(', ii, ') =', rr
                  END IF
                  IF ( get( Z_VALS ) ) THEN
                     CALL PRESOLVE_correct_z_for_dy( ii , rr )
                     prob%Z( jj ) = ZERO
                  END IF
               ELSE
                  rr = ZERO
               END IF

!              Obtain the value of the reactivated constraint from the
!              multiplier.

               prob%C( ii ) = PRESOLVE_c_from_y( ii, rr )
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    setting c(', ii, ') =', prob%C( ii )

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_YZ_UP ) ! row deactivation (upper singleton row)
!           -------------------------------------------------------------------

               ii = s%hist_i( l )    ! the index of the singleton row
               kk = s%hist_j( l )    ! the position of the singleton element
               rr = s%hist_r( l )    ! the value of the singleton element

!              Reactivate the constraint.

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') [upper singleton row]'
               prob%C_status( ii ) = ACTIVE

!              Obtain the column index of the singleton element from its
!              position.

               jj = prob%A%col( kk )

!              Obtain the dual quantities, if requested

               IF ( prob%Z( jj ) <= ZERO ) THEN
                  rr = prob%Z( jj ) / rr
                  IF ( get( Y_VALS ) ) THEN
                     prob%Y( ii ) = rr
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    setting y(', ii, ') =', rr
                  END IF
                  IF ( get( Z_VALS ) ) THEN
                     CALL PRESOLVE_correct_z_for_dy( ii , rr )
                     prob%Z( jj ) = ZERO
                  END IF
               ELSE
                  rr = ZERO
               END IF

!              Obtain the value of the reactivated constraint from the
!              multiplier.

               prob%C( ii ) = PRESOLVE_c_from_y( ii, rr )
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    setting c(', ii, ') =', prob%C( ii )

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_YZ_EQU ) ! row deactivation (equ. singleton row)
!           -------------------------------------------------------------------

               ii = s%hist_i( l )     ! the index of the singleton row
               kk = s%hist_j( l )     ! the position of the singleton element
               rr = s%hist_r( l )     ! the value of the singleton element

!              Reactivate the constraint.

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') [equ. singleton row]'
               prob%C_status( ii ) = ACTIVE

!              Obtain the column index of the singleton element from its
!              position.

               jj = prob%A%col( kk )

!              Obtain the the dual quantities, if requested.

               rr = prob%Z( jj ) / rr
               IF ( get( Y_VALS ) ) THEN
                  prob%Y( ii ) = rr
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting y(', ii, ') =', rr
               END IF
               IF ( get( Z_VALS ) ) THEN
                  CALL PRESOLVE_correct_z_for_dy( ii , rr )
                  prob%Z( jj ) = ZERO
               END IF

!              Obtain the value of the reactivated constraint.

               prob%C( ii ) = HALF * ( prob%C_l( ii ) + prob%C_u( ii ) )
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    setting c(', ii, ') =', prob%C( ii )

!           -------------------------------------------------------------------
            CASE ( C_REMOVED_GY ) ! row deactivation (column doubleton)
!           -------------------------------------------------------------------

               ii = s%hist_i( l )   ! the index of the deactivated row

!              Reactivate the constraint.

               prob%C_status( ii ) = ACTIVE

!              Obtain its value.

               prob%C( ii ) = HALF * ( prob%C_l( ii ) + prob%C_u( ii ) )
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] reactivating c(', ii, ') =', prob%C( ii ),     &
                  ' [doubleton column]'

!              note that the computation of y(ii) (from g and y) is postposed
!              (see A_ROWS_MERGED)

!           -------------------------------------------------------------------
            CASE ( C_LOWER_UPDATED ) ! lhs update
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the constraint
               rr = s%hist_r( l ) ! the value of c_l to be restored
               IF ( get( C_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating c_l(', ii, ') to', rr
                  prob%C_l( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating c_l(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( C_UPPER_UPDATED ) ! rhs updated
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the constraint
               rr = s%hist_r( l ) ! the value of c_u to be restored
               IF ( get( C_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating c_u(', ii, ') to', rr
                  prob%C_u( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating c_u(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( Y_LOWER_UPDATED ) ! lower bound on multiplier tightened
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the constraint
               rr = s%hist_r( l ) ! the value of y_l to be restored
               IF ( control%get_y_bounds ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating y_l(', ii, ') to', rr
                  prob%Y_l( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating y_l(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( Y_UPPER_UPDATED ) ! upper bound on multiplier tightened
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the constraint
               rr = s%hist_r( l ) ! the value of y_u to be restored
               IF ( control%get_y_bounds ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating y_u(', ii, ') to', rr
                  prob%Y_u( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating y_u(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( Y_FIXED ) ! multiplier fixed
!           -------------------------------------------------------------------

               ii = s%hist_i( l )   ! the index of the constraint
               rr = s%hist_r( l )   ! the value of y to be imposed
               IF ( get( Y_VALS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] fixing y(', ii, ') to', rr
                  prob%Y( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not fixing y(', ii, ') to', rr
               END IF

!           ========================
!           Variable transformations
!           ========================

!           -------------------------------------------------------------------
            CASE ( X_LOWER_UPDATED ) ! tightening of a variable lower bound
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the variable
               rr = s%hist_r( l ) ! the value of x_l to be restored
               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_l(', ii, ') to', rr
                  prob%X_l( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_l(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( X_UPPER_UPDATED ) ! tightening of a variable upper bound
!           -------------------------------------------------------------------

               ii = s%hist_i( l ) ! the index of the variable
               rr = s%hist_r( l ) ! the value of x_u to be restored
               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_u(', ii, ') to', rr
                  prob%X_u( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_u(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( X_LOWER_UPDATED_P ) ! tightening of a variable lower bound
                                       ! in primal constraint analysis
!           -------------------------------------------------------------------

               jj = s%hist_i( l )      ! the index of the variable
               rr = s%hist_r( l )      ! the value of x_l to be restored

               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_l(', jj, ') to', rr, '(p)'
                     prob%X_l( jj ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_l(', jj, ') to', rr, '(p)'
               END IF

!              Update the dual quantities, if requested.

               IF ( get( Y_VALS ) .OR. get( Z_VALS ) ) THEN

!                 Obtain the index of the constraint from the position of
!                 the element from which the bound was derived.

                  kk  = s%hist_j( l )    ! the position of the element
                  ii  = s%A_row( kk )
                  aij = prob%A%val( kk )

!                 Obtain the dual variables.

                  IF ( prob%Z( jj ) > control%z_accuracy ) THEN
                     IF ( s%level >= ACTION ) THEN
                        WRITE( s%out, * )                                      &
                             '            and correcting y(', ii,              &
                             ') and associated duals'
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '            from z(', jj, ') =', prob%Z( jj ),     &
                           '[lower]'
                     END IF

!                    Obtain the multiplier.

                     rr = prob%Z( jj ) / aij
                     IF ( get( Y_VALS ) ) THEN
                        prob%Y( ii ) = prob%Y( ii ) + rr
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    setting y(', ii, ') =', prob%Y( ii )
                     END IF
                     IF ( get( Z_VALS ) ) THEN
                        CALL PRESOLVE_correct_z_for_dy( ii, rr )
                        prob%Z( jj ) = ZERO
                     END IF
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_UPPER_UPDATED_P ) ! tightening of a variable upper bound
!           -------------------------------------------------------------------

               jj = s%hist_i( l )      ! the index of the variable
               rr = s%hist_r( l )      ! the value of x_l to be restored
               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_u(', jj, ') to', rr, '(p)'
                  prob%X_u( jj ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_u(', jj, ') to', rr, '(p)'
               END IF

!              Update the dual quantities, if requested.

               IF ( get( Y_VALS ) .OR. get( Z_VALS ) ) THEN

!                 Obtain the index of the constraint from the position of
!                 the element from which the bound was derived.

                  kk  = s%hist_j( l )    ! the position of the element
                  ii  = s%A_row( kk )
                  aij = prob%A%val( kk )

!                 Obtain the dual variables.

                  IF ( prob%Z( jj ) < - control%z_accuracy ) THEN
                     IF ( s%level >= ACTION ) THEN
                        WRITE( s%out, * )                                      &
                             '            and correcting y(', ii,              &
                             ') and associated duals'
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '            from z(', jj, ') =', prob%Z( jj ),     &
                           '[upper]'
                     END IF

!                    Obtain the multiplier.

                     rr = prob%Z( jj ) / aij
                     IF ( get( Y_VALS ) ) THEN
                        prob%Y( ii ) = prob%Y( ii ) + rr
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    setting y(', ii, ') =', prob%Y( ii )
                     END IF
                     IF ( get( Z_VALS ) ) THEN
                        CALL PRESOLVE_correct_z_for_dy( ii, rr )
                        prob%Z( jj ) = ZERO
                     END IF
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_LOWER_UPDATED_D ) ! tightening of a variable lower bound
                                       ! in dual constraint analysis
!           -------------------------------------------------------------------

               jj = s%hist_i( l )      ! the index of the variable
               rr = s%hist_r( l )      ! the value of x_l to be restored

               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_l(', jj, ') to', rr
                     prob%X_l( jj ) = rr
               END IF

!              Obtain the dual quantities, if requested.

               IF ( get( Y_VALS ) .OR. get( Z_VALS ) ) THEN
                  IF ( PRESOLVE_is_pos( prob%Z( jj ) ) ) THEN
                     IF ( s%level >= ACTION ) WRITE( s%out, * )                &
                        '            and correcting associated duals from z(', &
                        jj, ') [lower]'
                     CALL PRESOLVE_correct_yz_for_z( jj, get( Y_VALS ),        &
                                                                get( Z_VALS ) )
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_UPPER_UPDATED_D ) ! tightening of a variable upper bound
                                       ! in dual constraint analysis
!           -------------------------------------------------------------------

               jj = s%hist_i( l )      ! the index of the variable
               rr = s%hist_r( l )      ! the value of x_u to be restored

               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_u(', jj, ') to', rr
                  prob%X_u( jj ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_u(', jj, ') to', rr
               END IF

!              Obtain the dual quantities, if requested.

               IF ( get( Y_VALS ) .OR. get( Z_VALS ) ) THEN
                  IF ( PRESOLVE_is_neg( prob%Z( jj ) ) ) THEN
                     IF ( s%level >= ACTION ) WRITE( s%out, * )                &
                        '            and correcting associated duals from z(', &
                        jj, ') [upper]'
                     CALL PRESOLVE_correct_yz_for_z( jj, get( Y_VALS ),        &
                                                                get( Z_VALS ) )
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_LOWER_UPDATED_S ) ! tightening of a variable lower bound
                                       ! in a doubleton shift
!           -------------------------------------------------------------------

               kk = s%hist_i( l )      ! the index of the variable
               rr = s%hist_r( l )      ! the value of x_l to be restored

               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_l(', kk, ') to', rr
                  prob%X_l( kk ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_l(', kk, ') to', rr
               END IF

!              Obtain the dual quantities, if requested.

               IF ( get( Y_VALS ) .OR. get( Z_VALS ) ) THEN

!                 Obtain the row and column indices of the doubleton element
!                 from its position.

                  ie = s%hist_j( l )     ! the position of the doubleton element
                  ii = s%A_row( ie )
                  jj = prob%A%col( ie )

!                 Compute the duals.

                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '            and unshifting the bounds between x(',       &
                     jj, ') and x(', kk, ') [lower]'
                  IF ( prob%Z( kk ) > control%z_accuracy ) THEN
                     CALL PRESOLVE_duals_from_shift( ii, jj, kk, get( Y_VALS ),&
                                                     get( Z_VALS ) )
                     IF ( inform%status /= OK ) RETURN
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_UPPER_UPDATED_S ) ! tightening of a variable upper bound
                                       ! in a doubleton shift
!           -------------------------------------------------------------------

               kk = s%hist_i( l )      ! the index of the variable
               rr = s%hist_r( l )      ! the value of x_u to be restored

               IF ( get( X_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating x_u(', kk, ') to', rr
                  prob%X_u( kk ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating x_u(', kk, ') to', rr
               END IF

!              Obtain the dual quantities, if requested.

               IF ( get( Y_VALS ) .OR. get( Z_VALS ) ) THEN

!                 Obtain the row and column indices of the doubleton entry
!                 from its position.

                  ie = s%hist_j( l )     ! the position of the doubleton entry
                  ii = s%A_row( ie )
                  jj = prob%A%col( ie )

!                 Compute the duals.

                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '            and unshifting the bounds between x(',       &
                     jj, ') and x(', kk, ') [upper]'
                  IF ( prob%Z( kk ) < - control%z_accuracy ) THEN
                     CALL PRESOLVE_duals_from_shift( ii, jj, kk, get( Y_VALS ),&
                                                     get( Z_VALS ) )
                     IF ( inform%status /= OK ) RETURN
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_REDUCTION )     ! reduced variable from constraint
!           -------------------------------------------------------------------

               jj = s%hist_j( l )    ! the reduced variable
               ii = s%hist_i( l )    ! the sign of the reduction

!              Note that ii = 0 if a (single) equality constraint is reduced
!              with x(jj) free.

!              Recompute the value of all redundant constraints
!              (without the contribution of variable jj)
!              because the values assigned by PRESOLVE_c_from_y are inadequate.

               k  = s%A_col_f( jj )
               IF ( END_OF_LIST /= k .AND. get( A_VALS ) ) THEN
                  DO ko = 1, prob%m
                     i  = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN
                        cval = ZERO
                        ic   = i
                        DO
                           DO ke = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                              kk = prob%A%col( ke )
                              IF ( prob%X_status( kk ) <= ELIMINATED ) CYCLE
                              aij = prob%A%val( ke )
                              IF ( aij == ZERO ) CYCLE
                              r = prob%X( kk )
                              IF ( r == ZERO ) CYCLE
                              cval = cval + aij * r
                           END DO
                           ic = s%conc( ic )
                           IF ( ic == END_OF_LIST ) EXIT
                        END DO
                        prob%C( i ) = cval
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    recomputed c(', i, ') =', cval
                     END IF
                     k = s%A_col_n( k )
                     if ( END_OF_LIST == k ) EXIT
                  END DO
               END IF

!              Determine the initial bound for the value of x(jj).

               IF ( ii > 0 ) THEN
                  alpha = prob%X_l( jj )
                  IF ( alpha > s%M_INFINITY ) THEN
                     xval = alpha + ONE
                  ELSE
                     xval = ZERO
                  END IF
               ELSE IF ( ii < 0 ) THEN
                  alpha = prob%X_u( jj )
                  IF ( alpha < s%P_INFINITY ) THEN
                     xval = alpha -  ONE
                  ELSE
                     xval = ZERO
                  END IF
               ELSE                ! Single equality constraint
                  xval = ZERO
               END IF
               IF ( s%level >= DEBUG  ) WRITE( s%out, * )                      &
                  '    initial x(', jj, ') =', xval, ' sign =', ii

!              Loop on all constraint involving variable jj to find its
!              maximum (if the reduction sign is positive) or minimum
!              (if the reduction sign is negative).

               k = s%A_col_f( jj )
               IF ( END_OF_LIST /= k .AND. get( A_VALS ) ) THEN
                  DO ko = 1, prob%m
                     i  = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN

!                       Get the value of the largest constraint default
!                       (excluding x(jj) since it has not yet been reactivated)

                        aij = prob%A%val( k )
                        IF ( .NOT. PRESOLVE_is_zero( aij ) ) THEN
                           IF ( ii > 0 )  THEN
                              IF ( aij > ZERO ) THEN
                                 cval = ( prob%C_l(i) - prob%C(i) + ONE ) / aij
                              ELSE
                                 cval = ( prob%C_u(i) - prob%C(i) - ONE ) / aij
                              END IF
                              xval = MAX( cval, xval )
                           ELSE IF ( ii < 0 ) THEN
                              IF ( aij > ZERO ) THEN
                                 cval = ( prob%C_u(i) - prob%C(i) - ONE ) / aij
                              ELSE
                                 cval = ( prob%C_l(i) - prob%C(i) + ONE ) / aij
                              END IF
                              xval = MIN( cval, xval )
                           ELSE                ! Single equality constraint
                              cval = HALF * ( prob%C_l( i ) + prob%C_u( i ) )
                              xval = ( cval - prob%C( i ) ) / aij
                           END IF
                           IF ( s%level >= DEBUG ) WRITE( s%out, * )           &
                              '    updating x(', jj, ') for c(', i, ') to', xval
                        END IF
                     END IF
                     k = s%A_col_n( k )
                     if ( k == END_OF_LIST ) EXIT
                  END DO
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    recomputed x(', jj, ') =', xval

!              Loop again on the constraints to adjust their value
!              for the reintroduced variable jj.

               k = s%A_col_f( jj )
               IF ( END_OF_LIST /= k .AND. get( A_VALS ) ) THEN
                  DO ko = 1, prob%m
                     i  = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN
                        prob%C( i ) = prob%C( i ) + prob%A%val( k ) * xval
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    readjusted c(', i, ') =', prob%C( i )
                     END IF
                     k = s%A_col_n( k )
                     if ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

!              Reactivate the reduced variable.

               prob%X( jj ) = xval
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                     '  [', ll, '] unreducing x(', jj, ') =', xval

               prob%X_status( jj ) = ACTIVE

!              Set the dual variable: it must be zero
!              because x( jj ) is inactive.

               IF ( get( Z_VALS ) ) THEN
                  prob%z( jj ) = ZERO
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                           '    setting z(', jj, ') = 0'
               END IF

!           -------------------------------------------------------------------
            CASE ( Z_LOWER_UPDATED ) ! lower bound on z tightened
!           -------------------------------------------------------------------

               ii = s%hist_i( l )   ! the index of the variable
               rr = s%hist_r( l )   ! the value of z_l to be restored
               IF ( get( Z_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating z_l(', ii, ') to', rr
                  prob%Z_l( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating z_l(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( Z_UPPER_UPDATED ) ! upper bound on z tightened
!           -------------------------------------------------------------------

               ii = s%hist_i( l )
               rr = s%hist_r( l )
               IF ( get( Z_BNDS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] downdating z_u(', ii, ') to', rr
                  prob%Z_u( ii ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not downdating z_u(', ii, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( Z_FIXED ) ! dual variable fixed
!           -------------------------------------------------------------------

               jj = s%hist_i( l )   ! the index of the variable
               rr = s%hist_r( l )   ! the value of z)u to be restored
               IF ( get( Z_VALS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] fixing z(', jj, ') to', rr
                  prob%Z( jj ) = rr
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting z(', jj, ') =', prob%Z( jj )
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not fixing z(', jj, ') to', rr
               END IF

!           -------------------------------------------------------------------
            CASE ( X_FIXED_DF ) ! fixed variable
!           -------------------------------------------------------------------

               jj   = s%hist_i( l )   ! the index of the variable
               xval = s%hist_r( l )   ! the value of the variable

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] fixing x(', jj, ') to', xval,                  &
                  '[z from dual feasibility]'

               CALL PRESOLVE_do_unfix_x( jj, xval, get( X_VALS), get( G_VALS ),&
                                      get( F_VAL ), get( Z_VALS), get( C_BNDS ))

!              Find the associated dual variable.

              IF ( get( Z_VALS ) ) THEN
                  prob%Z( jj ) = PRESOLVE_compute_zj( jj )
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting z(', jj, ') =', prob%Z( jj )
               ELSE
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    not setting z(', jj, ') =', prob%Z( jj )
               END IF

!           -------------------------------------------------------------------
            CASE ( X_FIXED_SL ) ! fixed variable
!           -------------------------------------------------------------------

               kk   = s%hist_i( l )   ! the index of the variable
               xval = s%hist_r( l )   ! the value of the variable

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] fixing x(', kk, ') to', xval,                  &
                  '[z from shift low]'

               CALL PRESOLVE_do_unfix_x( kk, xval, get( X_VALS), get( G_VALS ),&
                                      get( F_VAL ), get( Z_VALS), get( C_BNDS ))

!              Find the row and column indices of the entry used for the shift.

               k  = s%hist_j( l )      ! the position
               ii = s%A_row( k )
               jj = prob%A%col( k )

!              Find the associated dual variable.

               IF ( get( Z_VALS ) .OR. get( Y_VALS ) ) THEN
                  prob%Z( kk ) = PRESOLVE_compute_zj( kk )
                  IF ( prob%Z( kk ) >= control%z_accuracy ) THEN
                     CALL PRESOLVE_duals_from_shift( ii, jj, kk,               &
                                                  get( Y_VALS ), get( Z_VALS ) )
                  ELSE
                     IF ( get( Z_VALS ) ) THEN
                       prob%Z( kk ) = PRESOLVE_compute_zj( kk )
                       IF ( s%level >= DEBUG ) WRITE( s%out, * )               &
                          '    setting z(', kk , ') =', prob%Z( kk )
                     END IF
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_FIXED_SU ) ! fixed variable
!           -------------------------------------------------------------------

               kk   = s%hist_i( l )   ! the index of the variable
               xval = s%hist_r( l )   ! the value of the variable

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] fixing x(', kk, ') to', xval,                  &
                  '[z from shift up]'

               CALL PRESOLVE_do_unfix_x( kk, xval, get( X_VALS), get( G_VALS ),&
                                      get( F_VAL ), get( Z_VALS), get( C_BNDS ))

!              Find the row and column indices of the entry used for the shift.

               k  = s%hist_j( l )
               ii = s%A_row( k )
               jj = prob%A%col( k )

!              Find the associated dual variable.

               IF ( get( Z_VALS ) .OR. get( Y_VALS ) ) THEN
                  prob%Z( kk ) = PRESOLVE_compute_zj( kk )
                  IF ( prob%Z( kk ) <= - control%z_accuracy ) THEN
                      CALL PRESOLVE_duals_from_shift( ii, jj, kk,              &
                                                  get( Y_VALS ), get( Z_VALS ) )
                  ELSE
                     IF ( get( Z_VALS ) ) THEN
                        prob%Z( kk ) = PRESOLVE_compute_zj( kk )
                        IF ( s%level >= DEBUG ) WRITE( s%out, * )              &
                           '    setting z(', kk , ') =', prob%Z( kk )
                     END IF
                  END IF
               END IF

!           -------------------------------------------------------------------
            CASE ( X_FIXED_ZV ) ! fixed variable with known dual variable
!           -------------------------------------------------------------------

               jj   = s%hist_i( l )   ! the index of the variable
               xval = s%hist_r( l )   ! the value of the variable
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] fixing x(', jj, ') to', xval,                  &
                  '[ z(', jj, ') =', prob%Z( jj ), ']'

               CALL PRESOLVE_do_unfix_x( jj, xval, get( X_VALS), get( G_VALS ),&
                                      get( F_VAL ), get( Z_VALS), get( C_BNDS ))

!           -------------------------------------------------------------------
            CASE ( X_MERGE ) ! merging of two linear variables
!           -------------------------------------------------------------------

               jj    = s%hist_i( l )  ! the index of the deactivated variable
               j     = s%hist_j( l )  ! the index of the merged variable
               alpha = s%hist_r( l )  ! the merging coefficient

!              Reactivate the deactivated variable.

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] unmerging x(', jj, ') and x(', j,              &
                  ') with', alpha
               prob%X_status( jj ) = ACTIVE

               IF ( s%level >= DEBUG ) THEN
                  WRITE( s%out, * )  '    x(', jj, ') in [', prob%X_l( jj ),   &
                                     ',', prob%X_u( jj ), ']'
                  WRITE( s%out, * )  '    x(',  j, ') in [', prob%X_l(  j ),   &
                                     ',', prob%X_u(  j ), ']'
                  WRITE( s%out, * ) '    xval =', prob%X(j) ,                  &
                                       'zval  =', prob%Z(j)
               END IF

!              Choose values for the unmerged variables.

               IF ( get( X_VALS ) .OR. get( Z_VALS ) ) THEN
                  xval  = prob%X( j )

!                 Consider first the case where the merged variable is
!                 active. Note that, if this is the case, then each of
!                 the merged variable must be at one of its bounds.

                  zj = prob%Z( j )

!                 1) active lower bound

                  IF ( zj >= control%z_accuracy ) THEN
                     IF ( get( X_VALS ) ) THEN
                        prob%X( j ) = prob%X_l( j )
                        IF ( alpha > ZERO ) THEN
                           prob%X( jj ) = prob%X_l( jj )
                        ELSE
                           prob%X( jj ) = prob%X_u( jj )
                        END IF
                     END IF
                     IF ( get( Z_VALS ) ) prob%Z( jj ) = alpha * zj

!                 2) active upper bound

                  ELSE IF ( zj <= - control%z_accuracy ) THEN
                     IF ( get( X_VALS ) ) THEN
                        prob%X( j ) = prob%X_u( j )
                        IF ( alpha > ZERO ) THEN
                           prob%X( jj ) = prob%X_u( jj )
                        ELSE
                           prob%X( jj ) = prob%X_l( jj )
                        END IF
                     END IF
                     IF ( get( Z_VALS ) ) prob%Z( jj ) = alpha * zj

!                 Consider now the case where the merged variable is inactive.

                  ELSE

!                    Compute the values of the variables.

                     IF ( get( X_VALS ) ) THEN
                        CALL PRESOLVE_guess_x( j, prob%X( j ), prob, s )
                        prob%X( jj ) = ( xval - prob%X( j ) ) / alpha

!                       If x(jj) is unfeasible, project it on the feasible
!                       interval and adjust x(j).

                        IF ( prob%X( jj ) > prob%X_u( jj ) ) THEN
                           prob%X( jj ) = prob%X_u( jj )
                           prob%X( j )  = xval - alpha * prob%X( jj )
                        ELSE IF ( prob%X( jj ) < prob%X_l( jj ) ) THEN
                           prob%X( jj ) = prob%X_l( jj )
                           prob%X( j )  = xval - alpha * prob%X( jj )
                        END IF

!                       If x(j) is now unfeasible, we have a problem.

                        IF ( prob%X( j ) > prob%X_u( j ) .OR. &
                             prob%X( j ) < prob%X_l( j )      ) THEN
                           IF ( control%check_primal_feasibility == SEVERE )   &
                              THEN
                              inform%status = PRIMAL_INFEASIBLE
                              WRITE( inform%message( 1 ), * )                  &
                                   ' PRESOLVE INTERNAL ERROR:',                &
                                   ' unmerging x(', j, ') and x(', jj,         &
                                   ') impossible'
                              RETURN
                           ELSE
                              IF ( s%level >= TRACE ) WRITE( s%out, * )        &
                                 ' PRESOLVE WARNING: unmerging x(', j,         &
                                 ') and x(', jj, ') impossible'
                           END IF
                        END IF
                     END IF

!                    Compute the values of the dual variables.

                     IF ( get( Z_VALS ) ) THEN
                        IF ( PRESOLVE_is_zero( prob%X( j) - prob%X_l( j) ).OR. &
                             PRESOLVE_is_zero( prob%X( j) - prob%X_u( j) )    )&
                        THEN
                           prob%Z(  j ) = PRESOLVE_compute_zj(  j )
                        ELSE
                           prob%Z(  j ) = ZERO
                        END IF
                        IF ( PRESOLVE_is_zero( prob%X(jj) - prob%X_l(jj) ).OR. &
                             PRESOLVE_is_zero( prob%X(jj) - prob%X_u(jj) )    )&
                        THEN
                           prob%Z( jj ) = PRESOLVE_compute_zj( jj )
                        ELSE
                           prob%Z( jj ) = ZERO
                        END IF
                     END IF
                  END IF
               END IF
               IF ( s%level >= DEBUG ) THEN
                  WRITE( s%out, * ) '    unmerge results:'
                  WRITE( s%out, * ) '    x(', jj,') =', prob%X( jj ),          &
                        'z(', jj, ') =', prob%Z( jj )
                  WRITE( s%out, * ) '    x(', j,') =', prob%X( j ),            &
                        'z(', j, ') =', prob%Z( j )
               END IF

!           -------------------------------------------------------------------
            CASE ( X_IGNORED ) ! ignored variable
!           -------------------------------------------------------------------

               jj = s%hist_i( l )  ! the index of the ignored variable

               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                  '  [', ll, '] remembering x(', jj, ')'

!              Reactivate the ignored variable (with its saved value).

               prob%X( jj )        = s%hist_r( l )
               prob%X_status( jj ) = ACTIVE

!              Find the associated dual variable.

               IF ( get( Z_VALS ) )                                            &
                  prob%Z( jj ) = PRESOLVE_compute_zj( jj )

!           -------------------------------------------------------------------
            CASE ( X_SUBSTITUTED ) ! a variable is substituted from a constraint
!           -------------------------------------------------------------------

               j = s%hist_i( l )   ! the index of the variable to be substituted
               k = s%hist_j( l )   ! the position of its coefficient

!              Find the index of the constraint from which x(j) must be
!              substituted, using the position of its coefficient.

               i = s%A_row( k )

!              Find the value of the coefficient (aval), of the multiplier (rr)
!              and of the constraint (alpha).

               aval  = prob%A%val( k )
               rr    = prob%G( j ) / aval
               alpha = s%hist_r( l )

               IF ( get( X_VALS ) .OR. get( F_VAL ) .OR. &
                    get( Z_VALS ) .OR. get( G_VALS )     ) THEN

!                 substitute x(j)

                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] substituting x(', j, ') from c(', i,        &
                     ') =', alpha, '[s]'
                  CALL PRESOLVE_substitute_x_from_c( j, i, aval, alpha )
                  IF ( inform%status /= OK ) RETURN

                  IF ( rr /= ZERO ) THEN

!                    Restore the objective function and gradient.

                     IF ( get( F_VAL ) ) prob%f = prob%f - alpha * rr

                     IF ( get( G_VALS ) .OR. get( Z_VALS ) ) THEN

                        ic = i
                        DO
                           DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                              jj  = prob%A%col( k )
                              IF ( prob%X_status( jj ) <= ELIMINATED ) CYCLE
                              aij = prob%A%val( k )
                              IF ( aij == ZERO ) CYCLE
                              ccorr = aij * rr
                              prob%G( jj ) = prob%G( jj ) + ccorr
                           END DO
                           ic = s%conc( ic )
                           IF ( ic == END_OF_LIST ) EXIT
                        END DO

                     END IF
                  END IF
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not substituting x(', j, ') from c(', i,    &
                     ') =', alpha, '[s]'
               END IF

!              Reactivate the j-th variable.

               prob%X_status( j ) = ACTIVE

!              Find the associated dual variable.  Note that this is always
!              zero as only free variables are removed by substitution.

               IF ( get( Z_VALS ) ) THEN
                  prob%Z( j ) = ZERO
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting z(', j, ') =', ZERO
               END IF

!           -------------------------------------------------------------------
            CASE ( X_BOUNDS_TO_C ) ! transfer of the bounds of a variable to
                                   ! an equality constraint
!           -------------------------------------------------------------------

               kk  = s%hist_i( l )   ! the position of the transfer coefficient

!              Find the index of the transfer constraint from the position of
!              the transfer coefficient.

               ii = s%A_row( kk )

!              Find the index of the variable and the value of the
!              transfer coefficient

               jj  = ABS( s%hist_j( l ) )
               aij = prob%A%val( kk )
               IF ( s%level >= ACTION ) WRITE( s%out, * )                      &
                   '  [', ll, '] transfering the bounds from c(', ii,          &
                   ') back to x(', jj, ')'

!              Recovering the original (and equal) constraint bounds
!              is done automatically because of the UPDATE actions.

!              Determine the multiplier of the hidden equality, when the
!              latter is known (which is when s%hist_j( l ) = - j ).

               yval = prob%Y( ii )
               zval = - aij * yval
               IF ( s%hist_j( l ) < 0 ) THEN
                  IF ( get( Y_VALS ) ) THEN
                     prob%Y( ii ) = yval + s%hist_r( l )
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    setting y(', ii,' ) =', prob%Y( ii )
                  END IF
                  IF ( get( Z_VALS ) ) THEN
                     prob%Z( jj ) =  zval
                  END IF
               ELSE
                  prob%Z( jj ) = zval
                  IF ( s%level >= DEBUG ) WRITE( s%out, * )                    &
                     '    setting z(', jj,' ) =', prob%Z( jj )
               END IF

!           ------------------------
!           A transformations
!           ------------------------

!           -------------------------------------------------------------------
            CASE ( A_ENTRY_REMOVED ) ! removal of an entry of A for sparse
                                     ! storage
!           -------------------------------------------------------------------

               kk = s%hist_i( l )        !  the position of the removed entry
               ii = s%A_row( kk )        !  its row index
               rr = s%hist_r( l )        !  its value
               jj = prob%A%col( kk )
               IF ( get( A_VALS ) ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                   '  [', ll, '] reinserting A(', ii, ',', jj, ') =', rr, 'in A'
                  prob%A%val( kk ) = rr
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                   '  [', ll, '] not reinserting A(', ii, ',', jj, ') =', rr,  &
                   'in A'
               END IF

!           ------------------------
!           H transformations
!           ------------------------

!           -------------------------------------------------------------------
            CASE ( H_ELIMINATION ) ! simple elimination within a 2x2 of H
!           -------------------------------------------------------------------

               IF ( get( H_VALS ) .OR. get( X_VALS ) .OR. &
                    get( G_VALS ) .OR. get( F_VAL  )      ) THEN
                  ii   = s%hist_i( l )   ! the index of the eliminated variable
                  khpj = s%hist_j( l )   ! the position of H(p,j)

!                 Find the index of the eliminating variable

                  jj = prob%H%col( khpj )
                  IF ( jj == ii ) jj = s%H_row( khpj )
                  rr = s%hist_r( l )   ! the value of gjj/ hjj

                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                   '  [', ll, '] uneliminating x(', ii, ') using x(', jj, ')'

!                 Uneliminate x(ii)

                  hjj  = prob%H%val( prob%H%ptr( ii + 1 ) - 1 )
                  hpj  = prob%H%val( khpj )
                  gp   = prob%G( jj ) + rr * hpj
                  IF ( get( G_VALS ) ) prob%G( jj ) = gp
                  IF ( get( F_VAL  ) ) prob%f = prob%f + rr * gp
                  IF ( get( H_VALS ) ) THEN
                     khpp = prob%H%ptr( jj + 1 ) - 1
                     prob%H%val( khpp ) = prob%H%val( khpp ) + hpj * hpj / hjj
                  END IF
                  IF ( get( X_VALS ) ) THEN
                     prob%X( ii ) = - ( prob%G( ii ) + hpj * prob%X( jj ) ) /hjj
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    x(', ii, ') =', prob%X( ii )
                  END IF
                  prob%X_status( ii ) = ACTIVE
                  s%n_active = s%n_active + 1
               ELSE
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  [', ll, '] not uneliminating x(', ii,                  &
                     ') using x(', jj, ')'
               END IF

!-------------------------------------------------------------------------------

            END SELECT   ! End of select( transformation)

         END DO   ! End of historical loop (loop within the slices)

      END DO      ! End of historical loop (loop on the slices)

!-------------------------------------------------------------------------------

      IF ( s%level >= TRACE .AND. s%rts + s%rtm > 0 )                          &
         WRITE( s%out, * ) ' ===  end of the historical loop'

!-------------------------------------------------------------------------------
!
!     Recover the value of the constraints and the quadratic objective.
!     Zero the dual of inactive primal quantities and possibly check
!     the values of the dual variables, if requested by the user.
!
!-------------------------------------------------------------------------------

      CALL PRESOLVE_compute_q( prob )
      CALL PRESOLVE_compute_c( .TRUE., prob, s )
      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   values assigned to q and c'

!     Zero the inactive dual variables.

      IF ( control%get_z                       .AND. &
           control%get_x                       .AND. &
           control%get_x_bounds                .AND. &
           control%inactive_z == FORCE_TO_ZERO       ) THEN

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    zeroing the duals of inactive variables'

         DO j = 1, prob%n
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
            rr = prob%X( j )
            IF ( PRESOLVE_is_pos( rr - prob%X_l( j ) ) .AND.   &
                 PRESOLVE_is_pos( prob%X_u( j ) - rr )       ) &
               prob%Z( j ) = ZERO
         END DO
      END IF

!     Zero the inactive multipliers.

      IF ( control%get_y                       .AND. &
           control%get_c                       .AND. &
           control%get_c_bounds                .AND. &
           control%inactive_y == FORCE_TO_ZERO       ) THEN

         IF ( s%level >= DEBUG ) WRITE( s%out, * )    &
            '    zeroing the multipliers of inactive constraints'

         DO i = 1, prob%m
            IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
            rr = prob%C( i )
            IF ( PRESOLVE_is_pos( rr - prob%C_l( i ) ) .AND.   &
                 PRESOLVE_is_pos( prob%C_l( i ) - rr )       ) &
               prob%Y( i ) = ZERO
         END DO
      END IF

!-------------------------------------------------------------------------------

!             If debugging, verify the final optimality

!-------------------------------------------------------------------------------

      IF ( s%level >= DEBUG ) CALL PRESOLVE_check_optimality( .TRUE. )

!-------------------------------------------------------------------------------

!     Enforce the user's convention on the sign of the multipliers and dual
!     variables.

!-------------------------------------------------------------------------------

!     Dual variables

      IF ( control%z_sign /= POSITIVE .AND.            &
           ( control%get_z .OR. control%get_z_bounds ) ) THEN
         DO j = 1, prob%n
            IF ( control%get_z ) prob%Z( j ) = - prob%Z( j )
            IF ( control%get_z_bounds ) THEN
               zj            = - prob%Z_l( j )
               prob%Z_l( j ) = - prob%Z_u( j )
               prob%Z_u( j ) = zj
            END IF
         END DO
      END IF

!     Multipliers

      IF ( control%y_sign /= POSITIVE .AND.           &
           ( control%get_y .OR. control%get_y_bounds ) ) THEN
         DO i = 1, prob%m
            IF ( control%get_y ) prob%Y( i ) = - prob%Y( i )
            IF ( control%get_y_bounds ) THEN
               zj            = - prob%Y_l( i )
               prob%Y_l( i ) = - prob%Y_u( i )
               prob%Y_u( i ) = zj
            END IF
         END DO
      END IF

!-------------------------------------------------------------------------------
!
!           Recover sparse matrix coordinate storage if necessary
!
!-------------------------------------------------------------------------------

!     Restore A.

      IF ( ALLOCATED( prob%A%type ) ) DEALLOCATE( prob%A%type )
      CALL SMT_put( prob%A%type, 'SPARSE_BY_ROWS', smt_stat )
      IF ( prob%A%ne > 0 .AND. get( A_VALS ) ) THEN
         SELECT CASE ( s%a_type_original )
         CASE ( COORDINATE )
            CALL QPT_A_from_S_to_C( prob, inform%status )
            IF ( inform%status /= OK ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating a_row(',     &
                  prob%A%ne, ')'
               RETURN
            END IF
            IF( s%level >= DETAILS ) WRITE( s%out, * )                         &
              '   A restored to coordinate storage'
         CASE ( DENSE )
            CALL QPT_A_from_S_to_D( prob, inform%status )
            IF ( inform%status /= OK ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating a_row(',     &
                  prob%A%ne, ')'
               RETURN
            END IF
            IF( s%level >= DETAILS ) WRITE( s%out, * )                         &
              '   A restored to dense storage'
         END SELECT
      END IF

!     Restore H.

      IF ( prob%H%ne > 0 .AND. get( H_VALS ) ) THEN
         IF ( ALLOCATED( prob%H%type ) ) DEALLOCATE( prob%H%type )
         CALL SMT_put( prob%H%type, 'SPARSE_BY_ROWS', smt_stat )
         SELECT CASE ( s%h_type_original )
         CASE ( COORDINATE )
            CALL QPT_H_from_S_to_C( prob, inform%status )
            IF ( inform%status /= OK ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating h_row(',     &
                  prob%H%ne, ')'
               RETURN
            END IF
            IF( s%level >= DETAILS ) WRITE( s%out, * )                         &
              '   H restored to coordinate storage'
         CASE ( DENSE )
            CALL QPT_H_from_S_to_D( prob, inform%status )
            IF ( inform%status /= OK ) THEN
               inform%status = MEMORY_FULL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: no memory left for allocating h_row(',     &
                  prob%H%ne, ')'
               RETURN
            END IF
            IF( s%level >= DETAILS ) WRITE( s%out, * )                         &
              '   H restored to dense storage'
         CASE ( DIAGONAL )
            CALL QPT_H_from_S_to_Di( prob, inform%status )
            IF ( inform%status /= OK ) THEN
               inform%status = NOT_DIAGONAL
               WRITE( inform%message( 1 ), * )                                 &
                  ' PRESOLVE ERROR: the matrix is not diagonal'
               RETURN
            END IF
            IF( s%level >= DETAILS ) WRITE( s%out, * )                         &
              '   H restored to diagonal storage'
         END SELECT
      END IF

!     Set the stage indicator to allow cleanup.

      s%stage = RESTORED

      CALL PRESOLVE_say_goodbye( control, inform, s )

      RETURN

!     Formats

1000  FORMAT( '      A_val  =', 5 ( ES12.4, 1x ), /, ( 12x, 5 ( 1x,ES12.4) ) )
1001  FORMAT( '      H_val  =', 5 ( ES12.4, 1x ), /, ( 12x, 5 ( 1x,ES12.4) ) )
1003  FORMAT( '     ', i4, ' = ', i4, 1x, i4, 3x, a7, '   --> ', i4, 1x, i4 )
1007  FORMAT( '     ', i4, 4( 2x, ES12.4 ) )

!==============================================================================
!==============================================================================

   CONTAINS  ! The restore routines

!==============================================================================
!==============================================================================

      REAL ( KIND = wp ) FUNCTION PRESOLVE_c_from_y( i, yi )

!     Obtain the value of a constraint, given that of its associated multiplier.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the constraint.

      REAL ( KIND = wp ), INTENT( IN ) :: yi

!            the associated multiplier.

!     Programming: Ph. Toint, June 2001.

!==============================================================================
!
!     If the multiplier is positive, the lower bound must be active.

      IF ( yi > ZERO ) THEN

         PRESOLVE_c_from_y = prob%C_l( i )

!     If the multiplier is negative, the upper bound must be active.

      ELSE IF ( yi < ZERO ) THEN

         PRESOLVE_c_from_y = prob%C_u( i )

!     Otherwise, the value is arbitrary bewteen the lower and upper bounds.

      ELSE

!        If both bounds are finite, pick their midpoint.

         IF ( prob%C_l( i ) >= s%M_INFINITY .AND.  &
              prob%C_u( i ) <= s%P_INFINITY        ) THEN
            PRESOLVE_c_from_y = HALF * ( prob%C_l( i ) + prob%C_u( i ) )

!        Otherwise, pick the admissible value closest to zero.

         ELSE IF ( prob%C_u( i ) <= ZERO ) THEN
            PRESOLVE_c_from_y = prob%C_u( i )
         ELSE IF ( prob%C_l( i ) >= ZERO ) THEN
            PRESOLVE_c_from_y = prob%C_l( i )
         ELSE
            PRESOLVE_c_from_y = ZERO
         END IF
      END IF

      RETURN

      END FUNCTION PRESOLVE_c_from_y

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_solve_implications

!     Ensures that implications in needs are logically pursued.  In other
!     words, if output value i needs output value j, and output value j needs
!     output value k, this subroutine ensures that output value k is known
!     to be needed for restoring output value i. Note that several levels of
!     such implications are possible.

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER :: i, j, k
      LOGICAL :: none

      DO
         none = .TRUE.
         DO i = 1, 6
            DO j = 1, 6
               IF ( s%needs( i, j ) <= control%max_nbr_transforms ) THEN
                  DO k = 1, 10
                     IF ( k == i .OR. k == j ) CYCLE
                     IF ( s%needs( j, k ) < s%needs( i, k ) ) THEN
                        s%needs( i, k ) = s%needs( j, k )
                        none = .FALSE.
                     END IF
                  END DO
               END IF
            END DO
         END DO
         IF ( none ) RETURN
      END DO

      END SUBROUTINE PRESOLVE_solve_implications

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_check_dependencies( t, next, get )

!     Computes the index of the next transformation at which the dependencies
!     between restored quantities (as specified by the array s%needs(.,.)) must
!     be recomputed.

!     Arguments:

      INTEGER, INTENT( IN )  :: t

!              the index of the current transformation

      INTEGER, INTENT( OUT ) :: next

!              the index of the transformation at which the dependencies must
!              be recomputed

      LOGICAL, INTENT( INOUT ) :: get( 10 )

!              a vector whose j-th component is .TRUE. iff the corresponding
!              value must be restored for the forthcoming transformations

!     Programming: Ph. Toint, November 2000

! =============================================================================

!     Local variables

      INTEGER :: i, first
      LOGICAL :: getit

!     Set the index of the next transformation index where dependencies
!     must be recomputed.

      next = 0

!     Check the dependencies.
!
!     1) x values

      getit = control%get_x
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, X_VALS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( X_VALS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( X_VALS ) now set to .FALSE.'
         END IF
      END IF
      get( X_VALS ) = getit

!     2) x bounds

      getit = control%get_x_bounds
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, X_BNDS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( X_BNDS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( X_BNDS ) now set to .FALSE.'
         END IF
      END IF
      get( X_BNDS ) = getit

!     3) z values

      getit = control%get_z
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, Z_VALS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( Z_VALS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( Z_VALS ) now set to .FALSE.'
         END IF
      END IF
      get( Z_VALS ) = getit

!     4) z bounds

      getit = control%get_z_bounds
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, Z_BNDS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( Z_BNDS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( Z_BNDS ) now set to .FALSE.'
         END IF
      END IF
      get( Z_BNDS ) = getit

!     5) c bounds

      getit = control%get_c_bounds
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, C_BNDS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( C_BNDS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( C_BNDS ) now set to .FALSE.'
         END IF
         get( C_BNDS ) = getit
      ELSE
         get( C_BNDS ) = .TRUE.
      END IF

!     6) y values

      getit = control%get_y
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, Y_VALS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( Y_VALS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( Y_VALS ) now set to .FALSE.'
         END IF
      END IF
      get( Y_VALS ) = getit

!     7) f value

      getit = control%get_f
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, F_VAL )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( F_VAL ) .AND. .NOT. getit ) WRITE( s%out, * )            &
               '    get( F_VAL  ) now set to .FALSE.'
         END IF
      END IF
      get( F_VAL ) = getit

!     8) g values

      getit = control%get_g
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, G_VALS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( G_VALS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( G_VALS ) now set to .FALSE.'
         END IF
      END IF
      get( G_VALS ) = getit

!     9) H values

      getit = control%get_H
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, H_VALS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( H_VALS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( H_VALS ) now set to .FALSE.'
         END IF
      END IF
      get( H_VALS ) = getit

!     10) A values

      getit = control%get_A
      IF ( .NOT. getit ) THEN
         DO i = 1, 6
            first = s%needs( i, A_VALS )
            getit = getit .OR. ( first <= t )
            IF ( first < t ) next = MAX( next, first )
         END DO
         IF ( s%level >= DEBUG ) THEN
            IF ( get( A_VALS ) .AND. .NOT. getit ) WRITE( s%out, * )           &
               '    get( A_VALS ) now set to .FALSE.'
         END IF
      END IF
      get( A_VALS ) = getit

      RETURN

      END SUBROUTINE PRESOLVE_check_dependencies

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_duals_from_shift( i, j, k, get_y, get_z )

!     Recomputes the values of the dual variables z_j, z_k and multiplier y_i
!     when restoring a shift on the variables bounds using equality doubleton
!     i. The multiplier y_i is corrected iff get_y == .TRUE., and the dual
!     variables z_j and z_k are corrected iff get_z == .TRUE..

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!             the index of the considered equality doubleton constraint

      INTEGER, INTENT( IN ) :: j

!             the index of the variable in constraint i whose bounds have
!             been shifted to the other variable in the constraint

      INTEGER, INTENT( IN ) :: k

!             the index of the variable in constraint i whose bounds have
!             been modified to reflect those on the other variable of the
!             constraint

      LOGICAL, INTENT( IN ) :: get_y

!             .TRUE. iff one is interested in reconstructing the value of
!             the multiplier y( i )

      LOGICAL, INTENT( IN ) :: get_z

!             .TRUE. iff one is interested in reconstructing the values of
!             the dual variables z( j ) and z( k )

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: ie, ic, kk, jj
      REAL ( KIND = wp ) :: aij, aik, rr

      IF ( .NOT. ( get_y .OR. get_z ) ) RETURN

!     Search for the doubleton entries.

      ie = 0
      ic = i
lic:   DO
         DO kk = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
            jj  = prob%A%col( kk )
            IF ( prob%X_status( jj ) <= ELIMINATED ) CYCLE
            IF ( jj == j ) THEN
               aij = prob%A%val( kk )
               ie  = ie + 1
            ELSE IF ( jj ==  k ) THEN
               aik = prob%A%val( kk )
               ie  = ie + 1
            END IF
            IF ( ie == 2 ) EXIT lic
         END DO
         ic = s%conc( ic )
         IF ( ic == END_OF_LIST ) EXIT
      END DO lic
      IF ( ie < 2 ) THEN
         inform%status = NO_DOUBLETON_ROW
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE INTERNAL ERROR: doubleton row entries in row', i
         WRITE( inform%message( 2 ), * )                                       &
              '    (merged) not found (DUALS_FROM_SHIFT)'
         IF ( s%level >= DEBUG )                                               &
            CALL PRESOLVE_write_full_prob( prob, control, s )
         RETURN
      END IF

      rr = prob%Z( k ) / aik

!     Update the multiplier associated with the doubleton constraint.

      IF ( get_y ) THEN
         prob%Y( i ) = prob%Y( i ) + rr
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    setting y(', i, ') =', prob%Y( i )
      END IF

!     Update the values of the dual variables.

      IF ( get_z ) THEN
         prob%Z( j ) = - aij * rr
         prob%Z( k ) = ZERO
         IF ( s%level >= DEBUG ) THEN
            WRITE( s%out, * ) '    setting z(', j, ') =', prob%Z( j )
            WRITE( s%out, * ) '    setting z(', k, ') =', ZERO
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_duals_from_shift

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_duals_from_forcing( i, first, lowup, get_y, get_z )

!     Recomputes the values of the dual variables z and multiplier y_i
!     when restoring a forcing constraint i. The argument lowup must be equal
!     to either LOWER or UPPER and indicates whether the lower or upper bound
!     of constraint i is active. The multiplier y_i is corrected iff
!     get_y == .TRUE., and the dual variables corresponding to the
!     variables fixed by the elemination of the forcing constraint are
!     corrected iff get_z == .TRUE..

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!             the index of the forcing constraint

      INTEGER, INTENT( IN ) :: lowup

!             the active side of the forcing constraint (LOWER or UPPER)

      INTEGER, INTENT( IN ) :: first

!             the first in the list of variables fixed to their bounds
!             by the forcing constraint

      LOGICAL, INTENT( IN ) :: get_y

!             .TRUE. iff one is interested in reconstructing the value of
!             the multiplier y( i )

      LOGICAL, INTENT( IN ) :: get_z

!             .TRUE. iff one is interested in reconstructing the values of
!             the dual variables z( j ) and z( k )

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: k, kk, j, jj, ns
      REAL ( KIND = wp ) :: zj, aij, yi

!     Initialize the multiplier y(i).
!     We know y(i) >= 0 if lowup == LOWER and y(i) <= 0 if lowup == UPPER.

      yi = ZERO

!     Find the beginning of the first merged row in row i.
!     This position is stored in k, and the index of the corresponding
!     row in kk.

      kk = i
      k  = prob%A%ptr( kk )

!     Loop on the list of fixed variables, starting from the first variable
!     in the list (first).

      j = first
      DO

         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    correcting z(', j, ') and y(', i, ')'

!        Get the j-th component of g+Hx-ATy, which is equal to z(j)
!        Note that the value of z(j) at this point assumes that y(i)=0.

         zj = prob%Z( j )

!        Determine aij, the value of a(i,j).

         DO ns = 1, prob%n + 1
            IF ( prob%A%col( k ) == j ) THEN
               aij = prob%A%val( k )
               EXIT
            ELSE

!           If the end of row kk is reached, find row (in merged row i)
!           which follows row kk.

               IF ( k == prob%A%ptr( kk + 1 ) ) THEN
                  jj = s%conc( kk )
                  IF ( jj == END_OF_LIST ) THEN
                     kk = i
                  ELSE
                     kk = jj
                  END IF
                  k = prob%A%ptr( kk )
               ELSE
                  k = k + 1
               END IF
            END IF
         END DO
         IF ( ns == prob%n + 1 ) THEN
            inform%status = NO_DOUBLETON_ROW
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE INTERNAL ERROR: doubleton row entries in row', i
            WRITE( inform%message( 2 ), * )                                    &
                 '    (merged) not found (DUALS_FROM_FORCING)'
            IF ( s%level >= DEBUG )                                            &
                 CALL PRESOLVE_write_full_prob( prob, control, s )
            RETURN
         END IF

!        Now determine the minimal value of y(i) that makes it
!        consistent with the bound at which x(j) is fixed.

         SELECT CASE ( lowup )

         CASE ( LOWER )

!           If aij > 0, then x(j) must be fixed to its upper bound
!           for the constraint to be forcing (lower). Thus z(j) <= 0.
!           Determine the  minimal value of yi using the relation
!
!                       zj - aij * yi <= 0
!
!           If aij < 0, then x(j) must be fixed to its lower bound
!           for the constraint to be forcing (lower).  Thus z(j) >= 0.
!           Determine the  minimal value of yi using the relation
!
!                       zj - aij * yi >= 0
!
!           Both cases give the same bound.

            yi = MAX( yi, zj / aij )

         CASE ( UPPER )

!           If aij > 0, then x(j) must be fixed to its lower bound
!           for the constraint to be forcing (upper). Thus z(j) >= 0.
!           Determine the  minimal value of yi using the relation
!
!                       zj - aij * yi >= 0
!
!           If aij < 0, then x(j) must be fixed to its upper bound
!           for the constraint to be forcing (upper). Thus z(j) >= 0.
!           Determine the  minimal value of yi using the relation
!
!                       zj - aij * yi <= 0
!
!           Both cases give the same bound.

            yi = MIN( yi, zj / aij )

         END SELECT

!        Get the next fixed variable.

         j = s%h_perm( j )
         IF ( j == END_OF_LIST ) EXIT

      END DO

!     Set the multiplier.

      IF ( get_y .OR. get_z ) THEN
         prob%Y( i ) = yi
         IF ( s%level >= DEBUG ) WRITE( s%out, * )                             &
            '    setting y(', i, ') =', yi
      END IF

!     Now correct the dual variables for the final value of yi.

      IF ( get_z .AND. yi /= ZERO ) CALL PRESOLVE_correct_z_for_dy( i, yi )

      RETURN

      END SUBROUTINE PRESOLVE_duals_from_forcing

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_correct_z_for_dy( i, dyi )

!     Update the values of the dual variables to reflect a change in the
!     value of y(i) of size dyi. In other words, the operation
!                       z  <---  z - a_i^T * dy_i
!     is performed, where a_i is the i-th row of A.

!     Arguments:

      INTEGER, INTENT( IN ) :: i

!            the index of the considered constraint

      REAL ( KIND = wp ), INTENT( IN ) :: dyi

!            the increment in the value of the i-th multiplier

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER           :: k, j, ic
      REAL( KIND = wp ) :: aval

      ic = i
      DO
         DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
            j = prob%A%col( k )
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
            aval = prob%A%val( k )
            IF ( aval == ZERO ) CYCLE
            prob%Z( j ) = prob%Z( j ) - aval * dyi
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
                '    setting z(', j, ') =', prob%Z( j )
         END DO
         ic = s%conc( ic )
         IF ( ic == END_OF_LIST ) EXIT
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_correct_z_for_dy

!==============================================================================
!==============================================================================

      REAL( KIND = wp ) FUNCTION                                               &
          PRESOLVE_max_dual_correction( low, val, upp, yz, dyz )

!     see how much of the correction dyz to yz, the dual associated to the
!     primal constraint low <= val <= upp  can be imposed while preserving
!     complementarity.  The value returned is between 0 and 1.

!     Arguments:

      REAL( KIND = wp ), INTENT( IN ) :: low

!                the lower bound of the considered primal constraint

      REAL( KIND = wp ), INTENT( IN ) :: val

!                the actual value of the considered primal constraint

      REAL( KIND = wp ), INTENT( IN ) :: upp

!                the upper bound of the considered primal constraint

      REAL( KIND = wp ), INTENT( IN ) :: yz

!                the current value of the associated dual (y or z)

      REAL( KIND = wp ), INTENT( IN ) :: dyz

!                the proposed correction to the associated dual

!     Programming: Ph. Toint, April 2001.

!==============================================================================

      REAL( KIND = wp ) :: alpha

      alpha = ONE

!     Examine lower bound.

      IF ( ABS( val - low ) < control%c_accuracy ) THEN
         IF ( dyz <= ZERO ) THEN
            IF ( yz <= ZERO ) THEN
               alpha = ZERO
            ELSE
               alpha = MIN( alpha, ABS( dyz / yz ) )
            END IF
         END IF
      END IF

!     Examine upper bound.

      IF ( ABS( val - upp ) < control%c_accuracy ) THEN
         IF ( dyz >= ZERO ) THEN
            IF ( yz >= ZERO ) THEN
               alpha = ZERO
             ELSE
               alpha = MIN( alpha, ABS( dyz / yz ) )
             END IF
          END IF
       END IF

       PRESOLVE_max_dual_correction = alpha

       RETURN

       END FUNCTION PRESOLVE_max_dual_correction

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_correct_yz_for_z( j, get_y, get_z )

!     Update the values of the dual variables and multipliers to reflect the
!     fact that z(j) must be set to zero without altering dual feasibility.
!     This occurs when x(j) is at a bound derived from the analysis of a
!     dual constraint (which one is irrelevant).
!     The method used is to loop on the active constraints i with a nonzero
!     in column j whose active variables are all at a bound which is
!     compatible with the required change in z resulting itself from the
!     required change in y(i) that is necessary for setting z(j) = 0.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!            the index of the variable whose dual must be zeroed.  It is
!            assumed that prob%Z( j ) /= 0.

      LOGICAL, INTENT( IN ) :: get_y

!             .TRUE. iff one is interested in reconstructing the value of
!             the multiplier y( i )

      LOGICAL, INTENT( IN ) :: get_z

!             .TRUE. iff one is interested in reconstructing the values of
!             the dual variables z( j ) and z( k )

!     Programming: Ph. Toint, April 2001

!==============================================================================

!     Local variables

      INTEGER           :: i, k, ii, ic, kk, i2, p
      REAL( KIND = wp ) :: dyi, aij, aip, czp, czj, alpha, beta

!     If there are no linear constraints, then the fact that z(j) is zero
!     is impossible

      If ( s%m_active <= 0 ) THEN
         inform%status = Z_CANNOT_BE_ZEROED
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: z(', j,') cannot be zeroed while preserving'
         WRITE( inform%message( 2 ), * )                                       &
              '    dual feasibility'
         RETURN
      END IF

      czj = prob%Z( j )

!     Now loop over the j-th column to find an active constraint whose
!     multiplier may be modified to reflect the zeroing of z(j).

      k = s%A_col_f( j )
      IF ( END_OF_LIST /= k ) THEN
sli:     DO ii = 1, prob%m
            IF ( ii > 1 ) THEN
               k = s%A_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END IF
            i = s%A_row( k )
            IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
            aij = prob%A%val ( k )
            IF ( aij == ZERO ) CYCLE
            dyi = czj / aij
            alpha = PRESOLVE_max_dual_correction( prob%C_l( i ), prob%C( i ),  &
                                                  prob%C_u( i ), prob%Y( i ),  &
                                                  dyi )

            IF ( alpha <= ZERO ) THEN
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c(', i, ') unsuitable for correction',                  &
                  ' ( alpha =', alpha, ')'
               CYCLE sli
            END IF

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
               '    y(', i, ') can be corrected for', alpha

!           Now test if the dual variables of the variables occuring in
!           constraint i may be modified for dyi while maintaing
!           complementarity.

            ic    = i
            DO i2 = 1, prob%m
               DO kk = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
                  p  = prob%A%col( kk )
                  IF ( prob%X_status( p ) <= ELIMINATED .OR. p == j ) CYCLE
                  aip = prob%A%val( kk )
                  IF ( aip == ZERO ) CYCLE
                  czp = - aip * dyi
                  beta = PRESOLVE_max_dual_correction( prob%X_l( p ),          &
                                                       prob%X( p ),            &
                                                       prob%X_u( p ),          &
                                                       prob%Z( p ),  czp )
                  alpha = MIN( alpha, beta )
                  IF ( alpha <= ZERO ) THEN
                     IF ( s%level >= DEBUG ) WRITE( s%out, * )                 &
                        '    c(', i, ') unsuitable for correction',            &
                       ' ( alpha (', p, ') =', alpha, ')'
                     CYCLE sli
                  END IF
                END DO
               ic = s%conc( ic )
               IF ( ic == END_OF_LIST ) EXIT
            END DO

            IF ( s%level >= DEBUG ) WRITE( s%out, * )                          &
                  '    the duals of c(', i, ') can be corrected for', alpha

!           Modify the multiplier and associated duals.

            dyi         = alpha * dyi
            IF ( get_y ) prob%Y( i ) = prob%Y( i ) + dyi
            IF ( get_z ) CALL PRESOLVE_correct_z_for_dy( i, dyi )

!           Update the remaining correction.

            czj = czj - dyi * aij

!           If the correction is complete, zero z(j) and return satisfied

            IF ( PRESOLVE_is_zero( czj ) ) THEN
               IF ( get_z ) prob%Z( j ) = ZERO
               RETURN
            END IF
            IF ( s%level >= DEBUG ) WRITE( s%out, * )                            &
               '    remaining correction =', czj
         END DO sli
      END IF

!     We have a problem if we reach this point!

      IF ( control%check_dual_feasibility == SEVERE ) THEN
         inform%status = Z_CANNOT_BE_ZEROED
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: z(', j, ') cannot be zeroed while preserving'
         WRITE( inform%message( 2 ), * )                                       &
              '    dual feasibility'
         RETURN
      ELSE IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * )                                                     &
              '    PRESOLVE WARNING: z(', j,                                   &
              ') cannot be zeroed while preserving dual feasibility'
      END IF

      END SUBROUTINE PRESOLVE_correct_yz_for_z

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_substitute_x_from_c( j, i, aij, cval )

!     Substitutes x(j) from the equation c(i) = cval, returning the value of
!     x(j) in xval.  This assumes x(j) is still inactive.

!     Arguments:

      INTEGER, INTENT( IN ) :: j

!            the index of the variable whose value must be substituted

      INTEGER, INTENT( IN ) :: i

!            the index of the constraint to use for the substitution

      REAL ( KIND = wp ), INTENT( IN ) :: aij

!            the value of A(i,j) ( must be nonzero! )

      REAL ( KIND = wp ), INTENT( INOUT ) :: cval

!            on entry: the value to which c(i) must be equalled to extract
!                      x(j)
!
!     Programming: Ph. Toint, March 2001.

!==============================================================================

!     Local variables

      INTEGER :: k, ic, jj
      REAL ( KIND = wp ) :: aijj, ccorr, xval, xjj, xlj, xuj

      xval = cval
      xlj  = prob%X_l( j )
      xuj  = prob%X_u( j )

      ic = i
      DO
         DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
            jj  = prob%A%col( k )
            IF ( prob%X_status( jj ) <= ELIMINATED ) CYCLE
            aijj = prob%A%val( k )
            IF ( aijj == ZERO ) CYCLE
            xjj = prob%X( jj )
            IF ( xjj == ZERO ) CYCLE
            xval = xval - aijj * xjj
         END DO
         ic = s%conc( ic )
         IF ( ic == END_OF_LIST ) EXIT
      END DO

      prob%X( j ) = xval / aij

      IF ( s%level >= DEBUG ) WRITE( s%out, * )                                &
         '    setting x(', j, ') =', prob%X( j )

!     Attempt to correct an out-of-bounds value of x(j) due to
!     inaccurate primal feasibility in the solver's output by reprojecting
!     x(j) onto its feasible interval.

      IF ( prob%X( j ) < xlj .AND. control%final_x_bounds == TIGHTEST  ) THEN
         ccorr = aij * ( xlj - prob%X( j ) )
         IF ( ABS( ccorr ) <= control%c_accuracy ) THEN
            prob%X( j ) = xlj
            prob%C( i ) = cval + ccorr
            IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    corrected x(', j,   &
               ') =', prob%X( j ), 'and c(', i, ') =', cval
         ELSE
            inform%status = X_OUT_OF_BOUNDS
         END IF
      END IF

      IF ( prob%X( j ) > xuj .AND. control%final_x_bounds == TIGHTEST  ) THEN
         ccorr = aij * ( xuj - prob%X( j ) )
         IF ( ABS( ccorr ) <= control%c_accuracy ) THEN
            prob%X( j ) = xuj
            prob%C( i ) = cval + ccorr
            IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    corrected x(', j,   &
               ') =', prob%X( j ), 'and c(', i, ') =', cval
         ELSE
           inform%status = X_OUT_OF_BOUNDS
         END IF
      END IF

      IF ( inform%status == X_OUT_OF_BOUNDS ) THEN
         WRITE( inform%message( 1 ), * )' PRESOLVE ERROR: substituted value for'
         WRITE( inform%message( 2 ), * )'     x(', j, ') =',prob%X( j )
         WRITE( inform%message( 3 ), * ) '    not in [', xlj, ',', xuj, ']'
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_substitute_x_from_c

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_read_h( first, last, filename )

!===============================================================================

!     Reads the transformations from positions first to last in the memory
!     buffer from filename, starting after record last_written.

!     Arguments:

      INTEGER, INTENT( IN ) :: first

!              the index (in memory) of the first transformation to write

      INTEGER, INTENT( IN ) ::  last

!              the index (in memory) of the last transformation to write

      CHARACTER( LEN = 30 ), INTENT( IN ) :: filename

!              the name of the file from which the transformations have to
!              be read
!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: l, k, iostat

!     Save the current content of the history from position first to last.

      k = 0
      DO l = first + 1, last + 1
         k = k + 1
         READ ( control%transf_file_nbr, REC = l, IOSTAT = iostat )            &
            s%hist_type( k ), s%hist_i( k ), s%hist_j ( k ), s%hist_r( k )
         IF ( iostat /= 0 ) THEN
            inform%status = CORRUPTED_SAVE_FILE
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: file ', filename
            WRITE( inform%message( 2 ), * )                                    &
                 '    ( unit =', control%transf_file_nbr, ') has been corrupted'
            RETURN
         END IF
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_read_h

!===============================================================================
!===============================================================================

      REAL ( KIND = wp ) FUNCTION PRESOLVE_compute_zj( j )

!     Compute the j-th component of z = g + H*x - A^T*y, where j is the
!     index of an active variable.

!     Argument:

      INTEGER, INTENT( IN ) :: j

!              the index of the dual variable to be computed from
!              the dual feasibility equation

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: i, ii, jo, k, kk
      REAL ( KIND = wp ) :: tmp, val, yi

!     Initialize with the gradient component.

      tmp = prob%G( j )

!     Add the j-th component of Hx

      IF ( prob%H%ne > 0 ) THEN

         DO k = prob%H%ptr( j ), prob%H%ptr( j + 1 ) - 1
            kk = prob%H%col( k )
            IF ( prob%X_status( kk ) <= ELIMINATED ) CYCLE
            val = prob%H%val( k )
            IF ( val /= ZERO ) tmp = tmp + val * prob%X( kk )
         END DO
         k = s%H_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO jo = 1, prob%n
               kk = s%H_row( k )
               IF ( prob%X_status( kk ) > ELIMINATED ) THEN
                  val = prob%H%val( k )
                  IF ( val /= ZERO ) tmp = tmp + val * prob%X( kk )
               END IF
               k = s%H_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END DO
         END IF

      END IF

!     Substract the component of A^T y.

      IF ( prob%A%ne > 0 ) THEN

         k = s%A_col_f( j )
         IF ( END_OF_LIST /= k ) THEN
            DO ii = 1, prob%m
               i  = s%A_row( k )
               IF ( prob%C_status( i ) > ELIMINATED ) THEN
                  yi = prob%Y( i )
                  IF ( yi /= ZERO ) THEN
                     val = prob%A%val( k )
                     IF ( val /= ZERO ) tmp = tmp - yi * val
                  END IF
               END IF
               k = s%A_col_n( k )
               IF ( k == END_OF_LIST ) EXIT
            END DO
        END IF

      END IF

!     Set dual value and return.

      PRESOLVE_compute_zj = tmp

      RETURN

      END FUNCTION PRESOLVE_compute_zj

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_compute_z( z )

!     Compute the full vector of dual variables from the equation
!
!              (z-sign) * z = g + H*x - (y-sign) * A^T*y,
!
!     assuming that the problem is in reduced form (as opposed to the
!     extensive form that includes the eliminated variables and constraints).

!     Argument:

      REAL( KIND = wp ), DIMENSION( prob%n ), INTENT( OUT ) :: z

!            the vector of computed dual variables

!     Programming: Ph. Toint, Spring 2001

!==============================================================================

!     Local variables

      INTEGER            :: i, j, k
      REAL ( KIND = wp ) :: val, yi

!     Initialize z to the gradient.

      z = prob%G( :prob%n )

!     Add  Hx.

      IF ( prob%H%ne > 0 ) THEN

         DO i = 1, prob%n
            DO k = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
               j = prob%H%col( k )
               val = prob%H%val( k )
               IF ( val == ZERO ) CYCLE
               z( i ) = z( i ) + val * prob%X( j )
               IF ( j /= i  ) z( j ) = z( j ) + val * prob%X( i )
            END DO
         END DO

      END IF

!     Substract A^T y

      IF ( prob%A%ne > 0 ) THEN

         DO i = 1, prob%m
            yi = prob%Y( i )
            IF ( yi  /= ZERO ) THEN
               DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                  val = prob%A%val( k )
                  IF ( val == ZERO ) CYCLE
                  j = prob%A%col( k )
                  SELECT CASE ( control%y_sign )
                  CASE ( POSITIVE )
                     z( j ) = z( j ) - yi * val
                  CASE ( NEGATIVE )
                     z( j ) = z( j ) + yi * val
                  END SELECT
               END DO
            END IF
         END DO

      END IF

      IF ( control%z_sign == NEGATIVE ) z = - z

      RETURN

      END SUBROUTINE PRESOLVE_compute_z

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_do_unfix_x( jj, xval, getx, getg, getf, getz, getcb )

!     Performs the operations that are necessary to unfix variable jj.

!     Arguments:

      INTEGER, INTENT( IN ) :: jj

!            the variable to unfix

      REAL ( KIND = wp ) , INTENT( IN ) :: xval

!            the value to be given to variable jj

      LOGICAL, INTENT( IN ) :: getx

!            .TRUE. iff the values of X are requested

      LOGICAL, INTENT( IN ) :: getg

!            .TRUE. iff the values of G are requested

      LOGICAL, INTENT( IN ) :: getf

!            .TRUE. iff the value of F is requested

      LOGICAL, INTENT( IN ) :: getz

!            .TRUE. iff the values of Z are requested

      LOGICAL, INTENT( IN ) :: getcb

!            .TRUE. iff the values of the bounds on C are requested


!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER            :: k, hj, ii, i
      REAL ( KIND = wp ) :: ccorr, aval, f_add

      IF ( getx ) prob%X( jj ) = xval

      IF ( getg .OR. getf .OR. getz .OR. getcb  ) THEN

         IF ( xval /= ZERO ) THEN

!           Update the gradient for the change in the second order term.

            IF ( ( getg .OR. getz ) .AND. prob%H%ne > 0 ) THEN

!              Use the structure of the fixed columns of H to
!              update the relevant quantities.  Note that the loop
!              avoids the diagonal element since column jj is not yet
!              reactivated.

               DO k = prob%H%ptr( jj ), prob%H%ptr( jj + 1 ) - 1
                  hj = prob%H%col( k )
                  IF ( prob%X_status( hj ) <= ELIMINATED ) CYCLE
                  ccorr = xval * prob%H%val( k )
                  prob%G( hj ) = prob%G( hj ) - ccorr
               END DO
               k = s%H_col_f( jj )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = 1, prob%n
                     hj = s%H_row( k )
                     IF ( prob%X_status( hj ) > ELIMINATED ) THEN
                        ccorr = xval * prob%H%val( k )
                        prob%G( hj ) = prob%G( hj ) - ccorr
                     END IF
                     k = s%H_col_n( k )
                     IF ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF

!           Update the objective function for the change in the first
!           order term and Hessian diagonal.

            IF ( getf ) THEN
               prob%f = prob%f - xval * prob%G( jj )
               IF ( prob%H%ne > 0 ) THEN
                  k = prob%H%ptr( jj + 1 ) - 1
                  IF ( k >= prob%H%ptr( jj ) ) THEN
                     IF ( prob%H%col( k ) == jj ) THEN
                        f_add = prob%H%val( k )
                        IF ( f_add /= ZERO ) THEN
                           prob%f = prob%f - HALF * f_add * xval**2
                        END IF
                     END IF
                  END IF
               END IF
            END IF

!           Update the constraint bounds.

            IF ( getcb .AND. prob%A%ne > 0 ) THEN

!              Use the structure of the fixed column of A
!              to update the relevant quantities

               k = s%A_col_f( jj )
               IF ( END_OF_LIST /= k ) THEN
                  DO ii = 1, prob%m
                     i  = s%A_row( k )
                     IF ( prob%C_status( i ) > ELIMINATED ) THEN
                        aval = prob%A%val( k )
                        IF ( aval /= ZERO ) THEN
                           ccorr  = xval * aval
                           prob%C_l( i ) = prob%C_l( i ) + ccorr
                           prob%C_u( i ) = prob%C_u( i ) + ccorr
                        END IF
                     END IF
                     k = s%A_col_n( k )
                     IF ( k == END_OF_LIST ) EXIT
                  END DO
               END IF

            END IF
         END IF
      END IF

!     Update the status of variable jj.

      prob%X_status( jj ) = ACTIVE

      RETURN

      END SUBROUTINE PRESOLVE_do_unfix_x

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_check_optimality( detailed )

!     Argument:

      LOGICAL, INTENT( IN ) :: detailed

!         .TRUE. iff the details of the components must be output,
!          additionnally to the summary
!
!     Programming: Ph. Toint, November 2000.

!===============================================================================

!     Local variables

      INTEGER           :: jo, ko, zo, i, j
      REAL( KIND = wp ) :: mxinf, mxslc, mxzinf, xval, rr, zj, cval, ccorr

      jo = 0
      ko = 0
      zo = 0

!     Verify primal feasibility.

      IF ( control%get_c_bounds .AND. control%get_x_bounds ) THEN
         mxinf = ZERO
         mxslc = ZERO
         IF ( detailed ) WRITE( s%out, 1004 )

!        Accumulate the maximal primal feasibility error with respect to the
!        bound constraints and the maximal complementarity violation.

         DO j = 1, prob%n
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE

!           Compute the feasibility error.

            xval = prob%X( j )
            ccorr = MAX( ZERO, prob%X_l( j ) - xval, xval - prob%X_u( j ) )

!           Remember the error, if this is the largest so far.

            IF ( ccorr > mxinf ) THEN
               jo    = j
               mxinf = ccorr
            END IF

!           Compute the slackness error.

            rr  = prob%Z( j )
            IF ( rr > control%z_accuracy ) THEN
               rr = rr * ( xval - prob%X_l( j ) )
            ELSE IF ( rr < - control%z_accuracy ) THEN
               rr = rr * ( xval - prob%X_u( j ) )
            ELSE
               rr = ZERO
            END IF

!           Remember it, if this is the largest so far.

            IF ( rr > mxslc ) THEN
               ko    = j
               mxslc = rr
            END IF
            IF ( detailed ) WRITE( s%out, 1007 ) j, xval, ccorr, rr
         END DO

!        Accumulate the maximal primal feasibility error with respect to the
!        linear constraints and the maximal complementarity violation.

         CALL PRESOLVE_compute_c( .TRUE., prob, s )
         IF ( detailed ) WRITE( s%out, 1005 )
         DO i = 1, prob%m
            IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
            cval  = prob%C( i )
            ccorr = MAX( ZERO, prob%C_l( i ) - cval, cval - prob%C_u( i ) )
            IF ( ccorr > mxinf ) THEN
               jo    = -i
               mxinf = ccorr
            END IF
            rr  = prob%Y( i )
            IF ( rr > control%c_accuracy ) THEN
               rr = rr * ( cval - prob%C_l( i ) )
            ELSE IF ( rr < - control%c_accuracy ) THEN
               rr = rr * ( cval - prob%C_u( i ) )
            ELSE
               rr = ZERO
            END IF
            IF ( rr > mxslc ) THEN
               ko    = -i
               mxslc = rr
            END IF
            IF ( detailed ) WRITE( s%out, 1007 ) i, cval, ccorr, rr
         END DO
      ELSE
         WRITE( s%out, * ) '    unable to verify primal feasibility unless '
         WRITE( s%out, * )                                                     &
                 '       control%get_c_bounds = control%get_x_bounds = .TRUE.'
      END IF

!     Verify the values of the dual variables.

      IF ( control%get_z .AND. control%get_z_bounds ) THEN
         err    = ZERO
         mxzinf = ZERO
         IF ( detailed ) WRITE( s%out, 1006 )
         DO j = 1, prob%n
            IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE

!           Compute the ideal dual variable and the error.

            rr    = PRESOLVE_compute_zj( j )
            zj    = prob%Z( j )
            ccorr = ABS( rr - zj )

!           Remember it, if this is the largest so far.

            IF ( ccorr > err ) THEN
               zo  = j
               err = ccorr
            END IF
            IF ( detailed ) WRITE( s%out, 1007 ) j, zj, rr, ccorr

!           Compute the dual feasibility error.

            ccorr = MAX( ZERO, prob%Z_l( j ) - zj, zj - prob%Z_u( j ) )

!           Remember the error, if this is the largest so far.

            IF ( ccorr > mxzinf ) THEN
               zo     = -j
               mxzinf = ccorr
            END IF
         END DO
      ELSE
         WRITE( s%out, * ) '    unable to verify dual feasibility unless '
         WRITE( s%out, * )                                                     &
           '       control%get_z and control%get_z_bounds are .TRUE.'
      END IF

!     Write the summary of the verifications just performed.

      IF ( jo > 0 ) THEN
         WRITE( s%out, * ) '    maximum primal infeasibility =', mxinf,        &
              '--> x(', jo, ')'
      ELSE IF ( jo < 0 ) THEN
         WRITE( s%out, * ) '    maximum primal infeasibility =', mxinf,        &
              '--> c(', - jo, ')'
      END IF
      IF ( ko > 0 ) THEN
         WRITE( s%out, * ) '    maximum slackness violation  =', mxslc,        &
              '--> x(', ko, ')'
      ELSE IF ( ko < 0 ) THEN
         WRITE( s%out, * ) '    maximum slackness violation  =', mxslc,        &
              '--> c(', - ko, ')'
      END IF
      IF ( zo > 0 ) THEN
         WRITE( s%out, * ) '    maximum dual infeasibility   =', err,          &
              '--> z(', zo, ')'
      ELSE IF ( zo < 0 ) THEN
         WRITE( s%out, * ) '    maximum dual infeasibility  =', mxzinf,        &
              '--> z(', - zo, ')'
      END IF

      RETURN

1004  FORMAT( '     ---> verifying bound constraints' /,                       &
              '        j     restored    infeasibility   slackness' )
1005  FORMAT( '     ---> verifying linear constraints' /,                      &
              '        i     restored    infeasibility   slackness' )
1006  FORMAT( '     ---> verifying dual variables' /,                          &
              '        j     restored      verified     difference ' )
1007  FORMAT( '     ', i4, 4( 2x, ES12.4 ) )

      END SUBROUTINE PRESOLVE_check_optimality

!===============================================================================
!===============================================================================

      END SUBROUTINE PRESOLVE_restore

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

      SUBROUTINE PRESOLVE_terminate( control, inform, s )

!     Cleans up the workspace and the map space.

!     Arguments:

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

      IF ( s%level >= TRACE ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *         GALAHAD PRESOLVE for QPs         *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' *            workspace cleanup             *'
         WRITE( s%out, * ) ' *                                          *'
         WRITE( s%out, * ) ' ********************************************'
         WRITE( s%out, * ) ' '
      END IF

!    Check there is something to clean up.

      inform%status = OK
      IF ( s%stage == VOID ) THEN
         inform%status = STRUCTURE_NOT_SET
         WRITE( inform%message( 1 ), * )                                       &
              ' PRESOLVE ERROR: the problem structure has not been set up'
         RETURN
      END IF

!     Clean up the various workspace arrays.

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                    &
         ' cleaning up PRESOLVE temporaries'

      IF ( ALLOCATED( s%hist_i    ) ) DEALLOCATE( s%hist_i    )
      IF ( ALLOCATED( s%hist_j    ) ) DEALLOCATE( s%hist_j    )
      IF ( ALLOCATED( s%hist_type ) ) DEALLOCATE( s%hist_type )
      IF ( ALLOCATED( s%hist_r    ) ) DEALLOCATE( s%hist_r    )
      IF ( ALLOCATED( s%ztmp      ) ) DEALLOCATE( s%ztmp    )
      IF ( ALLOCATED( s%x_l2      ) ) DEALLOCATE( s%x_l2    )
      IF ( ALLOCATED( s%x_u2      ) ) DEALLOCATE( s%x_u2    )
      IF ( ALLOCATED( s%z_l2      ) ) DEALLOCATE( s%z_l2    )
      IF ( ALLOCATED( s%z_u2      ) ) DEALLOCATE( s%z_u2    )
      IF ( ALLOCATED( s%c_l2      ) ) DEALLOCATE( s%c_l2    )
      IF ( ALLOCATED( s%c_u2      ) ) DEALLOCATE( s%c_u2    )
      IF ( ALLOCATED( s%y_l2      ) ) DEALLOCATE( s%y_l2    )
      IF ( ALLOCATED( s%y_u2      ) ) DEALLOCATE( s%y_u2    )
      IF ( ALLOCATED( s%A_col_s   ) ) DEALLOCATE( s%A_col_s )
      IF ( ALLOCATED( s%A_row_s   ) ) DEALLOCATE( s%A_row_s )
      IF ( ALLOCATED( s%H_str     ) ) DEALLOCATE( s%H_str   )
      IF ( ALLOCATED( s%w_n       ) ) DEALLOCATE( s%w_n     )
      IF ( ALLOCATED( s%w_mn      ) ) DEALLOCATE( s%w_mn    )
      IF ( ALLOCATED( s%w_m       ) ) DEALLOCATE( s%w_m     )
      IF ( ALLOCATED( s%a_perm    ) ) DEALLOCATE( s%a_perm  )
      IF ( ALLOCATED( s%h_perm    ) ) DEALLOCATE( s%h_perm  )
      IF ( ALLOCATED( s%conc      ) ) DEALLOCATE( s%conc    )
      IF ( s%a_type == SPARSE ) THEN
         IF ( ALLOCATED( s%A_col_f ) ) DEALLOCATE( s%A_col_f )
         IF ( ALLOCATED( s%A_col_n ) ) DEALLOCATE( s%A_col_n )
         IF ( ALLOCATED( s%A_row   ) ) DEALLOCATE( s%A_row   )
      END IF
      IF ( s%h_type == SPARSE ) THEN
         IF ( ALLOCATED( s%H_col_f ) ) DEALLOCATE( s%H_col_f )
         IF ( ALLOCATED( s%H_col_n ) ) DEALLOCATE( s%H_col_n )
         IF ( ALLOCATED( s%H_row   ) ) DEALLOCATE( s%H_row   )
      END IF

      IF ( s%level >= DETAILS ) WRITE( s%out, * )                              &
         '   temporaries cleanup successful'

      s%stage = VOID

      CALL PRESOLVE_say_goodbye( control, inform, s )

      RETURN

      END SUBROUTINE PRESOLVE_terminate

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

      SUBROUTINE PRESOLVE_guess_x( j, xj, prob, s  )

!     Guess a value xj for the j-th variable, given its associated bounds

!     Arguments:

      INTEGER, INTENT( IN )  :: j

!            the index of the variable whose value should be guessed

      REAL ( KIND = wp ), INTENT( INOUT ) :: xj

!            the value guessed for the j-th variable

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      REAL ( KIND = wp ) :: xl, xu

      xl = prob%X_l( j )
      xu = prob%X_u( j )

!     Check the variables.

      IF ( xl <= s%M_INFINITY ) THEN
         IF ( xu >= s%P_INFINITY ) THEN                     ! free variable
!            xj = ONE
         ELSE
            xj = MIN( xj, xu )                              ! upper bounded
         END IF
      ELSE
         IF ( xu < s%P_INFINITY ) THEN                      ! fixed variable
            IF ( xu == xl ) THEN
               xj = xl
            ELSE                                            ! range bounded
!               xj = HALF * ( xl + xu )
                xj = MIN( MAX( xl, xj ), xu )
            END IF
         ELSE
             xj = MAX( xj, xl )                             ! lower bounded
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_guess_x

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_guess_z( j, zj, prob, s )

!     Guess a value zj for the j-th dual variable, given its associated bounds.

!     Arguments:

      INTEGER, INTENT( IN )  :: j

!             the index of the dual variable whose value should be guessed

      REAL ( KIND = wp ), INTENT( OUT ) :: zj

!             the valure guessed for the j-th dual variable

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      REAL ( KIND = wp ) :: xl, xu, zl, zu

      xl = prob%X_l( j )
      xu = prob%X_u( j )

!     Check the variables.

      IF ( xl <= s%M_INFINITY ) THEN
         IF ( xu >= s%P_INFINITY ) THEN                      ! free variable
            zj = ZERO
         ELSE
            zj = - ONE
         END IF
      ELSE
         IF ( xu < s%P_INFINITY ) THEN                       ! fixed variable
            IF ( xu == xl ) THEN
               zl = prob%Z_l( j )
               zu = prob%Z_u( j )
               IF ( zl >= ZERO ) THEN
                  zj = MIN( zl + ONE, HALF * ( zl + zu ) )
               ELSE IF ( zu <= ZERO ) THEN
                  zj = MAX( zu - ONE, HALF * ( zl + zu ) )
               ELSE
                  zj = ZERO
               END IF
            ELSE                                             ! range bounded
               zj = ZERO
            END IF
         ELSE
            zj = ONE
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_guess_z

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_guess_y( prob, s )

!     Guess values for the vector of Lagrange multipliers, given their bounds.

!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: i
      REAL ( KIND = wp ) :: cl, cu, yl, yu

!     Loop on all active rows

      DO i = 1, prob%m
         cl = prob%C_l( i )
         cu = prob%C_u( i )
         IF ( prob%C_status( i ) <= ELIMINATED ) CYCLE
         IF ( cl < s%M_INFINITY ) THEN
            prob%Y( i ) = -ONE
         ELSE
            IF ( cu < s%P_INFINITY ) THEN
               IF ( cu == cl ) THEN                        ! equality constraint
                  yl = prob%Y_l( i )
                  yu = prob%Y_u( i )
                  IF ( yl >= ZERO ) THEN
                     prob%Y( i ) = MIN( yl + ONE, HALF * ( yl + yu ) )
                  ELSE IF ( yu <= ZERO ) THEN
                     prob%Y( i ) = MAX( yu - ONE, HALF * ( yl + yu ) )
                  ELSE
                     prob%Y( i ) = ZERO
                  END IF
               ELSE                                        ! range
                  prob%Y( i ) = ZERO
               END IF
            ELSE                                           ! lower bounded
               prob%Y( i ) = ONE
            END IF
         END IF

      END DO

      IF ( s%level >= DETAILS ) WRITE( s%out, * ) '   values assigned to y'

      RETURN

      END SUBROUTINE PRESOLVE_guess_y

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_say_goodbye( control, inform, s )

!     Print diagnostic and say goodbye.

!     Arguments:

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. L. Toint, Spring 2002.

!==============================================================================

!     Local variable

      INTEGER :: line

      IF ( s%level >= TRACE ) THEN

!        Successful exit

         WRITE( s%out, * )' '
         IF ( inform%status == OK ) THEN
            SELECT CASE ( s%stage )
            CASE ( READY )
               WRITE( s%out, * ) ' Problem successfully set up.'
            CASE ( ANALYZED )
               WRITE( s%out, * ) ' Problem successfully analyzed:',            &
                    s%tt, 'transforms.'
            CASE ( PERMUTED )
               WRITE( s%out, * ) ' Problem successfully permuted:',            &
                    s%tt, 'transforms.'
            CASE ( FULLY_REDUCED )
               WRITE( s%out, * ) ' No permutation necessary.'
            CASE ( RESTORED )
               WRITE( s%out, * ) ' Problem successfully restored.'
            CASE ( VOID )
            END SELECT

!        Unsuccessful exit

         ELSE
            DO line = 1, 3
               IF ( LEN_TRIM( inform%message( line ) ) > 0 ) THEN
                  WRITE( control%errout, * ) TRIM( inform%message( line ) )
               ELSE
                  EXIT
               END IF
            END DO
         END IF

         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) ' ******************** Bye *******************'
         WRITE( s%out, * ) ' '
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_say_goodbye

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_revise_control( mode, prob, control, inform, s  )

!     Verifies and activates the changes in the PRESOLVE control parameters

!     Arguments:

      INTEGER, INTENT( IN ) :: mode

!              the mode of PRESOLVE execution, corresponding to the routine from
!              which the routine is being called.

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. L. Toint, Fall 2001.

!==============================================================================

!     Local variable

      INTEGER :: iostat

!     Restore the accuracy parameter.

      ACCURACY = s%ACCURACY

!     Print starter, if requested.

      IF ( s%level >= TRACE ) THEN
         WRITE( control%out, * )                                               &
              ' verifying user-defined presolve control parameters'
      END IF

!     Printout level

      IF ( control%print_level /= s%prev_control%print_level ) THEN
         IF ( control%print_level >= ACTION ) THEN
            SELECT CASE ( control%print_level )
            CASE ( ACTION )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, * )                                            &
                       '  print level changed from SILENT to ACTION'
               CASE ( TRACE )
                  WRITE( s%out, * )                                            &
                       '  print level changed from TRACE to ACTION'
               CASE ( DETAILS )
                  WRITE( s%out, * )                                            &
                       '  print level changed from DETAILS to ACTION'
               CASE ( DEBUG )
                  WRITE( s%out, * )                                            &
                       '  print level changed from DEBUG to ACTION'
               CASE ( CRAZY )
                  WRITE( s%out, * )                                            &
                       '  print level changed from CRAZY to ACTION'
               END SELECT
            CASE ( DETAILS )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, * )                                            &
                       '  print level changed from SILENT to DETAILS'
               CASE ( TRACE )
                  WRITE( s%out, * )                                            &
                       '  print level changed from TRACE to DETAILS'
               CASE ( ACTION )
                  WRITE( s%out, * )                                            &
                       '  print level changed from ACTION to DETAILS'
               CASE ( DEBUG )
                  WRITE( s%out, * )                                            &
                       '  print level changed from DEBUG to DETAILS'
               CASE ( CRAZY )
                  WRITE( s%out, * )                                            &
                       '  print level changed from CRAZY to DETAILS'
               END SELECT
            CASE ( DEBUG )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, * )                                            &
                       '  print level changed from SILENT to DEBUG'
               CASE ( TRACE )
                  WRITE( s%out, * )                                            &
                       '  print level changed from TRACE to DEBUG'
               CASE ( ACTION )
                  WRITE( s%out, * )                                            &
                       '  print level changed from ACTION to DEBUG'
               CASE ( DETAILS )
                  WRITE( s%out, * )                                            &
                       '  print level changed from DETAILS to DEBUG'
               CASE ( CRAZY )
                  WRITE( s%out, * )                                            &
                       '  print level changed from CRAZY to DEBUG'
               END SELECT
            CASE ( CRAZY: )
               SELECT CASE ( s%prev_control%print_level )
               CASE ( SILENT )
                  WRITE( s%out, * )                                            &
                       '  print level changed from SILENT to CRAZY'
               CASE ( TRACE )
                  WRITE( s%out, * )                                            &
                       '  print level changed from TRACE to CRAZY'
               CASE ( ACTION )
                  WRITE( s%out, * )                                            &
                       '  print level changed from ACTION to CRAZY'
               CASE ( DETAILS )
                  WRITE( s%out, * )                                            &
                       '  print level changed from DETAILS to CRAZY'
               CASE ( DEBUG )
                  WRITE( s%out, * )                                            &
                       '  print level changed from DEBUG to CRAZY'
               END SELECT
            END SELECT
         END IF
         s%prev_control%print_level = control%print_level
      ELSE IF ( control%print_level >= ACTION ) THEN
         SELECT CASE ( control%print_level )
         CASE ( ACTION )
            WRITE( s%out, * ) '  print level is ACTION'
         CASE ( DETAILS )
            WRITE( s%out, * ) '  print level is DETAILS'
         CASE ( DEBUG )
            WRITE( s%out, * ) '  print level is DEBUG'
         CASE ( CRAZY: )
            WRITE( s%out, * ) '  print level is CRAZY'
         END SELECT
      END IF
      s%level = control%print_level

!     Printout device

      IF ( control%out /= s%prev_control%out ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%prev_control%out, * )                                     &
                 '  printing device changed from', s%prev_control%out, 'to',   &
                  control%out
            WRITE( control%out, * )                                            &
                 '  printing device changed from', s%prev_control%out, 'to',   &
                 control%out
         END IF
         s%prev_control%out = control%out
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  printing device is', control%out
      END IF
      s%out = control%out

!     Error printout device

      IF ( control%errout /= s%prev_control%errout ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%prev_control%out, * )                                     &
                 '  error output device changed from', s%prev_control%errout,  &
                 'to', control%errout
            WRITE( control%out, * )                                            &
                 '  error output device changed from', s%prev_control%errout,  &
                 'to', control%errout
         END IF
         s%prev_control%errout = control%errout
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  error output device device is', control%errout
      END IF

!     The value of INFINITY

      IF ( control%infinity /= s%prev_control%infinity ) THEN
         IF ( control%infinity <= s%prev_control%infinity ) THEN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  infinity value changed from', s%prev_control%infinity, 'to', &
               control%infinity
            s%prev_control%infinity = control%infinity
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, * )  '  PRESOLVE WARNING: ',                      &
                    'attempt to increase the value of infinity from',          &
               s%prev_control%infinity, 'to', control%infinity
               WRITE( s%out, * )  '    infinity value kept at',                &
                    s%prev_control%infinity
            END IF
            control%infinity = s%prev_control%infinity
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  infinity value is', control%infinity
      END IF
      s%INFINITY   = TEN * control%infinity
      s%P_INFINITY =   control%infinity
      s%M_INFINITY = - control%infinity

!     See if one needs to remove redundant variables and constraints.

      IF ( control%redundant_xc  .NEQV. s%prev_control%redundant_xc ) THEN
         IF ( s%level >= ACTION ) THEN
            IF ( control%redundant_xc ) THEN
               WRITE( s%out, * )                                               &
             '  removal of redundant variables and constraints is now attempted'
            ELSE
               WRITE( s%out, * )                                               &
       '  removal of redundant variables and constraints is no longer attempted'
            END IF
         END IF
         s%prev_control%redundant_xc = control%redundant_xc
      ELSE IF ( s%level >= ACTION ) THEN
         IF ( control%redundant_xc ) THEN
            WRITE( s%out, * )                                                  &
                 '  removal of redundant variables and constraints is attempted'
         ELSE
            WRITE( s%out, * )                                                  &
             '  removal of redundant variables and constraints is not attempted'
         END IF
      END IF

!     See if dual transformations are allowed.

      IF ( control%dual_transformations  .NEQV.                                &
                                 s%prev_control%dual_transformations ) THEN
         IF ( s%level >= ACTION ) THEN
            IF ( control%dual_transformations ) THEN
               WRITE( s%out, * ) '  dual transformations are now allowed'
            ELSE
               WRITE( s%out, * ) '  dual transformations are no longer allowed'
            END IF
         END IF
         s%prev_control%dual_transformations = control%dual_transformations
      ELSE IF ( s%level >= ACTION ) THEN
         IF ( control%dual_transformations ) THEN
            WRITE( s%out, * ) '  dual transformations are allowed'
         ELSE
            WRITE( s%out, * ) '  dual transformations are not allowed'
         END IF
      END IF

!     Disable dual transformations if they are not allowed.

      IF ( .NOT. control%dual_transformations ) THEN
         control%get_z                  = .FALSE.
         control%get_z_bounds           = .FALSE.
         control%get_y                  = .FALSE.
         control%get_y_bounds           = .FALSE.
         control%check_dual_feasibility = NONE
         control%z_accuracy             = control%infinity
         control%dual_constraints_freq  = 0
         control%singleton_columns_freq = 0
         control%doubleton_columns_freq = 0
      END IF

!     Sign convention for the multipliers

      IF ( control%y_sign /= s%prev_control%y_sign ) THEN
         SELECT CASE ( control%y_sign )
         CASE ( POSITIVE )
             IF ( s%level >= ACTION ) WRITE( s%out, * )                        &
                '  sign convention for the mulipliers',                        &
                ' changed from NEGATIVE to POSITIVE'
            s%prev_control%y_sign = control%y_sign
         CASE ( NEGATIVE )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                '  sign convention for the mulipliers',                        &
                ' changed from POSITIVE to NEGATIVE'
            s%prev_control%y_sign = control%y_sign
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * )  '  PRESOLVE WARNING:',                    &
                   ' attempt to change the sign',                              &
                   ' convention for the multipliers to the unknown value',     &
                   control%y_sign
               SELECT CASE ( s%prev_control%y_sign )
               CASE ( POSITIVE )
                  WRITE( s%out, * )                                            &
                    '  sign convention for the multipliers kept POSITIVE'
               CASE ( NEGATIVE)
                  WRITE( s%out, * )                                            &
                    '  sign convention for the multipliers kept NEGATIVE'
               END SELECT
            END IF
            control%y_sign = s%prev_control%y_sign
         END SELECT
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%y_sign )
         CASE ( POSITIVE )
            WRITE( s%out, * )'  sign convention for the multipliers is POSITIVE'
         CASE ( NEGATIVE )
            WRITE( s%out, * )'  sign convention for the multipliers is NEGATIVE'
         END SELECT
      END IF

!     Policy for the multipliers of inactive constraints

      IF ( control%inactive_y /= s%prev_control%inactive_y ) THEN
         SELECT CASE ( control%inactive_y )
         CASE ( FORCE_TO_ZERO )
             IF ( s%level >= ACTION ) WRITE( s%out, * )                        &
                '  policy for the multipliers of inactive constraints',        &
                ' changed from LEAVE_AS_IS to FORCE_TO_ZERO'
            s%prev_control%inactive_y = control%inactive_y
         CASE ( LEAVE_AS_IS )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                '  policy for the multipliers of inactive constraints',        &
                ' changed from FORCE_TO_ZERO to LEAVE_AS_IS'
            s%prev_control%inactive_y = control%inactive_y
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * )  '  PRESOLVE WARNING:',                    &
                       ' attempt to change the policy for the multipliers'
                  WRITE( s%out, * ) '  of inactive constraints to the',        &
                       ' unknown value', control%inactive_y
               SELECT CASE ( s%prev_control%inactive_y )
               CASE ( FORCE_TO_ZERO)
                  WRITE( s%out, * )                                            &
                      '  policy for the multipliers of inactive constraints',  &
                      ' kept as FORCE_TO_ZERO'
               CASE ( LEAVE_AS_IS )
                  WRITE( s%out, * )                                            &
                      '  policy for the multipliers of inactive constraints',  &
                      ' kept as LEAVE_AS_IS'
               END SELECT
            END IF
            control%inactive_y = s%prev_control%inactive_y
         END SELECT
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%inactive_y )
         CASE ( FORCE_TO_ZERO )
            WRITE( s%out, * )                                                  &
                 '  policy for the multipliers of inactive constraints',       &
                 ' is FORCE_TO_ZERO'
         CASE ( LEAVE_AS_IS )
            WRITE( s%out, * )                                                  &
                 '  policy for the multipliers of inactive constraints',       &
                 ' is LEAVE_AS_IS'
         END SELECT
      END IF

!     Sign convention for the dual variables

      IF ( control%z_sign /= s%prev_control%z_sign ) THEN
         SELECT CASE ( control%z_sign )
         CASE ( POSITIVE )
             IF ( s%level >= ACTION ) WRITE( s%out, * )                        &
                '  sign convention for the dual variables',                    &
                ' changed from NEGATIVE to POSITIVE'
            s%prev_control%z_sign = control%z_sign
         CASE ( NEGATIVE )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                '  sign convention for the dual variables',                    &
                ' changed from POSITIVE to NEGATIVE'
            s%prev_control%z_sign = control%z_sign
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, * ) '  PRESOLVE WARNING:',                        &
                    '  attempt to change the sign convention for the dual'
               WRITE( s%out, * ) '  variables to the unknown value',           &
                    control%z_sign
               SELECT CASE ( s%prev_control%z_sign )
               CASE ( POSITIVE )
                  WRITE( s%out, * )                                            &
                    '  sign convention for the dual variables kept POSITIVE'
               CASE ( NEGATIVE)
                  WRITE( s%out, * )                                            &
                    '  sign convention for the dual variables kept NEGATIVE'
               END SELECT
            END IF
            control%z_sign = s%prev_control%z_sign
         END SELECT
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%z_sign )
         CASE ( POSITIVE )
            WRITE( s%out, * )                                                  &
                 '  sign convention for the dual variables is POSITIVE'
         CASE ( NEGATIVE )
            WRITE( s%out, * )                                                  &
                 '  sign convention for the dual variables is NEGATIVE'
         END SELECT
      END IF

!     Policy for the dual variables of inactive bounds

      IF ( control%inactive_z /= s%prev_control%inactive_z ) THEN
         SELECT CASE ( control%inactive_z )
         CASE ( FORCE_TO_ZERO )
             IF ( s%level >= ACTION ) WRITE( s%out, * )                        &
                '  policy for the dual variables of inactive bounds',          &
                ' changed from LEAVE_AS_IS to FORCE_TO_ZERO'
            s%prev_control%inactive_z = control%inactive_z
         CASE ( LEAVE_AS_IS )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                '  policy for the dual variables of inactive bounds',          &
                ' changed from FORCE_TO_ZERO to LEAVE_AS_IS'
            s%prev_control%inactive_z = control%inactive_z
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                       ' attempt to change the policy for the dual variables'
                  WRITE( s%out, * ) '  of inactive bounds to the unknown',     &
                       ' value', control%inactive_z
               SELECT CASE ( s%prev_control%inactive_z )
               CASE ( FORCE_TO_ZERO)
                  WRITE( s%out, * )                                            &
                       '  policy for the dual variables of inactive bounds',   &
                       ' kept as FORCE_TO_ZERO'
               CASE ( LEAVE_AS_IS )
                  WRITE( s%out, * )                                            &
                       '  policy for the dual variables of inactive bounds',   &
                       ' kept as LEAVE_AS_IS'
               END SELECT
            END IF
            control%inactive_z = s%prev_control%inactive_z
         END SELECT
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%inactive_z )
         CASE ( FORCE_TO_ZERO )
            WRITE( s%out, * )                                                  &
                 '  policy for the dual variables of inactive bounds',         &
                 ' is FORCE_TO_ZERO'
         CASE ( LEAVE_AS_IS )
            WRITE( s%out, * )                                                  &
                 '  policy for the dual variables of inactive bounds',         &
                 ' is LEAVE_AS_IS'
         END SELECT
      END IF

!     Return if no transformations are allowed.

      IF ( control%max_nbr_transforms == 0 ) RETURN

!     Obtention of the quadratic value

      IF ( control%get_q .NEQV. s%prev_control%get_q ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for the value of the quadratic changed from',           &
            s%prev_control%get_q, 'to', control%get_q
         s%prev_control%get_q = control%get_q
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the value of the quadratic is',      &
              control%get_q
      END IF

!     Obtention of the quadratic constant

      IF ( control%get_f .NEQV. s%prev_control%get_f ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for the constant of the quadratic changed from',        &
            s%prev_control%get_f, 'to', control%get_f
         s%prev_control%get_f = control%get_f
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the constant of the quadratic is',   &
              control%get_f
      END IF

!     Obtention of the quadratic gradient

      IF ( control%get_g .NEQV. s%prev_control%get_g ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for the gradient of the quadratic changed from',        &
            s%prev_control%get_g, 'to', control%get_g
         s%prev_control%get_g = control%get_g
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the gradient of the quadratic is',   &
              control%get_g
      END IF

!     Obtention of the quadratic Hessian

      IF ( control%get_H .NEQV. s%prev_control%get_H ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for the Hessian of the quadratic changed from',         &
            s%prev_control%get_H, 'to', control%get_H
         s%prev_control%get_H = control%get_H
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the Hessian of the quadratic is',    &
              control%get_H
      END IF

!     Obtention of the matrix A

      IF ( control%get_A .NEQV. s%prev_control%get_A ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for the matrix A changed from',                         &
            s%prev_control%get_A, 'to', control%get_A
         s%prev_control%get_A = control%get_A
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the matrix A is', control%get_A
      END IF

!     Obtention of x

      IF ( control%get_x .NEQV. s%prev_control%get_x ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for values of the variables changed from',              &
            s%prev_control%get_x, 'to', control%get_x
         s%prev_control%get_x = control%get_x
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the values of the variables is',     &
              control%get_x
      END IF

!     Obtention of the bounds on x

      IF ( control%get_x_bounds .NEQV. s%prev_control%get_x_bounds ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for bounds on the variables changed from',              &
            s%prev_control%get_x_bounds, 'to', control%get_x_bounds
         s%prev_control%get_x_bounds = control%get_x_bounds
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the bounds on the variables is',     &
              control%get_x_bounds
      END IF

!     Obtention of z

      IF ( control%get_z .NEQV. s%prev_control%get_z ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for values of the dual variables changed from',         &
            s%prev_control%get_z, 'to', control%get_z
         s%prev_control%get_z = control%get_z
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the values of the dual variables is',&
              control%get_z
      END IF

!     Obtention of the bounds on z

      IF ( control%get_z_bounds .NEQV. s%prev_control%get_z_bounds ) THEN
         IF ( s%stage == READY ) THEN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  request for values of the dual variables changed from',      &
               s%prev_control%get_z_bounds, 'to', control%get_z_bounds
            s%prev_control%get_z_bounds = control%get_z_bounds
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, * ) ' PRESOLVE WARNING:',                         &
                    ' attempt to change the request',                          &
                    ' for dual bounds to', control%get_z_bounds
               WRITE( s%out, * ) '  request for the bounds on the dual',       &
                    ' variables kept to', s%prev_control%get_z_bounds
            END IF
            control%get_z_bounds = s%prev_control%get_z_bounds
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the bounds on the dual variables',   &
              ' is', control%get_z_bounds
      END IF

!     Obtention of c

      IF ( control%get_c .NEQV. s%prev_control%get_c ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for values of the constraints changed from',            &
            s%prev_control%get_c, 'to', control%get_c
         s%prev_control%get_c = control%get_c
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the values of the constraints is',   &
              control%get_c
      END IF

!     Obtention of the bounds on c

      IF ( control%get_c_bounds .NEQV. s%prev_control%get_c_bounds ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for values of the bounds on the constraints changed',   &
            ' from', s%prev_control%get_c_bounds, 'to', control%get_c_bounds
         s%prev_control%get_c_bounds = control%get_c_bounds
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the bounds on the constraints is',   &
              control%get_c_bounds
      END IF

!     Obtention of y

      IF ( control%get_y .NEQV. s%prev_control%get_y ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  request for values of the multipliers changed from',            &
            s%prev_control%get_y, 'to', control%get_y
         s%prev_control%get_y = control%get_y
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the values of the multipliers is',   &
              control%get_y
      END IF

!     Obtention of the bounds on y

      IF ( control%get_y_bounds .NEQV. s%prev_control%get_y_bounds ) THEN
         IF ( s%stage == READY ) THEN
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  request for bounds on the multipliers changed from',         &
               s%prev_control%get_y_bounds, 'to', control%get_y_bounds
            s%prev_control%get_y_bounds = control%get_y_bounds
         ELSE
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, * ) '  PRESOLVE WARNING:',                        &
                    ' attempt to change the request',                          &
                    ' for the multiplier bounds to', control%get_y_bounds
               WRITE( s%out, * ) '  request for the bounds on the',            &
                    ' multipliers kept to', s%prev_control%get_y_bounds
            END IF
            control%get_y_bounds = s%prev_control%get_y_bounds
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  request for the bounds on the multipliers is',   &
              control%get_y_bounds
      END IF

!     Level of primal feasibility at RESTORE

      IF ( control%check_primal_feasibility /=   &
           s%prev_control%check_primal_feasibility ) THEN
         IF ( s%level >= ACTION ) THEN
            SELECT CASE ( control%check_primal_feasibility )
            CASE ( NONE )
               SELECT CASE ( s%prev_control%check_primal_feasibility )
               CASE ( BASIC )
                    WRITE( s%out, * ) '  level of primal feasibility at ',     &
                         'RESTORE changed from BASIC to NONE'
               CASE ( SEVERE )
                    WRITE( s%out, * ) '  level of primal feasibility at ',     &
                         'RESTORE changed from SEVERE to NONE'
               END SELECT
            CASE ( BASIC )
               SELECT CASE ( s%prev_control%check_primal_feasibility )
               CASE ( NONE )
                    WRITE( s%out, * ) '  level of primal feasibility at ',     &
                         'RESTORE changed from NONE to BASIC'
               CASE ( SEVERE )
                    WRITE( s%out, * ) '  level of primal feasibility at ',     &
                         'RESTORE changed from SEVERE to BASIC'
               END SELECT
            CASE ( SEVERE )
               SELECT CASE ( s%prev_control%check_primal_feasibility )
               CASE ( NONE )
                    WRITE( s%out, * ) '  level of primal feasibility at ',     &
                         'RESTORE changed from NONE to SEVERE'
               CASE ( BASIC )
                    WRITE( s%out, * ) '  level of primal feasibility at ',     &
                         'RESTORE changed from BASIC to SEVERE'
               END SELECT
            END SELECT
         END IF
         s%prev_control%check_primal_feasibility =                             &
                                                control%check_primal_feasibility
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%check_primal_feasibility )
         CASE ( NONE )
            WRITE( s%out, * ) '  level of primal feasibility at RESTORE is NONE'
         CASE ( BASIC )
            WRITE( s%out, * )                                                  &
                 '  level of primal feasibility at RESTORE is BASIC'
         CASE ( SEVERE )
            WRITE( s%out, * )                                                  &
                 '  level of primal feasibility at RESTORE is SEVERE'
         END SELECT
      END IF

!     Level of dual feasibility at RESTORE

      IF ( control%check_dual_feasibility /=   &
           s%prev_control%check_dual_feasibility ) THEN
         IF ( s%level >= ACTION ) THEN
            SELECT CASE ( control%check_dual_feasibility )
            CASE ( NONE )
               SELECT CASE ( s%prev_control%check_dual_feasibility )
               CASE ( BASIC )
                    WRITE( s%out, * ) '  level of dual feasibility at ',       &
                         'RESTORE changed from BASIC to NONE'
               CASE ( SEVERE )
                    WRITE( s%out, * ) '  level of dual feasibility at ',       &
                         'RESTORE changed from SEVERE to NONE'
               END SELECT
            CASE ( BASIC )
               SELECT CASE ( s%prev_control%check_dual_feasibility )
               CASE ( NONE )
                    WRITE( s%out, * ) '  level of dual feasibility at ',       &
                         'RESTORE changed from NONE to BASIC'
               CASE ( SEVERE )
                    WRITE( s%out, * ) '  level of dual feasibility at ',       &
                         'RESTORE changed from SEVERE to BASIC'
               END SELECT
            CASE ( SEVERE )
               SELECT CASE ( s%prev_control%check_dual_feasibility )
               CASE ( NONE )
                    WRITE( s%out, * ) '  level of dual feasibility at ',       &
                         'RESTORE changed from NONE to SEVERE'
               CASE ( BASIC )
                    WRITE( s%out, * ) '  level of dual feasibility at ',       &
                         'RESTORE changed from BASIC to SEVERE'
               END SELECT
            END SELECT
         END IF
         s%prev_control%check_dual_feasibility = control%check_dual_feasibility
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%check_dual_feasibility )
         CASE ( NONE )
            WRITE( s%out, * ) '  level of dual feasibility at RESTORE is NONE'
         CASE ( BASIC )
            WRITE( s%out, * ) '  level of dual feasibility at RESTORE is BASIC'
         CASE ( SEVERE )
            WRITE( s%out, * ) '  level of dual feasibility at RESTORE is SEVERE'
         END SELECT
      END IF

!     Accuracy for the linear constraints

      IF ( control%c_accuracy /= s%prev_control%c_accuracy ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * )                                                  &
               '  relative accuracy for the linear constraints changed'
            WRITE( s%out, * )                                                  &
               '     from', s%prev_control%c_accuracy, 'to', control%c_accuracy
         END IF
         s%prev_control%c_accuracy = control%c_accuracy
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  relative accuracy for the linear constraints is',&
              control%c_accuracy
      END IF

!     Accuracy for dual variables

      IF ( control%z_accuracy /= s%prev_control%z_accuracy ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * ) '  relative accuracy for dual variables changed'
            WRITE( s%out, * ) '     from', s%prev_control%z_accuracy, 'to',    &
                 control%z_accuracy
         END IF
         s%prev_control%z_accuracy = control%z_accuracy
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * )                                                     &
              '  relative accuracy for dual variables is', control%z_accuracy
      END IF

!     Name of the file used to store the problem transformations on disk

      IF ( control%transf_file_name /= s%prev_control%transf_file_name ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change the',     &
                 ' file to save problem transformations from',                 &
                 s%prev_control%transf_file_name, 'to', control%transf_file_name
            WRITE( s%out, * )                                                  &
                 '  file to save problem transformations kept to ', &
                 s%prev_control%transf_file_name
         END IF
         control%transf_file_name = s%prev_control%transf_file_name
      ELSE IF ( s%level >= ACTION  ) THEN
         WRITE( s%out, * ) '  file to store problem  transformations is ',     &
              control%transf_file_name
      END IF

!     Unit number for writing/reading the transformation file

      IF ( control%transf_file_nbr /= s%prev_control%transf_file_nbr ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  unit number for transformations changed from ',                 &
            s%prev_control%transf_file_nbr, 'to', control%transf_file_nbr
         s%prev_control%transf_file_nbr = control%transf_file_nbr
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  unit number for transformations saving is',      &
              control%transf_file_nbr
      END IF

!     KEEP or DELETE status for the transformation file

      IF ( control%transf_file_status /= s%prev_control%transf_file_status )THEN
         SELECT CASE ( control%transf_file_status )
         CASE ( KEEP )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  final status for transformation file changed from',          &
               ' DELETE to KEEP'
            s%prev_control%transf_file_status = control%transf_file_status
         CASE ( DELETE )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
               '  final status for transformation file changed from',          &
               ' KEEP to DELETE'
            s%prev_control%transf_file_status = control%transf_file_status
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, * )                                               &
                    '  PRESOLVE WARNING: attempt to set the final ',           &
                    'status for transformation file  to the undefined value',  &
                    control%transf_file_name
               SELECT CASE ( s%prev_control%transf_file_status )
               CASE ( KEEP )
                   WRITE( s%out, * ) '  final status for ',                    &
                        'transformation file set to KEEP'
               CASE ( DELETE )
                   WRITE( s%out, * ) '  final status for ',                    &
                        'transformation file set to DELETE'
               END SELECT
            END IF
            control%transf_file_status = s%prev_control%transf_file_status
         END SELECT
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%transf_file_status )
         CASE ( KEEP )
             WRITE( s%out, * )'  final status for transformation file is KEEP'
         CASE ( DELETE )
             WRITE( s%out, * )'  final status for transformation file is DELETE'
         END SELECT
      END IF

!-------------------------------------------------------------------------------

!     Nothing more to do if PERMUTE, RESTORE or TERMINATE

      IF ( mode == PERMUTE .OR. mode == RESTORE .OR. mode == TERMINATE ) RETURN

!-------------------------------------------------------------------------------

!     Minimum relative bound improvement

      s%mrbi = MAX( TEN * TEN * EPSMACH, control%min_rel_improve)
      IF ( s%mrbi /= s%prev_control%min_rel_improve ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * ) '  minimum relative bound improvement changed'
            WRITE( s%out, * ) '     from',                                     &
                 s%prev_control%min_rel_improve, 'to', s%mrbi
         END IF
         s%prev_control%min_rel_improve = s%mrbi
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  minimum relative bound improvement is',  s%mrbi
      END IF

!     Maximum growth factor between original and reduced problems

      IF ( control%max_growth_factor /= s%prev_control%max_growth_factor ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * ) '  maximum growth factor changed'
            WRITE( s%out, * ) '     from', s%prev_control%max_growth_factor,   &
                 'to', control%max_growth_factor
         END IF
         s%prev_control%max_growth_factor = control%max_growth_factor
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  maximum growth factor is',                       &
              control%max_growth_factor
      END IF

!     Strategy for presolve termination.

      IF ( control%termination /= s%prev_control%termination ) THEN
         SELECT CASE ( control%termination )
         CASE ( REDUCED_SIZE )
             IF ( s%level >= ACTION ) WRITE( s%out, * )                        &
                '  strategy for presolve termination',                         &
                ' changed from FULL_PRESOLVE to REDUCED_SIZE'
            s%prev_control%termination = control%termination
         CASE ( FULL_PRESOLVE )
            IF ( s%level >= ACTION ) WRITE( s%out, * )                         &
                '  strategy for presolve termination',                         &
                ' changed from REDUCED_SIZE to FULL_PRESOLVE'
            s%prev_control%termination = control%termination
         CASE DEFAULT
            IF ( s%level >= ACTION ) THEN
               WRITE( s%out, * ) '  PRESOLVE WARNING:',                        &
                    ' attempt to change the strategy for presolve termination'
               WRITE( s%out, * ) '  to the unknown value', control%termination
               SELECT CASE ( s%prev_control%termination )
               CASE ( REDUCED_SIZE )
                  WRITE( s%out, * )                                            &
                    '  strategy for presolve termination kept as REDUCED_SIZE'
               CASE ( FULL_PRESOLVE )
                  WRITE( s%out, * )                                            &
                   '  strategy for presolve termination kept as FULL_PRESOLVE'
               END SELECT
            END IF
            control%termination = s%prev_control%termination
         END SELECT
      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%termination )
         CASE ( REDUCED_SIZE )
            WRITE( s%out, * )                                                  &
                 '  strategy for presolve termination is REDUCED_SIZE'
         CASE ( FULL_PRESOLVE )
            WRITE( s%out, * )                                                  &
                 '  strategy for presolve termination is FULL_PRESOLVE'
         END SELECT
      END IF

!     Maximum number of analysis passes

      IF ( control%max_nbr_passes /= s%prev_control%max_nbr_passes ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  maximum number of analysis passes changed from',                &
            s%prev_control%max_nbr_passes, 'to', control%max_nbr_passes
         s%prev_control%max_nbr_passes = control%max_nbr_passes
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  maximum number of analysis passes is',           &
              control%max_nbr_passes
      END IF

!    Maximum number of problem transformations

      IF ( control%max_nbr_transforms /= s%prev_control%max_nbr_transforms) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  maximum number of transformations changed from',                &
            s%prev_control%max_nbr_transforms, 'to', control%max_nbr_transforms
         s%prev_control%max_nbr_transforms = control%max_nbr_transforms
         IF ( s%tt >= control%max_nbr_transforms ) THEN
            inform%status = MAX_NBR_TRANSF
            RETURN
         END IF
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  maximum number of transformations is',           &
              control%max_nbr_transforms
      END IF

!     Size of the transformation buffer in memory

      IF ( control%transf_buffer_size /= s%prev_control%transf_buffer_size )THEN
         control%transf_buffer_size = MIN( control%transf_buffer_size,         &
                                           control%max_nbr_transforms  )
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '    ze of transformation buffer changed from',                    &
            s%prev_control%transf_buffer_size, 'to', control%transf_buffer_size
         s%prev_control%transf_buffer_size = control%transf_buffer_size
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * )'  size of transformation buffer is',                &
              control%transf_buffer_size
      END IF
      s%max_tm = control%transf_buffer_size

!     Maximum percentage of row-wise fill in A

      IF ( control%max_fill /= s%prev_control%max_fill ) THEN
         IF ( s%level >= ACTION ) THEN
            IF ( control%max_fill >= 0 ) THEN
               IF ( s%prev_control%max_fill >= 0 ) THEN
                  WRITE( s%out, * )                                            &
                       '  maximum percentage of fill in a row changed from',   &
                        s%prev_control%max_fill, '% to', control%max_fill, '%'
               ELSE
                  WRITE( s%out, * )                                            &
                       '  maximum percentage of fill in a row changed from',   &
                       ' infinite to', control%max_fill, '%'
               END IF
            ELSE IF ( s%prev_control%max_fill >= 0 ) THEN
                  WRITE( s%out, * )                                            &
                       '  maximum percentage of fill in a row changed from',   &
                       s%prev_control%max_fill, '% to infinite'
            END IF
         END IF
         s%prev_control%max_fill = control%max_fill
      ELSE IF ( s%level >= ACTION ) THEN
         IF ( control%max_fill >= 0 ) THEN
            WRITE( s%out, * ) '  maximum percentage of fill in a row is',      &
                 control%max_fill, '%'
         ELSE
            WRITE( s%out, * )                                                  &
                 '  maximum percentage of fill in a row is infinite'
         END IF
      END IF
      IF ( control%max_fill >= 0 ) THEN
         s%max_fill_prop = HUNDRED + control%max_fill
         s%max_fill_prop = s%max_fill_prop / HUNDRED
      ELSE
         s%max_fill_prop = s%INFINITY
      END IF

!     Tolerance for pivoting in A

      IF ( control%pivot_tol /= s%prev_control%pivot_tol ) THEN
         IF ( s%level >= ACTION ) THEN
            WRITE( s%out, * ) '  relative tolerance for pivoting in A changed'
            WRITE( s%out, * )                                                  &
                 '     from', s%prev_control%pivot_tol, 'to', control%pivot_tol
         END IF
         s%prev_control%pivot_tol = control%pivot_tol
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  relative tolerance for pivoting in A is',        &
              control%pivot_tol
      END IF

!     The frequency of the various preprocessing actions

!     1) analysis of the primal constraints

      IF ( control%primal_constraints_freq /=                                  &
           s%prev_control%primal_constraints_freq ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of primal constraints analysis changed from',         &
            s%prev_control%primal_constraints_freq, 'to',                      &
            control%primal_constraints_freq
         s%prev_control%primal_constraints_freq =                              &
            control%primal_constraints_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of primal constraints analysis is',    &
              control%primal_constraints_freq
      END IF

!     2) analysis of the dual constraints

      IF ( control%dual_constraints_freq /=                                    &
           s%prev_control%dual_constraints_freq ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of dual constraints analysis changed from',           &
            s%prev_control%dual_constraints_freq, 'to',                        &
            control%dual_constraints_freq
         s%prev_control%dual_constraints_freq = control%dual_constraints_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of dual constraints analysis is',      &
              control%dual_constraints_freq
      END IF

!     3) analysis of the singleton columns

      IF ( control%singleton_columns_freq /=                                   &
           s%prev_control%singleton_columns_freq ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of singleton columns analysis changed from',          &
            s%prev_control%singleton_columns_freq, 'to',                       &
            control%singleton_columns_freq
         s%prev_control%singleton_columns_freq = control%singleton_columns_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of singleton columns analysis is',     &
              control%singleton_columns_freq
      END IF

!     4) analysis of the doubleton columns

      IF ( control%doubleton_columns_freq /=                                   &
           s%prev_control%doubleton_columns_freq ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of doubleton columns analysis changed from',          &
            s%prev_control%doubleton_columns_freq, 'to',                       &
            control%doubleton_columns_freq
         s%prev_control%doubleton_columns_freq = control%doubleton_columns_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of doubleton columns analysis is',     &
              control%doubleton_columns_freq
      END IF

!     5) analysis of the linearly unconstrained variables

      IF ( control%unc_variables_freq /= s%prev_control%unc_variables_freq )THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of unconstrained variables analysis changed from',    &
            s%prev_control%unc_variables_freq, 'to', control%unc_variables_freq
         s%prev_control%unc_variables_freq = control%unc_variables_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of unconstrained variables',           &
              ' analysis is', control%unc_variables_freq
      END IF

!     6) analysis of the linearly dependent variables

      IF ( control%dependent_variables_freq /=                                 &
           s%prev_control%dependent_variables_freq ) THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of dependent variables analysis changed from',        &
            s%prev_control%dependent_variables_freq, 'to',                     &
            control%dependent_variables_freq
         s%prev_control%dependent_variables_freq =                             &
            control%dependent_variables_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of dependent variables analysis is',   &
              control%dependent_variables_freq
      END IF

!     7) row sparsification frequency

      IF ( control%sparsify_rows_freq /= s%prev_control%sparsify_rows_freq )THEN
         IF ( s%level >= ACTION ) WRITE( s%out, * )                            &
            '  frequency of row sparsification analysis changed from',         &
            s%prev_control%sparsify_rows_freq, 'to', control%sparsify_rows_freq
         s%prev_control%sparsify_rows_freq = control%sparsify_rows_freq
      ELSE IF ( s%level >= ACTION ) THEN
         WRITE( s%out, * ) '  frequency of row sparsification analysis is',    &
              control%sparsify_rows_freq
      END IF

!-------------------------------------------------------------------------------

!     Final status of the bounds on the variables, dual variables and
!     multipliers.

!-------------------------------------------------------------------------------

!     The variables

      IF ( control%final_x_bounds /= s%prev_control%final_x_bounds ) THEN

!        Allowed change after INITIALIZE

         IF ( s%stage == READY ) THEN

            SELECT CASE ( control%final_x_bounds )

            CASE ( TIGHTEST )

               s%prev_control%final_x_bounds = control%final_x_bounds

!              NOTE: there is no need to deallocate the additional workspace
!                    here because the default is not to allocate them.

            CASE ( NON_DEGENERATE )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * )                                            &
                   '  changing the final status of the bounds on the variables'
                  WRITE( s%out, * ) '  from TIGHTEST to NON_DEGENERATE'
               END IF
               s%prev_control%final_x_bounds = control%final_x_bounds

!              Now allocate the required additional workspace
!              1) lower bounds on the primal variables

               ALLOCATE( s%x_l2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating x_l2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    x_l2(', prob%n, ') allocated'
               s%x_l2 = prob%X_l

!              2) upper bounds on the primal variables

               ALLOCATE( s%x_u2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating x_u2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    x_u2(', prob%n, ') allocated'
               s%x_u2 = prob%X_u

            CASE ( LOOSEST )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * )                                            &
                   '  changing the final status of the bounds on the variables'
                  WRITE( s%out, * ) '  from TIGHTEST to LOOSEST'
               END IF
               s%prev_control%final_x_bounds = control%final_x_bounds

!              Now allocate the required additional workspace
!              1) lower bounds on the primal variables

               ALLOCATE( s%x_l2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating x_l2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    x_l2(', prob%n, ') allocated'
               s%x_l2 = prob%X_l

!              2) upper bounds on the primal variables

               ALLOCATE( s%x_u2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating x_u2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    x_u2(', prob%n, ') allocated'
               s%x_u2 = prob%X_u

!           Change allowed, but erroneous value specified.

            CASE DEFAULT
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                      ' attempt to change the final status of the bounds'
                  WRITE( s%out, * )                                            &
                       '  on the variables to the unknown value',              &
                      control%final_x_bounds
                  SELECT CASE ( s%prev_control%final_x_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) '  final status of the bounds on',      &
                          ' the variables kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) '  final status of the bounds on',      &
                          ' the variables kept to NON_DEGENERATE'
                  END SELECT
               END IF
               control%final_x_bounds = s%prev_control%final_x_bounds
            END SELECT

!        Illegal attempt to change the final status of the bounds (that is
!        later than after INITIALIZE).

         ELSE
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( control%final_x_bounds )
               CASE ( TIGHTEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the variables'
                  SELECT CASE ( s%prev_control%final_x_bounds )
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) ' from NON_DEGENERATE to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' variables kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) ' from LOOSEST to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' variables kept to LOOSEST'
                  END SELECT
               CASE ( NON_DEGENERATE )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the variables'
                  SELECT CASE ( s%prev_control%final_x_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) ' from TIGHTEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' variables kept to TIGHTEST'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) ' from LOOSEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' variables kept to LOOSEST'
                  END SELECT
               CASE ( LOOSEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the variables'
                  SELECT CASE ( s%prev_control%final_x_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) ' from TIGHTEST to LOOSEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' variables kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) ' from NON_DEGENERATE to LOOSEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' variables kept to NON_DEGENERATE'
                   END SELECT
               CASE DEFAULT
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                      ' attempt to change the final status of the bounds'
                  WRITE( s%out, * ) '  on the variables to the unknown value', &
                      control%final_x_bounds
                  SELECT CASE ( s%prev_control%final_x_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )                                         &
                       '  final status of the bounds kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )                                         &
                       '  final status of the bounds kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )                                         &
                       '  final status of the bounds kept to LOOSEST'
                  END SELECT
               END SELECT
            END IF
            control%final_x_bounds = s%prev_control%final_x_bounds
         END IF

!     No change in the final status of the bounds has been requested.

      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%final_x_bounds )
         CASE ( TIGHTEST )
            WRITE( s%out, * )                                                  &
                 '  final status of the bounds on the variables is TIGHTEST'
         CASE ( NON_DEGENERATE )
            WRITE( s%out, * )                                                  &
             '  final status of the bounds on the variables is NON_DEGENERATE'
         CASE ( LOOSEST )
            WRITE( s%out, * )                                                  &
             '  final status of the bounds on the variables is LOOSEST'
         END SELECT
      END IF

!     The dual variables

      IF ( control%final_z_bounds /= s%prev_control%final_z_bounds ) THEN

!        Allowed change after INITIALIZE

         IF ( s%stage == READY ) THEN

            SELECT CASE ( control%final_z_bounds )

            CASE ( TIGHTEST )

               s%prev_control%final_z_bounds = control%final_z_bounds

!              NOTE: there is no need to deallocate the additional workspace
!                    here because the default is not to allocate them.

            CASE ( NON_DEGENERATE )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  changing the final status of the',      &
                       ' bounds on the dual variables'
                  WRITE( s%out, * ) '  from TIGHTEST to NON_DEGENERATE'
               END IF
               s%prev_control%final_z_bounds = control%final_z_bounds

!              Now allocate the required additional workspace.

!              1) lower bounds on the dual variables

               ALLOCATE( s%z_l2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating z_l2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    z_l2(', prob%n, ') allocated'
               s%z_l2 = prob%Z_l

!              2) upper bounds on the dual variables

               ALLOCATE( s%z_u2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating z_u2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    z_u2(', prob%n, ') allocated'
               s%z_u2 = prob%Z_u

            CASE ( LOOSEST )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  changing the final status of the',      &
                       ' bounds on the dual variables'
                  WRITE( s%out, * ) '  from TIGHTEST to LOOSEST'
               END IF
               s%prev_control%final_z_bounds = control%final_z_bounds

!              Now allocate the required additional workspace.

!              1) lower bounds on the dual variables

               ALLOCATE( s%z_l2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating z_l2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    z_l2(', prob%n, ') allocated'
               s%z_l2 = prob%Z_l

!              2) upper bounds on the dual variables

               ALLOCATE( s%z_u2( prob%n ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating z_u2(', &
                       prob%n, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    z_u2(', prob%n, ') allocated'
               s%z_u2 = prob%Z_u

!           Change allowed, but erroneous value specified.

            CASE DEFAULT
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                      ' attempt to change the final status of the bounds'
                  WRITE( s%out, * ) '  on the duals variable to the unknown',  &
                       ' value', control%final_z_bounds
                  SELECT CASE ( s%prev_control%final_z_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' dual variables kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' kept to LOOSEST'
                  END SELECT
               END IF
               control%final_z_bounds = s%prev_control%final_z_bounds
            END SELECT

!        Illegal attempt to change the final status of the bounds (that is
!        later than after INITIALIZE).

         ELSE
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( control%final_z_bounds )
               CASE ( TIGHTEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the dual variables'
                  SELECT CASE ( s%prev_control%final_z_bounds )
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) ' from NON_DEGENERATE to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' dual variables kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) ' from LOOSEST to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' dual variables kept to LOOSEST'
                  END SELECT
               CASE ( NON_DEGENERATE )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the dual variables'
                  SELECT CASE ( s%prev_control%final_z_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) ' from TIGHTEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' dual variables kept to TIGHTEST'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) ' from LOOSEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' dual variables kept to LOOSEST'
                  END SELECT
               CASE ( LOOSEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the dual variables'
                  SELECT CASE ( s%prev_control%final_z_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) ' from TIGHTEST to LOOSEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' dual variables kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) ' from NON_DEGENERATE to LOOSEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' dual variables kept to NON_DEGENERATE'
                  END SELECT
               CASE DEFAULT
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                      ' attempt to change the final status of the bounds'
                  WRITE( s%out, * ) '   on the dual variables to the',         &
                       ' unknown value', control%final_z_bounds
                  SELECT CASE ( s%prev_control%final_z_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' dual variables kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )'    final status of the bounds on the', &
                          ' dual variables kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )'    final status of the bounds on the', &
                          ' dual variables kept to LOOSEST'
                  END SELECT
               END SELECT
            END IF
            control%final_z_bounds = s%prev_control%final_z_bounds
         END IF

!     No change in the final status of the bounds has been requested.

      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%final_z_bounds )
         CASE ( TIGHTEST )
            WRITE( s%out, * ) '  final status of the bounds on the',           &
                 ' dual variables is TIGHTEST'
         CASE ( NON_DEGENERATE )
            WRITE( s%out, * ) '  final status of the bounds on the',           &
                 ' dual variables is NON_DEGENERATE'
         CASE ( LOOSEST )
            WRITE( s%out, * ) '  final status of the bounds on the',           &
                 ' dual variables is LOOSEST'
         END SELECT
      END IF

!     The constraints bounds

      IF ( control%final_c_bounds /= s%prev_control%final_c_bounds .AND.       &
           prob%m > 0                                                ) THEN

!        Allowed change after INITIALIZE

         IF ( s%stage == READY ) THEN

!           Make sure that the status of the final bounds on c is coherent
!           with that on x.

            IF ( control%final_c_bounds /= TIGHTEST ) THEN
               IF ( control%final_c_bounds /= control%final_x_bounds ) THEN
                  IF ( s%level >= ACTION ) WRITE( s%out, * )                   &
                     '  preserving the coherence between the status of the',   &
                     ' final bounds on x and c'
                  control%final_c_bounds = control%final_x_bounds
               END IF
            END IF

!           Print the selected value and possibly allocate the necessary
!           workspace.

            SELECT CASE ( control%final_c_bounds )

            CASE ( TIGHTEST )

               s%prev_control%final_c_bounds = control%final_c_bounds

!              NOTE: there is no need to deallocate the additional workspace
!                    here because the default is not to allocate them.

            CASE ( NON_DEGENERATE )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  changing the status of the final',      &
                       ' bounds on the constraints'
                  WRITE( s%out, * ) '  from TIGHTEST to NON_DEGENERATE'
               END IF
               s%prev_control%final_c_bounds = control%final_c_bounds

!              Now allocate the required additional workspace.

!              1) lower bounds on the constraints

               ALLOCATE( s%c_l2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating c_l2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c_l2(', prob%n, ') allocated'
               s%c_l2 = prob%C_l

!              2) upper bounds on the constraints

               ALLOCATE( s%c_u2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating c_u2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c_u2(', prob%m, ') allocated'
               s%c_u2 = prob%C_u

            CASE ( LOOSEST )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  changing the status of the final',      &
                       ' bounds on the constraints'
                  WRITE( s%out, * ) '  from TIGHTEST to LOOSEST'
               END IF
               s%prev_control%final_c_bounds = control%final_c_bounds

!              Now allocate the required additional workspace.

!              1) lower bounds on the constraints

               ALLOCATE( s%c_l2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating c_l2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c_l2(', prob%n, ') allocated'
               s%c_l2 = prob%C_l

!              2) upper bounds on the constraints

               ALLOCATE( s%c_u2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating c_u2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    c_u2(', prob%m, ') allocated'
               s%c_u2 = prob%C_u

!           Change allowed, but erroneous value specified.

            CASE DEFAULT
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                       ' attempt to change the final status of the bounds'
                  WRITE( s%out, *)'  on the constraints to the unknown value', &
                       control%final_c_bounds
                  SELECT CASE ( s%prev_control%final_c_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                       '  constraints kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                       ' constraints kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                       ' constraints kept to LOOSEST'
                  END SELECT
               END IF
               control%final_c_bounds = s%prev_control%final_c_bounds
            END SELECT

!        Illegal attempt to change the status of the final bounds (that is
!        later than after INITIALIZE)

         ELSE
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( control%final_c_bounds )
               CASE ( TIGHTEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the constraints'
                  SELECT CASE ( s%prev_control%final_c_bounds )
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) '  from NON_DEGENERATE to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' constraints kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) '  from LOOSEST to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' constraints kept to LOOSEST'
                  END SELECT
               CASE ( NON_DEGENERATE )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the constraints'
                  SELECT CASE ( s%prev_control%final_c_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) '  from TIGHTEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' constraints kept to TIGHTEST'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) '  from LOOSEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' constraints kept to LOOSEST'
                  END SELECT
               CASE ( LOOSEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the constraints'
                  SELECT CASE ( s%prev_control%final_c_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) '  from TIGHTEST to LOOSEST'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' constraints kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) '  from NON_DEGENERATE to LOOSEST'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' constraints kept to NON_DEGENERATE'
                  END SELECT
               CASE DEFAULT
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                       ' attempt to change the final status of the bounds'
                  WRITE( s%out, * ) '   on the multipliers to the unknown',    &
                       ' value', control%final_c_bounds
                  SELECT CASE ( s%prev_control%final_c_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' constraints kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' constraints kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' constraints kept to LOOSEST'
                  END SELECT
               END SELECT
            END IF
            control%final_c_bounds = s%prev_control%final_c_bounds
         END IF

!     No change in the status of the final bounds has been requested.

      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%final_c_bounds )
         CASE ( TIGHTEST )
            WRITE( s%out, * ) '  status of the final bounds on the',           &
                 ' constraints is TIGHTEST'
         CASE ( NON_DEGENERATE )
            WRITE( s%out, * ) '  status of the final bounds on the',           &
                 ' constraints is NON_DEGENERATE'
         CASE ( LOOSEST )
            WRITE( s%out, * ) '  status of the final bounds on the',           &
                 ' constraints is LOOSEST'
         END SELECT
      END IF

!     The multipliers

      IF ( control%final_y_bounds /= s%prev_control%final_y_bounds .AND.       &
           prob%m > 0                                                ) THEN

!        Allowed change after INITIALIZE

         IF ( s%stage == READY ) THEN

            SELECT CASE ( control%final_y_bounds )

            CASE ( TIGHTEST )

               s%prev_control%final_y_bounds = control%final_y_bounds

!              NOTE: there is no need to deallocate the additional workspace
!                    here because the default is not to allocate them.

            CASE ( NON_DEGENERATE )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  changing the status of the final',      &
                       ' bounds on the multipliers'
                  WRITE( s%out, * ) '  from TIGHTEST to NON_DEGENERATE'
               END IF
               s%prev_control%final_y_bounds = control%final_y_bounds

!              Now allocate the required additional workspace

!              1) lower bounds on the multipliers

               ALLOCATE( s%y_l2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating y_l2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    y_l2(', prob%n, ') allocated'
               s%y_l2 = prob%Y_l

!              2) upper bounds on the dual variables

               ALLOCATE( s%y_u2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating y_u2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    y_u2(', prob%m, ') allocated'
               s%y_u2 = prob%Y_u

            CASE ( LOOSEST )

               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  changing the status of the final',      &
                       ' bounds on the multipliers'
                  WRITE( s%out, * ) '  from TIGHTEST to LOOSEST'
               END IF
               s%prev_control%final_y_bounds = control%final_y_bounds

!              Now allocate the required additional workspace

!              1) lower bounds on the multipliers

               ALLOCATE( s%y_l2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating y_l2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    y_l2(', prob%n, ') allocated'
               s%y_l2 = prob%Y_l

!              2) upper bounds on the dual variables

               ALLOCATE( s%y_u2( prob%m ), STAT = iostat )
               IF ( iostat /= 0 ) THEN
                  inform%status = MEMORY_FULL
                  WRITE( inform%message( 1 ), * )                              &
                       ' PRESOLVE ERROR: no memory left for allocating y_u2(', &
                       prob%m, ')'
                  RETURN
               END IF
               IF ( s%level >= DEBUG ) WRITE( s%out, * )                       &
                  '    y_u2(', prob%m, ') allocated'
               s%y_u2 = prob%Y_u

!           Change allowed, but erroneous value specified.

            CASE DEFAULT
               IF ( s%level >= ACTION ) THEN
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                       ' attempt to change the final status of the bounds'
                  WRITE( s%out, *)'  on the multipliers to the unknown value', &
                       control%final_y_bounds
                  SELECT CASE ( s%prev_control%final_y_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                       '  multipliers kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                       ' multipliers kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                       ' multipliers kept to LOOSEST'
                  END SELECT
               END IF
               control%final_y_bounds = s%prev_control%final_y_bounds
            END SELECT

!        Illegal attempt to change the status of the final bounds (that is
!        later than after INITIALIZE).

         ELSE
            IF ( s%level >= ACTION ) THEN
               SELECT CASE ( control%final_y_bounds )
               CASE ( TIGHTEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the multipliers'
                  SELECT CASE ( s%prev_control%final_y_bounds )
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) '  from NON_DEGENERATE to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' multipliers kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) '  from LOOSEST to TIGHTEST'
                     WRITE( s%out, * ) '  final status of the bounds on the',  &
                          ' multipliers kept to LOOSEST'
                  END SELECT
               CASE ( NON_DEGENERATE )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the multipliers'
                  SELECT CASE ( s%prev_control%final_y_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) '  from TIGHTEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' multipliers kept to TIGHTEST'
                  CASE ( LOOSEST )
                     WRITE( s%out, * ) '  from LOOSEST to NON_DEGENERATE'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' multipliers kept to LOOSEST'
                  END SELECT
               CASE ( LOOSEST )
                  WRITE( s%out, * ) '  PRESOLVE WARNING: attempt to change',   &
                       ' the final status of the bounds on the multipliers'
                  SELECT CASE ( s%prev_control%final_y_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * ) '  from TIGHTEST to LOOSEST'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' multipliers kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * ) '  from NON_DEGENERATE to LOOSEST'
                     WRITE( s%out, * ) '  status of the final bounds on the',  &
                          ' multipliers kept to NON_DEGENERATE'
                  END SELECT
               CASE DEFAULT
                  WRITE( s%out, * ) '  PRESOLVE WARNING:',                     &
                       ' attempt to change the final status of the bounds'
                  WRITE( s%out, * ) '   on the multipliers to the unknown',    &
                       ' value', control%final_y_bounds
                  SELECT CASE ( s%prev_control%final_y_bounds )
                  CASE ( TIGHTEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' multipliers kept to TIGHTEST'
                  CASE ( NON_DEGENERATE )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' multipliers kept to NON_DEGENERATE'
                  CASE ( LOOSEST )
                     WRITE( s%out, * )'  final status of the bounds on the',   &
                          ' multipliers kept to LOOSEST'
                  END SELECT
               END SELECT
            END IF
            control%final_y_bounds = s%prev_control%final_y_bounds
         END IF

!     No change in the status of the final bounds has been requested.

      ELSE IF ( s%level >= ACTION ) THEN
         SELECT CASE ( control%final_y_bounds )
         CASE ( TIGHTEST )
            WRITE( s%out, * ) '  status of the final bounds on the',           &
                 ' multipliers is TIGHTEST'
         CASE ( NON_DEGENERATE )
            WRITE( s%out, * ) '  status of the final bounds on the',           &
                 ' multipliers is NON_DEGENERATE'
         CASE ( LOOSEST )
            WRITE( s%out, * ) '  status of the final bounds on the',           &
                 ' multipliers is LOOSEST'
         END SELECT
      END IF

!     Overide dual transformations parameters, if needed.

      IF ( .NOT. control%dual_transformations ) THEN
         control%dual_constraints_freq = 0

      END IF

      RETURN

      END SUBROUTINE PRESOLVE_revise_control

!==============================================================================
!==============================================================================

      LOGICAL FUNCTION PRESOLVE_is_too_big( val )

!     Decides whether or not val is too large (in absolute value)
!     compared to the maximum acceptable value for reduced problem data.

!     Argument:

      REAL ( KIND = wp ), INTENT( IN ) :: val

!             the value tested for being too large.

!     Programming: Ph. Toint, July 2002

!==============================================================================

      IF ( val <= M_INFINITY .OR. val >= P_INFINITY ) THEN
         PRESOLVE_is_too_big = .FALSE.
      ELSE
         PRESOLVE_is_too_big = ABS( val ) > MAX_GROWTH
      END IF

      RETURN

      END FUNCTION PRESOLVE_is_too_big

!==============================================================================
!==============================================================================

      LOGICAL FUNCTION PRESOLVE_is_pos( val )

!     Decides whether or not val is positive within problem dependent accuracy.

!     Argument:

      REAL ( KIND = wp ), INTENT( IN ) :: val

!             the value tested for being positive.

!     Programming: Ph. Toint, November 2000

!==============================================================================

      PRESOLVE_is_pos = val >= ACCURACY

      RETURN

      END FUNCTION PRESOLVE_is_pos

!==============================================================================
!==============================================================================

      LOGICAL FUNCTION PRESOLVE_is_neg( val )

!     Decides whether or not val is negative within problem dependent accuracy.

!     Argument:

      REAL ( KIND = wp ), INTENT( IN ) :: val

!             the value tested for being negative.

!     Programming: Ph. Toint, November 2000

!==============================================================================

      PRESOLVE_is_neg = val <= - ACCURACY

      RETURN

      END FUNCTION PRESOLVE_is_neg

!==============================================================================
!==============================================================================

      LOGICAL FUNCTION PRESOLVE_is_zero( val )

!     Decides whether or not val is zero within problem dependent accuracy.

!     Argument:

      REAL ( KIND = wp ), INTENT( IN ) :: val

!             the value tested for being zero.

!     Programming: Ph. Toint, November 2000

!==============================================================================

      IF ( val < ZERO ) THEN
         PRESOLVE_is_zero = val >= - ACCURACY
      ELSE
         PRESOLVE_is_zero = val <=   ACCURACY
      END IF

      RETURN

      END FUNCTION PRESOLVE_is_zero

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_swap( n, x, y )

!     Swaps the contents of the vectors x( 1:n ) and y( 1:n ).
!     Note that it can be replaced by the corresponding BLAS1 routine
!     (with increments equal to 1).

!     Arguments:

      INTEGER, INTENT( IN ) :: n

!             the size of the vectors to swap

      REAL ( KIND = wp ), INTENT( INOUT ) :: x( n ), y( n )

!             the two vectors whose contents are to be swapped.

!     Programming: Ph. Toint, November 2000

!==============================================================================

!     Local variables

      INTEGER            :: i
      REAL ( KIND = wp ) :: tmp

      DO i = 1, n
         tmp    = x( i )
         x( i ) = y( i )
         y( i ) = tmp
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_swap

!==============================================================================
!==============================================================================

      SUBROUTINE PRESOLVE_compute_q( prob )

!     Compute the value of the objective function, given the current values
!     of the variables.

!     Argument:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: i, j, k
      REAL ( KIND = wp ) :: xi

      prob%q = prob%f

      DO i = 1, prob%n
         IF ( prob%X_status( i ) <= ELIMINATED ) CYCLE
         xi = prob%X( i )
         prob%q = prob%q + xi * prob%G( i )
         IF ( prob%H%ne == 0 ) CYCLE
         DO k = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            j = prob%H%col( k )
            IF ( i == j ) THEN
               prob%q = prob%q + HALF * prob%H%val( k ) * xi * xi
            ELSE
               IF ( prob%X_status( j ) <= ELIMINATED ) CYCLE
               prob%q = prob%q + xi * prob%H%val( k ) * prob%X( j )
            END IF
         END DO
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_compute_q

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_compute_c( unreduced, prob, s )

!     Compute the value of the constraints, given the current values
!     of the variables.
!
!     Argument:

      LOGICAL, INTENT( IN ) :: unreduced

!             .TRUE. if the problem is in its unreduced form (that is if the
!             eliminated variables must be filtered out and if merged rows
!             can be present).
!
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER            :: i, j, k, ic
      REAL ( KIND = wp ) :: a, ci

!     Loop on the constraints.

      DO i = 1, prob%m
         ci = ZERO
         ic = i
         DO
            DO k = prob%A%ptr( ic ), prob%A%ptr( ic + 1 ) - 1
               j = prob%A%col( k )
               IF ( unreduced .AND. prob%X_status( j ) <= ELIMINATED ) CYCLE
               a = prob%A%val( k )
               IF ( a == ZERO ) CYCLE
               ci = ci + a * prob%X( j )
            END DO
            IF ( .NOT. unreduced ) EXIT
            ic = s%conc( ic )
            IF ( ic == END_OF_LIST ) EXIT
         END DO
         prob%C( i ) = ci
      END DO

      RETURN

      END SUBROUTINE PRESOLVE_compute_c

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_open_h( status, control, inform, s )

!     Opens the transformation file filename for writing, if not already
!     opened. If the file already exists, it is checked for consistency
!     with the current problem (using the file checksums).

!     Arguments:

      CHARACTER( LEN = 7 ), INTENT( IN ) :: status

!              the associated status, meaning:
!              'REPLACE' : the file must replace a possibly existing file
!                          with the same name,
!              'OLD    ' : the file is supposed to exist.

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

!              the PRESOLVE control structure (see above)

      TYPE ( PRESOLVE_inform_type ), INTENT( INOUT ) :: inform

!              the PRESOLVE exit information structure (see above)

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!
!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER           :: iostat, ic1, ic2, ic3
      LOGICAL           :: opened
      REAL( KIND = wp ) :: rc


      INQUIRE( UNIT = control%transf_file_nbr, OPENED = opened )

!     Open the file.

      IF ( .NOT. opened ) THEN
         IF ( s%level >= DEBUG ) WRITE( s%out, * ) '    opening file ',        &
                                      control%transf_file_name
         OPEN( UNIT     = control%transf_file_nbr,                             &
               FILE     = control%transf_file_name,                            &
               ACCESS   = 'DIRECT',                                            &
               RECL     = s%recl, IOSTAT = iostat,                             &
               STATUS   = TRIM( status ) )
         IF ( iostat > 0 ) THEN
            inform%status = FILE_NOT_OPENED
            WRITE( inform%message( 1 ), * )                                    &
                 ' PRESOLVE ERROR: could not open file',                       &
                 TRIM( control%transf_file_name ), ' as unit',                 &
                 control%transf_file_nbr
            RETURN
         END IF

!        Process the checksums.

         IF ( status == 'REPLACE' ) THEN
            WRITE( control%transf_file_nbr, REC = 1 )                          &
                 s%icheck1, s%icheck2, s%icheck3, s%rcheck
         ELSE
            READ( control%transf_file_nbr, REC = 1 ) ic1, ic2, ic3, rc
            IF ( ic1 /= s%icheck1 .OR. ic2 /= s%icheck2 .OR. &
                 ic3 /= s%icheck3 .OR. rc  /= s%rcheck       ) THEN
               inform%status = CORRUPTED_SAVE_FILE
               WRITE( inform%message( 1 ), * )                                 &
                    ' PRESOLVE ERROR: file ', TRIM( control%transf_file_name ),&
                    ' has been corrupted'
               WRITE( inform%message( 2 ), * )                                 &
                    '    since the last call to PRESOLVE'
               RETURN
            END IF
         END IF
      END IF

      RETURN

      END SUBROUTINE PRESOLVE_open_h

!===============================================================================
!===============================================================================

      SUBROUTINE PRESOLVE_write_full_prob( prob, control, s )

!     Writes the current full problem, including eliminated quantities
!     (for debugging) and relaxed bounds (NON_DEGENERATE or LOOSEST) when
!     applicable.
!
!     Arguments:

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob

!              the problem to which PRESOLVE is to be applied

      TYPE ( PRESOLVE_control_type ), INTENT( INOUT ) :: control

      TYPE ( PRESOLVE_data_type ), INTENT( INOUT ) :: s

!              the PRESOLVE saved information structure (see above)

!     Programming: Ph. Toint, November 2000.
!
!===============================================================================

!     Local variables

      INTEGER :: i, j, k, ic, n, m
      LOGICAL :: show_inactive
      REAL ( KIND = wp ) :: aij

      WRITE( s%out, * ) ' '
      WRITE( s%out, * ) '    =============== PROBLEM ===================='
      WRITE( s%out, * ) ' '

!     Determine the dimensions for printout.

      show_inactive = s%level >= DEBUG
      IF ( show_inactive ) THEN
         n = prob%n
         m = prob%m
      ELSE
         n = s%n_active
         m = s%m_active
      END IF

!     Write the variables.

      WRITE( s%out, * ) '    n = ', n

      IF ( show_inactive .AND. s%level >= CRAZY ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '     status of the variables'
         WRITE( s%out, * ) '      ', prob%X_status( 1:prob%n )
      END IF

      IF ( n > 0 ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '    variables '
         WRITE( s%out, * ) ' '
         IF ( control%final_x_bounds == TIGHTEST ) THEN
            WRITE( s%out, * )'      j           lower       actual      upper '
            WRITE( s%out, * ) ' '
            DO j = 1, prob%n
               SELECT CASE ( prob%X_status( j ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 100 )                                          &
                       j, prob%X_l( j ), prob%X( j ), prob%X_u( j )
               CASE ( INACTIVE )
                  IF ( show_inactive ) WRITE( s%out, 101 )                     &
                     j, prob%X_l( j ), prob%X( j ), prob%X_u( j )
               CASE ( ELIMINATED )
                  IF ( show_inactive ) WRITE( s%out, 102 )                     &
                     j, prob%X_l( j ), prob%X( j ), prob%X_u( j )
               END SELECT
            END DO
         ELSE
            WRITE( s%out, * )                                                  &
              '      j        lower-nd    lower    actual     upper    upper-nd'
            WRITE( s%out, * ) ' '
            DO j = 1, prob%n
               SELECT CASE ( prob%X_status( j ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 105 )                                          &
                       j, s%x_l2(j), prob%X_l(j),prob%X(j),prob%X_u(j),s%x_u2(j)
               CASE ( INACTIVE )
                  IF ( show_inactive ) WRITE( s%out, 106 )                     &
                     j, s%x_l2(j), prob%X_l(j), prob%X(j), prob%X_u(j),s%x_u2(j)
               CASE ( ELIMINATED )
                  IF ( show_inactive ) WRITE( s%out, 107 )                     &
                     j, s%x_l2(j), prob%X_l(j), prob%X(j), prob%X_u(j),s%x_u2(j)
               END SELECT
            END DO
         END IF

!        Write the dual variables.

         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '    z multipliers '
         WRITE( s%out, * ) ' '
         IF ( control%final_z_bounds == TIGHTEST ) THEN
            WRITE( s%out, * )'      j           lower       actual      upper '
            WRITE( s%out, * ) ' '
            DO j = 1, prob%n
               SELECT CASE ( prob%X_status( j ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 200 )                                          &
                       j, prob%Z_l( j ), prob%Z( j ), prob%Z_u( j )
               CASE ( INACTIVE )
                  IF ( show_inactive ) THEN
                     WRITE( s%out, 201 )                                       &
                       j, prob%Z_l( j ), prob%Z( j ), prob%Z_u( j )
                  END IF
               CASE ( ELIMINATED )
                  IF ( show_inactive ) THEN
                     WRITE( s%out, 202 )                                       &
                       j, prob%Z_l( j ), prob%Z( j ), prob%Z_u( j )
                  END IF
               END SELECT
            END DO
         ELSE
            WRITE( s%out, * )                                                  &
              '      j        lower-nd    lower    actual     upper    upper-nd'
            WRITE( s%out, * ) ' '
            DO j = 1, prob%n
               SELECT CASE ( prob%X_status( j ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 205 )                                          &
                       j, s%z_l2(j), prob%Z_l(j),prob%Z(j),prob%Z_u(j),s%z_u2(j)
               CASE ( INACTIVE )
                  IF ( show_inactive ) THEN
                     WRITE( s%out, 206 )                                       &
                       j, s%z_l2(j), prob%Z_l(j),prob%Z(j),prob%Z_u(j),s%z_u2(j)
                  END IF
               CASE ( ELIMINATED )
                  IF ( show_inactive ) THEN
                     WRITE( s%out, 207 )                                       &
                       j, s%z_l2(j), prob%Z_l(j),prob%Z(j),prob%Z_u(j),s%z_u2(j)
                  END IF
               END SELECT
            END DO
         END IF
      END IF

!    Write the constraints.

      WRITE( s%out, * ) ' '
      WRITE( s%out, * ) '    m = ', m

      IF ( show_inactive .AND. s%level >= CRAZY ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '     status of the constraints'
         WRITE( s%out, * ) '      ', prob%C_status( 1:prob%m )
      END IF

      IF ( m > 0 ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '    constraints '
         WRITE( s%out, * ) ' '
         IF ( control%final_c_bounds == TIGHTEST ) THEN
            WRITE( s%out, * )'      i           lower       actual      upper '
            WRITE( s%out, * ) ' '
            DO i = 1, prob%m
               SELECT CASE ( prob%C_status( i ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 300 )                                          &
                       i, prob%C_l( i ), prob%C( i ), prob%C_u( i )
               CASE ( INACTIVE )
                  IF ( show_inactive ) WRITE( s%out, 301 )                     &
                     i, prob%C_l( i ), prob%C( i ), prob%C_u( i )
               CASE ( ELIMINATED )
                  IF ( show_inactive ) WRITE( s%out, 303 )                     &
                     i, prob%C_l( i ), prob%C( i ), prob%C_u( i )
               END SELECT
            END DO
         ELSE
            WRITE( s%out, * )                                                  &
              '      i        lower-nd    lower    actual     upper    upper-nd'
            WRITE( s%out, * ) ' '
            DO i = 1, prob%m
               SELECT CASE ( prob%C_status( i ) )
               CASE ( ACTIVE:FREE )
                 WRITE( s%out, 305 )                                           &
                      i, s%c_l2(i), prob%C_l(i), prob%C(i),prob%C_u(i),s%c_u2(i)
               CASE ( INACTIVE )
                  IF ( show_inactive ) WRITE( s%out, 306 )                     &
                       i, s%c_l2(i), prob%C_l(i),prob%C(i),prob%C_u(i),s%c_u2(i)
               CASE ( ELIMINATED )
                  IF ( show_inactive ) WRITE( s%out, 308 )                     &
                       i, s%c_l2(i), prob%C_l(i),prob%C(i),prob%C_u(i),s%c_u2(i)
               END SELECT
            END DO
         END IF

!        Write the multipliers.

         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '    y multipliers '
         WRITE( s%out, * ) ' '
         IF ( control%final_y_bounds == TIGHTEST ) THEN
            WRITE( s%out, * )'      i           lower       actual      upper '
            WRITE( s%out, * ) ' '
            DO i = 1, prob%m
               SELECT CASE ( prob%C_status( i ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 400 )                                          &
                       i, prob%Y_l( i ), prob%Y( i ), prob%Y_u( i )
               CASE ( INACTIVE )
                  IF ( show_inactive ) WRITE( s%out, 401 )                     &
                          i, prob%Y_l( i ), prob%Y( i ), prob%Y_u( i )
               CASE ( ELIMINATED )
                  IF ( show_inactive ) WRITE( s%out, 403 )                     &
                          i, prob%Y_l( i ), prob%Y( i ), prob%Y_u( i )
               END SELECT
            END DO
         ELSE
            WRITE( s%out, * )                                                  &
              '      j        lower-nd    lower    actual     upper    upper-nd'
            WRITE( s%out, * ) ' '
            DO i = 1, prob%m
               SELECT CASE ( prob%C_status( i ) )
               CASE ( ACTIVE:FREE )
                  WRITE( s%out, 405 )                                          &
                       i, s%y_l2(i),prob%Y_l(i), prob%Y(i),prob%Y_u(i),s%y_u2(i)
               CASE ( INACTIVE )
                  IF ( show_inactive ) WRITE( s%out, 406 )                     &
                       i, s%y_l2(i),prob%Y_l(i), prob%Y(i),prob%Y_u(i),s%y_u2(i)
               CASE ( ELIMINATED )
                  IF ( show_inactive ) WRITE( s%out, 408 )                     &
                       i, s%y_l2(i),prob%Y_l(i), prob%Y(i),prob%Y_u(i),s%y_u2(i)
               END SELECT
            END DO
         END IF

!        Write the Jacobian.

         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '    Jacobian '
         WRITE( s%out, * ) ' '

         DO i = 1, prob%m
            DO k = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
               j   = prob%A%col( k )
               aij = prob%A%val( k )
               IF ( prob%C_status( i ) > ELIMINATED .AND. &
                    prob%X_status( j ) > ELIMINATED       ) THEN
                  WRITE( s%out, 600 ) i, j, aij
               ELSE IF ( show_inactive ) THEN
                  SELECT CASE ( prob%C_status( i ) )
                  CASE ( INACTIVE )
                     WRITE( s%out, 601 ) i, j, aij
                  CASE ( ELIMINATED )
                     WRITE( s%out, 603 ) i, j, aij
                  CASE ( ACTIVE )
                     SELECT CASE ( prob%X_status( j ) )
                     CASE ( INACTIVE )
                        WRITE( s%out, 601 ) i, j, aij
                     CASE ( ELIMINATED )
                        WRITE( s%out, 603 ) i, j, aij
                     END SELECT
                  END SELECT
               END IF
            END DO
            ic = s%conc( i )
            IF ( ic == END_OF_LIST ) CYCLE
            WRITE( s%out, * ) '    row', ic,                                   &
                 'concatenated at the end of row', i
         END DO

      END IF

!     Write the objective function.

      WRITE( s%out, * ) ' '
      WRITE( s%out, * ) '    objective function '
      WRITE( s%out, * ) ' '
      WRITE( s%out, "( 4x, ES12.4 )" ) prob%q

      WRITE( s%out, * ) ' '
      WRITE( s%out, * ) '    constant term '
      WRITE( s%out, * ) ' '
      WRITE( s%out, "( 4x, ES12.4 )" ) prob%f

!     Write the gradient.

      IF ( n > 0 ) THEN
         WRITE( s%out, * ) ' '
         WRITE( s%out, * ) '      j          gradient '
         WRITE( s%out, * ) ' '
         DO j = 1, prob%n
            SELECT CASE ( prob%X_status( j ) )
            CASE ( ACTIVE:FREE )
               WRITE( s%out, 500 ) j, prob%G( j )
            CASE ( INACTIVE )
               IF ( show_inactive ) WRITE( s%out, 501 ) j, prob%G( j )
            CASE ( ELIMINATED )
               IF ( show_inactive ) WRITE( s%out, 502 ) j, prob%G( j )
            END SELECT
         END DO

!        Write the Hessian.

         IF ( prob%H%ne > 0 ) THEN

            WRITE( s%out, * ) ' '
            WRITE( s%out, * ) '    Hessian '
            WRITE( s%out, * ) ' '

            DO i = 1, prob%n
               DO k = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                  j = prob%H%col( k )
                  IF ( prob%X_status(j) > ELIMINATED .AND. &
                       prob%X_status(i) > ELIMINATED       ) THEN
                     WRITE( s%out, 700 ) i, j, prob%H%val( k )
                  ELSE IF ( show_inactive ) THEN
                     IF ( prob%X_status(i) == INACTIVE .OR. &
                          prob%X_status(j) == INACTIVE      ) THEN
                        WRITE( s%out, 701 ) i, j, prob%H%val( k )
                     ELSE
                        WRITE( s%out, 702 ) i, j, prob%H%val( k )
                     END IF
                  END IF
               END DO
            END DO

         END IF
      END IF

      WRITE( s%out, * ) ' '
      WRITE( s%out, * ) '    ============ END OF PROBLEM =================='
      WRITE( s%out, * ) ' '

      RETURN

!     Formats

100   FORMAT( 3x, 'x(', i4, ') =', 3x, 3ES12.4 )
101   FORMAT( 3x, 'x(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
102   FORMAT( 3x, 'x(', i4, ') =', 3x, 3ES12.4, 3x, 'eliminated' )
105   FORMAT( 3x, 'x(', i4, ') =', 3x, 5ES10.3 )
106   FORMAT( 3x, 'x(', i4, ') =', 3x, 5ES10.3, 3x, 'inactive'   )
107   FORMAT( 3x, 'x(', i4, ') =', 3x, 5ES10.3, 3x, 'eliminated' )
200   FORMAT( 3x, 'z(', i4, ') =', 3x, 3ES12.4 )
201   FORMAT( 3x, 'z(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
202   FORMAT( 3x, 'z(', i4, ') =', 3x, 3ES12.4, 3x, 'eliminated' )
205   FORMAT( 3x, 'z(', i4, ') =', 3x, 5ES10.3 )
206   FORMAT( 3x, 'z(', i4, ') =', 3x, 5ES10.3, 3x, 'inactive'   )
207   FORMAT( 3x, 'z(', i4, ') =', 3x, 5ES10.3, 3x, 'eliminated' )
300   FORMAT( 3x, 'c(', i4, ') =', 3x, 3ES12.4 )
301   FORMAT( 3x, 'c(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
303   FORMAT( 3x, 'c(', i4, ') =', 3x, 3ES12.4, 3x, 'eliminated' )
305   FORMAT( 3x, 'c(', i4, ') =', 3x, 5ES10.3 )
306   FORMAT( 3x, 'c(', i4, ') =', 3x, 5ES10.3, 3x, 'inactive'   )
308   FORMAT( 3x, 'c(', i4, ') =', 3x, 5ES10.3, 3x, 'eliminated' )
400   FORMAT( 3x, 'y(', i4, ') =', 3x, 3ES12.4 )
401   FORMAT( 3x, 'y(', i4, ') =', 3x, 3ES12.4, 3x, 'inactive'   )
403   FORMAT( 3x, 'y(', i4, ') =', 3x, 3ES12.4, 3x, 'eliminated' )
405   FORMAT( 3x, 'y(', i4, ') =', 3x, 5ES10.3 )
406   FORMAT( 3x, 'y(', i4, ') =', 3x, 5ES10.3, 3x, 'inactive'   )
408   FORMAT( 3x, 'y(', i4, ') =', 3x, 5ES10.3, 3x, 'eliminated' )
500   FORMAT( 3x, 'g(', i4, ') =', 3x,  ES12.4 )
501   FORMAT( 3x, 'g(', i4, ') =', 3x,  ES12.4, 3x, 'inactive'   )
502   FORMAT( 3x, 'g(', i4, ') =', 3x,  ES12.4, 3x, 'eliminated' )
600   FORMAT( 3x, 'A(', i4, ',', i4, ') = ', ES12.4 )
601   FORMAT( 3x, 'A(', i4, ',', i4, ') = ', ES12.4, 3x, 'inactive'   )
603   FORMAT( 3x, 'A(', i4, ',', i4, ') = ', ES12.4, 3x, 'eliminated' )
700   FORMAT( 3x, 'H(', i4, ',', i4, ') = ', ES12.4 )
701   FORMAT( 3x, 'H(', i4, ',', i4, ') = ', ES12.4, 3x, 'inactive'   )
702   FORMAT( 3x, 'H(', i4, ',', i4, ') = ', ES12.4, 3x, 'eliminated' )

      END SUBROUTINE PRESOLVE_write_full_prob

!===============================================================================
!===============================================================================
!===============================================================================

!  End of module PRESOLVE

   END MODULE GALAHAD_PRESOLVE_double

!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                       *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*   END GALAHAD PRESOLVE  M O D U L E   *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*                                       *-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
!-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
