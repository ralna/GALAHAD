! THIS VERSION: GALAHAD 3.3 - 27/01/2020 AT 10:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ L P A    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   essentially an interface to John Reid's HSL code LA04
!   originally released in GALAHAD Version 3.1. October 7th 2018

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_LPA_double

!     ------------------------------------------------
!     |                                              |
!     | Minimize the linear objective function       |
!     |                                              |
!     |              g^T x + f                       |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |           c_l <= A x <= c_u                  |
!     |           x_l <=  x <= x_u                   |
!     |                                              |
!     | using an active-set (simplex) method         |
!     |                                              |
!     ------------------------------------------------

!$    USE omp_lib
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING, ONLY: STRING_pleural, STRING_verb_pleural,           &
                                       STRING_ies, STRING_are, STRING_ordinal
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_RPD_double, ONLY: RPD_inform_type, RPD_write_qp_problem_data
      USE GALAHAD_QPD_double, ONLY: QPD_SIF

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: LPA_initialize, LPA_read_specfile, LPA_solve, LPA_terminate,   &
                QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: a_coordinate = 0
      INTEGER, PARAMETER :: a_sparse_by_rows = 1
      INTEGER, PARAMETER :: a_dense = 2
      INTEGER, PARAMETER :: min_real_factor_size_default = 10000
      INTEGER, PARAMETER :: min_integer_factor_size_default = 20000
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LPA_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level
!    (>= 2 turns on LA)4 output)

        INTEGER :: print_level = 0

!   any printing will start on this iteration

        INTEGER :: start_print = - 1

!   any printing will stop on this iteration

        INTEGER :: stop_print = - 1

!   at most maxit inner iterations are allowed

        INTEGER :: maxit = 1000

!  maximum number of iterative refinements allowed

       INTEGER :: max_iterative_refinements = 0

!  initial size for real array for the factors and other data

       INTEGER :: min_real_factor_size = min_real_factor_size_default

!  initial size for integer array for the factors and other data

       INTEGER :: min_integer_factor_size = min_integer_factor_size_default

!  the initial seed used when generating random numbers

       INTEGER :: random_number_seed = 0

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 52

!    specifies the unit number to write generated QPLIB file describing the
!     current problem

        INTEGER :: qplib_file_device = 53

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!  the tolerable relative perturbation of the data (A,g,..) defining the problem

!      REAL ( KIND = wp ) :: tol_data = epsmch ** ( 2.0_wp / 3.0_wp )
       REAL ( KIND = wp ) :: tol_data = ten ** ( - 10 )

!   any constraint violated by less than feas_tol will be considered to be
!    satisfied

!      REAL ( KIND = wp ) :: feas_tol =  epsmch ** ( 2.0_wp / 3.0_wp )
       REAL ( KIND = wp ) :: feas_tol =  ten ** ( - 10 )

!  pivot threshold used to control the selection of pivot elements in the
!   matrix factorization. Any potential pivot which is less than the largest
!   entry in its row times the threshold is excluded as a candidate

       REAL ( KIND = wp ) :: relative_pivot_tolerance = 0.1_wp

!  limit to control growth in the upated basis factors. A refactorization
!   occurs if the growth exceeds this limit

       REAL ( KIND = wp ) :: growth_limit = one / SQRT( epsmch )

!  any entry in the basis smaller than this is considered zero

       REAL ( KIND = wp ) :: zero_tolerance = epsmch

!  any solution component whose change is smaller than a tolerence times the
!   largest change may be considered to be zero

!      REAL ( KIND = wp ) :: change_tolerance = epsmch ** ( 2.0_wp / 3.0_wp )
       REAL ( KIND = wp ) :: change_tolerance = ten ** ( - 10 )

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

       REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!   the maximum CPU time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

       REAL ( KIND = wp ) :: clock_time_limit = - one

!   if %scale is true, the problem will be automatically scaled prior to
!    solution. This may improve computation time and accuracy

       LOGICAL :: scale = .FALSE.

!  should the dual problem be solved rather than the primal?

       LOGICAL :: dual = .FALSE.

!  should a warm start using the data in C_stat and X_stat be attempted?

       LOGICAL :: warm_start = .FALSE.

!  should steepest-edge weights be used to detetrmine the variable
!   leaving the basis?

       LOGICAL :: steepest_edge = .TRUE.

!   if %space_critical is true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!   if %generate_qplib_file is .true. if a QPLIB file describing the current
!    problem is to be generated

        LOGICAL :: generate_qplib_file = .FALSE.

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "LPAPROB.SIF"  // REPEAT( ' ', 18 )

!  name of generated QPLIB file containing input problem

        CHARACTER ( LEN = 30 ) :: qplib_file_name =                            &
         "LPAPROB.qplib"  // REPEAT( ' ', 16 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

      END TYPE LPA_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LPA_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

        REAL ( KIND = wp ) :: clock_preprocess = 0.0

      END TYPE LPA_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: LPA_inform_type

!  return status. See LPA_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the total number of iterations required

        INTEGER :: iter = - 1

!  the final value of la04's job argument

        INTEGER :: la04_job = 0

!  any extra information from an unsuccesfull call to la04 (la04's RINFO(35))

        INTEGER :: la04_job_info = 0

!  the value of the objective function at the best estimate of the solution
!   determined by LPA_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  the information array from la04

        REAL ( KIND = wp ), DIMENSION( 40 ) :: RINFO

!  timings (see above)

        TYPE ( LPA_time_type ) :: time

!  inform parameters for RPD

        TYPE ( RPD_inform_type ) :: RPD_inform

      END TYPE LPA_inform_type

!  - - - - - - - - - -
!   data derived type
!  - - - - - - - - - -

      TYPE, PUBLIC :: LPA_data_type
        INTEGER :: trans
        INTEGER :: m, n, job, kb, lb, lws, liws
        LOGICAL :: tried_to_remove_deps, save_structure, new_problem_structure
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ptr, A_row, IX, JX, IWS
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: x_map, c_map
        REAL ( KIND = wp ) :: f
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_val, B, C
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X, Z, G, CS, WS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND
        REAL ( KIND = wp ), DIMENSION( 15 ) :: CNTL
        TYPE ( LPA_control_type ) :: control
      END TYPE LPA_data_type

      CONTAINS

!-*-*-*-*-*-   L P A _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LPA_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for LPA. This routine should be called before
!  LPA_solve
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

      TYPE ( LPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPA_control_type ), INTENT( OUT ) :: control
      TYPE ( LPA_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  revise control parameters (not all compilers currently support fortran 2013)

      control%tol_data = epsmch ** ( 2.0_wp / 3.0_wp )
      control%feas_tol =  epsmch ** ( 2.0_wp / 3.0_wp )
      control%change_tolerance = epsmch ** ( 2.0_wp / 3.0_wp )

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of LPA_initialize

      END SUBROUTINE LPA_initialize

!-*-*-*-*-   L P A _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE LPA_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by LPA_initialize could (roughly)
!  have been set as:

! BEGIN LPA SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  maximum-number-of-iterations                      1000
!  max-iterative-refinements                         0
!  minimum-real-factor-size                          10000
!  minimum-integer-factor-size                       10000
!  random-number-seed                                0
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  infinity-value                                    1.0D+19
!  tolerable-relative-data-perturbation              1.0D-12
!  feasibility-tolerance                             0.0
!  relative-pivot-tolerance                          0.1
!  growth-limit-tolerance                            1.0D+16
!  zero-basis-entry-tolerance                        1.0D-16
!  change-tolerance                                  1.0D-12
!  identical-bounds-tolerance                        1.0D-15
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  scale-problem-data                                no
!  solve-dual                                        no
!  warm-start                                        no
!  steepest-edge-weights                             yes
!  space-critical                                    no
!  deallocate-error-fatal                            no
!  generate-sif-file                                 no
!  generate-qplib-file                               no
!  sif-file-name                                     LPAPROB.SIF
!  qplib-file-name                                   LPAPROB.qplib
!  output-line-prefix                                ""
! END LPA SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( LPA_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: start_print = print_level + 1
      INTEGER, PARAMETER :: stop_print = start_print + 1
      INTEGER, PARAMETER :: maxit = stop_print + 1
      INTEGER, PARAMETER :: max_iterative_refinements = maxit + 1
      INTEGER, PARAMETER :: min_real_factor_size = max_iterative_refinements + 1
      INTEGER, PARAMETER :: min_integer_factor_size = min_real_factor_size + 1
      INTEGER, PARAMETER :: random_number_seed = min_integer_factor_size + 1
      INTEGER, PARAMETER :: sif_file_device = random_number_seed + 1
      INTEGER, PARAMETER :: qplib_file_device = sif_file_device + 1
      INTEGER, PARAMETER :: infinity = qplib_file_device + 1
      INTEGER, PARAMETER :: tol_data = infinity + 1
      INTEGER, PARAMETER :: feas_tol = tol_data + 1
      INTEGER, PARAMETER :: relative_pivot_tolerance = feas_tol + 1
      INTEGER, PARAMETER :: growth_limit = relative_pivot_tolerance + 1
      INTEGER, PARAMETER :: zero_tolerance = growth_limit + 1
      INTEGER, PARAMETER :: change_tolerance = zero_tolerance + 1
      INTEGER, PARAMETER :: identical_bounds_tol = change_tolerance + 1
      INTEGER, PARAMETER :: cpu_time_limit = identical_bounds_tol + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: scale = clock_time_limit + 1
      INTEGER, PARAMETER :: dual = scale + 1
      INTEGER, PARAMETER :: warm_start = dual + 1
      INTEGER, PARAMETER :: steepest_edge = warm_start + 1
      INTEGER, PARAMETER :: space_critical = steepest_edge + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: generate_qplib_file = generate_sif_file + 1
      INTEGER, PARAMETER :: sif_file_name = generate_qplib_file + 1
      INTEGER, PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER, PARAMETER :: prefix = qplib_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'LPA'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( start_print )%keyword = 'start-print'
      spec( stop_print )%keyword = 'stop-print'
      spec( maxit )%keyword = 'maximum-number-of-iterations'
      spec( max_iterative_refinements )%keyword = 'max-iterative-refinements'
      spec( min_real_factor_size )%keyword = 'minimum-real-factor-size'
      spec( min_integer_factor_size )%keyword = 'minimum-integer-factor-size'
      spec( random_number_seed )%keyword = 'random-number-seed'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( tol_data )%keyword = 'tolerable-relative-data-perturbation'
      spec( feas_tol )%keyword = 'feasibility-tolerance'
      spec( relative_pivot_tolerance )%keyword = 'relative-pivot-tolerance'
      spec( growth_limit )%keyword = 'growth-limit-tolerance'
      spec( zero_tolerance )%keyword = 'zero-basis-entry-tolerance'
      spec( change_tolerance )%keyword = 'change-tolerance'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( scale )%keyword = 'scale-problem-data'
      spec( dual )%keyword = 'solve-dual'
      spec( warm_start )%keyword = 'warm-start'
      spec( steepest_edge )%keyword = 'steepest-edge-weights'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

      IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' lpa: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( start_print ),                          &
                                 control%start_print,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_print ),                           &
                                 control%stop_print,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( maxit ),                                &
                                 control%maxit,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_iterative_refinements ),            &
                                 control%max_iterative_refinements,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_real_factor_size ),                 &
                                 control%min_real_factor_size,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_integer_factor_size ),              &
                                 control%min_integer_factor_size,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( random_number_seed ),                   &
                                 control%random_number_seed,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( sif_file_device ),                      &
                                 control%sif_file_device,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_device ),                    &
                                 control%qplib_file_device,                    &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( tol_data ),                             &
                                 control%tol_data,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( feas_tol ),                             &
                                 control%feas_tol,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( relative_pivot_tolerance ),             &
                                 control%relative_pivot_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( growth_limit ),                         &
                                 control%growth_limit,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_tolerance ),                       &
                                 control%zero_tolerance,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( change_tolerance ),                     &
                                 control%change_tolerance,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                 &
                                 control%identical_bounds_tol,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( cpu_time_limit ),                       &
                                 control%cpu_time_limit,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( clock_time_limit ),                     &
                                 control%clock_time_limit,                     &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( scale ),                                &
                                 control%scale,                                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( dual ),                                 &
                                 control%dual,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( warm_start ),                           &
                                 control%warm_start,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( steepest_edge ),                        &
                                 control%steepest_edge,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( space_critical ),                       &
                                 control%space_critical,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),               &
                                 control%deallocate_error_fatal,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_sif_file ),                    &
                                 control%generate_sif_file,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_qplib_file ),                  &
                                 control%generate_qplib_file,                  &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_name ),                      &
                                 control%qplib_file_name,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

      RETURN

!  end of subroutine LPA_read_specfile

    END SUBROUTINE LPA_read_specfile

!-*-*-*-*-*-*-*-*-*-   L P A _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE LPA_solve( prob, data, control, inform, C_stat, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the linear objective
!
!                   g^T x + f
!
!  where
!
!        (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!  and   (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A is sparse.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  prob is a structure of type QPT_problem_type, whose components hold
!   information about the problem on input, and its solution on output.
!   The following components must be set:
!
!   %new_problem_structure is a LOGICAL variable, which must be set to
!    .TRUE. by the user if this is the first problem with this "structure"
!    to be solved since the last call to LPA_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero, and the analytic centre of the
!        feasible region will be found. %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %G is a REAL array, which need only be set if %gradient_kind is not 0
!    or 1. If this is so, it must be of length at least %n, and contain the
!    linear terms g for the objective function.
!
!   %f is a REAL variable, which must be set by the user to the value of
!    the constant term f in the objective function. On exit, it may have
!    been changed to reflect variables which have been fixed.
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
!   %C is a REAL array of length %m, which is used to store the values of
!    A x. It need not be set on entry. On exit, it will have been filled
!    with appropriate values.
!
!   %X is a REAL array of length %n, which must be set by the user
!    to estimaes of the solution, x. On successful exit, it will contain
!    the required solution, x.
!
!   %C_l, %C_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %C_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %C_u to a value larger than
!    control%infinity. On exit, %C_l and %C_u will most likely have been
!    reordered.
!
!   %Y is a REAL array of length %m, which must be set by the user to
!    appropriate estimates of the values of the Lagrange multipliers
!    corresponding to the general constraints c_l <= A x <= c_u.
!    On successful exit, it will contain the required vector of Lagrange
!    multipliers.
!
!   %X_l, %X_u are REAL arrays of length %n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to control%infinity in
!    absolute value will be regarded as being infinite (see the entry
!    control%infinity). Thus, an infinite lower bound may be specified by
!    setting the appropriate component of %X_l to a value smaller than
!    -control%infinity, while an infinite upper bound can be specified by
!    setting the appropriate element of %X_u to a value larger than
!    control%infinity. On exit, %X_l and %X_u will most likely have been
!    reordered.
!
!   %Z is a REAL array of length %n, which must be set by the user to
!    appropriate estimates of the values of the dual variables
!    (Lagrange multipliers corresponding to the simple bound constraints
!    x_l <= x <= x_u). On successful exit, it will contain
!   the required vector of dual variables.
!
!  data is a structure of type LPA_data_type which holds private internal data
!
!  control is a structure of type LPA_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LPA_initialize. See the preamble
!   for details
!
!  inform is a structure of type LPA_inform_type that provides
!    information on exit from LPA_solve. The component status
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
!        prob%n     >=  1
!        prob%m     >=  0
!        prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated.
!
!    -4 The constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -8 The analytic center appears to be unbounded.
!
!    -9 The analysis phase of the factorization failed; the return status
!       from the factorization package is given in the component factor_status.
!
!   -10 The factorization failed; the return status from the factorization
!       package is given in the component factor_status.
!
!   -11 The solve of a required linear system failed; the return status from
!       the factorization package is given in the component factor_status.
!
!   -16 The problem is so ill-conditoned that further progress is impossible.
!
!   -17 The step is too small to make further impact.
!
!   -18 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!   -19 Too much time has passed. This may happen if control%cpu_time_limit or
!       control%clock_time_limit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!  On exit from LPA_solve, other components of inform are given in the preamble
!
!  C_stat is an optional INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set.
!
!   If control%warm_start is .TRUE., a warm start will be attempted using the
!   values input in C_stat (as above) to indicate whether each constraint is
!   initially to be on its lower bound, its upper bound or between the bounds.
!   If C_stat is absent, no warm start will occur.
!
!  X_stat is an optional  INTEGER array of length n, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set.
!
!   If control%warm_start is .TRUE., a warm start will be attempted using the
!   values input in X_stat (as above) to indicate whether each variable is
!   initially to be on its lower bound, its upper bound or between the bounds.
!   If X_stat is absent, no warm start will occur.
!
!   N.B. If a warm start is required, it is expected that the number of active
!   general constraints and simple bounds should add to n. If there are fewer
!   than n, the package will choose inactive constraints and variables to make
!   up the deficit, while if there are more than n, some will be freed. In
!   addition any constraint whose lower and upper bounds coincide will
!   automatically be in the active set.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPA_control_type ), INTENT( IN ) :: control
      TYPE ( LPA_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  Local variables

      INTEGER :: i, j, l, a_ne
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now
      REAL ( KIND = wp ) :: av_bnd, x_l, x_u, g, sigma
!     REAL ( KIND = wp ) :: fixed_sum, xi
      LOGICAL :: printi, printa, reset_bnd, stat_required
      LOGICAL :: restart, warm_start
      CHARACTER ( LEN = 80 ) :: array_name

!  functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering LPA_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

! -------------------------------------------------------------------
!  If desired, generate a QPLIB file for problem passed

      IF ( control%generate_qplib_file ) THEN
        CALL RPD_write_qp_problem_data( prob, control%qplib_file_name,         &
                    control%qplib_file_device, inform%rpd_inform )
      END IF

!  QPLIB file generated
! -------------------------------------------------------------------

!  initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  initialize counts

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%iter = - 1
      inform%obj = - one
      inform%feasible = .FALSE.
!$    inform%threads = OMP_GET_MAX_THREADS( )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( X_stat )
      warm_start = control%warm_start .AND. stat_required

!  basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1
      printa = control%out > 0 .AND. control%print_level >= 101

!  ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 900
      END IF

!  if required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 )                   &
        CALL QPT_summarize_problem( control%out, prob, lp = .TRUE. )

!  check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

      reset_bnd = .FALSE.
      DO i = 1, prob%n
        IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          GO TO 900
        ELSE IF ( prob%X_u( i ) == prob%X_l( i )  ) THEN
        ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
          prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( /, A, '   **  Warning: one or more variable bounds reset ' )" )     &
         prefix

      reset_bnd = .FALSE.
      DO i = 1, prob%m
        IF ( prob%C_l( i ) - prob%C_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          GO TO 900
        ELSE IF ( prob%C_u( i ) == prob%C_l( i ) ) THEN
        ELSE IF ( prob%C_u( i ) - prob%C_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%C_l( i ) + prob%C_u( i ) )
          prob%C_l( i ) = av_bnd ; prob%C_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( A, /, '   **  Warning: one or more constraint bounds reset ' )" )   &
          prefix

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

!  store the problem dimensions

      IF ( prob%new_problem_structure ) THEN
        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          a_ne = prob%A%ne
        END SELECT
      END IF

!  if the problem has no general constraints, solve it explicitly
!  --------------------------------------------------------------

      IF ( a_ne <= 0 ) THEN
        IF ( printi ) WRITE( control%out,                                      &
          "( /, A, ' Solving explicit bound-constrained LP -' )" ) prefix
        inform%obj = prob%f
        inform%feasible = .TRUE.

!  loop over the components of x

        DO i = 1, prob%n
          x_l = prob%X_l( i ) ; x_u = prob%X_u( i )

!  record the component of the gradient

          IF ( prob%gradient_kind == 0 ) THEN
            g = zero
          ELSE IF ( prob%gradient_kind == 1 ) THEN
            g = one
          ELSE
            g = prob%G( i )
          END IF

!  the objective increases along this component

          IF ( g > zero ) THEN
            prob%X( i ) = x_l

!  the objective is unbounded

            IF ( x_l < - control%infinity ) THEN
              inform%status = GALAHAD_error_unbounded
              GO TO 900

!  The minimizer occurs at the lower bound

            ELSE
              prob%Z( i ) = g
              IF ( PRESENT( X_stat ) ) X_stat( i ) = - 1
            END IF

!  the objective decreases along this component

          ELSE IF ( g < zero ) THEN
            prob%X( i ) = x_u

!  the objective is unbounded

            IF ( x_u > control%infinity ) THEN
              inform%status = GALAHAD_error_unbounded
              GO TO 900

!  the minimizer occurs at the upper bound

            ELSE
              prob%Z( i ) = g
              IF ( PRESENT( X_stat ) ) X_stat( i ) = 1
            END IF

!  the objective is constant along this component direction

          ELSE
            prob%Z( i ) = zero

!  pick an arbitrary minimizer between the bounds

            IF ( PRESENT( X_stat ) ) X_stat( i ) = 0
            IF ( x_l >= - control%infinity .AND. x_u <= control%infinity ) THEN
              prob%X( i ) = half * ( x_l + x_u )
            ELSE IF ( x_l >= - control%infinity ) THEN
              prob%X( i ) = x_l
            ELSE IF ( x_u <= control%infinity ) THEN
              prob%X( i ) = x_u
            ELSE
              prob%X( i ) = zero
            END IF
          END IF
          inform%obj = inform%obj + prob%X( i ) * g
        END DO

        IF ( printi ) THEN
          CALL CLOCK_time( clock_now )
          WRITE( control%out,                                                  &
             "( A, ' explicit bound-constrained LP: status = ',                &
          &   I0, ', time = ', F0.2, /, A, ' objective value =', ES12.4 )",    &
            advance = 'no' ) prefix, inform%status, inform%time%clock_total    &
              + clock_now - clock_start, prefix, inform%obj
          IF ( PRESENT( X_stat ) ) THEN
            WRITE( control%out, "( ', active bounds: ', I0, ' from ', I0 )" )  &
              COUNT( X_stat( : prob%n ) /= 0 ), prob%n
          ELSE
            WRITE( control%out, "( '' )" )
          END IF
        END IF
        GO TO 800
      END IF

!  reorder the problem data to the form required by LA04

      IF ( control%dual ) THEN
        IF ( warm_start ) THEN
          CALL LPA_reorder_dual( prob, data%f, data%m, data%n, data%A_val,     &
                                 data%A_row, data%A_ptr, data%B, data%C,       &
                                 data%kb, data%lb, sigma, data%BND,            &
                                 data%x_map, data%c_map, data%IX, data%JX,     &
                                 control, inform, C_stat, X_stat )
!         WRITE(6,"( '   IX', 11I5, /, ( 12I5 ) )" )  data%IX
        ELSE
          CALL LPA_reorder_dual( prob, data%f, data%m, data%n, data%A_val,     &
                                 data%A_row, data%A_ptr, data%B, data%C,       &
                                 data%kb, data%lb, sigma, data%BND,            &
                                 data%x_map, data%c_map, data%IX, data%JX,     &
                                 control, inform )
        END IF
      ELSE
        IF ( warm_start ) THEN
          CALL LPA_reorder( prob, data%f, data%m, data%n, data%A_val,          &
                            data%A_row, data%A_ptr, data%B, data%C,            &
                            data%kb, data%lb, sigma, data%BND,                 &
                            data%x_map, data%c_map, data%IX, data%JX,          &
                            control, inform, C_stat, X_stat )
!         WRITE(6,"( '   IX', 11I5, /, ( 12I5 ) )" )  data%IX
        ELSE
          CALL LPA_reorder( prob, data%f, data%m, data%n, data%A_val,          &
                            data%A_row, data%A_ptr, data%B, data%C,            &
                            data%kb, data%lb, sigma, data%BND,                 &
                            data%x_map, data%c_map, data%IX, data%JX,          &
                            control, inform )
        END IF
      END IF
      IF ( inform%status /= GALAHAD_ok ) GO TO 900
!     a_ne = data%A_ptr( data%n + 1 ) - 1
      a_ne = SIZE( data%A_val )
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%preprocess =                                                 &
        inform%time%preprocess + REAL( time_now - time_start, wp )
      inform%time%clock_preprocess =                                           &
        inform%time%clock_preprocess + clock_now - clock_start

!  debug printing of reordered problem

      IF ( control%out > 0 .AND. control%print_level >= 100 ) THEN
        WRITE( control%out, "( ' m, n, kb, lb = ', 3( I0, ', ' ), I0 )" )      &
          data%m, data%n, data%kb, data%lb
        DO j = 1, data%n
          WRITE( control%out, "( ' column ', I0, ' of A' )" ) j
          DO l = data%A_ptr( j ), data%A_ptr( j + 1 ) - 1
            WRITE( control%out, "( ' row ', I7, ' val ', ES12.4 )" )           &
              data%A_row( l ), data%A_val( l )
          END DO
        END DO

        WRITE( control%out, "( ' b ', /, 3( I7, ES12.4 ) )" )                  &
          ( i, data%B( i ), i = 1, data%m )

        WRITE( control%out, "( ' objective ', /, 3( I7, ES12.4 ) )" )          &
          ( i, data%C( i ), i = 1, data%n )
      END IF

!  set LA04 control parameters from LPA values

      data%CNTL( 1 ) = sigma
      data%CNTL( 2 ) = control%tol_data
      data%CNTL( 3 ) = control%feas_tol
      IF ( control%steepest_edge ) THEN
        data%CNTL( 4 ) = 1.0_wp
      ELSE
        data%CNTL( 4 ) = 0.0_wp
      END IF
      data%CNTL( 5 ) = REAL( control%max_iterative_refinements, KIND = wp )
      IF ( control%error <= 0 .OR. control%print_level <= 0 ) THEN
        data%CNTL( 6 ) = - 1.0_wp
      ELSE
        data%CNTL( 6 ) = REAL( control%error, KIND = wp )
      END IF
      IF ( control%out <= 0 .OR.  control%print_level <= 1 ) THEN
        data%CNTL( 7 ) = - 1.0_wp
      ELSE
        data%CNTL( 7 ) = REAL( control%out, KIND = wp )
      END IF
      data%CNTL( 8 ) = control%relative_pivot_tolerance
      data%CNTL( 9 ) = control%growth_limit
      data%CNTL( 10 ) = control%zero_tolerance
      data%CNTL( 11 ) = control%change_tolerance
      data%CNTL( 12 ) = REAL( control%random_number_seed, KIND = wp )
      data%CNTL( 13 : 15 ) = 0

!  allocate LA04 return arrays

      array_name = 'lpa: data%X'
      CALL SPACE_resize_array( data%m + data%n, data%X,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'lpa: data%Z'
      CALL SPACE_resize_array( data%n, data%Z,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'lpa: data%G'
      CALL SPACE_resize_array( data%n, data%G,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  allocate LA04 workspace arrays

      data%liws = MAX( control%min_integer_factor_size, ( 3 * a_ne ) / 2 ) +   &
                    10 * data%m + 12
      data%lws = MAX( 4 * ( data%m + data%n ) + 10, 3 * data%m + 4 +           &
                   MAX( control%min_real_factor_size, ( 3 * a_ne ) / 2 ) )

 100  CONTINUE
      array_name = 'lpa: data%IWS'
      CALL SPACE_resize_array( data%liws, data%IWS,                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'lpa: data%WS'
      CALL SPACE_resize_array( data%lws, data%WS,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      IF ( control%dual ) THEN
        IF ( printi ) WRITE( control%out,                                      &
          "( A, ' -- solve via the dual --', / )" ) prefix
      END IF

!  scale the problem if required

      IF ( control%scale ) THEN
        inform%la04_job = 0
        CALL LA04AD( data%A_val, a_ne, data%A_row, data%A_ptr, data%m,         &
                     data%n, data%B, data%C, data%BND, data%kb, data%lb,       &
                     inform%la04_job, data%CNTL, data%IX, data%JX,             &
                     data%X, data%Z, data%G, inform%RINFO,                     &
                     data%WS, data%lws, data%IWS, data%liws )

!  check return conditions

        IF ( inform%la04_job == - 13 .OR. inform%la04_job == - 14 ) THEN
          inform%la04_job_info = INT( inform%RINFO( 35 ) )
          data%liws = MAX( data%liws, ( 11 * inform%la04_job_info ) / 10 )
          data%lws = MAX( data%lws, ( 11 * inform%la04_job_info ) / 10 )
          GO TO 100
        ELSE IF ( inform%la04_job == - 101 ) THEN
          inform%status = GALAHAD_unavailable_option
          GO TO 900
        ELSE IF ( inform%la04_job < 0 ) THEN
          IF ( printi ) WRITE( control%out, "( A, '  scaling failed,',         &
         &  ' job value is ', I0 )" ) prefix, inform%la04_job
          GO TO 900
        END IF

!  remember the scalings

        array_name = 'lpa: data%CS'
        CALL SPACE_resize_array( data%n, data%CS,                              &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error,              &
               exact_size = .TRUE. )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        data%CS( : data%n ) =                                                  &
          data%WS( : data%n ) * data%WS( data%n + data%m + 2 )
      END IF

!  solve the problem using the simplex method

      inform%iter = 0
      IF ( warm_start ) THEN
        inform%la04_job = 2
!       WRITE(6,"( '   IX', 11I5, /, ( 12I5 ) )" )  data%IX
!       WRITE(6,"( '   JX', 11I5, /, ( 12I5 ) )" )  data%JX
!       WRITE(6,"( '   BND(1;)',  /, ( 5ES12.4 ) )" )  data%BND(1,:)
!       WRITE(6,"( '   BND(2;)',  /, ( 5ES12.4 ) )" )  data%BND(2,:)
      ELSE
        inform%la04_job = 1
      END IF
      DO
        inform%iter = inform%iter + 1
        IF ( inform%iter > control%maxit ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, ' LPA iteration limit reached' )" ) prefix
          inform%status = GALAHAD_error_max_iterations ; GO TO 900
        END IF

!  check that the CPU time limit has not been reached

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        IF ( ( control%cpu_time_limit >= zero .AND.                            &
             REAL( time_now - time_start, wp ) > control%cpu_time_limit ) .OR. &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( A, ' LPA time limit reached' )" ) prefix
          inform%status = GALAHAD_error_time_limit ; GO TO 900
        END IF

!  perform another simplex iteration

        CALL LA04AD( data%A_val, a_ne, data%A_row, data%A_ptr, data%m,         &
                     data%n, data%B, data%C, data%BND, data%kb, data%lb,       &
                     inform%la04_job, data%CNTL, data%IX, data%JX,             &
                     data%X, data%Z, data%G, inform%RINFO,                     &
                     data%WS, data%lws, data%IWS, data%liws )
!       WRITE(6,"( '   IX', 11I5, /, ( 12I5 ) )" )  data%IX

!  successful termination of the simplex method

        IF (inform%la04_job == 0 ) THEN
          inform%status = GALAHAD_ok
          EXIT

!  error returns

        ELSE  IF ( inform%la04_job < 0 ) THEN
          restart = .FALSE.
          SELECT CASE ( inform%la04_job )

!  the constraints have no solution

          CASE ( - 1 )
            inform%status = GALAHAD_error_primal_infeasible

!  the objective function is unbounded. The direction that leads
!  to an unbounded solution is available in data%WS

          CASE ( - 2 )
            inform%status = GALAHAD_error_unbounded

!  the bounds on m, n, kb and lb are not all satisfied

          CASE ( - 3 )
            inform%status = GALAHAD_error_restrictions

!  workspace length lws or liws is too small. They must be at least
!  ib + 3m + 4 and 2ib + 10m + 12, respectively, where ib = la04_job_info

          CASE ( - 4 )
            inform%status = GALAHAD_error_integer_ws
            inform%la04_job_info = INT( inform%RINFO( 35 ) )
            data%liws = MAX( data%liws, ( 11 * ( 2 * inform%la04_job_info      &
                             + 10 * data%m + 12 ) ) / 10 )
            data%lws = MAX( data%lws, ( 11 * ( inform%la04_job_info            &
                             + 3 * data%m + 4 ) ) / 10 )
            restart = .TRUE.

!  the inequalities - sigma <= BND(1,j) <= BND(2,j) <= sigma are not
!  satisfied, where j = la04_job_info

          CASE ( - 5 )
            inform%status = GALAHAD_error_bad_bounds
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  invalid A_ptr(j), where j = la04_job_info

          CASE ( - 6 )
            inform%status = GALAHAD_error_restrictions
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  invalid A_row(k), where k = la04_job_info

          CASE ( - 7 )
            inform%status = GALAHAD_error_restrictions
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  duplicate entry in A.  The index k is returned in la04_job_info

          CASE ( - 8 )
            inform%status = GALAHAD_warning_repeated_entry
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  job outside the range 0 <= job <= 7 on entry

          CASE ( - 9 )
            inform%status = GALAHAD_error_restrictions

! la is too small, and must be at least la04_job_info

          CASE ( - 10 )
            inform%status = GALAHAD_error_real_ws
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  the inequalities 1 <= IX(i) <= 3( m + n ) are not satisfied, where
!  i = la04_job_info

          CASE ( - 11 )
            inform%status = GALAHAD_error_restrictions
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  the inequalities - 1 <= JX(i)<= 2 are not satisfied, or JX(i) = 1 and
!  BND(1,j) = - sigma, or `JX(i) = 2 and BND(2,j) = sigma, where
!  i = la04_job_info

          CASE ( - 12 )
            inform%status = GALAHAD_error_restrictions
            inform%la04_job_info = INT( inform%RINFO( 35 ) )

!  lws is too small, and must be at least la04_job_info

          CASE ( - 13 )
            inform%status = GALAHAD_error_real_ws
            inform%la04_job_info = INT( inform%RINFO( 35 ) )
            data%liws = MAX( data%liws, ( 11 * inform%la04_job_info ) / 10 )
            data%lws = MAX( data%lws, ( 11 * inform%la04_job_info ) / 10 )
            restart = .TRUE.

!  liws is too small, and must be at least la04_job_info

          CASE ( - 14 )
            inform%status = GALAHAD_error_integer_ws
            inform%la04_job_info = INT( inform%RINFO( 35 ) )
            data%liws = MAX( data%liws, ( 11 * inform%la04_job_info ) / 10 )
            data%lws = MAX( data%lws, ( 11 * inform%la04_job_info ) / 10 )
            restart = .TRUE.

!  LA04 is not available

          CASE ( - 101 )
            inform%status = GALAHAD_unavailable_option
          END SELECT

          IF ( restart ) THEN
            IF ( printi ) WRITE( control%out, "( A, '  increasing workspace',  &
           &  '(integer,real=', I0, ',', O0, ') and restarting' )" )           &
              prefix, data%liws, data%lws
            GO TO 100
          ELSE
            IF ( printi ) WRITE( control%out, "( A, '  solution not found',    &
           &  ' after ', I0, ' iterations, job value is ', I0 )" )             &
              prefix, inform%iter, inform%la04_job
            GO TO 900
          END IF
        END IF
      END DO
!     WRITE(6,"( '   JX', 11I5, /, ( 12I5 ) )" )  data%JX

!  debug printing of solution to reordered problem

      IF ( control%out > 0 .AND. control%print_level >= 100 ) THEN
        WRITE( control%out, "( ' x ', /, 3( I7, ES12.4 ) )" )                  &
          ( i, data%X( i ), i = 1, data%n )
        WRITE( control%out, "( ' y ', /, 3( I7, ES12.4 ) )" )                  &
          ( i, data%WS( i ), i = 1, data%m )
        WRITE( control%out, "( ' z ', /, 3( I7, ES12.4 ) )" )                  &
          ( i, data%Z( i ), i = 1, data%n )
      END IF

!  recover the solution from that returned by LA04

!  add the bounds in use to X, and unscale it if necessary

      DO i = 1, data%kb
        IF (data%JX( i ) == 1 ) data%X( i ) = data%X( i ) + data%BND( 1, i )
        IF (data%JX( i ) == 2 ) data%X( i ) = data%X( i ) + data%BND( 2, i )
      END DO
      IF ( control%scale )                                                     &
        data%X( : data%n ) = data%X( : data%n ) * data%CS( : data%n )

!  unscramble the reordered version

      IF ( control%dual ) THEN
        CALL LPA_revert_dual( prob, data%x_map, data%c_map, data%n, data%X,    &
                              data%WS, data%Z, control, inform )
      ELSE
        CALL LPA_revert( prob, data%x_map, data%c_map, data%X,                 &
                         data%WS, data%Z, control, inform )
      END IF

!  if required, return the status of the general constraints

      IF ( PRESENT( C_stat ) ) THEN
        DO i = 1, prob%m
          IF ( prob%C( i ) <= prob%C_l( i ) + control%feas_tol ) THEN
            C_stat( i ) = - 1
          ELSE IF ( prob%C( i ) >= prob%C_u( i ) - control%feas_tol ) THEN
            C_stat( i ) = 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO
      END IF

!  if required, return the status of the variables

      IF ( PRESENT( X_stat ) ) THEN
        DO i = 1, prob%n
          IF ( prob%X( i ) <= prob%X_l( i ) + control%feas_tol ) THEN
            X_stat( i ) = - 1
          ELSE IF ( prob%X( i ) >= prob%X_u( i ) - control%feas_tol ) THEN
            X_stat( i ) = 1
          ELSE
            X_stat( i ) = 0
          END IF
        END DO
      END IF

!  compute total time

  800 CONTINUE
      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, wp )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
     "( /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=',                                            &
    &   /, A, 3X, ' =  LPA total time = ', F12.2, ', preprocess = ', F12.2,    &
    &   7X, '=',                                                               &
    &   /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=') ")                                         &
        prefix, prefix, inform%time%clock_total, inform%time%clock_preprocess, &
        prefix
        GO TO 990

!  error returns

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, wp )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Error return from -LPA_solve- status = ', I0 )" )           &
        prefix, inform%status
      GO TO 990

!  allocation error

  910 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, wp )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -LPA_solve-', /,  A,                           &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, prefix, inform%alloc_status

!  prepare to exit

  990 CONTINUE
      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving LPA_solve ' )" ) prefix
      RETURN

!  End of LPA_solve

      END SUBROUTINE LPA_solve

!-*-*-*-*-*-*-   L P A _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE LPA_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine LPA_initialize
!   control see Subroutine LPA_initialize
!   inform  see Subroutine LPA_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( LPA_data_type ), INTENT( INOUT ) :: data
      TYPE ( LPA_control_type ), INTENT( IN ) :: control
      TYPE ( LPA_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all remaing allocated arrays


      array_name = 'lpa: data%A_ptr'
      CALL SPACE_dealloc_array( data%A_ptr,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%A_row'
      CALL SPACE_dealloc_array( data%A_row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%IX'
      CALL SPACE_dealloc_array( data%IX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%JX'
      CALL SPACE_dealloc_array( data%JX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%IWS'
      CALL SPACE_dealloc_array( data%IWS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%A_val'
      CALL SPACE_dealloc_array( data%A_val,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%G'
      CALL SPACE_dealloc_array( data%G,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%X'
      CALL SPACE_dealloc_array( data%X,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%Z'
      CALL SPACE_dealloc_array( data%Z,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%WS'
      CALL SPACE_dealloc_array( data%WS,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%BND'
      CALL SPACE_dealloc_array( data%BND,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'lpa: data%CS'
      CALL SPACE_dealloc_array( data%CS,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine LPA_terminate

      END SUBROUTINE LPA_terminate

!-*-*-*-*-*-*-*-   L P A _ r e o r d e r    S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE LPA_reorder( prob, f, m, n, A_val, A_row, A_ptr, B, C, kb,    &
                              lb, sigma, BND, x_map, c_map, IX, JX,            &
                              control, inform, C_stat, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find a reordering of the data for the problem
!
!     minimize              l(x) = g(T) x + f
!         x
!     subject to the bounds  x_l <=  x  <= x_u
!     and constraints        c_l <= A x <= c_u ,
!
!  where x is a vector of n components ( x_1, .... , x_n ), A is an m by n
!  matrix, and any of the bounds x_l, x_u, c_l, c_u may be infinite.
!
!  The reordered problem takes the form
!
!     minimize                 c^T x_r + f'
!       x_r
!     subject to the equations A_r x_r = b,
!     the general bounds       bnd(1,i) <= x_r(i) <= bnd(2,i), i = 1,..., kb
!     and the non-negativities x_r(i) >= 0, i = lb, ...,  n
!
!  has the following properties:
!
!  * slack variables s will be introduced for any constraint for which
!    c_l < c_u. If both bounds c_l and c_u are finite, the slack will
!    be two-sided with 0 <= s <= s_u = c_u - c_l, while otherwise s_i >= 0

!  * all constraints will become equations A_r x_r = b by introducing
!    slack variables.
!
!  * the variables are ordered so that their bounds appear in the order
!
!    range              x_l <= x <= x_u (including one-sided infinite x_l/x_u)
!    two-sided slacks    0  <= s <= s_u
!    free                      x
!    non-negativity      0  <= x or non-positivity x <= 0
!    one-sided slacks          s >= 0 or surpluses s <= 0
!    fixed                     x = x_l = x_u

!    Fixed variables will be removed.
!
!    The signs of columns of A and entries of c corresponding to one-sided
!    upper bounds will be reversed.
!
!  * the constraint right-hand sides b wil be
!
!    equality           b = c_l
!    lower              b = c_l
!    range              b = c_l
!    upper              b = c_u
!
!    Free constraints will be removed.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  input variable that needs to be set:
!
!  prob is a structure of type QPT_type, whose components hold the
!   details of the problem. The following components must be set
!
!   f is a REAL variable that gives the term f in the objective function.
!
!   n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: n >= 1
!
!   m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m.  RESTRICTION: m >= 0
!
!   gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero. %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by G (see below)
!
!   G is a REAL array of length n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!
!   A is a structure of type SMT_type used to hold the matrix A.
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
!   X_l, X_u are REAL arrays of length n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to infinity in absolute value
!    will be regarded as being infinite (see the entry control%infinity).
!    Thus, an infinite lower bound may be specified by setting the appropriate
!    component of X_l to a value smaller than -infinity, while an infinite
!    upper bound can be specified by setting the appropriate element of X_u
!    to a value larger than infinity.
!
!   C_l, C_u are REAL array of length m, which must be set by the user to
!    the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to infinity in absolute value
!    will be regarded as being infinite (see the entry control%infinity).
!    Thus, an infinite lower bound may be specified by setting the appropriate
!    component of C_l to a value smaller than -infinity, while an infinite
!    upper bound can be specified by setting the appropriate element of C_u
!    to a value larger than infinity.
!
!  output variables that need not be set on input and describe the
!  reordered linear program:
!
!  f is a REAL variable that gives the term f' in the objective function.
!
!  n is an INTEGER variable that gives the number of optimization parameters
!   (including slacks)
!
!  m is an INTEGER variable that gives the number of general linear constraints
!   (after removing free constraints)
!
!  A_val is a REAL array whose components give the elements of A, stored by
!   column, with those for column i+1 immediately after those for column i;
!   the order within each column is arbitrary
!
!  A_row is an INTEGER array that holds the row indices of the components
!   of A corresponding to the entries in A_val
!
!  A_ptr is an INTEGER array of length n whose i-th entry gives the position
!   in A_val and A_row of the first entry in column i, i = 1, ..., m,
!   while A_ptr( n + 1 ) - 1 gives the lengths of A_val and A_row
!
!  B is a REAL array that holds the right-hand side vector b
!
!  C is a REAL array that holds the output objective gradient c
!
!  kb is an INTEGER variable that gives the number of two-sided bounds
!
!  lb is an INTEGER variable for which variables x_i >= 0 for i = lb, ..., n
!
!  sigma is a REAL variable whose value is used to flag infinite lower or
!    upper bounds
!
!  BND( : 2, : kb ) is a REAL array for which BND( 1, i ) and BND( 2, i )
!   are the lower and upper bounds on the variables for i = 1, ... , kb.
!   Any variable lower bound whose value is - sigma is considered to be
!   unbounded from below, and one whose upper bound is sigma is unbounded
!   from above
!
!  x_map is an INTEGER array whose j-th entry gives the position of the
!   original j-th variable in the reordered problem (0=removed)
!
!  c_map is an INTEGER array whose i-th entry gives the position of the
!   original i-th constraint row in the reordered problem (0=removed)
!
!  control is a structure of type LPA_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LPA_initialize. See
!   LPA_initialize for details
!
!  inform is a structure of type LPA_inform_type that provides information
!    on exit from LPA_reorder. The component status has possible values:
!
!     0 Normal termination
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 one of the restrictions
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated
!
!    -5 The constraints are inconsistent
!
!  C_stat is an optional INTEGER array of length prob%m, which if present
!   should be set on entry to indicate the initial status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the active set.
!
!  X_stat is an optional INTEGER array of length prob%n, which if present
!    should be set on entry to indicate the initial status of the simple bound
!   constraints. Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is  not in the active set.
!
!   N.B. If C_stat and B_stat are present, it is expected that the number of
!   active general constraints and simple bounds should add to n. If there
!   are fewer than n, the package will choose inactive constraints and
!   variables to make up the deficit, while if there are more than n,
!   some will be freed. In addition any constraint whose lower and upper
!   bounds coincide will automatically be in the active set.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPA_control_type ), INTENT( IN ) :: control
      TYPE ( LPA_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, INTENT( OUT ) :: m, n, kb, lb
      REAL ( KIND = wp ), INTENT( OUT ) :: f, sigma
      INTEGER, INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: x_map, c_map
      INTEGER, INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: IX, JX
      INTEGER, INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_ptr
      REAL ( KIND = wp ), INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: B, C
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  local variables

      INTEGER :: i, ii, j, jj, l, b0, b1, b2, r1, r2, s0, s1, s2
      INTEGER :: free, nonneg, lower, range, upper, nonpos, fixed, basic
      INTEGER :: c_lower, c_range, c_upper, c_equality, a_ne, a_type, v_stat

      REAL ( KIND = wp ) :: xl, xu, cl, cu
      CHARACTER ( LEN = 80 ) :: array_name

!  provide room for the mapping arrays and the residuals for fixed variables

      array_name = 'lpa: x_map'
      CALL SPACE_resize_array( prob%n, x_map,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: c_map'
      CALL SPACE_resize_array( prob%m, c_map,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: prob%C'
      CALL SPACE_resize_array( prob%m, prob%C,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  run through the bounds to see how many fall into each of the categories:
!  free(free), non-negativity(nonneg), lower(lower), range(range),
!  upper(upper), non-positivity (nonpos) and fixed (fixed)

!  also indicate whether a variable is fixed by temporarily setting
!  x_map(i) = 0 (fixed), x_map(i) = 1 (non-negativity/positivity),
!  x_map(i) = 2 (general one- or two-sided bounds) and x_map(i) = 3 (free)

      free = 0 ; nonneg = 0 ; lower = 0
      range = 0 ; upper = 0 ; nonpos = 0 ; fixed = 0

      DO i = 1, prob%n
        xl = prob%X_l( i ) ; xu = prob%X_u( i )

!  fixed variable

        IF ( xu == xl ) THEN
          fixed = fixed + 1
          x_map( i ) = 0

!  the variable has no lower bound

        ELSE IF ( xl <= - control%infinity ) THEN

!  free variable

          IF ( xu >= control%infinity ) THEN
            free = free + 1
            x_map( i ) = 3

!  non-positivity

          ELSE IF ( xu == zero ) THEN
            nonpos = nonpos + 1
            x_map( i ) = 1

!  upper-bounded variable

          ELSE
            upper = upper + 1
            x_map( i ) = 2
          END IF

!  the variable has a lower bound

        ELSE
          x_map( i ) = 1
          IF ( xu < control%infinity ) THEN

!  inconsistent upper bound

!           IF ( xu < xl ) THEN
!             inform%status = GALAHAD_error_primal_infeasible
!             RETURN

!  range-bounded variable

!           ELSE
              range = range + 1
              x_map( i ) = 2
!           END IF

!  non-negativity

          ELSE IF ( xl == zero ) THEN
            nonneg = nonneg + 1
            x_map( i ) = 1

!  lower-bounded variable

          ELSE
            lower = lower + 1
            x_map( i ) = 2
          END IF
        END IF
      END DO

!  see which storage scheme is used for A

      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        a_type = a_dense
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_type = a_sparse_by_rows
      ELSE
        a_type = a_coordinate
      END IF

!  count how many non-fixed entries there are in each constraint. c_map is
!  temporarily used for this: c_map(i) gives the number of entries in row i
!  Also set C = A_fx x_fx, where fx are the indices of the fixed variables

!  ** pass 1 through A **

      c_map( : prob%m ) = 0 ; prob%C = zero

!  original dense storage

      IF ( a_type == a_dense ) THEN
!       c_map( : prob%m ) = prob%n - fixed
        DO j = 1, prob%n
          l = j
          IF ( x_map( j ) == 0 ) THEN
            xl = PROB%X_l( j )
            DO i = 1, prob%m
              prob%C( i ) = prob%C( i ) + prob%A%val( l ) * xl
              l = l + prob%n
            END DO
          ELSE
            DO i = 1, prob%m
              IF ( prob%A%val( l ) /= zero ) c_map( i ) = c_map( i ) + 1
              l = l + prob%n
            END DO
          END IF
        END DO
      ELSE

!  original co-ordinate storage

        IF ( a_type == a_coordinate ) THEN
          DO l = 1, prob%A%ne
            i = prob%A%row( l )
            IF ( x_map(  prob%A%col( l ) ) > 0 ) THEN
              c_map( i ) = c_map( i ) + 1
            ELSE
              prob%C( i )                                                      &
                = prob%C( i ) + prob%A%val( l ) * prob%X_l( prob%A%col( l ) )
            END IF
          END DO

!  original row-wise storage

        ELSE
          DO i = 1, prob%m
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              IF ( x_map( prob%A%col( l ) ) > 0 ) THEN
                c_map( i ) = c_map( i ) + 1
              ELSE
                prob%C( i )                                                    &
                  = prob%C( i ) + prob%A%val( l ) * prob%X_l( prob%A%col( l ) )
              END IF
            END DO
          END DO
        END IF
      END IF

!  run through the constraints to see how many fall into each of the categories:
!  lower(c_lower), range(c_range), upper(c_upper) and equality (c_equality)

!  also count the number of non-free constraints (m) and record these in c_map

      m = 0 ; c_lower = 0 ; c_range = 0 ; c_upper = 0 ; c_equality = 0

      DO i = 1, prob%m
        cl = prob%C_l( i ) ; cu = prob%C_u( i )

!  equality constraint

        IF ( cu == cl ) THEN
          IF ( c_map( i ) > 0 ) THEN
            c_equality = c_equality + 1 ; m = m + 1 ; c_map( i ) = m

!  deal with null equality constraint

          ELSE
            IF ( cu == prob%C( i ) ) THEN
              c_map( i ) = 0
            ELSE
              inform%status = GALAHAD_error_primal_infeasible
              RETURN
            END IF
          END IF
        ELSE IF ( cl <= - control%infinity ) THEN

!  free constraint

          IF ( cu >= control%infinity ) THEN
            c_map( i ) = 0
          ELSE

!  upper-bounded constraint

            IF ( c_map( i ) > 0 ) THEN
              c_upper = c_upper + 1 ; m = m + 1 ; c_map( i ) = m

!  deal with null upper-bounded constraint

            ELSE
              IF ( cu >= prob%C( i ) ) THEN
                c_map( i ) = 0
              ELSE
                inform%status = GALAHAD_error_primal_infeasible
                RETURN
              END IF
            END IF
          END IF
        ELSE
          IF ( cu < control%infinity ) THEN

!  inconsistent constraints

!           IF ( cu < cl ) THEN
!             inform%status = GALAHAD_error_primal_infeasible
!             RETURN

!  range-bounded constraint

!           ELSE
              IF ( c_map( i ) > 0 ) THEN
                c_range = c_range + 1 ; m = m + 1 ; c_map( i ) = m
              ELSE

!  deal with null range bounded constraint

                IF ( cl <= prob%C( i ) .AND. cu >= prob%C( i ) ) THEN
                  c_map( i ) = 0
                ELSE
                  inform%status = GALAHAD_error_primal_infeasible
                  RETURN
                END IF
              END IF
!           END IF
          ELSE

!  lower-bounded constraint

            IF ( c_map( i ) > 0 ) THEN
              c_lower = c_lower + 1 ; m = m + 1 ; c_map( i ) = m
            ELSE

!  deal with null lower-bounded constraint

              IF ( cl <= prob%C( i ) ) THEN
                c_map( i ) = 0
              ELSE
                inform%status = GALAHAD_error_primal_infeasible
                RETURN
              END IF
            END IF
          END IF
        END IF
      END DO

!  set the starting and ending addresses as required

!        --------------------------------------------
!        | l<=  |  2-sided  |      | x>=0 | 1-sided |
!  A_r = |   x  |  slacks   | free |  or  | slacks  |
!        | <=u  | 0<=s<=s_u |      | x<=0 |  s>=0   |
!        --------------------------------------------
!       ^      ^ ^         ^ ^    ^ ^    ^ ^       ^
!       |      | |         | |    | |    | |       |
!      b2     r2 s2    kb=b0 s0  b1 lb  r1 s1      n

      b2 = 0
      r2 = range + lower + upper
      s2 = r2 + 1
      kb = r2 + c_range
      b0 = kb
      s0 = kb + 1
      b1 = b0 + free
      lb = b1 + 1
      r1 = b1 + nonneg + nonpos
      s1 = r1 + 1
      n = r1 + c_lower + c_upper
!  allocate space for the reordered problem

      array_name = 'lpa: A_ptr'
      CALL SPACE_resize_array( n + 1, A_ptr,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: C'
      CALL SPACE_resize_array( n, C,                                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: B'
      CALL SPACE_resize_array( m, B,                                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: BND'
      CALL SPACE_resize_array( 2, kb, BND,                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  reset x_map so that its j-th entry gives the position of the
!  original j-th variable in the reordered problem. Also assign
!  sigma, BND and C

      sigma = ten * control%infinity
      DO j = 1, prob%n

!  non-negativity/positivity

        IF ( x_map( j ) == 1 ) THEN
          b1 = b1 + 1
          x_map( j ) = b1
          IF ( prob%gradient_kind == 0 ) THEN
            C( b1 ) = zero
          ELSE IF ( prob%gradient_kind == 1 ) THEN
            C( b1 ) = one
          ELSE
            C( b1 ) = prob%G( j )
          END IF

!  general one- or two-sided bound

        ELSE IF ( x_map( j ) == 2 ) THEN
          b2 = b2 + 1
          x_map( j ) = b2
          IF ( prob%gradient_kind == 0 ) THEN
            C( b2 ) = zero
          ELSE IF ( prob%gradient_kind == 1 ) THEN
            C( b2 ) = one
          ELSE
            C( b2 ) = prob%G( j )
          END IF
          xl = prob%X_l( j ) ; xu = prob%X_u( j )
          IF ( xl <= - control%infinity ) THEN
            BND( 1, b2 ) = - sigma
          ELSE
            BND( 1, b2 ) = xl
          END IF
          IF ( xu >= control%infinity ) THEN
            BND( 2, b2 ) = sigma
          ELSE
            BND( 2, b2 ) = xu
          END IF

!  free variable

        ELSE IF ( x_map( j ) == 3 ) THEN
          b0 = b0 + 1
          x_map( j ) = b0
          IF ( prob%gradient_kind == 0 ) THEN
            C( b0 ) = zero
          ELSE IF ( prob%gradient_kind == 1 ) THEN
            C( b0 ) = one
          ELSE
            C( b0 ) = prob%G( j )
          END IF
        END IF
      END DO

!  assign C and BND(1) for the slacks

      C( s2 : kb ) = zero ; C( s1 : n ) = zero
      BND( 1, s2 : kb ) = zero

!  calculate the lengths of column j of A_r in A_ptr(j+1)

      A_ptr( 2 : r2 + 1 ) = 0 ; A_ptr( s0 + 1 : r1 + 1 ) = 0

!  ** pass 2 through A **

!  original dense storage

      IF ( a_type == a_dense ) THEN
        DO j = 1, prob%n
          jj = x_map( j )
          IF ( jj > 0 ) THEN
            l = j
            DO i = 1, prob%m
              IF ( c_map( i ) > 0 .AND. prob%A%val( l ) /= zero )              &
                A_ptr( jj + 1 ) = A_ptr( jj + 1 ) + 1
               l = l + prob%n
            END DO
          END IF
        END DO

!  original co-ordinate storage

      ELSE IF ( a_type == a_coordinate ) THEN
        DO l = 1, prob%A%ne
          jj = x_map( prob%A%col( l ) )
          IF ( jj > 0 ) THEN
            IF ( c_map( prob%A%row( l ) ) > 0 )                              &
              A_ptr( jj + 1 ) = A_ptr( jj + 1 ) + 1
          END IF
        END DO

!  original row-wise storage

      ELSE
        DO i = 1, prob%m
          IF ( c_map( i ) > 0 ) THEN
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              jj = x_map( prob%A%col( l ) )
              IF ( jj > 0 ) A_ptr( jj + 1 ) = A_ptr( jj + 1 ) + 1
            END DO
          END IF
        END DO
      END IF

!  include the slack columns

      A_ptr( s2 + 1 : kb + 1) = 1 ; A_ptr( s1 + 1 : n + 1 ) = 1

!  adjust A_ptr so that A_ptr(j) points to the storage location one before the
!  starting entry in column j (and A_ptr(n+1) points to the end of the array)

      A_ptr( 1 ) = 0
      DO j = 2, n + 1
        A_ptr( j ) = A_ptr( j ) + A_ptr( j - 1 )
      END DO

!  allocate space for values and row indices for the reordered problem

      IF ( control%scale ) THEN
        a_ne = A_ptr( n + 1 ) + n + m
      ELSE
        a_ne = A_ptr( n + 1 )
      END IF

      array_name = 'lpa: A_val'
      CALL SPACE_resize_array( a_ne, A_val,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: A_row'
      CALL SPACE_resize_array( a_ne, A_row,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  introduce the slack columns into A_r; also set B and BND(2) as appropriate

      DO i = 1, prob%m
        ii = c_map( i )
        IF ( ii > 0 ) THEN
          cl = prob%C_l( i ) ; cu = prob%C_u( i )

!  equality constraints

          IF ( cu == cl ) THEN
            B( ii ) = cl - prob%C( i )

!  upper-bounded constraint

          ELSE IF ( cl <= - control%infinity ) THEN
            IF ( cu < control%infinity ) THEN
              r1 = r1 + 1
              A_ptr( r1 ) = A_ptr( r1 ) + 1
              A_val( A_ptr( r1 ) ) = one
              A_row( A_ptr( r1 ) ) = ii
              B( ii ) = cu - prob%C( i )
            END IF

!  range-bounded constraint

          ELSE
            IF ( cu < control%infinity ) THEN
              r2 = r2 + 1
              A_ptr( r2 ) = A_ptr( r2 ) + 1
              A_val( A_ptr( r2 ) ) = - one
              A_row( A_ptr( r2 ) ) = ii
              B( ii ) = cl
              BND( 2, r2 ) = cu - cl

!  lower-bounded constraint

            ELSE
              r1 = r1 + 1
              A_ptr( r1 ) = A_ptr( r1 ) + 1
              A_val( A_ptr( r1 ) ) = - one
              A_row( A_ptr( r1 ) ) = ii
              B( ii ) = cl - prob%C( i )
            END IF
          END IF
        END IF
      END DO

!  introduce the entries of A into A_r, and update B as appropriate

!  ** pass 3 through A **

!  original dense storage

      IF ( a_type == a_dense ) THEN
        l = 0
        DO i = 1, prob%m
          ii = c_map( i )
          IF ( ii > 0 ) THEN
            DO j = 1, prob%n
              jj = x_map( j ) ; l = l + 1
              IF ( jj > 0 .AND. prob%A%val( l ) /= zero ) THEN
                A_ptr( jj ) = A_ptr( jj ) + 1
                A_val( A_ptr( jj ) ) = prob%A%val( l )
                A_row( A_ptr( jj ) ) = ii
              END IF
            END DO
          ELSE
            l = l + prob%n
          END IF
        END DO

!  original co-ordinate storage

      ELSE
        IF ( a_type == a_coordinate ) THEN
          DO l = 1, prob%A%ne
            j = prob%A%col( l )
            jj = x_map( j )
            IF ( jj > 0 ) THEN
              ii = c_map( prob%A%row( l ) )
              IF ( ii > 0 ) THEN
                A_ptr( jj ) = A_ptr( jj ) + 1
                A_val( A_ptr( jj ) ) = prob%A%val( l )
                A_row( A_ptr( jj ) ) = ii
              END IF
!           ELSE
!             ii = c_map( prob%A%row( l ) )
!             IF ( ii > 0 ) B( ii ) = B( ii ) - prob%A%val( l ) * prob%X_l( j )
            END IF
          END DO

!  original row-wise storage

        ELSE
          DO i = 1, prob%m
            ii = c_map( i )
            IF ( ii > 0 ) THEN
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                jj = x_map( j )
                IF ( jj > 0 ) THEN
                  A_ptr( jj ) = A_ptr( jj ) + 1
                  A_val( A_ptr( jj ) ) = prob%A%val( l )
                  A_row( A_ptr( jj ) ) = ii
!               ELSE
!                 B( ii ) = B( ii ) - prob%A%val( l ) * prob%X_l( j )
                END IF
              END DO
            END IF
          END DO
        END IF
      END IF

!  reset the column pointers

      DO i = n, 1, - 1
        A_ptr( i + 1 ) = A_ptr( i ) + 1
      END DO
      A_ptr( 1 ) = 1

!  reverse the signs of columns of A_r and entries of C
!  corresponding to one-sided upper bounds

      DO i = 1, prob%n
        IF ( prob%X_l( i ) <= - control%infinity .AND.                         &
             prob%X_u( i ) == zero ) THEN
          ii = x_map( i )
          A_val( A_ptr( ii ) : A_ptr( ii + 1 ) - 1 ) =                         &
            A_val( A_ptr( ii ) : A_ptr( ii + 1 ) - 1 )
          C( ii ) = - C( ii )
        END IF
      END DO

!  compute the constant term in the objective function, f'

      f = prob%f
      DO j = 1, prob%n
        IF ( x_map( j ) == 0 ) f = f + prob%G( j ) * prob%X_l( j )
      END DO

!  allocate space to hold the lists of basic and non-basic variables

      array_name = 'lpa: IX'
      CALL SPACE_resize_array( m, IX,                                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: JX'
      CALL SPACE_resize_array( kb, JX,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  if a warm start is requested, set the basic and non-basic variables

      IF ( PRESENT( C_stat ) .AND. PRESENT( X_stat ) ) THEN
        r2 = s2 - 1
        r1 = s1 - 1
        basic = 0

!  assess the status of the problem variables

        DO j = 1, prob%n
          jj = x_map( j )
          IF ( jj > 0 ) THEN
            v_stat = X_stat( j )
            IF ( v_stat < 0 ) THEN
              IF ( jj <= kb ) JX( jj ) = 1
            ELSE IF ( v_stat > 0 ) THEN
              IF ( jj <= kb ) JX( jj ) = 2
            ELSE
              IF ( basic < m ) THEN
                basic = basic + 1
                IF ( jj <= kb ) JX( jj ) = - 1
                IX( basic ) = jj
!write(6, "( ' basic         ', 2I7, 3ES12.4 )" ) &
!  jj, j,  prob%X_l(j), prob%X(j), prob%X_u(j)
              ELSE
!write(6, "( ' free nonbasic ', 2I7, 3ES12.4 )" ) &
!  jj, j,  prob%X_l(j), prob%X(j), prob%X_u(j)
                IF ( jj <= kb ) JX( jj ) = 0
              END IF
            END IF
          END IF
        END DO

!  assess the status of slack variables

        DO i = 1, prob%m
          ii = c_map( i )
          IF ( ii > 0 ) THEN
            cl = prob%C_l( i ) ; cu = prob%C_u( i )
            v_stat = C_stat( i )

!  equality constraints

            IF ( cu == cl ) THEN

!  upper-bounded constraint

            ELSE IF ( cl <= - control%infinity ) THEN
              IF ( cu < control%infinity ) THEN
                r1 = r1 + 1
                IF ( v_stat == 0 .AND. basic < m ) THEN
                  basic = basic + 1
                  IX( basic ) = r1
                END IF
              END IF

!  range-bounded constraint

            ELSE
              IF ( cu < control%infinity ) THEN
                r2 = r2 + 1
                IF ( v_stat < 0 ) THEN
                  JX( r2 ) = 1
                ELSE IF ( v_stat > 0 ) THEN
                  JX( r2 ) = 2
                ELSE
                  IF ( basic < m ) THEN
                    basic = basic + 1
                    JX( r2 ) = - 1
                    IX( basic ) = r2
                  ELSE
                    JX( r2 ) = 0
                  END IF
                END IF

!  lower-bounded constraint

              ELSE
                r1 = r1 + 1
                IF ( v_stat == 0 .AND. basic < m ) THEN
                  basic = basic + 1
                  IX( basic ) = r1
                END IF
              END IF
            END IF
          END IF
        END DO
        IX( basic + 1 : m ) = 0
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  Error returns

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      RETURN

!  End of LPA_reorder

      END SUBROUTINE LPA_reorder

!-*-*-*-*-*-*-*-   L P A _ r e v e r t    S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE LPA_revert( prob, x_map, c_map, X, Y, Z, control, inform )

!  recover the solution from the reordered problem solved by LA04

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPA_control_type ), INTENT( IN ) :: control
      TYPE ( LPA_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, INTENT( IN ), ALLOCATABLE, DIMENSION( : ) :: x_map, c_map
      REAL ( KIND = wp ), INTENT( IN ), ALLOCATABLE, DIMENSION( : ) :: X, Y, Z

!  local variables

      INTEGER :: i, ii, j, l
      REAL ( KIND = wp ) :: yi
      CHARACTER ( LEN = 80 ) :: array_name

!  allocate space for the solution if necessary

      array_name = 'lpa: prob%X'
      CALL SPACE_resize_array( prob%n, prob%X,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: prob%Y'
      CALL SPACE_resize_array( prob%m, prob%Y,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: prob%Z'
      CALL SPACE_resize_array( prob%n, prob%Z,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  recover x from the reordered solution; remember to swap the sign of
!  variables that originally had one-sided upper bounds

      DO i = 1, prob%n
        ii = x_map( i )
        IF ( ii > 0 ) THEN
          IF ( prob%X_l( i ) <= - control%infinity .AND.                       &
               prob%X_u( i ) == zero ) THEN
            prob%X( i ) = - X( ii )
!           prob%Z( i ) = - Z( ii )
          ELSE
            prob%X( i ) = X( ii )
!           prob%Z( i ) = Z( ii )
          END IF
        ELSE
          prob%X( i ) = prob%X_l( i )
        END IF
      END DO

!  record the Lagrange multipliers, y

      DO i = 1, prob%m
        ii = c_map( i )
        IF ( ii > 0 ) THEN
          prob%Y( i ) = Y( ii )
        ELSE
          prob%Y( i ) = zero
        END IF
      END DO

!  compute the objective function value

      inform%obj =                                                             &
        DOT_PRODUCT( prob%G( : prob%n ), prob%X( : prob%n ) ) + prob%f

!  compute the constraint values c and the dual variables, z

      prob%Z( : prob%n ) = prob%G( : prob%n )
      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        l = 0
        DO i = 1, prob%m
          prob%C( i ) =                                                        &
            DOT_PRODUCT( prob%A%val( l + 1 : l + prob%n ), prob%X( : prob%n ) )
          yi = prob%Y( i )
          prob%Z( : prob%n ) = prob%Z( : prob%n ) -                            &
            prob%A%val( i + 1 : i + prob%n ) * yi
          l = l + prob%n
        END DO
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        DO i = 1, prob%m
          prob%C( i ) = zero
          yi = prob%Y( i )
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            j = prob%A%col( l )
            prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( j )
            prob%Z( j ) = prob%Z( j ) - prob%A%val( l ) * yi
          END DO
        END DO
      ELSE
        prob%C( : prob%m ) = zero
        DO l = 1, prob%A%ne
          i = prob%A%row( l ) ; j = prob%A%col( l )
          prob%C( i ) = prob%C( i ) +                                          &
            prob%A%val( l ) * prob%X( prob%A%col( l ) )
          prob%Z( j ) = prob%Z( j ) -                                          &
            prob%A%val( l ) * prob%Y( prob%A%row( l ) )
        END DO
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  Error returns

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      RETURN

!  End of LPA_revert

      END SUBROUTINE LPA_revert

!-*-*-*-*-*-   L P A _ r e o r d e r _ d u a l   S U B R O U T I N E  -*-*-*-*-

      SUBROUTINE LPA_reorder_dual( prob, f, m, n, A_val, A_row, A_ptr, B, C,   &
                                   kb, lb, sigma, BND, x_map, c_map, IX, JX,   &
                                   control, inform, C_stat, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Find a reordering of the data for the dual of the problem
!
!     minimize              l(x) = g(T) x + f
!         x
!     subject to the bounds,   x_l <=  x  <= x_u
!     inequality constraints   c_l <= A x <= c_u ,
!     and equality constraints    A_e x = c_e
!
!  where x is a vector of n components ( x_1, .... , x_n ), A is an m by n
!  matrix, and any of the bounds x_l, x_u, c_l, c_u may be infinite.
!
!  The dual for this problem is
!
!        maximize    c_e^T y + c_l^T y_l - c_u^T y_u + x_l^T z_l - x_u^T z_u
!      y_e,y_l,y_u,z_l,z_u
!  i.e.,
!      - minimize  - c_e^T y - c_l^T y_l + c_u^T y_u - x_l^T z_l + x_u^T z_u
!      y_e,y_l,y_u,z_l,z_u
!     subject to the bounds  ( y_l, y_u, z_l, z_u ) >= 0
!     and constraints      A_e^T y +  A^T (y_l - y_u) + z_l - z_u = g
!
!  where components of y_l, y_u, z_l, z_u that correspond to infinite
!  c_l, c_u, x_l, x_u are zero, and fixed variables are removed
!
!  The reordered problem takes the form
!
!     minimize                 c^T v + f'
!       x_r
!     subject to the equations A_r v = b,
!     and the non-negativities v_r(i) >= 0, i = lb, ...,  n
!
!  where v^T = ( y_e^T,y_l^T,y_u^T,z_^Tl,z^T_u),
!        A_r = (A_e^T  A^T  -A^T  I  -I)
!        c^T = ( c_e^T,c_l^T,c_u^T,x_^Tl,x^T_u),
!        b = g
!  and lb = dim(c_e)+1
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Arguments:
!
!  input variable that needs to be set:
!
!  prob is a structure of type QPT_type, whose components hold the
!   details of the problem. The following components must be set
!
!   f is a REAL variable that gives the term f in the objective function.
!
!   n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: n >= 1
!
!   m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m.  RESTRICTION: m >= 0
!
!   gradient_kind is an INTEGER variable which defines the type of linear
!    term of the objective function to be used. Possible values are
!
!     0  the linear term g will be zero. %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by G (see below)
!
!   G is a REAL array of length n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!
!   A is a structure of type SMT_type used to hold the matrix A.
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
!   X_l, X_u are REAL arrays of length n, which must be set by the user
!    to the values of the arrays x_l and x_u of lower and upper bounds on x.
!    Any bound x_l_i or x_u_i larger than or equal to infinity in absolute value
!    will be regarded as being infinite (see the entry control%infinity).
!    Thus, an infinite lower bound may be specified by setting the appropriate
!    component of X_l to a value smaller than -infinity, while an infinite
!    upper bound can be specified by setting the appropriate element of X_u
!    to a value larger than infinity.
!
!   C_l, C_u are REAL array of length m, which must be set by the user to
!    the values of the arrays c_l and c_u of lower and upper bounds on A x.
!    Any bound c_l_i or c_u_i larger than or equal to infinity in absolute value
!    will be regarded as being infinite (see the entry control%infinity).
!    Thus, an infinite lower bound may be specified by setting the appropriate
!    component of C_l to a value smaller than -infinity, while an infinite
!    upper bound can be specified by setting the appropriate element of C_u
!    to a value larger than infinity.
!
!  output variables that need not be set on input and describe the
!  reordered linear program:
!
!  f is a REAL variable that gives the term f' in the objective function.
!
!  n is an INTEGER variable that gives the number of dual optimization
!    parameters
!
!  m is an INTEGER variable that gives the number of dual linear constraints
!   (after removing fixed primal variables)
!
!  A_val is a REAL array whose components give the elements of A, stored by
!   column, with those for column i+1 immediately after those for column i;
!   the order within each column is arbitrary
!
!  A_row is an INTEGER array that holds the row indices of the components
!   of A corresponding to the entries in A_val
!
!  A_ptr is an INTEGER array of length n whose i-th entry gives the position
!   in A_val and A_row of the first entry in column i, i = 1, ..., m,
!   while A_ptr( n + 1 ) - 1 gives the lengths of A_val and A_row
!
!  B is a REAL array that holds the right-hand side vector b
!
!  C is a REAL array that holds the output objective gradient c
!
!  kb is an INTEGER variable that will be set to 0
!
!  lb is an INTEGER variable for which variables v_i >= 0 for i = lb, ..., n
!
!  sigma is a REAL variable whose value is used to flag infinite lower or
!    upper bounds (not needed)
!
!  BND( : 2, : kb ) is a REAL array that is allocated but not otherwise
!   needed for the dual problem since kb = 0
!
!  x_map is an INTEGER array whose j-th entry gives the position of the
!   original j-th variable in the reordered problem (0=removed)
!
!  c_map is an INTEGER array whose i-th entry gives the position of the
!   original i-th constraint row in the reordered problem (0=removed)
!
!  control is a structure of type LPA_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to LPA_initialize. See
!   LPA_initialize for details
!
!  inform is a structure of type LPA_inform_type that provides information
!    on exit from LPA_reorder. The component status has possible values:
!
!     0 Normal termination
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 one of the restrictions
!          prob%n     >=  1
!          prob%m     >=  0
!          prob%A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' }
!       has been violated
!
!    -5 The constraints are inconsistent
!
!  C_stat is an optional INTEGER array of length prob%m, which if present
!   should be set on entry to indicate the initial status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the active set.
!
!  X_stat is an optional INTEGER array of length prob%n, which if present
!    should be set on entry to indicate the initial status of the simple bound
!   constraints. Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is  not in the active set.
!
!   N.B. If C_stat and B_stat are present, it is expected that the number of
!   active general constraints and simple bounds should add to n. If there
!   are fewer than n, the package will choose inactive constraints and
!   variables to make up the deficit, while if there are more than n,
!   some will be freed. In addition any constraint whose lower and upper
!   bounds coincide will automatically be in the active set.
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPA_control_type ), INTENT( IN ) :: control
      TYPE ( LPA_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, INTENT( OUT ) :: m, n, kb, lb
      REAL ( KIND = wp ), INTENT( OUT ) :: f, sigma
      INTEGER, INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: x_map, c_map
      INTEGER, INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: IX, JX
      INTEGER, INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: A_row, A_ptr
      REAL ( KIND = wp ), INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: A_val
      REAL ( KIND = wp ), INTENT( OUT ), ALLOCATABLE, DIMENSION( : ) :: B, C
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BND
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  local variables

      INTEGER :: i, ii, j, jj, l, m_e, m_l, m_u, n_l, n_u
      INTEGER :: free, lower, range, upper, fixed, basic
      INTEGER :: c_lower, c_range, c_upper, c_equality, a_ne, a_type, v_stat
      INTEGER :: m_es, m_is, n_s, ai_len

      REAL ( KIND = wp ) :: xl, xu, cl, cu
      CHARACTER ( LEN = 80 ) :: array_name

!  provide room for the mapping arrays and the residuals for fixed variables

      array_name = 'lpa: x_map'
      CALL SPACE_resize_array( prob%n, x_map,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: c_map'
      CALL SPACE_resize_array( prob%m, c_map,                                  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: prob%C'
      CALL SPACE_resize_array( prob%m, prob%C,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  original variable i will become x_map(i) in the reordered problem, with
!  x_map(i) = 0 if the variable is fixed. There will be m reordeed variables
!  in total (these correspond to dual constraints). Also count how many fall
!  into each of the categories: fixed, lower, upper, range and free

      free = 0 ; lower = 0 ;  upper = 0 ; range = 0 ; fixed = 0 ; m = 0

      DO j = 1, prob%n
        xl = prob%X_l( j ) ; xu = prob%X_u( j )

!  fixed variable

        IF ( xu == xl ) THEN
          fixed = fixed + 1
          x_map( j ) = 0

!  the variable has no lower bound

        ELSE IF ( xl <= - control%infinity ) THEN

!  free variable

          IF ( xu >= control%infinity ) THEN
            free = free + 1 ; m = m + 1
            x_map( j ) = m

!  upper-bounded variable

          ELSE
            upper = upper + 1 ; m = m + 1
            x_map( j ) = m
          END IF

!  the variable has a lower bound

        ELSE
          IF ( xu < control%infinity ) THEN

!  inconsistent upper bound

!           IF ( xu < xl ) THEN
!             inform%status = GALAHAD_error_primal_infeasible
!             RETURN

!  range-bounded variable

!           ELSE
              range = range + 1 ; m = m + 1
              x_map( j ) = m
!           END IF

!  lower-bounded variable

          ELSE
            lower = lower + 1 ; m = m + 1
            x_map( j ) = m
          END IF
        END IF
      END DO

!  see which storage scheme is used for A

      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        a_type = a_dense
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_type = a_sparse_by_rows
      ELSE
        a_type = a_coordinate
      END IF

!  count how many non-fixed entries there are in each constraint. c_map is
!  temporarily used for this: c_map(i) gives the number of entries in row i.
!  Also set C = A_fx x_fx, where fx are the indices of the fixed variables

!  ** pass 1 through A **

      c_map( : prob%m ) = 0 ; prob%C = zero

!  original dense storage

      IF ( a_type == a_dense ) THEN
        DO j = 1, prob%n
          l = j
          IF ( x_map( j ) == 0 ) THEN
            xl = PROB%X_l( j )
            DO i = 1, prob%m
              prob%C( i ) = prob%C( i ) + prob%A%val( l ) * xl
              l = l + prob%n
            END DO
          ELSE
            DO i = 1, prob%m
              IF ( prob%A%val( l ) /= zero ) c_map( i ) = c_map( i ) + 1
              l = l + prob%n
            END DO
          END IF
        END DO
      ELSE
        c_map( : prob%m ) = 0

!  original co-ordinate storage

        IF ( a_type == a_coordinate ) THEN
          DO l = 1, prob%A%ne
            i = prob%A%row( l )
            IF ( x_map(  prob%A%col( l ) ) > 0 ) THEN
              c_map( i ) = c_map( i ) + 1
            ELSE
              prob%C( i )                                                      &
                = prob%C( i ) + prob%A%val( l ) * prob%X_l( prob%A%col( l ) )
            END IF
          END DO

!  original row-wise storage

        ELSE
          DO i = 1, prob%m
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              IF ( x_map( prob%A%col( l ) ) > 0 ) THEN
                c_map( i ) = c_map( i ) + 1
              ELSE
                prob%C( i )                                                    &
                  = prob%C( i ) + prob%A%val( l ) * prob%X_l( prob%A%col( l ) )
              END IF
            END DO
          END DO
        END IF
      END IF

!  run through the constraints to see how many fall into each of the categories:
!  lower(c_lower), upper(c_upper), range(c_range), and equality (c_equality).
!  Also count the number of non-zeros in A_r (a_ne)

      c_lower = 0 ; c_upper = 0 ; c_range = 0 ; c_equality = 0 ; a_ne = 0

      DO i = 1, prob%m
        cl = prob%C_l( i ) ; cu = prob%C_u( i )

!  equality constraint

        IF ( cu == cl ) THEN
          IF ( c_map( i ) > 0 ) THEN
            a_ne = a_ne + c_map( i )
            c_equality = c_equality + 1

!  deal with an inconsistent null equality constraint

          ELSE
            IF ( ABS( cu - prob%C( i ) ) > control%feas_tol ) THEN
              inform%status = GALAHAD_error_primal_infeasible
              RETURN
            END IF
          END IF

!  upper-bounded constraint

        ELSE IF ( cl <= - control%infinity ) THEN
          IF ( cu < control%infinity ) THEN
            IF ( c_map( i ) > 0 ) THEN
              a_ne = a_ne + c_map( i )
              c_upper = c_upper + 1

!  deal with an inconsistent null upper-bounded constraint

            ELSE
              IF ( cu < prob%C( i ) ) THEN
                inform%status = GALAHAD_error_primal_infeasible
                RETURN
              END IF
            END IF
          END IF

!  inconsistent constraints

        ELSE
          IF ( cu < control%infinity ) THEN
!           IF ( cu < cl ) THEN
!             inform%status = GALAHAD_error_primal_infeasible
!             RETURN

!  range-bounded constraint

!           ELSE
              IF ( c_map( i ) > 0 ) THEN
                a_ne = a_ne + 2 * c_map( i )
                c_range = c_range + 1

!  deal with an inconsistent range-bounded constraint

              ELSE
                IF ( cl > prob%C( i ) .OR. cu < prob%C( i ) ) THEN
                  inform%status = GALAHAD_error_primal_infeasible
                  RETURN
                END IF
              END IF
!           END IF

!  lower-bounded constraint

          ELSE
            IF ( c_map( i ) > 0 ) THEN
              a_ne = a_ne + c_map( i )
              c_lower = c_lower + 1

!  deal with an inconsistent null lower-bounded constraint

            ELSE
              IF ( cl > prob%C( i ) ) THEN
                inform%status = GALAHAD_error_primal_infeasible
                RETURN
              END IF
            END IF
          END IF
        END IF
      END DO

!  record m_e, m_l, m_u the numbers of equality, >=, and <= constraints

      m_e = c_equality ; m_l = c_lower + c_range ; m_u = c_upper + c_range

!  record n_l, n_u the numbers of >=, and <= bound constraints

      n_l = lower + range ; n_u = upper + range

      a_ne = a_ne + n_l + n_u

!  set the starting and ending addresses as required
!
!         <- m_e -> <- m_l + m_u  ->  <-  n_l + n_u  ->
!        -----------------------------------------------
!  c   = |  c_e^T  |  c_l^T  - c_u^T |  x_l^T  - x_u^T !
!        -----------------------------------------------
!        |         |                 |                 | ^
!  A_r = |  A_e^T  |  A_l^T  - A_u^T |    I      - I   | m
!        |         |                 |                 | |
!        -----------------------------------------------
!       ^         ^ ^               ^                 ^
!       |         | |               |                 |
!       m_es   m_is lb             n_s                n
!
!
!  the equality and inequality constraints and simple bounds are arranged in
!  increasing input order, any pair of ranges will occur consecutively, lower
!  before upper

      kb = 0
      m_es = 0
      m_is = m_e
      lb = m_e + 1
      n_s = m_is + m_l + m_u
      n = n_s + n_l + n_u
      sigma = ten * control%infinity
      IF ( control%scale ) a_ne = a_ne + n + m

!  allocate space for the reordered problem

      array_name = 'lpa: A_val'
      CALL SPACE_resize_array( a_ne, A_val,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: A_row'
      CALL SPACE_resize_array( a_ne, A_row,                                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: A_ptr'
      CALL SPACE_resize_array( n + 1, A_ptr,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: C'
      CALL SPACE_resize_array( n, C,                                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: B'
      CALL SPACE_resize_array( m, B,                                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: BND'
      CALL SPACE_resize_array( 2, kb, BND,                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  run through the constraints again, setting A_ptr(j+1) to the number of
!  nonzeros in column j of A_r, and resetting c_map(i) to indicate the
!  column index of A_r of the (first) occurence of a_i^T as follows: if
!  1 <= c_map(i) <= n, - a_i^T will occur in column c_map(i) and the constraint
!  is just upper bounded, if - n <= c_map(i) <= - 1, a_i^T occurs in column
!  - c_map(i) and the constraint is just lower bounded, and if n + 1 <=
!  c_map(i) <= 2 n, a_i^T will occur in column c_map(i) - n, - a_i^T will
!  occur in column  c_map(i) - n + 1, and the constraint is range bounded.
!  In addition, set c

      DO i = 1, prob%m
        cl = prob%C_l( i ) ; cu = prob%C_u( i )
        ai_len = c_map( i )

!  equality constraint

        IF ( cu == cl ) THEN
          IF ( ai_len > 0 ) THEN
            m_es = m_es + 1
            A_ptr( m_es + 1 ) = ai_len
            C( m_es ) = prob%C( i ) - cl
            c_map( i ) = - m_es
          END IF

!  upper-bounded constraint

        ELSE IF ( cl <= - control%infinity ) THEN
          IF ( cu < control%infinity ) THEN
            IF ( c_map( i ) > 0 ) THEN
              m_is = m_is + 1
              A_ptr( m_is + 1 ) = ai_len
              C( m_is ) = cu - prob%C( i )
              c_map( i ) = m_is
            END IF
          END IF

!  range-bounded constraint

        ELSE
          IF ( cu < control%infinity ) THEN
            IF ( c_map( i ) > 0 ) THEN
              m_is = m_is + 1
              A_ptr( m_is + 1 ) = ai_len
              C( m_is ) = prob%C( i ) - cl
              c_map( i ) = m_is + n
              m_is = m_is + 1
              A_ptr( m_is + 1 ) = ai_len
              C( m_is ) = cu - prob%C( i )
            END IF

!  lower-bounded constraint

          ELSE
            IF ( c_map( i ) > 0 ) THEN
              m_is = m_is + 1
              A_ptr( m_is + 1 ) = ai_len
              C( m_is ) = prob%C( i ) - cl
              c_map( i ) = - m_is
            END IF
          END IF
        END IF
      END DO

!  adjust A_ptr so that A_ptr(j) points to the storage location one before the
!  starting entry in column j (and A_ptr(n_s+1) points to the end of the part
!  of the array containing A^T)

      A_ptr( 1 ) = 0
      DO j = 2, n_s + 1
        A_ptr( j ) = A_ptr( j ) + A_ptr( j - 1 )
      END DO

!  allocate space for values and row indices for the reordered problem

!  introduce the entries of A into A_r

!  ** pass 2 through A **

!  original dense storage

      IF ( a_type == a_dense ) THEN
        l = 0
        DO i = 1, prob%m
          ii = c_map( i )

!  range-bounded constraint

          IF ( ii > n ) THEN
            ii = ii - n
            DO j = 1, prob%n
              l = l + 1 ; jj = x_map( j )
              IF ( jj > 0 .AND. prob%A%val( l ) /= zero ) THEN
                A_ptr( ii ) = A_ptr( ii ) + 1
                A_val( A_ptr( ii ) ) = prob%A%val( l )
                A_row( A_ptr( ii ) ) = jj
                A_ptr( ii + 1 ) = A_ptr( ii + 1 ) + 1
                A_val( A_ptr( ii + 1 ) ) = - prob%A%val( l )
                A_row( A_ptr( ii + 1 ) ) = jj
              END IF
            END DO

!  upper-bounded constraint

          ELSE IF ( ii > 0 ) THEN
            DO j = 1, prob%n
              l = l + 1 ; jj = x_map( j )
              IF ( jj > 0 .AND. prob%A%val( l ) /= zero ) THEN
                A_ptr( ii ) = A_ptr( ii ) + 1
                A_val( A_ptr( ii ) ) = - prob%A%val( l )
                A_row( A_ptr( ii ) ) = jj
              END IF
            END DO

!  lower-bounded constraint

          ELSE IF ( ii < 0 ) THEN
            ii = - ii
            DO j = 1, prob%n
              l = l + 1 ; jj = x_map( j )
              IF ( jj > 0 .AND. prob%A%val( l ) /= zero ) THEN
                A_ptr( ii ) = A_ptr( ii ) + 1
                A_val( A_ptr( ii ) ) = prob%A%val( l )
                A_row( A_ptr( ii ) ) = jj
              END IF
            END DO

!  free constraint

          ELSE
            l = l + prob%n
          END IF
        END DO

!  original co-ordinate storage

      ELSE
        IF ( a_type == a_coordinate ) THEN
          DO l = 1, prob%A%ne
            j = prob%A%col( l )
            jj = x_map( j )
            IF ( jj > 0 ) THEN
              ii = c_map( prob%A%row( l ) )

!  range-bounded constraint

              IF ( ii > n ) THEN
                ii = ii - n
                A_ptr( ii ) = A_ptr( ii ) + 1
                A_val( A_ptr( ii ) ) = prob%A%val( l )
                A_row( A_ptr( ii ) ) = jj
                A_ptr( ii + 1 ) = A_ptr( ii + 1 ) + 1
                A_val( A_ptr( ii + 1 ) ) = - prob%A%val( l )
                A_row( A_ptr( ii + 1 ) ) = jj

!  upper-bounded constraint

              ELSE IF ( ii > 0 ) THEN
                A_ptr( ii ) = A_ptr( ii ) + 1
                A_val( A_ptr( ii ) ) = - prob%A%val( l )
                A_row( A_ptr( ii ) ) = jj

!  lower-bounded constraint

              ELSE IF ( ii < 0 ) THEN
                ii = - ii
                A_ptr( ii ) = A_ptr( ii ) + 1
                A_val( A_ptr( ii ) ) = prob%A%val( l )
                A_row( A_ptr( ii ) ) = jj
              END IF
            END IF
          END DO

!  original row-wise storage

        ELSE
          DO i = 1, prob%m
            ii = c_map( i )

!  range-bounded constraint

            IF ( ii > n ) THEN
              ii = ii - n
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                jj = x_map( j )
                IF ( jj > 0 ) THEN
                  A_ptr( ii ) = A_ptr( ii ) + 1
                  A_val( A_ptr( ii ) ) = prob%A%val( l )
                  A_row( A_ptr( ii ) ) = jj
                  A_ptr( ii + 1 ) = A_ptr( ii + 1 ) + 1
                  A_val( A_ptr( ii + 1 ) ) = - prob%A%val( l )
                  A_row( A_ptr( ii + 1 ) ) = jj
                END IF
              END DO

!  upper-bounded constraint

            ELSE IF ( ii > 0 ) THEN
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                jj = x_map( j )
                IF ( jj > 0 ) THEN
                  A_ptr( ii ) = A_ptr( ii ) + 1
                  A_val( A_ptr( ii ) ) = - prob%A%val( l )
                  A_row( A_ptr( ii ) ) = jj
                END IF
              END DO

!  lower-bounded constraint

            ELSE IF ( ii < 0 ) THEN
              ii = - ii
              DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                j = prob%A%col( l )
                jj = x_map( j )
                IF ( jj > 0 ) THEN
                  A_ptr( ii ) = A_ptr( ii ) + 1
                  A_val( A_ptr( ii ) ) = prob%A%val( l )
                  A_row( A_ptr( ii ) ) = jj
                END IF
              END DO
            END IF
          END DO
        END IF
      END IF

!  reset the column pointers

      DO i = n_s, 1, - 1
        A_ptr( i + 1 ) = A_ptr( i ) + 1
      END DO
      A_ptr( 1 ) = 1

!  now run through the simple bounds again, setting A_ptr, A_rwo and A_val
!  to identity columns in A_r, and resetting x_map(j) to indicate the
!  column index of A_r of the (first) occurence of e_j^T as follows: if
!  1 <= x_map(i) <= n, - e_i^T will occur in column x_map(j) and the variable
!  is just upper bounded, if - n <= x_map(j) <= - 1, e_j^T occurs in column
!  - x_map(i) and the variable is just lower bounded, and if !  n + 1 <=
!  x_map(i) <= 2 n, e_j^T will occur in column x_map(i) - n, - e_i^T will
!  occur in column x_map(j) - n + 1, and the constraint is range bounded. In
!  addition, continue setting c, and also set b to the appropriate entries of g

      m = 0 ;  f = prob%f
      DO j = 1, prob%n
        xl = prob%X_l( j ) ; xu = prob%X_u( j )

!  flag fixed variables with x_map = - n - 1, and compute the constant term
!  in the objective function, f

        IF ( xu == xl ) THEN
          f = f + prob%G( j ) * xl
          x_map( j ) = - n - 1
          CYCLE
        ELSE
          m = m + 1
          B( m ) = prob%G( j )
        END IF

!  upper bounded variable

        IF ( xl <= - control%infinity ) THEN
          IF ( xu < control%infinity ) THEN
            n_s = n_s + 1 ; l = A_ptr( n_s )
            A_row( l ) = m
            A_val( l ) = - one
            A_ptr( n_s + 1 ) = A_ptr( n_s ) + 1
            C( n_s ) = xu
            x_map( j ) = n_s

! free variable

          ELSE
            x_map( j ) = 0
          END IF

!  range bounded variable

        ELSE
          IF ( xu < control%infinity ) THEN
            n_s = n_s + 1 ; l = A_ptr( n_s )
            A_row( l ) = m
            A_val( l ) = one
            A_ptr( n_s + 1 ) = A_ptr( n_s ) + 1
            C( n_s ) = - xl
            n_s = n_s + 1 ; l = A_ptr( n_s )
            A_row( l ) = m
            A_val( l ) = - one
            A_ptr( n_s + 1 ) = A_ptr( n_s ) + 1
            C( n_s ) = xu
            x_map( j ) = n_s + n

!  lower bounded variable

          ELSE
            n_s = n_s + 1 ; l = A_ptr( n_s )
            A_row( l ) = m
            A_val( l ) = one
            A_ptr( n_s + 1 ) = A_ptr( n_s ) + 1
            C( n_s ) = - xl
            x_map( j ) = - n_s
          END IF
        END IF
      END DO

!  allocate space to hold the lists of basic and non-basic variables

      array_name = 'lpa: IX'
      CALL SPACE_resize_array( m, IX,                                          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: JX'
      CALL SPACE_resize_array( kb, JX,                                         &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error,                &
             exact_size = .TRUE. )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  if a warm start is requested, set the basic and non-basic variables

      IF ( PRESENT( C_stat ) .AND. PRESENT( X_stat ) ) THEN

!  assess the status of the problem variables

        basic = 0
        DO j = 1, prob%n
          v_stat = X_stat( j )
          IF ( v_stat == 0 ) CYCLE
          basic = basic + 1
          IF ( basic > m ) EXIT
          jj = x_map( j )

!  ignore fixed variables

          IF ( jj == - n - 1 ) THEN
            CYCLE

!  range-bounded variables

          ELSE IF ( jj > n ) THEN
            IF ( v_stat < 0 ) THEN
              IX( basic ) = jj - n
            ELSE
              IX( basic ) = jj - n + 1
            END IF

!  upper-bounded variables

          ELSE IF ( jj > 0 ) THEN
            IX( basic ) = jj

!  lower-bounded variables

          ELSE IF ( jj < 0 ) THEN
            IX( basic ) = - jj
          END IF
        END DO

!  now cosider the ststus of the constraints

        DO i = 1, prob%m
          v_stat = C_stat( i )
          IF ( v_stat == 0 ) CYCLE
          basic = basic + 1
          IF ( basic > m ) EXIT
          ii = c_map( i )

!  range-bounded constraint

          IF ( ii > n ) THEN
            IF ( v_stat < 0 ) THEN
              IX( basic ) = ii - n
            ELSE
              IX( basic ) = ii - n + 1
            END IF

!  upper-bounded constraint

          ELSE IF ( ii > 0 ) THEN
            IX( basic ) = ii

!  lower-bounded constraint

          ELSE IF ( ii < 0 ) THEN
            IX( basic ) = - ii
          END IF
        END DO

        IX( basic + 1 : m ) = 0
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  Error returns

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      RETURN

!  End of LPA_reorder_dual

      END SUBROUTINE LPA_reorder_dual

!-*-*-*-*-*-*-   L P A _ r e v e r t _ d u a l   S U B R O U T I N E  -*-*-*-*-

      SUBROUTINE LPA_revert_dual( prob, x_map, c_map, n, X, Y, Z,              &
                                  control, inform )

!  recover the solution from the reordered problem solved by LA04

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n
      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( LPA_control_type ), INTENT( IN ) :: control
      TYPE ( LPA_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, INTENT( IN ), ALLOCATABLE, DIMENSION( : ) :: x_map, c_map
      REAL ( KIND = wp ), INTENT( IN ), ALLOCATABLE, DIMENSION( : ) :: X, Y, Z

!  local variables

      INTEGER :: i, ii, j, l, m
      REAL ( KIND = wp ) :: xl, yi
      CHARACTER ( LEN = 80 ) :: array_name

!  allocate space for the solution if necessary

      array_name = 'lpa: prob%X'
      CALL SPACE_resize_array( prob%n, prob%X,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: prob%Y'
      CALL SPACE_resize_array( prob%m, prob%Y,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lpa: prob%Z'
      CALL SPACE_resize_array( prob%n, prob%Z,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  recover x from the reordered dual solution

      m = 0
      DO j = 1, prob%n
        xl = prob%X_l( j )
        IF ( xl == prob%X_u( j ) ) THEN
          prob%X( j ) = xl
        ELSE
          m = m + 1
          prob%X( j ) = - Y( m )
        END IF
      END DO

!  compute the objective function value

      inform%obj =                                                             &
        DOT_PRODUCT( prob%G( : prob%n ), prob%X( : prob%n ) ) + prob%f

!  compute the constraint values

      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        l = 0
        DO i = 1, prob%m
          prob%C( i ) =                                                        &
            DOT_PRODUCT( prob%A%val( i + 1 : i + prob%n ), prob%X( : prob%n ) )
          l = l + prob%n
        END DO
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        DO i = 1, prob%m
          prob%C( i ) = zero
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            prob%C( i ) =                                                      &
              prob%C( i ) + prob%A%val( l ) * prob%X( prob%A%col( l ) )
          END DO
        END DO
      ELSE
        prob%C( : prob%m ) = zero
        DO l = 1, prob%A%ne
          i = prob%A%row( l )
          prob%C( i ) =                                                        &
            prob%C( i ) + prob%A%val( l ) * prob%X( prob%A%col( l ) )
        END DO
      END IF

!  recover the Lagrange multipliers from the reordered dual solution

      DO i = 1, prob%m
        ii = c_map( i )

!  range-bounded constraint

        IF ( ii > n ) THEN
          ii = ii - n
          prob%Y( i ) = X( ii ) - X( ii + 1 )

!  upper-bounded constraint

        ELSE IF ( ii > 0 ) THEN
          prob%Y( i ) = - X( ii )

!  lower-bounded constraint

        ELSE IF ( ii < 0 ) THEN
          prob%Y( i ) = X( - ii )

!  free constraint

        ELSE
          prob%Y( i ) = zero
        END IF
      END DO

!  compute the constraint values c and the dual variables, z

      prob%Z( : prob%n ) = prob%G( : prob%n )
      IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
        l = 0
        DO i = 1, prob%m
          prob%C( i ) =                                                        &
            DOT_PRODUCT( prob%A%val( i + 1 : i + prob%n ), prob%X( : prob%n ) )
          yi = prob%Y( i )
          prob%Z( : prob%n ) = prob%Z( : prob%n ) -                            &
            prob%A%val( i + 1 : i + prob%n ) * yi
          l = l + prob%n
        END DO
      ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
        DO i = 1, prob%m
          prob%C( i ) = zero
          yi = prob%Y( i )
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            j = prob%A%col( l )
            prob%C( i ) = prob%C( i ) + prob%A%val( l ) * prob%X( j )
            prob%Z( j ) = prob%Z( j ) - prob%A%val( l ) * yi
          END DO
        END DO
      ELSE
        prob%C( : prob%m ) = zero
        DO l = 1, prob%A%ne
          i = prob%A%row( l ) ; j = prob%A%col( l )
          prob%C( i ) = prob%C( i ) +                                          &
            prob%A%val( l ) * prob%X( prob%A%col( l ) )
          prob%Z( j ) = prob%Z( j ) -                                          &
            prob%A%val( l ) * prob%Y( prob%A%row( l ) )
        END DO
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  Error returns

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      RETURN

!  End of LPA_revert_dual

      END SUBROUTINE LPA_revert_dual

!  End of module LPA

    END MODULE GALAHAD_LPA_double
