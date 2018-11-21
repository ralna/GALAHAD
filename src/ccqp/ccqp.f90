! THIS VERSION: GALAHAD 2.7 - 17/07/2015 AT 13:00 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ C C Q P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.7. July 17th 2015

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

MODULE GALAHAD_CCQP_double

!      ----------------------------------------------
!     |                                              |
!     | Minimize the quadratic objective function    |
!     |                                              |
!     |        1/2 x^T H x + g^T x + f               |
!     |                                              |
!     | or linear/seprable objective function        |
!     |                                              |
!     |  1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f   |
!     |                                              |
!     | subject to the linear constraints and bounds |
!     |                                              |
!     |           c_l <= A x <= c_u                  |
!     |           x_l <=  x <= x_u                   |
!     |                                              |
!     | for some positive-semi-definite Hessian H    |
!     | or |(possibly zero) diagonal matrix W using  |
!     | an infeasible-point primal-dual to dual      |
!     | gradient-projection crossover method         |
!     |                                              |
!      ----------------------------------------------

!$    USE omp_lib
!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_STRING_double, ONLY: STRING_pleural, STRING_verb_pleural,    &
                                       STRING_ies, STRING_are, STRING_ordinal
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPP_double, CCQP_dims_type => QPP_dims_type
      USE GALAHAD_QPD_double, CCQP_data_type => QPD_data_type,                 &
                              CCQP_AX => QPD_AX, CCQP_HX => QPD_HX,            &
                              CCQP_abs_AX => QPD_abs_AX,                       &
                              CCQP_abs_HX => QPD_abs_HX
      USE GALAHAD_LMS_double, ONLY: LMS_data_type, LMS_apply_lbfgs
      USE GALAHAD_SORT_double, ONLY: SORT_heapsort_build,                      &
                               SORT_heapsort_smallest, SORT_inverse_permute
      USE GALAHAD_FDC_double
      USE GALAHAD_CQP_double
      USE GALAHAD_DQP_double
      USE GALAHAD_CRO_double
      USE GALAHAD_ROOTS_double, ONLY: ROOTS_terminate
      USE GALAHAD_FIT_double, ONLY: FIT_terminate
      USE GALAHAD_GLTR_double, ONLY: GLTR_terminate
      USE GALAHAD_SBLS_double, ONLY: SBLS_terminate
      USE GALAHAD_SLS_double, ONLY: SLS_terminate
      USE GALAHAD_SCU_double, ONLY: SCU_terminate
      USE GALAHAD_NORMS_double, ONLY: TWO_norm
      USE GALAHAD_RPD_double, ONLY: RPD_inform_type, RPD_write_qp_problem_data

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CCQP_initialize, CCQP_read_specfile, CCQP_solve,               &
                CCQP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,  &
                CCQP_Ax, CCQP_data_type, CCQP_dims_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: no_last = - 1000
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: three = 3.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: eight = 8.0_wp
      REAL ( KIND = wp ), PARAMETER :: sixteen = 16.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: tenm5 = ten ** ( - 5 )
      REAL ( KIND = wp ), PARAMETER :: tenm7 = ten ** ( - 7 )
      REAL ( KIND = wp ), PARAMETER :: tenm10 = ten ** ( - 10 )
      REAL ( KIND = wp ), PARAMETER :: ten4 = ten ** 4
      REAL ( KIND = wp ), PARAMETER :: ten5 = ten ** 5
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: onemeps = one - epsmch
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: rminvr_zero = epsmch
      REAL ( KIND = wp ), PARAMETER :: twentyeps = two * teneps
      REAL ( KIND = wp ), PARAMETER :: stop_alpha = ten ** ( -15 )
      REAL ( KIND = wp ), PARAMETER :: relative_pivot_default = 0.01_wp

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CCQP_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   indicate whether and how much of the input problem
!    should be restored on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER :: restore_problem = 2

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 52

!    specifies the unit number to write generated QPLIB file describing the
!     current problem

        INTEGER :: qplib_file_device = 53

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!   perturb_h will be added to the Hessian

        REAL ( KIND = wp ) :: perturb_h = zero

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!   the equality constraints will be preprocessed to remove any linear
!    dependencies if true

        LOGICAL :: remove_dependencies = .TRUE.

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!  if %crossover is true, cross over the solution to one defined by
!   linearly-independent constraints if possible
!
        LOGICAL :: crossover = .TRUE.

!   if %space_critical true, every effort will be made to use as little
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
         "CCQPPROB.SIF"  // REPEAT( ' ', 18 )

!  name of generated QPLIB file containing input problem

        CHARACTER ( LEN = 30 ) :: qplib_file_name =                            &
         "CCQPPROB.qplib"  // REPEAT( ' ', 16 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control

!  control parameters for CQP

        TYPE ( CQP_control_type ) :: CQP_control

!  control parameters for DQP

        TYPE ( DQP_control_type ) :: DQP_control

!  control parameters for CRO

        TYPE ( CRO_control_type ) :: CRO_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CCQP_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ):: factorize = 0.0

!  the CPU time spent computing the search direction

        REAL ( KIND = wp ) :: solve = 0.0

!  the total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  the clock time spent preprocessing the problem

        REAL ( KIND = wp ) :: clock_preprocess = 0.0

!  the clock time spent detecting linear dependencies

        REAL ( KIND = wp ) :: clock_find_dependent = 0.0

!  the clock time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the clock time spent factorizing the required matrices

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the clock time spent computing the search direction

        REAL ( KIND = wp ) :: clock_solve = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: CCQP_inform_type

!  return status. See CCQP_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER  ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER  ( KIND = long ) :: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER :: nfacts = - 1

!  the number of threads used

        INTEGER :: threads = 1

!  the value of the objective function at the best estimate of the solution
!   determined by CCQP_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  the smallest pivot which was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  timings (see above)

        TYPE ( CCQP_time_type ) :: time

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform

!  inform parameters for CQP

        TYPE ( CQP_inform_type ) :: CQP_inform

!  inform parameters for DQP

        TYPE ( DQP_inform_type ) :: DQP_inform

!  inform parameters for CRO

        TYPE ( CRO_inform_type ) :: CRO_inform

!  inform parameters for RPD

        TYPE ( RPD_inform_type ) :: RPD_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   C C Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE CCQP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for CCQP. This routine should be called before
!  CCQP_solve
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

      TYPE ( CCQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( CCQP_control_type ), INTENT( OUT ) :: control
      TYPE ( CCQP_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Initalize CQP components

      CALL CQP_initialize( data, control%CQP_control,                          &
                           inform%CQP_inform  )
      control%CQP_control%prefix = '" - CQP:"                     '

!  Initalize DQP components

      CALL DQP_initialize( data, control%DQP_control,                          &
                           inform%DQP_inform  )
      control%DQP_control%prefix = '" - DQP:"                     '

!  Initalize FDC components

      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform  )
      control%FDC_control%max_infeas = control%CQP_control%stop_abs_p
      control%FDC_control%prefix = '" - FDC:"                     '

!  Set CRO control parameters

      CALL CRO_initialize( data%CRO_data, control%CRO_control,                 &
                           inform%CRO_inform )
      control%CRO_control%prefix = '" - CRO:"                     '

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of CCQP_initialize

      END SUBROUTINE CCQP_initialize

!-*-*-*-*-   C C Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE CCQP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by CCQP_initialize could (roughly)
!  have been set as:

! BEGIN CCQP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  start-print                                       -1
!  stop-print                                        -1
!  restore-problem-on-output                         2
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  identical-bounds-tolerance                        1.0D-15
!  perturb-hessian-by                                0.0
!  infinity-value                                    1.0D+19
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  remove-linear-dependencies                        T
!  treat-zero-bounds-as-general                      F
!  cross-over-solution                               T
!  array-syntax-worse-than-do-loop                   F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  generate-qplib-file                               F
!  sif-file-name                                     CCQPPROB.SIF
!  qplib-file-name                                   CCQPPROB.qplib
!  output-line-prefix                                ""
! END CCQP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( CCQP_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: restore_problem = print_level + 1
      INTEGER, PARAMETER :: sif_file_device = restore_problem + 1
      INTEGER, PARAMETER :: qplib_file_device = sif_file_device + 1
      INTEGER, PARAMETER :: infinity = qplib_file_device + 1
      INTEGER, PARAMETER :: identical_bounds_tol = infinity + 1
      INTEGER, PARAMETER :: perturb_h = identical_bounds_tol + 1
      INTEGER, PARAMETER :: cpu_time_limit = perturb_h + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: remove_dependencies = clock_time_limit + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general =                     &
                              remove_dependencies + 1
      INTEGER, PARAMETER :: crossover = treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: space_critical = crossover + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: generate_qplib_file = generate_sif_file + 1
      INTEGER, PARAMETER :: sif_file_name = generate_qplib_file + 1
      INTEGER, PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER, PARAMETER :: prefix = qplib_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'CCQP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( perturb_h )%keyword = 'perturb-hessian-by'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( crossover )%keyword = 'cross-over-solution'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

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
     CALL SPECFILE_assign_value( spec( restore_problem ),                      &
                                 control%restore_problem,                      &
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
     CALL SPECFILE_assign_value( spec( perturb_h ),                            &
                                 control%perturb_h,                            &
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

     CALL SPECFILE_assign_value( spec( remove_dependencies ),                  &
                                 control%remove_dependencies,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),         &
                                 control%treat_zero_bounds_as_general,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( crossover ),                            &
                                 control%crossover,                            &
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

!  Read the specfile for CQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL CQP_read_specfile( control%CQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CQP' )
      ELSE
        CALL CQP_read_specfile( control%CQP_control, device )
      END IF

!  Read the specfile for DQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL DQP_read_specfile( control%DQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-DQP' )
      ELSE
        CALL DQP_read_specfile( control%DQP_control, device )
      END IF

!  Read the specfile for FDC

      IF ( PRESENT( alt_specname ) ) THEN
        CALL FDC_read_specfile( control%FDC_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-FDC' )
      ELSE
        CALL FDC_read_specfile( control%FDC_control, device )
      END IF
      control%FDC_control%max_infeas = control%CQP_control%stop_abs_p

!  Read the specfile for CRO

      IF ( PRESENT( alt_specname ) ) THEN
        CALL CRO_read_specfile( control%CRO_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CRO' )
      ELSE
        CALL CRO_read_specfile( control%CRO_control, device )
      END IF

      RETURN

      END SUBROUTINE CCQP_read_specfile

!-*-*-*-*-*-*-*-*-*-   C C Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE CCQP_solve( prob, data, control, inform, C_stat, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Minimize the quadratic objective
!
!        1/2 x^T H x + g^T x + f
!
!  or the linear/separable objective
!
!        1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  where
!
!               (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!  and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
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
!    to be solved since the last call to CCQP_initialize, and .FALSE. if
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
!        feasible region will be found if in addition %Hessian_kind is 0.
!        %G (see below) need not be set
!
!     1  each component of the linear terms g will be one.
!        %G (see below) need not be set
!
!     any other value - the gradients will be those given by %G (see below)
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region will be found. %WEIGHT (see below) need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the positive semi-definite Hessian H will be used
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H (except for the L-BFGS case). Eight storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       H%type( 1 : 10 ) = TRANSFER( 'COORDINATE', H%type )
!       H%val( : )   the values of the components of H
!       H%row( : )   the row indices of the components of H
!       H%col( : )   the column indices of the components of H
!       H%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of H
!
!    ii) sparse, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 14 ) = TRANSFER( 'SPARSE_BY_ROWS', H%type )
!       H%val( : )   the values of the components of H, stored row by row
!       H%col( : )   the column indices of the components of H
!       H%ptr( : )   pointers to the start of each row, and past the end of
!                    the last row
!
!    iii) dense, by rows
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = TRANSFER( 'DENSE', H%type )
!       H%val( : )   the values of the components of H, stored row by row,
!                    with each the entries in each row in order of
!                    increasing column indicies.
!
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', H%type )
!       H%val( : )   the values of the diagonals of H, stored in order
!
!   v) scaled identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 15) = 'SCALED-IDENTITY'
!       H%val( 1 )  the value assigned to each diagonal of H
!
!   vi) identity
!
!       In this case, the following must be set:
!
!       H%type( 1 : 8 ) = 'IDENTITY'
!
!   vii) no Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 4 ) = 'ZERO' or 'NONE'
!
!   ix) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = 'LBFGS'
!
!       The Hessian in this case is available via the component H_lm below
!
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above,
!    except for scheme (ix), for which a permutation will be set within H_lm.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %H_lm is a structure of type LMS_data_type, whose components hold the
!     L-BFGS Hessian. Access to this structure is via the module GALAHAD_LMS,
!     and this component needs only be set if %H%type( 1 : 5 ) = 'LBFGS.'
!
!   %WEIGHT is a REAL array, which need only be set if %Hessian_kind is larger
!    than 1. If this is so, it must be of length at least %n, and contain the
!    weights W for the objective function.
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is not 1 or 2.
!    If this is so, it must be of length at least %n, and contain the
!    weights X^0 for the objective function.
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
!  data is a structure of type CCQP_data_type which holds private internal data
!
!  control is a structure of type CCQP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to CCQP_initialize. See the preamble
!   for details
!
!  inform is a structure of type CCQP_inform_type that provides
!    information on exit from CCQP_solve. The component status
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
!  On exit from CCQP_solve, other components of inform are given in the preamble
!
!  C_stat is an optional INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th constraint is likely not in the active set
!
!  B_stat is an optional  INTEGER array of length n, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   B_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( CCQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( CCQP_control_type ), INTENT( IN ) :: control
      TYPE ( CCQP_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: i, j, l, n_depen, nzc, nv, lbd, dual_starting_point
      REAL ( KIND = wp ) :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: time_analyse, time_factorize
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: clock_analyse, clock_factorize, cro_clock_matrix
      REAL ( KIND = wp ) :: av_bnd
!     REAL ( KIND = wp ) :: fixed_sum, xi
      LOGICAL :: printi, printa, remap_freed, reset_bnd, stat_required
      LOGICAL :: composite_g, diagonal_h, identity_h, scaled_identity_h
      LOGICAL :: separable_bqp, lbfgs
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( CQP_control_type ) :: CQP_control
      TYPE ( DQP_control_type ) :: DQP_control

!  functions

!$    INTEGER :: OMP_GET_MAX_THREADS

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' entering CCQP_solve ' )" ) prefix

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
      inform%factorization_status = 0
      inform%nfacts = - 1
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%obj = - one
      inform%non_negligible_pivot = zero
      inform%feasible = .FALSE.
!$    inform%threads = OMP_GET_MAX_THREADS( )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( B_stat )
      cro_clock_matrix = 0.0_wp

!  basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1
      printa = control%out > 0 .AND. control%print_level >= 101

!  ensure that input parameters are within allowed ranges

      IF ( prob%n < 1 .OR. prob%m < 0 .OR.                                     &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) prefix, inform%status
        GO TO 800
      END IF

!  assign CQP control parameters

      CQP_control = control%CQP_control

!  if required, write out problem

      IF ( prob%Hessian_kind < 0 ) THEN
        lbfgs = SMT_get( prob%H%type ) == 'LBFGS'
      ELSE
        lbfgs = .FALSE.
      END IF
      IF ( control%out > 0 .AND. control%print_level >= 20 )                   &
        CALL QPT_summarize_problem( control%out, prob )

!  check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

      reset_bnd = .FALSE.
      DO i = 1, prob%n
        IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
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
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
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

!  allocate integer workspace

      array_name = 'ccqp: data%B_stat'
      CALL SPACE_resize_array( prob%n, data%B_stat,                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'ccqp: data%C_stat'
      CALL SPACE_resize_array( prob%m, data%C_stat,                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

      IF ( prob%new_problem_structure ) THEN

!  store the problem dimensions

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%a_ne = prob%A%ne
        END SELECT

        IF ( prob%Hessian_kind < 0 ) THEN
          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'NONE', 'ZERO', 'IDENTITY' )
            data%h_ne = 0
          CASE ( 'SCALED_IDENTITY' )
            data%h_ne = 1
          CASE ( 'DIAGONAL' )
            data%h_ne = prob%n
          CASE ( 'DENSE' )
            data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
          CASE ( 'COORDINATE' )
            data%h_ne = prob%H%ne
          CASE ( 'LBFGS' )
            data%h_ne = 0
          END SELECT
        END IF
      END IF

!  if the problem has no general constraints, check to see if it is separable

      IF ( data%a_ne <= 0 ) THEN
        separable_bqp = .TRUE.
        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'NONE', 'ZERO', 'IDENTITY', 'SCALED_IDENTITY', 'DIAGONAL' )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                separable_bqp = .FALSE. ; GO TO 10
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, prob%n
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                separable_bqp = .FALSE. ; GO TO 10
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
              separable_bqp = .FALSE. ; GO TO 10
            END IF
          END DO
        CASE ( 'LBFGS' )
          separable_bqp = .FALSE.
        END SELECT
 10     CONTINUE

!  the problem is a separable bound-constrained QP. Solve it explicitly

        IF ( separable_bqp ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( /, A, ' Solving separable bound-constrained QP -' )" ) prefix
          IF ( PRESENT( B_stat ) ) THEN
            CALL QPD_solve_separable_BQP( prob, control%infinity,              &
                                          control%CQP_control%obj_unbounded,   &
                                          inform%obj,                          &
                                          inform%feasible, inform%status,      &
                                          B_stat = B_stat( : prob%n ) )
          ELSE
            CALL QPD_solve_separable_BQP( prob, control%infinity,              &
                                          control%CQP_control%obj_unbounded,   &
                                          inform%obj,                          &
                                          inform%feasible, inform%status )
          END IF
          IF ( printi ) THEN
            CALL CLOCK_time( clock_now )
            WRITE( control%out,                                                &
               "( A, ' On exit from QPD_solve_separable_BQP: status = ',       &
            &   I0, ', time = ', F0.2, /, A, ' objective value =', ES12.4 )",  &
              advance = 'no' ) prefix, inform%status, inform%time%clock_total  &
                + clock_now - clock_start, prefix, inform%obj
            IF ( PRESENT( B_stat ) ) THEN
              WRITE( control%out, "( ', active bounds: ', I0, ' from ', I0 )" )&
                COUNT( B_stat( : prob%n ) /= 0 ), prob%n
            ELSE
              WRITE( control%out, "( '' )" )
            END IF
          END IF
          inform%factorization_integer = 0 ; inform%factorization_real = 0

          IF ( printi ) then
            SELECT CASE( inform%status )
              CASE( GALAHAD_error_restrictions  ) ; WRITE( control%out,        &
                "( /, A, '  Warning - input paramters incorrect' )" ) prefix
              CASE( GALAHAD_error_primal_infeasible ) ; WRITE( control%out,    &
                "( /, A, '  Warning - the constraints appear to be',           &
               &   ' inconsistent' )" ) prefix
              CASE( GALAHAD_error_unbounded ) ; WRITE( control%out,            &
                "( /, A, '  Warning - problem appears to be unbounded from',   &
               & ' below' )") prefix
            END SELECT
          END IF
          IF ( inform%status /= GALAHAD_ok ) RETURN
          GO TO 800
        END IF
      END IF

!  perform the preprocessing

      IF ( prob%new_problem_structure ) THEN
        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before preprocessing: ', /,  A,   &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          GO TO 800
        END IF

!  record array lengths

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%a_ne = prob%A%ne
        END SELECT

        IF ( prob%Hessian_kind < 0 ) THEN
          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'NONE', 'ZERO', 'IDENTITY' )
            data%h_ne = 0
          CASE ( 'SCALED_IDENTITY' )
            data%h_ne = 1
          CASE ( 'DIAGONAL' )
            data%h_ne = prob%n
          CASE ( 'DENSE' )
            data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
          CASE ( 'COORDINATE' )
            data%h_ne = prob%H%ne
          CASE ( 'LBFGS' )
            data%h_ne = 0
          END SELECT
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "(  A, ' problem dimensions after preprocessing: ', /,  A,      &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform,                       &
                          prob, get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record

!  test for satisfactory termination

          IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 5 )              &
              WRITE( control%out, "( A, ' status ', I0, ' after QPP_apply')" ) &
               prefix, data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) prefix, inform%status
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            GO TO 800
          END IF
        END IF
        data%trans = data%trans + 1
      END IF

!  special case: no free variables

      IF ( prob%n == 0 ) THEN
        prob%Y( : prob%m ) = zero
        prob%Z( : prob%n ) = zero
        prob%C( : prob%m ) = zero
        CALL CCQP_AX( prob%m, prob%C( : prob%m ), prob%m,                      &
                      prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,                &
                      prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
        GO TO 700
      END IF

!  =================================================================
!  Check to see if the equality constraints are linearly independent
!  =================================================================

      time_analyse = inform%FDC_inform%time%analyse
      clock_analyse = inform%FDC_inform%time%clock_analyse
      time_factorize = inform%FDC_inform%time%factorize
      clock_factorize = inform%FDC_inform%time%clock_factorize

      IF ( prob%m > 0 .AND.                                                    &
           ( .NOT. data%tried_to_remove_deps .AND.                             &
              control%remove_dependencies ) ) THEN
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
           "( /, A, 1X, I0, ' equalit', A, ' from ', I0, ' constraint', A )" ) &
              prefix, data%dims%c_equality,                                    &
              TRIM( STRING_ies( data%dims%c_equality ) ),                      &
              prob%m, TRIM( STRING_pleural( prob%m ) )

!  set control parameters

        data%FDC_control = control%FDC_control
        data%FDC_control%max_infeas = control%CQP_control%stop_abs_p

!  find any dependent rows

        nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL FDC_find_dependent( prob%n, data%dims%c_equality,                 &
                                 prob%A%val( : nzc ),                          &
                                 prob%A%col( : nzc ),                          &
                                 prob%A%ptr( : data%dims%c_equality + 1 ),     &
                                 prob%C_l, n_depen, data%Index_C_freed,        &
                                 data%FDC_data, data%FDC_control,              &
                                 inform%FDC_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%find_dependent =                                           &
          inform%time%find_dependent + time_now - time_record
        inform%time%clock_find_dependent =                                     &
          inform%time%clock_find_dependent + clock_now - clock_record

!  record output parameters

        inform%status = inform%FDC_inform%status
        inform%non_negligible_pivot = inform%FDC_inform%non_negligible_pivot
        inform%alloc_status = inform%FDC_inform%alloc_status
        inform%factorization_status = inform%FDC_inform%factorization_status
        inform%factorization_integer = inform%FDC_inform%factorization_integer
        inform%factorization_real = inform%FDC_inform%factorization_real
        inform%bad_alloc = inform%FDC_inform%bad_alloc
        inform%nfacts = 1

        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now - time_start > control%cpu_time_limit ) .OR.           &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          GO TO 800
        END IF

        IF ( printi .AND. inform%non_negligible_pivot < thousand *             &
          control%FDC_control%SLS_control%absolute_pivot_tolerance )           &
            WRITE( control%out, "(                                             &
       &  /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /, A,                 &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, A, ' ***  perhaps increase',                                      &
       &     ' FDC_control%SLS_control%absolute_pivot_tolerance from',         &
       &    ES11.4,'  ***', /, A, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" )   &
           prefix, prefix, inform%non_negligible_pivot, prefix,                &
           control%FDC_control%SLS_control%absolute_pivot_tolerance, prefix

!  check for error exits

        IF ( inform%status /= 0 ) THEN

!  print details of the error exit

          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            IF ( inform%status /= GALAHAD_ok ) WRITE( control%error,           &
                 "( A, '    ** Error return ', I0, ' from ', A )" )            &
               prefix, inform%status, 'FDC_dependent'
          END IF
          GO TO 700
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, A, ' The following ', I0, ' constraint',    &
         &  A, ' appear', A, ' to be dependent', /, ( 4X, 8I8 ) )" )           &
              prefix, n_depen, TRIM(STRING_pleural( n_depen ) ),               &
              TRIM( STRING_verb_pleural( n_depen ) ), data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  special case: no free variables

        IF ( prob%n == 0 ) THEN
          prob%Y( : prob%m ) = zero
          prob%Z( : prob%n ) = zero
          prob%C( : prob%m ) = zero
          CALL CCQP_AX( prob%m, prob%C( : prob%m ), prob%m,                    &
                        prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,              &
                        prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
      END IF

      IF ( remap_freed ) THEN

!  some of the current constraints will be removed by freeing them

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, A, ' -> ', I0, ' constraint', A, ' ', A,   &
         & ' dependent and will be temporarily removed' )" ) prefix, n_depen,  &
           TRIM( STRING_pleural( n_depen ) ), TRIM( STRING_are( n_depen ) )

!  allocate arrays to indicate which constraints have been freed

          array_name = 'ccqp: data%C_freed'
          CALL SPACE_resize_array( n_depen, data%C_freed,                      &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          data%C_freed( i ) = prob%C_l( j )
          prob%C_l( j ) = - control%infinity
          prob%C_u( j ) = control%infinity
          prob%Y( j ) = zero
        END DO

        CALL QPP_initialize( data%QPP_map_freed, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

!  store the problem dimensions

        data%dims_save_freed = data%dims
        data%a_ne = prob%A%ne

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before removal of dependecies: ', &
              &   /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )          &
               prefix, prefix, prob%n, prob%m, data%a_ne

!  perform the preprocessing

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

        data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1
        data%dims%x_s = 1 ; data%dims%x_e = prob%n
        data%dims%c_s = data%dims%x_e + 1
        data%dims%c_e = data%dims%x_e + data%dims%nc
        data%dims%c_b = data%dims%c_e - prob%m
        data%dims%y_s = data%dims%c_e + 1
        data%dims%y_e = data%dims%c_e + prob%m
        data%dims%y_i = data%dims%c_s + prob%m
        data%dims%v_e = data%dims%y_e

!  test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 5 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) prefix, inform%status
          CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,            &
                              data%QPP_inform )
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          GO TO 800
        END IF

!  record revised array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions after removal of dependencies: ', &
             &    /, A, ' n = ', I8, ' m = ', I8, ' a_ne = ', I8 )" )          &
               prefix, prefix, prob%n, prob%m, data%a_ne
      END IF

!  compute the dimension of the KKT system

      data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1

!  arrays containing data relating to the composite vector ( x  c  y )
!  are partitioned as follows:

!   <---------- n --------->  <---- nc ------>  <-------- m --------->
!                             <-------- m --------->
!                        <-------- m --------->
!   -------------------------------------------------------------------
!   |                   |    |                 |    |                 |
!   |         x              |       c         |          y           |
!   |                   |    |                 |    |                 |
!   -------------------------------------------------------------------
!    ^                 ^    ^ ^               ^ ^    ^               ^
!    |                 |    | |               | |    |               |
!   x_s                |    |c_s              |y_s  y_i             y_e = v_e
!                      |    |                 |
!                     c_b  x_e               c_e

      data%dims%x_s = 1 ; data%dims%x_e = prob%n
      data%dims%c_s = data%dims%x_e + 1
      data%dims%c_e = data%dims%x_e + data%dims%nc
      data%dims%c_b = data%dims%c_e - prob%m
      data%dims%y_s = data%dims%c_e + 1
      data%dims%y_e = data%dims%c_e + prob%m
      data%dims%y_i = data%dims%c_s + prob%m
      data%dims%v_e = data%dims%y_e

!  ----------------
!  set up workspace
!  ----------------

      CALL CQP_workspace( prob%m, prob%n, data%dims, data%a_ne, data%h_ne,     &
                          prob%Hessian_kind, lbfgs, stat_required, data%order, &
                          data%GRAD_L, data%DIST_X_l, data%DIST_X_u, data%Z_l, &
                          data%Z_u, data%BARRIER_X, data%Y_l, data%DIST_C_l,   &
                          data%Y_u, data%DIST_C_u, data%C, data%BARRIER_C,     &
                          data%SCALE_C, data%RHS, data%OPT_alpha,              &
                          data%OPT_merit, data%BINOMIAL, data%CS_coef,         &
                          data%COEF, data%ROOTS, data%DX_zh, data%DY_zh,       &
                          data%DC_zh, data%DY_l_zh, data%DY_u_zh,              &
                          data%DZ_l_zh, data%DZ_u_zh, data%X_coef,             &
                          data%C_coef, data%Y_coef, data%Y_l_coef,             &
                          data%Y_u_coef, data%Z_l_coef, data%Z_u_coef,         &
                          data%H_s, data%A_s, data%Y_last, data%Z_last,        &
                          data%A_sbls, data%H_sbls,                            &
                          CQP_control, inform%CQP_inform )

!  ===========================
!  Solve the problem using CQP
!  ===========================

!  assign CQP control parameters

      CALL CPU_TIME( time_now ) ; time_record = time_now - time_start
      IF ( control%cpu_time_limit >= zero ) THEN
        IF ( CQP_control%cpu_time_limit < zero ) THEN
          CQP_control%cpu_time_limit = control%cpu_time_limit - time_record
        ELSE
          CQP_control%cpu_time_limit = MIN( CQP_control%cpu_time_limit,        &
                                         control%cpu_time_limit ) - time_record
        END IF
      END IF
      CALL CLOCK_time( clock_now ) ; clock_record = clock_now - clock_start
      IF ( control%clock_time_limit >= zero ) THEN
        IF ( CQP_control%clock_time_limit < zero ) THEN
          CQP_control%clock_time_limit = control%clock_time_limit - clock_record
        ELSE
          CQP_control%clock_time_limit = MIN( CQP_control%clock_time_limit,    &
                control%clock_time_limit ) - clock_record
        END IF
      END IF

!  solve the problem using an infeasible primal-dual interior-point method

      CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )

      IF ( prob%Hessian_kind == 0 ) THEN
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          CALL CQP_solve_main( data%dims, prob%n, prob%m,                      &
                               prob%A%val, prob%A%col, prob%A%ptr,             &
                               prob%C_l, prob%C_u, prob%X_l, prob%X_u,         &
                               prob%C, prob%X, prob%Y, prob%Z,                 &
                               data%GRAD_L, data%DIST_X_l, data%DIST_X_u,      &
                               data%Z_l, data%Z_u, data%BARRIER_X,             &
                               data%Y_l, data%DIST_C_l, data%Y_u,              &
                               data%DIST_C_u, data%C, data%BARRIER_C,          &
                               data%SCALE_C, data%RHS, prob%f,                 &
                               data%H_sbls, data%A_sbls, data%C_sbls,          &
                               data%order, data%X_coef, data%C_coef,           &
                               data%Y_coef, data%Y_l_coef, data%Y_u_coef,      &
                               data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,    &
                               data%CS_coef, data%COEF,                        &
                               data%ROOTS, data%ROOTS_data,                    &
                               data%DX_zh, data%DC_zh,                         &
                               data%DY_zh, data%DY_l_zh,                       &
                               data%DY_u_zh, data%DZ_l_zh,                     &
                               data%DZ_u_zh,                                   &
                               data%OPT_alpha, data%OPT_merit,                 &
                               data%SBLS_data, prefix,                         &
                               CQP_control, inform%CQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               C_last = data%A_s, X_last = data%H_s,           &
                               Y_last = data%Y_last, Z_last = data%Z_last,     &
                               C_stat = data%C_stat, B_Stat = data%B_Stat )
        ELSE
          CALL CQP_solve_main( data%dims, prob%n, prob%m,                      &
                               prob%A%val, prob%A%col, prob%A%ptr,             &
                               prob%C_l, prob%C_u, prob%X_l, prob%X_u,         &
                               prob%C, prob%X, prob%Y, prob%Z,                 &
                               data%GRAD_L, data%DIST_X_l, data%DIST_X_u,      &
                               data%Z_l, data%Z_u, data%BARRIER_X,             &
                               data%Y_l, data%DIST_C_l, data%Y_u,              &
                               data%DIST_C_u, data%C, data%BARRIER_C,          &
                               data%SCALE_C, data%RHS, prob%f,                 &
                               data%H_sbls, data%A_sbls, data%C_sbls,          &
                               data%order, data%X_coef, data%C_coef,           &
                               data%Y_coef, data%Y_l_coef, data%Y_u_coef,      &
                               data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,    &
                               data%CS_coef, data%COEF,                        &
                               data%ROOTS, data%ROOTS_data,                    &
                               data%DX_zh, data%DC_zh,                         &
                               data%DY_zh, data%DY_l_zh,                       &
                               data%DY_u_zh, data%DZ_l_zh,                     &
                               data%DZ_u_zh,                                   &
                               data%OPT_alpha, data%OPT_merit,                 &
                               data%SBLS_data, prefix,                         &
                               CQP_control, inform%CQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               G = prob%G,                                     &
                               C_last = data%A_s, X_last = data%H_s,           &
                               Y_last = data%Y_last, Z_last = data%Z_last,     &
                               C_stat = data%C_stat, B_Stat = data%B_Stat )
        END IF
      ELSE IF ( prob%Hessian_kind == 1 ) THEN
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          CALL CQP_solve_main( data%dims, prob%n, prob%m,                      &
                               prob%A%val, prob%A%col, prob%A%ptr,             &
                               prob%C_l, prob%C_u, prob%X_l, prob%X_u,         &
                               prob%C, prob%X, prob%Y, prob%Z,                 &
                               data%GRAD_L, data%DIST_X_l, data%DIST_X_u,      &
                               data%Z_l, data%Z_u, data%BARRIER_X,             &
                               data%Y_l, data%DIST_C_l, data%Y_u,              &
                               data%DIST_C_u, data%C, data%BARRIER_C,          &
                               data%SCALE_C, data%RHS, prob%f,                 &
                               data%H_sbls, data%A_sbls, data%C_sbls,          &
                               data%order, data%X_coef, data%C_coef,           &
                               data%Y_coef, data%Y_l_coef, data%Y_u_coef,      &
                               data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,    &
                               data%CS_coef, data%COEF,                        &
                               data%ROOTS, data%ROOTS_data,                    &
                               data%DX_zh, data%DC_zh,                         &
                               data%DY_zh, data%DY_l_zh,                       &
                               data%DY_u_zh, data%DZ_l_zh,                     &
                               data%DZ_u_zh,                                   &
                               data%OPT_alpha, data%OPT_merit,                 &
                               data%SBLS_data, prefix,                         &
                               CQP_control, inform%CQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               X0 = prob%X0,                                   &
                               C_last = data%A_s, X_last = data%H_s,           &
                               Y_last = data%Y_last, Z_last = data%Z_last,     &
                               C_stat = data%C_stat, B_Stat = data%B_Stat )
        ELSE
          CALL CQP_solve_main( data%dims, prob%n, prob%m,                      &
                               prob%A%val, prob%A%col, prob%A%ptr,             &
                               prob%C_l, prob%C_u, prob%X_l, prob%X_u,         &
                               prob%C, prob%X, prob%Y, prob%Z,                 &
                               data%GRAD_L, data%DIST_X_l, data%DIST_X_u,      &
                               data%Z_l, data%Z_u, data%BARRIER_X,             &
                               data%Y_l, data%DIST_C_l, data%Y_u,              &
                               data%DIST_C_u, data%C, data%BARRIER_C,          &
                               data%SCALE_C, data%RHS, prob%f,                 &
                               data%H_sbls, data%A_sbls, data%C_sbls,          &
                               data%order, data%X_coef, data%C_coef,           &
                               data%Y_coef, data%Y_l_coef, data%Y_u_coef,      &
                               data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,    &
                               data%CS_coef, data%COEF,                        &
                               data%ROOTS, data%ROOTS_data,                    &
                               data%DX_zh, data%DC_zh,                         &
                               data%DY_zh, data%DY_l_zh,                       &
                               data%DY_u_zh, data%DZ_l_zh,                     &
                               data%DZ_u_zh,                                   &
                               data%OPT_alpha, data%OPT_merit,                 &
                               data%SBLS_data, prefix,                         &
                               CQP_control, inform%CQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               X0 = prob%X0, G = prob%G,                       &
                               C_last = data%A_s, X_last = data%H_s,           &
                               Y_last = data%Y_last, Z_last = data%Z_last,     &
                               C_stat = data%C_stat, B_Stat = data%B_Stat )
        END IF
      ELSE IF ( prob%Hessian_kind == 2 ) THEN
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          CALL CQP_solve_main( data%dims, prob%n, prob%m,                      &
                               prob%A%val, prob%A%col, prob%A%ptr,             &
                               prob%C_l, prob%C_u, prob%X_l, prob%X_u,         &
                               prob%C, prob%X, prob%Y, prob%Z,                 &
                               data%GRAD_L, data%DIST_X_l, data%DIST_X_u,      &
                               data%Z_l, data%Z_u, data%BARRIER_X,             &
                               data%Y_l, data%DIST_C_l, data%Y_u,              &
                               data%DIST_C_u, data%C, data%BARRIER_C,          &
                               data%SCALE_C, data%RHS, prob%f,                 &
                               data%H_sbls, data%A_sbls, data%C_sbls,          &
                               data%order, data%X_coef, data%C_coef,           &
                               data%Y_coef, data%Y_l_coef, data%Y_u_coef,      &
                               data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,    &
                               data%CS_coef, data%COEF,                        &
                               data%ROOTS, data%ROOTS_data,                    &
                               data%DX_zh, data%DC_zh,                         &
                               data%DY_zh, data%DY_l_zh,                       &
                               data%DY_u_zh, data%DZ_l_zh,                     &
                               data%DZ_u_zh,                                   &
                               data%OPT_alpha, data%OPT_merit,                 &
                               data%SBLS_data, prefix,                         &
                               CQP_control, inform%CQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               WEIGHT = prob%WEIGHT, X0 = prob%X0,             &
                               C_last = data%A_s, X_last = data%H_s,           &
                               Y_last = data%Y_last, Z_last = data%Z_last,     &
                               C_stat = data%C_stat, B_Stat = data%B_Stat )
        ELSE
          CALL CQP_solve_main( data%dims, prob%n, prob%m,                      &
                               prob%A%val, prob%A%col, prob%A%ptr,             &
                               prob%C_l, prob%C_u, prob%X_l, prob%X_u,         &
                               prob%C, prob%X, prob%Y, prob%Z,                 &
                               data%GRAD_L, data%DIST_X_l, data%DIST_X_u,      &
                               data%Z_l, data%Z_u, data%BARRIER_X,             &
                               data%Y_l, data%DIST_C_l, data%Y_u,              &
                               data%DIST_C_u, data%C, data%BARRIER_C,          &
                               data%SCALE_C, data%RHS, prob%f,                 &
                               data%H_sbls, data%A_sbls, data%C_sbls,          &
                               data%order, data%X_coef, data%C_coef,           &
                               data%Y_coef, data%Y_l_coef, data%Y_u_coef,      &
                               data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,    &
                               data%CS_coef, data%COEF,                        &
                               data%ROOTS, data%ROOTS_data,                    &
                               data%DX_zh, data%DC_zh,                         &
                               data%DY_zh, data%DY_l_zh,                       &
                               data%DY_u_zh, data%DZ_l_zh,                     &
                               data%DZ_u_zh,                                   &
                               data%OPT_alpha, data%OPT_merit,                 &
                               data%SBLS_data, prefix,                         &
                               CQP_control, inform%CQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               WEIGHT = prob%WEIGHT, X0 = prob%X0,             &
                               G = prob%G,                                     &
                               C_last = data%A_s, X_last = data%H_s,           &
                               Y_last = data%Y_last, Z_last = data%Z_last,     &
                               C_stat = data%C_stat, B_Stat = data%B_Stat )
        END IF
      ELSE
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          IF ( lbfgs ) THEN
            CALL CQP_solve_main( data%dims, prob%n, prob%m,                    &
                                 prob%A%val, prob%A%col, prob%A%ptr,           &
                                 prob%C_l, prob%C_u, prob%X_l, prob%X_u,       &
                                 prob%C, prob%X, prob%Y, prob%Z,               &
                                 data%GRAD_L, data%DIST_X_l, data%DIST_X_u,    &
                                 data%Z_l, data%Z_u, data%BARRIER_X,           &
                                 data%Y_l, data%DIST_C_l, data%Y_u,            &
                                 data%DIST_C_u, data%C, data%BARRIER_C,        &
                                 data%SCALE_C, data%RHS, prob%f,               &
                                 data%H_sbls, data%A_sbls, data%C_sbls,        &
                                 data%order, data%X_coef, data%C_coef,         &
                                 data%Y_coef, data%Y_l_coef, data%Y_u_coef,    &
                                 data%Z_l_coef, data%Z_u_coef,                 &
                                 data%BINOMIAL, data%CS_coef, data%COEF,       &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 H_lm = prob%H_lm,                             &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last,                         &
                                 Z_last = data%Z_last,                         &
                                 C_stat = data%C_stat, B_Stat = data%B_Stat )
          ELSE
            CALL CQP_solve_main( data%dims, prob%n, prob%m,                    &
                                 prob%A%val, prob%A%col, prob%A%ptr,           &
                                 prob%C_l, prob%C_u, prob%X_l, prob%X_u,       &
                                 prob%C, prob%X, prob%Y, prob%Z,               &
                                 data%GRAD_L, data%DIST_X_l, data%DIST_X_u,    &
                                 data%Z_l, data%Z_u, data%BARRIER_X,           &
                                 data%Y_l, data%DIST_C_l, data%Y_u,            &
                                 data%DIST_C_u, data%C, data%BARRIER_C,        &
                                 data%SCALE_C, data%RHS, prob%f,               &
                                 data%H_sbls, data%A_sbls, data%C_sbls,        &
                                 data%order, data%X_coef, data%C_coef,         &
                                 data%Y_coef, data%Y_l_coef, data%Y_u_coef,    &
                                 data%Z_l_coef, data%Z_u_coef,                 &
                                 data%BINOMIAL, data%CS_coef, data%COEF,       &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 H_val = prob%H%val, H_col = prob%H%col,       &
                                 H_ptr = prob%H%ptr,                           &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last,                         &
                                 Z_last = data%Z_last,                         &
                                 C_stat = data%C_stat, B_Stat = data%B_Stat )
          END IF
        ELSE
          IF ( lbfgs ) THEN
            CALL CQP_solve_main( data%dims, prob%n, prob%m,                    &
                                 prob%A%val, prob%A%col, prob%A%ptr,           &
                                 prob%C_l, prob%C_u, prob%X_l, prob%X_u,       &
                                 prob%C, prob%X, prob%Y, prob%Z,               &
                                 data%GRAD_L, data%DIST_X_l, data%DIST_X_u,    &
                                 data%Z_l, data%Z_u, data%BARRIER_X,           &
                                 data%Y_l, data%DIST_C_l, data%Y_u,            &
                                 data%DIST_C_u, data%C, data%BARRIER_C,        &
                                 data%SCALE_C, data%RHS, prob%f,               &
                                 data%H_sbls, data%A_sbls, data%C_sbls,        &
                                 data%order, data%X_coef, data%C_coef,         &
                                 data%Y_coef, data%Y_l_coef, data%Y_u_coef,    &
                                 data%Z_l_coef, data%Z_u_coef,                 &
                                 data%BINOMIAL, data%CS_coef, data%COEF,       &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 H_lm = prob%H_lm, G = prob%G,                 &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last,                         &
                                 Z_last = data%Z_last,                         &
                                 C_stat = data%C_stat, B_Stat = data%B_Stat )
          ELSE
            CALL CQP_solve_main( data%dims, prob%n, prob%m,                    &
                                 prob%A%val, prob%A%col, prob%A%ptr,           &
                                 prob%C_l, prob%C_u, prob%X_l, prob%X_u,       &
                                 prob%C, prob%X, prob%Y, prob%Z,               &
                                 data%GRAD_L, data%DIST_X_l, data%DIST_X_u,    &
                                 data%Z_l, data%Z_u, data%BARRIER_X,           &
                                 data%Y_l, data%DIST_C_l, data%Y_u,            &
                                 data%DIST_C_u, data%C, data%BARRIER_C,        &
                                 data%SCALE_C, data%RHS, prob%f,               &
                                 data%H_sbls, data%A_sbls, data%C_sbls,        &
                                 data%order, data%X_coef, data%C_coef,         &
                                 data%Y_coef, data%Y_l_coef, data%Y_u_coef,    &
                                 data%Z_l_coef, data%Z_u_coef,                 &
                                 data%BINOMIAL, data%CS_coef, data%COEF,       &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 H_val = prob%H%val, H_col = prob%H%col,       &
                                 H_ptr = prob%H%ptr, G = prob%G,               &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last,                         &
                                 Z_last = data%Z_last,                         &
                                 C_stat = data%C_stat, B_Stat = data%B_Stat )
          END IF
        END IF
      END IF

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%CQP_inform%time%total = time_now - time_record
      inform%CQP_inform%time%clock_total = clock_now - clock_record

!  record output statistics

      inform%status = inform%CQP_inform%status
      inform%alloc_status = inform%CQP_inform%alloc_status
      inform%bad_alloc = inform%CQP_inform%bad_alloc
      inform%factorization_status = inform%CQP_inform%factorization_status
      inform%factorization_integer = inform%CQP_inform%factorization_integer
      inform%factorization_real = inform%CQP_inform%factorization_real
      inform%nfacts = inform%CQP_inform%nfacts
      inform%obj = inform%CQP_inform%obj
      inform%primal_infeasibility = inform%CQP_inform%primal_infeasibility
      inform%dual_infeasibility = inform%CQP_inform%dual_infeasibility
      inform%complementary_slackness = inform%CQP_inform%complementary_slackness
      inform%feasible = inform%CQP_inform%feasible
      inform%time%analyse = inform%CQP_inform%time%analyse +                   &
        inform%FDC_inform%time%analyse - time_analyse
      inform%time%factorize = inform%CQP_inform%time%factorize +               &
        inform%FDC_inform%time%factorize - time_factorize
      inform%time%solve = inform%CQP_inform%time%solve
      inform%time%clock_analyse = inform%CQP_inform%time%clock_analyse +       &
        inform%FDC_inform%time%clock_analyse - clock_analyse
      inform%time%clock_factorize = inform%CQP_inform%time%clock_factorize +   &
        inform%FDC_inform%time%clock_factorize - clock_factorize
      inform%time%clock_solve = inform%CQP_inform%time%clock_solve

      IF ( printi ) WRITE( control%out,                                        &
        "( /, A, ' on exit from CQP, status = ', I0 )" ) prefix, inform%status
      IF ( inform%status /= GALAHAD_ok ) GO TO 600

!  ==============================
!  crossover solution if required
!  ==============================

      IF ( control%crossover .AND. inform%status == GALAHAD_ok ) THEN
!write(6,*) ' n_active, m_active ',  &
!  COUNT( data%B_stat( : prob%n ) /= 0 ), COUNT( data%C_stat( : prob%m ) /= 0 )
         IF ( printa ) THEN
          WRITE( control%out, "( A, ' Before crossover:`' )" ) prefix
          WRITE( control%out, "( /, A, '      i       X_l             X   ',   &
         &   '          X_u            Z        st' )" ) prefix
          DO i = 1, prob%n
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i ), B_stat( i )
          END DO

          IF ( prob%m > 0 ) THEN
            WRITE( control%out, "( /, A, '      i       C_l             C ',   &
           &   '            C_u            Y        st' )" ) prefix
            DO i = 1, prob%m
              WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,        &
                prob%C_l( i ), prob%C( i ), prob%C_u( i ), prob%Y( i ),        &
                C_stat( i )
            END DO
          END IF
        END IF

        data%CRO_control = control%CRO_control
        data%CRO_control%feasibility_tolerance =                               &
          MAX( inform%primal_infeasibility, inform%dual_infeasibility,         &
               inform%complementary_slackness,                                 &
               control%CRO_control%feasibility_tolerance )
        IF ( data%CRO_control%feasibility_tolerance < infinity / two ) THEN
          data%CRO_control%feasibility_tolerance =                             &
            two * data%CRO_control%feasibility_tolerance
        ELSE
          data%CRO_control%feasibility_tolerance = infinity
        END IF
        time_analyse = inform%CRO_inform%time%analyse
        clock_analyse = inform%CRO_inform%time%clock_analyse
        time_factorize = inform%CRO_inform%time%factorize
        clock_factorize = inform%CRO_inform%time%clock_factorize
        IF ( lbfgs ) THEN
          CALL CRO_crossover( prob%n, prob%m, data%dims%c_equality,            &
                              prob%H_lm, prob%A%val,                           &
                              prob%A%col, prob%A%ptr, prob%G, prob%C_l,        &
                              prob%C_u, prob%X_l, prob%X_u, prob%C, prob%X,    &
                              prob%Y, prob%Z, data%C_stat, data%B_stat,        &
                              data%CRO_data, data%CRO_control,                 &
                              inform%CRO_inform )
        ELSE
          CALL CRO_crossover( prob%n, prob%m, data%dims%c_equality,            &
                              prob%H%val, prob%H%col, prob%H%ptr, prob%A%val,  &
                              prob%A%col, prob%A%ptr, prob%G, prob%C_l,        &
                              prob%C_u, prob%X_l, prob%X_u, prob%C, prob%X,    &
                              prob%Y, prob%Z, data%C_stat, data%B_stat,        &
                              data%CRO_data, data%CRO_control,                 &
                              inform%CRO_inform )
        END IF
        inform%time%analyse = inform%time%analyse +                            &
          inform%CRO_inform%time%analyse - time_analyse
        inform%time%clock_analyse = inform%time%clock_analyse +                &
          inform%CRO_inform%time%clock_analyse - clock_analyse
        inform%time%factorize = inform%time%factorize +                        &
          inform%CRO_inform%time%factorize - time_factorize
        inform%time%clock_factorize = inform%time%clock_factorize +            &
          inform%CRO_inform%time%clock_factorize - clock_factorize
        cro_clock_matrix =                                                     &
          inform%CRO_inform%time%clock_analyse - clock_analyse +               &
          inform%CRO_inform%time%clock_factorize - clock_factorize

        IF ( printa ) THEN
          WRITE( control%out, "( A, ' After crossover:' )" ) prefix
          WRITE( control%out, "( /, A, '      i       X_l             X   ',   &
         &   '          X_u            Z        st' )" ) prefix
          DO i = 1, prob%n
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i ),            &
            data%B_stat( i )
          END DO

          IF ( prob%m > 0 ) THEN
            WRITE( control%out, "( /, A, '      i       C_l             C ',   &
           &   '            C_u            Y        st' )" ) prefix
            DO i = 1, prob%m
              WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,        &
                prob%C_l( i ), prob%C( i ), prob%C_u( i ), prob%Y( i ),        &
                data%C_stat( i )
            END DO
          END IF
        END IF
      END IF
!write(6,*) ' n_active, m_active ',  &
!  COUNT( data%B_stat( : prob%n ) /= 0 ), COUNT( data%C_stat( : prob%m ) /= 0 )

!  =============================
!  Refine the solution using DQP
!  =============================

!  ----------------
!  set up workspace
!  ----------------

!  N.B. to save on storage, arrays used for CQP will be reused (and re-allocated
!  if necessary) for DQP. The exact mapping for data% is as follows:

!  CQP           DQP     CQP            DQP             CQP          DQP
!  ---           ---     ---            ---             ---          ---
!  GRAD_L    <-  G       DY_u_zh    <-  GY_u            SCALE_C  <-  VT
!  DIST_X_l  <-  ZC_l    DZ_l_zh    <-  GZ_l            DX_zh    <-  SOL
!  DIST_X_u  <-  ZC_u    DZ_u_zh    <-  GZ_u            DY_zh    <-  RES
!  DIST_C_l  <-  YC_l    C          <-  VECTOR          A_s      <-  GV
!  DIST_C_u  <-  YC_u    BARRIER_X  <-  BREAK_points    Y_last   <-  PV
!  DY_l_zh   <-  GY_l    BARRIER_C  <-  V0              Z_last   <-  DV

      IF ( prob%Hessian_kind >= 1 ) THEN
        composite_g = prob%target_kind /= 0
      ELSE
        composite_g = prob%gradient_kind == 0 .OR. prob%gradient_kind == 1
      END IF

      diagonal_h = .TRUE. ; identity_h = .TRUE.
      scaled_identity_h = SMT_get( prob%H%type ) == 'SCALED_IDENTITY'
      IF ( prob%Hessian_kind < 0 ) THEN
H_loop: DO i = 1, prob%n
          DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            j = prob%H%col( l )
            IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
              diagonal_h = .FALSE. ; EXIT H_loop
            END IF
            IF ( i == j .AND. prob%H%val( l ) /= one ) identity_h = .FALSE.
          END DO
        END DO H_loop
      END IF

      CALL DQP_workspace( prob%m, prob%n, data%dims, prob%A, prob%H,           &
                          composite_g, diagonal_h, identity_h,                 &
                          scaled_identity_h, nv, lbd,                          &
                          data%C_status, data%NZ_p, data%IUSED, data%INDEX_r,  &
                          data%INDEX_w, data%X_status, data%V_status,          &
                          data%X_status_old, data%C_status_old, data%C_active, &
                          data%X_active, data%CHANGES, data%ACTIVE_list,       &
                          data%ACTIVE_status, data%DX_zh, data%RHS,            &
                          data%DY_zh, data%H_s, data%Y_l, data%Y_u, data%Z_l,  &
                          data%Z_u, data%C, data%BARRIER_X, data%DIST_C_l,     &
                          data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,         &
                          data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,            &
                          data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,          &
                          data%A_s, data%GRAD_L, data%Y_last,                  &
                          data%HPV, data%Z_last,                               &
                          data%V_bnd, data%H_sbls, data%A_sbls, data%SCU_mat,  &
                          DQP_control, inform%DQP_inform )

!  assign DQP control parameters

      DQP_control = control%DQP_control
      CALL CPU_TIME( time_now ) ; time_record = time_now - time_start
      IF ( control%cpu_time_limit >= zero ) THEN
        IF ( DQP_control%cpu_time_limit < zero ) THEN
          DQP_control%cpu_time_limit = control%cpu_time_limit - time_record
        ELSE
          DQP_control%cpu_time_limit = MIN( DQP_control%cpu_time_limit,        &
                                         control%cpu_time_limit ) - time_record
        END IF
      END IF
      CALL CLOCK_time( clock_now ) ; clock_record = clock_now - clock_start
      IF ( control%clock_time_limit >= zero ) THEN
        IF ( DQP_control%clock_time_limit < zero ) THEN
          DQP_control%clock_time_limit = control%clock_time_limit - clock_record
        ELSE
          DQP_control%clock_time_limit = MIN( DQP_control%clock_time_limit,    &
                control%clock_time_limit ) - clock_record
        END IF
      END IF

!  ensure that the relative decrease in the criticality measures are
!  relative to the initial values from CQP not DQP

      DQP_control%stop_abs_p = MAX( DQP_control%stop_abs_p,                    &
        DQP_control%stop_rel_p * inform%CQP_inform%init_primal_infeasibility )
      DQP_control%stop_abs_d = MAX( DQP_control%stop_abs_d,                    &
        DQP_control%stop_rel_d * inform%CQP_inform%init_dual_infeasibility )
      DQP_control%stop_abs_c = MAX( DQP_control%stop_abs_c,                    &
        DQP_control%stop_rel_d * inform%CQP_inform%init_complementary_slackness)

      dual_starting_point = - 1

!  solve the problem using a dual projected-gradient method

      CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
      IF ( prob%Hessian_kind == 1 ) THEN
        IF ( prob%target_kind == 0 .OR. prob%target_kind == 1 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind, G = prob%G,   &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 X0 = prob%X0,                                 &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 X0 = prob%X0, G = prob%G,                     &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          END IF
        END IF
      ELSE IF ( prob%Hessian_kind == 2 ) THEN
        IF ( prob%target_kind == 0 .OR. prob%target_kind == 1 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT,                         &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT, G = prob%G,             &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT, X0 = prob%X0,           &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,        &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix,                       &
                                 DQP_control, inform%DQP_inform,               &
                                 prob%Hessian_kind, prob%gradient_kind,        &
                                 nv, lbd, data%m_ref, dual_starting_point,     &
                                 data%clock_total, data%cpu_total,             &
                                 data%SBLS_data, data%SLS_data,                &
                                 data%SCU_data, data%GLTR_data,                &
                                 data%SLS_control, data% SBLS_control,         &
                                 data%GLTR_control, data%C_status,             &
                                 data%NZ_p, data%IUSED, data%INDEX_r,          &
                                 data%INDEX_w, data%X_status, data%V_status,   &
                                 data%X_status_old, data%C_status_old,         &
                                 data%refactor, data%m_active, data%n_active,  &
                                 data%C_active, data%X_active, data%CHANGES,   &
                                 data%ACTIVE_list, data%ACTIVE_status,         &
                                 data%DX_zh, data%RHS, data%DY_zh, data%H_s,   &
                                 data%Y_l, data%Y_u, data%Z_l, data%Z_u,       &
                                 data%C, data%BARRIER_X, data%DIST_C_l,        &
                                 data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,  &
                                 data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,     &
                                 data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,   &
                                 data%A_s, data%GRAD_L,                        &
                                 data%Y_last, data%HPV, data%Z_last,           &
                                 data%V_bnd, data%H_sbls, data%A_sbls,         &
                                 data%C_sbls, data%SCU_mat,                    &
                                 target_kind = prob%target_kind,               &
                                 WEIGHT = prob%WEIGHT, X0 = prob%X0,           &
                                 G = prob%G,                                   &
                                 C_stat = data%C_stat, X_Stat = data%B_Stat )
          END IF
        END IF
      ELSE IF ( prob%Hessian_kind /= 0 ) THEN
        IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
          CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,          &
                               prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,     &
                               prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,     &
                               prob%C, prob%f, prefix,                         &
                               DQP_control, inform%DQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               nv, lbd, data%m_ref, dual_starting_point,       &
                               data%clock_total, data%cpu_total,               &
                               data%SBLS_data, data%SLS_data,                  &
                               data%SCU_data, data%GLTR_data,                  &
                               data%SLS_control, data% SBLS_control,           &
                               data%GLTR_control, data%C_status,               &
                               data%NZ_p, data%IUSED, data%INDEX_r,            &
                               data%INDEX_w, data%X_status, data%V_status,     &
                               data%X_status_old, data%C_status_old,           &
                               data%refactor, data%m_active, data%n_active,    &
                               data%C_active, data%X_active, data%CHANGES,     &
                               data%ACTIVE_list, data%ACTIVE_status,           &
                               data%DX_zh, data%RHS, data%DY_zh, data%H_s,     &
                               data%Y_l, data%Y_u, data%Z_l, data%Z_u,         &
                               data%C, data%BARRIER_X, data%DIST_C_l,          &
                               data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,    &
                               data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,       &
                               data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,     &
                               data%A_s, data%GRAD_L,                          &
                               data%Y_last, data%HPV, data%Z_last,             &
                               data%V_bnd, data%H_sbls, data%A_sbls,           &
                               data%C_sbls, data%SCU_mat,                      &
                               H_val = prob%H%val, H_col = prob%H%col,         &
                               H_ptr = prob%H%ptr,                             &
                               C_stat = data%C_stat, X_Stat = data%B_Stat )
        ELSE
          CALL DQP_solve_main( prob%n, prob%m, data%dims, prob%A%val,          &
                               prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,     &
                               prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,     &
                               prob%C, prob%f, prefix,                         &
                               DQP_control, inform%DQP_inform,                 &
                               prob%Hessian_kind, prob%gradient_kind,          &
                               nv, lbd, data%m_ref, dual_starting_point,       &
                               data%clock_total, data%cpu_total,               &
                               data%SBLS_data, data%SLS_data,                  &
                               data%SCU_data, data%GLTR_data,                  &
                               data%SLS_control, data% SBLS_control,           &
                               data%GLTR_control, data%C_status,               &
                               data%NZ_p, data%IUSED, data%INDEX_r,            &
                               data%INDEX_w, data%X_status, data%V_status,     &
                               data%X_status_old, data%C_status_old,           &
                               data%refactor, data%m_active, data%n_active,    &
                               data%C_active, data%X_active, data%CHANGES,     &
                               data%ACTIVE_list, data%ACTIVE_status,           &
                               data%DX_zh, data%RHS, data%DY_zh, data%H_s,     &
                               data%Y_l, data%Y_u, data%Z_l, data%Z_u,         &
                               data%C, data%BARRIER_X, data%DIST_C_l,          &
                               data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,    &
                               data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,       &
                               data%DZ_u_zh, data%BARRIER_C, data%SCALE_C,     &
                               data%A_s, data%GRAD_L,                          &
                               data%Y_last, data%HPV, data%Z_last,             &
                               data%V_bnd, data%H_sbls, data%A_sbls,           &
                               data%C_sbls, data%SCU_mat,                      &
                               H_val = prob%H%val, H_col = prob%H%col,         &
                               H_ptr = prob%H%ptr, G = prob%G,                 &
                               C_stat = data%C_stat, X_Stat = data%B_Stat )
        END IF
      END IF

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%DQP_inform%time%total = time_now - time_record
      inform%DQP_inform%time%clock_total = clock_now - clock_record

!  record output statistics

      inform%status = inform%DQP_inform%status
      inform%alloc_status = inform%DQP_inform%alloc_status
      inform%bad_alloc = inform%DQP_inform%bad_alloc
      inform%factorization_status = inform%DQP_inform%factorization_status
      inform%factorization_integer = MAX( inform%factorization_integer,        &
        inform%DQP_inform%factorization_integer )
      inform%factorization_real = MAX( inform%factorization_real,              &
        inform%DQP_inform%factorization_real )
      inform%nfacts = inform%nfacts + inform%DQP_inform%nfacts
      inform%obj = inform%DQP_inform%obj
      inform%primal_infeasibility = inform%DQP_inform%primal_infeasibility
      inform%dual_infeasibility = inform%DQP_inform%dual_infeasibility
      inform%complementary_slackness = inform%DQP_inform%complementary_slackness
      inform%feasible = inform%DQP_inform%feasible
      inform%time%analyse = inform%time%analyse                                &
        + inform%DQP_inform%time%analyse
      inform%time%factorize = inform%time%factorize                            &
        + inform%DQP_inform%time%factorize
      inform%time%solve = inform%time%solve                                    &
        + inform%DQP_inform%time%solve
      inform%time%clock_analyse = inform%time%clock_analyse                    &
        + inform%DQP_inform%time%clock_analyse
      inform%time%clock_factorize = inform%time%clock_factorize                &
        + inform%DQP_inform%time%clock_factorize
      inform%time%clock_solve = inform%time%clock_solve                        &
        + inform%DQP_inform%time%clock_solve
      IF ( printi ) WRITE( control%out,                                        &
        "( /, A, ' on exit from DQP, status = ', I0 )" ) prefix, inform%status

!  if some of the constraints were freed during the computation, refix them now

  600 CONTINUE
      IF ( remap_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        data%C_stat( prob%m + 1 : data%QPP_map_freed%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map_freed%m,                       &
                                   data%QPP_map_freed%c_map,                   &
                                   IX = data%C_stat( : data%QPP_map_freed%m ) )
        data%B_stat( prob%n + 1 : data%QPP_map_freed%n ) = - 1
        CALL SORT_inverse_permute( data%QPP_map_freed%n,                       &
                                   data%QPP_map_freed%x_map,                   &
                                   IX = data%B_stat( : data%QPP_map_freed%n ) )
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform, prob,           &
                          get_all = .TRUE.)
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed

!  fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO
      END IF
      data%tried_to_remove_deps = .FALSE.

!  retore the problem to its original form

  700 CONTINUE
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
!       data%IW( : prob%n + 1 ) = 0
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        data%C_stat( prob%m + 1 : data%QPP_map%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,         &
                                   IX = data%C_stat( : data%QPP_map%m ) )
        data%B_stat( prob%n + 1 : data%QPP_map%n ) = - 1
        CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,         &
                                   IX = data%B_stat( : data%QPP_map%n ) )

!  full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_all = .TRUE. )

!  restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  recover solution

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_x = .TRUE., get_y = .TRUE.,                    &
                            get_z = .TRUE., get_c = .TRUE. )
        END IF

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  if required, recond the variable and constraint status

      IF ( stat_required ) THEN
        B_stat( : prob%n ) = data%B_stat( : prob%n )
        C_stat( : prob%m ) = data%C_stat( : prob%m )
      END IF

!  compute total time

  800 CONTINUE
      CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start

      IF ( printi ) WRITE( control%out,                                        &
     "( /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=',                                            &
    &   /, A, 3X, ' =                          CCQP total time           ',    &
    &             '             =',                                            &
    &   /, A, 3X, ' =', 24X, 0P, F12.2, 29x, '='                               &
    &   /, A, 3X, ' =    preprocess    analyse    factorize     solve    ',    &
    &             ' crossover   =',                                            &
    &   /, A, 3X, ' =', 5F12.2, 5x, '=',                                       &
    &   /, A, 3X, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',    &
    &             '-=-=-=-=-=-=-=') ")                                         &
        prefix, prefix, prefix, inform%time%clock_total, prefix, prefix,       &
        inform%time%clock_preprocess, inform%time%clock_analyse,               &
        inform%time%clock_factorize, inform%time%clock_solve,                  &
        inform%CRO_inform%time%clock_total - cro_clock_matrix, prefix

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving CCQP_solve ' )" ) prefix
      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -CCQP_solve-', /,  A,                          &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving CCQP_solve ' )" ) prefix
      RETURN

!  non-executable statements

 2010 FORMAT( ' ', /, A, '    ** Error return ', I0, ' from CCQP ' )

!  End of CCQP_solve

      END SUBROUTINE CCQP_solve

!-*-*-*-*-*-*-   C C Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE CCQP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine CCQP_initialize
!   control see Subroutine CCQP_initialize
!   inform  see Subroutine CCQP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( CCQP_data_type ), INTENT( INOUT ) :: data
      TYPE ( CCQP_control_type ), INTENT( IN ) :: control
      TYPE ( CCQP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: scu_status
      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by FDC

      CALL FDC_terminate( data%FDC_data, data%FDC_control,                     &
                          inform%FDC_inform )
      IF ( inform%FDC_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%FDC_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated by CRO

      CALL CRO_terminate( data%CRO_data, control%CRO_control,                  &
                          inform%CRO_inform )
      IF ( inform%CRO_inform%status /= GALAHAD_ok )                            &
        inform%status = inform%CRO_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate FIT internal arrays

      CALL FIT_terminate( data%FIT_data, control%CQP_control%FIT_control,      &
                          inform%CQP_inform%FIT_inform )
      IF ( inform%CQP_inform%FIT_inform%status /= GALAHAD_ok )                 &
        inform%status = inform%CQP_inform%FIT_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate ROOTS internal arrays

      CALL ROOTS_terminate( data%ROOTS_data,                                   &
                            control%CQP_control%ROOTS_control,                 &
                            inform%CQP_inform%ROOTS_inform )
      IF ( inform%CQP_inform%ROOTS_inform%status /= GALAHAD_ok )               &
        inform%status = inform%CQP_inform%ROOTS_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data, control%DQP_control%SLS_control,      &
                          inform%DQP_inform%SLS_inform )
      inform%status = inform%DQP_inform%SLS_inform%status
      IF ( inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'DQP: data%SLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated within SBLS

      CALL SBLS_terminate( data%SBLS_data, control%DQP_control%SBLS_control,   &
                           inform%DQP_inform%SBLS_inform )
      inform%status = inform%DQP_inform%SBLS_inform%status
      IF ( inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%bad_alloc = 'DQP: data%SBLS'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate GLTR internal arrays

      CALL GLTR_terminate( data%GLTR_data, control%DQP_control%GLTR_control,   &
                           inform%DQP_inform%GLTR_inform )
      IF ( inform%DQP_inform%GLTR_inform%status /= GALAHAD_ok )                &
        inform%status = inform%DQP_inform%GLTR_inform%status
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate SCU internal arrays

      CALL SCU_terminate( data%SCU_data, scu_status,                           &
                          inform%DQP_inform%SCU_inform )
      IF ( scu_status /= GALAHAD_ok ) inform%status = GALAHAD_error_deallocate
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

!  Deallocate QPP internal arrays

      CALL QPP_terminate( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform)
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all arrays allocated for the preprocessing stage

      CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      IF ( data%QPP_inform%status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = 'ccqp: data%QPP'
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaing allocated arrays

      array_name = 'ccqp: data%INDEX_C_freed'
      CALL SPACE_dealloc_array( data%INDEX_C_freed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%GRAD_L'
      CALL SPACE_dealloc_array( data%GRAD_L,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DIST_X_l'
      CALL SPACE_dealloc_array( data%DIST_X_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DIST_X_u'
      CALL SPACE_dealloc_array( data%DIST_X_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Z_l'
      CALL SPACE_dealloc_array( data%Z_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Z_u'
      CALL SPACE_dealloc_array( data%Z_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%BARRIER_X'
      CALL SPACE_dealloc_array( data%BARRIER_X,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Y_l'
      CALL SPACE_dealloc_array( data%Y_l,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DY_l'
      CALL SPACE_dealloc_array( data%DY_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DIST_C_l'
      CALL SPACE_dealloc_array( data%DIST_C_l,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Y_u'
      CALL SPACE_dealloc_array( data%Y_u,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DY_u'
      CALL SPACE_dealloc_array( data%DY_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DIST_C_u'
      CALL SPACE_dealloc_array( data%DIST_C_u,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%C'
      CALL SPACE_dealloc_array( data%C,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%BARRIER_C'
      CALL SPACE_dealloc_array( data%BARRIER_C,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%SCALE_C'
      CALL SPACE_dealloc_array( data%SCALE_C,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%H_s'
      CALL SPACE_dealloc_array( data%H_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%A_s'
      CALL SPACE_dealloc_array( data%A_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%X_last'
      CALL SPACE_dealloc_array( data%X_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Y_last'
      CALL SPACE_dealloc_array( data%Y_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Z_last'
      CALL SPACE_dealloc_array( data%Z_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%OPT_alpha'
      CALL SPACE_dealloc_array( data%OPT_alpha,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%OPT_merit'
      CALL SPACE_dealloc_array( data%OPT_merit,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%X_coef'
      CALL SPACE_dealloc_array( data%X_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%C_coef'
      CALL SPACE_dealloc_array( data%C_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Y_coef'
      CALL SPACE_dealloc_array( data%Y_coef,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Y_l_coef'
      CALL SPACE_dealloc_array( data%Y_l_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Y_u_coef'
      CALL SPACE_dealloc_array( data%Y_u_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Z_l_coef'
      CALL SPACE_dealloc_array( data%Z_l_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%Z_u_coef'
      CALL SPACE_dealloc_array( data%Z_u_coef,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%BINOMIAL'
      CALL SPACE_dealloc_array( data%BINOMIAL,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DX_zh'
      CALL SPACE_dealloc_array( data%DX_zh,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DC_zh'
      CALL SPACE_dealloc_array( data%DC_zh,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DY_zh'
      CALL SPACE_dealloc_array( data%DY_zh,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DY_l_zh'
      CALL SPACE_dealloc_array( data%DY_l_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DY_u_zh'
      CALL SPACE_dealloc_array( data%DY_u_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DZ_l_zh'
      CALL SPACE_dealloc_array( data%DZ_l_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%DZ_u_zh'
      CALL SPACE_dealloc_array( data%DZ_u_zh,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%A_sbls%row'
      CALL SPACE_dealloc_array( data%A_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%A_sbls%col'
      CALL SPACE_dealloc_array( data%A_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%A_sbls%val'
      CALL SPACE_dealloc_array( data%A_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%H_sbls%ptr'
      CALL SPACE_dealloc_array( data%H_sbls%ptr,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%H_sbls%row'
      CALL SPACE_dealloc_array( data%H_sbls%row,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%H_sbls%col'
      CALL SPACE_dealloc_array( data%H_sbls%col,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%H_sbls%val'
      CALL SPACE_dealloc_array( data%H_sbls%val,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%X_status'
      CALL SPACE_dealloc_array( data%X_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%C_status'
      CALL SPACE_dealloc_array( data%C_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%V_status'
      CALL SPACE_dealloc_array( data%V_status,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%X_status_old'
      CALL SPACE_dealloc_array( data%X_status_old,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%C_status_old'
      CALL SPACE_dealloc_array( data%C_status_old,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%X_active'
      CALL SPACE_dealloc_array( data%X_active,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%C_active'
      CALL SPACE_dealloc_array( data%C_active,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%CHANGES'
      CALL SPACE_dealloc_array( data%CHANGES,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%ACTIVE_list'
      CALL SPACE_dealloc_array( data%ACTIVE_list,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%ACTIVE_status'
      CALL SPACE_dealloc_array( data%ACTIVE_status,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%NZ_p'
      CALL SPACE_dealloc_array( data%NZ_p,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%IUSED'
      CALL SPACE_dealloc_array( data%IUSED,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%INDEX_r'
      CALL SPACE_dealloc_array( data%INDEX_r,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%INDEX_w'
      CALL SPACE_dealloc_array( data%INDEX_w,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'ccqp: data%V_bnd'
      CALL SPACE_dealloc_array( data%V_bnd,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine CCQP_terminate

      END SUBROUTINE CCQP_terminate

!  End of module CCQP

   END MODULE GALAHAD_CCQP_double
