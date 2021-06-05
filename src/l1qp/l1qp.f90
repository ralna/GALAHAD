! THIS VERSION: GALAHAD 3.0 - 16/10/2017 AT 10:15 GMT.

!-*-*-*-*-*-*-*-*-  G A L A H A D _ L 1 Q P    M O D U L E  -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released in GALAHAD Version 3.0. June 28th 2017

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_L1QP_double

!     ----------------------------------------------------------
!     |                                                        |
!     | Minimize the l_1 penalty function                      |
!     |                                                        |
!     |    q(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1    |
!     |                                                        |
!     | or                                                     |
!     |                                                        |
!     |    s(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1    |
!     |                                                        |
!     | subject to the bound constraints x_l <= x <= x_u.      |
!     |                                                        |
!     | Here q(x) is the quadratic function                    |
!     |                                                        |
!     |         q(x) = 1/2 x^T H x + g^T x + f                 |
!     |                                                        |
!     | and s(x) is the linear/seprable function               |
!     |                                                        |
!     |     s(x) = 1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f   |
!     |                                                        |
!     ----------------------------------------------------------

!$    USE omp_lib
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_QPT_double
      USE GALAHAD_QPD_double, L1QP_data_type => QPD_data_type,                 &
                              L1QP_AX => QPD_AX, L1QP_HX => QPD_HX,            &
                              L1QP_abs_AX => QPD_abs_AX,                       &
                              L1QP_abs_HX => QPD_abs_HX
      USE GALAHAD_QPP_double
      USE GALAHAD_SORT_double, ONLY: SORT_inverse_permute
      USE GALAHAD_CRO_double
      USE GALAHAD_LPQP_double
      USE GALAHAD_CQP_double
      USE GALAHAD_DQP_double
      USE GALAHAD_DLP_double, ONLY: DLP_control_type, DLP_next_perturbation,   &
                                    DLP_read_specfile
      USE GALAHAD_RPD_double, ONLY: RPD_inform_type, RPD_write_qp_problem_data
      USE GALAHAD_ROOTS_double, ONLY: ROOTS_terminate
      USE GALAHAD_FIT_double, ONLY: FIT_terminate
      USE GALAHAD_GLTR_double, ONLY: GLTR_terminate
      USE GALAHAD_SBLS_double, ONLY: SBLS_terminate
      USE GALAHAD_SLS_double, ONLY: SLS_terminate
      USE GALAHAD_SCU_double, ONLY: SCU_terminate

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: L1QP_initialize, L1QP_read_specfile, L1QP_solve,               &
                L1QP_terminate, QPT_problem_type, SMT_type, SMT_put, SMT_get,  &
                L1QP_data_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: L1QP_control_type

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

!    specifies the unit number to write generated SIF file describing the
!     converted problem

        INTEGER :: converted_sif_file_device = 54

!    specifies the unit number to write generated QPLIB file describing the
!     converted problem

        INTEGER :: converted_qplib_file_device = 55

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   the required absolute and relative accuracies for the primal infeasibility

        REAL ( KIND = wp ) :: stop_abs_p = ten ** ( - 5 )
        REAL ( KIND = wp ) :: stop_rel_p = epsmch

!   the required absolute and relative accuracies for the dual infeasibility

        REAL ( KIND = wp ) :: stop_abs_d = ten ** ( - 5 )
        REAL ( KIND = wp ) :: stop_rel_d = epsmch

!   the required absolute and relative accuracies for the complementarity

        REAL ( KIND = wp ) :: stop_abs_c = ten ** ( - 5 )
        REAL ( KIND = wp ) :: stop_rel_c = epsmch

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!    the penalty weight, rho. The general constraints are not enforced
!     explicitly, but instead included in the objective as a penalty term
!     weighted by rho when rho > 0. If rho <= 0, the general constraints are
!     explicit (that is, there is no penalty term in the objective function)

        REAL ( KIND = wp ) :: rho = zero

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!  if %crossover is true, cross over the solution to one defined by
!   linearly-independent constraints if possible
!
        LOGICAL :: crossover = .TRUE.

!  if %refine is true, apply the dual projected-gradient method to refine
!   the solution
!
        LOGICAL :: refine = .FALSE.

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

!   if %generate_converted_sif_file is .true. if a SIF file describing the
!    converted problem is to be generated

        LOGICAL :: generate_converted_sif_file = .FALSE.

!   if %generate_converted_qplib_file is .true. if a QPLIB file describing the
!    converted problem is to be generated

        LOGICAL :: generate_converted_qplib_file = .FALSE.

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "L1QPPROB.SIF"  // REPEAT( ' ', 18 )

!  name of generated QPLIB file containing input problem

        CHARACTER ( LEN = 30 ) :: qplib_file_name =                            &
         "L1QPPROB.qplib"  // REPEAT( ' ', 16 )

!  name of generated SIF file containing converted problem

        CHARACTER ( LEN = 30 ) :: converted_sif_file_name =                    &
         "L1QPCPROB.SIF"  // REPEAT( ' ', 17 )

!  name of generated QPLIB file containing converted problem

        CHARACTER ( LEN = 30 ) :: converted_qplib_file_name =                  &
         "L1QPCPROB.qplib"  // REPEAT( ' ', 15 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for LPQP

        TYPE ( LPQP_control_type ) :: LPQP_control

!  control parameters for CQP

        TYPE ( CQP_control_type ) :: CQP_control

!  control parameters for CRO

        TYPE ( CRO_control_type ) :: CRO_control

!  control parameters for DLP

        TYPE ( DLP_control_type ) :: DLP_control

!  control parameters for DQP

        TYPE ( DQP_control_type ) :: DQP_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: L1QP_time_type

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

      TYPE, PUBLIC :: L1QP_inform_type

!  return status. See L1QP_solve for details

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

!  the number of threads used

        INTEGER :: threads = 1

!  the value of the objective function at the best estimate of the solution
!   determined by L1QP_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the value of the primal infeasibility

        REAL ( KIND = wp ) :: primal_infeasibility = HUGE( one )

!  the value of the dual infeasibility

        REAL ( KIND = wp ) :: dual_infeasibility = HUGE( one )

!  the value of the complementary slackness

        REAL ( KIND = wp ) :: complementary_slackness = HUGE( one )

!  is the returned "solution" feasible?

        LOGICAL :: feasible = .FALSE.

!  timings (see above)

        TYPE ( L1QP_time_type ) :: time

!  inform parameters for LPQP

        TYPE ( LPQP_inform_type ) :: LPQP_inform

!  inform parameters for CQP

        TYPE ( CQP_inform_type ) :: CQP_inform

!  inform parameters for DLP/DQP

        TYPE ( DQP_inform_type ) :: DQP_inform

!  inform parameters for CRO

        TYPE ( CRO_inform_type ) :: CRO_inform

!  inform parameters for RPD

        TYPE ( RPD_inform_type ) :: RPD_inform
      END TYPE

   CONTAINS

!-*-*-*-*-*-   L 1 Q P _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE L1QP_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for L1QP. This routine should be called before
!  L1QP_solve
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

      TYPE ( L1QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( L1QP_control_type ), INTENT( OUT ) :: control
      TYPE ( L1QP_inform_type ), INTENT( OUT ) :: inform

!  Set control parameters

      CALL LPQP_initialize( control%LPQP_control )
      control%LPQP_control%prefix = '" -LPQP:"                     '
      CALL CQP_initialize( data, control%CQP_control, inform%CQP_inform )
      control%CQP_control%prefix = '" - CQP:"                     '
      control%CQP_control%stop_abs_p = - one
      control%CQP_control%stop_rel_p = - one
      control%CQP_control%stop_abs_c = - one
      control%CQP_control%stop_rel_c = - one
      control%CQP_control%stop_abs_d = - one
      control%CQP_control%stop_rel_d = - one
!     CALL DLP_initialize( data, control%DLP_control, inform%DQP_inform )
      control%DLP_control%prefix = '" - DLP:"                     '
      control%DLP_control%stop_abs_p = - one
      control%DLP_control%stop_rel_p = - one
      control%DLP_control%stop_abs_c = - one
      control%DLP_control%stop_rel_c = - one
      control%DLP_control%stop_abs_d = - one
      control%DLP_control%stop_rel_d = - one
      CALL DQP_initialize( data, control%DQP_control, inform%DQP_inform )
      control%DQP_control%prefix = '" - DQP:"                     '
      control%DQP_control%stop_abs_p = - one
      control%DQP_control%stop_rel_p = - one
      control%DQP_control%stop_abs_c = - one
      control%DQP_control%stop_rel_c = - one
      control%DQP_control%stop_abs_d = - one
      control%DQP_control%stop_rel_d = - one
      CALL CRO_initialize( data%CRO_data, control%CRO_control,                 &
                           inform%CRO_inform )
      control%CRO_control%prefix = '" - CRO:"                     '

      inform%status = GALAHAD_ok

      RETURN

!  End of L1QP_initialize

      END SUBROUTINE L1QP_initialize

!-*-*-*-*-   L 1 Q P _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE L1QP_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by L1QP_initialize could (roughly)
!  have been set as:

! BEGIN QP SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  restore-problem-on-output                         2
!  sif-file-device                                   52
!  qplib-file-device                                 53
!  converted-sif-file-device                         54
!  converted-qplib-file-device                       55
!  infinity-value                                    1.0D+19
!  identical-bounds-tolerance                        1.0D-15
!  absolute-primal-accuracy                          1.0D-5
!  relative-primal-accuracy                          1.0D-15
!  absolute-dual-accuracy                            1.0D-5
!  relative-dual-accuracy                            1.0D-15
!  absolute-complementary-slackness-accuracy         1.0D-5
!  relative-complementary-slackness-accuracy         1.0D-15
!  penalty-weight                                    0.0D+0
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  cross-over-solution                               T
!  treat-zero-bounds-as-general                      F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  generate-sif-file                                 F
!  generate-qplib-file                               F
!  generate-converted-sif-file                       F
!  generate-converted-qplib-file                     F
!  sif-file-name                                     L1QPPROB.SIF
!  qplib-file-name                                   L1QPPROB.qplib
!  converted-sif-file-name                           L1QPCPROB.SIF
!  converted-qplib-file-name                         L1QPCPROB.qplib
!  output-line-prefix                                ""
! END QP SPECIFICATIONS (DEFAULT)

!  Dummy arguments

      TYPE ( L1QP_control_type ), INTENT( INOUT ) :: control
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
      INTEGER, PARAMETER :: converted_sif_file_device = qplib_file_device + 1
      INTEGER, PARAMETER :: converted_qplib_file_device                        &
                              = converted_sif_file_device + 1
      INTEGER, PARAMETER :: infinity = converted_qplib_file_device + 1
      INTEGER, PARAMETER :: stop_abs_p = infinity + 1
      INTEGER, PARAMETER :: stop_rel_p = stop_abs_p + 1
      INTEGER, PARAMETER :: stop_abs_d = stop_rel_p + 1
      INTEGER, PARAMETER :: stop_rel_d = stop_abs_d + 1
      INTEGER, PARAMETER :: stop_abs_c = stop_rel_d + 1
      INTEGER, PARAMETER :: stop_rel_c = stop_abs_c + 1
      INTEGER, PARAMETER :: rho = stop_rel_c + 1
      INTEGER, PARAMETER :: identical_bounds_tol = rho + 1
      INTEGER, PARAMETER :: cpu_time_limit = identical_bounds_tol + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: crossover = clock_time_limit + 1
      INTEGER, PARAMETER :: refine = crossover + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general = refine + 1
      INTEGER, PARAMETER :: space_critical = treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: generate_sif_file = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: generate_qplib_file = generate_sif_file + 1
      INTEGER, PARAMETER :: generate_converted_sif_file                        &
                              = generate_qplib_file + 1
      INTEGER, PARAMETER :: generate_converted_qplib_file                      &
                              = generate_converted_sif_file + 1
      INTEGER, PARAMETER :: sif_file_name = generate_converted_qplib_file + 1
      INTEGER, PARAMETER :: qplib_file_name = sif_file_name + 1
      INTEGER, PARAMETER :: converted_sif_file_name = qplib_file_name + 1
      INTEGER, PARAMETER :: converted_qplib_file_name                          &
                              = converted_sif_file_name + 1
      INTEGER, PARAMETER :: prefix = converted_qplib_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'L1QP'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( sif_file_device )%keyword = 'sif-file-device'
      spec( qplib_file_device )%keyword = 'qplib-file-device'
      spec( converted_sif_file_device )%keyword = 'converted-sif-file-device'
      spec( converted_qplib_file_device )%keyword                              &
        = 'converted-qplib-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( stop_abs_p )%keyword = 'absolute-primal-accuracy'
      spec( stop_rel_p )%keyword = 'relative-primal-accuracy'
      spec( stop_abs_d )%keyword = 'absolute-dual-accuracy'
      spec( stop_rel_d )%keyword = 'relative-dual-accuracy'
      spec( stop_abs_c )%keyword = 'absolute-complementary-slackness-accuracy'
      spec( stop_rel_c )%keyword = 'relative-complementary-slackness-accuracy'
      spec( rho )%keyword = 'penalty-weight'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( crossover )%keyword = 'cross-over-solution'
      spec( refine )%keyword = 'refine-solution'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( generate_sif_file )%keyword = 'generate-sif-file'
      spec( generate_qplib_file )%keyword = 'generate-qplib-file'
      spec( generate_converted_sif_file )%keyword =                            &
        'generate-converted-sif-file'
      spec( generate_converted_qplib_file )%keyword =                          &
        'generate-converted-qplib-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
      spec( qplib_file_name )%keyword = 'qplib-file-name'
      spec( converted_sif_file_name )%keyword = 'converted_sif-file-name'
      spec( converted_qplib_file_name )%keyword = 'converted_qplib-file-name'
      spec( prefix )%keyword = 'output-line-prefix'

!     IF ( PRESENT( alt_specname ) ) WRITE(6,*) ' liqp: ', alt_specname

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
     CALL SPECFILE_assign_value( spec( converted_sif_file_device ),            &
                                 control%converted_sif_file_device,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( converted_qplib_file_device ),          &
                                 control%converted_qplib_file_device,          &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( infinity ),                             &
                                 control%infinity,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_p ),                           &
                                 control%stop_abs_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_p ),                           &
                                 control%stop_rel_p,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_d ),                           &
                                 control%stop_abs_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_d ),                           &
                                 control%stop_rel_d,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_abs_c ),                           &
                                 control%stop_abs_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( stop_rel_c ),                           &
                                 control%stop_rel_c,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( rho ),                                  &
                                 control%rho,                                  &
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

     CALL SPECFILE_assign_value( spec( crossover ),                            &
                                 control%crossover,                            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( refine ),                               &
                                 control%refine,                               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),         &
                                 control%treat_zero_bounds_as_general,         &
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
     CALL SPECFILE_assign_value( spec( generate_converted_sif_file ),          &
                                 control%generate_converted_sif_file,          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_converted_qplib_file ),        &
                                 control%generate_converted_qplib_file,        &
                                 control%error )

!  Set character values

     CALL SPECFILE_assign_value( spec( sif_file_name ),                        &
                                 control%sif_file_name,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( qplib_file_name ),                      &
                                 control%qplib_file_name,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( converted_sif_file_name ),              &
                                 control%converted_sif_file_name,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( converted_qplib_file_name ),            &
                                 control%converted_qplib_file_name,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )

!  Make sure that inifinity is set consistently

      control%LPQP_control%infinity = control%infinity
      control%CQP_control%infinity = control%infinity
      control%DLP_control%infinity = control%infinity
      control%DQP_control%infinity = control%infinity

!  Read the specfiles for LPQP, CQP, DLP and DQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL LPQP_read_specfile( control%LPQP_control, device,                 &
                                alt_specname = TRIM( alt_specname ) // '-LPQP' )
        CALL CQP_read_specfile( control%CQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CQP' )
        CALL DLP_read_specfile( control%DLP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-DLP' )
        CALL DQP_read_specfile( control%DQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-DQP' )
      ELSE
        CALL LPQP_read_specfile( control%LPQP_control, device )
        CALL CQP_read_specfile( control%CQP_control, device )
        CALL DLP_read_specfile( control%DLP_control, device )
        CALL DQP_read_specfile( control%DQP_control, device )
      END IF

      RETURN

!  End of L1QP_read_specfile

      END SUBROUTINE L1QP_read_specfile

!-*-*-*-*-*-*-*-*-  L 1 Q P _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*

      SUBROUTINE L1QP_solve( prob, data, control, inform, C_stat, X_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Minimize the l_1 penalty function
!
!      q(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1
!
!  or
!
!      s(x) + rho || min( A x - c_l, c_u - A x, 0 )||_1
!
!  subject to the bound constraints x_l <= x <= x_u.
!
!  Here q(x) is the quadratic function
!
!           q(x) = 1/2 x^T H x + g^T x + f,
!
!  s(x) is the linear/seprable function
!
!       s(x) = 1/2 || W * ( x - x^0 ) ||_2^2 + g^T x + f
!
!  and x is a vector of n components ( x_1, .... , x_n ),
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite. The subroutine is particularly
!  appropriate when H and/or A are sparse.
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
!    to be solved since the last call to L1QP_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %Hessian_kind is an INTEGER variable which defines the type of objective
!    function to be used. Possible values are
!
!     0  all the weights will be zero, and the analytic centre of the
!        feasible region (if gradient_kind = 0) or the the minimizer of
!        the L1LP
!           g^T x + f + rho || min( A x - c_l, c_u - A x, 0 )||_1
!        (if gradient kind /= 0) will be found. %WEIGHT (see below)
!        need not be set
!
!     1  all the weights will be one. %WEIGHT (see below) need not be set
!
!     2  the weights will be those given by %WEIGHT (see below)
!
!    <0  the Hessian H will be used
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
!   viii) L-BFGS Hessian
!
!       In this case, the following must be set:
!
!       H%type( 1 : 5 ) = 'LBFGS'
!
!       The Hessian in this case is available via the component %H_lm below
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
!   %target_kind is an INTEGER variable which defines possible special
!     targets X0. Possible values are
!
!     0  X0 will be a vector of zeros.
!        %X0 (see below) need not be set
!
!     1  X0 will be a vector of ones.
!        %X0 (see below) need not be set
!
!     any other value - the values of X0 will be those given by %X0 (see below)
!
!   %X0 is a REAL array, which need only be set if %Hessian_kind is larger
!    that 0 and %target_kind /= 0,1. If this is so, it must be of length at
!    least %n, and contain the targets X^0 for the objective function.
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
!  data is a structure of type L1QP_data_type which holds private internal data
!
!  control is a structure of type L1QP_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to L1QP_initialize. See L1QP_initialize
!   for details
!
!  inform is a structure of type L1QP_inform_type that provides
!    information on exit from L1QP_solve. The component status
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
!   -18 Too many iterations have been performed.
!
!   -19 Too much elapsed CPU or system clock time has passed.
!
!  C_stat is an optional INTEGER array of length m, which if present will be
!   set on exit to indicate the likely ultimate status of the constraints.
!   Possible values are
!   C_stat( i ) < 0, the i-th constraint is likely in the active set,
!                    on its lower bound (-1 on bound, < -1 violated below bound)
!               > 0, the i-th constraint is likely in the active set
!                    on its upper bound (1 on bound, > 1 violated above bound)
!               = 0, the i-th constraint is likely not in the active set
!
!  X_stat is an optional INTEGER array of length n, which if present will be
!   set on exit to indicate the likely ultimate status of the simple bound
!   constraints. Possible values are
!   X_stat( i ) < 0, the i-th bound constraint is likely in the active set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is likely in the active set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is likely not in the active set
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      TYPE ( L1QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( L1QP_control_type ), INTENT( IN ) :: control
      TYPE ( L1QP_inform_type ), INTENT( OUT ) :: inform
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_stat

!  Local variables

      INTEGER :: i, j, l, nv, lbd
      INTEGER :: a_ne, h_ne, dual_starting_point
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: time_analyse, time_factorize, time_solve
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now, clock_solve
      REAL ( KIND = wp ) :: clock_analyse, clock_factorize, cro_clock_matrix
      REAL ( KIND = wp ) :: av_bnd
      REAL ( KIND = wp ) :: perturbation
!     REAL ( KIND = wp ) :: fixed_sum, xi
      REAL ( KIND = wp ) :: H_val( 1 )
      LOGICAL :: printi, printa, printe, reset_bnd, stat_required
      LOGICAL :: composite_g, diagonal_h, identity_h, scaled_identity_h
      LOGICAL :: separable_bqp, lbfgs, extrapolation_ok, initial
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
        WRITE( control%out, "( A, ' entering L1QP_solve ' )" ) prefix

!  see if the problem is an LP

      data%is_lp = .FALSE.
      IF ( prob%Hessian_kind < 0 ) THEN
        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'NONE', 'ZERO' )
          data%is_lp = .TRUE.
        CASE ( 'SPARSE_BY_ROWS' )
          IF ( prob%H%ptr( prob%n + 1 ) <= 1 ) data%is_lp = .TRUE.
        END SELECT
      ELSE IF ( prob%Hessian_kind == 0 ) THEN
        data%is_lp = .TRUE.
      END IF

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .NOT. data%is_lp )
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
      inform%obj = - one
!$    inform%threads = OMP_GET_MAX_THREADS( )
      stat_required = PRESENT( C_stat ) .AND. PRESENT( X_stat )
      cro_clock_matrix = 0.0_wp

!  basic single line of output per iteration

      printe = control%error > 0 .AND. control%print_level >= 1
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

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

!  store the problem dimensions

!write(6,*) '  Hessian_kind ',  prob%Hessian_kind
      IF ( prob%new_problem_structure ) THEN
        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          data%a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          data%a_ne = prob%A%ne
        END SELECT

        IF ( prob%Hessian_kind < 0 ) THEN
!write(6,*) ' new structure h_type ', SMT_get( prob%H%type )
          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'NONE', 'ZERO' )
            data%h_ne = 0
          CASE ( 'IDENTITY' )
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

      IF ( data%is_lp .AND. control%refine ) THEN
        prob%Hessian_kind = - 1
        CALL SMT_put( prob%H%type, 'SCALED_IDENTITY', i )
!write(6,*) ' refine lp structure h_type ', SMT_get( prob%H%type )
        array_name = 'l1qp: prob%H%val'
        CALL SPACE_resize_array( 1, prob%H%val,                                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900
        prob%H%val( 1 ) = control%DLP_control%initial_perturbation
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
          IF ( PRESENT( X_stat ) ) THEN
            CALL QPD_solve_separable_BQP( prob, control%infinity,              &
                                          control%CQP_control%obj_unbounded,   &
                                          inform%obj,                          &
                                          inform%feasible, inform%status,      &
                                          B_stat = X_stat( : prob%n ) )
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
            IF ( PRESENT( X_stat ) ) THEN
              WRITE( control%out, "( ', active bounds: ', I0, ' from ', I0 )" )&
                COUNT( X_stat( : prob%n ) /= 0 ), prob%n
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

!  convert the QP problem into an l_1 QP

      IF ( control%rho > zero ) THEN
        IF ( printi ) WRITE( control%out, "( /, A, ' l1_weight = ', ES9.2 )" ) &
           prefix, control%rho

        data%LPQP_control = control%LPQP_control
        data%LPQP_control%h_output_format = 'COORDINATE'
        IF ( data%is_lp ) data%LPQP_control%h_output_format = 'DIAGONAL'
        CALL LPQP_formulate( prob, control%rho, .TRUE.,                        &
                             data%LPQP_data, data%LPQP_control,                &
                             inform%LPQP_inform )

! -------------------------------------------------------------------
!  If desired, generate a SIF file for the converted problem

        IF ( control%generate_converted_sif_file ) THEN
          CALL QPD_SIF( prob, control%converted_sif_file_name,                 &
                        control%converted_sif_file_device,                     &
                        control%infinity, .NOT. data%is_lp .OR. control%refine )
        END IF

!  SIF file generated
! -------------------------------------------------------------------

! -------------------------------------------------------------------
!  If desired, generate a QPLIB file for the converted problem

        IF ( control%generate_converted_qplib_file ) THEN
          CALL RPD_write_qp_problem_data( prob,                                &
                     control%converted_qplib_file_name,                        &
                     control%converted_qplib_file_device, inform%rpd_inform )
        END IF

!  QPLIB file generated
! -------------------------------------------------------------------

        IF ( inform%LPQP_inform%status /= 0 ) THEN
          IF ( printe )                                                        &
            WRITE( control%error, "( ' On exit from LPQP_formulate',           &
           & ', status = ', I0 )" ) inform%LPQP_inform%status
          inform%status = GALAHAD_error_qp_solve ; GO TO 900
        END IF

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
!write(6,*) ' l1qp structure h_type ', SMT_get( prob%H%type ),  data%h_ne
        END IF
      END IF

!  perform the preprocessing by computing suitable maps

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
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
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
!write(6,*) ' h_ne = ',  data%h_ne
        IF ( printi ) WRITE( control%out,                                      &
               "(  A, ' problem dimensions after preprocessing: ', /,  A,      &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!       prob%new_problem_structure = .FALSE.
        data%trans = 1

!  preprocss the problem using the existing maps

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform,                       &
                          prob, get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + REAL( time_now - time_record, wp )
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
        CALL L1QP_AX( prob%m, prob%C( : prob%m ), prob%m,                      &
                      prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,                &
                      prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')

!  restore the original L1QP formulation from the preprocessed version

        data%trans = data%trans - 1
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

!  full restore

          IF ( control%restore_problem >= 2 ) THEN
            CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,             &
                              get_all = .TRUE. )

!  restore vectors and scalars

          ELSE IF ( control%restore_problem == 1 ) THEN
            CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,             &
                              get_f = .TRUE., get_g = .TRUE.,                  &
                              get_x = .TRUE., get_x_bounds = .TRUE.,           &
                              get_y = .TRUE., get_z = .TRUE.,                  &
                              get_c = .TRUE., get_c_bounds = .TRUE. )

!  recover solution

          ELSE
            CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,             &
                              get_x = .TRUE., get_y = .TRUE.,                  &
                              get_z = .TRUE., get_c = .TRUE. )
          END IF

          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + REAL( time_now - time_record, wp )
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record
          prob%new_problem_structure = data%new_problem_structure
          data%save_structure = .TRUE.
        END IF

!  assign the output status if required 

        IF ( stat_required ) THEN
          DO i = 1, prob%n
            IF ( prob%X( i ) < prob%X_l( i ) - control%stop_abs_p ) THEN
              X_stat( i ) = - 2
            ELSE IF ( prob%X( i ) > prob%X_u( i ) + control%stop_abs_p ) THEN
               X_stat( i ) = 2
            ELSE IF ( prob%X( i ) < prob%X_l( i ) + control%stop_abs_p ) THEN
               X_stat( i ) = - 1
            ELSE IF ( prob%X( i ) > prob%X_u( i ) - control%stop_abs_p ) THEN
               X_stat( i ) = 1
            ELSE
               X_stat( i ) = 0
            END IF
          END DO

          DO i = 1, prob%m
            IF ( prob%C( i ) < prob%C_l( i ) - control%stop_abs_p ) THEN
              C_stat( i ) = - 2
            ELSE IF ( prob%C( i ) > prob%C_u( i ) + control%stop_abs_p ) THEN
               C_stat( i ) = 2
            ELSE IF ( prob%C( i ) < prob%C_l( i ) + control%stop_abs_p ) THEN
               C_stat( i ) = - 1
            ELSE IF ( prob%C( i ) > prob%C_u( i ) - control%stop_abs_p ) THEN
               C_stat( i ) = 1
            ELSE
               C_stat( i ) = 0
            END IF
          END DO
        END IF
        GO TO 800
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

      IF ( data%is_lp ) data%h_ne = prob%n
      CALL CQP_workspace( prob%m, prob%n, data%dims, data%a_ne, data%h_ne,     &
!                         prob%Hessian_kind, lbfgs, stat_required, data%order, &
                          prob%Hessian_kind, lbfgs, .TRUE., data%order,        &
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

!  allocate integer workspace

      array_name = 'l1qp: data%X_stat'
      CALL SPACE_resize_array( data%QPP_map%n, data%X_stat,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'l1qp: data%C_stat'
      CALL SPACE_resize_array( data%QPP_map%m, data%C_stat,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

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
            control%cpu_time_limit ) - REAL( time_record, wp )
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

!  overwrite CQP stopping tolerances if required

      IF ( CQP_control%stop_abs_p < zero )                                     &
        CQP_control%stop_abs_p = control%stop_abs_p
      IF ( CQP_control%stop_rel_p < zero )                                     &
        CQP_control%stop_rel_p = control%stop_rel_p
      IF ( CQP_control%stop_abs_d < zero )                                     &
        CQP_control%stop_abs_d = control%stop_abs_d
      IF ( CQP_control%stop_rel_d < zero )                                     &
        CQP_control%stop_rel_d = control%stop_rel_d
      IF ( CQP_control%stop_abs_c < zero )                                     &
        CQP_control%stop_abs_c = control%stop_abs_c
      IF ( CQP_control%stop_rel_c < zero )                                     &
        CQP_control%stop_rel_c = control%stop_rel_c

      time_analyse = inform%CQP_inform%time%analyse
      clock_analyse = inform%CQP_inform%time%clock_analyse
      time_factorize = inform%CQP_inform%time%factorize
      clock_factorize = inform%CQP_inform%time%clock_factorize
      time_solve = inform%CQP_inform%time%solve
      clock_solve = inform%CQP_inform%time%clock_solve

!  solve the problem using an infeasible primal-dual interior-point method

      CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
!     IF ( prob%Hessian_kind == 0 ) THEN
      IF ( data%is_lp ) THEN
        IF ( control%refine ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
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
                                 data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,  &
                                 data%CS_coef, data%COEF,                      &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
!                                prob%Hessian_kind, prob%gradient_kind,        &
                                 - 1, prob%gradient_kind, H_val = prob%H%val,  &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last, Z_last = data%Z_last,   &
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
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
                                 data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,  &
                                 data%CS_coef, data%COEF,                      &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
!                                prob%Hessian_kind, prob%gradient_kind,        &
                                 - 1, prob%gradient_kind, H_val = prob%H%val,  &
                                 G = prob%G,                                   &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last, Z_last = data%Z_last,   &
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
          END IF
        ELSE
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
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
                                 data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,  &
                                 data%CS_coef, data%COEF,                      &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
!                                prob%Hessian_kind, prob%gradient_kind,        &
                                 0, prob%gradient_kind,                        &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last, Z_last = data%Z_last,   &
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
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
                                 data%Z_l_coef, data%Z_u_coef, data%BINOMIAL,  &
                                 data%CS_coef, data%COEF,                      &
                                 data%ROOTS, data%ROOTS_data,                  &
                                 data%DX_zh, data%DC_zh,                       &
                                 data%DY_zh, data%DY_l_zh,                     &
                                 data%DY_u_zh, data%DZ_l_zh,                   &
                                 data%DZ_u_zh,                                 &
                                 data%OPT_alpha, data%OPT_merit,               &
                                 data%SBLS_data, prefix,                       &
                                 CQP_control, inform%CQP_inform,               &
!                                prob%Hessian_kind, prob%gradient_kind,        &
                                 0, prob%gradient_kind,                        &
                                 G = prob%G,                                   &
                                 C_last = data%A_s, X_last = data%H_s,         &
                                 Y_last = data%Y_last, Z_last = data%Z_last,   &
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
          END IF
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
                               C_stat = data%C_stat, B_Stat = data%X_stat )
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
                               C_stat = data%C_stat, B_Stat = data%X_stat )
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
                               C_stat = data%C_stat, B_Stat = data%X_stat )
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
                               C_stat = data%C_stat, B_Stat = data%X_stat )
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
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
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
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
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
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
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
                                 C_stat = data%C_stat, B_Stat = data%X_stat )
          END IF
        END IF
      END IF

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%CQP_inform%time%total = REAL( time_now - time_record, wp )
      inform%CQP_inform%time%clock_total = clock_now - clock_record

!  record output statistics

      inform%status = inform%CQP_inform%status
      inform%alloc_status = inform%CQP_inform%alloc_status
      inform%bad_alloc = inform%CQP_inform%bad_alloc
      inform%factorization_status = inform%CQP_inform%factorization_status
      inform%factorization_integer = inform%CQP_inform%factorization_integer
      inform%factorization_real = inform%CQP_inform%factorization_real
      inform%obj = inform%CQP_inform%obj
      inform%primal_infeasibility = inform%CQP_inform%primal_infeasibility
      inform%dual_infeasibility = inform%CQP_inform%dual_infeasibility
      inform%complementary_slackness = inform%CQP_inform%complementary_slackness
      inform%time%analyse = inform%time%analyse +                              &
        inform%CQP_inform%time%analyse - time_analyse
      inform%time%factorize = inform%time%factorize +                          &
         inform%CQP_inform%time%factorize - time_factorize
      inform%time%solve = inform%time%solve +                                  &
         inform%CQP_inform%time%solve - time_solve
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%CQP_inform%time%clock_analyse - clock_analyse
      inform%time%clock_factorize = inform%time%clock_factorize +              &
         inform%CQP_inform%time%clock_factorize - clock_factorize
      inform%time%clock_solve = inform%time%clock_solve +                      &
         inform%CQP_inform%time%clock_solve - clock_solve

      IF ( printi ) WRITE( control%out,                                        &
        "( /, A, ' on exit from CQP, status = ', I0 )" ) prefix, inform%status

!  check if the solution has been found

      IF ( inform%status /= GALAHAD_ok ) THEN

!  if not, restore the original L1QP formulation from the preprocessed version

        data%trans = data%trans - 1
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          data%C_stat( prob%m + 1 : data%QPP_map%m ) = 0
          CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,       &
                                     IX = data%C_stat( : data%QPP_map%m ) )
          data%X_stat( prob%n + 1 : data%QPP_map%n ) = - 1
          CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,       &
                                     IX = data%X_stat( : data%QPP_map%n ) )
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + REAL( time_now - time_record, wp )
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record
!         prob%new_problem_structure = data%new_problem_structure
          data%save_structure = .TRUE.
        END IF

!  ... and restore the original problem and solution from the L1QP formulation

        IF ( control%rho > zero ) CALL LPQP_restore( prob, data%LPQP_data,     &
                                                     C_stat = data%C_stat )
        GO TO 600
      END IF

!  ==============================
!  crossover solution if required
!  ==============================

      IF ( control%crossover .AND. inform%status == GALAHAD_ok ) THEN
!write(6,*) ' n_active, m_active ',  &
!  COUNT( data%X_stat( : prob%n ) /= 0 ), COUNT( data%C_stat( : prob%m ) /= 0 )
         IF ( printa ) THEN
          WRITE( control%out, "( A, ' Before crossover:`' )" ) prefix
          WRITE( control%out, "( /, A, '      i       X_l             X   ',   &
         &   '          X_u            Z        st' )" ) prefix
          DO i = 1, prob%n
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i ),            &
            data%X_stat( i )
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
!data%CRO_control%print_level = 101
        IF ( printi ) WRITE( control%out,                                      &
          "( /, A, ' compute crossover solution' )" ) prefix
        time_analyse = inform%CRO_inform%time%analyse
        clock_analyse = inform%CRO_inform%time%clock_analyse
        time_factorize = inform%CRO_inform%time%factorize
        clock_factorize = inform%CRO_inform%time%clock_factorize
        IF ( lbfgs ) THEN
          CALL CRO_crossover( prob%n, prob%m, data%dims%c_equality,            &
                              prob%H_lm, prob%A%val,                           &
                              prob%A%col, prob%A%ptr, prob%G, prob%C_l,        &
                              prob%C_u, prob%X_l, prob%X_u, prob%C, prob%X,    &
                              prob%Y, prob%Z, data%C_stat, data%X_stat,        &
                              data%CRO_data, data%CRO_control,                 &
                              inform%CRO_inform )
        ELSE IF ( prob%Hessian_kind == 0 ) THEN
          CALL CRO_crossover( prob%n, prob%m, data%dims%c_equality,            &
                              prob%A%val, prob%A%col, prob%A%ptr, prob%G,      &
                              prob%C_l, prob%C_u, prob%X_l, prob%X_u, prob%C,  &
                              prob%X, prob%Y, prob%Z, data%C_stat,             &
                              data%X_stat, data%CRO_data, data%CRO_control,    &
                              inform%CRO_inform )
        ELSE
          CALL CRO_crossover( prob%n, prob%m, data%dims%c_equality,            &
                              prob%H%val, prob%H%col, prob%H%ptr, prob%A%val,  &
                              prob%A%col, prob%A%ptr, prob%G, prob%C_l,        &
                              prob%C_u, prob%X_l, prob%X_u, prob%C, prob%X,    &
                              prob%Y, prob%Z, data%C_stat, data%X_stat,        &
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
            data%X_stat( i )
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

!  restore the original L1QP formulation from the preprocessed version

      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
!       data%IW( : prob%n + 1 ) = 0
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        data%C_stat( prob%m + 1 : data%QPP_map%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,         &
                                   IX = data%C_stat( : data%QPP_map%m ) )
        data%X_stat( prob%n + 1 : data%QPP_map%n ) = - 1
        CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,         &
                                   IX = data%X_stat( : data%QPP_map%n ) )
        CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,                 &
                          get_all = .TRUE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
!       prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  restore the original problem and solution from the L1QP formulation

      IF ( control%rho > zero ) CALL LPQP_restore( prob, data%LPQP_data,       &
                                                   C_stat = data%C_stat )

!  ==============================================================
!  refine solution by applying a dual gradient projection method
!  ==============================================================

      IF ( .NOT. control%refine ) GO TO 600
      IF ( printi ) WRITE( control%out, "( /, A, ' refine solution' )" ) prefix

!  preprocess the problem data

      IF ( data%save_structure_dqp ) THEN
        data%new_problem_structure_dqp = prob%new_problem_structure
        data%save_structure_dqp = .FALSE.
      END IF

!  perform the preprocessing by computing suitable maps

      IF ( prob%new_problem_structure ) THEN

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          a_ne = prob%A%ne
        END SELECT

        IF ( prob%Hessian_kind < 0 ) THEN
          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'NONE', 'ZERO', 'IDENTITY' )
            h_ne = 0
          CASE ( 'SCALED_IDENTITY' )
            h_ne = 1
          CASE ( 'DIAGONAL' )
            h_ne = prob%n
          CASE ( 'DENSE' )
            h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            h_ne = prob%H%ptr( prob%n + 1 ) - 1
          CASE ( 'COORDINATE' )
            h_ne = prob%H%ne
          CASE ( 'LBFGS' )
            h_ne = 0
          END SELECT
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before preprocessing: ', /,  A,   &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, a_ne, h_ne

        CALL QPP_initialize( data%QPP_map_dqp, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          control%treat_zero_bounds_as_general

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map_dqp, data%QPP_control,                  &
                          data%QPP_inform, data%dims_dqp, prob,                &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
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
          CALL QPP_terminate( data%QPP_map_dqp, data%QPP_control,              &
                              data%QPP_inform )
          GO TO 800
        END IF

!  record array lengths

        SELECT CASE ( SMT_get( prob%A%type ) )
        CASE ( 'DENSE' )
          a_ne = prob%m * prob%n
        CASE ( 'SPARSE_BY_ROWS' )
          a_ne = prob%A%ptr( prob%m + 1 ) - 1
        CASE ( 'COORDINATE' )
          a_ne = prob%A%ne
        END SELECT

        IF ( prob%Hessian_kind < 0 ) THEN
          SELECT CASE ( SMT_get( prob%H%type ) )
          CASE ( 'NONE', 'ZERO', 'IDENTITY' )
            h_ne = 0
          CASE ( 'SCALED_IDENTITY' )
            h_ne = 1
          CASE ( 'DIAGONAL' )
            h_ne = prob%n
          CASE ( 'DENSE' )
            h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            h_ne = prob%H%ptr( prob%n + 1 ) - 1
          CASE ( 'COORDINATE' )
            h_ne = prob%H%ne
          CASE ( 'LBFGS' )
            h_ne = 0
          END SELECT
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "(  A, ' problem dimensions after preprocessing: ', /,  A,      &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, a_ne, h_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  preprocss the problem using the existing maps

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map_dqp, data%QPP_inform,                   &
                          prob, get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + REAL( time_now - time_record, wp )
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
            CALL QPP_terminate( data%QPP_map_dqp, data%QPP_control,            &
                                data%QPP_inform)
            GO TO 800
          END IF
        END IF
        data%trans = data%trans + 1
      END IF

!write(6,*) ' n_active, m_active ',  &
!  COUNT( data%X_stat( : prob%n ) /= 0 ), COUNT( data%C_stat( : prob%m ) /= 0 )

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

      IF ( data%is_lp ) THEN
        diagonal_h = .TRUE. ; identity_h = .FALSE. ; scaled_identity_h = .TRUE.
      ELSE
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
      END IF

      CALL DQP_workspace( prob%m, prob%n, data%dims_dqp, prob%A, prob%H,       &
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
                          data%A_s, data%GRAD_L, data%Y_last, data%HPV,        &
                          data%Z_last, data%V_bnd,                             &
                          data%H_sbls, data%A_sbls, data%SCU_mat,              &
                          DQP_control, inform%DQP_inform )

!  assign DQP control parameters

      IF ( data%is_lp ) THEN
        DQP_control = control%DLP_control
      ELSE
        DQP_control = control%DQP_control
      END IF
      CALL CPU_TIME( time_now ) ; time_record = time_now - time_start
      IF ( control%cpu_time_limit >= zero ) THEN
        IF ( DQP_control%cpu_time_limit < zero ) THEN
          DQP_control%cpu_time_limit =                                         &
            control%cpu_time_limit - REAL( time_record, wp )
        ELSE
          DQP_control%cpu_time_limit = MIN( DQP_control%cpu_time_limit,        &
            control%cpu_time_limit ) - REAL( time_record, wp )
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

!     DQP_control%stop_abs_p = MAX( DQP_control%stop_abs_p,                    &
!       DQP_control%stop_rel_p * inform%CQP_inform%init_primal_infeasibility )
!     DQP_control%stop_abs_d = MAX( DQP_control%stop_abs_d,                    &
!        DQP_control%stop_rel_d * inform%CQP_inform%init_dual_infeasibility )
!     DQP_control%stop_abs_c = MAX( DQP_control%stop_abs_c,                    &
!       DQP_control%stop_rel_d * inform%CQP_inform%init_complementary_slackness)
!     DQP_control%stop_abs_p = MAX( DQP_control%stop_abs_p,                    &
!       half * inform%CQP_inform%primal_infeasibility )
!     DQP_control%stop_abs_d = MAX( DQP_control%stop_abs_d,                    &
!       half * inform%CQP_inform%dual_infeasibility )
!     DQP_control%stop_abs_c = MAX( DQP_control%stop_abs_c,                    &
!        half * inform%CQP_inform%complementary_slackness )

!  overwrite DQP stopping tolerances if required

      IF ( DQP_control%stop_abs_p < zero )                                     &
        DQP_control%stop_abs_p = half * inform%CQP_inform%primal_infeasibility
      IF ( DQP_control%stop_rel_p < zero )                                     &
        DQP_control%stop_rel_p = control%stop_rel_p
      IF ( DQP_control%stop_abs_d < zero )                                     &
        DQP_control%stop_abs_d = half * inform%CQP_inform%dual_infeasibility
      IF ( DQP_control%stop_rel_d < zero )                                     &
        DQP_control%stop_rel_d = control%stop_rel_d
      IF ( DQP_control%stop_abs_c < zero )                                     &
        DQP_control%stop_abs_c = half *inform%CQP_inform%complementary_slackness
      IF ( DQP_control%stop_rel_c < zero )                                     &
        DQP_control%stop_rel_c = control%stop_rel_c
      dual_starting_point = - 1

!  solve the problem using a dual projected-gradient method

      CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
      time_analyse = inform%DQP_inform%time%analyse
      clock_analyse = inform%DQP_inform%time%clock_analyse
      time_factorize = inform%DQP_inform%time%factorize
      clock_factorize = inform%DQP_inform%time%clock_factorize
      time_solve = inform%DQP_inform%time%solve
      clock_solve = inform%DQP_inform%time%clock_solve

!  the problem is an L1LP. Solve as a perturbed QP

      IF ( data%is_lp ) THEN
        H_val( 1 ) = dqp_control%initial_perturbation
        initial = .TRUE.
        dqp_control%rho = control%rho
        dqp_control%factor_optimal_matrix = .TRUE.

!  loop over a sequence of decreasing perturbations until optimal

        DO
          IF ( printi ) WRITE( control%out, "( /,  A, 2X, 20( '-' ),           &
         &  ' quadratic perturbation = ', ES7.1, 1X, 20( '-' ) )" )            &
                prefix, H_val( 1 )
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,    &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, dqp_control,          &
                                 inform%DQP_inform, - 1, prob%gradient_kind,   &
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
                                 data%C_sbls, data%SCU_mat, H_val = H_val,     &
                                 C_stat = C_stat, X_stat = X_stat,             &
                                 initial = initial )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,    &
                                 prob%A%col, prob%A%ptr, prob%C_l, prob%C_u,   &
                                 prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z,   &
                                 prob%C, prob%f, prefix, dqp_control,          &
                                 inform%DQP_inform, - 1, prob%gradient_kind,   &
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
                                 H_val = H_val, G = prob%G,                    &
                                 C_stat = C_stat, X_stat = X_stat,             &
                                 initial = initial )
          END IF
!write(6,"( A, ( 5ES12.4 ) )" ) ' X ', prob%X
!write(6,"( A, ( 10I4 ) )" ) ' X_stat ', X_stat
!write(6,"( A, ( 10I4 ) )" ) ' C_stat ', C_stat
          IF ( inform%status /= GALAHAD_ok ) EXIT

!  reduce the perturbation, and prepare to warm start if the value is above
!  its lower limit

          perturbation = H_val( 1 )
          CALL DLP_next_perturbation( prob%n, prob%m, data%dims_dqp,           &
                                      prob%A%val,                              &
                                      prob%A%col, prob%A%ptr, prob%C_l,        &
                                      prob%C_u, prob%X_l, prob%X_u, prob%X,    &
                                      prob%Y, prob%Z, prob%C, prob%f, prefix,  &
                                      dqp_control, inform%DQP_inform,          &
                                      extrapolation_ok, perturbation,          &
                                      prob%gradient_kind, nv, data%m_ref,      &
                                      data%SBLS_data, data%SCU_data,           &
                                      data%GLTR_data, data% SBLS_control,      &
                                      data%GLTR_control, data%refactor,        &
                                      data%m_active, data%n_active,            &
                                      data%C_active, data%X_active,            &
                                      data%ACTIVE_status,  data%DX_zh,         &
                                      data%RHS, data%DY_zh, data%Y_l,          &
                                      data%Y_u, data%Z_l, data%Z_u, data%C,    &
                                      data%SCALE_C, data%A_s, data%Y_last,     &
                                      data%Z_last, data%A_sbls, data%C_sbls,   &
                                      data%SCU_mat, G = prob%G,                &
                                      C_stat = C_stat, X_stat = X_stat )

!  exit if the extrapolation leads to the optimal solution

          IF ( extrapolation_ok ) THEN
            IF ( printi ) WRITE( control%out,                                  &
                 "( /, A, ' Extrapolated solution is optimal' )" ) prefix
            EXIT

!  the extrapolation is infeasible, so reduce the perturbation so the new
!  perturbed problem lies on a different face of the dual fesaible set

          ELSE
            H_val( 1 ) = MIN( H_val( 1 ) * dqp_control%perturbation_reduction, &
                              0.99_wp * perturbation )
!           H_val( 1 ) = perturbation * dqp_control%perturbation_reduction

!  don't allow the perturbation to be too small

            IF ( H_val( 1 ) <= dqp_control%final_perturbation ) EXIT
            dual_starting_point = 0
            initial = .FALSE.
          END IF
        END DO

!  the problem is an L1QP

      ELSE
        dqp_control%rho = control%rho
        IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%target_kind == 0 .OR. prob%target_kind == 1 ) THEN
            IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            ELSE
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &

                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &

                                   target_kind = prob%target_kind, G = prob%G, &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            END IF
          ELSE
            IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   X0 = prob%X0,                               &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            ELSE
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   X0 = prob%X0, G = prob%G,                   &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            END IF
          END IF
        ELSE IF ( prob%Hessian_kind == 2 ) THEN
          IF ( prob%target_kind == 0 .OR. prob%target_kind == 1 ) THEN
            IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   WEIGHT = prob%WEIGHT,                       &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            ELSE
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   WEIGHT = prob%WEIGHT, G = prob%G,           &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            END IF
          ELSE
            IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   WEIGHT = prob%WEIGHT, X0 = prob%X0,         &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            ELSE
              CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,  &
                                   prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, &
                                   prob%X_l, prob%X_u, prob%X, prob%Y, prob%Z, &
                                   prob%C, prob%f, prefix,                     &
                                   DQP_control, inform%DQP_inform,             &
                                   prob%Hessian_kind, prob%gradient_kind,      &
                                   nv, lbd, data%m_ref, dual_starting_point,   &
                                   data%clock_total, data%cpu_total,           &
                                   data%SBLS_data, data%SLS_data,              &
                                   data%SCU_data, data%GLTR_data,              &
                                   data%SLS_control, data% SBLS_control,       &
                                   data%GLTR_control, data%C_status,           &
                                   data%NZ_p, data%IUSED, data%INDEX_r,        &
                                   data%INDEX_w, data%X_status, data%V_status, &
                                   data%X_status_old, data%C_status_old,       &
                                   data%refactor, data%m_active, data%n_active,&
                                   data%C_active, data%X_active, data%CHANGES, &
                                   data%ACTIVE_list, data%ACTIVE_status,       &
                                   data%DX_zh, data%RHS, data%DY_zh, data%H_s, &
                                   data%Y_l, data%Y_u, data%Z_l, data%Z_u,     &
                                   data%C, data%BARRIER_X, data%DIST_C_l,      &
                                   data%DIST_C_u, data%DIST_X_l, data%DIST_X_u,&
                                   data%DY_l_zh, data%DY_u_zh, data%DZ_l_zh,   &
                                   data%DZ_u_zh, data%BARRIER_C, data%SCALE_C, &
                                   data%A_s, data%GRAD_L,                      &
                                   data%Y_last, data%HPV, data%Z_last,         &
                                   data%V_bnd, data%H_sbls, data%A_sbls,       &
                                   data%C_sbls, data%SCU_mat,                  &
                                   target_kind = prob%target_kind,             &
                                   WEIGHT = prob%WEIGHT, X0 = prob%X0,         &
                                   G = prob%G,                                 &
                                   C_stat = data%C_stat, X_Stat = data%X_stat )
            END IF
          END IF
        ELSE IF ( prob%Hessian_kind /= 0 ) THEN
          IF ( prob%gradient_kind == 0 .OR. prob%gradient_kind == 1 ) THEN
            CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,    &
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
                                 H_val = prob%H%val, H_col = prob%H%col,       &
                                 H_ptr = prob%H%ptr,                           &
                                 C_stat = data%C_stat, X_Stat = data%X_stat )
          ELSE
            CALL DQP_solve_main( prob%n, prob%m, data%dims_dqp, prob%A%val,    &
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
                                 H_val = prob%H%val, H_col = prob%H%col,       &
                                 H_ptr = prob%H%ptr, G = prob%G,               &
                                 C_stat = data%C_stat, X_Stat = data%X_stat )
          END IF
        END IF
      END IF

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%DQP_inform%time%total = REAL( time_now - time_record, wp )
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
      inform%obj = inform%DQP_inform%obj
      inform%primal_infeasibility = inform%DQP_inform%primal_infeasibility
      inform%dual_infeasibility = inform%DQP_inform%dual_infeasibility
      inform%complementary_slackness = inform%DQP_inform%complementary_slackness
      inform%feasible = inform%DQP_inform%feasible
      inform%time%analyse = inform%time%analyse +                              &
        inform%DQP_inform%time%analyse - time_analyse
      inform%time%factorize = inform%time%factorize +                          &
         inform%DQP_inform%time%factorize - time_factorize
      inform%time%solve = inform%time%solve +                                  &
         inform%DQP_inform%time%solve - time_solve
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%DQP_inform%time%clock_analyse - clock_analyse
      inform%time%clock_factorize = inform%time%clock_factorize +              &
         inform%DQP_inform%time%clock_factorize - clock_factorize
      inform%time%clock_solve = inform%time%clock_solve +                      &
         inform%DQP_inform%time%clock_solve - clock_solve
      IF ( printi ) WRITE( control%out,                                        &
        "( /, A, ' on exit from DQP, status = ', I0 )" ) prefix, inform%status

      IF ( inform%status == GALAHAD_ok ) THEN
!write(6,*) ' n_active, m_active ',  &
!  COUNT( data%X_stat( : prob%n ) /= 0 ), COUNT( data%C_stat( : prob%m ) /= 0 )
         IF ( printa ) THEN
          WRITE( control%out, "( A, ' After DLP/DQP:`' )" ) prefix
          WRITE( control%out, "( /, A, '      i       X_l             X   ',   &
         &   '          X_u            Z        st' )" ) prefix
          DO i = 1, prob%n
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i ), X_stat( i )
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
      END IF

!  if some of the constraints were freed during the computation, refix them now

  600 CONTINUE

!  retore the problem to its original form

      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
!       data%IW( : prob%n + 1 ) = 0
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        data%C_stat( prob%m + 1 : data%QPP_map_dqp%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map_dqp%m, data%QPP_map_dqp%c_map, &
                                   IX = data%C_stat( : data%QPP_map_dqp%m ) )
        data%X_stat( prob%n + 1 : data%QPP_map_dqp%n ) = - 1
        CALL SORT_inverse_permute( data%QPP_map_dqp%n, data%QPP_map_dqp%x_map, &
                                   IX = data%X_stat( : data%QPP_map_dqp%n ) )

!  full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map_dqp, data%QPP_inform, prob,           &
                            get_all = .TRUE. )

!  restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map_dqp, data%QPP_inform, prob,           &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  recover solution

        ELSE
          CALL QPP_restore( data%QPP_map_dqp, data%QPP_inform, prob,           &
                            get_x = .TRUE., get_y = .TRUE.,                    &
                            get_z = .TRUE., get_c = .TRUE. )
        END IF

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess =                                               &
          inform%time%preprocess + REAL( time_now - time_record, wp )
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure_dqp
        data%save_structure = .TRUE.
      END IF

!  if required, recond the variable and constraint status

      IF ( stat_required ) THEN
        X_stat( : prob%n ) = data%X_stat( : prob%n )
        C_stat( : prob%m ) = data%C_stat( : prob%m )
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
    &   /, A, 3X, ' =                          L1QP total time           ',    &
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
        WRITE( control%out, "( A, ' leaving L1QP_solve ' )" ) prefix
      RETURN

!  allocation error

  900 CONTINUE
      inform%status = GALAHAD_error_allocate
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + REAL( time_now - time_start, wp )
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      IF ( printi ) WRITE( control%out,                                        &
        "( A, ' ** Message from -L1QP_solve-', /,  A,                          &
       &      ' Allocation error, for ', A, /, A, ' status = ', I0 ) " )       &
        prefix, prefix, inform%bad_alloc, inform%alloc_status

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' leaving L1QP_solve ' )" ) prefix
      RETURN

!  non-executable statements

 2010 FORMAT( ' ', /, A, '    ** Error return ', I0, ' from L1QP ' )

      RETURN

!  End of L1QP_solve

      END SUBROUTINE L1QP_solve

!-*-*-*-*-*-   L 1 Q P _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE L1QP_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!
!   data    see Subroutine QP_initialize
!   control see Subroutine QP_initialize
!   inform  see Subroutine QP_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( L1QP_data_type ), INTENT( INOUT ) :: data
      TYPE ( L1QP_control_type ), INTENT( IN ) :: control
      TYPE ( L1QP_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: scu_status
      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated by LPQP, CQP and DQP

      CALL LPQP_terminate( data%LPQP_data, control%LPQP_control,               &
                           inform%LPQP_inform )
      IF ( inform%LPQP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%LPQP_inform%alloc_status
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

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

      CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_dqp, data%QPP_control, data%QPP_inform )
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

!  End of subroutine L1QP_terminate

      END SUBROUTINE L1QP_terminate

!  End of module L1QP

   END MODULE GALAHAD_L1QP_double
