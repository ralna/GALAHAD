! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 14:30 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ Q P C   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released pre GALAHAD Version 1.0. October 17th 1997
!   update released with GALAHAD Version 2.0. August 11th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_QPC_double

!               ---------------------------------------------
!               |                                           |
!               | Solve the quadratic program               |
!               |                                           |
!               |    minimize     1/2 x(T) H x + g(T) x + f |
!               |    subject to     c_l <= A x <= c_u       |
!               |                   x_l <=  x  <= x_u       |
!               |                                           |
!               | using an interior-point trust-region      |
!               | approach to find an approximate solution, |
!               ! then crossing over to a working-set       |
!               | method to obtain an accurate solution     |
!               |                                           |
!               ---------------------------------------------

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_NORMS_double
      USE GALAHAD_SPACE_double
      USE GALAHAD_SORT_double
      USE GALAHAD_QPT_double
      USE GALAHAD_QPP_double
      USE GALAHAD_QPD_double, QPC_data_type => QPD_data_type,                  &
                              QPC_HX => QPD_HX, QPC_AX => QPD_AX
      USE GALAHAD_CQP_double
      USE GALAHAD_CRO_double
      USE GALAHAD_EQP_double
      USE GALAHAD_LSQP_double
      USE GALAHAD_QPA_double
      USE GALAHAD_QPB_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_FDC_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: QPC_initialize, QPC_read_specfile, QPC_solve, QPC_terminate,   &
                QPC_data_type, QPT_problem_type, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: max_real_store_ratio = 100
      INTEGER, PARAMETER :: max_integer_store_ratio = 100
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: point01 = 0.01_wp
      REAL ( KIND = wp ), PARAMETER :: point1 = 0.1_wp
      REAL ( KIND = wp ), PARAMETER :: point9 = 0.9_wp
      REAL ( KIND = wp ), PARAMETER :: point99 = 0.99_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: hundred = 100.0_wp
      REAL ( KIND = wp ), PARAMETER :: thousand = 1000.0_wp
      REAL ( KIND = wp ), PARAMETER :: tenm2 = ten ** ( - 2 )
      REAL ( KIND = wp ), PARAMETER :: tenm4 = ten ** ( - 4 )
      REAL ( KIND = wp ), PARAMETER :: k_diag = one
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: teneps = ten * epsmch
      REAL ( KIND = wp ), PARAMETER :: res_large = one
      REAL ( KIND = wp ), PARAMETER :: remote = ten ** 10
      REAL ( KIND = wp ), PARAMETER :: bar_min = zero
      REAL ( KIND = wp ), PARAMETER :: z_min = ten ** ( - 12 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPC_control_type

!   error and warning diagnostics occur on stream error

        INTEGER :: error = 6

!   general output occurs on stream out

        INTEGER :: out = 6

!   the level of output required is specified by print_level

        INTEGER :: print_level = 0

!   an initial guess as to the integer workspace required by SBLS     (OBSOLETE)

        INTEGER :: indmin = 1000

!   an initial guess as to the real workspace required by SBLS        (OBSOLETE)

        INTEGER :: valmin = 1000

!   indicate whether and how much of the input problem
!    should be restored on output. Possible values are

!      0 nothing restored
!      1 scalar and vector parameters
!      2 all parameters

        INTEGER :: restore_problem = 2

!    specifies the unit number to write generated SIF file describing the
!     current problem

        INTEGER :: sif_file_device = 54

!   any bound larger than infinity in modulus will be regarded as infinite

        REAL ( KIND = wp ) :: infinity = ten ** 19

!   any pair of constraint bounds (c_l,c_u) or (x_l,x_u) that are closer than
!    identical_bounds_tol will be reset to the average of their values

        REAL ( KIND = wp ) :: identical_bounds_tol = epsmch

!   the initial value of the penalty parameter used by QPA to penalize
!    general constraints. A non-positive value will be reset to 2 * infinity
!    norm of the Lagrange multipliers found by QPB or, if QPB has not been
!    used, 2 * m

        REAL ( KIND = wp ) :: rho_g = - one

!   the initial value of the penalty parameter used by QPA to penalize
!    simple bound constraints. A non-positive value will be reset to 2 *
!    infinity norm of the dual variables found by QPB or, if QPB has not been
!    used, 2 * n

        REAL ( KIND = wp ) :: rho_b = - one

!   the threshold pivot used by the matrix factorization when attempting to
!    detect linearly dependent constraints.
!    See the documentation for FDC for details                        (OBSOLETE)

        REAL ( KIND = wp ) :: pivot_tol_for_dependencies = half

!   any pivots smaller than zero_pivot in absolute value will be regarded to
!    be zero when attempting to detect linearly dependent constraints (OBSOLETE)

        REAL ( KIND = wp ) :: zero_pivot = epsmch

!   the maximum CPU time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: cpu_time_limit = - one

!   the maximum elapsed clock time allowed (-ve means infinite)

        REAL ( KIND = wp ) :: clock_time_limit = - one

!   the furthest variables/constraints are from one of their bounds to be
!    regarded as active

        REAL ( KIND = wp ) :: on_bound_tol = epsmch

!    is the problem convex?

        LOGICAL :: convex = .FALSE.

!    any problem bound with the value zero will be treated as if it were a
!     general value if true

        LOGICAL :: treat_zero_bounds_as_general = .FALSE.

!   if %array_syntax_worse_than_do_loop is true, f77-style do loops will be
!    used rather than f90-style array syntax for vector operations    (OBSOLETE)

        LOGICAL :: array_syntax_worse_than_do_loop = .FALSE.

!   if %space_critical true, every effort will be made to use as little
!     space as possible. This may result in longer computation time

        LOGICAL :: space_critical = .FALSE.

!   if %deallocate_error_fatal is true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue

        LOGICAL :: deallocate_error_fatal = .FALSE.

!   should the crossover phase to get an exact solution using the working set
!     solver QPA will be skipped?

        LOGICAL :: no_qpa = .FALSE.

!   should the interior-point phase using QPB will be skipped, and the
!    solution be found using the working set solver QPA?

        LOGICAL :: no_qpb = .FALSE.

!   does the user wishes to use the interior-point phase of the computation,
!    but only follow this with the working-set phase if the former is
!    unsuccessful? Otherwise both phases are to be used (subject to the
!    requests made in no_qpa and no_qpb).

        LOGICAL :: qpb_or_qpa = .FALSE.

!   if %generate_sif_file is .true. if a SIF file describing the current
!    problem is to be generated

        LOGICAL :: generate_sif_file = .FALSE.

!  name of generated SIF file containing input problem

        CHARACTER ( LEN = 30 ) :: sif_file_name =                              &
         "QPCPROB.SIF"  // REPEAT( ' ', 19 )

!  all output lines will be prefixed by %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in
!   quotes, e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for QPA

        TYPE ( QPA_control_type ) :: QPA_control

!  control parameters for QPB

        TYPE ( QPB_control_type ) :: QPB_control

!  control parameters for CQP

        TYPE ( CQP_control_type ) :: CQP_control

!  control parameters for CRO

        TYPE ( CRO_control_type ) :: CRO_control

!  control parameters for EQP

        TYPE ( EQP_control_type ) :: EQP_control

!  control parameters for FDC

        TYPE ( FDC_control_type ) :: FDC_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: QPC_time_type

!  the total CPU time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  the CPU time spent preprocessing the problem

        REAL ( KIND = wp ) :: preprocess = 0.0

!  the CPU time spent detecting linear dependencies

        REAL ( KIND = wp ) :: find_dependent = 0.0

!  the CPU time spent analysing the required matrices prior to factorization

        REAL ( KIND = wp ) :: analyse = 0.0

!  the CPU time spent factorizing the required matrices

        REAL ( KIND = wp ) :: factorize = 0.0

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

      TYPE, PUBLIC :: QPC_inform_type

!  return status. See QPB_solve for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the return status from the factorization

        INTEGER :: factorization_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_real = - 1

!  the total number of factorizations performed

        INTEGER :: nfacts = 0

!  the total number of factorizations which were modified to ensure that the
!   matrix was an appropriate preconditioner

        INTEGER :: nmods = 0

!  has the the crosover succeeded?

        LOGICAL :: p_found = .FALSE.

!  the value of the objective function at the best estimate of the solution
!   determined by QPB_solve

        REAL ( KIND = wp ) :: obj = HUGE( one )

!  the smallest pivot which was not judged to be zero when detecting linearly
!   dependent constraints

        REAL ( KIND = wp ) :: non_negligible_pivot = - one

!  timings (see above)

        TYPE ( QPC_time_type ) :: time

!  inform parameters for QPA

        TYPE ( QPA_inform_type ) :: QPA_inform

!  inform parameters for QPB

        TYPE ( QPB_inform_type ) :: QPB_inform

!  inform parameters for CQP

        TYPE ( CQP_inform_type ) :: CQP_inform

!  inform parameters for CRO

        TYPE ( CRO_inform_type ) :: CRO_inform

!  inform parameters for EQP

        TYPE ( EQP_inform_type ) :: EQP_inform

!  inform parameters for FDC

        TYPE ( FDC_inform_type ) :: FDC_inform
      END TYPE

!-------------------------------
!   I n t e r f a c e  B l o c k
!-------------------------------

!      INTERFACE TWO_NORM
!
!        FUNCTION SNRM2( n, X, incx )
!        REAL :: SNRM2
!        INTEGER, INTENT( IN ) :: n, incx
!        REAL, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!!       REAL, INTENT( IN ), DIMENSION( : ) :: X
!        END FUNCTION SNRM2
!
!        FUNCTION DNRM2( n, X, incx )
!        DOUBLE PRECISION :: DNRM2
!        INTEGER, INTENT( IN ) :: n, incx
!        DOUBLE PRECISION, INTENT( IN ), DIMENSION( incx * ( n - 1 ) + 1 ) :: X
!!       DOUBLE PRECISION, INTENT( IN ), DIMENSION( : ) :: X
!        END FUNCTION DNRM2
!
!      END INTERFACE

   CONTAINS

!-*-*-*-*-*-   Q P C _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE QPC_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for QPC. This routine should be called before
!  QPC_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. See preamble
!  inform   a structure containing output information. See preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( QPC_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPC_control_type ), INTENT( OUT ) :: control
      TYPE ( QPC_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  Set control parameters

      CALL CQP_initialize( data, control%CQP_control, inform%CQP_inform  )
      control%CQP_control%prefix = '" - CQP:"                     '
      CALL EQP_initialize( data, control%EQP_control, inform%EQP_inform  )
      control%EQP_control%prefix = '" - EQP:"                     '
      CALL QPA_initialize( data, control%QPA_control, inform%QPA_inform )
      control%QPA_control%prefix = '" - QPA:"                     '
      CALL QPB_initialize( data, control%QPB_control, inform%QPB_inform )
      control%QPB_control%prefix = '" - QPB:"                     '
      CALL CRO_initialize( data%CRO_data, control%CRO_control,                 &
                           inform%CRO_inform )
      control%CRO_control%prefix = '" - CRO:"                     '
      CALL FDC_initialize( data%FDC_data, control%FDC_control,                 &
                           inform%FDC_inform  )
      control%FDC_control%prefix = '" - FDC:"                     '

!  Real parameters

      control%zero_pivot = epsmch ** 0.75

!  initialise private data

      data%trans = 0 ; data%tried_to_remove_deps = .FALSE.
      data%save_structure = .TRUE.

      RETURN

!  End of QPC_initialize

      END SUBROUTINE QPC_initialize

!-*-*-*-*-   Q P C _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE QPC_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by QPC_initialize could (roughly)
!  have been set as:

! BEGIN QPC SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  initial-integer-workspace                         1000
!  initial-real-workspace                            1000
!  restore-problem-on-output                         2
!  sif-file-device                                   52
!  infinity-value                                    1.0D+19
!  identical-bounds-tolerance                        1.0D-15
!  initial-rho-g                                     -1.0
!  initial-rho-b                                     -1.0
!  pivot-tolerance-used-for-dependencies             0.5
!  zero-pivot-tolerance                              1.0D-12
!  on-bound-tolerance                                1.0D-15
!  maximum-cpu-time-limit                            -1.0
!  maximum-clock-time-limit                          -1.0
!  convex-problem                                    F
!  treat-zero-bounds-as-general                      F
!  array-syntax-worse-than-do-loop                   F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  no-qpa-phase                                      F
!  no-qpb-phase                                      F
!  qpb-or-qpa                                        F
!  generate-sif-file                                 F
!  sif-file-name                                     QPCPROB.SIF
!  output-line-prefix                                ""
! END QPC SPECIFICATIONS

!  Dummy arguments

      TYPE ( QPC_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: indmin = print_level + 1
      INTEGER, PARAMETER :: valmin = indmin + 1
      INTEGER, PARAMETER :: restore_problem = valmin + 1
      INTEGER, PARAMETER :: sif_file_device = restore_problem + 1
      INTEGER, PARAMETER :: infinity = sif_file_device + 1
      INTEGER, PARAMETER :: identical_bounds_tol = infinity + 1
      INTEGER, PARAMETER :: rho_g = identical_bounds_tol + 1
      INTEGER, PARAMETER :: rho_b = rho_g + 1
      INTEGER, PARAMETER :: pivot_tol_for_dependencies = rho_b + 1
      INTEGER, PARAMETER :: zero_pivot = pivot_tol_for_dependencies + 1
      INTEGER, PARAMETER :: on_bound_tol = zero_pivot + 1
      INTEGER, PARAMETER :: cpu_time_limit = on_bound_tol + 1
      INTEGER, PARAMETER :: clock_time_limit = cpu_time_limit + 1
      INTEGER, PARAMETER :: treat_zero_bounds_as_general = clock_time_limit + 1
      INTEGER, PARAMETER :: convex = treat_zero_bounds_as_general + 1
      INTEGER, PARAMETER :: array_syntax_worse_than_do_loop = convex + 1
      INTEGER, PARAMETER :: space_critical = array_syntax_worse_than_do_loop + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical + 1
      INTEGER, PARAMETER :: no_qpa = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: no_qpb = no_qpa + 1
      INTEGER, PARAMETER :: qpb_or_qpa = no_qpb + 1
      INTEGER, PARAMETER :: generate_sif_file = qpb_or_qpa + 1
      INTEGER, PARAMETER :: sif_file_name = generate_sif_file + 1
      INTEGER, PARAMETER :: prefix = sif_file_name + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 3 ), PARAMETER :: specname = 'QPC'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( indmin )%keyword = 'initial-integer-workspace'
      spec( valmin )%keyword = 'initial-real-workspace'
      spec( restore_problem )%keyword = 'restore-problem-on-output'
      spec( sif_file_device )%keyword = 'sif-file-device'

!  Real key-words

      spec( infinity )%keyword = 'infinity-value'
      spec( identical_bounds_tol )%keyword = 'identical-bounds-tolerance'
      spec( rho_g )%keyword = 'initial-rho-g'
      spec( rho_b )%keyword = 'initial-rho-b'
      spec( pivot_tol_for_dependencies )%keyword =                             &
        'pivot-tolerance-used-for-dependencies'
      spec( zero_pivot )%keyword = 'zero-pivot-tolerance'
      spec( on_bound_tol )%keyword = 'on-bound-tolerance'
      spec( cpu_time_limit )%keyword = 'maximum-cpu-time-limit'
      spec( clock_time_limit )%keyword = 'maximum-clock-time-limit'

!  Logical key-words

      spec( convex )%keyword = 'convex-problem'
      spec( treat_zero_bounds_as_general )%keyword =                           &
        'treat-zero-bounds-as-general'
      spec( array_syntax_worse_than_do_loop )%keyword =                        &
        'array-syntax-worse-than-do-loop'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'
      spec( no_qpa )%keyword = 'no-qpa-phase'
      spec( no_qpb )%keyword = 'no-qpb-phase'
      spec( qpb_or_qpa )%keyword = 'qpb-or-qpa'
      spec( generate_sif_file )%keyword = 'generate-sif-file'

!  Character key-words

      spec( sif_file_name )%keyword = 'sif-file-name'
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
      CALL SPECFILE_assign_value( spec( print_level ),                         &
                                  control%print_level,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( indmin ),                              &
                                  control%indmin,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( valmin ),                              &
                                  control%valmin,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( restore_problem ),                     &
                                  control%restore_problem,                     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( sif_file_device ),                     &
                                  control%sif_file_device,                     &
                                  control%error )
!  Set real value

      CALL SPECFILE_assign_value( spec( infinity ),                            &
                                  control%infinity,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( identical_bounds_tol ),                &
                                  control%identical_bounds_tol,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( rho_g ),                               &
                                  control%rho_g,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( rho_b ),                               &
                                  control%rho_b,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( pivot_tol_for_dependencies ),          &
                                  control%pivot_tol_for_dependencies,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( zero_pivot ),                          &
                                  control%zero_pivot,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( on_bound_tol ),                        &
                                   control%on_bound_tol,                       &
                                   control%error )
      CALL SPECFILE_assign_value( spec( cpu_time_limit ),                      &
                                  control%cpu_time_limit,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( clock_time_limit ),                    &
                                  control%clock_time_limit,                    &
                                  control%error )

!  Set logical values


      CALL SPECFILE_assign_value( spec( convex ),                              &
                                  control%convex,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( treat_zero_bounds_as_general ),        &
                                  control%treat_zero_bounds_as_general,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( array_syntax_worse_than_do_loop ),     &
                                  control%array_syntax_worse_than_do_loop,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( no_qpa ),                              &
                                  control%no_qpa,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( no_qpb ),                              &
                                  control%no_qpb,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( qpb_or_qpa ),                          &
                                  control%qpb_or_qpa,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( generate_sif_file ),                   &
                                  control%generate_sif_file,                   &
                                  control%error )

!  Set character value

      CALL SPECFILE_assign_value( spec( sif_file_name ),                       &
                                   control%sif_file_name,                      &
                                 control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Make sure that inifinity is set consistently

      control%QPA_control%infinity = control%infinity
      control%QPB_control%infinity = control%infinity

!  Read the specfiles for QPA, QPB and EQP

      IF ( PRESENT( alt_specname ) ) THEN
        CALL QPA_read_specfile( control%QPA_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPA' )
        CALL QPB_read_specfile( control%QPB_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-QPB' )
        CALL CQP_read_specfile( control%CQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CQP' )
        CALL EQP_read_specfile( control%EQP_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-EQP' )
        CALL CRO_read_specfile( control%CRO_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-CRO' )
      ELSE
        CALL QPA_read_specfile( control%QPA_control, device )
        CALL QPB_read_specfile( control%QPB_control, device )
        CALL CQP_read_specfile( control%CQP_control, device )
        CALL EQP_read_specfile( control%EQP_control, device )
        CALL CRO_read_specfile( control%CRO_control, device )
      END IF

      RETURN

      END SUBROUTINE QPC_read_specfile

!-*-*-*-*-*-*-*-*-   Q P C _ S O L V E  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE QPC_solve( prob, C_stat, B_stat, data, control, inform,      &
                            G_p, X_p, Y_p, Z_p )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Solve the quadratic program
!
!     minimize     q(x) = 1/2 x(T) H x + g(T) x + f
!
!     subject to    (c_l)_i <= (Ax)_i <= (c_u)_i , i = 1, .... , m,
!
!        and        (x_l)_i <=   x_i  <= (x_u)_i , i = 1, .... , n,
!
!  where x is a vector of n components ( x_1, .... , x_n ), const is a
!  constant, g is an n-vector, H is a symmetric matrix,
!  A is an m by n matrix, and any of the bounds (c_l)_i, (c_u)_i
!  (x_l)_i, (x_u)_i may be infinite, using a primal-dual method.
!  The subroutine is particularly appropriate when A and H are sparse
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
!    to be solved since the last call to QPC_initialize, and .FALSE. if
!    a previous call to a problem with the same "structure" (but different
!    numerical data) was made.
!
!   %n is an INTEGER variable, which must be set by the user to the
!    number of optimization parameters, n.  RESTRICTION: %n >= 1
!
!   %m is an INTEGER variable, which must be set by the user to the
!    number of general linear constraints, m. RESTRICTION: %m >= 0
!
!   %H is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of H. Four storage formats are permitted:
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
!    On exit, the components will most likely have been reordered.
!    The output  matrix will be stored by rows, according to scheme (ii) above.
!    However, if scheme (i) is used for input, the output H%row will contain
!    the row numbers corresponding to the values in H%val, and thus in this
!    case the output matrix will be available in both formats (i) and (ii).
!
!   %G is a REAL array of length %n, which must be set by
!    the user to the value of the gradient, g, of the linear term of the
!    quadratic objective function. The i-th component of G, i = 1, ....,
!    n, should contain the value of g_i.
!    On exit, G will most likely have been reordered.
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
!    to an estimate of the solution x. On successful exit, it will contain
!    the required solution.
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
!  C_stat is a INTEGER array of length m, which may be set by the user
!   on entry to QPC_solve to indicate which of the constraints are to
!   be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; C_stat
!   need not be set if control%QPA_control%cold_start is nonzero. On exit,
!   C_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   C_stat( i ) < 0, the i-th constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th constraint is not in the working set
!
!  B_stat is a INTEGER array of length n, which may be set by the user
!   on entry to QPC_solve to indicate which of the simple bound constraints
!   are to be included in the initial working set. If this facility is required,
!   the component control%cold_start must be set to 0 on entry; B_stat
!   need not be set if control%QPA_control%cold_start is nonzero. On exit,
!   B_stat will indicate which constraints are in the final working set.
!   Possible entry/exit values are
!   B_stat( i ) < 0, the i-th bound constraint is in the working set,
!                    on its lower bound,
!               > 0, the i-th bound constraint is in the working set
!                    on its upper bound, and
!               = 0, the i-th bound constraint is not in the working set
!
!  data is a structure of type QPC_data_type which holds private internal data
!
!  control is a structure of type QPC_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to QPC_initialize. See the preamble
!   for details
!
!  inform is a structure of type QPC_inform_type that provides
!    information on exit from QPC_solve. The component status
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
!        prob%H%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', 'DIAGONAL' }
!       has been violated.
!
!    -4 The bound constraints are inconsistent.
!
!    -5 The constraints appear to have no feasible point.
!
!    -7 The objective function appears to be unbounded from below on the
!       feasible set.
!
!    -9 The factorization failed; the return status from the factorization
!       package is given in the component factorization_status.
!
!    -13 The problem is so ill-conditoned that further progress is impossible.
!
!    -16 The step is too small to make further impact.
!
!    -17 Too many iterations have been performed. This may happen if
!       control%maxit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!    -19 Too much time has passed. This may happen if control%cpu_time_limit or
!       control%clock_time_limit is too small, but may also be symptomatic of
!       a badly scaled problem.
!
!    -23 an entry from the strict upper triangle of H has been input.
!
!  On exit from QPC_solve, other components of inform are fiven in the preamble
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      INTEGER, INTENT( INOUT ), DIMENSION( prob%m ) :: C_stat
      INTEGER, INTENT( INOUT ), DIMENSION( prob%n ) :: B_stat
      TYPE ( QPC_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPC_control_type ), INTENT( IN ) :: control
      TYPE ( QPC_inform_type ), INTENT( OUT ) :: inform

      REAL ( KIND = wp ), INTENT( IN ), OPTIONAL, DIMENSION( prob%n ) :: G_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: X_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: Z_p
      REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( prob%m ) :: Y_p

!  Local variables

      INTEGER :: i, j, l, tiny_x, tiny_c, n_sbls, nzc, n_appear_depen
      INTEGER :: ii, jj, ll, lbreak, k_n_max, lbd, m_link, n_depen, n_more_depen
      INTEGER :: hd_start, hd_end, hnd_start, hnd_end, type, n_pcg
      REAL :: time_start, time_record, time_now
      REAL ( KIND = wp ) :: clock_start, clock_record, clock_now
      REAL ( KIND = wp ) :: tol, f, q, av_bnd, a_x, a_norms, best_obj, viol8
      LOGICAL :: printi, printt, printm, printd, first_pass, center, reset_bnd
      LOGICAL :: remap_fixed, remap_freed, remap_more_freed, lsqp, cqp
      LOGICAL :: diagonal_qp, convex_diagonal_qp, gotsol
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( EQP_control_type ) :: EQP_control
      TYPE ( LSQP_control_type ) :: LSQP_control
      TYPE ( LSQP_inform_type ) :: LSQP_inform
      TYPE ( CQP_control_type ) :: CQP_control
      TYPE ( QPA_control_type ) :: QPA_control
      TYPE ( QPB_control_type ) :: QPB_control

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      CHARACTER ( LEN = LEN( TRIM( control%QPB_control%prefix ) ) - 2 ) ::     &
        qpb_prefix
      CHARACTER ( LEN = LEN( TRIM( control%QPA_control%prefix ) ) - 2 ) ::     &
        qpa_prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )
      IF ( LEN( TRIM( control%QPB_control%prefix ) ) > 2 )                     &
        qpb_prefix = control%QPB_control%prefix( 2 :                           &
                        LEN( TRIM( control%QPB_control%prefix ) ) - 1 )
      IF ( LEN( TRIM( control%QPA_control%prefix ) ) > 2 )                     &
        qpa_prefix = control%QPA_control%prefix( 2 :                           &
                        LEN( TRIM( control%QPA_control%prefix ) ) - 1 )

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( A, ' -- entering QPC_solve ' )" ) prefix

! -------------------------------------------------------------------
!  If desired, generate a SIF file for problem passed

      IF ( control%generate_sif_file ) THEN
        CALL QPD_SIF( prob, control%sif_file_name, control%sif_file_device,    &
                      control%infinity, .TRUE. )
      END IF

!  SIF file generated
! -------------------------------------------------------------------

!  Initialize time

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      inform%obj = infinity
      gotsol = .FALSE.

      inform%QPA_inform%status = - 1001
      inform%QPB_inform%status = - 1001
      inform%CQP_inform%status = - 1001
      inform%EQP_inform%status = - 1001

!  Basic single line of output per iteration

      printi = control%out > 0 .AND. control%print_level >= 1
      printt = control%out > 0 .AND. control%print_level >= 2
      printm = control%out > 0 .AND. control%print_level >= 3
      printd = control%out > 0 .AND. control%print_level >= 11

!  Make sure that QPA, QPB and LSQP control parameters follow those from QPC

      IF ( control%qpb_or_qpa .OR. .NOT. control%no_qpa ) THEN
        QPA_control = control%QPA_control
        QPA_control%restore_problem = control%restore_problem
        QPA_control%solve_qp = .TRUE.
        data%SLS_control = QPA_control%SLS_control

        IF ( control%cpu_time_limit >= zero ) THEN
          IF ( QPA_control%cpu_time_limit >= zero ) THEN
            QPA_control%cpu_time_limit = MIN( control%cpu_time_limit,          &
              QPA_control%cpu_time_limit )
          ELSE
            QPA_control%cpu_time_limit = control%cpu_time_limit
          END IF
        END IF
        IF ( control%clock_time_limit >= zero ) THEN
          IF ( QPA_control%clock_time_limit >= zero ) THEN
            QPA_control%clock_time_limit = MIN( control%clock_time_limit,      &
              QPA_control%clock_time_limit )
          ELSE
            QPA_control%clock_time_limit = control%clock_time_limit
          END IF
        END IF
      END IF

      IF ( control%qpb_or_qpa .OR. .NOT. control%no_qpb ) THEN
        QPB_control = control%QPB_control
        QPB_control%restore_problem = control%restore_problem
        QPB_control%LSQP_control%restore_problem = control%restore_problem

        IF ( control%cpu_time_limit >= zero ) THEN
          IF ( QPB_control%cpu_time_limit >= zero ) THEN
            QPB_control%cpu_time_limit = MIN( control%cpu_time_limit,          &
               QPB_control%cpu_time_limit )
          ELSE
            QPB_control%cpu_time_limit = control%cpu_time_limit
          END IF
          IF ( QPB_control%LSQP_control%cpu_time_limit >= zero ) THEN
            QPB_control%LSQP_control%cpu_time_limit =                          &
              MIN( control%cpu_time_limit,                                     &
                   QPB_control%LSQP_control%cpu_time_limit )
          ELSE
            QPB_control%LSQP_control%cpu_time_limit =                          &
              control%cpu_time_limit
          END IF
        END IF
        IF ( control%clock_time_limit >= zero ) THEN
          IF ( QPB_control%clock_time_limit >= zero ) THEN
            QPB_control%clock_time_limit = MIN( control%clock_time_limit,      &
               QPB_control%clock_time_limit )
          ELSE
            QPB_control%clock_time_limit = control%clock_time_limit
          END IF
          IF ( QPB_control%LSQP_control%clock_time_limit >= zero ) THEN
            QPB_control%LSQP_control%clock_time_limit =                        &
              MIN( control%clock_time_limit,                                   &
                   QPB_control%LSQP_control%clock_time_limit )
          ELSE
            QPB_control%LSQP_control%clock_time_limit =                        &
              control%clock_time_limit
          END IF
        END IF
      END IF
      EQP_control%max_infeasibility_absolute = QPB_control%stop_p

!  Ensure that input parameters are within allowed ranges

      IF ( prob%n <= 0 .OR. prob%m < 0 .OR.                                    &
           .NOT. QPT_keyword_H( prob%H%type ) .OR.                             &
           .NOT. QPT_keyword_A( prob%A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) inform%status
        GO TO 800
      END IF

!  If required, write out problem

      IF ( control%out > 0 .AND. control%print_level >= 20 ) THEN
        WRITE( control%out, "( ' n, m = ', I0, 1X, I0 )" ) prob%n, prob%m
        WRITE( control%out, "( ' f = ', ES12.4 )" ) prob%f
        WRITE( control%out, "( ' G = ', /, ( 5ES12.4 ) )" ) prob%G( : prob%n )
        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          WRITE( control%out, "( ' H (diagonal) = ', /, ( 5ES12.4 ) )" )       &
            prob%H%val( : prob%n )
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' H (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%H%val( : prob%n * ( prob%n + 1 ) / 2 )
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' H (row-wise) = ' )" )
          DO i = 1, prob%n
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%H%col( j ), prob%H%val( j ),                           &
                j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' H (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%H%row( i ), prob%H%col( i ), prob%H%val( i ), i = 1, prob%H%ne)
        END IF
        WRITE( control%out, "( ' X_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_l( : prob%n )
        WRITE( control%out, "( ' X_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%X_u( : prob%n )
        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          WRITE( control%out, "( ' A (dense) = ', /, ( 5ES12.4 ) )" )          &
            prob%A%val( : prob%n * prob%m )
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          WRITE( control%out, "( ' A (row-wise) = ' )" )
          DO i = 1, prob%m
            WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                   &
              ( i, prob%A%col( j ), prob%A%val( j ),                           &
                j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1 )
          END DO
        ELSE
          WRITE( control%out, "( ' A (co-ordinate) = ' )" )
          WRITE( control%out, "( ( 2( 2I8, ES12.4 ) ) )" )                     &
          ( prob%A%row( i ), prob%A%col( i ), prob%A%val( i ), i = 1, prob%A%ne)
        END IF
        WRITE( control%out, "( ' C_l = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_l( : prob%m )
        WRITE( control%out, "( ' C_u = ', /, ( 5ES12.4 ) )" )                  &
          prob%C_u( : prob%m )
      END IF

!  Check that problem bounds are consistent; reassign any pair of bounds
!  that are "essentially" the same

      reset_bnd = .FALSE.
      DO i = 1, prob%n
        IF ( prob%X_l( i ) - prob%X_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
          GO TO 800
        ELSE IF ( prob%X_u( i ) == prob%X_l( i ) ) THEN
        ELSE IF ( prob%X_u( i ) - prob%X_l( i )                                &
                  <= control%identical_bounds_tol ) THEN
          av_bnd = half * ( prob%X_l( i ) + prob%X_u( i ) )
          prob%X_l( i ) = av_bnd ; prob%X_u( i ) = av_bnd
          reset_bnd = .TRUE.
        END IF
      END DO
      IF ( reset_bnd .AND. printi ) WRITE( control%out,                        &
        "( ' ', /, '   **  Warning: one or more variable bounds reset ' )" )

      reset_bnd = .FALSE.
      DO i = 1, prob%m
        IF ( prob%C_l( i ) - prob%C_u( i ) > control%identical_bounds_tol ) THEN
          inform%status = GALAHAD_error_bad_bounds
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
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
        "( ' ', /, '   **  Warning: one or more constraint bounds reset ' )" )

!  Set Hessian and gradient types to generic - this may change in future

      prob%Hessian_kind = - 1 ; prob%gradient_kind = - 1

!  ===========================
!  Preprocess the problem data
!  ===========================

      IF ( data%save_structure ) THEN
        data%new_problem_structure = prob%new_problem_structure
        data%save_structure = .FALSE.
      END IF

      IF ( prob%new_problem_structure ) THEN
        CALL QPP_initialize( data%QPP_map, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                       &
          QPB_control%treat_zero_bounds_as_general

!  Store the problem dimensions

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          data%h_ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          data%h_ne = prob%H%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before preprocessing: ', /,  A,   &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Perform the preprocessing

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map, data%QPP_control,                      &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 1 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == GALAHAD_error_upper_entry )                    &
            WRITE( control%error, 2240 )
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          IF ( data%QPP_inform%status == GALAHAD_error_primal_infeasible )     &
            inform%status = GALAHAD_error_primal_infeasible
          GO TO 800
        END IF

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          data%h_ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          data%h_ne = prob%H%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
               "(  A, ' problem dimensions after preprocessing: ', /,  A,      &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

        prob%new_problem_structure = .FALSE.
        data%trans = 1

!  Recover the problem dimensions after preprocessing

      ELSE
        IF ( data%trans == 0 ) THEN
          CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
          CALL QPP_apply( data%QPP_map, data%QPP_inform, prob,                 &
                          get_all = .TRUE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
            inform%time%clock_preprocess + clock_now - clock_record

!  Test for satisfactory termination

          IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 1 )              &
              WRITE( control%out, "( ' status ', I0, ' after QPP_apply ')" )   &
               data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) inform%status
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            IF ( data%QPP_inform%status == GALAHAD_error_primal_infeasible)    &
              inform%status = GALAHAD_error_primal_infeasible
            GO TO 800
          END IF
        END IF
        data%trans = data%trans + 1

!  Record array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          data%h_ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          data%h_ne = prob%H%ne
        END IF
      END IF

!  Permute initial working sets if provided

      IF ( QPA_control%cold_start == 0 ) THEN
        CALL SORT_inplace_permute( data%QPP_map%m, data%QPP_map%c_map,         &
                                   IX = C_stat( : data%QPP_map%m ) )
        CALL SORT_inplace_permute( data%QPP_map%n, data%QPP_map%x_map,         &
                                   IX = B_stat( : data%QPP_map%n ) )
      END IF

      remap_freed = .FALSE. ; remap_more_freed = .FALSE. ; remap_fixed = .FALSE.

!  If all the variables have now been fixed, the solution has been found

      IF ( prob%n == 0 ) THEN

!  Check that the solution is feasible

        DO i = 1, prob%m
          IF ( prob%C_l( i ) > QPB_control%stop_p .OR.                         &
               prob%C_u( i ) < - QPB_control%stop_p ) THEN
            inform%status = GALAHAD_error_primal_infeasible
            GO TO 800
          END IF
        END DO
        prob%C( : prob%m ) = zero
        prob%Y( : prob%m ) = zero
        inform%obj = prob%f
        inform%status = 0
        GO TO 720
      END IF

      IF ( .NOT. control%qpb_or_qpa .AND. control%no_qpb ) GO TO 150

!  =================================================================
!  Check to see if the equality constraints are linearly independent
!  =================================================================

      IF ( prob%m > 0 .AND.                                                    &
           ( .NOT. data%tried_to_remove_deps .AND.                             &
              QPB_control%remove_dependencies ) ) THEN

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        IF ( control%out > 0 .AND. control%print_level >= 1 )                  &
          WRITE( control%out,                                                  &
            "( /, A, 1X, I0, ' equalities from ', I0, ' constraints ' )" )     &
            prefix, data%dims%c_equality, prob%m

!  Set control parameters

        data%FDC_control = control%FDC_control
        data%FDC_control%max_infeas = QPB_control%stop_p

!  Find any dependent rows

        nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1
        CALL FDC_find_dependent( prob%n, data%dims%c_equality,                 &
                                 prob%A%val( : nzc ),                          &
                                 prob%A%col( : nzc ),                          &
                                 prob%A%ptr( : data%dims%c_equality + 1 ),     &
                                 prob%C_l,  n_depen, data%Index_C_freed,       &
                                 data%FDC_data, data%FDC_control,              &
                                 inform%FDC_inform )

!  Record output parameters

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%find_dependent =                                           &
          inform%time%find_dependent + time_now - time_record
        inform%time%clock_find_dependent =                                     &
          inform%time%clock_find_dependent + clock_now - clock_record
        inform%alloc_status = inform%FDC_inform%alloc_status
        inform%factorization_status = inform%FDC_inform%factorization_status
        inform%factorization_integer = inform%FDC_inform%factorization_integer
        inform%factorization_real = inform%FDC_inform%factorization_real
        inform%bad_alloc = inform%FDC_inform%bad_alloc
        inform%non_negligible_pivot = inform%FDC_inform%non_negligible_pivot
        inform%nfacts = inform%nfacts + 1

        IF ( ( control%cpu_time_limit >= zero .AND.                            &
               time_now - time_start > control%cpu_time_limit ) .OR.           &
             ( control%clock_time_limit >= zero .AND.                          &
               clock_now - clock_start > control%clock_time_limit ) ) THEN
          inform%status = GALAHAD_error_cpu_limit
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
          GO TO 800
        END IF

        IF ( printi .AND. inform%non_negligible_pivot < thousand *             &
             data%FDC_control%SLS_control%absolute_pivot_tolerance )           &
            WRITE( control%out, "(                                             &
       &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /,                       &
       &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',    &
       &  /, ' ***  perhaps increase SQP_control%LSQP_control%',               &
       &     'FDC_control%SLS_control%absolute_pivot_tolerance from',          &
       &     ES11.4,'  ***', /, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" )     &
           inform%non_negligible_pivot,                                        &
           data%FDC_control%SLS_control%absolute_pivot_tolerance

!  Check for error exits

        IF ( inform%FDC_inform%status /=                &
             GALAHAD_ok ) THEN

!  Allocate arrays to hold the matrix vector product

          array_name = 'qpc: data%HX'
          CALL SPACE_resize_array( prob%n, data%HX, inform%status,             &
                 inform%alloc_status, array_name = array_name,                 &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  On error exit, compute the current objective function value

          data%HX( : prob%n ) = zero
          CALL QPC_HX( data%dims, prob%n, data%HX( : prob%n ),                 &
                       prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,               &
                       prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
          inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),                 &
                                           data%HX( : prob%n ) )               &
                       + DOT_PRODUCT( prob%X( : prob%n ),                      &
                                      prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

          IF ( inform%FDC_inform%status ==              &
               GALAHAD_error_analysis ) THEN
            inform%status = GALAHAD_error_analysis
            inform%factorization_status = GALAHAD_error_upper_entry
          ELSE IF ( inform%FDC_inform%status ==         &
                    GALAHAD_error_factorization )  THEN
            inform%status = GALAHAD_error_factorization
            inform%factorization_status = GALAHAD_error_bad_bounds
          ELSE
            inform%status = inform%FDC_inform%status
          END IF
          IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
            WRITE( control%out, "( ' ' )" )
            WRITE( control%error, 2040 )                                       &
              inform%FDC_inform%status,                 &
                'FDC_find_dependent'
          END IF
          GO TO 750
        END IF

        IF ( control%out > 0 .AND. control%print_level >= 2 .AND. n_depen > 0 )&
          WRITE( control%out, "(/, ' The following ',I0,' constraints appear', &
       &         ' to be dependent', /, ( 8I8 ) )" ) n_depen, data%Index_C_freed

        remap_freed = n_depen > 0 .AND. prob%n > 0

!  Special case: no free variables

        IF ( prob%n == 0 ) THEN
          IF ( QPB_control%array_syntax_worse_than_do_loop ) THEN
            DO i = 1, prob%m ; prob%Y( i ) = zero ; END DO
            DO i = 1, prob%n ; prob%Z( i ) = prob%G( i ) ; END DO
          ELSE
            prob%Y( : prob%m ) = zero
            prob%Z( : prob%n ) = prob%G( : prob%n )
          END IF
          CALL QPC_HX( data%dims, prob%n, prob%Z( : prob%n ),                  &
                       prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,               &
                       prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
          prob%C( : prob%m ) = zero
          CALL QPC_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                       prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,               &
                       prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          inform%obj = prob%f
          inform%status = GALAHAD_ok
          GO TO 700
        END IF
        data%tried_to_remove_deps = .TRUE.
      ELSE
        remap_freed = .FALSE.
      END IF

      IF ( remap_freed ) THEN

!  Some of the current constraints will be removed by freeing them

        IF ( control%error > 0 .AND. control%print_level >= 1 )                &
          WRITE( control%out, "( /, ' -> ', I0, ' constraints are',            &
         & ' dependent and will be temporarily removed' )" ) n_depen

!  Allocate arrays to indicate which constraints have been freed

        array_name = 'qpc: data%C_freed'
        CALL SPACE_resize_array( n_depen, data%C_freed, inform%status,         &
               inform%alloc_status, array_name = array_name,                   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Free the constraint bounds as required

        DO i = 1, n_depen
          j = data%Index_C_freed( i )
          data%C_freed( i ) = prob%C_l( j )
          prob%C_l( j ) = - control%infinity
          prob%C_u( j ) = control%infinity
          prob%Y( j ) = zero
        END DO

        CALL QPP_initialize( data%QPP_map_freed, data%QPP_control )
        data%QPP_control%infinity = control%infinity
        data%QPP_control%treat_zero_bounds_as_general =                        &
          QPB_control%treat_zero_bounds_as_general

!  Store the problem dimensions

        data%dims_save_freed = data%dims
        data%a_ne = prob%A%ne
        data%h_ne = prob%H%ne

        IF ( printi ) WRITE( control%out,                                      &
               "( /, A, ' problem dimensions before removal of dependencies:', &
     &            /, A,                                                        &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Perform the preprocessing

        CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
        CALL QPP_reorder( data%QPP_map_freed, data%QPP_control,                &
                          data%QPP_inform, data%dims, prob,                    &
                          .FALSE., .FALSE., .FALSE. )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        inform%QPB_inform%time%preprocess = inform%time%preprocess
        inform%QPB_inform%time%preprocess = inform%time%preprocess
        inform%QPA_inform%time%preprocess = inform%time%preprocess

        data%dims%nc = data%dims%c_u_end - data%dims%c_l_start + 1
        data%dims%x_s = 1 ; data%dims%x_e = prob%n
        data%dims%c_s = data%dims%x_e + 1
        data%dims%c_e = data%dims%x_e + data%dims%nc
        data%dims%c_b = data%dims%c_e - prob%m
        data%dims%y_s = data%dims%c_e + 1
        data%dims%y_e = data%dims%c_e + prob%m
        data%dims%y_i = data%dims%c_s + prob%m
        data%dims%v_e = data%dims%y_e

!  Test for satisfactory termination

        IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
          inform%status = data%QPP_inform%status
          IF ( control%out > 0 .AND. control%print_level >= 1 )                &
            WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')" ) &
             prefix, data%QPP_inform%status
          IF ( control%error > 0 .AND. control%print_level > 0 )               &
            WRITE( control%error, 2010 ) inform%status
          IF ( control%out > 0 .AND. control%print_level > 0 .AND.             &
               inform%status == GALAHAD_error_upper_entry )                    &
            WRITE( control%error, 2240 )
          CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,            &
                              data%QPP_inform )
          CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
          IF ( data%QPP_inform%status == - GALAHAD_error_primal_infeasible)    &
            inform%status = GALAHAD_error_primal_infeasible
          GO TO 800
        END IF

!  Record revised array lengths

        IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
          data%a_ne = prob%m * prob%n
        ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
        ELSE
          data%a_ne = prob%A%ne
        END IF

        IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
          data%h_ne = prob%n
        ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
          data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
        ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
          data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
        ELSE
          data%h_ne = prob%H%ne
        END IF

        IF ( printi ) WRITE( control%out,                                      &
          "( A, ' problem dimensions after removal of dependencies:', /, A,    &
     &          ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )  &
             prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Further permute initial working sets if provided

        IF ( QPA_control%cold_start == 0 ) THEN
          CALL SORT_inplace_permute( data%QPP_map_freed%m,                     &
                                     data%QPP_map_freed%c_map,                 &
                                     IX = C_stat( : data%QPP_map_freed%m ) )
          CALL SORT_inplace_permute( data%QPP_map_freed%n,                     &
                                     data%QPP_map_freed%x_map,                 &
                                     IX = B_stat( : data%QPP_map_freed%n ) )
        END IF
      END IF

!  Special case: Bound-constrained QP

!     IF ( data%a_ne == 0 .AND. data%h_ne /= 0 .AND.                           &
!          QPB_control%extrapolate > 0 .AND. printi )                          &
!       WRITE( control%error,                                                  &
!        "( A, ' >->->-> turned diagonal bqp solver off for testing ')" ) prefix

      IF ( data%a_ne == 0 .AND. data%h_ne /= 0 ) THEN

!  Check to see if the Hessian is diagonal

        diagonal_qp = .TRUE.
        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'DIAGONAL' )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                diagonal_qp = .FALSE. ; GO TO 3
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, prob%n
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
                diagonal_qp = .FALSE. ; GO TO 3
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( i /= j .AND. prob%H%val( l ) /= zero ) THEN
              diagonal_qp = .FALSE. ; GO TO 3
            END IF
          END DO
        END SELECT
  3     CONTINUE

        remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
        IF ( diagonal_qp ) THEN
          IF ( printi ) WRITE( control%out,                                    &
            "( /, A, ' solving separable bound-constrained QP ' )" ) prefix
          CALL QPD_solve_separable_BQP( prob, QPB_control%infinity,            &
                                        QPB_control%obj_unbounded,             &
                                        inform%QPB_inform%obj,                 &
                                        inform%QPB_inform%feasible,            &
                                        inform%QPB_inform%status,              &
                                        B_stat = B_stat( : prob%n ) )
          IF ( printi ) WRITE( control%out,                                    &
              "( A, ' on exit from QPB_optimal_for_SBQP: status = ', I0,       &
           &   ', time = ', F0.2, /, A, ' objective value =', ES12.4,          &
           &   /, A, ' active bounds: ', I0, ' from ', I0 )" )                 &
              prefix, inform%QPB_inform%status, inform%QPB_inform%time%total,  &
              prefix, inform%QPB_inform%obj, prefix,                           &
              COUNT( B_stat( : prob%n ) /= 0 ), prob%n
          inform%obj = inform%QPB_inform%obj
          inform%status = inform%QPB_inform%status
          inform%QPB_inform%iter = 0
          inform%QPB_inform%non_negligible_pivot = zero
          inform%QPB_inform%factorization_integer = 0
          inform%QPB_inform%factorization_real = 0
          GO TO 700
        ELSE
          CALL QPB_feasible_for_BQP( prob, data, QPB_control,                  &
                                     inform%QPB_inform )
        END IF
      END IF

      cqp = control%convex
      IF ( data%a_ne == 0 .AND. data%h_ne /= 0 .AND. .NOT. cqp ) THEN

!  Special case: equality-constrained QP

      ELSE IF ( prob%n == data%dims%x_free .AND.                               &
                prob%m == data%dims%c_equality ) THEN
        prob%C( : prob%m ) = - prob%C_l( : prob%m )
        IF ( printi ) WRITE( control%out,                                      &
          "( /, A, ' solving equality-constrained QP ' )" ) prefix
        CALL EQP_solve( prob, data, control%EQP_control, inform%EQP_inform )

!  Record the exit status

        inform%status = inform%EQP_inform%status
        inform%alloc_status = inform%EQP_inform%alloc_status
        inform%obj = inform%EQP_inform%obj
        inform%time%total = inform%time%total + inform%EQP_inform%time%total
        inform%time%factorize =                                                &
          inform%time%factorize + inform%EQP_inform%time%factorize
        inform%time%solve =inform%time%solve + inform%EQP_inform%time%solve

        IF ( printi ) THEN
          IF ( control%EQP_control%out > 0 .AND.                               &
               control%EQP_control%print_level > 0 )                           &
            WRITE( control%out, "( /, A, ' returned from EQP' )" ) prefix
          WRITE( control%out, "( A, ' on exit from EQP: status = ', I0,        &
         &   ', iterations = ', I0, ', time = ', F0.2 )" ) prefix,             &
            inform%EQP_inform%status, inform%EQP_inform%cg_iter,               &
            inform%EQP_inform%time%total
        END IF

!  Record the objective and constraint values

        IF ( inform%status /= GALAHAD_error_allocate .AND.                     &
             inform%status /= GALAHAD_error_deallocate .AND.                   &
             inform%status /= GALAHAD_error_analysis .AND.                     &
             inform%status /= GALAHAD_error_factorization .AND.                &
             inform%status /= GALAHAD_error_solve .AND.                        &
             inform%status /= GALAHAD_error_uls_analysis .AND.                 &
             inform%status /= GALAHAD_error_uls_solve ) THEN
          prob%C( : prob%m ) = zero
          CALL QPC_AX( prob%m, prob%C( : prob%m ), prob%m,                     &
                       prob%A%ptr( prob%m + 1 ) - 1, prob%A%val,               &
                       prob%A%col, prob%A%ptr, prob%n, prob%X, '+ ')
          prob%Z( : prob%n ) = zero
          CALL QPC_HX( data%dims, prob%n, prob%Z( : prob%n ),                  &
                       prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,               &
                       prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
          inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),                 &
                                           prob%Z( : prob%n ) )                &
                            + DOT_PRODUCT( prob%X( : prob%n ),                 &
                                           prob%G( : prob%n ) ) + prob%f
        END IF
        prob%Z( : prob%n ) = zero
        remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
        GO TO 700

!  General case: QP or LP

      ELSE
        first_pass = .TRUE.
        center = QPB_control%center
        f = prob%f

!  Check to see if the Hessian is diagonal and positive semi-definite

        convex_diagonal_qp = .TRUE.

!       IF ( convex_diagonal_qp .AND. QPB_control%extrapolate > 0 ) THEN
!         IF ( printi ) WRITE( control%error,                                  &
!       "( A, ' >->->-> turned convex diagonal qp solver off for testing ')" ) &
!            prefix
!         convex_diagonal_qp = .FALSE. ; GO TO 5
!       END IF

        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'DIAGONAL' )
          IF ( COUNT( prob%H%val( : prob%n ) < zero ) > 0 )                    &
            convex_diagonal_qp = .FALSE.
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( ( i /= j .AND. prob%H%val( l ) /= zero ) .OR.               &
                   ( i == j .AND. prob%H%val( l ) <  zero ) ) THEN
                convex_diagonal_qp = .FALSE. ; GO TO 5
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, prob%n
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( ( i /= j .AND. prob%H%val( l ) /= zero ) .OR.               &
                   ( i == j .AND. prob%H%val( l ) <  zero ) ) THEN
                convex_diagonal_qp = .FALSE. ; GO TO 5
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( ( i /= j .AND. prob%H%val( l ) /= zero ) .OR.                 &
                 ( i == j .AND. prob%H%val( l ) <  zero ) ) THEN
              convex_diagonal_qp = .FALSE. ; GO TO 5
            END IF
          END DO
        END SELECT
  5     CONTINUE

        cqp = control%convex
        IF ( data%h_ne == 0 .OR. convex_diagonal_qp .OR. cqp ) THEN
          prob%gradient_kind = 2
          IF ( NRM2( prob%n, prob%G, 1 ) <= epsmch ) THEN
            prob%gradient_kind = 0
          ELSE IF ( NRM2( prob%n, prob%G - one, 1 ) <= epsmch ) THEN
            prob%gradient_kind = 1
          END IF
        ELSE
          prob%gradient_kind = 0
        END IF

!  ==============================
!  Find an initial feasible point
!  ==============================

 10     CONTINUE

!  If the problem is said to be a convex QP, solve in a single phase
!  -----------------------------------------------------------------

        IF ( cqp ) THEN
          lsqp = .FALSE.

!  Set appropraiate control parameters

          CQP_control = control%CQP_control
          CQP_control%infinity = control%infinity
          CQP_control%identical_bounds_tol = control%identical_bounds_tol
          CQP_control%treat_zero_bounds_as_general =                           &
            control%treat_zero_bounds_as_general
          CQP_control%feasol = .FALSE.

!  Solve the CQP

          IF ( printi ) WRITE( control%out, "( /, A, ' entering CQP ' )" )     &
            prefix
          CALL CQP_solve( prob, data, CQP_control, inform%CQP_inform,          &
                          C_stat = C_stat( : prob%m ),                         &
                          B_stat = B_stat( : prob%n ) )
!         write(6,*) ' cqp - status ', CQP_inform%status
!         write(6,*) ' c_stat ', C_stat( : prob%m )
!         write(6,*) ' b_stat ', B_stat( : prob%n )

!         WRITE( control%out, "(' b_stat ', /, ( 10I5 ) )" ) B_stat( : prob%n )
!         WRITE( control%out, "(' c_stat ', /, ( 10I5 ) )" ) C_stat( : prob%m )

          IF ( printi ) THEN
            IF ( CQP_control%out > 0 .AND. CQP_control%print_level > 0 )       &
              WRITE( control%out, "( /, A, ' returned from CQP' )" ) prefix
            WRITE( control%out, "( A, ' on exit from CQP: status = ', I0,      &
           &   ', iterations = ', I0, ', time = ', F0.2 )" ) prefix,           &
              inform%CQP_inform%status, inform%CQP_inform%iter,                &
              inform%CQP_inform%time%total
            WRITE( control%out, "( A, ' objective value =', ES12.4,            &
           &   /, A, ' # active counstraints: ', I0, ' from ', I0,             &
           &   ', bounds: ', I0, ' from ', I0 )" ) prefix,                     &
              inform%CQP_inform%obj, prefix, COUNT( C_stat( : prob%m ) /= 0 ), &
              prob%m, COUNT( B_stat( : prob%n ) /= 0 ),  prob%n
          END IF

          prob%f = f

          IF ( printi )                                                        &
            WRITE( control%out, "( A, ' ', I0, ' integer and ', I0,            &
          &  ' real words required for factors in CQP' )" ) prefix,            &
             inform%CQP_inform%factorization_integer,                          &
             inform%CQP_inform%factorization_real

!  Record output information from CQP

          inform%alloc_status = inform%CQP_inform%alloc_status
          inform%nfacts = inform%nfacts + inform%CQP_inform%nfacts
          inform%obj = inform%CQP_inform%obj
          inform%time%total = inform%time%total + inform%CQP_inform%time%total
          inform%time%analyse =                                                &
            inform%time%analyse + inform%CQP_inform%time%analyse
          inform%time%factorize =                                              &
            inform%time%factorize + inform%CQP_inform%time%factorize
          inform%time%solve =inform%time%solve + inform%CQP_inform%time%solve
          inform%time%preprocess =                                             &
            inform%time%preprocess + inform%CQP_inform%time%preprocess

!  The problem is not thought to be a convex QP, so solve in two phases
!  --------------------------------------------------------------------

        ELSE

!  Set appropraiate control parameters for the phase 1

          LSQP_control = QPB_control%LSQP_control
          LSQP_control%print_level = QPB_control%print_level
          LSQP_control%out = QPB_control%out
          LSQP_control%error = QPB_control%error
          LSQP_control%infeas_max = QPB_control%infeas_max
          LSQP_control%restore_problem = QPB_control%restore_problem
          LSQP_control%infinity = control%infinity
          LSQP_control%stop_p = QPB_control%stop_p
          LSQP_control%stop_c = QPB_control%stop_c
          LSQP_control%stop_d = QPB_control%stop_d
          LSQP_control%muzero = QPB_control%muzero
          LSQP_control%reduce_infeas = QPB_control%reduce_infeas
          LSQP_control%identical_bounds_tol = control%identical_bounds_tol
          LSQP_control%remove_dependencies = QPB_control%remove_dependencies
          LSQP_control%treat_zero_bounds_as_general =                          &
            control%treat_zero_bounds_as_general
          LSQP_control%feasol = .FALSE.
          LSQP_control%array_syntax_worse_than_do_loop =                       &
            control%array_syntax_worse_than_do_loop

!  Either find the solution to the LP (if there is no Hessian term) ...

          IF ( data%h_ne == 0 ) THEN
            lsqp = .TRUE.
            LSQP_control%just_feasible = .FALSE.
            LSQP_control%maxit = QPB_control%maxit
            LSQP_control%prfeas =                                              &
              MAX( LSQP_control%prfeas, QPB_control%prfeas )
            LSQP_control%dufeas =                                              &
              MAX( LSQP_control%dufeas, QPB_control%dufeas )
            prob%Hessian_kind = 0

!  .. or find the solution to the Diagonal QP (if the Hessian is diagonal) ...

          ELSE IF ( convex_diagonal_qp ) THEN
            lsqp = .TRUE.
            LSQP_control%just_feasible = .FALSE.
            LSQP_control%maxit = QPB_control%maxit
            LSQP_control%prfeas =                                              &
              MAX( LSQP_control%prfeas, QPB_control%prfeas )
            LSQP_control%dufeas =                                              &
              MAX( LSQP_control%dufeas, QPB_control%dufeas )
            prob%Hessian_kind = 2

            array_name = 'qpc: prob%X0'
            CALL SPACE_resize_array( prob%n, prob%X0, inform%status,           &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: prob%WEIGHT'
            CALL SPACE_resize_array( prob%n, prob%WEIGHT, inform%status,       &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            SELECT CASE ( SMT_get( prob%H%type ) )
            CASE ( 'DIAGONAL' )
              prob%WEIGHT( : prob%n ) = prob%H%val( : prob%n )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, prob%n
                DO j = 1, i
                  l = l + 1
                  IF ( i == j ) prob%WEIGHT( i ) = prob%H%val( l )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              prob%WEIGHT( : prob%n ) = zero
              DO i = 1, prob%n
                DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                  j = prob%H%col( l )
                  IF ( i == j )                                                &
                    prob%WEIGHT( i ) = prob%WEIGHT( i ) + prob%H%val( l )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              prob%WEIGHT( : prob%n ) = zero
              DO l = 1, prob%H%ne
                i = prob%H%row( l ) ; j = prob%H%col( l )
                IF ( i == j )                                                  &
                  prob%WEIGHT( i ) = prob%WEIGHT( i ) + prob%H%val( l )
              END DO
            END SELECT
            prob%WEIGHT( : prob%n ) = SQRT( prob%WEIGHT( : prob%n ) )
            prob%X0( : prob%n ) = zero

!  .. or find a centered feasible point ...

          ELSE IF ( center ) THEN
            lsqp = .FALSE.
!           LSQP_control%potential_unbounded = -100.0_wp
            LSQP_control%just_feasible = .FALSE.
            LSQP_control%prfeas = QPB_control%prfeas
            LSQP_control%dufeas = QPB_control%dufeas
            prob%Hessian_kind = 0
            prob%f = zero

!  .. or minimize the distance to the nearest feasible point

          ELSE
            lsqp = .FALSE.
            LSQP_control%just_feasible = .TRUE.
            LSQP_control%prfeas = QPB_control%prfeas
            LSQP_control%dufeas = QPB_control%dufeas
            prob%Hessian_kind = 1

            array_name = 'qpc: prob%X0'
            CALL SPACE_resize_array( prob%n, prob%X0, inform%status,           &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: data%X0'
            CALL SPACE_resize_array( prob%n, data%X0, inform%status,           &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            data%X0( : prob%n ) = prob%X( : prob%n )
            prob%X0( : prob%n ) = data%X0( : prob%n )
            prob%f = zero
          END IF

!  Solve the LSQP or phase-1 problem

          LSQP_inform%time%analyse = 0.0 ; LSQP_inform%time%clock_analyse = 0.0
          LSQP_inform%time%factorize = 0.0
          LSQP_inform%time%clock_factorize = 0.0
          LSQP_inform%time%solve = 0.0 ; LSQP_inform%time%clock_solve = 0.0
          CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

          IF ( lsqp ) THEN
            IF ( printi ) WRITE( control%out, "( /, A, ' entering LSQP ' )" )  &
              prefix
            LSQP_control%indicator_type = QPB_control%indicator_type
            LSQP_control%indicator_tol_p = QPB_control%indicator_tol_p
            LSQP_control%indicator_tol_pd = QPB_control%indicator_tol_pd
            LSQP_control%indicator_tol_tapia = QPB_control%indicator_tol_tapia
            CALL LSQP_solve( prob, data, LSQP_control, LSQP_inform,            &
                             C_stat = C_stat( : prob%m ),                      &
                             B_stat = B_stat( : prob%n ) )
!           write(6,*) ' lsqp - status ', LSQP_inform%status
!           write(6,*) ' c_stat ', C_stat( : prob%m )
          ELSE
            IF ( printi )                                                      &
              WRITE( control%out, "( /, A, ' entering LSQP -- phase-one' )" )  &
                prefix
            CALL LSQP_solve( prob, data, LSQP_control, LSQP_inform,            &
                             C_stat = C_stat( : prob%m ),                      &
                             B_stat = B_stat( : prob%n ) )
          END IF

!  Record times for components of LSQP

          CALL CPU_TIME( time_now ) ; time_now = time_now - time_record
          CALL CLOCK_time( clock_now ) ; clock_now = clock_now - clock_record
          inform%QPB_inform%time%total = inform%QPB_inform%time%total + time_now
          inform%QPB_inform%time%clock_total =                                 &
            inform%QPB_inform%time%clock_total + clock_now
          inform%QPB_inform%time%phase1_total =                                &
            inform%QPB_inform%time%phase1_total + time_now
          inform%QPB_inform%time%clock_phase1_total =                          &
            inform%QPB_inform%time%clock_phase1_total + clock_now
          inform%QPB_inform%time%phase1_analyse =                              &
            inform%QPB_inform%time%phase1_analyse + LSQP_inform%time%analyse
          inform%QPB_inform%time%clock_phase1_analyse =                        &
            inform%QPB_inform%time%clock_phase1_analyse +                      &
              LSQP_inform%time%clock_analyse
          inform%QPB_inform%time%phase1_factorize =                            &
            inform%QPB_inform%time%phase1_factorize + LSQP_inform%time%factorize
          inform%QPB_inform%time%clock_phase1_factorize =                      &
            inform%QPB_inform%time%clock_phase1_factorize +                    &
              LSQP_inform%time%clock_factorize
          inform%QPB_inform%time%phase1_solve =                                &
            inform%QPB_inform%time%phase1_solve + LSQP_inform%time%solve
          inform%QPB_inform%time%clock_phase1_solve =                          &
            inform%QPB_inform%time%clock_phase1_solve +                        &
              LSQP_inform%time%clock_solve

!         WRITE( control%out, "(' b_stat ', /, ( 10I5 ) )" ) B_stat( : prob%n )
!         WRITE( control%out, "(' c_stat ', /, ( 10I5 ) )" ) C_stat( : prob%m )

!  Record the exit status

          inform%status = LSQP_inform%status
          inform%QPB_inform%status = LSQP_inform%status
          inform%QPB_inform%feasible = LSQP_inform%feasible

          IF ( printi ) THEN
            IF ( LSQP_control%out > 0 .AND. LSQP_control%print_level > 0 )     &
              WRITE( control%out, "( /, A, ' returned from LSQP' )" ) prefix
            WRITE( control%out, "( A, ' on exit from LSQP: status = ', I0,     &
           &   ', iterations = ', I0, ', time = ', F0.2 )" ) prefix,           &
              LSQP_inform%status, LSQP_inform%iter, clock_now
            IF ( lsqp ) WRITE( control%out, "( A, ' objective value =', ES12.4,&
           &   /, A, ' # active counstraints: ', I0, ' from ', I0,  &
           &   ', bounds: ', I0, ' from ', I0 )" ) prefix, LSQP_inform%obj,    &
              prefix, COUNT( C_stat( : prob%m ) /= 0 ),  prob%m,               &
              COUNT( B_stat( : prob%n ) /= 0 ),  prob%n
          END IF

!  If the analytic center appears to be unbounded, have another attempt
!  at getting feasible

          IF ( inform%QPB_inform%status == GALAHAD_error_upper_entry .OR.      &
               inform%QPB_inform%status == GALAHAD_error_factorization .OR.    &
               inform%QPB_inform%status == GALAHAD_error_ill_conditioned .OR.  &
               inform%QPB_inform%status == GALAHAD_error_tiny_step .OR.        &
               inform%QPB_inform%status == GALAHAD_error_max_iterations .OR.   &
             ( inform%QPB_inform%status == GALAHAD_error_unbounded             &
                .AND. .NOT. lsqp ) )  THEN
            IF ( inform%QPB_inform%feasible ) THEN
              inform%status = GALAHAD_ok
            ELSE
              IF ( first_pass .AND. .NOT. lsqp ) THEN
                center = .NOT. center
                first_pass = .FALSE.
                IF ( printi ) WRITE( control%out,                              &
                "( /, ' .... have a second attempt at getting feasible ....' )")
                GO TO 10
              END IF
            END IF
          END IF
          prob%f = f
!         IF ( .NOT. lsqp .AND. .NOT. center ) NULLIFY( prob%X0 )
          prob%Hessian_kind = - 1 ; prob%gradient_kind = - 1

          IF ( printi )                                                        &
            WRITE( control%out, "( A, ' ', I0, ' integer and ', I0,            &
          &  ' real words required for factors in LSQP' )" ) prefix,           &
             LSQP_inform%factorization_integer, LSQP_inform%factorization_real

!  Record output information from LSQP

          inform%QPB_inform%alloc_status = LSQP_inform%alloc_status
          inform%QPB_inform%iter = LSQP_inform%iter
          inform%QPB_inform%factorization_status =                             &
            LSQP_inform%factorization_status
          inform%QPB_inform%factorization_integer =                            &
            LSQP_inform%factorization_integer
          inform%QPB_inform%factorization_real =                               &
            LSQP_inform%factorization_real
          inform%QPB_inform%nfacts = LSQP_inform%nfacts
          inform%QPB_inform%nbacts = LSQP_inform%nbacts
          inform%QPB_inform%nmods = 0
          inform%QPB_inform%obj = LSQP_inform%obj

          inform%alloc_status = inform%QPB_inform%alloc_status
          inform%nfacts = inform%nfacts + inform%QPB_inform%nfacts
          inform%nmods = inform%nmods + inform%QPB_inform%nmods
          inform%obj = inform%QPB_inform%obj
        END IF

!  Check for error exits

        IF ( inform%status /= GALAHAD_ok .AND.                                 &
!            inform%status /= GALAHAD_error_primal_infeasible .AND.            &
             inform%status /= GALAHAD_error_ill_conditioned .AND.              &
             inform%status /= GALAHAD_error_tiny_step ) THEN
!            inform%status /= GALAHAD_error_max_iterations

!  On error exit, compute the current objective function value

          IF ( inform%status /= GALAHAD_error_allocate .AND.                   &
               inform%status /= GALAHAD_error_deallocate ) THEN
            data%HX( : prob%n ) = zero
            CALL QPC_HX( data%dims, prob%n, data%HX( : prob%n ),               &
                         prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,             &
                         prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
            inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),               &
                                             data%HX( : prob%n ) )             &
                              + DOT_PRODUCT( prob%X( : prob%n ),               &
                                        prob%G( : prob%n ) ) + prob%f
          END IF
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          GO TO 700
        END IF

!  Move to crossover or exit (as appropriate) if the problem is an LSQP

        IF ( lsqp .OR. cqp ) THEN
          remap_more_freed = .FALSE. ; remap_fixed = .FALSE.
          IF ( ( control%qpb_or_qpa  .AND. inform%status == GALAHAD_ok ) .OR.  &
                 control%no_qpa ) THEN
!         IF ( control%qpb_or_qpa .OR. control%no_qpa ) THEN
!         IF ( .NOT. control%qpb_or_qpa .AND. control%no_qpa ) THEN
!         IF ( control%no_qpa .AND.                                            &
!             inform%status /= GALAHAD_error_ill_conditioned .AND.             &
!             inform%status /= GALAHAD_error_tiny_step ) THEN
            GO TO 700
          ELSE

!  Save the solution in case QPA fails

            array_name = 'qpc: data%X_trial'
            CALL SPACE_resize_array( prob%n, data%X_trial, inform%status,      &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: data%Y_last'
            CALL SPACE_resize_array( prob%m, data%Y_last, inform%status,       &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: data%Z_last'
            CALL SPACE_resize_array( prob%n, data%Z_last, inform%status,       &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: data%C'
            CALL SPACE_resize_array( 1, prob%m, data%C, inform%status,         &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            gotsol = .TRUE.
            data%X_trial( : prob%n ) = prob%X( : prob%n )
            data%Y_last( : prob%m ) = prob%Y( : prob%m )
            data%Z_last( : prob%n ) = prob%Z( : prob%n )
            data%C( : prob%m ) = prob%C( : prob%m )
            best_obj = inform%obj

            GO TO 150
          END IF
        END IF

!  ============================
!  Initial feasible point found
!  ============================

!  Check to see if any variables/constraints are flagged as being fixed

        tol = MIN( QPB_control%stop_p / ten, SQRT( epsmch ) )
        tiny_x = COUNT( prob%X( data%dims%x_free + 1 : data%dims%x_l_start - 1)&
                        < tol ) +                                              &
                 COUNT( prob%X( data%dims%x_l_start: data%dims%x_l_end ) -     &
                        prob%X_l( data%dims%x_l_start : data%dims%x_l_end )    &
                        < tol ) +                                              &
                 COUNT( prob%X( data%dims%x_u_start: data%dims%x_u_end ) -     &
                        prob%X_u( data%dims%x_u_start : data%dims%x_u_end )    &
                        > - tol ) +                                            &
                 COUNT( prob%X( data%dims%x_u_end + 1 : prob%n )               &
                        > - tol )

        tiny_c = COUNT( prob%C( data%dims%c_l_start: data%dims%c_l_end ) -     &
                        prob%C_l( data%dims%c_l_start : data%dims%c_l_end )    &
                        < tol ) +                                              &
                 COUNT( prob%C( data%dims%c_u_start: data%dims%c_u_end ) -     &
                        prob%C_u( data%dims%c_u_start : data%dims%c_u_end )    &
                        > - tol )

        remap_fixed = tiny_x > 0 .OR. tiny_c > 0
        IF ( remap_fixed ) THEN

!  Some of the current variables/constraints will be fixed

          IF ( control%error > 0 .AND. control%print_level >= 1 )              &
            WRITE( control%out, "( /, ' -> ', I0, ' further variables and ',   &
           &       I0, ' further constraints will be fixed' )" ) tiny_x, tiny_c

!  Allocate arrays to record the bounds which will be altered

          IF ( tiny_x > 0 ) THEN
            array_name = 'qpc: data%X_fixed'
            CALL SPACE_resize_array( tiny_x, data%X_fixed, inform%status,      &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: data%Index_X_fixed'
            CALL SPACE_resize_array( tiny_x,                                   &
                   data%Index_X_fixed, inform%status,                          &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

          IF ( tiny_c > 0 ) THEN
            array_name = 'qpc: data%C_fixed'
            CALL SPACE_resize_array( tiny_c, data%C_fixed, inform%status,      &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

            array_name = 'qpc: data%Index_C_fixed'
            CALL SPACE_resize_array( tiny_c,                                   &
                   data%Index_C_fixed, inform%status,                          &
                   inform%alloc_status, array_name = array_name,               &
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900
          END IF

!  Fix the problem bounds as required

          IF ( tiny_x > 0 ) THEN
            tiny_x = 0

            DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i )
              IF ( prob%X( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_u( i ) = zero
              END IF
            END DO

            DO i = data%dims%x_l_start, data%dims%x_u_start - 1
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i ) - prob%X_l( i )
              IF ( prob%X( i ) - prob%X_l( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_u( i ) =  prob%X_l( i )
              END IF
            END DO

            DO i = data%dims%x_u_start, data%dims%x_l_end
!             write(6,"( I6, A1, ES12.4 )" ) i, 'l', prob%X( i ) - prob%X_l( i )
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', prob%X_u( i ) - prob%X( i )
              IF ( prob%X( i ) - prob%X_l( i ) < tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_u( i )
                data%Index_X_fixed( tiny_x ) = - i
                prob%X_u( i ) =  prob%X_l( i )
              ELSE IF ( prob%X( i ) - prob%X_u( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_l( i ) =  prob%X_u( i )
              END IF
            END DO

            DO i = data%dims%x_l_end + 1, data%dims%x_u_end
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', prob%X_u( i ) - prob%X( i )
              IF ( prob%X( i ) - prob%X_u( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_l( i ) =  prob%X_u( i )
              END IF
            END DO

            DO i = data%dims%x_u_end + 1, prob%n
!             write(6,"( I6, A1, ES12.4 )" ) i, 'u', - prob%X( i )
              IF ( prob%X( i ) > - tol ) THEN
                tiny_x = tiny_x + 1
                data%X_fixed( tiny_x ) = prob%X_l( i )
                data%Index_X_fixed( tiny_x ) = i
                prob%X_l( i ) = zero
              END IF
            END DO
          END IF

!  Do the same for the constraint bounds

          IF ( tiny_c > 0 ) THEN
            tiny_c = 0

            DO i = data%dims%c_l_start, data%dims%c_u_start - 1
              IF ( prob%C( i ) - prob%C_l( i ) < tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_u( i )
                data%Index_C_fixed( tiny_c ) = - i
                prob%C_u( i ) =  prob%C_l( i )
              END IF
            END DO

            DO i = data%dims%c_u_start, data%dims%c_l_end
              IF ( prob%C( i ) - prob%C_l( i ) < tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_u( i )
                data%Index_C_fixed( tiny_c ) = - i
                prob%C_u( i ) =  prob%C_l( i )
              ELSE IF ( prob%C( i ) - prob%C_u( i ) > - tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_l( i )
                data%Index_C_fixed( tiny_c ) = i
                prob%C_l( i ) =  prob%C_u( i )
              END IF
            END DO

            DO i = data%dims%c_l_end + 1, data%dims%c_u_end
              IF ( prob%C( i ) - prob%C_u( i ) > - tol ) THEN
                tiny_c = tiny_c + 1
                data%C_fixed( tiny_c ) = prob%C_l( i )
                data%Index_C_fixed( tiny_c ) = i
                prob%C_l( i ) =  prob%C_u( i )
              END IF
            END DO
          END IF

! write(6,"( ' tiny_x = ', I0, /, ( 10I5 ))" ) tiny_x, data%Index_X_fixed
! write(6,"( ' tiny_c = ', I0, /, ( 10I5 ))" ) tiny_c, data%Index_C_fixed
          CALL QPP_initialize( data%QPP_map_fixed, data%QPP_control )
          data%QPP_control%infinity = control%infinity
          data%QPP_control%treat_zero_bounds_as_general =                      &
            QPB_control%treat_zero_bounds_as_general

!  Store the problem dimensions

          data%dims_save_fixed = data%dims
          data%a_ne = prob%A%ne
          data%h_ne = prob%H%ne

          IF ( printi ) WRITE( control%out,                                    &
                 "( /, A, ' problem dimensions before preprocessing: ', /,  A, &
     &           ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" ) &
                 prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Perform the preprocessing

          CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
          CALL QPP_reorder( data%QPP_map_fixed, data%QPP_control,              &
                             data%QPP_inform, data%dims, prob,                 &
                             .FALSE., .FALSE., .FALSE. )
          CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%preprocess =                                             &
            inform%time%preprocess + time_now - time_record
          inform%time%clock_preprocess =                                       &
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

!  Test for satisfactory termination

          IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
            inform%status = data%QPP_inform%status
            IF ( control%out > 0 .AND. control%print_level >= 1 )              &
              WRITE( control%out, "( A, ' status ', I0, ' after QPP_reorder')")&
               prefix, data%QPP_inform%status
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, 2010 ) inform%status
            IF ( control%out > 0 .AND. control%print_level > 0 .AND.           &
                 inform%status == GALAHAD_error_upper_entry )                  &
              WRITE( control%error, 2240 )
            CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,          &
                                data%QPP_inform )
            CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,          &
                                data%QPP_inform )
            CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform)
            IF ( data%QPP_inform%status == GALAHAD_error_primal_infeasible)    &
              inform%status = GALAHAD_error_primal_infeasible
            GO TO 800
          END IF

!  Record revised array lengths

          IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
            data%a_ne = prob%m * prob%n
          ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
            data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
          ELSE
            data%a_ne = prob%A%ne
          END IF

          IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            data%h_ne = prob%n
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
          ELSE
            data%h_ne = prob%H%ne
          END IF

          IF ( printi ) WRITE( control%out,                                    &
                 "( A, ' problem dimensions after preprocessing: ', /,  A,     &
     &           ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" ) &
                 prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Further permute initial working sets if provided

          IF ( QPA_control%cold_start == 0 ) THEN
            CALL SORT_inplace_permute( data%QPP_map_fixed%m,                   &
                                       data%QPP_map_fixed%c_map,               &
                                       IX = C_stat( : data%QPP_map_fixed%m ) )
            CALL SORT_inplace_permute( data%QPP_map_fixed%n,                   &
                                       data%QPP_map_fixed%x_map,               &
                                       IX = B_stat( : data%QPP_map_fixed%n ) )
          END IF

!  If all the variables have now been fixed, the solution has been found

          IF ( prob%n == 0 ) THEN

!  Check that the solution is feasible

            DO i = 1, prob%m
              IF ( prob%C_l( i ) > QPB_control%stop_p .OR.             &
                   prob%C_u( i ) < - QPB_control%stop_p ) THEN
                inform%status = GALAHAD_error_primal_infeasible
                GO TO 800
              END IF
            END DO
            prob%C( : prob%m ) = zero
            prob%Y( : prob%m ) = zero
            inform%obj = prob%f
            inform%status = 0
            GO TO 720
          END IF

!  ====================================================================
!  Check to see if the equality constraints remain linearly independent
!  ====================================================================

          IF ( prob%m > 0 .AND. QPB_control%remove_dependencies ) THEN

            CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
            IF ( control%out > 0 .AND. control%print_level >= 1 )              &
              WRITE( control%out,                                              &
                "( /, A, 1X, I0, ' equalities from ', I0, ' constraints' )" )  &
                prefix, data%dims%c_equality, prob%m

!  Find any dependent rows

            nzc = prob%A%ptr( data%dims%c_equality + 1 ) - 1
            CALL FDC_find_dependent( prob%n, data%dims%c_equality,             &
                                     prob%A%val( : nzc ),                      &
                                     prob%A%col( : nzc ),                      &
                                     prob%A%ptr( : data%dims%c_equality + 1 ), &
                                     prob%C_l,                                 &
                                     n_more_depen, data%Index_C_more_freed,    &
                                     data%FDC_data, data%FDC_control,          &
                                     inform%FDC_inform )
            inform%non_negligible_pivot =                                      &
              MIN( inform%FDC_inform%non_negligible_pivot,                     &
                   inform%non_negligible_pivot )

!  Record output parameters

            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%find_dependent =                                       &
              inform%time%find_dependent + time_now - time_record
            inform%time%clock_find_dependent =                                 &
              inform%time%clock_find_dependent + clock_now - clock_record
            inform%alloc_status = inform%FDC_inform%alloc_status
            inform%factorization_status = inform%FDC_inform%factorization_status
            inform%factorization_integer =                                     &
              inform%FDC_inform%factorization_integer
            inform%factorization_real = inform%FDC_inform%factorization_real
            inform%bad_alloc = inform%FDC_inform%bad_alloc
            inform%non_negligible_pivot = inform%FDC_inform%non_negligible_pivot
            inform%nfacts = inform%nfacts + 1

            IF ( ( control%cpu_time_limit >= zero .AND.                        &
                   time_now - time_start > control%cpu_time_limit ) .OR.       &
                 ( control%clock_time_limit >= zero .AND.                      &
                   clock_now - clock_start > control%clock_time_limit ) ) THEN
              inform%status = GALAHAD_error_cpu_limit
              IF ( control%error > 0 .AND. control%print_level > 0 )           &
                WRITE( control%error, 2010 ) inform%status
              GO TO 800
            END IF

            IF ( printi .AND. inform%non_negligible_pivot < thousand *         &
                 data%FDC_control%SLS_control%absolute_pivot_tolerance )       &
                WRITE( control%out, "(                                         &
           &  /, 1X, 26 ( '*' ), ' WARNING ', 26 ( '*' ), /,                   &
           &  ' ***  smallest allowed pivot =', ES11.4,' may be too small ***',&
           &  /, ' ***  perhaps increase QPB_control%LSQP_control%',           &
           &     'FDC_control%SLS_control%absolute_pivot_tolerance from',      &
           &     ES11.4,'  ***', /, 1X, 26 ( '*' ), ' WARNING ', 26 ('*') )" ) &
               inform%non_negligible_pivot,                                    &
               data%FDC_control%SLS_control%absolute_pivot_tolerance

!  Check for error exits

            IF ( inform%FDC_inform%status /=            &
                 GALAHAD_ok ) THEN

!  Allocate arrays to hold the matrix vector product

              array_name = 'qpc: data%HX'
              CALL SPACE_resize_array( prob%n, data%HX, inform%status,         &
                     inform%alloc_status, array_name = array_name,             &
                     deallocate_error_fatal = control%deallocate_error_fatal,  &
                     exact_size = control%space_critical,                      &
                     bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  On error exit, compute the current objective function value

              data%HX( : prob%n ) = zero
              CALL QPC_HX( data%dims, prob%n, data%HX( : prob%n ),             &
                            prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,          &
                            prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
              inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),             &
                                               data%HX( : prob%n ) )           &
                           + DOT_PRODUCT( prob%X( : prob%n ),                  &
                                          prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

              IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
                WRITE( control%out, "( ' ' )" )
                WRITE( control%error, 2040 )                                   &
                  inform%FDC_inform%status,             &
                  'FDC_find_dependent'
              END IF
              inform%status = inform%FDC_inform%status
              GO TO 700
            END IF

            IF ( control%out > 0 .AND. control%print_level >= 2                &
                 .AND. n_more_depen > 0 )                                      &
              WRITE( control%out, "(/, ' The following ', I0,                  &
           &         ' constraints appear to be dependent', /, ( 8I8 ) )" )    &
                n_more_depen, data%Index_C_more_freed

            remap_more_freed = n_more_depen > 0
          ELSE
            remap_more_freed = .FALSE.
          END IF

          IF ( remap_more_freed ) THEN

!  Some of the current constraints will be removed by freeing them

            IF ( control%error > 0 .AND. control%print_level >= 1 )            &
              WRITE( control%out, "( /, ' -> ', I0, ' constraints are',        &
             & ' dependent and will be temporarily removed' )" ) n_more_depen

!  Allocate arrays to indicate which constraints have been freed

            array_name = 'qpc: data%C_more_freed'
            CALL SPACE_resize_array( n_more_depen, data%C_more_freed,          &
                   inform%status, inform%alloc_status, array_name = array_name,&
                   deallocate_error_fatal = control%deallocate_error_fatal,    &
                   exact_size = control%space_critical,                        &
                   bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Free the constraint bounds as required

            DO i = 1, n_more_depen
              j = data%Index_C_more_freed( i )
              data%C_more_freed( i ) = prob%C_l( j )
              prob%C_l( j ) = - control%infinity
              prob%C_u( j ) = control%infinity
              prob%Y( j ) = zero
            END DO

            CALL QPP_initialize( data%QPP_map_more_freed, data%QPP_control )
            data%QPP_control%infinity = control%infinity
            data%QPP_control%treat_zero_bounds_as_general =                    &
              QPB_control%treat_zero_bounds_as_general

!  Store the problem dimensions

            data%dims_save_more_freed = data%dims
            data%a_ne = prob%A%ne
            data%h_ne = prob%H%ne

            IF ( printi ) WRITE( control%out,                                  &
               "( /, A, ' problem dimensions before removal of dependencies:', &
     &            /, A,                                                        &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Perform the preprocessing

            CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
            CALL QPP_reorder( data%QPP_map_more_freed, data%QPP_control,       &
                              data%QPP_inform, data%dims, prob,                &
                              .FALSE., .FALSE., .FALSE. )
            CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
            inform%time%preprocess =                                           &
              inform%time%preprocess + time_now - time_record
            inform%time%clock_preprocess =                                     &
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

!  Test for satisfactory termination

            IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
              inform%status = data%QPP_inform%status
              IF ( control%out > 0 .AND. control%print_level >= 1 )            &
                WRITE( control%out,"( A, ' status ', I0,' after QPP_reorder')")&
                 prefix, data%QPP_inform%status
              IF ( control%error > 0 .AND. control%print_level > 0 )           &
                WRITE( control%error, 2010 ) inform%status
              IF ( control%out > 0 .AND. control%print_level > 0 .AND.         &
                   inform%status == GALAHAD_error_upper_entry )                &
                WRITE( control%error, 2240 )
              CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,   &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,        &
                                  data%QPP_inform )
              CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,        &
                                  data%QPP_inform )
              IF ( data%QPP_inform%status == GALAHAD_error_primal_infeasible ) &
                inform%status = GALAHAD_error_primal_infeasible
              GO TO 800
            END IF

!  Record revised array lengths

            IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
              data%a_ne = prob%m * prob%n
            ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
              data%a_ne = prob%A%ptr( prob%m + 1 ) - 1
            ELSE
              data%a_ne = prob%A%ne
            END IF

            IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
              data%h_ne = prob%n
            ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
              data%h_ne = ( prob%n * ( prob%n + 1 ) ) / 2
            ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
              data%h_ne = prob%H%ptr( prob%n + 1 ) - 1
            ELSE
              data%h_ne = prob%H%ne
            END IF

            IF ( printi ) WRITE( control%out,                                  &
               "( A, ' problem dimensions after removal of dependencies:',     &
     &            /, A,                                                        &
     &         ' n = ', I8, ' m = ', I8, ' a_ne = ', I8, ' h_ne = ', I8 )" )   &
               prefix, prefix, prob%n, prob%m, data%a_ne, data%h_ne

!  Further permute initial working sets if provided

            IF ( QPA_control%cold_start == 0 ) THEN
              CALL SORT_inplace_permute( data%QPP_map_more_freed%m,            &
                                         data%QPP_map_more_freed%c_map,        &
                                    IX = C_stat( : data%QPP_map_more_freed%m ) )
              CALL SORT_inplace_permute( data%QPP_map_more_freed%n,            &
                                         data%QPP_map_more_freed%x_map,        &
                                    IX = B_stat( : data%QPP_map_more_freed%n ) )
            END IF
          END IF

!  Experiment!!

!         GO TO 10
        ELSE
          remap_more_freed = .FALSE.
        END IF
      END IF

!  Allocate additional real workspace

      array_name = 'qpc: data%DZ_l'
      CALL SPACE_resize_array( data%dims%x_l_start, data%dims%x_l_end,         &
             data%DZ_l, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%DZ_u'
      CALL SPACE_resize_array( data%dims%x_u_start, data%dims%x_u_end,         &
             data%DZ_u, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%GRAD'
      CALL SPACE_resize_array( prob%n, data%GRAD, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%X_trial'
      CALL SPACE_resize_array( prob%n, data%X_trial, inform%status,            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%GRAD_X_phi'
      CALL SPACE_resize_array( prob%n, data%GRAD_X_phi, inform%status,         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%GRAD_C_phi'
      CALL SPACE_resize_array( data%dims%c_l_start, data%dims%c_u_end,         &
             data%GRAD_C_phi, inform%status,                                   &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%S'
      CALL SPACE_resize_array( data%dims%c_e, data%S, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%Y_trial'
      CALL SPACE_resize_array( prob%m, data%Y_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%A_s'
      CALL SPACE_resize_array( prob%m, data%A_s, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%H_s'
      CALL SPACE_resize_array( prob%n, data%H_s, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%Y_last'
      CALL SPACE_resize_array( prob%m, data%Y_last, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%Z_last'
      CALL SPACE_resize_array( prob%n, data%Z_last, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      IF ( QPB_control%extrapolate > 0 ) THEN
        data%hist = QPB_control%path_history
        data%deriv = QPB_control%path_derivatives
        IF ( QPB_control%fit_order > 0 ) THEN
          data%order = QPB_control%fit_order
        ELSE
          data%order = data%hist * ( data%deriv + 1 ) - 1
        END IF
      ELSE
        data%hist = 0
        data%deriv = 0
        data%order = 0
      END IF
      data%len_hist = 0

      array_name = 'qpb: data%BINOMIAL'
      CALL SPACE_resize_array( 0, data%deriv - 1, data%deriv, data%BINOMIAL,   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%fit_mu'
      CALL SPACE_resize_array( data%order + 1, data%fit_mu,                    &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%fit_f'
      CALL SPACE_resize_array( data%order + 1, data%fit_f,                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%X_coef'
      CALL SPACE_resize_array( 0, data%order, prob%n, data%X_coef,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%C_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%c_l_start,             &
             data%dims%c_u_end, data%C_coef,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_coef'
      CALL SPACE_resize_array( 0, data%order, prob%m, data%Y_coef,             &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_l_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%c_l_start,             &
             data%dims%c_l_end, data%Y_l_coef,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_u_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%c_u_start,             &
             data%dims%c_u_end, data%Y_u_coef,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_l_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%x_free + 1,            &
             data%dims%x_l_end, data%Z_l_coef,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_u_coef'
      CALL SPACE_resize_array( 0, data%order, data%dims%x_u_start,             &
             prob%n, data%Z_u_coef,                                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%list_hist'
      CALL SPACE_resize_array( data%hist, data%list_hist,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%mu_hist'
      CALL SPACE_resize_array( data%hist, data%mu_hist,                        &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%X_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, prob%n, data%X_hist,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%C_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%c_l_start,  &
             data%dims%c_u_end, data%C_hist,                                   &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, prob%m, data%Y_hist,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_l_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%c_l_start,  &
             data%dims%c_l_end, data%Y_l_hist,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Y_u_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%c_u_start,  &
             data%dims%c_u_end, data%Y_u_hist,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_l_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%x_free + 1, &
             data%dims%x_l_end, data%Z_l_hist,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpb: data%Z_u_hist'
      CALL SPACE_resize_array( data%hist, 0, data%deriv, data%dims%x_u_start,  &
             prob%n, data%Z_u_hist,                                            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      n_sbls =  prob%n + data%dims%nc

!  H will be in coordinate form

      CALL SMT_put( data%H_sbls%type, 'COORDINATE', inform%alloc_status )
      data%H_sbls%n = n_sbls
      data%H_sbls%m = n_sbls
      data%H_sbls%ne = data%h_ne + n_sbls

!  allocate space for H

      array_name = 'lsqp: data%H_sbls%row'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%H_sbls%col'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%H_sbls%val'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the components of the barrier terms in coordinate form ...

      DO i = 1, n_sbls
        data%H_sbls%row( i ) = i ; data%H_sbls%col( i ) = i
      END DO

!  ... and add the components of H

      DO i = 1, prob%n
        data%H_sbls%row( n_sbls + prob%H%ptr( i ) :                            &
                         n_sbls + prob%H%ptr( i + 1 ) - 1 ) = i
      END DO
      data%H_sbls%col( n_sbls + 1 : n_sbls + data%h_ne ) =                     &
        prob%H%col( : data%h_ne )
      data%H_sbls%val( n_sbls + 1 : n_sbls + data%h_ne ) =                     &
        prob%H%val( : data%h_ne )

!  A will be in coordinate form

      CALL SMT_put( data%A_sbls%type, 'COORDINATE', inform%alloc_status )
      data%A_sbls%n = n_sbls
      data%A_sbls%m = prob%m
      data%A_sbls%ne = data%a_ne + data%dims%nc

!  allocate space for A

      array_name = 'lsqp: data%A_sbls%row'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%A_sbls%col'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%A_sbls%val'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the components of A in coordinate form ...

      DO i = 1, prob%m
        data%A_sbls%row( prob%A%ptr( i ) : prob%A%ptr( i + 1 ) - 1 ) = i
      END DO
      data%A_sbls%col( : data%a_ne ) = prob%A%col( : data%a_ne )
      data%A_sbls%val( : data%a_ne ) = prob%A%val( : data%a_ne )

!  ... and include the coodinates corresponding to the slack variables

      DO i = 1, data%dims%nc
        data%A_sbls%row( data%a_ne + i ) = data%dims%c_equality + i
        data%A_sbls%col( data%a_ne + i ) = prob%n + i
      END DO

!  H will be in coordinate form

      CALL SMT_put( data%H_sbls%type, 'COORDINATE', inform%alloc_status )
      data%H_sbls%n = n_sbls
      data%H_sbls%m = n_sbls
      data%H_sbls%ne = data%h_ne + n_sbls

!  allocate space for H

      array_name = 'lsqp: data%H_sbls%row'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%H_sbls%col'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%H_sbls%val'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the components of the barrier terms in coordinate form ...

      DO i = 1, n_sbls
        data%H_sbls%row( i ) = i ; data%H_sbls%col( i ) = i
      END DO

!  ... and add the components of H

      DO i = 1, prob%n
        data%H_sbls%row( n_sbls + prob%H%ptr( i ) :                            &
                         n_sbls + prob%H%ptr( i + 1 ) - 1 ) = i
      END DO
      data%H_sbls%col( n_sbls + 1 : n_sbls + data%h_ne ) =                     &
        prob%H%col( : data%h_ne )
      data%H_sbls%val( n_sbls + 1 : n_sbls + data%h_ne ) =                     &
        prob%H%val( : data%h_ne )

!  A will be in coordinate form

      CALL SMT_put( data%A_sbls%type, 'COORDINATE', inform%alloc_status )
      data%A_sbls%n = n_sbls
      data%A_sbls%m = prob%m
      data%A_sbls%ne = data%a_ne + data%dims%nc

!  allocate space for A

      array_name = 'lsqp: data%A_sbls%row'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%A_sbls%col'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'lsqp: data%A_sbls%val'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  set the components of A in coordinate form ...

      DO i = 1, prob%m
        data%A_sbls%row( prob%A%ptr( i ) : prob%A%ptr( i + 1 ) - 1 ) = i
      END DO
      data%A_sbls%col( : data%a_ne ) = prob%A%col( : data%a_ne )
      data%A_sbls%val( : data%a_ne ) = prob%A%val( : data%a_ne )

!  ... and include the coodinates corresponding to the slack variables

      DO i = 1, data%dims%nc
        data%A_sbls%row( data%a_ne + i ) = data%dims%c_equality + i
        data%A_sbls%col( data%a_ne + i ) = prob%n + i
      END DO

!  the zero matrix C will be in zero form

      CALL SMT_put( data%C_sbls%type, 'ZERO', inform%alloc_status )

!  ===============================
!  Approximately solve the problem
!  ===============================

!  overlaps: DY_l => DIST_C_l_trial
!            DY_u => DIST_C_u_trial
!            DELTA => VECTOR
!            RHS( : c_e ) => R
!            DZ_l( x_l_start : x_l_end ) => DIST_X_l_trial
!            DZ_u( x_u_start : x_u_end ) => DIST_X_u_trial

      QPB_control%LSQP_control%indicator_type =                                &
        QPB_control%indicator_type
      QPB_control%LSQP_control%indicator_tol_p =                               &
        QPB_control%indicator_tol_p
      QPB_control%LSQP_control%indicator_tol_pd =                              &
        QPB_control%indicator_tol_pd
      QPB_control%LSQP_control%indicator_tol_tapia =                           &
        QPB_control%indicator_tol_tapia

      data%SBLS_control = QPB_control%SBLS_control
      data%SBLS_control%get_norm_residual = .TRUE.
      data%GLTR_control = QPB_control%GLTR_control

      IF ( printi ) WRITE( control%out, "( /, A, ' entering QPB ' )" ) prefix

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL QPB_solve_main( data%dims, prob%n, prob%m,                          &
                           prob%H%val, prob%H%col, prob%H%ptr,                 &
                           prob%G, prob%f, prob%A%val, prob%A%col,             &
                           prob%A%ptr, prob%C_l, prob%C_u, prob%X_l,           &
                           prob%X_u, prob%C, prob%X, prob%Y, prob%Z,           &
                           data%X_trial, data%Y_trial,                         &
                           data%HX, data%GRAD_L, data%DIST_X_l,                &
                           data%DIST_X_u, data%Z_l, data%Z_u,                  &
                           data%BARRIER_X, data%Y_l, data%DY_l,                &
                           data%DIST_C_l, data%Y_u, data%DY_u,                 &
                           data%DIST_C_u, data%C, data%BARRIER_C,              &
                           data%SCALE_C, data%DELTA,                           &
                           data%H_sbls, data%A_sbls, data%C_sbls,              &
                           data%RHS( : data%dims%v_e ),                        &
                           data%GRAD, data%GRAD_X_phi, data%GRAD_C_phi,        &
!                           data%DZ_l( data%dims%x_l_start :                   &
!                                      data%dims%x_l_end ),                    &
!                           data%DZ_u( data%dims%x_u_start :                   &
!                                     data%dims%x_u_end ), data%S,             &
                           data%DZ_l, data%DZ_u, data%S,                       &
                           data%hist, data%deriv, data%order, data%len_hist,   &
                           data%BINOMIAL, data%fit_mu, data%fit_f,             &
                           data%X_coef, data%C_coef, data%Y_coef,              &
                           data%Y_l_coef, data%Y_u_coef,                       &
                           data%Z_l_coef, data%Z_u_coef,                       &
                           data%list_hist, data%mu_hist,                       &
                           data%X_hist, data%C_hist, data%Y_hist,              &
                           data%Y_l_hist, data%Y_u_hist,                       &
                           data%Z_l_hist, data%Z_u_hist,                       &
                           data%SBLS_data, data%GLTR_data, data%FIT_data,      &
                           qpb_prefix, QPB_control, inform%QPB_inform,         &
                           data%SBLS_control, data%GLTR_control,               &
                           C_last = data%A_s, X_last = data%H_s,               &
                           Y_last = data%Y_last, Z_last = data%Z_last,         &
                           C_stat = C_stat( : prob%m ),                        &
                           B_stat = B_stat( : prob%n ) )

!    write(6,*) '     x_l          x            x_u            z'
!    do i = 1, prob%n
!    write(6,"(4ES12.4)" ) prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i)
!    end do

!    write(6,*) '     c_l          c            c_u            y'
!    do i = 1, prob%m
!    write(6,"(4ES12.4)" ) prob%C_l( i ), prob%C( i ), prob%C_u( i ), prob%Y( i)
!    end do

!  Record times for components of QPB

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      time_now = time_now - time_record ; clock_now = clock_now - clock_record
      inform%QPB_inform%time%total = inform%QPB_inform%time%total + time_now
      inform%QPB_inform%time%clock_total =                                     &
        inform%QPB_inform%time%clock_total + clock_now

      IF ( printd ) THEN
       WRITE( control%out, "( A, ' C_stat = ', /, ( 8I8 ) )" ) prefix,         &
         C_stat( : prob%m )
       WRITE( control%out, "( A, ' B_stat = ', /, ( 8I8 ) )" ) prefix,         &
         B_stat( : prob%n )
      END IF

      IF ( printi ) THEN
        IF ( QPB_control%out > 0 .AND. QPB_control%print_level > 0 )           &
          WRITE( control%out, "( /, A, ' returned from QPB' )" ) prefix
        WRITE( control%out, "( A, ' on exit from QPB: status = ', I0,          &
       &   ', iterations = ', I0, ', time = ', F0.2 )" ) prefix,               &
          inform%QPB_inform%status, inform%QPB_inform%iter, clock_now
        WRITE( control%out, "( A, ' objective value =', ES12.4,                &
       &   /, A, ' # active counstraints: ', I0, ' from ', I0,                 &
       &   ', bounds: ', I0, ' from ', I0 )" ) prefix, inform%QPB_inform%obj,  &
            prefix, COUNT( C_stat( : prob%m ) /= 0 ),  prob%m,                 &
            COUNT( B_stat( : prob%n ) /= 0 ),  prob%n
      END IF

!write(6,*) COUNT( ABS( C_stat( : prob%m ) ) == 2 ), ' degen constraints &',   &
!           COUNT( ABS( B_stat( : prob%m ) ) == 2 ),  ' degen bounds '

!  Record output information from QPB

      inform%status = inform%QPB_inform%status
      inform%alloc_status = inform%QPB_inform%alloc_status
      inform%nfacts = inform%nfacts + inform%QPB_inform%nfacts
      inform%nmods = inform%nmods + inform%QPB_inform%nmods
      inform%obj = inform%QPB_inform%obj

      IF ( printi )                                                            &
        WRITE( control%out, "( A, ' ', I0, ' integer and ', I0,                &
      &  ' real words required for factors in QPB' )" ) prefix,                &
         inform%QPB_inform%factorization_integer,                              &
         inform%QPB_inform%factorization_real

!  Check for error exits

      IF ( inform%status /= GALAHAD_ok .AND.                                   &
           inform%status /= GALAHAD_error_ill_conditioned .AND.                &
           inform%status /= GALAHAD_error_tiny_step ) THEN
!          inform%status /= GALAHAD_error_max_iterations ) THEN
        GO TO 700
      END IF

!  Check to see if crossover is required

      IF ( control%qpb_or_qpa .AND. inform%status == GALAHAD_ok ) GO TO 700
      IF ( .NOT. control%qpb_or_qpa .AND. control%no_qpa ) GO TO 700

!     IF ( control%no_qpa .AND.                                                &
!          inform%status /= GALAHAD_error_ill_conditioned .AND.                &
!          inform%status /= GALAHAD_error_tiny_step ) GO TO 700

!  Save the solution in case QPA fails

      array_name = 'qpc: data%X_trial'
      CALL SPACE_resize_array( prob%n, data%X_trial, inform%status,            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%Y_last'
      CALL SPACE_resize_array( prob%m, data%Y_last, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%Z_last'
      CALL SPACE_resize_array( prob%n, data%Z_last, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%C'
      CALL SPACE_resize_array( 1, prob%m, data%C, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      gotsol = .TRUE.
      data%X_trial( : prob%n ) = prob%X( : prob%n )
      data%Y_last( : prob%m ) = prob%Y( : prob%m )
      data%Z_last( : prob%n ) = prob%Z( : prob%n )
      data%C( 1 : prob%m ) = prob%C( 1 : prob%m )
      best_obj = inform%obj

!  if the problem is convex, crossover to a basic solution

  150 CONTINUE
!     IF ( .FALSE. ) then
      IF ( cqp ) then
        IF ( printd ) THEN
          WRITE( control%out, "( /, A, '      i       X_l             X   ',   &
         &   '          X_u            Z        st' )" ) prefix
          DO i = 1, prob%n
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i ), B_stat( i )
          END DO

          WRITE( control%out, "( /, A, '      i       C_l             C   ',   &
         &   '          C_u            Y        st' )" ) prefix
          DO i = 1, prob%m
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%C_l( i ), prob%C( i ), prob%C_u( i ), prob%Y( i ), C_stat( i )
          END DO
        END IF
        CALL CRO_crossover( prob%n, prob%m, data%dims%c_equality, prob%H%val,  &
                            prob%H%col, prob%H%ptr, prob%A%val,                &
                            prob%A%col, prob%A%ptr, prob%G, prob%C_l,          &
                            prob%C_u, prob%X_l, prob%X_u, prob%C, prob%X,      &
                            prob%Y, prob%Z, C_stat, B_stat, data%CRO_data,     &
                            control%CRO_control, inform%CRO_inform )

        IF ( printd ) THEN
          WRITE( control%out, "( /, A, '      i       X_l             X   ',   &
         &   '          X_u            Z        st' )" ) prefix
          DO i = 1, prob%n
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i ), B_stat( i )
          END DO

          WRITE( control%out, "( /, A, '      i       C_l             C   ',   &
         &   '          C_u            Y        st' )" ) prefix
          DO i = 1, prob%m
            WRITE( control%out, "( A, I7, 4ES15.7, I3 )" ) prefix, i,          &
            prob%C_l( i ), prob%C( i ), prob%C_u( i ), prob%Y( i ), C_stat( i )
          END DO
        END IF

!write(6,"( ' x ', /, ( 5ES12.4 ) )" )  prob%X( : prob%n )
!write(6,"( ' y ', /, ( 5ES12.4 ) )" )  prob%Y( : prob%m )
!write(6,"( ' z ', /, ( 5ES12.4 ) )" )  prob%Z( : prob%n )

!  flag redundant constraints as inactive

        DO i = 1, prob%n
          IF ( ABS( B_stat( i ) ) /= 1 ) B_stat( i ) = 0
        END DO
        DO i = 1, prob%m
          IF ( ABS( C_stat( i ) ) /= 1 ) C_stat( i ) = 0
        END DO

!stop
      END IF

!     OPEN( 56 )
!     WRITE( 56, "( '    i st    x - x_l       x_u - x        z ' )" )
!     DO i = 1, prob%n
!       IF ( B_stat( i ) /= 0 ) THEN
!         WRITE( 56, "( i6, i3, 3ES12.4 )" ) i, B_stat( i ),                   &
!           prob%X( i ) - prob%X_l( i ), prob%X_u( i ) - prob%X( i ), prob%Z( i)
!       END IF
!     END DO

!     WRITE( 56, "( '    i st    c - c_l       c_u - c        y ' )" )
!     DO i = 1, prob%m
!       IF ( C_stat( i ) /= 0 ) THEN
!         WRITE( 56, "( i6, i3, 3ES12.4 )" ) i, C_stat( i ),                   &
!           prob%C( i ) - prob%C_l( i ), prob%C_u( i ) - prob%C( i ), prob%Y( i)
!       END IF
!     END DO
!     CLOSE( 56 )

!  Store the current constraint statii for later checks

      array_name = 'qpc: data%IW'
      CALL SPACE_resize_array( prob%m + prob%n, data%IW, inform%status,        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      data%IW( : prob%m ) = C_stat( : prob%m )
      data%IW( prob%m + 1 : prob%m + prob%n ) = B_stat( : prob%n )

!  ===========================================================================
!  Check to see if the constraints in the working set are linearly independent
!  ===========================================================================

!  Allocate workspace arrays

      lbreak = prob%m + data%dims%c_l_end - data%dims%c_u_start +              &
               prob%n - data%dims%x_free + data%dims%x_l_end -                 &
               data%dims%x_u_start + 2

      array_name = 'qpc: data%IBREAK'
      CALL SPACE_resize_array( lbreak, data%IBREAK, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%RES_l'
      CALL SPACE_resize_array( 1, data%dims%c_l_end,                           &
             data%RES_l, inform%status,                                        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%RES_u'
      CALL SPACE_resize_array( data%dims%c_u_start, prob%m,                    &
             data%RES_u, inform%status,                                        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%A_norms'
      CALL SPACE_resize_array( prob%m, data%A_norms, inform%status,            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%H_s'
      CALL SPACE_resize_array( prob%n, data%H_s, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%X_trial'
      CALL SPACE_resize_array( prob%n, data%X_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%Y_trial'
      CALL SPACE_resize_array( prob%m, data%Y_trial,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%C'
      CALL SPACE_resize_array( 1, prob%m, data%C, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Compute the initial residuals

      DO i = 1, data%dims%c_u_start - 1
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'l', i, a_x, prob%C_l( i )
        data%RES_l( i ) = a_x - prob%C_l( i )
        data%A_norms( i ) = SQRT( a_norms )
      END DO

      DO i = data%dims%c_u_start, data%dims%c_l_end
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'l', i, a_x, prob%C_l( i )
!       write(6,*) 'u', i, a_x, prob%C_u( i )
        data%RES_l( i ) = a_x - prob%C_l( i )
        data%RES_u( i ) = prob%C_u( i ) - a_x
        data%A_norms( i ) = SQRT( a_norms )
      END DO

      DO i = data%dims%c_l_end + 1, prob%m
        a_norms = zero ; a_x = zero
        DO ii = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
          a_x = a_x + prob%A%val( ii ) * prob%X( prob%A%col( ii ) )
          a_norms = a_norms + prob%A%val( ii ) ** 2
        END DO
!       write(6,*) 'u', i, a_x, prob%C_u( i )
        data%RES_u( i ) = prob%C_u( i ) - a_x
        data%A_norms( i ) = SQRT( a_norms )
      END DO

!  If necessary, determine which constraints occur in the reference set

      IF ( control%qpb_or_qpa .OR. control%no_qpb ) THEN

!  cold start from the value set in X; constraints active
!  at X will determine the initial working set.

        IF ( QPA_control%cold_start == 1 ) THEN

!  constraints with lower bounds

          DO i = 1, data%dims%c_u_start - 1
            IF ( ABS( data%RES_l( i ) ) <= teneps ) THEN
!             write(6,*) i, data%RES_l( i )
              C_stat( i ) = - 1
            ELSE
              C_stat( i ) = 0
            END IF
          END DO

!  constraints with both lower and upper bounds

          DO i = data%dims%c_u_start, data%dims%c_l_end
            IF ( ABS( data%RES_l( i ) ) <= teneps ) THEN
!             write(6,*) i, data%RES_l( i )
              C_stat( i ) = - 1
            ELSE IF ( ABS( data%RES_u( i ) ) <= teneps ) THEN
!             write(6,*) i, data%RES_u( i )
              C_stat( i ) = 1
            ELSE
              C_stat( i ) = 0
            END IF
          END DO

!  constraints with upper bounds

          DO i = data%dims%c_l_end + 1, prob%m
            IF ( ABS( data%RES_u( i ) ) <= teneps ) THEN
!             write(6,*) i, data%RES_u( i )
              C_stat( i ) = 1
            ELSE
              C_stat( i ) = 0
            END IF
          END DO

!  free variables

          B_stat( : data%dims%x_free ) = 0

!  simple non-negativity

          DO i = data%dims%x_free + 1, data%dims%x_l_start - 1
            IF ( ABS( prob%X( i ) ) <= teneps ) THEN
!             write(6,*) prob%m + i, prob%X( i )
              B_stat( i ) = - 1
            ELSE
              B_stat( i ) = 0
            END IF
          END DO

!  simple bound from below

          DO i = data%dims%x_l_start, data%dims%x_u_start - 1
            IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= teneps ) THEN
!             write(6,*) prob%m + i, prob%X( i ) - prob%X_l( i )
              B_stat( i ) = - 1
            ELSE
              B_stat( i ) = 0
            END IF
          END DO

!  simple bound from below and above

          DO i = data%dims%x_u_start, data%dims%x_l_end
            IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= teneps ) THEN
!             write(6,*) prob%m + i, prob%X( i ) - prob%X_l( i )
              B_stat( i ) = - 1
            ELSE IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= teneps ) THEN
!             write(6,*) prob%m + i, prob%X( i ) - prob%X_u( i )
              B_stat( i ) = 1
            ELSE
              B_stat( i ) = 0
            END IF
          END DO

!  simple bound from above

          DO i = data%dims%x_l_end + 1, data%dims%x_u_end
            IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= teneps ) THEN
!             write(6,*) prob%m + i, prob%X( i ) - prob%X_u( i )
              B_stat( i ) = 1
            ELSE
              B_stat( i ) = 0
            END IF
          END DO

!  simple non-positivity

          DO i = data%dims%x_u_end + 1, prob%n
            IF ( ABS( prob%X( i ) ) <= teneps ) THEN
!             write(6,*) prob%m + i, prob%X( i )
              B_stat( i ) = 1
            ELSE
              B_stat( i ) = 0
            END IF
          END DO

!  cold start with only equality constraints active

        ELSE IF ( QPA_control%cold_start == 3 ) THEN
          B_stat = 0
          C_stat( : MIN( data%dims%c_equality, prob%n ) ) = 1
          C_stat( MIN( data%dims%c_equality, prob%n ) + 1 : ) = 0

!  cold start with as many active constraints as possible

        ELSE IF ( QPA_control%cold_start == 4 ) THEN
          B_stat = 0 ; C_stat = 0
          l = 0

!  equality constraints

          DO i = 1,  data%dims%c_equality
            IF ( l > prob%n ) EXIT
            C_stat( i ) = - 1
            l = l + 1
          END DO

!  simple bound from below

          DO i = data%dims%x_free + 1, data%dims%x_l_end
            IF ( l > prob%n ) EXIT
            B_stat( i ) = - 1
            l = l + 1
          END DO

!  simple bound from above

          DO i = data%dims%x_l_end + 1, data%dims%x_u_end
            IF ( l > prob%n ) EXIT
            B_stat( i ) = 1
            l = l + 1
          END DO

!  constraints with lower bounds

          DO i = data%dims%c_equality + 1, data%dims%c_l_end
            IF ( l > prob%n ) EXIT
            C_stat( i ) = - 1
            l = l + 1
          END DO

!  constraints with upper bounds

          DO i = data%dims%c_l_end + 1, prob%m
            IF ( l > prob%n ) EXIT
            C_stat( i ) = 1
            l = l + 1
          END DO

!  cold start with no active constraints

        ELSE IF ( QPA_control%cold_start /= 0 ) THEN
          B_stat = 0 ; C_stat = 0
        END IF
      END IF

      IF ( printm ) THEN
        WRITE( control%out, "(' b_stat ', /, ( 10I5 ) )" ) B_stat( : prob%n )
        WRITE( control%out, "(' c_stat ', /, ( 10I5 ) )" ) C_stat( : prob%m )
      END IF

      inform%QPA_inform%obj = inform%obj
      data%SLS_control = QPA_control%SLS_control

!  Assign appropriate values to the fixed constraints

      DO i = 1, prob%m
        IF ( C_stat( i ) < 0 ) THEN
          data%Y_trial( i ) = prob%C_l( i )
        ELSE IF ( C_stat( i ) > 0 ) THEN
          data%Y_trial( i ) = prob%C_u( i )
        ELSE
          data%Y_trial( i ) = zero
        END IF
      END DO

      DO i = 1, prob%n
        IF ( B_stat( i ) < 0 ) THEN
          data%X_trial( i ) = prob%X_l( i )
        ELSE IF ( B_stat( i ) > 0 ) THEN
          data%X_trial( i ) = prob%X_u( i )
        ELSE
          data%X_trial( i ) = zero
        END IF
      END DO

!  Remove any dependent working constraints, checking for consistency

      CALL CPU_TIME( time_record )  ; CALL CLOCK_time( clock_record )
      CALL QPA_remove_dependent( prob%n, prob%m, prob%A%val, prob%A%col,       &
                                 prob%A%ptr, data%K, data%SLS_data,            &
                                 data%SLS_control, C_stat, B_stat,             &
                                 data%IBREAK, data%P, data%SOL, data%D,        &
                                 qpa_prefix, QPA_control,                      &
                                 inform%QPA_inform, n_appear_depen,            &
                                 C_given = data%Y_trial,                       &
                                 X_given = data%X_trial )
      CALL CPU_TIME( time_now )  ; CALL CLOCK_time( clock_now )
      inform%time%preprocess = inform%time%preprocess + time_now - time_record
      inform%time%clock_preprocess =                                           &
        inform%time%clock_preprocess + clock_now - clock_record

!  Check for error exits

      inform%status = inform%QPA_inform%status
      IF ( inform%status /= GALAHAD_ok ) THEN

!  On error exit, compute the current objective function value

        data%H_s( : prob%n ) = zero
        CALL QPC_HX( data%dims, prob%n, data%H_s( : prob%n ),                  &
                     prob%H%ptr( prob%n + 1 ) - 1, prob%H%val,                 &
                     prob%H%col, prob%H%ptr, prob%X( : prob%n ), '+' )
        inform%obj = half * DOT_PRODUCT( prob%X( : prob%n ),                   &
                                         data%H_s( : prob%n ) )                &
                          + DOT_PRODUCT( prob%X( : prob%n ),                   &
                                         prob%G( : prob%n ) ) + prob%f

!  Print details of the error exit

        IF ( control%error > 0 .AND. control%print_level >= 1 ) THEN
          WRITE( control%out, "( ' ' )" )
          IF ( inform%status /= GALAHAD_ok ) WRITE( control%error, 2040 )      &
            inform%status, 'QPA_remove_dependent'
        END IF
        GO TO 700

!       IF ( control%out > 0 .AND. control%print_level >= 2 .AND.              &
!            n_appear_depen > 0 )                                              &
!       WRITE( control%out, "(/, ' The following ',I7,' constraints appear',   &
!      &       ' to be dependent', /, ( 8I8 ) )" ) n_appear_depen,             &
!                data%Index_C_freed

      END IF
      IF ( control%out > 0 .AND. control%print_level >= 1 .AND.                &
           n_appear_depen > 0 )                                                &
        WRITE( control%out, "(/, A, ' ', I0,' active constraints appear to be',&
       & ' dependent and will be ignored', /, A, ' => # active constraints: ', &
       & I0, ' from ', I0, ', bounds: ', I0, ' from ', I0 )" )                 &
          prefix, n_appear_depen, prefix, COUNT( C_stat( : prob%m ) /= 0 ),    &
          prob%m, COUNT( B_stat( : prob%n ) /= 0 ),  prob%n

!  ===========================================================================
!  Now try to see if the predicted working set is an optimal one by
!  solving the corresponding equality-constrained qp
!  ===========================================================================

!  Continue allocating workspace arrays

      m_link = prob%m + prob%n - data%dims%x_free
      k_n_max = prob%n + m_link

      array_name = 'qpc: data%A_s'
      CALL SPACE_resize_array( prob%m, data%A_s,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%H_s'
      CALL SPACE_resize_array( prob%n, data%H_s,                               &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%VECTOR'
      CALL SPACE_resize_array( MAX( prob%n, k_n_max ), data%VECTOR,            &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%P'
      CALL SPACE_resize_array( prob%n, data%P,                                 &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%GRAD'
      CALL SPACE_resize_array( prob%n, data%GRAD,                              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!write(6,*) 'x', prob%X( : prob%n )
!write(6,*) 'y', prob%Y( : proB%m )
!write(6,*) 'z', prob%Z( : prob%n )

!  Order the free variables

      data%H_sbls%n = 0
      DO i = 1, prob%n
        IF ( B_stat( i ) == 0 ) THEN
          data%H_sbls%n = data%H_sbls%n + 1
          data%P( i ) = data%H_sbls%n
        ELSE
          data%P( i ) = 0
        END IF
      END DO

!  Set up the data for the EQP:

      CALL SMT_put( data%A_sbls%type, 'COORDINATE', inform%alloc_status )
      CALL SMT_put( data%H_sbls%type, 'COORDINATE', inform%alloc_status )

!  the number of variables ....

      data%A_sbls%n = data%H_sbls%n

!  ... the number of constraints ...

      data%A_sbls%m = COUNT( C_stat( : prob%m ) /= 0 )
!write(6,*) ' EQP: n, m = ', data%A_sbls%n, data%A_sbls%m

!  ... the constant term from the objective ...

      f = prob%f

!  ... the linear term from the objective ...

      ii = 0 ; ll = 0
      DO i = 1, prob%n
        IF ( B_stat( i ) == 0 ) THEN
          ii = ii + 1
          data%GRAD( ii ) = prob%G( i )
          data%H_s( ii ) = prob%X( i )
        ELSE

!  update the constant term from the objective to account for the fixed gradient

          f = f + prob%G( i ) * data%X_trial( i )
        END IF
      END DO

!  ... the Hessian ...

!  find the number of nonzeros in the free-variable Hessian

      data%H_sbls%ne = 0
      DO i = 1, prob%n
        IF ( B_stat( i ) == 0 ) THEN
          DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            IF ( B_stat( prob%H%col( l ) ) == 0 )                              &
              data%H_sbls%ne = data%H_sbls%ne + 1
          END DO
        END IF
      END DO

!  allocate space to hold the free-variable Hessian

      array_name = 'qpc: data%H_sbls%row'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%H_sbls%col'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%H_sbls%val'
      CALL SPACE_resize_array( data%H_sbls%ne, data%H_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  transfer the components to the free-variable Hessian

      ii = 0 ; ll = 0
      DO i = 1, prob%n
        IF ( B_stat( i ) == 0 ) THEN
          ii = ii + 1
          DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            j = prob%H%col( l )
            jj = data%P( j )
            IF ( jj > 0 ) THEN
              ll = ll + 1
              IF ( ii >= jj ) THEN
                data%H_sbls%row( ll ) = ii ; data%H_sbls%col( ll ) = jj
              ELSE
                data%H_sbls%row( ll ) = jj ; data%H_sbls%col( ll ) = ii
              END IF
              data%H_sbls%val( ll ) = prob%H%val( l )

!  update the gradient term from the objective to account for the fixed Hessian

            ELSE
              data%GRAD( ii ) =  data%GRAD( ii ) +                             &
                prob%H%val( l ) * data%X_trial( j )
            END IF
          END DO
        ELSE
          DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            j = prob%H%col( l )
            jj = data%P( j )

!  update the gradient term from the objective to account for the fixed Hessian

            IF ( jj > 0 ) THEN
              data%GRAD( jj ) =  data%GRAD( jj ) +                             &
                prob%H%val( l ) * data%X_trial( i )
            ELSE

!  update the constant term from the objective to account for the fixed Hessian

              IF ( i /= j ) THEN
                f = f + prob%H%val( l ) * data%X_trial( i ) * data%X_trial( j )
              ELSE
                f = f + half * prob%H%val( l ) *                               &
                      data%X_trial( i ) * data%X_trial( j )
              END IF
            END IF
          END DO
        END IF
      END DO

!  ... the constant term from the constraints

      ii = 0
      DO i = 1, prob%m
        IF ( C_stat( i ) /= 0 ) THEN
          ii = ii + 1
          data%C( ii ) = - data%Y_trial( i )
          data%A_s( ii ) = prob%Y( i )
        END IF
      END DO

!  ... and the constraint Jacobian ...

!  find the number of nonzeros in the free-variable active-constraint Jacobian

      data%A_sbls%ne = 0
      DO i = 1, prob%m
        IF ( C_stat( i ) /= 0 ) THEN
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            IF ( B_stat( prob%A%col( l ) ) == 0 )                              &
              data%A_sbls%ne = data%A_sbls%ne + 1
          END DO
        END IF
      END DO

!  allocate space to hold the free-variable active-constraint Jacobian

      array_name = 'qpc: data%A_sbls%row'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%row,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%A_sbls%col'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%col,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%A_sbls%val'
      CALL SPACE_resize_array( data%A_sbls%ne, data%A_sbls%val,                &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  transfer the components to the free-variable active-constraint Jacobian

      ii = 0 ; ll = 0
      DO i = 1, prob%m
        IF ( C_stat( i ) /= 0 ) THEN
          ii = ii + 1
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            j = prob%A%col( l )
            jj = data%P( j )
            IF ( jj > 0 ) THEN
              ll = ll + 1
              data%A_sbls%row( ll ) = ii ; data%A_sbls%col( ll ) = jj
              data%A_sbls%val( ll ) = prob%A%val( l )

!  update the constraint constant term to account for the fixed Jacobian

            ELSE
              data%C( ii ) = data%C( ii ) + prob%A%val( l ) * data%X_trial( j )
            END IF
          END DO
        END IF
      END DO

!  solve the EQP

      data%SBLS_data%last_preconditioner = - 1000
      data%new_c = .TRUE.

      EQP_control = control%EQP_control
      EQP_control%GLTR_control%steihaug_toint  = .TRUE.
      IF ( printi ) WRITE( control%out, "( /, A, ' entering EQP ' )" ) prefix
      CALL EQP_solve_main( data%H_sbls%n, data%A_sbls%m,                       &
                           data%H_sbls, data%GRAD( : data%H_sbls%n ), f,       &
                           data%A_sbls, q, data%H_s( : data%H_sbls%n ),        &
                           data%A_s( : data%A_sbls%m ),                        &
                           data, EQP_control, inform%EQP_inform,               &
                           C = data%C( : data%A_sbls%m ) )
      IF ( printi ) THEN
        IF ( control%EQP_control%out > 0 .AND.                                 &
             control%EQP_control%print_level > 0 )                             &
          WRITE( control%out, "( /, A, ' returned from EQP' )" ) prefix
        WRITE( control%out, "( A, ' on exit from EQP: status = ', I0 )" )      &
          prefix, inform%EQP_inform%status
      END IF

!  if the constraints appear to be infeasible, quit

      IF ( inform%EQP_inform%status == GALAHAD_error_primal_infeasible ) THEN
        inform%status = inform%EQP_inform%status
        GO TO 700

!  if the eqp solution succeeded, map the solution onto the entire problem

      ELSE IF ( inform%EQP_inform%status == 0 ) THEN

!  spread the free variables through x

        ii = 0
        DO i = 1, prob%n
          IF ( B_stat( i ) == 0 ) THEN
            ii = ii + 1
            data%X_trial( i ) = data%H_s( ii )
          END IF
        END DO

!write(6,"( ' x ', /, ( 5ES12.4 ) )" )  data%X_trial( : prob%n )

!  spread the non-zero multipliers through y

        ii = 0
        DO i = 1, prob%m
          IF ( C_stat( i ) /= 0 ) THEN
            ii = ii + 1
            data%Y_trial( i ) = data%A_s( ii )
          ELSE
            data%Y_trial( i ) = zero
          END IF
        END DO

!write(6,"( ' y ', /, ( 5ES12.4 ) )" )  data%Y_trial( : prob%m )

!  recover the dual variables for the active bounds

!   z = g ...

        DO i = 1, prob%n
          IF ( B_stat( i ) == 0 ) THEN
            data%H_s( i ) = zero
          ELSE
            data%H_s( i ) = prob%G( i )
          END IF
        END DO

!  ... + H x ...

        DO i = 1, prob%n
          ii = data%P( i )
          DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
            j = prob%H%col( l )
            jj = data%P( j )
            IF ( ii == 0 .AND. jj == 0 ) THEN
              data%H_s( i ) =                                                  &
                data%H_s( i ) + prob%H%val( l ) * data%X_trial( j )
              IF ( i /= j ) data%H_s( j ) =                                    &
                data%H_s( j ) + prob%H%val( l ) * data%X_trial( i )
            ELSE IF ( ii == 0 ) THEN
              data%H_s( i ) =                                                  &
                data%H_s( i ) + prob%H%val( l ) * data%X_trial( j )
            ELSE IF ( jj == 0 ) THEN
              data%H_s( j ) =                                                  &
                data%H_s( j ) + prob%H%val( l ) * data%X_trial( i )
            END IF
          END DO
        END DO

!  ... - A^T y

        ii = 0
        DO i = 1, prob%m
          IF ( C_stat( i ) /= 0 ) THEN
            ii = ii + 1
            DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
              j = prob%A%col( l )
              IF ( B_stat( j ) /= 0 ) data%H_s( j ) =                          &
                data%H_s( j ) - prob%A%val( l ) * data%Y_trial( i )
            END DO
          END IF
        END DO

!  compute the values of the constraints Ax

        DO i = 1, prob%m
          data%C( i ) = zero
          DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
            data%C( i ) =                                                      &
              data%C( i ) + prob%A%val( l ) *  data%X_trial( prob%A%col( l ) )
          END DO
        END DO

!write(6,"( ' z ', /, ( 5ES12.4 ) )" )  data%H_s( : prob%n )

!       write(6,*) '     x_l          x            x_u            z'
!       do i = 1, prob%n
!       write(6,"(4ES12.4)" )                                                  &
!         prob%X_l( i ), data%X_trial( i ), prob%X_u( i ), data%H_s( i )
!       end do

!       write(6,*) '     c_l          c            c_u            y'
!       do i = 1, prob%m
!       write(6,"(4ES12.4)" )                                                  &
!         prob%C_l( i ), data%C( i ), prob%C_u( i ), data%Y_trial( i )
!       end do

!  Check to see if we are optimal

        tol = QPB_control%stop_p
        l = 0
        viol8 = zero
        DO i = 1, prob%n
          IF ( data%X_trial( i ) <= prob%X_l( i ) - tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%X_trial( i ) - prob%X_l( i ) ) )
!write( control%out, "( ' i, x, xl ', I7, 2ES10.2 )" )              &
!  i, data%X_trial( i ), prob%X_l( i )
          END IF
          IF ( data%X_trial( i ) >= prob%X_u( i ) + tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%X_trial( i ) - prob%X_u( i ) ) )
!write( control%out, "( ' i, x, xu ', I7, 2ES10.2 )" )              &
!  i, data%X_trial( i ), prob%X_u( i )
          END IF
          IF ( ABS( prob%X_u( i ) - prob%X_l( i )) <= tol ) CYCLE
          IF ( ABS( data%X_trial( i ) - prob%X_l( i ) ) <= tol .AND.           &
               data%H_s( i ) < - tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%H_s( i ) ) )
!write( control%out, "( ' i, zl ', I7, ES10.2 )" ) i, data%H_s( i )
          END IF
          IF ( ABS( data%X_trial( i ) - prob%X_u( i ) ) <= tol .AND.           &
               data%H_s( i ) > tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%H_s( i ) ) )
!write( control%out, "( ' i, zu ', I7, ES10.2 )" ) i, data%H_s( i )
          END IF
        END DO

        DO i = 1, prob%m
          IF ( data%C( i ) <= prob%C_l( i ) - tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%C( i ) - prob%C_l( i ) ) )
!write( control%out, "( ' i, c, cl ', I7, 2ES10.2 )" )              &
!  i, data%C( i ), prob%C_l( i )
          END IF
          IF ( data%C( i ) >= prob%C_u( i ) + tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%C( i ) - prob%C_u( i ) ) )
!write( control%out, "( ' i, c, cu ', I7, 2ES10.2 )" )
!  i, data%C( i ), prob%C_u( i )
          END IF
          IF ( ABS( prob%C_u( i ) - prob%C_l( i )) <= tol ) CYCLE
          IF ( ABS( data%C( i ) - prob%C_l( i ) ) <= tol .AND.                 &
               data%Y_trial( i ) < - tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%Y_trial( i ) ) )
!write( control%out, "( ' i, yl ', I7, ES10.2 )" )                 &
!  i, data%Y_trial( i )
          END IF
          IF ( ABS( data%C( i ) - prob%C_u( i ) ) <= tol .AND.                 &
               data%Y_trial( i ) > tol ) THEN
            l = l + 1
            viol8 = MAX( viol8, ABS( data%Y_trial( i ) ) )
!write( control%out, "( ' i, yu ', I7, ES10.2 )" )                 &
!  i, data%Y_trial( i )
          END IF
        END DO

!!$        DO i = 1, prob%n
!!$          IF ( data%X_trial( i ) <= prob%X_l( i ) - tol ) THEN
!!$            write(6,*) i, ABS( data%X_trial( i ) - prob%X_l( i ) )
!!$          END IF
!!$          IF ( data%X_trial( i ) >= prob%X_u( i ) + tol ) THEN
!!$            write(6,*) i, ABS( data%X_trial( i ) - prob%X_u( i ) )
!!$          END IF
!!$          IF ( ABS( prob%X_u( i ) - prob%X_l( i )) <= tol ) CYCLE
!!$          IF ( ABS( data%X_trial( i ) - prob%X_l( i ) ) <= tol .AND.        &
!!$               data%H_s( i ) < - tol ) THEN
!!$            write(6,*) i, ABS( data%H_s( i ) )
!!$          END IF
!!$          IF ( ABS( data%X_trial( i ) - prob%X_u( i ) ) <= tol .AND.        &
!!$               data%H_s( i ) > tol ) THEN
!!$            write(6,*) i, ABS( data%H_s( i ) )
!!$          END IF
!!$        END DO
!!$
!!$        DO i = 1, prob%m
!!$          IF ( data%C( i ) <= prob%C_l( i ) - tol ) THEN
!!$            write(6,*) i, ABS( data%C( i ) - prob%C_l( i ) )
!!$          END IF
!!$          IF ( data%C( i ) >= prob%C_u( i ) + tol ) THEN
!!$            write(6,*) i, ABS( data%C( i ) - prob%C_u( i ) )
!!$          END IF
!!$          IF ( ABS( prob%C_u( i ) - prob%C_l( i ) ) <= tol ) CYCLE
!!$          IF ( ABS( data%C( i ) - prob%C_l( i ) ) <= tol .AND.              &
!!$               data%Y_trial( i ) < - tol ) THEN
!!$            write(6,*) i, ABS( data%Y_trial( i ) )
!!$          END IF
!!$          IF ( ABS( data%C( i ) - prob%C_u( i ) ) <= tol .AND.              &
!!$               data%Y_trial( i ) > tol ) THEN
!!$            write(6,*) i, ABS( data%Y_trial( i ) )
!!$          END IF
!!$        END DO

        IF ( l == 0 ) THEN
          prob%X( : prob%n ) = data%X_trial( : prob%n )
          prob%Y( : prob%m ) = data%Y_trial( : prob%m )
          prob%Z( : prob%n ) = data%H_s( : prob%n )
          prob%C( : prob%m ) = data%C( : prob%m )
          inform%obj = q
          IF ( control%qpb_or_qpa .OR. .NOT. control%no_qpb ) THEN
            IF ( printi ) WRITE( control%out, "( /, A, ' ** EQP confirms',     &
         &  ' accurate solution found by QPB/CQP ... QPA not needed **' )" )   &
               prefix
          ELSE
            IF ( printi ) WRITE( control%out, "( /, A, ' ** EQP confirms',     &
         &  ' optimal guess ... QPA not needed **' )" ) prefix
          END IF

!  Find the solution to the system

!       (    H  A_active^T ) ( x_p ) = - ( g_p )
!       ( A_active     0   ) ( y_p )     (  0  )

!  (required by the GALAHAD module fastr)

          IF ( PRESENT( G_p ) .AND. PRESENT( X_p ) .AND.                       &
               PRESENT( Y_p ) .AND. PRESENT( Z_p ) ) THEN

!  Set up the data for the related EQP:

!  the constant term from the objective ...

            f = zero

!  ... the linear term from the objective ...

            ii = 0
            DO i = 1, prob%n
              IF ( B_stat( i ) == 0 ) THEN
                ii = ii + 1
                data%GRAD( ii ) = G_p( i )
              END IF
            END DO

!  ... and the constraint values

            data%C( : data%A_sbls%m ) = zero

!  solve the EQP

            IF ( printi ) WRITE( control%out, "( /, A, ' re-entering EQP ' )") &
              prefix
            CALL EQP_resolve_main( data%H_sbls%n, data%A_sbls%m,               &
                                   data%H_sbls, data%GRAD( : data%H_sbls%n ),  &
                                   f, data%A_sbls, q,                          &
                                   data%H_s( : data%H_sbls%n ),                &
                                   data%A_s( : data%A_sbls%m ),                &
                                   data, EQP_control, inform%EQP_inform,       &
                                   C = data%C( : data%A_sbls%m ) )
            IF ( printi ) THEN
              IF ( control%EQP_control%out > 0 .AND.                           &
                   control%EQP_control%print_level > 0 )                       &
                WRITE( control%out, "( /, A, ' returned from EQP' )" ) prefix
              WRITE( control%out, "( A, ' on exit from EQP: status = ', I0 )" )&
                prefix, inform%EQP_inform%status
            END IF

!  if the eqp solution succeeded, map the solution onto the entire problem

            IF ( inform%EQP_inform%status == 0 ) THEN

!  spread the free variables through x

              ii = 0
              DO i = 1, prob%n
                IF ( B_stat( i ) == 0 ) THEN
                  ii = ii + 1
                  X_p( i ) = data%H_s( ii )
                END IF
              END DO

!  spread the non-zero multipliers through y

              ii = 0
              DO i = 1, prob%m
                IF ( C_stat( i ) /= 0 ) THEN
                  ii = ii + 1
                  Y_p( i ) = data%A_s( ii )
                ELSE
                  Y_p( i ) = zero
                END IF
              END DO

!  recover the dual variables for the active bounds

!   z = g ...

              DO i = 1, prob%n
                IF ( B_stat( i ) == 0 ) THEN
                  Z_p( i ) = zero
                ELSE
                  Z_p( i ) = prob%G( i )
                END IF
              END DO

!  ... + H x ...

              DO i = 1, prob%n
                ii = data%P( i )
                DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                  j = prob%H%col( l )
                  jj = data%P( j )
                  IF ( ii == 0 .AND. jj == 0 ) THEN
                    Z_p( i ) = Z_p( i ) + prob%H%val( l ) * X_p( j )
                    IF ( i /= j )                                              &
                      Z_p( j ) = Z_p( j ) + prob%H%val( l ) * X_p( i )
                  ELSE IF ( ii == 0 ) THEN
                    Z_p( i ) = Z_p( i ) + prob%H%val( l ) * X_p( j )
                  ELSE IF ( jj == 0 ) THEN
                    Z_p( j ) = Z_p( j ) + prob%H%val( l ) * X_p( i )
                  END IF
                END DO
              END DO

!  ... - A^T y

              ii = 0
              DO i = 1, prob%m
                IF ( C_stat( i ) /= 0 ) THEN
                  ii = ii + 1
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    j = prob%A%col( l )
                    IF ( B_stat( j ) /= 0 )                                    &
                      Z_p( j ) = Z_p( j ) - prob%A%val( l ) * Y_p( ii )
                  END DO
                END IF
              END DO
            ELSE
            END IF
          END IF
          GO TO 700
        ELSE
          IF ( printi ) write( control%out,                                    &
           "( A, 1X, I0, ' primal-dual violations of norm ', ES10.4 )" )       &
              prefix, l, viol8
        END IF
      END IF

!  select the penalty parameters

      IF ( control%qpb_or_qpa .OR. control%no_qpb ) THEN
        IF( control%rho_g <= zero ) THEN
          prob%rho_g = 2 * prob%m
        ELSE
          prob%rho_g = control%rho_g
        END IF

        IF( control%rho_b <= zero ) THEN
          prob%rho_b = 2 * prob%n
        ELSE
          prob%rho_b = control%rho_b
        END IF
      ELSE
        IF( control%rho_g <= zero ) THEN
          prob%rho_g = prob%m
          prob%rho_g = two * MAX( prob%rho_g, MAXVAL( ABS( prob%Y ) ) )
        ELSE
          prob%rho_g = control%rho_g
        END IF

        IF( control%rho_b <= zero ) THEN
          prob%rho_b = prob%n
          prob%rho_b = two * MAX( prob%rho_b, MAXVAL( ABS( prob%Z ) ) )
        ELSE
          prob%rho_b = control%rho_b
        END IF

        IF ( printi ) WRITE( control%out,                                      &
          "( /, A, ' Warm-starting QPA with rho_g = ', ES8.2,                  &
       &     ', rho_b = ', ES8.2, /, A, ' ', I0, ' constraints and ', I0,      &
       &   ' bounds in the initial working set' )" ) prefix,                   &
          prob%rho_g, prob%rho_b, prefix, COUNT( C_stat( : prob%m ) /= 0 ),    &
          COUNT( B_stat( : prob%n ) /= 0 )
      END IF

!  Allocate more real workspace

      array_name = 'qpc: data%BREAKP'
      CALL SPACE_resize_array( lbreak, data%BREAKP, inform%status,             &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%PERT'
      CALL SPACE_resize_array( prob%m + prob%n, data%PERT, inform%status,      &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%GRAD '
      CALL SPACE_resize_array( prob%n, data%GRAD, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%RHS'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%RHS, inform%status,                                          &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%S'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%S, inform%status,                                            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%B'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%B, inform%status,                                            &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%RES'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%RES, inform%status,                                          &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%S_perm'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%S_perm, inform%status,                                       &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%DX'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%DX, inform%status,                                           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%RES_print'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%RES_print, inform%status,                                    &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      IF ( QPA_control%precon >= 0 ) THEN
        n_pcg = prob%n
      ELSE
        n_pcg = 0
      END IF

      array_name = 'qpc: data%R_pcg'
      CALL SPACE_resize_array( n_pcg, data%R_pcg, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%X_pcg'
      CALL SPACE_resize_array(n_pcg , data%X_pcg, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%P_pcg'
      CALL SPACE_resize_array( n_pcg, data%P_pcg, inform%status,               &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Allocate integer workspace arrays

      array_name = 'qpc: data%SC'
      CALL SPACE_resize_array(  QPA_control%max_sc + 1,                        &
             data%SC, inform%status,                                           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%REF'
      CALL SPACE_resize_array( m_link, data%REF, inform%status,                &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%C_up_or_low'
      CALL SPACE_resize_array( data%dims%c_u_start, data%dims%c_l_end,         &
             data%C_up_or_low, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%X_up_or_low'
      CALL SPACE_resize_array( data%dims%x_u_start, data%dims%x_l_end,         &
             data%X_up_or_low, inform%status,                                  &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%PERM'
      CALL SPACE_resize_array( k_n_max + QPA_control%max_sc,                   &
             data%PERM, inform%status,                                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Find the total length of the QPA_control%max_sc largest rows

      DO i = 1, prob%m
        data%IBREAK( i ) = prob%A%ptr( i ) - prob%A%ptr( i + 1 )
      END DO
      CALL SORT_heapsort_build( prob%m, data%IBREAK( : prob%m ),               &
                                inform%status )
      lbd = 0
      DO i = 1, MIN( QPA_control%max_sc, prob%m )
        ii = prob%m - i + 1
        CALL SORT_heapsort_smallest( ii, data%IBREAK( : ii ), inform%status )
        lbd = lbd - data%IBREAK( ii )
      END DO
      IF ( QPA_control%max_sc > prob%m )                                       &
        lbd = lbd + QPA_control%max_sc - prob%m

!  Allocate arrays

      array_name = 'qpc: data%SCU_mat%BD_col_start'
      CALL SPACE_resize_array( QPA_control%max_sc + 1,                         &
             data%SCU_mat%BD_col_start, inform%status,                         &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%SCU_mat%BD_val'
      CALL SPACE_resize_array( lbd, data%SCU_mat%BD_val, inform%status,        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%SCU_mat%BD_row'
      CALL SPACE_resize_array( lbd, data%SCU_mat%BD_row, inform%status,        &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

      array_name = 'qpc: data%DIAG'
      CALL SPACE_resize_array( 2, K_n_max, data%DIAG, inform%status,           &
             inform%alloc_status, array_name = array_name,                     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  decide on appropriate initial preconditioners and factorizations

      data%auto_prec = QPA_control%precon == 0
      data%auto_fact = QPA_control%factor == 0

!  If the Hessian has semi-bandwidth smaller than nsemib and the preconditioner
!  is to be picked automatically, use the full Hessian. Otherwise, use the
!  Hessian of the specified semi-bandwidth.

      IF ( data%auto_prec ) THEN
        data%prec_hist = 2

!  prec_hist indicates which factors are currently being used. Possible values:
!   1 full factors used
!   2 band factors used
!   3 diagonal factors used (as a last resort)

!  Check to see if the Hessian is banded

 dod :  DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = data%dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_free

          CASE ( 2 )

            hd_start  = data%dims%x_free + 1
            hd_end    = data%dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = data%dims%x_l_start
            hd_end    = data%dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = data%dims%x_u_start
            hd_end    = data%dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_l_end

          CASE ( 5 )

            hd_start  = data%dims%x_l_end + 1
            hd_end    = data%dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = data%dims%x_u_end

          CASE ( 6 )

            hd_start  = data%dims%x_u_end + 1
            hd_end    = data%dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = prob%n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, prob%n )
          DO i = hd_start, hd_end
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 2
              IF ( ABS( i - prob%H%col( l ) ) > QPA_control%nsemib) THEN
                data%prec_hist = 1
                EXIT dod
              END IF
            END DO
          END DO
          IF ( hd_end == prob%n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, prob%n )
          DO i = hnd_start, hnd_end
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              IF ( ABS( i - prob%H%col( l ) ) > QPA_control%nsemib) THEN
                data%prec_hist = 1
                EXIT dod
              END IF
            END DO
          END DO
          IF ( hd_end == prob%n ) EXIT

        END DO dod

      END IF

!  Ensure that the intial status vectors are assigned correctly

      DO i = 1, prob%n
        IF ( B_stat( i ) < 0 ) THEN
          B_stat( i ) = - 1
        ELSE IF ( B_stat( i ) > 0 ) THEN
          B_stat( i ) = 1
        END IF
      END DO
      DO i = 1, prob%m
        IF ( C_stat( i ) < 0 ) THEN
          C_stat( i ) = - 1
        ELSE IF ( C_stat( i ) > 0 ) THEN
          C_stat( i ) = 1
        END IF
      END DO

!  =============
!  Now crossover
!  =============

      data%SLS_control = QPA_control%SLS_control

      IF ( printi ) WRITE( control%out, "( /, A, ' entering QPA ' )" ) prefix

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      CALL QPA_solve_qp( data%dims, prob%n, prob%m,                            &
                         prob%H%val, prob%H%col, prob%H%ptr,                   &
                         prob%G, prob%f, prob%rho_g, prob%rho_b, prob%A%val,   &
                         prob%A%col, prob%A%ptr, prob%C_l, prob%C_u, prob%X_l, &
                         prob%X_u, prob%X, prob%Y, prob%Z, C_stat, B_stat,     &
                         m_link, K_n_max, lbreak, data%RES_l, data%RES_u,      &
                         data%A_norms, data%H_s, data%BREAKP, data%A_s,        &
                         data%PERT, data%GRAD, data%VECTOR, data%RHS, data%S,  &
                         data%B, data%RES, data%S_perm, data%DX, n_pcg,        &
                         data%R_pcg, data%X_pcg, data%P_pcg, data%Abycol_val,  &
                         data%Abycol_row, data%Abycol_ptr, data%S_val,         &
                         data%S_row, data%S_col, data%S_colptr, data%IBREAK,   &
                         data%SC, data%REF, data%RES_print, data%DIAG,         &
                         data%C_up_or_low, data%X_up_or_low, data%PERM,        &
                         data%P, data%SOL, data%D,                             &
                         data%SLS_data, data%SLS_control,                      &
                         data%SCU_mat, data%SCU_info, data%SCU_data, data%K,   &
                         data%seed, time_record, clock_record,                 &
                         data%start_print, data%stop_print,                    &
                         data%prec_hist, data%auto_prec, data%auto_fact,       &
                         QPA_control%print_level > 0 .AND.                     &
                         QPA_control%out > 0, qpa_prefix,                      &
                         QPA_control, inform%QPA_inform,                       &
                         G_p = G_p, X_p = X_p, Y_p = Y_p, Z_p = Z_p )

!   write(6,*) '     x_l          x            x_u            z'
!   do i = 1, prob%n
!   write(6,"(4ES12.4)" ) prob%X_l( i ), prob%X( i ), prob%X_u( i ), prob%Z( i )
!   end do

!   write(6,*) '     c_l          c            c_u            y'
!   do i = 1, prob%m
!   write(6,"(4ES12.4)" ) prob%C_l( i ), prob%C( i ), prob%C_u( i ), prob%Y( i )
!   end do

!  Record times for components of QPA

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
      time_now = time_now - time_record ; clock_now = clock_now - clock_record
      inform%QPA_inform%time%total = inform%QPA_inform%time%total + time_now
      inform%QPA_inform%time%clock_total =                                     &
        inform%QPA_inform%time%clock_total + clock_now
      IF ( printi ) THEN
        IF ( QPA_control%out > 0 .AND. QPA_control%print_level > 0 )           &
          WRITE( control%out, "( /, A, ' returned from QPA' )" ) prefix
        WRITE( control%out, "( A, ' on exit from QPA: status = ', I0,          &
       &   ', iterations = ', I0, ', time = ', F0.2 )" ) prefix,               &
          inform%QPA_inform%status, inform%QPA_inform%iter, clock_now
        WRITE( control%out, "( A, ' objective value =', ES12.4,                &
       &   /, A, ' # active counstraints: ', I0, ' from ', I0,                 &
       &   ', bounds: ', I0, ' from ', I0 )" ) prefix, inform%QPA_inform%obj,  &
            prefix, COUNT( C_stat( : prob%m ) /= 0 ),  prob%m,                 &
            COUNT( B_stat( : prob%n ) /= 0 ),  prob%n
      END IF

!  Record output information from QPA

!     IF ( .NOT. inform%QPB_inform%feasible )                                  &
        inform%status = inform%QPA_inform%status
      inform%alloc_status = inform%QPA_inform%alloc_status
      inform%nfacts = inform%nfacts + inform%QPA_inform%nfacts
      inform%nmods = inform%nmods + inform%QPA_inform%nmods
      inform%obj = inform%QPA_inform%obj

!  Record times for components of QPA

      inform%time%total = inform%time%total + inform%QPA_inform%time%total
      inform%time%analyse =                                                    &
        inform%time%analyse + inform%QPA_inform%time%analyse
      inform%time%factorize =                                                  &
        inform%time%factorize + inform%QPA_inform%time%factorize
      inform%time%solve =inform%time%solve + inform%QPA_inform%time%solve
      inform%time%preprocess =                                                 &
        inform%time%preprocess + inform%QPA_inform%time%preprocess

      IF ( printi ) THEN
        WRITE( control%out, "( A, ' ', I0, ' integer and ', I0,                &
      &  ' real words required for factors in QPA' )" ) prefix,                &
         inform%QPA_inform%factorization_integer,                              &
         inform%QPA_inform%factorization_real
      END IF

!  If the crosover fails for any reason, use the active-set predictions from
!  the original QP solve

      IF ( inform%QPA_inform%status < 0 ) THEN
        DO i = 1, prob%m
          IF ( data%IW( i ) > 0 ) THEN
            C_stat( i ) = 1
          ELSE IF ( data%IW( i ) < 0 ) THEN
            C_stat( i ) = - 1
          ELSE
            C_stat( i ) = 0
          END IF
        END DO
        DO i = 1, prob%n
          IF ( data%IW( prob%m + i ) > 0 ) THEN
            B_stat( i ) = 1
          ELSE IF ( data%IW( prob%m + i ) < 0 ) THEN
            B_stat( i ) = - 1
          ELSE
            B_stat( i ) = 0
          END IF
        END DO

!  If possible, restore the solution from QPB

        IF ( gotsol ) THEN
          prob%X( : prob%n ) = data%X_trial( : prob%n )
          prob%Y( : prob%m ) = data%Y_last( : prob%m )
          prob%Z( : prob%n ) = data%Z_last( : prob%n )
          prob%C( : prob%m ) = data%C( : prob%m )
          inform%obj = best_obj
          inform%status = GALAHAD_ok
        END IF
      ELSE
        inform%p_found = .TRUE.
      END IF

      IF ( control%qpb_or_qpa .OR. .NOT. control%no_qpb ) THEN
        IF ( printd ) THEN
          DO i = 1, prob%m
            IF ( data%IW( i ) /= C_stat( i ) ) WRITE( control%out,             &
              "( ' C_stat ', I6, ' QPB/LSQP/CQP = ', I2, ' vs QPA = ', I2 )" ) &
                i, data%IW( i ), C_stat( i )
          END DO
          DO i = 1, prob%n
            IF ( data%IW( prob%m + i ) /= B_stat( i ) ) WRITE( control%out,    &
              "( ' B_stat ', I6, ' QPB/LSQP/CQP = ', I2, ' vs QPA = ', I2 )" ) &
                i, data%IW( prob%m + i ), B_stat( i )
          END DO
        ELSE
          IF ( printi ) WRITE( control%out,                                    &
            "( A, ' # changes to C_stat, B_stat: QPB/LSQP/CQP vs QPA = ', I0,  &
            &  ', ', I0 )" ) prefix,                                           &
             COUNT( C_stat( : prob%m ) /= data%IW( : prob%m ) ),               &
             COUNT( B_stat( : prob%n ) /=                                      &
                   data%IW( prob%m + 1 : prob%m + prob%n ) )
        END IF

        DO i = 1, prob%n
          IF ( data%IW( prob%m + i ) /= B_stat( i ) ) THEN
            IF ( printt ) WRITE( control%out,                                  &
              "( '  B_stat ', I6, ' QPB/LSQP/CQP = ', I2, ' vs QPA = ', I2, /, &
           &     '  dist_x_l, dist_x_u, z = ', 3ES16.8 )" ) i,                 &
              data%IW( prob%m + i ), B_stat( i ), prob%X( i ) - prob%X_l( i ), &
              prob%X_u( i ) - prob%X( i ), prob%Z( i )
            IF ( ABS( prob%X( i ) - prob%X_l( i ) ) <= control%on_bound_tol )  &
              B_stat( i ) = - 1
            IF ( ABS( prob%X( i ) - prob%X_u( i ) ) <= control%on_bound_tol )  &
              B_stat( i ) = 1
          END IF
        END DO
        DO i = 1, prob%m
          IF ( data%IW( i ) /= C_stat( i ) ) THEN
            IF ( printt ) WRITE( control%out,                                  &
              "( '  C_stat ', I6, ' QPB/LSQP/CQP = ', I2, ' vs QPA = ', I2, /, &
           &     '  dist_c_l, dist_c_u, y = ', 3ES16.8 )" ) i,                 &
              data%IW( i ), C_stat( i ), prob%C( i ) - prob%C_l( i ),          &
              prob%C_u( i ) - prob%C( i ), prob%Y( i )
            IF ( ABS( prob%C( i ) - prob%C_l( i ) ) <= control%on_bound_tol )  &
              C_stat( i ) = - 1
            IF ( ABS( prob%C( i ) - prob%C_u( i ) ) <= control%on_bound_tol )  &
              C_stat( i ) = 1
          END IF
        END DO
      END IF

  700 CONTINUE

!     IF ( control%yes ) THEN
!      OPEN( 56 )
!      WRITE( 56, "( '    i st        dist_x_l             dist_x_u        z')")
!      DO i = 1, prob%n
!        WRITE( 56, "( I6, I3, 3ES22.14 )" ) i, B_stat( i ),                   &
!         prob%X( i ) - prob%X_l( i ), prob%X_u( i ) - prob%X( i ), prob%Z( i )
!      END DO

!      WRITE( 56, "( '    i st        dist_c_l             dist_c_u        y')")
!      DO i = 1, prob%m
!        WRITE( 56, "( I6, I3, 3ES22.14 )" ) i, C_stat( i ),                   &
!         prob%C( i ) - prob%C_l( i ), prob%C_u( i ) - prob%C( i ), prob%Y( i )
!      END DO
!      CLOSE( 56 )
!     END IF

!  If some of the constraints were freed having first been fixed during
!  the computation, refix them now

      IF ( remap_more_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( inform%p_found ) THEN
          IF ( PRESENT( X_p ) ) THEN
            X_p( prob%n + 1 : data%QPP_map_more_freed%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map_more_freed%n,              &
                                       data%QPP_map_more_freed%x_map,          &
                                       X = X_p( : data%QPP_map_more_freed%n ) )
          END IF
          IF ( PRESENT( Y_p ) ) THEN
            Y_p( prob%m + 1 : data%QPP_map_more_freed%m ) = zero
            CALL SORT_inverse_permute( data%QPP_map_more_freed%m,              &
                                       data%QPP_map_more_freed%c_map,          &
                                       X = Y_p( : data%QPP_map_more_freed%m ) )
          END IF
          IF ( PRESENT( Z_p ) ) THEN
            Z_p( prob%n + 1 : data%QPP_map_more_freed%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map_more_freed%n,              &
                                       data%QPP_map_more_freed%x_map,          &
                                       X = Z_p( : data%QPP_map_more_freed%n ) )
          END IF
        END IF
        C_stat( prob%m + 1 : data%QPP_map_more_freed%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map_more_freed%m,                  &
                                   data%QPP_map_more_freed%c_map,              &
                                   IX = C_stat( : data%QPP_map_more_freed%m ) )
        B_stat( prob%n + 1 : data%QPP_map_more_freed%n ) = - 1
!       B_stat( prob%n + 1 : data%QPP_map_more_freed%n ) = 0
        CALL SORT_inverse_permute( data%QPP_map_more_freed%n,                  &
                                   data%QPP_map_more_freed%x_map,              &
                                   IX = B_stat( : data%QPP_map_more_freed%n ) )
        CALL QPP_restore( data%QPP_map_more_freed, data%QPP_inform,            &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,         &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_more_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_more_depen
          j = data%Index_c_more_freed( i )
          prob%C_l( j ) = data%C_more_freed( i )
          prob%C_u( j ) = data%C_more_freed( i )
        END DO
      END IF

!  If some of the variables/constraints were fixed during the computation,
!  free them now

  720 CONTINUE
      IF ( remap_fixed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( inform%p_found ) THEN
          IF ( PRESENT( X_p ) ) THEN
            X_p( prob%n + 1 : data%QPP_map_fixed%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map_fixed%n,                   &
                                       data%QPP_map_fixed%x_map,               &
                                       X = X_p( : data%QPP_map_fixed%n ) )
          END IF
          IF ( PRESENT( Y_p ) ) THEN
            Y_p( prob%m + 1 : data%QPP_map_fixed%m ) = zero
            CALL SORT_inverse_permute( data%QPP_map_fixed%m,                   &
                                       data%QPP_map_fixed%c_map,               &
                                       X = Y_p( : data%QPP_map_fixed%m ) )
          END IF
          IF ( PRESENT( Z_p ) ) THEN
            Z_p( prob%n + 1 : data%QPP_map_fixed%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map_fixed%n,                   &
                                       data%QPP_map_fixed%x_map,               &
                                       X = Z_p( : data%QPP_map_fixed%n ) )
          END IF
        END IF
        C_stat( prob%m + 1 : data%QPP_map_fixed%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map_fixed%m,                       &
                                   data%QPP_map_fixed%c_map,                   &
                                   IX = C_stat( : data%QPP_map_fixed%m ) )
        B_stat( prob%n + 1 : data%QPP_map_fixed%n ) = - 1
!       B_stat( prob%n + 1 : data%QPP_map_fixed%n ) = 0
        CALL SORT_inverse_permute( data%QPP_map_fixed%n,                       &
                                   data%QPP_map_fixed%x_map,                   &
                                   IX = B_stat( : data%QPP_map_fixed%n ) )
        CALL QPP_restore( data%QPP_map_fixed, data%QPP_inform,                 &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_fixed

!  Release the temporarily fixed problem bounds

        DO i = 1, tiny_x
          j = data%Index_X_fixed( i )
          IF ( j > 0 ) THEN
            prob%X_l( j ) = data%X_fixed( i )
            IF ( B_stat( j ) < 0 ) THEN
              B_stat( j ) = - B_stat( j )
!             prob%Z( j ) = -  prob%Z( j )
            END IF
          ELSE
            prob%X_u( - j ) = data%X_fixed( i )
          END IF
        END DO

!  Do the same for the constraint bounds

        DO i = 1, tiny_c
          j = data%Index_C_fixed( i )
          IF ( j > 0 ) THEN
            prob%C_l( j ) = data%C_fixed( i )
            IF ( C_stat( j ) < 0 ) THEN
              C_stat( j ) = - C_stat( j )
!             prob%Y( j ) = -  prob%Y( j )
            END IF
          ELSE
            prob%C_u( - j ) = data%C_fixed( i )
          END IF
        END DO

      END IF

!  If some of the constraints were freed during the computation, refix them now

      IF ( remap_freed ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( inform%p_found ) THEN
          IF ( PRESENT( X_p ) ) THEN
            X_p( prob%n + 1 : data%QPP_map_freed%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map_freed%n,                   &
                                       data%QPP_map_freed%x_map,               &
                                       X = X_p( : data%QPP_map_freed%n ) )
          END IF
          IF ( PRESENT( Y_p ) ) THEN
            Y_p( prob%m + 1 : data%QPP_map_freed%m ) = zero
            CALL SORT_inverse_permute( data%QPP_map_freed%m,                   &
                                       data%QPP_map_freed%c_map,               &
                                       X = Y_p( : data%QPP_map_freed%m ) )
          END IF
          IF ( PRESENT( Z_p ) ) THEN
            Z_p( prob%n + 1 : data%QPP_map_freed%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map_freed%n,                   &
                                       data%QPP_map_freed%x_map,               &
                                       X = Z_p( : data%QPP_map_freed%n ) )
          END IF
        END IF
        C_stat( prob%m + 1 : data%QPP_map_freed%m ) = 0
        CALL SORT_inverse_permute( data%QPP_map_freed%m,                       &
                                   data%QPP_map_freed%c_map,                   &
                                   IX = C_stat( : data%QPP_map_freed%m ) )
!       B_stat( prob%n + 1 : data%QPP_map_freed%n ) = 0
        B_stat( prob%n + 1 : data%QPP_map_freed%n ) = - 1
        CALL SORT_inverse_permute( data%QPP_map_freed%n,                       &
                                   data%QPP_map_freed%x_map,                   &
                                   IX = B_stat( : data%QPP_map_freed%n ) )
        CALL QPP_restore( data%QPP_map_freed, data%QPP_inform,                 &
                          prob, get_all = .TRUE. )
!       CALL QPP_terminate( data%QPP_map_freed, data%QPP_control,              &
!                           data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        data%dims = data%dims_save_freed

!  Fix the temporarily freed constraint bounds

        DO i = 1, n_depen
          j = data%Index_c_freed( i )
          prob%C_l( j ) = data%C_freed( i )
          prob%C_u( j ) = data%C_freed( i )
        END DO

      END IF
      data%tried_to_remove_deps = .FALSE.

!  Retore the problem to its original form

  750 CONTINUE
      data%trans = data%trans - 1
      IF ( data%trans == 0 ) THEN
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( inform%p_found ) THEN
          IF ( PRESENT( X_p ) ) THEN
            X_p( prob%n + 1 : data%QPP_map%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,     &
                                       X = X_p( : data%QPP_map%n ) )
          END IF
          IF ( PRESENT( Y_p ) ) THEN
            Y_p( prob%m + 1 : data%QPP_map%m ) = zero
            CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,     &
                                       X = Y_p( : data%QPP_map%m ) )
          END IF
          IF ( PRESENT( Z_p ) ) THEN
            Z_p( prob%n + 1 : data%QPP_map%n ) = zero
            CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,     &
                                       X = Z_p( : data%QPP_map%n ) )
          END IF
        END IF
        CALL SORT_inverse_permute( data%QPP_map%m, data%QPP_map%c_map,         &
                                   IX = C_stat( : data%QPP_map%m ) )
        B_stat( prob%n + 1 : data%QPP_map%n ) = - 1
        CALL SORT_inverse_permute( data%QPP_map%n, data%QPP_map%x_map,         &
                                   IX = B_stat( : data%QPP_map%n ) )

!  Full restore

        IF ( control%restore_problem >= 2 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_all = .TRUE. )

!  Restore vectors and scalars

        ELSE IF ( control%restore_problem == 1 ) THEN
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                            get_f = .TRUE., get_g = .TRUE.,                    &
                            get_x = .TRUE., get_x_bounds = .TRUE.,             &
                            get_y = .TRUE., get_z = .TRUE.,                    &
                            get_c = .TRUE., get_c_bounds = .TRUE. )

!  Solution recovery

        ELSE
          CALL QPP_restore( data%QPP_map, data%QPP_inform, prob,               &
                             get_x = .TRUE., get_y = .TRUE., get_z = .TRUE.,   &
                             get_c = .TRUE. )
        END IF
!       CALL QPP_terminate( data%QPP_map, data%QPP_control, data%QPP_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%preprocess = inform%time%preprocess + time_now - time_record
        inform%time%clock_preprocess =                                         &
          inform%time%clock_preprocess + clock_now - clock_record
        prob%new_problem_structure = data%new_problem_structure
        data%save_structure = .TRUE.
      END IF

!  Compute total time

  800 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total =                                                      &
        inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      inform%time%analyse = inform%time%analyse +                              &
        inform%FDC_inform%time%analyse +                                       &
        inform%QPA_inform%time%analyse +                                       &
        inform%QPB_inform%time%analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%FDC_inform%time%clock_analyse +                                 &
        inform%QPA_inform%time%clock_analyse +                                 &
        inform%QPB_inform%time%clock_analyse
      inform%time%factorize = inform%time%factorize +                          &
        inform%FDC_inform%time%factorize +                                     &
        inform%QPA_inform%time%factorize +                                     &
        inform%QPB_inform%time%factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%FDC_inform%time%clock_factorize +                               &
        inform%QPA_inform%time%clock_factorize +                               &
        inform%QPB_inform%time%clock_factorize
      inform%time%solve = inform%time%solve +                                  &
        inform%QPA_inform%time%solve +                                         &
        inform%QPB_inform%time%solve
      inform%time%clock_solve = inform%time%clock_solve +                      &
        inform%QPA_inform%time%clock_solve +                                   &
        inform%QPB_inform%time%clock_solve

      IF ( control%out > 0 .AND. control%print_level >= 2 )                    &
        WRITE( control%out, 2000 )                                             &
          inform%time%total, inform%time%preprocess,                           &
          inform%time%analyse, inform%time%factorize, inform%time%solve,       &
          inform%QPA_inform%time%total, inform%QPB_inform%time%total,          &
          inform%QPA_inform%time%analyse, inform%QPA_inform%time%factorize,    &
          inform%QPA_inform%time%solve, inform%QPB_inform%time%analyse,        &
          inform%QPB_inform%time%factorize, inform%QPB_inform%time%solve

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPC_solve ' )" )
      RETURN

!  Allocation error

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total =                                                      &
        inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      inform%time%analyse = inform%time%analyse +                              &
        inform%FDC_inform%time%analyse +                                       &
        inform%QPA_inform%time%analyse +                                       &
        inform%QPB_inform%time%analyse
      inform%time%clock_analyse = inform%time%clock_analyse +                  &
        inform%FDC_inform%time%clock_analyse +                                 &
        inform%QPA_inform%time%clock_analyse +                                 &
        inform%QPB_inform%time%clock_analyse
      inform%time%factorize = inform%time%factorize +                          &
        inform%FDC_inform%time%factorize +                                     &
        inform%QPA_inform%time%factorize +                                     &
        inform%QPB_inform%time%factorize
      inform%time%clock_factorize = inform%time%clock_factorize +              &
        inform%FDC_inform%time%clock_factorize +                               &
        inform%QPA_inform%time%clock_factorize +                               &
        inform%QPB_inform%time%clock_factorize
      inform%time%solve = inform%time%solve +                                  &
        inform%QPA_inform%time%solve +                                         &
        inform%QPB_inform%time%solve
      inform%time%clock_solve = inform%time%clock_solve +                      &
        inform%QPA_inform%time%clock_solve +                                   &
        inform%QPB_inform%time%clock_solve

      IF ( control%out > 0 .AND. control%print_level >= 5 )                    &
        WRITE( control%out, "( ' leaving QPC_solve ' )" )

      RETURN

!  Non-executable statements

 2000 FORMAT( /, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',               &
              '-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-=',                             &
              /, ' =', 5X, 'QPC timing statistics:  total', 0P, F12.2,         &
                 ' preprocess', F12.2, 5X, '=',                                &
              /, ' =', 23X, 'analyse   factorize    solve', 23X, '=',          &
              /, ' =', 18X, 3F11.2, 23X, '=',                                  &
              /, ' =', 12X, 'QPA: total',  F12.2,                              &
                       14X, 'QPB: total',  F12.2, 4X, '=',                     &
              /, ' =      analyse    factorize     solve',                     &
                 '      analyse    factorize     solve  =',                    &
              /, ' =', 6F12.2, 2x, '=',                                        &
              /, ' =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=',               &
                 '-=-=-=-=-=-=-==-=-=-=-=-=-=-=-=-=' )
 2010 FORMAT( ' ', /, '   **  Error return ',I3,' from QPC ' )
 2040 FORMAT( '   **  Error return ', I6, ' from ', A )
 2240 FORMAT( /, '  Warning - an entry from strict upper triangle of H given ' )

!  End of QPC_solve

      END SUBROUTINE QPC_solve

!-*-*-*-*-*-*-   Q P C _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*-*

      SUBROUTINE QPC_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========
!
!   data    see Subroutine QPC_initialize
!   control see Subroutine QPC_initialize
!   inform  see Subroutine QPC_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPC_data_type ), INTENT( INOUT ) :: data
      TYPE ( QPC_control_type ), INTENT( IN ) :: control
      TYPE ( QPC_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

      inform%status = GALAHAD_ok

!  Deallocate all arrays allocated by QPA, QPB, EQP and QPP

      CALL QPA_terminate( data, control%QPA_control, inform%QPA_inform )
      IF ( inform%QPA_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%QPA_inform%alloc_status
!       inform%bad_alloc = inform%QPA_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPB_terminate( data, control%QPB_control, inform%QPB_inform )
      IF ( inform%QPB_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%QPB_inform%alloc_status
!       inform%bad_alloc = inform%QPB_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL CRO_terminate( data%CRO_data, control%CRO_control,                  &
                          inform%CRO_inform )
      IF ( inform%CRO_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%CRO_inform%alloc_status
!       inform%bad_alloc = inform%CRO_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL EQP_terminate( data, control%EQP_control, inform%EQP_inform )
      IF ( inform%EQP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%EQP_inform%alloc_status
!       inform%bad_alloc = inform%EQP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL FDC_terminate( data%FDC_data, control%FDC_control,                  &
                          inform%FDC_inform )
      IF ( inform%FDC_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = inform%FDC_inform%alloc_status
!       inform%bad_alloc = inform%EQP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

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

      CALL QPP_terminate( data%QPP_map_fixed, data%QPP_control,                &
                          data%QPP_inform)
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

      CALL QPP_terminate( data%QPP_map_more_freed, data%QPP_control,           &
                          data%QPP_inform )
      IF ( data%QPP_inform%status /= GALAHAD_ok ) THEN
        inform%status = GALAHAD_error_deallocate
        inform%alloc_status = data%QPP_inform%alloc_status
        inform%bad_alloc = data%QPP_inform%bad_alloc
        IF ( control%deallocate_error_fatal ) RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'qpc: data%HX'
      CALL SPACE_dealloc_array( data%HX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%C_freed'
      CALL SPACE_dealloc_array( data%C_freed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%X0'
      CALL SPACE_dealloc_array( data%X0,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%X_fixed'
      CALL SPACE_dealloc_array( data%X_fixed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%Index_X_fixed'
      CALL SPACE_dealloc_array( data%Index_X_fixed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%C_fixed'
      CALL SPACE_dealloc_array( data%C_fixed,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%Index_C_fixed'
      CALL SPACE_dealloc_array( data%Index_C_fixed,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%HX'
      CALL SPACE_dealloc_array( data%HX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%C_more_freed'
      CALL SPACE_dealloc_array( data%C_more_freed,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%DZ_l'
      CALL SPACE_dealloc_array( data%DZ_l,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%DZ_u'
      CALL SPACE_dealloc_array( data%DZ_u,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%GRAD'
      CALL SPACE_dealloc_array( data%GRAD,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%X_trial'
      CALL SPACE_dealloc_array( data%X_trial,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%GRAD_X_phi'
      CALL SPACE_dealloc_array( data%GRAD_X_phi,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%GRAD_C_phi'
      CALL SPACE_dealloc_array( data%GRAD_C_phi,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%A_s'
      CALL SPACE_dealloc_array( data%A_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%H_s'
      CALL SPACE_dealloc_array( data%H_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%Y_last'
      CALL SPACE_dealloc_array( data%Y_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%Z_last'
      CALL SPACE_dealloc_array( data%Z_last,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%IW'
      CALL SPACE_dealloc_array( data%IW,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%IBREAK'
      CALL SPACE_dealloc_array( data%IBREAK,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%RES_l'
      CALL SPACE_dealloc_array( data%RES_l,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%RES_u'
      CALL SPACE_dealloc_array( data%RES_u,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%A_norms'
      CALL SPACE_dealloc_array( data%A_norms,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%H_s'
      CALL SPACE_dealloc_array( data%H_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%BREAKP'
      CALL SPACE_dealloc_array( data%BREAKP,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%A_s'
      CALL SPACE_dealloc_array( data%A_s,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%PERT'
      CALL SPACE_dealloc_array( data%PERT,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%GRAD'
      CALL SPACE_dealloc_array( data%GRAD,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%VECTOR'
      CALL SPACE_dealloc_array( data%VECTOR,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%RHS'
      CALL SPACE_dealloc_array( data%RHS,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%S'
      CALL SPACE_dealloc_array( data%S,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%B'
      CALL SPACE_dealloc_array( data%B,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%RES'
      CALL SPACE_dealloc_array( data%RES,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%S_perm'
      CALL SPACE_dealloc_array( data%S_perm,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%DX'
      CALL SPACE_dealloc_array( data%DX,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%RES_print'
      CALL SPACE_dealloc_array( data%RES_print,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%R_pcg'
      CALL SPACE_dealloc_array( data%R_pcg,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%X_pcg'
      CALL SPACE_dealloc_array( data%X_pcg,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%P_pcg'
      CALL SPACE_dealloc_array( data%P_pcg,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%SC'
      CALL SPACE_dealloc_array( data%SC,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%REF'
      CALL SPACE_dealloc_array( data%REF,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%C_up_or_low'
      CALL SPACE_dealloc_array( data%C_up_or_low,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%X_up_or_low'
      CALL SPACE_dealloc_array( data%X_up_or_low,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%PERM'
      CALL SPACE_dealloc_array( data%PERM,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%SCU_mat%BD_col_start'
      CALL SPACE_dealloc_array( data%SCU_mat%BD_col_start,                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%SCU_mat%BD_val'
      CALL SPACE_dealloc_array( data%SCU_mat%BD_val,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%SCU_mat%BD_row'
      CALL SPACE_dealloc_array( data%SCU_mat%BD_row,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'qpc: data%DIAG'
      CALL SPACE_dealloc_array( data%DIAG,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine QPC_terminate

      END SUBROUTINE QPC_terminate

!  End of module QPC

   END MODULE GALAHAD_QPC_double
