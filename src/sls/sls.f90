! THIS VERSION: GALAHAD 2.6 - 24/12/2013 AT 14:30 GMT.

!-*-*-*-*-*-*-*-*- G A L A H A D _ S L S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. June 15th 2009
!   LAPACK solvers and band ordering added Version 2.5, November 21st 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SLS_double

!     ---------------------------------------------
!     |                                           |
!     |  Provide interfaces from various packages |
!     |  to allow the solution of                 |
!     |                                           |
!     |        Symmetric Linear Systems           |
!     |                                           |
!     |  Packages currently supported:            |
!     |                                           |
!     |               MA27/SILS                   |
!     |               MA57                        |
!     |               MA77                        |
!     |               MA86                        |
!     |               MA87                        |
!     |               MA97                        |
!     |               SSIDS from SPRAL            |
!     |               PARDISO                     |
!     |               WSMP                        |
!     |               POTR from LAPACK            |
!     |               SYTR from LAPACK            |
!     |               PBTR from LAPACK            |
!     |                                           |
!     ---------------------------------------------

   use iso_c_binding
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SORT_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_STRING_double, ONLY: STRING_lower_word
     USE GALAHAD_SMT_double
     USE GALAHAD_SILS_double
     USE GALAHAD_BLAS_interface, ONLY : TRSV, TBSV
     USE GALAHAD_LAPACK_interface, ONLY : POTRF, POTRS, SYTRF, SYTRS, PBTRF,  &
                                          PBTRS
     USE HSL_ZD11_double
     USE HSL_MA57_double
     USE HSL_MA77_double
     USE HSL_MA86_double
     USE HSL_MA87_double
     USE HSL_MA97_double
     USE HSL_MC64_double
     USE HSL_MC68_integer
     USE SPRAL_SSIDS

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SLS_initialize, SLS_analyse, SLS_factorize, SLS_solve,          &
               SLS_fredholm_alternative,                                       &
               SLS_terminate, SLS_enquire, SLS_alter_d, SLS_part_solve,        &
               SLS_sparse_forward_solve, SLS_read_specfile,                    &
               SLS_initialize_solver, SLS_coord_to_extended_csr,               &
               SLS_coord_to_sorted_csr, SMT_type, SMT_get, SMT_put

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
     INTEGER, PARAMETER :: real_bytes = 8
     INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!  other parameters

     INTEGER, PARAMETER :: len_solver = 20
     REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( 1.0_wp )

!  default control values

     INTEGER, PARAMETER :: bits_default = 32
     INTEGER, PARAMETER :: block_size_kernel_default = 40
     INTEGER, PARAMETER :: block_size_elimination_default = 256
     INTEGER, PARAMETER :: blas_block_size_factor_default = 16
     INTEGER, PARAMETER :: blas_block_size_solve_default = 16
     INTEGER, PARAMETER :: node_amalgamation_default = 32
     INTEGER, PARAMETER :: initial_pool_size_default = 100000
     INTEGER, PARAMETER :: min_real_factor_size_default = 10000
     INTEGER, PARAMETER :: min_integer_factor_size_default = 10000
     INTEGER, PARAMETER :: full_row_threshold_default = 100
     INTEGER, PARAMETER :: row_search_indefinite_default = 10

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLS_control_type

!  unit for error messages

       INTEGER :: error = 6

!  unit for warning messages

       INTEGER :: warning = 6

!  unit for monitor output

       INTEGER :: out = 6

!  unit for statistical output

       INTEGER :: statistics = 0

!  controls level of diagnostic output

       INTEGER :: print_level = 0

!  controls level of diagnostic output from external solver

       INTEGER :: print_level_solver = 0

!  number of bits used in architecture

       INTEGER :: bits = bits_default

!  the target blocksize for kernel factorization

       INTEGER :: block_size_kernel = block_size_kernel_default

!  the target blocksize for parallel elimination

       INTEGER :: block_size_elimination = block_size_elimination_default

!  level 3 blocking in factorize

       INTEGER :: blas_block_size_factorize = blas_block_size_factor_default

!  level 2 and 3 blocking in solve

       INTEGER :: blas_block_size_solve = blas_block_size_solve_default

!  a child node is merged with its parent if they both involve fewer than
!  node_amalgamation eliminations

       INTEGER :: node_amalgamation = node_amalgamation_default

!  initial size of task-pool arrays for parallel elimination

       INTEGER :: initial_pool_size = initial_pool_size_default

!  initial size for real array for the factors and other data

       INTEGER :: min_real_factor_size = min_real_factor_size_default

!  initial size for integer array for the factors and other data

       INTEGER :: min_integer_factor_size = min_integer_factor_size_default

!  maximum size for real array for the factors and other data

       INTEGER ( KIND = long ) :: max_real_factor_size = HUGE( 0 )

!  maximum size for integer array for the factors and other data

       INTEGER ( KIND = long ) :: max_integer_factor_size = HUGE( 0 )

!  amount of in-core storage to be used for out-of-core factorization

       INTEGER ( KIND = long ) :: max_in_core_store = HUGE( 0 ) / real_bytes

!  factor by which arrays sizes are to be increased if they are too small

       REAL ( KIND = wp ) :: array_increase_factor = 2.0_wp

!  if previously allocated internal workspace arrays are greater than
!  array_decrease_factor times the currently required sizes, they are reset
!  to current requirements

       REAL ( KIND = wp ) :: array_decrease_factor = 2.0_wp

!  pivot control:
!   1  Numerical pivoting will be performed.
!   2  No pivoting will be performed and an error exit will
!      occur immediately a pivot sign change is detected.
!   3  No pivoting will be performed and an error exit will
!      occur if a zero pivot is detected.
!   4  No pivoting is performed but pivots are changed to all be positive

       INTEGER :: pivot_control = 1

!  controls ordering (ignored if explicit PERM argument present)
!  <0  calculated internally by package with appropriate ordering -ordering
!   0  chosen package default (or the AMD ordering if no package default)
!   1  Approximate minimum degree (AMD) with provisions for "dense" rows/columns
!   2  Minimum degree
!   3  Nested disection
!   4  indefinite ordering to generate a combination of 1x1 and 2x2 pivots
!   5  Profile/Wavefront reduction
!   6  Bandwidth reduction
!  >6  ordering chosen depending on matrix characteristics (not yet implemented)

       INTEGER :: ordering = 0

!  controls threshold for detecting full rows in analyse, registered as
!  percentage of matrix order. If 100, only fully dense rows detected (default)

       INTEGER :: full_row_threshold = full_row_threshold_default

!  number of rows searched for pivot when using indefinite ordering

       INTEGER :: row_search_indefinite = row_search_indefinite_default

!  controls scaling (ignored if explicit SCALE argument present)
!  <0  calculated internally by package with appropriate scaling -scaling
!   0  No scaling
!   1  Scaling using MC64
!   2  Scaling using MC77 based on the row one-norm
!   3  Scaling using MC77 based on the row infinity-norm

       INTEGER :: scaling = 0

!  the number of scaling iterations performed (default 10 used if
!   %scale_maxit < 0)

       INTEGER :: scale_maxit = 0

!  the scaling iteration stops as soon as the row/column norms are less
!   than 1+/-%scale_thresh

       REAL ( KIND = wp ) :: scale_thresh = 0.1_wp

!  pivot threshold

       REAL ( KIND = wp ) :: relative_pivot_tolerance = 0.01_wp

!  smallest permitted relative pivot threshold

       REAL ( KIND = wp ) :: minimum_pivot_tolerance = 0.01_wp

!  any pivot small than this is considered zero

       REAL ( KIND = wp ) :: absolute_pivot_tolerance = EPSILON( 1.0_wp )

!  any entry smaller than this is considered zero

       REAL ( KIND = wp ) :: zero_tolerance = 0.0_wp

!  any pivot smaller than this is considered zero for positive-definite solvers

       REAL ( KIND = wp ) :: zero_pivot_tolerance = EPSILON( 1.0_wp )

!  any pivot smaller than this is considered to be negative for p-d solvers

       REAL ( KIND = wp ) :: negative_pivot_tolerance                          &
                               = - 0.5_wp * HUGE( 1.0_wp )

!  used for setting static pivot level

       REAL ( KIND = wp ) :: static_pivot_tolerance = 0.0_wp

!  used for switch to static

       REAL ( KIND = wp ) :: static_level_switch = 0.0_wp

!  used to determine whether a system is consistent when seeking a Fredholm
!   alternative

       REAL ( KIND = wp ) :: consistency_tolerance = EPSILON( 1.0_wp )

!  maximum number of iterative refinements allowed

       INTEGER :: max_iterative_refinements = 0

!  refinement will cease as soon as the residual ||Ax-b|| falls below
!     max( acceptable_residual_relative * ||b||, acceptable_residual_absolute )

       REAL ( KIND = wp ) :: acceptable_residual_relative = 10.0_wp * epsmch
       REAL ( KIND = wp ) :: acceptable_residual_absolute = 10.0_wp * epsmch

!  set %multiple_rhs to .true. if there is possibility that the solver
!   will be required to solve systems with more than one right-hand side.
!   More efficient execution may be possible when  %multiple_rhs = .false.

       LOGICAL :: multiple_rhs = .TRUE.

!   if %generate_matrix_file is .true. if a file describing the current
!    matrix is to be generated

        LOGICAL :: generate_matrix_file = .FALSE.

!    specifies the unit number to write the input matrix (in co-ordinate form)

        INTEGER :: matrix_file_device = 74

!  name of generated matrix file containing input problem

        CHARACTER ( LEN = 30 ) :: matrix_file_name =                           &
         "MATRIX.out"  // REPEAT( ' ', 20 )

!  directory name for out of core factorization
!  and additional real workspace in the indefinite case, respectively

       CHARACTER ( LEN = 400 ) :: out_of_core_directory = REPEAT( ' ', 400 )

!  out of core superfile names for integer and real factor data, real workspace
!  and additional real workspace in the indefinite case, respectively

       CHARACTER ( LEN = 400 ) :: out_of_core_integer_factor_file =            &
                                   'factor_integer_ooc' // REPEAT( ' ', 382 )
       CHARACTER ( LEN = 400 ) :: out_of_core_real_factor_file =               &
                                   'factor_real_ooc' // REPEAT( ' ', 385 )
       CHARACTER ( LEN = 400 ) :: out_of_core_real_work_file =                 &
                                   'work_real_ooc' // REPEAT( ' ', 387 )
       CHARACTER ( LEN = 400 ) :: out_of_core_indefinite_file =                &
                                   'work_indefinite_ooc' // REPEAT( ' ', 381 )
       CHARACTER ( LEN = 500 ) :: out_of_core_restart_file =                   &
                                   'restart_ooc' // REPEAT( ' ', 489 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '

     END TYPE SLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLS_time_type

!  the total cpu time spent in the package

       REAL ( KIND = wp ) :: total = 0.0

!  the total cpu time spent in the analysis phase

       REAL ( KIND = wp ) :: analyse = 0.0

!  the total cpu time spent in the factorization phase

       REAL ( KIND = wp ) :: factorize = 0.0

!  the total cpu time spent in the solve phases

       REAL ( KIND = wp ) :: solve = 0.0

!  the total cpu time spent by the external solver in the ordering phase

       REAL ( KIND = wp ) :: order_external = 0.0

!  the total cpu time spent by the external solver in the analysis phase

       REAL ( KIND = wp ) :: analyse_external = 0.0

!  the total cpu time spent by the external solver in the factorization phase

       REAL ( KIND = wp ) :: factorize_external = 0.0

!  the total cpu time spent by the external solver in the solve phases

       REAL ( KIND = wp ) :: solve_external = 0.0

!  the total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0

!  the total clock time spent in the analysis phase

       REAL ( KIND = wp ) :: clock_analyse = 0.0

!  the total clock time spent in the factorization phase

       REAL ( KIND = wp ) :: clock_factorize = 0.0

!  the total clock time spent in the solve phases

       REAL ( KIND = wp ) :: clock_solve = 0.0

!  the total clock time spent by the external solver in the ordering phase

       REAL ( KIND = wp ) :: clock_order_external = 0.0

!  the total clock time spent by the external solver in the analysis phase

       REAL ( KIND = wp ) :: clock_analyse_external = 0.0

!  the total clock time spent by the external solver in the factorization phase

       REAL ( KIND = wp ) :: clock_factorize_external = 0.0

!  the total clock time spent by the external solver in the solve phases

       REAL ( KIND = wp ) :: clock_solve_external = 0.0

     END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -20  alegedly +ve definite matrix is not
!   -29  unavailable option
!   -31  input order is not a permutation or is faulty in some other way
!   -32  > control%max_integer_factor_size integer space required for factors
!   -33  > control%max_real_factor_size real space required for factors
!   -40  not possible to alter the diagonals
!   -41  no access to permutation or pivot sequence used
!   -42  no access to diagonal perturbations
!   -43  direct-access file error
!   -50  solver-specific error; see the solver's info parameter
!  -101  unknown solver

       INTEGER :: status = 0

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  further information on failure

       INTEGER :: more_info = 0

!  number of entries

       INTEGER :: entries = - 1

!  number of indices out-of-range

       INTEGER :: out_of_range = 0

!  number of duplicates

       INTEGER :: duplicates = 0

!  number of entries from the strict upper triangle

       INTEGER :: upper = 0

!  number of missing diagonal entries for an allegedly-definite matrix

       INTEGER :: missing_diagonals = 0

!  maximum depth of the assembly tree

       INTEGER :: max_depth_assembly_tree = - 1

!  nodes in the assembly tree (= number of elimination steps)

       INTEGER :: nodes_assembly_tree = - 1

!  desirable or actual size for real array for the factors and other data

       INTEGER ( KIND = long ) :: real_size_desirable = - 1

!  desirable or actual size for integer array for the factors and other data

       INTEGER ( KIND = long ) :: integer_size_desirable = - 1

!  necessary size for real array for the factors and other data

       INTEGER ( KIND = long ) :: real_size_necessary = - 1

!  necessary size for integer array for the factors and other data

       INTEGER ( KIND = long ):: integer_size_necessary = - 1

!  predicted or actual number of reals to hold factors

       INTEGER ( KIND = long ) :: real_size_factors  = - 1

!  predicted or actual number of integers to hold factors

       INTEGER ( KIND = long ) :: integer_size_factors = - 1

!  number of entries in factors

       INTEGER ( KIND = long ) :: entries_in_factors = - 1_long

!   maximum number of tasks in the factorization task pool

       INTEGER :: max_task_pool_size = - 1

!  forecast or actual size of largest front

       INTEGER :: max_front_size = - 1

!  number of compresses of real data

       INTEGER :: compresses_real = - 1

!  number of compresses of integer data

       INTEGER :: compresses_integer = - 1

!  number of 2x2 pivots

       INTEGER :: two_by_two_pivots = - 1

!  semi-bandwidth of matrix following bandwidth reduction

       INTEGER :: semi_bandwidth = - 1

!  number of delayed pivots (total)

       INTEGER :: delayed_pivots = - 1

!  number of pivot sign changes if no pivoting is used successfully

       INTEGER :: pivot_sign_changes = - 1

!  number of static pivots chosen

       INTEGER :: static_pivots = - 1

!  first pivot modification when static pivoting

       INTEGER :: first_modified_pivot = - 1

!  estimated rank of the matrix

       INTEGER :: rank = - 1

!  number of negative eigenvalues

       INTEGER :: negative_eigenvalues = - 1

!  number of pivots that are considered zero (and ignored)

       INTEGER :: num_zero = 0

!  number of iterative refinements performed

       INTEGER :: iterative_refinements = 0

!  anticipated or actual number of floating-point operations in assembly

       INTEGER ( KIND = long ) :: flops_assembly = - 1_long

!  anticipated or actual number of floating-point operations in elimination

       INTEGER ( KIND = long ) :: flops_elimination = - 1_long

!  additional number of floating-point operations for BLAS

       INTEGER ( KIND = long ) :: flops_blas = - 1_long

!  largest diagonal modification when static pivoting or ensuring definiteness

       REAL ( KIND = wp ) :: largest_modified_pivot = - 1.0_wp

!  minimum scaling factor

       REAL ( KIND = wp ) :: minimum_scaling_factor = 1.0_wp

!  maximum scaling factor

       REAL ( KIND = wp ) :: maximum_scaling_factor = 1.0_wp

!  esimate of the condition number of the matrix (category 1 equations)

       REAL ( KIND = wp ) :: condition_number_1 = - 1.0_wp

!  estimate of the condition number of the matrix (category 2 equations)

       REAL ( KIND = wp ) :: condition_number_2 = - 1.0_wp

!  esimate of the backward error (category 1 equations)

       REAL ( KIND = wp ) :: backward_error_1 = - 1.0_wp

!  esimate of the backward error (category 2 equations)

       REAL ( KIND = wp ) :: backward_error_2 = - 1.0_wp

!  estimate of forward error

       REAL ( KIND = wp ) :: forward_error = - 1.0_wp

!  has an "alternative" y: A y = 0 and yT b > 0 been found when trying to
!  solve A x = b ?

       LOGICAL :: alternative = .FALSE.

!  timings (see above)

        TYPE ( SLS_time_type ) :: time

!  the output structure from sils

       TYPE ( SILS_ainfo ) :: sils_ainfo
       TYPE ( SILS_finfo ) :: sils_finfo
       TYPE ( SILS_sinfo ) :: sils_sinfo

!  the output structure from ma57

       TYPE ( MA57_ainfo ) :: ma57_ainfo
       TYPE ( MA57_finfo ) :: ma57_finfo
       TYPE ( MA57_sinfo ) :: ma57_sinfo

!  the output structure from ma77

       TYPE ( MA77_info ) :: ma77_info

!  the output structure from ma86

       TYPE ( MA86_info ) :: ma86_info

!  the output structure from ma87

       TYPE ( MA87_info ) :: ma87_info

!  the output structure from ma97

       TYPE ( MA97_info ) :: ma97_info

!  the output structure from ssids

       TYPE ( SSIDS_inform ) :: ssids_inform

!  the integer and real output arrays from mc61

       INTEGER, DIMENSION( 10 ) :: mc61_info
       REAL ( KIND = wp ), DIMENSION( 15 ) :: mc61_rinfo

!  the output structure from mc64

       TYPE ( MC64_info ) :: mc64_info

!  the output structure from mc68

       TYPE ( MC68_info ) :: mc68_info

!  the integer output array from mc77

       INTEGER, DIMENSION( 10 ) :: mc77_info

!  the real output status from mc77

        REAL ( KIND = wp ), DIMENSION( 10 ) :: mc77_rinfo

!  the output scalars and arrays from pardiso

       INTEGER :: pardiso_error = 0
       INTEGER, DIMENSION( 64 ) :: pardiso_iparm = - 1
       REAL ( KIND = wp ), DIMENSION( 64 ) :: pardiso_dparm = - 1.0_wp

!  the output scalars and arrays from wsmp

       INTEGER :: wsmp_error = 0
       INTEGER, DIMENSION( 64 ) :: wsmp_iparm = - 1
       REAL ( KIND = wp ), DIMENSION( 64 ) :: wsmp_dparm = - 1.0_wp

!  the output scalars and arrays from LAPACK routines

       INTEGER :: lapack_error = 0

     END TYPE SLS_inform_type

!  ...................
!   data derived type
!  ...................

     TYPE, PUBLIC :: SLS_data_type
       PRIVATE
       INTEGER :: len_solver = - 1
       INTEGER :: n, ne, matrix_ne, matrix_scale_ne, pardiso_mtype, mc61_lirn
       INTEGER :: mc61_liw, mc77_liw, mc77_ldw, sytr_lwork
       CHARACTER ( LEN = len_solver ) :: solver = '                    '
       LOGICAL :: must_be_definite, explicit_scaling, reordered
       INTEGER :: set_res = - 1
       INTEGER :: set_res2 = - 1
       LOGICAL :: got_maps_scale = .FALSE.
       INTEGER, DIMENSION( 64 ) :: PARDISO_PT
       INTEGER, DIMENSION( 64 ) :: pardiso_iparm = - 1
       INTEGER, DIMENSION( 0 ) :: wsmp_aux
       INTEGER, DIMENSION( 64 ) :: wsmp_iparm = 0
       INTEGER, DIMENSION( 10 ) :: mc61_ICNTL                                  &
         = (/ 6, 6, 0, 0, 0, 0, 0, 0, 0, 0 /)
       REAL ( KIND = wp ), DIMENSION( 5 ) :: mc61_CNTL                        &
         = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp /)
       INTEGER, DIMENSION( 10 ) :: mc77_ICNTL                                  &
         = (/ 6, 6, - 1, 0, 0, 1, 10, 0, 0, 0 /)
       REAL ( KIND = wp ), DIMENSION( 10 ) :: mc77_CNTL                        &
         = (/ 0.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp,                          &
              0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp, 0.0_wp /)
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ORDER, MAPS, PIVOTS, MAPS_scale
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INVP, MRP
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: MAP
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: mc64_PERM
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: mc61_IW, mc77_IW
       REAL ( KIND = wp ), DIMENSION( 64 ) :: pardiso_dparm = 0.0_wp
       REAL ( KIND = wp ), DIMENSION( 64 ) :: wsmp_dparm = - 1.0_wp
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RESIDUALS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RESIDUALS_zero
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCALE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: B2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: RES2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: X2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: matrix_dense
       REAL ( KIND = wp ), DIMENSION( 0 : 0 ) :: DIAG
       LOGICAL, ALLOCATABLE, DIMENSION( : ) :: LFLAG

       TYPE ( ZD11_type ) :: matrix, matrix_scale

       TYPE ( SILS_factors ) :: sils_factors
       TYPE ( SILS_control ) :: sils_control
       TYPE ( SILS_ainfo ) :: sils_ainfo
       TYPE ( SILS_finfo ) :: sils_finfo
       TYPE ( SILS_sinfo ) :: sils_sinfo

       TYPE ( MA57_factors ) :: ma57_factors
       TYPE ( MA57_control ) :: ma57_control
       TYPE ( MA57_ainfo ) :: ma57_ainfo
       TYPE ( MA57_finfo ) :: ma57_finfo
       TYPE ( MA57_sinfo ) :: ma57_sinfo

       TYPE ( MA77_keep ) :: ma77_keep
       TYPE ( MA77_control ) :: ma77_control
       TYPE ( MA77_info ) :: ma77_info

       TYPE ( MA86_keep ) :: ma86_keep
       TYPE ( MA86_control ) :: ma86_control
       TYPE ( MA86_info ) :: ma86_info

       TYPE ( MA87_keep ) :: ma87_keep
       TYPE ( MA87_control ) :: ma87_control
       TYPE ( MA87_info ) :: ma87_info

       TYPE ( MA97_akeep ) :: ma97_akeep
       TYPE ( MA97_fkeep ) :: ma97_fkeep
       TYPE ( MA97_control ) :: ma97_control
       TYPE ( MA97_info ) :: ma97_info

       TYPE ( SSIDS_akeep ) :: ssids_akeep
       TYPE ( SSIDS_fkeep ) :: ssids_fkeep
       TYPE ( SSIDS_options ) :: ssids_options
       TYPE ( SSIDS_inform ) :: ssids_inform

       TYPE ( MC64_control ) :: mc64_control
       TYPE ( MC64_info ) :: mc64_info

       TYPE ( MC68_control ) :: mc68_control
       TYPE ( MC68_info ) :: mc68_info

     END TYPE SLS_data_type

!----------------------------------
!   I n t e r f a c e  B l o c k s
!----------------------------------

     INTERFACE SLS_solve
       MODULE PROCEDURE SLS_solve_ir, SLS_solve_ir_multiple
     END INTERFACE


   CONTAINS

!-*-*-*-*-*-   S L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE SLS_initialize( solver, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Set initial values, including default control data and solver used, for SLS.
!  This routine must be called before the first call to SLS_analyse

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( SLS_data_type ), INTENT( OUT ) :: data
     TYPE ( SLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLS_inform_type ), INTENT( OUT ) :: inform

!  initialize the solver-specific data

     CALL SLS_initialize_solver( solver, data, inform )

!  initialize solver-specific controls

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       IF ( control%ordering == 0 )                                            &
         control%ordering = - data%sils_control%ordering
       IF ( control%scaling == 0 )                                             &
         control%scaling = - data%sils_control%scaling
       control%relative_pivot_tolerance = data%sils_control%u

!  = MA57 =

     CASE ( 'ma57' )
       IF ( control%ordering == 0 )                                            &
         control%ordering = - data%ma57_control%ordering
       IF ( control%scaling == 0 )                                             &
         control%scaling = - data%ma57_control%scaling
       control%relative_pivot_tolerance = data%ma57_control%u

!  = MA77 =

     CASE ( 'ma77' )
!IS64  control%max_in_core_store = HUGE( 0_long ) / real_bytes

!  = MA86 =

     CASE ( 'ma86' )
       control%absolute_pivot_tolerance = data%ma86_control%small
!86V2  IF ( control%scaling == 0 )                                             &
!86V2    control%scaling = - data%ma86_control%scaling

!  = MA87 =

     CASE ( 'ma87' )
       control%zero_pivot_tolerance = SQRT( EPSILON( 1.0_wp ) )

!  = MA97 =

     CASE ( 'ma97' )
       IF ( control%scaling == 0 )                                             &
         control%scaling = - data%ma97_control%scaling
!      control%node_amalgamation = 8
       IF ( control%ordering == 0 )                                            &
         control%ordering = - data%ma97_control%ordering

!  = SSIDS =

     CASE ( 'ssids' )
       IF ( control%scaling == 0 )                                             &
         control%scaling = - data%ssids_options%scaling
!      control%node_amalgamation = 8
       IF ( control%ordering == 0 )                                            &
         control%ordering = - data%ssids_options%ordering

!  = PARDISO =

     CASE ( 'pardiso' )
       control%node_amalgamation = 80

!  = WSMP =

     CASE ( 'wsmp' )

!  = POTR =

     CASE ( 'potr' )

!  = SYTR =

     CASE ( 'sytr' )

!  = PBTR =

     CASE ( 'pbtr' )
       IF ( control%ordering == 0 ) control%ordering = 6

     END SELECT

     RETURN

!  End of SLS_initialize

     END SUBROUTINE SLS_initialize

!-*-*-   S L S _ I N I T I A L I Z E _ S O L V E R  S U B R O U T I N E   -*-*-

     SUBROUTINE SLS_initialize_solver( solver, data, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Set initial values, including default control data and solver used, for SLS.
!  This routine must be called before the first call to SLS_analyse

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_inform_type ), INTENT( OUT ) :: inform

!  record the solver

     data%len_solver = MIN( len_solver, LEN_TRIM( solver ) )
     data%solver( 1 : data%len_solver ) = solver( 1 : data%len_solver )
     CALL STRING_lower_word( data%solver( 1 : data%len_solver ) )

     data%set_res = - 1 ; data%set_res2 = - 1
     data%got_maps_scale = .FALSE.
     inform%status = GALAHAD_ok

!  initialize solver-specific controls

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       data%must_be_definite = .FALSE.
       CALL SILS_initialize( FACTORS = data%sils_factors,                     &
                             CONTROL = data%sils_control )

!  = MA57 =

     CASE ( 'ma57' )
       data%must_be_definite = .FALSE.
       CALL MA57_initialize( factors = data%ma57_factors,                     &
                             control = data%ma57_control )

!  = MA77 =

     CASE ( 'ma77' )
       data%must_be_definite = .FALSE.

!  = MA86 =

     CASE ( 'ma86' )
       data%must_be_definite = .FALSE.

!  = MA87 =

     CASE ( 'ma87' )
       data%must_be_definite = .TRUE.

!  = MA97 =

     CASE ( 'ma97' )
       data%must_be_definite = .FALSE.

!  = SSIDS =

     CASE ( 'ssids' )
       data%must_be_definite = .FALSE.

!  = PARDISO =

     CASE ( 'pardiso' )
       data%must_be_definite = .FALSE.

!  = WSMP =

     CASE ( 'wsmp' )
       data%must_be_definite = .FALSE.

!  = POTR =

     CASE ( 'potr' )
       data%must_be_definite = .TRUE.

!  = SYTR =

     CASE ( 'sytr' )
       data%must_be_definite = .FALSE.

!  = PBTR =

     CASE ( 'pbtr' )
       data%must_be_definite = .TRUE.

!  = unavailable solver =

     CASE DEFAULT
       inform%status = GALAHAD_error_unknown_solver
     END SELECT

     RETURN

!  End of SLS_initialize_solver

     END SUBROUTINE SLS_initialize_solver

!-*-   S L S _ C O P Y _ C O N T R O L _ T O _ S I L S  S U B R O U T I N E  -*-

     SUBROUTINE SLS_copy_control_to_sils( control, control_sils )

!  copy control parameters to their SILS equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SILS_control ), INTENT( INOUT ) :: control_sils

     IF ( control%print_level_solver > 0 ) THEN
       control_sils%lp = control%error
       control_sils%wp = control%warning
       control_sils%mp = control%out
       control_sils%sp = control%statistics
     ELSE
       control_sils%lp = - 1
       control_sils%wp = - 1
       control_sils%mp = - 1
       control_sils%sp = - 1
     END IF
     control_sils%ldiag = control%print_level_solver
     control_sils%factorblocking = control%blas_block_size_factorize
     IF ( control_sils%factorblocking < 1 )                                    &
       control_sils%factorblocking = blas_block_size_factor_default
     control_sils%solveblocking = control%blas_block_size_solve
     IF ( control_sils%solveblocking < 1 )                                     &
       control_sils%solveblocking = blas_block_size_solve_default
     control_sils%la = MAX( control%min_real_factor_size, 1 )
     IF ( control_sils%la < 1 )                                                &
        control_sils%la = control%min_real_factor_size
     control_sils%liw = MAX( control%min_integer_factor_size, 1 )
     IF ( control_sils%liw < 1 )                                               &
       control_sils%liw = min_integer_factor_size_default
     control_sils%maxla = INT( control%max_real_factor_size )
     control_sils%maxliw = INT( control%max_integer_factor_size )
     control_sils%pivoting = control%pivot_control
     control_sils%thresh = control%full_row_threshold
     IF ( control_sils%thresh < 1 .OR.                                         &
          control_sils%thresh > 100 )                                          &
       control_sils%thresh = full_row_threshold_default
     IF ( control%ordering < 0 ) control_sils%ordering = - control%ordering
     IF ( control%scaling <= 0 ) control_sils%scaling = - control%scaling
     control_sils%u= control%relative_pivot_tolerance
     control_sils%multiplier = control%array_increase_factor
     control_sils%reduce = control%array_decrease_factor
     control_sils%static_tolerance= control%static_pivot_tolerance
     control_sils%static_level = control%static_level_switch
     control_sils%tolerance = control%zero_tolerance

     RETURN

!  End of SLS_copy_control_to_sils

     END SUBROUTINE SLS_copy_control_to_sils

!-*-   S L S _ C O P Y _ C O N T R O L _ T O _ M A 5 7  S U B R O U T I N E  -*-

     SUBROUTINE SLS_copy_control_to_ma57( control, control_ma57 )

!  copy control parameters to their MA57 equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( MA57_control ), INTENT( INOUT ) :: control_ma57


     IF ( control%print_level_solver > 0 ) THEN
       control_ma57%lp = control%error
       control_ma57%wp = control%warning
       control_ma57%mp = control%out
       control_ma57%sp = control%statistics
     ELSE
       control_ma57%lp = - 1
       control_ma57%wp = - 1
       control_ma57%mp = - 1
       control_ma57%sp = - 1
     END IF
     control_ma57%ldiag = control%print_level_solver
     control_ma57%factorblocking = control%blas_block_size_factorize
     IF ( control_ma57%factorblocking < 1 )                                    &
       control_ma57%factorblocking = blas_block_size_factor_default
     control_ma57%solveblocking = control%blas_block_size_solve
     IF ( control_ma57%solveblocking < 1 )                                     &
       control_ma57%solveblocking = blas_block_size_solve_default
     control_ma57%la = control%min_real_factor_size
     IF ( control_ma57%la < 1 )                                                &
        control_ma57%la = control%min_real_factor_size
     control_ma57%liw = control%min_integer_factor_size
     IF ( control_ma57%liw < 1 )                                               &
       control_ma57%liw = min_integer_factor_size_default
     control_ma57%maxla = INT( control%max_real_factor_size )
     control_ma57%maxliw = INT( control%max_integer_factor_size )
     control_ma57%pivoting = control%pivot_control
     control_ma57%thresh = control%full_row_threshold
     IF ( control_ma57%thresh < 1 .OR.                                         &
          control_ma57%thresh > 100 )                                          &
       control_ma57%thresh = full_row_threshold_default
     IF ( control%ordering < 0 ) control_ma57%ordering = - control%ordering
     IF ( control%scaling <= 0 ) control_ma57%scaling = - control%scaling
     control_ma57%u= control%relative_pivot_tolerance
     control_ma57%multiplier = control%array_increase_factor
     control_ma57%reduce = control%array_decrease_factor
     control_ma57%static_tolerance= control%static_pivot_tolerance
     control_ma57%static_level = control%static_level_switch
     control_ma57%tolerance = control%zero_tolerance
     control_ma57%consist = control%consistency_tolerance

     RETURN

!  End of SLS_copy_control_to_ma57

     END SUBROUTINE SLS_copy_control_to_ma57

!-*-   S L S _ C O P Y _ C O N T R O L _ T O _ M A 7 7  S U B R O U T I N E  -*-

     SUBROUTINE SLS_copy_control_to_ma77( control, control_ma77 )

!  copy control parameters to their MA77 equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( MA77_control ), INTENT( INOUT ) :: control_ma77

     IF ( control%print_level_solver > 0 ) THEN
       control_ma77%unit_error = control%error
       control_ma77%unit_warning = control%warning
       control_ma77%unit_diagnostics = control%out
     ELSE
       control_ma77%unit_error = - 1
       control_ma77%unit_warning = - 1
       control_ma77%unit_diagnostics = - 1
     END IF
     control_ma77%print_level = control%print_level_solver
     control_ma77%small = control%absolute_pivot_tolerance
     control_ma77%u = control%relative_pivot_tolerance
     control_ma77%umin = control%minimum_pivot_tolerance
     control_ma77%nemin = control%node_amalgamation
     IF ( control_ma77%nemin < 1 )                                             &
       control_ma77%nemin = node_amalgamation_default
     control_ma77%nbi = control%block_size_kernel
     IF ( control_ma77%nbi < 1 )                                               &
       control_ma77%nbi = block_size_kernel_default
     control_ma77%bits = control%bits
     IF ( control_ma77%bits /= 64 ) control_ma77%bits = bits_default
!IS64control_ma77%bits = 64 ! replace the "!IS64" with space if 64bit arch.
     control_ma77%maxstore = control%max_in_core_store
     control_ma77%consist_tol = control%consistency_tolerance

     RETURN

!  End of SLS_copy_control_to_ma77

     END SUBROUTINE SLS_copy_control_to_ma77

!-*-   S L S _ C O P Y _ C O N T R O L _ T O _ M A 8 6  S U B R O U T I N E  -*-

     SUBROUTINE SLS_copy_control_to_ma86( control, control_ma86 )

!  copy control parameters to their MA86 equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( MA86_control ), INTENT( INOUT ) :: control_ma86

     IF ( control%print_level_solver > 0 ) THEN
       control_ma86%unit_error = control%error
       control_ma86%unit_warning = control%warning
       control_ma86%unit_diagnostics = control%out
     ELSE
       control_ma86%unit_error = - 1
       control_ma86%unit_warning = - 1
       control_ma86%unit_diagnostics = - 1
     END IF
     control_ma86%diagnostics_level = control%print_level_solver
     control_ma86%nemin = control%node_amalgamation
     control_ma86%nbi = block_size_kernel_default
     IF ( control_ma86%nemin < 1 )                                             &
       control_ma86%nemin = node_amalgamation_default
     control_ma86%nb = control%block_size_elimination
     IF ( control_ma86%nb < 1 )                                                &
       control_ma86%nb = block_size_elimination_default
     control_ma86%pool_size = control%initial_pool_size
     IF ( control_ma86%pool_size < 1 )                                         &
        control_ma86%pool_size = initial_pool_size_default
     control_ma86%small = control%absolute_pivot_tolerance
     IF ( control%scaling == - 1 ) THEN
!86V2  control_ma86%scaling = 1
     ELSE
!86V2  control_ma86%scaling = 0
     END IF
     IF ( control%pivot_control == 2 ) THEN
       control_ma86%static = 0.0_wp
       control_ma86%u = 0.0_wp
       control_ma86%umin = 0.0_wp
       control_ma86%action = .TRUE.
     ELSE IF ( control%pivot_control == 3 ) THEN
       control_ma86%static = 0.0_wp
       control_ma86%u = 0.0_wp
       control_ma86%umin = 0.0_wp
       control_ma86%action = .FALSE.
     ELSE IF ( control%pivot_control == 4 ) THEN
       control_ma86%static = control%static_pivot_tolerance
       control_ma86%u = 0.0_wp
       control_ma86%umin = 0.0_wp
       control_ma86%action = .TRUE.
     ELSE
       control_ma86%static = 0.0_wp
       control_ma86%u = control%relative_pivot_tolerance
       control_ma86%umin = control%minimum_pivot_tolerance
       control_ma86%action = .TRUE.
     END IF

     RETURN

!  End of SLS_copy_control_to_ma86

     END SUBROUTINE SLS_copy_control_to_ma86

!-*-   S L S _ C O P Y _ C O N T R O L _ T O _ M A 8 7  S U B R O U T I N E  -*-

     SUBROUTINE SLS_copy_control_to_ma87( control, control_ma87 )

!  copy control parameters to their MA87 equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( MA87_control ), INTENT( INOUT ) :: control_ma87

     IF ( control%print_level_solver > 0 ) THEN
       control_ma87%unit_error = control%error
       control_ma87%unit_warning = control%warning
       control_ma87%unit_diagnostics = control%out
     ELSE
       control_ma87%unit_error = - 1
       control_ma87%unit_warning = - 1
       control_ma87%unit_diagnostics = - 1
     END IF
     control_ma87%diagnostics_level = control%print_level_solver
     control_ma87%nemin = control%node_amalgamation
     IF ( control_ma87%nemin < 1 )                                             &
       control_ma87%nemin = node_amalgamation_default
     control_ma87%nb = control%block_size_elimination
     IF ( control_ma87%nb < 1 )                                                &
       control_ma87%nb = block_size_elimination_default
     control_ma87%pool_size = control%initial_pool_size
     IF ( control_ma87%pool_size < 1 )                                         &
        control_ma87%pool_size = initial_pool_size_default
     IF ( control%pivot_control == 4 ) THEN
        control_ma87%diag_zero_plus = control%zero_pivot_tolerance
        control_ma87%diag_zero_minus = control%negative_pivot_tolerance
!write(6,*) ' diag_zero_plus ', control_ma87%diag_zero_plus
!write(6,*) ' diag_zero_minus ', control_ma87%diag_zero_minus
     END IF

     RETURN

!  End of SLS_copy_control_to_ma87

     END SUBROUTINE SLS_copy_control_to_ma87

!-*-   S L S _ C O P Y _ C O N T R O L _ T O _ M A 9 7  S U B R O U T I N E  -*-

     SUBROUTINE SLS_copy_control_to_ma97( control, control_ma97 )

!  copy control parameters to their MA97 equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( MA97_control ), INTENT( INOUT ) :: control_ma97

     IF ( control%print_level_solver > 0 ) THEN
       control_ma97%unit_error = control%error
       control_ma97%unit_warning = control%warning
       control_ma97%unit_diagnostics = control%out
     ELSE
       control_ma97%unit_error = - 1
       control_ma97%unit_warning = - 1
       control_ma97%unit_diagnostics = - 1
     END IF
     control_ma97%print_level = control%print_level_solver
     control_ma97%nemin = control%node_amalgamation
     IF ( control%scaling == - 1 ) THEN
       control_ma97%scaling = 1
     ELSE IF ( control%scaling == - 2 ) THEN
       control_ma97%scaling = 2
     ELSE IF ( control%scaling == - 3 ) THEN
       control_ma97%scaling = 3
     ELSE
       control_ma97%scaling = 0
     END IF
     control_ma97%small = control%absolute_pivot_tolerance
     control_ma97%consist_tol = control%consistency_tolerance
     IF ( control%pivot_control == 2 ) THEN
       control_ma97%u = 0.0_wp
       control_ma97%action = .TRUE.
     ELSE IF ( control%pivot_control == 3 ) THEN
       control_ma97%u = 0.0_wp
       control_ma97%action = .FALSE.
     ELSE IF ( control%pivot_control == 4 ) THEN
       control_ma97%u = 0.0_wp
       control_ma97%action = .TRUE.
     ELSE
       control_ma97%u = control%relative_pivot_tolerance
       control_ma97%action = .TRUE.
     END IF

     RETURN

!  End of SLS_copy_control_to_ma97

     END SUBROUTINE SLS_copy_control_to_ma97

!-   S L S _ C O P Y _ C O N T R O L _ T O _ S S I D S  S U B R O U T I N E  -

     SUBROUTINE SLS_copy_control_to_ssids( control, control_ssids )

!  copy control parameters to their SSIDS equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SSIDS_options ), INTENT( INOUT ) :: control_ssids

     IF ( control%print_level_solver > 0 ) THEN
       control_ssids%unit_error = control%error
       control_ssids%unit_warning = control%warning
       control_ssids%unit_diagnostics = control%out
     ELSE
       control_ssids%unit_error = - 1
       control_ssids%unit_warning = - 1
       control_ssids%unit_diagnostics = - 1
     END IF
     control_ssids%print_level = control%print_level_solver
     control_ssids%nemin = control%node_amalgamation
     IF ( control%scaling == - 1 ) THEN
       control_ssids%scaling = 1
     ELSE IF ( control%scaling == - 2 ) THEN
       control_ssids%scaling = 2
     ELSE IF ( control%scaling == - 3 ) THEN
       control_ssids%scaling = 3
     ELSE
       control_ssids%scaling = 0
     END IF
     control_ssids%small = control%absolute_pivot_tolerance
!    control_ssids%presolve = 0
!    control_ssids%consist_tol = control%consistency_tolerance
     IF ( control%pivot_control == 2 ) THEN
       control_ssids%u = 0.0_wp
!      control_ssids%presolve = 1
       control_ssids%action = .TRUE.
     ELSE IF ( control%pivot_control == 3 ) THEN
       control_ssids%u = 0.0_wp
!      control_ssids%action = .TRUE.
!      control_ssids%presolve = 1
       control_ssids%action = .FALSE.
     ELSE IF ( control%pivot_control == 4 ) THEN
       control_ssids%u = 0.0_wp
       control_ssids%action = .TRUE.
     ELSE
       control_ssids%u = control%relative_pivot_tolerance
       control_ssids%action = .TRUE.
     END IF
!    IF ( control%multiple_rhs ) control_ssids%presolve = 1

     RETURN

!  End of SLS_copy_control_to_ssids

     END SUBROUTINE SLS_copy_control_to_ssids

!-  S L S _ C O P Y _ C O N T R O L _ T O _ P A R D I S O  S U B R O U T I N E -

     SUBROUTINE SLS_copy_control_to_pardiso( control, iparm )

!  copy control parameters to their PARDISO equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: iparm

     IF ( control%ordering <= 0 ) THEN
       iparm( 2 ) = 2
!    ELSE IF ( control%ordering == 0 ) THEN
!      iparm( 2 ) = 0
     END IF
     iparm( 8 ) = control%max_iterative_refinements
     IF ( control%pivot_control == 1 ) THEN
       iparm( 21 ) = 1
     ELSE
       iparm( 21 ) = 0
     END IF
!    iparm( 30 ) = control%node_amalgamation

     RETURN

!  End of SLS_copy_control_to_pardiso

     END SUBROUTINE SLS_copy_control_to_pardiso

!-*-  S L S _ C O P Y _ C O N T R O L _ T O _ W S M P  S U B R O U T I N E -*-

     SUBROUTINE SLS_copy_control_to_wsmp( control, iparm, dparm )

!  copy control parameters to their WSMP equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: iparm
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( 64 ) :: dparm

!  set default values as defined by the wsmp documentation

!    iparm = 0
!    iparm( 5 ) = 1
!    iparm( 6 ) = 1
!    iparm( 7 ) = 3
!    iparm( 16 ) = 1
!    iparm( 18 ) = 1

!    dparm = 0.0_wp
!    dparm( 6 ) = 10.0_wp * EPSILON( 1.0_wp )
!    dparm( 10 ) = 10.0_wp ** ( - 18 )
!    dparm( 11 ) = 10.0_wp ** ( - 3 )
!    dparm( 12 ) = 2.0_wp * EPSILON( 1.0_wp )
!    dparm( 21 ) = 10.0_wp ** 200
!    dparm( 22 ) = SQRT( 2.0_wp * EPSILON( 1.0_wp ) )

     iparm( 6 ) = control%max_iterative_refinements
     IF ( control%ordering < 0 ) THEN
       iparm( 8 ) = - control%ordering
       iparm( 16 ) = - control%ordering
     ELSE IF ( control%ordering == 0 ) THEN
!      iparm( 8 ) = 2
       iparm( 16 ) = - 1
     END IF
     IF ( control%pivot_control == 2 ) THEN
       iparm( 31 ) = 0
       iparm( 11 ) = 0
     ELSE IF ( control%pivot_control == 3 ) THEN
       iparm( 31 ) = 1
       iparm( 11 ) = 0
     ELSE IF ( control%pivot_control == 4 ) THEN
       iparm( 31 ) = 5
       iparm( 11 ) = 2
       iparm( 13 ) = - 1
     ELSE
       iparm( 31 ) = 2
       iparm( 11 ) = 2
       iparm( 13 ) = - 1
     END IF
     IF ( control%scaling < 0 ) THEN
       iparm( 10 ) = - control%scaling
     ELSE IF ( control%scaling == 0 ) THEN
       IF ( iparm( 31 ) == 2 .OR. iparm( 31 ) == 4 ) THEN
         iparm( 10 ) = 2
       ELSE
         iparm( 10 ) = 2
       END IF
     ELSE
       iparm( 10 ) = 2
     END IF
     iparm( 20 ) = 1
     IF ( control%blas_block_size_factorize < 1 ) THEN
       iparm( 26 ) = 0
     ELSE
       iparm( 26 ) = control%blas_block_size_factorize
     END IF
!    iparm( 27 ) = 1
     dparm( 6 ) = control%acceptable_residual_relative
!    dparm( 10 ) = 0.0_wp
     dparm( 11 ) = control%relative_pivot_tolerance

     RETURN

!  End of SLS_copy_control_to_wsmp

     END SUBROUTINE SLS_copy_control_to_wsmp

!-*   S L S _ C O P Y _ I N F O R M _ F R O M _ M A 7 7  S U B R O U T I N E  *-

     SUBROUTINE SLS_copy_inform_from_ma77( inform, info_ma77 )

!  copy inform parameters from their MA77 equivalents

!  Dummy arguments

     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( MA77_info ), INTENT( IN ) :: info_ma77

     inform%ma77_info = info_ma77
     inform%status = info_ma77%flag
     SELECT CASE( inform%status )
     CASE ( 0 : )
       inform%status = GALAHAD_ok
       inform%two_by_two_pivots = info_ma77%ntwo
       inform%rank = info_ma77%matrix_rank
       inform%negative_eigenvalues = info_ma77%num_neg
       inform%static_pivots = info_ma77%num_perturbed
       inform%delayed_pivots = info_ma77%ndelay
       inform%entries_in_factors = info_ma77%nfactor
       inform%flops_elimination = info_ma77%nflops
       inform%max_front_size  = info_ma77%maxfront
       inform%max_depth_assembly_tree = info_ma77%maxdepth
     CASE ( - 1 )
       inform%status = GALAHAD_error_allocate
       inform%alloc_status = info_ma77%stat
     CASE ( - 8 )
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = info_ma77%stat
     CASE( -4, - 18, - 19, - 20, - 24, - 25, - 33, -36, - 37, - 38 )
       inform%status = GALAHAD_error_restrictions
     CASE ( - 5, - 6, - 7, - 12, - 13, - 15, - 16, - 23, - 26, - 27, - 28 )
       inform%status = GALAHAD_error_direct_access
     CASE ( - 21 )
       inform%status = GALAHAD_error_permutation
     CASE ( - 11, - 29  )
       inform%status = GALAHAD_error_inertia
     CASE ( - 100 + GALAHAD_unavailable_option  )
       inform%status = GALAHAD_unavailable_option
     CASE DEFAULT
       inform%status = GALAHAD_error_technical
     END SELECT

     RETURN

!  End of SLS_copy_inform_from_ma77

     END SUBROUTINE SLS_copy_inform_from_ma77

!-*   S L S _ C O P Y _ I N F O R M _ F R O M _ M A 9 7  S U B R O U T I N E  *-

     SUBROUTINE SLS_copy_inform_from_ma97( inform, info_ma97 )

!  copy inform parameters from their MA97 equivalents

!  Dummy arguments

     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( MA97_info ), INTENT( IN ) :: info_ma97

     inform%ma97_info = info_ma97
     inform%status = info_ma97%flag
     SELECT CASE( inform%status )
     CASE ( 0 : )
       inform%status = GALAHAD_ok
       inform%duplicates = info_ma97%matrix_dup
       inform%out_of_range = info_ma97%matrix_outrange
       inform%two_by_two_pivots = info_ma97%num_two
       inform%rank = info_ma97%matrix_rank
       inform%negative_eigenvalues = info_ma97%num_neg
!      inform%static_pivots = info_ma97%num_perturbed
       inform%delayed_pivots = info_ma97%num_delay
       inform%entries_in_factors = info_ma97%num_factor
       inform%flops_elimination = info_ma97%num_flops
       inform%max_front_size  = info_ma97%maxfront
       inform%max_depth_assembly_tree = info_ma97%maxdepth
     CASE ( - 30  )
       inform%status = GALAHAD_error_allocate
       inform%alloc_status = info_ma97%stat
     CASE ( - 31  )
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = info_ma97%stat
     CASE( - 1, - 2, - 3, - 4, - 5, - 6, - 9, - 10, - 12, - 13, - 14, - 15 )
       inform%status = GALAHAD_error_restrictions
     CASE ( - 11 )
       inform%status = GALAHAD_error_permutation
     CASE ( - 7, - 8  )
       inform%status = GALAHAD_error_inertia
     CASE ( - 32, GALAHAD_unavailable_option  )
       inform%status = GALAHAD_unavailable_option
     CASE DEFAULT
       inform%status = GALAHAD_error_technical
     END SELECT

     RETURN

!  End of SLS_copy_inform_from_ma97

     END SUBROUTINE SLS_copy_inform_from_ma97

!-   S L S _ C O P Y _ I N F O R M _ F R O M _ S S I D S  S U B R O U T I N E  -

     SUBROUTINE SLS_copy_inform_from_ssids( inform, info_ssids )

!  copy inform parameters from their SSIDS equivalents

!  Dummy arguments

     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     TYPE ( SSIDS_inform ), INTENT( IN ) :: info_ssids

     inform%ssids_inform = info_ssids
     inform%status = info_ssids%flag
     SELECT CASE( inform%status )
     CASE ( 0 : )
       inform%status = GALAHAD_ok
       inform%duplicates = info_ssids%matrix_dup
       inform%out_of_range = info_ssids%matrix_outrange
       inform%two_by_two_pivots = info_ssids%num_two
       inform%rank = info_ssids%matrix_rank
       inform%negative_eigenvalues = info_ssids%num_neg
!      inform%static_pivots = info_ssids%num_perturbed
       inform%delayed_pivots = info_ssids%num_delay
       inform%entries_in_factors = info_ssids%num_factor
       inform%flops_elimination = info_ssids%num_flops
       inform%max_front_size  = info_ssids%maxfront
       inform%max_depth_assembly_tree = info_ssids%maxdepth
     CASE ( - 30  )
       inform%status = GALAHAD_error_allocate
       inform%alloc_status = info_ssids%stat
     CASE ( - 31  )
       inform%status = GALAHAD_error_deallocate
       inform%alloc_status = info_ssids%stat
     CASE( - 1, - 2, - 3, - 4, - 5, - 6, - 9, - 10, - 12, - 13, - 14, - 15 )
       inform%status = GALAHAD_error_restrictions
     CASE ( - 11 )
       inform%status = GALAHAD_error_permutation
     CASE ( - 7, - 8  )
       inform%status = GALAHAD_error_inertia
     CASE ( - 32, GALAHAD_unavailable_option  )
       inform%status = GALAHAD_unavailable_option
     CASE ( GALAHAD_error_unknown_solver  )
       inform%status = GALAHAD_error_unknown_solver
     CASE DEFAULT
       inform%status = GALAHAD_error_technical
     END SELECT

     RETURN

!  End of SLS_copy_inform_from_ssids

     END SUBROUTINE SLS_copy_inform_from_ssids

!-*-*-*-*-   S L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE SLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SLS_initialize could (roughly)
!  have been set as:

! BEGIN SLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  warning-printout-device                           6
!  printout-device                                   6
!  statistics-printout-device                        0
!  print-level                                       0
!  print-level-solver                                0
!  architecture-bits                                 32
!  block-size-for-kernel                             40
!  block-size-for--elimination                       32
!  blas-block-for-size-factorize                     16
!  blas-block-size-for-solve                         16
!  node-amalgamation-toleranace                      256
!  initial-pool-size                                 100000
!  minimum-real-factor-size                          10000
!  minimum-integer-factor-size                       10000
!  maximum-real-factor-size                          2147483647
!  maximum-integer-factor-size                       2147483647
!  maximum-in-core-store                             268435455
!  pivot-control                                     1
!  ordering                                          0
!  full-row-threshold                                100
!  pivot-row-search-when-indefinite                  10
!  scaling                                           0
!  scale-maxit                                       0
!  scale-thresh                                      0.1
!  max-iterative-refinements                         0
!  array-increase-factor                             2.0
!  array-decrease-factor                             2.0
!  relative-pivot-tolerance                          0.01
!  minimum-pivot-tolerance                           0.01
!  absolute-pivot-tolerance                          2.0D-16
!  zero-tolerance                                    0.0
!  zero-pivot-tolerance                              1.0D-10
!  negative-pivot-tolerance                          -1.0D+50
!  static-pivot-tolerance                            0.0
!  static-level-switch                               0.0
!  consistency-tolerance                             2.0D-16
!  acceptable-residual-relative                      2.0D-15
!  acceptable-residual-absolute                      2.0D-15
!  possibly-use-multiple-rhs                         YES
!  generate-matrix-file                              NO
!  matrix-file-device                                74
!  matrix-file-name                                  MATRIX.out
!  out-of-core-directory
!  out-of-core-integer-factor-file                   factor_integer_ooc
!  out-of-core-real-factor-file                      factor_real_ooc
!  out-of-core-real-work-file                        work_real_ooc
!  out-of-core-indefinite-file                       work_indefinite_ooc
!  out-of-core-restart-file                          restart_ooc
!  output-line-prefix                                ""
! END SLS SPECIFICATIONS

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: warning = error + 1
     INTEGER, PARAMETER :: out = warning + 1
     INTEGER, PARAMETER :: statistics = out + 1
     INTEGER, PARAMETER :: print_level = statistics + 1
     INTEGER, PARAMETER :: print_level_solver = print_level + 1
     INTEGER, PARAMETER :: block_size_kernel = print_level_solver + 1
     INTEGER, PARAMETER :: bits = block_size_kernel + 1
     INTEGER, PARAMETER :: block_size_elimination = bits + 1
     INTEGER, PARAMETER :: blas_block_size_factorize =                         &
                             block_size_elimination + 1
     INTEGER, PARAMETER :: blas_block_size_solve = blas_block_size_factorize + 1
     INTEGER, PARAMETER :: node_amalgamation = blas_block_size_solve + 1
     INTEGER, PARAMETER :: initial_pool_size = node_amalgamation + 1
     INTEGER, PARAMETER :: min_real_factor_size = initial_pool_size + 1
     INTEGER, PARAMETER :: min_integer_factor_size = min_real_factor_size + 1
     INTEGER, PARAMETER :: max_real_factor_size = min_integer_factor_size + 1
     INTEGER, PARAMETER :: max_integer_factor_size = max_real_factor_size + 1
     INTEGER, PARAMETER :: max_in_core_store = max_integer_factor_size + 1
     INTEGER, PARAMETER :: pivot_control = max_in_core_store + 1
     INTEGER, PARAMETER :: ordering = pivot_control + 1
     INTEGER, PARAMETER :: full_row_threshold = ordering + 1
     INTEGER, PARAMETER :: row_search_indefinite = full_row_threshold + 1
     INTEGER, PARAMETER :: scaling = row_search_indefinite + 1
     INTEGER, PARAMETER :: scale_maxit = scaling + 1
     INTEGER, PARAMETER :: scale_thresh = scale_maxit + 1
     INTEGER, PARAMETER :: max_iterative_refinements = scale_thresh + 1
     INTEGER, PARAMETER :: array_increase_factor = max_iterative_refinements + 1
     INTEGER, PARAMETER :: array_decrease_factor = array_increase_factor + 1
     INTEGER, PARAMETER :: relative_pivot_tolerance = array_decrease_factor + 1
     INTEGER, PARAMETER :: minimum_pivot_tolerance =                           &
                             relative_pivot_tolerance + 1
     INTEGER, PARAMETER :: absolute_pivot_tolerance =                          &
                             minimum_pivot_tolerance + 1
     INTEGER, PARAMETER :: zero_tolerance = absolute_pivot_tolerance + 1
     INTEGER, PARAMETER :: zero_pivot_tolerance = zero_tolerance + 1
     INTEGER, PARAMETER :: negative_pivot_tolerance = zero_pivot_tolerance + 1
     INTEGER, PARAMETER :: static_pivot_tolerance = negative_pivot_tolerance + 1
     INTEGER, PARAMETER :: static_level_switch = static_pivot_tolerance + 1
     INTEGER, PARAMETER :: consistency_tolerance = static_level_switch + 1
     INTEGER, PARAMETER :: matrix_file_device = consistency_tolerance + 1
     INTEGER, PARAMETER :: multiple_rhs = matrix_file_device + 1
     INTEGER, PARAMETER :: generate_matrix_file = multiple_rhs + 1
     INTEGER, PARAMETER :: matrix_file_name = generate_matrix_file + 1
     INTEGER, PARAMETER :: acceptable_residual_relative =                      &
                              matrix_file_name + 1
     INTEGER, PARAMETER :: acceptable_residual_absolute =                      &
                             acceptable_residual_relative + 1
     INTEGER, PARAMETER :: out_of_core_directory =                             &
                             acceptable_residual_absolute + 1
     INTEGER, PARAMETER :: out_of_core_integer_factor_file =                   &
                             out_of_core_directory + 1
     INTEGER, PARAMETER :: out_of_core_real_factor_file =                      &
                             out_of_core_integer_factor_file + 1
     INTEGER, PARAMETER :: out_of_core_real_work_file =                        &
                             out_of_core_real_factor_file + 1
     INTEGER, PARAMETER :: out_of_core_indefinite_file =                       &
                             out_of_core_real_work_file + 1
     INTEGER, PARAMETER :: out_of_core_restart_file =                          &
                             out_of_core_indefinite_file + 1
     INTEGER, PARAMETER :: prefix = out_of_core_restart_file + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 3 ), PARAMETER :: specname = 'SLS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( warning )%keyword = 'warning-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( statistics )%keyword = 'statistics-printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( print_level_solver )%keyword = 'print-level-solver'
     spec( bits )%keyword = 'architecture-bits'
     spec( block_size_kernel )%keyword = 'block-size-for-kernel'
     spec( block_size_elimination )%keyword = 'block-size-for--elimination'
     spec( blas_block_size_factorize )%keyword = 'blas-block-for-size-factorize'
     spec( blas_block_size_solve )%keyword = 'blas-block-size-for-solve'
     spec( node_amalgamation )%keyword = 'node-amalgamation-tolerance'
     spec( initial_pool_size )%keyword = 'initial-pool-size'
     spec( min_real_factor_size )%keyword = 'minimum-real-factor-size'
     spec( min_integer_factor_size )%keyword = 'minimum-integer-factor-size'
     spec( max_real_factor_size )%keyword = 'maximum-real-factor-size'
     spec( max_integer_factor_size )%keyword = 'maximum-integer-factor-size'
     spec( max_in_core_store )%keyword = 'maximum-in-core-store'
     spec( pivot_control )%keyword = 'pivot-control'
     spec( ordering )%keyword = 'ordering'
     spec( full_row_threshold )%keyword = 'full-row-threshold'
     spec( row_search_indefinite )%keyword = 'pivot-row-search-when-indefinite'
     spec( scaling )%keyword = 'scaling'
     spec( scale_maxit )%keyword = 'scale_maxit'
     spec( max_iterative_refinements )%keyword = 'max-iterative-refinements'
     spec( matrix_file_device )%keyword = 'matrix-file-device'

!  Real key-words

     spec( array_increase_factor )%keyword = 'array-increase-factor'
     spec( array_decrease_factor )%keyword = 'array-decrease-factor'
     spec( relative_pivot_tolerance )%keyword = 'relative-pivot-tolerance'
     spec( minimum_pivot_tolerance )%keyword = 'minimum-pivot-tolerance'
     spec( absolute_pivot_tolerance )%keyword = 'absolute-pivot-tolerance'
     spec( zero_tolerance )%keyword = 'zero-tolerance'
     spec( zero_pivot_tolerance )%keyword = 'zero-pivot-tolerance'
     spec( negative_pivot_tolerance )%keyword = 'negative-pivot-tolerance'
     spec( scale_thresh )%keyword = 'scale-thresh'
     spec( static_pivot_tolerance )%keyword = 'static-pivot-tolerance'
     spec( static_level_switch )%keyword = 'static-level-switch'
     spec( consistency_tolerance )%keyword = 'consistency-tolerance'
     spec( acceptable_residual_relative )%keyword                              &
       = 'acceptable-residual-relative'
     spec( acceptable_residual_absolute )%keyword                              &
       = 'acceptable-residual-absolute'

!  Logical key-words

    spec( multiple_rhs )%keyword = 'possibly-use-multiple-rhs'
    spec( generate_matrix_file )%keyword = 'generate-matrix-file'

!  Character key-words

     spec( matrix_file_name )%keyword = 'matrix-file-name'
     spec( out_of_core_directory )%keyword =                                   &
       'out-of-core-directory'
     spec( out_of_core_integer_factor_file )%keyword =                         &
       'out-of-core-integer-factor-file'
     spec( out_of_core_real_factor_file )%keyword  =                           &
       'out-of-core-real-factor-file'
     spec( out_of_core_real_work_file )%keyword  =                             &
        'out-of-core-real-work-file'
     spec( out_of_core_indefinite_file )%keyword  =                            &
        'out-of-core-indefinite-file'
     spec( out_of_core_restart_file )%keyword  =                               &
        'out-of-core-restart-file'
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
     CALL SPECFILE_assign_value( spec( warning ),                              &
                                 control%warning,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out ),                                  &
                                 control%out,                                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( statistics ),                           &
                                 control%statistics,                           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level_solver ),                   &
                                 control%print_level_solver,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( bits ),                                 &
                                 control%bits,                                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( block_size_kernel ),                    &
                                 control%block_size_kernel,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( block_size_elimination ),               &
                                 control%block_size_elimination,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( blas_block_size_factorize ),            &
                                 control%blas_block_size_factorize,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( blas_block_size_solve ),                &
                                 control%blas_block_size_solve,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( node_amalgamation ),                    &
                                 control%node_amalgamation,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_pool_size ),                    &
                                 control%initial_pool_size,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_real_factor_size ),                 &
                                 control%min_real_factor_size,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_integer_factor_size ),              &
                                 control%min_integer_factor_size,              &
                                 control%error )
     CALL SPECFILE_assign_long ( spec( max_real_factor_size ),                 &
                                 control%max_real_factor_size,                 &
                                 control%error )
     CALL SPECFILE_assign_long ( spec( max_integer_factor_size ),              &
                                 control%max_integer_factor_size,              &
                                 control%error )
     CALL SPECFILE_assign_long ( spec( max_in_core_store ),                    &
                                 control%max_in_core_store,                    &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivot_control ),                        &
                                 control%pivot_control,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( ordering ),                             &
                                 control%ordering,                             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( full_row_threshold ),                   &
                                 control%full_row_threshold,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( row_search_indefinite ),                &
                                 control%row_search_indefinite,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scaling ),                              &
                                 control%scaling,                              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_maxit ),                          &
                                 control%scale_maxit,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_iterative_refinements ),            &
                                 control%max_iterative_refinements,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( matrix_file_device ),                   &
                                 control%matrix_file_device,                   &
                                 control%error )

!  Set real values

     CALL SPECFILE_assign_value( spec( array_increase_factor ),                &
                                 control%array_increase_factor,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( array_decrease_factor ),                &
                                 control%array_decrease_factor,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( relative_pivot_tolerance ),             &
                                 control%relative_pivot_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minimum_pivot_tolerance ),              &
                                 control%minimum_pivot_tolerance,              &
                                 control%error )
     CALL SPECFILE_assign_value( spec( absolute_pivot_tolerance ),             &
                                 control%absolute_pivot_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_tolerance ),                       &
                                 control%zero_tolerance,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_pivot_tolerance ),                 &
                                 control%zero_pivot_tolerance,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( negative_pivot_tolerance ),             &
                                 control%negative_pivot_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( scale_thresh ),                         &
                                 control%scale_thresh,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( static_pivot_tolerance ),               &
                                 control%static_pivot_tolerance,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( static_level_switch ),                  &
                                 control%static_level_switch,                  &
                                 control%error )
     CALL SPECFILE_assign_value( spec( consistency_tolerance ),                &
                                 control%consistency_tolerance,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( acceptable_residual_relative ),         &
                                 control%acceptable_residual_relative,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( acceptable_residual_absolute ),         &
                                 control%acceptable_residual_absolute,         &
                                 control%error )

!  Set logical values

     CALL SPECFILE_assign_value( spec( multiple_rhs ),                         &
                                 control%multiple_rhs,                         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( generate_matrix_file ),                 &
                                 control%generate_matrix_file,                 &
                                 control%error )
!  Set character value

     CALL SPECFILE_assign_value( spec( matrix_file_name ),                     &
                                 control%matrix_file_name,                     &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out_of_core_directory ),                &
                                 control%out_of_core_directory,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out_of_core_integer_factor_file ),      &
                                 control%out_of_core_integer_factor_file,      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out_of_core_real_factor_file ),         &
                                 control%out_of_core_real_factor_file,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out_of_core_real_work_file ),           &
                                 control%out_of_core_real_work_file,           &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out_of_core_indefinite_file ),          &
                                 control%out_of_core_indefinite_file,          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( out_of_core_restart_file ),             &
                                 control%out_of_core_restart_file,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )
     RETURN

!  End of SLS_read_specfile

     END SUBROUTINE SLS_read_specfile

!-*-*-*-*-*-*-*-   S L S _ A N A L Y S E   S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SLS_analyse( matrix, data, control, inform, PERM )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Analyse the sparsity pattern to obtain a good potential ordering
!  for any subsequent factorization

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER, INTENT( IN ), OPTIONAL :: PERM( matrix%n )

!  local variables

     INTEGER :: i, j, k, l, l1, l2, ordering, pardiso_solver
     REAL :: time, time_start, time_now
     REAL ( KIND = wp ) :: clock, clock_start, clock_now
     LOGICAL :: mc6168_ordering
     CHARACTER ( LEN = 400 ), DIMENSION( 1 ) :: path
     CHARACTER ( LEN = 400 ), DIMENSION( 4 ) :: filename
     INTEGER :: ILAENV
     EXTERNAL :: ILAENV
!$   INTEGER :: OMP_GET_NUM_THREADS

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  check input data

     IF ( matrix%n < 1 .OR.                                                    &
         ( matrix%ne < 0 .AND. SMT_get( matrix%type ) == 'COORDINATE' ) .OR.   &
         .NOT. SLS_keyword( matrix%type ) ) THEN
       inform%status = GALAHAD_error_restrictions
       data%n = 0
       GO TO 900
     ELSE
       data%n = matrix%n
     END IF

     IF ( control%pivot_control == 2 .OR. control%pivot_control == 3 )         &
       data%must_be_definite = .TRUE.

     SELECT CASE ( SMT_get( matrix%type ) )
     CASE ( 'COORDINATE' )
       data%matrix_ne = matrix%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%matrix_ne = matrix%PTR( matrix%n + 1 ) - 1
     CASE ( 'DENSE' )
       data%matrix_ne = matrix%n * ( matrix%n + 1 ) / 2
     END SELECT
     data%matrix%ne = data%matrix_ne
     inform%entries = data%matrix_ne
     inform%semi_bandwidth = matrix%n - 1

!  check that any "permutation" presented is actually a permutation

     IF ( PRESENT( PERM ) ) THEN
       CALL SPACE_resize_array( matrix%n, data%ORDER,                          &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%ORDER' ; GO TO 900 ; END IF
       data%ORDER = 0
       DO i = 1, matrix%n
         l = PERM( i )
         IF ( l < 1 .OR. l > matrix%n ) THEN
           inform%status = GALAHAD_error_permutation ; GO TO 900 ;  END IF
         data%ORDER( l ) = 1
       END DO
       IF ( COUNT( data%ORDER /= 1 ) /= 0 ) THEN
         inform%status = GALAHAD_error_permutation ; GO TO 900 ; END IF
     END IF

!  decide if the ordering should be chosen by mc68

     SELECT CASE( data%solver( 1 : data%len_solver ) )
     CASE ( 'ma77', 'ma86', 'ma87', 'ma97', 'ssids' )
       mc6168_ordering = control%ordering >= 0 .AND. .NOT. PRESENT( PERM )
     CASE DEFAULT
       mc6168_ordering = control%ordering > 0 .AND. .NOT. PRESENT( PERM )
     END SELECT

!  convert the data to sorted compressed-sparse row format

     IF ( mc6168_ordering ) THEN
       CALL SPACE_resize_array( data%matrix_ne, data%MAPS,                     &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%MAPS' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( matrix%n + 1, data%matrix%PTR,                 &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%PTR' ; GO TO 900 ; END IF

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         CALL SLS_coord_to_sorted_csr( matrix%n, matrix%ne,                    &
                                       matrix%ROW,  matrix%COL,                &
                                       data%MAPS, data%matrix%PTR,             &
                                       inform%duplicates,                      &
                                       inform%out_of_range, inform%upper,      &
                                       inform%missing_diagonals,               &
                                       inform%status, inform%alloc_status )
       CASE ( 'SPARSE_BY_ROWS' )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,             &
                                  inform%status, inform%alloc_status )
         DO i = 1, matrix%n
           data%matrix%ROW( matrix%PTR( i ) : matrix%PTR( i + 1 ) - 1 ) = i
         END DO
         CALL SLS_coord_to_sorted_csr( matrix%n, data%matrix_ne,               &
                                       data%matrix%ROW, matrix%COL,            &
                                       data%MAPS, data%matrix%PTR,             &
                                       inform%duplicates,                      &
                                       inform%out_of_range, inform%upper,      &
                                       inform%missing_diagonals,               &
                                       inform%status, inform%alloc_status )
       CASE ( 'DENSE' )
         CALL SMT_put( data%matrix%type, 'DENSE', inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,             &
                                  inform%status, inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,             &
                                  inform%status, inform%alloc_status )
         l = 0
         DO i = 1, matrix%n
           DO j = 1, i
             l = l + 1
             data%matrix%ROW( l ) = i ; data%matrix%COL( l ) = j
           END DO
         END DO
         CALL SLS_coord_to_sorted_csr( matrix%n, data%matrix_ne,               &
                                       data%matrix%ROW, data%matrix%COL,       &
                                       data%MAPS, data%matrix%PTR,             &
                                       inform%duplicates,                      &
                                       inform%out_of_range, inform%upper,      &
                                       inform%missing_diagonals,               &
                                       inform%status, inform%alloc_status )
       END SELECT
       IF ( inform%status /= GALAHAD_ok ) GO TO 900

       IF ( data%must_be_definite .AND. inform%missing_diagonals > 0 ) THEN
         inform%status = GALAHAD_error_inertia
         GO TO 900
       END IF

!  now map the column data

       data%matrix%n = matrix%n
       data%ne = data%matrix%PTR( matrix%n + 1 ) - 1
       data%matrix%ne = data%ne
!      CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS', inform%alloc_status )
       CALL SPACE_resize_array( data%ne, data%matrix%ROW, inform%status,       &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%ROW' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%ne, data%matrix%COL, inform%status,       &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%COL' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%ne, data%matrix%VAL, inform%status,       &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF

       DO i = 1, matrix%n
         l = data%matrix%PTR( i )
         data%matrix%COL( l ) = i
       END DO

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, matrix%ne
           k = data%MAPS( l )
           IF ( k > 0 ) THEN
             data%matrix%ROW( k ) = MIN( matrix%ROW( l ), matrix%COL( l ) )
             data%matrix%COL( k ) = MAX( matrix%ROW( l ), matrix%COL( l ) )
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             k = data%MAPS( l )
             IF ( k > 0 ) THEN
               data%matrix%ROW( k ) = MIN( i, matrix%COL( l ) )
               data%matrix%COL( k ) = MAX( i, matrix%COL( l ) )
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, matrix%n
           DO j = 1, i
             l = l + 1
             k = data%MAPS( l )
             IF ( k > 0 ) THEN
               data%matrix%ROW( k ) = MIN( i, j )
               data%matrix%COL( k ) = MAX( i, j )
             END IF
           END DO
         END DO
       END SELECT

!  now compute the required ordering

       CALL ZD11_put( data%matrix%type, 'pattern', inform%alloc_status )
       CALL SPACE_resize_array( matrix%n, data%ORDER,                          &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%ORDER' ; GO TO 900 ; END IF

       IF ( control%ordering > 6 .OR.control%ordering < 1 ) THEN
         IF ( data%solver( 1 : data%len_solver ) == 'pbtr' ) THEN
           ordering = 6
         ELSE
           ordering = 1
         END IF
       ELSE
         ordering = control%ordering
       END IF

!  mc61 band ordering

       IF ( ordering > 4 ) THEN
         data%mc61_lirn = 2 * ( data%matrix%PTR( data%matrix%n + 1 ) - 1 )
         data%mc61_liw = 8 * data%matrix%n + 2

         CALL SPACE_resize_array( data%mc61_liw, data%mc61_IW,                 &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mc61_IW' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( data%matrix%n, data%WORK,                    &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%WORK' ; GO TO 900 ; END IF

         CALL SPACE_resize_array( data%mc61_lirn, data%MRP,                    &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MRP' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n + 1, data%INVP,                     &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%INVP' ; GO TO 900 ; END IF

         data%INVP( : data%matrix%n + 1 ) =                                    &
           data%matrix%PTR( : data%matrix%n + 1 )
         data%MRP( : data%matrix%PTR( data%matrix%n + 1 ) - 1 ) =              &
           data%matrix%COL( : data%matrix%PTR( data%matrix%n + 1 ) - 1 )
         IF ( control%print_level <= 0 .OR. control%out <= 0 )                 &
           data%mc61_ICNTL( 1 : 2 ) = - 1
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL MC61AD( ordering - 4, data%matrix%n, data%mc61_lirn, data%MRP,   &
                      data%INVP, data%ORDER, data%mc61_liw, data%mc61_IW,      &
                      data%WORK, data%mc61_ICNTL, data%mc61_CNTL,              &
                      inform%mc61_info, inform%mc61_rinfo )
         CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%order_external = time_now - time
         inform%time%clock_order_external = clock_now - clock
         IF ( inform%mc61_info( 1 ) == GALAHAD_unavailable_option ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' mc61 is not available ' )" ) prefix
           inform%status = GALAHAD_error_unknown_solver ; GO TO 900
         END IF
         inform%semi_bandwidth = INT( inform%mc61_rinfo( 7 ) ) - 1

!  mc68 sparsity ordering

       ELSE
         data%mc68_control%row_full_thresh = control%full_row_threshold
         IF ( data%mc68_control%row_full_thresh < 1 .OR.                       &
              data%mc68_control%row_full_thresh > 100 )                        &
            data%mc68_control%row_full_thresh = full_row_threshold_default
         IF ( ordering == 4 ) THEN
           data%mc68_control%row_search = control%row_search_indefinite
           IF ( data%mc68_control%row_search < 1 )                             &
             data%mc68_control%row_search = row_search_indefinite_default
         END IF
         IF ( control%print_level <= 0 .OR. control%out <= 0 )                 &
           data%mc68_control%lp = - 1

         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL MC68_order( ordering,                                            &
                          data%matrix%n, data%matrix%PTR, data%matrix%COL,     &
                          data%ORDER, data%mc68_control, data%mc68_info )
         CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%order_external = time_now - time
         inform%time%clock_order_external = clock_now - clock
         IF ( data%mc68_info%flag == - 9 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' MeTiS is not available ' )" ) prefix
           inform%status = GALAHAD_unavailable_option ; GO TO 900
         ELSE IF ( data%mc68_info%flag == GALAHAD_unavailable_option ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' mc68 is not available ' )" ) prefix
           inform%status = GALAHAD_error_unknown_solver ; GO TO 900
         END IF
       END IF
       CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS',                       &
                     inform%alloc_status )
     END IF

!  solver-dependent analysis

!write(6,*) data%solver( 1 : data%len_solver )
     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS or MA57 =

     CASE ( 'sils', 'ma27', 'ma57' )

!  if the input matrix is not in co-ordinate form, make a copy

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'SPARSE_BY_ROWS' )
         data%matrix%n = matrix%n
         data%matrix%ne = matrix%PTR( matrix%n + 1 ) - 1
         CALL SMT_put( data%matrix%type, 'COORDINATE', inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,             &
                                  inform%status, inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,             &
                                  inform%status, inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%VAL,             &
                                  inform%status, inform%alloc_status )
         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             data%matrix%ROW( l ) = i
             data%matrix%COL( l ) = matrix%COL( l )
           END DO
         END DO
       CASE ( 'DENSE' )
         data%matrix%n = matrix%n
         data%matrix%ne = matrix%n * ( matrix%n + 1 ) / 2
         CALL SMT_put( data%matrix%type, 'COORDINATE', inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,             &
                                  inform%status, inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,             &
                                  inform%status, inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%VAL,             &
                                  inform%status, inform%alloc_status )
         l = 0
         DO i = 1, matrix%n
           DO j = 1, i
             l = l + 1
             data%matrix%ROW( l ) = i ; data%matrix%COL( l ) = j
           END DO
         END DO
       END SELECT

       IF ( .NOT. mc6168_ordering ) THEN
         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           inform%upper = COUNT( matrix%ROW( : matrix%ne ) <                   &
                                 matrix%COL( : matrix%ne ) )
         CASE ( 'SPARSE_BY_ROWS' )
           inform%upper = 0
           DO i = 1, matrix%n
             DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
               IF ( i < matrix%COL( l ) ) inform%upper = inform%upper + 1
             END DO
           END DO
         END SELECT
       END IF
       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

       CASE ( 'sils', 'ma27' )
         CALL SLS_copy_control_to_sils( control, data%sils_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         IF ( mc6168_ordering ) THEN
           SELECT CASE ( SMT_get( MATRIX%type ) )
           CASE ( 'COORDINATE' )
             CALL SILS_analyse( matrix, data%sils_factors, data%sils_control,  &
                                data%sils_ainfo, data%ORDER )
           CASE DEFAULT
             CALL SILS_analyse( data%matrix, data%sils_factors,                &
                                data%sils_control,                             &
                                data%sils_ainfo, data%ORDER )
           END SELECT
         ELSE
           IF ( PRESENT( PERM ) ) data%sils_control%ordering = 1
           SELECT CASE ( SMT_get( MATRIX%type ) )
           CASE ( 'COORDINATE' )
             CALL SILS_analyse( matrix, data%sils_factors, data%sils_control,  &
                                data%sils_ainfo, PERM )
           CASE DEFAULT
             CALL SILS_analyse( data%matrix, data%sils_factors,                &
                                data%sils_control,                             &
                                data%sils_ainfo, PERM )
           END SELECT
         END IF
         inform%sils_ainfo = data%sils_ainfo
         inform%status = data%sils_ainfo%flag
         IF ( inform%status == - 1 .OR. inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%sils_ainfo%stat
         ELSE IF ( inform%status == - 9 ) THEN
           inform%status = GALAHAD_error_permutation
         ELSE
           inform%status = GALAHAD_ok
           inform%alloc_status = data%sils_ainfo%stat
           inform%more_info = data%sils_ainfo%more
           inform%nodes_assembly_tree = data%sils_ainfo%nsteps
           inform%real_size_desirable = INT( data%sils_ainfo%nrltot, long )
           inform%integer_size_desirable = INT( data%sils_ainfo%nirtot, long )
           inform%real_size_necessary = INT( data%sils_ainfo%nrlnec, long )
           inform%integer_size_necessary = INT( data%sils_ainfo%nirnec, long )
           inform%entries_in_factors = INT( data%sils_ainfo%nrladu, long )
           inform%real_size_factors = INT( data%sils_ainfo%nrladu, long )
           inform%integer_size_factors = INT( data%sils_ainfo%niradu, long )
           inform%compresses_integer = data%sils_ainfo%ncmpa
           inform%out_of_range = data%sils_ainfo%oor
           inform%duplicates = data%sils_ainfo%dup
           inform%max_front_size = data%sils_ainfo%maxfrt
           inform%flops_assembly = INT( data%sils_ainfo%opsa, long )
           inform%flops_elimination = INT( data%sils_ainfo%opse, long )
         END IF

!  = MA57 =

       CASE ( 'ma57' )
         CALL SLS_copy_control_to_ma57( control, data%ma57_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         IF ( mc6168_ordering ) THEN
           data%ma57_control%ordering = 1
           SELECT CASE ( SMT_get( MATRIX%type ) )
           CASE ( 'COORDINATE' )
             CALL MA57_analyse( matrix, data%ma57_factors, data%ma57_control,  &
                                data%ma57_ainfo, data%ORDER )
           CASE DEFAULT
             CALL MA57_analyse( data%matrix, data%ma57_factors,                &
                                data%ma57_control,                             &
                                data%ma57_ainfo, data%ORDER )
           END SELECT
         ELSE
           IF ( PRESENT( PERM ) ) data%ma57_control%ordering = 1
           SELECT CASE ( SMT_get( MATRIX%type ) )
           CASE ( 'COORDINATE' )
             CALL MA57_analyse( matrix, data%ma57_factors, data%ma57_control,  &
                                data%ma57_ainfo, PERM )
           CASE DEFAULT
             CALL MA57_analyse( data%matrix, data%ma57_factors,                &
                                data%ma57_control,                             &
                                data%ma57_ainfo, PERM )
           END SELECT
         END IF
         inform%ma57_ainfo = data%ma57_ainfo
         inform%status = data%ma57_ainfo%flag
         IF ( inform%status == - 1 .OR. inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma57_ainfo%stat
         ELSE IF ( inform%status == - 9 ) THEN
           inform%status = GALAHAD_error_permutation
         ELSE IF ( inform%status == - 10 ) THEN
           inform%status = GALAHAD_error_metis
         ELSE IF ( inform%status == GALAHAD_unavailable_option ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' ma57 is not available ' )" ) prefix
           inform%status = GALAHAD_error_unknown_solver
         ELSE
           inform%status = GALAHAD_ok
           inform%more_info = data%ma57_ainfo%more
           inform%nodes_assembly_tree = data%ma57_ainfo%nsteps
           inform%real_size_desirable = INT( data%ma57_ainfo%nrltot, long )
           inform%integer_size_desirable = INT( data%ma57_ainfo%nirtot, long )
           inform%real_size_necessary = INT( data%ma57_ainfo%nrlnec, long )
           inform%integer_size_necessary = INT( data%ma57_ainfo%nirnec, long )
           inform%entries_in_factors = INT( data%ma57_ainfo%nrladu, long )
           inform%real_size_factors = INT( data%ma57_ainfo%nrladu, long )
           inform%integer_size_factors = INT( data%ma57_ainfo%niradu, long )
           inform%compresses_integer = data%ma57_ainfo%ncmpa
           inform%out_of_range = data%ma57_ainfo%oor
           inform%duplicates = data%ma57_ainfo%dup
           inform%max_front_size = data%ma57_ainfo%maxfrt
           inform%flops_assembly = INT( data%ma57_ainfo%opsa, long )
           inform%flops_elimination = INT( data%ma57_ainfo%opse, long )
         END IF
       END SELECT

!  = MA77 =

     CASE ( 'ma77' )

!  if the extended matrix has not yet been constructed, do so

!      IF ( mc6168_ordering ) THEN
!        CALL SPACE_resize_array( data%matrix%PTR( matrix%n + 1 ) - 1,         &
!                   data%matrix%ROW, inform%status, inform%alloc_status )
!        IF ( inform%status /= GALAHAD_ok ) THEN
!          inform%bad_alloc = 'sls: data%matrix%ROW' ; GO TO 900 ; END IF
!        data%matrix%ROW( : data%matrix%PTR( matrix%n + 1 ) - 1 ) =            &
!          data%matrix%COL( : data%matrix%PTR( matrix%n + 1 ) - 1 )
!        CALL SPACE_dealloc_array( data%matrix%COL, inform%status,             &
!                                  inform%alloc_status )

!  compute the map to move the matrix into its extended form

!      ELSE
         data%matrix%ne = data%matrix_ne
         CALL SPACE_resize_array( data%matrix_ne, 2, data%MAP,                 &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MAP' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n + 1, data%matrix%PTR,               &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%PTR' ; GO TO 900 ; END IF

         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           CALL SLS_coord_to_extended_csr( matrix%n, matrix%ne, matrix%ROW,    &
                                           matrix%COL, data%MAP,               &
                                           data%matrix%PTR, inform%duplicates, &
                                           inform%out_of_range, inform%upper,  &
                                           inform%missing_diagonals )
         CASE ( 'SPARSE_BY_ROWS' )
           CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS',                   &
                         inform%alloc_status )
           CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,           &
                                    inform%status, inform%alloc_status )
           DO i = 1, matrix%n
             data%matrix%ROW( matrix%PTR( i ) : matrix%PTR( i + 1 ) - 1 ) = i
           END DO
           CALL SLS_coord_to_extended_csr( matrix%n, data%matrix%ne,           &
                                           data%matrix%ROW,                    &
                                           matrix%COL, data%MAP,               &
                                           data%matrix%PTR, inform%duplicates, &
                                           inform%out_of_range, inform%upper,  &
                                           inform%missing_diagonals )
         CASE ( 'DENSE' )
           CALL SMT_put( data%matrix%type, 'DENSE', inform%alloc_status )
           CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,           &
                                    inform%status, inform%alloc_status )
           CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,           &
                                    inform%status, inform%alloc_status )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               data%matrix%ROW( l ) = i ; data%matrix%COL( l ) = j
             END DO
           END DO
           CALL SLS_coord_to_extended_csr( matrix%n, data%matrix%ne,           &
                                           data%matrix%ROW,                    &
                                           data%matrix%COL, data%MAP,          &
                                           data%matrix%PTR, inform%duplicates, &
                                           inform%out_of_range, inform%upper,  &
                                           inform%missing_diagonals )
         END SELECT
         IF ( data%must_be_definite .AND. inform%missing_diagonals > 0 ) THEN
           inform%status = GALAHAD_error_inertia
           GO TO 900
         END IF

!  extend the column data

         data%matrix%n = matrix%n
         data%matrix%ne = data%matrix%PTR( matrix%n + 1 ) - 1
         CALL SPACE_resize_array( data%matrix%ne,                              &
                    data%matrix%ROW, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%ROW' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( data%matrix%ne,                              &
                    data%matrix%VAL, inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF

         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, matrix%ne
             k = data%MAP( l, 1 )
             IF ( k > 0 ) data%matrix%ROW( k ) = matrix%COL( l )
             k = data%MAP( l, 2 )
             IF ( k > 0 ) data%matrix%ROW( k ) = matrix%ROW( l )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, matrix%n
             DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
               k = data%MAP( l, 1 )
               IF ( k > 0 ) data%matrix%ROW( k ) = matrix%COL( l )
               k = data%MAP( l, 2 )
               IF ( k > 0 ) data%matrix%ROW( k ) = i
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               k = data%MAP( l, 1 )
               IF ( k > 0 ) data%matrix%ROW( k ) = j
               k = data%MAP( l, 2 )
               IF ( k > 0 ) data%matrix%ROW( k ) = i
             END DO
           END DO
         END SELECT

!  specify the order to be used

         CALL SPACE_resize_array( matrix%n, data%ORDER,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%ORDER' ; GO TO 900 ; END IF
         IF ( PRESENT( PERM ) ) THEN
           data%ORDER( : matrix%n ) = PERM( : matrix%n )
         ELSE
           DO i = 1, matrix%n
             data%ORDER( i ) = i
           END DO
         END IF
!      END IF

!  open the direct access files

       filename( 1 ) = control%out_of_core_integer_factor_file
       filename( 2 ) = control%out_of_core_real_factor_file
       filename( 3 ) = control%out_of_core_real_work_file
       filename( 4 ) = control%out_of_core_indefinite_file
       DO i = 1, 2
         CALL SLS_copy_control_to_ma77( control, data%ma77_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         IF ( TRIM( control%out_of_core_directory ) == '' ) THEN
           CALL MA77_open( data%n, filename, data%ma77_keep,                   &
                           data%ma77_control, data%ma77_info )
         ELSE
           path( 1 ) = REPEAT( ' ', 400 )
           path( 1 ) = control%out_of_core_directory
           CALL MA77_open( data%n, filename, data%ma77_keep,                   &
                           data%ma77_control, data%ma77_info, path = path )
         END IF
         IF (  data%ma77_info%flag /= - 3 ) EXIT
         CALL MA77_finalise( data%ma77_keep, data%ma77_control, data%ma77_info )
       END DO

       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       IF (  data%ma77_info%flag == GALAHAD_unavailable_option ) THEN
         IF ( control%print_level > 0 .AND. control%out > 0 )                  &
           WRITE( control%out, "( A, ' ma77 is not available ' )" ) prefix
         inform%status = GALAHAD_error_unknown_solver
       END IF
       IF ( inform%status /= GALAHAD_ok ) GO TO 800

!  input the rows, one at a time

       DO i = 1, data%n
         l1 = data%matrix%PTR( i )
         l2 = data%matrix%PTR( i + 1 ) - 1
         CALL MA77_input_vars( i, l2 - l1 + 1, data%matrix%ROW( l1 : l2 ),     &
                               data%ma77_keep, data%ma77_control,              &
                               data%ma77_info )
         CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
       END DO
       CALL MA77_analyse( data%ORDER( : data%n ), data%ma77_keep,              &
                          data%ma77_control, data%ma77_info )
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )

!  = MA86, MA87, MA97, SSIDS, PARDISO or WSMP =

     CASE ( 'ma86', 'ma87', 'ma97', 'ssids', 'pardiso', 'wsmp' )

!  convert the data to sorted compressed-sparse row format

       IF ( .NOT. mc6168_ordering ) THEN
         CALL SPACE_resize_array( data%matrix_ne, data%MAPS,                   &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MAPS' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n + 1, data%matrix%PTR,               &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%PTR' ; GO TO 900 ; END IF
         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           CALL SLS_coord_to_sorted_csr( matrix%n, matrix%ne,                  &
                                         matrix%ROW,  matrix%COL,              &
                                         data%MAPS, data%matrix%PTR,           &
                                         inform%duplicates,                    &
                                         inform%out_of_range, inform%upper,    &
                                         inform%missing_diagonals,             &
                                         inform%status, inform%alloc_status )
         CASE ( 'SPARSE_BY_ROWS' )
           CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS',                   &
                         inform%alloc_status )
           CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,           &
                                    inform%status, inform%alloc_status )
           DO i = 1, matrix%n
             data%matrix%ROW( matrix%PTR( i ) : matrix%PTR( i + 1 ) - 1 ) = i
           END DO
           CALL SLS_coord_to_sorted_csr( matrix%n, data%matrix_ne,             &
                                         data%matrix%ROW, matrix%COL,          &
                                         data%MAPS, data%matrix%PTR,           &
                                         inform%duplicates,                    &
                                         inform%out_of_range, inform%upper,    &
                                         inform%missing_diagonals,             &
                                         inform%status, inform%alloc_status )
         CASE ( 'DENSE' )
           CALL SMT_put( data%matrix%type, 'DENSE', inform%alloc_status )
           CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,           &
                                    inform%status, inform%alloc_status )
           CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,           &
                                    inform%status, inform%alloc_status )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               data%matrix%ROW( l ) = i ; data%matrix%COL( l ) = j
             END DO
           END DO
           CALL SLS_coord_to_sorted_csr( matrix%n, data%matrix_ne,             &
                                         data%matrix%ROW, data%matrix%COL,     &
                                         data%MAPS, data%matrix%PTR,           &
                                         inform%duplicates,                    &
                                         inform%out_of_range, inform%upper,    &
                                         inform%missing_diagonals,             &
                                         inform%status, inform%alloc_status )
         END SELECT
         IF ( inform%status /= GALAHAD_ok ) GO TO 900

         data%ne = data%matrix%PTR( matrix%n + 1 ) - 1
         CALL SPACE_resize_array( data%ne, data%matrix%COL, inform%status,     &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%COL' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( data%ne, data%matrix%VAL, inform%status,     &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF

!  now map the coordinates

         data%matrix%n = matrix%n
         data%matrix%ne = data%ne
         CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS', inform%alloc_status )

         DO i = 1, matrix%n
           l = data%matrix%PTR( i )
           data%matrix%COL( l ) = i
         END DO

         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           DO l = 1, matrix%ne
             k = data%MAPS( l )
             IF ( k > 0 )                                                      &
               data%matrix%COL( k ) = MAX( matrix%ROW( l ), matrix%COL( l ) )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, matrix%n
             DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
               k = data%MAPS( l )
               IF ( k > 0 ) data%matrix%COL( k ) = MAX( i, matrix%COL( l ) )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               k = data%MAPS( l )
               IF ( k > 0 ) data%matrix%ROW( k ) = MIN( i, j )
               IF ( k > 0 ) data%matrix%COL( k ) = MAX( i, j )
             END DO
           END DO
         END SELECT

!  specify the order to be used

         CALL SPACE_resize_array( matrix%n, data%ORDER,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%ORDER' ; GO TO 900 ; END IF
         IF ( PRESENT( PERM ) ) THEN
           data%ORDER( : matrix%n ) = PERM( : matrix%n )
         ELSE
           DO i = 1, matrix%n
             data%ORDER( i ) = i
           END DO
         END IF
       ELSE
         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
         CASE ( 'SPARSE_BY_ROWS' )
         CASE ( 'DENSE' )
!          CALL SMT_put( data%matrix%type, 'DENSE', inform%alloc_status )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               k = data%MAPS( l )
               IF ( k > 0 ) data%matrix%ROW( k ) = MIN( i, j )
               IF ( k > 0 ) data%matrix%COL( k ) = MAX( i, j )
             END DO
           END DO
         END SELECT
       END IF

       CALL SPACE_resize_array( data%ne, data%matrix%VAL,                      &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( matrix%n, 1, data%B2,                          &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF

       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = MA86 =

       CASE ( 'ma86' )
         CALL SLS_copy_control_to_ma86( control, data%ma86_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL MA86_analyse( data%matrix%n, data%matrix%PTR, data%matrix%COL,   &
                            data%ORDER, data%ma86_keep,                        &
                            data%ma86_control, data%ma86_info )
         inform%ma86_info  = data%ma86_info
         inform%status = data%ma86_info%flag
         IF ( inform%status == - 1 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma86_info%stat
         ELSE IF ( inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_permutation
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_inertia
         ELSE IF ( inform%status == - 4 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 5 ) THEN
           inform%status = GALAHAD_error_factorization
         ELSE IF ( inform%status == GALAHAD_unavailable_option ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' ma86 is not available ' )" ) prefix
           inform%status = GALAHAD_error_unknown_solver
         ELSE
           inform%max_depth_assembly_tree = data%ma86_info%maxdepth
           inform%nodes_assembly_tree = data%ma86_info%num_nodes
           inform%entries_in_factors = data%ma86_info%num_factor
           inform%flops_elimination = data%ma86_info%num_flops
         END IF

!  = MA87 =

       CASE ( 'ma87' )
         CALL SLS_copy_control_to_ma87( control, data%ma87_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!write(6,*) ' n ', data%matrix%n
!write(6,*) ' ptr ',  data%matrix%PTR( : data%matrix%n + 1 )
!write(6,*) ' col ',  data%matrix%COL( : data%matrix%PTR( data%matrix%n + 1 )-1)
!write(6,*) ' order ', data%ORDER( : data%matrix%n )
         CALL MA87_analyse( data%matrix%n, data%matrix%PTR, data%matrix%COL,   &
                            data%ORDER, data%ma87_keep,                        &
                            data%ma87_control, data%ma87_info )
         inform%ma87_info  = data%ma87_info
         inform%status = data%ma87_info%flag
         IF ( inform%status == - 1 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma87_info%stat
         ELSE IF ( inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_permutation
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_inertia
         ELSE IF ( inform%status == - 4 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 5 ) THEN
           inform%status = GALAHAD_error_factorization
         ELSE IF ( inform%status == GALAHAD_unavailable_option ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' ma87 is not available ' )" ) prefix
           inform%status = GALAHAD_error_unknown_solver
         ELSE
           inform%max_depth_assembly_tree = data%ma87_info%maxdepth
           inform%nodes_assembly_tree = data%ma87_info%num_nodes
           inform%entries_in_factors = data%ma87_info%num_factor
           inform%flops_elimination = data%ma87_info%num_flops
         END IF
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
         CALL SPACE_resize_array( matrix%n, data%INVP,                         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%INVP' ; GO TO 900 ; END IF
         DO i = 1, matrix%n
           data%INVP( data%ORDER( i ) ) = i
         END DO

!  = MA97 =

       CASE ( 'ma97' )
         CALL SLS_copy_control_to_ma97( control, data%ma97_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         IF ( mc6168_ordering ) THEN
           data%ma97_control%ordering = 0
           CALL MA97_analyse( .FALSE., data%matrix%n,                          &
                              data%matrix%PTR, data%matrix%COL,                &
                              data%ma97_akeep,                                 &
                              data%ma97_control, data%ma97_info,               &
                              order = data%ORDER )
         ELSE
           IF ( PRESENT( PERM ) ) THEN
             data%ma97_control%ordering = 0
             CALL MA97_analyse( .FALSE., data%matrix%n,                        &
                                data%matrix%PTR, data%matrix%COL,              &
                                data%ma97_akeep,                               &
                                data%ma97_control, data%ma97_info,             &
                                order = data%ORDER )
           ELSE
             data%ma97_control%ordering = - control%ordering
             CALL MA97_analyse( .FALSE., data%matrix%n,                        &
                                data%matrix%PTR, data%matrix%COL,              &
                                data%ma97_akeep,                               &
                                data%ma97_control, data%ma97_info,             &
                                order = data%ORDER )
           END IF
         END IF
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
         CALL SPACE_resize_array( matrix%n, data%LFLAG,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%LFLAG' ; GO TO 900 ; END IF
         DO i = 1, matrix%n
           data%LFLAG( i ) = .FALSE.
         END DO

!  = SSIDS =

       CASE ( 'ssids' )
         CALL SLS_copy_control_to_ssids( control, data%ssids_options )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         IF ( mc6168_ordering ) THEN
           data%ssids_options%ordering = 0
           CALL SSIDS_analyse( .FALSE., data%matrix%n,                         &
                               data%matrix%PTR, data%matrix%COL,               &
                               data%ssids_akeep,                               &
                               data%ssids_options, data%ssids_inform,          &
                               order = data%ORDER )
         ELSE
           IF ( PRESENT( PERM ) ) THEN
             data%ssids_options%ordering = 0
             CALL SSIDS_analyse( .FALSE., data%matrix%n,                       &
                                 data%matrix%PTR, data%matrix%COL,             &
                                 data%ssids_akeep,                             &
                                 data%ssids_options, data%ssids_inform,        &
                                 order = data%ORDER )
           ELSE
             data%ssids_options%ordering = - control%ordering
             CALL SSIDS_analyse( .FALSE., data%matrix%n,                       &
                                 data%matrix%PTR, data%matrix%COL,             &
                                 data%ssids_akeep,                             &
                                 data%ssids_options, data%ssids_inform,        &
                                 order = data%ORDER )
           END IF
         END IF
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
         CALL SPACE_resize_array( matrix%n, data%LFLAG,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%LFLAG' ; GO TO 900 ; END IF
         DO i = 1, matrix%n
           data%LFLAG( i ) = .FALSE.
         END DO

!  = PARDISO =

       CASE ( 'pardiso' )

         CALL SPACE_resize_array( matrix%n, 1, data%X2,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF

!  set intial pardiso storage

         IF ( data%must_be_definite ) THEN
           data%pardiso_mtype = 2
         ELSE
           data%pardiso_mtype = - 2
         END IF
         data%pardiso_iparm( 1 ) = 0
         pardiso_solver = 0
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!v3      CALL PARDISOINIT( data%PARDISO_PT, data%pardiso_mtype,                &
!                          data%pardiso_iparm )

         CALL PARDISOINIT( data%PARDISO_PT, data%pardiso_mtype, pardiso_solver,&
                           data%pardiso_iparm, data%pardiso_dparm,             &
                           inform%pardiso_error )

         IF ( inform%pardiso_error < 0 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 ) THEN
             IF ( inform%pardiso_error == GALAHAD_unavailable_option ) THEN
               WRITE( control%out, "( A, ' pardiso is not available ' )" )     &
                prefix
             ELSE
               WRITE( control%out, "( A, ' pardiso init error code = ', I0 )" )&
                prefix, inform%pardiso_error
             END IF
           END IF
           IF ( inform%pardiso_error == GALAHAD_unavailable_option ) THEN
             inform%status = GALAHAD_unavailable_option
           ELSE IF ( inform%pardiso_error == - 11 ) THEN
             inform%status = GALAHAD_unavailable_option
           ELSE
             inform%status = GALAHAD_error_pardiso
           END IF
           GO TO 800
         END IF
         data%pardiso_iparm( 3 ) = 1
!$       data%pardiso_iparm( 3 ) = OMP_GET_NUM_THREADS( )
         IF ( control%ordering > 0 .OR. PRESENT( PERM ) )                      &
           data%pardiso_iparm( 5 ) = 1

         CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
         CALL PARDISO( data%PARDISO_PT( 1 : 64 ), 1, 1, data%pardiso_mtype,    &
                       11, data%matrix%n, data%matrix%VAL( 1 : data%ne ),      &
                       data%matrix%PTR( 1 : data%matrix%n + 1 ),               &
                       data%matrix%COL( 1 : data%ne ),                         &
                       data%ORDER( 1 : data%matrix%n ), 1,                     &
                       data%pardiso_iparm( 1 : 64 ),                           &
                       control%print_level_solver,                             &
                       data%B2( 1 : matrix%n, 1 : 1 ),                         &
                       data%X2( 1 : matrix%n, 1 : 1 ), inform%pardiso_error,   &
                       data%pardiso_dparm( 1 : 64 ) )
!v3                    data%X2( 1 : matrix%n, 1 : 1 ), inform%pardiso_error )

         inform%pardiso_iparm = data%pardiso_iparm
         inform%pardiso_dparm = data%pardiso_dparm

         IF ( inform%pardiso_error < 0 .AND. control%print_level > 0 .AND.     &
              control%out > 0 ) WRITE( control%out,                            &
            "( A, ' pardiso error code = ', I0 )" ) prefix, inform%pardiso_error

         SELECT CASE( inform%pardiso_error )
         CASE ( - 1 )
           inform%status = GALAHAD_error_restrictions
         CASE ( GALAHAD_unavailable_option )
           inform%status = GALAHAD_unavailable_option
         CASE ( - 103 : GALAHAD_unavailable_option - 1,                        &
                GALAHAD_unavailable_option + 1 : - 2 )
           inform%status = GALAHAD_error_pardiso
         CASE DEFAULT
           inform%status = GALAHAD_ok
         END SELECT
         IF ( data%pardiso_iparm( 18 ) > 0 )                                   &
           inform%entries_in_factors = INT( data%pardiso_iparm( 18 ), long )

!  = WSMP =

       CASE ( 'wsmp' )

!  set up the inverse permutation

         CALL SPACE_resize_array( matrix%n, data%MRP,                          &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MRP' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n, data%INVP,                         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%INVP' ; GO TO 900 ; END IF
         DO i = 1, matrix%n
           data%INVP( data%ORDER( i ) ) = i
         END DO

!  fill 'iparm' and 'dparm' arrays with default values

         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL wsmp_initialize( )
         data%wsmp_iparm( 1 ) = 0
         data%wsmp_iparm( 2 ) = 0
         data%wsmp_iparm( 3 ) = 0
         CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),  &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%matrix%VAL( 1 : data%ne ),                           &
                     data%DIAG( 0 : 0 ),                                       &
                     data%ORDER( 1 : data%matrix%n ),                          &
                     data%INVP( 1 : data%matrix%n ),                           &
                     data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1,    &
                     data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),          &
                     data%wsmp_iparm, data%wsmp_dparm )
         inform%wsmp_error = data%wsmp_iparm( 64 )
         IF ( inform%wsmp_error < 0 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 ) THEN
             IF ( inform%wsmp_error == GALAHAD_unavailable_option ) THEN
               WRITE( control%out, "( A, ' wsmp is not available ' )" ) prefix
             ELSE
               WRITE( control%out, "( A, ' wsmp init error code = ', I0 )" )   &
                prefix, inform%wsmp_error
             END IF
           END IF
           IF ( inform%wsmp_error == GALAHAD_unavailable_option ) THEN
             inform%status = GALAHAD_unavailable_option
           ELSE IF ( inform%wsmp_error == - 900 ) THEN
             inform%status = GALAHAD_unavailable_option
           ELSE
             inform%status = GALAHAD_error_wsmp
           END IF
           GO TO 800
         END IF

         CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,              &
                                        data%wsmp_dparm )
!$       CALL wsetmaxthrds( OMP_GET_NUM_THREADS( ) )

         data%wsmp_iparm( 2 ) = 1
         data%wsmp_iparm( 3 ) = 2
         CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),  &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%matrix%VAL( 1 : data%ne ),                           &
                     data%DIAG( 0 : 0 ),                                       &
                     data%ORDER( 1 : data%matrix%n ),                          &
                     data%INVP( 1 : data%matrix%n ),                           &
                     data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1,    &
                     data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),          &
                     data%wsmp_iparm, data%wsmp_dparm )
         inform%wsmp_iparm = data%wsmp_iparm
         inform%wsmp_dparm = data%wsmp_dparm
         inform%wsmp_error = data%wsmp_iparm( 64 )

         IF ( inform%wsmp_error < 0 .AND. control%print_level > 0 .AND.        &
              control%out > 0 ) WRITE( control%out,                            &
            "( A, ' wsmp error code = ', I0 )" ) prefix, inform%wsmp_error

         SELECT CASE( inform%wsmp_error )
         CASE ( 0 )
           inform%status = GALAHAD_ok
         CASE ( - 102 )
           inform%status = GALAHAD_error_allocate
         CASE DEFAULT
           inform%status = GALAHAD_error_wsmp
         END SELECT

         IF ( data%wsmp_iparm( 23 ) >= 0 )                                     &
           inform%real_size_factors = 1000 * INT( data%wsmp_iparm( 23 ), long )
         IF ( data%wsmp_iparm( 24 ) >= 0 )                                     &
           inform%entries_in_factors = 1000 * INT( data%wsmp_iparm( 24 ), long )
       END SELECT

!  = POTR or SYTR =

     CASE ( 'potr', 'sytr' )

       data%matrix%n = data%n
       CALL SPACE_resize_array( data%matrix_ne, data%matrix%VAL,               &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF

       CALL SPACE_resize_array( data%n, data%n, data%matrix_dense,             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix_dense' ; GO TO 900 ; END IF

!write(6,*) ' mc6168_ordering ', mc6168_ordering
       IF ( .NOT. mc6168_ordering ) THEN

       CALL SPACE_resize_array( matrix%n, data%ORDER,                          &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%ORDER' ; GO TO 900 ; END IF


         IF ( PRESENT( PERM ) ) THEN
           data%ORDER( : matrix%n ) = PERM( : matrix%n )
         ELSE
           DO i = 1, matrix%n
             data%ORDER( i ) = i
           END DO
         END IF
       END IF

       data%reordered = .FALSE.
       DO i = 1, matrix%n
         IF ( data%ORDER( i ) /= i ) THEN
           data%reordered = .TRUE. ; EXIT
         END IF
       END DO

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       IF ( data%solver( 1 : data%len_solver ) == 'sytr' ) THEN
         CALL SPACE_resize_array( data%n, data%PIVOTS,                         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%sytr_lwork =                                                     &
           data%n * ILAENV( 1, 'DSYTRF', 'L', data%n, - 1, - 1, - 1 )
         CALL SPACE_resize_array( data%sytr_lwork, data%WORK,                  &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
       END IF

!  = PBTR =

     CASE ( 'pbtr' )

       data%matrix%n = data%n
       CALL SPACE_resize_array( data%matrix_ne, data%matrix%VAL,               &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF

       IF ( .NOT. mc6168_ordering ) THEN
         CALL SPACE_resize_array( matrix%n, data%ORDER,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%ORDER' ; GO TO 900 ; END IF

         IF ( PRESENT( PERM ) ) THEN
           data%ORDER( : matrix%n ) = PERM( : matrix%n )
         ELSE
           DO i = 1, matrix%n
             data%ORDER( i ) = i
           END DO
         END IF
       END IF

       data%reordered = .FALSE.
       DO i = 1, matrix%n
         IF ( data%ORDER( i ) /= i ) THEN
           data%reordered = .TRUE. ; EXIT
         END IF
       END DO

!  compute the semi-bandwith of the matrix

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         IF ( matrix%ne > 0 ) THEN
           inform%semi_bandwidth =                                             &
             MAXVAL( ABS( data%ORDER( matrix%ROW( : matrix%ne ) ) -            &
                          data%ORDER( matrix%COL( : matrix%ne ) ) ) )
         ELSE
           inform%semi_bandwidth = 0
         END IF
       CASE ( 'SPARSE_BY_ROWS' )
         inform%semi_bandwidth = 0
         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             inform%semi_bandwidth = MAX( inform%semi_bandwidth,               &
                ABS( data%ORDER( i ) - data%ORDER( matrix%COL( l ) ) ) )
           END DO
         END DO
       CASE ( 'DENSE' )
         inform%semi_bandwidth = matrix%n - 1
       END SELECT
!write(6,*) ' semi-bandwidth ', inform%semi_bandwidth
       CALL SPACE_resize_array( inform%semi_bandwidth + 1, data%n,             &
                                data%matrix_dense,                             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix_dense' ; GO TO 900 ; END IF

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

     CASE DEFAULT
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       inform%status = GALAHAD_error_unknown_solver
     END SELECT

!  record external analyse time

 800 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%analyse_external = time_now - time
     inform%time%clock_analyse_external = clock_now - clock

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%analyse =inform%time%analyse + time_now - time_start
     inform%time%clock_analyse =                                               &
       inform%time%clock_analyse + clock_now - clock_start
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of SLS_analyse

     END SUBROUTINE SLS_analyse

!-*-*-*-*-*-*-   S L S _ F A C T O R I Z E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE SLS_factorize( matrix, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Factorize the matrix using the ordering suggested from the analysis

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, ii, j, jj, k, l, l1, l2, job, matrix_type
     REAL :: time, time_start, time_now
     REAL ( KIND = wp ) :: clock, clock_start, clock_now
     REAL ( KIND = wp ) :: val
     LOGICAL :: filexx
     CHARACTER ( LEN = 400 ), DIMENSION( 1 ) :: path
     CHARACTER ( LEN = 400 ), DIMENSION( 4 ) :: filename
!    CHARACTER :: dumc( 20 )

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  if the input matrix is not in co-ordinate form and sils/ma27/ma57 are
!  to be used, make a copy

     data%explicit_scaling = control%scaling > 0 .AND. control%scaling < 4
     SELECT CASE( data%solver( 1 : data%len_solver ) )
     CASE ( 'sils', 'ma27', 'ma57' )
       SELECT CASE ( SMT_get( MATRIX%type ) )
       CASE ( 'COORDINATE' )
       CASE ( 'SPARSE_BY_ROWS', 'DENSE' )
         data%matrix%VAL( : data%matrix%ne ) = matrix%VAL( : data%matrix%ne )
       END SELECT
     END SELECT

!  if required, compute scale factors

     IF ( data%explicit_scaling ) THEN

!  convert to column storage. First compute scaling map

       IF ( .NOT. data%got_maps_scale ) THEN
         data%got_maps_scale = .TRUE.
         data%matrix_scale_ne = data%matrix_ne
         data%matrix_scale%ne = data%matrix_ne
         CALL SMT_put( data%matrix_scale%type, 'SPARSE_BY_ROWS',               &
                       inform%alloc_status )
         CALL SPACE_resize_array( data%matrix_scale_ne, data%MAPS_scale,       &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MAPS_scale' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n + 1, data%matrix_scale%PTR,         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix_scale%PTR' ; GO TO 900 ; END IF

         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           CALL SLS_coord_to_sorted_csr( matrix%n, matrix%ne,                  &
                                         matrix%ROW, matrix%COL,               &
                                         data%MAPS_scale,                      &
                                         data%matrix_scale%PTR,                &
                                         inform%duplicates,                    &
                                         inform%out_of_range, inform%upper,    &
                                         inform%missing_diagonals,             &
                                         inform%status, inform%alloc_status )
         CASE ( 'SPARSE_BY_ROWS' )
           CALL SMT_put( data%matrix_scale%type, 'SPARSE_BY_ROWS',             &
                         inform%alloc_status )
           CALL SPACE_resize_array( data%matrix_scale%ne,                      &
                                    data%matrix_scale%ROW,                     &
                                    inform%status, inform%alloc_status )
           DO i = 1, matrix%n
             data%matrix_scale%ROW( matrix%PTR( i ) :                          &
                                    matrix%PTR( i + 1 ) - 1 ) = i
           END DO
           CALL SLS_coord_to_sorted_csr( matrix%n, data%matrix_scale_ne,       &
                                         data%matrix_scale%ROW, matrix%COL,    &
                                         data%MAPS_scale,                      &
                                         data%matrix_scale%PTR,                &
                                         inform%duplicates,                    &
                                         inform%out_of_range, inform%upper,    &
                                         inform%missing_diagonals,             &
                                         inform%status, inform%alloc_status )
         CASE ( 'DENSE' )
           CALL SMT_put( data%matrix_scale%type, 'DENSE', inform%alloc_status )
           CALL SPACE_resize_array( data%matrix_scale%ne,                      &
                                    data%matrix_scale%ROW,                     &
                                    inform%status, inform%alloc_status )
           CALL SPACE_resize_array( data%matrix_scale%ne,                      &
                                    data%matrix_scale%COL,                     &
                                    inform%status, inform%alloc_status )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               data%matrix_scale%ROW( l ) = i ; data%matrix_scale%COL( l ) = j
             END DO
           END DO
           CALL SLS_coord_to_sorted_csr( matrix%n, data%matrix_scale_ne,       &
                                         data%matrix_scale%ROW,                &
                                         data%matrix_scale%COL,                &
                                         data%MAPS_scale,                      &
                                         data%matrix_scale%PTR,                &
                                         inform%duplicates,                    &
                                         inform%out_of_range, inform%upper,    &
                                         inform%missing_diagonals,             &
                                         inform%status, inform%alloc_status )
         END SELECT
         IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  now map the column data ...

         data%matrix_scale%m = matrix%n
         data%matrix_scale%n = matrix%n
         data%matrix_scale%ne =                                                &
           data%matrix_scale%PTR( data%matrix_scale%n + 1 ) - 1
         CALL SPACE_resize_array( data%matrix_scale%ne, data%matrix_scale%ROW, &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix_scale%ROW' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( data%matrix_scale%ne, data%matrix_scale%VAL, &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix_scale%VAL' ; GO TO 900 ; END IF

         DO i = 1, matrix%n
           data%matrix_scale%ROW( data%matrix_scale%PTR( i ) ) = i
         END DO

         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )

!  if necessary, provide make a copy of the matrix so that applying
!  scale factors does not alter the orginal

           SELECT CASE( data%solver( 1 : data%len_solver ) )
           CASE ( 'sils', 'ma27', 'ma57' )

             data%matrix%n = matrix%n ; data%matrix%ne = matrix%ne
             CALL SMT_put( data%matrix%type, 'COORDINATE',                     &
                           inform%alloc_status )
             CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,         &
                                      inform%status, inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) THEN
               inform%bad_alloc = 'sls: data%matrix%ROW' ; GO TO 900 ; END IF
             CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,         &
                                      inform%status, inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) THEN
               inform%bad_alloc = 'sls: data%matrix%COL' ; GO TO 900 ; END IF
             CALL SPACE_resize_array( data%matrix%ne, data%matrix%VAL,         &
                                      inform%status, inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) THEN
               inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 900 ; END IF
             data%matrix%row( : data%matrix%ne ) =                             &
               matrix%row( : data%matrix%ne )
             data%matrix%COL( : data%matrix%ne ) =                             &
               matrix%COL( : data%matrix%ne )
           END SELECT

           DO l = 1, matrix%ne
             k = data%MAPS_scale( l )
             IF ( k > 0 ) data%matrix_scale%ROW( k ) =                         &
                 MAX( matrix%ROW( l ), matrix%COL( l ) )
           END DO
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, matrix%n
             DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
               k = data%MAPS_scale( l )
               IF ( k > 0 ) data%matrix_scale%ROW( k ) =                       &
                 MAX( i, matrix%COL( l ) )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
               k = data%MAPS_scale( l )
               IF ( k > 0 ) data%matrix_scale%ROW( k ) = MAX( i, j )
             END DO
           END DO
         END SELECT
       END IF

!  ...  and the values

       DO i = 1, matrix%n
         data%matrix_scale%VAL( data%matrix_scale%PTR( i ) ) = 0.0_wp
       END DO

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, matrix%ne
           k = data%MAPS_scale( l )
           IF ( k > 0 ) THEN
             data%matrix_scale%VAL( k ) = matrix%VAL( l )
           ELSE IF ( k < 0 ) THEN
             data%matrix_scale%VAL( - k ) =                                    &
               data%matrix_scale%VAL( - k ) + matrix%VAL( l )
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             k = data%MAPS_scale( l )
             IF ( k > 0 ) THEN
               data%matrix_scale%VAL( k ) = matrix%VAL( l )
             ELSE IF ( k < 0 ) THEN
               data%matrix_scale%VAL( - k ) =                                  &
                 data%matrix_scale%VAL( - k ) + matrix%VAL( l )
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, matrix%n
           DO j = 1, i
             l = l + 1
             k = data%MAPS_scale( l )
             IF ( k > 0 ) THEN
               data%matrix_scale%VAL( k ) = matrix%VAL( l )
             ELSE IF ( k < 0 ) THEN
               data%matrix_scale%VAL( - k ) =                                  &
                 data%matrix_scale%VAL( - k ) + matrix%VAL( l )
             END IF
           END DO
         END DO
       END SELECT

!  scaling using MC64 lower triangle stored by columns
!  need matrix%id(1) = 'S'

       IF ( control%scaling == 1 ) THEN

!  allocate workspace

         CALL SPACE_resize_array( 2 * data%matrix_scale%n, data%mc64_PERM,     &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mc64_PERM' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( 2 * data%matrix_scale%n, data%SCALE,         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%SCALE' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( 1, data%matrix_scale%id,                     &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix_scale%id' ; GO TO 900 ; END IF

!  obtain scaling factors from MC64

         IF ( control%print_level <= 0 .OR. control%out <= 0 ) THEN
           data%mc64_control%lp = - 1
           data%mc64_control%wp = - 1
           data%mc64_control%sp = - 1
         END IF
         data%matrix_scale%id( 1 ) = 'S'
         CALL MC64_MATCHING( 5, data%matrix_scale, data%mc64_control,          &
                             inform%mc64_info, data%mc64_PERM,                 &
                             data%SCALE )
         IF ( inform%mc64_info%flag /= 0 .AND.                                 &
              inform%mc64_info%flag /= - 9 ) THEN
           inform%status = GALAHAD_error_mc64 ; GO TO 900
         END IF
         data%SCALE( : data%matrix_scale%n ) =                                 &
           EXP( data%SCALE( : data%matrix_scale%n ) )

!  scaling using MC77 based on the row one- or infinity-norm

       ELSE

!  allocate workspace

         data%mc77_liw = data%matrix_scale%n
         data%mc77_ldw = data%matrix_scale%ne + 2 * data%matrix_scale%n
         CALL SPACE_resize_array( data%mc77_liw, data%mc77_IW,                 &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mc77_IW' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( data%mc77_ldw, data%SCALE,                   &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%SCALE' ; GO TO 900 ; END IF

!  obtain scaling factors from MC77

         IF ( control%scaling == 2 ) THEN
           job = 1
         ELSE
           job = 0
         END IF
         IF ( control%print_level <= 0 .OR. control%out <= 0 )                 &
           data%mc77_ICNTL( 1 ) = - 1
         CALL MC77AD( job, data%matrix_scale%n, data%matrix_scale%n,           &
                      data%matrix_scale%ne, data%matrix_scale%PTR,             &
!                     data%matrix_scale%COL, data%matrix_scale%VAL,            &
                      data%matrix_scale%ROW, data%matrix_scale%VAL,            &
                      data%mc77_IW, data%mc77_liw,data%SCALE,                  &
                      data%mc77_ldw, data%mc77_ICNTL, data%mc77_CNTL,          &
                      inform%mc77_info, inform%mc77_rinfo )
         IF ( inform%mc77_info( 1 ) /= 0 ) THEN
           inform%status = GALAHAD_error_mc77 ; GO TO 900
         END IF
       END IF
     END IF

!  if desired, write the input matrix to an output file in co-ordinate form

     IF ( control%generate_matrix_file .AND.                                   &
          control%matrix_file_device > 0 ) THEN
       INQUIRE( FILE = control%matrix_file_name, EXIST = filexx )
       IF ( filexx ) THEN
         OPEN( control%matrix_file_device, FILE = control%matrix_file_name,    &
                FORM = 'FORMATTED', STATUS = 'OLD', IOSTAT = i )
       ELSE
         OPEN( control%matrix_file_device, FILE = control%matrix_file_name,    &
               FORM = 'FORMATTED', STATUS = 'NEW', IOSTAT = i )
       END IF

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         WRITE( control%matrix_file_device, * ) matrix%n, matrix%ne
         DO l = 1, matrix%ne
           WRITE( control%matrix_file_device, * )                              &
             matrix%ROW( l ), matrix%COL( l ), matrix%VAL( l )
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         WRITE( control%matrix_file_device, * )                                &
           matrix%n, matrix%PTR( matrix%n + 1 ) - 1
         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             WRITE( control%matrix_file_device, * )                            &
               i, matrix%COL( l ), matrix%VAL( l )
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         WRITE( control%matrix_file_device, * )                                &
           matrix%n, matrix%n * ( matrix%n + 1 ) / 2
         DO i = 1, matrix%n
           DO j = 1, i
             l = l + 1
             WRITE( control%matrix_file_device, * ) i, j, matrix%VAL( l )
           END DO
         END DO
       END SELECT

       CLOSE( control%matrix_file_device )
     END IF

!  solver-dependent factorization

     SELECT CASE( data%solver( 1 : data%len_solver ) )

     CASE ( 'sils', 'ma27', 'ma57' )

!  apply calculated scaling factors

       IF ( data%explicit_scaling ) THEN
!write(6,*) ' n, ne ', data%matrix%n, data%matrix%ne
         DO l = 1, data%matrix%ne
           i = data%matrix%ROW( l ) ; j = data%matrix%COL( l )
           data%matrix%VAL( l ) = matrix%VAL( l ) /                           &
             ( data%SCALE( i ) * data%SCALE( j ) )
         END DO
       END IF

!  = SILS =

       SELECT CASE( data%solver( 1 : data%len_solver ) )
       CASE ( 'sils', 'ma27' )
         CALL SLS_copy_control_to_sils( control, data%sils_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           IF ( data%explicit_scaling ) THEN
             CALL SILS_factorize( data%matrix, data%sils_factors,              &
                                  data%sils_control, data%sils_finfo )
           ELSE
             CALL SILS_factorize( matrix, data%sils_factors,                   &
                                  data%sils_control, data%sils_finfo )
           END IF
         CASE DEFAULT
           CALL SILS_factorize( data%matrix, data%sils_factors,                &
                                data%sils_control, data%sils_finfo )
         END SELECT
         inform%sils_finfo = data%sils_finfo
         inform%status = data%sils_finfo%flag
         IF ( inform%status == - 1 .OR. inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%sils_finfo%stat
         ELSE IF ( inform%status == - 5 .OR. inform%status == - 6 ) THEN
           inform%status = GALAHAD_error_inertia
           inform%more_info = data%sils_finfo%more
         ELSE IF ( inform%status == - 7 ) THEN
           inform%status = GALAHAD_error_real_ws
         ELSE IF ( inform%status == - 8 ) THEN
           inform%status = GALAHAD_error_integer_ws
         ELSE
!          IF ( inform%status == 4 ) inform%status = GALAHAD_error_inertia
           IF ( inform%status == 4 ) inform%status = GALAHAD_ok
           inform%more_info = data%sils_finfo%more
           inform%alloc_status = data%sils_finfo%stat
           inform%more_info = data%sils_finfo%more
           inform%max_front_size = data%sils_finfo%maxfrt
           inform%entries_in_factors = INT( data%sils_finfo%nebdu, long )
           inform%real_size_factors = INT( data%sils_finfo%nrlbdu, long )
           inform%integer_size_factors = INT( data%sils_finfo%nirbdu, long )
           inform%real_size_desirable = INT( data%sils_finfo%nrltot, long )
           inform%integer_size_desirable = INT( data%sils_finfo%nirtot, long )
           inform%real_size_necessary = INT( data%sils_finfo%nrlnec, long )
           inform%integer_size_necessary = INT( data%sils_finfo%nirnec, long )
           inform%compresses_real = data%sils_finfo%ncmpbr
           inform%compresses_integer = data%sils_finfo%ncmpbi
           inform%rank = data%sils_finfo%rank
           inform%two_by_two_pivots = data%sils_finfo%ntwo
           inform%negative_eigenvalues = data%sils_finfo%neig
           inform%delayed_pivots = data%sils_finfo%delay
           inform%pivot_sign_changes = data%sils_finfo%signc
           inform%static_pivots = data%sils_finfo%static
           inform%first_modified_pivot = data%sils_finfo%modstep
           inform%flops_assembly = INT( data%sils_finfo%opsa, long )
           inform%flops_elimination = INT( data%sils_finfo%opse, long )
           inform%flops_blas = INT( data%sils_finfo%opsb, long )
           inform%largest_modified_pivot = data%sils_finfo%maxchange
           inform%minimum_scaling_factor = data%sils_finfo%smin
           inform%maximum_scaling_factor = data%sils_finfo%smax
         END IF

!  = MA57 =

       CASE ( 'ma57' )
         CALL SLS_copy_control_to_ma57( control, data%ma57_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         SELECT CASE ( SMT_get( MATRIX%type ) )
         CASE ( 'COORDINATE' )
           IF ( data%explicit_scaling ) THEN
             CALL MA57_factorize( data%matrix, data%ma57_factors,              &
                                  data%ma57_control, data%ma57_finfo )
           ELSE
             CALL MA57_factorize( matrix, data%ma57_factors,                   &
                                  data%ma57_control, data%ma57_finfo )
           END IF
         CASE DEFAULT
           CALL MA57_factorize( data%matrix, data%ma57_factors,                &
                                data%ma57_control, data%ma57_finfo )
         END SELECT
         inform%ma57_finfo = data%ma57_finfo
         inform%status = data%ma57_finfo%flag
         IF ( inform%status == - 1 .OR. inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma57_finfo%stat
         ELSE IF ( inform%status == - 5 .OR. inform%status == - 6 ) THEN
           inform%status = GALAHAD_error_inertia
           inform%more_info = data%ma57_finfo%more
         ELSE IF ( inform%status == - 7 ) THEN
           inform%status = GALAHAD_error_real_ws
         ELSE IF ( inform%status == - 8 ) THEN
           inform%status = GALAHAD_error_integer_ws
         ELSE
!          IF ( inform%status == 4 ) inform%status = GALAHAD_error_inertia
           IF ( inform%status == 4 ) inform%status = GALAHAD_ok
           inform%more_info = data%ma57_finfo%more
           inform%max_front_size = data%ma57_finfo%maxfrt
           inform%entries_in_factors = INT( data%ma57_finfo%nebdu, long )
           inform%real_size_factors = INT( data%ma57_finfo%nrlbdu, long )
           inform%integer_size_factors = INT( data%ma57_finfo%nirbdu, long )
           inform%real_size_desirable = INT( data%ma57_finfo%nrltot, long )
           inform%integer_size_desirable = INT( data%ma57_finfo%nirtot, long )
           inform%real_size_necessary  = INT( data%ma57_finfo%nrlnec, long )
           inform%integer_size_necessary  = INT( data%ma57_finfo%nirnec, long )
           inform%compresses_real = data%ma57_finfo%ncmpbr
           inform%compresses_integer = data%ma57_finfo%ncmpbi
           inform%rank = data%ma57_finfo%rank
           inform%two_by_two_pivots = data%ma57_finfo%ntwo
           inform%negative_eigenvalues  = data%ma57_finfo%neig
           inform%delayed_pivots = data%ma57_finfo%delay
           inform%pivot_sign_changes = data%ma57_finfo%signc
           inform%static_pivots = data%ma57_finfo%static
           inform%first_modified_pivot = data%ma57_finfo%modstep
           inform%flops_assembly = INT( data%ma57_finfo%opsa, long )
           inform%flops_elimination = INT( data%ma57_finfo%opse, long )
           inform%flops_blas = INT( data%ma57_finfo%opsb, long )
           IF ( inform%first_modified_pivot > 0 ) THEN
             inform%largest_modified_pivot = data%ma57_finfo%maxchange
           ELSE
             inform%largest_modified_pivot = 0.0_wp
           END IF
           inform%minimum_scaling_factor = data%ma57_finfo%smin
           inform%maximum_scaling_factor = data%ma57_finfo%smax
         END IF
       END SELECT

!  = MA77 =

     CASE ( 'ma77' )
       DO j = 1, 2
         DO l = 1, data%matrix_ne
           k = data%MAP( l, j )
           IF ( k > 0 ) THEN
             data%matrix%VAL( k ) = matrix%VAL( l )
           ELSE IF ( k < 0 ) THEN
             data%matrix%VAL( - k ) = data%matrix%VAL( - k ) + matrix%VAL( l )
           END IF
         END DO
       END DO

!  apply calculated scaling factors

       IF ( data%explicit_scaling ) THEN
         DO i = 1, data%matrix%n
           DO l = data%matrix%PTR( i ), data%matrix%PTR( i + 1 ) - 1
             j = data%matrix%ROW( l )
             data%matrix%VAL( l ) = data%matrix%VAL( l ) /                     &
               ( data%SCALE( i ) * data%SCALE( j ) )
           END DO
         END DO
       END IF

       CALL SLS_copy_control_to_ma77( control, data%ma77_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       DO i = 1, data%n
         l1 = data%matrix%PTR( i )
         l2 = data%matrix%PTR( i + 1 ) - 1
         CALL MA77_input_reals( i, l2 - l1 + 1, data%matrix%VAL( l1 : l2 ),    &
                               data%ma77_keep, data%ma77_control,              &
                               data%ma77_info )
         CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
       END DO
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         CALL SPACE_resize_array( data%n, data%SCALE,                          &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 800 ; END IF
         CALL MA77_scale( data%SCALE,                                          &
                          data%ma77_keep, data%ma77_control, data%ma77_info )
         CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
         CALL MA77_factor( data%must_be_definite,                              &
                           data%ma77_keep, data%ma77_control,                  &
                           data%ma77_info, SCALE = data%SCALE )
       ELSE
         CALL MA77_factor( data%must_be_definite,                              &
                           data%ma77_keep, data%ma77_control, data%ma77_info )
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )

       CALL MA77_finalise( data%ma77_keep, data%ma77_control,                  &
                           data%ma77_info,                                     &
                           restart_file = control%out_of_core_restart_file )
       filename( 1 ) = control%out_of_core_integer_factor_file
       filename( 2 ) = control%out_of_core_real_factor_file
       filename( 3 ) = control%out_of_core_real_work_file
       filename( 4 ) = control%out_of_core_indefinite_file
       IF ( TRIM( control%out_of_core_directory ) == '' ) THEN
         CALL MA77_restart( control%out_of_core_restart_file, filename,        &
                         data%ma77_keep, data%ma77_control, data%ma77_info )
       ELSE
         path( 1 ) = REPEAT( ' ', 400 )
         path( 1 ) = control%out_of_core_directory
         CALL MA77_restart( control%out_of_core_restart_file, filename,        &
                          data%ma77_keep, data%ma77_control, data%ma77_info,   &
                          path = path )
       END IF

!  = MA86, MA87, MA97, SSIDS, PARDISO or WSMP =

     CASE ( 'ma86', 'ma87', 'ma97', 'ssids', 'pardiso', 'wsmp' )
       data%matrix%n = matrix%n
       DO i = 1, matrix%n
         l = data%matrix%PTR( i )
         data%matrix%VAL( l ) = 0.0_wp
       END DO

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, matrix%ne
           k = data%MAPS( l )
           IF ( k > 0 ) THEN
             data%matrix%VAL( k ) = matrix%VAL( l )
           ELSE IF ( k < 0 ) THEN
             data%matrix%VAL( - k ) = data%matrix%VAL( - k ) + matrix%VAL( l )
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             k = data%MAPS( l )
             IF ( k > 0 ) THEN
               data%matrix%VAL( k ) = matrix%VAL( l )
             ELSE IF ( k < 0 ) THEN
               data%matrix%VAL( - k ) = data%matrix%VAL( - k ) + matrix%VAL( l )
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, matrix%n
           DO j = 1, i
             l = l + 1
             k = data%MAPS( l )
             IF ( k > 0 ) THEN
               data%matrix%VAL( k ) = matrix%VAL( l )
             ELSE IF ( k < 0 ) THEN
               data%matrix%VAL( - k ) = data%matrix%VAL( - k ) + matrix%VAL( l )
             END IF
           END DO
         END DO
       END SELECT

!  apply calculated scaling factors

       IF ( data%explicit_scaling ) THEN
         DO i = 1, data%matrix%n
           DO l = data%matrix%PTR( i ), data%matrix%PTR( i + 1 ) - 1
             j = data%matrix%COL( l )
             data%matrix%VAL( l ) = data%matrix%VAL( l ) /                     &
               ( data%SCALE( i ) * data%SCALE( j ) )
           END DO
         END DO
       END IF

       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = MA86 =

       CASE ( 'ma86' )
         CALL SLS_copy_control_to_ma86( control, data%ma86_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL MA86_factor( data%matrix%n, data%matrix%PTR, data%matrix%COL,    &
                           data%matrix%VAL, data%ORDER, data%ma86_keep,        &
                           data%ma86_control, data%ma86_info )

         inform%ma86_info = data%ma86_info
         inform%status = data%ma86_info%flag
         IF ( inform%status == - 1 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma86_info%stat
         ELSE IF ( inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_permutation
         ELSE IF ( inform%status == - 3 .OR. inform%status == 2 .OR.           &
                   inform%status == 3 ) THEN
           inform%status = GALAHAD_error_inertia
         ELSE IF ( inform%status == - 4 ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( inform%status == - 5 ) THEN
           inform%status = GALAHAD_error_factorization
         ELSE
           inform%max_task_pool_size = data%ma86_info%pool_size
           inform%two_by_two_pivots = data%ma86_info%num_two
           inform%rank = data%ma86_info%matrix_rank
           inform%negative_eigenvalues = data%ma86_info%num_neg
           inform%delayed_pivots = data%ma86_info%num_delay
           inform%entries_in_factors = data%ma86_info%num_factor
           inform%flops_elimination = data%ma86_info%num_flops
           inform%static_pivots = data%ma86_info%num_perturbed
         END IF

!  = MA87 =

       CASE ( 'ma87' )
         CALL SLS_copy_control_to_ma87( control, data%ma87_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL MA87_factor( data%matrix%n, data%matrix%PTR, data%matrix%COL,    &
                           data%matrix%VAL, data%ORDER, data%ma87_keep,        &
                           data%ma87_control, data%ma87_info )

         inform%ma87_info = data%ma87_info
         inform%status = data%ma87_info%flag
         IF ( inform%status == - 1 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma87_info%stat
           GO TO 800
         ELSE IF ( inform%status == - 2 ) THEN
           inform%status = GALAHAD_error_permutation
           GO TO 800
         ELSE IF ( inform%status == - 3 ) THEN
           inform%status = GALAHAD_error_inertia
           GO TO 800
         ELSE IF ( inform%status == - 4 ) THEN
           inform%status = GALAHAD_error_restrictions
           GO TO 800
         ELSE IF ( inform%status == - 5 ) THEN
           inform%status = GALAHAD_error_factorization
           GO TO 800
         ELSE
           inform%max_task_pool_size = data%ma87_info%pool_size
           inform%entries_in_factors = data%ma87_info%num_factor
           inform%flops_elimination = data%ma87_info%num_flops
           inform%rank = data%matrix%n
           inform%negative_eigenvalues = 0
           inform%num_zero = data%ma87_info%num_zero
         END IF

         CALL SPACE_resize_array( data%matrix%n, data%WORK,                    &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%WORK' ; GO TO 900 ; END IF

!  = MA97 =

       CASE ( 'ma97' )
         CALL SLS_copy_control_to_ma97( control, data%ma97_control )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         IF ( data%must_be_definite ) THEN
           matrix_type = 3
         ELSE
           matrix_type = 4
         END IF
!    WRITE( 77, * ) data%matrix%n
!    WRITE( 77, * ) data%matrix%PTR( : data%matrix%n + 1 )
!    WRITE( 77, * ) data%matrix%COL( : data%matrix%PTR( data%matrix%n + 1 ) )
!    WRITE( 77, * ) data%matrix%VAL( : data%matrix%PTR( data%matrix%n + 1 ) )
!    stop
         IF ( data%ma97_control%scaling == 0 ) THEN
           CALL MA97_factor( matrix_type, data%matrix%VAL,                     &
                             data%ma97_akeep, data%ma97_fkeep,                 &
                             data%ma97_control, data%ma97_info,                &
                             ptr = data%matrix%PTR, row = data%matrix%COL )
         ELSE
           CALL SPACE_resize_array( data%n, data%SCALE,                        &
                                    inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 800 ; END IF
           CALL MA97_factor( matrix_type, data%matrix%VAL,                     &
                             data%ma97_akeep, data%ma97_fkeep,                 &
                             data%ma97_control, data%ma97_info,                &
                             scale = data%SCALE,                               &
                             ptr = data%matrix%PTR, row = data%matrix%COL )
         END IF
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )

!  = SSIDS =

       CASE ( 'ssids' )
         CALL SLS_copy_control_to_ssids( control, data%ssids_options )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!    WRITE( 77, * ) data%matrix%n
!    WRITE( 77, * ) data%matrix%PTR( : data%matrix%n + 1 )
!    WRITE( 77, * ) data%matrix%COL( : data%matrix%PTR( data%matrix%n + 1 ) )
!    WRITE( 77, * ) data%matrix%VAL( : data%matrix%PTR( data%matrix%n + 1 ) )
!    stop
         IF ( data%ssids_options%scaling == 0 ) THEN
           CALL SSIDS_factor( data%must_be_definite, data%matrix%VAL,          &
                              data%ssids_akeep, data%ssids_fkeep,              &
                              data%ssids_options, data%ssids_inform,           &
                              ptr = data%matrix%PTR, row = data%matrix%COL )
         ELSE
           CALL SPACE_resize_array( data%n, data%SCALE,                        &
                                    inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 800 ; END IF
           CALL SSIDS_factor( data%must_be_definite, data%matrix%VAL,          &
                              data%ssids_akeep, data%ssids_fkeep,              &
                              data%ssids_options, data%ssids_inform,           &
                              scale = data%SCALE,                              &
                              ptr = data%matrix%PTR, row = data%matrix%COL )
         END IF
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

       CASE ( 'pardiso' )
         CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL PARDISO( data%PARDISO_PT, 1, 1, data%pardiso_mtype, 22,          &
                       data%matrix%n, data%matrix%VAL( : data%ne ),            &
                       data%matrix%PTR( : data%matrix%n + 1 ),                 &
                       data%matrix%COL( : data%ne ),                           &
                       data%ORDER( : data%matrix%n ), 1,                       &
                       data%pardiso_iparm( : 64 ),                             &
                       control%print_level_solver, data%B2( : matrix%n, : 1 ), &
                       data%X2( : matrix%n, : 1 ), inform%pardiso_error,       &
                       data%pardiso_dparm( 1 : 64 ) )
!v3                    data%X2( : matrix%n, : 1 ), inform%pardiso_error )

         inform%pardiso_iparm = data%pardiso_iparm
         inform%pardiso_dparm = data%pardiso_dparm

         IF ( inform%pardiso_error < 0 .AND. control%print_level > 0 .AND.     &
              control%out > 0 ) WRITE( control%out,                            &
            "( A, ' pardiso error code = ', I0 )" ) prefix, inform%pardiso_error

         SELECT CASE( inform%pardiso_error )
         CASE ( - 1 )
           inform%status = GALAHAD_error_restrictions
         CASE ( GALAHAD_unavailable_option )
           inform%status = GALAHAD_unavailable_option
         CASE ( - 103 : GALAHAD_unavailable_option - 1,                        &
                GALAHAD_unavailable_option + 1 : - 2 )
           inform%status = GALAHAD_error_pardiso
         CASE DEFAULT
           inform%status = GALAHAD_ok
         END SELECT
         IF ( data%pardiso_iparm( 18 ) > 0 )                                   &
           inform%entries_in_factors = INT( data%pardiso_iparm( 18 ), long )
         inform%negative_eigenvalues = data%pardiso_iparm( 23 )
         inform%rank = data%pardiso_iparm( 22 ) + data%pardiso_iparm( 23 )
         IF ( data%must_be_definite .AND. inform%negative_eigenvalues > 0 )    &
           inform%status = GALAHAD_error_inertia

!  = WSMP =

       CASE ( 'wsmp' )
!       write( 71, * ) data%matrix%n, data%matrix%PTR( data%matrix%n + 1 ) - 1
!       do i = 1, data%matrix%n + 1
!         write( 71, * ) data%matrix%PTR( i )
!       end do
!       do l = 1, data%matrix%PTR( data%matrix%n + 1 ) - 1
!         write( 71, * ) data%matrix%COL( l ), data%matrix%VAL( l )
!       end do
         CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,              &
                                        data%wsmp_dparm )
         data%wsmp_iparm( 2 ) = 3
         data%wsmp_iparm( 3 ) = 3
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),  &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%matrix%VAL( 1 : data%ne ),                           &
                     data%DIAG( 0 : 0 ),                                       &
                     data%ORDER( 1 : data%matrix%n ),                          &
                     data%INVP( 1 : data%matrix%n ),                           &
                     data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1,    &
                     data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),          &
                     data%wsmp_iparm, data%wsmp_dparm )
         inform%wsmp_iparm = data%wsmp_iparm
         inform%wsmp_dparm = data%wsmp_dparm
         inform%wsmp_error = data%wsmp_iparm( 64 )
         IF ( inform%wsmp_error < 0 .AND. control%print_level > 0 .AND.        &
              control%out > 0 ) WRITE( control%out,                            &
            "( A, ' wsmp error code = ', I0 )" ) prefix, inform%wsmp_error

         SELECT CASE( inform%wsmp_error )
         CASE ( 0 )
           inform%status = GALAHAD_ok
         CASE ( - 102 )
           inform%status = GALAHAD_error_allocate
         CASE DEFAULT
           inform%status = GALAHAD_error_wsmp
         END SELECT

         IF ( data%wsmp_iparm( 23 ) > 0 )                                      &
           inform%real_size_factors = 1000 * INT( data%wsmp_iparm( 23 ), long )
         IF ( data%wsmp_iparm( 24 ) > 0 )                                      &
           inform%entries_in_factors = 1000 * INT( data%wsmp_iparm( 24 ), long )
         inform%negative_eigenvalues = data%wsmp_iparm( 22 )
         inform%rank = data%matrix%n - data%pardiso_iparm( 21 )
         IF ( data%must_be_definite .AND. inform%negative_eigenvalues > 0 )    &
           inform%status = GALAHAD_error_inertia
         IF ( data%must_be_definite .AND.inform%wsmp_error > 0 )               &
           inform%status = GALAHAD_error_inertia

       END SELECT

!  = LAPACK solvers POTR or SYTR =

     CASE ( 'potr', 'sytr' )

       data%matrix_dense( : matrix%n, : matrix%n ) = 0.0_wp

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, matrix%ne
           i = matrix%ROW( l ) ; ii = data%ORDER( i )
           j = matrix%COL( l ) ; jj = data%ORDER( j )
           IF ( data%explicit_scaling ) THEN
              val = matrix%VAL( l ) / ( data%SCALE( i ) * data%SCALE( j ) )
              data%matrix%VAL( l ) = val
           ELSE
              val = matrix%VAL( l )
           END IF
           IF ( ii >= jj ) THEN
             data%matrix_dense( ii, jj ) = data%matrix_dense( ii, jj ) + val
           ELSE
             data%matrix_dense( jj, ii ) = data%matrix_dense( jj, ii ) + val
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, matrix%n
           ii = data%ORDER( i )
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             j = matrix%COL( l ) ; jj = data%ORDER( j )
             IF ( data%explicit_scaling ) THEN
                val = matrix%VAL( l ) / ( data%SCALE( i ) * data%SCALE( j ) )
                data%matrix%VAL( l ) = val
             ELSE
                val = matrix%VAL( l )
             END IF
             IF ( ii >= jj ) THEN
               data%matrix_dense( ii, jj ) = data%matrix_dense( ii, jj ) + val
             ELSE
               data%matrix_dense( jj, ii ) = data%matrix_dense( jj, ii ) + val
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, matrix%n
           ii = data%ORDER( i )
           DO j = 1, i
             jj =  data%ORDER( j )
             l = l + 1
             IF ( data%explicit_scaling ) THEN
                val = matrix%VAL( l ) / ( data%SCALE( i ) * data%SCALE( j ) )
                data%matrix%VAL( l ) = val
             ELSE
                val = matrix%VAL( l )
             END IF
             IF ( ii >= jj ) THEN
               data%matrix_dense( ii, jj ) = data%matrix_dense( ii, jj ) + val
             ELSE
               data%matrix_dense( jj, ii ) = data%matrix_dense( jj, ii ) + val
             END IF
           END DO
         END DO
       END SELECT

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = POTR =

       CASE ( 'potr' )

         CALL POTRF( 'L', data%n, data%matrix_dense, data%n,                   &
                     inform%lapack_error )

         IF ( inform%lapack_error < 0 .AND. control%print_level > 0 .AND.      &
              control%out > 0 ) WRITE( control%out, "( A,                      &
        &  ' LAPACK POTRF error code = ', I0 )" ) prefix, inform%lapack_error

         SELECT CASE( inform%lapack_error )
         CASE ( 0 )
           inform%status = GALAHAD_ok
           inform%two_by_two_pivots = 0
           inform%rank = data%n
           inform%negative_eigenvalues = 0
           inform%entries_in_factors =                                         &
             INT( data%n * ( data%n + 1 ) / 2, long )
           inform%flops_elimination = INT( data%n ** 3 / 6, long )
         CASE ( 1 : )
           inform%status = GALAHAD_error_inertia
         CASE DEFAULT
           inform%status = GALAHAD_error_lapack
         END SELECT

!  = SYTR =

       CASE ( 'sytr' )

         CALL SYTRF( 'L', data%n, data%matrix_dense, data%n,                   &
                      data%PIVOTS, data%WORK, data%sytr_lwork,                 &
                      inform%lapack_error )

         IF ( inform%lapack_error < 0 .AND. control%print_level > 0 .AND.      &
              control%out > 0 ) WRITE( control%out, "( A,                      &
        &  ' LAPACK SYTRF error code = ', I0 )" ) prefix, inform%lapack_error

         SELECT CASE( inform%lapack_error )
         CASE ( 0 : )
           inform%status = GALAHAD_ok
           inform%two_by_two_pivots = 0
           inform%rank = data%n
           inform%negative_eigenvalues = 0
           k = 1
           DO                                   ! run through the pivots
             IF ( k > data%n ) EXIT
             IF ( data%PIVOTS( k ) > 0 ) THEN   ! a 1 x 1 pivot
               IF ( data%matrix_dense( k, k ) < 0.0_wp ) THEN
                 inform%negative_eigenvalues =                                 &
                   inform%negative_eigenvalues + 1
               ELSE IF ( data%matrix_dense( k, k ) == 0.0_wp ) THEN
                 inform%rank = inform%rank - 1
               END IF
               k = k + 1
             ELSE                               ! a 2 x 2 pivot
               inform%two_by_two_pivots = inform%two_by_two_pivots + 1
               IF ( data%matrix_dense( k, k ) *                                &
                    data%matrix_dense( k + 1, k + 1 ) <                        &
                    data%matrix_dense( k + 1, k ) ** 2 ) THEN
                 inform%negative_eigenvalues = inform%negative_eigenvalues + 1
               ELSE IF ( data%matrix_dense( k, k ) < 0.0_wp ) THEN
                 inform%negative_eigenvalues = inform%negative_eigenvalues + 2
               END IF
               k = k + 2
             END IF
           END DO
           inform%entries_in_factors =                                         &
             INT( data%n * ( data%n + 1 ) / 2, long )
           inform%flops_elimination = INT( data%n ** 3 / 3, long )
         CASE DEFAULT
           inform%status = GALAHAD_error_lapack
         END SELECT

       END SELECT

!  = PBTR =

     CASE ( 'pbtr' )

       data%matrix_dense( : inform%semi_bandwidth + 1, : matrix%n ) = 0.0_wp

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         DO l = 1, matrix%ne
           i = matrix%ROW( l ) ; ii = data%ORDER( i )
           j = matrix%COL( l ) ; jj = data%ORDER( j )
           IF ( data%explicit_scaling ) THEN
              val = matrix%VAL( l ) / ( data%SCALE( i ) * data%SCALE( j ) )
              data%matrix%VAL( l ) = val
           ELSE
              val = matrix%VAL( l )
           END IF
           IF ( ii >= jj ) THEN
             data%matrix_dense( ii - jj + 1, jj ) =                            &
               data%matrix_dense( ii - jj + 1, jj ) + val
           ELSE
             data%matrix_dense( jj - ii + 1, ii ) =                            &
               data%matrix_dense( jj - ii + 1, ii ) + val
           END IF
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         DO i = 1, matrix%n
           ii = data%ORDER( i )
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             j = matrix%COL( l ) ; jj = data%ORDER( j )
             IF ( data%explicit_scaling ) THEN
                val = matrix%VAL( l ) / ( data%SCALE( i ) * data%SCALE( j ) )
                data%matrix%VAL( l ) = val
             ELSE
                val = matrix%VAL( l )
             END IF
             IF ( ii >= jj ) THEN
               data%matrix_dense( ii - jj + 1, jj ) =                          &
                 data%matrix_dense( ii - jj + 1, jj ) + val
             ELSE
               data%matrix_dense( jj - ii + 1, ii ) =                          &
                 data%matrix_dense( jj - ii + 1, ii ) + val
             END IF
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, matrix%n
           ii = data%ORDER( i )
           DO j = 1, i
             jj = data%ORDER( j )
             l = l + 1
             IF ( data%explicit_scaling ) THEN
                val = matrix%VAL( l ) / ( data%SCALE( i ) * data%SCALE( j ) )
                data%matrix%VAL( l ) = val
             ELSE
                val = matrix%VAL( l )
             END IF
             IF ( ii >= jj ) THEN
               data%matrix_dense( ii - jj + 1, jj ) =                          &
                 data%matrix_dense( ii - jj + 1, jj ) + val
             ELSE
               data%matrix_dense( jj - ii + 1, ii ) =                          &
                 data%matrix_dense( jj - ii + 1, ii ) + val
             END IF
           END DO
         END DO
       END SELECT

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PBTRF( 'L', data%n, inform%semi_bandwidth, data%matrix_dense,      &
                   inform%semi_bandwidth + 1, inform%lapack_error )

       IF ( inform%lapack_error < 0 .AND. control%print_level > 0 .AND.        &
            control%out > 0 ) WRITE( control%out, "( A,                        &
      &  ' LAPACK POTRF error code = ', I0 )" ) prefix, inform%lapack_error

       SELECT CASE( inform%lapack_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%two_by_two_pivots = 0
         inform%rank = data%n
         inform%negative_eigenvalues = 0
         inform%entries_in_factors =                                           &
           INT( ( ( 2 * data%n - inform%semi_bandwidth ) *                     &
                  ( inform%semi_bandwidth + 1 ) ) / 2, long )
         inform%flops_elimination =                                            &
           INT( 2 * data%n * ( inform%semi_bandwidth + 1 ) ** 2, long )
       CASE ( 1 : )
         inform%status = GALAHAD_error_inertia
       CASE DEFAULT
         inform%status = GALAHAD_error_lapack
       END SELECT

     END SELECT

!  record external factorize time

 800 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%factorize_external = time_now - time
     inform%time%clock_factorize_external = clock_now - clock

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%factorize =inform%time%factorize + time_now - time_start
     inform%time%clock_factorize =                                             &
       inform%time%clock_factorize + clock_now - clock_start
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of SLS_factorize

     END SUBROUTINE SLS_factorize

! -*-*-*-*-*-*-   S L S _ S O L V E _ I R   S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SLS_solve_ir( matrix, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Given a symmetric matrix A and its SLS factors, solve the system A x = b,
!  where b is input in X, and the solution x overwrites X

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( INOUT ) , DIMENSION ( : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER :: i, j, l, iter, n
     REAL :: time_start, time_now
     REAL ( KIND = wp ) :: clock_start, clock_now
     REAL ( KIND = wp ) :: residual, residual_zero, val
     LOGICAL :: filexx

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  Set default inform values

     inform%bad_alloc = ''

!  if desired, write the input rhs to an output file

     IF ( control%generate_matrix_file .AND.                                   &
          control%matrix_file_device > 0 ) THEN
       INQUIRE( FILE = control%matrix_file_name, EXIST = filexx )
       IF ( filexx ) THEN
         OPEN( control%matrix_file_device, FILE = control%matrix_file_name,    &
               FORM = 'FORMATTED', STATUS = 'OLD', POSITION = 'APPEND',        &
               IOSTAT = i )
       ELSE
         OPEN( control%matrix_file_device, FILE = control%matrix_file_name,    &
               FORM = 'FORMATTED', STATUS = 'NEW', IOSTAT = i )
       END IF
       DO i = 1, matrix%n
           WRITE( control%matrix_file_device, * ) X( i )
         END DO
       CLOSE( control%matrix_file_device )
     END IF

!  No refinement is required (or Pardiso is called with its internal refinement)
!  -----------------------------------------------------------------------------

     IF ( control%max_iterative_refinements <= 0 .OR.                          &
          data%solver( 1 : data%len_solver ) == 'pardiso' ) THEN

!  solve A x = b with calculated scaling factors

       IF ( data%explicit_scaling ) THEN
         n = MATRIX%n
         X( : n ) = X( : n ) / data%SCALE( : n )
         CALL SLS_solve_one_rhs( data%matrix, X, data, control, inform )
         X( : n ) = X( : n ) / data%SCALE( : n )

!  solve A x = b without calculated scaling factors

       ELSE
         CALL SLS_solve_one_rhs( matrix, X, data, control, inform )
       END IF

!  Iterative refinement is required
!  --------------------------------

     ELSE

!  Allocate space if necessary

       n = MATRIX%n
       IF ( data%set_res /= n ) THEN
         CALL SPACE_resize_array( n, data%RES,                                 &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%RES' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( n, data%B,                                   &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
         data%set_res = n
       END IF

!  Compute the original residual

       IF ( data%explicit_scaling ) THEN
         data%B( : n ) = X( : n ) / data%SCALE( : n )
       ELSE
         data%B( : n ) = X( : n )
       END IF
       data%RES( : n ) = data%B( : n )
       X( : n ) = 0.0_wp
       residual_zero = MAXVAL( ABS( data%B( : n ) ) )

!  Solve the system with iterative refinement

       IF ( control%print_level > 1 .AND. control%out > 0 )                    &
         WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )    &
           prefix, residual_zero, 0.0_wp

       DO iter = 0, control%max_iterative_refinements
         inform%iterative_refinements = iter

!  Use factors of the system matrix to solve for the correction

         IF ( data%explicit_scaling ) THEN
           CALL SLS_solve_one_rhs( data%matrix, data%RES( : n ), data,         &
                                   control, inform )
         ELSE
           CALL SLS_solve_one_rhs( matrix, data%RES( : n ), data, control,     &
                                   inform )
         END IF
         IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Update the estimate of the solution

         X( : n ) = X( : n ) + data%RES( : n )

!  Form the residuals

         IF ( iter < control%max_iterative_refinements ) THEN
           data%RES( : n ) = data%B( : n )
           IF ( data%explicit_scaling ) THEN
             IF ( data%solver( 1 : data%len_solver ) /= 'potr' .AND.           &
                  data%solver( 1 : data%len_solver ) /= 'sytr' .AND.           &
                  data%solver( 1 : data%len_solver ) /= 'pbtr' ) THEN
               SELECT CASE ( SMT_get( data%matrix%type ) )
               CASE ( 'COORDINATE' )
                 DO l = 1, data%matrix%ne
                   i = data%matrix%ROW( l ) ; j = data%matrix%COL( l )
                   IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                     val = data%matrix%VAL( l )
                     data%RES( i ) = data%RES( i ) - val * X( j )
                     IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                   END IF
                 END DO
               CASE ( 'SPARSE_BY_ROWS', 'DENSE' )
                 IF ( data%solver( 1 : data%len_solver ) == 'ma77' ) THEN
                   DO i = 1, n
                     DO l = data%matrix%PTR( i ), data%matrix%PTR( i + 1 ) - 1
                       j = data%matrix%ROW( l )
                       IF ( j >= 1 .AND. j <= n ) data%RES( i ) =              &
                         data%RES( i ) - data%matrix%VAL( l ) * X( j )
                     END DO
                   END DO
                 ELSE
                   DO i = 1, n
                     DO l = data%matrix%PTR( i ), data%matrix%PTR( i + 1 ) - 1
                       j = data%matrix%COL( l )
                       IF ( j >= 1 .AND. j <= n ) THEN
                         val = data%matrix%VAL( l )
                         data%RES( i ) = data%RES( i ) - val * X( j )
                         IF ( i /= j )                                         &
                           data%RES( j ) = data%RES( j ) - val * X( i )
                       END IF
                     END DO
                   END DO
                 END IF
               END SELECT
             ELSE
               SELECT CASE ( SMT_get( matrix%type ) )
               CASE ( 'COORDINATE' )
                 DO l = 1, matrix%ne
                   i = matrix%ROW( l ) ; j = matrix%COL( l )
                   IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                     val = data%matrix%VAL( l )
                     data%RES( i ) = data%RES( i ) - val * X( j )
                     IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                   END IF
                 END DO
               CASE ( 'SPARSE_BY_ROWS' )
                 DO i = 1, n
                   DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
                     j = matrix%COL( l )
                     IF ( j >= 1 .AND. j <= n ) THEN
                       val = data%matrix%VAL( l )
                       data%RES( i ) = data%RES( i ) - val * X( j )
                       IF ( i /= j )                                           &
                         data%RES( j ) = data%RES( j ) - val * X( i )
                     END IF
                   END DO
                 END DO
               CASE ( 'DENSE' )
                 l = 0
                 DO i = 1, n
                   DO j = 1, i
                     l = l + 1
                     val = data%matrix%VAL( l )
                     data%RES( i ) = data%RES( i ) - val * X( j )
                     IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                   END DO
                 END DO
               END SELECT
             END IF
           ELSE
             SELECT CASE ( SMT_get( matrix%type ) )
             CASE ( 'COORDINATE' )
               DO l = 1, matrix%ne
                 i = matrix%ROW( l ) ; j = matrix%COL( l )
                 IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                   val = matrix%VAL( l )
                   data%RES( i ) = data%RES( i ) - val * X( j )
                   IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                 END IF
               END DO
             CASE ( 'SPARSE_BY_ROWS' )
               DO i = 1, n
                 DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
                   j = matrix%COL( l )
                   IF ( j >= 1 .AND. j <= n ) THEN
                     val = matrix%VAL( l )
                     data%RES( i ) = data%RES( i ) - val * X( j )
                     IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                   END IF
                 END DO
               END DO
             CASE ( 'DENSE' )
               l = 0
               DO i = 1, n
                 DO j = 1, i
                   l = l + 1
                   val = matrix%VAL( l )
                   data%RES( i ) = data%RES( i ) - val * X( j )
                   IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                 END DO
               END DO
             END SELECT
           END IF
         END IF

!  Check for convergence

         residual = MAXVAL( ABS( data%RES( : n ) ) )
         IF ( control%print_level >= 1 .AND. control%out > 0 )                 &
           WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )  &
             prefix, residual, MAXVAL( ABS( X( : n ) ) )

         IF ( residual < MAX( control%acceptable_residual_absolute,            &
                control%acceptable_residual_relative * residual_zero ) ) EXIT
       END DO
       IF ( data%explicit_scaling ) X( : n ) = X( : n ) / data%SCALE( : n )
     END IF

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve =inform%time%solve + time_now - time_start
     inform%time%clock_solve =                                                 &
       inform%time%clock_solve + clock_now - clock_start
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of subroutine SLS_solve_ir

     END SUBROUTINE SLS_solve_ir

! -*-*-   S L S _ S O L V E _ I R _ M U L T I P L E   S U B R O U T I N E  -*-*-

     SUBROUTINE SLS_solve_ir_multiple( matrix, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Given a symmetric matrix A and its SLS factors, solve the system A X = B,
!  where B is input in X, and the solution X overwrites X

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
      REAL ( KIND = wp ), INTENT( INOUT ) , DIMENSION ( : , : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER :: i, j, l, iter, n, nrhs
     REAL :: time_start, time_now
     REAL ( KIND = wp ) :: clock_start, clock_now
     REAL ( KIND = wp ) :: val
     LOGICAL :: too_big

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  Set default inform values

     inform%bad_alloc = ''

!  No refinement is required (or Pardiso is called with its internal refinement)
!  -----------------------------------------------------------------------------

     IF ( control%max_iterative_refinements <= 0 .OR.                          &
          data%solver( 1 : data%len_solver ) == 'pardiso' ) THEN

!  Solve A X = B

       IF ( data%explicit_scaling ) THEN
         n = matrix%n ; nrhs = SIZE( X, 2 )
         DO i = 1, nrhs
           X( : n, i ) = X( : n, i ) / data%SCALE( : n )
         END DO
!write(6,"( /,  ' x ', ( 5ES12.5 ) )" ) X( : n, 1 )
         CALL SLS_solve_multiple_rhs( data%matrix_scale, X, data,              &
                                      control, inform )
!write(6,"( /,  ' x ', ( 5ES12.5 ) )" ) X( : n, 1 )
         DO i = 1, nrhs
           X( : n, i ) = X( : n, i ) / data%SCALE( : n )
         END DO
       ELSE
!write(6,"( /,  ' x ', ( 5ES12.5 ) )" ) X( : matrix%n, 1 )
         CALL SLS_solve_multiple_rhs( matrix, X, data, control, inform )
!write(6,"( /,  ' x ', ( 5ES12.5 ) )" ) X( : matrix%n, 1 )
       END IF

!  Iterative refinement is required
!  --------------------------------

     ELSE

!  Allocate space if necessary

       n = matrix%n ; nrhs = SIZE( X, 2 )
       IF ( data%set_res2 /= n ) THEN
         CALL SPACE_resize_array( n, nrhs, data%B2,                            &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( n, nrhs, data%RES2,                          &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%RES2' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( nrhs, data%RESIDUALS,                        &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%RESIDUALS' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( nrhs, data%RESIDUALS_zero,                   &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%RESIDUALS_zero' ; GO TO 900 ; END IF
         data%set_res2 = n
       END IF

!  Compute the original residuals

       IF ( data%explicit_scaling ) THEN
         DO i = 1, nrhs
           data%B2( : n, i ) = X( : n, i ) / data%SCALE( : n )
         END DO
       ELSE
         data%B2( : n, : nrhs ) = X( : n, : nrhs )
       END IF
       data%RES2( : n, : nrhs ) = data%B2( : n, : nrhs )
       X( : n, : nrhs ) = 0.0_wp

       DO i = 1, nrhs
         data%RESIDUALS_zero( i ) = MAXVAL( ABS( data%B2( : n, i ) ) )
       END DO

!  Solve the block system with iterative refinement

       IF ( control%print_level >= 1 .AND. control%out > 0 )                   &
         WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )    &
           prefix, MAXVAL( data%RESIDUALS_zero( : nrhs ) ),                    &
           MAXVAL( ABS( X( : n, : nrhs ) ) )

       DO iter = 0, control%max_iterative_refinements
         inform%iterative_refinements = iter

!  Use factors of the system matrix to solve for the corrections

         IF ( data%explicit_scaling ) THEN
           CALL SLS_solve_multiple_rhs( data%matrix,                           &
                                        data%RES2( : n, : nrhs ),              &
                                        data, control, inform )
         ELSE
           CALL SLS_solve_multiple_rhs( matrix, data%RES2( : n, : nrhs ),      &
                                        data, control, inform )
         END IF
         IF ( inform%status /= GALAHAD_ok ) GO TO 900

!  Update the estimates of the solutions

         X( : n, : nrhs ) = X( : n, : nrhs ) + data%RES2( : n, : nrhs )

!  Form the residuals

         IF ( iter < control%max_iterative_refinements ) THEN
           data%RES2( : n, : nrhs ) = data%B2( : n, : nrhs )
           IF ( data%explicit_scaling ) THEN
             IF ( data%solver( 1 : data%len_solver ) /= 'potr' .AND.           &
                  data%solver( 1 : data%len_solver ) /= 'sytr' .AND.           &
                  data%solver( 1 : data%len_solver ) /= 'pbtr' ) THEN
               SELECT CASE ( SMT_get( data%matrix%type ) )
               CASE ( 'COORDINATE' )
                 DO l = 1, data%matrix%ne
                   i = data%matrix%ROW( l ) ; j = data%matrix%COL( l )
                   IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                     val = data%matrix%VAL( l )
                     data%RES2( i, : nrhs )                                    &
                       = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                     IF ( i /= j ) data%RES2( j, : nrhs )                      &
                       = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                   END IF
                 END DO
               CASE ( 'SPARSE_BY_ROWS', 'DENSE' )
                 IF ( data%solver( 1 : data%len_solver ) == 'ma77' ) THEN
                   DO i = 1, n
                     DO l = data%matrix%PTR( i ), data%matrix%PTR( i + 1 ) - 1
                       j = data%matrix%ROW( l )
                       IF ( j >= 1 .AND. j <= n )                              &
                          data%RES2( i, : nrhs ) = data%RES2( i, : nrhs )      &
                            - data%matrix%VAL( l ) * X( j, : nrhs )
                     END DO
                   END DO
                 ELSE
                   DO i = 1, n
                     DO l = data%matrix%PTR( i ), data%matrix%PTR( i + 1 ) - 1
                       j = data%matrix%COL( l )
                       IF ( j >= 1 .AND. j <= n ) THEN
                         val = data%matrix%VAL( l )
                         data%RES2( i, : nrhs )                                &
                           = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                         IF ( i /= j ) data%RES2( j, : nrhs )                  &
                           = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                       END IF
                     END DO
                   END DO
                 END IF
               END SELECT
             ELSE
               SELECT CASE ( SMT_get( matrix%type ) )
               CASE ( 'COORDINATE' )
                 DO l = 1, matrix%ne
                   i = matrix%ROW( l ) ; j = matrix%COL( l )
                   IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                     val = data%matrix%VAL( l )
                     data%RES2( i, : nrhs )                                    &
                       = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                     IF ( i /= j ) data%RES2( j, : nrhs )                      &
                       = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                   END IF
                 END DO
               CASE ( 'SPARSE_BY_ROWS' )
                 DO i = 1, n
                   DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
                     j = matrix%COL( l )
                     IF ( j >= 1 .AND. j <= n ) THEN
                       val = data%matrix%VAL( l )
                       data%RES2( i, : nrhs )                                  &
                         = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                       IF ( i /= j ) data%RES2( j, : nrhs )                    &
                         = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                     END IF
                   END DO
                 END DO
               CASE ( 'DENSE' )
                 l = 0
                 DO i = 1, n
                   DO j = 1, i
                     l = l + 1
                     val = data%matrix%VAL( l )
                     data%RES2( i, : nrhs )                                    &
                       = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                     IF ( i /= j ) data%RES2( j, : nrhs )                      &
                       = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                   END DO
                 END DO
               END SELECT
             END IF
           ELSE
             SELECT CASE ( SMT_get( matrix%type ) )
             CASE ( 'COORDINATE' )
               DO l = 1, matrix%ne
                 i = matrix%ROW( l ) ; j = matrix%COL( l )
                 IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                   val = matrix%VAL( l )
                   data%RES2( i, : nrhs )                                      &
                     = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                   IF ( i /= j ) data%RES2( j, : nrhs )                        &
                     = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                 END IF
               END DO
             CASE ( 'SPARSE_BY_ROWS' )
               DO i = 1, n
                 DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
                   j = matrix%COL( l )
                   IF ( j >= 1 .AND. j <= n ) THEN
                     val = matrix%VAL( l )
                     data%RES2( i, : nrhs )                                    &
                       = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                     IF ( i /= j ) data%RES2( j, : nrhs )                      &
                       = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                   END IF
                 END DO
               END DO
             CASE ( 'DENSE' )
               l = 0
               DO i = 1, n
                 DO j = 1, i
                   l = l + 1
                   val = matrix%VAL( l )
                   data%RES2( i, : nrhs )                                      &
                     = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                   IF ( i /= j ) data%RES2( j, : nrhs )                        &
                     = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                 END DO
               END DO
             END SELECT
           END IF
         END IF

         DO i = 1, nrhs
           data%RESIDUALS( i ) = MAXVAL( ABS( data%RES2( : n, i ) ) )
         END DO

         IF ( control%print_level >= 1 .AND. control%out > 0 )                 &
           WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )  &
             prefix, MAXVAL( data%RESIDUALS( : nrhs ) ),                       &
             MAXVAL( ABS( X( : n, : nrhs ) ) )

!  Check for convergence

         too_big = .FALSE.
         DO i = 1, nrhs
           IF ( data%RESIDUALS( i ) >                                          &
                  MAX( control%acceptable_residual_absolute,                   &
                    control%acceptable_residual_relative *                     &
                      data%residuals_zero( i ) ) ) THEN
             too_big = .TRUE. ; EXIT ; END IF
         END DO
         IF ( .NOT. too_big ) EXIT
       END DO
       IF ( data%explicit_scaling ) THEN
         DO i = 1, nrhs
           X( : n, i ) = X( : n, i ) / data%SCALE( : n )
         END DO
       END IF
     END IF

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve =inform%time%solve + time_now - time_start
     inform%time%clock_solve =                                                 &
       inform%time%clock_solve + clock_now - clock_start
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of subroutine SLS_solve_ir_multiple

     END SUBROUTINE SLS_solve_ir_multiple

!-*-*-*-*-   S L S _ S O L V E _ O N E _ R H S   S U B R O U T I N E   -*-*-*-

     SUBROUTINE SLS_solve_one_rhs( matrix, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the linear system using the factors obtained in the factorization

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( INOUT ) :: X( : )
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     REAL :: time, time_now
     REAL ( KIND = wp ) :: clock, clock_now

!  solver-dependent solution

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SLS_copy_control_to_sils( control, data%sils_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         CALL SILS_solve( matrix, data%sils_factors, X, data%sils_control,     &
                          data%sils_sinfo )
       CASE DEFAULT
         CALL SILS_solve( data%matrix, data%sils_factors, X,                   &
                          data%sils_control, data%sils_sinfo )
       END SELECT
       inform%sils_sinfo = data%sils_sinfo
       inform%status = data%sils_sinfo%flag
       inform%alloc_status = data%sils_sinfo%stat
       inform%condition_number_1 = data%sils_sinfo%cond
       inform%condition_number_2 = data%sils_sinfo%cond2
       inform%backward_error_1 = data%sils_sinfo%berr
       inform%backward_error_2 = data%sils_sinfo%berr2
       inform%forward_error = data%sils_sinfo%error

!  = MA57 =

     CASE ( 'ma57' )
       CALL SLS_copy_control_to_ma57( control, data%ma57_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         CALL MA57_solve( matrix, data%ma57_factors, X, data%ma57_control,     &
                          data%ma57_sinfo )
       CASE DEFAULT
         CALL MA57_solve( data%matrix, data%ma57_factors, X,                   &
                          data%ma57_control, data%ma57_sinfo )
       END SELECT
       inform%ma57_sinfo = data%ma57_sinfo
       inform%status = data%ma57_sinfo%flag
       IF ( inform%status == - 3 )                                             &
         inform%alloc_status = data%ma57_sinfo%stat

!  = MA77 =

     CASE ( 'ma77' )
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_ma77( control, data%ma77_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                  &
                          data%ma77_control, data%ma77_info,                   &
                          scale = data%SCALE )
       ELSE
         CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                  &
                          data%ma77_control, data%ma77_info )
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       IF ( inform%status /= GALAHAD_ok ) GO TO 800
       X( : data%n ) = data%X2( : data%n, 1 )

!  = MA86 =

     CASE ( 'ma86' )
       CALL SLS_copy_control_to_ma86( control, data%ma86_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA86_solve( X( : data%n ), data%ORDER, data%ma86_keep,             &
                        data%ma86_control, data%ma86_info )
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag

!  = MA87 =

     CASE ( 'ma87' )
       CALL SLS_copy_control_to_ma87( control, data%ma87_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA87_solve( X( : data%n ), data%ORDER, data%ma87_keep,             &
                        data%ma87_control, data%ma87_info )
       inform%ma87_info = data%ma87_info
       inform%status = data%ma87_info%flag

!  = MA97 =

     CASE ( 'ma97' )
       CALL SLS_copy_control_to_ma97( control, data%ma97_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,       &
                        data%ma97_control, data%ma97_info )
       CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )

!  = SSIDS =

     CASE ( 'ssids' )
       CALL SLS_copy_control_to_ssids( control, data%ssids_options )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,    &
                        data%ssids_options, data%ssids_inform )
       CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

     CASE ( 'pardiso' )
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
       data%pardiso_iparm( 6 ) = 1
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%PARDISO_PT, 1, 1, data%pardiso_mtype, 33,            &
                     data%matrix%n, data%matrix%VAL( : data%ne ),              &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), 1,                         &
                     data%pardiso_iparm( : 64 ),                               &
                     control%print_level_solver,                               &
                     data%X2( : data%matrix%n, : 1 ),                          &
                     data%B2( : data%matrix%n, : 1 ), inform%pardiso_error,    &
                     data%pardiso_dparm( 1 : 64 ) )
!v3                  data%B2( : data%matrix%n, : 1 ), inform%pardiso_error )
       inform%pardiso_iparm = data%pardiso_iparm
       inform%pardiso_dparm = data%pardiso_dparm

       SELECT CASE( inform%pardiso_error )
       CASE ( - 1 )
         inform%status = GALAHAD_error_restrictions
       CASE ( GALAHAD_unavailable_option )
         inform%status = GALAHAD_unavailable_option
       CASE ( - 103 : GALAHAD_unavailable_option - 1,                          &
              GALAHAD_unavailable_option + 1 : - 2 )
         inform%status = GALAHAD_error_pardiso
       CASE DEFAULT
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%pardiso_iparm( 7 )
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%X2( : data%n, 1 )

!  = WSMP =

     CASE ( 'wsmp' )

       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,                &
                                      data%wsmp_dparm )
       data%wsmp_iparm( 2 ) = 4
       data%wsmp_iparm( 3 ) = 5
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1,      &
                   data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),            &
                   data%wsmp_iparm, data%wsmp_dparm )
!write(6,*) ' solv threads used = ', data%wsmp_iparm( 33 )
       inform%wsmp_iparm = data%wsmp_iparm
       inform%wsmp_dparm = data%wsmp_dparm
       inform%wsmp_error = data%wsmp_iparm( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_iparm( 6 )
       CASE ( - 102 )
         inform%status = GALAHAD_error_allocate
       CASE DEFAULT
         inform%status = GALAHAD_error_wsmp
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%B2( : data%n, 1 )

!  = LAPACK solvers POTR, SYTR or PBTR =

     CASE ( 'potr', 'sytr', 'pbtr' )

       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( data%ORDER( : data%n ), 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = POTR =

       CASE ( 'potr' )
         CALL POTRS( 'L', data%n, 1, data%matrix_dense, data%n, data%X2,       &
                     data%n, inform%lapack_error )

!  = SYTR =

       CASE ( 'sytr' )
         CALL SYTRS( 'L', data%n, 1, data%matrix_dense, data%n,                &
                     data%PIVOTS, data%X2, data%n, inform%lapack_error )

!  = PBTR =

       CASE ( 'pbtr' )
         CALL PBTRS( 'L', data%n, inform%semi_bandwidth, 1, data%matrix_dense, &
                      inform%semi_bandwidth + 1, data%X2, data%n,              &
                      inform%lapack_error )
       END SELECT

       SELECT CASE( inform%lapack_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
       CASE DEFAULT
         inform%status = GALAHAD_error_lapack
       END SELECT
       IF ( inform%status == GALAHAD_ok )                                      &
         X( : data%n )= data%X2( data%ORDER( : data%n ), 1 )

     END SELECT

!  record external solve time

 800 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve_external = time_now - time
     inform%time%clock_solve_external = clock_now - clock

 900 CONTINUE
     RETURN

!  End of SLS_solve_one_rhs

     END SUBROUTINE SLS_solve_one_rhs

!-*-*-  S L S _ S O L V E _ M U L T I P L E _ R H S   S U B R O U T I N E  -*-*-

     SUBROUTINE SLS_solve_multiple_rhs( matrix, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the linear system using the factors obtained in the factorization

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : , : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: lx, nrhs
     REAL :: time, time_now
     REAL ( KIND = wp ) :: clock, clock_now

!  solver-dependent solution

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SLS_copy_control_to_sils( control, data%sils_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         CALL SILS_solve( matrix, data%sils_factors, X, data%sils_control,     &
                          data%sils_sinfo )
       CASE DEFAULT
         CALL SILS_solve( data%matrix, data%sils_factors, X,                   &
                          data%sils_control, data%sils_sinfo )
       END SELECT
       inform%ma57_sinfo = data%ma57_sinfo
       inform%status = data%sils_sinfo%flag
       inform%alloc_status = data%sils_sinfo%stat
       inform%condition_number_1 = data%sils_sinfo%cond
       inform%condition_number_2 = data%sils_sinfo%cond2
       inform%backward_error_1 = data%sils_sinfo%berr
       inform%backward_error_2 = data%sils_sinfo%berr2
       inform%forward_error = data%sils_sinfo%error

!  = MA57 =

     CASE ( 'ma57' )
       CALL SLS_copy_control_to_ma57( control, data%ma57_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         CALL MA57_solve( matrix, data%ma57_factors, X, data%ma57_control,     &
                          data%ma57_sinfo )
       CASE DEFAULT
         CALL MA57_solve( data%matrix, data%ma57_factors, X,                   &
                          data%ma57_control, data%ma57_sinfo )
       END SELECT
       inform%sils_sinfo = data%sils_sinfo
       inform%status = data%ma57_sinfo%flag
       IF ( inform%status == - 3 )                                             &
         inform%alloc_status = data%ma57_sinfo%stat

!  = MA77 =

     CASE ( 'ma77' )
       lx = SIZE( X, 1 ) ; nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_ma77( control, data%ma77_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         CALL MA77_solve( nrhs, lx, X, data%ma77_keep,                         &
                          data%ma77_control, data%ma77_info,                   &
                          scale = data%SCALE )
       ELSE
         CALL MA77_solve( nrhs, lx, X, data%ma77_keep,                         &
                          data%ma77_control, data%ma77_info )
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )

!  = MA86 =

     CASE ( 'ma86' )
       lx = SIZE( X, 1 ) ; nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_ma86( control, data%ma86_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA86_solve( nrhs, lx, X, data%ORDER, data%ma86_keep,               &
                        data%ma86_control, data%ma86_info )
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag

!  = MA87 =

     CASE ( 'ma87' )
       lx = SIZE( X, 1 ) ; nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_ma87( control, data%ma87_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA87_solve( nrhs, lx, X, data%ORDER, data%ma87_keep,               &
                        data%ma87_control, data%ma87_info )
       inform%ma87_info = data%ma87_info
       inform%status = data%ma87_info%flag

!  = MA97 =

     CASE ( 'ma97' )
       lx = SIZE( X, 1 ) ; nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_ma97( control, data%ma97_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA97_solve( nrhs, X, lx, data%ma97_akeep, data%ma97_fkeep,         &
                        data%ma97_control, data%ma97_info )
       CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )

!  = SSIDS =

     CASE ( 'ssids' )
       inform%status = GALAHAD_unavailable_option
       lx = SIZE( X, 1 ) ; nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_ssids( control, data%ssids_options )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL SSIDS_solve( nrhs, X, lx, data%ssids_akeep, data%ssids_fkeep,      &
                        data%ssids_options, data%ssids_inform )
       CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

     CASE ( 'pardiso' )
       nrhs = SIZE( X, 2 )
       CALL SPACE_resize_array( data%n, nrhs, data%B2, inform%status,          &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
       data%pardiso_iparm( 6 ) = 1
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%PARDISO_PT, 1, 1, data%pardiso_mtype, 33,            &
                     data%matrix%n, data%matrix%VAL( : data%ne ),              &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), nrhs,                      &
                     data%pardiso_iparm( : 64 ),                               &
                     control%print_level_solver,                               &
                     X( : data%matrix%n, : nrhs ),                             &
                     data%B2( : data%matrix%n, : nrhs ), inform%pardiso_error, &
                     data%pardiso_dparm( 1 : 64 ) )
!v3                  data%B2( : data%matrix%n, : nrhs ), inform%pardiso_error )
       inform%pardiso_iparm = data%pardiso_iparm
       SELECT CASE( inform%pardiso_error )
       CASE ( - 1 )
         inform%status = GALAHAD_error_restrictions
       CASE ( GALAHAD_unavailable_option )
         inform%status = GALAHAD_unavailable_option
       CASE ( - 103 : GALAHAD_unavailable_option - 1,                          &
              GALAHAD_unavailable_option + 1 : - 2 )
         inform%status = GALAHAD_error_pardiso
       CASE DEFAULT
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%pardiso_iparm( 7 )
       END SELECT

!  = WSMP =

     CASE ( 'wsmp' )

       nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,                &
                                      data%wsmp_dparm )
       data%wsmp_iparm( 2 ) = 4
       data%wsmp_iparm( 3 ) = 5
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   X( : data%matrix%n, : nrhs ), matrix%n, nrhs,               &
                   data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),            &
                   data%wsmp_iparm, data%wsmp_dparm )

       inform%wsmp_iparm = data%wsmp_iparm
       inform%wsmp_dparm = data%wsmp_dparm
       inform%wsmp_error = data%wsmp_iparm( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_iparm( 6 )
       CASE ( - 102 )
         inform%status = GALAHAD_error_allocate
       CASE DEFAULT
         inform%status = GALAHAD_error_wsmp
       END SELECT

!  = LAPACK solvers POTR, SYTR or PBTR =

     CASE ( 'potr', 'sytr', 'pbtr' )
       nrhs = SIZE( X, 2 )

!  the matrix has been permuted

       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, nrhs, data%X2, inform%status,        &
                 inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
         data%X2( data%ORDER( : data%n ), : nrhs ) = X( : data%n, : nrhs )

         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = POTR =

         CASE ( 'potr' )
           CALL POTRS( 'L', data%n, nrhs, data%matrix_dense, data%n,           &
                        data%X2, data%n, inform%lapack_error )

!  = SYTR =

         CASE ( 'sytr' )
           CALL SYTRS( 'L', data%n, nrhs, data%matrix_dense, data%n,           &
                       data%PIVOTS, data%X2, data%n, inform%lapack_error )

!  = PBTR =

         CASE ( 'pbtr' )
           CALL PBTRS( 'L', data%n, inform%semi_bandwidth, nrhs,               &
                       data%matrix_dense, inform%semi_bandwidth + 1,           &
                       data%X2, data%n, inform%lapack_error )
         END SELECT
         IF ( inform%lapack_error == 0 )                                       &
           X( : data%n, : nrhs ) = data%X2( data%ORDER( : data%n ), : nrhs )

!  the matrix has not been permuted

       ELSE
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = POTR =

         CASE ( 'potr' )
           CALL POTRS( 'L', data%n, nrhs, data%matrix_dense, data%n,           &
                        X, data%n, inform%lapack_error )

!  = SYTR =

         CASE ( 'sytr' )
           CALL SYTRS( 'L', data%n, nrhs, data%matrix_dense, data%n,           &
                       data%PIVOTS, X, data%n, inform%lapack_error )

!  = PBTR =

         CASE ( 'pbtr' )
           CALL PBTRS( 'L', data%n, inform%semi_bandwidth, nrhs,               &
                       data%matrix_dense, inform%semi_bandwidth + 1,           &
                       X, data%n, inform%lapack_error )
         END SELECT
       END IF

       SELECT CASE( inform%lapack_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
       CASE DEFAULT
         inform%status = GALAHAD_error_lapack
       END SELECT

     END SELECT

!  record external solve time

     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve_external = time_now - time
     inform%time%clock_solve_external = clock_now - clock

 900 CONTINUE
     RETURN

!  End of SLS_solve_multiple_rhs

     END SUBROUTINE SLS_solve_multiple_rhs

!-*-*-*-*-*-*-*-   S L S _ T E R M I N A T E  S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE SLS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Deallocate all currently allocated arrays

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: info

!  solver-dependent termination

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SLS_copy_control_to_sils( control, data%sils_control )
       CALL SILS_finalize( data%sils_factors, data%sils_control, info )
       inform%status = info

!  = MA57 =

     CASE ( 'ma57' )
       CALL SLS_copy_control_to_ma57( control, data%ma57_control )
       CALL MA57_finalize( data%ma57_factors, data%ma57_control, info )
       inform%status = info

!  = MA77 =

     CASE ( 'ma77' )
       CALL SPACE_dealloc_array( data%PIVOTS, inform%status,                   &
                                 inform%alloc_status )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       CALL SPACE_dealloc_array( data%D, inform%status, inform%alloc_status )
       IF ( control%print_level <= 0 .OR. control%out <= 0 )                   &
           data%ma77_control%unit_error = - 1
       CALL MA77_finalise( data%ma77_keep, data%ma77_control, data%ma77_info )
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )

!  = MA86 =

     CASE ( 'ma86' )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       IF ( control%print_level <= 0 .OR. control%out <= 0 )                  &
           data%ma86_control%unit_error = - 1
       CALL MA86_finalise( data%ma86_keep, data%ma86_control )
       inform%status = 0

!  = MA87 =

     CASE ( 'ma87' )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       IF ( control%print_level <= 0 .OR. control%out <= 0 )                  &
           data%ma87_control%unit_error = - 1
       CALL MA87_finalise( data%ma87_keep, data%ma87_control )
       inform%status = 0

!  = MA97 =

     CASE ( 'ma97' )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       CALL MA97_finalise( data%ma97_akeep, data%ma97_fkeep )
       inform%status = 0

!  = SSIDS =

     CASE ( 'ssids' )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       CALL SSIDS_free( data%ssids_akeep, data%ssids_fkeep, inform%status )
       inform%status = 0

!  = PARDISO =

     CASE ( 'pardiso' )
       IF ( data%n > 0 ) THEN
         CALL PARDISO( data%PARDISO_PT, 1, 1, data%pardiso_mtype, - 1,         &
                       data%matrix%n, data%matrix%VAL( : data%ne ),            &
                       data%matrix%PTR( : data%matrix%n + 1 ),                 &
                       data%matrix%COL( : data%ne ),                           &
                       data%ORDER( : data%matrix%n ), 1,                       &
                       data%pardiso_iparm( : 64 ),                             &
                       control%print_level_solver,                             &
                       data%B2( : data%matrix%n, : 1 ),                        &
                       data%X2( : data%matrix%n, : 1 ), inform%pardiso_error,  &
                       data%pardiso_dparm( 1 : 64 ) )
!v3                    data%X2( : data%matrix%n, : 1 ), inform%pardiso_error )
         inform%pardiso_iparm = data%pardiso_iparm

         SELECT CASE( inform%pardiso_error )
         CASE ( - 1 )
           inform%status = GALAHAD_error_restrictions
         CASE ( GALAHAD_unavailable_option )
           inform%status = GALAHAD_unavailable_option
         CASE ( - 103 : GALAHAD_unavailable_option - 1,                        &
                GALAHAD_unavailable_option + 1 : - 2 )
           inform%status = GALAHAD_error_pardiso
         CASE DEFAULT
           inform%status = GALAHAD_ok
         END SELECT
       END IF
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       CALL SPACE_dealloc_array( data%MAPS, inform%status, inform%alloc_status )

!  = WSMP =

     CASE ( 'wsmp' )
       CALL wsmp_clear( )
       CALL wsffree( )
       CALL wsafree( )

!  = POTR =

     CASE ( 'potr' )

!  = SYTR =

     CASE ( 'sytr' )

!  = PBTR =

     CASE ( 'pbtr' )

     END SELECT

!  solver-independent termination

     IF ( ALLOCATED( data%matrix%type ) )                                      &
       DEALLOCATE( data%matrix%type, STAT = inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%ROW, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%COL, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%PTR, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%VAL, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix_dense, inform%status,               &
                               inform%alloc_status )
     IF ( ALLOCATED( data%matrix_scale%type ) )                                &
       DEALLOCATE( data%matrix_scale%type, STAT = inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix_scale%ROW, inform%status,           &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix_scale%COL, inform%status,           &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix_scale%PTR, inform%status,           &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix_scale%VAL, inform%status,           &
                               inform%alloc_status )

     CALL SPACE_dealloc_array( data%INVP, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%MRP, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%MAP, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%MAPS, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%MRP, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%INVP, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%LFLAG, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%ORDER, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%PIVOTS, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%B, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%RES, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%B2, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%RES2, inform%status, inform%alloc_status )
!if( ALLOCATED( data%SCALE ) ) THEN
!  write(6,*) ' data%SCALE allocated, length = ', SIZE( data%SCALE )
!endif
!  next line sometimes causes segfault under nvcc
     CALL SPACE_dealloc_array( data%SCALE, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%D, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%RESIDUALS, inform%status,                  &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%RESIDUALS_zero, inform%status,             &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%mc61_IW, inform%status,                    &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%mc77_IW, inform%status,                    &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%WORK, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%mc64_PERM, inform%status,                  &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%MAPS_scale, inform%status,                 &
                               inform%alloc_status )

     data%len_solver = - 1
     data%got_maps_scale = .FALSE.
     data%set_res = - 1
     data%set_res2 = - 1

     RETURN

!  End of SLS_terminate

     END SUBROUTINE SLS_terminate

!-*-*-*-*-*-*-*-   S L S _ E N Q U I R E  S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SLS_enquire( data, inform, PERM, PIVOTS, D, PERTURBATION )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Interogate the factorization to obtain additional information

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( data%n ) :: PIVOTS, PERM
     REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL, DIMENSION( 2, data%n ) :: D
     REAL ( KIND = wp ), INTENT( OUT ), OPTIONAL,                              &
       DIMENSION( data%n ) :: PERTURBATION

!  local variables

     INTEGER :: k
     REAL :: time_start, time_now
     REAL ( KIND = wp ) :: clock_start, clock_now

     inform%status = GALAHAD_ok

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  solver-dependent equiry

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SILS_enquire( data%sils_factors, PERM, PIVOTS, D, PERTURBATION )

!  = MA57 =

     CASE ( 'ma57' )
       CALL MA57_enquire( data%ma57_factors, PERM, PIVOTS, D, PERTURBATION )

!  = MA77 =

     CASE ( 'ma77' )

       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         IF ( data%must_be_definite ) THEN
           CALL MA77_enquire_posdef( D( 1, : ), data%ma77_keep,                &
                                     data%ma77_control, data%ma77_info )
           IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
         ELSE IF ( PRESENT( PIVOTS ) ) THEN
           CALL MA77_enquire_indef( PIVOTS, D, data%ma77_keep,                 &
                                    data%ma77_control, data%ma77_info)
         ELSE
           CALL SPACE_resize_array( data%n, data%PIVOTS,                       &
                                    inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           CALL MA77_enquire_indef( data%PIVOTS, D, data%ma77_keep,            &
                                    data%ma77_control, data%ma77_info )
         END IF
       ELSE
         IF ( PRESENT( PIVOTS ) ) THEN
           IF ( data%must_be_definite ) THEN
             inform%status = GALAHAD_error_access_pivots
           ELSE
             CALL SPACE_resize_array( 2, data%n, data%D, inform%status,        &
                                      inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) GO TO 900
             CALL MA77_enquire_indef( PIVOTS, data%D, data%ma77_keep,          &
                                      data%ma77_control, data%ma77_info )
           END IF
         END IF
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = MA86 =

     CASE ( 'ma86' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_wp ; D( 2, : ) = 0.0_wp
       END IF
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = MA87 =

     CASE ( 'ma87' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_wp ; D( 2, : ) = 0.0_wp
       END IF
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = MA97 =

     CASE ( 'ma97' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert
       IF ( data%must_be_definite ) THEN
         IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
         CALL MA97_enquire_posdef( data%ma97_akeep, data%ma97_fkeep,           &
                                   data%ma97_control, data%ma97_info, D( 1, :) )
         D( 2, : ) = 0.0_wp
       ELSE
         IF ( PRESENT( D ) ) THEN
           IF ( PRESENT( PIVOTS ) ) THEN
             CALL MA97_enquire_indef( data%ma97_akeep, data%ma97_fkeep,        &
                                      data%ma97_control, data%ma97_info,       &
                                      piv_order = PIVOTS, d = D )
           ELSE
             CALL SPACE_resize_array( data%n, data%PIVOTS,                     &
                                      inform%status, inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) GO TO 900
             CALL MA97_enquire_indef( data%ma97_akeep, data%ma97_fkeep,        &
                                      data%ma97_control, data%ma97_info, d = D )
           END IF
         ELSE
           CALL SPACE_resize_array( 2, data%n, data%D, inform%status,          &
                                    inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           IF ( PRESENT( PIVOTS ) ) THEN
             CALL MA97_enquire_indef( data%ma97_akeep, data%ma97_fkeep,        &
                                      data%ma97_control, data%ma97_info,       &
                                      piv_order = PIVOTS )
           END IF
         END IF
       END IF

!  = SSIDS =

     CASE ( 'ssids' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert
       IF ( data%must_be_definite ) THEN
         IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
         CALL SSIDS_enquire_posdef( data%ssids_akeep, data%ssids_fkeep,        &
                                    data%ssids_options, data%ssids_inform,     &
                                    D( 1, :) )
         D( 2, : ) = 0.0_wp
       ELSE
         IF ( PRESENT( D ) ) THEN
           IF ( PRESENT( PIVOTS ) ) THEN
             CALL SSIDS_enquire_indef( data%ssids_akeep, data%ssids_fkeep,     &
                                      data%ssids_options, data%ssids_inform,   &
                                      piv_order = PIVOTS, d = D )
           ELSE
             CALL SPACE_resize_array( data%n, data%PIVOTS,                     &
                                      inform%status, inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) GO TO 900
             CALL SSIDS_enquire_indef( data%ssids_akeep, data%ssids_fkeep,     &
                                      data%ssids_options, data%ssids_inform,   &
                                      d = D )
           END IF
         ELSE
           CALL SPACE_resize_array( 2, data%n, data%D, inform%status,          &
                                    inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           IF ( PRESENT( PIVOTS ) ) THEN
             CALL SSIDS_enquire_indef( data%ssids_akeep, data%ssids_fkeep,     &
                                      data%ssids_options, data%ssids_inform,   &
                                      piv_order = PIVOTS )
           END IF
         END IF
       END IF

!  = POTR =

     CASE ( 'potr' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_wp ; D( 2, : ) = 0.0_wp
       END IF
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = SYTR =

     CASE ( 'sytr' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         k = 1
         DO                                   ! run through the pivots
           IF ( k > data%n ) EXIT
           IF ( data%PIVOTS( k ) > 0 ) THEN   ! a 1 x 1 pivot
             IF ( data%matrix_dense( k, k ) < 0.0_wp ) THEN
               inform%negative_eigenvalues =                                 &
                 inform%negative_eigenvalues + 1
             ELSE IF ( data%matrix_dense( k, k ) == 0.0_wp ) THEN
               inform%rank = inform%rank - 1
             END IF
             D( 1, k ) = data%matrix_dense( k, k )
             D( 2, k ) = 0.0_wp
             k = k + 1
           ELSE                               ! a 2 x 2 pivot
             D( 1, k ) = data%matrix_dense( k, k )
             D( 1, k + 1 ) = data%matrix_dense( k + 1, k + 1 )
             D( 2, k ) = data%matrix_dense( k + 1, k )
             D( 2, k + 1 ) = 0.0_wp
             k = k + 2
           END IF
         END DO
       END IF
       IF ( PRESENT( PIVOTS ) ) PIVOTS = data%PIVOTS( : data%n )
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = PBTR =

     CASE ( 'pbtr' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_wp ; D( 2, : ) = 0.0_wp
       END IF
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = anything else =

     CASE DEFAULT
       inform%status = GALAHAD_unavailable_option
     END SELECT

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of SLS_enquire

     END SUBROUTINE SLS_enquire

!-*-*-*-*-*-*-*-   S L S _ A L T E R _ D   S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SLS_alter_d( data, D, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Alter the diagonal blocks

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = wp ), INTENT( INOUT ) :: D( 2, data%n )
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: info, k
     REAL :: time_start, time_now
     REAL ( KIND = wp ) :: clock_start, clock_now

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  solver-dependent data alteration

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SILS_alter_d( data%sils_factors, D, info )
       inform%status = info

!  = MA57 =

     CASE ( 'ma57' )
       CALL MA57_alter_d( data%ma57_factors, D, info )
       inform%status = info

!  = MA77 =

     CASE ( 'ma77' )
       CALL MA77_alter( D, data%ma77_keep, data%ma77_control, data%ma77_info )
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       inform%status = GALAHAD_ok

!  = MA86 =

     CASE ( 'ma86' )
       inform%status = GALAHAD_error_alter_diagonal

!  = MA87 =

     CASE ( 'ma87' )
       inform%status = GALAHAD_error_alter_diagonal

!  = MA97 =

     CASE ( 'ma97' )
       IF ( data%must_be_definite ) THEN
         inform%status = GALAHAD_ok
       ELSE
         CALL MA97_alter( D, data%ma97_akeep, data%ma97_fkeep,                 &
                          data%ma97_control, data%ma97_info )
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
       END IF

!  = SSIDS =

     CASE ( 'ssids' )
       IF ( data%must_be_definite ) THEN
         inform%status = GALAHAD_ok
       ELSE
         CALL SSIDS_alter( D, data%ssids_akeep, data%ssids_fkeep,              &
                           data%ssids_options, data%ssids_inform )
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
       END IF

!  = POTR =

     CASE ( 'potr' )
       inform%status = GALAHAD_error_alter_diagonal

!  = SYTR =

     CASE ( 'sytr' )
       k = 1
       DO                                   ! run through the pivots
         IF ( k > data%n ) EXIT
         IF ( data%PIVOTS( k ) > 0 ) THEN   ! a 1 x 1 pivot
           IF ( data%matrix_dense( k, k ) < 0.0_wp ) THEN
             inform%negative_eigenvalues =                                     &
               inform%negative_eigenvalues + 1
           ELSE IF ( data%matrix_dense( k, k ) == 0.0_wp ) THEN
             inform%rank = inform%rank - 1
           END IF
           data%matrix_dense( k, k ) = D( 1, k )
           k = k + 1
         ELSE                               ! a 2 x 2 pivot
           data%matrix_dense( k, k ) = D( 1, k )
           data%matrix_dense( k + 1, k + 1 ) = D( 1, k + 1 )
           data%matrix_dense( k + 1, k ) = D( 2, k )
           k = k + 2
         END IF
       END DO
       inform%status = GALAHAD_ok

!  = PBTR =

     CASE ( 'pbtr' )
       inform%status = GALAHAD_error_alter_diagonal

!  = anything else =

     CASE DEFAULT
       inform%status = GALAHAD_error_alter_diagonal
     END SELECT

!  record total time

     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of SLS_alter_d

     END SUBROUTINE SLS_alter_d


!-*-*-*-*-*-*-   S L S _ P A R T _ S O L V E   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE SLS_part_solve( part, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve a system involving individual factors

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER, INTENT( IN ) :: part
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, info
     REAL :: time, time_start, time_now
     REAL ( KIND = wp ) :: clock, clock_start, clock_now

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

     IF ( data%explicit_scaling ) THEN
       IF ( part == 'L' .OR. part == 'S' ) THEN
         X( : data%n ) = X( : data%n ) / data%SCALE( : data%n )
       END IF
     END IF

!  solver-dependent partial solution

!write(6,*) ' part solve using ', data%solver( 1 : data%len_solver )
     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SLS_copy_control_to_sils( control, data%sils_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'S' ) THEN
         CALL SILS_part_solve( data%sils_factors, data%sils_control, 'L',      &
                               X, info )
         inform%status = info
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL SILS_part_solve( data%sils_factors, data%sils_control, 'D',      &
                               data%WORK( : data%n ), info )
         inform%status = info
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
           IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.       &
                ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.         &
                ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_wp ) THEN
             X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
           ELSE
             X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
           END IF
         END DO
       ELSE
         CALL SILS_part_solve( data%sils_factors, data%sils_control, part,     &
                               X, info )
       END IF
       inform%status = info

!  = MA57 =

     CASE ( 'ma57' )
       CALL SLS_copy_control_to_ma57( control, data%ma57_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'S' ) THEN
         CALL MA57_part_solve( data%ma57_factors, data%ma57_control, 'L',      &
                               X, info )
         inform%status = info
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL MA57_part_solve( data%ma57_factors, data%ma57_control, 'D',      &
                               X, info )
         inform%status = info
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
           IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.       &
                ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.         &
                ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_wp ) THEN
             X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
           ELSE
             X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
           END IF
         END DO
       ELSE
         CALL MA57_part_solve( data%ma57_factors, data%ma57_control, part,     &
                               X, info )
       END IF
       inform%status = info

!  = MA77 =

     CASE ( 'ma77' )
       IF ( part == 'D' .AND. data%must_be_definite ) THEN
         inform%status = 0
         GO TO 900
       END IF
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         IF ( part == 'L' .OR.                                                 &
              ( part == 'S' .AND. data%must_be_definite ) ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 1, scale = data%SCALE )
         ELSE IF ( part == 'D' ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 2, scale = data%SCALE )
         ELSE  IF ( part == 'U' ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3, scale = data%SCALE )
         ELSE
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) == 0.0_wp ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) /= 0.0_wp ) .OR.    &
                  ( data%X2( i, 1 ) /= 0.0_wp .AND. X( i ) == 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_wp .AND. X( i ) < 0.0_wp ) .OR.      &
                  ( data%X2( i, 1 ) < 0.0_wp .AND. X( i ) > 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_wp ) THEN
               data%X2( i, 1 ) = SQRT( data%X2( i, 1 ) ) * SQRT( X( i ) )
             ELSE
               data%X2( i, 1 ) = - SQRT( - data%X2( i, 1 ) ) * SQRT( - X( i ) )
             END IF
           END DO
         END IF
       ELSE
         IF ( part == 'L' .OR.                                                 &
              ( part == 'S' .AND. data%must_be_definite ) ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 1 )
         ELSE IF ( part == 'D' ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 2 )
         ELSE  IF ( part == 'U' ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 3 )
         ELSE
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 1 )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 2 )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) == 0.0_wp ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) /= 0.0_wp ) .OR.    &
                  ( data%X2( i, 1 ) /= 0.0_wp .AND. X( i ) == 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_wp .AND. X( i ) < 0.0_wp ) .OR.      &
                  ( data%X2( i, 1 ) < 0.0_wp .AND. X( i ) > 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_wp ) THEN
               data%X2( i, 1 ) = SQRT( data%X2( i, 1 ) ) * SQRT( X( i ) )
             ELSE
               data%X2( i, 1 ) = - SQRT( - data%X2( i, 1 ) ) * SQRT( - X( i ) )
             END IF
           END DO
         END IF
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       IF ( inform%status == 0 ) X( : data%n ) = data%X2( : data%n, 1 )

!  = MA86 =

     CASE ( 'ma86' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'L' ) THEN
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 1 )
       ELSE IF ( part == 'D' ) THEN
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 2 )
       ELSE  IF ( part == 'U' ) THEN
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 3 )
       ELSE
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 1 )
         inform%ma86_info = data%ma86_info
         inform%status = data%ma86_info%flag
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL MA86_solve( data%WORK( : data%n ), data%ORDER, data%ma86_keep,   &
                          data%ma86_control, data%ma86_info, job = 2 )
         inform%ma86_info = data%ma86_info
         inform%status = data%ma86_info%flag
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
           IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.       &
                ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.         &
                ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_wp ) THEN
             X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
           ELSE
             X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
           END IF
         END DO
       END IF
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag

!  = MA87 =

     CASE ( 'ma87' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'D' ) THEN
         inform%status = 0
         GO TO 900
       END IF
       IF ( part == 'L' .OR. part == 'S') THEN
         CALL MA87_solve( X, data%ORDER, data%ma87_keep,                       &
                          data%ma87_control, data%ma87_info, job = 1 )
       ELSE  IF ( part == 'U' ) THEN
         CALL MA87_solve( X, data%ORDER, data%ma87_keep,                       &
                          data%ma87_control, data%ma87_info, job = 2 )
       END IF
       inform%ma87_info = data%ma87_info
       inform%status = data%ma87_info%flag

!  = MA97 =

     CASE ( 'ma97' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'L' .OR. ( part == 'S' .AND. data%must_be_definite ) ) THEN
         CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
                          data%ma97_control, data%ma97_info, job = 1 )
       ELSE IF ( part == 'D' ) THEN
         IF ( data%must_be_definite ) THEN
           inform%status = 0
           GO TO 900
         ELSE
           CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,   &
                            data%ma97_control, data%ma97_info, job = 2 )
         END IF
       ELSE  IF ( part == 'U' ) THEN
         CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
                          data%ma97_control, data%ma97_info, job = 3 )
       ELSE
         CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
                          data%ma97_control, data%ma97_info, job = 1 )
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL MA97_solve( data%WORK( : data%n ), data%ma97_akeep,              &
                          data%ma97_fkeep,                                     &
                          data%ma97_control, data%ma97_info, job = 2 )
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
           IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.       &
                ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.         &
                ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_wp ) THEN
             X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
           ELSE
             X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
           END IF
         END DO
       END IF
       CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )

!  = SSIDS =

     CASE ( 'ssids' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'L' .OR. ( part == 'S' .AND. data%must_be_definite ) ) THEN
         CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,  &
                           data%ssids_options, data%ssids_inform, job = 1 )
       ELSE IF ( part == 'D' ) THEN
         IF ( data%must_be_definite ) THEN
           inform%status = 0
           GO TO 900
         ELSE
           CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,&
                             data%ssids_options, data%ssids_inform, job = 2 )
         END IF
       ELSE  IF ( part == 'U' ) THEN
         CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,  &
                           data%ssids_options, data%ssids_inform, job = 3 )
       ELSE
         CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,  &
                           data%ssids_options, data%ssids_inform, job = 1 )
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL SSIDS_solve( data%WORK( : data%n ), data%ssids_akeep,            &
                           data%ssids_fkeep,                                   &
                           data%ssids_options, data%ssids_inform, job = 2 )
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
           IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.       &
                ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.         &
                ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_wp ) THEN
             X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
           ELSE
             X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
           END IF
         END DO
       END IF
       CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

     CASE ( 'pardiso' )
       IF ( part == 'D' ) THEN
         inform%status = 0
         GO TO 900
       END IF
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
       data%pardiso_iparm( 6 ) = 1
       IF ( part == 'L' .OR. part == 'S') THEN
         data%pardiso_iparm( 26 ) = 1
       ELSE
         data%pardiso_iparm( 26 ) = 2
       END IF
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%PARDISO_PT, 1, 1, data%pardiso_mtype, 33,            &
                     data%matrix%n, data%matrix%VAL( : data%ne ),              &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), 1,                         &
                     data%pardiso_iparm( : 64 ),                               &
                     control%print_level_solver,                               &
                     data%X2( : data%matrix%n, : 1 ),                          &
                     data%B2( : data%matrix%n, : 1 ), inform%pardiso_error,    &
                     data%pardiso_dparm( 1 : 64 ) )
       inform%pardiso_iparm = data%pardiso_iparm
       SELECT CASE( inform%pardiso_error )
       CASE ( - 1 )
         inform%status = GALAHAD_error_restrictions
       CASE ( GALAHAD_unavailable_option )
         inform%status = GALAHAD_unavailable_option
         CASE ( - 103 : GALAHAD_unavailable_option - 1,                        &
                GALAHAD_unavailable_option + 1 : - 2 )
         inform%status = GALAHAD_error_pardiso
       CASE DEFAULT
         inform%status = GALAHAD_ok
         inform%iterative_refinements =  data%pardiso_iparm( 7 )
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%X2( : data%n, 1 )

!  = WSMP =

     CASE ( 'wsmp' )
       CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,                &
                                      data%wsmp_dparm )
       IF ( data%wsmp_iparm( 31 ) == 0 .OR. data%wsmp_iparm( 31 ) == 5 ) THEN
         IF ( part == 'D' ) THEN
           inform%status = 0
           GO TO 900
         ELSE IF ( part == 'L' .OR. part == 'S') THEN
           data%wsmp_iparm( 30 ) = 1
         ELSE
           data%wsmp_iparm( 30 ) = 2
         END IF
       ELSE
         IF ( part == 'D' ) THEN
           data%wsmp_iparm( 30 ) = 3
         ELSE IF ( part == 'L' ) THEN
           data%wsmp_iparm( 30 ) = 1
         ELSE IF ( part == 'U' ) THEN
           data%wsmp_iparm( 30 ) = 2
         ELSE
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
       END IF
       data%wsmp_iparm( 2 ) = 4
       data%wsmp_iparm( 3 ) = 4

       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1,      &
                   data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),            &
                   data%wsmp_iparm, data%wsmp_dparm )

       inform%wsmp_iparm = data%wsmp_iparm
       inform%wsmp_dparm = data%wsmp_dparm
       inform%wsmp_error = data%wsmp_iparm( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_iparm( 6 )
       CASE ( - 102 )
         inform%status = GALAHAD_error_allocate
       CASE DEFAULT
         inform%status = GALAHAD_error_wsmp
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%B2( : data%n, 1 )

!  = POTR =

     CASE ( 'potr' )
       inform%status = 0
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, data%B, inform%status,               &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
!        data%B( : data%n ) = X( data%ORDER( : data%n ) )
         data%B( data%ORDER( : data%n ) ) = X( : data%n )
         IF ( part == 'L' .OR. part == 'S') THEN
           CALL TRSV ( 'L', 'N', 'N', data%n, data%matrix_dense, data%n,       &
                       data%B, 1 )
         ELSE IF ( part == 'U' ) THEN
           CALL TRSV ( 'L', 'T', 'N', data%n, data%matrix_dense, data%n,       &
                       data%B, 1 )
         ELSE ! if part = 'D'
           GO TO 900
         END IF
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
      ELSE
         IF ( part == 'L' .OR. part == 'S') THEN
           CALL TRSV ( 'L', 'N', 'N', data%n, data%matrix_dense, data%n, X, 1 )
         ELSE IF ( part == 'U' ) THEN
           CALL TRSV ( 'L', 'T', 'N', data%n, data%matrix_dense, data%n, X, 1 )
         ELSE ! if part = 'D'
           GO TO 900
         END IF
       END IF

!  = SYTR =

     CASE ( 'sytr' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, data%B, inform%status,               &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
!        data%B( : data%n ) = X( data%ORDER( : data%n ) )
         data%B( data%ORDER( : data%n ) ) = X( : data%n )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )

         CALL SLS_sytr_part_solve( part, data%n, data%n, data%matrix_dense,    &
                                   data%PIVOTS, data%B, inform%status )
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         CALL SLS_sytr_part_solve( part, data%n, data%n, data%matrix_dense,    &
                                   data%PIVOTS, X, inform%status )
       END IF

!  = PBTR =

     CASE ( 'pbtr' )
       inform%status = 0
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, data%B, inform%status,               &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
!        data%B( : data%n ) = X( data%ORDER( : data%n ) )
         data%B( data%ORDER( : data%n ) ) = X( : data%n )
         IF ( part == 'L' .OR. part == 'S') THEN
           CALL DTBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,           &
                       data%matrix_dense, inform%semi_bandwidth + 1, data%B, 1 )
         ELSE IF ( part == 'U' ) THEN
           CALL DTBSV( 'L', 'T', 'N', data%n, inform%semi_bandwidth,           &
                       data%matrix_dense, inform%semi_bandwidth + 1, data%B, 1 )
         ELSE ! if part = 'D'
           GO TO 900
         END IF
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         IF ( part == 'L' .OR. part == 'S') THEN
           CALL DTBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,           &
                       data%matrix_dense, inform%semi_bandwidth + 1, X, 1 )
         ELSE IF ( part == 'U' ) THEN
           CALL DTBSV( 'L', 'T', 'N', data%n, inform%semi_bandwidth,           &
                       data%matrix_dense, inform%semi_bandwidth + 1, X, 1 )
         ELSE ! if part = 'D'
           GO TO 900
         END IF
       END IF

!  = unavailable with other solvers =

     CASE DEFAULT
       inform%status = GALAHAD_unavailable_option
       GO TO 900
     END SELECT

     IF ( data%explicit_scaling ) THEN
       IF ( part == 'U' ) THEN
         X( : data%n ) = X( : data%n ) / data%SCALE( : data%n )
       END IF
     END IF

!  record external solve time

     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve_external = time_now - time
     inform%time%clock_solve_external = clock_now - clock

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve =inform%time%solve + time_now - time_start
     inform%time%clock_solve =                                                 &
       inform%time%clock_solve + clock_now - clock_start
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of SLS_part_solve

     END SUBROUTINE SLS_part_solve

!-*-  S L S _ S P A R S E _ F O R W A R D _ S O L V E   S U B R O U T I N E  -*-

     SUBROUTINE SLS_sparse_forward_solve( nnz_b, INDEX_b, B, nnz_x, INDEX_x,   &
                                          X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the system L x = b in which b is sparse and so is the output x

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER, INTENT( IN  ) :: nnz_b
     INTEGER, INTENT( OUT ) :: nnz_x
     INTEGER, INTENT( INOUT  ), DIMENSION( : ) :: INDEX_b
     INTEGER, INTENT( OUT ), DIMENSION( : ) :: INDEX_x
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( : ) :: B
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, info
     REAL :: time, time_start, time_now
     REAL ( KIND = wp ) :: clock, clock_start, clock_now

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

     IF ( data%explicit_scaling ) THEN
       X( : data%n ) = X( : data%n ) / data%SCALE( : data%n )
     END IF

!  for those solvers that don't offer a sparse-solution option, simply
!  use the standard "forawrd solve"

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  skip solvers that have sparse option

     CASE ( 'ma57', 'ma87', 'ma97', 'ssids' )

!  those that don't

     CASE DEFAULT
       X( : data%n ) = 0.0_wp
       DO i = 1, nnz_b
         X( INDEX_b( i ) ) = B( INDEX_b( i ) )
       END DO
     END SELECT

!  solver-dependent forward solution

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SLS_copy_control_to_sils( control, data%sils_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL SILS_part_solve( data%sils_factors, data%sils_control, 'L',        &
                             X, info )
       inform%status = info
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       CALL SPACE_resize_array( data%n, data%WORK,                             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%WORK( : data%n ) = X( : data%n )
       CALL SILS_part_solve( data%sils_factors, data%sils_control, 'D',        &
                             data%WORK( : data%n ), info )
       inform%status = info
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       DO i = 1, data%n
         IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
         IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.         &
              ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.           &
              ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( X( i ) > 0.0_wp ) THEN
           X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
         ELSE
           X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
         END IF
       END DO
       inform%status = info

!  = MA57 =

     CASE ( 'ma57' )
       CALL SLS_copy_control_to_ma57( control, data%ma57_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!      CALL MA57_part_solve( data%ma57_factors, data%ma57_control, 'L',        &
!                            X, info )
!      inform%status = info
!      IF ( inform%status /= GALAHAD_ok ) GO TO 900
!      CALL SPACE_resize_array( data%n, data%WORK,                             &
!                               inform%status, inform%alloc_status )
!      IF ( inform%status /= GALAHAD_ok ) GO TO 900
!      data%WORK( : data%n ) = X( : data%n )
!      CALL MA57_part_solve( data%ma57_factors, data%ma57_control, 'D',        &
!                            X, info )
!      inform%status = info
!      IF ( inform%status /= GALAHAD_ok ) GO TO 900
!      DO i = 1, data%n
!        IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
!        IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.         &
!             ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
!          inform%status = GALAHAD_error_inertia ; GO TO 900
!        END IF
!        IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.           &
!             ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
!          inform%status = GALAHAD_error_inertia ; GO TO 900
!        END IF
!        IF ( X( i ) > 0.0_wp ) THEN
!          X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
!        ELSE
!          X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
!        END IF
!      END DO

       DO i = 1, nnz_b
!        INDEX_x( i ) = INDEX_b( i )
         X( INDEX_b( i ) ) = B( INDEX_b( i ) )
!        B( INDEX_b( i ) ) = 0.0_wp
       END DO
       CALL ma57_sparse_lsolve( data%ma57_factors, data%ma57_control, nnz_b,   &
                                INDEX_b, nnz_x, INDEX_x, X,  data%ma57_sinfo )
       inform%ma57_sinfo = data%ma57_sinfo
       inform%status = data%ma57_sinfo%flag

!  = MA77 =

     CASE ( 'ma77' )
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         IF ( data%must_be_definite ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 1, scale = data%SCALE )
         ELSE
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) == 0.0_wp ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) /= 0.0_wp ) .OR.    &
                  ( data%X2( i, 1 ) /= 0.0_wp .AND. X( i ) == 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_wp .AND. X( i ) < 0.0_wp ) .OR.      &
                  ( data%X2( i, 1 ) < 0.0_wp .AND. X( i ) > 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_wp ) THEN
               data%X2( i, 1 ) = SQRT( data%X2( i, 1 ) ) * SQRT( X( i ) )
             ELSE
               data%X2( i, 1 ) = - SQRT( - data%X2( i, 1 ) ) * SQRT( - X( i ) )
             END IF
           END DO
         END IF
       ELSE
         IF ( data%must_be_definite ) THEN
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 1 )
         ELSE
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 1 )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1, data%n, data%X2, data%ma77_keep,                &
                            data%ma77_control, data%ma77_info, job = 2 )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) == 0.0_wp ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_wp .AND. X( i ) /= 0.0_wp ) .OR.    &
                  ( data%X2( i, 1 ) /= 0.0_wp .AND. X( i ) == 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_wp .AND. X( i ) < 0.0_wp ) .OR.      &
                  ( data%X2( i, 1 ) < 0.0_wp .AND. X( i ) > 0.0_wp ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_wp ) THEN
               data%X2( i, 1 ) = SQRT( data%X2( i, 1 ) ) * SQRT( X( i ) )
             ELSE
               data%X2( i, 1 ) = - SQRT( - data%X2( i, 1 ) ) * SQRT( - X( i ) )
             END IF
           END DO
         END IF
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       IF ( inform%status == 0 ) X( : data%n ) = data%X2( : data%n, 1 )

!  = MA86 =

     CASE ( 'ma86' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA86_solve( X, data%ORDER, data%ma86_keep,                         &
                        data%ma86_control, data%ma86_info, job = 1 )
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       CALL SPACE_resize_array( data%n, data%WORK,                             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%WORK( : data%n ) = X( : data%n )
       CALL MA86_solve( data%WORK( : data%n ), data%ORDER, data%ma86_keep,     &
                        data%ma86_control, data%ma86_info, job = 2 )
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       DO i = 1, data%n
         IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
         IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.         &
              ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.           &
              ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( X( i ) > 0.0_wp ) THEN
           X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
         ELSE
           X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
         END IF
       END DO
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag

!  = MA87 =

     CASE ( 'ma87' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA87_sparse_fwd_solve( nnz_b, INDEX_b, B, data%ORDER, data%INVP,   &
                                   nnz_x, INDEX_x, X, data%WORK,               &
                                   data%ma87_keep, data%ma87_control,          &
                                   data%ma87_info )
!      CALL MA87_sparse_fwd_solve( nnz_b, INDEX_b, B, data%ORDER,              &
!                                  nnz_x, INDEX_x, X,                          &
!                                  data%ma87_keep, data%ma87_control,          &
!                                  data%ma87_info )
       inform%ma87_info = data%ma87_info
       inform%status = data%ma87_info%flag

!  = MA97 =

     CASE ( 'ma97' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA97_sparse_fwd_solve( nnz_b, INDEX_b, B, data%ORDER, data%LFLAG,  &
                                   nnz_x, INDEX_x, X,                          &
                                   data%ma97_akeep, data%ma97_fkeep,           &
                                   data%ma97_control, data%ma97_info )
       data%LFLAG( INDEX_x( : nnz_x ) ) = .FALSE.
!      IF ( data%must_be_definite ) THEN
!        CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
!                         data%ma97_control, data%ma97_info, job = 1 )
!      ELSE
!        CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
!                         data%ma97_control, data%ma97_info, job = 1 )
!        CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
!        IF ( inform%status /= GALAHAD_ok ) GO TO 900
!        CALL SPACE_resize_array( data%n, data%WORK,                           &
!                                 inform%status, inform%alloc_status )
!        IF ( inform%status /= GALAHAD_ok ) GO TO 900
!        data%WORK( : data%n ) = X( : data%n )
!        CALL MA97_solve( data%WORK( : data%n ), data%ma97_akeep,              &
!                         data%ma97_fkeep,                                     &
!                         data%ma97_control, data%ma97_info, job = 2 )
!        CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
!        IF ( inform%status /= GALAHAD_ok ) GO TO 900
!        DO i = 1, data%n
!          IF ( X( i ) == 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) CYCLE
!          IF ( ( X( i ) == 0.0_wp .AND. data%WORK( i ) /= 0.0_wp ) .OR.       &
!               ( X( i ) /= 0.0_wp .AND. data%WORK( i ) == 0.0_wp ) ) THEN
!            inform%status = GALAHAD_error_inertia ; GO TO 900
!          END IF
!          IF ( ( X( i ) > 0.0_wp .AND. data%WORK( i ) < 0.0_wp ) .OR.         &
!               ( X( i ) < 0.0_wp .AND. data%WORK( i ) > 0.0_wp ) ) THEN
!            inform%status = GALAHAD_error_inertia ; GO TO 900
!          END IF
!          IF ( X( i ) > 0.0_wp ) THEN
!            X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
!          ELSE
!            X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
!          END IF
!        END DO
!      END IF
       CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )

!  = SSIDS =

     CASE ( 'ssids' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900
!      CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!      CALL SSIDS_sparse_fwd_solve( nnz_b, INDEX_b, B, data%ORDER, data%LFLAG, &
!                                   nnz_x, INDEX_x, X,                         &
!                                   data%ssids_akeep, data%ssids_fkeep,        &
!                                   data%ssids_options, data%ssids_inform )
!      data%LFLAG( INDEX_x( : nnz_x ) ) = .FALSE.
!      CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

     CASE ( 'pardiso' )
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
       data%pardiso_iparm( 6 ) = 1
       data%pardiso_iparm( 26 ) = 1
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%PARDISO_PT, 1, 1, data%pardiso_mtype, 33,            &
                     data%matrix%n, data%matrix%VAL( : data%ne ),              &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), 1,                         &
                     data%pardiso_iparm( : 64 ),                               &
                     control%print_level_solver,                               &
                     data%X2( : data%matrix%n, : 1 ),                          &
                     data%B2( : data%matrix%n, : 1 ), inform%pardiso_error,    &
                     data%pardiso_dparm( 1 : 64 ) )
       inform%pardiso_iparm = data%pardiso_iparm
       SELECT CASE( inform%pardiso_error )
       CASE ( - 1 )
         inform%status = GALAHAD_error_restrictions
       CASE ( GALAHAD_unavailable_option )
         inform%status = GALAHAD_unavailable_option
         CASE ( - 103 : GALAHAD_unavailable_option - 1,                        &
                GALAHAD_unavailable_option + 1 : - 2 )
         inform%status = GALAHAD_error_pardiso
       CASE DEFAULT
         inform%status = GALAHAD_ok
         inform%iterative_refinements =  data%pardiso_iparm( 7 )
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%X2( : data%n, 1 )

!  = WSMP =

     CASE ( 'wsmp' )
       CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,                &
                                      data%wsmp_dparm )
       IF ( data%wsmp_iparm( 31 ) == 0 .OR. data%wsmp_iparm( 31 ) == 5 ) THEN
         data%wsmp_iparm( 30 ) = 1
       ELSE
         inform%status = GALAHAD_error_inertia ; GO TO 900
       END IF
       data%wsmp_iparm( 2 ) = 4
       data%wsmp_iparm( 3 ) = 4

       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL wssmp( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1,      &
                   data%wsmp_aux, 0, data%MRP( 1 : data%matrix%n ),            &
                   data%wsmp_iparm, data%wsmp_dparm )

       inform%wsmp_iparm = data%wsmp_iparm
       inform%wsmp_dparm = data%wsmp_dparm
       inform%wsmp_error = data%wsmp_iparm( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_iparm( 6 )
       CASE ( - 102 )
         inform%status = GALAHAD_error_allocate
       CASE DEFAULT
         inform%status = GALAHAD_error_wsmp
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%B2( : data%n, 1 )

!  = POTR =

     CASE ( 'potr' )
       inform%status = 0
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, data%B, inform%status,               &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
!        data%B( : data%n ) = X( data%ORDER( : data%n ) )
         data%B( data%ORDER( : data%n ) ) = X( : data%n )
         CALL TRSV ( 'L', 'N', 'N', data%n, data%matrix_dense, data%n,         &
                     data%B, 1 )
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         CALL TRSV ( 'L', 'N', 'N', data%n, data%matrix_dense, data%n, X, 1 )
       END IF

!  = SYTR =

     CASE ( 'sytr' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, data%B, inform%status,               &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
!        data%B( : data%n ) = X( data%ORDER( : data%n ) )
         data%B( data%ORDER( : data%n ) ) = X( : data%n )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )

         CALL SLS_sytr_part_solve( 'S', data%n, data%n, data%matrix_dense,     &
                                   data%PIVOTS, data%B, inform%status )
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         CALL SLS_sytr_part_solve( 'S', data%n, data%n, data%matrix_dense,     &
                                   data%PIVOTS, X, inform%status )
       END IF

!  = PBTR =

     CASE ( 'pbtr' )
       inform%status = 0
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       IF ( data%reordered ) THEN
         CALL SPACE_resize_array( data%n, data%B, inform%status,               &
                                  inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B' ; GO TO 900 ; END IF
!        data%B( : data%n ) = X( data%ORDER( : data%n ) )
         data%B( data%ORDER( : data%n ) ) = X( : data%n )
         CALL DTBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,             &
                     data%matrix_dense, inform%semi_bandwidth + 1, data%B, 1 )
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         CALL DTBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,             &
                     data%matrix_dense, inform%semi_bandwidth + 1, X, 1 )
       END IF

!  = unavailable with other solvers =

     CASE DEFAULT
       inform%status = GALAHAD_unavailable_option
       GO TO 900
     END SELECT

!  record external solve time

     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve_external = time_now - time
     inform%time%clock_solve_external = clock_now - clock

!  for those solvers that don't offer a sparse-solution option,
!  record the nonzeros in the solution

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  skip solvers that have sparse option

     CASE ( 'ma57', 'ma87', 'ma97', 'ssids' )

!  those that don't

     CASE DEFAULT
       nnz_x = 0
       DO i = 1, data%n
         IF ( X( i ) == 0.0_wp ) CYCLE
         nnz_x = nnz_x + 1
         INDEX_x( nnz_x ) = i
       END DO
     END SELECT

!  record total time

 900 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve =inform%time%solve + time_now - time_start
     inform%time%clock_solve =                                                 &
       inform%time%clock_solve + clock_now - clock_start
     inform%time%total = inform%time%total + time_now - time_start
     inform%time%clock_total =                                                 &
       inform%time%clock_total + clock_now - clock_start
     RETURN

!  End of SLS_sparse_forward_solve

     END SUBROUTINE SLS_sparse_forward_solve

!-*-  S L S _ F R E D H O L M _ A L T E R N A T I V E   S U B R O U T I N E  -*-

     SUBROUTINE SLS_fredholm_alternative( matrix, X, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Find the Fredholm Alternative x, i.e. either A x = b or A^T x = 0 & b^T x > 0

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( INOUT ) :: X( : )
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     LOGICAL, DIMENSION( 1 ) :: flag_out
     REAL :: time, time_now
     REAL ( KIND = wp ) :: clock, clock_now

!  solver-dependent solution
! write(6,*) data%solver( 1 : data%len_solver )
     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL SLS_copy_control_to_sils( control, data%sils_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = MA57 =

     CASE ( 'ma57' )
       CALL SPACE_resize_array( data%n, data%WORK, inform%status,              &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%WORK' ; GO TO 900 ; END IF
       CALL SLS_copy_control_to_ma57( control, data%ma57_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         CALL MA57_fredholm_alternative( data%ma57_factors,                    &
                                         data%ma57_control,                    &
                                         X, data%WORK, data%ma57_sinfo )
       CASE DEFAULT
         CALL MA57_fredholm_alternative( data%ma57_factors,                    &
                                         data%ma57_control,                    &
                                         X, data%WORK, data%ma57_sinfo )
       END SELECT
       inform%alternative = data%ma57_sinfo%flag == 1

       SELECT CASE( data%ma57_sinfo%flag )
       CASE ( 0, 1 )
         inform%status = GALAHAD_ok
         IF ( inform%alternative ) THEN
           X( : data%n ) = data%WORK( : data%n )
         END IF
       CASE DEFAULT
         inform%status =  GALAHAD_error_allocate
         inform%alloc_status = data%ma57_sinfo%flag
       END SELECT

!  = MA77 =

     CASE ( 'ma77' )
       CALL SPACE_resize_array( data%n, 2, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_ma77( control, data%ma77_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         CALL MA77_solve_fredholm( 1, flag_out, data%n, data%X2,               &
                                   data%ma77_keep, data%ma77_control,          &
                                   data%ma77_info, scale = data%SCALE )
       ELSE
         CALL MA77_solve_fredholm( 1, flag_out, data%n, data%X2,               &
                                   data%ma77_keep, data%ma77_control,          &
                                   data%ma77_info )
       END IF
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
       inform%alternative = .NOT. flag_out( 1 )
       IF ( inform%status /= GALAHAD_ok ) GO TO 800
       IF ( inform%alternative ) THEN
         X( : data%n ) = data%X2( : data%n, 2 )
       ELSE
         X( : data%n ) = data%X2( : data%n, 1 )
       END IF

!  = MA86 =

     CASE ( 'ma86' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = MA87 =

     CASE ( 'ma87' )
       CALL SLS_copy_control_to_ma87( control, data%ma87_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = MA97 =

     CASE ( 'ma97' )
       CALL SPACE_resize_array( data%n, 2, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_ma97( control, data%ma97_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA97_solve_fredholm( 1, flag_out, data%X2, data%n,                 &
                                 data%ma97_akeep, data%ma97_fkeep,             &
                                 data%ma97_control, data%ma97_info )
       inform%alternative = .NOT. flag_out( 1 )
       CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
       IF ( inform%status /= GALAHAD_ok ) GO TO 800
       IF ( inform%alternative ) THEN
         X( : data%n ) = data%X2( : data%n, 2 )
       ELSE
         X( : data%n ) = data%X2( : data%n, 1 )
       END IF

!  = SSIDS =

     CASE ( 'ssids' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900
!      CALL SPACE_resize_array( data%n, 2, data%X2, inform%status,             &
!              inform%alloc_status )
!      IF ( inform%status /= GALAHAD_ok ) THEN
!        inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
!      data%X2( : data%n, 1 ) = X( : data%n )
!      CALL SLS_copy_control_to_ssids( control, data%ssids_options )
!      CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!      CALL SSIDS_solve_fredholm( 1, flag_out, data%X2, data%n,                &
!                                 data%ssids_akeep, data%ssids_fkeep,          &
!                                 data%ssids_options, data%ssids_inform )
!      inform%alternative = .NOT. flag_out( 1 )
!      CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
!      IF ( inform%status /= GALAHAD_ok ) GO TO 800
!      IF ( inform%alternative ) THEN
!        X( : data%n ) = data%X2( : data%n, 2 )
!      ELSE
!        X( : data%n ) = data%X2( : data%n, 1 )
!      END IF

!  = PARDISO =

     CASE ( 'pardiso' )
       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )

       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = WSMP =

     CASE ( 'wsmp' )

       CALL SPACE_resize_array( data%n, 1, data%B2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_wsmp( control, data%wsmp_iparm,                &
                                      data%wsmp_dparm )
       data%wsmp_iparm( 2 ) = 4
       data%wsmp_iparm( 3 ) = 5
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = LAPACK solvers POTR, SYTR or PBTR =

     CASE ( 'potr', 'sytr', 'pbtr' )

       CALL SPACE_resize_array( data%n, 1, data%X2, inform%status,             &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( data%ORDER( : data%n ), 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = POTR =

       CASE ( 'potr' )
         inform%status = GALAHAD_unavailable_option
         GO TO 900

!  = SYTR =

       CASE ( 'sytr' )
         inform%status = GALAHAD_unavailable_option
         GO TO 900

!  = PBTR =

       CASE ( 'pbtr' )
         inform%status = GALAHAD_unavailable_option
         GO TO 900
       END SELECT

       SELECT CASE( inform%lapack_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
       CASE DEFAULT
         inform%status = GALAHAD_error_lapack
       END SELECT
       IF ( inform%status == GALAHAD_ok )                                      &
         X( : data%n )= data%X2( data%ORDER( : data%n ), 1 )

     END SELECT

!  record external solve time

 800 CONTINUE
     CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
     inform%time%solve_external = time_now - time
     inform%time%clock_solve_external = clock_now - clock

 900 CONTINUE
     RETURN

!  End of SLS_fredholm_alternative

     END SUBROUTINE SLS_fredholm_alternative

!-*-*-*-   S L S _ S Y T R _ P A R T _ S O L V E   S U B R O U T I N E   -*-*-*-

     SUBROUTINE SLS_sytr_part_solve( part, n, lda, A, PIVOTS, X, status )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the systems Lx = b, D x = b or L^T x = b, where the factorization
!  A = L D L^T is found by the LAPACK routine SYTRF and b is input in x

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = 1 ), INTENT( IN ) :: part
     INTEGER, INTENT( IN ) :: n, lda
     INTEGER, INTENT( INOUT ) :: status
     INTEGER, INTENT( IN ), DIMENSION( n ) :: PIVOTS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lda, n ) :: A
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: X
!  local variables

     INTEGER :: k, km1, kp, kp1
     REAL ( KIND = wp ) :: akm1k, akm1, ak, denom, bkm1, bk, val

!  Replace x by L^-1 x

     status = 0
     IF ( part == 'L' ) THEN
       k = 1
       DO                                   ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN        ! a 1 x 1 pivot
           kp = PIVOTS( K )
           IF ( kp /= k ) THEN              ! interchange X k and PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF

!  multiply by L(k)^-1, where L(k) is the transformation stored in column k of A

           IF ( k < n ) X( k + 1 : n ) = X( k + 1 : n )                        &
               - A( k + 1 : n, k ) * X( k )
           k = k + 1
         ELSE                               ! a 2 x 2 pivot
           kp1 = k + 1
           kp = - PIVOTS( k )
           IF ( kp /= kp1 ) THEN            ! interchange X k+1 and -PIVOTS(k)
             val =  X( kp1 ) ; X( kp1 ) = X( kp ) ; X( kp ) = val
           END IF

!  multiply by L(k)^-1 where L(k) is stored in columns k and k+1 of A

           IF ( k < n - 1 ) X( k + 2 : n ) = X( k + 2 : n )                    &
                - A( k + 2 : n, k ) * X( k )                                   &
                - A( k + 2 : n, kp1 ) * X( kp1 )
           k = k + 2
         END IF
       END DO
     ELSE IF ( part == 'U' ) THEN
       k = n
       DO                                   ! run backwards through the pivots
         IF ( k < 1 ) EXIT
         IF ( pivots( k ) > 0 ) THEN        ! a 1 x 1 pivot

!  multiply by L(k)^-T, where L(k) is the transformation stored in column k of A

           IF ( k < n )                                                        &
             X( k ) = X( k ) - DOT_PRODUCT( X( k + 1 : n ), A( k + 1 : n , k ) )
           kp = PIVOTS( k )
           IF ( kp /= k ) THEN              ! interchange X k and PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF
           k = k - 1
         ELSE                               ! a 2 x 2 pivot

!  multiply by L(k)^-1 where L(k) is stored in columns k-1 and k of A

           IF ( k < n ) THEN
             km1 = k - 1
             kp1 = k + 1
             X( k ) = X( k ) - DOT_PRODUCT( X( kp1 : n ), A( kp1 : n, k ) )
             X( km1 ) = X( km1 ) - DOT_PRODUCT( X( kp1 : n ), A( kp1 : n, km1 ))
           END IF
           kp = - PIVOTS( k )
           IF ( kp /= k ) THEN              ! interchange X k and -PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF
           k = k - 2
         END IF
       END DO
     ELSE IF ( part == 'D' ) THEN
       k = 1
       DO                                   ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN   ! a 1 x 1 pivot
           X( k ) = X( k ) / A( k, k )
           k = k + 1
         ELSE                               ! a 2 x 2 pivot
           kp1 = k + 1
           akm1k = A( kp1, k )
           akm1 = A( k, k ) / akm1k
           ak = A( kp1, kp1 ) / akm1k
           denom = akm1 * ak - 1.0_wp
           bkm1 = X( k ) / akm1k
           bk = X( kp1 ) / akm1k
           X( k ) = ( ak * bkm1 - bk ) / denom
           X( kp1 ) = ( akm1 * bk - bkm1 ) / denom
           k = k + 2
         END IF
       END DO
     ELSE ! if part = 'S'
       k = 1
       DO                                   ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN        ! a 1 x 1 pivot
           kp = PIVOTS( K )
           IF ( kp /= k ) THEN              ! interchange X k and PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF

!  multiply by L(k)^-1, where L(k) is the transformation stored in column k of A

           IF ( k < n ) X( k + 1 : n ) = X( k + 1 : n )                        &
               - A( k + 1 : n, k ) * X( k )
           k = k + 1
         ELSE                               ! a 2 x 2 pivot
           kp1 = k + 1
           kp = - PIVOTS( k )
           IF ( kp /= kp1 ) THEN            ! interchange X k+1 and -PIVOTS(k)
             val =  X( kp1 ) ; X( kp1 ) = X( kp ) ; X( kp ) = val
           END IF

!  multiply by L(k)^-1 where L(k) is stored in columns k and k+1 of A

           IF ( k < n - 1 ) X( k + 2 : n ) = X( k + 2 : n )                    &
                - A( k + 2 : n, k ) * X( k )                                   &
                - A( k + 2 : n, kp1 ) * X( kp1 )
           k = k + 2
         END IF
       END DO

!  divide by the square roots of D(k) if possibe

       k = 1
       DO                                   ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN   ! a 1 x 1 pivot
           IF ( A( k, k ) > 0.0_wp ) THEN
             X( k ) = X( k ) / SQRT( A( k, k ) )
           ELSE
             status = GALAHAD_error_inertia ; EXIT
           END IF
           k = k + 1
         ELSE                               ! a 2 x 2 pivot
           status = GALAHAD_error_inertia ; EXIT
         END IF
       END DO
     END IF

     RETURN

!  End of SLS_sytr_part_solve

     END SUBROUTINE SLS_sytr_part_solve

!-*-*-*-*-*-*-*-*-*-   S L S _ K E Y W O R D    F U N C T I O N  -*-*-*-*-*-*-*-

     FUNCTION SLS_keyword( array )
     LOGICAL :: SLS_keyword

!  Dummy arguments

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array

!  Check to see if the string is an appropriate keyword

     SELECT CASE( SMT_get( array ) )

!  Keyword known

     CASE( 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' )
       SLS_keyword = .TRUE.

!  Keyword unknown

     CASE DEFAULT
       SLS_keyword = .FALSE.
     END SELECT

     RETURN

!  End of SLS_keyword

     END FUNCTION SLS_keyword

!-*-  S L S _ C O O R D _ T O _ E X T E N D E D _ C S R  S U B R O U T I N E -*-

     SUBROUTINE SLS_coord_to_extended_csr( n, ne, row, col, map, ptr, dup,     &
                                           oor, upper, missing_diagonals )

!  Compute a mapping from the co-ordinate scheme to the extended row storage
!  scheme used by MA77. The mapping records out-of-range components and
!  flags duplicates for summation.
!
!  Entry l is mapped to positions MAP( l, j ) for j = 1 and 2, l = 1, ne.
!  If MAP( l, 2 ) = 0, the entry is on the diagonal, and thus only mapped
!  to the single location MAP( l, 1 ). If MAP( l, 1 ) = 0, the entry is out of
!  range. If MAP( l, j ) < 0, the entry should be added to that in - MAP( l, j )
!
!  dup gives the number of duplicates, oor is the number of out-of-rangers and
!  missing_diagonals records the number of rows without a diagonal entry

! dummy arguments

     INTEGER, INTENT( IN ) :: n, ne
     INTEGER, INTENT( IN ), DIMENSION( ne ) :: ROW, COL
     INTEGER, INTENT( OUT ) :: dup, oor, upper, missing_diagonals
     INTEGER, INTENT( OUT ), DIMENSION( ne, 2 ) :: MAP
     INTEGER, INTENT( OUT ), DIMENSION( n + 1 ) :: PTR

!  local variables

     INTEGER :: i, j, k, l, ll
     INTEGER, DIMENSION( n + 1 ) :: IW

!  record the numbers of nonzeros in each row of the lower triangular
!  part of the matrix in IW

     upper = 0
     IW( 2 : n + 1 ) = 0
     DO l = 1, ne
       i = ROW( l ) ; j = COL( l )
       IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
       IF ( i >= j ) THEN
         IW( i + 1 ) = IW( i + 1 ) + 1
       ELSE
         IW( j + 1 ) = IW( j + 1 ) + 1
         upper = upper + 1
       END IF
     END DO

!  record starting addresses for each row

     IW( 1 ) = 1
     DO i = 1, n
       IW( i + 1 ) = IW( i + 1 ) + IW( i )
     END DO

!  map the lower triangular part into IW(:,1) and
!  use IW(:,2) to point back to the original storage.

     DO l = 1, ne
       i = ROW( l ) ; j = COL( l )
       IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
       IF ( i >= j ) THEN
         MAP( l, 1 ) = IW( i )
         MAP( IW( i ), 2 ) = l
         IW( i ) = IW( i ) + 1
       ELSE
         MAP( l, 1 ) = IW( j )
         MAP( IW( j ), 2 ) = l
         IW( j ) = IW( j ) + 1
       END IF
     END DO

!   restore the starting addresses for the rows

     DO i = n, 1, - 1
       IW( i + 1 ) = IW( i )
     END DO
     IW( 1 ) = 1

!  check for duplicate entries and zero diagonals

     dup = 0 ; missing_diagonals = 0
     PTR( 1 : n ) = 0
     DO i = 1, n
       k = IW( i )
       DO ll = IW( i ), IW( i + 1 ) - 1
         l = MAP( ll, 2 )
         j = MIN( ROW( l ), COL( l ) )

!  new entry

         IF ( PTR( j ) < IW( i ) ) THEN
           PTR( j ) = k
           k = k + 1

!  duplicate. Point at the original

         ELSE
           MAP( l, 1 ) = - MAP( PTR( j ), 2 )
           dup = dup + 1
         END IF
       END DO
       IF ( PTR( i ) < IW( i ) ) missing_diagonals = missing_diagonals + 1
     END DO

!  now find the number of nonzeros in each row of the expanded matrix

     PTR( 2 : n + 1 ) = 0
     DO l = 1, ne
       i = ROW( l ) ; j = COL( l )
       IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) THEN

!  new entries

       ELSE
         ll = MAP( l, 1 )
         IF ( ll > 0 ) THEN
           IF ( i /= j ) THEN
             PTR( i + 1 ) = PTR( i + 1 ) + 1
             PTR( j + 1 ) = PTR( j + 1 ) + 1
           ELSE
             PTR( i + 1 ) = PTR( i + 1 ) + 1
           END IF
         ELSE
         END IF
       END IF
     END DO

!  record starting addresses for each row of the expanded matrix

     PTR( 1 ) = 1
     DO i = 1, n
       PTR( i + 1 ) = PTR( i + 1 ) + PTR( i )
     END DO

!  compute the map into the extended structure

     oor = 0
     DO l = 1, ne
       i = ROW( l ) ; j = COL( l )

!  count and flag out-of-range indices

       IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) THEN
         oor = oor + 1
         MAP( l, 1 ) = 0
         MAP( l, 2 ) = 0

!  new entries

       ELSE
         ll = MAP( l, 1 )
         IF ( ll > 0 ) THEN
           IF ( i /= j ) THEN
             MAP( l, 1 ) = PTR( i )
             PTR( i ) = PTR( i ) + 1
             MAP( l, 2 ) = PTR( j )
             PTR( j ) = PTR( j ) + 1
           ELSE
             MAP( l, 1 ) = PTR( i )
             PTR( i ) = PTR( i ) + 1
             MAP( l, 2 ) = 0
           END IF

!  duplicates

         ELSE
           IF ( i /= j ) THEN
             MAP( l, 1 ) = - MAP( - ll, 1 )
             MAP( l, 2 ) = - MAP( - ll, 2 )
           ELSE
             MAP( l, 1 ) = - MAP( - ll, 1 )
             MAP( l, 2 ) = 0
           END IF
         END IF
       END IF
     END DO

!  restore the starting addresses for the rows in the expanded matrix

     DO i = n, 1, - 1
       PTR( i + 1 ) = PTR( i )
     END DO
     PTR( 1 ) = 1

     RETURN

!  End of SUBROUTINE SLS_coord_to_extended_csr

     END SUBROUTINE SLS_coord_to_extended_csr

!-*-   S L S _ C O O R D _ T O _ S O R T E D _ C S R  S U B R O U T I N E  -*-*-

     SUBROUTINE SLS_coord_to_sorted_csr( n, ne, row, col, map, ptr, dup, oor,  &
                                         upper, missing_diagonals, status,     &
                                         alloc_status )

!  Compute a mapping from the co-ordinate scheme to the row storage scheme
!  used by MA86, MA87, MA97, SSIDS, MC61, MC68, PARDISO and WSMP. The mapping
!  records out-of-range components and flags duplicates for summation.
!
!  Entry l is mapped to positions MAP( l ) for j = 1, l = 1, ne.
!  If MAP( l ) = 0, the entry is out of range.
!  If MAP( l ) < 0, the entry should be added to that in - MAP( l )
!
!  dup gives the number of duplicates, oor is the number of out-of-rangers and
!  missing_diagonals records the number of rows without a diagonal entry

! dummy arguments

     INTEGER, INTENT( IN ) :: n, ne
     INTEGER, INTENT( IN ), DIMENSION( ne ) :: ROW, COL
     INTEGER, INTENT( OUT ) :: dup, oor, upper, missing_diagonals
     INTEGER, INTENT( OUT ) :: status, alloc_status
     INTEGER, INTENT( OUT ), DIMENSION( ne ) :: MAP
     INTEGER, INTENT( OUT ), DIMENSION( n + 1 ) :: PTR

!  local variables

     INTEGER :: i, j, jj, j_old, k, l, ll, err, pt, size
     INTEGER, DIMENSION( n + 1 ) :: IW
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: COLS, ENTS
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: MAP2

     status = 0

!  record the numbers of nonzeros in each row of the upper triangular
!  part of the matrix in IW. Zero diagonals will be included

     upper = 0
     IW( 2 : n + 1 ) = 1
     DO l = 1, ne
       i = ROW( l ) ; j = COL( l )
       IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) CYCLE
       IF ( i < j ) THEN
         IW( i + 1 ) = IW( i + 1 ) + 1
         upper = upper + 1
       ELSE IF ( i == j ) THEN
         IW( i + 1 ) = IW( i + 1 ) + 1
       ELSE
         IW( j + 1 ) = IW( j + 1 ) + 1
       END IF
     END DO

!  record starting addresses for each row

     IW( 1 ) = 1
     DO i = 1, n
       IW( i + 1 ) = IW( i + 1 ) + IW( i )
     END DO
     size = IW( n + 1 ) - 1

     ALLOCATE( MAP2( size ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF

     DO i = 1, n
       MAP2( IW( i ) ) = 0
       IW( i ) = IW( i ) + 1
     END DO

     oor = 0
     DO l = 1, ne
       i = ROW( l ) ; j = COL( l )

!  flag and count out-of-range indices

       IF ( i < 1 .OR. i > n .OR. j < 1 .OR. j > n ) THEN
         oor = oor + 1
         MAP( l ) = 0

!  map the upper triangular part into IW(:)

       ELSE IF ( i <= j ) THEN
         MAP( l ) = IW( i )
         MAP2( IW( i ) ) = l
         IW( i ) = IW( i ) + 1
       ELSE IF ( i > j ) THEN
         MAP( l ) = IW( j )
         MAP2( IW( j ) ) = l
         IW( j ) = IW( j ) + 1
       END IF
     END DO

!   restore the starting addresses for the rows

     DO i = n, 1, - 1
       IW( i + 1 ) = IW( i )
     END DO
     IW( 1 ) = 1

!  allocate workspace

     size = MAXVAL( IW( 2 : n + 1 ) - IW( 1 : n ) - 1 )
     ALLOCATE( ENTS( size ), COLS( size ), STAT = alloc_status )
     IF ( alloc_status /= 0 ) THEN
       status = GALAHAD_error_deallocate
       RETURN
     END IF

!  check for duplicate entries and zero diagonals

     dup = 0 ; missing_diagonals = 0
     PTR( 1 ) = 1
     DO i = 1, n
       size = IW( i + 1 ) - IW( i ) - 1
       pt = PTR( i )

!  record the column indices in the i-th row

       IF ( size > 0 ) THEN
         k = 0
         DO ll = IW( i ) +  1, IW( i + 1 ) - 1
           k = k + 1
           l = MAP2( ll )
           ENTS( k ) = l
           COLS( k ) = MAX( ROW( l ), COL( l ) )
         END DO

!  sort the columns within the i-th row

         CALL SORT_quicksort( size, COLS, err, ivector = ENTS )

!  remap the pointers to the columns to take account of the sort

         j_old = i - 1
         IF ( COLS( 1 ) > i ) THEN
           pt = pt + 1
           missing_diagonals = missing_diagonals + 1
         END IF
         DO k = 1, size
           l = ENTS( k )
           jj = COLS( k )
           IF ( jj > j_old ) THEN
             j_old = jj
             MAP( l ) = pt
             ll = l
             pt = pt + 1
           ELSE
             dup = dup + 1
             IF ( l < ll ) THEN
               MAP( l ) = pt - 1
               MAP( ll ) = - pt + 1
               ll = l
             ELSE
               MAP( l ) = - pt + 1
             END IF
           END IF
         END DO
       ELSE
         pt = pt + 1
       END IF
       PTR( i + 1 ) = pt
     END DO

     DEALLOCATE( MAP2, ENTS, COLS, STAT = i )
     RETURN

!  End of SUBROUTINE SLS_coord_to_sorted_csr

     END SUBROUTINE SLS_coord_to_sorted_csr

!!-*-*-  S L S _ M A P _ T O _ E X T E N D E D _ C S R  S U B R O U T I N E -*-*-
!
!     SUBROUTINE SLS_map_to_extended_csr( matrix, map, ptr, dup, oor,           &
!                                         missing_diagonals )
!
!!  Compute a mapping from the co-ordinate scheme to the extended row storage
!!  scheme used by MA77. The mapping records out-of-range components
!!  and flags duplicates for summation.
!!
!!  Entry l is mapped to positions MAP( l, j ) for j = 1 and 2, l = 1, ne.
!!  If MAP( l, 2 ) = 0, the entry is on the diagonal, and thus only mapped
!!  to the single location MAP( l, 1 ). If MAP( l, 1 ) = 0, the entry is out of
!!  range. If MAP( l, j ) < 0, the entry should be added to that in - MAP( l, j )
!!
!!  dup gives the number of duplicates, oor is the number of out-of-rangers and
!!  missing_diagonals records the number of rows without a diagonal entry
!
!! dummy arguments
!
!     TYPE ( SMT_type ), INTENT( IN ) :: matrix
!     INTEGER, INTENT( out ) :: dup, oor, missing_diagonals
!     INTEGER, INTENT( out ), DIMENSION( matrix%ne, 2 ) :: MAP
!     INTEGER, INTENT( out ), DIMENSION( matrix%n + 1 ) :: PTR
!
!!  local variables
!
!     INTEGER :: i, j, k, l, ll
!     INTEGER, DIMENSION( matrix%n + 1 ) :: IW
!
!!  record the numbers of nonzeros in each row of the lower triangular
!!  part of the matrix in IW
!
!     IW( 2 : matrix%n + 1 ) = 0
!     DO l = 1, matrix%ne
!       i = matrix%ROW( l ) ; j = matrix%COL( l )
!       IF ( i < 1 .OR. i > matrix%n .OR. j < 1 .OR. j > matrix%n ) CYCLE
!       IF ( i >= j ) THEN
!         IW( i + 1 ) = IW( i + 1 ) + 1
!       ELSE
!         IW( j + 1 ) = IW( j + 1 ) + 1
!       END IF
!     END DO
!
!!  record starting addresses for each row
!
!     IW( 1 ) = 1
!     DO i = 1, matrix%n
!       IW( i + 1 ) = IW( i + 1 ) + IW( i )
!     END DO
!
!!  map the lower triangular part into IW(:,1) and
!!  use IW(:,2) to point back to the original storage.
!
!     DO l = 1, matrix%ne
!       i = matrix%ROW( l ) ; j = matrix%COL( l )
!       IF ( i < 1 .OR. i > matrix%n .OR. j < 1 .OR. j > matrix%n ) CYCLE
!       IF ( i >= j ) THEN
!         MAP( l, 1 ) = IW( i )
!         MAP( IW( i ), 2 ) = l
!         IW( i ) = IW( i ) + 1
!       ELSE
!         MAP( l, 1 ) = IW( j )
!         MAP( IW( j ), 2 ) = l
!         IW( j ) = IW( j ) + 1
!       END IF
!     END DO
!
!!   restore the starting addresses for the rows
!
!     DO i = matrix%n, 1, - 1
!       IW( i + 1 ) = IW( i )
!     END DO
!     IW( 1 ) = 1
!
!!  check for duplicate entries and zero diagonals
!
!     dup = 0 ; missing_diagonals = 0
!     PTR( 1 : matrix%n ) = 0
!     DO i = 1, matrix%n
!       k = IW( i )
!       DO ll = IW( i ), IW( i + 1 ) - 1
!         l = MAP( ll, 2 )
!         j = MIN( matrix%ROW( l ), matrix%COL( l ) )
!
!!  new entry
!
!         IF ( PTR( j ) < IW( i ) ) THEN
!           PTR( j ) = k
!           k = k + 1
!
!!  duplicate. Point at the original
!
!         ELSE
!           MAP( l, 1 ) = - MAP( PTR( j ), 2 )
!           dup = dup + 1
!         END IF
!       END DO
!       IF ( PTR( i ) < IW( i ) ) missing_diagonals = missing_diagonals + 1
!     END DO
!
!!  now find the number of nonzeros in each row of the expanded matrix
!
!     PTR( 2 : matrix%n + 1 ) = 0
!     DO l = 1, matrix%ne
!       i = matrix%ROW( l ) ; j = matrix%COL( l )
!       IF ( i < 1 .OR. i > matrix%n .OR. j < 1 .OR. j > matrix%n ) THEN
!
!!  new entries
!
!       ELSE
!         ll = MAP( l, 1 )
!         IF ( ll > 0 ) THEN
!           IF ( i /= j ) THEN
!             PTR( i + 1 ) = PTR( i + 1 ) + 1
!             PTR( j + 1 ) = PTR( j + 1 ) + 1
!           ELSE
!             PTR( i + 1 ) = PTR( i + 1 ) + 1
!           END IF
!         ELSE
!         END IF
!       END IF
!     END DO
!
!!  record starting addresses for each row of the expanded matrix
!
!     PTR( 1 ) = 1
!     DO i = 1, matrix%n
!       PTR( i + 1 ) = PTR( i + 1 ) + PTR( i )
!     END DO
!
!!  compute the map into the extended structure
!
!     oor = 0
!     DO l = 1, matrix%ne
!       i = matrix%ROW( l ) ; j = matrix%COL( l )
!
!!  count and flag out-of-range indices
!
!       IF ( i < 1 .OR. i > matrix%n .OR. j < 1 .OR. j > matrix%n ) THEN
!         oor = oor + 1
!         MAP( l, 1 ) = 0
!         MAP( l, 2 ) = 0
!
!!  new entries
!
!       ELSE
!         ll = MAP( l, 1 )
!         IF ( ll > 0 ) THEN
!           IF ( i /= j ) THEN
!             MAP( l, 1 ) = PTR( i )
!             PTR( i ) = PTR( i ) + 1
!             MAP( l, 2 ) = PTR( j )
!             PTR( j ) = PTR( j ) + 1
!           ELSE
!             MAP( l, 1 ) = PTR( i )
!             PTR( i ) = PTR( i ) + 1
!             MAP( l, 2 ) = 0
!           END IF
!
!!  duplicates
!
!         ELSE
!           IF ( i /= j ) THEN
!             MAP( l, 1 ) = - MAP( - ll, 1 )
!             MAP( l, 2 ) = - MAP( - ll, 2 )
!           ELSE
!             MAP( l, 1 ) = - MAP( - ll, 1 )
!             MAP( l, 2 ) = 0
!           END IF
!         END IF
!       END IF
!     END DO
!
!!  restore the starting addresses for the rows in the expanded matrix
!
!     DO i = matrix%n, 1, - 1
!       PTR( i + 1 ) = PTR( i )
!     END DO
!     PTR( 1 ) = 1
!
!     RETURN
!
!!  End of SUBROUTINE SLS_map_to_extended_csr
!
!     END SUBROUTINE SLS_map_to_extended_csr
!
!!-*-*-   S L S _ M A P _ T O _ S O R T E D _ C S R  S U B R O U T I N E  -*-*-*-
!
!     SUBROUTINE SLS_map_to_sorted_csr( matrix, map, ptr, dup, oor,             &
!                                       missing_diagonals, status,              &
!                                       alloc_status )
!
!!  Compute a mapping from the co-ordinate scheme to the row storage scheme
!!  used by MA86, MA87, MA97, SSIDS, MC61, MC68, PARDISO and WSMP. The mapping
!!  records out-of-range components and flags duplicates for summation.
!!
!!  Entry l is mapped to positions MAP( l ) for j = 1, l = 1, ne.
!!  If MAP( l ) = 0, the entry is out of range.
!!  If MAP( l ) < 0, the entry should be added to that in - MAP( l )
!!
!!  dup gives the number of duplicates, oor is the number of out-of-rangers and
!!  missing_diagonals records the number of rows without a diagonal entry
!
!! dummy arguments
!
!     TYPE ( SMT_type ), INTENT( IN ) :: matrix
!     INTEGER, INTENT( out ) :: dup, oor, missing_diagonals, status, alloc_status
!     INTEGER, INTENT( out ), DIMENSION( matrix%ne ) :: MAP
!     INTEGER, INTENT( out ), DIMENSION( matrix%n + 1 ) :: PTR
!
!!  local variables
!
!     INTEGER :: i, j, jj, j_old, k, l, ll, err, pt, size
!     INTEGER, DIMENSION( matrix%n + 1 ) :: IW
!     INTEGER, ALLOCATABLE, DIMENSION( : ) :: COLS, ENTS
!     INTEGER, ALLOCATABLE, DIMENSION( : ) :: MAP2
!
!     status = 0
!
!!  record the numbers of nonzeros in each row of the upper triangular
!!  part of the matrix in IW. Zero diagonals will be included
!
!     IW( 2 : matrix%n + 1 ) = 1
!     DO l = 1, matrix%ne
!       i = matrix%ROW( l ) ; j = matrix%COL( l )
!       IF ( i < 1 .OR. i > matrix%n .OR. j < 1 .OR. j > matrix%n ) CYCLE
!       IF ( i <= j ) THEN
!         IW( i + 1 ) = IW( i + 1 ) + 1
!       ELSE IF ( i > j ) THEN
!         IW( j + 1 ) = IW( j + 1 ) + 1
!       END IF
!     END DO
!
!!  record starting addresses for each row
!
!     IW( 1 ) = 1
!     DO i = 1, matrix%n
!       IW( i + 1 ) = IW( i + 1 ) + IW( i )
!     END DO
!     size = IW( matrix%n + 1 ) - 1
!
!     ALLOCATE( MAP2( size ), STAT = alloc_status )
!     IF ( alloc_status /= 0 ) THEN
!       status = GALAHAD_error_deallocate
!       RETURN
!     END IF
!
!     DO i = 1, matrix%n
!       MAP2( IW( i ) ) = 0
!       IW( i ) = IW( i ) + 1
!     END DO
!
!     oor = 0
!     DO l = 1, matrix%ne
!       i = matrix%ROW( l ) ; j = matrix%COL( l )
!
!!  flag and count out-of-range indices
!
!       IF ( i < 1 .OR. i > matrix%n .OR. j < 1 .OR. j > matrix%n ) THEN
!         oor = oor + 1
!         MAP( l ) = 0
!
!!  map the upper triangular part into IW(:)
!
!       ELSE IF ( i <= j ) THEN
!         MAP( l ) = IW( i )
!         MAP2( IW( i ) ) = l
!         IW( i ) = IW( i ) + 1
!       ELSE IF ( i > j ) THEN
!         MAP( l ) = IW( j )
!         MAP2( IW( j ) ) = l
!         IW( j ) = IW( j ) + 1
!       END IF
!     END DO
!
!!   restore the starting addresses for the rows
!
!     DO i = matrix%n, 1, - 1
!       IW( i + 1 ) = IW( i )
!     END DO
!     IW( 1 ) = 1
!
!!  allocate workspace
!
!     size = MAXVAL( IW( 2 : matrix%n + 1 ) - IW( 1 : matrix%n ) - 1 )
!     ALLOCATE( ENTS( size ), COLS( size ), STAT = alloc_status )
!     IF ( alloc_status /= 0 ) THEN
!       status = GALAHAD_error_deallocate
!       RETURN
!     END IF
!
!!  check for duplicate entries and zero diagonals
!
!     dup = 0 ; missing_diagonals = 0
!     PTR( 1 ) = 1
!     DO i = 1, matrix%n
!       size = IW( i + 1 ) - IW( i ) - 1
!       pt = PTR( i )
!
!!  record the column indices in the i-th row
!
!       IF ( size > 0 ) THEN
!         k = 0
!         DO ll = IW( i ) +  1, IW( i + 1 ) - 1
!           k = k + 1
!           l = MAP2( ll )
!           ENTS( k ) = l
!           COLS( k ) = MAX( matrix%ROW( l ), matrix%COL( l ) )
!         END DO
!
!!  sort the columns within the i-th row
!
!         CALL SORT_quicksort( size, COLS, err, ivector = ENTS )
!
!!  remap the pointers to the columns to take account of the sort
!
!         j_old = i - 1
!         IF ( COLS( 1 ) > i ) THEN
!           pt = pt + 1
!           missing_diagonals = missing_diagonals + 1
!         END IF
!         DO k = 1, size
!           l = ENTS( k )
!           jj = COLS( k )
!           IF ( jj > j_old ) THEN
!             j_old = jj
!             MAP( l ) = pt
!             ll = l
!             pt = pt + 1
!           ELSE
!             dup = dup + 1
!             IF ( l < ll ) THEN
!               MAP( l ) = pt - 1
!               MAP( ll ) = - pt + 1
!               ll = l
!             ELSE
!               MAP( l ) = - pt + 1
!             END IF
!           END IF
!         END DO
!       ELSE
!         pt = pt + 1
!       END IF
!       PTR( i + 1 ) = pt
!     END DO
!
!     DEALLOCATE( MAP2, ENTS, COLS, STAT = i )
!     RETURN
!
!!  End of SUBROUTINE SLS_map_to_sorted_csr
!
!     END SUBROUTINE SLS_map_to_sorted_csr

!  End of module GALAHAD_SLS_double

   END MODULE GALAHAD_SLS_double

