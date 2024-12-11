! THIS VERSION: GALAHAD 5.1 - 2024-11-18 AT 15:00 GMT

#include "galahad_modules.h"
#undef METIS_DBG_INFO

!-*-*-*-*-*-*-*-*- G A L A H A D _ S L S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. June 15th 2009
!   LAPACK solvers and band ordering added Version 2.5, November 21st 2011

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_SLS_precision

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
!     |               MUMPS                       |
!     |               PARDISO                     |
!     |               MKL PARDISO                 |
!     |               PASTIX                      |
!     |               WSMP                        |
!     |               POTR from LAPACK            |
!     |               SYTR from LAPACK            |
!     |               PBTR from LAPACK            |
!     |                                           |
!     ---------------------------------------------

     USE GALAHAD_KINDS_precision
     USE GALAHAD_CLOCK
     USE GALAHAD_SYMBOLS
     USE GALAHAD_STRING, ONLY: STRING_lower_word
     USE GALAHAD_AMD_precision
     USE GALAHAD_SORT_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_SPECFILE_precision
     USE GALAHAD_SMT_precision
     USE GALAHAD_SILS_precision
     USE GALAHAD_BLAS_inter_precision, ONLY : TRSV, TBSV, GEMV, GER, SWAP, SCAL
     USE GALAHAD_LAPACK_inter_precision, ONLY : LAENV, POTRF, POTRS, SYTRF,    &
                                                SYTRS, PBTRF, PBTRS ! , SYEV
     USE GALAHAD_HSL_inter_precision, ONLY: MA27I, MC61A, MC77A
     USE hsl_zd11_precision
     USE hsl_ma57_precision
     USE hsl_ma77_precision
     USE hsl_ma86_precision
     USE hsl_ma87_precision
     USE hsl_ma97_precision
     USE hsl_mc64_precision
     USE hsl_mc68_integer
     USE MKL_PARDISO
     USE SPRAL_SSIDS_precision
     USE GALAHAD_MUMPS_TYPES_precision, MPI_COMM_WORLD_mumps => MPI_COMM_WORLD
     USE spmf_enums, MPI_COMM_WORLD_pastix => MPI_COMM_WORLD
     USE spmf_interfaces_precision
     USE pastixf_enums, MPI_COMM_WORLD_pastix_duplic8 => MPI_COMM_WORLD
     USE pastixf_interfaces_precision
!    USE omp_lib

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: SLS_initialize, SLS_analyse, SLS_factorize, SLS_solve,          &
               SLS_fredholm_alternative, SLS_terminate, SLS_enquire,           &
               SLS_alter_d, SLS_part_solve, SLS_sparse_forward_solve,          &
               SLS_read_specfile, SLS_initialize_solver,                       &
               SLS_coord_to_extended_csr, SLS_coord_to_sorted_csr,             &
               SLS_full_initialize, SLS_full_terminate,                        &
               SLS_analyse_matrix, SLS_factorize_matrix, SLS_solve_system,     &
               SLS_reset_control, SLS_partial_solve, SLS_information,          &
               SMT_type, SMT_get, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE SLS_initialize
       MODULE PROCEDURE SLS_initialize, SLS_full_initialize
     END INTERFACE SLS_initialize

     INTERFACE SLS_solve
       MODULE PROCEDURE SLS_solve_ir, SLS_solve_ir_multiple
     END INTERFACE

     INTERFACE SLS_terminate
       MODULE PROCEDURE SLS_terminate, SLS_full_terminate
     END INTERFACE SLS_terminate

!  other parameters

     INTEGER ( KIND = ip_ ), PARAMETER :: len_solver = 20
     REAL ( KIND = rp_ ), PARAMETER :: epsmch = EPSILON( 1.0_rp_ )

!  default control values

     INTEGER ( KIND = ip_ ), PARAMETER :: bits_default = 32
     INTEGER ( KIND = ip_ ), PARAMETER :: block_size_kernel_default = 40
     INTEGER ( KIND = ip_ ), PARAMETER :: block_size_elimination_default = 256
     INTEGER ( KIND = ip_ ), PARAMETER :: blas_block_size_factor_default = 16
     INTEGER ( KIND = ip_ ), PARAMETER :: blas_block_size_solve_default = 16
     INTEGER ( KIND = ip_ ), PARAMETER :: node_amalgamation_default = 32
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_pool_size_default = 100000
     INTEGER ( KIND = ip_ ), PARAMETER :: min_real_factor_size_default = 10000
     INTEGER ( KIND = ip_ ), PARAMETER :: min_integer_factor_size_default= 10000
     INTEGER ( KIND = ip_ ), PARAMETER :: full_row_threshold_default = 100
     INTEGER ( KIND = ip_ ), PARAMETER :: row_search_indefinite_default = 10

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLS_control_type

!  unit for error messages

       INTEGER ( KIND = ip_ ) :: error = 6

!  unit for warning messages

       INTEGER ( KIND = ip_ ) :: warning = 6

!  unit for monitor output

       INTEGER ( KIND = ip_ ) :: out = 6

!  unit for statistical output

       INTEGER ( KIND = ip_ ) :: statistics = 0

!  controls level of diagnostic output

       INTEGER ( KIND = ip_ ) :: print_level = 0

!  controls level of diagnostic output from external solver

       INTEGER ( KIND = ip_ ) :: print_level_solver = 0

!  number of bits used in architecture

       INTEGER ( KIND = ip_ ) :: bits = bits_default

!  the target blocksize for kernel factorization

       INTEGER ( KIND = ip_ ) :: block_size_kernel = block_size_kernel_default

!  the target blocksize for parallel elimination

       INTEGER ( KIND = ip_ ) :: block_size_elimination                        &
                                   = block_size_elimination_default

!  level 3 blocking in factorize

       INTEGER ( KIND = ip_ ) :: blas_block_size_factorize                     &
                                   = blas_block_size_factor_default

!  level 2 and 3 blocking in solve

       INTEGER ( KIND = ip_ ) :: blas_block_size_solve                         &
                                   = blas_block_size_solve_default

!  a child node is merged with its parent if they both involve fewer than
!  node_amalgamation eliminations

       INTEGER ( KIND = ip_ ) :: node_amalgamation = node_amalgamation_default

!  initial size of task-pool arrays for parallel elimination

       INTEGER ( KIND = ip_ ) :: initial_pool_size = initial_pool_size_default

!  initial size for real array for the factors and other data

       INTEGER ( KIND = ip_ ) :: min_real_factor_size                          &
                                   = min_real_factor_size_default

!  initial size for integer array for the factors and other data

       INTEGER ( KIND = ip_ ) :: min_integer_factor_size                       &
                                   = min_integer_factor_size_default

!  maximum size for real array for the factors and other data

       INTEGER ( KIND = long_ ) :: max_real_factor_size = HUGE( 0 )

!  maximum size for integer array for the factors and other data

       INTEGER ( KIND = long_ ) :: max_integer_factor_size = HUGE( 0 )

!  amount of in-core storage to be used for out-of-core factorization

!      INTEGER ( KIND = long_ ) :: max_in_core_store = HUGE( 0 ) / real_bytes_
       INTEGER ( KIND = long_ ) :: max_in_core_store =                         &
           FLOOR( REAL( HUGE( 0 ), KIND = dp_ ) /                              &
                  REAL( real_bytes_, KIND = dp_ ), KIND = long_ )

!  factor by which arrays sizes are to be increased if they are too small

       REAL ( KIND = rp_ ) :: array_increase_factor = 2.0_rp_

!  if previously allocated internal workspace arrays are greater than
!  array_decrease_factor times the currently required sizes, they are reset
!  to current requirements

       REAL ( KIND = rp_ ) :: array_decrease_factor = 2.0_rp_

!  pivot control:
!   1  Numerical pivoting will be performed.
!   2  No pivoting will be performed and an error exit will
!      occur immediately a pivot sign change is detected.
!   3  No pivoting will be performed and an error exit will
!      occur if a zero pivot is detected.
!   4  No pivoting is performed but pivots are changed to all be positive

       INTEGER ( KIND = ip_ ) :: pivot_control = 1

!  controls ordering (ignored if explicit PERM argument present)
!  <0  calculated internally by package with appropriate ordering -ordering
!   0  chosen package default (or the AMD ordering if no package default)
!   1  Approximate minimum degree (AMD) with provisions for "dense" rows/columns
!   2  Minimum degree
!   3  Nested disection
!   4  indefinite ordering to generate a combination of 1x1 and 2x2 pivots
!   5  Profile/Wavefront reduction
!   6  Bandwidth reduction
!   7  earlier implementation of AMD with no provisions for "dense" rows/columns
!  >7  ordering chosen depending on matrix characteristics (not yet implemented)

       INTEGER ( KIND = ip_ ) :: ordering = 0

!  controls threshold for detecting full rows in analyse, registered as
!  percentage of matrix order. If 100, only fully dense rows detected (default)

       INTEGER ( KIND = ip_ ) :: full_row_threshold = full_row_threshold_default

!  number of rows searched for pivot when using indefinite ordering

       INTEGER ( KIND = ip_ ) :: row_search_indefinite                         &
                                   = row_search_indefinite_default

!  controls scaling (ignored if explicit SCALE argument present)
!  <0  calculated internally by package with appropriate scaling -scaling
!   0  No scaling
!   1  Scaling using MC64
!   2  Scaling using MC77 based on the row one-norm
!   3  Scaling using MC77 based on the row infinity-norm

       INTEGER ( KIND = ip_ ) :: scaling = 0

!  the number of scaling iterations performed (default 10 used if
!   %scale_maxit < 0)

       INTEGER ( KIND = ip_ ) :: scale_maxit = 0

!  the scaling iteration stops as soon as the row/column norms are less
!   than 1+/-%scale_thresh

       REAL ( KIND = rp_ ) :: scale_thresh = 0.1_rp_

!  pivot threshold

       REAL ( KIND = rp_ ) :: relative_pivot_tolerance = 0.01_rp_

!  smallest permitted relative pivot threshold

       REAL ( KIND = rp_ ) :: minimum_pivot_tolerance = 0.01_rp_

!  any pivot small than this is considered zero

       REAL ( KIND = rp_ ) :: absolute_pivot_tolerance = EPSILON( 1.0_rp_ )

!  any entry smaller than this is considered zero

       REAL ( KIND = rp_ ) :: zero_tolerance = 0.0_rp_

!  any pivot smaller than this is considered zero for positive-definite solvers

       REAL ( KIND = rp_ ) :: zero_pivot_tolerance = EPSILON( 1.0_rp_ )

!  any pivot smaller than this is considered to be negative for p-d solvers

       REAL ( KIND = rp_ ) :: negative_pivot_tolerance                         &
                                = - 0.5_rp_ * HUGE( 1.0_rp_ )

!  used for setting static pivot level

       REAL ( KIND = rp_ ) :: static_pivot_tolerance = 0.0_rp_

!  used for switch to static

       REAL ( KIND = rp_ ) :: static_level_switch = 0.0_rp_

!  used to determine whether a system is consistent when seeking a Fredholm
!   alternative

       REAL ( KIND = rp_ ) :: consistency_tolerance = EPSILON( 1.0_rp_ )

!  maximum number of iterative refinements allowed

       INTEGER ( KIND = ip_ ) :: max_iterative_refinements = 0

!  refinement will cease as soon as the residual ||Ax-b|| falls below
!     max( acceptable_residual_relative * ||b||, acceptable_residual_absolute )

       REAL ( KIND = rp_ ) :: acceptable_residual_relative = 10.0_rp_ * epsmch
       REAL ( KIND = rp_ ) :: acceptable_residual_absolute = 10.0_rp_ * epsmch

!  set %multiple_rhs to .true. if there is possibility that the solver
!   will be required to solve systems with more than one right-hand side.
!   More efficient execution may be possible when  %multiple_rhs = .false.

       LOGICAL :: multiple_rhs = .TRUE.

!   if %generate_matrix_file is .true. if a file describing the current
!    matrix is to be generated

        LOGICAL :: generate_matrix_file = .FALSE.

!    specifies the unit number to write the input matrix (in co-ordinate form)

        INTEGER ( KIND = ip_ ) :: matrix_file_device = 74

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

       CHARACTER ( LEN = 30 ) :: prefix = '""' // REPEAT( ' ', 28 )
     END TYPE SLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: SLS_time_type

!  the total cpu time spent in the package

       REAL ( KIND = rp_ ) :: total = 0.0

!  the total cpu time spent in the analysis phase

       REAL ( KIND = rp_ ) :: analyse = 0.0

!  the total cpu time spent in the factorization phase

       REAL ( KIND = rp_ ) :: factorize = 0.0

!  the total cpu time spent in the solve phases

       REAL ( KIND = rp_ ) :: solve = 0.0

!  the total cpu time spent by the external solver in the ordering phase

       REAL ( KIND = rp_ ) :: order_external = 0.0

!  the total cpu time spent by the external solver in the analysis phase

       REAL ( KIND = rp_ ) :: analyse_external = 0.0

!  the total cpu time spent by the external solver in the factorization phase

       REAL ( KIND = rp_ ) :: factorize_external = 0.0

!  the total cpu time spent by the external solver in the solve phases

       REAL ( KIND = rp_ ) :: solve_external = 0.0

!  the total clock time spent in the package

       REAL ( KIND = rp_ ) :: clock_total = 0.0

!  the total clock time spent in the analysis phase

       REAL ( KIND = rp_ ) :: clock_analyse = 0.0

!  the total clock time spent in the factorization phase

       REAL ( KIND = rp_ ) :: clock_factorize = 0.0

!  the total clock time spent in the solve phases

       REAL ( KIND = rp_ ) :: clock_solve = 0.0

!  the total clock time spent by the external solver in the ordering phase

       REAL ( KIND = rp_ ) :: clock_order_external = 0.0

!  the total clock time spent by the external solver in the analysis phase

       REAL ( KIND = rp_ ) :: clock_analyse_external = 0.0

!  the total clock time spent by the external solver in the factorization phase

       REAL ( KIND = rp_ ) :: clock_factorize_external = 0.0

!  the total clock time spent by the external solver in the solve phases

       REAL ( KIND = rp_ ) :: clock_solve_external = 0.0

     END TYPE SLS_time_type

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

       INTEGER ( KIND = ip_ ) :: status = 0

!  STAT value after allocate failure

       INTEGER ( KIND = ip_ ) :: alloc_status = 0

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  further information on failure

       INTEGER ( KIND = ip_ ) :: more_info = 0

!  number of entries

       INTEGER ( KIND = ip_ ) :: entries = - 1

!  number of indices out-of-range

       INTEGER ( KIND = ip_ ) :: out_of_range = 0

!  number of duplicates

       INTEGER ( KIND = ip_ ) :: duplicates = 0

!  number of entries from the strict upper triangle

       INTEGER ( KIND = ip_ ) :: upper = 0

!  number of missing diagonal entries for an allegedly-definite matrix

       INTEGER ( KIND = ip_ ) :: missing_diagonals = 0

!  maximum depth of the assembly tree

       INTEGER ( KIND = ip_ ) :: max_depth_assembly_tree = - 1

!  nodes in the assembly tree (= number of elimination steps)

       INTEGER ( KIND = ip_ ) :: nodes_assembly_tree = - 1

!  desirable or actual size for real array for the factors and other data

       INTEGER ( KIND = long_ ) :: real_size_desirable = - 1

!  desirable or actual size for integer array for the factors and other data

       INTEGER ( KIND = long_ ) :: integer_size_desirable = - 1

!  necessary size for real array for the factors and other data

       INTEGER ( KIND = long_ ) :: real_size_necessary = - 1

!  necessary size for integer array for the factors and other data

       INTEGER ( KIND = long_ ):: integer_size_necessary = - 1

!  predicted or actual number of reals to hold factors

       INTEGER ( KIND = long_ ) :: real_size_factors  = - 1

!  predicted or actual number of integers to hold factors

       INTEGER ( KIND = long_ ) :: integer_size_factors = - 1

!  number of entries in factors

       INTEGER ( KIND = long_ ) :: entries_in_factors = - 1_long_

!   maximum number of tasks in the factorization task pool

       INTEGER ( KIND = ip_ ) :: max_task_pool_size = - 1

!  forecast or actual size of largest front

       INTEGER ( KIND = ip_ ) :: max_front_size = - 1

!  number of compresses of real data

       INTEGER ( KIND = ip_ ) :: compresses_real = - 1

!  number of compresses of integer data

       INTEGER ( KIND = ip_ ) :: compresses_integer = - 1

!  number of 2x2 pivots

       INTEGER ( KIND = ip_ ) :: two_by_two_pivots = - 1

!  semi-bandwidth of matrix following bandwidth reduction

       INTEGER ( KIND = ip_ ) :: semi_bandwidth = - 1

!  number of delayed pivots (total)

       INTEGER ( KIND = ip_ ) :: delayed_pivots = - 1

!  number of pivot sign changes if no pivoting is used successfully

       INTEGER ( KIND = ip_ ) :: pivot_sign_changes = - 1

!  number of static pivots chosen

       INTEGER ( KIND = ip_ ) :: static_pivots = - 1

!  first pivot modification when static pivoting

       INTEGER ( KIND = ip_ ) :: first_modified_pivot = - 1

!  estimated rank of the matrix

       INTEGER ( KIND = ip_ ) :: rank = - 1

!  number of negative eigenvalues

       INTEGER ( KIND = ip_ ) :: negative_eigenvalues = - 1

!  number of pivots that are considered zero (and ignored)

       INTEGER ( KIND = ip_ ) :: num_zero = 0

!  number of iterative refinements performed

       INTEGER ( KIND = ip_ ) :: iterative_refinements = 0

!  anticipated or actual number of floating-point operations in assembly

       INTEGER ( KIND = long_ ) :: flops_assembly = - 1_long_

!  anticipated or actual number of floating-point operations in elimination

       INTEGER ( KIND = long_ ) :: flops_elimination = - 1_long_

!  additional number of floating-point operations for BLAS

       INTEGER ( KIND = long_ ) :: flops_blas = - 1_long_

!  largest diagonal modification when static pivoting or ensuring definiteness

       REAL ( KIND = rp_ ) :: largest_modified_pivot = - 1.0_rp_

!  minimum scaling factor

       REAL ( KIND = rp_ ) :: minimum_scaling_factor = 1.0_rp_

!  maximum scaling factor

       REAL ( KIND = rp_ ) :: maximum_scaling_factor = 1.0_rp_

!  esimate of the condition number of the matrix (category 1 equations)

       REAL ( KIND = rp_ ) :: condition_number_1 = - 1.0_rp_

!  estimate of the condition number of the matrix (category 2 equations)

       REAL ( KIND = rp_ ) :: condition_number_2 = - 1.0_rp_

!  esimate of the backward error (category 1 equations)

       REAL ( KIND = rp_ ) :: backward_error_1 = - 1.0_rp_

!  esimate of the backward error (category 2 equations)

       REAL ( KIND = rp_ ) :: backward_error_2 = - 1.0_rp_

!  estimate of forward error

       REAL ( KIND = rp_ ) :: forward_error = - 1.0_rp_

!  has an "alternative" y: A y = 0 and yT b > 0 been found when trying to
!  solve A x = b ?

       LOGICAL :: alternative = .FALSE.

!  name of linear solver used

       CHARACTER ( LEN = len_solver ) :: solver = REPEAT( ' ', len_solver )

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

       INTEGER ( KIND = ip_ ), DIMENSION( 10 ) :: mc61_info = 0
       REAL ( KIND = rp_ ), DIMENSION( 15 ) :: mc61_rinfo = - 1.0_rp_

!  the output structure from mc64

       TYPE ( MC64_info ) :: mc64_info

!  the output structure from mc68

       TYPE ( MC68_info ) :: mc68_info

!  the integer output array from mc77

       INTEGER ( KIND = ip_ ), DIMENSION( 10 ) :: mc77_info = 0

!  the real output status from mc77

        REAL ( KIND = rp_ ), DIMENSION( 10 ) :: mc77_rinfo  = -1.0_rp_

!  the output scalars and arrays from mumps

       INTEGER ( KIND = ip_ ) :: mumps_error = 0
       INTEGER ( KIND = ip_ ), DIMENSION( 80 ) :: mumps_info = - 1
       REAL ( KIND = rp_ ), DIMENSION( 40 ) :: mumps_rinfo = - 1.0_rp_

!  the output scalars and arrays from pardiso

       INTEGER ( KIND = ip_ ) :: pardiso_error = 0
       INTEGER ( KIND = ip_ ), DIMENSION( 64 ) :: pardiso_IPARM = - 1
       REAL ( KIND = rp_ ), DIMENSION( 64 ) :: pardiso_DPARM = - 1.0_rp_

!  the output scalars and arrays from mkl_pardiso

       INTEGER ( KIND = ip_ ) :: mkl_pardiso_error = 0
       INTEGER ( KIND = ip_ ), DIMENSION( 64 ) :: mkl_pardiso_IPARM = - 1

!  the output scalar from pastix

       INTEGER ( KIND = ip_ ) :: pastix_info = 0

!  the output scalars and arrays from wsmp

       INTEGER ( KIND = ip_ ) :: wsmp_error = 0
       INTEGER ( KIND = ip_ ), DIMENSION( 64 ) :: wsmp_iparm = - 1
       REAL ( KIND = rp_ ), DIMENSION( 64 ) :: wsmp_dparm = - 1.0_rp_

!  the output scalar from mpi

       INTEGER ( KIND = ip_ ) :: mpi_ierr = 0

!  the output scalar from LAPACK routines

       INTEGER ( KIND = ip_ ) :: lapack_error = 0

     END TYPE SLS_inform_type

!  ...................
!   data derived type
!  ...................

     TYPE, PUBLIC :: SLS_data_type
       PRIVATE
       INTEGER ( KIND = ip_ ) :: len_solver = - 1
       INTEGER ( KIND = ip_ ) :: n, ne, matrix_ne, matrix_scale_ne
       INTEGER ( KIND = ip_ ) :: pardiso_mtype, mc61_lirn
       INTEGER ( KIND = ip_ ) :: mc61_liw, mc77_liw, mc77_ldw, sytr_lwork
       CHARACTER ( LEN = len_solver ) :: solver = REPEAT( ' ', len_solver )
       LOGICAL :: explicit_scaling, reordered
       LOGICAL ( KIND = lp_ ) :: must_be_definite
       INTEGER ( KIND = ip_ ) :: set_res = - 1
       INTEGER ( KIND = ip_ ) :: set_res2 = - 1
       LOGICAL :: got_maps_scale = .FALSE.
       LOGICAL :: no_mpi = .FALSE.
       LOGICAL :: no_mumps = .FALSE.
       LOGICAL :: no_pastix = .FALSE.
       LOGICAL :: no_sils = .FALSE.
       LOGICAL :: no_ma57 = .FALSE.
       LOGICAL :: no_ssids = .FALSE.
       LOGICAL :: trivial_matrix_type = .FALSE.
       INTEGER ( KIND = long_ ), DIMENSION( 64 ) :: pardiso_PT
       TYPE ( MKL_PARDISO_HANDLE ), DIMENSION( 64 ) :: mkl_pardiso_PT
       INTEGER ( KIND = ip_ ), DIMENSION( 1 ) :: idum
       INTEGER ( KIND = ip_ ), DIMENSION( 64 ) :: pardiso_IPARM = - 1
       INTEGER ( KIND = ip_ ), DIMENSION( 64 ) :: mkl_pardiso_IPARM = - 1
       INTEGER ( KIND = ip_ ), DIMENSION( 0 ) :: wsmp_aux
       INTEGER ( KIND = ip_ ), DIMENSION( 64 ) :: wsmp_IPARM = 0
       INTEGER ( KIND = ip_ ), DIMENSION( 10 ) :: mc61_ICNTL                   &
         = (/ 6, 6, 0, 0, 0, 0, 0, 0, 0, 0 /)
       REAL ( KIND = rp_ ), DIMENSION( 1 ) :: ddum
       REAL ( KIND = rp_ ), DIMENSION( 5 ) :: mc61_CNTL                        &
         = (/ 2.0_rp_,  1.0_rp_,  0.0_rp_,  0.0_rp_,  0.0_rp_ /)
       INTEGER ( KIND = ip_ ), DIMENSION( 10 ) :: mc77_ICNTL                   &
         = (/ 6, 6, - 1, 0, 0, 1, 10, 0, 0, 0 /)
       REAL ( KIND = rp_ ), DIMENSION( 10 ) :: mc77_CNTL                       &
         = (/ 0.0_rp_,  1.0_rp_,  0.0_rp_,  0.0_rp_,  0.0_rp_,                 &
              0.0_rp_,  0.0_rp_,  0.0_rp_,  0.0_rp_,  0.0_rp_ /)
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: ORDER, MAPS
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: PIVOTS, MAPS_scale
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: INVP, MRP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : , : ) :: MAP
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: mc64_PERM
       INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: mc61_IW, mc77_IW
       REAL ( KIND = rp_ ), DIMENSION( 64 ) :: pardiso_DPARM = 0.0_rp_
       REAL ( KIND = rp_ ), DIMENSION( 64 ) :: wsmp_DPARM = - 1.0_rp_
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RESIDUALS
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RESIDUALS_zero
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: B1
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: RES
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: SCALE
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: WORK
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: B2
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: RES2
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: X2
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: D
       REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : , : ) :: matrix_dense
       REAL ( KIND = rp_ ), DIMENSION( 0 : 0 ) :: DIAG
!      LOGICAL, ALLOCATABLE, DIMENSION( : ) :: LFLAG
       LOGICAL ( KIND = lp_ ), ALLOCATABLE, DIMENSION( : ) :: LFLAG

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

       TYPE ( AMD_data_type ) :: amd_data
       TYPE ( AMD_control_type ) :: amd_control
       TYPE ( AMD_inform_type ) :: amd_inform

       TYPE ( MC64_control ) :: mc64_control
       TYPE ( MC64_info ) :: mc64_info

       TYPE ( MC68_control ) :: mc68_control
       TYPE ( MC68_info ) :: mc68_info

       INTEGER ( KIND = ip_ ), POINTER, DIMENSION( : ) :: ROW, COL, PTR
       REAL ( KIND = rpc_ ), POINTER, DIMENSION( : ) :: VAL
       TYPE ( pastix_data_t ), POINTER :: pastix_data
       TYPE ( spmatrix_t ), POINTER :: spm, spm_check
       TYPE ( pastix_order_t ), POINTER :: order_pastix => NULL( )
       INTEGER ( KIND = pastix_int_t ), DIMENSION( : ), POINTER :: PERMTAB
       INTEGER ( KIND = pastix_int_t ) :: iparm_pastix( 75 )
       REAL ( KIND = rp_ ) :: dparm_pastix( 24 )
       TYPE ( MUMPS_STRUC ) :: mumps_par

     END TYPE SLS_data_type

     TYPE, PUBLIC :: SLS_full_data_type
       LOGICAL :: f_indexing = .TRUE.
       TYPE ( SLS_data_type ) :: SLS_data
       TYPE ( SLS_control_type ) :: SLS_control
       TYPE ( SLS_inform_type ) :: SLS_inform
       TYPE ( SMT_type ) :: matrix
     END TYPE SLS_full_data_type

!--------------------------------
!   I n t e r f a c e  B l o c k
!--------------------------------

     INTERFACE
       SUBROUTINE MUMPS_precision( mumps_par )
       USE GALAHAD_MUMPS_TYPES_precision
       TYPE ( MUMPS_STRUC ) :: mumps_par
       END SUBROUTINE MUMPS_precision
     END INTERFACE

   CONTAINS

     SUBROUTINE SLS_initialize( solver, data, control, inform, check )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Set initial values, including default control data and solver used, for SLS.
!  This routine must be called before the first call to SLS_analyse

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( SLS_data_type ), INTENT( OUT ) :: data
     TYPE ( SLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLS_inform_type ), INTENT( OUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: check

!  local variables

     INTEGER ( KIND = ip_ ), PARAMETER :: n_dummy = 2
     INTEGER ( KIND = ip_ ), DIMENSION( n_dummy + 1 )  :: PTR = (/ 1, 2, 3 /)
     INTEGER ( KIND = ip_ ), DIMENSION( n_dummy ) :: ROW = (/ 2, 1 /)
     INTEGER ( KIND = ip_ ), DIMENSION( n_dummy ) :: ROWL = (/ 1, 2 /)
     INTEGER ( KIND = ip_ ), DIMENSION( 8 ) :: ICNTL_metis
     INTEGER ( KIND = ip_ ), DIMENSION( n_dummy ) :: PERM, INVP
     TYPE ( mc68_control ) :: control_mc68
     TYPE ( mc68_info ) :: info_mc68

     LOGICAL :: check_available, hsl_available, metis_available

!    write(6,*) ' omp cancellation = ', omp_get_cancellation()
!  initialize the solver-specific data

     CALL SLS_initialize_solver( solver, data, control%error, inform, check )
     IF ( inform%status == GALAHAD_error_unknown_solver .OR.                   &
          inform%status == GALAHAD_error_omp_env ) RETURN

!  check to see if HSL ordering packages are available

     control_mc68%lp = - 1
     CALL MC68_order( 1_ip_, n_dummy, PTR, ROWL, PERM, control_mc68, info_mc68 )
     hsl_available = info_mc68%flag >= 0

!  check to see if the MeTiS ordering packages is available

     CALL galahad_metis_setopt( ICNTL_metis )
     CALL galahad_metis( n_dummy, PTR, ROW, 1_ip_, ICNTL_metis, INVP, PERM )
     metis_available = PERM( 1 ) > 0
! write(6,*) ' hsl_available, metis_available ', hsl_available, metis_available
! write(6,*) ' solver ', TRIM( inform%solver )
! stop

!  if required, check to see which ordering options are available

     IF ( PRESENT( check ) ) THEN
       check_available = check
     ELSE
       check_available = .FALSE.
     END IF

!  set the ordering so that, in the worst case, it defaults to early AMD

     IF ( check_available ) THEN
       IF ( control%ordering > 0 ) THEN
         IF ( hsl_available ) THEN
           IF ( control%ordering == 3 .AND. .NOT. metis_available )            &
             control%ordering = 1
         ELSE
           control%ordering = 7
         END IF
       END IF

     END IF

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
!IS64  control%max_in_core_store = HUGE( 0_long ) / real_bytes_

!  = MA86 =

     CASE ( 'ma86' )
       control%absolute_pivot_tolerance = data%ma86_control%small
!86V2  IF ( control%scaling == 0 )                                             &
!86V2    control%scaling = - data%ma86_control%scaling
       IF ( control%ordering == 0 ) THEN
         IF ( hsl_available ) THEN
           IF ( metis_available ) THEN
             control%ordering = 3
           ELSE
             control%ordering = 1
           END IF
         ELSE
           control%ordering = 7
         END IF
       END IF

!  = MA87 =

     CASE ( 'ma87' )
       control%zero_pivot_tolerance = SQRT( EPSILON( 1.0_rp_ ) )
       IF ( control%ordering == 0 ) THEN
         IF ( hsl_available ) THEN
           IF ( metis_available ) THEN
             control%ordering = 3
           ELSE
             control%ordering = 1
           END IF
         ELSE
           control%ordering = 7
         END IF
       END IF

!  = MA97 =

     CASE ( 'ma97' )
       IF ( control%scaling == 0 )                                             &
         control%scaling = - data%ma97_control%scaling
!      control%node_amalgamation = 8
       IF ( control%ordering == 0 ) THEN
         IF ( check_available ) THEN
           IF ( metis_available ) THEN
             control%ordering = - 3
           ELSE
             control%ordering = - 5
           END IF
         ELSE
           control%ordering = - 5
         END IF
       END IF

!  = SSIDS =

     CASE ( 'ssids' )
       IF ( control%scaling == 0 )                                             &
         control%scaling = - data%ssids_options%scaling
!      control%node_amalgamation = 8
       IF ( control%ordering == 0 ) THEN
         IF ( hsl_available ) THEN
           IF ( metis_available ) THEN
             control%ordering = 3
           ELSE
             control%ordering = 1
           END IF
         ELSE
           control%ordering = 7
         END IF
       END IF

!  = PARDISO =

     CASE ( 'pardiso', 'mkl_pardiso' )
       control%node_amalgamation = 80

!  = WSMP =

     CASE ( 'wsmp' )

!  = PaStiX =

     CASE ( 'pastix' )

!  = MUMPS =

     CASE ( 'mumps' )

!  = POTR =

     CASE ( 'potr' )

!  = SYTR =

     CASE ( 'sytr' )

!  = PBTR =

     CASE ( 'pbtr' )
       IF ( control%ordering == 0 ) THEN
         IF ( hsl_available ) THEN
           control%ordering = 6
         ELSE
           control%ordering = 7
         END IF
       END IF
     END SELECT

     RETURN

!  End of SLS_initialize

     END SUBROUTINE SLS_initialize

!- G A L A H A D -  S L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E -

     SUBROUTINE SLS_full_initialize( solver, data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for SLS controls

!   Arguments:

!   solver   name of solver to be used
!   data     private internal data
!   control  a structure containing control information. See preamble
!   inform   a structure containing output information. See preamble

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( OUT ) :: control
     TYPE ( SLS_inform_type ), INTENT( OUT ) :: inform

     CALL SLS_initialize( solver, data%sls_data, control, inform )
     data%sls_inform%solver = inform%solver

     RETURN

!  End of subroutine SLS_full_initialize

     END SUBROUTINE SLS_full_initialize

!-*-*-   S L S _ I N I T I A L I Z E _ S O L V E R  S U B R O U T I N E   -*-*-

     SUBROUTINE SLS_initialize_solver( solver, data, error, inform, check )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Set initial values, including default control data and solver used, for SLS.
!  This routine must be called before the first call to SLS_analyse.
!  If check is present and true, attempts will be made to ensure that the
!  requested solver is available, and if not to provide a suitable alternative

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: error
     TYPE ( SLS_inform_type ), INTENT( OUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: check

!  local variables

     INTEGER ( KIND = ip_ ) :: flag_ssids
     LOGICAL :: check_available, mpi_initialzed_flag
     INTEGER ( KIND = ip_ ), DIMENSION( 30 ) :: ICNTL_ma27
     REAL ( KIND = rp_ ), DIMENSION( 5 ) :: CNTL_ma27
     TYPE ( MA57_control ) :: control_ma57
     TYPE ( ssids_akeep ) :: akeep_ssids
!$   LOGICAL :: OMP_GET_CANCELLATION
!$   INTEGER ( KIND = ip_ ) :: OMP_GET_PROC_BIND

!  record the solver

     IF ( PRESENT( check ) ) THEN
       check_available = check
     ELSE
       check_available = .FALSE.
     END IF

     data%len_solver = MIN( len_solver, LEN_TRIM( solver ) )
     data%solver = REPEAT( ' ', len_solver )
     data%solver( 1 : data%len_solver ) = solver( 1 : data%len_solver )
     CALL STRING_lower_word( data%solver( 1 : data%len_solver ) )

!  ensure that OpenMP has been correctly initialized

!$   SELECT CASE( data%solver( 1 : data%len_solver ) )
!$   CASE ( 'ssids', 'mumps' )
!!!!$     write(6,*) 'omp', OMP_GET_CANCELLATION( ), OMP_GET_PROC_BIND( )
!$     IF ( .NOT. OMP_GET_CANCELLATION( ) .OR.                                 &
!$          OMP_GET_PROC_BIND( ) == 0 ) THEN
!!!!$          OMP_GET_PROC_BIND( ) /= 1 ) THEN
!$       IF ( error > 0 ) WRITE( error,                                        &
!$         "( ' WARNING: To use the requested linear solver ', A,              &
!$      &     ', the environment variables', /,  '          OMP_CANCELLATION', &
!$      &     ' and OMP_PROC_BIND must both be set to TRUE' )" )               &
!$            data%solver( 1 : data%len_solver )
!$       inform%status = GALAHAD_error_omp_env ; RETURN
!$     END IF
!$   END SELECT

!  initialize solver-specific controls

  10 CONTINUE
     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       CALL MA27I( ICNTL_ma27, CNTL_ma27 )
       data%no_sils = ICNTL_ma27( 4 ) == - 1
       IF ( data%no_sils ) THEN
         IF ( check_available ) THEN ! if sils is unavailble, use ssids instead
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 5
           data%solver( 1 : data%len_solver ) = 'ssids'
           GO TO 10
         ELSE
           inform%status = GALAHAD_unavailable_option ; RETURN
         END IF
       END IF
       data%must_be_definite = .FALSE.
       CALL SILS_initialize( FACTORS = data%sils_factors,                     &
                             CONTROL = data%sils_control )

!  = MA57 =

     CASE ( 'ma57' )
       CALL MA57_initialize( control = control_ma57 )
       data%no_ma57 = control_ma57%lp == - 1
       IF ( data%no_ma57 ) THEN
         IF ( check_available ) THEN ! if ma57 is not availble, try sils instead
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 4
           data%solver( 1 : data%len_solver ) = 'sils'
           GO TO 10
         ELSE
           inform%status = GALAHAD_unavailable_option ; RETURN
         END IF
       END IF
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
       CALL SSIDS_free( akeep_ssids, flag_ssids )
       data%no_ssids = flag_ssids == GALAHAD_unavailable_option
       IF ( data%no_ssids ) THEN
         IF ( check_available ) THEN ! if ssids is unavailble, use sytr instead
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 4
           data%solver( 1 : data%len_solver ) = 'sytr'
           GO TO 10
         ELSE
           inform%status = GALAHAD_unavailable_option ; RETURN
         END IF
       END IF
       data%must_be_definite = .FALSE.

!  = PARDISO =

     CASE ( 'pardiso', 'mkl_pardiso' )
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

!  = PaStiX =

     CASE ( 'pastix' )
       ALLOCATE( data%spm )
       CALL spmInit( data%spm )
       data%no_pastix = data%spm%mtxtype == - 1
       IF ( data%no_pastix ) THEN
         IF ( check_available ) THEN ! if pastix is unavailble, use ma57 instead
           DEALLOCATE( data%spm )
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 4
           data%solver( 1 : data%len_solver ) = 'ma57'
           GO TO 10
         ELSE
!          DEALLOCATE( data%spm )
           inform%status = GALAHAD_unavailable_option ; RETURN
         END IF
       END IF

       CALL pastixInitParam( data%iparm_pastix, data%dparm_pastix )
       data%iparm_pastix( 1 ) = 0
       data%iparm_pastix( 6 ) = 0
!      IF ( unsymmetric ) THEN
!        data%iparm_pastix( 44 ) = 2
!      ELSE
         data%iparm_pastix( 44 ) = 1
!      END IF
!      OPEN( 2, FILE = "/dev/null", STATUS = "OLD" ) ! try to get rid of msgs
       CALL pastixInit( data%pastix_data, MPI_COMM_WORLD_pastix,               &
                        data%iparm_pastix, data%dparm_pastix )
!      CLOSE( 2 )
!      OPEN( 2, FILE = "/dev/stdout", STATUS = "OLD" )
       data%must_be_definite = .FALSE.

!  = MUMPS =

     CASE ( 'mumps' )
       CALL MPI_INITIALIZED( mpi_initialzed_flag, inform%mpi_ierr )
       IF ( mpi_initialzed_flag ) THEN
         data%no_mpi = .FALSE.
       ELSE
         CALL MPI_INIT( inform%mpi_ierr )
         data%no_mpi = inform%mpi_ierr < 0
       END IF

       IF ( data%no_mpi ) THEN
         data%no_mumps = .TRUE.
       ELSE
         data%mumps_par%COMM = MPI_COMM_WORLD_mumps
         data%mumps_par%JOB = - 1
         data%mumps_par%SYM = 2 ! symmetric
         data%mumps_par%PAR = 1 ! parallel solve
         CALL MUMPS_precision( data%mumps_par )
         data%no_mumps = data%mumps_par%INFOG( 1 ) == - 999
       END IF

       IF ( data%no_mumps ) THEN
         data%mumps_par%MYID = 1
         IF ( check_available ) THEN ! if mumps is unavailable, use ma57 instead
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 4
           data%solver( 1 : data%len_solver ) = 'ma57'
           GO TO 10
         ELSE
           inform%status = GALAHAD_unavailable_option ; RETURN
         END IF
       END IF
       data%must_be_definite = .FALSE.

!  = unavailable solver =

     CASE DEFAULT
       inform%status = GALAHAD_error_unknown_solver
       RETURN
     END SELECT

!  record the name of the solver actually used

     inform%solver = REPEAT( ' ', len_solver )
     inform%solver( 1 : data%len_solver ) = data%solver( 1 : data%len_solver )

     data%set_res = - 1 ; data%set_res2 = - 1
     data%got_maps_scale = .FALSE.
     inform%status = GALAHAD_ok

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
       control_ma86%static = 0.0_rp_
       control_ma86%u = 0.0_rp_
       control_ma86%umin = 0.0_rp_
       control_ma86%action = .TRUE.
     ELSE IF ( control%pivot_control == 3 ) THEN
       control_ma86%static = 0.0_rp_
       control_ma86%u = 0.0_rp_
       control_ma86%umin = 0.0_rp_
       control_ma86%action = .FALSE.
     ELSE IF ( control%pivot_control == 4 ) THEN
       control_ma86%static = control%static_pivot_tolerance
       control_ma86%u = 0.0_rp_
       control_ma86%umin = 0.0_rp_
       control_ma86%action = .TRUE.
     ELSE
       control_ma86%static = 0.0_rp_
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
       control_ma97%u = 0.0_rp_
       control_ma97%action = .TRUE.
     ELSE IF ( control%pivot_control == 3 ) THEN
       control_ma97%u = 0.0_rp_
       control_ma97%action = .FALSE.
     ELSE IF ( control%pivot_control == 4 ) THEN
       control_ma97%u = 0.0_rp_
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
       control_ssids%u = 0.0_rp_
!      control_ssids%presolve = 1
       control_ssids%action = .TRUE.
     ELSE IF ( control%pivot_control == 3 ) THEN
       control_ssids%u = 0.0_rp_
!      control_ssids%action = .TRUE.
!      control_ssids%presolve = 1
       control_ssids%action = .FALSE.
     ELSE IF ( control%pivot_control == 4 ) THEN
       control_ssids%u = 0.0_rp_
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
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 64 ) :: iparm

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

!-  S L S _ C O P Y _ C O N T R O L _ T O _ M K L _ P A R D I S O  SUBROUTINE -

     SUBROUTINE SLS_copy_control_to_mkl_pardiso( control, iparm )

!  copy control parameters to their MKL PARDISO equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 64 ) :: iparm

!  MKL defaults

     iparm( 1 ) = 1 ! don't use defaults (0 = use defaults)
     iparm( 2 ) = 2 ! METIS fill-in reordering (0 = min degree, 3 = nested diss)
     iparm( 3 ) = 0 ! currently not used
     iparm( 4 ) = 0 ! direct algorithm (> 0 for CGS)
     iparm( 5 ) = 0 ! no user fill-in reducing permutation (> 0 provided)
     iparm( 6 ) = 0 ! solution in x (1 = overwrites b)
     iparm( 7 ) = 0 ! currently not used
     iparm( 8 ) = 2 ! number of iterative refinement steps (<0 extended precis)
     iparm( 9 ) = 0 ! currently not used
     iparm( 10 ) = 8 ! perturb the pivot elements by 10**-iparm( 10 )
     iparm( 11 ) = 0 ! disable scaling (1 = enable)
     iparm( 12 ) = 0 ! usual solve (2 = transposed system)
     iparm( 13 ) = 0 ! no maximum weighted algorithm (1 = on)
     iparm( 14 ) = 0 ! (OUTPUT) number of perturbed pivots
     iparm( 15 ) = 0 ! (OUTPUT) peak memory symbolic factorization
     iparm( 16 ) = 0 ! (OUTPUT) permenent memory symbolic factorization
     iparm( 17 ) = 0 ! (OUTPUT) peak memory numerical factorization
     iparm( 18 ) = - 1 ! report number nonzeros in factors (>= 0, don't)
     iparm( 19 ) = - 1 ! report Mflops for factorization (>= 0, don't)
     iparm( 20 ) = 0 ! (OUTPUT) number of CG Iterations
     iparm( 21 ) = 1 ! 1x1 & 2x2 pivoting (1 = just 1x1)
     iparm( 22 ) = - 1 ! (OUTPUT) number of +ve eigenvalues
     iparm( 23 ) = - 1 ! (OUTPUT) number of -ve eigenvalues
     iparm( 24 ) = 0 ! parallel factorization control (> 0 2-level control)
     iparm( 25 ) = 0 ! parallel solve (1 = sequential, 2 = parallel in-core)
     iparm( 26 ) = 0 ! currently not used
     iparm( 27 ) = 0 ! do not check matrix on input (1 = check)
     iparm( 28 ) = 0 ! input in double precision (1 = single precision)
     iparm( 29 ) = 0 ! currently not used
     iparm( 30 ) = - 1 ! (OUTPUT) number of zero & -ve pivots
     iparm( 31 ) = 0 ! full solve (> 0 partial solve)
     iparm( 32 : 33 ) = 0 ! currently not used
     iparm( 34 ) = 0 ! conditional numerical reproducibility mode off (> 0 on)
     iparm( 35 ) = 0 ! one-based indexing (1 = zero-based)
     iparm( 36 ) = 0 ! do not use Schur complements (1 use it)
     iparm( 37 ) = 0 ! CSR input-matrix format (>0 BSR, <0 VBSR)
     iparm( 38 ) = 0 ! currently not used
     iparm( 39 ) = 0 ! do not use low rank update functionality (1 = do)
     iparm( 40 : 42 ) = 0 ! currently not used
     iparm( 43 ) = 0 ! do not compute diagonal of inverse (1 = do)
     iparm( 44 : 55 ) = 0 ! currently not used
     iparm( 56 ) = 0 ! turn off diagonal and pivoting control (1 = on)
     iparm( 57 : 59 ) = 0 ! currently not used
     iparm( 60 ) = 0 ! in-core mode (2 = out-of-core, 1 = switch as needed)
     iparm( 61 : 62 ) = 0 ! currently not used
     iparm( 63 ) = - 1 ! (OUTPUT) minimum size of out-of-core memory needed)
     iparm( 64 ) = 0 ! currently not used

!  values changed by input controls

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
     IF ( control%max_in_core_store == 0 ) iparm( 60 ) = 2

     RETURN

!  End of SLS_copy_control_to_mkl_pardiso

     END SUBROUTINE SLS_copy_control_to_mkl_pardiso

!-*-  S L S _ C O P Y _ C O N T R O L _ T O _ W S M P  S U B R O U T I N E -*-

     SUBROUTINE SLS_copy_control_to_wsmp( control, iparm, dparm )

!  copy control parameters to their WSMP equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 64 ) :: iparm
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 64 ) :: dparm

!  set default values as defined by the wsmp documentation

!    iparm = 0
!    iparm( 5 ) = 1
!    iparm( 6 ) = 1
!    iparm( 7 ) = 3
!    iparm( 16 ) = 1
!    iparm( 18 ) = 1

!    dparm = 0.0_rp_
!    dparm( 6 ) = 10.0_rp_ * EPSILON( 1.0_rp_ )
!    dparm( 10 ) = 10.0_rp_ ** ( - 18 )
!    dparm( 11 ) = 10.0_rp_ ** ( - 3 )
!    dparm( 12 ) = 2.0_rp_ * EPSILON( 1.0_rp_ )
!    dparm( 21 ) = 10.0_rp_ ** 200
!    dparm( 22 ) = SQRT( 2.0_rp_ * EPSILON( 1.0_rp_ ) )

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
!    dparm( 10 ) = 0.0_rp_
     dparm( 11 ) = control%relative_pivot_tolerance

     RETURN

!  End of SLS_copy_control_to_wsmp

     END SUBROUTINE SLS_copy_control_to_wsmp

!-*-  S L S _ C O P Y _ C O N T R O L _ T O _ M U M P S  S U B R O U T I N E -*-

     SUBROUTINE SLS_copy_control_to_mumps( control, ICNTL, CNTL )

!  copy control parameters to their MUMPS equivalents

!  Dummy arguments

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     INTEGER ( KIND = ip_ ), INTENT( INOUT ), DIMENSION( 60 ) :: ICNTL
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( 15 ) :: CNTL

!  set default values as defined by the mumps 5.5 documentation

!    ICNTL(1) = 6 ! output stream for error messages
!    ICNTL(2) = 0 ! output stream for diagnostic messages
!    ICNTL(3) = 6 ! output stream for global information
!    ICNTL(4) = 2 ! level of printing
!    ICNTL(5) = 0 ! matrix input format
!    ICNTL(6) = 7 ! permutes the matrix to a zero-free diagonal
!    ICNTL(7) = 7 ! ordering(1=AMD,2=AMF,3=Scotch,4=Pord,5=Metis,6=AMDD,7=auto)
!    ICNTL(8) = 77 ! scaling strategy(-1=user,0=no,1=diag,4=inf,7=equib,77=auto)
!    ICNTL(9)  = 1 ! Solve A x=b (1) or A^Tx = b (else)
!    ICNTL(10) = 0 ! Max steps iterative refinement
!    ICNTL(11) = 0 ! Error analysis stats (1=all,2=some,else=off)
!    ICNTL(12) = 0 ! ordering strategy for symmetric matrices
!    ICNTL(13) = 0 ! parallelism of the root node
!    ICNTL(14) = 35 ! percentage increase in the working space
!    ICNTL(15) = 0 ! input matrix compression
!    ICNTL(16) = 0 ! the number of OpenMP threads
!    ICNTL(18) = 0 ! distributed input matrix strategy
!    ICNTL(19) = 0 ! compute the Schur complement matrix
!    ICNTL(20) = 0 ! RHS format (dense, sparse, or distributed)
!    ICNTL(21) = 0 ! solution distribution (centralized or distributed)
!    ICNTL(22) = 0 ! OOC factorization and solve
!    ICNTL(23) = 0 ! maximum size of the per processor memory (Mbytes)
!    ICNTL(24) = 0 ! detection of “null pivot rows”
!    ICNTL(25) = 0 ! solve defficient systems?
!    ICNTL(26) = 0 ! the solution phase with Schur complement matrix
!    ICNTL(27) = - 32  ! blocking size for multiple RHS
!    ICNTL(28) = 0 ! parallel ordering?
!    ICNTL(29) = 0 ! parallel ordering tool
!    ICNTL(30) = 0 ! computes some entries in the inverse of A
!    ICNTL(31) = 0 ! discard factors during factorization
!    ICNTL(32) = 0 ! forward RHS elimination during factorization
!    ICNTL(33) = 0 ! computes the determinant of the input matrix
!    ICNTL(34) = 0 ! delete files in case of save/restore
!    ICNTL(35) = 0 ! activate Block Low-Rank feature
!    ICNTL(36) = 0 ! choice of BLR factorization variant
!    ICNTL(37) = 0 ! unused
!    ICNTL(38) = 600 ! estimates compression rate of LU factors
!    ICNTL(39:57) = 0 ! unused
!    ICNTL(58) = 0 ! defines options for symbolic factorization
!    ICNTL(59_60)  = 0 ! unused
!    CNTL(1) = 0.0_rp_ ! Threshold for numerical pivoting
!    CNTL(2) = 0.0_rp_ ! Iterative refinement stopping tolerance
!    CNTL(3) = 0.0_rp_ ! Null pivot detection threshold
!    CNTL(4) = 0.0_rp_ ! Threshold for static pivoting
!    CNTL(5) = 0.0_rp_ ! Fixation for null pivots
!    CNTL(6) = 0.0_rp_ ! Not used
!    CNTL(7) = 0.0_rp_ ! Dropping threshold for BLR compression
!    CNTL(8-15) = 0.0_rp_ ! unused
     IF ( control%print_level_solver > 0 ) THEN
       ICNTL( 1 ) = control%error ; ICNTL( 2 ) = control%out
       ICNTL( 3 ) = control%statistics
       ICNTL( 4 ) = control%print_level_solver
     ELSE
       ICNTL( 1 ) = 0 ; ICNTL( 2 ) = 0 ; ICNTL( 3 ) = 0 ; ICNTL( 4 ) = 0
     END IF
     IF ( control%scaling < 0 ) THEN
       ICNTL( 8 ) = - control%scaling
     ELSE IF ( control%scaling == 4 ) THEN
     ELSE
       ICNTL( 8 ) = 0
     END IF
     ICNTL( 10 ) = control%max_iterative_refinements
     IF ( control%pivot_control == 2 ) THEN
       CNTL( 1 ) = 0.0_rp_
     ELSE IF ( control%pivot_control == 3 ) THEN
       CNTL( 1 ) = 0.0_rp_
     ELSE IF ( control%pivot_control == 4 ) THEN
       CNTL( 1 ) = 0.0_rp_
     ELSE
       CNTL( 1 ) = control%relative_pivot_tolerance
     END IF
     ICNTL( 24 ) = 1 ! always detect singularity

!  End of SLS_copy_control_to_mumps

     END SUBROUTINE SLS_copy_control_to_mumps

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
     CASE ( - 50  )
       inform%status = GALAHAD_error_allocate
       inform%alloc_status = info_ssids%stat
     CASE( - 1, - 2, - 3, - 4, - 7, - 10, - 11, - 12, - 13, - 14 )
       inform%status = GALAHAD_error_restrictions
     CASE ( - 8, - 9, - 15 )
       inform%status = GALAHAD_error_permutation
     CASE ( - 5, - 6  )
       inform%status = GALAHAD_error_inertia
     CASE ( GALAHAD_unavailable_option  )
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
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER ( KIND = ip_ ), PARAMETER :: error = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: warning = error + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out = warning + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: statistics = out + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level = statistics + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: print_level_solver = print_level + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: block_size_kernel                    &
                                            = print_level_solver + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: bits = block_size_kernel + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: block_size_elimination = bits + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: blas_block_size_factorize            &
                                            = block_size_elimination + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: blas_block_size_solve                &
                                            = blas_block_size_factorize + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: node_amalgamation                    &
                                            = blas_block_size_solve + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: initial_pool_size                    &
                                            = node_amalgamation + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: min_real_factor_size                 &
                                            = initial_pool_size + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: min_integer_factor_size              &
                                            = min_real_factor_size + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_real_factor_size                 &
                                            = min_integer_factor_size + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_integer_factor_size              &
                                            = max_real_factor_size + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_in_core_store                    &
                                            = max_integer_factor_size + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: pivot_control = max_in_core_store + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: ordering = pivot_control + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: full_row_threshold = ordering + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: row_search_indefinite                &
                                            = full_row_threshold + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: scaling = row_search_indefinite + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: scale_maxit = scaling + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: scale_thresh = scale_maxit + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: max_iterative_refinements            &
                                            = scale_thresh + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: array_increase_factor                &
                                            = max_iterative_refinements + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: array_decrease_factor                &
                                            = array_increase_factor + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: relative_pivot_tolerance             &
                                            = array_decrease_factor + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: minimum_pivot_tolerance              &
                                            = relative_pivot_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: absolute_pivot_tolerance             &
                                            = minimum_pivot_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: zero_tolerance                       &
                                            = absolute_pivot_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: zero_pivot_tolerance                 &
                                            = zero_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: negative_pivot_tolerance             &
                                            = zero_pivot_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: static_pivot_tolerance               &
                                            = negative_pivot_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: static_level_switch                  &
                                            = static_pivot_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: consistency_tolerance                &
                                            = static_level_switch + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: matrix_file_device                   &
                                            = consistency_tolerance + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: multiple_rhs = matrix_file_device + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: generate_matrix_file                 &
                                            = multiple_rhs + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: matrix_file_name                     &
                                            = generate_matrix_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: acceptable_residual_relative         &
                                            = matrix_file_name + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: acceptable_residual_absolute         &
                                            = acceptable_residual_relative + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out_of_core_directory                &
                                            = acceptable_residual_absolute + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out_of_core_integer_factor_file      &
                                            = out_of_core_directory + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out_of_core_real_factor_file         &
                                           = out_of_core_integer_factor_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out_of_core_real_work_file           &
                                            = out_of_core_real_factor_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out_of_core_indefinite_file          &
                                            = out_of_core_real_work_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: out_of_core_restart_file             &
                                            = out_of_core_indefinite_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: prefix = out_of_core_restart_file + 1
     INTEGER ( KIND = ip_ ), PARAMETER :: lspec = prefix
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
     INTEGER ( KIND = ip_ ), INTENT( IN ), OPTIONAL :: PERM( matrix%n )

!  local variables

     INTEGER ( KIND = ip_ ) :: i, j, k, l, l1, l2, ordering, pardiso_solver
     INTEGER( ipc_ ) :: pastix_info
     REAL :: time, time_start, time_now
     REAL ( KIND = rp_ ) :: clock, clock_start, clock_now
     LOGICAL :: mc6168_ordering
     CHARACTER ( LEN = 400 ), DIMENSION( 1 ) :: path
     CHARACTER ( LEN = 400 ), DIMENSION( 4 ) :: filename
!$   INTEGER ( KIND = ip_ ) :: OMP_GET_NUM_THREADS

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases, no analysis

     SELECT CASE ( SMT_get( matrix%type ) )
     CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY', 'ZERO', 'NONE' )
       IF ( matrix%n < 1 ) THEN
         inform%status = GALAHAD_error_restrictions
       ELSE
         inform%entries = matrix%n
         inform%real_size_desirable = 0
         inform%integer_size_desirable = 0
         inform%status = GALAHAD_ok
       END IF
       data%trivial_matrix_type = .TRUE.
       data%n = matrix%n
       GO TO 900
     CASE DEFAULT
       data%trivial_matrix_type = .FALSE.
     END SELECT

!  check whether solver is available

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = SILS =

     CASE ( 'sils', 'ma27' )
       IF ( data%no_sils ) THEN
         inform%status = GALAHAD_error_unknown_solver
         GO TO 900
       END IF

!  = MA57 =

     CASE ( 'ma57' )
       IF ( data%no_ma57 ) THEN
         inform%status = GALAHAD_error_unknown_solver
         GO TO 900
       END IF

!  = SSIDS =

     CASE ( 'ssids' )
       IF ( data%no_ssids ) THEN
         inform%status = GALAHAD_error_unknown_solver
         GO TO 900
       END IF

!  = PaStiX =

     CASE ( 'pastix' )
       IF ( data%no_pastix ) THEN
         inform%status = GALAHAD_error_unknown_solver
         GO TO 900
       END IF

!  = MUMPS =

     CASE ( 'mumps' )
       IF ( data%no_mumps ) THEN
         inform%status = GALAHAD_error_unknown_solver
         GO TO 900
       END IF
     END SELECT

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

!  decide if the ordering should be chosen by one of mc61, mc68 or amd

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
       CALL SPACE_resize_array( matrix%n + 1_ip_, data%matrix%PTR,             &
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

       IF ( control%ordering > 7 .OR.control%ordering < 1 ) THEN
         IF ( data%solver( 1 : data%len_solver ) == 'pbtr' ) THEN
           ordering = 6
         ELSE
           ordering = 1
         END IF
       ELSE
         ordering = control%ordering
       END IF

!  AMD ordering (ACM TOMS rather than HSL implementation)

       IF ( ordering == 7 ) THEN
         IF ( control%print_level <= 0 .OR. control%out <= 0 )                 &
           data%amd_control%out = - 1

         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL AMD_order( data%matrix%n, data%matrix%PTR, data%matrix%COL,      &
                         data%ORDER, data%amd_data, data%amd_control,          &
                         data%amd_inform )
         CALL CPU_time( time_now ) ; CALL CLOCK_time( clock_now )
         inform%time%order_external = time_now - time
         inform%time%clock_order_external = clock_now - clock
         IF ( data%amd_inform%status == GALAHAD_error_allocate ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' amd allocation error' )" ) prefix
           inform%status = GALAHAD_error_allocate ; GO TO 900
         ELSE IF ( data%amd_inform%status == GALAHAD_error_deallocate ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' amd deallocation error' )" ) prefix
           inform%status = GALAHAD_error_deallocate ; GO TO 900
         ELSE IF ( data%amd_inform%status == GALAHAD_error_restrictions ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' amd restrictions error' )" ) prefix
           inform%status = GALAHAD_error_restrictions ; GO TO 900
         END IF

!  mc68 sparsity ordering

       ELSE IF ( ordering <= 4 ) THEN
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
         IF ( data%mc68_info%flag == - 1 .OR. data%mc68_info%flag == - 6 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' mc68 allocation error' )" ) prefix
           inform%status = GALAHAD_error_allocate ; GO TO 900
         ELSE IF ( data%mc68_info%flag == - 2 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' mc68 deallocation error' )" ) prefix
           inform%status = GALAHAD_error_deallocate ; GO TO 900
         ELSE IF ( data%mc68_info%flag == - 3 .OR.                             &
                   data%mc68_info%flag == - 4 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' mc68 restriction error' )" ) prefix
           inform%status = GALAHAD_error_restrictions ; GO TO 900
         ELSE IF ( data%mc68_info%flag == - 5 ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' MeTiS is not available' )" ) prefix
           inform%status = GALAHAD_unavailable_option ; GO TO 900
         ELSE IF ( data%mc68_info%flag == GALAHAD_unavailable_option ) THEN
           IF ( control%print_level > 0 .AND. control%out > 0 )                &
             WRITE( control%out, "( A, ' mc68 is not available' )" ) prefix
           inform%status = GALAHAD_error_unknown_solver ; GO TO 900
         END IF

!  mc61 band ordering

       ELSE
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
         CALL SPACE_resize_array( matrix%n + 1_ip_, data%INVP,                 &
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
         CALL MC61A( ordering - 4, data%matrix%n, data%mc61_lirn, data%MRP,    &
                     data%INVP, data%ORDER, data%mc61_liw, data%mc61_IW,       &
                     data%WORK, data%mc61_ICNTL, data%mc61_CNTL,               &
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
       END IF
       CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS',                       &
                     inform%alloc_status )
     END IF

!  solver-dependent analysis

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
           inform%real_size_desirable = INT( data%sils_ainfo%nrltot, long_ )
           inform%integer_size_desirable = INT( data%sils_ainfo%nirtot, long_ )
           inform%real_size_necessary = INT( data%sils_ainfo%nrlnec, long_ )
           inform%integer_size_necessary = INT( data%sils_ainfo%nirnec, long_ )
           inform%entries_in_factors = INT( data%sils_ainfo%nrladu, long_ )
           inform%real_size_factors = INT( data%sils_ainfo%nrladu, long_ )
           inform%integer_size_factors = INT( data%sils_ainfo%niradu, long_ )
           inform%compresses_integer = data%sils_ainfo%ncmpa
           inform%out_of_range = data%sils_ainfo%oor
           inform%duplicates = data%sils_ainfo%dup
           inform%max_front_size = data%sils_ainfo%maxfrt
           inform%flops_assembly = INT( data%sils_ainfo%opsa, long_ )
           inform%flops_elimination = INT( data%sils_ainfo%opse, long_ )
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
           inform%real_size_desirable = INT( data%ma57_ainfo%nrltot, long_ )
           inform%integer_size_desirable = INT( data%ma57_ainfo%nirtot, long_ )
           inform%real_size_necessary = INT( data%ma57_ainfo%nrlnec, long_ )
           inform%integer_size_necessary = INT( data%ma57_ainfo%nirnec, long_ )
           inform%entries_in_factors = INT( data%ma57_ainfo%nrladu, long_ )
           inform%real_size_factors = INT( data%ma57_ainfo%nrladu, long_ )
           inform%integer_size_factors = INT( data%ma57_ainfo%niradu, long_ )
           inform%compresses_integer = data%ma57_ainfo%ncmpa
           inform%out_of_range = data%ma57_ainfo%oor
           inform%duplicates = data%ma57_ainfo%dup
           inform%max_front_size = data%ma57_ainfo%maxfrt
           inform%flops_assembly = INT( data%ma57_ainfo%opsa, long_ )
           inform%flops_elimination = INT( data%ma57_ainfo%opse, long_ )
         END IF
       END SELECT

!  = MA77 =

     CASE ( 'ma77' )

!  if the extended matrix has not yet been constructed, do so

!      IF ( mc6168_ordering ) THEN
!        CALL SPACE_resize_array( data%matrix%PTR( matrix%n + 1 ) - 1_ip_,     &
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
         CALL SPACE_resize_array( data%matrix_ne, 2_ip_, data%MAP,             &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MAP' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n + 1_ip_, data%matrix%PTR,           &
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
         CALL MA77_input_vars( i, l2 - l1 + 1_ip_, data%matrix%ROW( l1 : l2 ), &
                               data%ma77_keep, data%ma77_control,              &
                               data%ma77_info )
         CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 800
       END DO
       CALL MA77_analyse( data%ORDER( : data%n ), data%ma77_keep,              &
                          data%ma77_control, data%ma77_info )
       CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )

!  = MA86, MA87, MA97, SSIDS, PARDISO or WSMP =

     CASE ( 'ma86', 'ma87', 'ma97', 'ssids', 'pardiso', 'mkl_pardiso',         &
            'wsmp', 'pastix' )

!  convert the data to sorted compressed-sparse row format

       IF ( .NOT. mc6168_ordering ) THEN
         CALL SPACE_resize_array( data%matrix_ne, data%MAPS,                   &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%MAPS' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( matrix%n + 1_ip_, data%matrix%PTR,           &
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
       CALL SPACE_resize_array( matrix%n, 1_ip_, data%B2,                      &
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
           CALL MA97_analyse( .FALSE._lp_, data%matrix%n,                      &
                              data%matrix%PTR, data%matrix%COL,                &
                              data%ma97_akeep,                                 &
                              data%ma97_control, data%ma97_info,               &
                              order = data%ORDER )
         ELSE
           IF ( PRESENT( PERM ) ) THEN
             data%ma97_control%ordering = 0
             CALL MA97_analyse( .FALSE._lp_, data%matrix%n,                    &
                                data%matrix%PTR, data%matrix%COL,              &
                                data%ma97_akeep,                               &
                                data%ma97_control, data%ma97_info,             &
                                order = data%ORDER )
           ELSE
             data%ma97_control%ordering = - control%ordering
             CALL MA97_analyse( .FALSE._lp_, data%matrix%n,                    &
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

         CALL SPACE_resize_array( matrix%n, 1_ip_, data%X2,                    &
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
         CALL PARDISOINIT( data%pardiso_PT, data%pardiso_mtype, pardiso_solver,&
                           data%pardiso_IPARM, data%pardiso_DPARM,             &
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
         data%pardiso_IPARM( 3 ) = 1
!$       data%pardiso_IPARM( 3 ) = OMP_GET_NUM_THREADS( )
         IF ( control%ordering > 0 .OR. PRESENT( PERM ) )                      &
           data%pardiso_IPARM( 5 ) = 1

         CALL SLS_copy_control_to_pardiso( control, data%pardiso_iparm )
         CALL PARDISO( data%pardiso_PT( 1 : 64 ), 1_ip_, 1_ip_,                &
                       data%pardiso_mtype, 11_ip_,                             &
                       data%matrix%n, data%matrix%VAL( 1 : data%ne ),          &
                       data%matrix%PTR( 1 : data%matrix%n + 1 ),               &
                       data%matrix%COL( 1 : data%ne ),                         &
                       data%ORDER( 1 : data%matrix%n ), 1_ip_,                 &
                       data%pardiso_iparm( 1 : 64 ),                           &
                       control%print_level_solver,                             &
                       data%B2( 1 : matrix%n, 1 : 1 ),                         &
                       data%X2( 1 : matrix%n, 1 : 1 ), inform%pardiso_error,   &
                       data%pardiso_dparm( 1 : 64 ) )

         inform%pardiso_IPARM = data%pardiso_IPARM
         inform%pardiso_DPARM = data%pardiso_DPARM

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
         IF ( data%pardiso_IPARM( 18 ) > 0 )                                   &
           inform%entries_in_factors = INT( data%pardiso_IPARM( 18 ), long_ )

!  = MKL PARDISO =

       CASE ( 'mkl_pardiso' )

         CALL SPACE_resize_array( matrix%n, 1_ip_, data%X2,                    &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF

!  set intial pardiso storage

         IF ( data%must_be_definite ) THEN
           data%pardiso_mtype = 2
         ELSE
           data%pardiso_mtype = - 2
         END IF
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )

         data%mkl_pardiso_PT( 1 : 64 )%DUMMY =  0

         CALL SLS_copy_control_to_mkl_pardiso( control, data%mkl_pardiso_IPARM )
         IF ( control%ordering > 0 .OR. PRESENT( PERM ) )                      &
           data%mkl_pardiso_IPARM( 5 ) = 1
         CALL MKL_PARDISO_SOLVE(                                               &
                       data%mkl_pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,  &
                       11_ip_, data%matrix%n, data%matrix%VAL( 1 : data%ne ),  &
                       data%matrix%PTR( 1 : data%matrix%n + 1 ),               &
                       data%matrix%COL( 1 : data%ne ),                         &
                       data%ORDER( 1 : data%matrix%n ), 1_ip_,                 &
                       data%mkl_pardiso_IPARM, control%print_level_solver,     &
                       data%ddum, data%ddum, inform%pardiso_error )
         inform%mkl_pardiso_IPARM = data%mkl_pardiso_IPARM

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
           inform%entries_in_factors = INT( data%pardiso_iparm( 18 ), long_ )

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
         CALL WSMP_INITIALIZE( )
         data%wsmp_iparm( 1 ) = 0
         data%wsmp_iparm( 2 ) = 0
         data%wsmp_iparm( 3 ) = 0
         CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),  &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%matrix%VAL( 1 : data%ne ),                           &
                     data%DIAG( 0 : 0 ),                                       &
                     data%ORDER( 1 : data%matrix%n ),                          &
                     data%INVP( 1 : data%matrix%n ),                           &
                     data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n,       &
                     1_ip_, data%wsmp_AUX, 0_ip_,                              &
                     data%MRP( 1 : data%matrix%n ),                            &
                     data%wsmp_IPARM, data%wsmp_DPARM )
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

         CALL SLS_copy_control_to_wsmp( control, data%wsmp_IPARM,              &
                                        data%wsmp_DPARM )
!$       CALL WSETMAXTHRDS( OMP_GET_NUM_THREADS( ) )

         data%wsmp_iparm( 2 ) = 1
         data%wsmp_iparm( 3 ) = 2
         CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),  &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%matrix%VAL( 1 : data%ne ),                           &
                     data%DIAG( 0 : 0 ),                                       &
                     data%ORDER( 1 : data%matrix%n ),                          &
                     data%INVP( 1 : data%matrix%n ),                           &
                     data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n,       &
                     1_ip_, data%wsmp_AUX, 0_ip_,                              &
                     data%MRP( 1 : data%matrix%n ),                            &
                     data%wsmp_iparm, data%wsmp_DPARM )
         inform%wsmp_IPARM = data%wsmp_IPARM
         inform%wsmp_DPARM = data%wsmp_DPARM
         inform%wsmp_error = data%wsmp_IPARM( 64 )

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

         IF ( data%wsmp_IPARM( 23 ) >= 0 ) inform%real_size_factors            &
                = 1000 * INT( data%wsmp_IPARM( 23 ), long_ )
         IF ( data%wsmp_IPARM( 24 ) >= 0 ) inform%entries_in_factors           &
                = 1000 * INT( data%wsmp_IPARM( 24 ), long_ )

!  = PaStiX =

       CASE ( 'pastix' )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         data%spm%baseval = 1
         data%spm%mtxtype = SpmSymmetric
#ifdef REAL_32
         data%spm%flttype = SpmFloat
#elif REAL_128
         data%spm%flttype = SpmDouble  !!! this is undoubtedly wrong!
#else
         data%spm%flttype = SpmDouble
#endif
         data%spm%fmttype = SpmCSC
         data%spm%n = data%matrix%n
         data%spm%nnz = data%ne
         data%spm%dof = 1
         CALL spmUpdateComputedFields( data%spm )
         CALL spmAlloc( data%spm )
#ifdef REAL_32
         CALL spmGetArray( data%spm, colptr = data%PTR,                        &
                           rowptr = data%ROW, svalues = data%VAL )
#elif REAL_128
         CALL spmGetArray( data%spm, colptr = data%PTR,                        &
                           rowptr = data%ROW, qvalues = data%VAL )
#else
         CALL spmGetArray( data%spm, colptr = data%PTR,                        &
                           rowptr = data%ROW, dvalues = data%VAL )
#endif

!  set the matrix

         data%PTR( 1 : data%matrix%n + 1 )                                     &
           = data%matrix%PTR( 1 : data%matrix%n + 1 )
         data%ROW( 1 : data%ne ) = data%matrix%COL( 1 : data%ne )

         CALL pastix_task_analyze( data%pastix_data, data%spm, pastix_info )
         inform%pastix_info = INT( pastix_info )
         IF ( pastix_info == PASTIX_SUCCESS ) THEN
           inform%status = GALAHAD_ok
         ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( pastix_info == PASTIX_ERR_OUTOFMEMORY ) THEN
           inform%status = GALAHAD_error_allocate
         ELSE IF ( pastix_info < 0 ) THEN
           inform%status = GALAHAD_error_unknown_solver
         ELSE
           inform%status = GALAHAD_error_pastix
         END IF
       END SELECT

!  = MUMPS =

     CASE ( 'mumps' )

       IF ( data%mumps_par%MYID == 0 ) THEN
         data%mumps_par%N = matrix%n
         data%mumps_par%NNZ = data%matrix_ne
         CALL SPACE_resize_pointer( data%matrix_ne, data%mumps_par%IRN,        &
                                    inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mumps_par%IRN' ; GO TO 900 ; END IF
         CALL SPACE_resize_pointer( data%matrix_ne, data%mumps_par%JCN,        &
                                    inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mumps_par%JCN' ; GO TO 900 ; END IF
         CALL SPACE_resize_pointer( data%matrix_ne, data%mumps_par%A,          &
                                    inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mumps_par%A' ; GO TO 900 ; END IF
         IF ( mc6168_ordering .OR. PRESENT( PERM ) ) THEN
           data%mumps_par%ICNTL( 7 ) = 1
           CALL SPACE_resize_pointer( matrix%n, data%mumps_par%PERM_IN,        &
                                      inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'sls: data%mumps_par%PERM_IN' ; GO TO 900
           END IF
         ELSE
           data%mumps_par%ICNTL( 7 ) = 0
         END IF

!  copy the matrix into the relevant components of the MUMPS derived type
!  on the host

         SELECT CASE ( SMT_get( matrix%type ) )
         CASE ( 'COORDINATE' )
           data%mumps_par%IRN( 1 : matrix%ne ) = matrix%ROW( 1 : matrix%ne )
           data%mumps_par%JCN( 1 : matrix%ne ) = matrix%COL( 1 : matrix%ne )
         CASE ( 'SPARSE_BY_ROWS' )
           DO i = 1, matrix%n
             DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
               data%mumps_par%IRN( l ) = i
               data%mumps_par%JCN( l ) = matrix%COL( l )
             END DO
           END DO
         CASE ( 'DENSE' )
           l = 0
           DO i = 1, matrix%n
             DO j = 1, i
               l = l + 1
                data%mumps_par%IRN( l ) = i ; data%mumps_par%JCN( l ) = j
             END DO
           END DO
         END SELECT

!  copy non-default order into the relevant components of the MUMPS derived type

         IF ( mc6168_ordering ) THEN
           data%mumps_par%PERM_IN( 1 : matrix%n ) = data%ORDER( 1 : matrix%n )
         ELSE IF ( PRESENT( PERM ) ) THEN
           data%mumps_par%PERM_IN( 1 : matrix%n ) = PERM( 1 : matrix%n )
         END IF
       END IF

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL SLS_copy_control_to_mumps( control, data%mumps_par%ICNTL,          &
                                       data%mumps_par%CNTL )

       data%mumps_par%JOB = 1
       CALL MUMPS_precision( data%mumps_par )
       inform%mumps_info = data%mumps_par%INFOG
       inform%mumps_rinfo = data%mumps_par%RINFOG
       inform%mumps_error = data%mumps_par%INFOG( 1 )
!write(6,"( ' analysis status = ', I0 )" ) inform%mumps_error

       IF ( inform%mumps_error < 0 .AND. control%print_level > 0 .AND.         &
              control%out > 0 ) WRITE( control%out,                            &
            "( A, ' mumps error code = ', I0 )" ) prefix, inform%mumps_error

       inform%rank = matrix%n
       SELECT CASE( inform%mumps_error )
       CASE ( - 6, 0 : )
         inform%status = GALAHAD_ok
         IF ( data%mumps_par%INFOG( 3 ) >= 0 ) THEN
           inform%real_size_desirable                                          &
             = INT( data%mumps_par%INFOG( 3 ), long_ )
         ELSE
           inform%real_size_desirable                                          &
             = INT( - 1000000 * data%mumps_par%INFOG( 3 ), long_ )
         END IF

         IF ( data%mumps_par%INFOG( 4 ) >= 0 ) THEN
           inform%integer_size_desirable                                       &
             = INT( data%mumps_par%INFOG( 4 ), long_ )
         ELSE
           inform%integer_size_desirable                                       &
             = INT( - 1000000 * data%mumps_par%INFOG( 4 ), long_ )
         END IF
         inform%max_front_size =  data%mumps_par%INFOG( 5 )
         IF ( inform%mumps_error == - 6 )                                      &
           inform%rank = data%mumps_par%INFOG( 2 )
         inform%flops_elimination = INT( data%mumps_par%RINFOG( 1 ), long_ )
       CASE ( - 2, - 16 )
          inform%status = GALAHAD_error_restrictions
       CASE ( - 4 )
         inform%status = GALAHAD_error_permutation
       CASE ( - 5, - 7, - 13, - 22 )
         inform%status = GALAHAD_error_allocate
       CASE ( - 8, - 14, - 15 )
          inform%status = GALAHAD_error_integer_ws
       CASE ( - 9, - 11 )
          inform%status = GALAHAD_error_real_ws
       CASE DEFAULT
         inform%status = GALAHAD_error_mumps
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
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       IF ( data%solver( 1 : data%len_solver ) == 'sytr' ) THEN
         CALL SPACE_resize_array( data%n, data%PIVOTS,                         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%sytr_lwork =                                                     &
           data%n * LAENV( 1_ip_, 'DSYTRF', 'L', data%n,                    &
                           - 1_ip_, - 1_ip_, - 1_ip_ )
         CALL SPACE_resize_array( data%sytr_lwork, data%WORK,                  &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
       END IF
       inform%entries_in_factors = matrix%n * ( matrix%n + 1 ) / 2

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
       CALL SPACE_resize_array( inform%semi_bandwidth + 1_ip_, data%n,         &
                                data%matrix_dense,                             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix_dense' ; GO TO 900 ; END IF

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       inform%entries_in_factors = matrix%n * inform%semi_bandwidth

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

     INTEGER ( KIND = ip_ ) :: i, ii, j, jj, k, l, l1, l2, job, matrix_type
     INTEGER( ipc_ ) :: pastix_info
     REAL :: time, time_start, time_now
     REAL ( KIND = rp_ ) :: clock, clock_start, clock_now
     REAL ( KIND = rp_ ) :: val
     LOGICAL :: filexx, must_be_definite
     CHARACTER ( LEN = 400 ), DIMENSION( 1 ) :: path
     CHARACTER ( LEN = 400 ), DIMENSION( 4 ) :: filename
!    CHARACTER :: dumc( 20 )
!    REAL ( KIND = rp_ ) :: eigenvalues( 25 )

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases, no factorization

     SELECT CASE ( SMT_get( matrix%type ) )
     CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY', 'ZERO', 'NONE' )
       inform%real_size_necessary = 0
       inform%integer_size_necessary = 0
       inform%real_size_factors  = 0
       inform%integer_size_factors = 0
       inform%entries_in_factors = matrix%n
       inform%two_by_two_pivots = 0
       inform%semi_bandwidth = 0
       inform%largest_modified_pivot = 0.0_rp_
       inform%solver = 'none' // REPEAT( ' ', len_solver - 4 )

!  store the matrix in work

       CALL SPACE_resize_array( matrix%n, data%WORK,                           &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%WORK' ; GO TO 900 ; END IF

       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'DIAGONAL' )
         inform%rank                                                           &
           = COUNT( matrix%val( : matrix%n ) /= 0.0_rp_ )
         inform%negative_eigenvalues                                           &
           = COUNT( matrix%val( : matrix%n ) < 0.0_rp_ )
          data%WORK( : matrix%n ) = matrix%val( : matrix%n )
       CASE ( 'SCALED_IDENTITY' )
         IF ( matrix%val( 1 ) > 0.0_rp_ ) THEN
           inform%rank = matrix%n
           inform%negative_eigenvalues = 0
         ELSE IF ( matrix%val( 1 ) < 0.0_rp_ ) THEN
           inform%rank = matrix%n
           inform%negative_eigenvalues = matrix%n
         ELSE
           inform%rank = 0
           inform%negative_eigenvalues = 0
         END IF
         data%WORK( : matrix%n ) = matrix%val( 1 )
       CASE ( 'IDENTITY' )
         inform%rank = matrix%n
         inform%negative_eigenvalues = 0
         data%WORK( : matrix%n ) = 1.0_rp_
       CASE ( 'ZERO', 'NONE' )
         inform%rank = 0
         inform%negative_eigenvalues = 0
         data%WORK( : matrix%n ) = 0.0_rp_
       END SELECT
       inform%status = GALAHAD_ok
       GO TO 900
     END SELECT

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
         CALL SPACE_resize_array( matrix%n + 1_ip_, data%matrix_scale%PTR,     &
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
         data%matrix_scale%VAL( data%matrix_scale%PTR( i ) ) = 0.0_rp_
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
         CALL SPACE_resize_array( 1_ip_, data%matrix_scale%id,                 &
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
         CALL MC64_MATCHING( 5_ip_, data%matrix_scale, data%mc64_control,      &
                             inform%mc64_info, data%mc64_PERM,                 &
                             data%SCALE )
         IF ( inform%mc64_info%flag /= 0 .AND.                                 &
              inform%mc64_info%flag /= - 9 ) THEN
           inform%status = GALAHAD_error_mc64 ; GO TO 900
         END IF
         data%SCALE( : data%matrix_scale%n ) =                                 &
           EXP( data%SCALE( : data%matrix_scale%n ) )

!!  scaling using MC77 based on the row one- or infinity-norm

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
         CALL MC77A( job, data%matrix_scale%n, data%matrix_scale%n,            &
                     data%matrix_scale%ne, data%matrix_scale%PTR,              &
!                    data%matrix_scale%COL, data%matrix_scale%VAL,             &
                     data%matrix_scale%ROW, data%matrix_scale%VAL,             &
                     data%mc77_IW, data%mc77_liw, data%SCALE,                  &
                     data%mc77_ldw, data%mc77_ICNTL, data%mc77_CNTL,           &
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
           data%matrix%VAL( l ) = matrix%VAL( l ) /                            &
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
           inform%entries_in_factors = INT( data%sils_finfo%nebdu, long_ )
           inform%real_size_factors = INT( data%sils_finfo%nrlbdu, long_ )
           inform%integer_size_factors = INT( data%sils_finfo%nirbdu, long_ )
           inform%real_size_desirable = INT( data%sils_finfo%nrltot, long_ )
           inform%integer_size_desirable = INT( data%sils_finfo%nirtot, long_ )
           inform%real_size_necessary = INT( data%sils_finfo%nrlnec, long_ )
           inform%integer_size_necessary = INT( data%sils_finfo%nirnec, long_ )
           inform%compresses_real = data%sils_finfo%ncmpbr
           inform%compresses_integer = data%sils_finfo%ncmpbi
           inform%rank = data%sils_finfo%rank
           inform%two_by_two_pivots = data%sils_finfo%ntwo
           inform%negative_eigenvalues = data%sils_finfo%neig
           inform%delayed_pivots = data%sils_finfo%delay
           inform%pivot_sign_changes = data%sils_finfo%signc
           inform%static_pivots = data%sils_finfo%static
           inform%first_modified_pivot = data%sils_finfo%modstep
           inform%flops_assembly = INT( data%sils_finfo%opsa, long_ )
           inform%flops_elimination = INT( data%sils_finfo%opse, long_ )
           inform%flops_blas = INT( data%sils_finfo%opsb, long_ )
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
           inform%entries_in_factors = INT( data%ma57_finfo%nebdu, long_ )
           inform%real_size_factors = INT( data%ma57_finfo%nrlbdu, long_ )
           inform%integer_size_factors = INT( data%ma57_finfo%nirbdu, long_ )
           inform%real_size_desirable = INT( data%ma57_finfo%nrltot, long_ )
           inform%integer_size_desirable = INT( data%ma57_finfo%nirtot, long_ )
           inform%real_size_necessary  = INT( data%ma57_finfo%nrlnec, long_ )
           inform%integer_size_necessary  = INT( data%ma57_finfo%nirnec, long_ )
           inform%compresses_real = data%ma57_finfo%ncmpbr
           inform%compresses_integer = data%ma57_finfo%ncmpbi
           inform%rank = data%ma57_finfo%rank
           inform%two_by_two_pivots = data%ma57_finfo%ntwo
           inform%negative_eigenvalues  = data%ma57_finfo%neig
           inform%delayed_pivots = data%ma57_finfo%delay
           inform%pivot_sign_changes = data%ma57_finfo%signc
           inform%static_pivots = data%ma57_finfo%static
           inform%first_modified_pivot = data%ma57_finfo%modstep
           inform%flops_assembly = INT( data%ma57_finfo%opsa, long_ )
           inform%flops_elimination = INT( data%ma57_finfo%opse, long_ )
           inform%flops_blas = INT( data%ma57_finfo%opsb, long_ )
           IF ( inform%first_modified_pivot > 0 ) THEN
             inform%largest_modified_pivot = data%ma57_finfo%maxchange
           ELSE
             inform%largest_modified_pivot = 0.0_rp_
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
         CALL MA77_input_reals( i, l2 - l1 + 1_ip_,                            &
                               data%matrix%VAL( l1 : l2 ), data%ma77_keep,     &
                               data%ma77_control, data%ma77_info )
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

     CASE ( 'ma86', 'ma87', 'ma97', 'ssids', 'pardiso', 'mkl_pardiso',         &
            'wsmp', 'pastix' )
       data%matrix%n = matrix%n
       DO i = 1, matrix%n
         l = data%matrix%PTR( i )
         data%matrix%VAL( l ) = 0.0_rp_
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
         must_be_definite = data%must_be_definite
         IF ( data%ssids_options%scaling == 0 ) THEN
           CALL SSIDS_factor( must_be_definite, data%matrix%VAL,               &
                              data%ssids_akeep, data%ssids_fkeep,              &
                              data%ssids_options, data%ssids_inform,           &
                              ptr = data%matrix%PTR, row = data%matrix%COL )
         ELSE
           CALL SPACE_resize_array( data%n, data%SCALE,                        &
                                    inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'sls: data%matrix%VAL' ; GO TO 800 ; END IF
           CALL SSIDS_factor( must_be_definite, data%matrix%VAL,               &
                              data%ssids_akeep, data%ssids_fkeep,              &
                              data%ssids_options, data%ssids_inform,           &
                              scale = data%SCALE,                              &
                              ptr = data%matrix%PTR, row = data%matrix%COL )
         END IF
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

       CASE ( 'pardiso' )
         CALL SLS_copy_control_to_pardiso( control, data%pardiso_IPARM )
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL PARDISO( data%pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,      &
                       22_ip_, data%matrix%n, data%matrix%VAL( : data%ne ),    &
                       data%matrix%PTR( : data%matrix%n + 1 ),                 &
                       data%matrix%COL( : data%ne ),                           &
                       data%ORDER( : data%matrix%n ), 1_ip_,                   &
                       data%pardiso_IPARM( : 64 ),                             &
                       control%print_level_solver, data%B2( : matrix%n, : 1 ), &
                       data%X2( : matrix%n, : 1 ), inform%pardiso_error,       &
                       data%pardiso_DPARM( 1 : 64 ) )

         inform%pardiso_IPARM = data%pardiso_IPARM
         inform%pardiso_DPARM = data%pardiso_DPARM

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
         IF ( data%pardiso_IPARM( 18 ) > 0 )                                   &
           inform%entries_in_factors = INT( data%pardiso_IPARM( 18 ), long_ )
         inform%negative_eigenvalues = data%pardiso_IPARM( 23 )
         inform%rank = data%pardiso_IPARM( 22 ) + data%pardiso_IPARM( 23 )
         IF ( data%must_be_definite .AND. inform%negative_eigenvalues > 0 )    &
           inform%status = GALAHAD_error_inertia

!  = MKL PARDISO =

       CASE ( 'mkl_pardiso' )
         data%mkl_pardiso_IPARM = inform%mkl_pardiso_IPARM
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL MKL_PARDISO_SOLVE(                                               &
                       data%mkl_pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,  &
                       22_ip_, data%matrix%n, data%matrix%VAL( 1 : data%ne ),  &
                       data%matrix%PTR( 1 : data%matrix%n + 1 ),               &
                       data%matrix%COL( 1 : data%ne ),                         &
                       data%ORDER( 1 : data%matrix%n ), 1_ip_,                 &
                       data%mkl_pardiso_IPARM, control%print_level_solver,     &
                       data%ddum, data%ddum, inform%pardiso_error )
         inform%mkl_pardiso_IPARM = data%mkl_pardiso_IPARM

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
           inform%entries_in_factors = INT( data%pardiso_iparm( 18 ), long_ )
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
         CALL SLS_copy_control_to_wsmp( control, data%wsmp_IPARM,              &
                                        data%wsmp_DPARM )
         data%wsmp_IPARM( 2 ) = 3
         data%wsmp_IPARM( 3 ) = 3
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),  &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%matrix%VAL( 1 : data%ne ),                           &
                     data%DIAG( 0 : 0 ),                                       &
                     data%ORDER( 1 : data%matrix%n ),                          &
                     data%INVP( 1 : data%matrix%n ),                           &
                     data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n,       &
                     1_ip_, data%wsmp_AUX, 0_ip_,                              &
                     data%MRP( 1 : data%matrix%n ),                            &
                     data%wsmp_IPARM, data%wsmp_DPARM )
         inform%wsmp_IPARM = data%wsmp_IPARM
         inform%wsmp_DPARM = data%wsmp_DPARM
         inform%wsmp_error = data%wsmp_IPARM( 64 )
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

         IF ( data%wsmp_IPARM( 23 ) > 0 )                                      &
           inform%real_size_factors = 1000 * INT( data%wsmp_IPARM( 23 ), long_ )
         IF ( data%wsmp_IPARM( 24 ) > 0 )                                      &
           inform%entries_in_factors = 1000 * INT( data%wsmp_IPARM( 24 ), long_)
         inform%negative_eigenvalues = data%wsmp_IPARM( 22 )
         inform%rank = data%matrix%n - data%wsmp_IPARM( 21 )
         IF ( data%must_be_definite .AND. inform%negative_eigenvalues > 0 )    &
           inform%status = GALAHAD_error_inertia
         IF ( data%must_be_definite .AND.inform%wsmp_error > 0 )               &
           inform%status = GALAHAD_error_inertia

!  = PaStiX =

       CASE ( 'pastix' )
         data%VAL( 1 : data%ne ) = data%matrix%VAL( 1 : data%ne  )

         CALL pastix_task_numfact( data%pastix_data, data%spm, pastix_info )

         inform%pastix_info = INT( pastix_info )
         IF ( pastix_info == PASTIX_SUCCESS ) THEN
           inform%status = GALAHAD_ok
         ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
           inform%status = GALAHAD_error_restrictions
         ELSE IF ( pastix_info == PASTIX_ERR_OUTOFMEMORY ) THEN
           inform%status = GALAHAD_error_allocate
         ELSE
           inform%status = GALAHAD_error_pastix
         END IF
       END SELECT

!  = MUMPS =

     CASE ( 'mumps' )
       CALL SLS_copy_control_to_mumps( control, data%mumps_par%ICNTL,          &
                                       data%mumps_par%CNTL )

       IF ( data%mumps_par%MYID == 0 ) THEN
         IF ( data%explicit_scaling ) THEN
           DO l = 1, data%matrix_ne
             i = data%mumps_par%IRN( l ) ; j = data%mumps_par%JCN( l )
             data%mumps_par%A( l ) = matrix%VAL( l ) /                         &
               ( data%SCALE( i ) * data%SCALE( j ) )
           END DO
         ELSE
           data%mumps_par%A( : data%matrix_ne ) = matrix%VAL( : data%matrix_ne )
         END IF
!write(6,"( ' n, nnz = ', 2I9 )" ) data%mumps_par%N, data%mumps_par%NNZ
!write(6,"( ' R = ', 8I9 )" ) data%mumps_par%IRN( : data%matrix_ne )
!write(6,"( ' C = ', 8I9 )" ) data%mumps_par%JCN( : data%matrix_ne )
!write(6,"( ' V = ', 8ES9.1 )" ) data%mumps_par%A( : data%matrix_ne )
       END IF
!write(6, "( ' icntl ', /, ( 10I6 ) )" ) data%mumps_par%ICNTL
!write(6, "( ' cntl ', /, ( 5ES10.2 ) )" ) data%mumps_par%CNTL
       data%mumps_par%JOB = 2
       CALL MUMPS_precision( data%mumps_par )
       inform%mumps_info = data%mumps_par%INFOG
       inform%mumps_rinfo = data%mumps_par%RINFOG
       inform%mumps_error = data%mumps_par%INFOG( 1 )
!write(6,"( ' factorize status = ', I0 )" ) inform%mumps_error

       IF ( inform%mumps_error < 0 .AND. control%print_level > 0 .AND.         &
              control%out > 0 ) WRITE( control%out,                            &
            "( A, ' mumps error code = ', I0 )" ) prefix, inform%mumps_error

       inform%rank = matrix%n
       SELECT CASE( inform%mumps_error )
       CASE ( - 6, 0 : )
         inform%status = GALAHAD_ok
         IF ( data%mumps_par%INFOG( 9 ) >= 0 ) THEN
           inform%real_size_factors                                            &
             = INT( data%mumps_par%INFOG( 9 ), long_ )
         ELSE
           inform%real_size_factors                                            &
             = INT( - 1000000 * data%mumps_par%INFOG( 9 ), long_ )
         END IF
         IF ( data%mumps_par%INFOG( 10 ) >= 0 ) THEN
           inform%integer_size_factors                                         &
             = INT( data%mumps_par%INFOG( 10 ), long_ )
         ELSE
           inform%integer_size_factors                                         &
             = INT( - 1000000 * data%mumps_par%INFOG( 10 ), long_ )
         END IF
         inform%integer_size_desirable = inform%integer_size_factors
         inform%real_size_necessary = inform%real_size_factors
         IF ( data%mumps_par%INFOG( 29 ) >= 0 ) THEN
           inform%entries_in_factors                                           &
            = INT( data%mumps_par%INFOG( 29 ), long_ )
         ELSE
           inform%entries_in_factors                                           &
             = INT( - 1000000 * data%mumps_par%INFOG( 29 ), long_ )
         END IF
         inform%max_front_size = data%mumps_par%INFOG( 11 )
         IF ( inform%mumps_error == - 6 )                                      &
           inform%rank = data%mumps_par%INFOG( 2 )
         inform%negative_eigenvalues = data%mumps_par%INFOG( 12 )
         inform%rank =  data%matrix%n - data%mumps_par%INFOG( 28 )
         inform%delayed_pivots = data%mumps_par%INFOG( 13 )
         inform%compresses_real = data%mumps_par%INFOG( 14 )
         inform%static_pivots = data%mumps_par%INFOG( 25 )
         inform%flops_assembly = INT( data%mumps_par%RINFOG( 2 ), long_ )
         inform%flops_elimination = INT( data%mumps_par%RINFOG( 3 ), long_ )
         IF ( data%must_be_definite .AND. inform%negative_eigenvalues > 0 )    &
           inform%status = GALAHAD_error_inertia
         IF ( data%must_be_definite .AND.inform%wsmp_error > 0 )               &
           inform%status = GALAHAD_error_inertia
       CASE ( - 2, - 16 )
          inform%status = GALAHAD_error_restrictions
       CASE ( - 4 )
         inform%status = GALAHAD_error_permutation
       CASE ( - 5, - 7, - 13, - 22 )
         inform%status = GALAHAD_error_allocate
       CASE ( - 8, - 14, - 15 )
          inform%status = GALAHAD_error_integer_ws
       CASE ( - 9, - 11 )
          inform%status = GALAHAD_error_real_ws
       CASE DEFAULT
         inform%status = GALAHAD_error_mumps
       END SELECT

!  = LAPACK solvers POTR or SYTR =

     CASE ( 'potr', 'sytr' )

       data%matrix_dense( : matrix%n, : matrix%n ) = 0.0_rp_
       data%matrix%type =  matrix%type

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
             INT( data%n * ( data%n + 1 ) / 2, long_ )
           inform%flops_elimination = INT( data%n ** 3 / 6, long_ )
         CASE ( 1 : )
           inform%status = GALAHAD_error_inertia
         CASE DEFAULT
           inform%status = GALAHAD_error_lapack
         END SELECT

!  = SYTR =

       CASE ( 'sytr' )
!        CALL SYEV( 'N', 'L',  data%n, data%matrix_dense, data%n, &
!                     eigenvalues, data%WORK, data%sytr_lwork,                 &
!                     inform%lapack_error )
!         WRITE( 6, "( 'eigenvalues', /, (5ES12.4 ) )" ) eigenvalues( : data%n )
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
               IF ( data%matrix_dense( k, k ) < 0.0_rp_ ) THEN
                 inform%negative_eigenvalues =                                 &
                   inform%negative_eigenvalues + 1
               ELSE IF ( data%matrix_dense( k, k ) == 0.0_rp_ ) THEN
                 inform%rank = inform%rank - 1
               END IF
               k = k + 1
             ELSE                               ! a 2 x 2 pivot
               inform%two_by_two_pivots = inform%two_by_two_pivots + 1
               IF ( data%matrix_dense( k, k ) *                                &
                    data%matrix_dense( k + 1, k + 1 ) <                        &
                    data%matrix_dense( k + 1, k ) ** 2 ) THEN
                 inform%negative_eigenvalues = inform%negative_eigenvalues + 1
               ELSE IF ( data%matrix_dense( k, k ) < 0.0_rp_ ) THEN
                 inform%negative_eigenvalues = inform%negative_eigenvalues + 2
               END IF
               k = k + 2
             END IF
           END DO
           inform%entries_in_factors =                                         &
             INT( data%n * ( data%n + 1 ) / 2, long_ )
           inform%flops_elimination = INT( data%n ** 3 / 3, long_ )

           CALL SPACE_resize_array( data%n, data%INVP,                         &
                                    inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'sls: data%INVP' ; GO TO 900 ; END IF
           DO i = 1, data%n
             data%INVP( data%ORDER( i ) ) = i
           END DO
         CASE DEFAULT
           inform%status = GALAHAD_error_lapack
         END SELECT

       END SELECT

!  = PBTR =

     CASE ( 'pbtr' )

       data%matrix_dense( : inform%semi_bandwidth + 1, : matrix%n ) = 0.0_rp_
       data%matrix%type = matrix%type

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
                   inform%semi_bandwidth + 1_ip_, inform%lapack_error )

       IF ( inform%lapack_error < 0 .AND. control%print_level > 0 .AND.        &
            control%out > 0 ) WRITE( control%out, "( A,                        &
      &  ' LAPACK PBTRF error code = ', I0 )" ) prefix, inform%lapack_error

       SELECT CASE( inform%lapack_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%two_by_two_pivots = 0
         inform%rank = data%n
         inform%negative_eigenvalues = 0
         inform%entries_in_factors =                                           &
           INT( ( ( 2 * data%n - inform%semi_bandwidth ) *                     &
                  ( inform%semi_bandwidth + 1 ) ) / 2, long_ )
         inform%flops_elimination =                                            &
           INT( 2 * data%n * ( inform%semi_bandwidth + 1 ) ** 2, long_ )
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
     REAL ( KIND = rp_ ), INTENT( INOUT ) , DIMENSION ( : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER ( KIND = ip_ ) :: i, j, l, iter, n
     REAL :: time_start, time_now
     REAL ( KIND = rp_ ) :: clock_start, clock_now
     REAL ( KIND = rp_ ) :: residual, residual_zero, val
     LOGICAL :: filexx

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases

     SELECT CASE ( SMT_get( matrix%type ) )
     CASE ( 'DIAGONAL' )
       IF ( inform%rank == matrix%n ) THEN
         X( : matrix%n ) = X( : matrix%n ) / matrix%val( : matrix%n )
         inform%status = GALAHAD_ok
       ELSE
         inform%status = GALAHAD_error_solve
       END IF
       GO TO 900
     CASE ( 'SCALED_IDENTITY' )
       IF ( inform%rank == matrix%n ) THEN
         X( : matrix%n ) = X( : matrix%n ) / matrix%val( 1 )
         inform%status = GALAHAD_ok
       ELSE
         inform%status = GALAHAD_error_solve
       END IF
       GO TO 900
     CASE ( 'IDENTITY' )
       inform%status = GALAHAD_ok
!      X( : matrix%n ) = X( : matrix%n )
       GO TO 900
     CASE ( 'ZERO', 'NONE' )
       inform%status = GALAHAD_error_solve
       GO TO 900
     END SELECT

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
          data%solver( 1 : data%len_solver ) == 'pardiso' .OR.                 &
          data%solver( 1 : data%len_solver ) == 'mkl_pardiso' ) THEN

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
       X( : n ) = 0.0_rp_
       residual_zero = MAXVAL( ABS( data%B( : n ) ) )

!  Solve the system with iterative refinement

       IF ( control%print_level > 1 .AND. control%out > 0 )                    &
         WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )    &
           prefix, residual_zero, 0.0_rp_

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
             IF ( data%solver( 1 : data%len_solver ) == 'mumps' ) THEN
               DO l = 1, data%matrix_ne
                 i = data%mumps_par%IRN( l ) ; j = data%mumps_par%JCN( l )
                 IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                   val = data%mumps_par%A( l )
                   data%RES( i ) = data%RES( i ) - val * X( j )
                   IF ( i /= j ) data%RES( j ) = data%RES( j ) - val * X( i )
                 END IF
               END DO
             ELSE IF ( data%solver( 1 : data%len_solver ) /= 'potr' .AND.      &
                       data%solver( 1 : data%len_solver ) /= 'sytr' .AND.      &
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
      REAL ( KIND = rp_ ), INTENT( INOUT ) , DIMENSION ( : , : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

     INTEGER ( KIND = ip_ ) :: i, j, l, iter, n, nrhs
     REAL :: time_start, time_now
     REAL ( KIND = rp_ ) :: clock_start, clock_now
     REAL ( KIND = rp_ ) :: val
     LOGICAL :: too_big

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases

     nrhs = SIZE( X, 2 )
     SELECT CASE ( SMT_get( matrix%type ) )
     CASE ( 'DIAGONAL' )
       IF ( inform%rank == matrix%n ) THEN
         DO i = 1, nrhs
           X( : matrix%n, i ) = X( : matrix%n, i ) / matrix%val( : matrix%n )
         END DO
         inform%status = GALAHAD_ok
       ELSE
         inform%status = GALAHAD_error_solve
       END IF
       GO TO 900
     CASE ( 'SCALED_IDENTITY' )
       IF ( inform%rank == matrix%n ) THEN
         X( : matrix%n, : nrhs ) = X( : matrix%n, : nrhs ) / matrix%val( 1 )
         inform%status = GALAHAD_ok
       ELSE
         inform%status = GALAHAD_error_solve
       END IF
       GO TO 900
     CASE ( 'IDENTITY' )
       inform%status = GALAHAD_ok
!      X( : matrix%n, : nhs ) = X( : matrix%n, : nhs )
       GO TO 900
     CASE ( 'ZERO', 'NONE' )
       inform%status = GALAHAD_error_solve
       GO TO 900
     END SELECT

!  Set default inform values

     inform%bad_alloc = ''

!  No refinement is required (or Pardiso is called with its internal refinement)
!  -----------------------------------------------------------------------------

     IF ( control%max_iterative_refinements <= 0 .OR.                          &
          data%solver( 1 : data%len_solver ) == 'pardiso' .OR.                 &
          data%solver( 1 : data%len_solver ) == 'mkl_pardiso' ) THEN

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
       X( : n, : nrhs ) = 0.0_rp_

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
             IF ( data%solver( 1 : data%len_solver ) == 'mumps' ) THEN
               DO l = 1, data%matrix_ne
                 i = data%mumps_par%IRN( l ) ; j = data%mumps_par%JCN( l )
                 IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= n ) THEN
                   val = data%mumps_par%A( l )
                   data%RES2( i, : nrhs )                                      &
                     = data%RES2( i, : nrhs ) - val * X( j, : nrhs )
                   IF ( i /= j ) data%RES2( j, : nrhs )                        &
                     = data%RES2( j, : nrhs ) - val * X( i, : nrhs )
                 END IF
               END DO
             ELSE IF ( data%solver( 1 : data%len_solver ) /= 'potr' .AND.      &
                       data%solver( 1 : data%len_solver ) /= 'sytr' .AND.      &
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
     REAL ( KIND = rp_ ), INTENT( INOUT ) :: X( : )
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER( ipc_ ) :: pastix_info
     REAL :: time, time_now
     REAL ( KIND = rp_ ) :: clock, clock_now, tiny

!  trivial cases

     IF ( data%solver( 1 : data%len_solver ) /= 'mumps' ) THEN
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'DIAGONAL' )
         IF ( inform%rank == matrix%n ) THEN
           X( : matrix%n ) = X( : matrix%n ) / matrix%val( : matrix%n )
           inform%status = GALAHAD_ok
         ELSE
           inform%status = GALAHAD_error_solve
         END IF
         GO TO 900
       CASE ( 'SCALED_IDENTITY' )
         IF ( inform%rank == matrix%n ) THEN
           X( : matrix%n ) = X( : matrix%n ) / matrix%val( 1 )
           inform%status = GALAHAD_ok
         ELSE
           inform%status = GALAHAD_error_solve
         END IF
         GO TO 900
       CASE ( 'IDENTITY' )
         inform%status = GALAHAD_ok
  !      X( : matrix%n ) = X( : matrix%n )
         GO TO 900
       CASE ( 'ZERO', 'NONE' )
         inform%status = GALAHAD_error_solve
         GO TO 900
       END SELECT
     END IF

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
       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_ma77( control, data%ma77_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,              &
                          data%ma77_control, data%ma77_info,                   &
                          scale = data%SCALE )
       ELSE
         CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,              &
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
       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_pardiso( control, data%pardiso_IPARM )
       data%pardiso_IPARM( 6 ) = 1
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,        &
                     33_ip_, data%matrix%n, data%matrix%VAL( : data%ne ),      &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), 1_ip_,                     &
                     data%pardiso_IPARM( : 64 ), control%print_level_solver,   &
                     data%X2( : data%matrix%n, : 1 ),                          &
                     data%B2( : data%matrix%n, : 1 ), inform%pardiso_error,    &
                     data%pardiso_DPARM( 1 : 64 ) )
       inform%pardiso_IPARM = data%pardiso_IPARM
       inform%pardiso_DPARM = data%pardiso_DPARM

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
         inform%iterative_refinements = data%pardiso_IPARM( 7 )
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%X2( : data%n, 1 )

!  = MKL PARDISO =

     CASE ( 'mkl_pardiso' )
       CALL SPACE_resize_array( data%n, data%B1, inform%status,                &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B1' ; GO TO 900 ; END IF
       data%B1( : data%n ) = X( : data%n )

       data%mkl_pardiso_IPARM = inform%mkl_pardiso_IPARM
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MKL_PARDISO_SOLVE(                                                 &
                     data%mkl_pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,    &
                     33_ip_, data%matrix%n, data%matrix%VAL( 1 : data%ne ),    &
                     data%matrix%PTR( 1 : data%matrix%n + 1 ),                 &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%ORDER( 1 : data%matrix%n ), 1_ip_,                   &
                     data%mkl_pardiso_IPARM, control%print_level_solver,       &
                     data%B1( 1 : matrix%n ), X( 1 : matrix%n ),               &
                     inform%pardiso_error )
       inform%mkl_pardiso_IPARM = data%mkl_pardiso_IPARM

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

       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_wsmp( control, data%wsmp_IPARM,                &
                                      data%wsmp_DPARM )
       data%wsmp_iparm( 2 ) = 4
       data%wsmp_iparm( 3 ) = 5
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1_ip_,  &
                   data%wsmp_AUX, 0_ip_, data%MRP( 1 : data%matrix%n ),        &
                   data%wsmp_IPARM, data%wsmp_DPARM )
!write(6,*) ' solv threads used = ', data%wsmp_iparm( 33 )
       inform%wsmp_IPARM = data%wsmp_IPARM
       inform%wsmp_DPARM = data%wsmp_DPARM
       inform%wsmp_error = data%wsmp_IPARM( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_iparm( 6 )
         X( : data%n ) = data%B2( : data%n, 1 )
       CASE ( - 102 )
         inform%status = GALAHAD_error_allocate
       CASE DEFAULT
         inform%status = GALAHAD_error_wsmp
       END SELECT

!  = PaStiX =

     CASE ( 'pastix' )

       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       data%X2( : data%n, 1 ) = X( : data%n )
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL pastix_task_solve( data%pastix_data, data%n, 1_ip_, data%X2,       &
                               data%spm%nexp, pastix_info )

       inform%pastix_info = INT( pastix_info )
       IF ( pastix_info == PASTIX_SUCCESS ) THEN
         inform%status = GALAHAD_ok
       ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
         inform%status = GALAHAD_error_restrictions ; GO TO 900
       ELSE
         inform%status = GALAHAD_error_pastix ; GO TO 900
       END IF

       CALL pastix_task_refine( data%pastix_data, data%spm%nexp, 1_ip_,        &
                                data%B2, data%spm%nexp, data%X2,               &
                                data%spm%nexp, pastix_info )

       inform%pastix_info = INT( pastix_info )
       IF ( pastix_info == PASTIX_SUCCESS ) THEN
         X( : data%n ) = data%X2( : data%n, 1 )
         inform%status = GALAHAD_ok
       ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
         inform%status = GALAHAD_error_restrictions ; GO TO 900
       ELSE
         inform%status = GALAHAD_error_pastix ; GO TO 900
       END IF

!  = MUMPS =

     CASE ( 'mumps' )

       IF ( data%mumps_par%MYID == 0 ) THEN
         data%mumps_par%NRHS = 1
         data%mumps_par%LRHS = data%n
         CALL SPACE_resize_pointer( data%n, data%mumps_par%RHS,                &
                                    inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mumps_par%RHS' ; GO TO 900 ; END IF
         data%mumps_par%RHS( : data%n ) = X( : data%n )
!write(6,"('rhs', 5ES10.2 )" ) data%mumps_par%RHS( : data%n )
       END IF
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )

       data%mumps_par%JOB = 3
       CALL MUMPS_precision( data%mumps_par )
       inform%mumps_info = data%mumps_par%INFOG
       inform%mumps_rinfo = data%mumps_par%RINFOG
       inform%mumps_error = data%mumps_par%INFOG( 1 )

       SELECT CASE( inform%mumps_error )
       CASE ( 0 : )
         inform%status = GALAHAD_ok
         X( : data%n ) = data%mumps_par%RHS( : data%n )
!write(6,"('x', 5ES10.2 )" ) data%mumps_par%RHS( : data%n )
         inform%iterative_refinements = data%mumps_par%INFOG( 15 )
       CASE DEFAULT
         inform%status = GALAHAD_error_mumps ; GO TO 900
       END SELECT

!  = LAPACK solvers POTR, SYTR or PBTR =

     CASE ( 'potr', 'sytr', 'pbtr' )

       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( data%ORDER( : data%n ), 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = POTR =

       CASE ( 'potr' )
         CALL POTRS( 'L', data%n, 1_ip_, data%matrix_dense, data%n, data%X2,   &
                     data%n, inform%lapack_error )

!  = SYTR =

       CASE ( 'sytr' )
         IF ( inform%rank == data%n ) THEN
           CALL SYTRS( 'L', data%n, 1_ip_, data%matrix_dense, data%n,          &
                       data%PIVOTS, data%X2, data%n, inform%lapack_error )
         ELSE
           tiny = 100.0_rp_ * control%absolute_pivot_tolerance
           CALL SLS_sytr_singular_solve( data%n, 1_ip_, data%matrix_dense,     &
                                         data%n, data%PIVOTS, data%X2,         &
                                         data%n, tiny, inform%lapack_error )
           END IF

!  = PBTR =

       CASE ( 'pbtr' )
         CALL PBTRS( 'L', data%n, inform%semi_bandwidth, 1_ip_,                &
                      data%matrix_dense, inform%semi_bandwidth + 1_ip_,        &
                      data%X2, data%n, inform%lapack_error )
       END SELECT

       SELECT CASE( inform%lapack_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
       CASE ( GALAHAD_error_primal_infeasible )
         inform%status = GALAHAD_error_primal_infeasible
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
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : , : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER ( KIND = ip_ ) :: i, lx, nrhs
     INTEGER( ipc_ ) :: pastix_info
     REAL :: time, time_now
     REAL ( KIND = rp_ ) :: clock, clock_now, tiny

!  trivial cases

     IF ( data%solver( 1 : data%len_solver ) /= 'mumps' ) THEN
       nrhs = SIZE( X, 2 )
       SELECT CASE ( SMT_get( matrix%type ) )
       CASE ( 'DIAGONAL' )
         IF ( inform%rank == matrix%n ) THEN
           DO i = 1, nrhs
             X( : matrix%n, i ) = X( : matrix%n, i ) / matrix%val( : matrix%n )
           END DO
           inform%status = GALAHAD_ok
         ELSE
           inform%status = GALAHAD_error_solve
         END IF
         GO TO 900
       CASE ( 'SCALED_IDENTITY' )
         IF ( inform%rank == matrix%n ) THEN
           X( : matrix%n, : nrhs ) = X( : matrix%n, : nrhs ) / matrix%val( 1 )
           inform%status = GALAHAD_ok
         ELSE
           inform%status = GALAHAD_error_solve
         END IF
         GO TO 900
       CASE ( 'IDENTITY' )
         inform%status = GALAHAD_ok
!        X( : matrix%n, : nhs ) = X( : matrix%n, : nhs )
         GO TO 900
       CASE ( 'ZERO', 'NONE' )
         inform%status = GALAHAD_error_solve
         GO TO 900
       END SELECT
     END IF

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
       CALL SLS_copy_control_to_pardiso( control, data%pardiso_IPARM )
       data%pardiso_IPARM( 6 ) = 1
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,        &
                     33_ip_, data%matrix%n, data%matrix%VAL( : data%ne ),      &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), nrhs,                      &
                     data%pardiso_IPARM( : 64 ),                               &
                     control%print_level_solver,                               &
                     X( : data%matrix%n, : nrhs ),                             &
                     data%B2( : data%matrix%n, : nrhs ), inform%pardiso_error, &
                     data%pardiso_DPARM( 1 : 64 ) )
       inform%pardiso_IPARM = data%pardiso_IPARM
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
         inform%iterative_refinements = data%pardiso_IPARM( 7 )
       END SELECT

!  = MKL PARDISO =

     CASE ( 'mkl_pardiso' )
       nrhs = SIZE( X, 2 )
       CALL SPACE_resize_array( data%n, nrhs, data%X2, inform%status,          &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF

       data%mkl_pardiso_IPARM = inform%mkl_pardiso_IPARM
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MKL_PARDISO_SOLVE(                                                 &
                     data%mkl_pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,    &
                     33_ip_, data%matrix%n, data%matrix%VAL( 1 : data%ne ),    &
                     data%matrix%PTR( 1 : data%matrix%n + 1 ),                 &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%ORDER( 1 : data%matrix%n ), nrhs,                    &
                     data%mkl_pardiso_IPARM, control%print_level_solver,       &
                     X( : data%matrix%n, : nrhs ),                             &
                     data%X2( : data%matrix%n, : nrhs ), inform%pardiso_error )
       inform%mkl_pardiso_IPARM = data%mkl_pardiso_IPARM

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
       X( : data%matrix%n, : nrhs ) = data%X2( : data%matrix%n, : nrhs )

!  = WSMP =

     CASE ( 'wsmp' )
       nrhs = SIZE( X, 2 )
       CALL SLS_copy_control_to_wsmp( control, data%wsmp_IPARM,                &
                                      data%wsmp_DPARM )
       data%wsmp_IPARM( 2 ) = 4
       data%wsmp_IPARM( 3 ) = 5
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   X( : data%matrix%n, : nrhs ), matrix%n, nrhs,               &
                   data%wsmp_AUX, 0_ip_, data%MRP( 1 : data%matrix%n ),        &
                   data%wsmp_IPARM, data%wsmp_DPARM )

       inform%wsmp_IPARM = data%wsmp_IPARM
       inform%wsmp_DPARM = data%wsmp_DPARM
       inform%wsmp_error = data%wsmp_IPARM( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_IPARM( 6 )
       CASE ( - 102 )
         inform%status = GALAHAD_error_allocate
       CASE DEFAULT
         inform%status = GALAHAD_error_wsmp
       END SELECT

!  = PaStiX =

     CASE ( 'pastix' )

       IF ( .TRUE. ) THEN ! fix as pastix can't handle multiple refinement
         CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,       &
                 inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
         CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,       &
                 inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         nrhs = SIZE( X, 2 )
         DO i = 1, nrhs ! loop over the right-hand sides one at a time
           data%X2( : data%n, 1 ) = X( : data%n, i )
           data%B2( : data%n, 1 ) = X( : data%n, i )

           CALL pastix_task_solve( data%pastix_data, data%n, 1_ip_, data%X2,   &
                                   data%spm%nexp, pastix_info )

           inform%pastix_info = INT( pastix_info )
           IF ( pastix_info == PASTIX_SUCCESS ) THEN
             inform%status = GALAHAD_ok
           ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
             inform%status = GALAHAD_error_restrictions ; GO TO 900
           ELSE
             inform%status = GALAHAD_error_pastix ; GO TO 900
           END IF

           CALL pastix_task_refine( data%pastix_data, data%spm%nexp, 1_ip_,    &
                                    data%B2, data%spm%nexp, data%X2,           &
                                    data%spm%nexp, pastix_info )

           inform%pastix_info = INT( pastix_info )
           IF ( pastix_info == PASTIX_SUCCESS ) THEN
             inform%status = GALAHAD_ok
           ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
             inform%status = GALAHAD_error_restrictions ; GO TO 900
           ELSE
             inform%status = GALAHAD_error_pastix ; GO TO 900
           END IF

           X( : data%n, i ) = data%X2( : data%n, 1 )
         END DO
       ELSE ! disable as pastix doesn't have this feature
         nrhs = SIZE( X, 2 )
         CALL SPACE_resize_array( data%n, nrhs, data%B2, inform%status,        &
                 inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
         CALL CPU_time( time ) ; CALL CLOCK_time( clock )
         data%B2( : data%n, : nrhs ) = X( : data%n, : nrhs )

         CALL pastix_task_solve( data%pastix_data, data%n, nrhs, X,            &
                                 data%spm%nexp, pastix_info )

         inform%pastix_info = INT( pastix_info )
         IF ( pastix_info == PASTIX_SUCCESS ) THEN
           inform%status = GALAHAD_ok
         ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
           inform%status = GALAHAD_error_restrictions ; GO TO 900
         ELSE
           inform%status = GALAHAD_error_pastix ; GO TO 900
         END IF

         CALL pastix_task_refine( data%pastix_data, data%spm%nexp, nrhs,       &
                                  data%B2, data%spm%nexp, X, data%spm%nexp,    &
                                  pastix_info )

         inform%pastix_info = INT( pastix_info )
         IF ( pastix_info == PASTIX_SUCCESS ) THEN
           inform%status = GALAHAD_ok
         ELSE IF ( pastix_info == PASTIX_ERR_BADPARAMETER ) THEN
           inform%status = GALAHAD_error_restrictions ; GO TO 900
         ELSE
           inform%status = GALAHAD_error_pastix ; GO TO 900
         END IF
       END IF

!  = MUMPS =

     CASE ( 'mumps' )
       nrhs = SIZE( X, 2 )
       IF ( data%mumps_par%MYID == 0 ) THEN
         data%mumps_par%NRHS = nrhs
         data%mumps_par%LRHS = data%n
         CALL SPACE_resize_pointer( data%n * nrhs, data%mumps_par%RHS,         &
                                    inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%mumps_par%RHS' ; GO TO 900 ; END IF
         lx = 0
         DO i = 1, nrhs
           data%mumps_par%RHS( lx + 1 : lx + data%n ) = X( : data%n, i )
           lx = lx + data%n
         END DO
       END IF
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       data%mumps_par%JOB = 3
       CALL MUMPS_precision( data%mumps_par )
       inform%mumps_info = data%mumps_par%INFOG
       inform%mumps_rinfo = data%mumps_par%RINFOG
       inform%mumps_error = data%mumps_par%INFOG( 1 )

       SELECT CASE( inform%mumps_error )
       CASE ( 0 : )
         inform%status = GALAHAD_ok
         lx = 0
         DO i = 1, nrhs
           X( : data%n, i ) = data%mumps_par%RHS( lx + 1 : lx + data%n )
           lx = lx + data%n
         END DO
         inform%iterative_refinements = data%mumps_par%INFOG( 15 )
       CASE DEFAULT
         inform%status = GALAHAD_error_mumps ; GO TO 900
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
           IF ( inform%rank == data%n ) THEN
             CALL SYTRS( 'L', data%n, nrhs, data%matrix_dense, data%n,         &
                         data%PIVOTS, data%X2, data%n, inform%lapack_error )
           ELSE
             tiny = 100.0_rp_ * control%absolute_pivot_tolerance
             CALL SLS_sytr_singular_solve( data%n, nrhs, data%matrix_dense,    &
                                           data%n, data%PIVOTS, data%X2,       &
                                           data%n, tiny, inform%lapack_error )
           END IF

!  = PBTR =

         CASE ( 'pbtr' )
           CALL PBTRS( 'L', data%n, inform%semi_bandwidth, nrhs,               &
                       data%matrix_dense, inform%semi_bandwidth + 1_ip_,       &
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
           IF ( inform%rank == data%n ) THEN
             CALL SYTRS( 'L', data%n, nrhs, data%matrix_dense, data%n,         &
                         data%PIVOTS, X, data%n, inform%lapack_error )
           ELSE
             tiny = 100.0_rp_ * control%absolute_pivot_tolerance
             CALL SLS_sytr_singular_solve( data%n, nrhs, data%matrix_dense,    &
                                           data%n, data%PIVOTS, X,             &
                                           data%n, tiny, inform%lapack_error )
           END IF

!  = PBTR =

         CASE ( 'pbtr' )
           CALL PBTRS( 'L', data%n, inform%semi_bandwidth, nrhs,               &
                       data%matrix_dense, inform%semi_bandwidth + 1_ip_,       &
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

     INTEGER ( KIND = ip_ ) :: info

!  trivial cases

     IF ( data%trivial_matrix_type ) THEN
       CALL SPACE_dealloc_array( data%WORK, inform%status, inform%alloc_status )
       RETURN
     END IF

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
       IF ( control%print_level <= 0 .OR. control%out <= 0 )                   &
           data%ma86_control%unit_error = - 1
       CALL MA86_finalise( data%ma86_keep, data%ma86_control )
       inform%status = 0

!  = MA87 =

     CASE ( 'ma87' )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       IF ( control%print_level <= 0 .OR. control%out <= 0 )                   &
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
         CALL PARDISO( data%pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,      &
                       - 1_ip_, data%matrix%n, data%matrix%VAL( : data%ne ),   &
                       data%matrix%PTR( : data%matrix%n + 1 ),                 &
                       data%matrix%COL( : data%ne ),                           &
                       data%ORDER( : data%matrix%n ), 1_ip_,                   &
                       data%pardiso_IPARM( : 64 ),                             &
                       control%print_level_solver,                             &
                       data%B2( : data%matrix%n, : 1 ),                        &
                       data%X2( : data%matrix%n, : 1 ), inform%pardiso_error,  &
                       data%pardiso_DPARM( 1 : 64 ) )
         inform%pardiso_IPARM = data%pardiso_IPARM

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

!  = MKL PARDISO =

     CASE ( 'mkl_pardiso' )
       IF ( data%n > 0 ) THEN
         CALL SLS_copy_control_to_mkl_pardiso( control, data%mkl_pardiso_IPARM )
         CALL MKL_PARDISO_SOLVE( data%mkl_pardiso_PT, 1_ip_, 1_ip_,            &
                                 data%pardiso_mtype, - 1_ip_, data%matrix%n,   &
                                 data%ddum, data%idum, data%idum, data%idum,   &
                                 1_ip_, data%mkl_pardiso_IPARM,                &
                                 control%print_level_solver,                   &
                                 data%ddum, data%ddum, inform%pardiso_error )

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
       CALL SPACE_dealloc_array( data%B1, inform%status, inform%alloc_status )
       CALL SPACE_dealloc_array( data%B2, inform%status, inform%alloc_status )
       CALL SPACE_dealloc_array( data%X2, inform%status, inform%alloc_status )
       CALL SPACE_dealloc_array( data%MAPS, inform%status, inform%alloc_status )

!  = WSMP =

     CASE ( 'wsmp' )
       CALL WSMP_CLEAR( )
       CALL WSSFREE( )

!  = PaStiX =

     CASE ( 'pastix' )
       CALL pastixFinalize( data%pastix_data )
       CALL spmExit( data%spm )
       DEALLOCATE( data%spm )
       data%no_pastix = .FALSE.

!  = MUMPS =

     CASE ( 'mumps' )
       IF ( data%mumps_par%MYID == 0 ) THEN
         CALL SPACE_dealloc_pointer( data%mumps_par%IRN, inform%status,        &
                                     inform%alloc_status )
         CALL SPACE_dealloc_pointer( data%mumps_par%JCN, inform%status,        &
                                     inform%alloc_status )
         CALL SPACE_dealloc_pointer( data%mumps_par%A, inform%status,          &
                                     inform%alloc_status )
         CALL SPACE_dealloc_pointer( data%mumps_par%PERM_IN, inform%status,    &
                                     inform%alloc_status )
         CALL SPACE_dealloc_pointer( data%mumps_par%RHS, inform%status,        &
                                     inform%alloc_status )
       END IF
       IF ( control%print_level_solver <= 0 ) data%mumps_par%ICNTL( 4 ) = 0
       data%mumps_par%JOB = - 2
       CALL MUMPS_precision( data%mumps_par )
       data%no_mumps = .FALSE. ; data%no_mpi = .FALSE.

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

! -  G A L A H A D -  S L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE SLS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

     CALL SLS_terminate( data%sls_data, control, inform )

!  deallocate any internal problem arrays

     CALL SPACE_dealloc_array( data%matrix%ptr,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%row,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%col,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%val,                                &
                               inform%status, inform%alloc_status )

     RETURN

!  End of subroutine SLS_full_terminate

     END SUBROUTINE SLS_full_terminate

!-*-*-*-*-*-*-*-   S L S _ E N Q U I R E  S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE SLS_enquire( data, inform, PERM, PIVOTS, D, PERTURBATION )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Interogate the factorization to obtain additional information

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ), OPTIONAL,                          &
                               DIMENSION( data%n ) :: PIVOTS, PERM
     REAL ( KIND = rp_ ), INTENT( OUT ), OPTIONAL, DIMENSION( 2, data%n ) :: D
     REAL ( KIND = rp_ ), INTENT( OUT ), OPTIONAL,                             &
                               DIMENSION( data%n ) :: PERTURBATION

!  local variables

     INTEGER ( KIND = ip_ ) :: i, ip, k, pk
     REAL :: time_start, time_now
     REAL ( KIND = rp_ ) :: clock_start, clock_now
!    INTEGER, DIMENSION( data%n ) :: INV

!  trivial cases

     IF ( data%trivial_matrix_type ) THEN
       IF ( PRESENT( PERM ) ) THEN
         DO i = 1, data%n
           PERM( i ) = i
         END DO
         inform%status = GALAHAD_ok
       END IF
       IF ( PRESENT( PIVOTS ) ) THEN
         DO i = 1, data%n
           PIVOTS( i ) = i
         END DO
         inform%status = GALAHAD_ok
       END IF
       IF ( PRESENT( PERTURBATION ) ) THEN
         DO i = 1, data%n
           PERTURBATION( i ) = 0.0_rp_
         END DO
         inform%status = GALAHAD_ok
       END IF
       IF ( PRESENT( D ) ) inform%status = GALAHAD_error_access_diagonal
       GO TO 900
     END IF

     inform%status = GALAHAD_ok

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  solver-dependent enquiry

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
             CALL SPACE_resize_array( 2_ip_, data%n, data%D, inform%status,    &
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
         D( 1, : ) = 1.0_rp_ ; D( 2, : ) = 0.0_rp_
       END IF
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = MA87 =

     CASE ( 'ma87' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_rp_ ; D( 2, : ) = 0.0_rp_
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
         D( 2, : ) = 0.0_rp_
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
           CALL SPACE_resize_array( 2_ip_, data%n, data%D, inform%status,      &
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
                                    D( 1, : ) )
         D( 2, : ) = 0.0_rp_
       ELSE

         IF ( PRESENT( PIVOTS ) ) THEN
           CALL SPACE_resize_array( data%n, data%INVP,                         &
                                    inform%status, inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) THEN
             inform%bad_alloc = 'sls: data%INVP' ; RETURN
           END IF
         END IF

         IF ( PRESENT( D ) ) THEN
           IF ( PRESENT( PIVOTS ) ) THEN
             CALL SSIDS_enquire_indef( data%ssids_akeep, data%ssids_fkeep,     &
                                       data%ssids_options, data%ssids_inform,  &
                                       piv_order = data%INVP, d = D )
           ELSE
             CALL SPACE_resize_array( data%n, data%PIVOTS,                     &
                                      inform%status, inform%alloc_status )
             IF ( inform%status /= GALAHAD_ok ) GO TO 900
             CALL SSIDS_enquire_indef( data%ssids_akeep, data%ssids_fkeep,     &
                                       data%ssids_options, data%ssids_inform,  &
                                       d = D )
           END IF
         ELSE
           CALL SPACE_resize_array( 2_ip_, data%n, data%D, inform%status,      &
                                    inform%alloc_status )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           IF ( PRESENT( PIVOTS ) ) THEN
             CALL SSIDS_enquire_indef( data%ssids_akeep, data%ssids_fkeep,     &
                                       data%ssids_options, data%ssids_inform,  &
                                       piv_order = data%INVP )
           END IF
         END IF
         IF ( PRESENT( PIVOTS ) ) THEN
           DO i = 1, data%n
             IF ( data%INVP( i ) > 0 ) THEN
               PIVOTS( data%INVP( i ) ) = i
             ELSE
               PIVOTS( - data%INVP( i ) ) = - i
             END IF
           END DO
         END IF
       END IF

!  = PaStiX =

     CASE ( 'pastix' )
       IF ( PRESENT( PERM ) ) THEN
         CALL pastixOrderGet( data%pastix_data, data%order_pastix )
         CALL pastixOrderGetArray( data%order_pastix,                      &
                                       permtab = data%PERMTAB )
         PERM = data%PERMTAB( : data%n ) + 1
       END IF
       IF ( PRESENT( D ) )  inform%status = GALAHAD_error_access_diagonal
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = MUMPS =

     CASE ( 'mumps' )
       IF ( PRESENT( PERM ) ) THEN
         PERM = data%mumps_par%SYM_PERM( : data%n )
       END IF
       IF ( PRESENT( D ) )  inform%status = GALAHAD_error_access_diagonal
       IF ( PRESENT( PIVOTS ) ) inform%status = GALAHAD_error_access_pivots
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = POTR =

     CASE ( 'potr' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_rp_ ; D( 2, : ) = 0.0_rp_
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
             IF ( data%matrix_dense( k, k ) < 0.0_rp_ ) THEN
               inform%negative_eigenvalues =                                 &
                 inform%negative_eigenvalues + 1
             ELSE IF ( data%matrix_dense( k, k ) == 0.0_rp_ ) THEN
               inform%rank = inform%rank - 1
             END IF
             D( 1, k ) = data%matrix_dense( k, k )
             D( 2, k ) = 0.0_rp_
             k = k + 1
           ELSE                               ! a 2 x 2 pivot
             D( 1, k ) = data%matrix_dense( k, k )
             D( 1, k + 1 ) = data%matrix_dense( k + 1, k + 1 )
             D( 2, k ) = data%matrix_dense( k + 1, k )
             D( 2, k + 1 ) = 0.0_rp_
             k = k + 2
           END IF
         END DO
       END IF
       IF ( PRESENT( PIVOTS ) ) THEN
         DO k = 1, data%n
           PIVOTS( k ) = k
         END DO
         k = 1
         DO                                   ! run through the pivots
           IF ( k > data%n ) EXIT
           ip = data%PIVOTS( k )
           IF ( ip > 0 ) THEN   ! a 1 x 1 pivot, swap columns k and ip
             pk = PIVOTS( k )
             PIVOTS( k ) = PIVOTS( ip )
             PIVOTS( ip ) = pk
             k = k + 1
           ELSE ! a 2 x 2 pivot, swap columns k + 1 and -ip
             pk = PIVOTS( k + 1 )
             PIVOTS( k ) = - PIVOTS( k )
             PIVOTS( k + 1 ) = - PIVOTS( - ip )
             PIVOTS( - ip ) = pk
             k = k + 2
           END IF
         END DO
         DO i = 1, data%n
           k = PIVOTS( i )
           IF ( k > 0 ) then
             PIVOTS( i ) = data%INVP( k )
           ELSE
             PIVOTS( i ) = - data%INVP( - k )
           END IF
         END DO
       END IF
       IF ( PRESENT( PERTURBATION ) ) inform%status = GALAHAD_error_access_pert

!  = PBTR =

     CASE ( 'pbtr' )
       IF ( PRESENT( PERM ) ) PERM = data%ORDER( : data%n )
       IF ( PRESENT( D ) ) THEN
         D( 1, : ) = 1.0_rp_ ; D( 2, : ) = 0.0_rp_
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
     REAL ( KIND = rp_ ), INTENT( INOUT ) :: D( 2, data%n )
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER ( KIND = ip_ ) :: info, k
     REAL :: time_start, time_now
     REAL ( KIND = rp_ ) :: clock_start, clock_now

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases

     IF ( data%trivial_matrix_type ) THEN
       inform%status = GALAHAD_error_alter_diagonal
       RETURN
     END IF

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
           IF ( data%matrix_dense( k, k ) < 0.0_rp_ ) THEN
             inform%negative_eigenvalues =                                     &
               inform%negative_eigenvalues + 1
           ELSE IF ( data%matrix_dense( k, k ) == 0.0_rp_ ) THEN
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

!  Solve a system involving individual factors. The matrix A is presumed
!  to have been factorized as A = L D U (with U = L^T).  The character
!  "part" is one of 'L', 'D', 'U' or 'S', where 'S' refers to L * SQRT(D)

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = 1 ), INTENT( IN ) :: part
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER ( KIND = ip_ ) :: i, info, phase
     REAL :: time, time_start, time_now
     REAL ( KIND = rp_ ) :: clock, clock_start, clock_now

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases

     IF ( data%trivial_matrix_type ) THEN
       IF ( part == 'D' ) THEN
         DO i = 1, data%n
           IF ( data%WORK( i ) /= 0.0_rp_ ) THEN
             X( i ) = X( i ) / data%WORK( i )
           ELSE
             inform%status = GALAHAD_error_solve
             GO TO 900
           END IF
         END DO
       END IF
       IF ( part == 'S' ) THEN
         DO i = 1, data%n
           IF ( data%WORK( i ) > 0.0_rp_ ) THEN
             X( i ) = X( i ) / SQRT( data%WORK( i ) )
           ELSE
             inform%status = GALAHAD_error_solve
             GO TO 900
           END IF
         END DO
       END IF
       inform%status = GALAHAD_ok
       GO TO 900
     END IF

!  scale if required

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
           IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
           IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.     &
                ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.       &
                ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_rp_ ) THEN
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
           IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
           IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.     &
                ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.       &
                ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_rp_ ) THEN
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
       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         IF ( part == 'L' .OR.                                                 &
              ( part == 'S' .AND. data%must_be_definite ) ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 1_ip_, scale = data%SCALE )
         ELSE IF ( part == 'D' ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 2_ip_, scale = data%SCALE )
         ELSE  IF ( part == 'U' ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3_ip_, scale = data%SCALE )
         ELSE
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3_ip_, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3_ip_, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) /= 0.0_rp_ ) .OR.  &
                  ( data%X2( i, 1 ) /= 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_rp_ .AND. X( i ) < 0.0_rp_ ) .OR.    &
                  ( data%X2( i, 1 ) < 0.0_rp_ .AND. X( i ) > 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_rp_ ) THEN
               data%X2( i, 1 ) = SQRT( data%X2( i, 1 ) ) * SQRT( X( i ) )
             ELSE
               data%X2( i, 1 ) = - SQRT( - data%X2( i, 1 ) ) * SQRT( - X( i ) )
             END IF
           END DO
         END IF
       ELSE
         IF ( part == 'L' .OR.                                                 &
              ( part == 'S' .AND. data%must_be_definite ) ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 1_ip_ )
         ELSE IF ( part == 'D' ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 2_ip_ )
         ELSE  IF ( part == 'U' ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 3_ip_ )
         ELSE
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 1_ip_ )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 2_ip_ )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) /= 0.0_rp_ ) .OR.  &
                  ( data%X2( i, 1 ) /= 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_rp_ .AND. X( i ) < 0.0_rp_ ) .OR.    &
                  ( data%X2( i, 1 ) < 0.0_rp_ .AND. X( i ) > 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_rp_ ) THEN
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
                          data%ma86_control, data%ma86_info, job = 1_ip_ )
       ELSE IF ( part == 'D' ) THEN
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 2_ip_ )
       ELSE  IF ( part == 'U' ) THEN
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 3_ip_ )
       ELSE
         CALL MA86_solve( X, data%ORDER, data%ma86_keep,                       &
                          data%ma86_control, data%ma86_info, job = 1_ip_ )
         inform%ma86_info = data%ma86_info
         inform%status = data%ma86_info%flag
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL MA86_solve( data%WORK( : data%n ), data%ORDER, data%ma86_keep,   &
                          data%ma86_control, data%ma86_info, job = 2_ip_ )
         inform%ma86_info = data%ma86_info
         inform%status = data%ma86_info%flag
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
           IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.     &
                ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.       &
                ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_rp_ ) THEN
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
                          data%ma87_control, data%ma87_info, job = 1_ip_ )
       ELSE  IF ( part == 'U' ) THEN
         CALL MA87_solve( X, data%ORDER, data%ma87_keep,                       &
                          data%ma87_control, data%ma87_info, job = 2_ip_ )
       END IF
       inform%ma87_info = data%ma87_info
       inform%status = data%ma87_info%flag

!  = MA97 =

     CASE ( 'ma97' )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( part == 'L' .OR. ( part == 'S' .AND. data%must_be_definite ) ) THEN
         CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
                          data%ma97_control, data%ma97_info, job = 1_ip_ )
       ELSE IF ( part == 'D' ) THEN
         IF ( data%must_be_definite ) THEN
           inform%status = 0
           GO TO 900
         ELSE
           CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,   &
                            data%ma97_control, data%ma97_info, job = 2_ip_ )
         END IF
       ELSE  IF ( part == 'U' ) THEN
         CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
                          data%ma97_control, data%ma97_info, job = 3_ip_ )
       ELSE
         CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
                          data%ma97_control, data%ma97_info, job = 1_ip_ )
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL MA97_solve( data%WORK( : data%n ), data%ma97_akeep,              &
                          data%ma97_fkeep,                                     &
                          data%ma97_control, data%ma97_info, job = 2_ip_ )
         CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
           IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.     &
                ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.       &
                ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_rp_ ) THEN
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
                           data%ssids_options, data%ssids_inform, job = 1_ip_ )
       ELSE IF ( part == 'D' ) THEN
         IF ( data%must_be_definite ) THEN
           inform%status = 0
           GO TO 900
         ELSE
           CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,&
                           data%ssids_options, data%ssids_inform, job = 2_ip_ )
         END IF
       ELSE  IF ( part == 'U' ) THEN
         CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,  &
                           data%ssids_options, data%ssids_inform, job = 3_ip_ )
       ELSE
         CALL SSIDS_solve( X( : data%n ), data%ssids_akeep, data%ssids_fkeep,  &
                           data%ssids_options, data%ssids_inform, job = 1_ip_ )
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         CALL SPACE_resize_array( data%n, data%WORK,                           &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         data%WORK( : data%n ) = X( : data%n )
         CALL SSIDS_solve( data%WORK( : data%n ), data%ssids_akeep,            &
                           data%ssids_fkeep,                                   &
                           data%ssids_options, data%ssids_inform, job = 2_ip_ )
         CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )
         IF ( inform%status /= GALAHAD_ok ) GO TO 900
         DO i = 1, data%n
           IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
           IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.     &
                ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.       &
                ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
             inform%status = GALAHAD_error_inertia ; GO TO 900
           END IF
           IF ( X( i ) > 0.0_rp_ ) THEN
             X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
           ELSE
             X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
           END IF
         END DO
       END IF
       CALL SLS_copy_inform_from_ssids( inform, data%ssids_inform )

!  = PARDISO =

     CASE ( 'pardiso' )
       IF ( part == 'S' ) THEN
         inform%status = 0
         GO TO 900
       END IF
       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL SLS_copy_control_to_pardiso( control, data%pardiso_IPARM )
       data%pardiso_IPARM( 6 ) = 0
       IF ( part == 'L' ) THEN
         data%pardiso_IPARM( 26 ) = - 1
       ELSE IF ( part == 'D' ) THEN
         data%pardiso_IPARM( 26 ) = - 2
       ELSE IF ( part == 'U' ) THEN
         data%pardiso_IPARM( 26 ) = - 3
       END IF
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,        &
                     33_ip_, data%matrix%n, data%matrix%VAL( : data%ne ),      &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), 1_ip_,                     &
                     data%pardiso_IPARM( : 64 ),                               &
                     control%print_level_solver,                               &
                     data%X2( : data%matrix%n, : 1 ),                          &
                     data%B2( : data%matrix%n, : 1 ), inform%pardiso_error,    &
                     data%pardiso_DPARM( 1 : 64 ) )
       inform%pardiso_IPARM = data%pardiso_IPARM
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
         inform%iterative_refinements =  data%pardiso_IPARM( 7 )
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%X2( : data%n, 1 )

!  = MKL PARDISO =

     CASE ( 'mkl_pardiso' )
       IF ( part == 'L' ) THEN
         phase = 331
       ELSE IF ( part == 'D' ) THEN
         phase = 332
       ELSE IF ( part == 'U' ) THEN
         phase = 333
       ELSE
         inform%status = GALAHAD_unavailable_option
         GO TO 900
       END IF

       CALL SPACE_resize_array( data%n, data%B1, inform%status,                &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B1' ; GO TO 900 ; END IF
       data%B1( : data%n ) = X( : data%n )

       data%mkl_pardiso_IPARM = inform%mkl_pardiso_IPARM
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MKL_PARDISO_SOLVE(                                                 &
                     data%mkl_pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,    &
                     phase, data%matrix%n, data%matrix%VAL( 1 : data%ne ),     &
                     data%matrix%PTR( 1 : data%matrix%n + 1 ),                 &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%ORDER( 1 : data%matrix%n ), 1_ip_,                   &
                     data%mkl_pardiso_IPARM, control%print_level_solver,       &
                     data%B1( 1 : data%n ), X( 1 : data%n ),                   &
                     inform%pardiso_error )
       inform%mkl_pardiso_IPARM = data%mkl_pardiso_IPARM

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
         inform%iterative_refinements =  data%pardiso_iparm( 7 )
       END SELECT

!  = WSMP =

     CASE ( 'wsmp' )
       CALL SLS_copy_control_to_wsmp( control, data%wsmp_IPARM,                &
                                      data%wsmp_DPARM )
       IF ( data%wsmp_IPARM( 31 ) == 0 .OR. data%wsmp_IPARM( 31 ) == 5 ) THEN
         IF ( part == 'D' ) THEN
           inform%status = 0
           GO TO 900
         ELSE IF ( part == 'L' .OR. part == 'S') THEN
           data%wsmp_IPARM( 30 ) = 1
         ELSE
           data%wsmp_IPARM( 30 ) = 2
         END IF
       ELSE
         IF ( part == 'D' ) THEN
           data%wsmp_IPARM( 30 ) = 3
         ELSE IF ( part == 'L' ) THEN
           data%wsmp_IPARM( 30 ) = 1
         ELSE IF ( part == 'U' ) THEN
           data%wsmp_IPARM( 30 ) = 2
         ELSE
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
       END IF
       data%wsmp_IPARM( 2 ) = 4
       data%wsmp_IPARM( 3 ) = 4

       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1_ip_,  &
                   data%wsmp_AUX, 0_ip_, data%MRP( 1 : data%matrix%n ),        &
                   data%wsmp_IPARM, data%wsmp_DPARM )

       inform%wsmp_IPARM = data%wsmp_IPARM
       inform%wsmp_DPARM = data%wsmp_DPARM
       inform%wsmp_error = data%wsmp_IPARM( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_IPARM( 6 )
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
                       data%B, 1_ip_ )
         ELSE IF ( part == 'U' ) THEN
           CALL TRSV ( 'L', 'T', 'N', data%n, data%matrix_dense, data%n,       &
                       data%B, 1_ip_ )
         ELSE ! if part = 'D'
           GO TO 900
         END IF
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
      ELSE
         IF ( part == 'L' .OR. part == 'S') THEN
           CALL TRSV ( 'L', 'N', 'N', data%n, data%matrix_dense, data%n,       &
                        X, 1_ip_ )
         ELSE IF ( part == 'U' ) THEN
           CALL TRSV ( 'L', 'T', 'N', data%n, data%matrix_dense, data%n,       &
                        X, 1_ip_ )
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
           CALL TBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,            &
                      data%matrix_dense, inform%semi_bandwidth + 1_ip_,        &
                      data%B, 1_ip_ )
         ELSE IF ( part == 'U' ) THEN
           CALL TBSV( 'L', 'T', 'N', data%n, inform%semi_bandwidth,            &
                      data%matrix_dense, inform%semi_bandwidth + 1_ip_,        &
                      data%B, 1_ip_ )
         ELSE ! if part = 'D'
           GO TO 900
         END IF
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         IF ( part == 'L' .OR. part == 'S') THEN
           CALL TBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,            &
                      data%matrix_dense, inform%semi_bandwidth + 1_ip_,        &
                      X, 1_ip_ )
         ELSE IF ( part == 'U' ) THEN
           CALL TBSV( 'L', 'T', 'N', data%n, inform%semi_bandwidth,            &
                      data%matrix_dense, inform%semi_bandwidth + 1_ip_,        &
                      X, 1_ip_ )
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

     INTEGER ( KIND = ip_ ), INTENT( IN  ) :: nnz_b
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nnz_x
     INTEGER ( KIND = ip_ ), INTENT( INOUT  ), DIMENSION( : ) :: INDEX_b
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( : ) :: INDEX_x
     REAL ( KIND = rp_ ), INTENT( IN  ), DIMENSION( : ) :: B
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: X
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER ( KIND = ip_ ) :: i, info
     REAL :: time, time_start, time_now
     REAL ( KIND = rp_ ) :: clock, clock_start, clock_now

!  start timimg

     CALL CPU_time( time_start ) ; CALL CLOCK_time( clock_start )

!  trivial cases

     IF ( data%trivial_matrix_type ) THEN
       nnz_x = nnz_b
       INDEX_x( : nnz_x ) = INDEX_b( : nnz_b )
       inform%status = GALAHAD_ok
       GO TO 900
     END IF

!  scale if necessary

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
       X( : data%n ) = 0.0_rp_
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
         IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
         IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.       &
              ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.         &
              ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( X( i ) > 0.0_rp_ ) THEN
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
!        IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
!        IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.       &
!             ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
!          inform%status = GALAHAD_error_inertia ; GO TO 900
!        END IF
!        IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.         &
!             ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
!          inform%status = GALAHAD_error_inertia ; GO TO 900
!        END IF
!        IF ( X( i ) > 0.0_rp_ ) THEN
!          X( i ) = SQRT( X( i ) ) * SQRT( data%WORK( i ) )
!        ELSE
!          X( i ) = - SQRT( - X( i ) ) * SQRT( - data%WORK( i ) )
!        END IF
!      END DO

       DO i = 1, nnz_b
!        INDEX_x( i ) = INDEX_b( i )
         X( INDEX_b( i ) ) = B( INDEX_b( i ) )
!        B( INDEX_b( i ) ) = 0.0_rp_
       END DO
       CALL ma57_sparse_lsolve( data%ma57_factors, data%ma57_control, nnz_b,   &
                                INDEX_b, nnz_x, INDEX_x, X,  data%ma57_sinfo )
       inform%ma57_sinfo = data%ma57_sinfo
       inform%status = data%ma57_sinfo%flag

!  = MA77 =

     CASE ( 'ma77' )
       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         IF ( data%must_be_definite ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 1_ip_, scale = data%SCALE )
         ELSE
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3_ip_, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info,                 &
                            job = 3_ip_, scale = data%SCALE )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) /= 0.0_rp_ ) .OR.  &
                  ( data%X2( i, 1 ) /= 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_rp_ .AND. X( i ) < 0.0_rp_ ) .OR.    &
                  ( data%X2( i, 1 ) < 0.0_rp_ .AND. X( i ) > 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_rp_ ) THEN
               data%X2( i, 1 ) = SQRT( data%X2( i, 1 ) ) * SQRT( X( i ) )
             ELSE
               data%X2( i, 1 ) = - SQRT( - data%X2( i, 1 ) ) * SQRT( - X( i ) )
             END IF
           END DO
         END IF
       ELSE
         IF ( data%must_be_definite ) THEN
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 1_ip_ )
         ELSE
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 1_ip_ )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           X( : data%n ) = data%X2( : data%n, 1 )
           CALL MA77_solve( 1_ip_, data%n, data%X2, data%ma77_keep,            &
                            data%ma77_control, data%ma77_info, job = 2_ip_ )
           CALL SLS_copy_inform_from_ma77( inform, data%ma77_info )
           IF ( inform%status /= GALAHAD_ok ) GO TO 900
           DO i = 1, data%n
             IF ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) CYCLE
             IF ( ( data%X2( i, 1 ) == 0.0_rp_ .AND. X( i ) /= 0.0_rp_ ) .OR.  &
                  ( data%X2( i, 1 ) /= 0.0_rp_ .AND. X( i ) == 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( ( data%X2( i, 1 ) > 0.0_rp_ .AND. X( i ) < 0.0_rp_ ) .OR.    &
                  ( data%X2( i, 1 ) < 0.0_rp_ .AND. X( i ) > 0.0_rp_ ) ) THEN
               inform%status = GALAHAD_error_inertia ; GO TO 900
             END IF
             IF ( data%X2( i, 1 ) > 0.0_rp_ ) THEN
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
                        data%ma86_control, data%ma86_info, job = 1_ip_ )
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       CALL SPACE_resize_array( data%n, data%WORK,                             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       data%WORK( : data%n ) = X( : data%n )
       CALL MA86_solve( data%WORK( : data%n ), data%ORDER, data%ma86_keep,     &
                        data%ma86_control, data%ma86_info, job = 2_ip_ )
       inform%ma86_info = data%ma86_info
       inform%status = data%ma86_info%flag
       IF ( inform%status /= GALAHAD_ok ) GO TO 900
       DO i = 1, data%n
         IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
         IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.       &
              ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.         &
              ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
           inform%status = GALAHAD_error_inertia ; GO TO 900
         END IF
         IF ( X( i ) > 0.0_rp_ ) THEN
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
!                         data%ma97_control, data%ma97_info, job = 1_ip_ )
!      ELSE
!        CALL MA97_solve( X( : data%n ), data%ma97_akeep, data%ma97_fkeep,     &
!                         data%ma97_control, data%ma97_info, job = 1_ip_ )
!        CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
!        IF ( inform%status /= GALAHAD_ok ) GO TO 900
!        CALL SPACE_resize_array( data%n, data%WORK,                           &
!                                 inform%status, inform%alloc_status )
!        IF ( inform%status /= GALAHAD_ok ) GO TO 900
!        data%WORK( : data%n ) = X( : data%n )
!        CALL MA97_solve( data%WORK( : data%n ), data%ma97_akeep,              &
!                         data%ma97_fkeep,                                     &
!                         data%ma97_control, data%ma97_info, job = 2_ip_ )
!        CALL SLS_copy_inform_from_ma97( inform, data%ma97_info )
!        IF ( inform%status /= GALAHAD_ok ) GO TO 900
!        DO i = 1, data%n
!          IF ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) CYCLE
!          IF ( ( X( i ) == 0.0_rp_ .AND. data%WORK( i ) /= 0.0_rp_ ) .OR.     &
!               ( X( i ) /= 0.0_rp_ .AND. data%WORK( i ) == 0.0_rp_ ) ) THEN
!            inform%status = GALAHAD_error_inertia ; GO TO 900
!          END IF
!          IF ( ( X( i ) > 0.0_rp_ .AND. data%WORK( i ) < 0.0_rp_ ) .OR.       &
!               ( X( i ) < 0.0_rp_ .AND. data%WORK( i ) > 0.0_rp_ ) ) THEN
!            inform%status = GALAHAD_error_inertia ; GO TO 900
!          END IF
!          IF ( X( i ) > 0.0_rp_ ) THEN
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
       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_pardiso( control, data%pardiso_IPARM )
       data%pardiso_IPARM( 6 ) = 1
       data%pardiso_IPARM( 26 ) = 1
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL PARDISO( data%pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,        &
                     33_ip_, data%matrix%n, data%matrix%VAL( : data%ne ),      &
                     data%matrix%PTR( : data%matrix%n + 1 ),                   &
                     data%matrix%COL( : data%ne ),                             &
                     data%ORDER( : data%matrix%n ), 1_ip_,                     &
                     data%pardiso_IPARM( : 64 ),                               &
                     control%print_level_solver,                               &
                     data%X2( : data%matrix%n, : 1 ),                          &
                     data%B2( : data%matrix%n, : 1 ), inform%pardiso_error,    &
                     data%pardiso_DPARM( 1 : 64 ) )
       inform%pardiso_iparm = data%pardiso_IPARM
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
         inform%iterative_refinements =  data%pardiso_IPARM( 7 )
       END SELECT
       IF ( inform%status == GALAHAD_ok ) X( : data%n ) = data%X2( : data%n, 1 )

!  = MKL PARDISO =

     CASE ( 'mkl_pardiso' )

!  inefficient simulation

       CALL SPACE_resize_array( data%n, data%B1, inform%status,                &
                                inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B1' ; GO TO 900 ; END IF

       data%B1( : data%n ) = 0.0_rp_
       data%B1( INDEX_b( : nnz_b ) ) = B( INDEX_b( : nnz_b ) )

       data%mkl_pardiso_IPARM = inform%mkl_pardiso_IPARM
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MKL_PARDISO_SOLVE(                                                 &
                     data%mkl_pardiso_PT, 1_ip_, 1_ip_, data%pardiso_mtype,    &
                     331_ip_, data%matrix%n, data%matrix%VAL( 1 : data%ne ),   &
                     data%matrix%PTR( 1 : data%matrix%n + 1 ),                 &
                     data%matrix%COL( 1 : data%ne ),                           &
                     data%ORDER( 1 : data%matrix%n ), 1_ip_,                   &
                     data%mkl_pardiso_IPARM, control%print_level_solver,       &
                     data%B1( 1 : data%n ), X( 1 : data%n ),                   &
                     inform%pardiso_error )
       inform%mkl_pardiso_IPARM = data%mkl_pardiso_IPARM

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

!  record the nonzeros

       IF ( inform%status == GALAHAD_ok ) THEN
         nnz_x = 0
         DO i = 1, data%matrix%n
           IF ( X( i ) /= 0.0_rp_ ) THEN
             nnz_x = nnz_x + 1
             INDEX_x( nnz_x ) = i
           END IF
         END DO
       END IF

!  = WSMP =

     CASE ( 'wsmp' )
       CALL SLS_copy_control_to_wsmp( control, data%wsmp_IPARM,                &
                                      data%wsmp_DPARM )
       IF ( data%wsmp_IPARM( 31 ) == 0 .OR. data%wsmp_IPARM( 31 ) == 5 ) THEN
         data%wsmp_IPARM( 30 ) = 1
       ELSE
         inform%status = GALAHAD_error_inertia ; GO TO 900
       END IF
       data%wsmp_IPARM( 2 ) = 4
       data%wsmp_IPARM( 3 ) = 4

       CALL SPACE_resize_array( data%n, 1_ip_, data%B2, inform%status,         &

               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%B2' ; GO TO 900 ; END IF
       data%B2( : data%n, 1 ) = X( : data%n )

       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL WSSMP( data%matrix%n, data%matrix%PTR( 1 : data%matrix%n + 1 ),    &
                   data%matrix%COL( 1 : data%ne ),                             &
                   data%matrix%VAL( 1 : data%ne ),                             &
                   data%DIAG( 0 : 0 ),                                         &
                   data%ORDER( 1 : data%matrix%n ),                            &
                   data%INVP( 1 : data%matrix%n ),                             &
                   data%B2( 1 : data%matrix%n, 1 : 1 ), data%matrix%n, 1_ip_,  &
                   data%wsmp_AUX, 0_ip_, data%MRP( 1 : data%matrix%n ),        &
                   data%wsmp_IPARM, data%wsmp_DPARM )

       inform%wsmp_IPARM = data%wsmp_IPARM
       inform%wsmp_DPARM = data%wsmp_DPARM
       inform%wsmp_error = data%wsmp_IPARM( 64 )

       SELECT CASE( inform%wsmp_error )
       CASE ( 0 )
         inform%status = GALAHAD_ok
         inform%iterative_refinements = data%wsmp_IPARM( 6 )
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
                     data%B, 1_ip_ )
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         CALL TRSV ( 'L', 'N', 'N', data%n, data%matrix_dense, data%n,         &
                     X, 1_ip_ )
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
         CALL TBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,              &
                    data%matrix_dense, inform%semi_bandwidth + 1_ip_,          &
                    data%B, 1_ip_ )
!        X( data%ORDER( : data%n ) ) = data%B( : data%n )
         X( : data%n ) = data%B( data%ORDER( : data%n ) )
       ELSE
         CALL TBSV( 'L', 'N', 'N', data%n, inform%semi_bandwidth,              &
                    data%matrix_dense, inform%semi_bandwidth + 1_ip_,          &
                    X, 1_ip_ )
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
         IF ( X( i ) == 0.0_rp_ ) CYCLE
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
     REAL ( KIND = rp_ ), INTENT( INOUT ) :: X( : )
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER ( KIND = ip_ ) :: i
!    LOGICAL, DIMENSION( 1 ) :: flag_out
     LOGICAL ( KIND = lp_ ), DIMENSION( 1 ) :: flag_out
     REAL :: time, time_now
     REAL ( KIND = rp_ ) :: clock, clock_now

!  start timimg

     CALL CPU_time( time ) ; CALL CLOCK_time( clock )

!  trivial cases

     SELECT CASE ( SMT_get( matrix%type ) )
     CASE ( 'DIAGONAL' )
       inform%alternative = .FALSE.
       DO i = 1, matrix%n
         IF ( matrix%val( i ) /= 0.0_rp_ ) THEN
           X( i ) = X( i ) / matrix%val( i )
         ELSE
           IF ( X( i ) /= 0.0_rp_ ) THEN
             inform%alternative = .TRUE.
             EXIT
           END IF
         END IF
       END DO
       IF ( inform%alternative ) THEN
         DO i = 1, matrix%n
           IF ( matrix%val( i ) /= 0.0_rp_ ) X( i ) = 0.0_rp_
         END DO
       END IF
       inform%status = GALAHAD_ok
       GO TO 900
     CASE ( 'SCALED_IDENTITY' )
       inform%alternative = .FALSE.
       IF ( matrix%val( 1 ) /= 0.0_rp_ ) THEN
          X( : matrix%n ) = X( : matrix%n ) / matrix%val( 1 )
       ELSE
         DO i = 1, matrix%n
           IF ( X( i ) /= 0.0_rp_ ) THEN
             inform%alternative = .TRUE.
             EXIT
           END IF
         END DO
       END IF
       inform%status = GALAHAD_ok
       GO TO 900
     CASE ( 'IDENTITY' )
       inform%alternative = .FALSE.
       inform%status = GALAHAD_ok
!      X( : matrix%n ) = X( : matrix%n )
       GO TO 900
     CASE ( 'ZERO', 'NONE' )
       inform%alternative = .FALSE.
       DO i = 1, matrix%n
         IF ( X( i ) /= 0.0_rp_ ) THEN
           inform%alternative = .TRUE.
           EXIT
         END IF
       END DO
       inform%status = GALAHAD_ok
       GO TO 900
     END SELECT

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
       CALL SPACE_resize_array( data%n, 2_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_ma77( control, data%ma77_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       IF ( control%scaling == - 2 .OR. control%scaling == - 3 ) THEN
         CALL MA77_solve_fredholm( 1_ip_, flag_out, data%n, data%X2,           &
                                   data%ma77_keep, data%ma77_control,          &
                                   data%ma77_info, scale = data%SCALE )
       ELSE
         CALL MA77_solve_fredholm( 1_ip_, flag_out, data%n, data%X2,           &
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
       CALL SPACE_resize_array( data%n, 2_ip_, data%X2, inform%status,         &
               inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
       data%X2( : data%n, 1 ) = X( : data%n )
       CALL SLS_copy_control_to_ma97( control, data%ma97_control )
       CALL CPU_time( time ) ; CALL CLOCK_time( clock )
       CALL MA97_solve_fredholm( 1_ip_, flag_out, data%X2, data%n,             &
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
!      CALL SPACE_resize_array( data%n, 2_ip_, data%X2, inform%status,         &
!              inform%alloc_status )
!      IF ( inform%status /= GALAHAD_ok ) THEN
!        inform%bad_alloc = 'sls: data%X2' ; GO TO 900 ; END IF
!      data%X2( : data%n, 1 ) = X( : data%n )
!      CALL SLS_copy_control_to_ssids( control, data%ssids_options )
!      CALL CPU_time( time ) ; CALL CLOCK_time( clock )
!      CALL SSIDS_solve_fredholm( 1_ip_, flag_out, data%X2, data%n,            &
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

     CASE ( 'pardiso', 'mkl_pardiso' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = WSMP =

     CASE ( 'wsmp' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = PaStiX =

     CASE ( 'pastix' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = MUMPS =

     CASE ( 'mumps' )
       inform%status = GALAHAD_unavailable_option
       GO TO 900

!  = LAPACK solvers POTR, SYTR or PBTR =

     CASE ( 'potr', 'sytr', 'pbtr' )

       CALL SPACE_resize_array( data%n, 1_ip_, data%X2, inform%status,         &
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
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, lda
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: PIVOTS
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( n ) :: X

!  local variables

     INTEGER ( KIND = ip_ ) :: k, km1, kp, kp1
     REAL ( KIND = rp_ ) :: akm1k, akm1, ak, denom, bkm1, bk, val

!  Replace x by L^-1 x

     status = GALAHAD_ok
     IF ( part == 'L' ) THEN
       k = 1
       DO  ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN  ! a 1 x 1 pivot
           kp = PIVOTS( K )
           IF ( kp /= k ) THEN  ! interchange X k and PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF

!  multiply by L(k)^-1, where L(k) is the transformation stored in column k of A

           IF ( k < n ) X( k + 1 : n ) = X( k + 1 : n )                        &
               - A( k + 1 : n, k ) * X( k )
           k = k + 1
         ELSE  ! a 2 x 2 pivot
           kp1 = k + 1
           kp = - PIVOTS( k )
           IF ( kp /= kp1 ) THEN  ! interchange X k+1 and -PIVOTS(k)
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
       DO  ! run backwards through the pivots
         IF ( k < 1 ) EXIT
         IF ( pivots( k ) > 0 ) THEN  ! a 1 x 1 pivot

!  multiply by L(k)^-T, where L(k) is the transformation stored in column k of A

           IF ( k < n )                                                        &
             X( k ) = X( k ) - DOT_PRODUCT( X( k + 1 : n ), A( k + 1 : n , k ) )
           kp = PIVOTS( k )
           IF ( kp /= k ) THEN  ! interchange X k and PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF
           k = k - 1

!  multiply by L(k)^-1 where L(k) is stored in columns k-1 and k of A

         ELSE  ! a 2 x 2 pivot
           IF ( k < n ) THEN
             km1 = k - 1
             kp1 = k + 1
             X( k ) = X( k ) - DOT_PRODUCT( X( kp1 : n ), A( kp1 : n, k ) )
             X( km1 ) = X( km1 ) - DOT_PRODUCT( X( kp1 : n ), A( kp1 : n, km1 ))
           END IF
           kp = - PIVOTS( k )
           IF ( kp /= k ) THEN  ! interchange X k and -PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF
           k = k - 2
         END IF
       END DO
     ELSE IF ( part == 'D' ) THEN
       k = 1
       DO  ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN  ! a 1 x 1 pivot
           X( k ) = X( k ) / A( k, k )
           k = k + 1
         ELSE  ! a 2 x 2 pivot
           kp1 = k + 1
           akm1k = A( kp1, k )
           akm1 = A( k, k ) / akm1k
           ak = A( kp1, kp1 ) / akm1k
           denom = akm1 * ak - 1.0_rp_
           bkm1 = X( k ) / akm1k
           bk = X( kp1 ) / akm1k
           X( k ) = ( ak * bkm1 - bk ) / denom
           X( kp1 ) = ( akm1 * bk - bkm1 ) / denom
           k = k + 2
         END IF
       END DO
     ELSE  ! if part = 'S'
       k = 1
       DO  ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN  ! a 1 x 1 pivot
           kp = PIVOTS( K )
           IF ( kp /= k ) THEN  ! interchange X k and PIVOTS(k)
             val =  X( k ) ; X( k ) = X( kp ) ; X( kp ) = val
           END IF

!  multiply by L(k)^-1, where L(k) is the transformation stored in column k of A

           IF ( k < n ) X( k + 1 : n ) = X( k + 1 : n )                        &
               - A( k + 1 : n, k ) * X( k )
           k = k + 1
         ELSE  ! a 2 x 2 pivot
           kp1 = k + 1
           kp = - PIVOTS( k )
           IF ( kp /= kp1 ) THEN ! interchange X k+1 and -PIVOTS(k)
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
       DO ! run forward through the pivots
         IF ( k > n ) EXIT
         IF ( PIVOTS( k ) > 0 ) THEN  ! a 1 x 1 pivot
           IF ( A( k, k ) > 0.0_rp_ ) THEN
             X( k ) = X( k ) / SQRT( A( k, k ) )
           ELSE
             status = GALAHAD_error_inertia ; EXIT
           END IF
           k = k + 1
         ELSE  ! a 2 x 2 pivot
           status = GALAHAD_error_inertia ; EXIT
         END IF
       END DO
     END IF

     RETURN

!  End of SLS_sytr_part_solve

     END SUBROUTINE SLS_sytr_part_solve

!-*-   S L S _ S Y T R _ S I N G U L A R _ S O L V E   S U B R O U T I N E   -*-

     SUBROUTINE SLS_sytr_singular_solve( n, nrhs, A, lda, PIVOTS, B, ldb,      &
                                         tiny, status )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the systems A X = B, where the factoriization A = L D L^T is found
!  by the LAPACK routine SYTRF, A is reported singular, and B is input in X

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, nrhs, lda, ldb
     INTEGER ( KIND = ip_ ), INTENT( INOUT ) :: status
     REAL ( KIND = rp_ ), INTENT( IN ) :: tiny
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( n ) :: PIVOTS
     REAL ( KIND = rp_ ), INTENT( IN ), DIMENSION( lda, n ) :: A
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( ldb, nrhs ) :: B

!  local variables

     INTEGER ( KIND = ip_ ) :: j, k, kp
     REAL ( KIND = rp_ ) :: ak, akm1, akm1k, bk, bkm1, denom

!  first solve L*D*X = B, overwriting B with X. k is the main loop index,
!  increasing from 1 to n in steps of 1 or 2, depending on the size of
!  the diagonal blocks

     k = 1
     DO

!  if k > n, exit from loop

       IF ( k > n ) EXIT

!  1 x 1 diagonal block:  interchange rows k and PIVOTS(k)

       IF ( PIVOTS( k ) > 0 ) THEN
         kp = PIVOTS( k )
         IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), ldb, B( kp, : ), ldb )

!  multiply by inv(L(k)), where L(k) is the transformation stored in column
!  k of A

         IF ( k < n ) CALL GER( n - k, nrhs, - 1.0_rp_,                        &
                                A( k + 1 : , k ), 1_ip_,                       &
                                B( k, : ), ldb, B( k + 1 : , : ), ldb )

!  multiply by the inverse of the diagonal block if the diagonal is sufficiently
!  nonzero. If not, check to see if corresponding entries in B are also almost
!  zero, and skip the scaling. Otherwise flag the system as inconsistent

         IF ( ABS( A( k, k ) ) > tiny ) THEN
           CALL SCAL( nrhs, 1.0_rp_ / A( k, k ), B( k, : ), ldb )
         ELSE
           IF ( MAXVAL( ABS( B( k, 1 : nrhs ) ) ) > tiny ) THEN
             status = GALAHAD_error_primal_infeasible ; RETURN
           END IF
         END IF
         k = k + 1

!  2 x 2 diagonal block: interchange rows k + 1 and - PIVOTS(k)

       ELSE
         kp = - PIVOTS( k )
         IF ( kp /= k + 1 ) CALL SWAP( nrhs, B( k + 1, : ), ldb,               &
                                       B( kp, : ), ldb )

!  Multiply by inv(L(k)), where L(k) is the transformation stored in columns
!  k and k + 1 of A

         IF ( k < n - 1 ) THEN
           CALL GER( n - k - 1_ip_, nrhs, - 1.0_rp_,  A( k + 2 : , k ), 1_ip_, &
                     B( k, : ), ldb, B( k + 2 : , : ), ldb )
           CALL GER( n - k - 1_ip_, nrhs, - 1.0_rp_,  A( k + 2 : , k + 1 ),    &
                     1_ip_, B( k + 1, : ), ldb, B( k + 2 : , : ), ldb )
         END IF

!  multiply by the inverse of the diagonal block if its scaled determinant
!  is sufficiently nonzero. If not, check to see if corresponding entries
!  in B are also almost zero, and skip the scaling. Otherwise flag the system
!  as inconsistent

         akm1k = A( k + 1, k )
         akm1 = A( k, k ) / akm1k
         ak = A( k + 1, k + 1 ) / akm1k
         denom = akm1 * ak - 1.0_rp_
         IF ( ABS( denom ) > tiny ) THEN
           DO j = 1, nrhs
             bkm1 = B( k, j ) / akm1k
             bk = B( k + 1, j ) / akm1k
             B( k, j ) = ( ak * bkm1 - bk ) / denom
             B( k + 1, j ) = ( akm1 * bk - bkm1 ) / denom
           END DO
         ELSE
           IF ( MAXVAL( ABS( B( k : k + 1, 1 : nrhs ) ) ) > tiny ) THEN
             status = GALAHAD_error_primal_infeasible ; RETURN
           END IF
         END IF
         k = k + 2
       END IF
     END DO

!  next solve L**T * X = B, overwriting B with X. k is the main loop index,
!  decreasing from n to 1 in steps of 1 or 2, depending on the size of the
!  diagonal blocks

     k = n
     DO

!  if K < 1, exit from loop

       IF ( k < 1 ) EXIT

!  1 x 1 diagonal block: multiply by inv(L**T(K)), where L(K) is the
!  transformation stored in column k of A

       IF ( PIVOTS( k ) > 0 ) THEN
         IF ( k < n ) CALL GEMV( 'T', n - k, nrhs, - 1.0_rp_,                  &
                                 B( k + 1 : , : ), ldb, A( k + 1 : , k ),      &
                                 1_ip_, 1.0_rp_,  B( k, : ), ldb )

!  interchange rows k and PIVOTS(k)

         kp = PIVOTS( k )
         IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), ldb, B( kp, : ), ldb )
         k = k - 1

!  2x2 diagonal block: multiply by inv(L**T(k-1)), where L(k-1) is the
!  transformation stored in columns k-1 and k of A

       ELSE
         IF ( k < n ) THEN
           CALL GEMV( 'T', n - k, nrhs, - 1.0_rp_,  B( k + 1 : , : ), ldb,     &
                       A( k + 1 : , k ), 1_ip_, 1.0_rp_,  B( k, : ), ldb )
           CALL GEMV( 'T', n - k, nrhs, - 1.0_rp_,  B( k + 1 : , : ), ldb,     &
                       A( k + 1 : , k - 1 ), 1_ip_, 1.0_rp_,                   &
                       B( k - 1, : ), ldb )
         END IF

!  interchange rows k and - PIVOTS(k)

         kp = - PIVOTS( k )
         IF ( kp /= k ) CALL SWAP( nrhs, B( k, : ), ldb, B( kp, : ), ldb )
         k = k - 2
       END IF
     END DO
     status = GALAHAD_ok

     RETURN

!  End of SLS_sytr_singular_solve

     END SUBROUTINE SLS_sytr_singular_solve

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

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: ROW, COL
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: dup, oor, upper, missing_diagonals
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( ne, 2 ) :: MAP
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n + 1 ) :: PTR

!  local variables

     INTEGER ( KIND = ip_ ) :: i, j, k, l, ll
     INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: IW

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

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, ne
     INTEGER ( KIND = ip_ ), INTENT( IN ), DIMENSION( ne ) :: ROW, COL
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: dup, oor, upper, missing_diagonals
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status, alloc_status
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( ne ) :: MAP
     INTEGER ( KIND = ip_ ), INTENT( OUT ), DIMENSION( n + 1 ) :: PTR

!  local variables

     INTEGER ( KIND = ip_ ) :: i, j, jj, j_old, k, l, ll, err, pt, size
     INTEGER ( KIND = ip_ ), DIMENSION( n + 1 ) :: IW
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: COLS, ENTS
     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP2

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

         CALL SORT_quicksort( size, COLS, err, ix = ENTS )

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

!!-*-*-  S L S _ M A P _ T O _ E X T E N D E D _ C S R  S U B R O U T I N E -*-
!
!     SUBROUTINE SLS_map_to_extended_csr( matrix, map, ptr, dup, oor,          &
!                                         missing_diagonals )
!
!!  Compute a mapping from the co-ordinate scheme to the extended row storage
!!  scheme used by MA77. The mapping records out-of-range components
!!  and flags duplicates for summation.
!!
!!  Entry l is mapped to positions MAP( l, j ) for j = 1 and 2, l = 1, ne.
!!  If MAP( l, 2 ) = 0, the entry is on the diagonal, and thus only mapped
!!  to the single location MAP( l, 1 ). If MAP( l, 1 ) = 0, the entry is out of
!!  range. If MAP( l, j ) < 0, the entry should be added to that in - MAP( l, j)
!!
!!  dup gives the number of duplicates, oor is the number of out-of-rangers and
!!  missing_diagonals records the number of rows without a diagonal entry
!
!! dummy arguments
!
!     TYPE ( SMT_type ), INTENT( IN ) :: matrix
!     INTEGER ( KIND = ip_ ), INTENT( out ) :: dup, oor, missing_diagonals
!     INTEGER ( KIND = ip_ ), INTENT( out ), DIMENSION( matrix%ne, 2 ) :: MAP
!     INTEGER ( KIND = ip_ ), INTENT( out ), DIMENSION( matrix%n + 1 ) :: PTR
!
!!  local variables
!
!     INTEGER ( KIND = ip_ ) :: i, j, k, l, ll
!     INTEGER ( KIND = ip_ ), DIMENSION( matrix%n + 1 ) :: IW
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
!!-*-*-   S L S _ M A P _ T O _ S O R T E D _ C S R  S U B R O U T I N E  -*-*-
!
!     SUBROUTINE SLS_map_to_sorted_csr( matrix, map, ptr, dup, oor,            &
!                                       missing_diagonals, status,             &
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
!     INTEGER ( KIND = ip_ ), INTENT( out ) :: dup, oor, missing_diagonals
!     INTEGER ( KIND = ip_ ), INTENT( out ) :: status, alloc_status
!     INTEGER ( KIND = ip_ ), INTENT( out ), DIMENSION( matrix%ne ) :: MAP
!     INTEGER ( KIND = ip_ ), INTENT( out ), DIMENSION( matrix%n + 1 ) :: PTR
!
!!  local variables
!
!     INTEGER ( KIND = ip_ ) :: i, j, jj, j_old, k, l, ll, err, pt, size
!     INTEGER ( KIND = ip_ ), DIMENSION( matrix%n + 1 ) :: IW
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: COLS, ENTS
!     INTEGER ( KIND = ip_ ), ALLOCATABLE, DIMENSION( : ) :: MAP2
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
!         CALL SORT_quicksort( size, COLS, err, ix = ENTS )
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

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

!-  G A L A H A D -  S L S _ a n a l y s e _ m a t r i x _ S U B R O U T I N E -

     SUBROUTINE SLS_analyse_matrix( control, data, status, n,                  &
                                    matrix_type, matrix_ne,                    &
                                    matrix_row, matrix_col, matrix_ptr )

!  import structural matrix data into internal storage, and analyse the
!  structure prior to factorization

!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to SLS_solve
!
!  data is a scalar variable of type SLS_full_data_type used for internal data
!
!  status is a scalar variable of type default integer that indicates the
!   success or otherwise of the import and analysis. Possible values are:
!
!    0. The analysis was succesful, and the package is ready for the
!       factorization phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in statusrm.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0 or requirement that type contains
!       its relevant string 'DENSE', 'COORDINATE' or 'SPARSE_BY_ROWS',
!       has been violated.
!
!  n is a scalar variable of type default integer, that holds the number of
!   rows (and columns) of the matrix A
!
!  matrix_type is a character string that specifies the storage scheme used
!   for A. It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
!   lower or upper case variants are allowed.
!
!  matrix_ne is a scalar variable of type default integer, that holds the
!   number of entries in the  lower triangular part of A in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other schemes.
!
!  matrix_row is a rank-one array of type default integer, that holds
!   the row indices of the  lower triangular part of A in the sparse
!   co-ordinate storage scheme. It need not be set for any of the other
!   three schemes, and in this case can be of length 0
!
!  matrix_col is a rank-one array of type default integer,
!   that holds the column indices of the  lower triangular part of A in either
!   the sparse co-ordinate, or the sparse row-wise storage scheme. It need not
!   be set when the dense, diagonal, scaled identity, identity or zero schemes
!   are used, and in this case can be of length 0
!
!  matrix_ptr is a rank-one array of dimension n+1 and type default
!   integer, that holds the starting position of  each row of the  lower
!   triangular part of A, as well as the total number of entries plus one,
!   in the sparse row-wise storage scheme. It need not be set when the
!   other schemes are used, and in this case can be of length 0
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLS_control_type ), INTENT( INOUT ) :: control
     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( IN ) :: n, matrix_ne
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: matrix_type
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL,                         &
                                             INTENT( IN ) :: matrix_row
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL,                         &
                                             INTENT( IN ) :: matrix_col
     INTEGER ( KIND = ip_ ), DIMENSION( : ), OPTIONAL,                         &
                                             INTENT( IN ) :: matrix_ptr

!  copy control to data

     data%SLS_control = control

!  set A appropriately in the smt storage type

     data%matrix%n = n ; data%matrix%m = n
     SELECT CASE ( matrix_type )
     CASE ( 'coordinate', 'COORDINATE' )
      IF ( .NOT. ( PRESENT( matrix_row ) .AND. PRESENT( matrix_col ) ) ) THEN
         data%sls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%matrix%type, 'COORDINATE',                           &
                     data%sls_inform%alloc_status )
       data%matrix%ne = matrix_ne

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%row,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%col,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%matrix%row( : data%matrix%ne ) = matrix_row( : data%matrix%ne )
         data%matrix%col( : data%matrix%ne ) = matrix_col( : data%matrix%ne )
       ELSE
         data%matrix%row( : data%matrix%ne )                                   &
           = matrix_row( : data%matrix%ne ) + 1
         data%matrix%col( : data%matrix%ne )                                   &
           = matrix_col( : data%matrix%ne ) + 1
       END IF

     CASE ( 'sparse_by_rows', 'SPARSE_BY_ROWS' )
      IF ( .NOT. ( PRESENT( matrix_ptr ) .AND. PRESENT( matrix_col ) ) ) THEN
         data%sls_inform%status = GALAHAD_error_optional
         GO TO 900
       END IF
       CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS',                       &
                     data%sls_inform%alloc_status )
       IF ( data%f_indexing ) THEN
         data%matrix%ne = matrix_ptr( n + 1 ) - 1
       ELSE
         data%matrix%ne = matrix_ptr( n + 1 )
       END IF

       CALL SPACE_resize_array( n + 1_ip_, data%matrix%ptr,                    &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%col,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%matrix%ptr( : n + 1 ) = matrix_ptr( : n + 1 )
         data%matrix%col( : data%matrix%ne ) = matrix_col( : data%matrix%ne )
       ELSE
         data%matrix%ptr( : n + 1 ) = matrix_ptr( : n + 1 ) + 1
         data%matrix%col( : data%matrix%ne )                                   &
           = matrix_col( : data%matrix%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%matrix%type, 'DENSE',                                &
                     data%sls_inform%alloc_status )
       data%matrix%ne = ( n * ( n + 1 ) ) / 2

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

     CASE ( 'diagonal', 'DIAGONAL' )
       CALL SMT_put( data%matrix%type, 'DIAGONAL',                             &
                     data%sls_inform%alloc_status )
       data%matrix%ne = n

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

     CASE ( 'scaled_identity', 'SCALED_IDENTITY' )
       CALL SMT_put( data%matrix%type, 'SCALED_IDENTITY',                      &
                     data%sls_inform%alloc_status )
       data%matrix%ne = 1

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%sls_inform%status, data%sls_inform%alloc_status )
       IF ( data%sls_inform%status /= 0 ) GO TO 900

     CASE ( 'identity', 'IDENTITY' )
       CALL SMT_put( data%matrix%type, 'IDENTITY',                             &
                     data%sls_inform%alloc_status )
       data%matrix%ne = 0

     CASE ( 'zero', 'ZERO', 'none', 'NONE' )
       CALL SMT_put( data%matrix%type, 'ZERO',                                 &
                     data%sls_inform%alloc_status )
       data%matrix%ne = 0

     CASE DEFAULT
       data%sls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT

!  analyse the sparsity structure of the matrix prior to factorization

     CALL SLS_analyse( data%matrix, data%sls_data, data%sls_control,           &
                       data%sls_inform )

     status = data%sls_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%sls_inform%status
     RETURN

!  End of subroutine SLS_analyse_matrix

     END SUBROUTINE SLS_analyse_matrix

!-  G A L A H A D -  S L S _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE SLS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See SLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLS_control_type ), INTENT( IN ) :: control
     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  set control in internal data

     data%sls_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SLS_reset_control

     END SUBROUTINE SLS_reset_control

! G A L A H A D - S L S _ f a c t o r i z e _ m a t r i x  S U B R O U T I N E -

     SUBROUTINE SLS_factorize_matrix( data, status, matrix_val )

!  factorize the matrix A

!  Arguments are as follows:

!  data is a scalar variable of type SLS_full_data_type used for internal data
!
!  status is a scalar variable of type default integer that indicates the
!   success or otherwise of the factorization. Possible values are:
!
!    0. The factorization was succesful, and the package is ready for the
!       solve phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!-  20. The matrix is not positive definite while the solver used expected
!       it to be.
!  -29. This option is not available with this solver.
!  -32. More than control%max integer factor size words of internal integer
!       storage are required for in-core factorization.
!  -34. The package PARDISO failed; check the solver-specific information
!       components inform%pardiso_iparm and inform%pardiso_dparm along with
!       PARDISO’s documentation for more details.
!  -35. The package WSMP failed; check the solver-specific information
!       components inform%wsmp_iparm and inform%wsmp_dparm along with
!       WSMP’s documentation for more details.
!  -36. The scaling package HSL MC64 failed; check the solver-specific
!       information component inform%mc64_info along with HSL MC64’s
!       documentation for more details.
!  -37. The scaling package MC77 failed; check the solver-specific information
!      components inform%mc77_info and inform%mc77_rinfo along with MC77’s
!      documentation for more details.
!  -43. A direct-access file error occurred. See the value of
!      inform%ma77_info%flag for more details.
!  -50. A solver-specific error occurred; check the solver-specific
!       information component of inform along with the solver’s documentation
!       for more details.
!
!  matrix_val is a rank-one array of type default real, that holds the
!   values of  the  lower triangular part of A input in precisely the same
!   order as those for the row and column indices in SLS_analyse_matrix

!  See SLS_form_and_factorize for a description of the required arguments

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: matrix_val

!  save the values of matrix

     IF ( data%matrix%ne > 0 )                                                 &
       data%matrix%val( : data%matrix%ne ) = matrix_val( : data%matrix%ne )

!  factorize the matrix

     CALL SLS_factorize( data%matrix, data%sls_data, data%sls_control,         &
                         data%sls_inform )

     status = data%sls_inform%status
     RETURN

!  end of subroutine SLS_factorize_matrix

     END SUBROUTINE SLS_factorize_matrix

!--  G A L A H A D -  S L S _ s o l v e _ s y s t e m   S U B R O U T I N E  -

     SUBROUTINE SLS_solve_system( data, status, SOL )

!  solve the linear system A x = b

!  Arguments are as follows:

!  data is a scalar variable of type SLS_full_data_type used for internal data
!
!  status is a scalar variable of type default integer that indicates the
!   success or otherwise of the import. Possible values are:
!
!    0. The solve was succesful
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!  -50. A solver-specific error occurred; check the solver-specific
!       information component of inform along with the solver’s documentation
!       for more details.
!
!  SOL is a rank-one array of type default real, that holds the RHS b on
!      entry, and the solution x on a successful exit

!  See SLS_solve for a description of the required arguments

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: SOL

!  solve the block linear system

     CALL SLS_solve( data%matrix, SOL, data%sls_data, data%sls_control,        &
                     data%sls_inform )

     status = data%sls_inform%status
     RETURN

!  end of subroutine SLS_solve_system

     END SUBROUTINE SLS_solve_system

!--  G A L A H A D -  S L S _ p a r t i a l _ s o l v e  S U B R O U T I N E -

     SUBROUTINE SLS_partial_solve( part, data, status, SOL )

!  Given the factorization A = L D U (with U = L^T), solve one of the
!  linear system M x = b, where M is L, D, U or S = L * SQRT(D), and
!  SOL holds the right-hand side b on input, and the solution x on output.
!  See SLS_part_solve for a description of the required arguments

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     CHARACTER ( LEN = 1 ), INTENT( IN ) :: part
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = rp_ ), INTENT( INOUT ), DIMENSION( : ) :: SOL

!  solve the block linear system

     CALL SLS_part_solve( part, SOL, data%sls_data, data%sls_control,          &
                          data%sls_inform )

     status = data%sls_inform%status
     RETURN

!  end of subroutine SLS_partial_solve

     END SUBROUTINE SLS_partial_solve

!-*-  G A L A H A D -  S L S _ i n f o r m a t i o n   S U B R O U T I N E  -*-

     SUBROUTINE SLS_information( data, inform, status )

!  return solver information during or after solution by SLS
!  See SLS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( SLS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_inform_type ), INTENT( OUT ) :: inform
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%sls_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine SLS_information

     END SUBROUTINE SLS_information

!  End of module GALAHAD_SLS

   END MODULE GALAHAD_SLS_precision
