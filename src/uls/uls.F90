! THIS VERSION: GALAHAD 4.1 - 2022-10-16 AT 11:50 GMT.

!-*-*-*-*-*-*-*-*- G A L A H A D _ U L S    M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.4. August 24th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_ULS_double

!     ---------------------------------------
!     |                                     |
!     |  Provide interfaces from various    |
!     |  packages to allow the solution of  |
!     |                                     |
!     |     Unsymmetric Linear Systems      |
!     |                                     |
!     |  Packages currently supported:      |
!     |                                     |
!     |          MA28/GLS                   |
!     |          MA48                       |
!     |          GETRF from LAPACK          |
!     |                                     |
!     ---------------------------------------

     USE GALAHAD_SYMBOLS
     USE GALAHAD_SORT_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_SPECFILE_double
     USE GALAHAD_SMT_double
     USE GALAHAD_STRING, ONLY: STRING_put, STRING_get, STRING_lower_word
     USE GALAHAD_GLS_double
     USE GALAHAD_LAPACK_interface, ONLY : GETRF, GETRS
     USE HSL_ZD11_double
     USE HSL_MA48_double

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: ULS_initialize, ULS_factorize, ULS_solve,                       &
               ULS_fredholm_alternative, ULS_terminate,                        &
               ULS_enquire, ULS_read_specfile, ULS_initialize_solver,          &
               ULS_full_initialize, ULS_full_terminate,                        &
               ULS_factorize_matrix, ULS_solve_system,                         &
               ULS_reset_control, ULS_information,                             &
               SMT_type, SMT_get, SMT_put

!----------------------
!   I n t e r f a c e s
!----------------------

     INTERFACE ULS_initialize
       MODULE PROCEDURE ULS_initialize, ULS_full_initialize
     END INTERFACE ULS_initialize

     INTERFACE ULS_terminate
       MODULE PROCEDURE ULS_terminate, ULS_full_terminate
     END INTERFACE ULS_terminate

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

     INTEGER, PARAMETER :: min_real_factor_size_default = 10000
     INTEGER, PARAMETER :: min_integer_factor_size_default = 10000
     INTEGER, PARAMETER :: blas_block_size_factor_default = 16
     INTEGER, PARAMETER :: blas_block_size_solve_default = 16

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: ULS_control_type

!  unit for error messages

       INTEGER :: error = 6

!  unit for warning messages

       INTEGER :: warning = 6

!  unit for monitor output

       INTEGER :: out = 6

!  controls level of diagnostic output

       INTEGER :: print_level = 0

!  controls level of diagnostic output from external solver

       INTEGER :: print_level_solver = 0

!  prediction of factor by which the fill-in will exceed the initial
!  number of nonzeros in A

       INTEGER :: initial_fill_in_factor = 3

!  initial size for real array for the factors and other data

       INTEGER :: min_real_factor_size = min_real_factor_size_default

!  initial size for integer array for the factors and other data

       INTEGER :: min_integer_factor_size = min_integer_factor_size_default

!  maximum size for real array for the factors and other data

       INTEGER ( KIND = long ) :: max_factor_size = HUGE( 0 )

!  level 3 blocking in factorize

       INTEGER :: blas_block_size_factorize = blas_block_size_factor_default

!  level 2 and 3 blocking in solve

       INTEGER :: blas_block_size_solve = blas_block_size_solve_default

!  pivot control:
!   1  Threshold Partial Pivoting is desired
!   2  Threshold Rook Pivoting is desired
!   3  Threshold Complete Pivoting is desired
!   4  Threshold Symmetric Pivoting is desired
!   5  Threshold Diagonal Pivoting is desired

       INTEGER :: pivot_control = 1

!  number of rows/columns pivot selection restricted to (0 = no restriction)

       INTEGER :: pivot_search_limit = 0

!  the minimum permitted size of blocks within the block-triangular form

       INTEGER :: minimum_size_for_btf = 1

!  maximum number of iterative refinements allowed

       INTEGER :: max_iterative_refinements = 0

!  stop if the matrix is found to be structurally singular

       LOGICAL :: stop_if_singular = .FALSE.

!  factor by which arrays sizes are to be increased if they are too small

       REAL ( KIND = wp ) :: array_increase_factor = 2.0_wp

!  switch to full code when the density exceeds this factor

       REAL ( KIND = wp ) :: switch_to_full_code_density = 0.5_wp

!  if previously allocated internal workspace arrays are greater than
!  array_decrease_factor times the currently required sizes, they are reset
!  to current requirements

       REAL ( KIND = wp ) :: array_decrease_factor = 2.0_wp

!  pivot threshold

       REAL ( KIND = wp ) :: relative_pivot_tolerance = 0.01_wp

!  any pivot small than this is considered zero

       REAL ( KIND = wp ) :: absolute_pivot_tolerance = EPSILON( 1.0_wp )

!  any entry smaller than this in modulus is reset to zero

       REAL ( KIND = wp ) :: zero_tolerance = 0.0_wp

!  refinement will cease as soon as the residual ||Ax-b|| falls below
!     max( acceptable_residual_relative * ||b||, acceptable_residual_absolute )

       REAL ( KIND = wp ) :: acceptable_residual_relative = 10.0_wp * epsmch
       REAL ( KIND = wp ) :: acceptable_residual_absolute = 10.0_wp * epsmch

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

       CHARACTER ( LEN = 30 ) :: prefix = '""                            '
     END TYPE ULS_control_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

     TYPE, PUBLIC :: ULS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -26  unknown solver
!   -29  unavailable option
!   -31  input order is not a permutation or is faulty in some other way
!   -32  error with integer workspace
!   -33  error with real workspace
!   -34  error from PARDISO
!   -50  solver-specific error; see the solver's info parameter

       INTEGER :: status = 0

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  further information on failure

       INTEGER :: more_info = 0

!  number of indices out-of-range

       INTEGER ( KIND = long ) :: out_of_range = 0

!  number of duplicates

       INTEGER ( KIND = long ) :: duplicates = 0

!  number of entries dropped during the factorization

       INTEGER ( KIND = long ) :: entries_dropped = 0

!  predicted or actual number of reals and integers to hold factors

       INTEGER ( KIND = long ) :: workspace_factors  = - 1

!  number of compresses of data required

       INTEGER :: compresses = - 1

!  number of entries in factors

       INTEGER ( KIND = long ) :: entries_in_factors = - 1_long

!  estimated rank of the matrix

       INTEGER :: rank = - 1

!  structural rank of the matrix

       INTEGER :: structural_rank = - 1

!  pivot control:
!   1  Threshold Partial Pivoting has been used
!   2  Threshold Rook Pivoting has been used
!   3  Threshold Complete Pivoting has been desired
!   4  Threshold Symmetric Pivoting has been desired
!   5  Threshold Diagonal Pivoting has been desired

       INTEGER :: pivot_control = - 1

!  number of iterative refinements performed

       INTEGER :: iterative_refinements = 0

!  has an "alternative" y: A^T y = 0 and yT b > 0 been found when trying to
!  solve A x = b ?

       LOGICAL :: alternative = .FALSE.

!  name of linear solver used

       CHARACTER ( LEN = len_solver ) :: solver = REPEAT( ' ', len_solver )

!  the output array from

       TYPE ( GLS_ainfo ) :: gls_ainfo
       TYPE ( GLS_finfo ) :: gls_finfo
       TYPE ( GLS_sinfo ) :: gls_sinfo

!  the output array from

       TYPE ( MA48_ainfo ) :: ma48_ainfo
       TYPE ( MA48_finfo ) :: ma48_finfo
       TYPE ( MA48_sinfo ) :: ma48_sinfo

!  the output scalars and arrays from LAPACK routines

       INTEGER :: lapack_error = 0

     END TYPE ULS_inform_type

!  ...................
!   data derived type
!  ...................

     TYPE, PUBLIC :: ULS_data_type
       PRIVATE
       INTEGER :: len_solver = - 1
       INTEGER :: m, n, ne, matrix_ne, pardiso_mtype
       CHARACTER ( LEN = len_solver ) :: solver = '                    '
       LOGICAL :: set_res = .FALSE.

       INTEGER, DIMENSION( 64 ) :: PARDISO_PT
       INTEGER, DIMENSION( 64 ) :: pardiso_iparm = - 1
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROWS, MAPS, PIVOTS
       INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: MAP
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RESIDUALS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RESIDUALS_zero
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCALE
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: RHS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: B2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: RES2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: matrix_dense

       TYPE ( ZD11_type ) :: matrix

       TYPE ( GLS_factors ) :: gls_factors
       TYPE ( GLS_control ) :: gls_control
       TYPE ( GLS_ainfo ) :: gls_ainfo
       TYPE ( GLS_finfo ) :: gls_finfo
       TYPE ( GLS_sinfo ) :: gls_sinfo

       TYPE ( MA48_factors ) :: ma48_factors
       TYPE ( MA48_control ) :: ma48_control
       TYPE ( MA48_ainfo ) :: ma48_ainfo
       TYPE ( MA48_finfo ) :: ma48_finfo
       TYPE ( MA48_sinfo ) :: ma48_sinfo

     END TYPE ULS_data_type

     TYPE, PUBLIC :: ULS_full_data_type
       LOGICAL :: f_indexing = .TRUE.
       TYPE ( ULS_data_type ) :: ULS_data
       TYPE ( ULS_control_type ) :: ULS_control
       TYPE ( ULS_inform_type ) :: ULS_inform
       TYPE ( SMT_type ) :: matrix
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
     END TYPE ULS_full_data_type

!--------------------------------
!   I n t e r f a c e  B l o c k
!--------------------------------

!!$      INTERFACE
!!$        SUBROUTINE pardiso( PT, maxfct, mnum, mtype, phase, n, A, IA, JA,   &
!!$                            PERM, nrhs, IPARM, msglvl, B, X, error )
!!$        INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
!!$        INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: PT
!!$        INTEGER, INTENT( IN ) :: maxfct, mnum, mtype, phase, n, nrhs, msglvl
!!$        INTEGER, INTENT( OUT ) :: error
!!$        INTEGER, INTENT( INOUT ), DIMENSION( 64 ) :: IPARM
!!$        INTEGER, INTENT( IN ), DIMENSION( n ) :: PERM
!!$        INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: IA
!!$        INTEGER, INTENT( IN ), DIMENSION( : ) :: JA
!!$        REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: A
!!$        REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n , nrhs ) :: B
!!$        REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n , nrhs ) :: X
!!$        END SUBROUTINE pardiso
!!$      END INTERFACE

   CONTAINS

!-*-*-*-*-*-   U L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE ULS_initialize( solver, data, control, inform, check )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Set initial values, including default control data and solver used, for ULS.
!  This routine must be called before the first call to ULS_factorize

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( OUT ) :: control
     TYPE ( ULS_inform_type ), INTENT( OUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: check

!  initialize the solver-specific data

     CALL ULS_initialize_solver( solver, data, inform, check )

!  record the name of the solver used

     inform%solver = REPEAT( ' ', len_solver )
     inform%solver( 1 : data%len_solver ) = data%solver( 1 : data%len_solver )

!  initialize solver-specific controls

!     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

!     CASE ( 'gls', 'ma28' )
       CALL GLS_initialize( data%gls_factors, data%gls_control )
!      control%print_level = data%gls_control%ldiag
       control%relative_pivot_tolerance = data%gls_control%u

!  = MA48 =

!     CASE ( 'ma48' )
       CALL MA48_initialize( data%ma48_factors, data%ma48_control )
!      control%print_level = data%ma48_control%ldiag
!       control%relative_pivot_tolerance = data%ma48_control%u
!     END SELECT

     RETURN

!  End of ULS_initialize

     END SUBROUTINE ULS_initialize

!- G A L A H A D -  S B L S _ F U L L _ I N I T I A L I Z E  S U B R O U T I N E

     SUBROUTINE ULS_full_initialize( solver, data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Provide default values for ULS controls

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
     TYPE ( ULS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( OUT ) :: control
     TYPE ( ULS_inform_type ), INTENT( OUT ) :: inform

     CALL ULS_initialize( solver, data%uls_data, control, inform )

     RETURN

!  End of subroutine ULS_full_initialize

     END SUBROUTINE ULS_full_initialize

!-*-*-*-*-*-   U L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE ULS_initialize_solver( solver, data, inform, check )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Set initial values, including default control data and solver used, for ULS.
!  This routine must be called before the first call to ULS_factorize

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     CHARACTER ( LEN = * ), INTENT( IN ) :: solver
     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_inform_type ), INTENT( OUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: check

!  local variables

     LOGICAL :: check_available
     INTEGER, DIMENSION( 10 ) :: ICNTL_ma33
     REAL ( KIND = wp ), DIMENSION( 5 ) :: CNTL_ma33
     TYPE ( MA48_control ) :: control_ma48

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

!  initialize solver-specific controls

  10 CONTINUE
     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

     CASE ( 'gls', 'ma33' )
       IF ( check_available ) THEN ! if ma33 is not availble, use getr instead
         CALL MA33ID( ICNTL_ma33, CNTL_ma33 )
         IF ( ICNTL_ma33( 4 ) == - 1 ) THEN
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 4
           data%solver( 1 : data%len_solver ) = 'getr'
           GO TO 10
         END IF
       END IF

!  = MA48 =

     CASE ( 'ma48' )
       IF ( check_available ) THEN ! if ma48 is not availble, try sils instead
         CALL MA48_initialize( control = control_ma48 )
         IF ( control_ma48%lp == - 1 ) THEN
           data%solver = REPEAT( ' ', len_solver )
           data%len_solver = 4
           data%solver( 1 : data%len_solver ) = 'gls'
           GO TO 10
         END IF
       END IF

!  = GETR =

     CASE ( 'getr' )

!  = unavailable solver =

     CASE DEFAULT
       inform%status = GALAHAD_error_unknown_solver
     END SELECT

     data%set_res = .FALSE.
     inform%status = GALAHAD_ok

     RETURN

!  End of ULS_initialize_solver

     END SUBROUTINE ULS_initialize_solver

!-*-*-*-*-   U L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE ULS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by ULS_initialize could (roughly)
!  have been set as:

! BEGIN ULS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  warning-printout-device                           6
!  printout-device                                   6
!  print-level                                       0
!  print-level-solver                                0
!  minimum-block-size-for-btf                        1
!  blas-block-for-size-factorize                     16
!  blas-block-size-for-solve                         16
!  initial-fill-in-factor                            3
!  minimum-real-factor-size                          10000
!  minimum-integer-factor-size                       10000
!  maximum-factor-size                               2147483647
!  pivot-control                                     1
!  pivot-search-limit                                0
!  max-iterative-refinements                         0
!  array-increase-factor                             2.0
!  array-decrease-factor                             2.0
!  stop-if-singular                                  NO
!  relative-pivot-tolerance                          0.01
!  absolute-pivot-tolerance                          2.0D-16
!  zero-tolerance                                    0.0
!  switch-to-full-code-density                       0.5
!  acceptable-residual-relative                      2.0D-15
!  acceptable-residual-absolute                      2.0D-15
!  output-line-prefix                                ""
! END ULS SPECIFICATIONS

!  Dummy arguments

     TYPE ( ULS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: warning = error + 1
     INTEGER, PARAMETER :: out = warning + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: print_level_solver = print_level + 1
     INTEGER, PARAMETER :: minimum_size_for_btf = print_level_solver + 1
     INTEGER, PARAMETER :: blas_block_size_factorize = minimum_size_for_btf + 1
     INTEGER, PARAMETER :: blas_block_size_solve = blas_block_size_factorize + 1
     INTEGER, PARAMETER :: initial_fill_in_factor = blas_block_size_solve + 1
     INTEGER, PARAMETER :: min_real_factor_size = initial_fill_in_factor + 1
     INTEGER, PARAMETER :: min_integer_factor_size = min_real_factor_size + 1
     INTEGER, PARAMETER :: max_factor_size = min_integer_factor_size + 1
     INTEGER, PARAMETER :: pivot_control = max_factor_size + 1
     INTEGER, PARAMETER :: pivot_search_limit = pivot_control + 1
     INTEGER, PARAMETER :: max_iterative_refinements = pivot_search_limit + 1
     INTEGER, PARAMETER :: array_increase_factor = max_iterative_refinements + 1
     INTEGER, PARAMETER :: array_decrease_factor = array_increase_factor + 1
     INTEGER, PARAMETER :: stop_if_singular = array_decrease_factor + 1
     INTEGER, PARAMETER :: relative_pivot_tolerance = stop_if_singular + 1
     INTEGER, PARAMETER :: absolute_pivot_tolerance =                          &
                             relative_pivot_tolerance + 1
     INTEGER, PARAMETER :: zero_tolerance = absolute_pivot_tolerance + 1
     INTEGER, PARAMETER :: switch_to_full_code_density = zero_tolerance + 1
     INTEGER, PARAMETER :: acceptable_residual_relative =                      &
                             switch_to_full_code_density + 1
     INTEGER, PARAMETER :: acceptable_residual_absolute =                      &
                             acceptable_residual_relative + 1
     INTEGER, PARAMETER :: prefix = acceptable_residual_absolute + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 3 ), PARAMETER :: specname = 'ULS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( warning )%keyword = 'warning-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( print_level_solver )%keyword = 'print-level-solver'
     spec( minimum_size_for_btf )%keyword = 'minimum-block-size-for-btf'
     spec( initial_fill_in_factor )%keyword = 'initial-fill-in-factor'
     spec( min_real_factor_size )%keyword = 'minimum-real-factor-size'
     spec( min_integer_factor_size )%keyword = 'minimum-integer-factor-size'
     spec( max_factor_size )%keyword = 'maximum-factor-size'
     spec( blas_block_size_factorize )%keyword = 'blas-block-for-size-factorize'
     spec( blas_block_size_solve )%keyword = 'blas-block-size-for-solve'
     spec( pivot_control )%keyword = 'pivot-control'
     spec( pivot_search_limit )%keyword = 'pivot-search-limit'
     spec( max_iterative_refinements )%keyword = 'max-iterative-refinements'
     spec( switch_to_full_code_density )%keyword = 'switch-to-full-code-density'

!  Real key-words

     spec( array_increase_factor )%keyword = 'array-increase-factor'
     spec( array_decrease_factor )%keyword = 'array-decrease-factor'
     spec( relative_pivot_tolerance )%keyword = 'relative-pivot-tolerance'
     spec( absolute_pivot_tolerance )%keyword = 'absolute-pivot-tolerance'
     spec( zero_tolerance )%keyword = 'zero-tolerance'
     spec( acceptable_residual_relative )%keyword                              &
       = 'acceptable-residual-relative'
     spec( acceptable_residual_absolute )%keyword                              &
       = 'acceptable-residual-absolute'

!  Logical key-words

     spec( stop_if_singular )%keyword = 'stop-if-singular'

!  Character key-words

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
     CALL SPECFILE_assign_value( spec( print_level ),                          &
                                 control%print_level,                          &
                                 control%error )
     CALL SPECFILE_assign_value( spec( print_level_solver ),                   &
                                 control%print_level_solver,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( minimum_size_for_btf ),                 &
                                 control%minimum_size_for_btf,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_real_factor_size ),                 &
                                 control%min_real_factor_size,                 &
                                 control%error )
     CALL SPECFILE_assign_value( spec( min_integer_factor_size ),              &
                                 control%min_integer_factor_size,              &
                                 control%error )
     CALL SPECFILE_assign_long ( spec( max_factor_size ),                      &
                                 control%max_factor_size,                      &
                                 control%error )
     CALL SPECFILE_assign_value( spec( initial_fill_in_factor ),               &
                                 control%initial_fill_in_factor,               &
                                 control%error )
     CALL SPECFILE_assign_value( spec( blas_block_size_factorize ),            &
                                 control%blas_block_size_factorize,            &
                                 control%error )
     CALL SPECFILE_assign_value( spec( blas_block_size_solve ),                &
                                 control%blas_block_size_solve,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivot_control ),                        &
                                 control%pivot_control,                        &
                                 control%error )
     CALL SPECFILE_assign_value( spec( pivot_search_limit ),                   &
                                 control%pivot_search_limit,                   &
                                 control%error )
     CALL SPECFILE_assign_value( spec( max_iterative_refinements ),            &
                                 control%max_iterative_refinements,            &
                                 control%error )

!  Set real value

     CALL SPECFILE_assign_value( spec( array_increase_factor ),                &
                                 control%array_increase_factor,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( array_decrease_factor ),                &
                                 control%array_decrease_factor,                &
                                 control%error )
     CALL SPECFILE_assign_value( spec( relative_pivot_tolerance ),             &
                                 control%relative_pivot_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( absolute_pivot_tolerance ),             &
                                 control%absolute_pivot_tolerance,             &
                                 control%error )
     CALL SPECFILE_assign_value( spec( zero_tolerance ),                       &
                                 control%zero_tolerance,                       &
                                 control%error )
     CALL SPECFILE_assign_value( spec( switch_to_full_code_density ),          &
                                 control%switch_to_full_code_density,          &
                                 control%error )

     CALL SPECFILE_assign_value( spec( acceptable_residual_relative ),         &
                                 control%acceptable_residual_relative,         &
                                 control%error )
     CALL SPECFILE_assign_value( spec( acceptable_residual_absolute ),         &
                                 control%acceptable_residual_absolute,         &
                                 control%error )
!  Set logical values

     CALL SPECFILE_assign_value( spec( stop_if_singular ),                     &
                                 control%stop_if_singular,                     &
                                 control%error )
!  Set character value

     CALL SPECFILE_assign_value( spec( prefix ),                               &
                                 control%prefix,                               &
                                 control%error )
     RETURN

!  End of ULS_read_specfile

     END SUBROUTINE ULS_read_specfile

!-*-*-*-*-*-*-   U L S _ F A C T O R I Z E   S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE ULS_factorize( matrix, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Analyse the sparsity pattern to obtain a good potential ordering and then
!  factorize the matrix

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: i, j, l, out
!!$   INTEGER :: OMP_GET_NUM_THREADS
     LOGICAL :: printi

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  basic single line of output per iteration

      out =  control%out
      printi = out > 0 .AND. control%print_level >= 1

!  check input data

     IF ( matrix%n < 1 .OR. matrix%m < 1 .OR.                                  &
         ( matrix%ne < 0 .AND. STRING_get( matrix%type ) == 'COORDINATE' )     &
         .OR. .NOT. ULS_keyword( matrix%type ) ) THEN
       inform%status = GALAHAD_error_restrictions
       data%m = 0
       data%n = 0
       RETURN
     ELSE
       data%m = matrix%m
       data%n = matrix%n
     END IF

     SELECT CASE ( STRING_get( matrix%type ) )
     CASE ( 'COORDINATE' )
       data%matrix_ne = matrix%ne
     CASE ( 'SPARSE_BY_ROWS' )
       data%matrix_ne = matrix%PTR( matrix%n + 1 ) - 1
     CASE ( 'DENSE' )
       data%matrix_ne = matrix%m * matrix%n
     END SELECT
     data%matrix%ne = data%matrix_ne

     IF ( printi ) WRITE( out, "( A, ' unsymmetric solver ', A, ' used' )" )   &
       prefix,  data%solver( 1 : data%len_solver )

!  convert the input matrix into extended compressed-sparse-row format

!  solver-dependent factorization

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS or MA48 =

     CASE ( 'gls', 'ma28', 'ma48' )

!  if the input matrix is not in co-ordinate form, make a copy

       SELECT CASE ( STRING_get( matrix%type ) )
       CASE ( 'SPARSE_BY_ROWS' )
         data%matrix%m = matrix%m
         data%matrix%n = matrix%n
         data%matrix%ne = matrix%PTR( matrix%n + 1 ) - 1
         CALL STRING_put( data%matrix%type, 'COORDINATE', inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,             &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%row' ; GO TO 900 ; END IF

         CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,             &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%col' ; GO TO 900 ; END IF

         DO i = 1, matrix%n
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             data%matrix%ROW( l ) = i
             data%matrix%COL( l ) = matrix%COL( l )
           END DO
         END DO
       CASE ( 'DENSE' )
         data%matrix%m = matrix%m
         data%matrix%n = matrix%n
         data%matrix%ne = matrix%m * matrix%n
         CALL STRING_put( data%matrix%type, 'COORDINATE', inform%alloc_status )
         CALL SPACE_resize_array( data%matrix%ne, data%matrix%ROW,             &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%row' ; GO TO 900 ; END IF

         CALL SPACE_resize_array( data%matrix%ne, data%matrix%COL,             &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%matrix%col' ; GO TO 900 ; END IF

         l = 0
         DO i = 1, matrix%m
           DO j = 1, matrix%n
             l = l + 1
             data%matrix%ROW( l ) = i ; data%matrix%COL( l ) = j
           END DO
         END DO
       END SELECT
       CALL SPACE_resize_array( data%matrix%ne, data%matrix%VAL,               &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix%val' ; GO TO 900 ; END IF

       SELECT CASE ( STRING_get( MATRIX%type ) )
       CASE ( 'SPARSE_BY_ROWS', 'DENSE' )
         data%matrix%VAL( : data%matrix%ne ) = matrix%VAL( : data%matrix%ne )
       END SELECT

       SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

       CASE ( 'gls', 'ma28' )
         CALL ULS_copy_control_to_gls( control, data%gls_control )
         IF ( control%print_level <= 0 .OR. control%out <= 0 ) THEN
           data%gls_control%lp = - 1
           data%gls_control%wp = - 1
           data%gls_control%mp = - 1
         END IF
         SELECT CASE ( STRING_get( MATRIX%type ) )
         CASE ( 'SPARSE_BY_ROWS', 'DENSE' )
           CALL GLS_analyse( data%matrix, data%gls_factors, data%gls_control,  &
                             data%gls_ainfo, data%gls_finfo )
         CASE DEFAULT
           CALL GLS_analyse( MATRIX, data%gls_factors, data%gls_control,       &
                             data%gls_ainfo, data%gls_finfo )
         END SELECT
         inform%gls_ainfo = data%gls_ainfo
         inform%gls_finfo = data%gls_finfo
         inform%status = data%gls_finfo%flag
         IF ( data%gls_ainfo%flag == - 1 .OR. data%gls_ainfo%flag == - 2 .OR.  &
              data%gls_ainfo%flag == - 3 ) THEN
           inform%status = GALAHAD_error_restrictions
           inform%more_info = data%gls_ainfo%more
         ELSE IF ( data%gls_ainfo%flag == - 4 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%gls_ainfo%stat
         ELSE IF ( data%gls_ainfo%flag == - 7 ) THEN
           inform%status = GALAHAD_error_real_ws
         ELSE IF ( data%gls_ainfo%flag == GALAHAD_unavailable_option ) THEN
           inform%status = GALAHAD_unavailable_option
         ELSE
           inform%alloc_status = data%gls_ainfo%stat
           inform%more_info = data%gls_ainfo%more
           inform%workspace_factors = INT( data%gls_ainfo%len_factorize, long )
           inform%entries_dropped = data%gls_ainfo%drop
           inform%rank = data%gls_ainfo%rank
           inform%structural_rank = data%gls_ainfo%struc_rank
           inform%compresses = data%gls_ainfo%ncmpa
           inform%out_of_range = data%gls_ainfo%oor
           inform%duplicates = data%gls_ainfo%dup
           IF ( data%gls_finfo%flag == - 1 .OR. data%gls_finfo%flag == - 2 .OR.&
                data%gls_finfo%flag == - 3 ) THEN
             inform%status = GALAHAD_error_restrictions
             inform%more_info = data%gls_finfo%more
           ELSE IF ( data%gls_finfo%flag == - 4 ) THEN
             inform%status = GALAHAD_error_allocate
             inform%alloc_status = data%gls_finfo%stat
           ELSE IF ( data%gls_finfo%flag == - 7 ) THEN
             inform%status = GALAHAD_error_real_ws
           ELSE
             inform%status = GALAHAD_ok
             inform%alloc_status = data%gls_finfo%stat
             inform%more_info = data%gls_finfo%more
             inform%entries_in_factors = INT( data%gls_finfo%size_factor, long )
             inform%workspace_factors                                          &
               = INT( data%gls_finfo%len_factorize, long )
             inform%entries_dropped = data%gls_finfo%drop
             inform%rank = data%gls_finfo%rank
           END IF
         END IF

!  = MA48 =

       CASE ( 'ma48' )
         CALL ULS_copy_control_to_ma48( control, data%ma48_control )
         IF ( control%print_level <= 0 .OR. control%out <= 0 ) THEN
           data%ma48_control%lp = - 1
           data%ma48_control%wp = - 1
           data%ma48_control%mp = - 1
         END IF
         SELECT CASE ( STRING_get( MATRIX%type ) )
         CASE ( 'SPARSE_BY_ROWS', 'DENSE' )
           CALL MA48_analyse( data%matrix, data%ma48_factors,                  &
                              data%ma48_control, data%ma48_ainfo,              &
                              data%ma48_finfo )
         CASE DEFAULT
           CALL MA48_analyse( MATRIX, data%ma48_factors, data%ma48_control,    &
                              data%ma48_ainfo, data%ma48_finfo )
         END SELECT
         inform%ma48_ainfo = data%ma48_ainfo
         inform%ma48_finfo = data%ma48_finfo
         inform%status = data%ma48_finfo%flag
         IF ( data%ma48_ainfo%flag == - 1 .OR. data%ma48_ainfo%flag == - 2     &
              .OR. data%ma48_ainfo%flag == - 3 ) THEN
           inform%status = GALAHAD_error_restrictions
           inform%more_info = data%ma48_ainfo%more
         ELSE IF ( data%ma48_ainfo%flag == - 4 ) THEN
           inform%status = GALAHAD_error_allocate
           inform%alloc_status = data%ma48_ainfo%stat
         ELSE IF ( data%ma48_ainfo%flag == - 7 ) THEN
           inform%status = GALAHAD_error_real_ws
         ELSE IF ( data%ma48_ainfo%flag == GALAHAD_unavailable_option ) THEN
           inform%status = GALAHAD_unavailable_option
         ELSE
           inform%alloc_status = data%ma48_ainfo%stat
           inform%more_info = data%ma48_ainfo%more
!          inform%workspace_factors = INT( data%ma48_ainfo%len_factorize, long )
           inform%workspace_factors = INT( data%ma48_ainfo%lena_factorize +    &
             data%ma48_ainfo%leni_factorize, long )
           inform%entries_dropped = data%ma48_ainfo%drop
           inform%rank = data%ma48_ainfo%rank
           inform%structural_rank = data%ma48_ainfo%struc_rank
           inform%compresses = data%ma48_ainfo%ncmpa
           inform%out_of_range = data%ma48_ainfo%oor
           inform%duplicates = data%ma48_ainfo%dup
           IF ( data%ma48_finfo%flag == - 1 .OR. data%ma48_finfo%flag == - 2   &
               .OR. data%ma48_finfo%flag == - 3 ) THEN
             inform%status = GALAHAD_error_restrictions
             inform%more_info = data%ma48_finfo%more
           ELSE IF ( data%ma48_finfo%flag == - 4 ) THEN
             inform%status = GALAHAD_error_allocate
             inform%alloc_status = data%ma48_finfo%stat
           ELSE IF ( data%ma48_finfo%flag == - 7 ) THEN
             inform%status = GALAHAD_error_real_ws
           ELSE
             inform%status = GALAHAD_ok
             inform%alloc_status = data%ma48_finfo%stat
             inform%more_info = data%ma48_finfo%more
             inform%entries_in_factors                                         &
               = INT( data%ma48_finfo%size_factor, long )
             inform%workspace_factors = INT( data%ma48_finfo%lena_factorize +  &
               data%ma48_finfo%leni_factorize, long )
             inform%entries_dropped = data%ma48_finfo%drop
             inform%rank = data%ma48_finfo%rank
           END IF
         END IF
       END SELECT

!  = GETR =

     CASE ( 'getr' )

!  copy the input matrix into the required 2-D array

       CALL SPACE_resize_array( MIN( data%m, data%n ), data%ROWS,              &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%rows' ; GO TO 900 ; END IF

       CALL SPACE_resize_array( data%m, data%n, data%matrix_dense,             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= GALAHAD_ok ) THEN
         inform%bad_alloc = 'sls: data%matrix_dense' ; GO TO 900 ; END IF

       IF ( data%m == data%n ) THEN  ! for LAPACK solvers
         CALL SPACE_resize_array( data%n, 1, data%RHS,                         &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'sls: data%RHS2' ; GO TO 900 ; END IF
       END IF

       SELECT CASE ( STRING_get( matrix%type ) )
       CASE ( 'COORDINATE' )
         data%matrix_dense = 0.0_wp
         DO l = 1, matrix%ne
           data%matrix_dense( matrix%ROW( l ), matrix%COL( l ) )               &
             = matrix%VAL( l )
         END DO
       CASE ( 'SPARSE_BY_ROWS' )
         data%matrix_dense = 0.0_wp
         DO i = 1, matrix%m
           DO l = matrix%PTR( i ), matrix%PTR( i + 1 ) - 1
             data%matrix_dense( i, matrix%COL( l ) ) = matrix%VAL( l )
           END DO
         END DO
       CASE ( 'DENSE' )
         l = 0
         DO i = 1, matrix%m
           DO j = 1, matrix%n
             l = l + 1
             data%matrix_dense( i, j ) = matrix%VAL( l )
           END DO
         END DO
       END SELECT
       CALL GETRF( data%m, data%n, data%matrix_dense, data%m, data%ROWS,       &
                   inform%lapack_error )
       IF ( inform%lapack_error < 0 ) THEN
         inform%status = GALAHAD_error_restrictions ; RETURN
       END IF
       IF ( inform%lapack_error > 0 ) THEN
         inform%rank = inform%lapack_error
       ELSE
         inform%rank = MIN( data%m, data%n )
       END IF
       inform%rank = 0
       DO i = 1, MIN( data%m, data%n )
         if ( ABS( data%matrix_dense( i, i ) ) <= epsmch ** 0.8 )              &
           write(6,*) ' diagonal ', i, ' is zero ', data%matrix_dense( i, i )
         if ( ABS( data%matrix_dense( i, i ) ) > epsmch ** 0.9 )               &
           inform%rank = inform%rank + 1
       END DO
       inform%entries_in_factors = INT( data%m * data%n, KIND = long )
       inform%status = GALAHAD_ok

     END SELECT

     RETURN

!  error returns

 900 CONTINUE
     RETURN

!  End of ULS_factorize

     END SUBROUTINE ULS_factorize

! -*-*-*-*-*-*-*-   U L S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

     SUBROUTINE ULS_solve( matrix, RHS, X, data, control, inform, trans )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Given an unsymmetric matrix A and its ULS factors, solve the system A x = b
!  or A^T x = b, where b is input in RHS, and the solution x output in X

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( IN ) , DIMENSION ( : ) :: RHS
     REAL ( KIND = wp ), INTENT( INOUT ) , DIMENSION ( : ) :: X
     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform
     LOGICAL, INTENT( IN ) :: trans

!  Local variables

     INTEGER :: i, j, l, iter, m, n, itrans
     REAL ( KIND = wp ) :: residual, residual_zero

     CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
     IF ( LEN( TRIM( control%prefix ) ) > 2 )                                  &
       prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )
     itrans = 1

!  Set default inform values

     inform%bad_alloc = ''

!  No refinement is required (or Pardiso is called with its internal refinement)
!  -----------------------------------------------------------------------------

     IF ( control%max_iterative_refinements <= 0 .OR.                          &
          data%solver( 1 : data%len_solver ) == 'pardiso' ) THEN

!  Solve A^T x = b

       IF ( trans ) THEN
         CALL ULS_solve_one_rhs( matrix, RHS, X, data, control, inform,        &
                                 TRANS = itrans )
!  Solve A x = b

       ELSE
         CALL ULS_solve_one_rhs( matrix, RHS, X, data, control, inform )
       END IF

!  Iterative refinement is required
!  --------------------------------

     ELSE

!  Allocate space if necessary

       m = MATRIX%m ; n = MATRIX%n
       IF ( .NOT. data%set_res ) THEN
         CALL SPACE_resize_array( MAX( m, n ), data%RES,                       &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'uls: data%RES' ; RETURN ; END IF
         CALL SPACE_resize_array( MAX( m, n ), data%SOL,                       &
                                  inform%status, inform%alloc_status )
         IF ( inform%status /= GALAHAD_ok ) THEN
           inform%bad_alloc = 'uls: data%SOL' ; RETURN ; END IF
         data%set_res = .TRUE.
       END IF

!  Solve the system A^T x = b with iterative refinement

       IF ( trans ) THEN

!  Compute the original residual

         data%RES( : n ) = RHS( : n )
         X( : m ) = 0.0_wp
         residual_zero = MAXVAL( ABS( RHS( : n ) ) )

         IF ( control%print_level > 1 .AND. control%out > 0 )                  &
           WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )  &
             prefix, residual_zero, 0.0_wp

         DO iter = 0, control%max_iterative_refinements
           inform%iterative_refinements = iter

!  Use factors of the system matrix to solve for the correction

           CALL ULS_solve_one_rhs( matrix, data%RES( : n ), data%SOL( : m ),   &
                                   data, control, inform, TRANS = itrans )
           IF ( inform%status /= GALAHAD_ok ) RETURN

!  Update the estimate of the solution

           X( : m ) = X( : m ) + data%SOL( : m )

!  Form the residuals

           IF ( iter < control%max_iterative_refinements ) THEN
             data%RES( : n ) = RHS( : n )
             SELECT CASE ( STRING_get( MATRIX%type ) )
             CASE ( 'COORDINATE' )
               DO l = 1, MATRIX%ne
                 i = MATRIX%ROW( l ) ; j = MATRIX%COL( l )
                 IF ( i >= 1 .AND. i <= m .AND. j >= 1 .AND. j <= n )          &
                   data%RES( j ) = data%RES( j ) - MATRIX%val( l ) * X( i )
               END DO
             CASE ( 'SPARSE_BY_ROWS' )
               DO i = 1, m
                 DO l = MATRIX%ptr( i ), MATRIX%ptr( i + 1 ) - 1
                   j = MATRIX%COL( l )
                   IF ( j >= 1 .AND. j <= n )                                  &
                     data%RES( j ) = data%RES( j ) - MATRIX%val( l ) * X( i )
                 END DO
               END DO
             CASE ( 'DENSE' )
               l = 0
               DO i = 1, m
                 DO j = 1, n
                   l = l + 1
                   data%RES( j ) = data%RES( j ) - MATRIX%val( l ) * X( i )
                 END DO
               END DO
             END SELECT
           END IF

!  Check for convergence

           residual = MAXVAL( ABS( data%RES( : n ) ) )
           IF ( control%print_level >= 1 .AND. control%out > 0 )               &
            WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" ) &
               prefix, residual, MAXVAL( ABS( X( : m ) ) )

           IF ( residual < MAX( control%acceptable_residual_absolute,          &
                  control%acceptable_residual_relative * residual_zero ) ) EXIT
         END DO

!  Solve the system A^T x = b with iterative refinement

       ELSE

!  Compute the original residual

         data%RES( : m ) = RHS( : m )
         X( : n ) = 0.0_wp
         residual_zero = MAXVAL( ABS( RHS( : m ) ) )

         IF ( control%print_level > 1 .AND. control%out > 0 )                  &
           WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )" )  &
             prefix, residual_zero, 0.0_wp

         DO iter = 0, control%max_iterative_refinements
           inform%iterative_refinements = iter

!  Use factors of the system matrix to solve for the correction

           CALL ULS_solve_one_rhs( matrix, data%RES( : m ), data%SOL( : n ),   &
                                   data, control, inform )
           IF ( inform%status /= GALAHAD_ok ) RETURN

!  Update the estimate of the solution

           X( : n ) = X( : n ) + data%SOL( : n )

!  Form the residuals

           IF ( iter < control%max_iterative_refinements ) THEN
             data%RES( : m ) = RHS( : m )
             SELECT CASE ( STRING_get( MATRIX%type ) )
             CASE ( 'COORDINATE' )
               DO l = 1, MATRIX%ne
                 i = MATRIX%ROW( l ) ; j = MATRIX%COL( l )
                 IF ( i >= 1 .AND. i <= m .AND. j >= 1 .AND. j <= n )          &
                   data%RES( i ) = data%RES( i ) - MATRIX%val( l ) * X( j )
               END DO
             CASE ( 'SPARSE_BY_ROWS' )
               DO i = 1, m
                 DO l = MATRIX%ptr( i ), MATRIX%ptr( i + 1 ) - 1
                   j = MATRIX%COL( l )
                   IF ( j >= 1 .AND. j <= n )                                  &
                     data%RES( i ) = data%RES( i ) - MATRIX%val( l ) * X( j )
                 END DO
               END DO
             CASE ( 'DENSE' )
               l = 0
               DO i = 1, m
                 DO j = 1, n
                   l = l + 1
                   data%RES( i ) = data%RES( i ) - MATRIX%val( l ) * X( j )
                 END DO
               END DO
             END SELECT
           END IF

!  Check for convergence

           residual = MAXVAL( ABS( data%RES( : m ) ) )
           IF ( control%print_level >= 1 .AND. control%out > 0 )               &
            WRITE( control%out, "( A, ' maximum residual, sol ', 2ES24.16 )")  &
               prefix, residual, MAXVAL( ABS( X( : n ) ) )

           IF ( residual < MAX( control%acceptable_residual_absolute,          &
                  control%acceptable_residual_relative * residual_zero ) ) EXIT
         END DO
       END IF
     END IF

     RETURN

!  End of subroutine ULS_solve

     END SUBROUTINE ULS_solve

!-*-*-*-*-   U L S _ S O L V E _ O N E _ R H S   S U B R O U T I N E   -*-*-*-

     SUBROUTINE ULS_solve_one_rhs( matrix, RHS, X, data, control, inform,      &
                                   trans )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Solve the linear system using the factors obtained in the factorization

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: RHS
     REAL ( KIND = wp ), INTENT( INOUT ) :: X( : )
     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform
     INTEGER, OPTIONAL, INTENT( IN ) :: trans

!  local variables

!    INTEGER :: pardiso_error
     LOGICAL :: trans_true

!  solver-dependent solution

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

     CASE ( 'gls', 'ma28' )
       CALL ULS_copy_control_to_gls( control, data%gls_control )
       SELECT CASE ( STRING_get( MATRIX%type ) )
       CASE ( 'COORDINATE' )
         CALL GLS_solve( matrix, data%gls_factors, RHS, X,                     &
                         data%gls_control, data%gls_sinfo, trans )
       CASE DEFAULT
         CALL GLS_solve( data%matrix, data%gls_factors, RHS, X,                &
                         data%gls_control, data%gls_sinfo, trans )
       END SELECT
       inform%gls_sinfo = data%gls_sinfo
       inform%status = data%gls_sinfo%flag
       IF ( inform%status == - 1 .OR. inform%status == - 2 .OR.                &
            inform%status == - 3 ) THEN
         inform%status = GALAHAD_error_restrictions
       ELSE IF ( inform%status == - 4 ) THEN
         inform%status = GALAHAD_error_allocate
         inform%alloc_status = data%gls_sinfo%stat
       ELSE
         inform%more_info = data%gls_sinfo%more
         inform%alloc_status = data%gls_sinfo%stat
       END IF

!  = MA48 =

     CASE ( 'ma48' )
       CALL ULS_copy_control_to_ma48( control, data%ma48_control )
       SELECT CASE ( STRING_get( MATRIX%type ) )
       CASE ( 'COORDINATE' )
         CALL MA48_solve( matrix, data%ma48_factors, RHS, X,                   &
                          data%ma48_control, data%ma48_sinfo, trans )
       CASE DEFAULT
         CALL MA48_solve( data%matrix, data%ma48_factors, RHS, X,              &
                          data%ma48_control, data%ma48_sinfo, trans )
       END SELECT
       inform%ma48_sinfo = data%ma48_sinfo
       inform%status = data%ma48_sinfo%flag
       IF ( inform%status == - 1 .OR. inform%status == - 2 .OR.                &
            inform%status == - 3 ) THEN
         inform%status = GALAHAD_error_restrictions
       ELSE IF ( inform%status == - 4 ) THEN
         inform%status = GALAHAD_error_allocate
         inform%alloc_status = data%ma48_sinfo%stat
       ELSE
         inform%more_info = data%ma48_sinfo%more
         inform%alloc_status = data%ma48_sinfo%stat
       END IF
     CASE ( 'getr' )
       data%RHS( : data%n, 1 ) = RHS( : data%n )
       IF ( PRESENT( trans ) ) THEN
         trans_true = trans == 1
       ELSE
         trans_true = .FALSE.
       END IF
       IF ( trans_true ) THEN
         CALL GETRS( 'T', data%n, 1, data%matrix_dense, data%m, data%ROWS,     &
                     data%RHS, data%n, inform%lapack_error )
       ELSE
         CALL GETRS( 'N', data%n, 1, data%matrix_dense, data%m, data%ROWS,     &
                     data%RHS, data%n, inform%lapack_error )
       END IF
       IF ( inform%lapack_error < 0 ) THEN
         inform%status = GALAHAD_error_restrictions ; RETURN
       ELSE 
         inform%status = GALAHAD_ok
       END IF
       X( : data%n ) = data%RHS( : data%n, 1 )
     CASE DEFAULT
       WRITE( 6, * ) ' solver ', data%solver( 1 : data%len_solver )
       WRITE( 6, * ) ' should not be here'
     END SELECT

     RETURN

!  End of ULS_solve_one_rhs

     END SUBROUTINE ULS_solve_one_rhs

!-*-  U L S _ F R E D H O L M _ A L T E R N A T I V E   S U B R O U T I N E  -*-

     SUBROUTINE ULS_fredholm_alternative( matrix, RHS, X, data, control,       &
                                          inform, alternative )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
! GLS_fredholm_alternative uses the factors produced by GLS_factorize
!   to find either x so that Ax=b or a "direction of linear infinite descent"
!   y so that A^T y = 0 and b^T y > 0
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( SMT_type ), INTENT( IN ) :: matrix
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: RHS
     REAL ( KIND = wp ), INTENT( INOUT ) :: X( : )
     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform
     LOGICAL, INTENT( OUT ) :: alternative

!    INTEGER :: j, l
!    REAL ( KIND = wp ) :: RES( matrix%n )

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

     CASE ( 'gls', 'ma28' )
       CALL ULS_copy_control_to_gls( control, data%gls_control )
       SELECT CASE ( STRING_get( MATRIX%type ) )
       CASE ( 'COORDINATE' )
         CALL GLS_fredholm_alternative( matrix, data%gls_factors, RHS, X,      &
                         data%gls_control, data%gls_sinfo, alternative )
!        IF ( alternative ) THEN
! compute A^T x
!         RES = 0.0_wp
!         DO l = 1, matrix%ne
!           j = matrix%COL( l )
!           RES( j ) = RES( j ) + matrix%VAL( l ) * X( matrix%ROW( l ) )
!         END DO
!         write(6,*) ' A^T y ', MAXVAL( ABS( RES ) )
! compute rhs^T x
!          write(6,*) 'b^Ty ', DOT_PRODUCT( RHS( : matrix%m ), X( : matrix%m ) )
!        END IF
       CASE DEFAULT
         CALL GLS_fredholm_alternative( data%matrix, data%gls_factors, RHS, X, &
                         data%gls_control, data%gls_sinfo, alternative )
!        IF ( alternative ) THEN
! compute A^T x
!         RES = 0.0_wp
!         DO l = 1, data%matrix%ne
!           j = data%matrix%COL( l )
!           RES( j ) = RES( j ) + data%matrix%VAL( l ) * X( data%matrix%ROW(l) )
!         END DO
!         write(6,*) ' A^T y ', MAXVAL( ABS( RES ) )
! compute rhs^T x
!          write(6,*) 'b^Ty ', DOT_PRODUCT( RHS( : matrix%m ), X( : matrix%m ) )
!        END IF
       END SELECT
!write(6,*) ' X ', X
       inform%gls_sinfo = data%gls_sinfo
       inform%status = data%gls_sinfo%flag
       IF ( inform%status == - 1 .OR. inform%status == - 2 .OR.                &
            inform%status == - 3 ) THEN
         inform%status = GALAHAD_error_restrictions
       ELSE IF ( inform%status == - 4 ) THEN
         inform%status = GALAHAD_error_allocate
         inform%alloc_status = data%gls_sinfo%stat
       ELSE
         inform%more_info = data%gls_sinfo%more
         inform%alloc_status = data%gls_sinfo%stat
         inform%alternative = alternative
       END IF

!  = MA48 =

     CASE ( 'ma48' )
       CALL ULS_copy_control_to_ma48( control, data%ma48_control )
       inform%status = GALAHAD_unavailable_option
!       SELECT CASE ( STRING_get( MATRIX%type ) )
!       CASE ( 'COORDINATE' )
!         CALL MA48_solve( matrix, data%ma48_factors, RHS, X,                  &
!                          data%ma48_control, data%ma48_sinfo, trans )
!       CASE DEFAULT
!         CALL MA48_solve( data%matrix, data%ma48_factors, RHS, X,             &
!                          data%ma48_control, data%ma48_sinfo, trans )
!      END SELECT
!      inform%ma48_sinfo = data%ma48_sinfo
!      inform%status = data%ma48_sinfo%flag
!      IF ( inform%status == - 1 .OR. inform%status == - 2 .OR.                &
!           inform%status == - 3 ) THEN
!        inform%status = GALAHAD_error_restrictions
!      ELSE IF ( inform%status == - 4 ) THEN
!        inform%status = GALAHAD_error_allocate
!        inform%alloc_status = data%ma48_sinfo%stat
!      ELSE
!        inform%more_info = data%ma48_sinfo%more
!        inform%alloc_status = data%ma48_sinfo%stat
!      END IF

     CASE ( 'getr' )
       inform%status = GALAHAD_unavailable_option
     END SELECT

     RETURN

!  End of ULS_fredholm_alternative

     END SUBROUTINE ULS_fredholm_alternative

!-*-*-*-*-*-*-*-   U L S _ E N Q U I R E  S U B R O U T I N E   -*-*-*-*-*-*-

     SUBROUTINE ULS_enquire( data, inform, ROWS, COLS )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Interogate the factorization to obtain additional information

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform

     INTEGER, INTENT( OUT ), DIMENSION( : ) :: ROWS
     INTEGER, INTENT( OUT ), DIMENSION( : ) :: COLS

!  local variables

     INTEGER :: i, ip, ri, info, rank

!  solver-dependent solution

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

     CASE ( 'gls', 'ma28' )
       CALL GLS_special_rows_and_cols( data%gls_factors, rank,                 &
                                        ROWS( :  ),                            &
                                        COLS( :  ), info )
!                                       ROWS( : data%m ),                      &
!                                       COLS( : data%n ), info )
       inform%status = info
       inform%rank = rank

!  = MA48 =

     CASE ( 'ma48' )
       CALL MA48_special_rows_and_cols( data%ma48_factors, rank, ROWS, COLS,   &
                                        data%ma48_control, info )
       IF ( info == - 1 ) THEN
         inform%status = GALAHAD_error_allocate
       ELSE IF ( info == - 2 ) THEN
         inform%status = GALAHAD_error_deallocate
       ELSE
         inform%status = info
         inform%rank = rank
       END IF

!  = getr =

     CASE ( 'getr' )
       ROWS = (/ ( i, i = 1, data%m ) /)
       DO i = 1, MIN( data%m, data%n ) ! run through the pivots
         ip = data%ROWS( i ) ! swap rows i and ip
         ri = ROWS( i ) ; ROWS( i ) = ROWS( ip ) ; ROWS( ip ) = ri
       END DO
       COLS = (/ ( i, i = 1, data%n ) /)
     END SELECT

     RETURN

!  End of ULS_enquire

     END SUBROUTINE ULS_enquire

!-*-*-*-*-*-*-*-   U L S _ T E R M I N A T E  S U B R O U T I N E   -*-*-*-*-*-

     SUBROUTINE ULS_terminate( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Deallocate all currently allocated arrays

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

     TYPE ( ULS_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform

!  local variables

     INTEGER :: info
!    INTEGER :: pardiso_error

!  solver-dependent termination

     SELECT CASE( data%solver( 1 : data%len_solver ) )

!  = GLS =

     CASE ( 'gls', 'ma28' )
       CALL ULS_copy_control_to_gls( control, data%gls_control )
       CALL GLS_finalize( data%gls_factors, data%gls_control, info )
       inform%status = info

!  = MA48 =

     CASE ( 'ma48' )
       CALL ULS_copy_control_to_ma48( control, data%ma48_control )
       CALL MA48_finalize( data%ma48_factors, data%ma48_control, info )
       inform%status = info

     END SELECT

!  solver-independent termination

     IF ( ALLOCATED( data%matrix%type ) )                                      &
       DEALLOCATE( data%matrix%type, STAT = inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%row, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%col, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%ptr, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%val, inform%status,                 &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%type, inform%status,                &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( data%ROWS, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%MAP, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%SOL, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%RES, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%RHS, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%SCALE, inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix_dense, inform%status,               &
                               inform%alloc_status )
     data%len_solver = - 1

     RETURN

!  End of ULS_terminate

     END SUBROUTINE ULS_terminate

! -  G A L A H A D -  U L S _ f u l l _ t e r m i n a t e  S U B R O U T I N E -

     SUBROUTINE ULS_full_terminate( data, control, inform )

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!   Deallocate all private storage

!  *-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ULS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_inform_type ), INTENT( INOUT ) :: inform

!  deallocate workspace

     CALL ULS_terminate( data%uls_data, control, inform )

!  deallocate any internal problem arrays

     CALL SPACE_dealloc_array( data%matrix%ptr,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%row,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%col,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%matrix%val,                                &
                               inform%status, inform%alloc_status )
     CALL SPACE_dealloc_array( data%RHS,                                       &
                               inform%status, inform%alloc_status )

     RETURN

!  End of subroutine ULS_full_terminate

     END SUBROUTINE ULS_full_terminate

!-*   U L S _ C O P Y _ C O N T R O L _ T O _ S I L S  S U B R O U T I N E   *-

     SUBROUTINE ULS_copy_control_to_gls( control, control_gls )

!  copy control parameters to their GLS equivalents

!  Dummy arguments

     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( GLS_control ), INTENT( INOUT ) :: control_gls

     IF ( control%print_level_solver > 0 ) THEN
       control_gls%lp = control%error
       control_gls%wp = control%warning
       control_gls%mp = control%out
     ELSE
       control_gls%lp = 0
       control_gls%wp = 0
       control_gls%mp = 0
     END IF
     control_gls%ldiag = control%print_level_solver
     control_gls%fill_in = control%initial_fill_in_factor
     control_gls%la = control%min_real_factor_size
     control_gls%la_int = control%min_integer_factor_size
     control_gls%maxla = INT( control%max_factor_size )
     control_gls%multiplier = control%array_increase_factor
     control_gls%reduce = control%array_decrease_factor
     control_gls%btf = control%minimum_size_for_btf
     IF ( control%pivot_control == 5 ) THEN
       control_gls%diagonal_pivoting = .TRUE.
!       inform%pivot_control = 5
     ELSE
       control_gls%diagonal_pivoting = .FALSE.
!       inform%pivot_control = 1
     END IF
     control_gls%pivoting = control%pivot_search_limit
     control_gls%u = control%relative_pivot_tolerance
     control_gls%tolerance = control%absolute_pivot_tolerance
     control_gls%drop = control%zero_tolerance
     control_gls%factor_blocking = control%blas_block_size_factorize
     IF ( control_gls%factor_blocking < 1 )                                    &
       control_gls%factor_blocking = blas_block_size_factor_default
     control_gls%solve_blas = control%blas_block_size_solve
     IF ( control_gls%solve_blas < 1 )                                         &
       control_gls%solve_blas = blas_block_size_solve_default
     control_gls%struct = control%stop_if_singular
     control_gls%switch = control%switch_to_full_code_density

     RETURN

!  End of ULS_copy_control_to_gls

     END SUBROUTINE ULS_copy_control_to_gls

!-*-   U L S _ C O P Y _ C O N T R O L _ T O _ M A 5 7  S U B R O U T I N E  -*-

     SUBROUTINE ULS_copy_control_to_ma48( control, control_ma48 )

!  copy control parameters to their MA48 equivalents

!  Dummy arguments

     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( MA48_control ), INTENT( INOUT ) :: control_ma48

     control_ma48%lp = control%error
     control_ma48%wp = control%warning
     control_ma48%mp = control%out
     control_ma48%ldiag = control%print_level_solver
     control_ma48%fill_in = control%initial_fill_in_factor
!    control_ma48%la = control%min_real_factor_size_default
!    control_ma48%maxla = INT( control%max_factor_size )
     control_ma48%multiplier = control%array_increase_factor
!    control_ma48%reduce = control%array_decrease_factor
     control_ma48%btf = control%minimum_size_for_btf
     IF ( control%pivot_control == 5 ) THEN
       control_ma48%diagonal_pivoting = .TRUE.
!       inform%pivot_control == 5
     ELSE
       control_ma48%diagonal_pivoting = .FALSE.
!       inform%pivot_control == 1
     END IF
     control_ma48%pivoting = control%pivot_search_limit
     control_ma48%struct = control%stop_if_singular
     control_ma48%u = control%relative_pivot_tolerance
     control_ma48%tolerance = control%absolute_pivot_tolerance
     control_ma48%drop = control%zero_tolerance
     control_ma48%factor_blocking = control%blas_block_size_factorize
     IF ( control_ma48%factor_blocking < 1 )                                   &
       control_ma48%factor_blocking = blas_block_size_factor_default
     control_ma48%solve_blas = control%blas_block_size_solve
     IF ( control_ma48%solve_blas < 1 )                                        &
       control_ma48%solve_blas = blas_block_size_solve_default
     control_ma48%switch = control%switch_to_full_code_density

     RETURN

!  End of ULS_copy_control_to_ma48

     END SUBROUTINE ULS_copy_control_to_ma48

!-*-*-*-*-*-*-*-*-*-   U L S _ K E Y W O R D    F U N C T I O N  -*-*-*-*-*-*-*-

     FUNCTION ULS_keyword( array )
     LOGICAL :: ULS_keyword

!  Dummy arguments

     CHARACTER, ALLOCATABLE, DIMENSION( : ) :: array

!  Check to see if the string is an appropriate keyword

     SELECT CASE( STRING_get( array ) )

!  Keyword known

     CASE( 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE' )
       ULS_keyword = .TRUE.

!  Keyword unknown

     CASE DEFAULT
       ULS_keyword = .FALSE.
     END SELECT

     RETURN

!  End of ULS_keyword

     END FUNCTION ULS_keyword

! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------
!              specific interfaces to make calls from C easier
! -----------------------------------------------------------------------------
! =============================================================================
! -----------------------------------------------------------------------------

! G A L A H A D - U L S _ f a c t o r i z e _ m a t r i x  S U B R O U T I N E -

     SUBROUTINE ULS_factorize_matrix( control, data, status, m, n,             &
                                      matrix_type, matrix_ne, matrix_val,      &
                                      matrix_row, matrix_col, matrix_ptr )

!  import structural matrix data into internal storage, analyse the structure,
!  and then perform a factorization

!  Arguments are as follows:

!  control is a derived type whose components are described in the leading
!   comments to ULS_solve
!
!  data is a scalar variable of type ULS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
!   success or otherwise of the import and analysis. Possible values are:
!
!    0. The factorization was succesful, and the package is ready for the
!       solution phase
!
!   -1. An allocation error occurred. A message indicating the offending
!       array is written on unit control.error, and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -2. A deallocation error occurred.  A message indicating the offending
!       array is written on unit control.error and the returned allocation
!       status and a string containing the name of the offending array
!       are held in inform.alloc_status and inform.bad_alloc respectively.
!   -3. The restriction n > 0, m >= 0 or requirement that type contains
!       its relevant string 'DENSE', 'COORDINATE' or 'SPARSE_BY_ROWS'
!       has been violated.
!
!  m is a scalar variable of type default integer, that holds the number of
!   rows of the matrix A
!
!  n is a scalar variable of type default integer, that holds the number of
!   columns of the matrix A
!
!  matrix_type is a character string that specifies the storage scheme used
!   for A. It should be one of 'coordinate', 'sparse_by_rows' or 'dense';
!   lower or upper case variants are allowed.
!
!  matrix_ne is a scalar variable of type default integer, that holds the
!   number of entries in A in the sparse co-ordinate storage scheme.
!   It need not be set for any of the other schemes.
!
!  matrix_val is a rank-one array of type default real, that holds the
!   values of A input in precisely the same order as those for the row
!   and column indices
!
!  matrix_row is a rank-one array of type default integer, that holds the
!   row indices of A in the sparse co-ordinate storage scheme. It need not
!   be set for any of the other schemes, and in this case can be of length 0
!
!  matrix_col is a rank-one array of type default integer, that holds the
!   column indices of A in either the sparse co-ordinate, or the sparse
!   row-wise storage scheme. It need not  be set when the dense scheme is
!   used, and in this case can be of length 0
!
!  matrix_ptr is a rank-one array of dimension m+1 and type default
!   integer, that holds the starting position of each row of A, as well as
!   the total number of entries plus one, in the sparse row-wise storage
!   scheme. It need not be set when the other schemes are used, and in
!   this case can be of length 0
!
!  See ULS_factorize for a description of the required arguments
!
!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ULS_control_type ), INTENT( INOUT ) :: control
     TYPE ( ULS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( IN ) :: m, n, matrix_ne
     INTEGER, INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: matrix_type
     INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: matrix_row
     INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: matrix_col
     INTEGER, DIMENSION( : ), OPTIONAL, INTENT( IN ) :: matrix_ptr
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: matrix_val

!  copy control to data

     data%ULS_control = control

!  set A appropriately in the smt storage type

     data%matrix%m = m ; data%matrix%n = n
     SELECT CASE ( matrix_type )
     CASE ( 'coordinate', 'COORDINATE' )
       CALL SMT_put( data%matrix%type, 'COORDINATE',                           &
                     data%uls_inform%alloc_status )
       data%matrix%ne = matrix_ne

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%row,               &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%col,               &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

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
       CALL SMT_put( data%matrix%type, 'SPARSE_BY_ROWS',                       &
                     data%uls_inform%alloc_status )
       IF ( data%f_indexing ) THEN
         data%matrix%ne = matrix_ptr( m + 1 ) - 1
       ELSE
         data%matrix%ne = matrix_ptr( m + 1 )
       END IF

       CALL SPACE_resize_array( m + 1, data%matrix%ptr,                        &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%col,               &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

       IF ( data%f_indexing ) THEN
         data%matrix%ptr( : m + 1 ) = matrix_ptr( : m + 1 )
         data%matrix%col( : data%matrix%ne ) = matrix_col( : data%matrix%ne )
       ELSE
         data%matrix%ptr( : m + 1 ) = matrix_ptr( : m + 1 ) + 1
         data%matrix%col( : data%matrix%ne )                                   &
           = matrix_col( : data%matrix%ne ) + 1
       END IF

     CASE ( 'dense', 'DENSE' )
       CALL SMT_put( data%matrix%type, 'DENSE',                                &
                     data%uls_inform%alloc_status )
       data%matrix%ne = m * n

       CALL SPACE_resize_array( data%matrix%ne, data%matrix%val,               &
              data%uls_inform%status, data%uls_inform%alloc_status )
       IF ( data%uls_inform%status /= 0 ) GO TO 900

     CASE DEFAULT
       data%uls_inform%status = GALAHAD_error_unknown_storage
       GO TO 900
     END SELECT


!  save the values of matrix

     IF ( data%matrix%ne > 0 )                                                 &
       data%matrix%val( : data%matrix%ne ) = matrix_val( : data%matrix%ne )

!  factorize the matrix

     CALL ULS_factorize( data%matrix, data%uls_data, data%uls_control,         &
                         data%uls_inform )

!  set up space for the solution vector

     CALL SPACE_resize_array( MAX( m, n ), data%RHS,                           &
            data%uls_inform%status, data%uls_inform%alloc_status )
     IF ( data%uls_inform%status /= 0 ) GO TO 900

     status = data%uls_inform%status
     RETURN

!  error returns

 900 CONTINUE
     status = data%uls_inform%status
     RETURN

!  End of subroutine ULS_factorize_matrix

     END SUBROUTINE ULS_factorize_matrix

!-  G A L A H A D -  U L S _ r e s e t _ c o n t r o l   S U B R O U T I N E -

     SUBROUTINE ULS_reset_control( control, data, status )

!  reset control parameters after import if required.
!  See ULS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ULS_control_type ), INTENT( IN ) :: control
     TYPE ( ULS_full_data_type ), INTENT( INOUT ) :: data
     INTEGER, INTENT( OUT ) :: status

!  set control in internal data

     data%uls_control = control

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine ULS_reset_control

     END SUBROUTINE ULS_reset_control

!--  G A L A H A D -  U L S _ s o l v e _ s y s t e m   S U B R O U T I N E  -

     SUBROUTINE ULS_solve_system( data, status, SOL, trans )

!  solve the linear system A x = b or A^T x = b

!  Arguments are as follows:

!  data is a scalar variable of type ULS_full_data_type used for internal data
!
!  status is a scalar variable of type default intege that indicates the
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
!
!  trans is a scalr variable of type logical that should be set true
!      if the solution to the system A^T x = b is sought, and false
!      if the solution to the system A x = b is required

!  See ULS_solve for a description of the required arguments

!--------------------------------
!   D u m m y   A r g u m e n t s
!--------------------------------

     INTEGER, INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: trans
     TYPE ( ULS_full_data_type ), INTENT( INOUT ) :: data
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( : ) :: SOL

!  set up the right-hand side b

     IF ( trans ) THEN
       data%RHS( : data%matrix%n ) = SOL( : data%matrix%n )
     ELSE
       data%RHS( : data%matrix%m ) = SOL( : data%matrix%m )
     END IF

!  solve the block linear system

     CALL ULS_solve( data%matrix, data%RHS, SOL, data%uls_data,                &
                     data%uls_control, data%uls_inform, trans )

     status = data%uls_inform%status
     RETURN

!  end of subroutine ULS_solve_system

     END SUBROUTINE ULS_solve_system

!-*-  G A L A H A D -  U L S _ i n f o r m a t i o n   S U B R O U T I N E  -*-

     SUBROUTINE ULS_information( data, inform, status )

!  return solver information during or after solution by ULS
!  See ULS_solve for a description of the required arguments

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     TYPE ( ULS_full_data_type ), INTENT( INOUT ) :: data
     TYPE ( ULS_inform_type ), INTENT( OUT ) :: inform
     INTEGER, INTENT( OUT ) :: status

!  recover inform from internal data

     inform = data%uls_inform

!  flag a successful call

     status = GALAHAD_ok
     RETURN

!  end of subroutine ULS_information

     END SUBROUTINE ULS_information

!  End of module GALAHAD_ULS_double

   END MODULE GALAHAD_ULS_double
