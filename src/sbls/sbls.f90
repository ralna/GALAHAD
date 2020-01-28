! THIS VERSION: GALAHAD 3.1 - 26/08/2018 AT 14:20 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ S B L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started May 12th 2004
!   originally released GALAHAD Version 2.0. February 16th 2005
!   modified to enable sls in GALAHAD Version 2.4. August 19th 2009

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

    MODULE GALAHAD_SBLS_double

!      ---------------------------------------------------------------
!     |                                                               |
!     | Given matrices A and (symmetric) H and C, provide and apply   |
!     | preconditioners for the symmetric block linear system         |
!     |                                                               |
!     |    ( H   A^T ) ( x ) = ( a )                                  |
!     |    ( A   -C  ) ( y )   ( b )                                  |
!     |                                                               |
!     | of the form                                                   |
!     |                                                               |
!     |    K = ( G   A^T )                                            |
!     |        ( A   -C  )                                            |
!     |                                                               |
!     | involving some suitable symmetric, second-order sufficient G  |
!     |                                                               |
!      ---------------------------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double, ONLY: QPT_keyword_H, QPT_keyword_A
      USE GALAHAD_LMT_double, LMS_control_type => LMT_control_type,            &
                              LMS_data_type => LMT_data_type
      USE GALAHAD_SORT_double, ONLY: SORT_reorder_by_rows
      USE GALAHAD_ROOTS_double, ONLY : ROOTS_quadratic
      USE GALAHAD_SLS_double
      USE GALAHAD_ULS_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_NORMS_double, ONLY: TWO_norm
      USE GALAHAD_BLAS_interface, ONLY : GEMV
      USE GALAHAD_LAPACK_interface, ONLY : POTRF, POTRS, SYTRF, SYTRS

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SBLS_initialize, SBLS_read_specfile, SBLS_basis_solve,         &
                SBLS_form_and_factorize, SBLS_solve, SBLS_solve_explicit,      &
                SBLS_solve_implicit, SBLS_solve_null_space, SBLS_terminate,    &
                SBLS_fredholm_alternative, SBLS_part_solve, SBLS_eigs,         &
                SBLS_solve_iterative, SBLS_cond, SMT_type, SMT_put, SMT_get,   &
                LMS_control_type, LMS_data_type

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: real_bytes = 8
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: no_last = - 1000
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: rminvr_zero = ten * epsmch
      LOGICAL :: roots_debug = .FALSE.

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: SBLS_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  initial estimate of integer workspace for sls (obsolete)

        INTEGER :: indmin = 1000

!  initial estimate of real workspace for sls (obsolete)

        INTEGER :: valmin = 1000

!  initial estimate of workspace for uls (obsolete)

        INTEGER :: len_ulsmin = 1000

!  maximum number of iterative refinements with preconditioner

        INTEGER :: itref_max = 1

!  maximum number of projected CG iterations

        INTEGER :: maxit_pcg = 1000

!  how much has A changed since last factorization:
!   0 = not changed, 1 = values changed, 2 = structure changed

        INTEGER :: new_a = 2

!  how much has H changed since last factorization:
!   0 = not changed, 1 = values changed, 2 = structure changed

        INTEGER :: new_h = 2

!  how much has C changed since last factorization:
!   0 = not changed, 1 = values changed, 2 = structure changed

        INTEGER :: new_c = 2

!  which preconditioner to use:
!    0 automatic
!    1 explicit with G = I
!    2 explicit with G = H
!    3 explicit with G = diag(max(H,min_diag))
!    4 explicit with G = band(H)
!    5 explicit with G = (optional, diagonal) D
!    6 explicit with G = (optional, limited-memory) H_lm
!    7 explicit with G = H + (optional, limited-memory) H_lm
!    8 explicit with G = (optional, diagonal + limited-memory) D + H_lm
!   11 explicit with G_11 = 0, G_21 = 0, G_22 = H_22
!   12 explicit with G_11 = 0, G_21 = H_21, G_22 = H_22
!   -1 implicit with G_11 = 0, G_21 = 0, G_22 = I
!   -2 implicit with G_11 = 0, G_21 = 0, G_22 = H_22

        INTEGER :: preconditioner = 0

!  the semi-bandwidth for band(H)

        INTEGER :: semi_bandwidth = 5

!  the explicit factorization used:
!   0 automatic
!   1 Schur-complement if G = diag and successful otherwise augmented system
!   2 augmented system
!   3 null-space
!   4 Schur-complement if G = diag and successful otherwise failure
!   5 Schur-complement with pivoting if G=diag and successful otherwise failure

        INTEGER :: factorization = 0

!  maximum number of nonzeros in a column of A for Schur-complement
!  factorization

        INTEGER :: max_col = 35

!  not used at present

        INTEGER :: scaling = 0
        INTEGER :: ordering = 0

!  the relative pivot tolerance used by uls (obsolete)

        REAL ( KIND = wp ) :: pivot_tol = 0.01_wp

!  the relative pivot tolerance used by uls when determining
!  the basis matrix

        REAL ( KIND = wp ) :: pivot_tol_for_basis = 0.5

!  the absolute pivot tolerance used by uls                           (OBSOLETE)

!       REAL ( KIND = wp ) :: zero_pivot = epsmch ** 0.75_wp
        REAL ( KIND = wp ) :: zero_pivot = epsmch

!  not used at present

        REAL ( KIND = wp ) :: static_tolerance = 0.0_wp
        REAL ( KIND = wp ) :: static_level  = 0.0_wp

!  the minimum permitted diagonal in diag(max(H,min_diag))

        REAL ( KIND = wp ) :: min_diagonal = 0.00001_wp

!   the required absolute and relative accuracies

        REAL ( KIND = wp ) :: stop_absolute = epsmch
        REAL ( KIND = wp ) :: stop_relative = epsmch

!  preprocess equality constraints to remove linear dependencies

        LOGICAL :: remove_dependencies = .TRUE.

!  determine implicit factorization preconditioners using a
!  basis of A found by examining A's transpose

        LOGICAL :: find_basis_by_transpose = .TRUE.

!  can the right-hand side c be assumed to be zero?

        LOGICAL :: affine = .FALSE.

!  do we tolerate "singular" preconditioners?

        LOGICAL :: allow_singular = .FALSE.
!       LOGICAL :: allow_singular = .TRUE.

!   if the initial attempt at finding a preconditioner is unsuccessful,
!   should the diagonal be perturbed so that a second attempt succeeds?
!
        LOGICAL :: perturb_to_make_definite = .TRUE.

!  compute the residual when applying the preconditioner?

        LOGICAL :: get_norm_residual = .FALSE.

!  if an implicit or null-space preconditioner is used, assess and
!  correct for ill conditioned basis matrices

        LOGICAL :: check_basis = .TRUE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  indefinite linear equation solver

        CHARACTER ( LEN = 30 ) :: symmetric_linear_solver =                    &
           "sils" // REPEAT( ' ', 26 )

!  definite linear equation solver

        CHARACTER ( LEN = 30 ) :: definite_linear_solver =                     &
           "sils" // REPEAT( ' ', 26 )

!  unsymmetric linear equation solver

        CHARACTER ( LEN = 30 ) :: unsymmetric_linear_solver =                  &
           "gls" // REPEAT( ' ', 27 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix = '""                            '

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for ULS

        TYPE ( ULS_control_type ) :: ULS_control
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: SBLS_time_type

!  total cpu time spent in the package

        REAL ( KIND = wp ) :: total = 0.0

!  cpu time spent forming the preconditioner K_G

        REAL ( KIND = wp ) :: form = 0.0

!  cpu time spent factorizing K_G

        REAL ( KIND = wp ) :: factorize = 0.0

!  cpu time spent solving linear systems inolving K_G

        REAL ( KIND = wp ) :: apply = 0.0

!  total clock time spent in the package

        REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time spent forming the preconditioner K_g

        REAL ( KIND = wp ) :: clock_form = 0.0

!  clock time spent factorizing K_G

        REAL ( KIND = wp ) :: clock_factorize = 0.0

!  clock time spent solving linear systems inolving K_G

        REAL ( KIND = wp ) :: clock_apply = 0.0
      END TYPE

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: SBLS_inform_type

!  return status. See SBLS_form_and_factorize for details

        INTEGER :: status = 0

!  the status of the last attempted allocation/deallocation

        INTEGER :: alloc_status = 0

!  the name of the array for which an allocation/deallocation error ocurred

        CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  obsolete return status from the factorization routines

        INTEGER :: sils_analyse_status = 0      !                     (OBSOLETE)
        INTEGER :: sils_factorize_status = 0    !                     (OBSOLETE)
        INTEGER :: sils_solve_status = 0        !                     (OBSOLETE)
        INTEGER :: sls_analyse_status = 0       !                     (OBSOLETE)
        INTEGER :: sls_factorize_status = 0     !                     (OBSOLETE)
        INTEGER :: sls_solve_status = 0         !                     (OBSOLETE)
        INTEGER :: uls_analyse_status = 0       !                     (OBSOLETE)
        INTEGER :: uls_factorize_status = 0     !                     (OBSOLETE)
        INTEGER :: uls_solve_status = 0         !                     (OBSOLETE)

!  the return status from the sorting routines

        INTEGER :: sort_status = 0

!  the total integer workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_integer = - 1

!  the total real workspace required for the factorization

        INTEGER ( KIND = long ) :: factorization_real = - 1

!  the preconditioner used

        INTEGER :: preconditioner = 0

!  the factorization used

        INTEGER :: factorization = 0

!  how many of the diagonals in the factorization are positive

        INTEGER :: d_plus = - 1

!  the computed rank of A

        INTEGER :: rank = - 1

!  is the matrix A rank defficient?

        LOGICAL :: rank_def = .FALSE.

!  has the used preconditioner been perturbed to guarantee correct inertia?

        LOGICAL :: perturbed = .FALSE.

!  the total number of projected CG iterations required

        INTEGER :: iter_pcg = 0

!  the norm of the residual

        REAL ( KIND = wp ) :: norm_residual = - 1.0_wp

!  has an "alternative" y: K y = 0 and yT c > 0 been found when trying to
!  solve "K y = c" ?

        LOGICAL :: alternative = .FALSE.

!  timings (see above)

        TYPE ( SBLS_time_type ) :: time

!  inform parameters for SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

!  inform parameters for ULS

        TYPE ( ULS_inform_type ) :: ULS_inform
      END TYPE

!  ...............................
!   explicit factors derived type
!  ...............................

      TYPE, PUBLIC :: SBLS_explicit_factors_type
        PRIVATE
        INTEGER :: rank_a, rank_k, g_ne, k_g, k_c, k_pert, len_s, len_s_max
        INTEGER :: len_sol_workspace = - 1
        INTEGER :: len_sol_workspace_lm = - 1
        INTEGER :: len_part_sol_workspace = - 1
        LOGICAL :: analyse = .TRUE.
        LOGICAL :: lm = .FALSE.
        TYPE ( SMT_type ) :: K, B
        TYPE ( SLS_data_type ) :: K_data
        TYPE ( SLS_control_type ) :: K_control
        TYPE ( ULS_data_type ) :: B_data
        TYPE ( ULS_control_type ) :: B_control
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_ROWS
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_COLS
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_COLS_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_col_ptr
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_by_rows
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_row_ptr
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_by_cols
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW2
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IPIV
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RW
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_diag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_orig
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_o
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_u
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_w
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: S
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Z
      END TYPE

!  ...............................
!   implicit factors derived type
!  ...............................

      TYPE, PUBLIC :: SBLS_implicit_factors_type
        PRIVATE
        INTEGER :: rank_a, m, n, k_n, n_r
        INTEGER :: len_sol_workspace = - 1
        LOGICAL :: unitb22, unitb31, unitp22, zerob32, zerob33, zerop11, zerop21
        TYPE ( SMT_type ) :: A1, A2, P11, P21, B22, B32, B33
        TYPE ( ULS_data_type ) :: A1_data
        TYPE ( ULS_control_type ) :: A1_control
        TYPE ( SLS_data_type ) :: B22_data
        TYPE ( SLS_control_type ) :: B22_control
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_COLS_order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_COLS_basic
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_orig
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL_perm
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL_current
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERT
      END TYPE

!  .................................
!   null-space factors derived type
!  .................................

      TYPE, PUBLIC :: SBLS_null_space_factors_type
        PRIVATE
        INTEGER :: rank_a, m, n, k_n, n_r
        INTEGER :: len_sol_workspace = - 1
        TYPE ( SMT_type ) :: A1, A2, H11, H21, H22
        TYPE ( ULS_data_type ) :: A1_data
        TYPE ( ULS_control_type ) :: A1_control
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_COLS_order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_COLS_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_orig
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL_current
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERT
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: R
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: R_factors
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: SOL_perm
      END TYPE

!  ...................
!   data derived type
!  ...................

      TYPE, PUBLIC :: SBLS_data_type
        INTEGER :: last_preconditioner = no_last
        INTEGER :: last_factorization = no_last
        INTEGER :: last_n = no_last
        INTEGER :: last_npm = no_last
        INTEGER :: len_sol
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_d
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V
        TYPE ( SBLS_explicit_factors_type ) :: efactors
        TYPE ( SBLS_implicit_factors_type ) :: ifactors
        TYPE ( SBLS_null_space_factors_type ) :: nfactors
      END TYPE

   CONTAINS

!-*-*-*-*-*-   S B L S  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE SBLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for SBLS. This routine should be called before
!  SBLS_solve
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

      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( OUT ) :: control
      TYPE ( SBLS_inform_type ), INTENT( OUT ) :: inform

      control%stop_absolute = epsmch ** 0.33
      control%stop_relative = epsmch ** 0.33
      control%SLS_control%prefix = '" - SLS:"                    '
      control%ULS_control%prefix = '" - ULS:"                    '
      data%last_preconditioner = no_last
      data%last_factorization = no_last
      data%efactors%len_sol_workspace = - 1
      data%efactors%len_sol_workspace_lm = - 1
      data%efactors%len_part_sol_workspace = - 1
      data%ifactors%len_sol_workspace = - 1
      data%nfactors%len_sol_workspace = - 1

      inform%status = GALAHAD_ok
      RETURN

!  End of SBLS_initialize

      END SUBROUTINE SBLS_initialize

!-*-*-*-   S B L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE SBLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SBLS_initialize could (roughly)
!  have been set as:

! BEGIN SBLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  initial-workspace-for-unsymmetric-solver          1000
!  initial-integer-workspace                         1000       (obsolete)
!  initial-real-workspace                            1000       (obsolete)
!  maximum-refinements                               1
!  maximum-pcg-iterations                            1000
!  preconditioner-used                               0
!  semi-bandwidth-for-band-preconditioner            5
!  factorization-used                                0
!  maximum-column-nonzeros-in-schur-complement       35
!  ordering-used                                     3          (obsolete)
!  scaling-used                                      0          (obsolete)
!  has-a-changed                                     2
!  has-h-changed                                     2
!  has-c-changed                                     2
!  minimum-diagonal                                  1.0D-5
!  pivot-tolerance-used                              1.0D-12    (obsolete)
!  pivot-tolerance-used-for-basis                    0.5
!  zero-pivot-tolerance                              1.0D-12    (obsolete)
!  static-pivoting-diagonal-perturbation             0.0D+0     (obsolete)
!  level-at-which-to-switch-to-static                0.0D+0     (obsolete)
!  absolute-accuracy                                 1.0D-6
!  relative-accuracy                                 1.0D-6
!  find-basis-by-transpose                           T
!  check-for-reliable-basis                          T
!  affine-constraints                                F
!  allow-singular-preconditioner                     F
!  remove-linear-dependencies                        T
!  get-norm-residual                                 F
!  perturb-to-make-+ve-definite                      T
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  symmetric-linear-equation-solver                  sils
!  definite-linear-equation-solver                   sils
!  unsymmetric-linear-equation-solver                gls
!  output-line-prefix                                ""
! END SBLS SPECIFICATIONS

!  Dummy arguments

      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: control
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: error = 1
      INTEGER, PARAMETER :: out = error + 1
      INTEGER, PARAMETER :: print_level = out + 1
      INTEGER, PARAMETER :: indmin = print_level + 1
      INTEGER, PARAMETER :: valmin = indmin + 1
      INTEGER, PARAMETER :: len_ulsmin = valmin + 1
      INTEGER, PARAMETER :: itref_max = len_ulsmin + 1
      INTEGER, PARAMETER :: maxit_pcg = itref_max + 1
      INTEGER, PARAMETER :: new_a = maxit_pcg + 1
      INTEGER, PARAMETER :: new_h = new_a + 1
      INTEGER, PARAMETER :: new_c = new_h + 1
      INTEGER, PARAMETER :: preconditioner = new_c + 1
      INTEGER, PARAMETER :: semi_bandwidth = preconditioner + 1
      INTEGER, PARAMETER :: factorization = semi_bandwidth + 1
      INTEGER, PARAMETER :: max_col = factorization + 1
      INTEGER, PARAMETER :: scaling = max_col + 1
      INTEGER, PARAMETER :: ordering = scaling + 1
      INTEGER, PARAMETER :: pivot_tol = ordering + 1
      INTEGER, PARAMETER :: pivot_tol_for_basis = pivot_tol + 1
      INTEGER, PARAMETER :: zero_pivot = pivot_tol_for_basis + 1
      INTEGER, PARAMETER :: static_tolerance = zero_pivot + 1
      INTEGER, PARAMETER :: static_level  = static_tolerance + 1
      INTEGER, PARAMETER :: min_diagonal = static_level  + 1
      INTEGER, PARAMETER :: stop_absolute = min_diagonal + 1
      INTEGER, PARAMETER :: stop_relative = stop_absolute + 1
      INTEGER, PARAMETER :: remove_dependencies = stop_relative + 1
      INTEGER, PARAMETER :: find_basis_by_transpose = remove_dependencies + 1
      INTEGER, PARAMETER :: affine = find_basis_by_transpose + 1
      INTEGER, PARAMETER :: allow_singular = affine + 1
      INTEGER, PARAMETER :: perturb_to_make_definite = allow_singular + 1
      INTEGER, PARAMETER :: get_norm_residual = perturb_to_make_definite + 1
      INTEGER, PARAMETER :: check_basis = get_norm_residual + 1
      INTEGER, PARAMETER :: space_critical  = check_basis + 1
      INTEGER, PARAMETER :: deallocate_error_fatal = space_critical  + 1
      INTEGER, PARAMETER :: symmetric_linear_solver = deallocate_error_fatal + 1
      INTEGER, PARAMETER :: definite_linear_solver = symmetric_linear_solver + 1
      INTEGER, PARAMETER :: unsymmetric_linear_solver =                        &
                              definite_linear_solver + 1
      INTEGER, PARAMETER :: prefix = unsymmetric_linear_solver + 1
      INTEGER, PARAMETER :: lspec = prefix
      CHARACTER( LEN = 4 ), PARAMETER :: specname = 'SBLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

      spec%keyword = ''

!  Integer key-words

      spec( error )%keyword = 'error-printout-device'
      spec( out )%keyword = 'printout-device'
      spec( print_level )%keyword = 'print-level'
      spec( indmin )%keyword = 'initial-integer-workspace'
      spec( valmin )%keyword = 'initial-real-workspace'
      spec( len_ulsmin )%keyword = 'initial-workspace-for-unsymmetric-solver'
      spec( itref_max )%keyword = 'maximum-refinements'
      spec( maxit_pcg )%keyword = 'maximum-pcg-iterations'
      spec( new_a )%keyword = 'has-a-changed'
      spec( new_h )%keyword = 'has-h-changed'
      spec( new_c )%keyword = 'has-c-changed'
      spec( preconditioner )%keyword = 'preconditioner-used'
      spec( semi_bandwidth )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( factorization )%keyword = 'factorization-used'
      spec( max_col )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec( scaling )%keyword = 'scaling-used'
      spec( ordering )%keyword = 'ordering-used'

!  Real key-words

      spec( pivot_tol )%keyword = 'pivot-tolerance-used'
      spec( pivot_tol_for_basis )%keyword = 'pivot-tolerance-used-for-basis'
      spec( zero_pivot )%keyword = 'zero-pivot-tolerance'
      spec( static_tolerance )%keyword = 'static-pivoting-diagonal-perturbation'
      spec( static_level )%keyword = 'level-at-which-to-switch-to-static'
      spec( min_diagonal )%keyword = 'minimum-diagonal'
      spec( stop_absolute )%keyword = 'absolute-accuracy'
      spec( stop_relative )%keyword = 'relative-accuracy'

!  Logical key-words

      spec( remove_dependencies )%keyword = 'remove-linear-dependencies'
      spec( find_basis_by_transpose )%keyword = 'find-basis-by-transpose'
      spec( affine )%keyword = 'affine-constraints'
      spec( allow_singular )%keyword = 'allow-singular-preconditioner'
      spec( perturb_to_make_definite )%keyword = 'perturb-to-make-+ve-definite'
      spec( get_norm_residual )%keyword = 'get-norm-residual'
      spec( check_basis )%keyword = 'check-for-reliable-basis'
      spec( space_critical )%keyword = 'space-critical'
      spec( deallocate_error_fatal )%keyword = 'deallocate-error-fatal'

!  Character key-words

      spec( symmetric_linear_solver )%keyword =                                &
        'symmetric-linear-equation-solver'
      spec( definite_linear_solver )%keyword = 'definite-linear-equation-solver'
      spec( unsymmetric_linear_solver )%keyword =                              &
        'unsymmetric-linear-equation-solver'
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
      CALL SPECFILE_assign_value( spec( len_ulsmin ),                          &
                                  control%len_ulsmin,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( itref_max ),                           &
                                  control%itref_max,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( maxit_pcg ),                           &
                                  control%maxit_pcg,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_a ),                               &
                                  control%new_a,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_h ),                               &
                                  control%new_h,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( new_c ),                               &
                                  control%new_c,                               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( preconditioner ),                      &
                                  control%preconditioner,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( semi_bandwidth ),                      &
                                  control%semi_bandwidth,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( factorization ),                       &
                                  control%factorization,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_col ),                             &
                                  control%max_col,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec(ordering ),                             &
                                  control%ordering,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( scaling ),                             &
                                  control%scaling,                             &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( pivot_tol ),                           &
                                  control%pivot_tol,                           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( pivot_tol_for_basis ),                 &
                                  control%pivot_tol_for_basis,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( zero_pivot ),                          &
                                  control%zero_pivot,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( static_tolerance ),                    &
                                  control%static_tolerance,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( static_level ),                        &
                                  control%static_level,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( min_diagonal ),                        &
                                  control%min_diagonal,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_absolute ),                       &
                                  control%stop_absolute,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( stop_relative ),                       &
                                  control%stop_relative,                       &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( remove_dependencies ),                 &
                                  control%remove_dependencies,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( find_basis_by_transpose ),             &
                                  control%find_basis_by_transpose,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( affine ),                              &
                                  control%affine,                              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( allow_singular ),                      &
                                  control%allow_singular,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( perturb_to_make_definite ),            &
                                  control%perturb_to_make_definite,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( get_norm_residual ),                   &
                                  control%get_norm_residual,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( check_basis ),                         &
                                  control%check_basis,                         &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal ),              &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

      CALL SPECFILE_assign_value( spec( symmetric_linear_solver ),             &
                                  control%symmetric_linear_solver,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( definite_linear_solver ),              &
                                  control%definite_linear_solver,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( unsymmetric_linear_solver ),           &
                                  control%unsymmetric_linear_solver,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( prefix ),                              &
                                  control%prefix,                              &
                                  control%error )

!  Read the specfile for SLS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL SLS_read_specfile( control%SLS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-SLS' )
      ELSE
        CALL SLS_read_specfile( control%SLS_control, device )
      END IF

!  Read the specfile for ULS

      IF ( PRESENT( alt_specname ) ) THEN
        CALL ULS_read_specfile( control%ULS_control, device,                   &
                                alt_specname = TRIM( alt_specname ) // '-ULS' )
      ELSE
        CALL ULS_read_specfile( control%ULS_control, device )
      END IF

      RETURN

      END SUBROUTINE SBLS_read_specfile

!-*-*-*-*-*-   S B L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE SBLS_terminate( data, control, inform )

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
!   data    see Subroutine SBLS_initialize
!   control see Subroutine SBLS_initialize
!   inform  see Subroutine SBLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Ensure that future calls know that arrays have been deallocated

      data%efactors%len_sol_workspace = - 1
      data%efactors%len_sol_workspace_lm = - 1
      data%efactors%len_part_sol_workspace = - 1
      data%ifactors%len_sol_workspace = - 1
      data%nfactors%len_sol_workspace = - 1
      data%last_preconditioner = no_last
      data%last_factorization = no_last
      data%last_n = no_last
      data%last_npm = no_last

!  Deallocate all arrays allocated within SLS and ULS

      CALL SLS_terminate( data%efactors%K_data, data%efactors%K_control,       &
                          inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL ULS_terminate( data%efactors%B_data, data%efactors%B_control,       &
                          inform%ULS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%ULS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL ULS_terminate( data%ifactors%A1_data, data%ifactors%A1_control,     &
                          inform%ULS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%ULS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL SLS_terminate( data%ifactors%B22_data, data%ifactors%B22_control,   &
                          inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL ULS_terminate( data%nfactors%A1_data, data%nfactors%A1_control,     &
                          inform%ULS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%ULS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'sbls: G'
      CALL SPACE_dealloc_array( data%G,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: P'
      CALL SPACE_dealloc_array( data%P,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: R'
      CALL SPACE_dealloc_array( data%R,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: RHS_d'
      CALL SPACE_dealloc_array( data%RHS_d,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: V'
      CALL SPACE_dealloc_array( data%V,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%A_col_ptr'
      CALL SPACE_dealloc_array( data%efactors%A_col_ptr,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%A_row_ptr'
      CALL SPACE_dealloc_array( data%efactors%A_row_ptr,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%A_by_rows'
      CALL SPACE_dealloc_array( data%efactors%A_by_rows,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%A_by_cols'
      CALL SPACE_dealloc_array( data%efactors%A_by_cols,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%A_row'
      CALL SPACE_dealloc_array( data%efactors%A_row,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%W'
      CALL SPACE_dealloc_array( data%efactors%W,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%RW'
      CALL SPACE_dealloc_array( data%efactors%RW,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%IW'
      CALL SPACE_dealloc_array( data%efactors%IW,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%IW2'
      CALL SPACE_dealloc_array( data%efactors%IW2,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%IPIV'
      CALL SPACE_dealloc_array( data%efactors%IPIV,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%K%row'
      CALL SPACE_dealloc_array( data%efactors%K%row,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%K%col'
      CALL SPACE_dealloc_array( data%efactors%K%col,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%K%val'
      CALL SPACE_dealloc_array( data%efactors%K%val,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%G_diag'
      CALL SPACE_dealloc_array( data%efactors%G_diag,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%B%row'
      CALL SPACE_dealloc_array( data%efactors%B%row,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%B%col'
      CALL SPACE_dealloc_array( data%efactors%B%col,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%B%val'
      CALL SPACE_dealloc_array( data%efactors%B%val,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%B_ROWS'
      CALL SPACE_dealloc_array( data%efactors%B_ROWS,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%B_COLS'
      CALL SPACE_dealloc_array( data%efactors%B_COLS,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%B_COLS_basic'
      CALL SPACE_dealloc_array( data%efactors%B_COLS_basic,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A1%row'
      CALL SPACE_dealloc_array( data%ifactors%A1%row,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A1%col'
      CALL SPACE_dealloc_array( data%ifactors%A1%col,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A1%val'
      CALL SPACE_dealloc_array( data%ifactors%A1%val,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A_ROWS_basic'
      CALL SPACE_dealloc_array( data%ifactors%A_ROWS_basic,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A_COLS_basic'
      CALL SPACE_dealloc_array( data%ifactors%A_COLS_basic,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A_ROWS_order'
      CALL SPACE_dealloc_array( data%ifactors%A_ROWS_order,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A_COLS_order'
      CALL SPACE_dealloc_array( data%ifactors%A_COLS_order,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A1%row'
      CALL SPACE_dealloc_array( data%ifactors%A1%row,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A1%col'
      CALL SPACE_dealloc_array( data%ifactors%A1%col,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A1%val'
      CALL SPACE_dealloc_array( data%ifactors%A1%val,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A2%row'
      CALL SPACE_dealloc_array( data%ifactors%A2%row,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A2%col'
      CALL SPACE_dealloc_array( data%ifactors%A2%col,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%A2%val'
      CALL SPACE_dealloc_array( data%ifactors%A2%val,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%B22%row'
      CALL SPACE_dealloc_array( data%ifactors%B22%row,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%B22%col'
      CALL SPACE_dealloc_array( data%ifactors%B22%col,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%B22%val'
      CALL SPACE_dealloc_array( data%ifactors%B22%val,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%RHS_orig'
      CALL SPACE_dealloc_array( data%ifactors%RHS_orig,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%SOL_perm'
      CALL SPACE_dealloc_array( data%ifactors%SOL_perm,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%SOL_current'
      CALL SPACE_dealloc_array( data%ifactors%SOL_current,                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: ifactors%PERT'
      CALL SPACE_dealloc_array( data%ifactors%PERT,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A1%row'
      CALL SPACE_dealloc_array( data%nfactors%A1%row,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A1%col'
      CALL SPACE_dealloc_array( data%nfactors%A1%col,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A1%val'
      CALL SPACE_dealloc_array( data%nfactors%A1%val,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A_ROWS_basic'
      CALL SPACE_dealloc_array( data%nfactors%A_ROWS_basic,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A_COLS_basic'
      CALL SPACE_dealloc_array( data%nfactors%A_COLS_basic,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A_ROWS_order'
      CALL SPACE_dealloc_array( data%nfactors%A_ROWS_order,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A_COLS_order'
      CALL SPACE_dealloc_array( data%nfactors%A_COLS_order,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%IW'
      CALL SPACE_dealloc_array( data%nfactors%IW,                              &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A2%row'
      CALL SPACE_dealloc_array( data%nfactors%A2%row,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A2%col'
      CALL SPACE_dealloc_array( data%nfactors%A2%col,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%A2%val'
      CALL SPACE_dealloc_array( data%nfactors%A2%val,                          &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H11%row'
      CALL SPACE_dealloc_array( data%nfactors%H11%row,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H11%col'
      CALL SPACE_dealloc_array( data%nfactors%H11%col,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H11%val'
      CALL SPACE_dealloc_array( data%nfactors%H11%val,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H21%row'
      CALL SPACE_dealloc_array( data%nfactors%H21%row,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H21%col'
      CALL SPACE_dealloc_array( data%nfactors%H21%col,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H21%val'
      CALL SPACE_dealloc_array( data%nfactors%H21%val,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H22%row'
      CALL SPACE_dealloc_array( data%nfactors%H22%row,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H22%col'
      CALL SPACE_dealloc_array( data%nfactors%H22%col,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H22%ptr'
      CALL SPACE_dealloc_array( data%nfactors%H22%ptr,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%H22%val'
      CALL SPACE_dealloc_array( data%nfactors%H22%val,                         &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%V'
      CALL SPACE_dealloc_array( data%nfactors%V,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%W'
      CALL SPACE_dealloc_array( data%nfactors%W,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%RHS'
      CALL SPACE_dealloc_array( data%efactors%RHS,                             &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%RHS_orig'
      CALL SPACE_dealloc_array( data%efactors%RHS_orig,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%RHS_o'
      CALL SPACE_dealloc_array( data%efactors%RHS_o,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%RHS_u'
      CALL SPACE_dealloc_array( data%efactors%RHS_u,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%RHS_w'
      CALL SPACE_dealloc_array( data%efactors%RHS_w,                           &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: efactors%Z'
      CALL SPACE_dealloc_array( data%efactors%Z,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%SOL_current'
      CALL SPACE_dealloc_array( data%nfactors%SOL_current,                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%PERT'
      CALL SPACE_dealloc_array( data%nfactors%PERT,                            &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%R'
      CALL SPACE_dealloc_array( data%nfactors%R,                               &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%R_factors'
      CALL SPACE_dealloc_array( data%nfactors%R_factors,                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%SOL_perm'
      CALL SPACE_dealloc_array( data%nfactors%SOL_perm,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      RETURN

!  End of subroutine SBLS_terminate

      END SUBROUTINE SBLS_terminate

!-*-   S B L S _ F O R M _ A N D _ F A C T O R I Z E  S U B R O U T I N E   -*-

      SUBROUTINE SBLS_form_and_factorize( n, m, H, A, C, data,                 &
                                          control, inform, D, H_lm )

!  Form and factorize
!
!        K = ( G   A^T )
!            ( A   -C  )
!
!  for various approximations G of H

! inform%status:
!
!   0  successful termination
!  +1 A is rank defficient
!  -1  allocation error
!  -2  deallocation error
!  -3 input parameter out of range
!  -4 SLS analyse error
!  -5 SLS factorize error
!  -6 SLS solve error
!  -7 ULS factorize error
!  -8 ULS solve error
!  -9 insufficient preconditioner

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: D
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( IN ) :: H_lm

!  Local variables

      INTEGER :: c_ne
!     INTEGER :: i
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!     WRITE(6,"( ' H type = ', A )" ) SMT_get( H%type )
!     WRITE(6,"( ' H, m = ', I0, ', n = ', I0 )" ) H%m, H%n
!     DO i = 1, H%ne
!       WRITE( 6, "( 3I7, ES12.4 )" ) i, H%row( i ), H%col( i ), H%val( i )
!     END DO

!     WRITE(6,"( ' A type = ', A )" ) SMT_get( A%type )
!     WRITE(6,"( ' A, m = ', I0, ', n = ', I0 )" ) A%m, A%n
!     DO i = 1, A%ne
!       WRITE( 6, "( 3I7, ES12.4 )" ) i, A%row( i ), A%col( i ), A%val( i )
!     END DO

!     WRITE(6,"( ' C type = ', A )" ) SMT_get( C%type )
!     WRITE(6,"( ' C, m = ', I0, ', n = ', I0 )" ) C%m, C%n
!     DO i = 1, C%ne
!       WRITE( 6, "( 3I7, ES12.4 )" ) i, C%row( i ), C%col( i ), C%val( i )
!     END DO

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Set default information values

      inform%status = GALAHAD_ok
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%sls_analyse_status = 0 ; inform%sls_factorize_status = 0
      inform%sls_solve_status = 0
      inform%uls_factorize_status = 0 ; inform%uls_solve_status = 0
      inform%sort_status = 0
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%rank = m ; inform%rank_def = .FALSE.
      IF ( inform%perturbed )  data%efactors%analyse = .TRUE.
      inform%perturbed = .FALSE.
      inform%norm_residual = - one

!  Check for faulty dimensions

      IF ( n <= 0 .OR. m < 0 .OR.                                              &
           .NOT. QPT_keyword_H( H%type ) .OR.                                  &
           .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) prefix, inform%status
        GO TO 900
      END IF

      IF ( control%out >= 0 .AND. control%print_level >= 1 ) THEN
        WRITE( control%out,                                                    &
          "( /, A, ' n = ', I0, ', m = ', I0 )" ) prefix, n, m
      END IF

!  Ensure automatic choices for the preconditioner/factorization have been made

      IF ( data%last_preconditioner /= control%preconditioner ) THEN
        IF ( control%preconditioner == 0 ) THEN
          IF ( data%last_preconditioner == no_last ) THEN
            IF ( PRESENT( H_lm ) ) THEN
              inform%preconditioner = 6
            ELSE
              inform%preconditioner = 2
            END IF
          ELSE
            inform%preconditioner = data%last_preconditioner
          END IF
        ELSE
          inform%preconditioner = control%preconditioner
        END IF
      ELSE
        inform%preconditioner = control%preconditioner
      END IF
      IF ( data%last_factorization /= control%factorization ) THEN
        IF ( control%factorization == 0 ) THEN
          IF ( data%last_factorization == no_last ) THEN
            inform%factorization = 1
          ELSE
            inform%factorization = data%last_factorization
          END IF
        ELSE
          inform%factorization = control%factorization
        END IF
      ELSE
        inform%factorization = control%factorization
      END IF

!  Only allow the null-space method if C = 0

      IF ( inform%factorization == 3 ) THEN
        IF ( SMT_get( C%type ) == 'ZERO' .OR.                                  &
             SMT_get( C%type ) == 'NONE' ) THEN
          c_ne = 0
        ELSE IF ( SMT_get( C%type ) == 'DIAGONAL' .OR.                         &
                  SMT_get( C%type ) == 'SCALED_IDENTITY' .OR.                  &
                  SMT_get( C%type ) == 'IDENTITY' ) THEN
          c_ne = m
        ELSE IF ( SMT_get( C%type ) == 'DENSE' ) THEN
          c_ne = ( m * ( m + 1 ) ) / 2
        ELSE IF ( SMT_get( C%type ) == 'SPARSE_BY_ROWS' ) THEN
          c_ne = C%ptr( m + 1 ) - 1
        ELSE
          c_ne = C%ne
        END IF
        IF ( c_ne /= 0 ) inform%factorization = 1
      END IF

      inform%rank_def = .FALSE.

      IF ( control%out >= 0 .AND. control%print_level >= 1 ) THEN
        IF ( control%factorization == 1 ) THEN
          WRITE( control%out,                                                  &
            "( A, ' preconditioner = ', I0, ', factorization = ', I0,          &
         &     ', solver = ', A )" ) prefix, inform%preconditioner,            &
             inform%factorization, TRIM( control%definite_linear_solver )
        ELSE
          WRITE( control%out,                                                  &
            "( A, ' preconditioner = ', I0, ', factorization = ', I0,          &
         &     ', solver = ', A )" ) prefix, inform%preconditioner,            &
             inform%factorization, TRIM( control%symmetric_linear_solver )
        END IF
      END IF

!  Form and factorize the preconditioner

      IF ( inform%factorization == 3 ) THEN
        CALL SBLS_form_n_factorize_nullspace( n, m, H, A, data%nfactors,       &
                                              data%last_factorization,         &
                                              control, inform )
        data%len_sol = data%nfactors%k_n
      ELSE IF ( inform%preconditioner >= 0 ) THEN
        CALL SBLS_form_n_factorize_explicit( n, m, H, A, C, data%efactors,     &
                                             data%last_factorization,          &
                                             control, inform, D, H_lm )
        data%len_sol = data%efactors%K%n
      ELSE
!       CALL SBLS_form_n_factorize_implicit( n, m, H, A, C, data%ifactors,     &
        CALL SBLS_form_n_factorize_implicit( n, m, H, A, data%ifactors,        &
                                             data%last_factorization,          &
                                             control, inform )
        data%len_sol = data%ifactors%k_n
      END IF

      data%last_preconditioner = inform%preconditioner
      data%last_factorization = inform%factorization

!  record total time

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  Non-executable statements

 2010 FORMAT( ' ', /, A, '   **  Error return ',I0,' from SBLS ' )

!  End of subroutine SBLS_form_and_factorize

      END SUBROUTINE SBLS_form_and_factorize

!-*-*-   S B L S _ FORM _ N _ FACTORIZE _ EXPLICIT   S U B R O U T I N E   -*-

      SUBROUTINE SBLS_form_n_factorize_explicit( n, m, H, A, C, efactors,      &
                                                 last_factorization,           &
                                                 control, inform, D, H_lm )

!  Form and explicitly factorize
!
!        K = ( G   A^T )
!            ( A    -C )
!
!  for various approximations G of H

!  G is chosen according to the parameter control%proconditioner as follows:

!      1  no preconditioner, G = I
!      2  full factorization, G = H
!      3  diagonal, G = diag( H ) with minimum diagonal control%min_diagonal
!      4  banded, G = band(H) with semi-bandwidth control%semi_bandwidth
!      5  optionally supplied diagonal, G = D
!      6  optionally suppled limited-memory G = H_lm
!      7  optionally suppled limited-memory plus full, G = H_lm + H
!      8  optionally supplied limited-memory plus diagonal, G = H_lm + D
!     11  G_11 = 0, G_21 = 0, G_22 = H_22 (**)
!     12  G_11 = 0, G_21 = H_21, G_22 = H_22 (**)

!  (**) Here the columns of A and H are reordered so that A = (A_1 : A:2) P
!       for some P for which A_1 is non-singular and reasonably conditioned

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, last_factorization
      TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SBLS_explicit_factors_type ), INTENT( INOUT ) :: efactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), OPTIONAL, INTENT( IN ), DIMENSION( n ) :: D
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( IN ) :: H_lm

!  Local variables

      INTEGER :: i, ii, ip, j, jp, k, kk, l, g_ne, kzero, kminus, nb, out, c_ne
      INTEGER :: nnz_col_j, nnz_aat_old, nnz_aat, max_len, new_pos, a_ne, h_ne
      INTEGER :: new_h, new_a, new_c, k_c, k_ne, np1, npm, oi, oj, lw, lrw
      REAL :: time_start, time, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now, clock
      REAL ( KIND = wp ) :: al, val
      LOGICAL :: printi, resize, use_schur_complement
      LOGICAL :: fail_if_not_sc, numerical_pivoting
      CHARACTER ( LEN = 80 ) :: array_name
      INTEGER :: ILAENV
      EXTERNAL :: ILAENV

!     REAL ( KIND = wp ) :: DD(2,A%m+A%n)

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      out = control%out
      printi = control%print_level >= 1 .AND. out >= 0

      IF ( inform%preconditioner <= 0 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Use SBLS_form_n_factorize_implicit subroutine instead' )" )  &
            prefix
        inform%status = GALAHAD_error_call_order ; RETURN
      END IF

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne
      END IF

      IF ( SMT_get( H%type ) == 'DIAGONAL' .OR.                                &
           SMT_get( H%type ) == 'SCALED_IDENTITY' .OR.                         &
           SMT_get( H%type ) == 'IDENTITY' ) THEN
        h_ne = n
      ELSE IF ( SMT_get( H%type ) == 'DENSE' ) THEN
        h_ne = ( n * ( n + 1 ) ) / 2
      ELSE IF ( SMT_get( H%type ) == 'SPARSE_BY_ROWS' ) THEN
        h_ne = H%ptr( n + 1 ) - 1
      ELSE
        h_ne = H%ne
      END IF

      IF ( SMT_get( C%type ) == 'ZERO' .OR.                                    &
           SMT_get( C%type ) == 'NONE' ) THEN
        c_ne = 0
      ELSE IF ( SMT_get( C%type ) == 'DIAGONAL' .OR.                           &
                SMT_get( C%type ) == 'SCALED_IDENTITY' .OR.                    &
                SMT_get( C%type ) == 'IDENTITY' ) THEN
        c_ne = m
      ELSE IF ( SMT_get( C%type ) == 'DENSE' ) THEN
        c_ne = ( m * ( m + 1 ) ) / 2
      ELSE IF ( SMT_get( C%type ) == 'SPARSE_BY_ROWS' ) THEN
        c_ne = C%ptr( m + 1 ) - 1
      ELSE
        c_ne = C%ne
      END IF

      IF ( last_factorization /= inform%factorization ) THEN
        new_h = 2
        new_a = 2
        new_c = 2
      ELSE
        new_h = control%new_h
        new_a = control%new_a
        new_c = control%new_c
      END IF

!  check that optional arguments are available if required, and provide
!  remedial alternatives

      IF ( new_h > 0 ) THEN

!  Check to see D is present

        IF ( inform%preconditioner == 5 ) THEN
          IF ( .NOT. PRESENT( D ) ) inform%preconditioner = 1

!  Check to see H_lm is present

        ELSE IF ( inform%preconditioner == 6 .OR.                              &
                  inform%preconditioner == 7 ) THEN
          IF ( .NOT. PRESENT( H_lm ) ) inform%preconditioner = 1

!  Check to see D and H_lm are present

        ELSE IF ( inform%preconditioner == 8 ) THEN
          IF ( .NOT. ( PRESENT( D ) .AND. PRESENT( H_lm ) ) )                  &
            inform%preconditioner = 6
        END IF
      END IF

!IF (  PRESENT( H_lm ) ) write(6,*) ' n, H_lm%n_restriction, H_lm%n',          &
! n, H_lm%n_restriction, H_lm%n

!  Form the preconditioner

!  First, see if we can get away with a factorization of the Schur complement.

!   =======================
!    USE SCHUR COMPLEMENT
!   =======================

      IF ( inform%factorization == 0 .OR. inform%factorization == 1 .OR.       &
           inform%factorization == 4 .OR. inform%factorization == 5 ) THEN
        fail_if_not_sc = inform%factorization >= 4
        numerical_pivoting = inform%factorization == 5
        inform%factorization = 1
        array_name = 'sbls: efactors%IW'
        CALL SPACE_resize_array( n, efactors%IW,                               &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        IF ( new_h > 0 ) THEN
          IF ( inform%preconditioner > 8 ) THEN
            inform%factorization = 2

!  Check to see if there are off-diagonal entries, and that the diagonal
!  is present

          ELSE IF ( inform%preconditioner == 2 .OR.                            &
                    inform%preconditioner == 4  ) THEN
            efactors%IW = 0
            SELECT CASE ( SMT_get( H%type ) )
            CASE ( 'ZERO', 'NONE' )
              inform%factorization = 2
            CASE ( 'DIAGONAL' )
              IF ( COUNT( H%val( : n ) == zero ) > 0 ) inform%factorization = 2
            CASE ( 'SCALED_IDENTITY' )
              IF ( H%val( 1 ) == zero ) inform%factorization = 2
            CASE ( 'IDENTITY' )
            CASE ( 'DENSE' )
              inform%factorization = 2
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, n
                DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                  IF ( i /= H%col( l ) ) THEN
                    IF ( inform%preconditioner == 2 .OR.                       &
                         ( inform%preconditioner == 4                          &
                         .AND. control%semi_bandwidth /= 0 ) ) THEN
                      inform%factorization = 2
                      EXIT
                    END IF
                  ELSE IF ( H%val( l ) /= zero ) THEN
                    efactors%IW( i ) = efactors%IW( i ) + 1
                  END IF
                END DO
              END DO
              IF ( COUNT( efactors%IW > 0 ) /= n ) THEN
                IF ( fail_if_not_sc ) THEN
                  inform%status = GALAHAD_error_inertia
                  GO TO 900
                ELSE
                  inform%factorization = 2
                END IF
              END IF
            CASE ( 'COORDINATE' )
              DO l = 1, H%ne
                i = H%row( l )
                IF ( i /= H%col( l ) ) THEN
                  IF ( inform%preconditioner == 2 .OR.                         &
                       ( inform%preconditioner == 4                            &
                       .AND. control%semi_bandwidth /= 0 ) ) THEN
                    inform%factorization = 2
                    EXIT
                  END IF
                ELSE IF ( H%val( l ) /= zero ) THEN
                  efactors%IW( i ) = efactors%IW( i ) + 1
                END IF
              END DO
              IF ( COUNT( efactors%IW > 0 ) /= n ) THEN
                IF ( fail_if_not_sc ) THEN
                  inform%status = GALAHAD_error_inertia
                  GO TO 900
                ELSE
                  inform%factorization = 2
                END IF
              END IF
            END SELECT

!  Check to see D is nonzero

          ELSE IF ( inform%preconditioner == 5 ) THEN
            IF ( COUNT( D( : n ) == zero ) > 0 ) THEN
              IF ( fail_if_not_sc ) THEN
                inform%status = GALAHAD_error_inertia
                GO TO 900
              ELSE
                inform%factorization = 2
              END IF
            END IF

!  Check to see if there are off-diagonal entries

          ELSE IF ( inform%preconditioner == 7 ) THEN
            SELECT CASE ( SMT_get( H%type ) )
            CASE ( 'ZERO', 'NONE' )
            CASE ( 'DIAGONAL' )
            CASE ( 'SCALED_IDENTITY' )
            CASE ( 'IDENTITY' )
            CASE ( 'DENSE' )
              inform%factorization = 2
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, n
                DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                  IF ( i /= H%col( l ) ) THEN
                    IF ( fail_if_not_sc ) THEN
                      inform%status = GALAHAD_error_inertia
                      GO TO 900
                    ELSE
                      inform%factorization = 2 ; EXIT
                    END IF
                  END IF
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, H%ne
                IF ( H%row( l ) /= H%col( l ) ) THEN
                  IF ( fail_if_not_sc ) THEN
                    inform%status = GALAHAD_error_inertia
                    GO TO 900
                  ELSE
                    inform%factorization = 2 ; EXIT
                  END IF
                END IF
              END DO
            END SELECT
          END IF

!  If G is not non-singular and diagonal, use a factorization of the
!  augmented matrix instead

          IF ( inform%factorization == 2 ) THEN
            array_name = 'sbls: efactors%IW'
            CALL SPACE_dealloc_array( efactors%IW,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( control%deallocate_error_fatal .AND.                          &
                 inform%status /= GALAHAD_ok ) RETURN

            IF ( last_factorization /= inform%factorization ) THEN
              new_h = 2
              new_a = 2
              new_c = 2
            END IF
            GO TO 100
          END IF

!  G is diagonal. Now check to see if there are not too many entries in
!  any column of A. Find the number of entries in each column
!  (Only do this for sparse_by_row and coordinate storage)

          IF ( SMT_get( A%type ) /= ' DENSE' ) THEN
            array_name = 'sbls: efactors%A_col_ptr'
            CALL SPACE_resize_array( n + 1, efactors%A_col_ptr,                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN
          END IF
        END IF

        IF ( new_a == 2 ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            max_len = m
          CASE ( 'SPARSE_BY_ROWS' )
            efactors%A_col_ptr( 2 : ) = 0
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l ) + 1
                efactors%A_col_ptr( j ) = efactors%A_col_ptr( j ) + 1
              END DO
            END DO
            max_len = MAXVAL( efactors%A_col_ptr( 2 : ) )
          CASE ( 'COORDINATE' )
            efactors%A_col_ptr( 2 : ) = 0
            DO l = 1, A%ne
              j = A%col( l ) + 1
              efactors%A_col_ptr( j ) = efactors%A_col_ptr( j ) + 1
            END DO
            max_len = MAXVAL( efactors%A_col_ptr( 2 : ) )
          END SELECT

!  If the column with the largest number of entries exceeds max_col,
!  use a factorization of the augmented matrix instead

          IF ( printi ) WRITE( out, "( A,                                      &
         &  ' maximum, average column lengths of A = ', I0, ', ', F0.1, /,     &
         &  A, ' number of columns of A longer than maxcol = ', I0,            &
         &     ' is ',  I0 )" ) prefix,                                        &
            max_len, float( SUM( efactors%A_col_ptr( 2 : ) ) ) / float( n ),   &
            prefix,                                                            &
            control%max_col, COUNT( efactors%A_col_ptr( 2 : ) > control%max_col)

!         IF ( control%factorization == 0 .AND. max_len > control%max_col .AND.&
          IF ( inform%factorization == 1 .AND. max_len > control%max_col .AND. &
               m > max_sc ) THEN

            IF ( fail_if_not_sc ) THEN
              inform%status = GALAHAD_error_schur_complement
              GO TO 900
            END IF

            IF ( printi ) WRITE( out, "(                                       &
           &  A, ' - abandon the Schur-complement factorization', /, A,        &
           &  ' in favour of one of the augmented matrix')") prefix, prefix

            inform%factorization = 2

            array_name = 'sbls: efactors%IW'
            CALL SPACE_dealloc_array( efactors%IW,                             &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( control%deallocate_error_fatal .AND.                          &
                inform%status /= GALAHAD_ok ) RETURN

            array_name = 'sbls: efactors%A_col_ptr'
            CALL SPACE_dealloc_array( efactors%A_col_ptr,                      &
               inform%status, inform%alloc_status, array_name = array_name,    &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( control%deallocate_error_fatal .AND.                          &
                inform%status /= GALAHAD_ok ) RETURN

            IF ( last_factorization /= inform%factorization ) THEN
              new_h = 2
              new_a = 2
              new_c = 2
            END IF
            GO TO 100
          END IF

!  Now store A by rows. First find the number of entries in each row
!  (Only do this for coordinate storage, as it is already available
!  for storage by rows!)

          IF ( SMT_get( A%type ) == 'COORDINATE' ) THEN
            array_name = 'sbls: efactors%A_row_ptr'
            CALL SPACE_resize_array( m + 1, efactors%A_row_ptr,                &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN

            array_name = 'sbls: efactors%A_by_rows'
            CALL SPACE_resize_array( a_ne, efactors%A_by_rows,                 &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN

            efactors%A_row_ptr( 2 : ) = 0
            DO l = 1, A%ne
              i = A%row( l ) + 1
              efactors%A_row_ptr( i ) = efactors%A_row_ptr( i ) + 1
            END DO

!  Next assign row pointers

            efactors%A_row_ptr( 1 ) = 1
            DO i = 2, m + 1
              efactors%A_row_ptr( i )                                          &
               = efactors%A_row_ptr( i ) + efactors%A_row_ptr( i - 1 )
            END DO

!  Now record where the entries in each row occur in the original matrix

            DO l = 1, A%ne
              i = A%row( l )
              new_pos = efactors%A_row_ptr( i )
              efactors%A_by_rows( new_pos ) = l
              efactors%A_row_ptr( i ) = new_pos + 1
            END DO

!  Finally readjust the row pointers

            DO i = m + 1, 2, - 1
              efactors%A_row_ptr( i ) = efactors%A_row_ptr( i - 1 )
            END DO
            efactors%A_row_ptr( 1 ) = 1
          END IF

!  Also store A by columns, but with the entries sorted in increasing
!  row order within each column. First assign column pointers

          IF ( SMT_get( A%type ) /= 'DENSE' ) THEN
            efactors%A_col_ptr( 1 ) = 1
            DO j = 2, n + 1
              efactors%A_col_ptr( j )                                          &
               = efactors%A_col_ptr( j ) + efactors%A_col_ptr( j - 1 )
            END DO

!  Now record where the entries in each colum occur in the original matrix

            array_name = 'sbls: efactors%A_by_cols'
            CALL SPACE_resize_array( a_ne, efactors%A_by_cols,                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'SPARSE_BY_ROWS' )

              array_name = 'sbls: efactors%A_row'
              CALL SPACE_resize_array( a_ne, efactors%A_row,                   &
                 inform%status, inform%alloc_status, array_name = array_name,  &
                 deallocate_error_fatal = control%deallocate_error_fatal,      &
                 exact_size = control%space_critical,                          &
                 bad_alloc = inform%bad_alloc, out = control%error )
              IF ( inform%status /= GALAHAD_ok ) RETURN

              DO i = 1, m
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l )
                  new_pos = efactors%A_col_ptr( j )
                  efactors%A_row( new_pos ) = i
                  efactors%A_by_cols( new_pos ) = l
                  efactors%A_col_ptr( j ) = new_pos + 1
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO i = 1, m
                DO k = efactors%A_row_ptr( i ), efactors%A_row_ptr( i + 1 ) - 1
                  l = efactors%A_by_rows( k )
                  j = A%col( l )
                  new_pos = efactors%A_col_ptr( j )
                  efactors%A_by_cols( new_pos ) = l
                  efactors%A_col_ptr( j ) = new_pos + 1
                END DO
              END DO
            END SELECT

!  Finally readjust the column pointers

            DO j = n + 1, 2, - 1
              efactors%A_col_ptr( j ) = efactors%A_col_ptr( j - 1 )
            END DO
            efactors%A_col_ptr( 1 ) = 1
          END IF

!  Now build the sparsity structure of A diag(G)(inverse) A(trans)

          IF ( SMT_get( A%type ) == 'DENSE' ) THEN
            array_name = 'sbls: efactors%W'
            CALL SPACE_resize_array( n, efactors%W,                            &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = .FALSE.,                                           &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN
          ELSE
            array_name = 'sbls: efactors%IW'
            CALL SPACE_resize_array( m, efactors%IW,                           &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = .FALSE.,                                           &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN

            array_name = 'sbls: efactors%IW2'
            CALL SPACE_resize_array( m, efactors%IW2,                          &
               inform%status, inform%alloc_status,  array_name = array_name,   &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= GALAHAD_ok ) RETURN
          END IF

!  Compute the total storage for the (lower triangle) of
!    A diag(G)(inverse) A(transpose)

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            nnz_aat = m * ( m + 1 ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            nnz_aat = 0
            efactors%IW2 = 0

!  For the j-th column of A diag(G)(inverse) A(trans) ...

            DO j = 1, m
              nnz_col_j = 0
              DO k = A%ptr( j ), A%ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

                l = A%col( k )
                DO kk = efactors%A_col_ptr( l ), efactors%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                  i = efactors%A_row( kk )
                  IF ( efactors%IW2( i ) < j .AND. i >= j ) THEN
                    nnz_col_j = nnz_col_j + 1
                    efactors%IW2( i ) = j
                  END IF
                END DO
              END DO
              nnz_aat = nnz_aat + nnz_col_j
            END DO
          CASE ( 'COORDINATE' )
            nnz_aat = 0
            efactors%IW2 = 0

!  For the j-th column of A diag(G)(inverse) A(trans) ...

            DO j = 1, m
              nnz_col_j = 0
              DO k = efactors%A_row_ptr( j ), efactors%A_row_ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

                l = A%col( efactors%A_by_rows( k ) )
                DO kk = efactors%A_col_ptr( l ), efactors%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                  i = A%row( efactors%A_by_cols( kk ) )
                  IF ( efactors%IW2( i ) < j .AND. i >= j ) THEN
                    nnz_col_j = nnz_col_j + 1
                    efactors%IW2( i ) = j
                  END IF
                END DO
              END DO
              nnz_aat = nnz_aat + nnz_col_j
            END DO
          END SELECT

!  Allocate space to hold C + A diag(G)(inverse) A(transpose) in K

          efactors%K%n = m
          efactors%K%ne = nnz_aat + c_ne

          IF ( printi ) WRITE( out, "( A, ' Allocate arrays on length ', I0,   &
         &  ' to hold Schur complement' )" ) prefix, efactors%K%ne

          array_name = 'sbls: efactors%K%row'
          CALL SPACE_resize_array( efactors%K%ne, efactors%K%row,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'sbls: efactors%K%col'
          CALL SPACE_resize_array( efactors%K%ne, efactors%K%col,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'sbls: efactors%K%val'
          CALL SPACE_resize_array( efactors%K%ne, efactors%K%val,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

        END IF

!  Allocate space to store diag(G)

        IF ( new_h == 2 ) THEN
          array_name = 'sbls: efactors%G_diag'
          CALL SPACE_resize_array( n, efactors%G_diag,                         &
             inform%status, inform%alloc_status,  array_name = array_name,     &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN
        END IF

        use_schur_complement = .TRUE.
        IF ( printi ) WRITE( out, "( A, ' Schur complement used ' )" ) prefix

!  Now store diag(G)

        SELECT CASE( inform%preconditioner )

!  The identity matrix

        CASE( 1 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = I ' )" ) prefix
          efactors%G_diag( : n ) = one

!  The diagonal from the full matrix

        CASE( 2, 4 )
          IF ( printi ) THEN
            IF ( inform%preconditioner == 2 ) THEN
              WRITE( out, "( A, ' preconditioner: G = H ' )" ) prefix
            ELSE
              WRITE( out, "( A, ' preconditioner: G = diag(H) ' )" ) prefix
            END IF
          END IF
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            efactors%G_diag( : n ) = H%val( : n )
          CASE ( 'SCALED_IDENTITY' )
            efactors%G_diag( : n ) = H%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%G_diag( : n ) = one
          CASE ( 'DENSE' )
            WRITE( out, "( ' should not be here ... ' )" )
          CASE ( 'SPARSE_BY_ROWS' )
            efactors%G_diag( : n ) = zero
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            efactors%G_diag( : n ) = zero
            DO l = 1, H%ne
              i = H%row( l )
              efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
            END DO
          END SELECT

!  The (possibly modified) diagonal of the full matrix

        CASE( 3 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = diag(H) ' )" ) &
            prefix
          efactors%G_diag( : n ) = zero
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            efactors%G_diag( : n ) = H%val( : n )
          CASE ( 'SCALED_IDENTITY' )
            efactors%G_diag( : n ) = H%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%G_diag( : n ) = one
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              l = l + i
              efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( H%col( l ) == i )                                         &
                  efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              i = H%row( l )
              IF ( H%col( l ) == i )                                           &
                efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
            END DO
          END SELECT

          efactors%G_diag( : n ) =                                             &
            MAX( efactors%G_diag( : n ), control%min_diagonal )

!  The matrix D

        CASE( 5 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = D ' )" )       &
            prefix
          efactors%G_diag( : n ) = D

!  The scaled diagonal from the limited-memory formula

        CASE( 6 )
          IF ( printi )                                                        &
            WRITE( out, "( A, ' preconditioner: G = H_lm' )" ) prefix
!         efactors%G_diag( : n ) = zero
          IF ( H_lm%restricted == 0 ) THEN
            efactors%G_diag( : n ) = H_lm%delta
          ELSE
            efactors%G_diag( : n ) = control%min_diagonal
            DO i = 1, H_lm%n_restriction
              IF ( H_lm%RESTRICTION( i ) <= H_lm%n )                           &
                efactors%G_diag( i ) = H_lm%delta
            END DO
          END IF

!  The diagonal from the full matrix plus the scaled diagonal from the
!  limited-memory formula

        CASE( 7 )
          IF ( printi )                                                        &
            WRITE( out, "( A, ' preconditioner: G = H + H_lm' )" ) prefix
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'ZERO', 'NONE' )
            efactors%G_diag( : n ) = zero
          CASE ( 'DIAGONAL' )
            efactors%G_diag( : n ) = H%val( : n )
          CASE ( 'SCALED_IDENTITY' )
            efactors%G_diag( : n ) = H%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%G_diag( : n ) = one
          CASE ( 'DENSE' )
            WRITE( out, "( ' should not be here ... ' )" )
          CASE ( 'SPARSE_BY_ROWS' )
            efactors%G_diag( : n ) = zero
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            efactors%G_diag( : n ) = zero
            DO l = 1, H%ne
              i = H%row( l )
              efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
            END DO
          END SELECT
          IF ( H_lm%restricted == 0 ) THEN
            efactors%G_diag( : n ) = efactors%G_diag( : n ) + H_lm%delta
          ELSE
            DO i = 1, H_lm%n_restriction
              IF ( H_lm%RESTRICTION( i ) <= H_lm%n )                           &
                efactors%G_diag( i ) = efactors%G_diag( i ) + H_lm%delta
            END DO
          END IF

!  The diagonal D plus the scaled diagonal from the limited-memory formula

        CASE( 8 )
          IF ( printi )                                                        &
            WRITE( out, "( A, ' preconditioner: G = D + H_lm' )" ) prefix
          efactors%G_diag( : n ) = D( : n )
          IF ( H_lm%restricted == 0 ) THEN
            efactors%G_diag( : n ) = efactors%G_diag( : n ) + H_lm%delta
          ELSE
            DO i = 1, H_lm%n_restriction
              IF ( H_lm%RESTRICTION( i ) <= H_lm%n )                           &
                efactors%G_diag( i ) = efactors%G_diag( i ) + H_lm%delta
            END DO
          END IF
        END SELECT

!  count how many of the diagonal are positive

        inform%d_plus = COUNT( efactors%G_diag > zero )

!  Now insert the (row/col/val) entries of
!  A diag(G)(inverse) A(transpose) into K

        IF ( new_a == 2 ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            nnz_aat = 0
            l = 0
            DO i = 1, m
              efactors%W( : n ) = A%val( l + 1 : l + n ) / efactors%G_diag
              k = 0
              DO j = 1, i
                nnz_aat = nnz_aat + 1
                efactors%K%row( nnz_aat ) = i
                efactors%K%col( nnz_aat ) = j
                efactors%K%val( nnz_aat ) =                                    &
                  DOT_PRODUCT( efactors%W( : n ), A%val( k + 1 : k + n ) )
                k = k + n
              END DO
              l = l + n
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            nnz_aat_old = 0
            nnz_aat = 0
            efactors%IW( : n ) = efactors%A_col_ptr( : n )
            efactors%IW2 = 0

!  For the j-th column of A diag(G)(inverse) A(transpose) ...

            DO j = 1, m
              DO k = A%ptr( j ), A%ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

                l = A%col( k )
                al = A%val( k ) / efactors%G_diag( l )
                DO kk = efactors%IW( l ), efactors%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                  i = efactors%A_row( kk )

!  ... and which are in lower-triangular part

                  IF ( i >= j ) THEN

!  The first entry in this position ...

                    IF ( efactors%IW2( i ) < j .AND. i >= j ) THEN
                      nnz_aat = nnz_aat + 1
                      efactors%IW2( i ) = j + nnz_aat
                      efactors%K%row( nnz_aat ) = i
                      efactors%K%col( nnz_aat ) = j
                      efactors%K%val( nnz_aat ) =                              &
                        al * A%val( efactors%A_by_cols( kk ) )

!  ... or a subsequent one

                    ELSE
                      ii = efactors%IW2( i ) - j
                      efactors%K%val( ii ) = efactors%K%val( ii ) +            &
                        al * A%val( efactors%A_by_cols( kk ) )
                    END IF

!  IW is incremented since all entries above lie in the upper triangle

                  ELSE
                    efactors%IW( l ) = efactors%IW( l ) + 1
                  END IF
                END DO
              END DO
              DO l = nnz_aat_old + 1, nnz_aat
                efactors%IW2( efactors%K%row( l ) ) = j
              END DO
              nnz_aat_old  = nnz_aat
            END DO
          CASE ( 'COORDINATE' )
            nnz_aat_old = 0
            nnz_aat = 0
            efactors%IW( : n ) = efactors%A_col_ptr( : n )
            efactors%IW2 = 0

!  For the j-th column of A diag(G)(inverse) A(transpose) ...

            DO j = 1, m
              DO k = efactors%A_row_ptr( j ), efactors%A_row_ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

                l = A%col( efactors%A_by_rows( k ) )
                al = A%val( efactors%A_by_rows( k ) ) / efactors%G_diag( l )
                DO kk = efactors%IW( l ), efactors%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                  i = A%row( efactors%A_by_cols( kk ) )

!  ... and which are in lower-triangular part

                  IF ( i >= j ) THEN

!  The first entry in this position ...

                    IF ( efactors%IW2( i ) < j .AND. i >= j ) THEN
                      nnz_aat = nnz_aat + 1
                      efactors%IW2( i ) = j + nnz_aat
                      efactors%K%row( nnz_aat ) = i
                      efactors%K%col( nnz_aat ) = j
                      efactors%K%val( nnz_aat ) =                              &
                        al * A%val( efactors%A_by_cols( kk ) )

!  ... or a subsequent one

                    ELSE
                      ii = efactors%IW2( i ) - j
                      efactors%K%val( ii ) = efactors%K%val( ii ) +            &
                        al * A%val( efactors%A_by_cols( kk ) )
                    END IF

!  IW is incremented since all entries above lie in the upper triangle

                  ELSE
                    efactors%IW( l ) = efactors%IW( l ) + 1
                  END IF
                END DO
              END DO
              DO l = nnz_aat_old + 1, nnz_aat
                efactors%IW2( efactors%K%row( l ) ) = j
              END DO
              nnz_aat_old  = nnz_aat
            END DO
          END SELECT

!  Now insert the (val) entries of A diag(G)(inverse) A(transpose) into K

        ELSE
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            nnz_aat = 0
            l = 0
            DO i = 1, m
              efactors%W( : n ) = A%val( l + 1 : l + n ) / efactors%G_diag
              k = 0
              DO j = 1, i
                nnz_aat = nnz_aat + 1
                efactors%K%val( nnz_aat ) =                                    &
                  DOT_PRODUCT( efactors%W( : n ), A%val( k + 1 : k + n ) )
                k = k + n
              END DO
              l = l + n
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            nnz_aat_old = 0
            nnz_aat = 0
            efactors%IW( : n ) = efactors%A_col_ptr( : n )
            efactors%IW2 = 0

!  For the j-th column of A diag(G)(inverse) A(transpose) ...

            DO j = 1, m
              DO k = A%ptr( j ), A%ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

                l = A%col( k )
                al = A%val( k ) / efactors%G_diag( l )
                DO kk = efactors%IW( l ), efactors%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                  i = efactors%A_row( kk )

!  ... and which are in lower-triangular part

                  IF ( i >= j ) THEN

!  The first entry in this position ...

                    IF ( efactors%IW2( i ) < j .AND. i >= j ) THEN
                      nnz_aat = nnz_aat + 1
                      efactors%IW2( i ) = j + nnz_aat
                      efactors%K%val( nnz_aat ) =                              &
                        al * A%val( efactors%A_by_cols( kk ) )

!  ... or a subsequent one

                    ELSE
                      ii = efactors%IW2( i ) - j
                      efactors%K%val( ii ) = efactors%K%val( ii ) +            &
                        al * A%val( efactors%A_by_cols( kk ) )
                    END IF

!  IW is incremented since all entries above lie in the upper triangle

                  ELSE
                    efactors%IW( l ) = efactors%IW( l ) + 1
                  END IF
                END DO
              END DO
              DO l = nnz_aat_old + 1, nnz_aat
                efactors%IW2( efactors%K%row( l ) ) = j
              END DO
              nnz_aat_old  = nnz_aat
            END DO
          CASE ( 'COORDINATE' )
            nnz_aat_old = 0
            nnz_aat = 0
            efactors%IW( : n ) = efactors%A_col_ptr( : n )
            efactors%IW2 = 0

!  For the j-th column of A diag(G)(inverse) A(transpose) ...

            DO j = 1, m
              DO k = efactors%A_row_ptr( j ), efactors%A_row_ptr( j + 1 ) - 1

!  ... include the contributions from column l of A

                l = A%col( efactors%A_by_rows( k ) )
                al = A%val( efactors%A_by_rows( k ) ) / efactors%G_diag( l )
                DO kk = efactors%IW( l ), efactors%A_col_ptr( l + 1 ) - 1

!  ... which have nonzeros in rows i

                  i = A%row( efactors%A_by_cols( kk ) )

!  ... and which are in lower-triangular part

                  IF ( i >= j ) THEN

!  The first entry in this position ...

                    IF ( efactors%IW2( i ) < j .AND. i >= j ) THEN
                      nnz_aat = nnz_aat + 1
                      efactors%IW2( i ) = j + nnz_aat
                      efactors%K%val( nnz_aat ) =                              &
                        al * A%val( efactors%A_by_cols( kk ) )

!  ... or a subsequent one

                    ELSE
                      ii = efactors%IW2( i ) - j
                      efactors%K%val( ii ) = efactors%K%val( ii ) +            &
                        al * A%val( efactors%A_by_cols( kk ) )
                    END IF

!  IW is incremented since all entries above lie in the upper triangle

                  ELSE
                    efactors%IW( l ) = efactors%IW( l ) + 1
                  END IF
                END DO
              END DO
              DO l = nnz_aat_old + 1, nnz_aat
                efactors%IW2( efactors%K%row( l ) ) = j
              END DO
              nnz_aat_old  = nnz_aat
            END DO
          END SELECT
        END IF

!  Now insert the (val) entries of C into K ...

        IF ( new_a /= 2 .AND. new_c /= 2 ) THEN
          IF ( new_c > 0 .AND. c_ne > 0 ) THEN
            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'SCALED_IDENTITY' )
              efactors%K%val( nnz_aat + 1 : ) = C%val( 1 )
            CASE ( 'IDENTITY' )
              efactors%K%val( nnz_aat + 1 : ) = one
            CASE DEFAULT
              efactors%K%val( nnz_aat + 1 : ) = C%val
            END SELECT
          END IF

!   ... or the (row/col/val) entries of C into K

        ELSE
          IF ( c_ne > 0 ) THEN
            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
              DO i = 1, m
                efactors%K%row( nnz_aat + i ) = i
                efactors%K%col( nnz_aat + i ) = i
              END DO
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                DO j = 1, i
                  l = l + 1
                  efactors%K%row( nnz_aat + l ) = i
                  efactors%K%col( nnz_aat + l ) = j
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                DO l = C%ptr( i ), C%ptr( i + 1 ) - 1
                  efactors%K%row( nnz_aat + l ) = i
                  efactors%K%col( nnz_aat + l ) = C%col( l )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, c_ne
                efactors%K%row( nnz_aat + l ) = C%row( l )
                efactors%K%col( nnz_aat + l ) = C%col( l )
              END DO
            END SELECT

            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'SCALED_IDENTITY' )
              efactors%K%val( nnz_aat + 1 : ) = C%val( 1 )
            CASE ( 'IDENTITY' )
              efactors%K%val( nnz_aat + 1 : ) = one
            CASE DEFAULT
              DO l = 1, c_ne
                efactors%K%val( nnz_aat + l ) = C%val( l )
              END DO
            END SELECT
          END IF
        END IF

!       WRITE( out, "( ' K: m, nnz ', I0, 1X, I0 )" )                          &
!         efactors%K%n, efactors%K%ne
!       WRITE( out, "( A, /, ( 10I7) )" ) ' rows =', ( efactors%K%row )
!       WRITE( out, "( A, /, ( 10I7) )" ) ' cols =', ( efactors%K%col )
!       WRITE( out, "( A, /, ( 10F7.2) )" ) ' vals =', ( efactors%K%val )

        GO TO 200
      END IF

!   ======================
!    USE AUGMENTED SYSTEM
!   ======================

 100  CONTINUE
      use_schur_complement = .FALSE.
      IF ( printi ) WRITE( out, "( A, ' augmented matrix used ' )" ) prefix

!  Find the space required to store G and K

      IF ( new_a > 0 .OR. new_h > 0 ) THEN

        SELECT CASE( inform%preconditioner )

!  The identity matrix

        CASE( 1 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = I ' )" ) prefix
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  The full matrix

        CASE( 2 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = H ' )" ) prefix
          g_ne = h_ne
          efactors%K%n = n + m
          efactors%K%ne = h_ne + a_ne + c_ne

!  The (possibly modified) diagonal of the full matrix

        CASE( 3 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = diag(H) ' )" ) &
            prefix
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  A band (or semi-bandwith control%semibadwith) from the full matrix

        CASE( 4 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner G = band(H) ' )" )  &
            prefix

          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
            g_ne = n
          CASE ( 'DENSE' )
            g_ne = 0
            DO i = 1, n
              DO j = 1, i
                IF ( ABS( i - j ) <= control%semi_bandwidth ) g_ne = g_ne + 1
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            g_ne = 0
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                j = H%col( l )
                IF ( ABS( i - j ) <= control%semi_bandwidth ) g_ne = g_ne + 1
              END DO
            END DO
          CASE ( 'COORDINATE' )
            g_ne = COUNT( ABS( H%row( : H%ne ) - H%col( : H%ne ) )             &
                     <= control%semi_bandwidth )
          END SELECT
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  The user-supplied D

        CASE( 5 )
          IF ( PRESENT( D ) ) THEN
            IF ( printi ) WRITE( out, "( A, ' preconditioner: G = D' )" ) prefix
          ELSE
            inform%preconditioner = 1
            IF ( printi ) WRITE( out, "( A, ' preconditioner: G = I' )" ) prefix
          END IF
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  The full matrix + H_lm

        CASE( 6 )
          IF ( printi )                                                        &
            WRITE( out, "( A, ' preconditioner: G = H_lm' )" ) prefix
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  The full matrix + H_lm

        CASE( 7 )
          IF ( printi )                                                        &
            WRITE( out, "( A, ' preconditioner: G = H + H_lm' )" ) prefix
          g_ne = h_ne + n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  The full matrix + H_lm

        CASE( 8 )
          IF ( printi )                                                        &
            WRITE( out, "( A, ' preconditioner: G = D + H_lm' )" ) prefix
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  Non-basic submatrices of H

        CASE( 11 : 12 )
          IF ( new_a > 0 ) THEN

!  Find sets of basic rows and columns

            CALL SBLS_find_basis( m, n, a_ne, A, efactors%B,                   &
                                  efactors%B_data, efactors%B_control,         &
                                  efactors%B_ROWS, efactors%B_COLS,            &
                                  efactors%rank_a,                             &
                                  control%find_basis_by_transpose,             &
                                  prefix, 'sbls: efactors%', out, printi,      &
                                  control, inform )

!  Make a copy of the "basis" matrix

            array_name = 'sbls: efactors%B_COLS_basic'
            CALL SPACE_resize_array( n, efactors%B_COLS_basic,                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) GO TO 900

            efactors%B_COLS_basic = 0
            DO i = 1, efactors%rank_a
              efactors%B_COLS_basic( efactors%B_COLS( i ) ) = i
            END DO

!  Mark the non-basic columns

            nb = 0
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) <= 0 ) THEN
                nb = nb + 1
                efactors%B_COLS_basic( i ) = - nb
              END IF
            END DO
          END IF

          g_ne = 0
          IF ( inform%preconditioner == 11 ) THEN
            IF ( printi )                                                      &
              WRITE( out, "( A, ' preconditioner: G = H_22 ' )" ) prefix
            SELECT CASE ( SMT_get( H%type ) )
            CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
              DO i = 1, n
                IF ( efactors%B_COLS_basic( i ) < 0 ) g_ne = g_ne + 1
              END DO
            CASE ( 'DENSE' )
              DO i = 1, n
                DO j = 1, i
                  IF ( efactors%B_COLS_basic( i ) < 0 .AND.                    &
                       efactors%B_COLS_basic( j ) < 0 ) g_ne = g_ne + 1
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, n
                DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                  IF ( efactors%B_COLS_basic( i ) < 0 .AND.                    &
                       efactors%B_COLS_basic( H%col( l ) ) < 0 ) g_ne = g_ne + 1
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, H%ne
                IF ( efactors%B_COLS_basic( H%row( l ) ) < 0 .AND.             &
                     efactors%B_COLS_basic( H%col( l ) ) < 0 ) g_ne = g_ne + 1
              END DO
            END SELECT
          ELSE IF ( inform%preconditioner == 12 ) THEN
            IF ( printi )                                                      &
              WRITE( out, "( A, ' preconditioner: G = H_22 & H_21' )" ) prefix
            SELECT CASE ( SMT_get( H%type ) )
            CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
              DO i = 1, n
                IF ( efactors%B_COLS_basic( i ) < 0 ) g_ne = g_ne + 1
              END DO
            CASE ( 'DENSE' )
              DO i = 1, n
                DO j = 1, i
                  IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.            &
                     efactors%B_COLS_basic( j ) > 0 ) ) g_ne = g_ne + 1
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, n
                DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                  IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.            &
                     efactors%B_COLS_basic( h%col( l ) ) > 0 ) ) g_ne = g_ne + 1
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, H%ne
                IF ( .NOT. ( efactors%B_COLS_basic( H%row( l ) ) > 0 .AND.     &
                     efactors%B_COLS_basic( h%col( l ) ) > 0 ) ) g_ne = g_ne + 1
              END DO
            END SELECT
          END IF

          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

!  Anything else

        CASE DEFAULT
          IF ( printi ) WRITE( out,                                            &
            "( A, ' no option control%preconditioner = ', I8, ' at present')") &
               prefix, inform%preconditioner
          inform%status = GALAHAD_error_unknown_precond ; RETURN

        END SELECT
        efactors%g_ne = g_ne
      ELSE
        g_ne = efactors%g_ne
        IF ( c_ne == 2 ) efactors%K%ne = g_ne + a_ne + c_ne
      END IF

      IF ( control%perturb_to_make_definite ) THEN
        k_ne = efactors%K%ne + n
      ELSE
        k_ne = efactors%K%ne
      END IF

!  Check to see if we need to reallocate the space to hold K

      resize = .FALSE.
      IF ( ALLOCATED( efactors%K%row ) ) THEN
         IF ( SIZE( efactors%K%row ) < k_ne .OR. ( control%space_critical      &
              .AND. SIZE( efactors%K%row ) /= k_ne ) ) resize = .TRUE.
      ELSE
        resize = .TRUE.
      END IF
      IF ( ALLOCATED( efactors%K%col ) ) THEN
         IF ( SIZE( efactors%K%col ) < k_ne .OR. ( control%space_critical      &
              .AND. SIZE( efactors%K%col ) /= k_ne ) ) resize = .TRUE.
      ELSE
        resize = .TRUE.
      END IF
      IF ( ALLOCATED( efactors%K%val ) ) THEN
         IF ( SIZE( efactors%K%val ) < k_ne .OR. ( control%space_critical      &
              .AND. SIZE( efactors%K%val ) /= k_ne ) ) resize = .TRUE.
      ELSE
        resize = .TRUE.
      END IF

!  Allocate sufficient space to hold K

      IF ( resize ) THEN
        array_name = 'sbls: efactors%K%row'
        CALL SPACE_resize_array( k_ne, efactors%K%row,                         &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%K%col'
        CALL SPACE_resize_array( k_ne, efactors%K%col,                         &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%K%val'
        CALL SPACE_resize_array( k_ne, efactors%K%val,                         &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

!  Now insert A, G and C into K

      k_c = a_ne + g_ne
      efactors%k_g = a_ne
      efactors%k_c = k_c
      efactors%k_pert = efactors%k_c + c_ne

!  Start by storing A

      IF ( resize .OR. new_a > 1 ) THEN
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              efactors%K%row( l ) = i + n
              efactors%K%col( l ) = j
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              efactors%K%row( l ) = i + n
              efactors%K%col( l ) = A%col( l )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          efactors%K%row( : a_ne ) = A%row( : a_ne ) + n
          efactors%K%col( : a_ne ) = A%col( : a_ne )
        END SELECT
      END IF
      IF ( resize .OR. new_a > 0 ) efactors%K%val( : a_ne ) = A%val( : a_ne )

!  Now store G

      SELECT CASE( inform%preconditioner )

!  The identity matrix

      CASE( 1 )
        IF ( resize .OR. new_a > 1 ) THEN
          DO i = 1, g_ne
            efactors%K%row( a_ne + i ) = i
            efactors%K%col( a_ne + i ) = i
            efactors%K%val( a_ne + i ) = one
          END DO
        END IF

!  The full matrix

      CASE( 2 )
        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
            DO i = 1, n
              efactors%K%row( a_ne + i ) = i
              efactors%K%col( a_ne + i ) = i
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                efactors%K%row( a_ne + l ) = i
                efactors%K%col( a_ne + l ) = j
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                efactors%K%row( a_ne + l ) = i
                efactors%K%col( a_ne + l ) = H%col( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            efactors%K%row( a_ne + 1 : a_ne + h_ne ) = H%row( : h_ne )
            efactors%K%col( a_ne + 1 : a_ne + h_ne ) = H%col( : h_ne )
          END SELECT
        END IF
        IF ( resize .OR. new_a > 1 .OR. new_h > 0 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'SCALED_IDENTITY' )
            efactors%K%val( a_ne + 1 : a_ne + h_ne ) = H%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%K%val( a_ne + 1 : a_ne + h_ne ) = one
          CASE DEFAULT
            efactors%K%val( a_ne + 1 : a_ne + h_ne ) = H%val( : h_ne )
          END SELECT
        END IF

!  The (possibly modified) diagonal of the full matrix

      CASE( 3 )
        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          DO i = 1, g_ne
            efactors%K%row( a_ne + i ) = i
            efactors%K%col( a_ne + i ) = i
          END DO
        END IF

        IF ( resize .OR. new_a > 1 .OR. new_h > 0 ) THEN
          efactors%K%val( a_ne + 1 : a_ne + g_ne ) = zero
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            efactors%K%val( a_ne + 1 : a_ne + n ) = H%val( : n )
          CASE ( 'SCALED_IDENTITY' )
            efactors%K%val( a_ne + 1 : a_ne + n ) = H%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%K%val( a_ne + 1 : a_ne + n ) = one
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( j == i ) efactors%K%val( a_ne + i ) =                     &
                  efactors%K%val( a_ne + i ) + H%val( l )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( H%col( l ) == i ) efactors%K%val( a_ne + i ) =            &
                  efactors%K%val( a_ne + i ) + H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              i = H%row( l )
              IF ( H%col( l ) == i ) efactors%K%val( a_ne + i ) =              &
                efactors%K%val( a_ne + i ) + H%val( l )
            END DO
          END SELECT
          efactors%K%val( a_ne + 1 : a_ne + g_ne ) =                           &
            MAX( efactors%K%val( a_ne + 1 : a_ne + g_ne ), control%min_diagonal)
        END IF

!  A band (or semi-bandwith control%semibadwith) from the full matrix

      CASE( 4 )
        g_ne = a_ne
        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              g_ne = g_ne + 1
              efactors%K%row( g_ne ) = i
              efactors%K%col( g_ne ) = i
              efactors%K%val( g_ne ) = H%val( i )
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              g_ne = g_ne + 1
              efactors%K%row( g_ne ) = i
              efactors%K%col( g_ne ) = i
              efactors%K%val( g_ne ) = H%val( 1 )
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              g_ne = g_ne + 1
              efactors%K%row( g_ne ) = i
              efactors%K%col( g_ne ) = i
              efactors%K%val( g_ne ) = one
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( ABS( i - j ) <= control%semi_bandwidth ) THEN
                  g_ne = g_ne + 1
                  efactors%K%row( g_ne ) = i
                  efactors%K%col( g_ne ) = j
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                j = H%col( l )
                IF ( ABS( i - j ) <= control%semi_bandwidth ) THEN
                  g_ne = g_ne + 1
                  efactors%K%row( g_ne ) = i
                  efactors%K%col( g_ne ) = j
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              i = H%row( l ) ; j = H%col( l )
              IF ( ABS( i - j ) <= control%semi_bandwidth ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = j
                efactors%K%val( g_ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        ELSE IF ( new_h > 0 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              g_ne = g_ne + 1
              efactors%K%val( g_ne ) = H%val( i )
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              g_ne = g_ne + 1
              efactors%K%val( g_ne ) = H%val( 1 )
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              g_ne = g_ne + 1
              efactors%K%val( g_ne ) = one
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( ABS( i - j ) <= control%semi_bandwidth ) THEN
                  g_ne = g_ne + 1
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( ABS( i - H%col( l ) ) <= control%semi_bandwidth ) THEN
                  g_ne = g_ne + 1
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              IF ( ABS( H%col( l ) - H%row( l ) )                              &
                  <= control%semi_bandwidth ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        END IF

!  The user-supplied D

      CASE( 5 )
        IF ( resize .OR. new_a > 1 ) THEN
          DO i = 1, g_ne
            efactors%K%row( a_ne + i ) = i
            efactors%K%col( a_ne + i ) = i
            efactors%K%val( a_ne + i ) = D( i )
          END DO
        END IF

!  The limited-memory matrix H_lm

      CASE( 6 )
        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          DO i = 1, n
            efactors%K%row( a_ne + i ) = i
            efactors%K%col( a_ne + i ) = i
          END DO
        END IF
        IF ( resize .OR. new_a > 1 .OR. new_h > 0 ) THEN
          IF ( H_lm%restricted == 0 ) THEN
            efactors%K%val( a_ne + 1 : a_ne + n ) = H_lm%delta
          ELSE
            efactors%K%val( a_ne + 1 : a_ne + n ) = zero
            DO i = 1, H_lm%n_restriction
              IF ( H_lm%RESTRICTION( i ) <= H_lm%n )                           &
                efactors%K%val( a_ne + i ) = H_lm%delta
            END DO
          END IF
        END IF

!  The full matrix + H_lm

      CASE( 7 )
        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
            DO i = 1, n
              efactors%K%row( a_ne + i ) = i
              efactors%K%col( a_ne + i ) = i
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                efactors%K%row( a_ne + l ) = i
                efactors%K%col( a_ne + l ) = j
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                efactors%K%row( a_ne + l ) = i
                efactors%K%col( a_ne + l ) = H%col( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            efactors%K%row( a_ne + 1 : a_ne + h_ne ) = H%row( : h_ne )
            efactors%K%col( a_ne + 1 : a_ne + h_ne ) = H%col( : h_ne )
          END SELECT
          DO i = 1, n
            efactors%K%row( a_ne + h_ne + i ) = i
            efactors%K%col( a_ne + h_ne + i ) = i
          END DO
        END IF
        IF ( resize .OR. new_a > 1 .OR. new_h > 0 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'SCALED_IDENTITY' )
            efactors%K%val( a_ne + 1 : a_ne + h_ne ) = H%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%K%val( a_ne + 1 : a_ne + h_ne ) = one
          CASE DEFAULT
            efactors%K%val( a_ne + 1 : a_ne + h_ne ) = H%val( : h_ne )
          END SELECT
          IF ( H_lm%restricted == 0 ) THEN
            efactors%K%val( a_ne + h_ne + 1 : a_ne + h_ne + n ) = H_lm%delta
          ELSE
            efactors%K%val( a_ne + h_ne + 1 : a_ne + h_ne + n ) = zero
            DO i = 1, H_lm%n_restriction
              IF ( H_lm%RESTRICTION( i ) <= H_lm%n )                           &
                efactors%K%val( a_ne + h_ne + i ) = H_lm%delta
            END DO
          END IF
        END IF

!  The limited-memory matrix H_lm

      CASE( 8 )
        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          DO i = 1, n
            efactors%K%row( a_ne + i ) = i
            efactors%K%col( a_ne + i ) = i
          END DO
        END IF
        IF ( resize .OR. new_a > 1 .OR. new_h > 0 ) THEN
          IF ( H_lm%restricted == 0 ) THEN
            efactors%K%val( a_ne + 1 : a_ne + n ) = D( : n ) + H_lm%delta
          ELSE
            efactors%K%val( a_ne + 1 : a_ne + n ) = D( : n )
            DO i = 1, H_lm%n_restriction
              IF ( H_lm%RESTRICTION( i ) <= H_lm%n ) efactors%K%val( a_ne + i )&
                     = efactors%K%val( a_ne + i ) + H_lm%delta
            END DO
          END IF
        END IF

!  Non-basic submatrices of H

      CASE( 11 )
        g_ne = a_ne
        IF ( resize.OR. new_a > 1  .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = i
                efactors%K%val( g_ne ) = H%val( i )
              END IF
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = i
                efactors%K%val( g_ne ) = H%val( 1 )
              END IF
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = i
                efactors%K%val( g_ne ) = one
              END IF
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( efactors%B_COLS_basic( i ) < 0 .AND.                      &
                     efactors%B_COLS_basic( j ) < 0 ) THEN
                  g_ne = g_ne + 1
                  efactors%K%row( g_ne ) = i
                  efactors%K%col( g_ne ) = j
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                j = H%col( l )
                IF ( efactors%B_COLS_basic( i ) < 0 .AND.                      &
                     efactors%B_COLS_basic( j ) < 0 ) THEN
                  g_ne = g_ne + 1
                  efactors%K%row( g_ne ) = i
                  efactors%K%col( g_ne ) = j
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              i = H%row( l ) ; j = H%col( l )
              IF ( efactors%B_COLS_basic( i ) < 0 .AND.                        &
                   efactors%B_COLS_basic( j ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = j
                efactors%K%val( g_ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        ELSE IF ( new_h > 0 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( i )
              END IF
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( 1 )
              END IF
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = one
              END IF
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( efactors%B_COLS_basic( i ) < 0 .AND.                      &
                     efactors%B_COLS_basic( j ) < 0 ) THEN
                  g_ne = g_ne + 1
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( efactors%B_COLS_basic( i ) < 0 .AND.                      &
                     efactors%B_COLS_basic( H%col( l ) ) < 0 ) THEN
                  g_ne = g_ne + 1
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              IF ( efactors%B_COLS_basic( H%row( l ) ) < 0 .AND.               &
                   efactors%B_COLS_basic( H%col( l ) ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        END IF

      CASE( 12 )

        g_ne = a_ne
        IF ( resize.OR. new_a > 1  .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = i
                efactors%K%val( g_ne ) = H%val( i )
              END IF
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = i
                efactors%K%val( g_ne ) = H%val( 1 )
              END IF
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = i
                efactors%K%val( g_ne ) = one
              END IF
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.              &
                             efactors%B_COLS_basic( j ) > 0 ) ) THEN
                  g_ne = g_ne + 1
                  efactors%K%row( g_ne ) = i
                  efactors%K%col( g_ne ) = j
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                j = H%col( l )
                IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.              &
                             efactors%B_COLS_basic( j ) > 0 ) ) THEN
                  g_ne = g_ne + 1
                  efactors%K%row( g_ne ) = i
                  efactors%K%col( g_ne ) = j
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              i = H%row( l ) ; j = H%col( l )
              IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.                &
                           efactors%B_COLS_basic( j ) > 0 ) ) THEN
                g_ne = g_ne + 1
                efactors%K%row( g_ne ) = i
                efactors%K%col( g_ne ) = j
                efactors%K%val( g_ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        ELSE IF ( new_h > 0 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( i )
              END IF
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( 1 )
              END IF
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              IF ( efactors%B_COLS_basic( i ) < 0 ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = one
              END IF
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.              &
                             efactors%B_COLS_basic( j ) > 0 ) ) THEN
                  g_ne = g_ne + 1
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.              &
                             efactors%B_COLS_basic( H%col( l ) ) > 0 ) ) THEN
                  g_ne = g_ne + 1
                  efactors%K%val( g_ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              IF ( .NOT. ( efactors%B_COLS_basic( H%row( l ) ) > 0 .AND.       &
                           efactors%B_COLS_basic( H%col( l ) ) > 0 ) ) THEN
                g_ne = g_ne + 1
                efactors%K%val( g_ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        END IF

      END SELECT

!  Finally store -C in K

      IF ( c_ne > 0 ) THEN
        IF ( new_a == 2 .OR. new_a == 2 .OR. new_c == 2 ) THEN
          SELECT CASE ( SMT_get( C%type ) )
          CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
            DO i = 1, m
              efactors%K%row( k_c + i ) = n + i
              efactors%K%col( k_c + i ) = n + i
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, m
              DO j = 1, i
                l = l + 1
                efactors%K%row( k_c + l ) = n + i
                efactors%K%col( k_c + l ) = n + j
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = C%ptr( i ), C%ptr( i + 1 ) - 1
                efactors%K%row( k_c + l ) = n + i
                efactors%K%col( k_c + l ) = n + C%col( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            efactors%K%row( k_c + 1 : efactors%K%ne ) = n + C%row( : c_ne )
            efactors%K%col( k_c + 1 : efactors%K%ne ) = n + C%col( : c_ne )
          END SELECT

          SELECT CASE ( SMT_get( C%type ) )
          CASE ( 'SCALED_IDENTITY' )
            efactors%K%val( k_c + 1 : efactors%K%ne ) = - C%val( 1 )
          CASE ( 'IDENTITY' )
            efactors%K%val( k_c + 1 : efactors%K%ne ) = - one
          CASE DEFAULT
            efactors%K%val( k_c + 1 : efactors%K%ne ) = - C%val( : c_ne )
          END SELECT
        ELSE
          IF ( new_c > 0 ) THEN
            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'SCALED_IDENTITY' )
              efactors%K%val( k_c + 1 : efactors%K%ne ) = - C%val( 1 )
            CASE ( 'IDENTITY' )
              efactors%K%val( k_c + 1 : efactors%K%ne ) = - one
            CASE DEFAULT
              efactors%K%val( k_c + 1 : efactors%K%ne ) = - C%val( : c_ne )
            END SELECT
          END IF
        END IF
      END IF

!  record time to form preconditioner

 200  CONTINUE
      CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )
      inform%time%form =inform%time%form + time - time_start
      inform%time%clock_form = inform%time%clock_form + clock - clock_start

!  ===========
!   FACTORIZE
!  ===========

      IF ( printi ) THEN
        IF ( inform%factorization == 1 ) THEN
          WRITE( out,                                                          &
           "( /, A, ' Using SLS(', A, ') to factorize the Schur complement' )")&
            prefix, TRIM( control%definite_linear_solver )
        ELSE
          WRITE( out,                                                          &
           "( /, A, ' Using SLS(', A, ') to factorize the augmented matrix' )")&
            prefix, TRIM( control%symmetric_linear_solver )
        END IF
      END IF

!  A diagonal perturbation is to be added to the matrix to be factored
!  to make the resultant diaginally dominant

      IF ( inform%perturbed .AND. .NOT. use_schur_complement ) THEN
        efactors%K%val( efactors%K%ne + 1 : k_ne ) = zero
        DO l = a_ne + 1, k_c
          i = efactors%K%row( l )
          IF ( i > n ) CYCLE
          j = efactors%K%col( l )
          IF ( j > n ) CYCLE
          val = efactors%K%val( l )
          IF ( i == j ) THEN
            efactors%K%val( efactors%K%ne + i ) =                              &
              efactors%K%val( efactors%K%ne + i ) + val
          ELSE
            efactors%K%val( efactors%K%ne + i ) =                              &
              efactors%K%val( efactors%K%ne + i ) - ABS( val )
            efactors%K%val( efactors%K%ne + j ) =                              &
              efactors%K%val( efactors%K%ne + j ) - ABS( val )
          END IF
        END DO
        DO i = 1, n
          j = efactors%K%ne + i
          efactors%K%row( j ) = i
          efactors%K%col( j ) = i
          IF ( efactors%K%val( j ) > zero ) THEN
            efactors%K%val( j ) = zero
          ELSE
            efactors%K%val( j ) = control%min_diagonal - efactors%K%val( j )
          END IF
        END DO
        efactors%K%ne = k_ne
      END IF

!     WRITE( 77, "( ' n, nnz ', I7, I10 )" ) efactors%K%n, efactors%K%ne
!!    WRITE( 77, * ) ( efactors%K%row( i ),  efactors%K%col( i ),              &
!!      efactors%K%val( i ),  i = 1, efactors%K%ne )
!     WRITE( 77, "( ' n, nnz ', I7, I10 )" ) efactors%K%n, efactors%K%ne
!     WRITE( 77, "( A, /, ( 10I7) )" ) ' rows =',                              &
!        ( efactors%K%row( : efactors%K%ne ) )
!     WRITE( 77, "( A, /, ( 10I7) )" ) ' cols =',                              &
!        ( efactors%K%col( : efactors%K%ne ) )
!     WRITE( 77, "( A, /, ( 10F7.2) )" ) ' vals =',                            &
!        ( efactors%K%val( : efactors%K%ne ) )

      IF ( control%print_level >= 11 .AND. out >= 0 ) THEN
        WRITE( out, "( ' n, nnz ', I7, I10 )" ) efactors%K%n, efactors%K%ne
        WRITE( out, "( ' K: row, col, val ', /, 2( 2I6, ES24.16 ) )" )         &
          ( efactors%K%row( i ), efactors%K%col( i ), efactors%K%val( i ),     &
            i = 1, efactors%K%ne )
      END IF

      IF ( new_a > 1 .OR. new_h > 1 .OR. efactors%analyse ) THEN

!  Initialize the factorization data

        efactors%K_control = control%SLS_control
!IS64   efactors%K_control%max_in_core_store = HUGE( 0_long ) / real_bytes

!  Analyse the preconditioner

        IF ( efactors%K%n > 0 ) THEN
          CALL SMT_put( efactors%K%type, 'COORDINATE', i )
          IF ( use_schur_complement ) THEN
            IF ( numerical_pivoting ) THEN
              efactors%K_control%pivot_control = 1
            ELSE IF ( control%allow_singular ) THEN
              IF ( control%perturb_to_make_definite ) THEN
                efactors%K_control%pivot_control = 4
              ELSE
                efactors%K_control%pivot_control = 2
              END IF
            ELSE
              efactors%K_control%pivot_control = 3
            END IF
!           efactors%K_control%pivot_control = 1
!write(6,*) efactors%K_control%pivot_control
!           CALL SLS_initialize_solver( control%symmetric_linear_solver,       &
            CALL SLS_initialize_solver( control%definite_linear_solver,        &
                                        efactors%K_data, inform%SLS_inform )
          ELSE
            CALL SLS_initialize_solver( control%symmetric_linear_solver,       &
                                        efactors%K_data, inform%SLS_inform )
          END IF
          IF ( inform%SLS_inform%status == GALAHAD_error_unknown_solver ) THEN
            inform%status = GALAHAD_error_unknown_solver
            GO TO 900
          END IF
!         efactors%K_control = control%SLS_control
          CALL SLS_analyse( efactors%K, efactors%K_data,                       &
                            efactors%K_control, inform%SLS_inform )
          inform%sls_analyse_status = inform%SLS_inform%status
          IF ( printi ) WRITE( out, "(  A, ' SLS: analysis complete: status',  &
         &  ' = ', I0, ', ordering = ', I0 )" ) prefix,                        &
             inform%sls_analyse_status, efactors%K_control%ordering
          IF ( inform%sls_analyse_status < 0 ) THEN
            IF ( inform%sls_analyse_status ==                                  &
                 GALAHAD_error_unknown_solver ) THEN
              inform%status = GALAHAD_error_unknown_solver
            ELSE
              inform%status = GALAHAD_error_analysis
            END IF
            GO TO 900
          END IF
          IF ( printi ) WRITE( out, "( A, ' K n = ', I0,                       &
         &  ', nnz(prec,predicted factors) = ', I0, ', ', I0 )" )              &
               prefix, efactors%K%n, efactors%K%ne,                            &
               inform%SLS_inform%entries_in_factors
          IF ( inform%SLS_inform%entries_in_factors < 0 ) THEN
            inform%status = GALAHAD_error_analysis
            GO TO 900
          END IF
        ELSE
          IF ( printi ) WRITE( out,                                            &
             "(  A, ' no analysis need for matrix of order 0 ')" ) prefix
          inform%sls_analyse_status = 0
        END IF
        efactors%analyse = .FALSE.
      END IF

!  Factorize the preconditioner

      IF ( efactors%K%n > 0 ) THEN
        CALL SLS_factorize( efactors%K, efactors%K_data,                       &
                            efactors%K_control, inform%SLS_inform )
        inform%sls_factorize_status = inform%SLS_inform%status
        inform%factorization_integer = inform%SLS_inform%integer_size_factors
        inform%factorization_real = inform%SLS_inform%real_size_factors
        IF ( printi ) WRITE( out,                                              &
          "( A, ' SLS: factorization complete: status = ', I0,                 &
         &   ', pivoting = ', I0 )" ) prefix, inform%sls_factorize_status,     &
            efactors%K_control%pivot_control
!CALL SLS_enquire( efactors%K_data, inform%SLS_inform, D = DD )
!DO i = 1, efactors%K%n
!write(6,"( I6, 2ES12.4 )" ) i, DD( 1, i ), DD( 2, i )
!END DO
        IF ( inform%sls_factorize_status < 0 ) THEN
          SELECT CASE( inform%sls_factorize_status )
          CASE ( GALAHAD_error_unknown_solver  )
            inform%status = GALAHAD_error_unknown_solver
          CASE (  GALAHAD_error_inertia )
            inform%status = GALAHAD_error_inertia
          CASE DEFAULT
            inform%status = GALAHAD_error_factorization
          END SELECT
          GO TO 900
        END IF
        efactors%rank_k = inform%SLS_inform%rank
        IF ( inform%SLS_inform%rank < efactors%K%n ) THEN
          inform%rank_def = .TRUE.
          IF ( inform%factorization == 2 ) THEN
            inform%rank = inform%SLS_inform%rank - n
          ELSE
            inform%rank = inform%SLS_inform%rank
          END IF
        END IF
        IF ( printi ) WRITE( out,                                              &
             "( A, ' K nnz(prec,factors) = ', I0, ', ', I0 )" )                &
          prefix, efactors%K%ne, inform%SLS_inform%entries_in_factors

        kzero = efactors%K%n - inform%SLS_inform%rank
        kminus = inform%SLS_inform%negative_eigenvalues
        IF ( use_schur_complement ) THEN
!write(6,*) ' kminus = ', kminus
!         IF ( kzero + kminus > 0 ) THEN
          IF ( kminus > 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, 1X, I0, ' -ve and ' , I0,              &
             &  ' zero eigevalues > 0 required ones ' )" )                     &
               prefix, kminus, kzero
            IF ( control%perturb_to_make_definite .AND.                        &
                 .NOT. inform%perturbed ) THEN
              IF ( fail_if_not_sc ) THEN
                inform%status = GALAHAD_error_inertia
                GO TO 900
              ELSE
                IF ( control%out > 0 .AND. control%print_level > 0 )           &
                  WRITE( control%out,                                          &
                    "( A, ' Perturbing G to try to correct this ' )" ) prefix
                inform%factorization = 2
                inform%perturbed = .TRUE.
                efactors%analyse = .TRUE.
                GO TO 100
              END IF
            ELSE
!             IF ( kminus > 0 ) THEN
                inform%status = GALAHAD_error_inertia
!             ELSE
!               inform%status = GALAHAD_error_preconditioner
!             END IF
              GO TO 900
            END IF
          END IF
        ELSE
!         IF ( kzero + kminus > m ) THEN
!         write(6,*) ' k_0, k_- ', kzero,  kminus
          IF ( kminus > m ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, 1X, I0, ' -ve and ' , I0,              &
             &  ' zero eigevalues > ', I0, ' required ones ' )" )              &
               prefix, kminus, kzero, m
            IF ( control%perturb_to_make_definite .AND.                        &
                 .NOT. inform%perturbed ) THEN
              IF ( control%out > 0 .AND. control%print_level > 0 )             &
                WRITE( control%out,                                            &
                  "( A, ' Perturbing G to try to correct this ' )" ) prefix
              inform%perturbed = .TRUE.
              efactors%analyse = .TRUE.
              GO TO 200
            ELSE
              IF ( kminus > m ) THEN
                inform%status = GALAHAD_error_inertia
              ELSE
                inform%status = GALAHAD_error_preconditioner
              END IF
              GO TO 900
            END IF
          END IF
        END IF
      ELSE
        IF ( printi ) WRITE( out,                                              &
           "(  A, ' no factorization need for matrix of order 0 ')" ) prefix
        inform%sls_factorize_status = 0
      END IF

!  if a limited-memory Hessian is used, build the relevant Schur complement

!  S = [ -D       L^T    ] - [     Y^T R^T    0 ] K^-1 [ R Y : delta R S ]
!      [ L   delta S^T S ]   [ delta S^T R^T  0 ]      [  0 :      0     ]

      efactors%lm = inform%preconditioner == 6 .OR.                            &
                    inform%preconditioner == 7 .OR.                            &
                    inform%preconditioner == 8

      IF ( efactors%lm ) THEN

!  make space for the Schur complement S

        efactors%len_s_max = 2 * H_lm%m
        efactors%len_s = 2 * H_lm%length
        array_name = 'sbls: efactors%S'
        CALL SPACE_resize_array( efactors%len_s_max, efactors%len_s_max,       &
           efactors%S,                                                         &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  set up space for the Bunch-Kaufman factorization of S

        np1 = H_lm%n_restriction + 1 ; npm = n + m
        nb = ILAENV( 1, 'DSYTRF', 'L', efactors%len_s_max, - 1, - 1, - 1 )

        array_name = 'sbls: efactors%IPIV'
        CALL SPACE_resize_array( efactors%len_s_max, efactors%IPIV,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .FALSE.,                                               &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        IF ( H_lm%restricted == 0 ) THEN
          lw = MAX( efactors%len_s_max * nb, npm )
        ELSE
          lw = MAX( efactors%len_s_max * nb, H_lm%n + m )
          lrw = npm
        END IF

        array_name = 'sbls: efactors%W'
        CALL SPACE_resize_array( lw, efactors%W,                               &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .FALSE.,                                               &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        IF ( H_lm%restricted > 0 ) THEN
          array_name = 'sbls: efactors%RW'
          CALL SPACE_resize_array( lrw, efactors%RW,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = .FALSE.,                                             &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN
        END IF

!  form the Schur complement. Initialize as [ -D       L^T    ]
!                                           [ L   delta S^T S ]

        efactors%S( : efactors%len_s, : efactors%len_s ) = zero

        DO j = 1, H_lm%length
          oj = H_lm%ORDER( j ) ; jp = H_lm%length + j
          val = H_lm%L_scaled( j, j )
          efactors%S( j, j ) = - val * val
          DO i = j, H_lm%length
            oi = H_lm%ORDER( i ) ; ip = H_lm%length + i
            IF ( i > j ) THEN
              efactors%S( ip, j ) = H_lm%L_scaled( i, j ) * val
              efactors%S( j, ip ) = efactors%S( ip, j )
            END IF
            efactors%S( ip, jp ) = H_lm%delta * H_lm%STS( oi, oj )
            efactors%S( jp, ip ) = efactors%S( ip, jp )
          END DO
        END DO

!  for each column y_j of Y,

        DO j = 1, H_lm%length
          oj = H_lm%ORDER( j )
          efactors%W( : H_lm%n ) = H_lm%Y( : H_lm%n, oj )

!  unrestricted case:  solve K ( u_j ) = ( y_j )
!                              ( v_j )   (  0  )

          IF ( H_lm%restricted == 0 ) THEN
            efactors%W( np1 : npm ) = zero
            CALL SBLS_solve_explicit( n, m, A, C, efactors, control, inform,   &
                                      efactors%W( : npm ) )

!  restricted case: form R y_j ...

          ELSE
            efactors%RW( : n ) = zero
            DO i = 1, H_lm%n_restriction
              ii = H_lm%RESTRICTION( i )
              IF ( ii <= H_lm%n ) efactors%RW( i ) = efactors%W( ii )
            END DO
!           efactors%RW( : H_lm%n_restriction )                                &
!             = efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )
            efactors%RW( np1 : npm ) = zero

!  ... solve K ( u_j ) = ( R y_j ) ...
!              ( v_j )   (   0   )

            CALL SBLS_solve_explicit( n, m, A, C, efactors, control, inform,   &
                                      efactors%RW( : npm ) )

!  ... and form R^T u_j

            efactors%W( : H_lm%n ) = zero
            DO i = 1, H_lm%n_restriction
              ii = H_lm%RESTRICTION( i )
              IF ( ii <= H_lm%n ) efactors%W( ii ) = efactors%RW( i )
            END DO
!           efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )             &
!             = efactors%RW( : H_lm%n_restriction )
          END IF

!  subtract [     Y^T R^T u_j   ] from the ith column of S
!           [ delta S^T R^T u_j ]

          DO i = j, H_lm%length
            oi = H_lm%ORDER( i )
            efactors%S( i, j ) = efactors%S( i, j )                            &
              - DOT_PRODUCT( H_lm%Y( : H_lm%n, oi ),  efactors%W( : H_lm%n ) )
            efactors%S( j, i ) = efactors%S( i, j )
          END DO

          val = H_lm%delta
          DO i = 1, H_lm%length
            oi = H_lm%ORDER( i ) ; ip = H_lm%length + i
            efactors%S( ip, j ) = efactors%S( ip, j ) - val                    &
              * DOT_PRODUCT( H_lm%S( : H_lm%n, oi ),  efactors%W( : H_lm%n ) )
            efactors%S( j, ip ) = efactors%S( ip, j )
          END DO
        END DO

!  for each column s_j of S,

        DO j = 1, H_lm%length
          oj = H_lm%ORDER( j ) ; jp = H_lm%length + j
          efactors%W( : H_lm%n ) = H_lm%S( : H_lm%n, oj )

!  unrestricted case:  solve K ( u_j ) = ( s_j )
!                              ( v_j )   (  0  )

          IF ( H_lm%restricted == 0 ) THEN
            efactors%W( np1 : npm ) = zero
            CALL SBLS_solve_explicit( n, m, A, C, efactors, control, inform,   &
                                      efactors%W( : npm ) )

!  restricted case: form R s_j ...

          ELSE
            efactors%RW( : n ) = zero
            DO i = 1, H_lm%n_restriction
              ii = H_lm%RESTRICTION( i )
              IF ( ii <= H_lm%n ) efactors%RW( i ) = efactors%W( ii )
            END DO
!           efactors%RW( : H_lm%n_restriction )                                &
!             = efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )
            efactors%RW( np1 : npm ) = zero

!  ... solve K ( u_j ) = ( R s_j ) ...
!              ( v_j )   (   0   )

            CALL SBLS_solve_explicit( n, m, A, C, efactors, control, inform,   &
                                      efactors%RW( : npm ) )

!  ... and form R^T u_j

            efactors%W( : H_lm%n ) = zero
            DO i = 1, H_lm%n_restriction
              ii = H_lm%RESTRICTION( i )
              IF ( ii <= H_lm%n ) efactors%W( ii ) = efactors%RW( i )
            END DO
!           efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )             &
!             = efactors%RW( : H_lm%n_restriction )
          END IF

!  subtract [  delta Y^T R^T u_l+j  ] from the ith column of S
!           [ delta^2 S^T R^T u_l+j ]

          val = H_lm%delta * H_lm%delta
          DO i = j, H_lm%length
            oi = H_lm%ORDER( i ) ; ip = H_lm%length + i
            efactors%S( ip, jp ) = efactors%S( ip, jp ) - val                  &
              * DOT_PRODUCT( H_lm%S( : H_lm%n, oi ),  efactors%W( : H_lm%n ) )
            efactors%S( jp, ip ) = efactors%S( ip, jp )
          END DO
        END DO

!  compute the Bunch-Kaufman factors of S

        CALL SYTRF( 'L', efactors%len_s, efactors%S, efactors%len_s_max,       &
                    efactors%IPIV, efactors%W, lw, i )
        IF ( i /= 0 ) THEN
          IF ( printi )                                                        &
            WRITE( out, "( A, ' Bunch-Kaufman error ', I0 )" ) prefix, i
          inform%status = GALAHAD_error_factorization
          RETURN
        END IF
      END IF
      inform%status = GALAHAD_ok

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      IF ( printi ) WRITE( out,                                                &
         "( A, ' time to form and factorize explicit preconditioner ', F6.2 )")&
        prefix, time_now - time_start
      inform%time%factorize =inform%time%factorize + time_now - time
      inform%time%clock_factorize =                                            &
        inform%time%clock_factorize + clock_now - clock
      RETURN

!  End of subroutine SBLS_form_n_factorize_explicit

      END SUBROUTINE SBLS_form_n_factorize_explicit

!-*-*-  S B L S _ FORM _ AD _ FACTORIZE _ IMPLICIT   S U B R O U T I N E   -*-*-

!     SUBROUTINE SBLS_form_n_factorize_implicit( n, m, H, A, C, ifactors,      &
      SUBROUTINE SBLS_form_n_factorize_implicit( n, m, H, A, ifactors,         &
                                                 last_factorization,           &
                                                 control, inform )

!  Form an implicit factorization of
!
!        K = ( G   A^T )
!            ( A    -C )
!
!  for various approximations G of H

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, last_factorization
!     TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SMT_type ), INTENT( IN ) :: H, A
      TYPE ( SBLS_implicit_factors_type ), INTENT( INOUT ) :: ifactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, jj, l, out, a_ne
      INTEGER :: new_h, new_a
!     INTEGER :: new_c
      REAL :: time_start, time, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now, clock
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      out = control%out
      printi = control%print_level >= 1

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne
      END IF

      IF ( inform%preconditioner >= 0 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Use SBLS_form_n_factorize_explicit subroutine instead' )" )  &
            prefix
        inform%status = GALAHAD_error_call_order ; RETURN
      END IF
      inform%status = GALAHAD_ok

      IF ( last_factorization /= inform%factorization ) THEN
        new_h = 2
        new_a = 2
!       new_c = 2
      ELSE
        new_h = control%new_h
        new_a = control%new_a
!       new_c = control%new_c
      END IF

!  Find a permutation of A = IR ( A1  A2 ) IC^T for which the "basis"
!                               (  0   0 )
!  matrix A1 is nonsingular

!  This induces a re-ordering IC^T P IC of P

! ----------
! Find basis
! ----------

!     CALL CPU_TIME( t1 )

!  Find a "basic" set of rows and colums, that is a non-singular submatrix A1 of
!  maximal rank. Also set up the complement matrix A2 of columns of A not in A1

      IF ( new_a > 0 ) THEN
        ifactors%m = m ; ifactors%n = n
        CALL SBLS_find_A1_and_A2( m, n, a_ne, A, ifactors%A1, ifactors%A1_data,&
                                  ifactors%A1_control, ifactors%A2,            &
                                  ifactors%A_ROWS_basic,                       &
                                  ifactors%A_COLS_basic,                       &
                                  ifactors%A_ROWS_order, ifactors%A_COLS_order,&
                                  ifactors%rank_a, ifactors%k_n, ifactors%n_r, &
                                  prefix, 'sbls: ifactors%', out, printi,      &
                                  control, inform,                             &
                                  ifactors%RHS_orig, ifactors%SOL_current )
      END IF

!  Form the preconditioner

      ifactors%unitb31 = .TRUE.
      ifactors%unitp22 = .TRUE.

      SELECT CASE( inform%preconditioner )

      CASE( - 1 )

        IF ( printi ) WRITE( out, "( /, A, ' Preconditioner G_22 = I' )" )     &
          prefix

!  G_11 = 0, G_21 = 0, G_22 = I

        ifactors%unitb22 = .TRUE.
        ifactors%zerob32 = .TRUE.
        ifactors%zerob33 = .TRUE.
        ifactors%zerop11 = .TRUE.
        ifactors%zerop21 = .TRUE.

      CASE( - 2 )

        IF ( printi ) WRITE( out, "( /, A, ' Preconditioner G_22 = H_22' )" )  &
          prefix

!  G_11 = 0, G_21 = 0, G_22 = H_22

        ifactors%unitb22 = .FALSE.!      3  diagonal, G = diag( H )
!      4  G_11 = 0, G_21 = 0, !      3  diagonal, G = diag( H )
!      4  G_11 = 0, G_21 = 0, !      3  diagonal, G = diag( H )
!      4  G_11 = 0, G_21 = 0,
!      5  G_11 = 0, G_21 = H_21, G_22 = H_22
!
!      5  G_11 = 0, G_21 = H_21, G_22 = H_22
!
!      5  G_11 = 0, G_21 = H_21, G_22 = H_22
!
        ifactors%zerob32 = .TRUE.
        ifactors%zerob33 = .TRUE.
        ifactors%zerop11 = .TRUE.
        ifactors%zerop21 = .TRUE.
!      3  diagonal, G = diag( H )
!      4  G_11 = 0, G_21 = 0,
!      5  G_11 = 0, G_21 = H_21, G_22 = H_22
!
!  Store H_22 in B22; see how much space is required

        IF ( new_h > 1 ) THEN
          ifactors%B22%n = ifactors%n_r
          ifactors%B22%ne = 0
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
            DO i = 1, n
              IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a )              &
                ifactors%B22%ne = ifactors%B22%ne + 1
            END DO
          CASE ( 'DENSE' )
            DO i = 1, n
              DO j = 1, i
                IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a .AND.        &
                     ifactors%A_COLS_order( j ) > ifactors%rank_a )            &
                  ifactors%B22%ne = ifactors%B22%ne + 1
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a .AND.        &
                     ifactors%A_COLS_order( H%col( l ) ) > ifactors%rank_a )   &
                  ifactors%B22%ne = ifactors%B22%ne + 1
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              IF ( ifactors%A_COLS_order( H%row( l ) ) > ifactors%rank_a .AND. &
                   ifactors%A_COLS_order( H%col( l ) ) > ifactors%rank_a )     &
                ifactors%B22%ne = ifactors%B22%ne + 1
            END DO
          END SELECT

!  Allocate sufficient space ...

          array_name = 'sbls: ifactors%B22%row'
          CALL SPACE_resize_array( ifactors%B22%ne, ifactors%B22%row,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'sbls: ifactors%B22%col'
          CALL SPACE_resize_array( ifactors%B22%ne, ifactors%B22%col,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          array_name = 'sbls: ifactors%B22_VAL'
          CALL SPACE_resize_array( ifactors%B22%ne, ifactors%B22%val,          &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN
        END IF

!  ... and store H_22

        ifactors%B22%ne = 0
        IF ( new_a > 0 .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              ii = ifactors%A_COLS_order( i ) - ifactors%rank_a
              IF ( ii > 0 ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%row( ifactors%B22%ne ) = ii
                ifactors%B22%col( ifactors%B22%ne ) = ii
                ifactors%B22%val( ifactors%B22%ne ) = H%val( i )
              END IF
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              ii = ifactors%A_COLS_order( i ) - ifactors%rank_a
              IF ( ii > 0 ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%row( ifactors%B22%ne ) = ii
                ifactors%B22%col( ifactors%B22%ne ) = ii
                ifactors%B22%val( ifactors%B22%ne ) = H%val( 1 )
              END IF
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              ii = ifactors%A_COLS_order( i ) - ifactors%rank_a
              IF ( ii > 0 ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%row( ifactors%B22%ne ) = ii
                ifactors%B22%col( ifactors%B22%ne ) = ii
                ifactors%B22%val( ifactors%B22%ne ) = one
              END IF
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                ii = ifactors%A_COLS_order( i ) - ifactors%rank_a
                jj = ifactors%A_COLS_order( j ) - ifactors%rank_a
                IF ( ii > 0.AND. jj > 0 ) THEN
                  ifactors%B22%ne = ifactors%B22%ne + 1
                  ifactors%B22%row( ifactors%B22%ne ) = ii
                  ifactors%B22%col( ifactors%B22%ne ) = jj
                  ifactors%B22%val( ifactors%B22%ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                ii = ifactors%A_COLS_order( i ) - ifactors%rank_a
                jj = ifactors%A_COLS_order( H%col( l ) ) - ifactors%rank_a
                IF ( ii > 0 .AND. jj > 0 ) THEN
                  ifactors%B22%ne = ifactors%B22%ne + 1
                  ifactors%B22%row( ifactors%B22%ne ) = ii
                  ifactors%B22%col( ifactors%B22%ne ) = jj
                  ifactors%B22%val( ifactors%B22%ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              ii = ifactors%A_COLS_order( H%row( l ) ) - ifactors%rank_a
              jj = ifactors%A_COLS_order( H%col( l ) ) - ifactors%rank_a
              IF ( ii > 0 .AND. jj > 0 ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%row( ifactors%B22%ne ) = ii
                ifactors%B22%col( ifactors%B22%ne ) = jj
                ifactors%B22%val( ifactors%B22%ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        ELSE
          ifactors%B22%row = ifactors%B22%row - ifactors%rank_a
          ifactors%B22%col = ifactors%B22%col - ifactors%rank_a
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' )
            DO i = 1, n
              IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%val( ifactors%B22%ne ) = H%val( i )
              END IF
            END DO
          CASE ( 'SCALED_IDENTITY' )
            DO i = 1, n
              IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%val( ifactors%B22%ne ) = H%val( 1 )
              END IF
            END DO
          CASE ( 'IDENTITY' )
            DO i = 1, n
              IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%val( ifactors%B22%ne ) = one
              END IF
            END DO
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, n
              DO j = 1, i
                l = l + 1
                IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a .AND.        &
                     ifactors%A_COLS_order( j ) > ifactors%rank_a ) THEN
                  ifactors%B22%ne = ifactors%B22%ne + 1
                  ifactors%B22%val( ifactors%B22%ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                IF ( ifactors%A_COLS_order( i ) > ifactors%rank_a .AND.        &
                     ifactors%A_COLS_order( H%col( l ) ) > ifactors%rank_a) THEN
                  ifactors%B22%ne = ifactors%B22%ne + 1
                  ifactors%B22%val( ifactors%B22%ne ) = H%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, H%ne
              IF ( ifactors%A_COLS_order( H%row( l ) ) > ifactors%rank_a .AND. &
                   ifactors%A_COLS_order( H%col( l ) ) > ifactors%rank_a ) THEN
                ifactors%B22%ne = ifactors%B22%ne + 1
                ifactors%B22%val( ifactors%B22%ne ) = H%val( l )
              END IF
            END DO
          END SELECT
        END IF

!  record time to form preconditioner

        CALL CPU_TIME( time ) ; CALL CLOCK_time( clock )
        inform%time%form =inform%time%form + time - time_start
        inform%time%clock_form = inform%time%clock_form + clock - clock_start

!  Now factorize H_22

        IF ( printi ) WRITE( out, "( A, ' Using SLS' )" ) prefix
        ifactors%B22_control = control%SLS_control
!IS64   ifactors%B22_control%max_in_core_store = HUGE( 0_long ) / real_bytes
        IF ( control%perturb_to_make_definite )                                &
          ifactors%B22_control%pivot_control = 4
        CALL SMT_put( ifactors%B22%type, 'COORDINATE', i )
        CALL SLS_initialize_solver( control%symmetric_linear_solver,           &
                                    ifactors%B22_data, inform%SLS_inform )
        IF ( inform%SLS_inform%status == GALAHAD_error_unknown_solver ) THEN
          inform%status = GALAHAD_error_unknown_solver
          GO TO 900
        END IF
        ifactors%B22_control = control%SLS_control
        CALL SLS_analyse( ifactors%B22, ifactors%B22_data,                     &
                          ifactors%B22_control, inform%SLS_inform )
        inform%sls_analyse_status = inform%SLS_inform%status
        IF ( printi ) WRITE( out,                                              &
       "( A, ' Analysis complete:      status = ', I0 )" ) prefix,             &
          inform%sls_analyse_status
        IF ( inform%sls_analyse_status < 0 ) THEN
           inform%status = GALAHAD_error_analysis
           GO TO 900
        END IF
        IF ( printi ) WRITE( out, "( A, ' B22 n = ', I0,                       &
       &  ', nnz(prec,predicted factors) = ', I0, ', ', I0 )" )                &
             prefix, ifactors%B22%n, ifactors%B22%ne,                          &
             inform%SLS_inform%real_size_factors

        CALL SLS_factorize( ifactors%B22, ifactors%B22_data,                   &
                            ifactors%B22_control, inform%SLS_inform )
        inform%sls_factorize_status = inform%SLS_inform%status
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Factorization complete: status = ', I0 )" )                  &
            prefix, inform%sls_factorize_status
        IF ( inform%sls_factorize_status < 0 ) THEN
          SELECT CASE( inform%sls_factorize_status )
          CASE ( GALAHAD_error_unknown_solver  )
            inform%status = GALAHAD_error_unknown_solver
          CASE DEFAULT
            inform%status = GALAHAD_error_factorization
          END SELECT
          GO TO 900
        END IF
        IF ( inform%SLS_inform%rank < ifactors%B22%n ) THEN
          inform%rank_def = .TRUE.
          inform%rank = inform%SLS_inform%rank
        END IF
        IF ( printi ) WRITE( out,                                              &
             "( A, ' B22 nnz(prec,factors) = ', I0, ', ', I0 )" )              &
          prefix, ifactors%B22%ne, inform%SLS_inform%entries_in_factors

!  Check to ensure that the preconditioner is definite

        IF ( inform%SLS_inform%rank < ifactors%B22%n .OR.                      &
             inform%SLS_inform%negative_eigenvalues > 0 ) THEN
          inform%perturbed = .TRUE.
          array_name = 'sbls: ifactors%PERT'
          CALL SPACE_resize_array( ifactors%B22%n, ifactors%PERT,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 900

          CALL SLS_ENQUIRE( ifactors%B22_data, inform%SLS_inform,              &
                            PERTURBATION = ifactors%PERT )
          IF ( printi ) WRITE( out, "( A, ' H_22 perturbed by', ES11.4 )" )    &
            prefix, MAXVAL( ABS( ifactors%PERT ) )
        ELSE
          inform%perturbed = .FALSE.
        END IF

        IF ( inform%SLS_inform%negative_eigenvalues + ifactors%B22%n -         &
             inform%SLS_inform%rank > 0 ) THEN
          WRITE( out, "( A, ' SLS_factorize reports B22 is indefinite ' )" )   &
            prefix
           inform%status = GALAHAD_error_preconditioner ; GO TO 900
        END IF
        IF ( printi ) WRITE( out,                                              &
             "( A, ' B22 nnz(prec,factors) = ', I0, ', ', I0 )" )              &
          prefix, ifactors%B22%ne, inform%SLS_inform%real_size_factors

!  Restore the row and colum indices to make matrix-vector products efficient

        ifactors%B22%row = ifactors%B22%row + ifactors%rank_a
        ifactors%B22%col = ifactors%B22%col + ifactors%rank_a

!  record total time

  900   CONTINUE
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time
        inform%time%clock_factorize =                                          &
          inform%time%clock_factorize + clock_now - clock

      CASE( - 3 )

!  G_11 = 0, G_21 = H_21, G_22 = H_22

        IF ( printi ) WRITE( out,                                              &
          "( /, A, ' Preconditioner G_22 = H_22 and G_21 = H_21 ' )" ) prefix

        ifactors%unitb22 = .FALSE.
        ifactors%zerob32 = .TRUE.
        ifactors%zerob33 = .TRUE.
        ifactors%zerop11 = .TRUE.
        ifactors%zerop21 = .FALSE.

      CASE DEFAULT

!  Anything else

        IF ( printi ) WRITE( out,                                              &
          "( A, ' No option control%preconditioner = ', I8, ' at present' )" ) &
             prefix, inform%preconditioner
        inform%status = GALAHAD_error_unknown_precond ; RETURN

      END SELECT

      CALL CPU_TIME( time_now )
      IF ( printi ) WRITE( out,                                                &
        "( A, ' time to form and factorize implicit preconditioner ', F6.2 )" )&
        prefix, time_now - time_start

      RETURN

!  End of subroutine SBLS_form_n_factorize_implicit

      END SUBROUTINE SBLS_form_n_factorize_implicit

!-*-*-*-*-*-*-*-*-   S B L S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE SBLS_solve( n, m, A, C, data, control, inform, SOL, H_lm )

!  Solve

!    ( G   A^T ) ( x ) = ( a )
!    ( A   -C  ) ( y )   ( b )

!  with the right-hand side ( a ) input and the solution ( x ) output in SOL
!                           ( b )                        ( y )

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A, C
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL
      TYPE ( LMS_data_type ), OPTIONAL, INTENT( INOUT ) :: H_lm

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      IF ( control%print_level >= 10 .AND. control%out > 0 )                   &
        WRITE( control%out, "( A, ' rhs = ', /, ( 5ES16.8 ) )" )               &
          prefix, SOL( : n + m )

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Solve the preconditioned system

      IF ( inform%factorization == 3 ) THEN
        CALL SBLS_solve_null_space( data%nfactors, control, inform, SOL )
      ELSE IF ( data%efactors%lm ) THEN
        CALL SBLS_solve_explicit_lm( n, m, A, C, data%efactors, control,       &
                                     inform, SOL, H_lm )
      ELSE IF ( inform%preconditioner >= 0 ) THEN
        CALL SBLS_solve_explicit( n, m, A, C, data%efactors, control, inform,  &
                                  SOL )
      ELSE
        CALL SBLS_solve_implicit( data%ifactors, control, inform, SOL )
      END IF

!  record total time

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%apply = inform%time%apply + time_now - time_start
      inform%time%clock_apply =                                                &
        inform%time%clock_apply + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      inform%alternative = .FALSE.

      IF ( control%print_level >= 10 .AND. control%out > 0 )                   &
        WRITE( control%out, "( A, ' sol = ', /, ( 5ES16.8 ) )" )               &
          prefix, SOL( : n + m )


      RETURN

!  End of subroutine SBLS_solve

      END SUBROUTINE SBLS_solve

!-*-*-*-*-*-*-   S B L S _ P A R T _ S O L V E   S U B R O U T I N E   -*-*-*-*-

      SUBROUTINE SBLS_part_solve( part, n, m, A, data, control, inform, SOL )

!  Solve

!    ( G   A^T ) ( x ) = ( a )
!    ( A   -C  ) ( y )   ( b )

!  with the right-hand side ( a ) input and the solution ( x ) output in SOL
!                           ( b )                        ( y )

!  Dummy arguments

      CHARACTER( LEN = 1 ) :: part
      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL

!  Local variables

      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_start, clock_now

!  start timimg

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  Solve the preconditioned system

      IF ( inform%factorization == 3 ) THEN
! ** to do
!       CALL SBLS_solve_null_space( data%nfactors, control, inform, SOL )
      ELSE IF ( inform%preconditioner >= 0 ) THEN
        CALL SBLS_part_solve_explicit( part, n, m, A, data%efactors, control,  &
                                       inform, SOL )
      ELSE
! ** to do
!       CALL SBLS_solve_implicit( data%ifactors, control, inform, SOL )
      END IF

!  record total time

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%apply = inform%time%apply + time_now - time_start
      inform%time%clock_apply =                                                &
        inform%time%clock_apply + clock_now - clock_start
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total =                                                &
        inform%time%clock_total + clock_now - clock_start
      RETURN

!  End of subroutine SBLS_part_solve

      END SUBROUTINE SBLS_part_solve

!-*-*-*-   S B L S _ S O L V E _ E X P L I C I T   S U B R O U T I N E   -*-*-*-

      SUBROUTINE SBLS_solve_explicit( n, m, A, C, efactors, control, inform,  &
                                      SOL )

!  Solve

!    ( G   A^T ) ( x ) = ( a )
!    ( A    -C ) ( y )   ( b )

!  using an explicit factorization of K or C + A G(inv) A(transpose)

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A, C
      TYPE ( SBLS_explicit_factors_type ), INTENT( INOUT ) :: efactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL

!  Local variables

      INTEGER :: iter, i, ii, j, l, np1, npm
      REAL ( KIND = wp ) :: val
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Allocate workspace arrays

      npm = n + m
      IF ( inform%factorization == 1 ) np1 = n + 1
      IF ( npm /= efactors%len_sol_workspace ) THEN
        array_name = 'sbls: efactors%RHS'
        CALL SPACE_resize_array( npm, efactors%RHS,                            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%RHS_orig'
        CALL SPACE_resize_array( npm, efactors%RHS_orig,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        efactors%len_sol_workspace = npm
      END IF

!  Compute the original residual

      efactors%RHS_orig( : npm ) = SOL( : npm )
      efactors%RHS( : npm ) = SOL( : npm )
      SOL( : npm ) = zero

!  Solve the system with iterative refinement

      IF ( control%print_level >= 11 .AND. control%out >= 0 )                  &
        WRITE( control%out, "( ' RHS:', /, ( 3ES24.16 ) )" )                   &
          ( efactors%RHS( i ), i = 1, npm )

!     IF ( control%print_level >= 4 .AND. control%out > 0 ) THEN
!       WRITE( control%out, "( A, ' residuals = ', /, ( 5ES16.8 ) )" )         &
!         prefix, refactors%RHS( : npm )
      IF ( control%print_level >= 4 .AND. control%out > 0 ) THEN
        WRITE( control%out, "( A, ' maximum residual = ', ES10.4 )" )          &
          prefix, MAXVAL( ABS( efactors%RHS( : npm ) ) )
      END IF

      DO iter = 0, control%itref_max

!  Use factors of the Schur complement

        IF ( inform%factorization == 1 ) THEN

!  Form a <- diag(G)(inverse) a

          efactors%RHS( : n ) = efactors%RHS( : n ) / efactors%G_diag( : n )

          IF ( m > 0 ) THEN

!  Form b <- A a - b

            efactors%RHS( np1 : npm ) = - efactors%RHS( np1 : npm )
            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                ii = n + i
                efactors%RHS( ii ) = efactors%RHS( ii ) +                      &
                  DOT_PRODUCT( A%val( l + 1 : l + n ), efactors%RHS( : n ) )
                l = l + n
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                ii = n + i
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  efactors%RHS( ii ) = efactors%RHS( ii ) +                    &
                    A%val( l ) * efactors%RHS( A%col( l ) )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                ii = n + A%row( l )
                efactors%RHS( ii ) = efactors%RHS( ii ) +                      &
                  A%val( l ) * efactors%RHS( A%col( l ) )
              END DO
            END SELECT

!  Solve  ( C + A G(inv) A(transpose) ) y = A diag(G)(inverse) a - b
!  and place the result in a

            CALL SLS_solve( efactors%K, efactors%RHS( np1 : npm ),             &
                            efactors%K_data, efactors%K_control,               &
                            inform%SLS_inform )
            inform%sls_solve_status = inform%SLS_inform%status
            IF ( inform%sls_solve_status < 0 ) THEN
              IF ( control%out > 0 .AND. control%print_level > 0 )             &
                WRITE( control%out, "( A, ' solve exit status = ', I0 )" )     &
                  prefix, inform%sls_solve_status
              inform%status = GALAHAD_error_solve
              RETURN
            END IF

!  Form a <- diag(G)(inverse) ( a - A(trans) y )

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                DO j = 1, n
                  l = l + 1
                  efactors%RHS( j ) = efactors%RHS( j ) - A%val( l ) *         &
                    efactors%RHS( n + i ) / efactors%G_diag( j )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l )
                  efactors%RHS( j ) = efactors%RHS( j ) - A%val( l ) *         &
                    efactors%RHS( n + i ) / efactors%G_diag( j )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                j = A%col( l )
                efactors%RHS( j ) = efactors%RHS( j ) - A%val( l ) *           &
                  efactors%RHS( n + A%row( l ) ) / efactors%G_diag( j )
              END DO
            END SELECT
          END IF

!  Use factors of the augmented system

        ELSE
          CALL SLS_solve( efactors%K, efactors%RHS, efactors%K_data,           &
                           efactors%K_control, inform%SLS_inform )
          inform%sls_solve_status = inform%SLS_inform%status
          IF ( inform%sls_solve_status < 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' solve exit status = ', I0 )" )       &
                prefix, inform%sls_solve_status
            inform%status = GALAHAD_error_solve
            RETURN
          END IF
        END IF

!  Update the estimate of the solution

!write(6,"( ( I6, 2ES22.14 ) )" ) ( i, SOL( i ), efactors%RHS( i ), i = 1, npm )
        SOL( : npm ) = SOL( : npm ) + efactors%RHS( : npm )

        IF ( control%print_level >= 4 .AND. control%out > 0 )                  &
        WRITE( control%out, "( A, ' norm of solution and correction =',        &
       &  2ES11.4 )" ) prefix, MAXVAL( ABS( SOL( : npm ) ) ),                  &
            MAXVAL( ABS( efactors%RHS( : npm ) ) )

!  Form the residuals

        IF ( iter < control%itref_max .OR. control%get_norm_residual ) THEN

!  ... for the case where G is diagonal ...

          IF ( inform%factorization == 1 ) THEN
            efactors%RHS( : n ) =                                              &
              efactors%RHS_orig( : n ) - efactors%G_diag( : n ) * SOL( : n )

            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'ZERO', 'NONE' )
              efactors%RHS( np1 : npm ) = efactors%RHS_orig( np1 : npm )
            CASE ( 'DIAGONAL' )
              efactors%RHS( np1 : npm ) =                                      &
                efactors%RHS_orig( np1 : npm ) + C%val( : m ) * SOL( np1 : npm )
            CASE ( 'SCALED_IDENTITY' )
              efactors%RHS( np1 : npm ) =                                      &
                efactors%RHS_orig( np1 : npm ) + C%val( 1 ) * SOL( np1 : npm )
            CASE ( 'IDENTITY' )
              efactors%RHS( np1 : npm ) =                                      &
                efactors%RHS_orig( np1 : npm ) + SOL( np1 : npm )
            CASE ( 'DENSE' )
              efactors%RHS( np1 : npm ) = efactors%RHS_orig( np1 : npm )
              l = 0
              DO i = 1, m
                DO j = 1, i
                  l = l + 1
                  efactors%RHS( n + i ) =                                      &
                    efactors%RHS( n + i ) + C%val( l ) * SOL( n + j )
                  IF ( i /= j ) efactors%RHS( n + j ) =                        &
                    efactors%RHS( n + j ) + C%val( l ) * SOL( n + i )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              efactors%RHS( np1 : npm ) = efactors%RHS_orig( np1 : npm )
              DO i = 1, m
                DO l = C%ptr( i ), C%ptr( i + 1 ) - 1
                  j = C%col( l )
                  efactors%RHS( n + i ) =                                      &
                    efactors%RHS( n + i ) + C%val( l ) * SOL( n + j )
                  IF ( i /= j ) efactors%RHS( n + j ) =                        &
                    efactors%RHS( n + j ) + C%val( l ) * SOL( n + i )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              efactors%RHS( np1 : npm ) = efactors%RHS_orig( np1 : npm )
              DO l = 1, C%ne
                i = C%row( l ) ; j = C%col( l )
                efactors%RHS( n + i ) =                                        &
                  efactors%RHS( n + i ) + C%val( l ) * SOL( n + j )
                IF ( i /= j ) efactors%RHS( n + j ) =                          &
                  efactors%RHS( n + j ) + C%val( l ) * SOL( n + i )
              END DO
            END SELECT

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                ii = n + i
                DO j = 1, n
                  l = l + 1
                  val = A%val( l )
                  efactors%RHS( j ) = efactors%RHS( j ) - val * SOL( ii )
                  efactors%RHS( ii ) = efactors%RHS( ii ) - val * SOL( j )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                ii = n + i
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l ) ; val = A%val( l )
                  efactors%RHS( j ) = efactors%RHS( j ) - val * SOL( ii )
                  efactors%RHS( ii ) = efactors%RHS( ii ) - val * SOL( j )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                ii = n + A%row( l ) ; j = A%col( l ) ;  val = A%val( l )
                efactors%RHS( j ) = efactors%RHS( j ) - val * SOL( ii )
                efactors%RHS( ii ) = efactors%RHS( ii ) - val * SOL( j )
              END DO
            END SELECT

!  ... or the case of general G

          ELSE
            efactors%RHS = efactors%RHS_orig

!  include terms from A and A^T

            DO l = 1, efactors%k_g
              i = efactors%K%row( l ) ; j = efactors%K%col( l )
              val = efactors%K%val( l )
              efactors%RHS( i ) = efactors%RHS( i ) - val * SOL( j )
              efactors%RHS( j ) = efactors%RHS( j ) - val * SOL( i )
            END DO

!  include terms from G and C

            DO l = efactors%k_g + 1, efactors%k_pert
              i = efactors%K%row( l ) ; j = efactors%K%col( l )
              val = efactors%K%val( l )
              efactors%RHS( i ) = efactors%RHS( i ) - val * SOL( j )
              IF ( i /= j )                                                    &
                efactors%RHS( j ) = efactors%RHS( j ) - val * SOL( i )
            END DO

!  include terms from any diagonal perturbation to G

            DO l = efactors%k_pert + 1, efactors%K%ne
              i = efactors%K%row( l )
              efactors%RHS( i ) =                                              &
                efactors%RHS( i ) - efactors%K%val( l ) * SOL( i )
            END DO
          END IF

          IF ( control%print_level >= 4 .AND. control%out > 0 )                &
            WRITE( control%out, "( A, ' maximum residual = ', ES10.4 )" )      &
              prefix, MAXVAL( ABS( efactors%RHS( : npm ) ) )
        END IF
      END DO

      IF ( control%get_norm_residual )                                         &
        inform%norm_residual = MAXVAL( ABS( efactors%RHS( : npm ) ) )

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_solve_explicit

      END SUBROUTINE SBLS_solve_explicit

!-*-*   S B L S _ S O L V E _ E X P L I C I T _ L M  S U B R O U T I N E  * -*-

      SUBROUTINE SBLS_solve_explicit_lm( n, m, A, C, efactors, control,        &
                                         inform, SOL, H_lm )

!  Limited-memory secant approximations may generically be written as
!  B = D - U W^{-1} U^T, where D, U and W encode a sequence of t secant
!  triples (s,y,delta), and W and U are 2t by 2t and n by 2t; see Byrd,
!  Nodedal and Schnabel, Math. Prog. 63:2 (1994) 129-156, for details.

!  Using GALAHAD_LMS to build B to account for a sequence of (s,y,delta)
!  using the L-BFGS method, this subroutine solves linear systems

!   ( R B R^T + H   A^T ) ( x ) = ( b )    (1)
!   (       A       -C  ) ( y )   ( c )

!  and computes residuals

!   ( R B R^T + H   A^T ) ( x ) - ( b )
!   (       A       -C  ) ( y )   ( c )

!  for a given restriction matrix R, symmetric matrices H and C,
!  rectangular matrix A and right-hand sides b and c

!  [ ** a RESTRICTION matrix is a rectangular (or square) matrix involving
!     rows of the identity matrix that identifies specific rows. For example,
!     the restriction matrix

!       R = ( 0 0 1 0 )
!           ( 0 1 0 0 )

!     identifies rows 3 and 2, and R B R^T gives the two-by-two submatrix

!          ( b_33 b_32 )
!          ( b_23 b_22 )

!     of the four-by-four matrix B; the common case with R = I is treated
!     specially by the package when %restricted = .FALSE.. If  %restricted
!     = .TRUE., %restricted(i), i = 1, .., %n_restricted, gives the index
!     in the original (identity) ordering of the ith restricted component;
!     for the example above, %n_restricted = 2, and %restricted(:n_restricted)
!     = (/3,2/) ** ]

!  Writing G = H + R D R^T and V = R U, system (1) becomes

!   ( G - V W^{-1} V^T   A^T ) ( x ) = ( b )
!   (          A         -C  ) ( y )   ( c )

! and, on introducing the auxiliary variable z = W^{-1} V^T x,
! the solution also satisfies

!   (  G   A^T  V ) (  x  )   ( b )
!   (  A   -C   0 ) (  y  ) = ( c ).
!   ( V^T   0   W ) ( - z )   ( 0 )

! Thus, so long as the leading two-by-two block

!   K = ( G  A^T )
!       ( A  -C  )

!  may be factorized, the solution x and y may be found using these
!  factors and those of the 2t x 2t Schur complement

!    S = W - ( V^T 0 ) K^{-1} ( V ) = W - ( V^T R^T 0 ) K^{-1} ( R U )
!                             ( 0 )                            (  0  )

!  as found by SBLS_form_n_factorize_explicit. In particular, letting

!  P = ( V ) = ( R U ), v = ( x ) and a = ( b )
!      ( 0 )   (  0  )      ( y )         ( c )

!  v = u + w, where K u = P z, S z = P^T w and K w = a

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A, C
      TYPE ( SBLS_explicit_factors_type ), INTENT( INOUT ) :: efactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL
      TYPE ( LMS_data_type ), INTENT( INOUT ) :: H_lm

!  Local variables

      INTEGER :: iter, i, ii, j, l, np1, npm, oi, oj
      REAL ( KIND = wp ) :: val, z
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Allocate workspace arrays

      np1 = n + 1 ;  npm = n + m
      IF ( npm /= efactors%len_sol_workspace_lm ) THEN
        array_name = 'sbls: efactors%RHS_o'
        CALL SPACE_resize_array( npm, efactors%RHS_o,                          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%RHS_u'
        CALL SPACE_resize_array( npm, efactors%RHS_u,                          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%RHS_w'
        CALL SPACE_resize_array( npm, efactors%RHS_w,                          &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%Z'
        CALL SPACE_resize_array( efactors%len_s_max, 1, efactors%Z,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .FALSE.,                                               &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        efactors%len_sol_workspace_lm = npm
      END IF

!  compute the original residual

      efactors%RHS_o( : npm ) = SOL( : npm )
      efactors%RHS_w( : npm ) = efactors%RHS_o( : npm )
      SOL( : npm ) = zero

!  solve the system with iterative refinement

      IF ( control%print_level >= 11 .AND. control%out >= 0 )                  &
        WRITE( control%out, "( ' RHS_lm:', /, ( 3ES24.16 ) )" )                &
          ( efactors%RHS_w( i ), i = 1, npm )

      IF ( control%print_level >= 4 .AND. control%out > 0 )                    &
        WRITE( control%out, "( A, ' maximum residual_lm = ', ES10.4 )" )       &
          prefix, MAXVAL( ABS( efactors%RHS_w( : npm ) ) )

!write(6,*) control%itref_max
      DO iter = 0, control%itref_max

!  solve to obtain w <- K^{-1} w

!write(6,"( ' w ', 5ES12.4)") efactors%RHS_w( : n + m )
        CALL SBLS_solve_explicit( n, m, A, C, efactors, control, inform,       &
                                  efactors%RHS_w )
!write(6,"( ' w ', 5ES12.4)") efactors%RHS_w( : n + m )

!  unrestricted case:  form the product z = P^T w, where P^T = (     Y^T  : 0  )
!                                                              ( delta S^T : 0 )

        IF ( H_lm%restricted == 0 ) THEN
          DO j = 1, H_lm%length
            oj = H_lm%ORDER( j )
            efactors%Z( j, 1 ) =                                               &
              DOT_PRODUCT( efactors%RHS_w( : H_lm%n ), H_lm%Y( : H_lm%n, oj ) )
            efactors%Z( j + H_lm%length, 1 ) = H_lm%delta *                    &
              DOT_PRODUCT( efactors%RHS_w( : H_lm%n ), H_lm%S( : H_lm%n, oj ) )
          END DO

!  restricted case: form R^T w

        ELSE
          efactors%W( : H_lm%n ) = zero
          DO i = 1, H_lm%n_restriction
            j = H_lm%RESTRICTION( i )
            IF ( j <= H_lm%n ) efactors%W( j ) = efactors%RHS_w( i )
          END DO
!         efactors%W( H_lm%RESTRICTION( :  H_lm%n_restriction ) )              &
!           = efactors%RHS_w( :  H_lm%n_restriction )

!   form the product z = P^T w, where P^T = (     Y^T R^T   : 0 )
!                                           ( delta S^T R^T : 0 )

          DO j = 1, H_lm%length
            oj = H_lm%ORDER( j )
            efactors%Z( j, 1 ) =                                               &
              DOT_PRODUCT( efactors%W( :  H_lm%n ), H_lm%Y( :  H_lm%n, oj ) )
            efactors%Z( j + H_lm%length, 1 ) = H_lm%delta *                    &
              DOT_PRODUCT( efactors%W( :  H_lm%n ), H_lm%S( :  H_lm%n, oj ) )
          END DO
        END IF
!write(6,"( ' z ', 5ES12.4)") efactors%Z( : efactors%len_s, 1 )

!  solve to obtain z <- S^{-1} z

        CALL SYTRS( 'L', efactors%len_s, 1, efactors%S, efactors%len_s_max,    &
                    efactors%IPIV, efactors%Z, efactors%len_s_max, i )

        IF ( i < 0 ) THEN
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' solve exit status = ', I0 )" )         &
              prefix, inform%sls_solve_status
          inform%status = GALAHAD_error_solve
          RETURN
        END IF

!write(6,"( ' z ', 5ES12.4)") efactors%Z( : efactors%len_s, 1 )

!  unrestricted case: form the product u = P z, where P = ( R Y : R delta S )
!                                                         (  0  :     0     )

        IF ( H_lm%restricted == 0 ) THEN
          efactors%RHS_u( : npm ) = zero
          DO j = 1, H_lm%length
            oj = H_lm%ORDER( j )
            z = efactors%Z( j + H_lm%length, 1 )
            efactors%RHS_u( : H_lm%n )                                         &
              = efactors%RHS_u( : H_lm%n ) + H_lm%S( : H_lm%n, oj ) * z
          END DO
          efactors%RHS_u( : H_lm%n ) = efactors%RHS_u( : H_lm%n ) * H_lm%delta
          DO j = 1, H_lm%length
            oj = H_lm%ORDER( j )
            z = efactors%Z( j, 1 )
            efactors%RHS_u( : H_lm%n )                                         &
             = efactors%RHS_u( : H_lm%n ) + H_lm%Y( : H_lm%n, oj ) * z
          END DO

!  restricted case: form the product u = P z, where P = ( R Y : R delta S )
!                                                       (  0  :     0     )

        ELSE
          efactors%W( : n ) = zero
          DO j = 1, H_lm%length
            oj = H_lm%ORDER( j )
            z = efactors%Z( j + H_lm%length, 1 )
            efactors%W( : H_lm%n )                                             &
              = efactors%W( : H_lm%n ) + H_lm%S( : H_lm%n, oj ) * z
          END DO
          efactors%W( : H_lm%n ) = efactors%W( : H_lm%n ) * H_lm%delta
          DO j = 1, H_lm%length
            oj = H_lm%ORDER( j )
            z = efactors%Z( j, 1 )
            efactors%W( : H_lm%n )                                             &
              = efactors%W( : H_lm%n ) + H_lm%Y( : H_lm%n, oj ) * z
          END DO
          DO i = 1, H_lm%n_restriction
            j = H_lm%RESTRICTION( i )
            IF ( j <= H_lm%n ) THEN
              efactors%RHS_u( i ) = efactors%W( j )
            ELSE
              efactors%RHS_u( i ) = zero
            END IF
          END DO
!         efactors%RHS_u( : H_lm%n_restriction )                               &
!           = efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )
          efactors%RHS_u(  H_lm%n_restriction + 1 : npm ) = zero
        END IF

!write(6,"( ' u ', 5ES12.4)") efactors%RHS_u( : n + m )

!  solve to obtain u <- K^{-1} u

        CALL SBLS_solve_explicit( n, m, A, C, efactors, control, inform,       &
                                  efactors%RHS_u )

!write(6,"( ' u ', 5ES12.4)") efactors%RHS_u( : n + m )

!  update w <- w + u

        efactors%RHS_w( : npm )                                                &
          = efactors%RHS_w( : npm ) + efactors%RHS_u( : npm )

!  Update the estimate of the solution

!write(6,"( ( I6, 2ES22.14 ) )" ) ( i, SOL( i ), efactors%RHS( i ), i = 1, npm )
        SOL( : npm ) = SOL( : npm ) + efactors%RHS_w( : npm )

!write(6,"( ' sol ', 5ES12.4)") SOL( : npm )

        IF ( control%print_level >= 4 .AND. control%out > 0 )                  &
        WRITE( control%out, "( A, ' norm of solution and correction_lm =',     &
       &  2ES11.4 )" ) prefix, MAXVAL( ABS( SOL( : npm ) ) ),                  &
            MAXVAL( ABS( efactors%RHS_w( : npm ) ) )

!  Form the residuals

        IF ( iter < control%itref_max .OR. control%get_norm_residual ) THEN

!  first form the residuals

!   r = ( b ) - (  G   A^T  V ) (  x  )
!       ( c )   (  A   -C   0 ) (  y  )

!  ... for the case where G is diagonal ...

          IF ( inform%factorization == 1 ) THEN
            efactors%RHS_w( : n ) =                                            &
              efactors%RHS_o( : n ) - efactors%G_diag( : n ) * SOL( : n )

            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'ZERO', 'NONE' )
              efactors%RHS_w( np1 : npm ) = efactors%RHS_o( np1 : npm )
            CASE ( 'DIAGONAL' )
              efactors%RHS_w( np1 : npm ) =                                    &
                efactors%RHS_o( np1 : npm ) + C%val( : m ) * SOL( np1 : npm )
            CASE ( 'SCALED_IDENTITY' )
              efactors%RHS_w( np1 : npm ) =                                    &
                efactors%RHS_o( np1 : npm ) + C%val( 1 ) * SOL( np1 : npm )
            CASE ( 'IDENTITY' )
              efactors%RHS_w( np1 : npm ) =                                    &
                efactors%RHS_o( np1 : npm ) + SOL( np1 : npm )
            CASE ( 'DENSE' )
              efactors%RHS_w( np1 : npm ) = efactors%RHS_o( np1 : npm )
              l = 0
              DO i = 1, m
                DO j = 1, i
                  l = l + 1
                  efactors%RHS_w( n + i ) =                                    &
                    efactors%RHS_w( n + i ) + C%val( l ) * SOL( n + j )
                  IF ( i /= j ) efactors%RHS_w( n + j ) =                      &
                    efactors%RHS_w( n + j ) + C%val( l ) * SOL( n + i )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              efactors%RHS_w( np1 : npm ) = efactors%RHS_o( np1 : npm )
              DO i = 1, m
                DO l = C%ptr( i ), C%ptr( i + 1 ) - 1
                  j = C%col( l )
                  efactors%RHS_w( n + i ) =                                    &
                    efactors%RHS_w( n + i ) + C%val( l ) * SOL( n + j )
                  IF ( i /= j ) efactors%RHS_W( n + j ) =                      &
                    efactors%RHS_w( n + j ) + C%val( l ) * SOL( n + i )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              efactors%RHS_w( np1 : npm ) = efactors%RHS_o( np1 : npm )
              DO l = 1, C%ne
                i = C%row( l ) ; j = C%col( l )
                efactors%RHS_w( n + i ) =                                      &
                  efactors%RHS_w( n + i ) + C%val( l ) * SOL( n + j )
                IF ( i /= j ) efactors%RHS_w( n + j ) =                        &
                  efactors%RHS_w( n + j ) + C%val( l ) * SOL( n + i )
              END DO
            END SELECT

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                ii = n + i
                DO j = 1, n
                  l = l + 1
                  val = A%val( l )
                  efactors%RHS_w( j ) = efactors%RHS_w( j ) - val * SOL( ii )
                  efactors%RHS_w( ii ) = efactors%RHS_w( ii ) - val * SOL( j )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                ii = n + i
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l ) ; val = A%val( l )
                  efactors%RHS_w( j ) = efactors%RHS_w( j ) - val * SOL( ii )
                  efactors%RHS_w( ii ) = efactors%RHS_w( ii ) - val * SOL( j )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                ii = n + A%row( l ) ; j = A%col( l ) ;  val = A%val( l )
                efactors%RHS_w( j ) = efactors%RHS_w( j ) - val * SOL( ii )
                efactors%RHS_w( ii ) = efactors%RHS_w( ii ) - val * SOL( j )
              END DO
            END SELECT

!  ... or the case of general G

          ELSE
            efactors%RHS_w = efactors%RHS_o

!  include terms from A and A^T

            DO l = 1, efactors%k_g
              i = efactors%K%row( l ) ; j = efactors%K%col( l )
              val = efactors%K%val( l )
              efactors%RHS_w( i ) = efactors%RHS_W( i ) - val * SOL( j )
              efactors%RHS_w( j ) = efactors%RHS_W( j ) - val * SOL( i )
            END DO

!  include terms from G and C

            DO l = efactors%k_g + 1, efactors%k_pert
              i = efactors%K%row( l ) ; j = efactors%K%col( l )
              val = efactors%K%val( l )
              efactors%RHS_w( i ) = efactors%RHS_W( i ) - val * SOL( j )
              IF ( i /= j )                                                    &
                efactors%RHS_w( j ) = efactors%RHS_W( j ) - val * SOL( i )
            END DO

!  include terms from any diagonal perturbation to G

!write(6,*) ' pert ',  efactors%K%val( efactors%k_pert + 1 : efactors%K%ne )
            DO l = efactors%k_pert + 1, efactors%K%ne
              i = efactors%K%row( l )
              efactors%RHS_w( i ) =                                            &
                efactors%RHS_w( i ) - efactors%K%val( l ) * SOL( i )
            END DO
          END IF

!  now add R U W^{-1} U^T R^T x to the first block, see LMS_apply in GALAHAD_LMS

!  ----------------------------------------------------------------------------
!  apply the limited-memory BFGS (L-BFGS) secant approximation formula to v:
!
!   r_1 = r_1 + R [ Y : delta S ] [ -D      L^T     ]^{-1} [     Y^T   ] R^T x.
!                                 [ L   delta S^T S ]      [ delta S^T ]
!
!  Since
!
!   [ -D      L^T    ] = [   D^{1/2}   0 ] [ -I 0 ] [ D^{1/2} -D^{-1/2} L^T ]
!   [ L  delta S^T S ]   [ -L D^{-1/2} I ] [  0 C ] [    0          I       ]
!
!
!  with C = delta S^T S + L D^{-1} L^T,
!
!   r_1 =  r_1 + R [ Y : delta S ] [ p ], where                (L-BFGS-1)
!                                  [ q ]
!
!   [ D^{1/2} -D^{-1/2} L^T ] [ p ] = [ p2 ],                  (L-BFGS-2)
!   [    0          I       ] [ q ]   [ q2 ]
!
!   [ -I 0 ] [ p2 ] = [ p1 ],                                  (L-BFGS-3)
!   [  0 C ] [ q2 ]   [ q1 ]
!
!   [   D^{1/2}   0 ] [ p1 ] = [ p0 ] and                      (L-BFGS-4)
!   [ -L D^{-1/2} I ] [ q1 ]   [ q0 ]
!
!   [ p0 ] = [     Y^T R^T x   ]                               (L-BFGS-5)
!   [ q0 ]   [ delta S^T R^T x ]
!
!  ----------------------------------------------------------------------------

!  the vector (q,p) is stored as the two columns of the matrix QP

!  unrestricted case (L-BFGS-5): [ p ] = [     Y^T x   ]
!                                [ q ]   [ delta S^T x ]

          IF ( H_lm%restricted == 0 ) THEN
            CALL GEMV( 'T', H_lm%n, H_lm%length, H_lm%delta, H_lm%S, H_lm%n,   &
                        SOL, 1, zero, H_lm%QP( : , 1 ), 1 )
            CALL GEMV( 'T', H_lm%n, H_lm%length, one, H_lm%Y, H_lm%n,          &
                        SOL, 1, zero, H_lm%QP( : , 2 ), 1 )

!  restricted case (L-BFGS-5): [ p ] = [     Y^T R^T x   ]
!                              [ q ]   [ delta S^T R^T x ]

          ELSE
            efactors%W( H_lm%n ) = zero
            DO i = 1, H_lm%n_restriction
              j = H_lm%RESTRICTION( i )
              IF ( j <= H_lm%n ) efactors%W( j ) = SOL( i )
            END DO
!           efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )             &
!             = SOL( : H_lm%n_restriction )
            CALL GEMV( 'T', H_lm%n, H_lm%length, H_lm%delta, H_lm%S, H_lm%n,   &
                        efactors%W, 1, zero, H_lm%QP( : , 1 ), 1 )
            CALL GEMV( 'T', H_lm%n, H_lm%length, one, H_lm%Y, H_lm%n,          &
                        efactors%W, 1, zero, H_lm%QP( : , 2 ), 1 )
          END IF

!  permute q and p

          DO i = 1, H_lm%length
            oi = H_lm%ORDER( i )
            H_lm%QP_perm( i, 1 ) = H_lm%QP( oi, 1 )
            H_lm%QP_perm( i, 2 ) = H_lm%QP( oi, 2 )
          END DO

!  apply (L-BFGS-4) p -> D^{-1/2} p and q -> q + L D^{-1/2} p

          DO i = 1, H_lm%length
            H_lm%QP_PERM( i, 2 ) = H_lm%QP_PERM( i, 2 ) / H_lm%L_scaled( i, i )
          END DO

          DO i = 2, H_lm%length
            val = H_lm%QP_PERM( i, 1 )
            DO j = 1, i - 1
              val = val + H_lm%L_scaled( i, j ) * H_lm%QP_PERM( j, 2 )
            END DO
            H_lm%QP_PERM( i, 1 ) = val
          END DO

!  apply (L-BFGS-3) q -> C^{-1} q (using the Cholesky factors of C)

          i = 0
          CALL POTRS( 'L', H_lm%length, 1, H_lm%C, H_lm%len_c, H_lm%QP_PERM,   &
                      H_lm%m, i )
          IF ( i /= 0 ) THEN
            IF ( control%error > 0 .AND. control%print_level > 0 )             &
              WRITE( control%error, "( A, ' Cholesky solve error ', I0 )" )    &
                prefix, i
            inform%status = GALAHAD_error_factorization
            RETURN
          END IF

!  apply (L-BFGS-2) p -> D^{-1/2} ( - p + D^{-1/2} L^T q )

          DO i = 1, H_lm%length - 1
            val = - H_lm%QP_PERM( i, 2 )
            DO j = i + 1, H_lm%length
              val = val + H_lm%L_scaled( j, i ) * H_lm%QP_PERM( j, 1 )
            END DO
            H_lm%QP_PERM( i, 2 ) = val
          END DO
          H_lm%QP_PERM( H_lm%length, 2 ) = - H_lm%QP_PERM( H_lm%length, 2 )

          DO i = 1, H_lm%length
            H_lm%QP_PERM( i, 2 ) = H_lm%QP_PERM( i, 2 ) / H_lm%L_scaled( i, i )
          END DO

!  unpermute q and p

          DO i = 1, H_lm%length
            oi = H_lm%ORDER( i )
            H_lm%QP( oi, 1 ) = H_lm%QP_perm( i, 1 )
            H_lm%QP( oi, 2 ) = H_lm%QP_perm( i, 2 )
          END DO

!  unrestricted case: apply (L-BFGS-1) r_1 <- r_1 + delta S q + Y p

          IF ( H_lm%restricted == 0 ) THEN
            CALL GEMV( 'N', H_lm%n, H_lm%length, H_lm%delta, H_lm%S, H_lm%n,   &
                       H_lm%QP( : , 1 ), 1, one, efactors%RHS_w( : n ), 1 )
            CALL GEMV( 'N', H_lm%n, H_lm%length, one, H_lm%Y, H_lm%n,          &
                       H_lm%QP( : , 2 ), 1, one, efactors%RHS_w( : n ), 1 )

!  restricted case: apply (L-BFGS-1) r_1 <- r_1 + R( delta S q + Y p )

          ELSE
            CALL GEMV( 'N', H_lm%n, H_lm%length, H_lm%delta, H_lm%S, H_lm%n,   &
                       H_lm%QP( : , 1 ), 1, zero, efactors%W( : H_lm%n ), 1 )
            CALL GEMV( 'N', H_lm%n, H_lm%length, one, H_lm%Y, H_lm%n,          &
                       H_lm%QP( : , 2 ), 1, one, efactors%W( : H_lm%n ), 1 )
            DO i = 1, H_lm%n_restriction
              j = H_lm%RESTRICTION( i )
              IF ( j <= H_lm%n )                                               &
                efactors%RHS_w( i ) = efactors%RHS_w( i ) + efactors%W( j )
            END DO
!           efactors%RHS_w( : H_lm%n_restriction )                             &
!             = efactors%RHS_w( : H_lm%n_restriction )                         &
!               + efactors%W( H_lm%RESTRICTION( : H_lm%n_restriction ) )
          END IF

!write(6,*) 'res ', efactors%RHS_w( : npm )
          IF ( control%print_level >= 4 .AND. control%out > 0 )                &
            WRITE( control%out, "( A, ' maximum residual_lm = ', ES10.4 )" )   &
              prefix, MAXVAL( ABS( efactors%RHS_w( : npm ) ) )
        END IF
      END DO

      IF ( control%get_norm_residual )                                         &
        inform%norm_residual = MAXVAL( ABS( efactors%RHS_w( : npm ) ) )

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_solve_explicit_lm

      END SUBROUTINE SBLS_solve_explicit_lm

!-*-   S B L S _ P A R T _S O L V E _ E X P L I C I T   S U B R O U T I N E  -*-

      SUBROUTINE SBLS_part_solve_explicit( part, n, m, A, efactors,            &
                                           control, inform, SOL )

!  If the factors L D L(transpose) of ( G   A(transpose) ) are available, solve
!                                     ( A        -C      )
!   L x = b            (part = 'L'),
!   D x = b            (part = 'D') or
!   L(transpose) x = b (part = 'U')

!  where b is input in SOL and SOL is overwritten by x on exit

!  Otherwise if the factors L D L(transpose) = C + A G(inv) A(transpose) are
!  available, then since

!  ( G   A^T ) = ( I         0 ) ( H  0 ) (I  H(inv) A(transpose) )
!  ( A    -C )   ( A H(inv)  L ) ( 0 -D ) (0     L(transpose)     )

!  solve

!  ( I         0 ) ( x ) = ( a )           (part = 'L'),
!  ( A H(inv)  L ) ( y )   ( b )
!  ( H  0 ) ( x ) = ( a )                  (part = 'D') or
!  ( 0 -D ) ( y )   ( b )
!  (I  H(inv) A(transpose) ) ( x ) = ( a ) (part = 'U')
!  (0     L(transpose)     ) ( y )   ( b )

!  where ( a ) is input in SOL and SOL is overwritten by ( x ) on exit
!        ( b )                                           ( y )

!  Dummy arguments

      CHARACTER( LEN = 1 ) :: part
      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SBLS_explicit_factors_type ), INTENT( INOUT ) :: efactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL

!  Local variables

      INTEGER :: i, ii, j, l, np1, npm
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Allocate workspace arrays

      npm = n + m
      IF ( n /= efactors%len_part_sol_workspace .AND. m > 0 ) THEN
        array_name = 'sbls: efactors%W'
        CALL SPACE_resize_array( n, efactors%W,                                &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        efactors%len_sol_workspace = n
      END IF

!  Solve the system with iterative refinement

      IF ( control%print_level >= 11 .AND. control%out >= 0 )                  &
        WRITE( control%out, "( ' RHS:', /, ( 3ES24.16 ) )" )                   &
          ( SOL( i ), i = 1, npm )

      IF ( control%print_level >= 4 .AND. control%out > 0 )                    &
        WRITE( control%out, "( A, ' maximum residual = ', ES10.4 )" )          &
          prefix, MAXVAL( ABS( SOL( : npm ) ) )

!  Use factors of the Schur complement

      IF ( inform%factorization == 1 ) THEN
        np1 = n + 1

!  set x = a and solve L y = b - A H(inv) x

        IF ( part == 'L' ) THEN
          IF ( m > 0 ) THEN

!  form w = H(inv) a

            efactors%W( : n ) = SOL( : n ) / efactors%G_diag( : n )

!  overwrite b <- b - A w

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                ii = n + i
                SOL( ii ) = SOL( ii ) -                                        &
                  DOT_PRODUCT( A%val( l + 1 : l + n ), efactors%W( : n ) )
                l = l + n
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                ii = n + i
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  SOL( ii ) = SOL( ii ) - A%val( l ) * efactors%W( A%col( l ) )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                ii = n + A%row( l )
                SOL( ii ) = SOL( ii ) - A%val( l ) * efactors%W( A%col( l ) )
              END DO
            END SELECT

!  solve L y = b

            CALL SLS_part_solve( part, SOL( np1 : npm ), efactors%K_data,      &
                                 efactors%K_control, inform%SLS_inform )
          END IF

!  set x = H(inv) a and y = - D(inv) b

        ELSE IF ( part == 'D' ) THEN
          SOL( : n ) = SOL( : n ) / efactors%G_diag( : n )
          IF ( m > 0 ) THEN
            CALL SLS_part_solve( part, SOL( np1 : npm ), efactors%K_data,      &
                                 efactors%K_control, inform%SLS_inform )
            SOL( np1 : npm ) = - SOL( np1 : npm )
          END IF

!  solve L(transpose) y = b and set x = a - H(inv) A(transpose) y

        ELSE IF ( part == 'U' ) THEN
          IF ( m > 0 ) THEN
            CALL SLS_part_solve( part, SOL( np1 : npm ), efactors%K_data,      &
                                 efactors%K_control, inform%SLS_inform )

!  form w =  A(transpose) y

            efactors%W( : n ) = zero
            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, m
                DO j = 1, n
                  l = l + 1
                  efactors%W( j ) = efactors%W( j ) + A%val( l ) * SOL( n + i )
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, m
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l )
                  efactors%W( j ) = efactors%W( j ) + A%val( l ) * SOL( n + i )
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                j = A%col( l )
                efactors%W( j )                                                &
                  = efactors%W( j ) + A%val( l ) * SOL( n + A%row( l ) )
              END DO
            END SELECT

!  form x = a - H(inv) w

            SOL( : n ) = SOL( : n ) - efactors%W( : n ) / efactors%G_diag( : n )
          END IF
        END IF

!  Use factors of the augmented system

      ELSE
        CALL SLS_part_solve( part, SOL, efactors%K_data,                       &
                             efactors%K_control, inform%SLS_inform )
        inform%sls_solve_status = inform%SLS_inform%status
        IF ( inform%sls_solve_status < 0 ) THEN
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' solve exit status = ', I0 )" )         &
              prefix, inform%sls_solve_status
          inform%status = GALAHAD_error_solve
          RETURN
        END IF
      END IF

      inform%status = GALAHAD_ok
      RETURN

! end of subroutine SBLS_part_solve_explicit

      END SUBROUTINE SBLS_part_solve_explicit

!-*-*-*-   S B L S _ S O L V E _ I M P L I C I T   S U B R O U T I N E   -*-*-*-

      SUBROUTINE SBLS_solve_implicit( ifactors, control, inform, SOL )

!  To solve

!    Kz = ( P  A^T ) z = b
!         ( A   -C )

!   (i) transform b to c = IP b
!   (ii) solve perm(K) w = c
!   (iii) recover z = IP^T w

!  where IP = (IC   0 )
!             ( 0  IR )

!  and the permutations IR and IC are such that
!  A = IR ( A1  A2 ) IC^T and the "basis" matrix A1 is nonsingular
!         (  0   0 )
!  This induces a re-ordering IC^T P IC of P

!  Iterative refinement may be used

!  Dummy arguments

      TYPE ( SBLS_implicit_factors_type ), INTENT( INOUT ) :: ifactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( ifactors%n + ifactors%m ) :: SOL

!  Local variables

      INTEGER :: i, iter, j, l, rank_a, n, k_n
      INTEGER :: start_1, end_1, start_2, end_2, start_3, end_3
      REAL ( KIND = wp ) :: val
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      k_n = ifactors%k_n

!  Allocate workspace arrays

      IF ( k_n /= ifactors%len_sol_workspace ) THEN
        array_name = 'sbls: ifactors%SOL_current'
        CALL SPACE_resize_array( k_n, ifactors%SOL_current,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: ifactors%SOL_perm'
        CALL SPACE_resize_array( k_n, ifactors%SOL_perm,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: ifactors%RHS_orig'
        CALL SPACE_resize_array( k_n, ifactors%RHS_orig,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        ifactors%len_sol_workspace = k_n
      END IF

      n = ifactors%n ; rank_a = ifactors%rank_a

!  Partition the variables r and v

      start_1 = 1 ; end_1 = rank_a
      start_2 = rank_a + 1 ; end_2 = n
      start_3 = n + 1 ; end_3 = n + rank_a

!  Permute the variables

      DO i = 1, n
        ifactors%RHS_orig( i ) = SOL( ifactors%A_COLS_basic( i ) )
      END DO

      DO i = 1, rank_a
        ifactors%RHS_orig( n + i ) = SOL( n + ifactors%A_ROWS_basic( i ) )
      END DO

!  Solve the system with iterative refinement

      DO iter = 0, control%itref_max

!    ********************
!    *  RESIDUAL STAGE  *
!    ********************

!  Compute the current residual

        IF ( iter > 0 ) THEN

          SOL( : k_n ) = ifactors%RHS_orig
          ifactors%SOL_perm = zero

!  1. First, form
!     ===========

!    ( r_1 )   (  P_11^T   P_21^T  P_31^T ) ( v_1 ), where P_31^T = B_31^-1
!    ( r_2 ) = (  0        P_22^T    0    ) ( v_2 )
!    ( r_3 )   (  A_1       A_2      0    ) ( v_3 )

!  with v in ifactors%SOL_current and r in SOL

!  r_1 = P_31^T v_3

          IF ( ifactors%unitb31 ) THEN
            SOL( start_1 : end_1 ) = ifactors%SOL_current( start_3 : end_3 )
          ELSE
            SOL( start_1 : end_1 ) = zero
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

!  r_1 <- r_1 + P_11^T v_1

          IF ( .NOT. ifactors%zerop11 ) THEN
            DO l = 1, ifactors%P11%ne
              j = ifactors%P11%col( l )
              SOL( j ) = SOL( j ) + ifactors%P11%val( l )                      &
               * ifactors%SOL_perm( ifactors%P11%row( l ) )
            END DO
          END IF

!  r_1 <- r_1 + P_21^T v_2

          IF ( .NOT. ifactors%zerop21 ) THEN
            DO l = 1, ifactors%P21%ne
              j = ifactors%P21%col( l )
              SOL( j ) = SOL( j ) + ifactors%P21%val( l )                      &
               * ifactors%SOL_perm( ifactors%P21%row( l ) )
            END DO
          END IF

!  r_2 = P_22^T v_2

          IF ( ifactors%unitp22 ) THEN
            SOL( start_2 : end_2 ) = ifactors%SOL_current( start_2 : end_2 )
          ELSE
            SOL( start_2 : end_2 ) = zero
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb22 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

!  r_3 = A_1 v_1

          SOL( start_3 : end_3 ) = zero
          DO l = 1, ifactors%A1%ne
            i = n + ifactors%A1%row( l )
            SOL( i ) = SOL( i ) + ifactors%A1%val( l )                         &
              * ifactors%SOL_current( ifactors%A1%col( l ) )
          END DO

!  r_3 r3 + A_2 v_2

          DO l = 1, ifactors%A2%ne
            i = n + ifactors%A2%row( l )
            SOL( i ) = SOL( i ) + ifactors%A2%val( l )                         &
              * ifactors%SOL_current( ifactors%A2%col( l ) )
          END DO

!  2. Next form
!     =========

!     ( v_1 )   (   0     0    B_31^T ) ( r_1 )
!     ( v_2 ) = (   0    B_22  B_32^T ) ( r_2 )
!     ( v_3 )   (  B_31  B_32   B_33  ) ( r_3 )

!  with r in SOL and v in ifactors%SOL_perm

!  v_1 = B_31^T r_3

          IF ( ifactors%unitb31 ) THEN
            ifactors%SOL_perm( start_1 : end_1 ) = SOL( start_3 : end_3 )
          ELSE
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

!  v_2 = B_22 r_2

          IF ( ifactors%unitb22 ) THEN
            ifactors%SOL_perm( start_2 : end_2 ) = SOL( start_2 : end_2 )
          ELSE
            ifactors%SOL_perm( start_2 : end_2 ) = zero
            DO l = 1, ifactors%B22%ne
              i = ifactors%B22%row( l )
              j = ifactors%B22%col( l )
              val = ifactors%B22%val( l )
              ifactors%SOL_perm( j ) = ifactors%SOL_perm( j ) + val * SOL( i )
              IF ( i /= j )                                                    &
                ifactors%SOL_perm( i ) = ifactors%SOL_perm( i ) + val * SOL( j )
            END DO
            IF ( ALLOCATED( ifactors%PERT ) )                                  &
              ifactors%SOL_perm( start_2 : end_2 ) =                           &
                ifactors%SOL_perm( start_2 : end_2 )+                          &
                  ifactors%PERT * SOL( start_2 : end_2 )
          END IF

!  v_2 <- v_2 + B_32^T r_3

          IF ( .NOT. ifactors%zerob32 ) THEN
            DO l = 1, ifactors%B32%ne
              j = ifactors%B32%col( l )
              ifactors%SOL_perm( j ) = ifactors%SOL_perm( j ) +                &
                 ifactors%B32%val( l ) * SOL( ifactors%B32%row( l ) )
            END DO
          END IF

!  v_3 = B_31 r_1

          IF ( ifactors%unitb31 ) THEN
            ifactors%SOL_perm( start_3 : end_3 ) = SOL( start_1 : end_1 )
          ELSE
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

!  v_3 <- v_3 + B_32 r_2

          IF ( .NOT. ifactors%zerob32 ) THEN
            DO l = 1, ifactors%B32%ne
              i = ifactors%B32%row( l )
              ifactors%SOL_perm( i ) = ifactors%SOL_perm( i ) +                &
                 ifactors%B32%val( l ) * SOL( ifactors%B32%col( l ) )
            END DO
          END IF

!  v_3 <- v_3 + B_33 r_3

          IF ( .NOT.  ifactors%zerob33 ) THEN
            DO l = 1, ifactors%B33%ne
              i = ifactors%B33%row( l )
              ifactors%SOL_perm( i ) = ifactors%SOL_perm( i ) +                &
                 ifactors%B33%val( l ) * SOL( ifactors%B33%col( l ) )
            END DO
          END IF

!  3. Last, form
!     ==========

!     ( r_1 )   ( b_1 )   (  P_11    0    A_1^T ) ( v_1 ), where P_31 = B_31^-T
!     ( r_2 ) = ( b_2 ) - (  P_21   P_22  A_2^T ) ( v_2 )
!     ( r_3 )   ( b_3 )   (  P_31    0     0    ) ( v_3 )

!  with v in ifactors%SOL_perm and r in SOL

          SOL( : k_n ) = ifactors%RHS_orig

!  r_1 <- r_1 - P_11 v_1

          IF ( .NOT. ifactors%zerop11 ) THEN
            DO l = 1, ifactors%P11%ne
              i = ifactors%P11%row( l )
              SOL( i ) = SOL( i ) - ifactors%P11%val( l )                      &
                * ifactors%SOL_perm( ifactors%P11%col( l ) )
            END DO
          END IF

!  r_1 <- r_1 - A_1^T v_3

          DO l = 1, ifactors%A1%ne
            j = ifactors%A1%col( l )
            SOL( j ) = SOL( j ) - ifactors%A1%val( l )                         &
             * ifactors%SOL_perm( n + ifactors%A1%row( l ) )
          END DO

!  r_2 <- r_2 - P_21 v_1

          IF ( .NOT. ifactors%zerop21 ) THEN
            DO l = 1, ifactors%P21%ne
              i = ifactors%P21%row( l )
              SOL( i ) = SOL( i ) - ifactors%P21%val( l )                      &
                * ifactors%SOL_perm( ifactors%P21%col( l ) )
            END DO
          END IF

!  r_2 <- r_2 - P_22 v_2

          IF ( ifactors%unitp22 ) THEN
            SOL( start_2 : end_2 ) = SOL( start_2 : end_2 ) -                  &
              ifactors%SOL_perm( start_2 : end_2 )
          ELSE
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb22 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

!  r_2 <- r_2 - A_2^T v_3

          DO l = 1, ifactors%A2%ne
            j = ifactors%A2%col( l )
            SOL( j ) = SOL( j ) - ifactors%A2%val( l )                         &
             * ifactors%SOL_perm( n + ifactors%A2%row( l ) )
          END DO

!  r_3 <- r_3 - P_31 v_1

          IF ( ifactors%unitb31 ) THEN
            SOL( start_3 : end_3 ) = SOL( start_3 : end_3 ) -                  &
              ifactors%SOL_perm( start_1 : end_1 )
          ELSE
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

          IF ( control%get_norm_residual )                                     &
            inform%norm_residual = MAXVAL( ABS( SOL( : k_n ) ) )

!  No residual required

        ELSE
          SOL( : k_n ) = ifactors%RHS_orig
        END IF

!    *****************
!    *  SOLVE STAGE  *
!    *****************

!  1. First solve
!     ===========

!     (  P_11    0    A_1^T ) ( v_1 )   ( r_1 ), where P_31 = B_31^-T
!     (  P_21   P_22  A_2^T ) ( v_2 ) = ( r_2 )
!     (  P_31    0     0    ) ( v_3 )   ( r_3 )

        IF ( control%affine ) THEN
          ifactors%SOL_perm( start_1 : end_1 ) = zero
        ELSE

!  1a. Solve P_31 v_1 = r_3

          IF ( ifactors%unitb31 ) THEN
            ifactors%SOL_perm( start_1 : end_1 ) = SOL( start_3 : end_3 )
          ELSE
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

!  1b. Replace r_1 <- r_1 - P_11 v_1 ...

          IF ( .NOT. ifactors%zerop11 ) THEN
            DO l = 1, ifactors%P11%ne
              i = ifactors%P11%row( l )
              SOL( i ) = SOL( i ) - ifactors%P11%val( l )                      &
                * ifactors%SOL_perm( ifactors%P11%col( l ) )
            END DO
          END IF

!  ... and r_2 <- r_2 - P_21 v_1

          IF ( .NOT. ifactors%zerop21 ) THEN
            DO l = 1, ifactors%P21%ne
              i = ifactors%P21%row( l )
              SOL( i ) = SOL( i ) - ifactors%P21%val( l )                      &
                * ifactors%SOL_perm( ifactors%P21%col( l ) )
            END DO
          END IF
        END IF

!  1c. Solve A^1^T v_3 = r_1

        CALL ULS_SOLVE( ifactors%A1, SOL( start_1 : end_1 ),                   &
                        ifactors%SOL_perm( start_3 : end_3 ),                  &
                        ifactors%A1_data, ifactors%A1_control,                 &
                        inform%ULS_inform, .TRUE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_uls_solve ; RETURN
        END IF

!  1d. Replace r_2 <- r_2 - A_2^T v_3

        DO l = 1, ifactors%A2%ne
          j = ifactors%A2%col( l )
          SOL( j ) = SOL( j ) - ifactors%A2%val( l )                           &
           * ifactors%SOL_perm( n + ifactors%A2%row( l ) )
        END DO

!  1e. Solve P_22 v_2 = r_2

        IF ( ifactors%unitp22 ) THEN
          ifactors%SOL_perm( start_2 : end_2 ) = SOL( start_2 : end_2 )
        ELSE
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' unitpb22 not yet written' )" ) prefix
          inform%status = GALAHAD_not_yet_implemented ; RETURN
        END IF

!  2. Next solve
!     ==========

!     (   0     0    B_31^T ) ( r_1 )   ( v_1 )
!     (   0    B_22  B_32^T ) ( r_2 ) = ( v_2 )
!     (  B_31  B_32   B_33  ) ( r_3 )   ( v_3 )

!  2a. Solve B_31^T v_3 = r_1

        IF ( ifactors%unitb31 ) THEN
          SOL( start_3 : end_3 ) = ifactors%SOL_perm( start_1 : end_1 )
        ELSE
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
          inform%status = GALAHAD_not_yet_implemented ; RETURN
        END IF

!  2b. Replace r_2 <- r_2 - B_32^T v_3 and r_3 <- r_3 - B_33 v_3

        IF ( .NOT. ifactors%zerob32 ) THEN
          DO l = 1, ifactors%B32%ne
            j = ifactors%B32%col( l )
            ifactors%SOL_perm( j ) = ifactors%SOL_perm( j ) -                 &
               ifactors%B32%val( l ) * SOL( ifactors%B32%row( l ) )
          END DO
        END IF

        IF ( .NOT.  ifactors%zerob33 ) THEN
          DO l = 1, ifactors%B33%ne
            i = ifactors%B33%row( l )
            ifactors%SOL_perm( i ) = ifactors%SOL_perm( i ) -                 &
               ifactors%B33%val( l ) * SOL( ifactors%B33%col( l ) )
          END DO
        END IF

!  2c. Solve B_22 v_2 = r_2

        SOL( start_2 : end_2 ) = ifactors%SOL_perm( start_2 : end_2 )
        IF ( .NOT. ifactors%unitb22 ) THEN
          CALL SLS_solve( ifactors%B22, SOL( start_2 : end_2 ),                &
                          ifactors%B22_data, ifactors%B22_control,             &
                          inform%SLS_inform )
          inform%sls_solve_status = inform%SLS_inform%status
          IF ( inform%sls_solve_status < 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' solve exit status = ', I0 )" )       &
                prefix, inform%sls_solve_status
            inform%status = GALAHAD_error_solve ; RETURN
          END IF
        END IF

!  2d. Replace r_3 <- r_3 - B_32 v_2

        IF ( .NOT. ifactors%zerob32 ) THEN
          DO l = 1, ifactors%B32%ne
            i = ifactors%B32%row( l )
            ifactors%SOL_perm( i ) = ifactors%SOL_perm( i ) -                  &
               ifactors%B32%val( l ) * SOL( ifactors%B32%col( l ) )
          END DO
        END IF

!  2e. Solve B_31 v_1 = r3

        IF ( ifactors%unitb31 ) THEN
          SOL( start_1 : end_1 ) = ifactors%SOL_perm( start_3 : end_3 )
        ELSE
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
          inform%status = GALAHAD_not_yet_implemented ; RETURN
        END IF

!  3. Finally solve
!     =============

!     (  P_11^T   P_21^T  P_31^T ) ( v_1 )   ( r_1 ), where P_31^T = B_31^-1
!     (  0        P_22^T    0    ) ( v_2 ) = ( r_2 )
!     (  A_1       A_2      0    ) ( v_3 )   ( r_3 )

!  3a. Solve P_22^T v_2 = r_2

        IF ( ifactors%unitp22 ) THEN
          ifactors%SOL_perm( start_2 : end_2 ) = SOL( start_2 : end_2 )
        ELSE
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' unitpb22 not yet written' )" ) prefix
          inform%status = GALAHAD_not_yet_implemented ; RETURN
        END IF

!  3b. Replace r_1 <- r_1 - P_21^T v_2 ...

        IF ( .NOT. ifactors%zerop21 ) THEN
          DO l = 1, ifactors%P21%ne
            j = ifactors%P21%col( l )
            SOL( j ) = SOL( j ) - ifactors%P21%val( l )                        &
              * ifactors%SOL_perm( ifactors%P21%row( l ) )
          END DO
        END IF

!  ... and r_3 <- r_3 - A_2 v_2

        DO l = 1, ifactors%A2%ne
          i = n + ifactors%A2%row( l )
          SOL( i ) = SOL( i ) - ifactors%A2%val( l )                           &
            * ifactors%SOL_perm( ifactors%A2%col( l ) )
        END DO

!  3c. Solve A_1 v_1 = r_3

        CALL ULS_SOLVE( ifactors%A1, SOL( start_3 : end_3 ),                   &
                        ifactors%SOL_perm( start_1 : end_1 ),                  &
                        ifactors%A1_data, ifactors%A1_control,                 &
                        inform%ULS_inform, .FALSE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_uls_solve ; RETURN
        END IF

!  3d. Replace r_1 <- r_1 - P_11^T v_1

        IF ( .NOT. ifactors%zerop11 ) THEN
          DO l = 1, ifactors%P11%ne
            j = ifactors%P11%col( l )
            SOL( j ) = SOL( j ) - ifactors%P11%val( l )                        &
              * ifactors%SOL_perm( ifactors%P11%row( l ) )
          END DO
        END IF

!  3e. Solve P_31^T v_3 = r_1

        IF ( ifactors%unitb31 ) THEN
          ifactors%SOL_perm( start_3 : end_3 ) = SOL( start_1 : end_1 )
        ELSE
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' unitpb31 not yet written' )" ) prefix
          inform%status = GALAHAD_not_yet_implemented ; RETURN
        END IF

!    ******************
!    *  UPDATE STAGE  *
!    ******************

!  Update the solution

        IF ( iter > 0 ) ifactors%SOL_perm =                                    &
          ifactors%SOL_current + ifactors%SOL_perm
        IF ( iter < control%itref_max )                                        &
          ifactors%SOL_current = ifactors%SOL_perm
      END DO

!  Unpermute the variables

      DO i = 1, n
        SOL( ifactors%A_COLS_basic( i ) ) = ifactors%SOL_perm( i )
      END DO

      IF ( rank_a < ifactors%m ) SOL( n + 1 : n + ifactors%m ) = zero
      DO i = 1, rank_a
        SOL( n + ifactors%A_ROWS_basic( i ) ) = ifactors%SOL_perm( n + i )
      END DO

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_solve_implicit

      END SUBROUTINE SBLS_solve_implicit

!-*-*-*-*-  S B L S _ B A S I S _ S O L V E    S U B R O U T I N E  -*-*-*-*-

      SUBROUTINE SBLS_basis_solve( ifactors, control, inform, SOL )

!  To find a solution to A x = b

!   (i) transform b to c = IR b
!   (ii) solve A1 w1 = c and set w = (w1 0)
!   (iii) recover x = IC^T w

!  and the permutations IR and IC are such that
!  A = IR ( A1  A2 ) IC^T and the "basis" matrix A1 is nonsingular
!         (  0   0 )

!  Iterative refinement may be used

!  Dummy arguments

      TYPE ( SBLS_implicit_factors_type ), INTENT( INOUT ) :: ifactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( ifactors%n + ifactors%m ) :: SOL

!  Local variables

      INTEGER :: i, iter, l, rank_a, n, k_n
      INTEGER :: start_1, end_1, start_2, end_2, start_3, end_3
      CHARACTER ( LEN = 80 ) :: array_name

      k_n =  ifactors%k_n

!  Allocate workspace arrays

      IF ( k_n /= ifactors%len_sol_workspace ) THEN
        array_name = 'sbls: ifactors%SOL_current'
        CALL SPACE_resize_array( k_n, ifactors%SOL_current,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: ifactors%SOL_perm'
        CALL SPACE_resize_array( k_n, ifactors%SOL_perm,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: ifactors%RHS_orig'
        CALL SPACE_resize_array( k_n, ifactors%RHS_orig,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        ifactors%len_sol_workspace = k_n
      END IF

      n = ifactors%n ; rank_a = ifactors%rank_a

!  Partition the variables x

      start_1 = 1 ; end_1 = rank_a
      start_2 = rank_a + 1 ; end_2 = n
      start_3 = n + 1 ; end_3 = n + rank_a

!  Permute the RHS

      DO i = 1, rank_a
        ifactors%RHS_orig( n + i ) = SOL( n + ifactors%A_ROWS_basic( i ) )
      END DO

!  Solve the system with iterative refinement

      DO iter = 0, control%itref_max

!  Store c in SOL and w in SOL_perm

        SOL( start_3 : end_3 ) = ifactors%RHS_orig( start_3 : end_3 )

!  Compute the current residual

        IF ( iter > 0 ) THEN
          DO l = 1, ifactors%A1%ne
            i = n + ifactors%A1%row( l )
            SOL( i ) = SOL( i ) - ifactors%A1%val( l )                         &
              * ifactors%SOL_current( ifactors%A1%col( l ) )
          END DO
        END IF

!  1. Solve A_1 w_1 = c ...

        CALL ULS_SOLVE( ifactors%A1, SOL( start_3 : end_3 ),                   &
                        ifactors%SOL_perm( start_1 : end_1 ), ifactors%A1_data,&
                        ifactors%A1_control, inform%ULS_inform, .FALSE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_uls_solve ; RETURN
        END IF

!  Update the solution

        IF ( iter > 0 ) ifactors%SOL_perm( start_1 : end_1 ) =                 &
          ifactors%SOL_current( start_1 : end_1 ) +                            &
            ifactors%SOL_perm( start_1 : end_1 )
        IF ( iter < control%itref_max )                                        &
          ifactors%SOL_current( start_1 : end_1 ) =                            &
            ifactors%SOL_perm( start_1 : end_1 )
      END DO

!   .. and set w_2 = 0

      ifactors%SOL_perm( start_2 : end_2 ) = zero

!  Unpermute the variables

      DO i = 1, n
        SOL( ifactors%A_COLS_basic( i ) ) = ifactors%SOL_perm( i )
      END DO

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_basis_solve

      END SUBROUTINE SBLS_basis_solve

!-*-*-   S B L S _ FORM _ A _ FACTORIZE _ NULLSPACE  S U B R O U T I N E  -*-*-

      SUBROUTINE SBLS_form_n_factorize_nullspace( n, m, H, A, nfactors,        &
                                                     last_factorization,       &
                                                     control, inform )

!  Form a null-space factorization of
!
!        K = ( G   A^T )
!            ( A    -C )
!
!  for various approximations G of H

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, last_factorization
      TYPE ( SMT_type ), INTENT( IN ) :: H, A
      TYPE ( SBLS_null_space_factors_type ), INTENT( INOUT ) :: nfactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, jj, k, l, out, a_ne, potrf_info
      INTEGER :: liw, lptr, new_h, new_a, error
      LOGICAL :: printi, printe
      CHARACTER ( LEN = 80 ) :: array_name

      REAL ( KIND = wp ) :: val
      REAL :: time_start, time_end
!     REAL :: t1, t2, t3

      LOGICAL :: dense = .TRUE.
!     LOGICAL :: printd = .TRUE.
      LOGICAL :: printd = .FALSE.

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start )

      out = control%out
      printi = out >= 0 .AND. control%print_level >= 1
      error = control%error
      printe = error >= 0 .AND. control%print_level >= 0

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne
      END IF

!     IF ( inform%preconditioner >= 0 ) THEN
!       IF ( printi ) WRITE( out,                                              &
!         "( A, ' Use SBLS_form_n_factorize_explicit subroutine instead' )" )  &
!           prefix
!       inform%status = GALAHAD_error_call_order ; RETURN
!     END IF
      inform%status = GALAHAD_ok

      IF ( last_factorization /= inform%factorization ) THEN
        new_h = 2
        new_a = 2
      ELSE
        new_h = control%new_h
        new_a = control%new_a
      END IF

!  Find a permutation of A = IR ( A1  A2 ) IC^T for which the "basis"
!                               (  0   0 )
!  matrix A1 is nonsingular

!  This induces a re-ordering IC^T P IC of P

! ----------
! Find basis
! ----------

!     CALL CPU_TIME( t1 )

      IF ( new_a > 0 ) THEN

!  Find a "basic" set of rows and colums, that is a non-singular submatrix A1 of
!  maximal rank. Also set up the complement matrix A2 of columns of A not in A1

        nfactors%m = m ; nfactors%n = n
        CALL SBLS_find_A1_and_A2( m, n, a_ne, A, nfactors%A1,                  &
                                  nfactors%A1_data,                            &
                                  nfactors%A1_control, nfactors%A2,            &
                                  nfactors%A_ROWS_basic,                       &
                                  nfactors%A_COLS_basic,                       &
                                  nfactors%A_ROWS_order, nfactors%A_COLS_order,&
                                  nfactors%rank_a, nfactors%k_n, nfactors%n_r, &
                                  prefix, 'sbls: nfactors%', out, printi,      &
                                  control, inform,                             &
                                  nfactors%RHS_orig, nfactors%SOL_current )

!  Reorder A2 so that the matrix is stored by columns (its transpose by rows)

        IF ( nfactors%A2%ne > 0 ) THEN
          lptr = nfactors%n_r + 1
          array_name = 'sbls: nfactors%A2%ptr'
          CALL SPACE_resize_array( lptr, nfactors%A2%ptr,                      &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          liw = MAX( nfactors%n_r, nfactors%rank_a ) + 1
          array_name = 'sbls: nfactors%IW'
          CALL SPACE_resize_array( liw, nfactors%IW,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          nfactors%A2%col( : nfactors%A2%ne ) =                                &
            nfactors%A2%col( : nfactors%A2%ne ) - nfactors%rank_a

          CALL SORT_reorder_by_rows(                                           &
            nfactors%n_r, nfactors%rank_a, nfactors%A2%ne, nfactors%A2%col,    &
            nfactors%A2%row, nfactors%A2%ne, nfactors%A2%val, nfactors%A2%ptr, &
            lptr, nfactors%IW, liw, control%error, control%out,                &
            inform%sort_status )
          IF ( inform%sort_status /= GALAHAD_ok ) THEN
            IF ( printe ) WRITE( error,                                        &
         "( A, ' ULS: sort of A2, info = ', I0 )" )  prefix, inform%sort_status
            IF ( inform%sort_status > 0 ) THEN
              inform%status = GALAHAD_error_sort ; RETURN
            END IF
          END IF

          nfactors%A2%ne = nfactors%A2%ptr( nfactors%n_r + 1 ) - 1
          nfactors%A2%col( : nfactors%A2%ne ) =                                &
            nfactors%A2%col( : nfactors%A2%ne ) + nfactors%rank_a
        END IF
      END IF

!  Reorder H according to the partitions induced by A_1

      IF ( new_a > 0 .OR. new_h > 0 ) THEN

!  Calculate the space needed for H_11, H_21 and H_22 - note that both
!  triangles of H22 will be stored so that it may be accessed by columns

        nfactors%H11%ne = 0
        nfactors%H21%ne = 0
        nfactors%H22%ne = 0
        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL', 'SCALED_IDENTITY', 'IDENTITY' )
          DO l = 1, n
            i = nfactors%A_COLS_order( l )
            j = nfactors%A_COLS_order( l )
            IF ( i > nfactors%rank_a ) THEN
              nfactors%H22%ne = nfactors%H22%ne + 1
            ELSE
              nfactors%H11%ne = nfactors%H11%ne + 1
            END IF
          END DO
        CASE ( 'DENSE' )
          l = 0
          DO ii = 1, n
            i = nfactors%A_COLS_order( ii )
            DO jj = 1, ii
              j = nfactors%A_COLS_order( jj )
              IF ( i > nfactors%rank_a .AND. j > nfactors%rank_a ) THEN
                IF ( i == j ) THEN
                  nfactors%H22%ne = nfactors%H22%ne + 1
                ELSE
                  nfactors%H22%ne = nfactors%H22%ne + 2
                END IF
              ELSE IF ( i <= nfactors%rank_a .AND. j <= nfactors%rank_a ) THEN
                nfactors%H11%ne = nfactors%H11%ne + 1
              ELSE
                nfactors%H21%ne = nfactors%H21%ne + 1
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO ii = 1, n
            i = nfactors%A_COLS_order( ii )
            DO l = H%ptr( ii ), H%ptr( ii + 1 ) - 1
              j = nfactors%A_COLS_order( H%col( l ) )
              IF ( i > nfactors%rank_a .AND. j > nfactors%rank_a ) THEN
                IF ( i == j ) THEN
                  nfactors%H22%ne = nfactors%H22%ne + 1
                ELSE
                  nfactors%H22%ne = nfactors%H22%ne + 2
                END IF
              ELSE IF ( i <= nfactors%rank_a .AND. j <= nfactors%rank_a ) THEN
                nfactors%H11%ne = nfactors%H11%ne + 1
              ELSE
                nfactors%H21%ne = nfactors%H21%ne + 1
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            i = nfactors%A_COLS_order( H%row( l ) )
            j = nfactors%A_COLS_order( H%col( l ) )
            IF ( i > nfactors%rank_a .AND. j > nfactors%rank_a ) THEN
              IF ( i == j ) THEN
                nfactors%H22%ne = nfactors%H22%ne + 1
              ELSE
                nfactors%H22%ne = nfactors%H22%ne + 2
              END IF
            ELSE IF ( i <= nfactors%rank_a .AND. j <= nfactors%rank_a ) THEN
              nfactors%H11%ne = nfactors%H11%ne + 1
            ELSE
              nfactors%H21%ne = nfactors%H21%ne + 1
            END IF
          END DO
        END SELECT

! Allocate the space to store the reordered partitions H_11, H_21 and H_22

        array_name = 'sbls: nfactors%H11%row'
        CALL SPACE_resize_array( nfactors%H11%ne, nfactors%H11%row,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H11%col'
        CALL SPACE_resize_array( nfactors%H11%ne, nfactors%H11%col,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H11%val'
        CALL SPACE_resize_array( nfactors%H11%ne, nfactors%H11%val,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H21%row'
        CALL SPACE_resize_array( nfactors%H21%ne, nfactors%H21%row,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H21%col'
        CALL SPACE_resize_array( nfactors%H21%ne, nfactors%H21%col,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H21%val'
        CALL SPACE_resize_array( nfactors%H21%ne, nfactors%H21%val,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        lptr = nfactors%n_r + 1
        array_name = 'sbls: nfactors%H21%ptr'
        CALL SPACE_resize_array( lptr, nfactors%H21%ptr,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H22%row'
        CALL SPACE_resize_array( nfactors%H22%ne, nfactors%H22%row,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H22%col'
        CALL SPACE_resize_array( nfactors%H22%ne, nfactors%H22%col,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%H22%val'
        CALL SPACE_resize_array( nfactors%H22%ne, nfactors%H22%val,            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        lptr = nfactors%n_r + 1
        array_name = 'sbls: nfactors%H22%ptr'
        CALL SPACE_resize_array( lptr, nfactors%H22%ptr,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  Now fill H_11, H_21 and H_22

        nfactors%H11%ne = 0
        nfactors%H21%ne = 0
        nfactors%H22%ne = 0
        SELECT CASE ( SMT_get( H%type ) )
        CASE ( 'DIAGONAL' )
          DO l = 1, n
            i = nfactors%A_COLS_order( l )
            j = nfactors%A_COLS_order( l )
            IF ( i > nfactors%rank_a ) THEN
              nfactors%H22%ne = nfactors%H22%ne + 1
              nfactors%H22%row( nfactors%H22%ne ) = i - nfactors%rank_a
              nfactors%H22%col( nfactors%H22%ne ) = j - nfactors%rank_a
              nfactors%H22%val( nfactors%H22%ne ) = H%val( l )
            ELSE
              nfactors%H11%ne = nfactors%H11%ne + 1
              nfactors%H21%row( nfactors%H21%ne ) = i
              nfactors%H21%col( nfactors%H21%ne ) = j
              nfactors%H21%val( nfactors%H21%ne ) = H%val( l )
            END IF
          END DO
        CASE ( 'SCALED_IDENTITY' )
          DO l = 1, n
            i = nfactors%A_COLS_order( l )
            j = nfactors%A_COLS_order( l )
            IF ( i > nfactors%rank_a ) THEN
              nfactors%H22%ne = nfactors%H22%ne + 1
              nfactors%H22%row( nfactors%H22%ne ) = i - nfactors%rank_a
              nfactors%H22%col( nfactors%H22%ne ) = j - nfactors%rank_a
              nfactors%H22%val( nfactors%H22%ne ) = H%val( 1 )
            ELSE
              nfactors%H11%ne = nfactors%H11%ne + 1
              nfactors%H21%row( nfactors%H21%ne ) = i
              nfactors%H21%col( nfactors%H21%ne ) = j
              nfactors%H21%val( nfactors%H21%ne ) = H%val( 1 )
            END IF
          END DO
        CASE ( 'IDENTITY' )
          DO l = 1, n
            i = nfactors%A_COLS_order( l )
            j = nfactors%A_COLS_order( l )
            IF ( i > nfactors%rank_a ) THEN
              nfactors%H22%ne = nfactors%H22%ne + 1
              nfactors%H22%row( nfactors%H22%ne ) = i - nfactors%rank_a
              nfactors%H22%col( nfactors%H22%ne ) = j - nfactors%rank_a
              nfactors%H22%val( nfactors%H22%ne ) = one
            ELSE
              nfactors%H11%ne = nfactors%H11%ne + 1
              nfactors%H21%row( nfactors%H21%ne ) = i
              nfactors%H21%col( nfactors%H21%ne ) = j
              nfactors%H21%val( nfactors%H21%ne ) = one
            END IF
          END DO
        CASE ( 'DENSE' )
          l = 0
          DO ii = 1, n
            i = nfactors%A_COLS_order( ii )
            DO jj = 1, ii
              j = nfactors%A_COLS_order( jj )
              l = l + 1
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO ii = 1, n
            i = nfactors%A_COLS_order( ii )
            DO l = H%ptr( ii ), H%ptr( ii + 1 ) - 1
              j = nfactors%A_COLS_order( H%col( l ) )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, H%ne
            i = nfactors%A_COLS_order( H%row( l ) )
            j = nfactors%A_COLS_order( H%col( l ) )
            IF ( i > nfactors%rank_a .AND. j > nfactors%rank_a ) THEN
              nfactors%H22%ne = nfactors%H22%ne + 1
              nfactors%H22%row( nfactors%H22%ne ) = i - nfactors%rank_a
              nfactors%H22%col( nfactors%H22%ne ) = j - nfactors%rank_a
              nfactors%H22%val( nfactors%H22%ne ) = H%val( l )
              IF ( i /= j ) THEN
                nfactors%H22%ne = nfactors%H22%ne + 1
                nfactors%H22%row( nfactors%H22%ne ) = j - nfactors%rank_a
                nfactors%H22%col( nfactors%H22%ne ) = i - nfactors%rank_a
                nfactors%H22%val( nfactors%H22%ne ) = H%val( l )
              END IF
            ELSE IF ( i <= nfactors%rank_a .AND. j <= nfactors%rank_a ) THEN
              nfactors%H11%ne = nfactors%H11%ne + 1
              nfactors%H11%row( nfactors%H11%ne ) = i
              nfactors%H11%col( nfactors%H11%ne ) = j
              nfactors%H11%val( nfactors%H11%ne ) = H%val( l )
            ELSE
              nfactors%H21%ne = nfactors%H21%ne + 1
              IF ( i > nfactors%rank_a ) THEN
                nfactors%H21%row( nfactors%H21%ne ) = i - nfactors%rank_a
                nfactors%H21%col( nfactors%H21%ne ) = j
              ELSE
                nfactors%H21%row( nfactors%H21%ne ) = j - nfactors%rank_a
                nfactors%H21%col( nfactors%H21%ne ) = i
              END IF
              nfactors%H21%val( nfactors%H21%ne ) = H%val( l )
            END IF
          END DO
        END SELECT

        IF ( printd ) THEN
          WRITE( 6, "( ' H11: m, n, nnz ', 3I4 )" )                            &
            nfactors%rank_a, nfactors%rank_a, nfactors%H11%ne
          WRITE( 6, "( 3 ( 2I7, ES12.4 ) )" )                                  &
             ( nfactors%H11%row( i ), nfactors%H11%col( i ),                   &
               nfactors%H11%val( i ), i = 1, nfactors%H11%ne )
        END IF

!  Finally. reorder H21 and H22 so that they are stored by rows

        IF ( nfactors%H21%ne > 0 ) THEN

          liw = MAX( nfactors%n_r, nfactors%rank_a ) + 1
          array_name = 'sbls: nfactors%IW'
          CALL SPACE_resize_array( liw, nfactors%IW,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          nfactors%A2%col( : nfactors%A2%ne ) =                                &
            nfactors%A2%col( : nfactors%A2%ne ) - nfactors%rank_a

          CALL SORT_reorder_by_rows(                                           &
            nfactors%n_r, nfactors%rank_a, nfactors%H21%ne, nfactors%H21%row,  &
            nfactors%H21%col, nfactors%H21%ne, nfactors%H21%val,               &
            nfactors%H21%ptr, lptr, nfactors%IW, liw, control%error,           &
            control%out, inform%sort_status )
          IF ( inform%sort_status /= 0 ) THEN
            IF ( printe ) WRITE( error,                                        &
         "( A, ' ULS: sort of H21, info = ', I0 )" )  prefix, inform%sort_status
            IF ( inform%sort_status > 0 ) THEN
              inform%status = GALAHAD_error_sort ; RETURN
            END IF
          END IF

          nfactors%H21%ne = nfactors%H21%ptr( nfactors%n_r + 1 ) - 1
          nfactors%H21%row( : nfactors%H21%ne ) =                              &
            nfactors%H21%row( : nfactors%H21%ne ) + nfactors%rank_a

          IF ( printd ) THEN
            WRITE( 6, "( ' H21: m, n, nnz ', 3I4 )" )                          &
              nfactors%n_r, nfactors%rank_a, nfactors%H21%ne
            WRITE( 6, "( 3 ( 2I7, ES12.4 ) )" )                                &
               ( nfactors%H21%row( i ), nfactors%H21%col( i ),                 &
                 nfactors%H21%val( i ), i = 1, nfactors%H21%ne )
          END IF

        ELSE
          liw = nfactors%n_r + 1
          array_name = 'sbls: nfactors%IW'
          CALL SPACE_resize_array( liw, nfactors%IW,                           &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          nfactors%H21%ptr( : nfactors%n_r + 1 ) = 0
        END IF

        IF ( nfactors%H22%ne > 0 ) THEN
          CALL SORT_reorder_by_rows(                                           &
            nfactors%n_r, nfactors%n_r, nfactors%H22%ne, nfactors%H22%row,     &
            nfactors%H22%col, nfactors%H22%ne, nfactors%H22%val,               &
            nfactors%H22%ptr, lptr, nfactors%IW, liw, control%error,           &
            control%out, inform%sort_status )
          IF ( inform%sort_status /= 0 ) THEN
            IF ( printe ) WRITE( error,                                        &
         "( A, ' ULS: sort of H22, info = ', I0 )" )  prefix, inform%sort_status
            IF ( inform%sort_status > 0 ) THEN
              inform%status = GALAHAD_error_sort ; RETURN
            END IF
          END IF

          nfactors%H22%ne = nfactors%H22%ptr( nfactors%n_r + 1 ) - 1
          nfactors%H22%row( : nfactors%H22%ne ) =                              &
            nfactors%H22%row( : nfactors%H22%ne ) + nfactors%rank_a
          nfactors%H22%col( : nfactors%H22%ne ) =                              &
            nfactors%H22%col( : nfactors%H22%ne ) + nfactors%rank_a

          IF ( printd ) THEN
            WRITE( 6, "( ' H22: m, n, nnz ', 3I4 )" )                          &
              nfactors%n_r, nfactors%n_r, nfactors%H22%ne
            WRITE( 6, "( 3 ( 2I7, ES12.4 ) )" )                                &
               ( nfactors%H22%row( i ), nfactors%H22%col( i ),                 &
                 nfactors%H22%val( i ), i = 1, nfactors%H22%ne )
          END IF
        ELSE
          nfactors%H22%ptr( : nfactors%n_r + 1 ) = 0
        END IF
      END IF

!  Factorize the preconditioner

      SELECT CASE( inform%preconditioner )

      CASE( 2 )

!  Form the reduced Hessian
!    R = ( - A_2^T A_1^-T  I ) ( H_11  H_21^T ) ( - A_1^-1 A_2 )
!                              ( H_21   H_22  ) (        I     )
!  column by column

! Allocate the space to store workspace arrays v and w

        array_name = 'sbls: nfactors%V'
        CALL SPACE_resize_array( nfactors%rank_a, nfactors%V,                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%W'
        CALL SPACE_resize_array( nfactors%rank_a, nfactors%W,                  &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%R'
        CALL SPACE_resize_array( nfactors%n_r, nfactors%n_r, nfactors%R,       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%R_factors'
        CALL SPACE_resize_array( nfactors%n_r, nfactors%n_r,                   &
           nfactors%R_factors,                                                 &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  Loop over the columns

        DO k = 1, nfactors%n_r
          IF ( printd ) WRITE( 6, "( /, ' column ', I0 )" ) k

!  1. First form - A_2 e_k - > v

          nfactors%V = zero
          DO l = nfactors%A2%ptr( k ), nfactors%A2%ptr( k + 1 ) - 1
            i = nfactors%A2%row( l )
            nfactors%V( i ) = nfactors%V( i ) - nfactors%A2%val( l )
          END DO
          IF ( printd ) WRITE( 6, "( ' v ', ( 4ES12.4 ) )" ) nfactors%V

!  2. Now form A_1^-1 v -> w

          CALL ULS_SOLVE( nfactors%A1, nfactors%V( : nfactors%rank_a ),        &
                          nfactors%W( : nfactors%rank_a ), nfactors%A1_data,   &
                          nfactors%A1_control, inform%ULS_inform, .FALSE. )
          inform%uls_solve_status = inform%ULS_inform%status
          IF ( inform%uls_solve_status < 0 ) THEN
            IF ( printi ) WRITE( out, "( A, ' column ', I0, ' phase 2 ' )" )   &
              prefix, k
            inform%status = GALAHAD_error_uls_solve ; RETURN
          END IF
          IF ( printd ) WRITE( 6, "( ' w ', ( 4ES12.4 ) )" ) nfactors%W

!  3. Next form H_21 w + H_22 e_k -> r

          nfactors%R( : , k ) = zero

          DO l = nfactors%H22%ptr( k ), nfactors%H22%ptr( k + 1 ) - 1
            i = nfactors%H22%col( l ) - nfactors%rank_a
            nfactors%R( i, k ) = nfactors%R( i, k ) + nfactors%H22%val( l )
          END DO
          IF ( printd ) WRITE( 6, "( ' r ', ( 4ES12.4 ) )" ) nfactors%R( : , k )

          DO l = 1, nfactors%H21%ne
            i = nfactors%H21%row( l ) - nfactors%rank_a
            nfactors%R( i, k ) = nfactors%R( i, k ) +                          &
              nfactors%H21%val( l ) * nfactors%W( nfactors%H21%col( l ) )
          END DO
          IF ( printd ) WRITE( 6, "( ' r ', ( 4ES12.4 ) )" ) nfactors%R( : , k )

!  4. Now form the product H_11 w + H_21^T e_k -> v

          DO l = nfactors%A2%ptr( k ), nfactors%A2%ptr( k + 1 ) - 1
            nfactors%V( nfactors%A2%row( l ) ) = zero
          END DO
          IF ( printd ) WRITE( 6, "( ' v ', ( 4ES12.4 ) )" ) nfactors%V

          DO l = nfactors%H21%ptr( k ), nfactors%H21%ptr( k + 1 ) - 1
            i = nfactors%H21%col( l )
            nfactors%V( i ) = nfactors%V( i ) + nfactors%H21%val( l )
          END DO
          IF ( printd ) WRITE( 6, "( ' v ', ( 4ES12.4 ) )" ) nfactors%V

          DO l = 1, nfactors%H11%ne
            i = nfactors%H11%row( l )
            j = nfactors%H11%col( l )
            nfactors%V( i ) =                                                  &
              nfactors%V( i ) + nfactors%H11%val( l ) * nfactors%W( j )
            IF ( i /= j ) nfactors%V( j ) =                                    &
              nfactors%V( j ) + nfactors%H11%val( l ) * nfactors%W( i )
          END DO
          IF ( printd ) WRITE( 6, "( ' v ', ( 4ES12.4 ) )" ) nfactors%V

!  5. Next form w = A_1^-T v

          CALL ULS_SOLVE( nfactors%A1, nfactors%V( : nfactors%rank_a ),        &
                          nfactors%W( : nfactors%rank_a ), nfactors%A1_data,   &
                          nfactors%A1_control, inform%ULS_inform, .TRUE. )
          inform%uls_solve_status = inform%ULS_inform%status
          IF ( inform%uls_solve_status < 0 ) THEN
              IF ( printi ) WRITE( out, "( A, ' column ', I0, ' phase 5 ' )" ) &
                prefix, k
            inform%status = GALAHAD_error_uls_solve ; RETURN
          END IF
          IF ( printd ) WRITE( 6, "( ' w ', ( 4ES12.4 ) )" ) nfactors%W

!  6. Finally form r - A_2^T w -> r

          DO l = 1, nfactors%A2%ne
            i = nfactors%A2%col( l ) - nfactors%rank_a
            nfactors%R( i, k ) = nfactors%R( i, k ) -                          &
              nfactors%A2%val( l ) * nfactors%W( nfactors%A2%row( l ) )
          END DO
          IF ( printd ) WRITE( 6, "( ' r ', ( 4ES12.4 ) )" ) nfactors%R( : , k )

        END DO

        IF ( printd ) THEN
          WRITE( 6, "( ' R: ' )" )
          DO k = 1, nfactors%n_r
            WRITE( 6, "( I6, 6X, 5ES12.4, /, ( 6ES12.4 ) )" )                 &
              k, nfactors%R( : , k )
           END DO
        END IF

        val = zero
        DO i = 1, nfactors%n_r
          DO j = i, nfactors%n_r
            val = MAX( val, ABS( nfactors%R( i, j ) - nfactors%R( j, i ) ) )
          END DO
        END DO
        IF ( printi ) WRITE(6,"(A,' max asymmetry of R ', ES12.4)" ) prefix, val

      CASE DEFAULT

!  Anything else

        IF ( printi ) WRITE( out,                                              &
          "( A, ' No option control%preconditioner = ', I8, ' at present' )" ) &
             prefix, inform%preconditioner
        inform%status = GALAHAD_error_unknown_precond ; RETURN

      END SELECT

!  Now factorize R

      IF ( dense ) THEN
        nfactors%R_factors = nfactors%R
        CALL POTRF( 'L', nfactors%n_r, nfactors%R_factors, nfactors%n_r,       &
                    potrf_info )
      ELSE
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Sparse reduced Hessian option not implemented at present' )")&
             prefix
        inform%status = GALAHAD_not_yet_implemented ; RETURN
      END IF

      CALL CPU_TIME( time_end )
      IF ( printi ) WRITE( out,                                                &
        "( A, ' time to form and factorize null-space preconditioner ', F6.2)")&
        prefix, time_end - time_start

      inform%status = GALAHAD_ok
      RETURN

!  End of  subroutine SBLS_form_n_factorize_nullspace

      END SUBROUTINE SBLS_form_n_factorize_nullspace

!-*-*-   S B L S _ S O L V E _ N U L L _ S P A C E   S U B R O U T I N E   -*-*-

      SUBROUTINE SBLS_solve_null_space( nfactors, control, inform, SOL )

!  To solve

!    Kz = ( P  A^T ) z = b
!         ( A   0  )

!   (i) transform b to c = IP b
!   (ii) solve perm(K) w = c
!   (iii) recover z = IP^T w

!  where IP = (IC   0 )
!             ( 0  IR )

!  and the permutations IR and IC are such that
!  A = IR ( A1  A2 ) IC^T and the "basis" matrix A1 is nonsingular
!         (  0   0 )
!  This induces a re-ordering IC^T P IC of P

!  Iterative refinement may be used

!  Dummy arguments

      TYPE ( SBLS_null_space_factors_type ), INTENT( INOUT ) :: nfactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ),                                     &
                          DIMENSION( nfactors%n + nfactors%m ) :: SOL

!  Local variables

      INTEGER :: i, iter, j, l, rank_a, k_n, n, out, potrs_info
      INTEGER :: start_1, end_1, start_2, end_2, start_3, end_3
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      out = control%out
      printi = control%print_level >= 4

      n =  nfactors%n
      k_n = nfactors%k_n

!  Allocate workspace arrays

      IF ( k_n /= nfactors%len_sol_workspace ) THEN
        array_name = 'sbls: nfactors%RHS'
        CALL SPACE_resize_array( k_n, nfactors%RHS,                            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%SOL_current'
        CALL SPACE_resize_array( k_n, nfactors%SOL_current,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%SOL_perm'
        CALL SPACE_resize_array( k_n, 1, nfactors%SOL_perm,                    &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: nfactors%RHS_orig'
        CALL SPACE_resize_array( k_n, nfactors%RHS_orig,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        nfactors%len_sol_workspace = k_n
      END IF

      rank_a = nfactors%rank_a

!  Partition the variables r and v

      start_1 = 1 ; end_1 = rank_a
      start_2 = rank_a + 1 ; end_2 = n
      start_3 = n + 1 ; end_3 = n + rank_a

!  Permute the variables

      DO i = 1, n
        nfactors%RHS_orig( i ) = SOL( nfactors%A_COLS_basic( i ) )
      END DO

      DO i = 1, rank_a
        nfactors%RHS_orig( n + i ) = SOL( n + nfactors%A_ROWS_basic( i ) )
      END DO

!  ------------------------------------------
!  Solve the system with iterative refinement
!  ------------------------------------------

      DO iter = 0, control%itref_max

!    ********************
!    *  RESIDUAL STAGE  *
!    ********************

!  Compute the current residual

        IF ( iter > 0 ) THEN

          nfactors%RHS( : k_n ) = nfactors%RHS_orig
          nfactors%SOL_perm = zero

!  Residuals are

!  ( r_1 ) - ( H_11  H_21^T  A_1^T ) ( s_1 )
!  ( r_2 )   ( H_21    W     A_2^T ) ( s_2 )
!  ( r_3 )   ( A_1    A_2      0   ) ( s_3 )
!
!  for given W

!  Terms involving A_1 and A_1^T

          DO l = 1, nfactors%A1%ne
            i = nfactors%A1%row( l )
            j = nfactors%A1%col( l )
            nfactors%RHS( n + i ) = nfactors%RHS( n + i ) -                    &
              nfactors%A1%val( l ) * nfactors%SOL_current( j )
            nfactors%RHS( j ) = nfactors%RHS( j ) -                            &
              nfactors%A1%val( l ) * nfactors%SOL_current( n + i )
          END DO

!  Terms involving A_2 and A_2^T

          DO l = 1, nfactors%A2%ne
            i = nfactors%A2%row( l )
            j = nfactors%A2%col( l )
            nfactors%RHS( n + i ) = nfactors%RHS( n + i ) -                    &
              nfactors%A2%val( l ) * nfactors%SOL_current( j )
            nfactors%RHS( j ) = nfactors%RHS( j ) -                            &
              nfactors%A2%val( l ) * nfactors%SOL_current( n + i )
          END DO

!  Case: W = H_22

          IF ( inform%preconditioner == 2 ) THEN

            DO l = 1, nfactors%H11%ne
              i = nfactors%H11%row( l )
              j = nfactors%H11%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                          &
                nfactors%H11%val( l ) * nfactors%SOL_current( j )
              IF ( i /= j ) nfactors%RHS( j ) = nfactors%RHS( j ) -            &
                nfactors%H11%val( l ) * nfactors%SOL_current( i )
            END DO
            DO l = 1, nfactors%H21%ne
              i = nfactors%H21%row( l )
              j = nfactors%H21%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                          &
                nfactors%H21%val( l ) * nfactors%SOL_current( j )
              nfactors%RHS( j ) = nfactors%RHS( j ) -                          &
                nfactors%H21%val( l ) * nfactors%SOL_current( i )
            END DO
            DO l = 1, nfactors%H22%ne
              i = nfactors%H22%row( l )
              j = nfactors%H22%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                          &
                nfactors%H22%val( l ) * nfactors%SOL_current( j )
            END DO

!  Terms involving H

          ELSE

!  Case: W = R + ( - A_2^T A_1^-T  I ) ( H_11  H_21^T ) ( - A_1^-1 A_2 )
!                                      ( H_21   H_22  ) (        I     )

!  Terms involving H_11 and H_12

            DO l = 1, nfactors%H11%ne
              i = nfactors%H11%row( l )
              j = nfactors%H11%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                          &
                nfactors%H11%val( l ) * nfactors%SOL_current( j )
              IF ( i /= j ) nfactors%RHS( j ) = nfactors%RHS( j ) -            &
                nfactors%H11%val( l ) * nfactors%SOL_current( i )
            END DO
            DO l = 1, nfactors%H21%ne
              i = nfactors%H21%row( l )
              j = nfactors%H21%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                          &
                nfactors%H21%val( l ) * nfactors%SOL_current( j )
              nfactors%RHS( j ) = nfactors%RHS( j ) -                          &
                nfactors%H21%val( l ) * nfactors%SOL_current( i )
            END DO

!  Terms involving W = R + ( H_21   A_2^T ) ( H_11  A_1 )^-1 ( H_21^T )
!                                            (  A_1   0  )    (  A_2   )

            IF ( printi ) WRITE( out,                                          &
              "( A, ' general residuals not implemented at present' )")        &
                 prefix
            inform%status = GALAHAD_not_yet_implemented ; RETURN
          END IF

          IF ( control%get_norm_residual )                                     &
            inform%norm_residual = MAXVAL( ABS( nfactors%RHS( : k_n ) ) )

!  No residual required

!         WRITE( 6, "( A, /, ( 5ES12.4 ) )" ) ' solution ',                    &
!           nfactors%SOL_current( : k_n )
        ELSE
          nfactors%RHS( : k_n ) = nfactors%RHS_orig
        END IF
!       WRITE( 6, "( A, /, ( 5ES12.4 ) )" ) ' residuals ',  nfactors%RHS( : k_n)
        IF ( printi ) WRITE( out, "( A, ' maximum residual = ', ES10.4 )" )   &
          prefix, MAXVAL( ABS( nfactors%RHS( : k_n ) ) )

!    *****************
!    *  SOLVE STAGE  *
!    *****************

!  1. Solve A_1 y_1 = r_3 (store y_1 in v)

        IF ( .NOT. control%affine ) THEN
          CALL ULS_SOLVE( nfactors%A1, nfactors%RHS( start_3 : end_3 ),        &
                          nfactors%V, nfactors%A1_data, nfactors%A1_control,   &
                          inform%ULS_inform, .FALSE. )
          inform%uls_solve_status = inform%ULS_inform%status
          IF ( inform%uls_solve_status < 0 ) THEN
            inform%status = GALAHAD_error_uls_solve ; RETURN
          END IF
        END IF

!  2. Solve A_1^T y_3 = r_1 - H_11 y_1
!     and
!  3. form r_2 - H_21 y_1 - A_2^T y_3

!  2a. Form r_1 - H_11 y_1 -> w

        nfactors%W( start_1 : end_1 ) = nfactors%RHS( start_1 : end_1 )
        IF ( .NOT. control%affine ) THEN
          DO l = 1, nfactors%H11%ne
            i = nfactors%H11%row( l )
            j = nfactors%H11%col( l )
            nfactors%W( i ) = nfactors%W( i ) -                                &
              nfactors%H11%val( l ) * nfactors%V( j )
            IF ( i /= j ) nfactors%W( j ) = nfactors%W( j ) -                  &
              nfactors%H11%val( l ) * nfactors%V( i )
          END DO

!  3a. Form r_2 - H_21 y_1 -> r_2

          DO l = 1, nfactors%H21%ne
            i = nfactors%H21%row( l )
            j = nfactors%H21%col( l )
            nfactors%RHS( i ) = nfactors%RHS( i ) -                            &
              nfactors%H21%val( l ) * nfactors%V( j )
          END DO
        END IF

!  2b. Solve A_1^T y_3 = w (store y_3 in v)

        CALL ULS_SOLVE( nfactors%A1, nfactors%W, nfactors%V, nfactors%A1_data, &
                        nfactors%A1_control, inform%ULS_inform, .TRUE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_uls_solve ; RETURN
        END IF

!  3b. Form r_2 - A_2^T y_3 -> r_2

        DO l = 1, nfactors%A2%ne
          j = nfactors%A2%col( l )
          nfactors%RHS( j ) = nfactors%RHS( j ) -                              &
            nfactors%A2%val( l ) * nfactors%V( nfactors%A2%row( l ) )
        END DO

!  4. Find S^-1 r_2 -> x_2

        nfactors%SOL_perm( start_2 : end_2, 1 ) =                              &
          nfactors%RHS( start_2 : end_2 )
        CALL POTRS( 'L', nfactors%n_r, 1, nfactors%R_factors, nfactors%n_r,    &
                    nfactors%SOL_perm( start_2 : end_2, : ), nfactors%n_r,     &
                    potrs_info )

!  5. Solve A_1 x_1 = r_3 - A^2 x_2

!  5a. Form r_3 - A^2 x_2 -> r_3

        DO l = 1, nfactors%A2%ne
          i = n + nfactors%A2%row( l )
          nfactors%RHS( i ) = nfactors%RHS( i ) -                              &
            nfactors%A2%val( l ) * nfactors%SOL_perm( nfactors%A2%col( l ), 1 )
        END DO

!  5b. Solve A_1 x_1 = r_3

        CALL ULS_SOLVE( nfactors%A1, nfactors%RHS( start_3 : end_3 ),          &
                        nfactors%SOL_perm( start_1 : end_1, 1 ),               &
                        nfactors%A1_data, nfactors%A1_control,                 &
                        inform%ULS_inform, .FALSE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_uls_solve ; RETURN
        END IF

!  6. Solve A_1^T x_3 = r_1 - H_11 x_1 - H_21^T x_2

!  6a. Form r_1 - H_11 x_1 - H_21^T x_2 -> r_1

        DO l = 1, nfactors%H11%ne
          i = nfactors%H11%row( l )
          j = nfactors%H11%col( l )
          nfactors%RHS( i ) = nfactors%RHS( i ) -                              &
            nfactors%H11%val( l ) * nfactors%SOL_perm( j, 1 )
          IF ( i /= j ) nfactors%RHS( j ) = nfactors%RHS( j ) -                &
            nfactors%H11%val( l ) * nfactors%SOL_perm( i, 1 )
        END DO
        DO l = 1, nfactors%H21%ne
          i = nfactors%H21%row( l )
          j = nfactors%H21%col( l )
          nfactors%RHS( j ) = nfactors%RHS( j ) -                              &
            nfactors%H21%val( l ) * nfactors%SOL_perm( i, 1 )
        END DO

!  6b. Solve A_1^T x_3 = r_1

        CALL ULS_SOLVE( nfactors%A1, nfactors%RHS( start_1 : end_1 ),          &
                        nfactors%SOL_perm( start_3 : end_3, 1 ),               &
                        nfactors%A1_data, nfactors%A1_control,                 &
                        inform%ULS_inform, .TRUE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_uls_solve ; RETURN
        END IF

!    ******************
!    *  UPDATE STAGE  *
!    ******************

!  Update the solution

        IF ( iter > 0 ) nfactors%SOL_perm( : , 1 ) =                           &
          nfactors%SOL_current + nfactors%SOL_perm( : , 1 )
        IF ( iter < control%itref_max )                                        &
          nfactors%SOL_current = nfactors%SOL_perm( : , 1 )
!       WRITE( 6, "( A, ( 5ES12.4 ) )" ) ' solution ',  nfactors%SOL_current

!  ---------------------------
!  End of iterative refinement
!  ---------------------------

      END DO

!  Unpermute the variables

      DO i = 1, n
        SOL( nfactors%A_COLS_basic( i ) ) = nfactors%SOL_perm( i, 1 )
      END DO

      IF ( rank_a < nfactors%m ) SOL( n + 1 : n + nfactors%m ) = zero
      DO i = 1, rank_a
        SOL( n + nfactors%A_ROWS_basic( i ) ) = nfactors%SOL_perm( n + i, 1 )
      END DO

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_solve_null_space

      END SUBROUTINE SBLS_solve_null_space

!-*-*-*-*-*-*-*-*-*-   S B L S _ c o n d  S U B R O U T I N E  -*-*-*-*-*-*-*-

      SUBROUTINE SBLS_cond( data, out, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the smallest and largest eigenvalues of the block diagonal
!  part of the explicit factors
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: out
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, nroots, n, rank
      REAL ( KIND = wp ) :: root1, root2, dmax, dmin
      LOGICAL ::  twobytwo
      INTEGER :: P( data%efactors%K%n )
      REAL ( KIND = wp ) :: D( 2, data%efactors%K%n )

      n = data%efactors%K%n
      IF ( n <= 0 ) RETURN
      rank = data%efactors%rank_k
      CALL SLS_enquire( data%efactors%K_data, inform%SLS_inform,               &
                        PIVOTS = P, D = D )

      twobytwo = .FALSE.
      dmax = zero
      dmin = HUGE( one )
      DO i = 1, rank
        IF ( twobytwo ) THEN
          twobytwo = .FALSE.
          CYCLE
        END IF
        IF ( i < rank ) THEN
          IF ( D( 2, i ) /= zero ) THEN
            twobytwo = .TRUE.
            CALL ROOTS_quadratic( D( 1, i ) * D( 1, i + 1 ) - D( 2, i ) ** 2,  &
              - D( 1, i ) - D( 1, i + 1 ), one, epsmch, nroots, root1, root2,  &
              roots_debug )
            dmax = MAX( ABS( root1 ), ABS( root2 ), dmax )
            dmin = MIN( ABS( root1 ), ABS( root2 ), dmin )
          ELSE
            dmax = MAX( ABS( D( 1, i ) ), dmax )
            dmin = MIN( ABS( D( 1, i ) ), dmin )
          END IF
        ELSE
          dmax = MAX( ABS( D( 1, i ) ), dmax )
          dmin = MIN( ABS( D( 1, i ) ), dmin )
        END IF
      END DO

      IF ( n > rank ) dmax = HUGE( one )

      IF ( dmin == zero .OR. dmax == zero ) THEN
        WRITE( out, "( ' 1/ smallest,largest eigenvalues =',  2ES12.4 )" )     &
          dmin, dmax
      ELSE
        WRITE( out, "( ' smallest,largest eigenvalues =',  2ES12.4 )" )        &
          one / dmax, one / dmin
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_cond

      END SUBROUTINE SBLS_cond

!-*-*-*-*-*-*-*-*-*-   S B L S _ e i g s  S U B R O U T I N E  -*-*-*-*-*-*-*-

      SUBROUTINE SBLS_eigs( data, out, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute the eigenvalues of the block diagonal part of the explicit factors
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: out
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, nroots, n, rank
      REAL ( KIND = wp ) :: root1, root2, dmax, dmin
      LOGICAL ::  twobytwo
!     INTEGER :: P( data%efactors%K%n )
      REAL ( KIND = wp ) :: D( 2, data%efactors%K%n )

      n = data%efactors%K%n
      IF ( n <= 0 ) RETURN
      rank = data%efactors%rank_k
      CALL SLS_enquire( data%efactors%K_data, inform%SLS_inform, D = D )

      twobytwo = .FALSE.
      dmax = zero
      dmin = HUGE( one )
      DO i = 1, rank
        IF ( twobytwo ) THEN
          twobytwo = .FALSE.
          CYCLE
        END IF
        IF ( i < rank ) THEN
          IF ( D( 2, i ) /= zero ) THEN
            twobytwo = .TRUE.
            CALL ROOTS_quadratic( D( 1, i ) * D( 1, i + 1 ) - D( 2, i ) ** 2,  &
              - D( 1, i ) - D( 1, i + 1 ), one, epsmch, nroots, root1, root2,  &
              roots_debug )
            D( 1, i ) = one / root1
            D( 1, i + 1 ) = one / root2
          ELSE
            IF ( D( 1, i ) /= zero ) D( 1, i ) = one / D( 1, i )
          END IF
        ELSE
          IF ( D( 1, i ) /= zero ) D( 1, i ) = one / D( 1, i )
        END IF
      END DO

      IF ( n > rank ) D( 1, rank + 1 : n ) = zero
      WRITE( out, "( ' eigenvalues = ', 4ES12.4, /, ( 3X, 5ES12.4 ) )" )       &
         D( 1, 1 : n )

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_eigs

      END SUBROUTINE SBLS_eigs

!-*-*-*-   S B L S _ F I N D _ A 1 _ A N D _ A 2   S U B R O U T I N E   -*-*-*-

      SUBROUTINE SBLS_find_A1_and_A2( m, n, a_ne, A, A1, A1_data,              &
                                      A1_control, A2, A_ROWS_basic,            &
                                      A_COLS_basic, A_ROWS_order, A_COLS_order,&
                                      rank_a, k_n, n_r, prefix, resize_prefix, &
                                      out, printi, control, inform,            &
                                      RHS, SOL )

!  Given a rectangular matrix A, find a "basic" set of rows and colums, that is
!  a non-singular submatrix A1 of maximal rank. Also set up the complement
!  matrix A2 of columns of A not in A1

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n, a_ne, out
      INTEGER, INTENT( OUT ) :: rank_a, n_r, k_n
      LOGICAL, INTENT( IN ) :: printi
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A1, A2
      TYPE ( ULS_data_type ), INTENT( INOUT ) :: A1_data
      TYPE ( ULS_control_type ), INTENT( INOUT ) :: A1_control
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic, A_COLS_basic
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_order, A_COLS_order
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS, SOL
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = * ), INTENT( IN ) :: resize_prefix
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, jj, l, dependent, nb
      LOGICAL :: find_basis_by_transpose
      CHARACTER ( LEN = 80 ) :: array_name

!     LOGICAL :: printd = .TRUE.
      LOGICAL :: printd = .FALSE.

      find_basis_by_transpose = control%find_basis_by_transpose

  100 CONTINUE

!  Find sets of basic rows and columns

      CALL SBLS_find_basis( m, n, a_ne, A, A1, A1_data, A1_control,            &
                            A_ROWS_basic, A_COLS_basic, rank_a,                &
                            find_basis_by_transpose, prefix,                   &
                            resize_prefix, out, printi, control, inform )

!     CALL CPU_TIME( t2 )
!     WRITE(6,"(' time to find basis ',F6.2)") t2 - t1
      IF ( inform%status /= GALAHAD_ok ) RETURN

      k_n = n + rank_a ; n_r = n - rank_a

!  Print out rank and identifying vectors

      IF ( out > 0 .AND. control%print_level >= 2 ) THEN
        WRITE( out, "( /, A, ' First-pass factorization ' )" ) prefix
        WRITE( out, "( A, A, I0, 1X, I0, 1X, I0 )" ) prefix, ' m, rank, n = ', &
          m, rank_a, n
        WRITE( out, "( A, A, I0, 1X, I0 )" ) prefix, ' A_ne, factors_ne = ',   &
           a_ne, inform%SLS_inform%entries_in_factors
!       WRITE( out, "( A, A, /, ( 10I7 ) )" ) prefix,                          &
!         ' rows =', ( A_ROWS_basic( : rank_a ) )
!       WRITE( out, "( A, A, /, ( 10I7) )" ) prefix,                           &
!         ' cols =', ( A_COLS_basic( : rank_a ) )
      END IF

! ---------------
! factorize basis
! ---------------

!  Make a copy of the "basis" matrix

      array_name = resize_prefix // 'A_ROWS_order'
      CALL SPACE_resize_array( m, A_ROWS_order,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A_COLS_order'
      CALL SPACE_resize_array( n, A_COLS_order,                                &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      A_COLS_order = 0
      DO i = 1, rank_a
        A_COLS_order( A_COLS_basic( i ) ) = i
      END DO

      A_ROWS_order = 0
      DO i = 1, rank_a
        A_ROWS_order( A_ROWS_basic( i ) ) = i
      END DO

!  Ensure that the full-rank rows appear before the dependent ones

      rank_a = 0
      dependent = m + 1
      DO i = 1, m
        IF ( A_ROWS_order( i ) > 0 ) THEN
          rank_a = rank_a + 1
          A_ROWS_order( i ) = rank_a
          A_ROWS_basic( rank_a ) = i
        ELSE
          dependent = dependent - 1
          A_ROWS_order( i ) = dependent
          A_ROWS_basic( dependent ) = i
        END IF
      END DO

!  Mark the non-basic columns

      nb = 0
      DO i = 1, n
        IF ( A_COLS_order( i ) <= 0 ) THEN
          nb = nb + 1
          A_COLS_order( i ) = rank_a + nb
          A_COLS_basic( rank_a + nb ) = i
        END IF
      END DO

!  Determine the space required for A1 and A2

      A1%ne = 0
      A2%ne = 0

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        DO i = 1, m
          IF ( A_ROWS_order( i ) > rank_a ) CYCLE
          DO j = 1, n
            IF ( A_COLS_order( j ) <= rank_a ) THEN
              A1%ne = A1%ne + 1
            ELSE
              A2%ne = A2%ne + 1
            END IF
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          IF ( A_ROWS_order( i ) > rank_a ) CYCLE
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            IF ( A_COLS_order( A%col( l ) ) <= rank_a ) THEN
              A1%ne = A1%ne + 1
            ELSE
              A2%ne = A2%ne + 1
            END IF
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          IF ( A_ROWS_order( A%row( l ) ) > rank_a ) CYCLE
          IF ( A_COLS_order( A%col( l ) ) <= rank_a ) THEN
            A1%ne = A1%ne + 1
          ELSE
            A2%ne = A2%ne + 1
          END IF
        END DO
      END SELECT

!  Reallocate the space to store the basis matrix, A1, if necessary

      array_name = resize_prefix // 'A1%row'
      CALL SPACE_resize_array( A1%ne, A1%row,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A1%col'
      CALL SPACE_resize_array( A1%ne, A1%col,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A1%val'
      CALL SPACE_resize_array( A1%ne, A1%val,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  Assemble A1

      A1%m = rank_a ; A1%n = rank_a ; A1%ne = 0
      CALL SMT_put( A1%type, 'COORDINATE', i )

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        l = 0
        DO i = 1, m
          ii = A_ROWS_order( i )
          IF ( ii > rank_a ) THEN
            l = l + n
            CYCLE
          END IF
          DO j = 1, n
            l = l + 1
            jj = A_COLS_order( j )
            IF ( jj > rank_a ) CYCLE
            A1%ne = A1%ne + 1
            A1%row( A1%ne ) = ii
            A1%col( A1%ne ) = jj
            A1%val( A1%ne ) = A%val( l )
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          ii = A_ROWS_order( i )
          IF ( ii > rank_a ) CYCLE
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            jj = A_COLS_order( A%col( l ) )
            IF ( jj > rank_a ) CYCLE
            A1%ne = A1%ne + 1
            A1%row( A1%ne ) = ii
            A1%col( A1%ne ) = jj
            A1%val( A1%ne ) = A%val( l )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          ii = A_ROWS_order( A%row( l ) )
          IF ( ii > rank_a ) CYCLE
          jj = A_COLS_order( A%col( l ) )
          IF ( jj > rank_a ) CYCLE
          A1%ne = A1%ne + 1
          A1%row( A1%ne ) = ii
          A1%col( A1%ne ) = jj
          A1%val( A1%ne ) = A%val( l )
        END DO
      END SELECT

      IF ( printd ) THEN
        WRITE( 6, "( ' A1: m, n, nnz ', 3I4 )" ) A1%m, A1%n, A1%ne
        WRITE( 6, "( 3 ( 2I7, ES12.4 ) )" )                                    &
          ( A1%row( i ), A1%col( i ), A1%val( i ), i = 1, A1%ne )
      END IF

!     WRITE( 29, "( 3( 2I4, ES12.4 ) ) " ) ( A1%row( i ),                      &
!        A1%col( i ), A1%val( i ), i = 1, A1%ne )

!  Factorize A1

      A1_control = control%ULS_control
      CALL ULS_initialize( control%unsymmetric_linear_solver, A1_data,         &
                           A1_control, inform%ULS_inform )
      IF ( control%print_level <= 0 ) THEN
        A1_control%print_level = 0
        A1_control%error = - 1 ; A1_control%out = - 1 ; A1_control%warning = - 1
      ELSE
        A1_control%print_level = control%print_level - 1
        A1_control%error = control%error
        A1_control%out = control%out ; A1_control%warning = control%out
      END IF

      CALL ULS_factorize( A1, A1_data, A1_control, inform%ULS_inform )
      inform%uls_factorize_status = inform%ULS_inform%status
      IF ( printi ) WRITE( out,                                                &
         "( A, ' ULS: factorization of A1 complete: status = ', I0 )" )        &
             prefix, inform%uls_factorize_status
      IF ( inform%uls_factorize_status < 0 ) THEN
         inform%status = GALAHAD_error_uls_factorization ; RETURN
      END IF
      IF ( inform%uls_inform%rank < MIN( m, n ) ) THEN
        inform%rank_def = .TRUE.
        inform%rank = inform%ULS_inform%rank
      END IF

      IF ( printi ) WRITE( out, "( A, ' A1 nnz(prec,factors)', 2( 1X, I0 ))")  &
        prefix, A1%ne, inform%ULS_inform%entries_in_factors

!  If required, check to see if the basis found is reasobale. Do this
!  by solving the equations A_1 x = A_1 e to check that x is a reasoble
!  approximation to e

      IF ( control%check_basis ) THEN
        array_name = resize_prefix // 'RHS'
        CALL SPACE_resize_array( k_n, RHS,                                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = resize_prefix // 'SOL'
        CALL SPACE_resize_array( k_n, SOL,                                     &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        RHS( : rank_a ) = zero
        DO l = 1, A1%ne
          RHS( A1%row( l ) ) = RHS( A1%row( l ) ) + A1%val( l )
        END DO

        CALL ULS_SOLVE( A1, RHS( : rank_a ), SOL( : rank_a ), A1_data,         &
                        A1_control, inform%ULS_inform, .FALSE. )
        inform%uls_solve_status = inform%ULS_inform%status
        IF ( inform%uls_solve_status < 0 ) THEN
          IF ( find_basis_by_transpose .EQV.                                   &
               control%find_basis_by_transpose ) THEN
            find_basis_by_transpose = .NOT. control%find_basis_by_transpose
            IF ( printi )                                                      &
              WRITE( out, "( A, ' basis unstable - recomputing ' )" ) prefix
            GO TO 100
          ELSE
            IF ( printi )                                                      &
              WRITE( out, "( A, ' error return - basis unstable ' )" ) prefix
            inform%status = GALAHAD_error_uls_solve ; RETURN
          END IF
        END IF
      END IF

! Allocate the space to store the non-basic matrix, A2

      array_name = resize_prefix // 'A2%row'
      CALL SPACE_resize_array( A2%ne, A2%row,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A2%col'
      CALL SPACE_resize_array( A2%ne, A2%col,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A2%val'
      CALL SPACE_resize_array( A2%ne, A2%val,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  Assemble A2

      A2%m = rank_a ; A2%n = rank_a ; A2%ne = 0

      SELECT CASE ( SMT_get( A%type ) )
      CASE ( 'DENSE' )
        l = 0
        DO i = 1, m
          ii = A_ROWS_order( i )
          IF ( ii > rank_a ) THEN
            l = l + n
            CYCLE
          END IF
          DO j = 1, n
            l = l + 1
            jj =  A_COLS_order( j )
            IF ( jj <= rank_a ) CYCLE
            A2%ne = A2%ne + 1
            A2%row( A2%ne ) = ii
            A2%col( A2%ne ) = jj
            A2%val( A2%ne ) = A%val( l )
          END DO
        END DO
      CASE ( 'SPARSE_BY_ROWS' )
        DO i = 1, m
          ii = A_ROWS_order( i )
          IF ( ii > rank_a ) CYCLE
          DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
            jj =  A_COLS_order( A%col( l ) )
            IF ( jj <= rank_a ) CYCLE
            A2%ne = A2%ne + 1
            A2%row( A2%ne ) = ii
            A2%col( A2%ne ) = jj
            A2%val( A2%ne ) = A%val( l )
          END DO
        END DO
      CASE ( 'COORDINATE' )
        DO l = 1, A%ne
          ii = A_ROWS_order( A%row( l ) )
          IF ( ii > rank_a ) CYCLE
          jj =  A_COLS_order( A%col( l ) )
          IF ( jj <= rank_a ) CYCLE
          A2%ne = A2%ne + 1
          A2%row( A2%ne ) = ii
          A2%col( A2%ne ) = jj
          A2%val( A2%ne ) = A%val( l )
        END DO
      END SELECT

      IF ( printd ) THEN
        WRITE( 6, "( ' A2: m, n, nnz ', 3I4 )" ) A2%m, A2%n, A2%ne
        WRITE( 6, "( 3 ( 2I7, ES12.4 ) )" )                                    &
          ( A2%row( i ), A2%col( i ), A2%val( i ), i = 1, A2%ne )
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_find_A1_and_A2

      END SUBROUTINE SBLS_find_A1_and_A2

!-*-*-*-*-*-   S B L S _ F I N D _ B A S I S   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE SBLS_find_basis( m, n, a_ne, A, A1, A1_data, A1_control,      &
                                  A_ROWS_basic, A_COLS_basic, rank_a,          &
                                  find_basis_by_transpose, prefix,             &
                                  resize_prefix, out, printi, control, inform )

!  Given a rectangular matrix A, find a "basic" set of rows and colums, that is
!  those that give a non-singular submatrix A1 of maximal rank

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n, a_ne, out
      INTEGER, INTENT( OUT ) :: rank_a
      LOGICAL, INTENT( IN ) :: printi, find_basis_by_transpose
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: A1
      TYPE ( ULS_data_type ), INTENT( INOUT ) :: A1_data
      TYPE ( ULS_control_type ), INTENT( INOUT ) :: A1_control
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic, A_COLS_basic
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = * ), INTENT( IN ) :: resize_prefix
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, l
      CHARACTER ( LEN = 80 ) :: array_name

!  Set up storage for A1. Initially A1 will be used to hold A

      A1%ne = a_ne

      array_name = resize_prefix // 'A1%row'
      CALL SPACE_resize_array( a_ne, A1%row,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A1%col'
      CALL SPACE_resize_array( a_ne, A1%col,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

      array_name = resize_prefix // 'A1%val'
      CALL SPACE_resize_array( a_ne, A1%val,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         deallocate_error_fatal = control%deallocate_error_fatal,              &
         exact_size = control%space_critical,                                  &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) RETURN

!  =====================
!  Investigate using A^T
!  =====================

      IF ( find_basis_by_transpose ) THEN
        A1%m = n ; A1%n = m
        CALL SMT_put( A1%type, 'COORDINATE', i )

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              A1%row( l ) = j
              A1%col( l ) = i
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              A1%row( l ) = A%col( l )
              A1%col( l ) = i
            END DO
          END DO
        CASE ( 'COORDINATE' )
          A1%row( : a_ne ) = A%col( : A%ne )
          A1%col( : a_ne ) = A%row( : A%ne )
        END SELECT
        A1%val( : a_ne ) = A%val( : a_ne )

! Initialize the structures

!       out = 6
!       WRITE( out, "( ' A: m, n, nnz ', 3I4 )" )                              &
!         A1%m, A1%n, A1%ne
!       WRITE( out, "( A, /, ( 10I7) )" ) ' rows =', ( A1%row )
!       WRITE( out, "( A, /, ( 10I7) )" ) ' cols =', ( A1%col )
!       WRITE( out, "( A, /, ( F7.2) )" ) ' vals =', ( A1%val )

! Analyse and factorize

!       CALL CPU_TIME( t3 )
        A1_control = control%ULS_control
        CALL ULS_initialize( control%unsymmetric_linear_solver, A1_data,       &
                             A1_control, inform%ULS_inform )
        IF ( control%print_level <= 0 ) THEN
          A1_control%print_level = 0
          A1_control%error = - 1 ; A1_control%out = - 1
          A1_control%warning = - 1
        ELSE
          A1_control%print_level = control%print_level - 1
          A1_control%error = control%error
          A1_control%out = control%out ; A1_control%warning = control%out
        END IF
        A1_control%relative_pivot_tolerance = control%pivot_tol_for_basis
        A1_control%pivot_search_limit = MAX( A1%m, A1%n ) + 1

        CALL ULS_factorize( A1, A1_data, A1_control, inform%ULS_inform )
!       CALL CPU_TIME( t2 )

        inform%uls_factorize_status = inform%ULS_inform%status
        IF ( printi ) WRITE( out,                                              &
           "( /, A, ' ULS: factorization of A  complete: status = ', I0 )" )   &
               prefix, inform%uls_factorize_status
        IF ( printi ) WRITE( out, "( A, ' A nnz(prec,factors)', 2( 1X, I0))" ) &
          prefix, A1%ne, inform%ULS_inform%entries_in_factors

        IF ( inform%uls_factorize_status < 0 ) THEN
           inform%status = GALAHAD_error_uls_factorization ; RETURN
        END IF
        IF ( inform%uls_inform%rank < MIN( m, n ) ) THEN
          inform%rank_def = .TRUE.
          inform%rank = inform%ULS_inform%rank
        END IF

        array_name = resize_prefix // 'A_ROWS_basic'
        CALL SPACE_resize_array( m, A_ROWS_basic,                              &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = resize_prefix // 'A_COLS_basic'
        CALL SPACE_resize_array( n, A_COLS_basic,                              &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  Determine the nonsingular submatrix of the factorization

        CALL ULS_enquire( A1_data, inform%ULS_inform,                          &
                          A_COLS_basic( : n ) , A_ROWS_basic( : m ) )
        rank_a = inform%ULS_inform%rank

!  ===================
!  Investigate using A
!  ===================

      ELSE
        A1%m = m ; A1%n = n
        CALL SMT_put( A1%type, 'COORDINATE', i )

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, m
            DO j = 1, n
              l = l + 1
              A1%row( l ) = i
              A1%col( l ) = j
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, m
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              A1%row( l ) = i
              A1%col( l ) = A%col( l )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          A1%row( : a_ne ) = A%row( : A%ne )
          A1%col( : a_ne ) = A%col( : A%ne )
        END SELECT
        A1%val( : a_ne ) = A%val( : a_ne )

! Analyse and factorize

!       CALL CPU_TIME( t3 )
        A1_control = control%ULS_control
        CALL ULS_initialize( control%unsymmetric_linear_solver, A1_data,       &
                             A1_control, inform%ULS_inform )
        IF ( control%print_level <= 0 ) THEN
          A1_control%print_level = 0
          A1_control%error = - 1 ; A1_control%out = - 1
          A1_control%warning = - 1
        ELSE
          A1_control%print_level = control%print_level - 1
          A1_control%error = control%error
          A1_control%out = control%out ; A1_control%warning = control%out
        END IF
        A1_control%relative_pivot_tolerance = control%pivot_tol_for_basis

        CALL ULS_factorize( A1, A1_data, A1_control, inform%ULS_inform )
!       CALL CPU_TIME( t2 )

        IF ( printi ) WRITE( out, "( A, ' A nnz(prec,factors)', 2( 1X, I0 ))") &
          prefix, A1%ne, inform%ULS_inform%entries_in_factors

        inform%uls_factorize_status = inform%ULS_inform%status
        IF ( printi ) WRITE( out,                                              &
           "( /, A, ' ULS: factorization of A1  complete: status = ', I0 )" )  &
               prefix, inform%uls_factorize_status
        IF ( inform%uls_factorize_status < 0 ) THEN
           inform%status = GALAHAD_error_uls_factorization ; RETURN
        END IF
        IF ( inform%uls_inform%rank < MIN( m, n ) ) THEN
          inform%rank_def = .TRUE.
          inform%rank = inform%ULS_inform%rank
        END IF

        array_name = resize_prefix // 'A_ROWS_basic'
        CALL SPACE_resize_array( m, A_ROWS_basic,                              &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )

        IF ( inform%status /= GALAHAD_ok ) RETURN
        array_name = resize_prefix // 'A_COLS_basic'
        CALL SPACE_resize_array( n, A_COLS_basic,                              &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

!  Determine the nonsingular submatrix of the factorization

        CALL ULS_enquire( A1_data, inform%ULS_inform,                          &
                          A_ROWS_basic( : m ), A_COLS_basic( : n ) )
        rank_a = inform%ULS_inform%rank

      END IF

!     CALL CPU_TIME( t2 )
!     WRITE(6,"(' time to find basis ',F6.2)") t2 - t1

!  Record rank-defficient problems

      IF ( rank_a < MIN( m, n ) ) THEN
        inform%rank_def = .TRUE.
        IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,              &
          "( /, A, ' ** WARNING nullity A = ', I0 )" ) prefix,                 &
          MIN( m, n ) - rank_a
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_find_basis

      END SUBROUTINE SBLS_find_basis

!-*-*-   S B L S _ S O L V E _ I T E R A T I V E   S U B R O U T I N E   -*-*-

      SUBROUTINE SBLS_solve_iterative( n, m, H, A, SOL, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  solve the linear system
!
!    ( H   A^T ) ( x ) =  ( rhs_d ) = rhs
!    ( A    0  ) ( y )    ( rhs_p )
!
!  Here H must be positive semi-definite on the null-space of A
!
!  The rhs must be input in SOL, and ( x ) will be returned in SOL
!                                    ( y )
!
!  The vectors G, P, RHS_d, R, V are all used for workspace

!  Essentially the same as EQP_resolve_main merged with GLTR_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL
      TYPE ( SMT_type ), INTENT( IN ) :: A, H
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data

      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: G, P, RHS_d
!     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n + m ) :: R, V
!     REAL ( KIND = wp ), DIMENSION( n ) :: GG

!  local variables

      INTEGER :: out, error, i, j, l, np1, npi, npm, iter, itmax
      LOGICAL :: negative_curvature, rank_def
      LOGICAL :: printe, printi, printt, printc, printw
      REAL :: time_record, time_now
      REAL ( KIND = wp ) :: clock_record, clock_now
      REAL ( KIND = wp ) :: alpha, beta, rminvr, rminvr_old, res_p, res_d
      REAL ( KIND = wp ) :: stop, normp, diag, offdiag, val, maxres, curv, piv
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( SMT_type ) :: C0
      TYPE ( SBLS_control_type ) :: SBLS_control

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL SMT_put( C0%type, 'ZERO', i )

!  ===========================
!  Control the output printing
!  ===========================

      out = control%out ; error = control%error

!  all error output

      printe = error > 0 .AND. control%print_level >= 0

!  basic single line of output per iteration

      printi = out > 0 .AND. control%print_level >= 1

!  as per printi, but with additional timings for various operations

      printt = out > 0 .AND. control%print_level >= 2

!  as per printt, but with indication of progress in the CG iteration

      printc = out > 0 .AND. control%print_level >= 3

!  as per printc, but with checking of residuals, etc, and also with an
!  indication of where in the code we are

      printw = out > 0 .AND. control%print_level >= 4

      SBLS_control = control
      rank_def = inform%rank_def
      np1 = n + 1 ; npm = n + m

!  allocate workspace if necessary

      IF ( n >= data%last_n ) THEN
        data%last_n = n

        array_name = 'sbls: G'
        CALL SPACE_resize_array( n, data%G,                                    &
           inform%status, inform%alloc_status,  array_name = array_name,       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: P'
        CALL SPACE_resize_array( n, data%P,                                    &
           inform%status, inform%alloc_status,  array_name = array_name,       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: RHS_d'
        CALL SPACE_resize_array( n, data%RHS_d,                                &
           inform%status, inform%alloc_status,  array_name = array_name,       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
      END IF

      IF ( npm >= data%last_npm ) THEN
        data%last_npm = npm

        array_name = 'sbls: R'
        CALL SPACE_resize_array( npm, data%R,                                  &
           inform%status, inform%alloc_status,  array_name = array_name,       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: V'
        CALL SPACE_resize_array( npm, data%V,                                  &
           inform%status, inform%alloc_status,  array_name = array_name,       &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = control%space_critical,                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
     END IF

!  ------------------
!   Solve the system
!  ------------------

      data%RHS_d( : n ) = SOL( : n )

!  ------------------------------------------------
!   a. Find a point x_f that satisfies A x = rhs_p
!  ------------------------------------------------

!   find a suitable feasible point x_f (stored in V)

      IF ( MAXVAL( ABS( SOL( np1 : npm ) ) ) /= zero ) THEN
        data%V( : n  ) = SOL( : n )
!       data%V( : n  ) = zero
        data%V( np1 : npm ) = SOL( np1 : npm )

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        IF ( inform%factorization == 3 ) THEN
          CALL SBLS_solve_null_space( data%nfactors, SBLS_control, inform,     &
                                      data%V( : npm ) )
        ELSE IF ( inform%preconditioner > 0 ) THEN
          CALL SBLS_solve_explicit( n, m, A, C0, data%efactors, SBLS_control,  &
                                    inform, data%V( : npm ) )
        ELSE
          CALL SBLS_basis_solve( data%ifactors, SBLS_control,                  &
                                 inform, data%V( : npm ) )
        END IF
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
          inform%time%apply = inform%time%apply + time_now - time_record
          inform%time%clock_apply = inform%time%clock_apply +                  &
                                      clock_now - clock_record

        IF ( inform%status < 0 ) THEN
          inform%status = inform%status ; RETURN
        END IF

!  compute the constraint residuals

        data%R( np1 : npm ) = - SOL( np1 : npm )
        DO l = 1, A%ne
          npi = n + A%row( l )
          data%R( npi ) = data%R( npi ) + A%val( l ) * data%V( A%col( l ) )
        END DO

!  check to see if the residuals of potentially inconsistent constraints
!  are satisfactory

        maxres = MAX( MAXVAL( ABS( A%val( : A%ne ) ) ),                        &
                      MAXVAL( ABS( data%R( np1 : npm ) ) ),                    &
                      MAXVAL( ABS( data%V( : n ) ) ) )
        maxres = MAX( control%stop_relative * maxres, control%stop_absolute )

        IF ( rank_def ) THEN
          IF ( MAXVAL( ABS( data%R( np1 : npm ) ) ) <= maxres ) THEN
            IF ( printw ) WRITE( out,                                          &
              "( A, ' residual ', ES12.4, ' acceptably small, consistent',     &
           &    ' constraints' )" ) prefix, maxres
          ELSE
            IF ( printi ) WRITE( out,                                          &
              "( A, ' residual ', ES12.4, ' too large, constraints likely',    &
           &     ' inconsistent' )" ) prefix, maxres
            inform%status = GALAHAD_error_primal_infeasible ; RETURN
          END IF
        ELSE
          IF ( MAXVAL( ABS( data%R( np1 : npm ) ) ) > maxres ) THEN
            IF ( printi ) WRITE( out,                                          &
              "( A, ' residual ', ES12.4, ' too large, factorization likely',  &
           &     ' innacurate' )" ) prefix, maxres
            inform%status = GALAHAD_error_ill_conditioned ; RETURN
          END IF
        END IF

!  compute the gradient value at x_f (stored in G)

!       data%G( : n ) = - data%RHS_d( : n )
        data%G( : n ) = - SOL( : n )
        DO l = 1, H%ne
          i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
          data%G( i ) = data%G( i ) + val * data%V( j )
          IF ( i /= j ) data%G( j ) = data%G( j ) + val * data%V( i )
        END DO
        SOL( : n ) = data%V( : n )
        IF ( printt ) THEN
          IF ( m > 0 ) THEN
            WRITE( out, "( A, ' Feasibility phase primal KKT residual = ',     &
           &  ES10.4 )" ) prefix, MAXVAL( ABS( data%R( np1 : npm ) ) )
          ELSE
            WRITE( out, "( A, ' Feasibility phase primal KKT residual = ',     &
           &  ES10.4 )" ) prefix, zero
          END IF
        END IF
      ELSE
!       data%G( : n ) = - data%RHS_d( : n )
        data%G( : n ) = - SOL( : n )
        SOL( : n ) = zero
        IF ( printt )                                                          &
          WRITE( out, "( A, ' phase 1 primal KKT residual = ',       &
         &  ES10.4 )" ) prefix, zero
      END IF

!  ---------------------------------------------------
!   b. From x_f, use projected CG to find a point that
!       also satisfies the remaining equations
!  ---------------------------------------------------

!  compute the correction s from x_f (stored in S)

!  set initial data

      SBLS_control%affine = .TRUE.

      CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

!  ========================================
!  start of the Projected CG iteration loop
!  ========================================

      iter = 0
      itmax = control%maxit_pcg ; IF ( itmax < 0 ) itmax = n
      negative_curvature = .FALSE.

      DO

!  ----------------------------------
!  obtain the preconditioned residual
!  ----------------------------------

        IF ( printw ) WRITE( out,                                              &
           "( A, ' ............... precondition  ............... ' )" ) prefix
        data%V( : n ) = data%G( : n ) ; data%V( np1 : npm ) = zero

        CALL SBLS_solve( n, m, A, C0, data, SBLS_control, inform,              &
                         data%V( : npm ) )
        IF ( inform%status < 0 ) EXIT

!  compute the residuals ...

        data%R( : n ) = data%G( : n )
        data%R( np1 : npm ) = - SOL( np1 : npm )
        DO l = 1, A%ne
          i = A%row( l ) ; j = A%col( l ) ; npi = n + i ; val = A%val( l )
          data%R( npi ) = data%R( npi ) + val * SOL( j )
          data%R( j ) = data%R( j ) - val * data%V( npi )
        END DO

!  ... and their norms

        res_d = MAXVAL( ABS( data%R( : n ) ) )
        res_p = MAXVAL( ABS( data%R( np1 : npm ) ) )

!  obtain the scaled norm of the "gradient"

        rminvr = DOT_PRODUCT( data%G( : n ), data%V( : n ) )
        IF ( ABS( rminvr ) < rminvr_zero ) rminvr = zero
        IF ( rminvr < zero ) THEN
          IF ( MAXVAL( ABS( data%V( : n ) ) )                                  &
               < epsmch * MAXVAL( ABS( data%G( : n ) ) ) ) THEN
            rminvr = zero
          ELSE
            IF ( printe ) WRITE( error,                                        &
               "( A, ' The matrix M appears to be indefinite.',                &
              &   ' Inner product = ', ES12.4  )" ) prefix, rminvr
            inform%status = GALAHAD_error_preconditioner ; EXIT
          END IF
        END IF

        IF ( iter > 0 ) THEN
          beta = rminvr / rminvr_old
          diag = beta / alpha
          offdiag = SQRT( beta ) / ABS( alpha )
        ELSE

!  compute the stopping tolerance

          diag = zero
          stop = MAX( control%stop_relative * res_d, control%stop_absolute )
!         IF ( printc ) WRITE( out,                                            &
!           "( /, A, ' stopping tolerance = ', ES10.4 )" ) prefix, stop
!write(6,*) ' H_max = ', MAXVAL( ABS( H%val( : H%ne ) ) )
        END IF

!  print details of the latest iteration

        IF ( printc ) THEN
          IF ( MOD( iter, 25 ) == 0 ) THEN
            WRITE( out, "( /, A, '   Iter   res_p    res_d    step ',          &
           &  '   norm p  - res_d_stop =', ES9.2 )" ) prefix, stop
          END IF

          IF ( iter /= 0 ) THEN
            WRITE( out, "( A, I7, 4ES9.2 )" )                                  &
              prefix, iter, res_p, res_d, alpha, normp
          ELSE
            WRITE( out, "( A, I7, 2ES9.2, 4X, '-', 8X, '-' )" )                &
              prefix, iter, res_p, res_d
          END IF
        END IF

!  test for convergence

        IF ( res_d <= stop ) THEN
          IF ( printw ) WRITE( out,                                            &
            "( A, ' res_p ', ES10.4, ' < ', ES10.4 )" )                        &
               prefix, res_p, stop
          inform%status = GALAHAD_ok ; EXIT
        END IF

!  test to see that iteration limit has not been exceeded

        IF ( iter >= itmax ) THEN
          IF ( printc ) WRITE( out,                                            &
            "( /, A, ' Iteration limit exceeded ' ) " ) prefix
          inform%status = GALAHAD_error_max_iterations ; EXIT
        END IF

!  obtain the search direction P

        IF ( iter > 0 ) THEN
          data%P( : n ) = - data%V( : n ) + beta * data%P( : n )
!         pmp = rminvr + pmp * beta * beta

!  special case for the first iteration

        ELSE
          data%P( : n ) = - data%V( : n )
!         pmp = rminvr
        END IF
        rminvr_old = rminvr

!  compute the 2-norm of the search direction

        normp = TWO_NORM( data%P( : n ) )

!  test that the step is non trivial

        IF ( normp <= ten * epsmch ) THEN
          IF ( printc ) WRITE( out, "( A, ' pnorm ', ES12.4, ' < ', ES12.4 )" )&
            prefix, normp, ten * epsmch
          inform%status = GALAHAD_ok ; EXIT
        END IF

        iter = iter + 1

!  ------------------------------
!  obtain the product of H with p
!  ------------------------------

        IF ( printw ) WRITE( out,                                              &
          "( A, ' ............ matrix-vector product ..........' )" ) prefix

        data%V( : n ) = zero
        DO l = 1, H%ne
          i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
          data%V( i ) = data%V( i ) + val * data%P( j )
          IF ( i /= j ) data%V( j ) = data%V( j ) + val * data%P( i )
        END DO

!  obtain the curvature

        curv = DOT_PRODUCT( data%V( : n ), data%P( : n ) )
!       rayleigh = curv / pmp

!  obtain the stepsize and the new diagonal of the Lanczos tridiagonal

        IF ( curv > zero ) THEN
          alpha = rminvr / curv
          diag = diag + one / alpha
        ELSE
          negative_curvature = .TRUE.
        END IF

!  check that the Lanczos tridiagonal is still positive definite

        IF ( .NOT. negative_curvature ) THEN
          IF ( iter > 1 ) THEN
            piv = diag - ( offdiag / piv ) * offdiag
          ELSE
            piv = diag
          END IF
          negative_curvature = piv <= zero
        END IF

!  the matrix H is indefinite on the null-space of A

        IF ( negative_curvature ) THEN
          IF ( printe ) WRITE( error,                                          &
             "( A, ' The matrix H appears to be indefinite on Null(A)' )" )    &
                prefix
          inform%status = GALAHAD_error_inertia ; EXIT
        END IF

!  update the solution and residual

        SOL( : n ) = SOL( : n ) + alpha * data%P( : n )
!       data%G( : n ) = data%G( : n ) + alpha * data%V( : n )
!       res_d = MMAXVAL( ABS( data%G( : n ) ) )

!  recompute the residual for more accuracy

        data%G( : n ) = - data%RHS_d( : n )
        DO l = 1, H%ne
          i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
          data%G( i ) = data%G( i ) + val * SOL( j )
          IF ( i /= j ) data%G( j ) = data%G( j ) + val * SOL( i )
        END DO
!       write(6,*) ' res_d (rec,acc) = ', res_d, MAXVAL( ABS( data%G( : n ) ) )

!  ============================
!  end of the CG iteration loop
!  ============================

      END DO
      inform%iter_pcg = inform%iter_pcg + iter

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%apply = inform%time%apply + time_now - time_record
      inform%time%clock_apply = inform%time%clock_apply +                      &
                                  clock_now - clock_record

!  compute the residuals

      data%R( : n ) = - data%RHS_d( : n )
      data%R( np1 : npm ) = - SOL( np1 : npm )
      DO l = 1, A%ne
        i = A%row( l ) ; j = A%col( l ) ; npi = n + i ; val = A%val( l )
        data%R( npi ) = data%R( npi ) + val * SOL( j )
        data%R( j ) = data%R( j ) - val * data%V( npi )
      END DO
      DO l = 1, H%ne
        i = H%row( l ) ; j = H%col( l ) ; val = H%val( l )
        data%R( i ) = data%R( i ) + val * SOL( j )
        IF ( i /= j ) data%R( j ) = data%R( j ) + val * SOL( i )
      END DO
      res_d = MAXVAL( ABS( data%R( : n ) ) )
      res_p = MAXVAL( ABS( data%R( np1 : npm ) ) )

      IF ( printt ) WRITE( out, "(  A, ' on exit from projected CG: status =', &
     &      1X, I0, ', PCG iterations = ', I0, ', time = ', F0.2, /,           &
     &      A, ' primal-dual KKT residuals = ', ES10.4, ', ', ES10.4 )" )      &
           prefix, inform%status, iter, inform%time%clock_apply, prefix,       &
           res_p, res_d

!  compute the Lagrange multiplier estimates

      SOL( np1 : npm ) = - data%V( np1 : npm )

      inform%status = GALAHAD_ok
      RETURN

!  End of SBLS_solve_iterative

      END SUBROUTINE SBLS_solve_iterative

!-*- S B L S _ F R E D H O L M _ A L T E R N A T I V E   S U B R O U T I N E -*-

      SUBROUTINE SBLS_fredholm_alternative( n, m, A, efactors,                 &
                                            control, inform, SOL )

!  Find the Fredholm Alternative (x,y), i.e. either

!    ( G   A^T ) ( x ) = ( a )
!    ( A    -C ) ( y )   ( b )

!  or

!    ( G   A^T ) ( x ) = ( 0 )  and ( a^T  b^T ) ( x ) > 0
!    ( A    -C ) ( y )   ( 0 )                   ( y )

!  using an explicit factorization of K or C + A G(inv) A(transpose)

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SBLS_explicit_factors_type ), INTENT( INOUT ) :: efactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL

!  Local variables

      INTEGER :: i, ii, j, l, npm

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      npm = n + m

      IF ( control%print_level >= 11 .AND. control%out >= 0 )                  &
        WRITE( control%out, "( ' RHS:', /, ( 3ES24.16 ) )" )                   &
          ( SOL( i ), i = 1, npm )

      IF ( control%print_level >= 4 .AND. control%out > 0 )                    &
        WRITE( control%out, "( A, ' maximum residual = ', ES10.4 )" )          &
          prefix, MAXVAL( ABS( SOL( : npm ) ) )

!  Use factors of the Schur complement

!write(6,*) ' inform%factorization ', inform%factorization

      IF ( inform%factorization == 1 ) THEN
!       IF ( control%error > 0 .AND. control%print_level > 0 )                 &
!           WRITE( control%error, "( A, ' error, factorization = 1 disabled ', &
!          & 'in solve_fredholm_alternative')" ) prefix
!       inform%status = GALAHAD_unavailable_option
!       RETURN

!  Form a <- diag(G)(inverse) a

        SOL( : n ) = SOL( : n ) / efactors%G_diag( : n )

        IF ( m > 0 ) THEN

!  Form b <- A a - b

          SOL( n + 1 : npm ) = - SOL( n + 1 : npm )
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, m
              ii = n + i
              SOL( ii ) = SOL( ii ) +                                          &
                DOT_PRODUCT( A%val( l + 1 : l + n ), SOL( : n ) )
              l = l + n
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              ii = n + i
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                SOL( ii ) = SOL( ii ) + A%val( l ) * SOL( A%col( l ) )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              ii = n + A%row( l )
              SOL( ii ) = SOL( ii ) + A%val( l ) * SOL( A%col( l ) )
            END DO
          END SELECT

!  Solve  ( C + A G(inv) A(transpose) ) y = A diag(G)(inverse) a - b
!  and place the result in a

          CALL SLS_fredholm_alternative( efactors%K, SOL( n + 1 : npm ),       &
                                         efactors%K_data, efactors%K_control,  &
                                         inform%SLS_inform )

          inform%sls_solve_status = inform%SLS_inform%status
          inform%alternative = inform%SLS_inform%alternative
!write(6,*) ' sls status ', inform%sls_solve_status, inform%alternative
          IF ( inform%sls_solve_status < 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' solve exit status = ', I0 )" )       &
                prefix, inform%sls_solve_status
            inform%status = GALAHAD_error_solve
            RETURN
          END IF

!  Form a <- diag(G)(inverse) ( a - A(trans) y )

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, m
              DO j = 1, n
                l = l + 1
                SOL( j ) = SOL( j ) - A%val( l ) *                             &
                  SOL( n + i ) / efactors%G_diag( j )
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, m
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                SOL( j ) = SOL( j ) - A%val( l ) *                             &
                  SOL( n + i ) / efactors%G_diag( j )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              j = A%col( l )
              SOL( j ) = SOL( j ) - A%val( l ) *                               &
                SOL( n + A%row( l ) ) / efactors%G_diag( j )
            END DO
          END SELECT
        END IF

!  Use factors of the augmented system

      ELSE
        CALL SLS_fredholm_alternative( efactors%K, SOL, efactors%K_data,       &
                                       efactors%K_control, inform%SLS_inform )
        inform%sls_solve_status = inform%SLS_inform%status
        inform%alternative = inform%SLS_inform%alternative
!write(6,*) ' sls status ', inform%sls_solve_status, inform%alternative
        IF ( inform%sls_solve_status < 0 ) THEN
          IF ( control%out > 0 .AND. control%print_level > 0 )                 &
            WRITE( control%out, "( A, ' solve exit status = ', I0 )" )         &
              prefix, inform%sls_solve_status
          inform%status = GALAHAD_error_solve
          RETURN
        END IF
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_fredholm_alternative

      END SUBROUTINE SBLS_fredholm_alternative

!  End of module SBLS

    END MODULE GALAHAD_SBLS_double
