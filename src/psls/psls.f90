! THIS VERSION: GALAHAD 3.3 - 26/07/2021 AT 14:45 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ P S L S   M O D U L E  -*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   Based on LANCELOT B module PRECN (originally LANCELOT A) ~1992
!   originally released GALAHAD Version 2.2. April 13th 2008
!   Incorporated SLS in place of SILS GALAHAD Version 2.4, January 27th 2010

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

   MODULE GALAHAD_PSLS_double

!      -----------------------------------------------
!     |                                               |
!     | Given a symmetric matrix A, provide and       |
!     ! apply a symmetric, positive-definite or       |
!     | strictly-diagonally-dominat preconditioner P  |
!     |                                               |
!      -----------------------------------------------

      USE GALAHAD_CLOCK
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_QPT_double, ONLY : QPT_keyword_H
      USE GALAHAD_SLS_double
      USE GALAHAD_SCU_double, ONLY : SCU_matrix_type, SCU_data_type,           &
        SCU_inform_type, SCU_factorize, SCU_solve, SCU_append, SCU_terminate
      USE GALAHAD_SORT_double, ONLY : SORT_reorder_by_cols
      USE GALAHAD_EXTEND_double, ONLY : EXTEND_arrays
      USE LANCELOT_BAND_double
      USE HSL_MI28_double
      USE GALAHAD_SPECFILE_double
      USE GALAHAD_NORMS_double, ONLY : TWO_NORM

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: PSLS_read_specfile, PSLS_initialize, PSLS_terminate,           &
                PSLS_form_and_factorize, PSLS_update_factors, PSLS_solve,      &
                PSLS_index_submatrix, PSLS_product, PSLS_norm, PSLS_name,      &
                PSLS_build, SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )
      INTEGER, PARAMETER :: long = SELECTED_INT_KIND( 18 )

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: liwmin = 1, lwmin = 1
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one )

!  non-standard error returns

     INTEGER, PUBLIC, PARAMETER :: GALAHAD_preconditioner_unknown = - 51
     INTEGER, PUBLIC, PARAMETER :: GALAHAD_norm_unknown = - 52

!  preconditioners

      INTEGER, PARAMETER :: preconditioner_none = - 1
      INTEGER, PARAMETER :: preconditioner_auto = 0
      INTEGER, PARAMETER :: preconditioner_diagonal = 1
      INTEGER, PARAMETER :: preconditioner_band = 2
      INTEGER, PARAMETER :: preconditioner_reordered_band = 3
      INTEGER, PARAMETER :: preconditioner_full_se = 4
      INTEGER, PARAMETER :: preconditioner_full_gmps = 5
      INTEGER, PARAMETER :: preconditioner_incomplete_lm = 6
      INTEGER, PARAMETER :: preconditioner_incomplete_mi28 = 7
      INTEGER, PARAMETER :: preconditioner_incomplete_munks = 8
      INTEGER, PARAMETER :: preconditioner_expanding_band = 9

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

!  - - - - - - - - - - - - - - - - - - - - - - -
!   control derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: PSLS_control_type

!  unit for error messages

        INTEGER :: error = 6

!  unit for monitor output

        INTEGER :: out = 6

!  controls level of diagnostic output

        INTEGER :: print_level = 0

!  which preconditioner to use:
!   <0  no preconditioning, P = I
!    0  automatic
!    1  diagonal, P = diag( max( A, %min_diagonal ) )
!    2  banded, P = band( A ) with semi-bandwidth %semi_bandwidth
!    3  re-ordered band, P = band(order(A)) with semi-bandwidth %semi_bandwidth
!    4  full factorization, P = A, Schnabel-Eskow modification
!    5  full factorization, P = A, GMPS modification
!    6  incomplete factorization, Lin-More'
!    7  incomplete factorization, HSL_MI28
!    8  incomplete factorization, Munskgaard
!    9  expanding band

        INTEGER :: preconditioner = 0

!  the semi-bandwidth for band(H)

        INTEGER :: semi_bandwidth = 5

!  not used at present

        INTEGER :: scaling = 0
        INTEGER :: ordering = 0

!  maximum number of nonzeros in a column of A for Schur-complement
!  factorization  to accommodate newly fixed variables

        INTEGER :: max_col = 100

!  number of extra vectors of length n required by the Lin-More' incomplete
!  Cholesky preconditioner

        INTEGER :: icfs_vectors = 10

!  the maximum number of fill entries within each column of the incomplete
!  factor L computed by HSL_MI28. In general, increasing mi28_lsize improves
!  the quality of the preconditioner but increases the time to compute
!  and then apply the preconditioner. Values less than 0 are treated as 0

        INTEGER :: mi28_lsize = 10

!  the maximum number of entries within each column of the strictly lower
!  triangular matrix R used in the computation of the preconditioner by
!  HSL_MI28.  Rank-1 arrays of size mi28_rsize *  n are allocated internally
!  to hold R. Thus the amount of memory used, as well as the amount of work
!  involved in computing the preconditioner, depends on mi28_rsize. Setting
!  mi28_rsize > 0 generally leads to a higher quality preconditioner than
!  using mi28_rsize = 0, and choosing mi28_rsize >= mi28_lsize is generally
!  recommended

        INTEGER :: mi28_rsize = 10

!  the minimum permitted diagonal in diag(max(H,min_diag))

        REAL ( KIND = wp ) :: min_diagonal = 0.00001_wp

!  set new_structure true if the storage structure for the input matrix has
!  changed, and false if only the values have changed

        LOGICAL :: new_structure = .TRUE.

!  set get_semi_bandwidth true if the semi-bandwidth of the submatrix is to be
!  calculated

        LOGICAL :: get_semi_bandwidth = .TRUE.

!  set get_norm_residual true if the residual when applying the preconditioner
!  are to be calculated

        LOGICAL :: get_norm_residual = .FALSE.

!  if space is critical, ensure allocated arrays are no bigger than needed

        LOGICAL :: space_critical = .FALSE.

!  exit if any deallocation fails

        LOGICAL :: deallocate_error_fatal  = .FALSE.

!  definite linear equation solver

        CHARACTER ( LEN = 30 ) :: definite_linear_solver =                     &
           "sils" // REPEAT( ' ', 26 )

!  all output lines will be prefixed by
!    prefix(2:LEN(TRIM(%prefix))-1)
!  where prefix contains the required string enclosed in quotes,
!  e.g. "string" or 'string'

        CHARACTER ( LEN = 30 ) :: prefix

!  control parameters for SLS

        TYPE ( SLS_control_type ) :: SLS_control

!  control parameters for HSL_MI28

        TYPE ( MI28_control ) :: MI28_control
      END TYPE PSLS_control_type

!  - - - - - - - - - - - - - - - - - - - - - -
!   time derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: PSLS_time_type

!  total time

       REAL :: total = 0.0

!  time to assemble the preconditioner prior to factorization

        REAL :: assemble = 0.0

!  time for the analysis phase

       REAL :: analyse = 0.0

!  time for the factorization phase

       REAL :: factorize = 0.0

!  time for the linear solution phase

       REAL :: solve = 0.0

!  time to update the factorization

        REAL :: update = 0.0

!  total clock time spent in the package

       REAL ( KIND = wp ) :: clock_total = 0.0

!  clock time to assemble the preconditioner prior to factorization

        REAL ( KIND = wp ) :: clock_assemble = 0.0

!  clock time for the analysis phase

       REAL ( KIND = wp ) :: clock_analyse = 0.0

!  clock time for the factorization phase

       REAL ( KIND = wp ) :: clock_factorize = 0.0

!  clock time for the linear solution phase

       REAL ( KIND = wp ) :: clock_solve = 0.0

!  clock time to update the factorization

        REAL ( KIND = wp ) :: clock_update = 0.0

      END TYPE PSLS_time_type

!  - - - - - - - - - - - - - - - - - - - - - - -
!   inform derived type with component defaults
!  - - - - - - - - - - - - - - - - - - - - - - -

      TYPE, PUBLIC :: PSLS_inform_type

!  reported return status:
!     0  success
!    -1  allocation error
!    -2  deallocation error
!    -3  matrix data faulty (%n < 1, %ne < 0)
!   -20  alegedly +ve definite matrix is not

       INTEGER :: status = 1

!  STAT value after allocate failure

       INTEGER :: alloc_status = 0

!  status return from factorization

       INTEGER :: analyse_status = 0

!  status return from factorization

       INTEGER :: factorize_status = 0

!  status return from solution phase

       INTEGER :: solve_status = 0

!  number of integer words to hold factors

       INTEGER :: factorization_integer = - 1

!  number of real words to hold factors

       INTEGER :: factorization_real = - 1

!  code for the actual preconditioner used (see control%preconditioner)

       INTEGER :: preconditioner = - 100

!  the actual semi-bandwidth

       INTEGER :: semi_bandwidth = - 1

!  the semi-bandwidth following reordering (if any)

       INTEGER :: reordered_semi_bandwidth = - 1

!  number of indices out-of-range

       INTEGER :: out_of_range = 0

!  number of duplicates

       INTEGER :: duplicates = 0

!  number of entries from the strict upper triangle

       INTEGER :: upper = 0

!  number of missing diagonal entries for an allegedly-definite matrix

       INTEGER :: missing_diagonals = 0

!  the semi-bandwidth used

       INTEGER :: semi_bandwidth_used = - 1

!  number of 1 by 1 pivots in the factorization

       INTEGER :: neg1 = - 1

!  number of 2 by 2 pivots in the factorization

       INTEGER :: neg2 = - 1

!  has the preconditioner been perturbed during the fctorization?

       LOGICAL :: perturbed = .FALSE.

!  ratio of fill in to original nonzeros

       REAL ( KIND = wp ) :: fill_in_ratio

!  the norm of the solution residual

       REAL ( KIND = wp ) :: norm_residual

!  name of array which provoked an allocate failure

       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )

!  the integer and real output arrays from mc61

       INTEGER, DIMENSION( 10 ) :: mc61_info
       REAL ( KIND = wp ), DIMENSION( 15 ) :: mc61_rinfo

!  times for various stages

       TYPE ( PSLS_time_type ) :: time

!  inform values from SLS

        TYPE ( SLS_inform_type ) :: SLS_inform

!  the output structure from mi28

       TYPE ( MI28_info ) :: mi28_info
      END TYPE PSLS_inform_type

      TYPE, PUBLIC :: PSLS_data_type
        INTEGER :: n, n_sub, n_fixed, n_update, max_col, semi_bandwidth_used
        INTEGER :: n_pert, scu_status, p_ne, l_ne, mc61_lirn, mc61_liw
        REAL ( KIND = wp ) :: perturbation
        LOGICAL :: sub_matrix, perturbed
        INTEGER, DIMENSION( 10 ) :: mc61_ICNTL                                 &
          = (/ 6, 6, 0, 0, 0, 0, 0, 0, 0, 0 /)
        REAL ( KIND = wp ), DIMENSION( 5 ) :: mc61_CNTL                        &
          = (/ 2.0_wp, 1.0_wp, 0.0_wp, 0.0_wp, 0.0_wp /)
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: SUB, INDEX, IW, PERM, MAPS
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: P_colptr, P_row, P_col
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: L_colptr, L_row
        INTEGER, ALLOCATABLE, DIMENSION( : , : ) :: IKEEP, IW1
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W, RHS_sub, RHS_scu
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL_scu, G, DIAG
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERT, SOL_sub
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P_diag, P_offd
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: L_diag, L_offd
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: W1, OFFDIA, D
        TYPE ( SMT_type ) :: P, P_csr
        TYPE ( SLS_control_type ) :: SLS_control
        TYPE ( SLS_data_type ) :: SLS_data
        TYPE ( SCU_matrix_type ) :: SCU_matrix
        TYPE ( SCU_data_type ) :: SCU_data
        TYPE ( SCU_inform_type ) :: SCU_inform
        TYPE ( MI28_control ) :: MI28_control
        TYPE ( MI28_keep ) ::  MI28_keep
      END TYPE PSLS_data_type

!  ================================
!  The PSLS_save_type derived type
!  ================================

      TYPE :: PSLS_save_type
        INTEGER :: liw, lw, nsemiw, liccgg, nextra, nz01, iaj
        REAL ( KIND = KIND( 1.0E0 ) ) :: tfactr, t1stsl, tupdat, tsolve
        INTEGER :: ICNTL_iccg( 5 ), KEEP_iccg( 12 ), INFO_iccg( 10 )
        REAL ( KIND = wp ) :: CNTL_iccg( 3 )
      END TYPE PSLS_save_type

   CONTAINS

!-*-*-*-*-*-   P S L S _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

     SUBROUTINE PSLS_initialize( data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for PSLS. This routine should be called before
!  PSLS_solve
!
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  data     private internal data
!  control  a structure containing control information. Components are
!           described above
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

     TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( PSLS_control_type ), INTENT( OUT ) :: control
     TYPE ( PSLS_inform_type ), INTENT( OUT ) :: inform

      inform%status = GALAHAD_ok

!  initialize control parameters for SLS (see GALAHAD_SLS for details)

     control%SLS_control%prefix = '" - SLS:"                     '

     RETURN

!  End of PSLS_initialize

     END SUBROUTINE PSLS_initialize

!-*-*-*-   P S L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

     SUBROUTINE PSLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given could (roughly)
!  have been set as:

! BEGIN PSLS SPECIFICATIONS (DEFAULT)
!  error-printout-device                             6
!  printout-device                                   6
!  print-level                                       0
!  maximum-refinements                               1
!  maximum-schur-complement                          100
!  preconditioner-used                               0
!  semi-bandwidth-for-band-preconditioner            5
!  number-of-lin-more-vectors                        10
!  mi28-l-fill-size                                  10
!  mi28-r-entry-size                                 10
!  ordering-used                                     3
!  scaling-used                                      0
!  minimum-diagonal                                  1.0D-5
!  pivot-tolerance-used                              1.0D-12
!  zero-pivot-tolerance                              1.0D-12
!  static-pivoting-diagonal-perturbation             0.0D+0
!  level-at-which-to-switch-to-static                0.0D+0
!  new-structure                                     T
!  get-semi-bandwidth                                T
!  get-norm-of-residual                              F
!  space-critical                                    F
!  deallocate-error-fatal                            F
!  definite-linear-equation-solver                   sils
!  output-line-prefix                                ""
! END PSLS SPECIFICATIONS

!  Dummy arguments

     TYPE ( PSLS_control_type ), INTENT( INOUT ) :: control
     INTEGER, INTENT( IN ) :: device
     CHARACTER( LEN = * ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

     INTEGER, PARAMETER :: error = 1
     INTEGER, PARAMETER :: out = error + 1
     INTEGER, PARAMETER :: print_level = out + 1
     INTEGER, PARAMETER :: preconditioner = print_level + 1
     INTEGER, PARAMETER :: semi_bandwidth = preconditioner + 1
     INTEGER, PARAMETER :: scaling = semi_bandwidth + 1
     INTEGER, PARAMETER :: ordering = scaling + 1
     INTEGER, PARAMETER :: max_col = ordering + 1
     INTEGER, PARAMETER :: icfs_vectors = max_col + 1
     INTEGER, PARAMETER :: mi28_lsize = icfs_vectors + 1
     INTEGER, PARAMETER :: mi28_rsize = mi28_lsize + 1
     INTEGER, PARAMETER :: min_diagonal = mi28_rsize + 1
     INTEGER, PARAMETER :: new_structure = min_diagonal + 1
     INTEGER, PARAMETER :: get_semi_bandwidth = new_structure + 1
     INTEGER, PARAMETER :: get_norm_residual = get_semi_bandwidth + 1
     INTEGER, PARAMETER :: space_critical = get_norm_residual + 1
     INTEGER, PARAMETER :: deallocate_error_fatal  = space_critical + 1
     INTEGER, PARAMETER :: definite_linear_solver = deallocate_error_fatal  + 1
     INTEGER, PARAMETER :: prefix = definite_linear_solver + 1
     INTEGER, PARAMETER :: lspec = prefix
     CHARACTER( LEN = 4 ), PARAMETER :: specname = 'PSLS'
     TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

     spec( error )%keyword = 'error-printout-device'
     spec( out )%keyword = 'printout-device'
     spec( print_level )%keyword = 'print-level'
     spec( preconditioner )%keyword = 'preconditioner-used'
     spec( semi_bandwidth )%keyword = 'semi-bandwidth-for-band-preconditioner'
     spec( scaling )%keyword = 'scaling-used'
     spec( ordering )%keyword = 'ordering-used'
     spec( max_col )%keyword = 'maximum-schur-complement'
     spec( icfs_vectors )%keyword = 'number-of-lin-more-vectors'
     spec( mi28_lsize )%keyword = 'mi28-l-fill-size'
     spec( mi28_rsize )%keyword = 'mi28-r-entry-size'

!  Real key-words

     spec( min_diagonal )%keyword = 'minimum-diagonal'

!  Logical key-words

     spec( new_structure )%keyword = 'new-structure'
     spec( get_semi_bandwidth )%keyword = 'get-semi-bandwidth'
     spec( get_norm_residual )%keyword = 'get-norm-of-residual'
     spec( space_critical )%keyword = 'space-critical'
     spec( deallocate_error_fatal  )%keyword = 'deallocate-error-fatal'

!  Character key-words

     spec( definite_linear_solver )%keyword = 'definite-linear-equation-solver'
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
      CALL SPECFILE_assign_value( spec( preconditioner ),                      &
                                  control%preconditioner,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( semi_bandwidth ),                      &
                                  control%semi_bandwidth,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( scaling ),                             &
                                  control%scaling,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( ordering ),                            &
                                  control%ordering,                            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( max_col ),                             &
                                  control%max_col,                             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( icfs_vectors ),                        &
                                  control%icfs_vectors,                        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( mi28_lsize ),                          &
                                  control%mi28_lsize,                          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( mi28_rsize ),                          &
                                  control%mi28_rsize,                          &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( min_diagonal ),                        &
                                  control%min_diagonal,                        &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( new_structure ),                       &
                                  control%new_structure,                       &
                                  control%error )
      CALL SPECFILE_assign_value( spec( get_semi_bandwidth ),                  &
                                  control%get_semi_bandwidth,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( get_norm_residual ),                   &
                                  control%get_norm_residual,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( space_critical ),                      &
                                  control%space_critical,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( deallocate_error_fatal  ),             &
                                  control%deallocate_error_fatal ,             &
                                  control%error )

!  Set xharacter values

      CALL SPECFILE_assign_value( spec( definite_linear_solver ),              &
                                  control%definite_linear_solver,              &
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

      RETURN

!  End of PSLS_read_specfile

      END SUBROUTINE PSLS_read_specfile

!-*-   P S L S _ F O R M _ A N D _ F A C T O R I Z E  S U B R O U T I N E   -*-

      SUBROUTINE PSLS_form_and_factorize( A, data, control, inform, SUB )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Form and factorize a symmetric, positive definite approximation P to a
!  symmetric submatrix of given symmetrix matrix A

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!   A is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of A. Four storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of A
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
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       A%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', A%type )
!       A%val( : )   the values of the diagonals of A, stored in order
!
!  data is a structure of type PSLS_data_type which holds private internal data
!
!  control is a structure of type PSLS_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to PSLS_initialize. See PSLS_initialize
!   for details
!
!  inform is a structure of type PSLS_inform_type that provides information on
!   exit from PSLS_solve. The component status has possible values:
!
!     0 Normal termination.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 one of the restrictions
!        A%n    >=  1
!        A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', , 'DIAGONAL' }
!       has been violated.
!
!    -9 the ordering (analysis) phase failed.
!
!    -10 the factorization phase failed.
!
!  On exit from PSLS_form_and_factorize, other components of inform give the
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation
!     analyse_status = The return status from the ordering phase of the
!      factorization (if any)
!     factorize_status = The return status from the factorization phase
!     solve_status = The return status from the solve phase
!     factorization_integer = The total integer workspace required for the
!       factorization
!     factorization_real = The total real workspace required for the
!       factorization
!     preconditioner = Code for the actual preconditioner computed
!     semi_bandwidth = The semi-bandwidth of the original submatrix
!     neg1, neg2 - the number of -ve 1x1 and 2x2 pivots found during the
!       factorization
!     perturbed = true if the initial preconditioner was perturbed to ensure
!      that it is definite
!     fill_in_ratio = ratio of nonzeros in factors to the original matrix
!     norm_residual = norm of the residual during the solve phase
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package
!     time%assemble = the time spent building the preconditioner
!     time%analyse = the time spent reordering the preconditioner prior to
!       factorization
!     time%factorize = the time spent factorizing the preconditioner
!     time%solve = the time spent in the solution phase
!
!   SUB is an optional rank-one integer assumed-sized array whose components
!    list the indices of the required submatrix. The indices should be in
!    increasing order. If SUB is not present, the entire matrix A will be
!    considered.

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform
      INTEGER, OPTIONAL, INTENT( IN ), DIMENSION( : ) :: SUB

!  Local variables

      INTEGER :: i, ii, iii, ij, j, jj, jjj, k, l, icfact, out, print_level
      INTEGER ( KIND = long ) :: predicted
      LOGICAL :: printi, printt
      CHARACTER ( LEN = 80 ) :: array_name
      REAL :: time_now, time_start, time_record
      REAL ( KIND = wp ) :: clock_now, clock_start, clock_record
      REAL ( KIND = wp ) :: de, df

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set initial values

      IF ( control%new_structure ) THEN
        inform%analyse_status = 0 ; inform%factorize_status = 0
        inform%solve_status = 0
        inform%factorization_integer = 0 ; inform%factorization_real = 0
        inform%preconditioner = 0 ; inform%semi_bandwidth = 0
        inform%neg1 = 0 ; inform%neg2 = 0 ; inform%perturbed = .FALSE.
        inform%fill_in_ratio = one ; inform%norm_residual = zero
        inform%bad_alloc  = ''
        inform%time%assemble = 0.0 ; inform%time%analyse  = 0.0
        inform%time%factorize = 0.0 ; inform%time%solve = 0.0
        inform%time%update = 0.0 ; inform%time%total = 0.0
      END IF

!  return if no preconditioning (P=I) is required

      IF ( control%preconditioner == preconditioner_none ) THEN
        inform%status = 0
        RETURN
      END IF

!  record desired output level

      out = control%out
      print_level = control%print_level
      printi = out > 0 .AND. print_level > 0
      printt = out > 0 .AND. print_level > 1

!  Ensure that input parameters are within allowed ranges

      IF ( A%n < 0 .OR. .NOT. QPT_keyword_H( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 930
      END IF

      data%sub_matrix = PRESENT( SUB )

      IF ( control%new_structure .OR. data%sub_matrix ) THEN
!     IF ( control%new_structure ) THEN
        data%n_update = 0
        data%max_col = control%max_col
        IF ( data%max_col <= 0 ) data%max_col = 100

!  Record if a variable is contained in the submatrix

        data%n = A%n
        IF ( data%sub_matrix ) THEN
          data%n_sub = SIZE( SUB )
          array_name = 'psls: data%SUB'
          CALL SPACE_resize_array( data%n_sub, data%SUB,                       &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%INDEX'
          CALL SPACE_resize_array( data%n, data%INDEX,                         &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          data%INDEX = 0
          DO i = 1, data%n_sub
            data%SUB( i ) = SUB( i )
            data%INDEX( SUB( i ) ) = i
          END DO
        ELSE
          data%n_sub = data%n
        END IF

!  Record the variables contained in the submatrix

!  If desired, compute the semi-bandwidth of the selected sub-matrix

        IF ( control%get_semi_bandwidth ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            inform%semi_bandwidth = 0
          CASE ( 'DENSE' )
            inform%semi_bandwidth = data%n_sub - 1
          CASE ( 'SPARSE_BY_ROWS' )
            IF ( data%sub_matrix ) THEN
              DO ii = 1, data%n
                i = data%INDEX( ii )
                IF ( i > 0 ) THEN
                  DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                    j = data%INDEX( A%col( l ) )
                    IF ( j > 0 ) inform%semi_bandwidth =                       &
                      MAX( inform%semi_bandwidth, ABS( i - j ) )
                  END DO
                END IF
              END DO
            ELSE
              DO i = 1, data%n
                DO jj = A%ptr( i ), A%ptr( i + 1 ) - 1
                  inform%semi_bandwidth = MAX( inform%semi_bandwidth,          &
                    ABS( i - A%col( jj ) ) )
                END DO
              END DO
            END IF
          CASE ( 'COORDINATE' )
            inform%semi_bandwidth = 0
            IF ( data%sub_matrix ) THEN
              DO l = 1, A%ne
                i = data%INDEX( A%row( l ) )
                IF ( i > 0 ) THEN
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) inform%semi_bandwidth =                         &
                    MAX( inform%semi_bandwidth, ABS( i - j ) )
                END IF
              END DO
            ELSE
              DO l = 1, A%ne
                inform%semi_bandwidth = MAX( inform%semi_bandwidth,            &
                  ABS( A%row( l ) - A%col( l ) ) )
              END DO
            END IF
          END SELECT
        END IF
        inform%reordered_semi_bandwidth = inform%semi_bandwidth
      END IF
!write(6,*) inform%semi_bandwidth, SMT_get( A%type ), data%sub_matrix

!  --------------------------------
!  Stage 1 - Form the preliminary P
!  --------------------------------

      data%P%n = data%n_sub ; data%P%m = data%P%n

      IF ( printt ) WRITE( out, "( /, A, ' Form and factorize ' )" ) prefix
      SELECT CASE ( control%preconditioner )

!  a diagonal matrix will be used

      CASE ( preconditioner_diagonal )

!  allocate space to hold the diagonal

 !      IF ( control%new_structure ) THEN
        IF ( control%new_structure .OR. data%sub_matrix ) THEN
          array_name = 'psls: data%DIAG'
          CALL SPACE_resize_array( data%n_sub, data%DIAG,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910
        END IF

!  fit the data into the diagonal

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DIAGONAL' )
          IF ( data%sub_matrix ) THEN
            DO l = 1, data%n_sub
              data%DIAG( l ) = A%val( SUB( l ) )
            END DO
!           data%DIAG( : data%n_sub ) = A%val( SUB( : data%n_sub ) )
          ELSE
            data%DIAG = A%val( : data%n )
          END IF
        CASE ( 'DENSE' )
          IF ( data%sub_matrix ) THEN
            l = 0 ; j = 0
            DO i = 1, data%n
              l = l + i
              IF ( data%INDEX( i ) > 0 ) THEN
                j = j + 1 ; data%DIAG( j ) = A%val( l )
              END IF
            END DO
          ELSE
            l = 0
            DO i = 1, data%n
              l = l + i
              data%DIAG( i ) = A%val( l )
            END DO
          END IF
        CASE ( 'SPARSE_BY_ROWS' )
          data%DIAG = zero
          IF ( data%sub_matrix ) THEN
            DO ii = 1, data%n
              i = data%INDEX( ii )
              IF ( i > 0 ) THEN
                DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                  IF ( ii == A%col( l ) )                                      &
                    data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                END DO
              END IF
            END DO
          ELSE
            DO i = 1, data%n
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                IF ( i == j ) data%DIAG( i ) = data%DIAG( i ) + A%val( l )
              END DO
            END DO
          END IF
        CASE ( 'COORDINATE' )
          data%DIAG = zero
          IF ( data%sub_matrix ) THEN
            DO l = 1, A%ne
              ii = A%row( l ) ; i = data%INDEX( ii )
              IF ( i > 0 ) THEN
                IF ( ii == A%col( l ) )                                        &
                  data%DIAG( i ) = data%DIAG( i ) + A%val( l )
              END IF
            END DO
          ELSE
            DO l = 1, A%ne
              i = A%row( l ) ; j = A%col( l )
              IF ( i == j ) data%DIAG( i ) = data%DIAG( i ) + A%val( l )
            END DO
          END IF
        END SELECT

!  ensure that the diaginal is sufficiently positive

        data%DIAG( : data%n_sub ) =                                            &
          MAX( data%DIAG( : data%n_sub ), control%min_diagonal )

!  A band matrix will be used

      CASE ( preconditioner_band )

        IF ( control%new_structure ) THEN

!  compute the maximum semi-bandwidth required

          IF ( data%sub_matrix ) THEN

!  submatrix case

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DIAGONAL' )
              data%semi_bandwidth_used = 0
!             data%P%ne = data%n_sub
            CASE ( 'DENSE' )
              data%semi_bandwidth_used =                                       &
                MIN( control%semi_bandwidth, data%n_sub - 1 )
!             data%P%ne = ( data%n_sub * ( data%n_sub + 1 ) ) / 2
            CASE ( 'SPARSE_BY_ROWS' )
              data%semi_bandwidth_used = 0
!             data%P%ne = 0
              DO ii = 1, data%n
                i = data%INDEX( ii )
                IF ( i > 0 ) THEN
                  DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                    j = data%INDEX( A%col( l ) )
                    IF ( j > 0 ) THEN
                      ij = ABS( i - j )
                      IF ( ij <= control%semi_bandwidth ) THEN
                        data%semi_bandwidth_used =                             &
                          MAX( data%semi_bandwidth_used, ij )
!                       data%P%ne = data%P%ne + 1
                      END IF
                    END IF
                  END DO
                END IF
              END DO
            CASE ( 'COORDINATE' )
              data%semi_bandwidth_used = 0
!             data%P%ne = 0
              DO l = 1, A%ne
                i = data%INDEX( A%row( l ) )
                IF ( i > 0 ) THEN
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) THEN
                    ij = ABS( i - j )
                    IF ( ij <= control%semi_bandwidth ) THEN
                      data%semi_bandwidth_used =                               &
                        MAX( data%semi_bandwidth_used, ij )
!                     data%P%ne = data%P%ne + 1
                    END IF
                  END IF
                END IF
              END DO
            END SELECT
          ELSE

!  complete matrix case

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DIAGONAL' )
              data%semi_bandwidth_used = 0
!             data%P%ne = data%n_sub
            CASE ( 'DENSE' )
              data%semi_bandwidth_used =                                       &
                MIN( control%semi_bandwidth, data%n_sub - 1 )
!             data%P%ne = ( data%n_sub * ( data%n_sub + 1 ) ) / 2
            CASE ( 'SPARSE_BY_ROWS' )
              data%semi_bandwidth_used = 0
!             data%P%ne = 0
              DO i = 1, data%n
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  ij = ABS( i - A%col( l ) )
                  IF ( ij <= control%semi_bandwidth ) THEN
                    data%semi_bandwidth_used =                                 &
                      MAX( data%semi_bandwidth_used, ij )
!                   data%P%ne = data%P%ne + 1
                  END IF
                END DO
              END DO
            CASE ( 'COORDINATE' )
              data%semi_bandwidth_used = 0
!             data%P%ne = 0
              DO l = 1, A%ne
                i = A%row( l ) ; j = A%col( l )
                ij = ABS( i - j )
                IF ( ij <= control%semi_bandwidth ) THEN
                  data%semi_bandwidth_used = MAX( data%semi_bandwidth_used, ij )
!                 data%P%ne = data%P%ne + 1
                END IF
              END DO
            END SELECT
          END IF

!  allocate space to hold the band matrix in both band and co-ordinate form

          array_name = 'psls: data%DIAG'
          CALL SPACE_resize_array( data%n_sub, data%DIAG,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%OFFDIA'
          CALL SPACE_resize_array( data%semi_bandwidth_used, data%n_sub,       &
              data%OFFDIA,                                                     &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%PERT'
          CALL SPACE_resize_array( data%n_sub, data%PERT,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         array_name = 'psls: data%P%row'
!         CALL SPACE_resize_array( data%P%ne, data%P%row,                      &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         array_name = 'psls: data%P%col'
!         CALL SPACE_resize_array( data%P%ne, data%P%col,                      &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         array_name = 'psls: data%P%val'
!         CALL SPACE_resize_array( data%P%ne, data%P%val,                      &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         CALL SMT_put( data%P%type, 'COORDINATE', inform%alloc_status )
!         IF ( inform%alloc_status /= 0 ) THEN
!           inform%status = - 1 ; GO TO 910 ; END IF

        END IF

!  initialize an empty band

        data%DIAG = zero ; data%OFFDIA = zero

!  fit the data into the band

        IF ( data%sub_matrix ) THEN

!  submatrix case

!         data%P%ne = 0
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            data%DIAG = A%val( SUB( : data%n_sub ) )
!           DO ii = 1, data%n_sub
!             i = SUB( ii )
!             data%P%ne = data%P%ne + 1
!             IF ( control%new_structure ) THEN
!               data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = i
!             END IF
!           END DO
!           data%P%val( : data%n_sub ) = A%val( SUB( : data%n_sub ) )
          CASE ( 'DENSE' )
            l = 0
            DO ii = 1, data%n
              i = data%INDEX( ii )
              IF ( i > 0 ) THEN
                DO jj = 1, ii
                  j = data%INDEX( jj )
                  l = l + 1
                  IF ( j > 0 ) THEN
                    ij = ABS( i - j )
                    IF ( ij <= data%semi_bandwidth_used ) THEN
                      IF ( i == j ) THEN
                        data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                      ELSE
                        k = MIN( i, j )
                        data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                      END IF
!                     data%P%ne = data%P%ne + 1
!                     IF ( control%new_structure ) THEN
!                       data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne) = j
!                     END IF
!                     data%P%val( data%P%ne ) = A%val( l )
                    END IF
                  END IF
                END DO
              ELSE
                l = l + ii
              END IF
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO ii = 1, data%n
              i = data%INDEX( ii )
              IF ( i > 0 ) THEN
                DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) THEN
                    ij = ABS( i - j )
                    IF ( ij <= data%semi_bandwidth_used ) THEN
                      IF ( i == j ) THEN
                        data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                      ELSE
                        k = MIN( i, j )
                        data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                      END IF
!                     data%P%ne = data%P%ne + 1
!                     IF ( control%new_structure ) THEN
!                       data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne) = j
!                     END IF
!                     data%P%val( data%P%ne ) = A%val( l )
                    END IF
                  END IF
                END DO
              END IF
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = data%INDEX( A%row( l ) )
              IF ( i > 0 ) THEN
                j = data%INDEX( A%col( l ) )
                IF ( j > 0 ) THEN
                  ij = ABS( i - j )
                  IF ( ij <= data%semi_bandwidth_used ) THEN
                    IF ( i == j ) THEN
                      data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                    ELSE
                      k = MIN( i, j )
                      data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                    END IF
!                   data%P%ne = data%P%ne + 1
!                   IF ( control%new_structure ) THEN
!                     data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
!                   END IF
!                   data%P%val( data%P%ne ) = A%val( l )
                  END IF
                END IF
              END IF
            END DO
          END SELECT
        ELSE

!  complete matrix case

!         data%P%ne = 0
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            data%DIAG = A%val( : data%n )
!           DO i = 1, data%n
!             data%P%ne = data%P%ne + 1
!             IF ( control%new_structure ) THEN
!               data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = i
!             END IF
!           END DO
!           data%P%val( : data%n ) = A%val( : data%n )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, data%n
              DO j = 1, i
                l = l + 1
                ij = ABS( i - j )
                IF ( ij <= data%semi_bandwidth_used ) THEN
                  IF ( i == j ) THEN
                    data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                  ELSE
                    k = MIN( i, j )
                    data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                  END IF
!                 data%P%ne = data%P%ne + 1
!                 IF ( control%new_structure ) THEN
!                   data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
!                 END IF
!                 data%P%val( data%P%ne ) = A%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%n
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                ij = ABS( i - j )
                IF ( ij <= data%semi_bandwidth_used ) THEN
                  IF ( i == j ) THEN
                    data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                  ELSE
                    k = MIN( i, j )
                    data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                  END IF
!                 data%P%ne = data%P%ne + 1
!                  IF ( control%new_structure ) THEN
!                   data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
!                 END IF
!                 data%P%val( data%P%ne ) = A%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l ) ; j = A%col( l )
              ij = ABS( i - j )
              IF ( ij <= data%semi_bandwidth_used ) THEN
                IF ( i == j ) THEN
                  data%DIAG( i ) = data%DIAG( i ) + A%val( l )
                ELSE
                  k = MIN( i, j )
                  data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                END IF
!               data%P%ne = data%P%ne + 1
!               IF ( control%new_structure ) THEN
!                 data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
!               END IF
!               data%P%val( data%P%ne ) = A%val( l )
              END IF
            END DO
          END SELECT
        END IF

!  A re-ordered band matrix will be used

      CASE ( preconditioner_reordered_band )

        IF ( control%new_structure ) THEN

!  compute the number of nonzeros in A

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            data%p_ne = data%n_sub
          CASE ( 'DENSE' )
            data%p_ne = ( data%n_sub * ( data%n_sub + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            IF ( data%sub_matrix ) THEN
              data%p_ne = 0
              DO ii = 1, data%n
                i = data%INDEX( ii )
                IF ( i > 0 ) THEN
                  DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                    IF ( data%INDEX( A%col( l ) ) > 0) data%p_ne = data%p_ne + 1
                  END DO
                END IF
              END DO
            ELSE
              data%p_ne = A%ptr( data%n + 1 ) - 1
            END IF
          CASE ( 'COORDINATE' )
            IF ( data%sub_matrix ) THEN
              data%p_ne = 0
              DO l = 1, A%ne
                IF ( data%INDEX( A%row( l ) ) > 0 .AND.                        &
                     data%INDEX( A%col( l ) ) > 0 ) data%p_ne = data%p_ne + 1
              END DO
            ELSE
              data%p_ne = A%ne
            END IF
          END SELECT

!  allocate workspace

          data%mc61_lirn = 2 * data%p_ne
          data%mc61_liw = 8 * data%n_sub + 2

          array_name = 'psls: data%P_row'
          CALL SPACE_resize_array( data%mc61_lirn, data%P_row,                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_col'
          CALL SPACE_resize_array( data%p_ne, data%P_col,                      &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_colptr'
          CALL SPACE_resize_array( data%n_sub + 1, data%P_colptr,              &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%PERM'
          CALL SPACE_resize_array( data%n_sub, data%PERM,                      &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%IW'
          CALL SPACE_resize_array( data%mc61_liw, data%IW,                     &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%W'
          CALL SPACE_resize_array( data%n_sub, data%W,                         &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  convert the matrix to co-ordinate form ...

          data%p_ne = 0
          IF ( data%sub_matrix ) THEN

!  submatrix case

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DIAGONAL' )
              DO ii = 1, data%n_sub
                i = SUB( ii )
                data%p_ne = data%p_ne + 1
                data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = i
              END DO
            CASE ( 'DENSE' )
              l = 0
              DO ii = 1, data%n
                i = data%INDEX( ii )
                IF ( i > 0 ) THEN
                  DO jj = 1, ii
                    j = data%INDEX( jj )
                    l = l + 1
                    IF ( j > 0 ) THEN
                      data%p_ne = data%p_ne + 1
                      IF ( i >= j ) THEN
                        data%P_row( data%p_ne ) = i; data%P_col( data%p_ne ) = j
                      ELSE
                        data%P_row( data%p_ne ) = j; data%P_col( data%p_ne ) = i
                      END IF
                    END IF
                  END DO
                ELSE
                  l = l + ii
                END IF
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO ii = 1, data%n
                i = data%INDEX( ii )
                IF ( i > 0 ) THEN
                  DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                    j = data%INDEX( A%col( l ) )
                    IF ( j > 0 ) THEN
                      data%p_ne = data%p_ne + 1
                      IF ( i >= j ) THEN
                        data%P_row( data%p_ne ) = i; data%P_col( data%p_ne ) = j
                      ELSE
                        data%P_row( data%p_ne ) = j; data%P_col( data%p_ne ) = i
                      END IF
                    END IF
                  END DO
                END IF
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                i = data%INDEX( A%row( l ) )
                IF ( i > 0 ) THEN
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) THEN
                    data%p_ne = data%p_ne + 1
                    IF ( i >= j ) THEN
                      data%P_row( data%p_ne ) = i; data%P_col( data%p_ne ) = j
                    ELSE
                      data%P_row( data%p_ne ) = j; data%P_col( data%p_ne ) = i
                    END IF
                  END IF
                END IF
              END DO
            END SELECT
          ELSE

!  complete matrix case

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DIAGONAL' )
              data%DIAG = A%val( : data%n )
              DO i = 1, data%n
                data%p_ne = data%p_ne + 1
                data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = i
              END DO
            CASE ( 'DENSE' )
              l = 0
              DO i = 1, data%n
                DO j = 1, i
                  l = l + 1
                  data%p_ne = data%p_ne + 1
                  IF ( i >= j ) THEN
                    data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = j
                  ELSE
                    data%P_row( data%p_ne ) = j ; data%P_col( data%p_ne ) = i
                  END IF
                END DO
              END DO
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, data%n
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l )
                  data%p_ne = data%p_ne + 1
                  IF ( i >= j ) THEN
                    data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = j
                  ELSE
                    data%P_row( data%p_ne ) = j ; data%P_col( data%p_ne ) = i
                  END IF
                END DO
              END DO
            CASE ( 'COORDINATE' )
              DO l = 1, A%ne
                i = A%row( l ) ; j = A%col( l )
                data%p_ne = data%p_ne + 1
                IF ( i >= j ) THEN
                  data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = j
                ELSE
                  data%P_row( data%p_ne ) = j ; data%P_col( data%p_ne ) = i
                END IF
              END DO
            END SELECT
          END IF

!  ... and now convert the matrix to column form

          CALL SORT_reorder_by_cols( data%n_sub, data%n_sub, data%p_ne,        &
                                     data%P_row, data%P_col, data%p_ne,        &
                                     data%P_colptr, data%n_sub + 1,            &
                                     data%IW, data%mc61_liw, 0, 0, i )
          IF ( i > 0 ) THEN
            write(6,"( ' error ', I0, ' from SORT_reorder_by_cols' )" ) i
            stop
          ELSE IF ( i < 0 ) THEN
            write(6,"( ' warning ', I0, ' from SORT_reorder_by_cols' )" ) i
          END IF

!  find a bandwidth-reducing ordering

          IF ( control%print_level <= 0 .OR. control%out <= 0 )                &
            data%mc61_ICNTL( 1 : 2 ) = - 1
          CALL MC61AD( 2, data%n_sub, data%mc61_lirn,                          &
                       data%P_row, data%P_colptr, data%PERM, data%mc61_liw,    &
                       data%IW, data%W, data%mc61_ICNTL, data%mc61_CNTL,       &
                       inform%mc61_info, inform%mc61_rinfo )

!write(6,*) ' perm ', data%PERM
          IF ( inform%mc61_info( 1 ) == GALAHAD_unavailable_option ) THEN
            IF ( control%print_level > 0 .AND. control%out > 0 )               &
              WRITE( control%out, "( A, ' mc61 is not available ' )" ) prefix
            inform%status = GALAHAD_error_unknown_solver ; GO TO 930
          END IF
          inform%reordered_semi_bandwidth = INT( inform%mc61_rinfo( 7 ) ) - 1
!write(6,*) ' reordered semi-bandwidth ', inform%semi_bandwidth

!  compute the maximum semi-bandwidth required

          IF ( data%sub_matrix ) THEN

!  submatrix case

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DIAGONAL' )
              data%semi_bandwidth_used = 0
            CASE ( 'DENSE' )
              data%semi_bandwidth_used =                                       &
                MIN( control%semi_bandwidth, data%n_sub - 1 )
            CASE ( 'SPARSE_BY_ROWS' )
              data%semi_bandwidth_used = 0
              DO iii = 1, data%n
                i = data%INDEX( iii )
                IF ( i > 0 ) THEN
                  ii = data%PERM( i )
                  DO l = A%ptr( iii ), A%ptr( iii + 1 ) - 1
                    j = data%INDEX( A%col( l ) )
                    IF ( j > 0 ) THEN
                      jj = data%PERM( j )
                      ij = ABS( ii - jj )
                      IF ( ij <= control%semi_bandwidth ) THEN
                        data%semi_bandwidth_used =                             &
                          MAX( data%semi_bandwidth_used, ij )
                      END IF
                    END IF
                  END DO
                END IF
              END DO
            CASE ( 'COORDINATE' )
              data%semi_bandwidth_used = 0
              DO l = 1, A%ne
                i = data%INDEX( A%row( l ) )
                IF ( i > 0 ) THEN
                  ii = data%PERM( i )
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) THEN
                    jj = data%PERM( j )
                    ij = ABS( ii - jj )
                    IF ( ij <= control%semi_bandwidth ) THEN
                      data%semi_bandwidth_used =                               &
                        MAX( data%semi_bandwidth_used, ij )
                    END IF
                  END IF
                END IF
              END DO
            END SELECT

          ELSE

!  complete matrix case

            SELECT CASE ( SMT_get( A%type ) )
            CASE ( 'DIAGONAL' )
              data%semi_bandwidth_used = 0
            CASE ( 'DENSE' )
              data%semi_bandwidth_used =                                       &
                MIN( control%semi_bandwidth, data%n_sub - 1 )
            CASE ( 'SPARSE_BY_ROWS' )
              data%semi_bandwidth_used = 0
              DO i = 1, data%n
                ii = data%PERM( i )
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = A%col( l ) ; jj = data%PERM( j )
                  ij = ABS( ii - jj )
                  IF ( ij <= control%semi_bandwidth ) THEN
                    data%semi_bandwidth_used =                                 &
                      MAX( data%semi_bandwidth_used, ij )
                  END IF
                END DO
              END DO
            CASE ( 'COORDINATE' )
              data%semi_bandwidth_used = 0
              DO l = 1, A%ne
                i = A%row( l ) ; ii = data%PERM( i )
                j = A%col( l ) ; jj = data%PERM( j )
                ij = ABS( ii - jj )
                IF ( ij <= control%semi_bandwidth ) THEN
                  data%semi_bandwidth_used = MAX( data%semi_bandwidth_used, ij )
                END IF
              END DO
            END SELECT
          END IF

!  allocate space to hold the band matrix in both band and co-ordinate form

          array_name = 'psls: data%DIAG'
          CALL SPACE_resize_array( data%n_sub, data%DIAG,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%OFFDIA'
          CALL SPACE_resize_array( data%semi_bandwidth_used, data%n_sub,       &
              data%OFFDIA,                                                     &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%PERT'
          CALL SPACE_resize_array( data%n_sub, data%PERT,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         array_name = 'psls: P%row'
!         CALL SPACE_resize_array( data%P%ne, data%P%row,                      &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         array_name = 'psls: P%col'
!         CALL SPACE_resize_array( data%P%ne, data%P%col,                      &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         array_name = 'psls: P%val'
!         CALL SPACE_resize_array( data%P%ne, data%P%val,                      &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

!         CALL SMT_put( data%P%type, 'COORDINATE', inform%alloc_status )
!         IF ( inform%alloc_status /= 0 ) THEN
!           inform%status = - 1 ; GO TO 910 ; END IF

        END IF

!  initialize an empty band

        data%DIAG = zero ; data%OFFDIA = zero

!  fit the data into the band

        IF ( data%sub_matrix ) THEN

!  submatrix case

!         data%P%ne = 0
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            data%DIAG = A%val( SUB( : data%n_sub ) )
!           DO ii = 1, data%n_sub
!             i = SUB( ii )
!             data%P%ne = data%P%ne + 1
!             IF ( control%new_structure ) THEN
!               data%P%row( data%P%ne ) = ii ; data%P%col( data%P%ne ) = ii
!             END IF
!           END DO
!           data%P%val( : data%n_sub ) = A%val( SUB( : data%n_sub ) )
          CASE ( 'DENSE' )
            l = 0
            DO iii = 1, data%n
              i = data%INDEX( iii )
              IF ( i > 0 ) THEN
                ii = data%PERM( i )
                DO jjj = 1, iii
                  j = data%INDEX( jjj )
                  l = l + 1
                  IF ( j > 0 ) THEN
                    jj = data%PERM( j )
                    ij = ABS( ii - jj )
                    IF ( ij <= data%semi_bandwidth_used ) THEN
                      IF ( ii == jj ) THEN
                        data%DIAG( ii ) = data%DIAG( ii ) + A%val( l )
                      ELSE
                        k = MIN( ii, jj )
                        data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                      END IF
!                     data%P%ne = data%P%ne + 1
!                     IF ( control%new_structure ) THEN
!                       data%P%row( data%P%ne) = ii; data%P%col( data%P%ne) = jj
!                     END IF
!                     data%P%val( data%P%ne ) = A%val( l )
                    END IF
                  END IF
                END DO
              ELSE
                l = l + iii
              END IF
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO iii = 1, data%n
              i = data%INDEX( iii )
              IF ( i > 0 ) THEN
                ii = data%PERM( i )
                DO l = A%ptr( iii ), A%ptr( iii + 1 ) - 1
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) THEN
                    jj = data%PERM( j )
                    ij = ABS( ii - jj )
                    IF ( ij <= data%semi_bandwidth_used ) THEN
                      IF ( ii == jj ) THEN
                        data%DIAG( ii ) = data%DIAG( ii ) + A%val( l )
                      ELSE
                        k = MIN( ii, jj )
                        data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                      END IF
!                     data%P%ne = data%P%ne + 1
!                     IF ( control%new_structure ) THEN
!                       data%P%row( data%P%ne) = ii; data%P%col( data%P%ne) = jj
!                     END IF
!                     data%P%val( data%P%ne ) = A%val( l )
                    END IF
                  END IF
                END DO
              END IF
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = data%INDEX( A%row( l ) )
              IF ( i > 0 ) THEN
                ii = data%PERM( i )
                j = data%INDEX( A%col( l ) )
                IF ( j > 0 ) THEN
                  jj = data%PERM( j )
                  ij = ABS( ii - jj )
                  IF ( ij <= data%semi_bandwidth_used ) THEN
                    IF ( ii == jj ) THEN
                      data%DIAG( ii ) = data%DIAG( ii ) + A%val( l )
                    ELSE
                      k = MIN( ii, jj )
                      data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                    END IF
!                   data%P%ne = data%P%ne + 1
!                   IF ( control%new_structure ) THEN
!                     data%P%row( data%P%ne ) = ii; data%P%col( data%P%ne ) = jj
!                   END IF
!                   data%P%val( data%P%ne ) = A%val( l )
                  END IF
                END IF
              END IF
            END DO
          END SELECT

        ELSE

!  complete matrix case

!         data%P%ne = 0
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            data%DIAG = A%val( : data%n )
!           DO i = 1, data%n
!             data%P%ne = data%P%ne + 1
!             IF ( control%new_structure ) THEN
!               data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = i
!             END IF
!           END DO
!           data%P%val( : data%n ) = A%val( : data%n )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, data%n
              ii = data%PERM( i )
              DO j = 1, i
                l = l + 1
                jj = data%PERM( j )
                ij = ABS( ii - jj )
                IF ( ij <= data%semi_bandwidth_used ) THEN
                  IF ( ii == jj ) THEN
                    data%DIAG( ii ) = data%DIAG( ii ) + A%val( l )
                  ELSE
                    k = MIN( ii, jj )
                    data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                  END IF
!                 data%P%ne = data%P%ne + 1
!                 IF ( control%new_structure ) THEN
!                   data%P%row( data%P%ne ) = ii ; data%P%col( data%P%ne ) = jj
!                 END IF
!                 data%P%val( data%P%ne ) = A%val( l )
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%n
              ii = data%PERM( i )
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                jj = data%PERM( j )
                ij = ABS( ii - jj )
                IF ( ij <= data%semi_bandwidth_used ) THEN
                  IF ( ii == jj ) THEN
                    data%DIAG( ii ) = data%DIAG( ii ) + A%val( l )
                  ELSE
                    k = MIN( ii, jj )
                    data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                  END IF
!                 data%P%ne = data%P%ne + 1
!                  IF ( control%new_structure ) THEN
!                   data%P%row( data%P%ne ) = ii ; data%P%col( data%P%ne ) = jj
!                 END IF
!                 data%P%val( data%P%ne ) = A%val( l )
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l ) ; ii = data%PERM( i )
              j = A%col( l ) ; jj = data%PERM( j )
              ij = ABS( ii - jj )
              IF ( ij <= data%semi_bandwidth_used ) THEN
                IF ( ii == jj ) THEN
                  data%DIAG( ii ) = data%DIAG( ii ) + A%val( l )
                ELSE
                  k = MIN( ii, jj )
                  data%OFFDIA( ij, k ) = data%OFFDIA( ij, k ) + A%val( l )
                END IF
!               data%P%ne = data%P%ne + 1
!               IF ( control%new_structure ) THEN
!                 data%P%row( data%P%ne ) = ii ; data%P%col( data%P%ne ) = jj
!               END IF
!               data%P%val( data%P%ne ) = A%val( l )
              END IF
            END DO
          END SELECT
        END IF

!  The full matrix will be used

      CASE ( preconditioner_full_se, preconditioner_full_gmps,                 &
             preconditioner_incomplete_lm, preconditioner_incomplete_mi28 )

!  compute the space required

        IF ( control%new_structure ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            data%P%ne = data%n_sub
          CASE ( 'DENSE' )
            data%P%ne = ( data%n_sub * ( data%n_sub - 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            IF ( data%sub_matrix ) THEN
              data%P%ne = 0
              DO ii = 1, data%n
                i = data%INDEX( ii )
                IF ( i > 0 ) THEN
                  DO l = A%ptr( ii ), A%ptr( ii + 1 ) - 1
                    IF ( data%INDEX( A%col( l ) ) > 0) data%P%ne = data%P%ne + 1
                  END DO
                END IF
              END DO
            ELSE
              data%P%ne = A%ptr( data%n + 1 ) - 1
            END IF
          CASE ( 'COORDINATE' )
            IF ( data%sub_matrix ) THEN
              data%P%ne = 0
              DO l = 1, A%ne
                IF ( data%INDEX( A%row( l ) ) > 0 ) THEN
                  IF ( data%INDEX( A%col( l ) ) > 0 ) data%P%ne = data%P%ne + 1
                END IF
              END DO
            ELSE
              data%P%ne = A%ne
            END IF
          END SELECT

!  allocate space to hold the full matrix in co-ordinate form

          array_name = 'psls: data%P%row'
          CALL SPACE_resize_array( data%P%ne, data%P%row,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P%col'
          CALL SPACE_resize_array( data%P%ne, data%P%col,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P%val'
          CALL SPACE_resize_array( data%P%ne, data%P%val,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SMT_put( data%P%type, 'COORDINATE', inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN
            inform%status = - 1 ; GO TO 910 ; END IF

        END IF

!  fit the data into the coordinate storage scheme provided

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DIAGONAL' )
          IF ( data%sub_matrix ) THEN
            IF ( control%new_structure ) THEN
              DO i = 1, data%n_sub
                data%P%row( i ) = i ; data%P%col( i ) = i
              END DO
            END IF
            data%P%val( : data%P%ne ) = A%val( SUB( : data%n_sub ) )
          ELSE
            IF ( control%new_structure ) THEN
              DO i = 1, data%n
                data%P%row( i ) = i ; data%P%col( i ) = i
              END DO
            END IF
            data%P%val( : A%ne ) = A%val( : A%ne )
          END IF
        CASE ( 'DENSE' )
          IF ( data%sub_matrix ) THEN
            l = 0 ; data%P%ne = 0
            DO ii = 1, data%n
              i = data%INDEX( ii )
              IF ( i > 0 ) THEN
                DO jj = 1, ii
                  l = l + 1
                  j = data%INDEX( jj )
                  IF ( j > 0 ) THEN
                    data%P%ne = data%P%ne + 1
                    IF ( control%new_structure ) THEN
                      data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
                    END IF
                    data%P%val( data%P%ne ) = A%val( l )
                  END IF
                END DO
              ELSE
                l = l + ii
              END IF
            END DO
          ELSE
            IF ( control%new_structure ) THEN
              l = 0
              DO i = 1, data%n
                DO j = 1, i
                  l = l + 1
                  data%P%row( l ) = i ; data%P%col( l ) = j
                END DO
              END DO
            END IF
            data%P%val( : A%ne ) = A%val( : A%ne )
          END IF
        CASE ( 'SPARSE_BY_ROWS' )
          IF ( data%sub_matrix ) THEN
            data%P%ne = 0
            DO ii = 1, data%n
              i = data%INDEX( ii )
              IF ( i > 0 ) THEN
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  j = data%INDEX( A%col( l ) )
                  IF ( j > 0 ) THEN
                    data%P%ne = data%P%ne + 1
                    IF ( control%new_structure ) THEN
                      data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
                    END IF
                    data%P%val( data%P%ne ) = A%val( l )
                  END IF
                END DO
              END IF
            END DO
          ELSE
            IF ( control%new_structure ) THEN
              DO i = 1, data%n
                DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                  data%P%row( l ) = i
                  data%P%col( l ) = A%col( l )
                END DO
              END DO
            END IF
            data%P%val( 1 : A%ne ) = A%val( : A%ne )
          END IF
        CASE ( 'COORDINATE' )
          IF ( data%sub_matrix ) THEN
            data%P%ne = 0
            DO l = 1, A%ne
              i = data%INDEX( A%row( l ) )
              IF ( i > 0 ) THEN
                j = data%INDEX( A%col( l ) )
                IF ( j > 0 ) THEN
                  data%P%ne = data%P%ne + 1
                  IF ( control%new_structure ) THEN
                    data%P%row( data%P%ne ) = i ; data%P%col( data%P%ne ) = j
                  END IF
                  data%P%val( data%P%ne ) = A%val( l )
                END IF
              END IF
            END DO
          ELSE
            IF ( control%new_structure ) THEN
              data%P%row( 1 : A%ne ) = A%row( : A%ne )
              data%P%col( 1 : A%ne ) = A%col( : A%ne )
            END IF
            data%P%val( 1 : A%ne ) = A%val( : A%ne )
          END IF
        END SELECT

!  Choose initial values for the control parameters

!        IF ( control%print_level > 4 ) THEN
!          control%SLS_control%out = out ; control%SLS_control%error = out
!          control%SLS_control%print_level = 1
!        ELSE
!          control%SLS_control%out = 0 ; control%SLS_control%error = 0
!          control%SLS_control%print_level = 0
!        END IF

!  One of the other cases will be used ... eventually

      CASE DEFAULT
        WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )           &
          control%preconditioner
        inform%status = GALAHAD_preconditioner_unknown
        RETURN
      END SELECT

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%assemble =inform%time%assemble + time_now - time_start
      inform%time%clock_assemble                                               &
        = inform%time%clock_assemble + clock_now - clock_start
      IF ( printt ) WRITE( out, "( /, A,  ' time( assembly ) = ', F10.2 )" )   &
        prefix, time_now - time_start

      inform%semi_bandwidth_used = data%semi_bandwidth_used
!     write(6,*) inform%semi_bandwidth, inform%semi_bandwidth_used

!  -------------------------------------------
!  Stage 2 - Factorize (and possibly modify) P
!  -------------------------------------------

      SELECT CASE ( control%preconditioner )

!  The diagonal matrix needs not be factorized

      CASE ( preconditioner_diagonal )

!  The band or re-ordered band matrix will be factorized

      CASE ( preconditioner_band, preconditioner_reordered_band )

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL BAND_factor( data%n_sub, data%semi_bandwidth_used, data%DIAG,     &
                          data%OFFDIA, data%semi_bandwidth_used, i,            &
                          PERT = data%PERT, n_pert = data%n_pert )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize                                            &
          = inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, time_now - time_record

!  Incorporate any diagonal perturbations made into P

        data%perturbed = data%n_pert > 0
        inform%perturbed = data%perturbed
        IF ( data%perturbed ) data%DIAG( : data%n_sub ) =                      &
          data%DIAG( : data%n_sub ) + data%PERT( : data%n_sub )

        IF ( printi ) WRITE( out, "( /, A, ' -- Preconditioner formed.', /, A, &
       &           ' -- order = ', I0, ', semi-bandwidth = ', I0,              &
       &           ', true semi-bandwith = ', I0 )" ) prefix, prefix,          &
                   data%n_sub, data%semi_bandwidth_used, inform%semi_bandwidth

!  The full matrix will be factorized with Schnabel-Eskow modifications

      CASE ( preconditioner_full_se )

!  initialize solver-specific data

        data%SLS_control = control%SLS_control
        CALL SLS_INITIALIZE( control%definite_linear_solver,                   &
                             data%SLS_data, data%SLS_control,                  &
                             inform%SLS_inform )

!  Choose the pivot sequence for the factorization by analyzing the
!  sparsity pattern of P

        data%SLS_control%pivot_control = 4
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_analyse( data%P, data%SLS_data, data%SLS_control,             &
                          inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse                                              &
          = inform%time%clock_analyse + clock_now - clock_record
        IF ( printt ) WRITE( out, 2000 ) prefix, time_now - time_record

!  Test that the analysys succeeded

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printi ) WRITE( out, "( /, A, ' error return from ',            &
         &  'SLS_analyse: status = ', I0 )") prefix, inform%SLS_inform%status
          inform%status = GALAHAD_error_analysis ;  GO TO 930 ; END IF
         predicted = inform%SLS_inform%entries_in_factors

!  Factorize the matrix P, using the Schnabel-Eskow modified Cholesky method

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( data%P, data%SLS_data, data%SLS_control,           &
                             inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize                                            &
          = inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, time_now - time_record

!  Test that the factorization succeeded

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printi ) WRITE( out, "( /, A, ' error return from ',            &
         &  'SLS_factorize: status = ', I0 )" ) prefix,                        &
             inform%SLS_inform%status
          inform%status = GALAHAD_error_factorization ;  GO TO 930 ; END IF

        data%perturbed = inform%SLS_inform%first_modified_pivot > 0
        inform%perturbed = data%perturbed

!  Calculate the perturbation made to the diagonals of A

        IF ( data%perturbed ) THEN
          array_name = 'psls: data%PERT'
          CALL SPACE_resize_array( data%n_sub, data%PERT,                      &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SLS_ENQUIRE( data%SLS_data, inform%SLS_inform,                  &
                            PERTURBATION = data%PERT )
          inform%SLS_inform%largest_modified_pivot =                           &
            MAXVAL( ABS( data%PERT( : data%n_sub ) ) )
        ELSE
          inform%SLS_inform%largest_modified_pivot = zero
        END IF

        IF ( printi ) WRITE( out, "( /, A,                                     &
       &   ' Preconditioner: SE full factorization using ', A,                 &
       &     /, A, '  diagonals are perturbed by at most',                     &
       &     ES11.4, /, A, '  order of preconditioner         = ', I0,         &
       &     /, A, '  # nonzeros in preconditioner    = ', I0,                 &
       &     /, A, '  predicted # nonzeros in factors = ', I0,                 &
       &     /, A, '  actual    # nonzeros in factors = ', I0 )" )             &
           prefix, control%definite_linear_solver,                             &
           prefix, inform%SLS_inform%largest_modified_pivot,                   &
           prefix, data%n_sub,                                                 &
           prefix, data%P%ne, prefix, predicted,                               &
           prefix, inform%SLS_inform%entries_in_factors

!  Record the relative fill-in

        df = inform%SLS_inform%entries_in_factors ; de = data%P%ne
        IF ( data%P%ne > 0 ) inform%fill_in_ratio = df / de

!  The full matrix will be factorized with GMPS modifications

      CASE ( preconditioner_full_gmps )

!  initialize solver-specific data

        data%SLS_control = control%SLS_control
        CALL SLS_INITIALIZE( control%definite_linear_solver,                   &
                             data%SLS_data, data%SLS_control,                  &
                             inform%SLS_inform )

!  Choose the pivot sequence for the factorization by analyzing the
!  sparsity pattern of P

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_analyse( data%P, data%SLS_data, data%SLS_control,             &
                          inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%analyse = inform%time%analyse + time_now - time_record
        inform%time%clock_analyse                                              &
          = inform%time%clock_analyse + clock_now - clock_record
        IF ( printt ) WRITE( out, 2000 ) prefix, time_now - time_record

!  Test that the analysys succeeded

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printi ) WRITE( out, "( /, A, ' error return from ',            &
         &  'SLS_analyse: status = ', I0 )") prefix, inform%SLS_inform%status
          inform%status = GALAHAD_error_analysis ;  GO TO 930 ; END IF
         predicted = inform%SLS_inform%entries_in_factors

!  Factorize the matrix P, using the Gill-Murray-Ponceleon-Saunders
!  modification to the symmetric indefinite factorization

        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )
        CALL SLS_factorize( data%P, data%SLS_data, data%SLS_control,           &
                             inform%SLS_inform )
        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize                                            &
          = inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, time_now - time_record

!  Test that the factorization succeeded

        IF ( inform%SLS_inform%status < 0 ) THEN
          IF ( printi ) WRITE( out, "( /, A, ' error return from ',            &
         &  'SLS_factorize: status = ', I0 )" ) prefix,                        &
             inform%SLS_inform%status
          inform%status = GALAHAD_error_factorization ;  GO TO 930 ; END IF

!  Calculate the perturbation made to the diagonals of A

        array_name = 'psls: data%PERM'
        CALL SPACE_resize_array( data%n_sub, data%PERM,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%D'
        CALL SPACE_resize_array( 2, data%n_sub, data%D,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  Modify the pivots

        IF ( printt ) CALL CPU_TIME( time_record )
        CALL PSLS_gmps( data%n_sub, inform%SLS_inform%rank,                    &
                        data%SLS_data, inform%SLS_inform, inform%neg1,         &
                        inform%neg2, data%PERM, data%D )
        IF ( printt ) THEN
          CALL CPU_TIME( time_now )
          WRITE( out, "( /, A, ' time( PSLS_gmps ) = ', F10.2 )" )             &
            prefix, time_now - time_record
        END IF

        IF ( printi ) WRITE( out, "( /, A,                                     &
       &   ' Preconditioner: GMPS full factorization using ', A,               &
       &     /, A, '  order of preconditioner         = ', I0,                 &
       &     /, A, '  # nonzeros in preconditioner    = ', I0,                 &
       &     /, A, '  predicted # nonzeros in factors = ', I0,                 &
       &     /, A, '  actual    # nonzeros in factors = ', I0,                 &
       &     /, A, '  # negative 1 x 1 block pivots   = ', I0,                 &
       &     /, A, '  # negative 2 x 2 block pivots   = ', I0 )" )             &
           prefix, control%definite_linear_solver,                             &
           prefix, data%n_sub, prefix, data%P%ne, prefix, predicted,           &
           prefix, inform%SLS_inform%entries_in_factors, prefix,               &
           inform%neg1, inform%neg2

!  Record the relative fill-in

        df = inform%SLS_inform%entries_in_factors ; de = data%P%ne
        IF ( data%P%ne > 0 ) inform%fill_in_ratio = df / de

!  The Lin-More' incomplete factorization of the full matrix will be used

      CASE ( preconditioner_incomplete_lm )
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

!  Allocate workspace arrays for ICFS

        array_name = 'psls: data%IW'
        CALL SPACE_resize_array( 3 * data%n_sub, data%IW,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%DIAG'
        CALL SPACE_resize_array( data%n_sub, data%DIAG,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%W'
        CALL SPACE_resize_array( data%n_sub, data%W,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%P_row'
        CALL SPACE_resize_array( data%n_sub, data%P_row,                       &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%P_col'
        CALL SPACE_resize_array( data%n_sub, data%P_col,                       &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%P_diag'
        CALL SPACE_resize_array( data%n_sub, data%P_diag,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%P_colptr'
        CALL SPACE_resize_array( data%n_sub + 1, data%P_colptr,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%L_diag'
        CALL SPACE_resize_array( data%n_sub, data%L_diag,                      &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%L_colptr'
        CALL SPACE_resize_array( data%n_sub + 1, data%L_colptr,                &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  Count how many off-diagonal entries there are

         data%p_ne = COUNT( data%P%row( : data%P%ne ) /=                       &
                            data%P%col( : data%P%ne ) )

!  Decide how much room is available for the incomplete factorization.

         icfact = MAX( 0, control%icfs_vectors )
         data%l_ne = data%p_ne + data%n_sub * icfact

!  Allocate space to hold the off diagonals and factors in comprossed
!  column format

        array_name = 'psls: data%P_offd'
        CALL SPACE_resize_array( data%p_ne, data%P_offd,                       &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%P_row'
        CALL SPACE_resize_array( data%p_ne, data%P_row,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%L_offd'
        CALL SPACE_resize_array( data%l_ne, data%L_offd,                       &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

        array_name = 'psls: data%L_row'
        CALL SPACE_resize_array( data%l_ne, data%L_row,                        &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  reorder H so that its lower triangle is stored in compressed column format.
!  First count how many nonzeros there are in each column

         data%IW( : data%n_sub ) = 0
         DO l = 1, data%P%ne
          j = data%P%col( l )
          IF ( j /= data%P%row( l ) ) data%IW( j ) = data%IW( j ) + 1
         END DO

!  now find the starting address for each columm in the storage format

         data%P_colptr( 1 ) = 1
         DO i = 1, data%n_sub
           data%P_colptr( i + 1 ) = data%P_colptr( i ) + data%IW( i )
         END DO

!  finally copy the data into its correct position ...

         data%P_diag( : data%n_sub ) = zero
         DO l = 1, data%P%ne
           j = data%P%col( l )
           IF ( j /= data%P%row( l ) ) THEN  !  off-diagonal term
             data%P_offd( data%P_colptr( j ) ) = data%P%val( l )
             data%P_row( data%P_colptr( j ) ) = data%P%row( l )
             data%P_colptr( j ) = data%P_colptr( j ) + 1
           ELSE   !  diagonal term
             data%P_diag( j ) = data%P%val( l )
           END IF
         END DO

!   ... and reposition the starting addresses

         data%P_colptr( 1 ) = 1
         DO i = 1, data%n_sub
           data%P_colptr( i + 1 ) = data%P_colptr( i ) + data%IW( i )
         END DO

!         WRITE( out, "( ' P ' )" )
!         DO l = 1, data%P%ne
!           WRITE( out, "( ' ', 2I7, ES12.4 )" )                               &
!             data%P%row( l ), data%P%col( l ), data%P%val( l )
!         END DO

!         WRITE( out, "( ' P by rows ' )" )
!         DO i = 1, data%n_sub
!           WRITE( out, "( 'd', 2I7, ES12.4 )" )                               &
!              i, i, data%P_diag( i )
!           DO l = data%P_colptr( i ), data%P_colptr( i + 1 ) - 1
!               WRITE( out, "( 'o', 2I7, ES12.4 )" )                           &
!                 data%P_row( l ), i, data%P_offd( l )
!           END DO
!         END DO

!  form and factorize Lin and More's preconditioner

         data%perturbation = zero
         i = data%n_sub
         IF ( printi ) THEN
           data%IW( 1 ) = out
         ELSE
           data%IW( 1 ) = - 1
         END IF
         CALL DICFS( i, data%p_ne,                                             &
                     data%P_offd( : data%p_ne ), data%P_diag( : data%n_sub ),  &
                     data%P_colptr( : data%n_sub + 1 ),                        &
                     data%P_row( : data%p_ne ),                                &
                     data%L_offd( : data%l_ne ), data%L_diag( : data%n_sub ),  &
                     data%L_colptr( : data%n_sub + 1 ),                        &
                     data%L_row( : data%l_ne ),                                &
                     icfact, data%perturbation, data%IW( : 3 * data%n_sub ),   &
                     data%DIAG( : data%n_sub ), data%W( : data%n_sub ) )
         IF ( i == - 26 ) THEN
           inform%status = GALAHAD_error_unknown_solver ; GO TO 910 ; END IF

        IF ( printi ) WRITE( out, "( /, A,                                     &
       &   ' Preconditioner: Lin-More incomplete factorizatio',                &
       &     /, A, '  order of preconditioner         = ', I0,                 &
       &     /, A, '  # nonzeros in preconditioner    = ', I0 )" )             &
           prefix, prefix, data%n_sub, prefix, data%P%ne

!        WRITE( out, "( ' L by rows ' )" )
!        DO i = 1, data%n_sub
!          WRITE( out, "( 'd', 2I7, ES12.4 )" )                                &
!             i, i, data%L_diag( i )
!          DO l = data%L_colptr( i ), data%L_colptr( i + 1 ) - 1
!              WRITE( out, "( 'o', 2I7, ES12.4 )" )                            &
!                data%L_row( l ), i, data%L_offd( l )
!          END DO
!        END DO

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize                                            &
          = inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, time_now - time_record

!  the HSL_MI28 incomplete factorization of the full matrix will be used

      CASE ( preconditioner_incomplete_mi28 )
        CALL CPU_TIME( time_record ) ; CALL CLOCK_time( clock_record )

!  build a mapping from co-ordinate to CSR format

        IF ( control%new_structure ) THEN

          array_name = 'psls: data%MAPS'
          CALL SPACE_resize_array( data%P%ne, data%MAPS,                       &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_csr%ptr'
          CALL SPACE_resize_array( data%P%n + 1, data%P_csr%ptr,               &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SLS_coord_to_sorted_csr( data%P%n, data%P%ne,                   &
                                        data%P%row, data%P%col,                &
                                        data%MAPS, data%P_csr%ptr,             &
                                        inform%duplicates,                     &
                                        inform%out_of_range, inform%upper,     &
                                        inform%missing_diagonals,              &
                                        inform%status, inform%alloc_status )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  now map the row and column data ...

          data%P_csr%n = data%P%n
          data%P_csr%ne = data%P_csr%ptr( data%P%n + 1 ) - 1

!         array_name = 'psls: data%P_csr%row'
!         CALL SPACE_resize_array( data%P_csr%ne, data%P_csr%row,              &
!             inform%status, inform%alloc_status, array_name = array_name,     &
!             deallocate_error_fatal = control%deallocate_error_fatal,         &
!             exact_size = control%space_critical,                             &
!             bad_alloc = inform%bad_alloc, out = control%error )
!         IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_csr%col'
          CALL SPACE_resize_array( data%P_csr%ne, data%P_csr%col,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_csr%val'
          CALL SPACE_resize_array( data%P_csr%ne, data%P_csr%val,              &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          DO i = 1, data%P%n
            l = data%P_csr%ptr( i )
            data%P_csr%col( l ) = i
          END DO
          DO l = 1, data%P%ne
            k = data%MAPS( l )
            IF ( k > 0 ) THEN
!             data%P_csr%row( k ) = MIN( data%P%row( l ), data%P%col( l ) )
              data%P_csr%col( k ) = MAX( data%P%row( l ), data%P%col( l ) )
            END IF
          END DO
        END IF

!  ...  and the values

        data%P_csr%val( data%P_csr%ptr( 1 : data%P%n ) ) = zero
        DO l = 1, data%P%ne
          k = data%MAPS( l )
          IF ( k > 0 ) THEN
            data%P_csr%val( k ) = data%P%val( l )
          ELSE IF ( k < 0 ) THEN
            data%P_csr%val( - k ) = data%P_csr%val( - k ) + data%P%val( l )
          END IF
        END DO

        data%mi28_control = control%mi28_control
        IF ( control%print_level <= 0 .OR. control%out <= 0 )                  &
          data%mi28_control%unit_warning = - 1
        IF ( control%print_level <= 0 .OR. control%error <= 0 )                &
          data%mi28_control%unit_error = - 1

!  form the preconditioner

        CALL MI28_factorize( data%P_csr%n, data%P_csr%ptr, data%P_csr%col,     &
           data%P_csr%val, control%mi28_lsize, control%mi28_rsize,             &
           data%mi28_keep, data%mi28_control, inform%mi28_info )

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%factorize = inform%time%factorize + time_now - time_record
        inform%time%clock_factorize                                            &
          = inform%time%clock_factorize + clock_now - clock_record
        IF ( printt ) WRITE( out, 2010 ) prefix, time_now - time_record

        IF ( inform%mi28_info%stat < 0 ) THEN
          SELECT CASE( inform%mi28_info%stat )
          CASE ( - 1 )
            inform%status = GALAHAD_error_allocate
            inform%alloc_status = inform%mi28_info%stat
          CASE ( - 14 )
            inform%status = GALAHAD_error_deallocate
            inform%alloc_status = inform%mi28_info%stat
          CASE( -  6 : - 2 )
            inform%status = GALAHAD_error_restrictions
          CASE ( - 7  )
             inform%status = GALAHAD_error_mc77
          CASE ( - 8  )
             inform%status = GALAHAD_error_mc64
          CASE ( - 9  )
             inform%status = GALAHAD_error_inertia
          CASE ( - 10  )
             inform%status = GALAHAD_error_scale
          CASE ( - 11 )
            inform%status = GALAHAD_error_permutation
          CASE ( - 12  )
             inform%status = GALAHAD_error_mc61
          CASE ( - 13  )
             inform%status = GALAHAD_error_mc68
          CASE ( GALAHAD_unavailable_option  )
             inform%status = GALAHAD_unavailable_option
             IF ( control%print_level > 0 .AND. control%out > 0 )              &
               WRITE( control%out, "( A, ' hsl_mi28 is not available' )") prefix
          CASE DEFAULT
            inform%status = GALAHAD_error_technical
          END SELECT
          RETURN
        END IF

!  One of the other cases will be used ... eventually

      CASE DEFAULT
        WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )           &
          control%preconditioner
        inform%status = GALAHAD_preconditioner_unknown
        RETURN
      END SELECT

!  ----------------------------------------------------
!  Stage 3 - Prepare for further potential restrictions
!  ----------------------------------------------------

      data%n_fixed = 0

!  Allocate workspace for updates

      data%SCU_matrix%n = data%n_sub
      data%SCU_matrix%m = data%n_fixed
      data%SCU_matrix%m_max = data%max_col
      data%SCU_matrix%class = 4

      array_name = 'lancelot: data%SCU_matrix%BD_col_start'
      CALL SPACE_resize_array( data%SCU_matrix%m_max + 1,                      &
             data%SCU_matrix%BD_col_start,                                     &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'lancelot: data%SCU_matrix%BD_row'
      CALL SPACE_resize_array( data%SCU_matrix%m_max, data%SCU_matrix%BD_row,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'lancelot: data%SCU_matrix%BD_val'
      CALL SPACE_resize_array( data%SCU_matrix%m_max, data%SCU_matrix%BD_val,  &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'psls: data%RHS_sub'
      CALL SPACE_resize_array( data%n_sub, data%RHS_sub,                       &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      IF ( control%preconditioner == preconditioner_incomplete_mi28 ) THEN
        array_name = 'psls: data%SOL_sub'
        CALL SPACE_resize_array( data%n_sub, data%SOL_sub,                     &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910
      END IF

      data%SCU_matrix%BD_col_start( 1 ) = 1
      data%scu_status = 1
      CALL SCU_factorize( data%SCU_matrix, data%SCU_data,                      &
                          data%RHS_sub( : data%SCU_matrix%n ),                 &
                          data%scu_status, data%SCU_inform )

!  Allocate further workspace

      array_name = 'psls: data%RHS_scu'
      CALL SPACE_resize_array( data%n_sub + data%max_col, data%RHS_scu,        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'psls: data%SOL_scu'
      CALL SPACE_resize_array( data%n_sub + data%max_col, data%SOL_scu,        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

      array_name = 'psls: data%G'
      CALL SPACE_resize_array( data%n_sub, data%G,                             &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  Record the time taken to assemble and factorize the preconditioner

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      RETURN

!  Allocation error

  910 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      RETURN

!  Other error

  930 CONTINUE
      IF ( printi ) WRITE( out, "( ' ', /, A, '   **  Error return ', I0,     &
     & ' from PSLS ' )" ) prefix, inform%status
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      RETURN

! Non-executable statements

 2000 FORMAT( /, A, ' time( SLS_analyse ) = ', F0.2 )
 2010 FORMAT( /, A, ' time( SLS_factorize ) = ', F0.2 )

!  End of subroutine PSLS_form_and_factorize

      END SUBROUTINE PSLS_form_and_factorize

!-*-*-*-*-*-*-*-   P S L S _ B U I L D   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE PSLS_build( A, P, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Build a symmetric, strictly-diagonally dominant approximation P to a
!  given symmetrix matrix A

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!   A is a structure of type SMT_type used to hold the LOWER TRIANGULAR part
!    of A. Four storage formats are permitted:
!
!    i) sparse, co-ordinate
!
!       In this case, the following must be set:
!
!       A%type( 1 : 10 ) = TRANSFER( 'COORDINATE', A%type )
!       A%val( : )   the values of the components of A
!       A%row( : )   the row indices of the components of A
!       A%col( : )   the column indices of the components of A
!       A%ne         the number of nonzeros used to store
!                    the LOWER TRIANGULAR part of A
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
!    iv) diagonal
!
!       In this case, the following must be set:
!
!       A%type( 1 : 8 ) = TRANSFER( 'DIAGONAL', A%type )
!       A%val( : )   the values of the diagonals of A, stored in order
!
!  data is a structure of type PSLS_data_type which holds private internal data
!
!  control is a structure of type PSLS_control_type that controls the
!   execution of the subroutine and must be set by the user. Default values for
!   the elements may be set by a call to PSLS_initialize. See PSLS_initialize
!   for details
!
!  inform is a structure of type PSLS_inform_type that provides information on
!   exit from PSLS_solve. The component status has possible values:
!
!     0 Normal termination.
!
!    -1 An allocation error occured; the status is given in the component
!       alloc_status.
!
!    -2 A deallocation error occured; the status is given in the component
!       alloc_status.
!
!    -3 one of the restrictions
!        A%n    >=  1
!        A%type in { 'DENSE', 'SPARSE_BY_ROWS', 'COORDINATE', , 'DIAGONAL' }
!       has been violated.
!
!    -9 the ordering (analysis) phase failed.
!
!    -10 the factorization phase failed.
!
!  On exit from PSLS_build, other components of inform give the
!  following:
!
!     alloc_status = The status of the last attempted allocation/deallocation
!     analyse_status = The return status from the ordering phase of the
!      factorization (if any)
!     factorize_status = The return status from the factorization phase
!     solve_status = The return status from the solve phase
!     factorization_integer = The total integer workspace required for the
!       factorization
!     factorization_real = The total real workspace required for the
!       factorization
!     preconditioner = Code for the actual preconditioner computed
!     semi_bandwidth = The semi-bandwidth of the original submatrix
!     neg1, neg2 - the number of -ve 1x1 and 2x2 pivots found during the
!       factorization
!     perturbed = true if the initial preconditioner was perturbed to ensure
!      that it is definite
!     fill_in_ratio = ratio of nonzeros in factors to the original matrix
!     norm_residual = norm of the residual during the solve phase
!     bad_alloc = the name of the array for which an allocation/deallocation
!       error ocurred
!     time%total = the total time spent in the package
!     time%assemble = the time spent building the preconditioner
!     time%analyse = the time spent reordering the preconditioner prior to
!       factorization
!     time%factorize = the time spent factorizing the preconditioner
!     time%solve = the time spent in the solution phase
!
!   SUB is an optional rank-one integer assumed-sized array whose components
!    list the indices of the required submatrix. The indices should be in
!    increasing order. If SUB is not present, the entire matrix A will be
!    considered.

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( SMT_type ), INTENT( INOUT ) :: P
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, ij, j, jj, l, out, print_level
      LOGICAL :: printi, printt
      CHARACTER ( LEN = 80 ) :: array_name
      REAL :: time_now, time_start
      REAL ( KIND = wp ) :: clock_start, clock_now
      REAL ( KIND = wp ) :: val

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

!  set initial values

      IF ( control%new_structure ) THEN
        inform%analyse_status = 0 ; inform%factorize_status = 0
        inform%solve_status = 0
        inform%factorization_integer = 0 ; inform%factorization_real = 0
        inform%preconditioner = 0 ; inform%semi_bandwidth = 0
        inform%neg1 = 0 ; inform%neg2 = 0 ; inform%perturbed = .FALSE.
        inform%fill_in_ratio = one ; inform%norm_residual = zero
        inform%bad_alloc  = ''
        inform%time%assemble = 0.0 ; inform%time%analyse  = 0.0
        inform%time%factorize = 0.0 ; inform%time%solve = 0.0
        inform%time%update = 0.0 ; inform%time%total = 0.0
      END IF

!  return if no preconditioning (P=I) is required

      IF ( control%preconditioner == preconditioner_none ) THEN
        inform%status = GALAHAD_ok
        RETURN
      END IF

!  record desired output level

      out = control%out
      print_level = control%print_level
      printi = out > 0 .AND. print_level > 0
      printt = out > 0 .AND. print_level > 1

!  Ensure that input parameters are within allowed ranges

      IF ( A%n < 0 .OR. .NOT. QPT_keyword_H( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        GO TO 930
      END IF

      IF ( control%new_structure ) THEN
        data%n_update = 0
        data%max_col = control%max_col
        IF ( data%max_col <= 0 ) data%max_col = 100

!  Record if a variable is contained in the submatrix

        data%n = A%n
        data%n_sub = data%n

!  Record the variables contained in the submatrix

!  If desired, compute the semi-bandwidth of the selected sub-matrix

        IF ( control%get_semi_bandwidth ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DIAGONAL' )
            inform%semi_bandwidth = 0
          CASE ( 'DENSE' )
            inform%semi_bandwidth = data%n_sub - 1
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%n
              DO jj = A%ptr( i ), A%ptr( i + 1 ) - 1
                inform%semi_bandwidth = MAX( inform%semi_bandwidth,            &
                  ABS( i - A%col( jj ) ) )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            inform%semi_bandwidth = 0
            DO l = 1, A%ne
              inform%semi_bandwidth = MAX( inform%semi_bandwidth,              &
                ABS( A%row( l ) - A%col( l ) ) )
            END DO
          END SELECT
        END IF
        inform%reordered_semi_bandwidth = inform%semi_bandwidth
      END IF

!  --------------------------------
!  Stage 1 - Form the preliminary P
!  --------------------------------

      IF ( printt ) WRITE( out, "( /, A, ' Form P' )" ) prefix
      P%n = data%n_sub ; P%m = P%n

!  special case: A is diagonal

      IF ( SMT_get( A%type ) == 'DIAGONAL' ) THEN

!  allocate space to hold the matrix in diagonal form

        IF ( control%new_structure ) THEN
          P%ne = P%n
          CALL SPACE_resize_array( P%ne, P%val,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SMT_put( P%type, 'DIAGONAL', inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN
            inform%status = - 1 ; GO TO 910 ; END IF
        END IF

!  fit the data into the diagonal and ensure that the diagonal is
!  sufficiently positive

        P%val = MAX( A%val( : data%n ), control%min_diagonal )
        GO TO 900
      END IF

!  usual case: A is not diagonal

!  the matrix P will be ordered so that its diagonal entries occur in
!  positions 1:n

      SELECT CASE ( control%preconditioner )

!  a diagonal matrix will be used

      CASE ( preconditioner_diagonal )

!  allocate space to hold the matrix in diagonal form

        IF ( control%new_structure ) THEN
          P%ne = P%n
          CALL SPACE_resize_array( P%ne, P%val,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SMT_put( P%type, 'DIAGONAL', inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN
            inform%status = - 1 ; GO TO 910 ; END IF
        END IF

!  fit the data into the diagonal

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          P%val = zero
          DO i = 1, data%n
            l = l + i
            P%val( i ) = P%val( i ) + A%val( l )
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          P%val = zero
          DO i = 1, data%n
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              IF ( i == j ) P%val( i ) = P%val( i ) + A%val( l )
            END DO
          END DO
        CASE ( 'COORDINATE' )
          P%val = zero
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l )
            IF ( i == j ) P%val( i ) = P%val( i ) + A%val( l )
          END DO
        END SELECT

!  ensure that the diaginal is sufficiently positive

        P%val = MAX( P%val, control%min_diagonal )
        GO TO 900

!  A band matrix will be used

      CASE ( preconditioner_band )

        IF ( control%new_structure ) THEN

!  compute the maximum semi-bandwidth required

          data%semi_bandwidth_used = 0
          P%ne = data%n
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            DO i = 1, data%n
              DO j = 1, i
                ij = ABS( i - j )
                IF ( ij > 0 .AND. ij <= control%semi_bandwidth ) THEN
                  data%semi_bandwidth_used =                                   &
                    MAX( data%semi_bandwidth_used, ij )
                  P%ne = P%ne + 1
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%n
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                ij = ABS( i - A%col( l ) )
                IF ( ij > 0 .AND. ij <= control%semi_bandwidth ) THEN
                  data%semi_bandwidth_used =                                   &
                    MAX( data%semi_bandwidth_used, ij )
                  P%ne = P%ne + 1
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l ) ; j = A%col( l )
              ij = ABS( i - j )
              IF ( ij > 0 .AND. ij <= control%semi_bandwidth ) THEN
                data%semi_bandwidth_used = MAX( data%semi_bandwidth_used, ij )
                P%ne = P%ne + 1
              END IF
            END DO
          END SELECT

!  allocate space to hold the band matrix in both band and co-ordinate form

          array_name = 'psls: P%row'
          CALL SPACE_resize_array( P%ne, P%row,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: P%col'
          CALL SPACE_resize_array( P%ne, P%col,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: P%val'
          CALL SPACE_resize_array( P%ne, P%val,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SMT_put( P%type, 'COORDINATE', inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN
            inform%status = - 1 ; GO TO 910 ; END IF
        END IF

!  fit the data into the band

        IF ( control%new_structure ) THEN
          DO i = 1, data%n
            P%row( i ) = i ; P%col( i ) = i
          END DO
        END IF
        P%val( : data%n ) = zero
        P%ne = data%n
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, data%n
            DO j = 1, i
              l = l + 1
              ij = ABS( i - j )
              IF ( ij <= data%semi_bandwidth_used ) THEN
                IF ( ij == 0 ) THEN
                  P%val( i ) = P%val( i ) + A%val( l )
                ELSE
                  P%ne = P%ne + 1
                  IF ( control%new_structure ) THEN
                    P%row( P%ne ) = i ; P%col( P%ne ) = j
                  END IF
                  P%val( P%ne ) = A%val( l )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, data%n
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              ij = ABS( i - j )
              IF ( ij <= data%semi_bandwidth_used ) THEN
                IF ( ij == 0 ) THEN
                  P%val( i ) = P%val( i ) + A%val( l )
                ELSE
                  P%ne = P%ne + 1
                  IF ( control%new_structure ) THEN
                    P%row( P%ne ) = i ; P%col( P%ne ) = j
                  END IF
                  P%val( P%ne ) = A%val( l )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l )
            ij = ABS( i - j )
            IF ( ij <= data%semi_bandwidth_used ) THEN
              IF ( ij == 0 ) THEN
                P%val( i ) = P%val( i ) + A%val( l )
              ELSE
                P%ne = P%ne + 1
                IF ( control%new_structure ) THEN
                  P%row( P%ne ) = i ; P%col( P%ne ) = j
                END IF
                P%val( P%ne ) = A%val( l )
              END IF
            END IF
          END DO
        END SELECT

        IF ( printi ) WRITE( out, "( /, A, ' -- Preconditioner formed.', /, A, &
       &           ' -- order = ', I0, ', semi-bandwidth = ', I0,              &
       &           ', true semi-bandwith = ', I0 )" ) prefix, prefix,          &
                   data%n_sub, data%semi_bandwidth_used, inform%semi_bandwidth

!  A re-ordered band matrix will be used

      CASE ( preconditioner_reordered_band )

!       WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )           &
!         control%preconditioner
!       inform%status = GALAHAD_preconditioner_unknown
!       RETURN

        IF ( control%new_structure ) THEN

!  compute the number of nonzeros in A

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            data%p_ne = ( data%n_sub * ( data%n_sub + 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            data%p_ne = A%ptr( data%n + 1 ) - 1
          CASE ( 'COORDINATE' )
            data%p_ne = A%ne
          END SELECT

!  allocate workspace

          data%mc61_lirn = 2 * data%p_ne
          data%mc61_liw = 8 * data%n_sub + 2

          array_name = 'psls: data%P_row'
          CALL SPACE_resize_array( data%mc61_lirn, data%P_row,                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_col'
          CALL SPACE_resize_array( data%p_ne, data%P_col,                      &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
           IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%P_colptr'
          CALL SPACE_resize_array( data%n_sub + 1, data%P_colptr,              &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%PERM'
          CALL SPACE_resize_array( data%n_sub, data%PERM,                      &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%IW'
          CALL SPACE_resize_array( data%mc61_liw, data%IW,                     &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: data%W'
          CALL SPACE_resize_array( data%n_sub, data%W,                         &
                inform%status, inform%alloc_status, array_name = array_name,   &
                deallocate_error_fatal = control%deallocate_error_fatal,       &
                exact_size = control%space_critical,                           &
                bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  convert the matrix to co-ordinate form ...

          data%p_ne = 0
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            l = 0
            DO i = 1, data%n
              DO j = 1, i
                l = l + 1
                data%p_ne = data%p_ne + 1
                IF ( i >= j ) THEN
                  data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = j
                ELSE
                  data%P_row( data%p_ne ) = j ; data%P_col( data%p_ne ) = i
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%n
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l )
                data%p_ne = data%p_ne + 1
                IF ( i >= j ) THEN
                  data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = j
                ELSE
                  data%P_row( data%p_ne ) = j ; data%P_col( data%p_ne ) = i
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l ) ; j = A%col( l )
              data%p_ne = data%p_ne + 1
              IF ( i >= j ) THEN
                data%P_row( data%p_ne ) = i ; data%P_col( data%p_ne ) = j
              ELSE
                data%P_row( data%p_ne ) = j ; data%P_col( data%p_ne ) = i
              END IF
            END DO
          END SELECT

!  ... and now convert the matrix to column form

          CALL SORT_reorder_by_cols( data%n_sub, data%n_sub, data%p_ne,        &
                                     data%P_row, data%P_col, data%p_ne,        &
                                     data%P_colptr, data%n_sub + 1,            &
                                     data%IW, data%mc61_liw, 0, 0, i )
          IF ( i > 0 ) THEN
            write(6,"( ' error ', I0, ' from SORT_reorder_by_cols' )" ) i
            stop
          ELSE IF ( i < 0 ) THEN
            write(6,"( ' warning ', I0, ' from SORT_reorder_by_cols' )" ) i
          END IF

!  find a bandwidth-reducing ordering

          IF ( control%print_level <= 0 .OR. control%out <= 0 )                &
            data%mc61_ICNTL( 1 : 2 ) = - 1
          CALL MC61AD( 2, data%n_sub, data%mc61_lirn,                          &
                       data%P_row, data%P_colptr, data%PERM, data%mc61_liw,    &
                       data%IW, data%W, data%mc61_ICNTL, data%mc61_CNTL,       &
                       inform%mc61_info, inform%mc61_rinfo )

!write(6,*) ' perm ', data%PERM
          IF ( inform%mc61_info( 1 ) == GALAHAD_unavailable_option ) THEN
            IF ( control%print_level > 0 .AND. control%out > 0 )               &
              WRITE( control%out, "( A, ' mc61 is not available ' )" ) prefix
            inform%status = GALAHAD_error_unknown_solver ; GO TO 930
          END IF
          inform%reordered_semi_bandwidth = INT( inform%mc61_rinfo( 7 ) ) - 1
!write(6,*) ' reordered semi-bandwidth ', inform%semi_bandwidth

!  compute the maximum semi-bandwidth required

          data%semi_bandwidth_used = 0
          P%ne = data%n
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            DO i = 1, data%n
              ii = data%PERM( i )
              DO j = 1, i
                jj = data%PERM( j )
                ij = ABS( ii - jj )
                IF ( ij > 0 .AND. ij <= control%semi_bandwidth ) THEN
                  data%semi_bandwidth_used =                                   &
                    MAX( data%semi_bandwidth_used, ij )
                  P%ne = P%ne + 1
                END IF
              END DO
            END DO
          CASE ( 'SPARSE_BY_ROWS' )
            DO i = 1, data%n
              ii = data%PERM( i )
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                j = A%col( l ) ; jj = data%PERM( j )
                ij = ABS( ii - jj )
                IF ( ij > 0 .AND. ij <= control%semi_bandwidth ) THEN
                  data%semi_bandwidth_used =                                   &
                    MAX( data%semi_bandwidth_used, ij )
                  P%ne = P%ne + 1
                END IF
              END DO
            END DO
          CASE ( 'COORDINATE' )
            DO l = 1, A%ne
              i = A%row( l ) ; ii = data%PERM( i )
              j = A%col( l ) ; jj = data%PERM( j )
              ij = ABS( ii - jj )
              IF ( ij > 0 .AND. ij <= control%semi_bandwidth ) THEN
                data%semi_bandwidth_used = MAX( data%semi_bandwidth_used, ij )
                P%ne = P%ne + 1
              END IF
            END DO
          END SELECT

!  allocate space to hold the band matrix in both band and co-ordinate form

          array_name = 'psls: P%row'
          CALL SPACE_resize_array( P%ne, P%row,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: P%col'
          CALL SPACE_resize_array( P%ne, P%col,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: P%val'
          CALL SPACE_resize_array( P%ne, P%val,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SMT_put( P%type, 'COORDINATE', inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN
            inform%status = - 1 ; GO TO 910 ; END IF
        END IF

!  fit the data into the band

        IF ( control%new_structure ) THEN
          DO i = 1, data%n
            P%row( i ) = i ; P%col( i ) = i
          END DO
        END IF
        P%val( : data%n ) = zero
        P%ne = data%n
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, data%n
            ii = data%PERM( i )
            DO j = 1, i
              l = l + 1
              jj = data%PERM( j )
              ij = ABS( ii - jj )
              IF ( ij <= data%semi_bandwidth_used ) THEN
                IF ( ij == 0 ) THEN
                  P%val( i ) = P%val( i ) + A%val( l )
                ELSE
                  P%ne = P%ne + 1
                  IF ( control%new_structure ) THEN
                    P%row( P%ne ) = i ; P%col( P%ne ) = j
                  END IF
                  P%val( P%ne ) = A%val( l )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, data%n
            ii = data%PERM( i )
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              jj = data%PERM( j )
              ij = ABS( ii - jj )
              IF ( ij <= data%semi_bandwidth_used ) THEN
                IF ( ij == 0 ) THEN
                  P%val( i ) = P%val( i ) + A%val( l )
                ELSE
                  P%ne = P%ne + 1
                  IF ( control%new_structure ) THEN
                    P%row( P%ne ) = i ; P%col( P%ne ) = j
                  END IF
                  P%val( P%ne ) = A%val( l )
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; ii = data%PERM( i )
            j = A%col( l ) ; jj = data%PERM( j )
            ij = ABS( ii - jj )
            IF ( ij <= data%semi_bandwidth_used ) THEN
              IF ( ij == 0 ) THEN
                P%val( i ) = P%val( i ) + A%val( l )
              ELSE
                P%ne = P%ne + 1
                IF ( control%new_structure ) THEN
                  P%row( P%ne ) = i ; P%col( P%ne ) = j
                END IF
                P%val( P%ne ) = A%val( l )
              END IF
            END IF
          END DO
        END SELECT

        IF ( printi ) WRITE( out, "( /, A, ' -- Preconditioner formed.', /, A, &
       &           ' -- order = ', I0, ', semi-bandwidth = ', I0,              &
       &           ', true semi-bandwith = ', I0 )" ) prefix, prefix,          &
                   data%n_sub, data%semi_bandwidth_used, inform%semi_bandwidth

!  The full matrix will be used

      CASE ( preconditioner_full_se, preconditioner_full_gmps,                 &
             preconditioner_incomplete_lm, preconditioner_incomplete_mi28,     &
             preconditioner_incomplete_munks )

!  compute the space required

        IF ( control%new_structure ) THEN
          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' )
            P%ne = ( data%n_sub * ( data%n_sub - 1 ) ) / 2
          CASE ( 'SPARSE_BY_ROWS' )
            P%ne = data%n
            DO i = 1, data%n
              DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
                IF ( i /= A%col( l ) ) P%ne = P%ne + 1
              END DO
            END DO
          CASE ( 'COORDINATE' )
            P%ne = data%n
            DO l = 1, A%ne
              IF ( A%row( l ) /= A%col( l ) ) P%ne = P%ne + 1
            END DO
          END SELECT

!  allocate space to hold the full matrix in co-ordinate form

          array_name = 'psls: P%row'
          CALL SPACE_resize_array( P%ne, P%row,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: P%col'
          CALL SPACE_resize_array( P%ne, P%col,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          array_name = 'psls: P%val'
          CALL SPACE_resize_array( P%ne, P%val,                                &
              inform%status, inform%alloc_status, array_name = array_name,     &
              deallocate_error_fatal = control%deallocate_error_fatal,         &
              exact_size = control%space_critical,                             &
              bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) GO TO 910

          CALL SMT_put( P%type, 'COORDINATE', inform%alloc_status )
          IF ( inform%alloc_status /= 0 ) THEN
            inform%status = - 1 ; GO TO 910 ; END IF

        END IF

!  fit the data into the coordinate storage scheme provided

        IF ( control%new_structure ) THEN
          DO i = 1, data%n
            P%row( i ) = i ; P%col( i ) = i
          END DO
        END IF
        P%val( : data%n ) = zero
        P%ne = data%n
        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, data%n
            DO j = 1, i
              l = l + 1
              IF ( i == j ) THEN
                P%val( i ) = P%val( i ) + A%val( l )
              ELSE
                P%ne = P%ne + 1
                IF ( control%new_structure ) THEN
                  P%row( P%ne ) = i ; P%col( P%ne ) = j
                END IF
                P%val( P%ne ) = A%val( l )
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, data%n
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%col( l )
              IF ( i == j ) THEN
                P%val( i ) = P%val( i ) + A%val( l )
              ELSE
                P%ne = P%ne + 1
                IF ( control%new_structure ) THEN
                  P%row( P%ne ) = i ; P%col( P%ne ) = j
                END IF
                P%val( P%ne ) = A%val( l )
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%row( l ) ; j = A%col( l )
            IF ( i == j ) THEN
              P%val( i ) = P%val( i ) + A%val( l )
            ELSE
              P%ne = P%ne + 1
              IF ( control%new_structure ) THEN
                P%row( P%ne ) = i ; P%col( P%ne ) = j
              END IF
              P%val( P%ne ) = A%val( l )
            END IF
          END DO
        END SELECT

!  Choose initial values for the control parameters

!        IF ( control%print_level > 4 ) THEN
!          control%SLS_control%out = out ; control%SLS_control%error = out
!          control%SLS_control%print_level = 1
!        ELSE
!          control%SLS_control%out = 0 ; control%SLS_control%error = 0
!          control%SLS_control%print_level = 0
!        END IF

!  One of the other cases will be used ... eventually

      CASE DEFAULT
        WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )           &
          control%preconditioner
        inform%status = GALAHAD_preconditioner_unknown
        RETURN
      END SELECT

      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%assemble = time_now - time_start
      inform%time%clock_assemble = clock_now - clock_start
      IF ( printt ) WRITE( out, "( /, A,  ' time( assembly ) = ', F10.2 )" )   &
        prefix, inform%time%assemble

      inform%semi_bandwidth_used = data%semi_bandwidth_used
!     write(6,*) inform%semi_bandwidth, inform%semi_bandwidth_used

!  ---------------------------
!  Stage 2 - possibly modify P
!  ---------------------------

!  allocate space to hold sums of absolute values of off-diagonal emtries of P

      array_name = 'psls: data%P_offd'
      CALL SPACE_resize_array( data%n_sub, data%P_offd,                        &
          inform%status, inform%alloc_status, array_name = array_name,         &
          deallocate_error_fatal = control%deallocate_error_fatal,             &
          exact_size = control%space_critical,                                 &
          bad_alloc = inform%bad_alloc, out = control%error )
      IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  compute the sums of the absolute values of off-diagonal terms of P

      data%P_offd = zero
      DO l = 1, P%ne
        i = P%row( l ) ; j = P%col( l ) ; val = P%val( l )
        IF ( i /= j ) THEN
          data%P_offd( i ) = data%P_offd( i ) + ABS( val )
          data%P_offd( j ) = data%P_offd( j ) + ABS( val )
        END IF
      END DO

!  increase any diagonal entry of P to ensure that is larger than the sum
!  of the absolute values of off-diagonal terms

      DO i = 1, P%n
        P%val( i ) = MAX( P%val( i ), data%P_offd( i ) + control%min_diagonal )
      END DO

!  Record the time taken to build the preconditioner

  900 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      inform%status = GALAHAD_ok
      RETURN

!  Allocation error

  910 CONTINUE
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      RETURN

!  Other error

  930 CONTINUE
      IF ( printi ) WRITE( out, "( ' ', /, A, '   **  Error return ', I0,     &
     & ' from PSLS ' )" ) prefix, inform%status
      CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
      inform%time%total = inform%time%total + time_now - time_start
      inform%time%clock_total                                                  &
        = inform%time%clock_total + clock_now - clock_start
      RETURN

!  End of subroutine PSLS_build

      END SUBROUTINE PSLS_build

!-*-*-   P S L S _ U P D A T E  _ F A C T O R S   S U B R O U T I N E   -*-*-*-

      SUBROUTINE PSLS_update_factors( FIX, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Given a symmetrix positive matrix P, update its factors to account
!  for the case where a set of its rows/columns are removed.

!  Specifically, the matrix P has been found as a suitable approximation to
!  the n_sub by n_sub symmetric submatrix, involving rows/columns indexed by
!  SUB, of a given n by n matrix A, using subroutine PSLS_form_and_factorize.
!  Schemetically (implicitly permuting the rows/columns in SUB to the front),
!
!     ( A_sub  A_off^T ) -> A_sub ~= P
!     ( A_off  A_left  )
!
!  Subsequently a further n_fixed rows/columns have been removed by previous
!  calls to PSLS_update_factors. Now another set of rows/columns of A, listed
!  in FIX, are to be fixed. Again, schematically (implicitly permuting the
!  fixed rows/columns to the end),
!
!     ( P_free  P_off^T ) -> P_free
!     ( P_off   P_fixed )
!
!  Our aim is to solve systems
!
!      P_free x_free = rhs_free
!
!  involving P_free. We achieve by reformulating the system as
!
!     ( P_free  P_off^T  0 ) ( x_free  )   ( rhs_free  )
!     ( P_off   P_fixed  I ) ( x_fixed ) = ( rhs_fixed );  (*)
!     (   0       I      0 ) (   y     )   (    0      )
!
!  the actual values of rhs_fixed and thus y are irrelevant and may be
!  chosen for convenience. To solve (*) we use the Schur complement method
!  implemented in GALAHAD_SCU; this uses the available factors of P and
!  the computed ones of the Schur-complement
!
!     S = ( 0  I ) ( P_free  P_off^T )^{-1} ( 0 )
!                  ( P_off   P_fixed )      ( I )
!
!  and it is the latter that we need to obtain. S is obtained one row
!  at a time. Once S exceeds a specified dimension, P will be reformed
!  and refactorized, the fixed variables removed from SUB, and n_fixed
!  reset to zero

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Arguments:

!   FIX is a rank-one integer assumed-sized array whose components list the
!    indices of the A that are to be fixed

!   data, control, inform are as for PSLS_form_and_factorize

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) , DIMENSION ( : ) :: FIX
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, n_fix, n_stat, band_status, scu_status
      CHARACTER ( LEN = 60 ) :: task

!  how many variables will be fixed?

      n_fix = SIZE( FIX )

!  does this exceed the space alllowed for the Schur complement?

      IF ( data%SCU_matrix%m + n_fix > data%max_col ) THEN
        IF ( control%print_level >= 2 .AND. control%out > 0 )                  &
          WRITE( control%out,                                                  &
          "( /, ' Refactorizing: required Schur complement dimension ', I0,    &
       &     ' exceeds the allowed total of ', I0 )" )                         &
            data%SCU_matrix%m + n_fix, data%max_col
        data%INDEX( FIX( : n_fix ) ) = 0
        inform%status = 1
        RETURN
      END IF

!  run through the list of variables to be fixed. For each, find the
!  index in the free list, j

      DO i = 1, n_fix
        j = data%INDEX( FIX( i ) )
        IF ( j > 0 ) THEN

!  update the factorization of the Schur complement to allow for the removal of
!  the j-th row and column of the preconditioner P - this removal is effected by
!  by appending the j-th row and column of the identity matrix to P

          data%SCU_matrix%BD_val( data%n_fixed + 1 ) = one
          data%SCU_matrix%BD_row( data%n_fixed + 1 ) = j
          data%SCU_matrix%BD_col_start( data%n_fixed + 2 ) = data%n_fixed + 2
          scu_status = 1
  10      CONTINUE

!  call SCU_append to update the Schur-complement

          CALL SCU_append( data%SCU_matrix, data%SCU_data, data%RHS_sub,       &
                           scu_status, data%SCU_inform )

!  SCU_append requires additional information. Compute the solution to
!  the system P * x_sub = rhs_sub, returning the solution in rhs_sub

          IF ( scu_status > 0 ) THEN
            SELECT CASE ( control%preconditioner )

!  P is a diagonal matrix

            CASE ( preconditioner_diagonal )
              data%RHS_sub( : data%n_sub ) =                                   &
                data%RHS_sub( : data%n_sub ) / data%DIAG( : data%n_sub )

!  P is a band or re-ordered band matrix

            CASE ( preconditioner_band, preconditioner_reordered_band )
              CALL BAND_solve( data%n_sub, data%semi_bandwidth_used,           &
                               data%DIAG, data%OFFDIA,                         &
                               data%semi_bandwidth_used,                       &
                               data%RHS_sub, band_status )

!  P is the full matrix with Schnabel-Eskow or GMPS modifications

            CASE ( preconditioner_full_se, preconditioner_full_gmps )
              CALL SLS_solve( data%P, data%RHS_sub, data%SLS_data,             &
                              data%SLS_control, inform%SLS_inform )

!  P is the Lin-More' incomplete factorization

            CASE ( preconditioner_incomplete_lm )
              task = 'N'
              n_stat = data%n_sub
              CALL DSTRSOL( n_stat, data%L_offd( : data%l_ne ),                &
                            data%L_diag( : data%n_sub ),                       &
                            data%L_colptr( : data%n_sub + 1 ),                 &
                            data%L_row( : data%l_ne ), data%RHS_sub, task )
              IF ( n_stat == - 26 ) THEN
               inform%status = GALAHAD_error_unknown_solver ; RETURN ; END IF
              task = 'T'
              CALL DSTRSOL( n_stat, data%L_offd( : data%l_ne ),                &
                            data%L_diag( : data%n_sub ),                       &
                            data%L_colptr( : data%n_sub + 1 ),                 &
                            data%L_row( : data%l_ne ), data%RHS_sub, task )
              IF ( n_stat == - 26 ) THEN
               inform%status = GALAHAD_error_unknown_solver ; RETURN ; END IF

!  P is one of the other cases ... eventually

            CASE DEFAULT
              WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )     &
                control%preconditioner
              inform%status = GALAHAD_preconditioner_unknown
              RETURN
            END SELECT
            GO TO 10

!  If the Schur-complement is numerically indefinite, refactorize
!  the preconditioning matrix to alleviate the effect of rounding

          ELSE IF ( scu_status < 0 ) THEN
            IF ( control%print_level >= 2 .AND. control%out > 0 )              &
              WRITE( control%out,                                              &
              "( /, ' Refactorizing: status value on return from Schur',       &
           &      ' complement update = ', I0 )" ) scu_status
            data%INDEX( FIX( i : n_fix ) ) = 0
            inform%status = 1
            RETURN
          END IF

!  Record that the relevant variable is now fixed

          data%INDEX( FIX( i ) ) = 0
          data%n_fixed = data%n_fixed + 1
        END IF
      END DO
      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine PSLS_update_factors

      END SUBROUTINE PSLS_update_factors


!-*-*-   P S L S _ I N D E X _ S U B M A T R I X   S U B R O U T I N E   -*-*-

      SUBROUTINE PSLS_index_submatrix( n_sub, SUB, data )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!   n_sub is an integer that gives the size of the current submatrix allowing
!    for indices that have may been fixed by PSLS_update_factors

!   SUB is an rank-one integer assumed-sized array whose components list the
!    indices of the current submatrix. The indices will be in increasing order

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( OUT ) :: n_sub
      INTEGER, INTENT( OUT ), DIMENSION ( : ) :: SUB
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      INTEGER :: i, j

      n_sub = 0
      DO j = 1, data%n_sub
        i = data%SUB( j )
        IF ( data%INDEX( i ) /= 0 ) THEN
          n_sub = n_sub + 1
          SUB( n_sub ) = i
        END IF
      END DO

      RETURN

!  End of subroutine  PSLS_index_submatrix

      END SUBROUTINE  PSLS_index_submatrix

!-*-*-*-*-*-*-*-*-   P S L S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE PSLS_solve( SOL, data, control, inform )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Given a symmetrix positive matrix P, solve the system P x = b.
!  b is input in SOL, and the solution x overwrites SOL

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      REAL ( KIND = wp ), INTENT( INOUT ) , DIMENSION ( : ) :: SOL
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: n_stat, band_status, scu_status
      REAL :: time_start, time_now
      REAL ( KIND = wp ) :: clock_now, clock_start
      CHARACTER ( LEN = 60 ) :: task

!  prefix for all output

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      IF ( LEN( TRIM( control%prefix ) ) > 2 )                                 &
        prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start ) ; CALL CLOCK_time( clock_start )

      IF ( control%print_level > 10 .AND. control%out > 0 )                    &
        WRITE( control%out, "( /, A, ' solve ' )" ) prefix

!  - - - - - - - - - - - - - - - - - - - - - - - -
!  Initial solve using the original factorization
!  - - - - - - - - - - - - - - - - - - - - - - - -

      IF ( data%n_fixed == 0 ) THEN

!  return if no preconditioning (P=I) is required

        IF ( control%preconditioner == preconditioner_none ) THEN
          inform%status = 0
          RETURN
        END IF

!  gather the sub-components of the right-hand side

        IF ( control%preconditioner == preconditioner_reordered_band ) THEN
          IF ( data%sub_matrix ) THEN
            data%RHS_sub( data%PERM( : data%n_sub ) )                          &
              = SOL( data%SUB( : data%n_sub ) )
          ELSE
            data%RHS_sub( data%PERM( : data%n_sub ) ) = SOL( : data%n_sub )
          END IF
        ELSE
          IF ( data%sub_matrix ) THEN
            data%RHS_sub( : data%n_sub ) = SOL( data%SUB( : data%n_sub ) )
          ELSE
            data%RHS_sub( : data%n_sub ) = SOL( : data%n_sub )
          END IF
        END IF

!  solve the system P * x_sub = rhs_sub, returning the solution in
!  rhs_sub

        SELECT CASE ( control%preconditioner )

!  P is a diagonal matrix

        CASE ( preconditioner_diagonal )
          data%RHS_sub( : data%n_sub ) =                                       &
            data%RHS_sub( : data%n_sub ) / data%DIAG( : data%n_sub )

!  P is a band or re-ordered band matrix

        CASE ( preconditioner_band, preconditioner_reordered_band )
          CALL BAND_solve( data%n_sub, data%semi_bandwidth_used, data%DIAG,    &
                           data%OFFDIA, data%semi_bandwidth_used,              &
                           data%RHS_sub, band_status )

!  P is the full matrix with Schnabel-Eskow or GMPS modifications

        CASE ( preconditioner_full_se, preconditioner_full_gmps )
          CALL SLS_solve( data%P, data%RHS_sub, data%SLS_data,                 &
                          data%SLS_control, inform%SLS_inform )

!  P is the Lin-More' incomplete factorization

        CASE ( preconditioner_incomplete_lm )
          task = 'N'
          n_stat = data%n_sub
          CALL DSTRSOL( n_stat, data%L_offd( : data%l_ne ),                    &
                        data%L_diag( : data%n_sub ),                           &
                        data%L_colptr( : data%n_sub + 1 ),                     &
                        data%L_row( : data%l_ne ), data%RHS_sub, task )
          IF ( n_stat == - 26 ) THEN
            inform%status = GALAHAD_error_unknown_solver ; RETURN ; END IF
          task = 'T'
          CALL DSTRSOL( n_stat, data%L_offd( : data%l_ne ),                    &
                        data%L_diag( : data%n_sub ),                           &
                        data%L_colptr( : data%n_sub + 1 ),                     &
                        data%L_row( : data%l_ne ), data%RHS_sub, task )
          IF ( n_stat == - 26 ) THEN
            inform%status = GALAHAD_error_unknown_solver ; RETURN ; END IF

!  P is the HSL_MI28 incomplete factorization

        CASE ( preconditioner_incomplete_mi28 )
          CALL MI28_precondition( data%n_sub, data%mi28_keep,                  &
                                  data%RHS_sub( : data%n_sub ),                &
                                  data%SOL_sub( : data%n_sub ),                &
                                  inform%mi28_info )

!  P is one of the other cases ... eventually

        CASE DEFAULT
          WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )         &
            control%preconditioner
          inform%status = GALAHAD_preconditioner_unknown
          RETURN
        END SELECT

!  scatter the solution back to the correct sub-components

        IF ( control%preconditioner == preconditioner_reordered_band ) THEN
          IF ( data%sub_matrix ) THEN
            SOL( data%SUB( : data%n_sub ) )                                    &
              = data%RHS_sub( data%PERM( : data%n_sub ) )
          ELSE
            SOL( : data%n_sub ) = data%RHS_sub( data%PERM( : data%n_sub ) )
          END IF
        ELSE IF ( control%preconditioner == preconditioner_incomplete_mi28) THEN
          IF ( data%sub_matrix ) THEN
            SOL( data%SUB( : data%n_sub ) ) = data%SOL_sub( : data%n_sub )
          ELSE
            SOL( : data%n_sub ) = data%SOL_sub( : data%n_sub )
          END IF
        ELSE
          IF ( data%sub_matrix ) THEN
            SOL( data%SUB( : data%n_sub ) ) = data%RHS_sub( : data%n_sub )
          ELSE
            SOL( : data%n_sub ) = data%RHS_sub( : data%n_sub )
          END IF
        END IF

        CALL CPU_TIME( time_now ) ; CALL CLOCK_time( clock_now )
        inform%time%solve = inform%time%solve + time_now - time_start
        inform%time%clock_solve = clock_now - clock_start

!  - - - - - - - - - - - - - - - - - - - - - - - - -
!  Subsequent solves using the original factorization
!  and the factorization of the Schur-complement
!  - - - - - - - - - - - - - - - - - - - - - - - - -

      ELSE
!       WRITE( 6, "( ' update not yet implemented ' )" )
!       inform%status = GALAHAD_not_yet_implemented

!  Solve for the preconditioned gradient using the Schur complement update.
!  Gather the sub-components of SOL into RHS_scu

        IF ( control%preconditioner == preconditioner_reordered_band ) THEN
          data%RHS_scu( data%PERM( : data%n_sub ) )                            &
            = SOL( data%SUB( : data%n_sub ) )
        ELSE
          data%RHS_scu( : data%n_sub ) = SOL( data%SUB( : data%n_sub ) )
        END IF
        data%RHS_scu( data%n_sub + 1 : data%SCU_matrix%n + data%SCU_matrix%m ) &
          = zero

!  Solve the linear system P * sol_scu = RHS_scu

        scu_status = 1
  10    CONTINUE

!  Call SCU_solve to solve the system

        CALL SCU_solve( data%SCU_matrix, data%SCU_data, data%RHS_scu,          &
                        data%SOL_scu, data%RHS_sub, scu_status )

!  SCU_solve requires additional information. Compute the solution to
!  the system P * x_sub = rhs_sub, returning the solution in rhs_sub

        IF ( scu_status > 0 ) THEN
          SELECT CASE ( control%preconditioner )

!  P is a diagonal matrix

          CASE ( preconditioner_diagonal )
            data%RHS_sub( : data%n_sub ) =                                     &
              data%RHS_sub( : data%n_sub ) / data%DIAG( : data%n_sub )

!  P is a band or re-ordered band matrix

          CASE ( preconditioner_band, preconditioner_reordered_band )
            CALL BAND_solve( data%n_sub, data%semi_bandwidth_used,             &
                             data%DIAG, data%OFFDIA,                           &
                             data%semi_bandwidth_used,                         &
                             data%RHS_sub, band_status )

!  P is the full matrix with Schnabel-Eskow or GMPS modifications

          CASE ( preconditioner_full_se, preconditioner_full_gmps )
            CALL SLS_solve( data%P, data%RHS_sub, data%SLS_data,               &
                            data%SLS_control, inform%SLS_inform )

!  P is the Lin-More' incomplete factorization

          CASE ( preconditioner_incomplete_lm )
            task = 'N'
            n_stat = data%n_sub
            CALL DSTRSOL( n_stat, data%L_offd( : data%l_ne ),                  &
                          data%L_diag( : data%n_sub ),                         &
                          data%L_colptr( : data%n_sub + 1 ),                   &
                          data%L_row( : data%l_ne ), data%RHS_sub, task )
            IF ( n_stat == - 26 ) THEN
              inform%status = GALAHAD_error_unknown_solver ; RETURN ; END IF
            task = 'T'
            CALL DSTRSOL( n_stat, data%L_offd( : data%l_ne ),                  &
                          data%L_diag( : data%n_sub ),                         &
                          data%L_colptr( : data%n_sub + 1 ),                   &
                          data%L_row( : data%l_ne ), data%RHS_sub, task )
            IF ( n_stat == - 26 ) THEN
              inform%status = GALAHAD_error_unknown_solver ; RETURN ; END IF

!  P is the HSL_MI28 incomplete factorization

          CASE ( preconditioner_incomplete_mi28 )
            CALL MI28_precondition( data%n_sub, data%mi28_keep,                &
                                    data%RHS_sub( : data%n_sub ),              &
                                    data%SOL_sub( : data%n_sub ),              &
                                    inform%mi28_info )

!  P is one of the other cases ... eventually

          CASE DEFAULT
            WRITE( 6, "( ' PSLS: case ', I0, ' not yet implemented' )" )       &
              control%preconditioner
            inform%status = GALAHAD_preconditioner_unknown
            RETURN
          END SELECT
          GO TO 10
        END IF

!  Scatter the free components of the solution into SOL

        IF ( control%preconditioner == preconditioner_reordered_band ) THEN
          SOL( data%SUB( : data%n_sub ) )                                      &
            = data%SOL_scu( data%PERM( : data%n_sub ) )
        ELSE IF ( control%preconditioner == preconditioner_incomplete_mi28) THEN
          IF ( data%sub_matrix ) THEN
            SOL( data%SUB( : data%n_sub ) ) = data%SOL_sub( : data%n_sub )
          ELSE
            SOL( : data%n_sub ) = data%SOL_sub( : data%n_sub )
          END IF
        ELSE
          SOL( data%SUB( : data%n_sub ) ) = data%SOL_scu( : data%n_sub )
        END IF
      END IF

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine PSLS_solve

      END SUBROUTINE PSLS_solve

!-*-*-*-*-*-*-*-   P S L S _ P R O D U C T   S U B R O U T I N E   -*-*-*-*-*-*-

      SUBROUTINE PSLS_product( PROD, data, control, inform )

!  Given a symmetrix positive matrix P, forms the product P v
!  v is input in PROD, and the product P v overwrites PROD

!  Dummy arguments

      REAL ( KIND = wp ), INTENT( INOUT ) , DIMENSION ( : ) :: PROD
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine PSLS_product

      END SUBROUTINE PSLS_product

!-*-*-*-*-*-*-*-*-*-   P S L S _ N O R M   F U N C T I O N   -*-*-*-*-*-*-*-*-

      FUNCTION PSLS_norm( A, V, data, control, inform )
      REAL ( KIND = wp ) :: PSLS_norm

!  Given a symmetrix positive matrix P and vector v, forms the norm ||v||_P
!  for which ||v||_P^2 = v^T P v

!  Dummy arguments

      TYPE ( SMT_type ), INTENT( IN ) :: A
      REAL ( KIND = wp ), INTENT( IN ) , DIMENSION ( : ) :: V
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, j, k, l, m
      INTEGER ( KIND = long ) :: j_long
      REAL ( KIND = wp ) :: row_sum, val
      CHARACTER ( LEN = 80 ) :: array_name

      inform%status = GALAHAD_ok

!  compute the norm for the available preconditioners

      SELECT CASE ( control%preconditioner )

!  identity "preconditioner"

      CASE ( preconditioner_none )
        PSLS_norm = TWO_NORM( V )

!  diagonal preconditioner

      CASE ( preconditioner_diagonal )

!  the required norm is || SQRT(D) v ||_2

        PSLS_norm = TWO_NORM( SQRT( data%DIAG ) * V )

!  band preconditioner

      CASE ( preconditioner_band, preconditioner_reordered_band )

!  allocate necessary workspace

        array_name = 'psls: data%W'
        CALL SPACE_resize_array( data%n_sub, data%W,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  form the "backward" product w = L^T v

        DO i = data%n_sub, 1, - 1
          m = MIN( data%semi_bandwidth_used, data%n_sub - i )
!         data%W( i ) =                                                        &
!           V( i ) + SUM( data%OFFDIA( : m, i ) * V( i + 1 : i + m ) )
          row_sum = V( i )
          DO j = 1, m
            row_sum = row_sum + data%OFFDIA( j, i ) *  V( i + j )
          END DO
          data%W( i ) = row_sum
        END DO

!  the required norm is || SQRT(D) w ||_2

        PSLS_norm = TWO_NORM( SQRT( data%DIAG ) * data%W )

!  Schnabel-Eskow modified factorization

      CASE ( preconditioner_full_se )

!  allocate necessary workspace

        array_name = 'psls: data%W'
        CALL SPACE_resize_array( data%n_sub, data%W,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  check to see if there have been any diagonal perturbations, and
!  compute w = Pert * v

        CALL SLS_ENQUIRE( data%SLS_data, inform%SLS_inform,                    &
                          PERTURBATION = data%W )
        data%W( : data%n_sub ) = data%W( : data%n_sub ) * V( : data%n_sub )

!  compute w = w + A * v

        SELECT CASE ( SMT_get( A%type ) )
        CASE ( 'COORDINATE' )
          DO l = 1, A%ne
            i = A%ROW( l ) ; j = A%COL( l )
            IF ( MIN( i, j ) >= 1 .AND. MAX( i, j ) <= data%n_sub ) THEN
              val = A%val( l )
              data%W( i ) = data%W( i ) + val * V( j )
              IF ( i /= j ) data%W( j ) = data%W( j ) + val * V( i )
            END IF
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, data%n_sub
            DO l = A%ptr( i ), A%ptr( i + 1 ) - 1
              j = A%COL( l )
              IF ( j >= 1 .AND. j <= data%n_sub ) THEN
                val = A%val( l )
                data%W( i ) = data%W( i ) + val * V( j )
                IF ( i /= j ) data%W( j ) = data%W( j ) + val * V( i )
              END IF
            END DO
          END DO
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, data%n_sub
            DO j = 1, i
              l = l + 1
              val = A%val( l )
              data%W( i ) = data%W( i ) + val * V( j )
              IF ( i /= j ) data%W( j ) = data%W( j ) + val * V( i )
            END DO
          END DO
        CASE ( 'DIAGNAL' )
          data%W( : data%n_sub ) = data%W( : data%n_sub )                      &
            + A%val(  data%n_sub ) * V( : data%n_sub )
        END SELECT

!  the required norm is sqrt( v^T w )

        PSLS_norm = SQRT( ABS( DOT_PRODUCT( data%W( : data%n_sub ),            &
                                            V( : data%n_sub ) ) ) )

!  GMPS modified factorization

      CASE ( preconditioner_full_gmps )
        inform%status = GALAHAD_norm_unknown
        GO TO 910

!  Lin-More' incomplete Cholesky factorization,

      CASE ( preconditioner_incomplete_lm )

!  allocate necessary workspace

        array_name = 'psls: data%W'
        CALL SPACE_resize_array( data%n_sub, data%W,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  form the "backward" product w = L^T v

         data%W( data%n_sub ) = data%L_diag( data%n_sub ) * V( data%n_sub )
         DO j = data%n_sub - 1, 1, - 1
           row_sum = data%L_diag( j ) * V( j )
           DO k = data%L_colptr( j ), data%L_colptr( j + 1 ) - 1
             row_sum = row_sum + data%L_offd( k ) * V( data%L_row( k ) )
           END DO
           data%W( j ) = row_sum
         END DO

!  the required norm is || w ||_2

        PSLS_norm = TWO_NORM( data%W )

!  HSL_MI28 incomplete Cholesky factorization

      CASE ( preconditioner_incomplete_mi28 )

!  allocate necessary workspace

        array_name = 'psls: data%W'
        CALL SPACE_resize_array( data%n_sub, data%W,                           &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) GO TO 910

!  form the "backward" product w = L^T S^-1 Q^T v

        IF ( ALLOCATED( data%mi28_keep%invp ) ) THEN
          IF ( ALLOCATED( data%mi28_keep%scale ) ) THEN
            data%W( : data%n_sub ) = V( data%mi28_keep%invp( : data%n_sub ) ) &
              / data%mi28_keep%scale( : data%n_sub )
          ELSE
            data%W( : data%n_sub ) = V( data%mi28_keep%invp( : data%n_sub ) )
          END IF
        ELSE
          IF ( ALLOCATED( data%mi28_keep%scale ) ) THEN
            data%W( : data%n_sub )                                             &
              = V( : data%n_sub) / data%mi28_keep%scale( : data%n_sub )
          ELSE
            data%W( : data%n_sub ) = V( : data%n_sub )
          END IF
        END IF

        DO i = 1, data%n_sub
          row_sum = zero
          DO j_long = data%mi28_keep%fact_ptr( i ),                            &
                 data%mi28_keep%fact_ptr( i + 1 ) - 1
            k = data%mi28_keep%fact_row( j_long )
            row_sum = row_sum + data%mi28_keep%fact_val( j_long ) * data%W( k )
          END DO
          data%W( i ) = row_sum
        END DO

!  the required norm is || w ||_2

        PSLS_norm = TWO_NORM( data%W )

!  One of the other cases will be used ... eventually

      CASE DEFAULT
        WRITE( 6, "( ' PSLS_norm: case ', I0, ' not yet implemented' )" )      &
          control%preconditioner
        inform%status = GALAHAD_not_yet_implemented
        GO TO 910
      END SELECT
      RETURN

!  error return

 910 CONTINUE
     PSLS_norm = infinity
     RETURN

!  End of subroutine PSLS_norm

      END FUNCTION PSLS_norm

!-*-*-*-*-*-   P S L S _ T E R M I N A T E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE PSLS_terminate( data, control, inform )

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
!   data    see Subroutine PSLS_initialize
!   control see Subroutine PSLS_initialize
!   inform  see Subroutine PSLS_solve

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( PSLS_control_type ), INTENT( IN ) :: control
      TYPE ( PSLS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( PSLS_data_type ), INTENT( INOUT ) :: data

!  Local variables

      CHARACTER ( LEN = 80 ) :: array_name

!  Deallocate all arrays allocated within SLS

      CALL SLS_terminate( data%SLS_data,                                       &
                          control%SLS_control, inform%SLS_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%SLS_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!  Deallocate all arrays allocated for and within SCU

      array_name = 'cro: data%SCU_matrix%BD_col_start'
      CALL SPACE_dealloc_array( data%SCU_matrix%BD_col_start,                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%SCU_matrix%BD_val'
      CALL SPACE_dealloc_array( data%SCU_matrix%BD_val,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'cro: data%SCU_matrix%BD_row'
      CALL SPACE_dealloc_array( data%SCU_matrix%BD_row,                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      CALL SCU_terminate( data%SCU_data, data%scu_status, data%SCU_inform )
      IF ( control%deallocate_error_fatal .AND.                                &
        data%SCU_inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'psls: data%SUB'
      CALL SPACE_dealloc_array( data%SUB,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%INDEX'
      CALL SPACE_dealloc_array( data%INDEX,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%IW'
      CALL SPACE_dealloc_array( data%IW,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%PERM'
      CALL SPACE_dealloc_array( data%PERM,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%MAPS'
      CALL SPACE_dealloc_array( data%MAPS,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%IKEEP'
      CALL SPACE_dealloc_array( data%IKEEP,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%IW1'
      CALL SPACE_dealloc_array( data%IW1,                                      &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%W'
      CALL SPACE_dealloc_array( data%W,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%RHS_sub'
      CALL SPACE_dealloc_array( data%RHS_sub,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%SOL_sub'
      CALL SPACE_dealloc_array( data%RHS_sub,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%RHS_scu'
      CALL SPACE_dealloc_array( data%RHS_scu,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%PERT'
      CALL SPACE_dealloc_array( data%PERT,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%SOL_scu'
      CALL SPACE_dealloc_array( data%SOL_scu,                                  &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%G'
      CALL SPACE_dealloc_array( data%G,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%DIAG'
      CALL SPACE_dealloc_array( data%DIAG,                                     &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%W1'
      CALL SPACE_dealloc_array( data%W1,                                       &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%OFFDIA'
      CALL SPACE_dealloc_array( data%OFFDIA,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%D'
      CALL SPACE_dealloc_array( data%D,                                        &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%P_diag'
      CALL SPACE_dealloc_array( data%P_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%P_offd'
      CALL SPACE_dealloc_array( data%P_offd,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%L_diag'
      CALL SPACE_dealloc_array( data%L_diag,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%L_offd'
      CALL SPACE_dealloc_array( data%L_offd,                                   &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%P_colptr'
      CALL SPACE_dealloc_array( data%P_colptr,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%P_row'
      CALL SPACE_dealloc_array( data%P_row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%L_colptr'
      CALL SPACE_dealloc_array( data%L_colptr,                                 &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      array_name = 'psls: data%L_row'
      CALL SPACE_dealloc_array( data%L_row,                                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND. inform%status /= GALAHAD_ok )  &
        RETURN

      RETURN

!  End of subroutine PSLS_terminate

      END SUBROUTINE PSLS_terminate

!-*-*-*-*-*-*-*-*-*-   P S L S _ N A M E   F U N C T I O N   -*-*-*-*-*-*-*-*-

      FUNCTION PSLS_name( preconditioner, semi_bandwidth, icfs_vectors )

!  provide character descriptions of the preconditioners provided

      CHARACTER ( LEN = 80 ) :: PSLS_name
      INTEGER, INTENT( IN ) :: preconditioner, semi_bandwidth, icfs_vectors
      PSLS_name =  REPEAT( ' ', 80 )
      SELECT CASE ( preconditioner )
      CASE ( : - 1 )
        PSLS_name =                                                            &
          "no preconditioner"
      CASE ( 0 )
        PSLS_name =                                                            &
          "automatic preconditioner"
      CASE ( 1 )
        PSLS_name =                                                            &
          "diagonal preconditioner"
      CASE ( 2 )
        WRITE( PSLS_name, "( A, I0 )" )                                        &
          "banded preconditioner with semi-bandwidth ", semi_bandwidth
      CASE ( 3 )
        WRITE( PSLS_name, "( A, I0 )" )                                        &
          "reordered banded preconditioner with semi-bandwidth ", semi_bandwidth
      CASE ( 4 )
        PSLS_name =                                                            &
          "full factorization preconditioner with Schnabel-Eskow modification"
      CASE ( 5 )
        PSLS_name =                                                            &
          "full factorization preconditioner with GMPS modification"
      CASE ( 6 )
        WRITE( PSLS_name, "( A, I0, A )" )                                     &
          "Lin-More' incomplete factorization preconditioner with ",           &
          icfs_vectors, " vectors"
      CASE ( 7 )
        PSLS_name =                                                            &
          "Munskgaard incomplete factorization preconditioner"
      CASE ( 8 )
        PSLS_name =                                                            &
          "expanding band preconditioner"
      CASE DEFAULT
        PSLS_name =                                                            &
          "unknown preconditioner"
      END SELECT

!  End of function PSLS_name

      END FUNCTION PSLS_name

!-*-*-*-*-*-*-*-*-   P S L S _ G M P S   S U B R O U T I N E   -*-*-*-*-*-*-*-*

     SUBROUTINE PSLS_gmps( n, rank, data, inform, neg1, neg2, PERM, D )

!   The Gill-Murray-Ponceleon-Saunders code for modifying the negative
!   eigen-components obtained when factorizing a symmetric indefinite
!   matrix (see SOL 90-8, P.19-21) using the GALAHAD package SILS

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, rank
     INTEGER, INTENT( OUT ) :: neg1, neg2
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: PERM
     TYPE ( SLS_data_type ), INTENT( INOUT ) :: data
     TYPE ( SLS_inform_type ), INTENT( INOUT ) :: inform
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( 2, n ) :: D

!-----------------------------------------------
!   L o c a l   V a r i a b l e
!-----------------------------------------------

     INTEGER :: i
     REAL ( KIND = wp ) :: alpha, beta, gamma, tau
     REAL ( KIND = wp ) :: t, c , s, e1, e2, eigen, eigen_zero
     LOGICAL :: oneby1

     eigen_zero = EPSILON( one )
     CALL SLS_enquire( data, inform, PIVOTS = PERM, D = D )
     D( 1, rank + 1 : n ) = zero

!  neg1 and neg2 are the number of negative eigenvalues which arise
!  from small or negative 1x1 and 2x2 block pivots

     neg1 = 0 ; neg2 = 0

!  Loop over all the block pivots

     oneby1 = .TRUE.
     DO i = 1, n

!  Decide if the current block is a 1x1 or 2x2 pivot block

       IF ( oneby1 ) THEN
         IF ( i < n ) THEN
           oneby1 = PERM( i ) > 0
         ELSE
           oneby1 = .TRUE.
         END IF

         alpha = D( 1, i )

!  =========
!  1x1 block
!  =========

         IF ( oneby1 ) THEN

!  Record the eigenvalue

           IF ( alpha /= zero ) THEN
             eigen = one / alpha
           ELSE
             eigen = zero
           END IF

!  Negative 1x1 block
!  ------------------

           IF ( eigen < - eigen_zero ) THEN
             neg1 = neg1 + 1
             D( 1, i ) = - alpha

!  Small 1x1 block
!  ---------------

           ELSE IF ( eigen < eigen_zero ) THEN
             neg1 = neg1 + 1
             D( 1, i ) = one / eigen_zero
           END IF

!  =========
!  2x2 block
!  =========

         ELSE

           beta = D( 2, i )
           gamma = D( 1, i + 1 )

!  2x2 block: (  alpha  beta   ) = (  c  s  ) (  e1     ) (  c  s  )
!             (  beta   gamma  )   (  s -c  ) (     e2  ) (  s -c  )

           IF ( alpha * gamma < beta ** 2 ) THEN
             tau = ( gamma - alpha ) / ( two * beta )
             t = - one / ( ABS( tau ) + SQRT( one + tau ** 2 ) )
             IF ( tau < zero ) t = - t
             c = one / SQRT( one + t ** 2 ) ; s = t * c
             e1 = alpha + beta * t ; e2 = gamma - beta * t

!  Record the first eigenvalue

             eigen = one / e1

!  Change e1 and e2 to their modified values and then multiply the
!  three 2 * 2 matrices to get the modified alpha, beta and gamma

!  Negative first eigenvalue
!  -------------------------

             IF ( eigen < - eigen_zero ) THEN
               neg2 = neg2 + 1
               e1 = - e1

!  Small first eigenvalue
!  ----------------------

             ELSE IF ( eigen < eigen_zero ) THEN
               neg2 = neg2 + 1
               e1 = one / eigen_zero
             END IF

!  Record the second eigenvalue

             eigen = one / e2

!  Negative second eigenvalue
!  --------------------------

             IF ( eigen < - eigen_zero ) THEN
               neg2 = neg2 + 1
               e2 = - e2

!  Small second eigenvalue
!  -----------------------

             ELSE IF ( eigen < eigen_zero ) THEN
               neg2 = neg2 + 1
               e2 = one / eigen_zero
             END IF

!  Record the modified block

             D( 1, i ) = c ** 2 * e1 + s ** 2 * e2
             D( 2, i ) = c * s * ( e1 - e2 )
             D( 1, i + 1 ) = s ** 2 * e1 + c ** 2 * e2
           END IF
         END IF
       ELSE
         oneby1 = .TRUE.
       END IF
     END DO

!  Register the (possibly modified) diagonal blocks

     CALL SLS_alter_d( data, D, inform )

     RETURN

!  End of subroutine PSLS_gmps

     END SUBROUTINE PSLS_gmps

!  End of module GALAHAD_PSLS_double

   END MODULE GALAHAD_PSLS_double
