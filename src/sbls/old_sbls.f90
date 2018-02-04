! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-*-*-*-*- G A L A H A D _ S B L S   M O D U L E -*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   development started May 12th 2004
!   originally released GALAHAD Version 2.0. February 16th 2005

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

!NOT95USE GALAHAD_CPU_time
      USE GALAHAD_SYMBOLS
      USE GALAHAD_SPACE_double
      USE GALAHAD_SMT_double
      USE GALAHAD_SMT_double, ONLY: GLS_type => SMT_type
      USE GALAHAD_QPT_double, ONLY: QPT_keyword_H, QPT_keyword_A
      USE GALAHAD_SORT_double, ONLY: SORT_reorder_by_rows
      USE GALAHAD_SILS_double
      USE GALAHAD_GLS_DOUBLE
      USE GALAHAD_SPECFILE_double
   
      IMPLICIT NONE

      PRIVATE
      PUBLIC :: SBLS_initialize, SBLS_read_specfile, SBLS_basis_solve,         &
                SBLS_form_and_factorize, SBLS_solve, SBLS_solve_explicit,      &
                SBLS_solve_implicit, SBLS_solve_null_space, SBLS_terminate,    &
                SMT_type, SMT_put, SMT_get

!--------------------
!   P r e c i s i o n
!--------------------

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!-------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n s
!-------------------------------------------------

      TYPE, PUBLIC :: SBLS_control_type
        INTEGER :: error, out, print_level
        INTEGER :: indmin, valmin, len_glsmin, itref_max, scaling
        INTEGER :: ordering, preconditioner, factorization, max_col
        INTEGER :: new_a, new_h, new_c, semi_bandwidth
        REAL ( KIND = wp ) :: pivot_tol, pivot_tol_for_basis, zero_pivot
        REAL ( KIND = wp ) :: static_tolerance, static_level, min_diagonal
        LOGICAL :: remove_dependencies, find_basis_by_transpose, affine
        LOGICAL :: perturb_to_make_definite, get_norm_residual, check_basis
        LOGICAL :: space_critical, deallocate_error_fatal
        CHARACTER ( LEN = 30 ) :: prefix
      END TYPE

      TYPE, PUBLIC :: SBLS_explicit_factors_type
        PRIVATE
        INTEGER :: rank_a, len_sol_workspace, g_ne, k_g, k_c, k_pert
        TYPE ( SMT_type ) :: K
        TYPE ( GLS_type ) :: B
        TYPE ( SILS_factors ) :: K_factors
        TYPE ( GLS_factors ) :: B_factors
        TYPE ( SILS_control ) :: K_control
        TYPE ( GLS_control ) :: B_control
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
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_diag
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_orig
      END TYPE

      TYPE, PUBLIC :: SBLS_implicit_factors_type
        PRIVATE
        INTEGER :: rank_a, m, n, k_n, len_sol_workspace, n_r
        LOGICAL :: unitb22, unitb31, unitp22, zerob32, zerob33, zerop11, zerop21
        TYPE ( SMT_type ) :: A2, P11, P21, B22, B32, B33
        TYPE ( GLS_type ) :: A1, B31, P22
        TYPE ( GLS_factors ) :: A1_factors, B31_factors, P22_factors
        TYPE ( GLS_control ) :: A1_control, B31_control, P22_control
        TYPE ( SILS_factors ) :: B22_factors
        TYPE ( SILS_control ) :: B22_control
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_COLS_order
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic
        INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_COLS_basic
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS_orig
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL_perm
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL_current
        REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERT
      END TYPE
      
      TYPE, PUBLIC :: SBLS_null_space_factors_type
        PRIVATE
        INTEGER :: rank_a, m, n, k_n, len_sol_workspace, n_r
        TYPE ( SMT_type ) :: A2, H11, H21, H22, R_sparse
        TYPE ( GLS_type ) :: A1
        TYPE ( GLS_factors ) :: A1_factors
        TYPE ( GLS_control ) :: A1_control
        TYPE ( SILS_factors ) :: R_sparse_factors
        TYPE ( SILS_control ) :: R_sparse_control
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
      
      TYPE, PUBLIC :: SBLS_data_type
        INTEGER :: last_preconditioner, last_factorization, len_sol
        TYPE ( SBLS_explicit_factors_type ) :: efactors
        TYPE ( SBLS_implicit_factors_type ) :: ifactors
        TYPE ( SBLS_null_space_factors_type ) :: nfactors
      END TYPE

      TYPE, PUBLIC :: SBLS_inform_type
        INTEGER :: status, alloc_status
        INTEGER :: sils_analyse_status, sils_factorize_status
        INTEGER :: sils_solve_status
        INTEGER :: gls_analyse_status, gls_solve_status, sort_status
        INTEGER :: factorization_integer, factorization_real
        INTEGER :: preconditioner, factorization, rank
        LOGICAL :: rank_def, perturbed
        REAL ( KIND = wp ) :: norm_residual
        CHARACTER ( LEN = 80 ) :: bad_alloc
      END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

      INTEGER, PARAMETER :: max_sc = 200
      INTEGER, PARAMETER :: no_last = - 1000
      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: epsmch = EPSILON( one )

!---------------------------------
!   I n t e r f a c e  B l o c k s
!---------------------------------

      INTERFACE POTRF

        SUBROUTINE SPOTRF( uplo, n, A, lda, info )
        CHARACTER, INTENT( IN ) :: uplo
        INTEGER, INTENT( IN ) :: n, lda
        INTEGER, INTENT( OUT ) :: info
        REAL, INTENT( INOUT ), DIMENSION( lda, n ) :: A
        END SUBROUTINE SPOTRF

        SUBROUTINE DPOTRF( uplo, n, A, lda, info )
        CHARACTER, INTENT( IN ) :: uplo
        INTEGER, INTENT( IN ) :: n, lda
        INTEGER, INTENT( OUT ) :: info
        DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( lda, n ) :: A
        END SUBROUTINE DPOTRF

      END INTERFACE 

      INTERFACE POTRS

        SUBROUTINE SPOTRS( uplo, n, nrhs, A, lda, B, ldb, info )
        CHARACTER, INTENT( IN ) :: uplo
        INTEGER, INTENT( IN ) :: n, nrhs, lda, ldb
        INTEGER, INTENT( OUT ) :: info
        REAL, INTENT( IN ), DIMENSION( lda, n ) :: A
        REAL, INTENT( INOUT ), DIMENSION( ldb, n ) :: B
        END SUBROUTINE SPOTRS

        SUBROUTINE DPOTRS( uplo, n, nrhs, A, lda, B, ldb, info )
        CHARACTER, INTENT( IN ) :: uplo
        INTEGER, INTENT( IN ) :: n, nrhs, lda, ldb
        INTEGER, INTENT( OUT ) :: info
        DOUBLE PRECISION, INTENT( IN ), DIMENSION( lda, n ) :: A
        DOUBLE PRECISION, INTENT( INOUT ), DIMENSION( ldb, n ) :: B
        END SUBROUTINE DPOTRS

      END INTERFACE 

   CONTAINS

!-*-*-*-*-*-   S B L S  _ I N I T I A L I Z E   S U B R O U T I N E   -*-*-*-*-*

      SUBROUTINE SBLS_initialize( data, control )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Default control data for SBLS. This routine should be called before
!  SBLS_solve
! 
!  --------------------------------------------------------------------
!
!  Arguments:
!
!  control  a structure containing control information. Components are -
!
!  INTEGER control parameters:
!
!   error. Error and warning diagnostics occur on stream error 
!   
!   out. General output occurs on stream out
!   
!   print_level. The level of output required is specified by print_level
!   
!   indmin. An initial guess as to the integer workspace required by SILS
!
!   valmin. An initial guess as to the real workspace required by SILS
! 
!   len_glsmin. An initial guess as to the workspace required by GLS
!
!   itref_max. The maximum number of iterative refinements allowed
!
!   preconditioner. The preconditioner to be used for the CG is defined by 
!    preconditioner. Possible values are
!
!    variable:
!      0  automatic 
!
!    explicit factorization:
!
!      1  no preconditioner, G = I
!      2  full factorization, G = H
!      3  diagonal, G = diag( max( H, %min_diagonal ) )
!      4  banded, G = band( H ) with semi-bandwidth %semi_bandwidth
!     11  G_11 = 0, G_21 = 0, G_22 = H_22
!     12  G_11 = 0, G_21 = H_21, G_22 = H_22
!
!    implicit factorization:
!
!      -1  G_11 = 0, G_21 = 0, G_22 = I
!      -2  G_11 = 0, G_21 = 0, G_22 = H_22
!
!   semi_bandwidth. The semi-bandwidth of a band preconditioner, if appropriate
!
!   factorization. The factorization to be used.
!    Possible values are
!
!      0  automatic 
!      1  Schur-complement factorization
!      2  augmented-system factorization
!      3  null-space factorization
!
!   max_col. The maximum number of nonzeros in a column of A which is permitted
!    with the Schur-complement factorization
!
!   new_h. How much of H has changed since the previous factorization.
!    Possible values are
!
!      0  unchanged
!      1  values but not indices have changed
!      2  values and indices have changed
!
!   new_a. How much of A has changed since the previous factorization.
!    Possible values are
!
!      0  unchanged
!      1  values but not indices have changed
!      2  values and indices have changed 
!
!   new_c. How much of C has changed since the previous factorization.
!    Possible values are
!
!      0  unchanged
!      1  values but not indices have changed
!      2  values and indices have changed
!
!  REAL control parameters:
!
!   pivot_tol. The threshold pivot used by the matrix factorization.
!    See the documentation for SILS for details
!
!   pivot_tol_for_basis. The threshold pivot used by the matrix 
!    factorization when attempting to construct the basis.
!    See the documentation for GLS for details
!
!   zero_pivot. Any pivots smaller than zero_pivot in absolute value will 
!    be regarded to be zero when attempting to detect linearly dependent 
!    constraints
!
!   static_tolerance & static_level (may be) used by SILS
!
!   min_diagonal. Diagonal preconditioners will have diagonals no smaller
!    than  min_diagonal
!
!  LOGICAL control parameters:
!
!   find_basis_by_transpose. If true, implicit factorization preconditioners
!    will be based on a basis of A found by examining A's transpose
!
!   remove_dependencies. If true, the equality constraints will be preprocessed
!    to remove any linear dependencies
!
!   check_basis. If true and an implicit or null-space preconditioner is
!     used, the computed basis matrix will be assessed for ill conditioning 
!     and, if necessary an attempt will be made to correct for this
!
!   affine. If true, the second block component of the right-hand side c
!    will be assumed to be zero. This can lead to some efficiencies 
!    in the solve stage
!
!   perturb_to_make_definite. If true and the initial attempt at finding
!     a preconditioner is unsuccessful, the diagonal will be perturbed so
!     that a second attempt succeeds
!
!  get_norm_residual. If true, the residual when applying the preconditioner
!    will be calculated
!
!   space_critical. If true, every effort will be made to use as little
!    space as possible. This may result in longer computation times
!
!   deallocate_error_fatal. If true, any array/pointer deallocation error
!     will terminate execution. Otherwise, computation will continue
!
!  CHARACTER control parameters:
!
!  prefix (len=30). All output lines will be prefixed by 
!    %prefix(2:LEN(TRIM(%prefix))-1)
!   where %prefix contains the required string enclosed in 
!   quotes, e.g. "string" or 'string'
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( OUT ) :: control        

!  Set control parameters

!  Integer parameters

      control%error  = 6
      control%out  = 6
      control%print_level = 0
      control%indmin = 1000
      control%valmin = 1000
      control%len_glsmin = 1000
      control%itref_max = 1
      control%preconditioner = 0
      control%semi_bandwidth = 5
      control%factorization = 0
      control%max_col = 35
      control%ordering = 3
!57V2 control%ordering = 2
!57V3 control%ordering = 5
      control%scaling = 0
      control%new_h = 2
      control%new_a = 2
      control%new_c = 2

!  Real parameters

      control%pivot_tol = 0.01_wp
!     control%pivot_tol = epsmch ** 0.75
      control%pivot_tol_for_basis = half
      control%zero_pivot = epsmch ** 0.75
      control%static_tolerance = 0.0_wp
      control%static_level = 0.0_wp
      control%min_diagonal = 0.00001_wp

!  Logical parameters

      control%remove_dependencies = .TRUE.
      control%find_basis_by_transpose = .TRUE.
      control%perturb_to_make_definite = .TRUE.
      control%check_basis = .TRUE.
      control%affine = .FALSE.
      control%space_critical = .FALSE.
      control%get_norm_residual = .FALSE.
      control%deallocate_error_fatal  = .FALSE.

!  Character parameters

      control%prefix = '""                            '

!  Ensure that the private data arrays have the correct initial status

      data%last_preconditioner = no_last
      data%last_factorization = no_last
      data%efactors%len_sol_workspace = - 1
      data%ifactors%len_sol_workspace = - 1

!  Initialize control parameters and arrays used within SILS and M48

      CALL GLS_INITIALIZE( data%efactors%B_factors,                            &
                           data%efactors%B_control )
      CALL GLS_INITIALIZE( data%ifactors%A1_factors,                           &
                           data%ifactors%A1_control )
      CALL GLS_INITIALIZE( data%nfactors%A1_factors,                           &
                           data%nfactors%A1_control )
      CALL SILS_INITIALIZE( data%efactors%K_factors,                           &
                            data%efactors%K_control )
      CALL SILS_INITIALIZE( data%ifactors%B22_factors,                         &
                            data%ifactors%B22_control )
      CALL SILS_INITIALIZE( data%nfactors%R_sparse_factors,                    &
                            data%nfactors%R_sparse_control )

      RETURN  

!  End of SBLS_initialize

      END SUBROUTINE SBLS_initialize

!-*-*-*-   S B L S _ R E A D _ S P E C F I L E  S U B R O U T I N E   -*-*-*-

      SUBROUTINE SBLS_read_specfile( control, device, alt_specname )

!  Reads the content of a specification file, and performs the assignment of 
!  values associated with given keywords to the corresponding control parameters

!  The defauly values as given by SBLS_initialize could (roughly) 
!  have been set as:

!  BEGIN SBLS SPECIFICATIONS (DEFAULT)
!   error-printout-device                           6
!   printout-device                                 6
!   print-level                                     0
!   initial-workspace-for-unsymmetric-solver        1000
!   initial-integer-workspace                       1000
!   initial-real-workspace                          1000
!   maximum-refinements                             1
!   preconditioner-used                             0
!   semi-bandwidth-for-band-preconditioner          5
!   factorization-used                              0
!   maximum-column-nonzeros-in-schur-complement     35
!   ordering-used                                   3
!   scaling-used                                    0
!   has-a-changed                                   2
!   has-h-changed                                   2
!   has-c-changed                                   2
!   minimum-diagonal                                1.0D-5
!   pivot-tolerance-used                            1.0D-12
!   pivot-tolerance-used-for-basis                  0.5
!   zero-pivot-tolerance                            1.0D-12
!   static-pivoting-diagonal-perturbation           0.0D+0
!   level-at-which-to-switch-to-static              0.0D+0
!   find-basis-by-transpose                         T
!   check-for-reliable-basis                        T
!   remove-linear-dependencies                      T
!   perturb-to-make-+ve-definite                    T
!   space-critical                                  F
!   deallocate-error-fatal                          F
!   output-line-prefix                              ""
!  END SBLS SPECIFICATIONS

!  Dummy arguments

      TYPE ( SBLS_control_type ), INTENT( INOUT ) :: control        
      INTEGER, INTENT( IN ) :: device
      CHARACTER( LEN = 16 ), OPTIONAL :: alt_specname

!  Programming: Nick Gould and Ph. Toint, January 2002.

!  Local variables

      INTEGER, PARAMETER :: lspec = 36
      CHARACTER( LEN = 16 ), PARAMETER :: specname = 'SBLS'
      TYPE ( SPECFILE_item_type ), DIMENSION( lspec ) :: spec

!  Define the keywords

     spec%keyword = ''

!  Integer key-words

      spec(  1 )%keyword = 'error-printout-device'
      spec(  2 )%keyword = 'printout-device'
      spec(  3 )%keyword = 'print-level' 
      spec(  8 )%keyword = 'initial-workspace-for-unsymmetric-solver'
      spec(  9 )%keyword = 'initial-integer-workspace'
      spec( 10 )%keyword = 'initial-real-workspace'
      spec( 11 )%keyword = 'maximum-refinements'
      spec( 14 )%keyword = 'preconditioner-used'
      spec( 15 )%keyword = 'semi-bandwidth-for-band-preconditioner'
      spec( 16 )%keyword = 'factorization-used'
      spec( 17 )%keyword = 'maximum-column-nonzeros-in-schur-complement'
      spec( 18 )%keyword = 'ordering-used'
      spec( 20 )%keyword = 'scaling-used'
      spec( 21 )%keyword = 'has-a-changed'
      spec( 22 )%keyword = 'has-h-changed'
      spec( 23 )%keyword = 'has-c-changed'

!  Real key-words

      spec( 25 )%keyword = 'minimum-diagonal'
      spec( 26 )%keyword = 'pivot-tolerance-used'
      spec( 27 )%keyword = 'pivot-tolerance-used-for-basis'
      spec( 28 )%keyword = 'zero-pivot-tolerance'
      spec( 29 )%keyword = 'static-pivoting-diagonal-perturbation'
      spec( 30 )%keyword = 'level-at-which-to-switch-to-static'

!  Logical key-words

      spec( 31 )%keyword = 'perturb-to-make-+ve-definite'
      spec( 32 )%keyword = 'space-critical'
      spec( 33 )%keyword = 'find-basis-by-transpose'
      spec( 24 )%keyword = 'check-for-reliable-basis'
      spec( 34 )%keyword = 'remove-linear-dependencies'
      spec( 35 )%keyword = 'deallocate-error-fatal'

!  Character key-words

!     spec( 36 )%keyword = 'output-line-prefix'

!  Read the specfile

     IF ( PRESENT( alt_specname ) ) THEN
       CALL SPECFILE_read( device, alt_specname, spec, lspec, control%error )
     ELSE
       CALL SPECFILE_read( device, specname, spec, lspec, control%error )
     END IF

!  Interpret the result

!  Set integer values

      CALL SPECFILE_assign_value( spec( 1 ), control%error,                    &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 2 ), control%out,                      &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 3 ), control%print_level,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 8 ), control%len_glsmin,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 9 ), control%indmin,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 10 ), control%valmin,                  &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 11 ), control%itref_max,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 14 ), control%preconditioner,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 15 ), control%semi_bandwidth,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 16 ), control%factorization,           &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 17 ), control%max_col,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 18 ), control%ordering,                &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 20 ), control%scaling,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 21 ), control%new_a,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 22 ), control%new_h,                   &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 23 ), control%new_c,                   &
                                  control%error )

!  Set real values

      CALL SPECFILE_assign_value( spec( 25 ), control%min_diagonal,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 26 ), control%pivot_tol,               &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 27 ),                                  &
                                  control%pivot_tol_for_basis,                 &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 28 ), control%zero_pivot,              &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 29 ), control%static_tolerance,        &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 30 ), control%static_level,            &
                                  control%error )

!  Set logical values

      CALL SPECFILE_assign_value( spec( 31 ),                                  &
                                  control%perturb_to_make_definite,            &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 32 ), control%space_critical,          &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 33 ),                                  &
                                  control%find_basis_by_transpose,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 24 ), control%check_basis,             &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 34 ), control%remove_dependencies,     &
                                  control%error )
      CALL SPECFILE_assign_value( spec( 35 ),                                  &
                                  control%deallocate_error_fatal,              &
                                  control%error )

!  Set charcter values

!     CALL SPECFILE_assign_value( spec( 36 ), control%prefix,                  &
!                                 control%error )

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

!  Deallocate all arrays allocated within SILS and GLS

      CALL SILS_finalize( data%efactors%K_factors,                             &
                          data%efactors%K_control, inform%alloc_status )
      IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL GLS_finalize( data%efactors%B_factors,                              &
                         data%efactors%B_control, inform%alloc_status )
      IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL GLS_finalize( data%ifactors%A1_factors,                             &
                         data%ifactors%A1_control, inform%alloc_status )
      IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!     CALL GLS_finalize( data%ifactors%B31_factors,                            &
!                        data%ifactors%B31_control, inform%alloc_status )
!     IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
!       inform%status = GALAHAD_error_deallocate ; RETURN
!     END IF

!     CALL GLS_finalize( data%ifactors%P22_factors,                            &
!                         data%ifactors%P22_control, inform%alloc_status )
!     IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
!       inform%status = GALAHAD_error_deallocate ; RETURN
!     END IF

      CALL SILS_finalize( data%ifactors%B22_factors,                           &
                          data%ifactors%B22_control, inform%alloc_status )
      IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL GLS_finalize( data%nfactors%A1_factors,                             &
                         data%nfactors%A1_control, inform%alloc_status )
      IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

      CALL SILS_finalize( data%nfactors%R_sparse_factors,                      &
                          data%nfactors%R_sparse_control, inform%alloc_status )
      IF ( control%deallocate_error_fatal .AND. inform%alloc_status /= 0 ) THEN
        inform%status = GALAHAD_error_deallocate ; RETURN
      END IF

!  Deallocate all remaining allocated arrays

      array_name = 'sbls: efactors%IW'
      CALL SPACE_dealloc_array( data%efactors%IW,                              &
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

      array_name = 'sbls: nfactors%R_sparse%row'
      CALL SPACE_dealloc_array( data%nfactors%R_sparse%row,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%R_sparse%col'
      CALL SPACE_dealloc_array( data%nfactors%R_sparse%col,                    &
         inform%status, inform%alloc_status, array_name = array_name,          &
         bad_alloc = inform%bad_alloc, out = control%error )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN

      array_name = 'sbls: nfactors%R_sparse%val'
      CALL SPACE_dealloc_array( data%nfactors%R_sparse%val,                    &
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

      SUBROUTINE SBLS_form_and_factorize( n, m, H, A, C, data, control, inform )

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
!  -4 SILS analyse error
!  -5 SILS factorize error
!  -6 SILS solve error
!  -7 GLS analyse error
!  -8 GLS solve error
!  -9 insufficient preconditioner

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: c_ne

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Set default information values

      inform%status = GALAHAD_ok 
      inform%alloc_status = 0 ; inform%bad_alloc = ''
      inform%sils_analyse_status = 0 ; inform%sils_factorize_status = 0
      inform%sils_solve_status = 0
      inform%gls_analyse_status = 0 ; inform%gls_solve_status = 0
      inform%sort_status = 0
      inform%factorization_integer = - 1 ; inform%factorization_real = - 1
      inform%rank = m ; inform%rank_def = .FALSE.
      inform%perturbed = .FALSE.
      inform%norm_residual = - one

!  Check for faulty dimensions

      IF ( n <= 0 .OR. m < 0 .OR.                                              &
           .NOT. QPT_keyword_H( H%type ) .OR.                                  &
           .NOT. QPT_keyword_A( A%type ) ) THEN
        inform%status = GALAHAD_error_restrictions
        IF ( control%error > 0 .AND. control%print_level > 0 )                 &
          WRITE( control%error, 2010 ) prefix, inform%status 
        RETURN
      END IF 

      IF ( control%out >= 0 .AND. control%print_level >= 1 ) THEN
        WRITE( control%out,                                                    &
          "( /, A, ' n = ', I0, ', m = ', I0 )" ) prefix, n, m
      END IF

!  Ensure automatic choices for the preconditioner/factorization have been made

      IF ( data%last_preconditioner /= control%preconditioner ) THEN
        IF ( control%preconditioner == 0 ) THEN
          IF ( data%last_preconditioner == no_last ) THEN
            inform%preconditioner = 1
          ELSE
            inform%preconditioner = data%last_preconditioner
          END IF
        ELSE
          inform%preconditioner = control%preconditioner
        END IF
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
      END IF

!  Only allow the null-space method if C = 0

      IF ( inform%factorization == 3 ) THEN
        IF ( SMT_get( C%type ) == 'ZERO' ) THEN
          c_ne = 0
        ELSE IF ( SMT_get( C%type ) == 'DIAGONAL' ) THEN
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
        WRITE( control%out,                                                    &
          "( A, ' preconditioner = ', I0, ', factorization  = ', I0 )" )       &
            prefix, inform%preconditioner, inform%factorization
      END IF

!  Form and factorize the preconditioner

      IF ( inform%factorization == 3 ) THEN
        CALL SBLS_form_n_factorize_nullspace( n, m, H, A, data%nfactors,       &
                                                 data%last_factorization,      &
                                                 control, inform )
        data%len_sol = data%nfactors%k_n
      ELSE IF ( inform%preconditioner >= 0 ) THEN
        CALL SBLS_form_n_factorize_explicit( n, m, H, A, C, data%efactors,     &
                                              data%last_factorization,         &
                                              control, inform )
        data%len_sol = data%efactors%K%n
      ELSE
        CALL SBLS_form_n_factorize_implicit( n, m, H, A, C, data%ifactors,     &
                                              data%last_factorization,         &
                                              control, inform )
        data%len_sol = data%ifactors%k_n
      END IF

      data%last_preconditioner = inform%preconditioner
      data%last_factorization = inform%factorization

      RETURN

!  Non-executable statements

 2010 FORMAT( ' ', /, A, '   **  Error return ',I3,' from SBLS ' ) 

!  End of subroutine SBLS_form_and_factorize

      END SUBROUTINE SBLS_form_and_factorize

!-*-*-   S B L S _ FORM _ N _ FACTORIZE _ EXPLICIT   S U B R O U T I N E   -*-

      SUBROUTINE SBLS_form_n_factorize_explicit( n, m, H, A, C, efactors,      &
                                                   last_factorization,         &
                                                   control, inform )

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

!  Local variables

      INTEGER :: i, ii, j, k, kk, l, g_ne, kzero, kminus, nb, out, c_ne
      INTEGER :: nnz_col_j, nnz_aat_old, nnz_aat, max_len, new_pos, a_ne, h_ne
      INTEGER :: new_h, new_a, new_c, k_c, k_ne
      REAL ( KIND = wp ) :: al, val
      LOGICAL :: printi, resize, use_schur_complement
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( SILS_AINFO ) :: AINFO_SILS
      TYPE ( SILS_FINFO ) :: FINFO_SILS
      TYPE ( GLS_AINFO ) :: AINFO_GLS
      TYPE ( GLS_FINFO ) :: FINFO_GLS

      REAL :: time_start, time_end

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      out = control%out
      printi = control%print_level >= 1 .AND. out >= 0
      inform%perturbed = .FALSE.

      IF ( inform%preconditioner <= 0 ) THEN
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Use SBLS_form_n_factorize_implicit subroutine instead' )" )  &
            prefix

        inform%status = - 31 ; RETURN
      END IF

      IF ( SMT_get( A%type ) == 'DENSE' ) THEN
        a_ne = m * n
      ELSE IF ( SMT_get( A%type ) == 'SPARSE_BY_ROWS' ) THEN
        a_ne = A%ptr( m + 1 ) - 1
      ELSE
        a_ne = A%ne 
      END IF

      IF ( SMT_get( H%type ) == 'DIAGONAL' ) THEN
        h_ne = n
      ELSE IF ( SMT_get( H%type ) == 'DENSE' ) THEN
        h_ne = ( n * ( n + 1 ) ) / 2
      ELSE IF ( SMT_get( H%type ) == 'SPARSE_BY_ROWS' ) THEN
        h_ne = H%ptr( n + 1 ) - 1
      ELSE
        h_ne = H%ne 
      END IF

      IF ( SMT_get( C%type ) == 'ZERO' ) THEN
        c_ne = 0
      ELSE IF ( SMT_get( C%type ) == 'DIAGONAL' ) THEN
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

!  Form the preconditioner

      CALL CPU_TIME( time_start )

!  First, see if we can get away with a factorization of the Schur complement.

!   =======================
!    USE SCHUR COMPLEMENT
!   =======================

      IF ( inform%factorization == 0 .OR. inform%factorization == 1 ) THEN
         array_name = 'sbls: efactors%IW'
         CALL SPACE_resize_array( n, efactors%IW,                              &
            inform%status, inform%alloc_status, array_name = array_name,       &
            deallocate_error_fatal = control%deallocate_error_fatal,           &
            exact_size = control%space_critical,                               &
            bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        IF ( new_h > 0 ) THEN
          IF ( inform%preconditioner > 4 ) THEN
            inform%factorization = 2
          ELSE IF ( inform%preconditioner == 2 .OR.                            &
                    inform%preconditioner == 4  ) THEN

!  Check to see if there are off-diagonal entries, and that the diagonal
!  is present

            efactors%IW = 0
            SELECT CASE ( SMT_get( H%type ) )
            CASE ( 'DIAGONAL' ) 
              IF ( COUNT( H%val( : n ) == zero ) > 0 ) inform%factorization = 2
            CASE ( 'DENSE' ) 
              inform%factorization = 2
            CASE ( 'SPARSE_BY_ROWS' )
              DO i = 1, n
                DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                  IF ( i /= H%col( l ) .OR.                                    &
                       ( i == H%col( l ) .AND. H%val( l ) == zero ) ) THEN
                    IF ( inform%preconditioner == 2 .OR.                       &
                         ( inform%preconditioner == 4                          &
                         .AND. control%semi_bandwidth /= 0 ) ) THEN
                      inform%factorization = 2
                      EXIT
                    END IF
                  ELSE
                    efactors%IW( i ) = efactors%IW( i ) + 1
                  END IF
                END DO
              END DO
              IF ( COUNT( efactors%IW > 0 ) /= n ) inform%factorization = 2
            CASE ( 'COORDINATE' )
              DO l = 1, H%ne
                i = H%row( l )
                IF ( i /= H%col( l ) .OR.                                      &
                     ( i == H%col( l ) .AND. H%val( l ) == zero ) ) THEN
                  IF ( inform%preconditioner == 2 .OR.                         &
                       ( inform%preconditioner == 4                            &
                       .AND. control%semi_bandwidth /= 0 ) ) THEN
                    inform%factorization = 2
                    EXIT
                  END IF
                ELSE
                  efactors%IW( i ) = efactors%IW( i ) + 1
                END IF
              END DO
              IF ( COUNT( efactors%IW > 0 ) /= n ) inform%factorization = 2
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
            control%max_col, COUNT( efactors%A_col_ptr( 2 : ) > control%max_col )

          IF ( control%factorization == 0 .AND. max_len > control%max_col .AND.&
               m > max_sc ) THEN
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
          efactors%G_diag = one

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
            efactors%G_diag = H%val( : n )
          CASE ( 'DENSE' ) 
            write(6,*) prefix, " shouldn't be here ... "
          CASE ( 'SPARSE_BY_ROWS' )
            efactors%G_diag = zero
            DO i = 1, n
              DO l = H%ptr( i ), H%ptr( i + 1 ) - 1
                efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
              END DO
            END DO
          CASE ( 'COORDINATE' )
            efactors%G_diag = zero
            DO l = 1, H%ne
              i = H%row( l )
              efactors%G_diag( i ) = efactors%G_diag( i ) + H%val( l )
            END DO
          END SELECT

!  The (possibly modified) diagonal of the full matrix

        CASE( 3 )
          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = diag(H) ' )" ) &
            prefix
          efactors%G_diag = zero
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' ) 
            efactors%G_diag = H%val( : n )
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

          efactors%G_diag = MAX( efactors%G_diag, control%min_diagonal )
        END SELECT

        IF ( new_a == 2 ) THEN

!  Now insert the (row/col/val) entries of 
!  A diag(G)(inverse) A(transpose) into K

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            nnz_aat = 0
            l = 0
            DO i = 1, m
              efactors%W = A%val( l + 1 : l + n ) / efactors%G_diag           
              k = 0
              DO j = 1, i
                nnz_aat = nnz_aat + 1
                efactors%K%row( nnz_aat ) = i
                efactors%K%col( nnz_aat ) = j
                efactors%K%val( nnz_aat ) =                                    &
                  DOT_PRODUCT( efactors%W, A%val( k + 1 : k + n ) )
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
 
        ELSE

!  Now insert the (val) entries of A diag(G)(inverse) A(transpose) into K

          SELECT CASE ( SMT_get( A%type ) )
          CASE ( 'DENSE' ) 
            nnz_aat = 0
            l = 0
            DO i = 1, m
              efactors%W = A%val( l + 1 : l + n ) / efactors%G_diag           
              k = 0
              DO j = 1, i
                nnz_aat = nnz_aat + 1
                efactors%K%val( nnz_aat ) =                                    &
                  DOT_PRODUCT( efactors%W, A%val( k + 1 : k + n ) )
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
          IF ( new_c > 0 .AND. c_ne > 0 )                                      &
            efactors%K%val( nnz_aat + 1 : ) = C%val

!   ... or the (row/col/val) entries of C into K

        ELSE
          IF ( c_ne > 0 ) THEN
            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'DIAGONAL' ) 
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
              efactors%K%row( nnz_aat + 1 : ) = C%row
              efactors%K%col( nnz_aat + 1 : ) = C%col
            END SELECT
            efactors%K%val( nnz_aat + 1 : ) = C%val
          END IF
        END IF

!       WRITE( out, "( ' K: m, nnz ', 2I4 )" ) efactors%K%n, efactors%K%ne
!       WRITE( out, "( A, /, ( 10I7) )" ) ' rows =', ( efactors%K%row )
!       WRITE( out, "( A, /, ( 10I7) )" ) ' cols =', ( efactors%K%col )
!       WRITE( out, "( A, /, ( F7.2) )" ) ' vals =', ( efactors%K%val )

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

        CASE( 1 )

!  The identity matrix        

          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = I ' )" ) prefix
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

        CASE( 2 )

!  The full matrix

          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = H ' )" ) prefix
          g_ne = h_ne
          efactors%K%n = n + m
          efactors%K%ne = h_ne + a_ne + c_ne

        CASE( 3 )

!  The (possibly modified) diagonal of the full matrix

          IF ( printi ) WRITE( out, "( A, ' preconditioner: G = diag(H) ' )" ) &
            prefix
          g_ne = n
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

        CASE( 4 )

!  A band (or semi-bandwith control%semibadwith) from the full matrix

          IF ( printi ) WRITE( out, "( A, ' preconditioner G = band(H) ' )" )  &
            prefix
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' ) 
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
            g_ne = COUNT( ABS( H%row - H%col ) <= control%semi_bandwidth )
          END SELECT
          efactors%K%n = n + m
          efactors%K%ne = g_ne + a_ne + c_ne

        CASE( 11 : 12 )

!  Non-basic submatrices of H

          IF ( new_a > 0 ) THEN

!  Find sets of basic rows and columns

            CALL SBLS_find_basis( m, n, a_ne, A, efactors%B,                   &
                                  efactors%B_factors, efactors%B_control,      &
                                  efactors%B_ROWS, efactors%B_COLS,            &
                                  efactors%rank_a,                             &
                                  control%find_basis_by_transpose,             &
                                  prefix, 'sbls: efactors%', out, printi,      &
                                  control, inform, AINFO_GLS, FINFO_GLS )

!  Make a copy of the "basis" matrix

            array_name = 'sbls: efactors%B_COLS_basic'
            CALL SPACE_resize_array( n, efactors%B_COLS_basic,                 &
               inform%status, inform%alloc_status, array_name = array_name,    &
               deallocate_error_fatal = control%deallocate_error_fatal,        &
               exact_size = control%space_critical,                            &
               bad_alloc = inform%bad_alloc, out = control%error )
            IF ( inform%status /= 0 ) RETURN

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
            IF ( printi )                                                       &
              WRITE( out, "( A, ' preconditioner: G = H_22 ' )" ) prefix
            SELECT CASE ( SMT_get( H%type ) )
            CASE ( 'DIAGONAL' ) 
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
            CASE ( 'DIAGONAL' ) 
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
                  IF ( .NOT. ( efactors%B_COLS_basic( i ) > 0 .AND.           &
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

        CASE DEFAULT

!  Anything else

          IF ( printi ) WRITE( out,                                            &
            "( A, ' no option control%preconditioner = ', I8, ' at present')") &
               prefix, inform%preconditioner
          inform%status = - 32 ; RETURN

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

      k_c = g_ne + a_ne
      efactors%k_g = a_ne
      efactors%k_c = efactors%k_g + g_ne
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

      CASE( 1 )

!  The identity matrix        

        IF ( resize .OR. new_a > 1 ) THEN
          DO i = 1, g_ne
            efactors%K%row( a_ne + i ) = i
            efactors%K%col( a_ne + i ) = i
            efactors%K%val( a_ne + i ) = one
          END DO
        END IF

      CASE( 2 )

!  The full matrix

        IF ( resize .OR. new_a > 1 .OR. new_h > 1 ) THEN
          SELECT CASE ( SMT_get( H%type ) )
          CASE ( 'DIAGONAL' ) 
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
        IF ( resize .OR. new_a > 1 .OR. new_h > 0 )                            &
          efactors%K%val( a_ne + 1 : a_ne + h_ne ) = H%val( : h_ne )

      CASE( 3 )

!  The (possibly modified) diagonal of the full matrix

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
            MAX( efactors%K%val( a_ne + 1 : a_ne + g_ne ), control%min_diagonal )
        END IF

      CASE( 4 )

!  A band (or semi-bandwith control%semibadwith) from the full matrix

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

      CASE( 11 )

!  Non-basic submatrices of H

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
                IF ( efactors%B_COLS_basic( i ) < 0 .AND.                       &
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
          CASE ( 'DIAGONAL' ) 
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
            efactors%K%row( k_c + 1 : efactors%K%ne ) = n + C%row
            efactors%K%col( k_c + 1 : efactors%K%ne ) = n + C%col
          END SELECT
          efactors%K%val( k_c + 1 : efactors%K%ne ) = - C%val
        ELSE
          IF ( new_c > 0 ) efactors%K%val( k_c + 1 : efactors%K%ne ) = - C%val
        END IF
      END IF

!  ===========
!   FACTORIZE
!  ===========

 200  CONTINUE

      IF ( printi ) THEN
        IF ( inform%factorization == 1 ) THEN
          WRITE( out,                                                          &
            "( /, A, ' Using SILS to factorize the Schur complement' )" ) prefix
        ELSE
          WRITE( out,                                                          &
            "( /, A, ' Using SILS to factorize the augmented matrix' )" ) prefix
        END IF
      END IF

!  A diagonal perturbation is to be added to the matrix to be factored 
!  to make the resultant diaginally dominant

      IF ( inform%perturbed .AND. .NOT. use_schur_complement ) THEN
        efactors%K%val( efactors%K%ne : k_ne ) = zero
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
!     WRITE( 77, "( A, /, ( 10I7) )" ) ' rows =', ( efactors%K%row )
!     WRITE( 77, "( A, /, ( 10I7) )" ) ' cols =', ( efactors%K%col )
!     WRITE( 77, "( A, /, ( 10F7.2) )" ) ' vals =', ( efactors%K%val )

!     WRITE( 25, "( ' K: row, col, val ', /, 3( 2I6, ES24.16 ) )" )            &
!       ( efactors%K%row( i ), efactors%K%col( i ), efactors%K%val( i ),       &
!         i = 1, efactors%K%ne )

      IF ( new_a > 1 .OR. new_h > 1 .OR. inform%perturbed ) THEN

!  Initialize the factorization data

!       WRITE( out, "( ' n, nnz ', I7, I10 )" ) efactors%K%n, efactors%K%ne
        IF ( control%print_level <= 0 ) THEN
          efactors%K_control%ldiag = 0
          efactors%K_control%lp = - 1
          efactors%K_control%mp = - 1
          efactors%K_control%wp = - 1
          efactors%K_control%sp = - 1
        ELSE
          efactors%K_control%ldiag = control%print_level - 1
          efactors%K_control%lp = control%error
          efactors%K_control%mp = control%out
          efactors%K_control%wp = control%out
          efactors%K_control%sp = control%out
        END IF
!57V2   efactors%K_control%ordering = control%ordering
!57V2   efactors%K_control%scaling = control%scaling
!57V2   efactors%K_control%static_tolerance = control%static_tolerance
!57V2   efactors%K_control%static_level = control%static_level

!       write(6,*) control%ordering, control%scaling
!       write(6,*) control%static_tolerance, control%static_level

!  Analyse the preconditioner

        IF ( efactors%K%n > 0 ) THEN
          CALL SILS_ANALYSE( efactors%K, efactors%K_factors,                   &
                             efactors%K_control, AINFO_SILS )
          inform%sils_analyse_status = AINFO_SILS%flag
          IF ( printi ) WRITE( out,                                            &
             "(  A, ' SILS: analysis complete:      status = ', I0 )" )        &
                 prefix, inform%sils_analyse_status
          IF ( inform%sils_analyse_status < 0 ) THEN
             inform%status = GALAHAD_error_analysis ; RETURN
          END IF
        ELSE
          IF ( printi ) WRITE( out,                                            &
             "(  A, ' no analysis need for matrix of order 0 ')" ) prefix
          inform%sils_analyse_status = 0
        END IF
      END IF

!  Factorize the preconditioner

      IF ( efactors%K%n > 0 ) THEN
        efactors%K_control%U = control%pivot_tol
        CALL SILS_FACTORIZE( efactors%K, efactors%K_factors,                   &
                             efactors%K_control, FINFO_SILS )
        inform%sils_factorize_status = FINFO_SILS%FLAG
        IF ( printi ) WRITE( out,                                              &
          "( A, ' SILS: factorization complete: status = ', I0 )" ) prefix,    &
             inform%sils_factorize_status
        IF ( inform%sils_factorize_status < 0 ) THEN
           inform%status = GALAHAD_error_factorization ; RETURN
        END IF
        IF ( inform%sils_factorize_status == 4 ) THEN
          inform%rank_def = .TRUE.
          inform%rank = FINFO_SILS%rank
        END IF

        IF ( printi ) WRITE( out, "( A, ' K nnz(prec,factors)', 2( 1X, I0 ) )")&
          prefix, efactors%K%ne, FINFO_SILS%NEBDU

        kzero = efactors%K%n - FINFO_SILS%rank
        kminus = FINFO_SILS%NEIG
        IF ( use_schur_complement ) THEN
          IF ( kzero + kminus > 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, 1X, I0, ' -ve and ' , I0,              &
             &  ' zero eigevalues > 0 required ones ' )" )                     &
               prefix, kminus, kzero
            IF ( control%perturb_to_make_definite .AND.                        &
                 .NOT. inform%perturbed ) THEN
              IF ( control%out > 0 .AND. control%print_level > 0 )             &
                WRITE( control%out,                                            &
                  "( A, ' Perturbing G to try to correct this ' )" ) prefix
              inform%factorization = 2 
              inform%perturbed = .TRUE.
              GO TO 100
            ELSE
              inform%status = GALAHAD_error_preconditioner
              RETURN
            END IF
          END IF
        ELSE
          IF ( kzero + kminus > m ) THEN
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
              GO TO 200
            ELSE
              inform%status = GALAHAD_error_preconditioner
              RETURN
            END IF
          END IF
        END IF
      ELSE
        IF ( printi ) WRITE( out,                                              &
           "(  A, ' no factorization need for matrix of order 0 ')" ) prefix
        inform%sils_factorize_status = 0
      END IF

      CALL CPU_TIME( time_end )
      IF ( printi ) WRITE( out,                                                &
         "( A, ' time to form and factorize explicit preconditioner ', F6.2 )")&
        prefix, time_end - time_start

      inform%status = GALAHAD_ok
      RETURN

!  End of subroutine SBLS_form_n_factorize_explicit

      END SUBROUTINE SBLS_form_n_factorize_explicit

!-*-*-   S B L S _ FORM _ AD _ FACTORIZE _ IMPLICIT   S U B R O U T I N E   -*-*-

      SUBROUTINE SBLS_form_n_factorize_implicit( n, m, H, A, C, ifactors,      &
                                                  last_factorization,          &
                                                  control, inform )

!  Form an implicit factorization of
!
!        K = ( G   A^T )
!            ( A    -C )
!
!  for various approximations G of H

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m, last_factorization
      TYPE ( SMT_type ), INTENT( IN ) :: H, A, C
      TYPE ( SBLS_implicit_factors_type ), INTENT( INOUT ) :: ifactors
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform

!  Local variables

      INTEGER :: i, ii, j, jj, l, out, a_ne
      INTEGER :: new_h, new_a, new_c
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( SILS_AINFO ) :: AINFO_SILS
      TYPE ( SILS_FINFO ) :: FINFO_SILS
      TYPE ( GLS_AINFO ) AINFO_GLS
      TYPE ( GLS_FINFO ) FINFO_GLS

      REAL :: time_start, time_end
!     REAL :: t1, t2, t3

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      CALL CPU_TIME( time_start )

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
        inform%status = - 31 ; RETURN
      END IF
      inform%status = GALAHAD_ok

      IF ( last_factorization /= inform%factorization ) THEN
        new_h = 2
        new_a = 2
        new_c = 2
      ELSE
        new_h = control%new_h
        new_a = control%new_a
        new_c = control%new_c
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
        CALL SBLS_find_A1_and_A2( m, n, a_ne, A, ifactors%A1,                  &
                                  ifactors%A1_factors,                         &
                                  ifactors%A1_control, ifactors%A2,            &
                                  ifactors%A_ROWS_basic,                       &
                                  ifactors%A_COLS_basic,                       &
                                  ifactors%A_ROWS_order, ifactors%A_COLS_order,&
                                  ifactors%rank_a, ifactors%k_n, ifactors%n_r, &
                                  prefix, 'sbls: ifactors%', out, printi,      &
                                  control, inform, AINFO_GLS, FINFO_GLS,       &
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
          CASE ( 'DIAGONAL' ) 
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
                     ifactors%A_COLS_order( H%col( l ) ) > ifactors%rank_a ) THEN
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

!       WRITE( out, "( ' m, nnz ', 2I4 )" ) ifactors%B22%n, ifactors%B22%ne
!       WRITE( out, "( A, /, ( 10I7) )" ) ' rows =', ( ifactors%B22%row )
!       WRITE( out, "( A, /, ( 10I7) )" ) ' cols =', ( ifactors%B22%col )
!       WRITE( out, "( A, /, ( 10F7.2) )" ) ' vals =', ( ifactors%B22%val )

!  Now factorize H_22

        IF ( printi ) WRITE( out, "( A, ' Using SILS' )" ) prefix
        IF ( control%print_level <= 0 ) THEN
          ifactors%B22_control%ldiag = 0
          ifactors%B22_control%lp = - 1
          ifactors%B22_control%mp = - 1
          ifactors%B22_control%wp = - 1
          ifactors%B22_control%sp = - 1
        ELSE
          ifactors%B22_control%ldiag = control%print_level - 1
          ifactors%B22_control%lp = control%error
          ifactors%B22_control%mp = control%out
          ifactors%B22_control%wp = control%out
          ifactors%B22_control%sp = control%out
        END IF
!57V2   ifactors%B22_control%ordering = control%ordering
!57V2   ifactors%B22_control%scaling = control%scaling
!57V2   ifactors%B22_control%static_tolerance = control%static_tolerance
!57V2   ifactors%B22_control%static_level = control%static_level
        IF ( control%perturb_to_make_definite ) ifactors%B22_control%pivoting = 4
        CALL SILS_ANALYSE( ifactors%B22, ifactors%B22_factors,                 &
                           ifactors%B22_control, AINFO_SILS )
        inform%sils_analyse_status = AINFO_SILS%FLAG
        IF ( printi ) WRITE( out,                                              &
       "( A, ' Analysis complete:      status = ', I0 )" ) prefix,             &
          inform%sils_analyse_status
        IF ( inform%sils_analyse_status < 0 ) THEN
           inform%status = GALAHAD_error_analysis
           RETURN
        END IF

        CALL SILS_FACTORIZE( ifactors%B22, ifactors%B22_factors,               &
                             ifactors%B22_control, FINFO_SILS )
        inform%sils_factorize_status = FINFO_SILS%FLAG
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Factorization complete: status = ', I0 )" )                  &
            prefix, inform%sils_factorize_status
        IF ( inform%sils_factorize_status < 0 ) THEN
           inform%status = GALAHAD_error_factorization
           RETURN
        END IF
        IF ( inform%sils_factorize_status == 4 ) THEN
          inform%rank_def = .TRUE.
          inform%rank = FINFO_SILS%rank
        END IF

!  Check to ensure that the preconditioner is definite

        IF ( FINFO_SILS%modstep /= 0 ) THEN
          inform%perturbed = .TRUE.
          array_name = 'sbls: ifactors%PERT'
          CALL SPACE_resize_array( ifactors%B22%n, ifactors%PERT,              &
             inform%status, inform%alloc_status, array_name = array_name,      &
             deallocate_error_fatal = control%deallocate_error_fatal,          &
             exact_size = control%space_critical,                              &
             bad_alloc = inform%bad_alloc, out = control%error )
          IF ( inform%status /= GALAHAD_ok ) RETURN

          CALL SILS_ENQUIRE( ifactors%B22_factors, PERTURBATION = ifactors%PERT )
          IF ( printi ) WRITE( out, "( A, ' H_22 perturbed by', ES11.4 )" )    &
            prefix, MAXVAL( ABS( ifactors%PERT ) )
        ELSE
          inform%perturbed = .FALSE.
        END IF

        IF ( FINFO_SILS%NEIG + ifactors%B22%n - FINFO_SILS%rank > 0 ) THEN
          WRITE( out, "( A, ' SILS_FACTORIZE reports B22 is indefinite ' )" )  &
            prefix  
           inform%status = GALAHAD_error_preconditioner ; RETURN
        END IF
        IF ( printi ) WRITE( out, "( A, ' B22 nnz(prec,factors)', 2( 1X, I0))")&
          prefix, ifactors%B22%ne, FINFO_SILS%nrlbdu
        
!  Restore the row and colum indices to make matrix-vector products efficient

        ifactors%B22%row = ifactors%B22%row + ifactors%rank_a
        ifactors%B22%col = ifactors%B22%col + ifactors%rank_a

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
        inform%status = - 32 ; RETURN

      END SELECT

      CALL CPU_TIME( time_end )
      IF ( printi ) WRITE( out,                                                &
        "( A, ' time to form and factorize implicit preconditioner ', F6.2 )" )&
        prefix, time_end - time_start

      RETURN

!  End of  subroutine SBLS_form_n_factorize_implicit

      END SUBROUTINE SBLS_form_n_factorize_implicit

!-*-*-*-*-*-*-*-*-   S B L S _ S O L V E   S U B R O U T I N E   -*-*-*-*-*-*-*-

      SUBROUTINE SBLS_solve( n, m, A, C, data, control, inform, SOL )

!  Solve

!    ( G   A^T ) ( x ) = ( a )
!    ( A   -C  ) ( y )   ( b )

!  Dummy arguments

      INTEGER, INTENT( IN ) :: n, m
      TYPE ( SMT_type ), INTENT( IN ) :: A, C
      TYPE ( SBLS_data_type ), INTENT( INOUT ) :: data
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n + m ) :: SOL

!  Solve the preconditioned system

      IF ( inform%factorization == 3 ) THEN
        CALL SBLS_solve_null_space( data%nfactors, control, inform, SOL )
      ELSE IF ( inform%preconditioner >= 0 ) THEN
        CALL SBLS_solve_explicit( n, m, A, C, data%efactors, control, inform, &
                                  SOL )
      ELSE
        CALL SBLS_solve_implicit( data%ifactors, control, inform, SOL )
      END IF

      RETURN

!  End of subroutine SBLS_solve

      END SUBROUTINE SBLS_solve

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
      TYPE ( SILS_SINFO ) :: SINFO_SILS

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

!  Allocate workspace arrays

      npm = n + m
      IF ( inform%factorization == 1 ) np1 = n + 1
      IF ( efactors%K%n /= efactors%len_sol_workspace ) THEN
!       write(6,*) ' resize '
        array_name = 'sbls: efactors%RHS'
        CALL SPACE_resize_array( npm, efactors%RHS,                            &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN

        array_name = 'sbls: efactors%RHS_orig'
        CALL SPACE_resize_array( npm, efactors%RHS_orig,                       &
           inform%status, inform%alloc_status, array_name = array_name,        &
           deallocate_error_fatal = control%deallocate_error_fatal,            &
           exact_size = .TRUE.,                                                &
           bad_alloc = inform%bad_alloc, out = control%error )
        IF ( inform%status /= GALAHAD_ok ) RETURN
        efactors%len_sol_workspace = efactors%K%n
      END IF

!  Compute the original residual

!     WRITE( 25,"( ' sol ', /, ( 5ES24.16 ) )" ) SOL
      efactors%RHS_orig( : npm ) = SOL( : npm )
      efactors%RHS( : npm ) = SOL( : npm )
      SOL( : npm ) = zero

!  Solve the system with iterative refinement

      IF ( control%print_level > 1 .AND. control%out > 0 )                     &
        WRITE( control%out, "( A, ' maximum residual ', ES12.4 )" )            &
          prefix, MAXVAL( ABS( efactors%RHS( : npm ) ) )

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
!               write(6,*) ' ii, size( efactors%RHS ) ', ii, size( efactors%RHS )
                efactors%RHS( ii ) = efactors%RHS( ii ) +                      &
                  A%val( l ) * efactors%RHS( A%col( l ) )
              END DO
            END SELECT

!  Solve  ( C + A G(inv) A(transpose) ) y = A diag(G)(inverse) a - b
!  and place the result in a

!           WRITE(6,*) efactors%RHS( np1 : npm )
            CALL SILS_SOLVE( efactors%K, efactors%K_factors,                   &
              efactors%RHS( np1 : npm ), efactors%K_control, SINFO_SILS )
            inform%sils_solve_status = SINFO_SILS%flag
            IF ( inform%sils_solve_status < 0 ) THEN
              IF ( control%out > 0 .AND. control%print_level > 0 )             &
                WRITE( control%out, "( A, ' solve exit status = ', I3 )" )     &
                  prefix, inform%sils_solve_status
              inform%status = GALAHAD_error_solve
              RETURN
            END IF
!           WRITE(6,*) efactors%RHS( np1 : npm )

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
!           WRITE(6,*) ' sol ', efactors%RHS
          END IF

        ELSE

!  Use factors of the augmented system

          CALL SILS_SOLVE( efactors%K, efactors%K_factors, efactors%RHS,       &
                           efactors%K_control, SINFO_SILS )
          inform%sils_solve_status = SINFO_SILS%flag
          IF ( inform%sils_solve_status < 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' solve exit status = ', I3 )" )       &
                prefix, inform%sils_solve_status
            inform%status = GALAHAD_error_solve
            RETURN
          END IF
        END IF

!  Update the estimate of the solution

        SOL( : npm ) = SOL( : npm ) + efactors%RHS( : npm )

!  Form the residuals

        IF ( iter < control%itref_max .OR. control%get_norm_residual ) THEN  

!  ... for the case where G is diagonal ...

          IF ( inform%factorization == 1 ) THEN

            efactors%RHS( : n ) =                                              &
              efactors%RHS_orig( : n ) - efactors%G_diag( : n ) * SOL( : n )

            SELECT CASE ( SMT_get( C%type ) )
            CASE ( 'ZERO' ) 
              efactors%RHS( np1 : npm ) = efactors%RHS_orig( np1 : npm )
            CASE ( 'DIAGONAL' ) 
              efactors%RHS( np1 : npm ) =                                      &
                efactors%RHS_orig( np1 : npm ) + C%val( : m ) * SOL( np1 : npm )
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

          ELSE

!     WRITE( 25, "( ' kg, kpert, kne ', 3I10 )" ) &
!        efactors%k_g, efactors%k_pert, efactors%K%ne
!     WRITE( 25, "( ' K: row, col, val ', /, 3( 2I6, ES24.16 ) )" )            &
!       ( efactors%K%row( i ), efactors%K%col( i ), efactors%K%val( i ),       &
!         i = 1, efactors%K%ne )

!  ... or the case of general G

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

          IF ( control%print_level > 1 .AND. control%out > 0 )                 &
            WRITE( control%out, "( A, ' maximum residual ', ES12.4 )" )        &
              prefix, MAXVAL( ABS( efactors%RHS( : npm ) ) )
        END IF
      END DO

      IF ( control%get_norm_residual )                                         &
        inform%norm_residual = MAXVAL( ABS( efactors%RHS( : npm ) ) )

      RETURN

!  End of subroutine SBLS_solve_explicit

      END SUBROUTINE SBLS_solve_explicit

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

      INTEGER :: i, iter, j, l, rank_a, n, k_n, trans
      INTEGER :: start_1, end_1, start_2, end_2, start_3, end_3
      REAL ( KIND = wp ) :: val
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( SILS_SINFO ) :: SINFO_SILS
      TYPE ( GLS_SINFO ) :: SINFO_GLS

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

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
            WRITE( 6, "( ' not unitb31 not written yet ' )" )
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
            WRITE( 6, "( ' not unitp22 not written yet ' )" )
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
            WRITE( 6, "( ' not unitb31 not written yet ' )" )
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
            WRITE( 6, "( ' not unitb31 not written yet ' )" )
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
              SOL( i ) = SOL( i ) - ifactors%P11%val( l )                       &
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
            WRITE( 6, "( ' not unitp22 not written yet ' )" )
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
            WRITE( 6, "( ' not unitb31 not written yet ' )" )
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
            WRITE( 6, "( ' not unitb31 not written yet ' )" )
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

        trans = 1
        CALL GLS_SOLVE( ifactors%A1, ifactors%A1_factors,                      &
                        SOL( start_1 : end_1 ),                                &
                        ifactors%SOL_perm( start_3 : end_3 ),                  &
                        ifactors%A1_control, SINFO_GLS, TRANS = trans )
        inform%gls_solve_status = SINFO_GLS%flag
        IF ( inform%gls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_gls_solve ; RETURN
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
          WRITE( 6, "( ' not unitp22 not written yet ' )" )
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
          WRITE( 6, "( ' not unitb31 not written yet ' )" )
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
          CALL SILS_SOLVE( ifactors%B22, ifactors%B22_factors,                &
            SOL( start_2 : end_2 ),  ifactors%B22_control, SINFO_SILS )
          inform%sils_solve_status = SINFO_SILS%flag
          IF ( inform%sils_solve_status < 0 ) THEN
            IF ( control%out > 0 .AND. control%print_level > 0 )               &
              WRITE( control%out, "( A, ' solve exit status = ', I3 )" )       &
                prefix, inform%sils_solve_status
            inform%status = GALAHAD_error_solve
            RETURN
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
          WRITE( 6, "( ' not unitpb31 not written yet ' )" )
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
          WRITE( 6, "( ' not unitp22 not written yet ' )" )
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

        CALL GLS_SOLVE( ifactors%A1, ifactors%A1_factors,                      &
                        SOL( start_3 : end_3 ),                                &
                        ifactors%SOL_perm( start_1 : end_1 ),                  &
                        ifactors%A1_control, SINFO_GLS )
        inform%gls_solve_status = SINFO_GLS%flag
        IF ( inform%gls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_gls_solve ; RETURN
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
          WRITE( 6, "( ' not unitb31 not written yet ' )" )
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
      REAL ( KIND = wp ), INTENT( INOUT ),                                    &
                          DIMENSION( ifactors%n + ifactors%m ) :: SOL

!  Local variables

      INTEGER :: i, iter, l, rank_a, n, k_n
      INTEGER :: start_1, end_1, start_2, end_2, start_3, end_3
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( GLS_SINFO ) :: SINFO_GLS

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
!         WRITE(6,*) ' residual = ', MAXVAL( ABS( SOL( start_3 : end_3 ) ) ),  &
!           ' x ', MAXVAL( ABS( ifactors%SOL_current( start_1 : end_1 ) ) )
        END IF

!  1. Solve A_1 w_1 = c ...

        CALL GLS_SOLVE( ifactors%A1, ifactors%A1_factors,                      &
                        SOL( start_3 : end_3 ),                                &
                        ifactors%SOL_perm( start_1 : end_1 ),                  &
                        ifactors%A1_control, SINFO_GLS )
        inform%gls_solve_status = SINFO_GLS%flag
        IF ( inform%gls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_gls_solve ; RETURN
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
      INTEGER :: liw, lptr, trans, new_h, new_a, error
      LOGICAL :: printi, printe
      CHARACTER ( LEN = 80 ) :: array_name
!     TYPE ( SILS_AINFO ) :: AINFO_SILS
!     TYPE ( SILS_FINFO ) :: FINFO_SILS
      TYPE ( GLS_AINFO ) AINFO_GLS
      TYPE ( GLS_FINFO ) FINFO_GLS
      TYPE ( GLS_SINFO ) :: SINFO_GLS

      REAL ( KIND = wp ) :: val
      REAL :: time_start, time_end
!     REAL :: t1, t2, t3

      LOGICAL :: dense = .TRUE.
!     LOGICAL :: printd = .TRUE.
      LOGICAL :: printd = .FALSE.

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
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
!         "( A, ' Use SBLS_form_n_factorize_explicit subroutine instead' )" )&
!           prefix
!       inform%status = - 31 ; RETURN
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
                                  nfactors%A1_factors,                         &
                                  nfactors%A1_control, nfactors%A2,            &
                                  nfactors%A_ROWS_basic,                       &
                                  nfactors%A_COLS_basic,                       &
                                  nfactors%A_ROWS_order, nfactors%A_COLS_order,&
                                  nfactors%rank_a, nfactors%k_n, nfactors%n_r, &
                                  prefix, 'sbls: nfactors%', out, printi,      &
                                  control, inform, AINFO_GLS, FINFO_GLS,       &
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
         "( A, ' GLS: sort of A2, info = ', I0 )" )  prefix, inform%sort_status
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
        CASE ( 'DIAGONAL' ) 
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
          CALL SORT_reorder_by_rows(                                           &
            nfactors%n_r, nfactors%rank_a, nfactors%H21%ne, nfactors%H21%row,  &
            nfactors%H21%col, nfactors%H21%ne, nfactors%H21%val,               &
            nfactors%H21%ptr, lptr, nfactors%IW, liw, control%error,           &
            control%out, inform%sort_status )
          IF ( inform%sort_status /= 0 ) THEN
            IF ( printe ) WRITE( error,                                        &
         "( A, ' GLS: sort of H21, info = ', I0 )" )  prefix, inform%sort_status
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

        END IF

        IF ( nfactors%H22%ne > 0 ) THEN
          CALL SORT_reorder_by_rows(                                           &
            nfactors%n_r, nfactors%n_r, nfactors%H22%ne, nfactors%H22%row,     &
            nfactors%H22%col, nfactors%H22%ne, nfactors%H22%val,               &
            nfactors%H22%ptr, lptr, nfactors%IW, liw, control%error,           &
            control%out, inform%sort_status )
          IF ( inform%sort_status /= 0 ) THEN
            IF ( printe ) WRITE( error,                                        &
         "( A, ' GLS: sort of H22, info = ', I0 )" )  prefix, inform%sort_status
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

          CALL GLS_SOLVE( nfactors%A1, nfactors%A1_factors,                   &
                          nfactors%V( : nfactors%rank_a ),                    &
                          nfactors%W( : nfactors%rank_a ),                    & 
                          nfactors%A1_control, SINFO_GLS )
          inform%gls_solve_status = SINFO_GLS%flag
          IF ( inform%gls_solve_status < 0 ) THEN
            IF ( printi ) WRITE( out, "( A, ' column ', I0, ' phase 2 ' )" )   &
              prefix, k
            inform%status = GALAHAD_error_gls_solve ; RETURN
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
            nfactors%V( i ) =                                                 &
              nfactors%V( i ) + nfactors%H11%val( l ) * nfactors%W( j )
            IF ( i /= j ) nfactors%V( j ) =                                   &
              nfactors%V( j ) + nfactors%H11%val( l ) * nfactors%W( i )
          END DO
          IF ( printd ) WRITE( 6, "( ' v ', ( 4ES12.4 ) )" ) nfactors%V

!  5. Next form w = A_1^-T v

          trans = 1
          CALL GLS_SOLVE( nfactors%A1, nfactors%A1_factors,                    &
                          nfactors%V( : nfactors%rank_a ),                     &
                          nfactors%W( : nfactors%rank_a ),                     &
                          nfactors%A1_control, SINFO_GLS, TRANS = trans )
          inform%gls_solve_status = SINFO_GLS%flag
          IF ( inform%gls_solve_status < 0 ) THEN
              IF ( printi ) WRITE( out, "( A, ' column ', I0, ' phase 5 ' )" ) &
                prefix, k
            inform%status = GALAHAD_error_gls_solve ; RETURN
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
        inform%status = - 32 ; RETURN

      END SELECT

!  Now factorize R

      IF ( dense ) THEN
!       write(6,*) maxval( nfactors%R( : nfactors%n_r, : nfactors%n_r ) )
        nfactors%R_factors = nfactors%R
        CALL POTRF( 'L', nfactors%n_r, nfactors%R_factors, nfactors%n_r,       &
                    potrf_info )
      ELSE
        IF ( printi ) WRITE( out,                                              &
          "( A, ' Sparse reduced Hessian option not implemented at present' )")&
             prefix
        inform%status = - 32 ; RETURN
      END IF

      CALL CPU_TIME( time_end )
      IF ( printi ) WRITE( out,                                                &
        "( A, ' time to form and factorize null-space preconditioner ', F6.2)")&
        prefix, time_end - time_start

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

      INTEGER :: i, iter, j, l, rank_a, k_n, n, out, trans, potrs_info
      INTEGER :: start_1, end_1, start_2, end_2, start_3, end_3
      LOGICAL :: printi
      CHARACTER ( LEN = 80 ) :: array_name
!     TYPE ( SILS_SINFO ) :: SINFO_SILS
      TYPE ( GLS_SINFO ) :: SINFO_GLS

!  prefix for all output 

      CHARACTER ( LEN = LEN( TRIM( control%prefix ) ) - 2 ) :: prefix
      prefix = control%prefix( 2 : LEN( TRIM( control%prefix ) ) - 1 )

      out = control%out
      printi = control%print_level >= 1

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




! ***** testing remove ****

!write(6,*) ' ***** testing remove **** '





!         nfactors%RHS( : k_n ) = zero
!         nfactors%SOL_current( : rank_a ) = one
!         nfactors%SOL_current( rank_a + 1 : n ) = one
!         nfactors%SOL_current( n + 1: k_n ) = one

!         DO l = 1, nfactors%A1%ne
!            i = nfactors%A1%row( l )
!           j = nfactors%A1%col( l )
!           nfactors%RHS( n + i ) = nfactors%RHS( n + i ) +                   &
!             nfactors%A1%val( l ) * nfactors%SOL_current( j )
!           nfactors%RHS( j ) = nfactors%RHS( j ) +                           &
!             nfactors%A1%val( l ) * nfactors%SOL_current( n + i )
!         END DO

!         DO l = 1, nfactors%A2%ne
!           i = nfactors%A2%row( l )
!           j = nfactors%A2%col( l )
!           nfactors%RHS( n + i ) = nfactors%RHS( n + i ) +                   &
!             nfactors%A2%val( l ) * nfactors%SOL_current( j )
!           nfactors%RHS( j ) = nfactors%RHS( j ) +                           &
!             nfactors%A2%val( l ) * nfactors%SOL_current( n + i )
!         END DO

!           DO l = 1, nfactors%H11%ne
!             i = nfactors%H11%row( l )
!             j = nfactors%H11%col( l )
!             nfactors%RHS( i ) = nfactors%RHS( i ) +                         &
!               nfactors%H11%val( l ) * nfactors%SOL_current( j )
!             IF ( i /= j ) nfactors%RHS( j ) = nfactors%RHS( j ) +           &
!               nfactors%H11%val( l ) * nfactors%SOL_current( i )
!           END DO
!           DO l = 1, nfactors%H21%ne
!             i = nfactors%H21%row( l )
!             j = nfactors%H21%col( l )
!             nfactors%RHS( i ) = nfactors%RHS( i ) +                         &
!               nfactors%H21%val( l ) * nfactors%SOL_current( j )
!             nfactors%RHS( j ) = nfactors%RHS( j ) +                         &
!               nfactors%H21%val( l ) * nfactors%SOL_current( i )
!           END DO
!           DO l = 1, nfactors%H22%ne
!             i = nfactors%H22%row( l )
!             j = nfactors%H22%col( l )
!             nfactors%RHS( i ) = nfactors%RHS( i ) +                         &
!               nfactors%H22%val( l ) * nfactors%SOL_current( j )
!           END DO

!         nfactors%RHS_orig = nfactors%RHS( : k_n )

! ***** end of testing ****


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
            nfactors%RHS( n + i ) = nfactors%RHS( n + i ) -                   &
              nfactors%A1%val( l ) * nfactors%SOL_current( j )
            nfactors%RHS( j ) = nfactors%RHS( j ) -                           &
              nfactors%A1%val( l ) * nfactors%SOL_current( n + i )
          END DO

!  Terms involving A_2 and A_2^T

          DO l = 1, nfactors%A2%ne
            i = nfactors%A2%row( l )
            j = nfactors%A2%col( l )
            nfactors%RHS( n + i ) = nfactors%RHS( n + i ) -                   &
              nfactors%A2%val( l ) * nfactors%SOL_current( j )
            nfactors%RHS( j ) = nfactors%RHS( j ) -                           &
              nfactors%A2%val( l ) * nfactors%SOL_current( n + i )
          END DO

!  Case: W = H_22

          IF ( inform%preconditioner == 2 ) THEN

            DO l = 1, nfactors%H11%ne
              i = nfactors%H11%row( l )
              j = nfactors%H11%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                         &
                nfactors%H11%val( l ) * nfactors%SOL_current( j )
              IF ( i /= j ) nfactors%RHS( j ) = nfactors%RHS( j ) -           &
                nfactors%H11%val( l ) * nfactors%SOL_current( i )
            END DO
            DO l = 1, nfactors%H21%ne
              i = nfactors%H21%row( l )
              j = nfactors%H21%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                         &
                nfactors%H21%val( l ) * nfactors%SOL_current( j )
              nfactors%RHS( j ) = nfactors%RHS( j ) -                         &
                nfactors%H21%val( l ) * nfactors%SOL_current( i )
            END DO
            DO l = 1, nfactors%H22%ne
              i = nfactors%H22%row( l )
              j = nfactors%H22%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                         &
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
              nfactors%RHS( i ) = nfactors%RHS( i ) -                         &
                nfactors%H11%val( l ) * nfactors%SOL_current( j )
              IF ( i /= j ) nfactors%RHS( j ) = nfactors%RHS( j ) -           &
                nfactors%H11%val( l ) * nfactors%SOL_current( i )
            END DO
            DO l = 1, nfactors%H21%ne
              i = nfactors%H21%row( l )
              j = nfactors%H21%col( l )
              nfactors%RHS( i ) = nfactors%RHS( i ) -                         &
                nfactors%H21%val( l ) * nfactors%SOL_current( j )
              nfactors%RHS( j ) = nfactors%RHS( j ) -                         &
                nfactors%H21%val( l ) * nfactors%SOL_current( i )
            END DO

!  Terms involving W = R + ( H_21   A_2^T ) ( H_11  A_1 )^-1 ( H_21^T )
!                                            (  A_1   0  )    (  A_2   )

            IF ( printi ) WRITE( out,                                         &
              "( A, ' general residuals not implemented at present' )")       &
                 prefix
            inform%status = - 33 ; RETURN
          END IF

!  No residual required

!         WRITE( 6, "( A, /, ( 5ES12.4 ) )" ) ' solution ',                   &
!           nfactors%SOL_current( : k_n )
        ELSE
          nfactors%RHS( : k_n ) = nfactors%RHS_orig
        END IF
!       WRITE( 6, "( A, /, ( 5ES12.4 ) )" ) ' residuals ',  nfactors%RHS( : k_n )
        IF ( printi ) WRITE( out, "( A, ' maximum residual = ', ES12.4 )" )   &
          prefix, MAXVAL( ABS( nfactors%RHS( : k_n ) ) )

!    *****************
!    *  SOLVE STAGE  *
!    *****************

!         write(6,*) nfactors%RHS( start_1 : end_1 )
!         write(6,*) nfactors%RHS( start_2 : end_2 )
!         write(6,*) nfactors%RHS( start_3 : end_3 )

!  1. Solve A_1 y_1 = r_3 (store y_1 in v)

        IF ( .NOT. control%affine ) THEN
          CALL GLS_SOLVE( nfactors%A1, nfactors%A1_factors,                    &
                          nfactors%RHS( start_3 : end_3 ),                     &
                          nfactors%V,                                          &
                          nfactors%A1_control, SINFO_GLS )
          inform%gls_solve_status = SINFO_GLS%flag
          IF ( inform%gls_solve_status < 0 ) THEN
            inform%status = GALAHAD_error_gls_solve ; RETURN
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

        trans = 1
        CALL GLS_SOLVE( nfactors%A1, nfactors%A1_factors,                      &
                        nfactors%W,                                            &
                        nfactors%V,                                            &
                        nfactors%A1_control, SINFO_GLS, TRANS = trans )
        inform%gls_solve_status = SINFO_GLS%flag
        IF ( inform%gls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_gls_solve ; RETURN
        END IF

!  3b. Form r_2 - A_2^T y_3 -> r_2

        DO l = 1, nfactors%A2%ne
          j = nfactors%A2%col( l )
          nfactors%RHS( j ) = nfactors%RHS( j ) -                              &
            nfactors%A2%val( l ) * nfactors%V( nfactors%A2%row( l ) )
!         write(6,*) ' 2 ', j, nfactors%RHS( j ), nfactors%A2%row( l )
        END DO

!  4. Find S^-1 r_2 -> x_2

        nfactors%SOL_perm( start_2 : end_2, 1 ) =                              &
          nfactors%RHS( start_2 : end_2 )
!       write(6,*) maxval( nfactors%R_factors( : nfactors%n_r, : nfactors%n_r ) )
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

        CALL GLS_SOLVE( nfactors%A1, nfactors%A1_factors,                      &
                        nfactors%RHS( start_3 : end_3 ),                       &
                        nfactors%SOL_perm( start_1 : end_1, 1 ),               &
                        nfactors%A1_control, SINFO_GLS )
        inform%gls_solve_status = SINFO_GLS%flag
        IF ( inform%gls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_gls_solve ; RETURN
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

        trans = 1
        CALL GLS_SOLVE( nfactors%A1, nfactors%A1_factors,                      &
                        nfactors%RHS( start_1 : end_1 ),                       &
                        nfactors%SOL_perm( start_3 : end_3, 1 ),               &
                        nfactors%A1_control, SINFO_GLS, TRANS = trans )
        inform%gls_solve_status = SINFO_GLS%flag
        IF ( inform%gls_solve_status < 0 ) THEN
          inform%status = GALAHAD_error_gls_solve ; RETURN
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

      RETURN

!  End of subroutine SBLS_solve_null_space

      END SUBROUTINE SBLS_solve_null_space

!-*-*-*-   S B L S _ F I N D _ A 1 _ A N D _ A 2   S U B R O U T I N E   -*-*-*-

      SUBROUTINE SBLS_find_A1_and_A2( m, n, a_ne, A, A1, A1_factors,            &
                                      A1_control, A2, A_ROWS_basic,             &
                                      A_COLS_basic, A_ROWS_order, A_COLS_order, &
                                     rank_a, k_n, n_r, prefix, resize_prefix,   &
                                     out, printi, control, inform,              &
                                     AINFO_GLS, FINFO_GLS, RHS, SOL )

!  Given a rectangular matrix A, find a "basic" set of rows and colums, that is
!  a non-singular submatrix A1 of maximal rank. Also set up the complement
!  matrix A2 of columns of A not in A1

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n, a_ne, out
      INTEGER, INTENT( OUT ) :: rank_a, n_r, k_n
      LOGICAL, INTENT( IN ) :: printi
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( GLS_type ), INTENT( inout ) :: A1
      TYPE ( SMT_type ), INTENT( INOUT ) :: A2
      TYPE ( GLS_factors ), INTENT( INOUT ) :: A1_factors
      TYPE ( GLS_control ), INTENT( INOUT ) :: A1_control
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic, A_COLS_basic
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_order, A_COLS_order
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS, SOL
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = * ), INTENT( IN ) :: resize_prefix
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( GLS_AINFO ), INTENT( INOUT ) :: AINFO_GLS
      TYPE ( GLS_FINFO ), INTENT( INOUT ) :: FINFO_GLS

!  Local variables

      INTEGER :: i, ii, j, jj, l, info, dependent, nb
      LOGICAL :: find_basis_by_transpose
      CHARACTER ( LEN = 80 ) :: array_name
      TYPE ( GLS_SINFO ) :: SINFO_GLS

!     LOGICAL :: printd = .TRUE.
      LOGICAL :: printd = .FALSE.

      find_basis_by_transpose = control%find_basis_by_transpose
      
  100 CONTINUE

!  Find sets of basic rows and columns

      CALL SBLS_find_basis( m, n, a_ne, A, A1, A1_factors, A1_control,         &
                            A_ROWS_basic, A_COLS_basic, rank_a,                &
                            find_basis_by_transpose,                           &
                            prefix, resize_prefix, out, printi, control,       &
                            inform, AINFO_GLS, FINFO_GLS )

!     CALL CPU_TIME( t2 )
!     WRITE(6,"(' time to find basis ',F6.2)") t2 - t1
      IF ( inform%status /= GALAHAD_ok ) RETURN

      k_n = n + rank_a ; n_r = n - rank_a

!  Print out rank and identifying vectors

      IF ( out > 0 .AND. control%print_level >= 2 ) THEN
        WRITE( out, "( /, A, ' First-pass factorization ' )" ) prefix
        WRITE( out, "( A, A, 3I0 )" ) prefix, ' m, rank, n = ',                &
          m, rank_a, n
        WRITE( out, "( A, A, 2I0 )" ) prefix, ' A_ne, factors_ne = ',          &
           a_ne, FINFO_GLS%SIZE_FACTOR
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

!     WRITE( 6, "( A, /, (10I7) )" ) ' rbasics =', A_ROWS_basic( : m )
!     WRITE( 6, "( A, /, (10I7) )" ) ' cbasics =', A_COLS_basic( : n )
!     WRITE( 6, "( A, /, (10I7) )" ) ' order =', A_COLS_order( : n )

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

      CALL GLS_finalize( A1_factors, A1_control, info )
      IF ( control%deallocate_error_fatal .AND.                                &
           inform%status /= GALAHAD_ok ) RETURN
      CALL GLS_INITIALIZE( A1_factors, A1_control )
      IF ( control%print_level <= 0 ) THEN
        A1_control%ldiag = 0
        A1_control%lp = - 1 ; A1_control%mp = - 1 ; A1_control%wp = - 1
      ELSE
        A1_control%ldiag = control%print_level - 1
        A1_control%lp = control%error
        A1_control%mp = control%out ; A1_control%wp = control%out
      END IF
      A1_control%la = control%len_glsmin
      A1_control%fill_in = 3

      CALL GLS_ANALYSE( A1, A1_factors, A1_control, AINFO_GLS, FINFO_GLS )
      inform%gls_analyse_status = AINFO_GLS%flag
      IF ( printi ) WRITE( out,                                                &
         "( A, ' GLS: analysis of A1 complete: status = ', I0 )" )             &
             prefix, inform%gls_analyse_status
      IF ( inform%gls_analyse_status < 0 ) THEN
         inform%status = GALAHAD_error_gls_analysis ; RETURN
      END IF
      IF ( inform%gls_analyse_status == 4 ) THEN
        inform%rank_def = .TRUE.
        inform%rank = AINFO_GLS%rank
      END IF

      IF ( printi ) WRITE( out, "( A, ' A1 nnz(prec,factors)', 2( 1X, I0 ))")  &
        prefix, A1%ne, FINFO_GLS%size_factor

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

        CALL GLS_SOLVE( A1, A1_factors, RHS( : rank_a ), SOL( : rank_a ),      &
                        A1_control, SINFO_GLS )
        inform%gls_solve_status = SINFO_GLS%flag
!       write(6,*) ' gls solve ', inform%gls_solve_status
!       write(6,"( A, /, ( 5ES12.4 ) )" ) ' gls solve - solution ', SOL( : 5 )
        IF ( inform%gls_solve_status < 0 ) THEN
          IF ( find_basis_by_transpose .EQV.                                   &
               control%find_basis_by_transpose ) THEN
            find_basis_by_transpose = .NOT. control%find_basis_by_transpose
            IF ( printi )                                                      &
              WRITE( out, "( A, ' basis unstable - recomputing ' )" ) prefix
            GO TO 100
          ELSE
            IF ( printi )                                                      &
              WRITE( out, "( A, ' error return - basis unstable ' )" ) prefix
            inform%status = GALAHAD_error_gls_solve ; RETURN
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

      RETURN

!  End of subroutine SBLS_find_A1_and_A2

      END SUBROUTINE SBLS_find_A1_and_A2

!-*-*-*-*-*-   S B L S _ F I N D _ B A S I S   S U B R O U T I N E   -*-*-*-*-*-

      SUBROUTINE SBLS_find_basis( m, n, a_ne, A, A1, A1_factors, A1_control,    &
                                  A_ROWS_basic, A_COLS_basic, rank_a,           &
                                  find_basis_by_transpose,                      &
                                  prefix, resize_prefix, out, printi, control,  &
                                  inform, AINFO_GLS, FINFO_GLS )

!  Given a rectangular matrix A, find a "basic" set of rows and colums, that is
!  those that give a non-singular submatrix A1 of maximal rank

!  Dummy arguments

      INTEGER, INTENT( IN ) :: m, n, a_ne, out
      INTEGER, INTENT( OUT ) :: rank_a
      LOGICAL, INTENT( IN ) :: printi, find_basis_by_transpose
      TYPE ( SMT_type ), INTENT( IN ) :: A
      TYPE ( GLS_type ), INTENT( inout ) :: A1
      TYPE ( GLS_factors ), INTENT( INOUT ) :: A1_factors
      TYPE ( GLS_control ), INTENT( INOUT ) :: A1_control
      INTEGER, ALLOCATABLE, DIMENSION( : ) :: A_ROWS_basic, A_COLS_basic
      CHARACTER ( LEN = * ), INTENT( IN ) :: prefix
      CHARACTER ( LEN = * ), INTENT( IN ) :: resize_prefix
      TYPE ( SBLS_control_type ), INTENT( IN ) :: control
      TYPE ( SBLS_inform_type ), INTENT( INOUT ) :: inform
      TYPE ( GLS_AINFO ), INTENT( INOUT ) :: AINFO_GLS
      TYPE ( GLS_FINFO ), INTENT( INOUT ) :: FINFO_GLS

!  Local variables

      INTEGER :: i, j, l, info
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

        IF ( control%print_level <= 0 ) THEN
          A1_control%ldiag = 0
          A1_control%lp = - 1
          A1_control%mp = - 1
          A1_control%wp = - 1
        ELSE
          A1_control%ldiag = control%print_level - 1
          A1_control%lp = control%error
          A1_control%mp = control%out
          A1_control%wp = control%out
        END IF
        A1_control%u = control%pivot_tol_for_basis
        A1_control%la = control%len_glsmin
        A1_control%fill_in = 3
        A1_control%pivoting = MAX( A1%m, A1%n ) + 1

!       out = 6
!       WRITE( out, "( ' A: m, n, nnz ', 3I4 )" )                              &
!         A1%m, A1%n, A1%ne
!       WRITE( out, "( A, /, ( 10I7) )" ) ' rows =', ( A1%row )
!       WRITE( out, "( A, /, ( 10I7) )" ) ' cols =', ( A1%col )
!       WRITE( out, "( A, /, ( F7.2) )" ) ' vals =', ( A1%val )

! Analyse and factorize

!       A1_control%ldiag = 3
!       A1_control%lp = 6 ; A1_control%mp = 6 ; A1_control%wp = 6
!       CALL CPU_TIME( t3 )
        CALL GLS_ANALYSE( A1, A1_factors,                  &
                          A1_control, AINFO_GLS, FINFO_GLS )
!       CALL CPU_TIME( t2 )

        inform%gls_analyse_status = AINFO_GLS%flag
        IF ( printi ) WRITE( out,                                              &
           "( /, A, ' GLS: analysis of A  complete: status = ', I0 )" )        &
               prefix, inform%gls_analyse_status
        IF ( printi ) WRITE( out, "( A, ' A nnz(prec,factors)', 2( 1X, I0))" ) &
          prefix, A1%ne, FINFO_GLS%size_factor

        IF ( inform%gls_analyse_status < 0 ) THEN
           inform%status = GALAHAD_error_gls_analysis ; RETURN
        END IF
        IF ( inform%gls_analyse_status == 4 ) THEN
          inform%rank_def = .TRUE.
          inform%rank = AINFO_GLS%rank
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

!       CALL CPU_TIME( t3 )
        CALL GLS_SPECIAL_ROWS_AND_COLS( A1_factors, rank_a, A_COLS_basic,      &
                                        A_ROWS_basic, info )
!       CALL CPU_TIME( t2 )
!       WRITE(6,"(' time to find nonsingular submatrix ',F6.2)") t2 - t3


!  ===================
!  Investigate using A
!  ===================

      ELSE
        A1%m = m ; A1%n = n

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

! Initialize the structures

        IF ( control%print_level <= 0 ) THEN
          A1_control%ldiag = 0
          A1_control%lp = - 1
          A1_control%mp = - 1
          A1_control%wp = - 1
        ELSE
          A1_control%ldiag = control%print_level - 1
          A1_control%lp = control%error
          A1_control%mp = control%out
          A1_control%wp = control%out
        END IF
        A1_control%u = control%pivot_tol_for_basis
        A1_control%la = control%len_glsmin
        A1_control%fill_in = 3

! Analyse and factorize

!       A1_control%ldiag = 3
!       A1_control%lp = 6 ; A1_control%mp = 6 ; A1_control%wp = 6
!       CALL CPU_TIME( t3 )
        CALL GLS_ANALYSE( A1, A1_factors, A1_control, AINFO_GLS, FINFO_GLS )
!       CALL CPU_TIME( t2 )

        IF ( printi ) WRITE( out, "( A, ' A nnz(prec,factors)', 2( 1X, I0 ))") &
          prefix, A1%ne, FINFO_GLS%size_factor

        inform%gls_analyse_status = AINFO_GLS%flag
        IF ( printi ) WRITE( out,                                              &
           "( /, A, ' GLS: analysis of A  complete: status = ', I0 )" )        &
               prefix, inform%gls_analyse_status
        IF ( inform%gls_analyse_status < 0 ) THEN
           inform%status = GALAHAD_error_gls_analysis ; RETURN
        END IF
        IF ( inform%gls_analyse_status == 4 ) THEN
          inform%rank_def = .TRUE.
          inform%rank = AINFO_GLS%rank
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

        CALL GLS_SPECIAL_ROWS_AND_COLS( A1_factors, rank_a, A_ROWS_basic,      &
                                        A_COLS_basic, info )
      END IF    

!     CALL CPU_TIME( t2 )
!     WRITE(6,"(' time to find basis ',F6.2)") t2 - t1

!  Record rank-defficient problems

      IF ( rank_a < MIN( m, n ) ) THEN
        inform%rank_def = .TRUE.
        IF ( out > 0 .AND. control%print_level >= 1 ) WRITE( out,              &
          "( /, A, ' ** WARNING nullity A = ', I10 )" ) prefix,                &
          MIN( m, n ) - rank_a
      END IF

      RETURN

!  End of subroutine SBLS_find_basis

      END SUBROUTINE SBLS_find_basis

!  End of module SBLS

   END MODULE GALAHAD_SBLS_double
