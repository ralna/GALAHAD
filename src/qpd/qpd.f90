! THIS VERSION: GALAHAD 2.6 - 12/09/2013 AT 14:30 GMT.

!-*-*-*-*-*-*-*-*-*-  G A L A H A D _ Q P D  M O D U L E  -*-*-*-*-*-*-*-*-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal author: Nick Gould

!  History -
!   originally released with GALAHAD Version 2.0. August 10th 2005

!  For full documentation, see
!   http://galahad.rl.ac.uk/galahad-www/specs.html

!     -----------------------------------------------------
!     | Provides a generic derived type to hold and share |
!     | private data between the GALAHAD QP packages      |
!     |      NOT INTENDED FOR PUBLIC CONSUMPTION          |
!     -----------------------------------------------------

   MODULE GALAHAD_QPD_double

     USE GALAHAD_STRING_double, ONLY: STRING_real_12
     USE GALAHAD_SYMBOLS
     USE GALAHAD_RAND_double, ONLY: RAND_seed
     USE GALAHAD_SMT_double, ONLY: SMT_put, SMT_get
     USE GALAHAD_SILS_double, ONLY: SILS_factors, SILS_control,                &
                                    SILS_ainfo, SILS_finfo, SMT_type
     USE GALAHAD_ULS_double, ONLY: ULS_data_type, ULS_control_type
     USE GALAHAD_SLS_double, ONLY: SLS_data_type, SLS_control_type
     USE GALAHAD_SBLS_double, ONLY: SBLS_data_type, SBLS_control_type
     USE GALAHAD_CRO_double, ONLY: CRO_data_type, CRO_control_type
     USE GALAHAD_FDC_double, ONLY: FDC_data_type, FDC_control_type
     USE GALAHAD_GLTR_double, ONLY: GLTR_data_type, GLTR_control_type
     USE GALAHAD_LPQP_double, ONLY: LPQP_data_type, LPQP_control_type
     USE GALAHAD_FIT_double, ONLY: FIT_data_type
     USE GALAHAD_ROOTS_double, ONLY: ROOTS_data_type
     USE GALAHAD_SCU_double, ONLY: SCU_matrix_type, SCU_info_type,             &
                                   SCU_data_type
     USE GALAHAD_LMS_double, ONLY: LMS_control_type, LMS_inform_type,          &
                                   LMS_apply_lbfgs
!    USE GALAHAD_LMT_double, LMS_control_type => LMT_control_type,             &
!                            LMS_inform_type => LMT_inform_type
     USE GALAHAD_QPP_double, QPD_dims_type => QPP_dims_type
     USE GALAHAD_SCALE_double, ONLY: SCALE_trans_type, SCALE_data_type
     USE GALAHAD_PRESOLVE_double, ONLY: PRESOLVE_data_type,                    &
                                        PRESOLVE_control_type

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: QPD_HX, QPD_AX, QPD_abs_HX, QPD_abs_AX, QPD_SIF,                &
               QPD_solve_separable_BQP

!--------------------
!   P r e c i s i o n
!--------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  ==============================
!  The QPD_data_type derived type
!  ==============================

     TYPE, PUBLIC :: QPD_data_type

! -----------------
!  Scalar componets
! -----------------

!  Common scalar components

       INTEGER :: start_print
       INTEGER :: stop_print
       INTEGER :: m
       INTEGER :: n
       INTEGER :: a_ne
       INTEGER :: h_ne
       LOGICAL :: new_problem_structure

!  QPA scalar components

       INTEGER :: prec_hist
       LOGICAL :: auto_prec
       LOGICAL :: auto_fact

!  QPB/LSQP scalar components

       INTEGER :: trans = 0
       INTEGER :: hist = 0
       INTEGER :: deriv = 0
       INTEGER :: order = 0
       INTEGER :: len_hist = 0
       LOGICAL :: tried_to_remove_deps = .FALSE.
       LOGICAL :: save_structure = .TRUE.

!  L1QP components

       LOGICAL :: new_problem_structure_dqp
       LOGICAL :: save_structure_dqp = .TRUE.
       LOGICAL :: is_lp

!  EQP scalar components

       INTEGER :: n_depen = 0
       LOGICAL :: new_c = .TRUE.
       LOGICAL :: eqp_factors = .FALSE.

!  DQP/DLP scalar components

       INTEGER :: dual_starting_point = 0
       INTEGER :: m_active = 0
       INTEGER :: n_active = 0
       INTEGER :: m_ref = 0
       LOGICAL :: refactor = .TRUE.
       LOGICAL :: subspace_direct = .FALSE.
       REAL ( KIND = wp ) :: cpu_total = 0.0_wp
       REAL ( KIND = wp ) :: clock_total = 0.0_wp

! -----------------------
!  Allocatable components
! -----------------------

!  Common allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: B_stat
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_stat
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_row
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Abycol_ptr
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Abycol_val
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SOL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H_s
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_s
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SH
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SA
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_u

!  QPA, QPB and EQP allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S

!  QPA & QPB allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD

!  QPA & LSQP allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IBREAK

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAKP

!  QPB & LSQP allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Index_C_freed
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Index_C_more_freed
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IW
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: K_colptr
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: list_hist

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BARRIER_C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BARRIER_X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BEST
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BEST_y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_freed
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_more_freed
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DELTA
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIAG_C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIAG_X
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_X_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_X_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_C_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_C_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: fit_f
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: fit_mu
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_L
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HX
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: mu_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_x
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_y
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: SCALE_C
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X0
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_trial
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y_last
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Z_last

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: BINOMIAL
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: C_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: X_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Y_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Y_l_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Y_u_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Z_l_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: Z_u_coef

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: C_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: X_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: Y_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: Y_l_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: Y_u_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: Z_l_hist
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : , : ) :: Z_u_hist

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DC_zh
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX_zh
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_zh
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_l_zh
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_u_zh
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_l_zh
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_u_zh

!  QPA and EQP allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VECTOR

!  LPB allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_last

!  EQP allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_depen
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_eqp
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G_f
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: WORK

!  QPA allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_up_or_low
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: P
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: PERM
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: REF
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: SC
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: S_col
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: S_colptr
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: S_row
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_up_or_low

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: A_norms
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: B
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERT
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: P_pcg
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_print
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RES_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: R_pcg
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_perm
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: S_val
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_pcg

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: D
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: DIAG

!  QPB allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: H_band_ptr
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Index_C_fixed
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: Index_X_fixed

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C_fixed
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_X_phi
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GRAD_C_phi
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: X_fixed

!  LSQP and WPC allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COEF0
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COEF1
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COEF2
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COEF3
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COEF4
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DELTA_cor
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_cor_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DY_cor_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_cor_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DZ_cor_u

!  WPC allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_Y_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_Y_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_Z_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DIST_Z_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MU
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MU_C_L
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MU_C_U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MU_X_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: MU_X_U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_C_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_C_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_X_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_X_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_Y_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_Y_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_Z_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PERTURB_Z_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y

!  CQP allocatable components

       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: OPT_alpha
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: OPT_merit
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: CS_coef
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: COEF
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ROOTS
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: W

!  DQP allocatable components

       INTEGER, ALLOCATABLE, DIMENSION( : ) :: NZ_p
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: IUSED
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX_r
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: INDEX_w
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: V_status
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_status_old
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_status_old
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: X_active
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_active
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: CHANGES
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ACTIVE_list
       INTEGER, ALLOCATABLE, DIMENSION( : ) :: ACTIVE_status
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: BREAK_points
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: YC_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: YC_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ZC_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: ZC_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GY_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GY_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GZ_l
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GZ_u
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: V0
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: VT
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: GV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: G
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: PV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: HPV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DV
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: U
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: H
       REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: V_bnd

! -----------------------
!  Derived type componets
! -----------------------

!  Common derived type components

       TYPE ( QPD_dims_type ) :: dims
       TYPE ( SMT_type ) :: K, H_sbls, A_sbls, C_sbls, A_eqp, H_eqp
       TYPE ( SILS_factors ) :: FACTORS
       TYPE ( SILS_control ) :: CNTL
       TYPE ( QPP_control_type ) :: QPP_control
       TYPE ( QPP_inform_type ) :: QPP_inform
       TYPE ( QPP_map_type ) :: QPP_map

!  LSQP derived type components

       TYPE ( QPP_map_type ) :: QPP_map_fixed, QPP_map_freed
       TYPE ( QPP_map_type ) :: QPP_map_more_freed
       TYPE ( QPD_dims_type ) :: dims_save_fixed, dims_save_freed
       TYPE ( QPD_dims_type ) :: dims_save_more_freed

!  QPA derived type components

       TYPE ( RAND_seed ) :: seed
       TYPE ( SILS_control ) :: CNTLA
       TYPE ( SILS_ainfo ) :: AINFO
       TYPE ( SILS_finfo ) :: FINFO
       TYPE ( SCU_matrix_type ) :: SCU_mat
       TYPE ( SCU_info_type ) :: SCU_info
       TYPE ( SCU_data_type ) :: SCU_data

!  EQP derived type components

        TYPE ( SMT_type ) :: C0

!  L1QP derived type components

       TYPE ( QPP_map_type ) :: QPP_map_dqp
       TYPE ( QPD_dims_type ) :: dims_dqp

!  SCALE derived type components

        TYPE ( SCALE_trans_type ) :: SCALE_trans
        TYPE ( SCALE_data_type ) :: SCALE_data

!  PRESOLVE derived type components

       TYPE ( PRESOLVE_data_type ) :: PRESOLVE_data
       TYPE ( PRESOLVE_control_type ) :: PRESOLVE_control

!  LPQP derived type components

       TYPE ( LPQP_data_type ) :: LPQP_data
       TYPE ( LPQP_control_type ) :: LPQP_control

!  ULS derived type components

       TYPE ( ULS_data_type ) :: ULS_data
       TYPE ( ULS_control_type ) :: ULS_control

!  SLS derived type components

       TYPE ( SLS_data_type ) :: SLS_data
       TYPE ( SLS_control_type ) :: SLS_control

!  SBLS derived type components

       TYPE ( SBLS_data_type ) :: SBLS_data
       TYPE ( SBLS_control_type ) :: SBLS_control

!  FDC derived type components

       TYPE ( FDC_data_type ) :: FDC_data
       TYPE ( FDC_control_type ) :: FDC_control

!  CRO derived type components

       TYPE ( CRO_data_type ) :: CRO_data
       TYPE ( CRO_control_type ) :: CRO_control

!  GLTR derived type components

       TYPE ( GLTR_data_type ) :: GLTR_data
       TYPE ( GLTR_control_type ) :: GLTR_control

!  FIT derived type components

       TYPE ( FIT_data_type ) :: FIT_data

!  ROOTS derived type components

       TYPE ( ROOTS_data_type ) :: ROOTS_data

!  LMS derived type components

       TYPE ( LMS_control_type ) :: control
       TYPE ( LMS_inform_type ) :: inform

     END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

! -------------------------------------------
!  Subroutines shared between the QP packages
! -------------------------------------------

   CONTAINS

!-*-*-*-*-*-*-*-*-*-*-   Q P D _ H X  S U B R O U T I N E  -*-*-*-*-*-*-*-*-*-

      SUBROUTINE QPD_HX( dims, n, R, H_ne, H_val, H_col, H_ptr, X, op,         &
                         semibw, H_band_ptr )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ............................................
!      .                                          .
!      .  Perform the operation r := r op H * x   .
!         where op is + or -                      .
!      ............................................

!  Arguments:
!  =========
!
!   dims    see module GALAHAD_QPP
!   H_*     sparse storage by rows or band
!   X       the vector x
!   R       the result of adding H * x to r
!   semibw  if present, only those entries within a band of semi-bandwidth
!           semibw will be accessed
!   op      character string "+" or "-"

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPD_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, H_ne
      INTEGER, OPTIONAL, INTENT( IN ) :: semibw
      CHARACTER( LEN = 1 ), INTENT( IN ) :: op
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( n ) :: H_band_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ne ) ::  H_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( H_ne ) :: H_val
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: R

!  Local variables

      INTEGER :: hd_start, hd_end, hnd_start, hnd_end, i, j, l, type
      REAL ( KIND = wp ) :: xi, ri

!  For a banded portion of H

      IF ( PRESENT( semibw ) ) THEN

        IF ( op( 1 : 1 ) == '+' ) THEN

!  r <- r + H * x (commented out since it is not used at present)

!         DO type = 1, 6

!           SELECT CASE( type )
!           CASE ( 1 )

!             hd_start  = 1
!             hd_end    = dims%h_diag_end_free
!             hnd_start = hd_end + 1
!             hnd_end   = dims%x_free

!           CASE ( 2 )

!             hd_start  = dims%x_free + 1
!             hd_end    = dims%h_diag_end_nonneg
!             hnd_start = hd_end + 1
!             hnd_end   = dims%x_l_start - 1

!           CASE ( 3 )

!             hd_start  = dims%x_l_start
!             hd_end    = dims%h_diag_end_lower
!             hnd_start = hd_end + 1
!             hnd_end   = dims%x_u_start - 1

!           CASE ( 4 )

!             hd_start  = dims%x_u_start
!             hd_end    = dims%h_diag_end_range
!             hnd_start = hd_end + 1
!             hnd_end   = dims%x_l_end

!           CASE ( 5 )

!             hd_start  = dims%x_l_end + 1
!             hd_end    = dims%h_diag_end_upper
!             hnd_start = hd_end + 1
!             hnd_end   = dims%x_u_end

!           CASE ( 6 )

!             hd_start  = dims%x_u_end + 1
!             hd_end    = dims%h_diag_end_nonpos
!             hnd_start = hd_end + 1
!             hnd_end   = n

!           END SELECT

!  rows with a diagonal entry

!           hd_end = MIN( hd_end, n )
!           DO i = hd_start, hd_end
!             DO l = H_band_ptr( i ), H_ptr( i + 1 ) - 2
!               j = H_col( l )
!               R( j ) = R( j ) + H_val( l ) * X( i )
!               R( i ) = R( i ) + H_val( l ) * X( j )
!             END DO
!             R( i ) = R( i ) + H_val( H_ptr( i + 1 ) - 1 ) * X( i )
!           END DO
!           IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

!           hnd_end = MIN( hnd_end, n )
!           DO i = hnd_start, hnd_end
!             DO l = H_band_ptr( i ), H_ptr( i + 1 ) - 1
!               j = H_col( l )
!               R( j ) = R( j ) + H_val( l ) * X( i )
!               R( i ) = R( i ) + H_val( l ) * X( j )
!             END DO
!           END DO
!           IF ( hnd_end == n ) EXIT

!         END DO
        ELSE

!  r <- r - H * x

          DO type = 1, 6

            SELECT CASE( type )
            CASE ( 1 )

              hd_start  = 1
              hd_end    = dims%h_diag_end_free
              hnd_start = hd_end + 1
              hnd_end   = dims%x_free

            CASE ( 2 )

              hd_start  = dims%x_free + 1
              hd_end    = dims%h_diag_end_nonneg
              hnd_start = hd_end + 1
              hnd_end   = dims%x_l_start - 1

            CASE ( 3 )

              hd_start  = dims%x_l_start
              hd_end    = dims%h_diag_end_lower
              hnd_start = hd_end + 1
              hnd_end   = dims%x_u_start - 1

            CASE ( 4 )

              hd_start  = dims%x_u_start
              hd_end    = dims%h_diag_end_range
              hnd_start = hd_end + 1
              hnd_end   = dims%x_l_end

            CASE ( 5 )

              hd_start  = dims%x_l_end + 1
              hd_end    = dims%h_diag_end_upper
              hnd_start = hd_end + 1
              hnd_end   = dims%x_u_end

            CASE ( 6 )

              hd_start  = dims%x_u_end + 1
              hd_end    = dims%h_diag_end_nonpos
              hnd_start = hd_end + 1
              hnd_end   = n

            END SELECT

!  rows with a diagonal entry

            hd_end = MIN( hd_end, n )
            DO i = hd_start, hd_end
              xi = X( i )
              ri = R( i )
              DO l = H_band_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l )
                R( j ) = R( j ) - H_val( l ) * xi
                ri = ri - H_val( l ) * X( j )
              END DO
              R( i ) = ri - H_val( H_ptr( i + 1 ) - 1 ) * xi
            END DO
            IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

            hnd_end = MIN( hnd_end, n )
            DO i = hnd_start, hnd_end
              xi = X( i )
              ri = R( i )
              DO l = H_band_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                R( j ) = R( j ) - H_val( l ) * xi
                ri = ri - H_val( l ) * X( j )
              END DO
              R( i ) = ri
            END DO
            IF ( hnd_end == n ) EXIT

          END DO
        END IF

!  For the whole of H

      ELSE
        IF ( op( 1 : 1 ) == '+' ) THEN

!  r <- r + H * x

          DO type = 1, 6

            SELECT CASE( type )
            CASE ( 1 )

              hd_start  = 1
              hd_end    = dims%h_diag_end_free
              hnd_start = hd_end + 1
              hnd_end   = dims%x_free

            CASE ( 2 )

              hd_start  = dims%x_free + 1
              hd_end    = dims%h_diag_end_nonneg
              hnd_start = hd_end + 1
              hnd_end   = dims%x_l_start - 1

            CASE ( 3 )

              hd_start  = dims%x_l_start
              hd_end    = dims%h_diag_end_lower
              hnd_start = hd_end + 1
              hnd_end   = dims%x_u_start - 1

            CASE ( 4 )

              hd_start  = dims%x_u_start
              hd_end    = dims%h_diag_end_range
              hnd_start = hd_end + 1
              hnd_end   = dims%x_l_end

            CASE ( 5 )

              hd_start  = dims%x_l_end + 1
              hd_end    = dims%h_diag_end_upper
              hnd_start = hd_end + 1
              hnd_end   = dims%x_u_end

            CASE ( 6 )

              hd_start  = dims%x_u_end + 1
              hd_end    = dims%h_diag_end_nonpos
              hnd_start = hd_end + 1
              hnd_end   = n

            END SELECT

!  rows with a diagonal entry

            hd_end = MIN( hd_end, n )
            DO i = hd_start, hd_end
              xi = X( i )
              ri = R( i )
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l )
                R( j ) = R( j ) + H_val( l ) * xi
                ri = ri + H_val( l ) * X( j )
              END DO
              R( i ) = ri + H_val( H_ptr( i + 1 ) - 1 ) * xi
            END DO
            IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

            hnd_end = MIN( hnd_end, n )
            DO i = hnd_start, hnd_end
              xi = X( i )
              ri = R( i )
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                R( j ) = R( j ) + H_val( l ) * xi
                ri = ri + H_val( l ) * X( j )
              END DO
              R( i ) = ri
            END DO
            IF ( hnd_end == n ) EXIT

          END DO
        ELSE

!  r <- r - H * x

          DO type = 1, 6

            SELECT CASE( type )
            CASE ( 1 )

              hd_start  = 1
              hd_end    = dims%h_diag_end_free
              hnd_start = hd_end + 1
              hnd_end   = dims%x_free

            CASE ( 2 )

              hd_start  = dims%x_free + 1
              hd_end    = dims%h_diag_end_nonneg
              hnd_start = hd_end + 1
              hnd_end   = dims%x_l_start - 1

            CASE ( 3 )

              hd_start  = dims%x_l_start
              hd_end    = dims%h_diag_end_lower
              hnd_start = hd_end + 1
              hnd_end   = dims%x_u_start - 1

            CASE ( 4 )

              hd_start  = dims%x_u_start
              hd_end    = dims%h_diag_end_range
              hnd_start = hd_end + 1
              hnd_end   = dims%x_l_end

            CASE ( 5 )

              hd_start  = dims%x_l_end + 1
              hd_end    = dims%h_diag_end_upper
              hnd_start = hd_end + 1
              hnd_end   = dims%x_u_end

            CASE ( 6 )

              hd_start  = dims%x_u_end + 1
              hd_end    = dims%h_diag_end_nonpos
              hnd_start = hd_end + 1
              hnd_end   = n

            END SELECT

!  rows with a diagonal entry

            hd_end = MIN( hd_end, n )
            DO i = hd_start, hd_end
              xi = X( i )
              ri = R( i )
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
                j = H_col( l )
                R( j ) = R( j ) - H_val( l ) * xi
                ri = ri - H_val( l ) * X( j )
              END DO
              R( i ) = ri - H_val( H_ptr( i + 1 ) - 1 ) * xi
            END DO
            IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

            hnd_end = MIN( hnd_end, n )
            DO i = hnd_start, hnd_end
              xi = X( i )
              ri = R( i )
              DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
                j = H_col( l )
                R( j ) = R( j ) - H_val( l ) * xi
                ri = ri - H_val( l ) * X( j )
              END DO
              R( i ) = ri
            END DO
            IF ( hnd_end == n ) EXIT

          END DO
        END IF
      END IF
      RETURN

!  End of subroutine QPD_HX

      END SUBROUTINE QPD_HX

!-*-*-*-*-*-*-*-*-*-*-   Q P D _ A x  S U B R O U T I N E  -*-*-*-*-*-*-*-*-

      SUBROUTINE QPD_AX( dim_r, R, m, A_ne, A_val, A_col, A_ptr, dim_x, X, op )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Perform the operation r := r +/- A * x    .
!      .                     or r := r +/- A^T * x  .
!      .                                            .
!      ..............................................

!  Arguments:
!  =========

!   R      the result r of adding/subtracting A * x or A^T *x to/from r
!   X      the vector x
!   op     2 string character: possible values are
!          '+ '   r <- r + A * x
!          '+T'   r <- r + A^T * x
!          '- '   r <- r - A * x
!          '-T'   r <- r - A^T * x

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: dim_x, dim_r, m, A_ne
      CHARACTER( LEN = 2 ), INTENT( IN ) :: op
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) ::  A_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( dim_x ) :: X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( dim_r ) :: R

!  Local variables

      INTEGER :: i, l
      REAL ( KIND = wp ) :: xi, ri

      IF ( op( 1 : 1 ) == '+' ) THEN

!  r <- r + A^T * x

        IF ( op( 2 : 2 ) == 'T' .OR. op( 2 : 2 ) == 't' ) THEN
          DO i = 1, m
            xi = X( i )
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              R( A_col( l ) ) = R( A_col( l ) ) + A_val( l ) * xi
            END DO
          END DO

!  r <- r + A * x

        ELSE
          DO i = 1, m
            ri = R( i )
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              ri = ri + A_val( l ) * X( A_col( l ) )
            END DO
            R( i ) = ri
          END DO
        END IF

      ELSE

!  r <- r - A^T * x

        IF ( op( 2 : 2 ) == 'T' .OR. op( 2 : 2 ) == 't' ) THEN
          DO i = 1, m
            xi = X( i )
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              R( A_col( l ) ) = R( A_col( l ) ) - A_val( l ) * xi
            END DO
          END DO

!  r <- r - A * x

        ELSE
          DO i = 1, m
            ri = R( i )
            DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
              ri = ri - A_val( l ) * X( A_col( l ) )
            END DO
            R( i ) = ri
          END DO
        END IF

      END IF
      RETURN

!  End of subroutine QPD_Ax

      END SUBROUTINE QPD_Ax

!-*-*-*-*-*-*-*-*-   Q P D _ A B S  _ H X  S U B R O U T I N E  -*-*-*-*-*-*-

      SUBROUTINE QPD_abs_HX( dims, n, R, H_ne, H_val, H_col, H_ptr, X,         &
                             semibw, H_band_ptr )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..................................................
!      .                                                .
!      .  Perform the operation r := r + | H | * | x |  .
!      ..................................................

!  Arguments:
!  =========
!
!   dims    see module GALAHAD_QPP
!   H_*     sparse storage by rows or band
!   X       the vector x
!   R       the result of adding ABS( H ) * ABS( x ) to r
!   semibw  if present, only those entries within a band of semi-bandwidth
!           semibw will be accessed

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPD_dims_type ), INTENT( IN ) :: dims
      INTEGER, INTENT( IN ) :: n, H_ne
      INTEGER, OPTIONAL, INTENT( IN ) :: semibw
      INTEGER, INTENT( IN ), DIMENSION( n + 1 ) :: H_ptr
      INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( n ) :: H_band_ptr
      INTEGER, INTENT( IN ), DIMENSION( H_ne ) ::  H_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( H_ne ) :: H_val
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: R

!  Local variables

      INTEGER :: hd_start, hd_end, hnd_start, hnd_end, i, j, l, type
      REAL ( KIND = wp ) :: xi, ri

!  For a banded portion of H

      IF ( PRESENT( semibw ) ) THEN

!  r <- r + | H * x |

        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            xi = X( i )
            ri = R( i )
            DO l = H_band_ptr( i ), H_ptr( i + 1 ) - 2
              j = H_col( l )
              R( j ) = R( j ) - H_val( l ) * xi
              ri = ri + ABS( H_val( l ) * X( j ) )
            END DO
            R( i ) = ri + ABS( H_val( H_ptr( i + 1 ) - 1 ) * xi )
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            xi = X( i )
            ri = R( i )
            DO l = H_band_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l )
              R( j ) = R( j ) + ABS( H_val( l ) * xi )
              ri = ri + ABS( H_val( l ) * X( j ) )
            END DO
            R( i ) = ri
          END DO
          IF ( hnd_end == n ) EXIT

        END DO

!  For the whole of H

      ELSE

!  r <- r + | H * x |

        DO type = 1, 6

          SELECT CASE( type )
          CASE ( 1 )

            hd_start  = 1
            hd_end    = dims%h_diag_end_free
            hnd_start = hd_end + 1
            hnd_end   = dims%x_free

          CASE ( 2 )

            hd_start  = dims%x_free + 1
            hd_end    = dims%h_diag_end_nonneg
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_start - 1

          CASE ( 3 )

            hd_start  = dims%x_l_start
            hd_end    = dims%h_diag_end_lower
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_start - 1

          CASE ( 4 )

            hd_start  = dims%x_u_start
            hd_end    = dims%h_diag_end_range
            hnd_start = hd_end + 1
            hnd_end   = dims%x_l_end

          CASE ( 5 )

            hd_start  = dims%x_l_end + 1
            hd_end    = dims%h_diag_end_upper
            hnd_start = hd_end + 1
            hnd_end   = dims%x_u_end

          CASE ( 6 )

            hd_start  = dims%x_u_end + 1
            hd_end    = dims%h_diag_end_nonpos
            hnd_start = hd_end + 1
            hnd_end   = n

          END SELECT

!  rows with a diagonal entry

          hd_end = MIN( hd_end, n )
          DO i = hd_start, hd_end
            xi = X( i )
            ri = R( i )
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 2
              j = H_col( l )
              R( j ) = R( j ) + ABS( H_val( l ) * xi )
              ri = ri + ABS( H_val( l ) * X( j ) )
            END DO
            R( i ) = ri + H_val( H_ptr( i + 1 ) - 1 ) * xi
          END DO
          IF ( hd_end == n ) EXIT

!  rows without a diagonal entry

          hnd_end = MIN( hnd_end, n )
          DO i = hnd_start, hnd_end
            xi = X( i )
            ri = R( i )
            DO l = H_ptr( i ), H_ptr( i + 1 ) - 1
              j = H_col( l )
              R( j ) = R( j ) + ABS( H_val( l ) * xi )
              ri = ri + ABS( H_val( l ) * X( j ) )
            END DO
            R( i ) = ri
          END DO
          IF ( hnd_end == n ) EXIT

        END DO
      END IF
      RETURN

!  End of subroutine QPD_abs_HX

      END SUBROUTINE QPD_abs_HX

!-*-*-*-*-*-*-*-*-*-   Q P D _ A B S _ A X  S U B R O U T I N E  -*-*-*-*-*-*-*-

      SUBROUTINE QPD_abs_AX( dim_r, R, m, A_ne, A_val, A_col, A_ptr, dim_x,    &
                             X, op )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ...................................................
!      .                                                 .
!      .  Perform the operation r := r + | A | * | x |   .
!      .                     or r := r + | A^T | * | x | .
!      .                                                 .
!      ...................................................

!  Arguments:
!  =========

!   R      the result r of adding ABS(A) * ABS(x) or ABS(A^T) * ABS(x) to r
!   X      the vector x
!   op     1 string character: possible values are
!          ' '   r <- r + | A | * | x |
!          'T'   r <- r + | A^T | * | x |

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      INTEGER, INTENT( IN ) :: dim_x, dim_r, m, A_ne
      CHARACTER( LEN = 1 ), INTENT( IN ) :: op
      INTEGER, INTENT( IN ), DIMENSION( m + 1 ) :: A_ptr
      INTEGER, INTENT( IN ), DIMENSION( A_ne ) ::  A_col
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( dim_x ) :: X
      REAL ( KIND = wp ), INTENT( IN ), DIMENSION( A_ne ) :: A_val
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( dim_r ) :: R

!  Local variables

      INTEGER :: i, l
      REAL ( KIND = wp ) :: xi, ri


!  r <- r + A^T * x

      IF ( op( 1 : 1 ) == 'T' .OR. op( 1 : 1 ) == 't' ) THEN
        DO i = 1, m
          xi = X( i )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            R( A_col( l ) ) = R( A_col( l ) ) + ABS( A_val( l ) * xi )
          END DO
        END DO

!  r <- r + A * x

      ELSE
        DO i = 1, m
          ri = R( i )
          DO l = A_ptr( i ), A_ptr( i + 1 ) - 1
            ri = ri + ABS( A_val( l ) * X( A_col( l ) ) )
          END DO
          R( i ) = ri
        END DO
      END IF

      RETURN

!  End of subroutine QPD_abs_AX

      END SUBROUTINE QPD_abs_AX

!-*-*-*-*-*-*-*-*-*-*-   Q P D _ S I F  S U B R O U T I N E   -*-*-*-*-*-*-*-*-

      SUBROUTINE QPD_SIF( prob, file_name, sif, infinity, qp,                  &
                          no_linear, no_bounds, just_equality )

!  Write a SIF to the file file_name on unit sif for the input
!  linear (qp = .false.) or quadratic program (qp = .true.) from data
!  in prob (see QPP for details on the required data for prob).
!  problem bounds larger than infinity are regarded as infinite, while
!  general linear constraint information will not be provided when
!  no_linear is present

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      INTEGER, INTENT( IN ) :: sif
      CHARACTER ( LEN = 30 ) :: file_name
      REAL ( KIND = wp ), INTENT( IN ) :: infinity
      LOGICAL, INTENT( IN ) :: qp
      LOGICAL, OPTIONAL, INTENT( IN ) :: no_linear, no_bounds, just_equality

!  Local variables

      INTEGER :: i, j, l, iores
      REAL ( KIND = wp ) :: g
      LOGICAL :: filexx
      REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: DX, WORK_n

!  check if the file is old or new

      INQUIRE( FILE = file_name, EXIST = filexx )
      IF ( filexx ) THEN
         OPEN( sif, FILE = file_name, FORM = 'FORMATTED',                      &
               STATUS = 'OLD', IOSTAT = iores )
      ELSE
         OPEN( sif, FILE = file_name, FORM = 'FORMATTED',                      &
                STATUS = 'NEW', IOSTAT = iores )
      END IF

!  start the SIF file and assign the variables section (x)

      DO l = 1, 8
        IF ( file_name( l + 1 : l + 1 ) == '.' ) EXIT
      END DO
      WRITE( sif, "( 'NAME          ', A, //, 'VARIABLES', / )" )              &
        file_name( 1 : l )
      DO i = 1, prob%n
        WRITE( sif, "( '    X', I8 )" ) i
      END DO

!  assign the groups section (g and A)

      WRITE( sif, "( /, 'GROUPS', / )" )
      DO i = 1, prob%n
        IF ( prob%gradient_kind == 0 ) THEN
          g = zero
        ELSE IF ( prob%gradient_kind == 1 ) THEN
          g = one
        ELSE
          g = prob%G( i )
        END IF
        IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            g = g - one
          ELSE IF ( prob%target_kind /= 0 ) THEN
            g = g - prob%X0( i )
          END IF
        ELSE IF ( prob%Hessian_kind >= 2 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            g = g - prob%WEIGHT( i ) ** 2
          ELSE IF ( prob%target_kind /= 0 ) THEN
            g = g - prob%X0( i ) * prob%WEIGHT( i ) ** 2
          END IF
        END IF
        IF ( g /= zero )                                                       &
          WRITE( sif, "( ' N  OBJ      ', ' X', I8, ' ', A12 )" )              &
            i, STRING_real_12( g )
      END DO
      IF ( .NOT. PRESENT( no_linear ) ) THEN

!  general constraints

        IF ( .NOT. PRESENT( just_equality ) ) THEN
          IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
            l = 0
            DO i = 1, prob%m
              IF ( prob%C_l( i ) >= - infinity ) THEN
                IF ( prob%C_u( i ) <= infinity .AND.                           &
                     prob%C_l( i ) == prob%C_u( i ) ) THEN
                  DO j = 1, prob%n
                    l = l + 1
                    WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', A12 )" )        &
                      i, j, STRING_real_12( prob%A%val( l ) )
                  END DO
                ELSE
                  DO j = 1, prob%n
                    l = l + 1
                    WRITE( sif, "( ' G  C', I8, ' X', I8, ' ', A12 )" )        &
                      i, j, STRING_real_12( prob%A%val( l ) )
                  END DO
                END IF
              ELSE
                DO j = 1, prob%n
                  l = l + 1
                  WRITE( sif, "( ' L  C', I8, ' X', I8, ' ', A12 )" )          &
                    i, j, STRING_real_12( prob%A%val( l ) )
                END DO
              END IF
            END DO
          ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
            DO i = 1, prob%m
              IF ( prob%C_l( i ) >= - infinity ) THEN
                IF ( prob%C_u( i ) <= infinity .AND.                           &
                     prob%C_l( i ) == prob%C_u( i ) ) THEN
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', A12 )" )        &
                      i, prob%A%col( l ), STRING_real_12( prob%A%val( l ) )
                  END DO
                ELSE
                  DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                    WRITE( sif, "( ' G  C', I8, ' X', I8, ' ', A12 )" )        &
                      i, prob%A%col( l ), STRING_real_12( prob%A%val( l ) )
                  END DO
                END IF
              ELSE
                DO l = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                  WRITE( sif, "( ' L  C', I8, ' X', I8, ' ', A12 )" )          &
                    i, prob%A%col( l ), STRING_real_12( prob%A%val( l ) )
                END DO
              END IF
            END DO
          ELSE
            DO l = 1, prob%A%ne
              i = prob%A%row( l ) ; j = prob%A%col( l )
              IF ( prob%C_l( i ) >= - infinity ) THEN
                IF ( prob%C_u( i ) <= infinity .AND.                           &
                     prob%C_l( i ) == prob%C_u( i ) ) THEN
                  WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', A12 )" )          &
                    i, j, STRING_real_12( prob%A%val( l ) )
                ELSE
                  WRITE( sif, "( ' G  C', I8, ' X', I8, ' ', A12 )" )          &
                    i, j, STRING_real_12( prob%A%val( l ) )
                END IF
              ELSE
                WRITE( sif, "( ' L  C', I8, ' X', I8, ' ', A12 )" )            &
                  i, j, STRING_real_12( prob%A%val( l ) )
              END IF
            END DO
          END IF

!  assign the constants section (c_l or c_u)

          WRITE( sif, "( /, 'CONSTANTS', / )" )
          DO i = 1, prob%m
            IF ( prob%C_l( i ) >= - infinity ) THEN
              IF ( prob%C_l( i ) /= zero )                                     &
                WRITE( sif, "( '    RHS      ', ' C', I8, ' ', A12 )" )        &
                i, STRING_real_12( prob%C_l( i ) )
            ELSE IF ( prob%C_u( i ) <= infinity ) THEN
              IF ( prob%C_u( i ) /= zero )                                     &
                WRITE( sif, "( '    RHS      ', ' C', I8, ' ', A12 )" )        &
                i, STRING_real_12( prob%C_u( i ) )
            END IF
          END DO

!  assign the ranges section (c_u - c_l)

          l = 0
          DO i = 1, prob%m
            IF ( prob%C_l( i ) >= - infinity .AND. prob%C_u( i ) <= infinity   &
                 .AND. prob%C_l( i ) /= prob%C_u( i ) ) THEN
              IF ( l == 0 ) THEN
                WRITE( sif, "( /, 'RANGES', / )" )
                l = 1
              END IF
              WRITE( sif, "( '    RANGE    ', ' C', I8, ' ', A12 )" )          &
                i, STRING_real_12( prob%C_u( i ) - prob%C_l( i ) )
            END IF
          END DO

!  equality constraints

        ELSE
          IF ( SMT_get( prob%A%type ) == 'DENSE' ) THEN
            DO i = 1, prob%m
              DO j = 1, prob%n
                WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', A12 )" )            &
                  i, j, STRING_real_12( prob%A%val( j ) )
              END DO
            END DO
          ELSE IF ( SMT_get( prob%A%type ) == 'SPARSE_BY_ROWS' ) THEN
            DO i = 1, prob%m
              DO j = prob%A%ptr( i ), prob%A%ptr( i + 1 ) - 1
                WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', A12 )" )            &
                  i, prob%A%col( j ), STRING_real_12( prob%A%val( j ) )
              END DO
            END DO
          ELSE
            DO l = 1, prob%A%ne
              i = prob%A%row( l ) ; j = prob%A%col( l )
              WRITE( sif, "( ' E  C', I8, ' X', I8, ' ', A12 )" )              &
                i, j, STRING_real_12( prob%A%val( l ) )
            END DO
          END IF

!  assign the constants section (c_l or c_u)

          WRITE( sif, "( /, 'CONSTANTS', / )" )
          DO i = 1, prob%m
            IF ( prob%C( i ) /= zero )                                         &
              WRITE( sif, "( '    RHS      ', ' C', I8, ' ', A12 )" )          &
              i, STRING_real_12( prob%C( i ) )
          END DO
        END IF
      END IF

      IF ( .NOT. PRESENT( no_bounds ) ) THEN

!  assign the bounds section (x_l and x_u)

        IF ( COUNT( prob%X_l( : prob%n ) /= zero ) /= 0 .OR.                   &
             COUNT( prob%X_u( : prob%n ) <= infinity ) /= 0 ) THEN
            WRITE( sif, "( /, 'BOUNDS', /, /, ' FR BND       ''DEFAULT''' )" )
          DO i = 1, prob%n
            IF ( prob%X_l( i ) >= - infinity )                                 &
              WRITE( sif, "( ' LO BND       X', I8, ' ', A12 )" )              &
                i, STRING_real_12( prob%X_l( i ) )
            IF ( prob%X_u( i ) <= infinity )                                   &
              WRITE( sif, "( ' UP BND       X', I8, ' ', A12 )" )              &
                i, STRING_real_12( prob%X_u( i ) )
          END DO
        END IF
      ELSE
        WRITE( sif, "( /, 'BOUNDS', /, /, ' FR BND       ''DEFAULT''' )" )
      END IF

!  assign the start_point section (x_l and x_u)

      IF ( COUNT( prob%X( : prob%n ) /= zero ) /= 0 ) THEN
        WRITE( sif, "( /, 'START POINT', / )" )
        DO i = 1, prob%n
          IF ( prob%X( i ) /= zero )                                           &
            WRITE( sif, "( ' V  START    ', ' X', I8, ' ', A12 )" )            &
              i, STRING_real_12( prob%X( i ) )
        END DO
      END IF

!  assign the quadratic section (H)

      IF ( qp ) THEN
        IF ( prob%Hessian_kind < 0 ) THEN
          IF ( SMT_get( prob%H%type ) == 'IDENTITY' ) THEN
            WRITE( sif, "( /, 'QUADRATIC', / )" )
            DO i = 1, prob%n
              WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )              &
                i, i, STRING_real_12( one )
            END DO
          ELSE IF ( SMT_get( prob%H%type ) == 'SCALED_IDENTITY' ) THEN
            WRITE( sif, "( /, 'QUADRATIC', / )" )
            DO i = 1, prob%n
              WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )              &
                i, i, STRING_real_12(  prob%H%val( 1 ) )
            END DO
          ELSE IF ( SMT_get( prob%H%type ) == 'DIAGONAL' ) THEN
            WRITE( sif, "( /, 'QUADRATIC', / )" )
            DO i = 1, prob%n
              WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )              &
                i, i, STRING_real_12( prob%H%val( i ) )
            END DO
          ELSE IF ( SMT_get( prob%H%type ) == 'DENSE' ) THEN
            WRITE( sif, "( /, 'QUADRATIC', / )" )
            l = 0
            DO i = 1, prob%n
              DO j = 1, i
                l = l + 1
                WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )            &
                  i, j, STRING_real_12( prob%H%val( l ) )
              END DO
            END DO
          ELSE IF ( SMT_get( prob%H%type ) == 'SPARSE_BY_ROWS' ) THEN
            IF (  prob%H%ptr( prob%n + 1 ) > 1 ) THEN
              WRITE( sif, "( /, 'QUADRATIC', / )" )
              DO i = 1, prob%n
                DO j = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
                  WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )          &
                    i, prob%H%col( j ), STRING_real_12( prob%H%val( j ) )
                END DO
              END DO
            END IF
          ELSE IF ( SMT_get( prob%H%type ) == 'COORDINATE' ) THEN
            IF (  prob%H%ne > 0 ) THEN
              WRITE( sif, "( /, 'QUADRATIC', / )" )
              DO l = 1, prob%H%ne
               WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )             &
                 prob%H%row( l ), prob%H%col( l ),                             &
                 STRING_real_12( prob%H%val( l ) )
              END DO
            END IF
          ELSE
!           WRITE( sif, "( /, '* L-BFGS QUADRATIC (to do!)', / )" )
            ALLOCATE( DX( prob%H_lm%n_restriction ),                           &
                      WORK_n( prob%H_lm%n_restriction ) )
            WRITE( sif, "( /, 'QUADRATIC', / )" )
            DX = zero
            DO i = 1, prob%H_lm%n_restriction
              DX( i ) = one
              CALL LMS_apply_lbfgs( DX, prob%H_lm, j, RESULT = WORK_n )
              DO j = 1, i
                WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )            &
                  i, j, STRING_real_12( WORK_n( j ) )
              END DO
              DX( i ) = zero
            END DO
            DEALLOCATE( DX, WORK_n )
          END IF
        ELSE IF ( prob%Hessian_kind == 1 ) THEN
          WRITE( sif, "( /, 'QUADRATIC', / )" )
          DO i = 1, prob%n
           WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )                 &
             i, i, STRING_real_12( one )
          END DO
        ELSE IF ( prob%Hessian_kind >= 2 ) THEN
          WRITE( sif, "( /, 'QUADRATIC', / )" )
          DO i = 1, prob%n
           WRITE( sif, "( '    X', I8, ' X', I8, ' ', A12 )" )                 &
             i, i, STRING_real_12( prob%WEIGHT( i ) ** 2 )
          END DO
        END IF
      END IF

!  end the SIF file

      WRITE( sif, "( /, 'ENDATA' )" )
      CLOSE( sif )

      RETURN

!  End of subroutine QPD_SIF

      END SUBROUTINE QPD_SIF

!-*-   Q P D _ s o l v e _ s e p a r a b l e _ B Q P   S U B R O U T I N E  -*-

      SUBROUTINE QPD_solve_separable_BQP( prob, infinity, obj_unbounded, obj,  &
                                          feasible, status, B_stat )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
!
!  Compute a global minimizer of a separable bound-constrained quadratic program
!
! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  Dummy arguments

      TYPE ( QPT_problem_type ), INTENT( INOUT ) :: prob
      REAL ( KIND = wp ), INTENT( IN ) :: infinity, obj_unbounded
      INTEGER, INTENT( INOUT ) :: status
      REAL ( KIND = wp ), INTENT( INOUT ) :: obj
      LOGICAL, INTENT( INOUT ) :: feasible
      INTEGER, INTENT( OUT ), OPTIONAL, DIMENSION( prob%n ) :: B_stat

!  Local variables

      INTEGER :: i, j, l
      REAL ( KIND = wp ) :: g, h, x_l, x_u, x_unc
      LOGICAL :: stat_required

!  Parameters

      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp

      stat_required = PRESENT( B_stat )

!  Set information parameters

      obj = prob%f
      feasible = .TRUE.
      status = GALAHAD_ok

!  Temporarily store the diagonal Hessian in X

      IF ( prob%Hessian_kind < 0 ) THEN
        SELECT CASE ( SMT_get( prob%H%type ) )
        CASE ( 'IDENTITY' )
          prob%X( : prob%n ) = one
        CASE ( 'SCALED_IDENTITY' )
          prob%X( : prob%n ) = prob%H%val( 1 )
        CASE ( 'DIAGONAL' )
          prob%X( : prob%n ) = prob%H%val( : prob%n )
        CASE ( 'DENSE' )
          l = 0
          DO i = 1, prob%n
            DO j = 1, i
              l = l + 1
              IF ( i == j ) THEN
                prob%X( i ) = prob%H%val( l )
              ELSE
                IF ( prob%H%val( l ) /= zero ) THEN
                  status = GALAHAD_error_upper_entry ; RETURN
                END IF
              END IF
            END DO
          END DO
        CASE ( 'SPARSE_BY_ROWS' )
          DO i = 1, prob%n
            prob%X( i ) = zero
            DO l = prob%H%ptr( i ), prob%H%ptr( i + 1 ) - 1
              j = prob%H%col( l )
              IF ( i == j ) THEN
                prob%X( i ) = prob%X( i ) + prob%H%val( l )
              ELSE
                IF ( prob%H%val( l ) /= zero ) THEN
                  status = GALAHAD_error_upper_entry ; RETURN
                END IF
              END IF
            END DO
          END DO
        CASE ( 'COORDINATE' )
          prob%X = zero
          DO l = 1, prob%H%ne
            i = prob%H%row( l ) ; j = prob%H%col( l )
            IF ( i == j ) THEN
              prob%X( i ) = prob%X( i ) + prob%H%val( l )
            ELSE
              IF ( prob%H%val( l ) /= zero ) THEN
                status = GALAHAD_error_upper_entry ; RETURN
              END IF
            END IF
          END DO
        END SELECT
      ELSE IF ( prob%Hessian_kind == 0 ) THEN
        prob%X( : prob%n ) = zero
      ELSE IF ( prob%Hessian_kind == 1 ) THEN
        prob%X( : prob%n ) = one
      ELSE IF ( prob%Hessian_kind >= 2 ) THEN
        prob%X( : prob%n ) = prob%WEIGHT( : prob%n ) ** 2
      END IF

!  Now consider the solution, one component at a time

      DO i = 1, prob%n
        x_l = prob%X_l( i ) ; x_u = prob%X_u( i )
        IF ( x_l > x_u ) THEN
          feasible = .FALSE.
          status = GALAHAD_error_primal_infeasible ; RETURN
        END IF
        IF ( prob%gradient_kind == 0 ) THEN
          g = zero
        ELSE IF ( prob%gradient_kind == 1 ) THEN
          g = one
        ELSE
          g = prob%G( i )
        END IF
        IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            g = g - one
          ELSE IF ( prob%target_kind /= 0 ) THEN
            g = g - prob%X0( i )
          END IF
        ELSE IF ( prob%Hessian_kind >= 2 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            g = g - prob%WEIGHT( i ) ** 2
          ELSE IF ( prob%target_kind /= 0 ) THEN
            g = g - prob%X0( i ) * prob%WEIGHT( i ) ** 2
          END IF
        END IF
        h = prob%X( i )

!  The objective is strictly convex along this component direction

        IF ( h > zero ) THEN
          x_unc = - g / h

!  The minimizer occurs at the lower bound

          IF ( x_unc <= x_l ) THEN
            prob%X( i ) = x_l
            prob%Z( i ) = g + h * x_l
            IF ( stat_required ) B_stat( i ) = - 1

!  The minimizer occurs at the upper bound

          ELSE IF ( x_unc >= x_u ) THEN
            prob%X( i ) = x_u
            prob%Z( i ) = g + h * x_u
            IF ( stat_required ) B_stat( i ) = 1

!  The minimizer is unconstrained

          ELSE
            prob%X( i ) = x_unc
            prob%Z( i ) = zero
            IF ( stat_required ) B_stat( i ) = 0
          END IF

!  The objective is non-convex along this component direction

        ELSE IF ( h < zero ) THEN

!  The objective is unbounded

          IF ( x_l < - infinity ) THEN
            status = GALAHAD_error_unbounded
            prob%X( i ) = x_l
            prob%Z( i ) = zero
            IF ( stat_required ) B_stat( i ) = 0
          ELSE IF ( x_u > infinity ) THEN
            status = GALAHAD_error_unbounded
            prob%X( i ) = x_u
            prob%Z( i ) = zero
            IF ( stat_required ) B_stat( i ) = 0
          ELSE
            IF ( g * x_l + half * h * x_l ** 2 <                               &
                 g * x_u + half * h * x_u ** 2 ) THEN

!  The minimizer occurs at the lower bound

              prob%X( i ) = x_l
              prob%Z( i ) = g + h * x_l
              IF ( stat_required ) B_stat( i ) = - 1

!  The minimizer occurs at the upper bound

            ELSE
              prob%X( i ) = x_u
              prob%Z( i ) = g + h * x_u
              IF ( stat_required ) B_stat( i ) = 1
            END IF
          END IF

!  The objective has no curvature along this component direction

        ELSE
          IF ( g > zero ) THEN
            prob%X( i ) = x_l

!  The objective is unbounded

            IF ( x_l < - infinity ) THEN
              status = GALAHAD_error_unbounded
              prob%Z( i ) = zero
              IF ( stat_required ) B_stat( i ) = 0

!  The minimizer occurs at the lower bound

            ELSE
              prob%Z( i ) = g + h * x_l
              IF ( stat_required ) B_stat( i ) = - 1
            END IF
          ELSE IF ( g < zero ) THEN
            prob%X( i ) = x_u

!  The objective is unbounded

            IF ( x_u > infinity ) THEN
              status = GALAHAD_error_unbounded
              prob%Z( i ) = zero
              IF ( stat_required ) B_stat( i ) = 0

!  The minimizer occurs at the upper bound

            ELSE
              prob%Z( i ) = g + h * x_u
              IF ( stat_required ) B_stat( i ) = 1
            END IF
          ELSE

!  The objective is constant along this component direction

            prob%Z( i ) = zero

!  Pick an arbitrary minimizer between the bounds

            IF ( stat_required ) B_stat( i ) = 0
            IF ( x_l >= - infinity .AND. x_u <= infinity ) THEN
              prob%X( i ) = half * ( x_l + x_u )
            ELSE IF ( x_l >= - infinity ) THEN
              prob%X( i ) = x_l
            ELSE IF ( x_u <= infinity ) THEN
              prob%X( i ) = x_u
            ELSE
              prob%X( i ) = zero
            END IF
          END IF
        END IF
        obj = obj + prob%X( i ) * ( g + half * h * prob%X( i ) )
        IF ( prob%Hessian_kind == 1 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            obj = obj + half
          ELSE IF ( prob%target_kind /= 0 ) THEN
            obj = obj + half * prob%X0( i ) ** 2
          END IF
        ELSE IF ( prob%Hessian_kind >= 2 ) THEN
          IF ( prob%target_kind == 1 ) THEN
            obj = obj + half * prob%WEIGHT( i ) ** 2
          ELSE IF ( prob%target_kind /= 0 ) THEN
            obj = obj + half * ( prob%X0( i ) * prob%WEIGHT( i ) ) ** 2
          END IF
        END IF
      END DO
      IF ( obj < obj_unbounded ) status = GALAHAD_error_unbounded

      RETURN

!  End of subroutine QPD_solve_separable_BQP

      END SUBROUTINE QPD_solve_separable_BQP

!  End of module GALAHAD_QPD_double

   END MODULE GALAHAD_QPD_double
