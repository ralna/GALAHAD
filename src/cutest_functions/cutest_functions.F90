! THIS VERSION: GALAHAD 4.3 - 2024-02-01 AT 16:20 GMT.

#include "galahad_modules.h"
#include "cutest_routines.h"

!-*-*-*-*  G A L A H A D _ C U T E S T _ F U N C T I O N S  M O D U L E  *-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Daniel Robinson and Nick Gould
!
!  History -
!   originally released pre GALAHAD Version 2.2. February 22nd 2008
!
   MODULE GALAHAD_CUTEST_precision

     USE GALAHAD_KINDS_precision
     USE GALAHAD_STRING
     USE GALAHAD_SYMBOLS
     USE GALAHAD_SMT_precision
     USE GALAHAD_SPACE_precision
     USE GALAHAD_USERDATA_precision
     USE GALAHAD_NLPT_precision, ONLY: NLPT_problem_type, NLPT_cleanup
     USE CUTEST_INTERFACE_precision

     IMPLICIT NONE

!---------------------
!   P r e c i s i o n
!---------------------

     PRIVATE
     PUBLIC :: CUTEst_initialize, CUTEst_eval_F, CUTEst_eval_FC,               &
               CUTEst_eval_C, CUTEst_eval_G, CUTEst_eval_J,                    &
               CUTEst_eval_GJ, CUTEst_eval_SGJ,                                &
               CUTEst_eval_H, CUTEst_eval_HPROD, CUTEst_eval_SHPROD,           &
               CUTEst_eval_JPROD, CUTEst_eval_SJPROD, CUTEst_eval_HL,          &
               CUTEst_eval_HJ, CUTEst_eval_HLC, CUTEst_eval_HLPROD,            &
               CUTEst_eval_SHLPROD, CUTEst_eval_HLCPROD,                       &
               CUTEst_eval_SHLCPROD, CUTEst_eval_HCPRODS,                      &
               CUTEst_eval_HOCPRODS, CUTEst_start_timing, CUTEst_timing,       &
               CUTEst_terminate, NLPT_problem_type, GALAHAD_userdata_type

!------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n
!------------------------------------------------

     TYPE, PUBLIC :: CUTEST_control_type
       INTEGER ( KIND = ip_ ) :: input = 5
       INTEGER ( KIND = ip_ ) :: error = 6
       INTEGER ( KIND = ip_ ) :: io_buffer = 11
       LOGICAL :: separate_linear_constraints = .FALSE.
     END TYPE

     TYPE, PUBLIC :: CUTEST_inform_type
       INTEGER ( KIND = ip_ ) :: status = 0
       INTEGER ( KIND = ip_ ) :: alloc_status = 0
       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )
     END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = rp_ ), PARAMETER ::  zero  = 0.0_rp_
     REAL ( KIND = rp_ ), PARAMETER ::  one   = 1.0_rp_
     REAL ( KIND = rp_ ), PARAMETER ::  two   = 2.0_rp_
     REAL ( KIND = rp_ ), PARAMETER ::  ten   = 10.0_rp_
     REAL ( KIND = rp_ ), PARAMETER ::  small = ten ** ( -8 )
     REAL ( KIND = rp_ ), PARAMETER ::  huge  = ten ** ( 19 )

     INTEGER ( KIND = ip_ ), PARAMETER :: loc_m = 1
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_n = 2
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_m_a = 3
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_nnzg = 4
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_indg = 5
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_nnzh = 6
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_irnh = 7
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_icnh = 8
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_h = 9
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_nnzj = 10
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_indfun = 11
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_indvar = 12
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_cjac = 13
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_lohp = 14
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_ohpind = 15
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_lchp = 16
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_chpind = 17
     INTEGER ( KIND = ip_ ), PARAMETER :: loc_chpptr = 18

   CONTAINS

!-*-*-  C U T E R _ i n i t i a l i z e   S U B R O U T I N E  -*-*-*-*

     SUBROUTINE CUTEst_initialize( nlp, control, inform, userdata,             &
                                   no_hessian, no_jacobian, hessian_products,  &
                                   sparse_gradient )

     TYPE ( NLPT_problem_type ), INTENT( OUT ) :: nlp
     TYPE ( GALAHAD_userdata_type ), INTENT( OUT ) :: userdata
     TYPE ( CUTEST_control_type ), INTENT( IN ) :: control
     TYPE ( CUTEST_inform_type ), INTENT( OUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_hessian, no_jacobian
     LOGICAL, OPTIONAL, INTENT( IN ) :: hessian_products
     LOGICAL, OPTIONAL, INTENT( IN ) :: sparse_gradient

! local variables.

     INTEGER ( KIND = ip_ ) :: i, j, l, lcjac, lg, lh, h, status, iend, rend
     INTEGER ( KIND = ip_ ) :: m, n, nnzg, nnzj, nnzh, nnzchp, nnzohp
     INTEGER ( KIND = ip_ ) :: indfun, indvar, irnh, icnh, cjac, cutest_status
     INTEGER ( KIND = ip_ ) :: l_order, indg, iohpind, ichpind, ichpptr
     REAL ( KIND = rp_ ) :: f, f2, alpha, alpha_min
     LOGICAL :: no_hess, no_jac, hess_prods, sparse_grad
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u, C, X
     REAL ( KIND = rp_ ), ALLOCATABLE, DIMENSION( : ) :: C2, lin_const
     CHARACTER ( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: full_CNAMES
     CHARACTER ( LEN = 80 ) :: array_name

! get dimensions

     CALL CUTEST_cdimen_r( cutest_status, control%input, n, m )
     IF ( cutest_status /= 0 ) GO TO 930
     nlp%n = n
     nlp%m = m
     nlp%m_a = 0   ! dummy initialization.

!  Allocate sufficient space to hold the problem

     CALL SPACE_resize_array( nlp%n, nlp%X, inform%status,                     &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
        inform%bad_alloc = 'nlp%X' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( nlp%n, nlp%X_l, inform%status,                   &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
         inform%bad_alloc = 'nlp%X_l' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( nlp%n, nlp%X_u, inform%status,                   &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
         inform%bad_alloc = 'nlp%X_u' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( nlp%n, nlp%Z, inform%status,                     &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
        inform%bad_alloc = 'nlp%Z' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( nlp%n, nlp%G, inform%status,                     &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
        inform%bad_alloc = 'nlp%G' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( nlp%n, nlp%VNAMES, inform%status,                &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
         inform%bad_alloc = 'nlp%VNAMES' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( m, nlp%EQUATION, inform%status,                  &
                              inform%alloc_status)
     IF ( inform%status /= 0 ) THEN
         inform%bad_alloc = 'nlp%EQUATION' ; GO TO 910 ; END IF

     CALL SPACE_resize_array( m, nlp%LINEAR, inform%status,                    &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
         inform%bad_alloc = 'nlp%LINEAR' ; GO TO 910 ; END IF

!  check to see if the Hessian is required

     IF ( PRESENT( no_hessian ) ) THEN
       no_hess = no_hessian
     ELSE
       no_hess = .FALSE.
     END IF

!  check to see if products with Hessians of the constraints are required

     IF ( PRESENT( hessian_products ) .AND. m > 0 ) THEN
       hess_prods = hessian_products
     ELSE
       hess_prods = .FALSE.
     END IF

!  check to see if a sparse gradient is required

     IF ( PRESENT( sparse_gradient ) ) THEN
       sparse_grad = sparse_gradient
     ELSE
       sparse_grad = .FALSE.
     END IF

!  --------------------------------
!  -- The problem is constrained --
!  --------------------------------

     IF ( m > 0 ) THEN
       CALL SPACE_resize_array( m, Y, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'Y' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( m, C_l, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C_l' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( m, C_u, inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C_u' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( m, full_CNAMES, inform%status,                 &
                               inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'full_CNAMES' ; GO TO 910 ; END IF

!  get the problem with linear constraints possibly first

       i = n ; j = m
       IF ( control%separate_linear_constraints ) THEN
         l_order = 1
       ELSE
         l_order = 0
       END IF
       CALL CUTEST_csetup_r( cutest_status, control%input, control%error,      &
                             control%io_buffer, n, m,                          &
                             nlp%X, nlp%X_l, nlp%X_u, Y, C_l, C_u,             &
                             nlp%EQUATION, nlp%LINEAR, 0_ip_, l_order, 0_ip_ )
       IF ( cutest_status /= 0 ) GO TO 930

       nlp%m_a = 0
       IF ( control%separate_linear_constraints ) THEN
         DO l = 1, m
           IF ( nlp%LINEAR( l ) ) THEN
             nlp%m_a = nlp%m_a + 1
           ELSE
             EXIT
           END IF
         END DO
       END IF
       nlp%m = m - nlp%m_a

       CALL SPACE_resize_array( nlp%m, nlp%Y, inform%status,                   &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%Y' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%C, inform%status,                   &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%C_l, inform%status,                 &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C_l' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%C_u, inform%status,                 &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C_u' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m_a, nlp%Y_a, inform%status,               &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%Y_a' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m_a, nlp%A_l, inform%status,               &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%A_l' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m_a, nlp%Ax, inform%status,                &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%Ax' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m_a, nlp%A_u, inform%status,               &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%A_u' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%CNAMES, inform%status,              &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%CNAMES' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m_a, nlp%ANAMES, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%ANAMES' ; GO TO 910 ; END IF

!  obtain the names of the problem, its variables and general constraints

       CALL CUTEST_cnames_r( cutest_status, n, m, nlp%pname, nlp%VNAMES,       &
                             full_CNAMES )
       IF ( cutest_status /= 0 ) GO TO 930

!  define the "corrected" separated vectors

       nlp%Y_a = Y  ( 1 : nlp%m_a ) ;  nlp%Y   = Y  ( nlp%m_a + 1 : m )
       nlp%A_l = C_l( 1 : nlp%m_a ) ;  nlp%C_l = C_l( nlp%m_a + 1 : m )
       nlp%A_u = C_u( 1 : nlp%m_a ) ;  nlp%C_u = C_u( nlp%m_a + 1 : m )

       nlp%ANAMES = full_CNAMES( 1 : nlp%m_a )
       nlp%CNAMES = full_CNAMES( nlp%m_a + 1 : m )

!  deallocate arrays no longer needed

       array_name = 'cutest_functions : C_l'
       CALL SPACE_dealloc_array( C_l,                                          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'C_l' ; GO TO 920 ; END IF

       array_name = 'cutest_functions : C_u'
       CALL SPACE_dealloc_array( C_u,                                          &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'C_u' ; GO TO 920 ; END IF

       array_name = 'cutest_functions : full_CNAMES'
       CALL SPACE_dealloc_array( full_CNAMES,                                  &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'full_CNAMES' ; GO TO 920 ; END IF

!  check to see if the Jacobian is required

       IF ( PRESENT( no_jacobian ) ) THEN
         no_jac = no_jacobian
       ELSE
         no_jac = .FALSE.
       END IF

!  set up sparsity structure for A, J, and H - assume co-ordinate storage.
!  Determine number of non-zeros in the matrix of gradients of the
!  objective function AND constraint functions.

       IF ( no_jac ) THEN
         nnzj = 0
       ELSE
         CALL CUTEST_cdimsj_r( cutest_status, nnzj )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  determine how many nonzeros are required to store the Hessian matrix of the
!  Lagrangian, when the matrix is stored as a sparse matrix in "co-ordinate"
!  format (only the lower triangular part is stored).

       IF ( no_hess ) THEN
         nnzh = 0
       ELSE
         CALL CUTEST_cdimsh_r( cutest_status, nnzh )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  determine how many nonzeros are required to store the matrix of products
!  of the constraint Hessians with a vector, when the matrix is stored in
!  sparse column-wise format.

       IF (  hess_prods ) THEN
         CALL CUTEST_cdimohp_r( cutest_status, nnzohp )
         IF ( cutest_status /= 0 ) GO TO 930
         CALL CUTEST_cdimchp_r( cutest_status, nnzchp )
         IF ( cutest_status /= 0 ) GO TO 930
       ELSE
         nnzchp = 0
       END IF

!  determine how many nonzeros are required to store the sparse gradient
!  of the objective function

       IF ( sparse_grad ) THEN
         CALL CUTEST_cdimsg_r( cutest_status, nnzg )
         IF ( cutest_status /= 0 ) GO TO 930
          nlp%Go%ne = nnzg
       ELSE
         nnzg = 0
         nlp%Go%ne = n
       END IF

!  set starting addresses for workspace array partitions

       indg = loc_chpptr
       irnh = indg + nnzg
       icnh = irnh + nnzh
       indfun = icnh + nnzh
       indvar = indfun + nnzj
       iohpind = indvar + nnzj
       IF (  hess_prods ) THEN
         ichpind = iohpind + nnzohp
         ichpptr = ichpind + nnzchp
         iend = ichpptr + m + 1
       ELSE
         ichpind = iohpind
         ichpptr = ichpind
         iend = ichpptr
       END IF

       h = 0
       cjac = h + nnzh
       rend = MAX( cjac + nnzj, m, n )

!  allocate space to hold scalars/arrays needed for subsequent calls

       CALL SPACE_resize_array( iend, userdata%integer, inform%status,         &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%integer' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( rend, userdata%real, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%real' ; GO TO 910 ; END IF

!  record workspace partitions in userdata%integer

       userdata%integer( loc_m ) = m
       userdata%integer( loc_n ) = n
       userdata%integer( loc_m_a ) = nlp%m_a
       userdata%integer( loc_nnzg ) = nnzg
       userdata%integer( loc_indg ) = indg
       userdata%integer( loc_nnzh ) = nnzh
       userdata%integer( loc_irnh ) = irnh
       userdata%integer( loc_icnh ) = icnh
       userdata%integer( loc_h ) = h
       userdata%integer( loc_nnzj ) = nnzj
       userdata%integer( loc_indfun ) = indfun
       userdata%integer( loc_indvar ) = indvar
       userdata%integer( loc_cjac ) = cjac
       userdata%integer( loc_lohp ) = nnzohp
       userdata%integer( loc_ohpind ) = iohpind
       userdata%integer( loc_lchp ) = nnzchp
       userdata%integer( loc_chpind ) = ichpind
       userdata%integer( loc_chpptr ) = ichpptr

!  determine if there is a constant in the linear constraints. Adjust the
!  bounds if necessary

       IF ( nlp%m_a > 0 ) THEN
         CALL SPACE_resize_array( m, C, inform%status, inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%real' ; GO TO 910 ; END IF

         CALL SPACE_resize_array( m, C2, inform%status, inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%real' ; GO TO 910 ; END IF

         CALL SPACE_resize_array( nlp%m_a, lin_const, inform%status,           &
                                  inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%real' ; GO TO 910 ; END IF

         CALL SPACE_resize_array( n, X, inform%status, inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
            inform%bad_alloc = 'userdata%real' ; GO TO 910 ; END IF

!  make X feasible with respect to bounds

         X = zero
         X = MIN( nlp%X_u, MAX( nlp%X_l, X ) )

         alpha_min = two
         DO i = 1, n
           IF ( X(i) > zero ) THEN
             alpha = min( one + one / X( i ), nlp%X_u( i ) / X( i ) )
           ELSE IF ( X(i) < zero ) THEN
             alpha = min( one - one / X( i ), nlp%X_l( i ) / X( i ) )
           ELSE
             alpha = two
           END IF
           alpha_min = min( alpha_min, alpha )
         END DO
         alpha_min = max( alpha_min, 1.001_rp_ )

         CALL CUTEST_cfn_r( cutest_status, n, m, X, f, C )
         IF ( cutest_status /= 0 ) GO TO 930
         CALL CUTEST_cfn_r( cutest_status, n, m, alpha_min * X, f2, C2 )
         IF ( cutest_status /= 0 ) GO TO 930

         lin_const = alpha_min * C( : nlp%m_a ) - C2( : nlp%m_a )
         lin_const = lin_const / (alpha_min - one )

         DO i = 1, nlp%m_a
           IF ( nlp%A_l( i ) > - huge ) THEN
              nlp%A_l( i ) = nlp%A_l( i ) - lin_const( i )
           END IF
           IF ( nlp%A_u( i ) < huge ) THEN
              nlp%A_u( i ) = nlp%A_u( i ) - lin_const( i )
           END IF
         END DO

         array_name = 'cutest_functions : C'
         CALL SPACE_dealloc_array( C,                                          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'C' ; GO TO 920
         END IF

         array_name = 'cutest_functions : X'
         CALL SPACE_dealloc_array( X,                                          &
              inform%status, inform%alloc_status, array_name = array_name,     &
              bad_alloc = inform%bad_alloc, out = control%error )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'X' ; GO TO 920
         END IF
       END IF

!  evaluate the Jacobian and Hessian and get sparsity pattern

       IF ( no_hess .AND. no_jac ) THEN
       ELSE IF ( no_jac ) THEN
         lh = nnzh
         CALL CUTEST_cshp_r( cutest_status, nlp%n, nnzh, lh,                   &
                             userdata%integer( irnh + 1 : irnh + nnzh ),       &
                             userdata%integer( icnh + 1 : icnh + nnzh ) )
       ELSE IF ( no_hess ) THEN
         lcjac = nnzj
         CALL CUTEST_csgrp_r( cutest_status, nlp%n, nnzj, lcjac,               &
                              userdata%integer( indvar + 1 : indvar + nnzj ),  &
                              userdata%integer( indfun + 1 : indfun + nnzj ) )
       ELSE
         lcjac = nnzj
         lh = nnzh
         CALL CUTEST_csgrshp_r( cutest_status, nlp%n, nnzj, lcjac,             &
                              userdata%integer( indvar + 1 : indvar + nnzj ),  &
                              userdata%integer( indfun + 1 : indfun + nnzj ),  &
                              nnzh, lh,                                        &
                              userdata%integer( irnh + 1 : irnh + nnzh ),      &
                              userdata%integer( icnh + 1 : icnh + nnzh ) )
       END IF
       IF ( cutest_status /= 0 ) GO TO 930

!  evaluate the matrix of constraint Hessian-vector products to get its
!  sparsity pattern

       IF (  hess_prods ) THEN
         CALL SPACE_resize_array( nnzchp, nlp%P%val, inform%status,            &
                                  inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%P%val' ; GO TO 910 ; END IF
         CALL CUTEST_cchprodsp_r( cutest_status, m, nnzchp,                    &
                                  userdata%integer( ichpind + 1 :              &
                                                    ichpind + nnzchp),         &
                                  userdata%integer( ichpptr + 1 :              &
                                                    ichpptr + m + 1 ) )
       END IF

!  get the sparsity pattern of the objective-function gradient

       IF ( sparse_grad ) THEN
         CALL SPACE_resize_array( nnzg, nlp%Go%ind, inform%status,             &
                                  inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%Go%col' ; GO TO 910 ; END IF
         lg = nnzg
         CALL CUTEST_cisgrp_r( cutest_status, n, 0_ip_, nnzg, lg,              &
                             userdata%integer( indg + 1 : indg + nnzg ) )
         nlp%Go%ind( : nnzg ) = userdata%integer( indg + 1 : indg + nnzg )
         CALL SPACE_resize_array( nnzg, nlp%Go%val, inform%status,       &
                                  inform%alloc_status )
         IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%Go%ind' ; GO TO 910 ; END IF
       END IF

!  get the number of nonzeros in the linear and nonlinear constraint Jacobians

       nlp%J%ne = 0
       nlp%A%ne = 0
       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) == 0 ) THEN
!          nlp%Go%ne = nlp%Go%ne + 1
         ELSEIF ( userdata%integer( indfun + l ) <= nlp%m_a ) THEN
           nlp%A%ne = nlp%A%ne + 1
         ELSE
           nlp%J%ne = nlp%J%ne + 1
         END IF
       END DO

!  deallocate arrays no longer needed

       array_name = 'cutest_functions : Y'
       CALL SPACE_dealloc_array( Y,                                            &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'Y' ; GO TO 920 ; END IF

!  allocate arrays that are now of correct length

       CALL SPACE_resize_array( nlp%A%ne, nlp%A%row, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%A%row' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%A%ne, nlp%A%col, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%A%col' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%A%ne, nlp%A%val, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%A%val' ; GO TO 910
       END IF
       CALL SPACE_resize_array( nlp%J%ne, nlp%J%row, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%J%row' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%J%ne, nlp%J%col, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%J%col' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%J%ne, nlp%J%val, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%J%val' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%Go%ne, nlp%Go%ind,             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%Go%col' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%Go%ne, nlp%Go%val,             &
                                inform%status, inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%Go%val' ; GO TO 910
       END IF

!  untangle J: separate the gradient terms from the objective, linear
!  constraints and the general constraints in the Jacobian

!      nlp%Go%n = n ; nlp%Go%ne = 0
       nlp%A%n = n ; nlp%A%m = nlp%m_a ; nlp%A%ne = 0
       nlp%J%n = n ; nlp%J%m = m - nlp%m_a ; nlp%J%ne = 0

       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) == 0 ) THEN
!           nlp%Go%ne = nlp%Go%ne + 1
!           nlp%Go%col( nlp%Go%ne ) = userdata%integer( indvar + l )
         ELSE IF ( userdata%integer( indfun + l ) <= nlp%m_a ) THEN
           nlp%A%ne = nlp%A%ne + 1
           nlp%A%row( nlp%A%ne ) = userdata%integer( indfun + l )
           nlp%A%col( nlp%A%ne ) = userdata%integer( indvar + l )
           nlp%A%val( nlp%A%ne ) = userdata%real( cjac + l )
         ELSE
           nlp%J%ne = nlp%J%ne + 1
           nlp%J%row( nlp%J%ne ) = userdata%integer( indfun + l ) - nlp%m_a
           nlp%J%col( nlp%J%ne ) = userdata%integer( indvar + l )
         END IF
       END DO

!  define the storage type for J

       CALL SMT_put( nlp%A%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

       CALL SMT_put( nlp%J%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

!  ----------------------------------
!  -- The problem is unconstrained --
!  ----------------------------------

     ELSE

!  set up the correct data structures for subsequent computations

       CALL CUTEST_usetup_r( cutest_status, control%input, control%error,      &
                             control%io_buffer, n, nlp%X, nlp%X_l, nlp%X_u )
       IF ( cutest_status /= 0 ) GO TO 930

!  obtain the names of the problem and its variables

       CALL CUTEST_unames_r( cutest_status, n, nlp%pname, nlp%VNAMES )
       IF ( cutest_status /= 0 ) GO TO 930

!  set up sparsity structure for H. ( Assume co-ordinate storage )

!  determine how many nonzeros are required to store the Hessian matrix
!  when the matrix is stored as a sparse matrix in "co-ordinate" format
!  (only the lower triangular part is stored).

       IF ( no_hess ) THEN
         nnzh = 0
       ELSE
         CALL CUTEST_udimsh_r( cutest_status, nnzh )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  set starting addresses for workspace array partitions

       irnh = loc_h
       icnh = irnh + nnzh
       iend = icnh + nnzh
       h = 0
       rend = MAX( h + nnzh, n )

!  allocate space to hold scalars/arrays needed for subsequent calls

       CALL SPACE_resize_array( iend, userdata%integer, inform%status,         &
                               inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'userdata%integer' ; GO TO 910
       END IF

       CALL SPACE_resize_array( rend, userdata%real, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'userdata%real' ; GO TO 910
       END IF

!  record workspace partitions in userdata%integer

       userdata%integer( loc_m ) = m
       userdata%integer( loc_n ) = n
       userdata%integer( loc_m_a ) = nlp%m_a
       userdata%integer( loc_nnzh ) = nnzh
       userdata%integer( loc_irnh ) = irnh
       userdata%integer( loc_icnh ) = icnh
       userdata%integer( loc_h ) = h

!  evaluate the Hessian and get sparsity pattern

       IF ( .NOT. no_hess ) THEN
         lh = nnzh
!        CALL CUTEST_ush_r( cutest_status, nlp%n, nlp%X, nnzh, lh,             &
!                  userdata%real( h + 1 : h + nnzh ),                          &
!                  userdata%integer( irnh + 1 : irnh + nnzh ),                 &
!                  userdata%integer( icnh + 1 : icnh + nnzh ) )
         CALL CUTEST_ushp_r( cutest_status, nlp%n, nnzh, lh,                   &
                             userdata%integer( irnh + 1 : irnh + nnzh ),       &
                             userdata%integer( icnh + 1 : icnh + nnzh ) )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  define the storage type for the null J and A

       nlp%J%ne = 0
       nlp%A%ne = 0

       CALL SMT_put( nlp%A%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

       CALL SMT_put( nlp%J%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

!  allocate zero-length arrays to prevent errors

       CALL SPACE_resize_array( nlp%m, nlp%Y, inform%status,                   &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%Y' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%C, inform%status,                   &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%C_l, inform%status,                 &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C_l' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%C_u, inform%status,                 &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%C_u' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m, nlp%CNAMES, inform%status,              &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%CNAMES' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%m_a, nlp%ANAMES, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'nlp%ANAMES' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nlp%A%ne, nlp%A%row, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%A%row' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%A%ne, nlp%A%col, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%A%col' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%A%ne, nlp%A%val, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%A%val' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%J%ne, nlp%J%row, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%J%row' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%J%ne, nlp%J%col, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%J%col' ; GO TO 910
       END IF

       CALL SPACE_resize_array( nlp%J%ne, nlp%J%val, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%J%val' ; GO TO 910
       END IF

       CALL SPACE_resize_array( 0_ip_, nlp%Ax, inform%status,                  &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%Ax' ; GO TO 910
       END IF

       CALL SPACE_resize_array( 0_ip_, nlp%Y_a, inform%status,                 &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%Ax' ; GO TO 910
       END IF

     END IF

!  define reduced costs

     nlp%Z = zero

!  define the storage type for H

     CALL SMT_put( nlp%H%type, 'COORDINATE', status )
     IF ( status /= 0 ) GO TO 990

!  set the sparsity pattern for H

     nlp%H%n = n
     nlp%H%ne = nnzh

     CALL SPACE_resize_array( nlp%H%ne, nlp%H%row, inform%status,              &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
        inform%bad_alloc = 'nlp%H%row' ; GO TO 910
     END IF

     CALL SPACE_resize_array( nlp%H%ne, nlp%H%col, inform%status,              &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
        inform%bad_alloc = 'nlp%H%col' ; GO TO 910
     END IF

     CALL SPACE_resize_array( nlp%H%ne, nlp%H%val, inform%status,              &
                              inform%alloc_status )
     IF ( inform%status /= 0 ) THEN
        inform%bad_alloc = 'nlp%H%val' ; GO TO 910
     END IF

!  ensure that the lower triangle of H is stored

     DO l = 1, nlp%H%ne
       i = userdata%integer( irnh + l )
       j = userdata%integer( icnh + l )
       IF ( i > j ) THEN
         nlp%H%row( l ) = i
         nlp%H%col( l ) = j
       ELSE
         nlp%H%row( l ) = j
         nlp%H%col( l ) = i
       END IF
     END DO

!  set the space and sparsity pattern for P

     IF ( hess_prods ) THEN
       CALL SMT_put( nlp%P%type, 'SPARSE_BY_COLUMNS', inform%status )
       IF ( inform%status /= 0 ) GO TO 990

!      nnzchp = userdata%integer( ichpptr + m + 1 ) - 1

       CALL SPACE_resize_array( nnzchp, nlp%P%val, inform%status,              &
                                inform%alloc_status, exact_size = .TRUE. )
       IF ( inform%status /= 0 ) then
         inform%bad_alloc = 'nlp%P%val' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( nnzchp, nlp%P%row, inform%status,              &
                                inform%alloc_status, exact_size = .TRUE. )
       IF ( inform%status /= 0 ) then
         inform%bad_alloc = 'nlp%P%row' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( m + 1, nlp%P%ptr, inform%status,               &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) then
         inform%bad_alloc = 'nlp%P%ptr' ; GO TO 910 ; END IF

       nlp%P%row( : nnzchp ) = userdata%integer( ichpind + 1 : ichpind + nnzchp)
       nlp%P%ptr( : m + 1 ) = userdata%integer( ichpptr + 1 : ichpptr + m + 1 )
     END IF

     RETURN

!  error returns

 910 CONTINUE
     WRITE( control%error, "( ' ** ERROR - subroutine CUTEst_functions - ',    &
          &         'Allocation error (status = ', I0, ') for ', A, / )" )     &
          inform%status, inform%bad_alloc
     RETURN

 920 CONTINUE
     WRITE( control%error, "( ' ** ERROR - subroutine CUTEst_functions - ',    &
          &         'Deallocation error (status = ', I0, ') for ', A, / )" ) &
          inform%status, inform%bad_alloc
     RETURN

 930 CONTINUE
     WRITE( control%error, "( ' ** ERROR - subroutine CUTEst_functions - ',    &
          &         'CUTEst error (status = ', I0, ')' )" ) cutest_status
     status = cutest_status
     RETURN

 990 CONTINUE
     inform%status = status
     WRITE( control%error,"(' CUTEst_initialize: error using subroutine',      &
    &   ' SMT_put - status= ', I0 ) " )  status

     RETURN

     END SUBROUTINE CUTEst_initialize

!-*-*-*-*-*-*-*-   C U T E S T _ e v a l _ F   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_F( status, X, userdata, f )

!  Evaluate the objective function f(X)

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), INTENT( OUT ) :: f
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! local variables

     INTEGER ( KIND = ip_ ) :: m, n

!  Extract scalar addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )

     IF ( m > 0 ) THEN
       CALL CUTEST_cfn_r( status, n, m, X, f, userdata%real( : m ) )
     ELSE
       CALL CUTEST_ufn_r( status, n, X, f )
     END IF

     RETURN

     END SUBROUTINE CUTEst_eval_F

!-*-*-*-*-*-*-*-   C U T E S T _ e v a l _ C   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_C( status, X, userdata, C )

!  Evaluate the constraint functions C(X)

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: C
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL ( KIND = rp_ ) :: f

!  Extract scalar addresses

     m   = userdata%integer( loc_m )
     n   = userdata%integer( loc_n )
     m_a = userdata%integer( loc_m_a )

!    CALL CUTEST_cfn_r( status, n, m, X, f, userdata%real( : m ) )
     CALL CUTEST_cfn_r( status, n, m, X, f, userdata%real )
     IF ( status == 0 ) C = userdata%real( m_a + 1 : m )

     RETURN

     END SUBROUTINE CUTEst_eval_C

!-*-*-*-*-*-*-*-   C U T E S T _ e v a l _ F C   S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_FC( status, X, userdata, f, C )

!  Evaluate the objective function f(X) and constraint functions C(X)

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), OPTIONAL, INTENT( OUT ) :: f
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL ( KIND = rp_ ) :: f_dummy

!  Extract scalar addresses

     m   = userdata%integer( loc_m )
     m_a = userdata%integer( loc_m_a )
     n   = userdata%integer( loc_n )

!    CALL CUTEST_cfn_r( status, n, m, X, f_dummy, userdata%real( : m ) )
     CALL CUTEST_cfn_r( status, n, m, X, f_dummy, userdata%real )
     IF ( status == 0 ) THEN
       IF ( PRESENT( f ) ) f = f_dummy
       IF ( PRESENT( C ) ) C = userdata%real( m_a + 1 : m )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_FC

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ G   S U B R O U T I N E  -*-*-*-*-*-*-

     SUBROUTINE CUTEst_eval_G( status, X, userdata, G )

!  Evaluate the gradient of the objective function G(X)

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: m, n
     REAL ( KIND = rp_ ) :: f_dummy
!    INTEGER ( KIND = ip_ ) :: m, n, nnzj, indfun, indvar, cjac, lcjac, l
!    REAL ( KIND = rp_ ), DIMENSION( 1 ) :: Y_dummy = zero

!  Extract scalar and array addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )

     IF ( m > 0 ) THEN
       CALL CUTEST_cofg_r( status, n, X, f_dummy, G, .TRUE. )
       IF ( status /= 0 ) RETURN

!      nnzj = userdata%integer( loc_nnzj )
!      indfun = userdata%integer( loc_indfun )
!      indvar = userdata%integer( loc_indvar )
!      cjac = userdata%integer( loc_cjac )
!      lcjac = nnzj
!      CALL CUTEST_csgr_r( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,     &
!                          userdata%real( cjac + 1 : cjac + nnzj ),            &
!                          userdata%integer( indvar + 1 : indvar + nnzj ),     &
!                          userdata%integer( indfun + 1 : indfun + nnzj ) )
!      IF ( status /= 0 ) RETURN
! Untangle A: separate the gradient terms from the constraint Jacobian
!      G( : n ) = zero
!      DO l = 1, nnzj
!         IF ( userdata%integer( indfun + l ) == 0 ) THEN
!            G( userdata%integer( indvar + l ) ) = userdata%real( cjac + l )
!         END IF
!      END DO
     ELSE
       CALL CUTEST_ugr_r( status, n, X, G )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_G

!-*-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ J   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_J( status, X, userdata, J_val )

!  Evaluate the values of the constraint Jacobian Jval(X) for the nonzeros
!  corresponding to the sparse coordinate format set in CUTEst_initialize

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n, nnzj
     INTEGER ( KIND = ip_ ) :: indfun, indvar, cjac, Jne, lcjac, l
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: Y_dummy

!  Extract scalar and array addresses

     m = userdata%integer( loc_m )

     IF ( m > 0 ) THEN
       m_a    = userdata%integer( loc_m_a )
       n      = userdata%integer( loc_n )
       nnzj   = userdata%integer( loc_nnzj )
       indfun = userdata%integer( loc_indfun )
       indvar = userdata%integer( loc_indvar )
       cjac   = userdata%integer( loc_cjac )

       lcjac = nnzj ; Y_dummy = zero
       CALL CUTEST_csgr_r( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,     &
                           userdata%real( cjac + 1 : cjac + nnzj ),            &
                           userdata%integer( indvar + 1 : indvar + nnzj ),     &
                           userdata%integer( indfun + 1 : indfun + nnzj ) )
       IF ( status /= 0 ) RETURN

! Untangle A: separate the constraint Jacobian from the objective gradient
!             and the linear constraints.

       Jne = 0
       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) > m_a ) THEN
           Jne = Jne + 1
           J_val( Jne ) = userdata%real( cjac + l )
         END IF
       END DO
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_J

!-*-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ G J   S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_GJ( status, X, userdata, G, J_val )

!  Evaluate the gradient of the objective function G(X) and the values
!  of the constraint Jacobian Jval(X) for the nonzeros corresponding to
!  the sparse coordinate format set in CUTEst_initialize

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G, J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n, nnzj
     INTEGER ( KIND = ip_ ) :: indfun, indvar, cjac, Jne, lcjac, l
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: Y_dummy

!  Extract scalar and array addresses

     m      = userdata%integer( loc_m )
     n      = userdata%integer( loc_n )
     IF ( m > 0 ) THEN
       m_a    = userdata%integer( loc_m_a )
       nnzj   = userdata%integer( loc_nnzj )
       indfun = userdata%integer( loc_indfun )
       indvar = userdata%integer( loc_indvar )
       cjac   = userdata%integer( loc_cjac )

       lcjac = nnzj ; Y_dummy = zero
       CALL CUTEST_csgr_r( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,     &
                           userdata%real( cjac + 1 : cjac + nnzj ),            &
                           userdata%integer( indvar + 1 : indvar + nnzj ),     &
                           userdata%integer( indfun + 1 : indfun + nnzj ) )
       IF ( status /= 0 ) RETURN

! Untangle A: separate the gradient terms from the constraint Jacobian

       Jne = 0
       IF ( PRESENT( G ) ) G( : n ) = zero
       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) == 0 ) THEN
           IF ( PRESENT( G ) )                                                 &
             G( userdata%integer( indvar + l ) ) = userdata%real( cjac + l )
         ELSE IF ( userdata%integer( indfun + l ) > m_a ) then
           IF ( PRESENT( J_val ) ) THEN
             Jne = Jne + 1
             J_val( Jne ) = userdata%real( cjac + l )
           END IF
         END IF
       END DO
     ELSE IF ( PRESENT( G ) ) THEN
       CALL CUTEST_ugr_r( status, n, X, G )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_GJ

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ S G J   S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_SGJ( status, X, userdata, G, J_val )

!  Evaluate the values of the gradient of the objective function G(X) and
!  those of the constraint Jacobian Jval(X) for the nonzeros corresponding
!  to the sparse coordinate formats set in CUTEst_initialize

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: G, J_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n, nnzg, nnzj
     INTEGER ( KIND = ip_ ) :: indg, indfun, indvar, cjac, Jne, lg, lcjac, l
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: Y_dummy

!  Extract scalar and array addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )
     nnzg = userdata%integer( loc_nnzg )
     IF ( m > 0 ) THEN
       m_a = userdata%integer( loc_m_a )
       nnzj = userdata%integer( loc_nnzj )
       indfun = userdata%integer( loc_indfun )
       indvar = userdata%integer( loc_indvar )
       cjac = userdata%integer( loc_cjac )

       lcjac = nnzj ; Y_dummy = zero
       CALL CUTEST_csgr_r( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,     &
                           userdata%real( cjac + 1 : cjac + nnzj ),            &
                           userdata%integer( indvar + 1 : indvar + nnzj ),     &
                           userdata%integer( indfun + 1 : indfun + nnzj ) )
       IF ( status /= 0 ) RETURN

! Untangle A: separate the gradient terms from the constraint Jacobian

       Jne = 0
       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) > m_a ) THEN
           Jne = Jne + 1
           J_val( Jne ) = userdata%real( cjac + l )
         END IF
       END DO
     END IF

!  compute the sparse gradient

     lg = nnzg
     indg = userdata%integer( loc_indg )
     CALL CUTEST_cisgr_r( status, n, 0_ip_, X, nnzg, lg, G,                    &
                          userdata%integer( indg + 1 : indg + nnzg ) )

     RETURN

     END SUBROUTINE CUTEst_eval_SGJ

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ H    S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_H( status, X, userdata, H_val )

!  Evaluate the values of the Herssian of the objective function H_val(X)
!  for the nonzeros corresponding to the sparse coordinate format set in
!  CUTEst_initialize.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: n, nnzh, irnh, icnh, lh

!  Extract scalar and array addresses

     n = userdata%integer( loc_n )
     nnzh = userdata%integer( loc_nnzh )
     irnh = userdata%integer( loc_irnh )
     icnh = userdata%integer( loc_icnh )

! Evaluate the Hessian

     lh = nnzh
     CALL CUTEST_ush_r( status, n, X, nnzh, lh, H_val,                         &
                        userdata%integer( irnh + 1 : irnh + nnzh ),            &
                        userdata%integer( icnh + 1 : icnh + nnzh ) )
     RETURN

     END SUBROUTINE CUTEst_eval_H

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ H L   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_HL( status, X, Y, userdata, H_val, no_f )

!  Evaluate the values of the Herssian of the Lagrangian function Hval(X,Y)
!  for the nonzeros corresponding to the sparse coordinate format set in
!  CUTEst_initialize. By convention, the Lagrangian function is f - sum c_i y_i,
!  unless no_f is is PRESENT and TRUE in which case the Lagrangian function
!  is - sum c_i y_i

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_f

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n, nnzh, irnh, icnh, lh
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: Y_full
     LOGICAL :: no_f_value

     IF ( PRESENT( no_f ) ) THEN
       no_f_value = no_f
     ELSE
       no_f_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m    = userdata%integer( loc_m )
     m_a  = userdata%integer( loc_m_a )
     n    = userdata%integer( loc_n )
     nnzh = userdata%integer( loc_nnzh )
     irnh = userdata%integer( loc_irnh )
     icnh = userdata%integer( loc_icnh )

! Evaluate the Hessian

     Y_full                = zero
     Y_full( m_a + 1 : m ) = - Y

     lh = nnzh
     IF ( no_f_value ) THEN
       CALL CUTEST_cshc_r( status, n, m, X, Y_full, nnzh, lh, H_val,           &
                           userdata%integer( irnh + 1 : irnh + nnzh ),         &
                           userdata%integer( icnh + 1 : icnh + nnzh ) )
     ELSE
       CALL CUTEST_csh_r( status, n, m, X, Y_full, nnzh, lh, H_val,            &
                          userdata%integer( irnh + 1 : irnh + nnzh ),          &
                          userdata%integer( icnh + 1 : icnh + nnzh ) )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_HL

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ H J   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_HJ( status, X, y0, Y, userdata, H_val )

!  Evaluate the values of the Herssian of the John function Hval(X,y0,Y)
!  for the nonzeros corresponding to the sparse coordinate format set in
!  CUTEst_initialize. By convention, the John function is y_0 f - sum c_i y_i

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), INTENT( IN ) :: y0
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n, nnzh, irnh, icnh, lh
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: Y_full

!  Extract scalar and array addresses

     m    = userdata%integer( loc_m )
     m_a  = userdata%integer( loc_m_a )
     n    = userdata%integer( loc_n )
     nnzh = userdata%integer( loc_nnzh )
     irnh = userdata%integer( loc_irnh )
     icnh = userdata%integer( loc_icnh )

! Evaluate the Hessian

     Y_full = zero
     Y_full( m_a + 1 : m ) = - Y

     lh = nnzh
     CALL CUTEST_cshj_r( status, n, m, X, y0, Y_full, nnzh, lh, H_val,         &
                        userdata%integer( irnh + 1 : irnh + nnzh ),            &
                        userdata%integer( icnh + 1 : icnh + nnzh ) )
     RETURN

     END SUBROUTINE CUTEst_eval_HJ

!-*-*-*-*-*-*-  C U T E S T _ e v a l _ H L C   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_HLC( status, X, Y, userdata, H_val )

!  Evaluate the values of the Hessian of the constraint Lagrangian function,
!  sum c_i y_i for the nonzeros corresponding to the sparse coordinate format
!  set in CUTEst_initialize

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n, nnzh, irnh, icnh, lh
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: Y_full

!  Extract scalar and array addresses

     m    = userdata%integer( loc_m )
     m_a  = userdata%integer( loc_m_a )
     n    = userdata%integer( loc_n )
     nnzh = userdata%integer( loc_nnzh )
     irnh = userdata%integer( loc_irnh )
     icnh = userdata%integer( loc_icnh )

! Evaluate the Hessian

     Y_full                = zero
     Y_full( m_a + 1 : m ) = Y

     lh = nnzh
     CALL CUTEST_cshc_r( status,  n, m, X, Y_full, nnzh, lh, H_val,            &
                         userdata%integer( irnh + 1 : irnh + nnzh ),           &
                         userdata%integer( icnh + 1 : icnh + nnzh ) )
     RETURN

     END SUBROUTINE CUTEst_eval_HLC

!-*-*-*-*-*  C U T E S T _ e v a l _ J P R O D    S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_JPROD( status, X, userdata, transpose, U, V, got_j )

!  Compute the Jacobian-vector product
!    U = U + J(X) * V
!  (if transpose is .FALSE.) or
!    U = U + J(X)' * V
!  (if transpose is .TRUE.). If got_j is PRESENT and TRUE, the Jacobian is as
!  recorded at the last point at which it was evaluated.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: full_V
     LOGICAL :: got_j_value

     IF ( PRESENT( got_j ) ) THEN
       got_j_value = got_j
     ELSE
       got_j_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m    = userdata%integer( loc_m )
     m_a  = userdata%integer( loc_m_a )
     n    = userdata%integer( loc_n )

     IF ( transpose ) then
       full_V = zero
       full_V( m_a + 1 : m ) = V
     END IF

     IF ( got_j_value ) THEN
       IF ( transpose ) THEN
         CALL CUTEST_cjprod_r( status, n, m, .FALSE., transpose, X, full_V,    &
                               m, userdata%real( : n ), n )
       ELSE
         CALL CUTEST_cjprod_r( status, n, m, .FALSE., transpose, X, V, n,      &
                               userdata%real( : m ), m )
       END IF
     ELSE
       IF ( transpose ) THEN
         CALL CUTEST_cjprod_r( status, n, m, .TRUE., transpose,                &
                               userdata%real( : n ), full_V,                   &
                               m, userdata%real( : n ), n )
       ELSE
         CALL CUTEST_cjprod_r( status, n, m, .TRUE., transpose,                &
                               userdata%real( : n ), V,                        &
                               n, userdata%real( : m ), m )
       END IF
     END IF
     IF ( status /= 0 ) RETURN

     IF ( transpose ) THEN
       U( : n ) = U( : n ) + userdata%real( : n )
     ELSE
       U( : m - m_a ) = U( : m - m_a ) + userdata%real( m_a + 1 : m )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_JPROD

!-*-*-*-*-*  C U T E S T _ e v a l _ J P R O D    S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_SJPROD( status, X, userdata, transpose, nnz_v,     &
                                    INDEX_nz_v, V, nnz_u, INDEX_nz_u, U, got_j )

!  Compute the sparse Jacobian-vector product
!    U = J(X) * V
!  (if transpose is .FALSE.) or
!    U = J(X)' * V
!  (if transpose is .TRUE.) involving a sparse vector V. If got_j is PRESENT
!  and TRUE, the Jacobian is as recorded at the last point at which it was
!  evaluated.

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz_v
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nnz_u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j

! Local variables

     INTEGER ( KIND = ip_ ) :: m, n
     LOGICAL :: got_j_value

     IF ( PRESENT( got_j ) ) THEN
       got_j_value = got_j
     ELSE
       got_j_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m    = userdata%integer( loc_m )
     n    = userdata%integer( loc_n )

     IF ( got_j_value ) THEN
       IF ( transpose ) THEN
         CALL CUTEST_csjprod_r( status, n, m, .FALSE., transpose, X, nnz_v,    &
                                INDEX_nz_v, V, m, nnz_u, INDEX_nz_u, U, n )
       ELSE
         CALL CUTEST_csjprod_r( status, n, m, .FALSE., transpose, X, nnz_v,    &
                                INDEX_nz_v, V, n, nnz_u, INDEX_nz_u, U, m  )
       END IF
     ELSE
       IF ( transpose ) THEN
         CALL CUTEST_csjprod_r( status, n, m, .TRUE., transpose, X, nnz_v,     &
                                INDEX_nz_v, V, m, nnz_u, INDEX_nz_u, U, n )
       ELSE
         CALL CUTEST_csjprod_r( status, n, m, .TRUE., transpose, X, nnz_v,     &
                                INDEX_nz_v, V, n, nnz_u, INDEX_nz_u, U, m )
       END IF
     END IF

     RETURN

     END SUBROUTINE CUTEst_eval_SJPROD

!-*-*-*-*-  C U T E S T _ e v a l _ H P R O D    S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_HPROD( status, X, userdata, U, V, got_h )

!  Compute the product U = U + H(X) * V involving the Hessian of the objective
!  H(X). If got_h is present, the Hessian is as recorded at the last point at
!  which it was evaluated.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER ( KIND = ip_ ) :: n
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     n = userdata%integer( loc_n )

     IF ( got_h_value ) THEN
       CALL CUTEST_uhprod_r( status, n, .TRUE., X, V, userdata%real( : n ) )
     ELSE
       CALL CUTEST_uhprod_r( status, n, .FALSE., X, V, userdata%real( : n ) )
     END IF
     IF ( status /= 0 ) RETURN

     U( : n ) = U( : n ) + userdata%real( : n )
     RETURN

     END SUBROUTINE CUTEst_eval_HPROD

!-*-*-*-*-  C U T E S T _ e v a l _ S H P R O D    S U B R O U T I N E  -*-*-*-

     SUBROUTINE CUTEst_eval_SHPROD( status, X, userdata, nnz_v, INDEX_nz_v,    &
                                    V, nnz_u, INDEX_nz_u, U, got_h )

!  Compute the product U = H(X) * V involving the Hessian of the objective
!  H(X) with sparse V. If got_h is present, the Hessian is as recorded at
!  the last point at which it was evaluated.

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz_v
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nnz_u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER ( KIND = ip_ ) :: n
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     n = userdata%integer( loc_n )

     IF ( got_h_value ) THEN
       CALL CUTEST_ushprod_r( status, n, .TRUE., X,                            &
                              nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     ELSE
       CALL CUTEST_ushprod_r( status, n, .FALSE., X,                           &
                              nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_SHPROD

!-*-*-*-*-  C U T E S T _ e v a l _ H L P R O D    S U B R O U T I N E  -*-*-*-

     SUBROUTINE CUTEst_eval_HLPROD( status, X, Y, userdata, U, V, no_f, got_h )

!  Compute the product U = U + H(X,Y) * V involving the Hessian of the
!  Lagrangian H(X,Y). If got_h is PRESENT and TRUE, the Hessian is as
!  recorded at the last point at which it was evaluated. By convention, the
!  Lagrangian function is f - sum c_i y_i, unless no_f is is PRESENT and TRUE
!  in which case the Lagrangian function is - sum c_i y_i

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
     LOGICAL :: no_f_value, got_h_value

     IF ( PRESENT( no_f ) ) THEN
       no_f_value = no_f
     ELSE
       no_f_value = .FALSE.
     END IF

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m   = userdata%integer( loc_m )
     m_a = userdata%integer( loc_m_a )
     n   = userdata%integer( loc_n )

     IF ( no_f_value ) THEN
       IF ( got_h_value ) THEN
         CALL CUTEST_chcprod_r( status, n, m, .TRUE., X, - full_Y, V,          &
                                userdata%real( : n ) )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_chcprod_r( status, n, m, .FALSE., X, - full_Y, V,         &
                                userdata%real( : n ) )
       END IF
     ELSE
       IF ( got_h_value ) THEN
         CALL CUTEST_chprod_r( status, n, m, .TRUE., X, - full_Y, V,           &
                               userdata%real( : n ) )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_chprod_r( status, n, m, .FALSE., X, - full_Y, V,          &
                               userdata%real( : n ) )
       END IF
     END IF
     IF ( status /= 0 ) RETURN

     U( : n ) = U( : n ) + userdata%real( : n )
     RETURN

     END SUBROUTINE CUTEst_eval_HLPROD

!-*-*-*-  C U T E S T _ e v a l _ S H L P R O D    S U B R O U T I N E  -*-*-*-

     SUBROUTINE CUTEst_eval_SHLPROD( status, X, Y, userdata, nnz_v,            &
                                     INDEX_nz_v, V, nnz_u, INDEX_nz_u, U,      &
                                     no_f, got_h )

!  Compute the product U = H(X,Y) * V involving the Hessian of the
!  Lagrangian H(X,Y) and the sparse vector V. If got_h is PRESENT and TRUE,
!  the Hessian is as recorded at the last point at which it was evaluated. By
!  convention, the Lagrangian function is f - sum c_i y_i, unless no_f is is
!  PRESENT and TRUE in which case the Lagrangian function is - sum c_i y_i

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz_v
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nnz_u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
     LOGICAL :: no_f_value, got_h_value

     IF ( PRESENT( no_f ) ) THEN
       no_f_value = no_f
     ELSE
       no_f_value = .FALSE.
     END IF

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m   = userdata%integer( loc_m )
     m_a = userdata%integer( loc_m_a )
     n   = userdata%integer( loc_n )

     IF ( no_f_value ) THEN
       IF ( got_h_value ) THEN
         CALL CUTEST_cshcprod_r( status, n, m, .TRUE., X, - full_Y,            &
                                 nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_cshcprod_r( status, n, m, .FALSE., X, - full_Y,           &
                                 nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       END IF
     ELSE
       IF ( got_h_value ) THEN
         CALL CUTEST_cshprod_r( status, n, m, .TRUE., X, - full_Y,             &
                                nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_cshprod_r( status, n, m, .FALSE., X, - full_Y,            &
                                nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       END IF
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_SHLPROD

!-*-*-*-  C U T E S T _ e v a l _ H L C P R O D    S U B R O U T I N E  -*-*-*-

     SUBROUTINE CUTEst_eval_HLCPROD( status, X, Y, userdata, U, V, got_h )

!  Compute the product U = U + H(X,Y) * V involving the Hessian of the
!  Lagrangian of the constraints, sum c_i y_i. If got_h is PRESENT and TRUE,
!  the Hessian is as recorded at the last point at which it was evaluated.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m   = userdata%integer( loc_m )
     m_a = userdata%integer( loc_m_a )
     n   = userdata%integer( loc_n )

     IF ( got_h_value ) THEN
       CALL CUTEST_chcprod_r( status, n, m, .TRUE., X, full_Y, V,              &
                              userdata%real( : n ) )
     ELSE
       full_Y( 1 : m_a ) = zero
       full_Y( m_a + 1 : m  ) = Y
       CALL CUTEST_chcprod_r( status, n, m, .FALSE., X, full_Y, V,             &
                              userdata%real( : n ) )
     END IF
     IF ( status /= 0 ) RETURN

     U( : n ) = U( : n ) + userdata%real( : n )
     RETURN

     END SUBROUTINE CUTEst_eval_HLCPROD

!-*-*-*-*-  C U T E S T _ e v a l _ S H L P R O D    S U B R O U T I N E  -*-*-

     SUBROUTINE CUTEst_eval_SHLCPROD( status, X, Y, userdata, nnz_v,           &
                                      INDEX_nz_v, V, nnz_u, INDEX_nz_u, U,     &
                                      got_h )

!  Compute the product U = H(X,Y) * V involving the Hessian of the Lagrangian
!  H(X,Y) of the constraints, sum c_i y_i, and the sparse vector V. If got_h
!  is PRESENT and TRUE, the Hessian is as recorded at the last point at which
!  it was evaluated.

     INTEGER ( KIND = ip_ ), INTENT( IN ) :: nnz_v
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: nnz_u
     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER ( KIND = ip_ ), DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

!  local variables

     INTEGER ( KIND = ip_ ) :: m, m_a, n
     REAL ( KIND = rp_ ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m   = userdata%integer( loc_m )
     m_a = userdata%integer( loc_m_a )
     n   = userdata%integer( loc_n )

     IF ( got_h_value ) THEN
       CALL CUTEST_cshcprod_r( status, n, m, .TRUE., X, - full_Y,              &
                               nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     ELSE
       full_Y( 1 : m_a ) = zero
       full_Y( m_a + 1 : m  ) = Y
       CALL CUTEST_cshcprod_r( status, n, m, .FALSE., X, - full_Y,             &
                               nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_SHLCPROD

!-*-*-*-  C U T E S T _ e v a l _ H C P R O D S    S U B R O U T I N E  -*-*-*-

     SUBROUTINE CUTEst_eval_HCPRODS( status, X, V, userdata, P_val, got_h )

!  Compute the products P = H_i(X) * V involving the Hessians of the
!  constraints. If got_h is PRESENT and TRUE, the Hessians are as
!  recorded at the last point at which they were evaluated.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

!  local variables

     INTEGER ( KIND = ip_ ) :: m, n, ichpind, ichpptr, lchp
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  extract scalar and array addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )
     lchp = userdata%integer( loc_lchp )
     ichpind = userdata%integer( loc_chpind )
     ichpptr = userdata%integer( loc_chpptr )

!  compute the values of P

     CALL CUTEST_cchprods_r( status, n, m, got_h_value, X, V, lchp, P_val,     &
                             userdata%integer( ichpind + 1 : ichpind + lchp ), &
                             userdata%integer( ichpptr + 1 : ichpptr + m + 1 ) )
     RETURN

     END SUBROUTINE CUTEst_eval_HCPRODS

!-*-*-*-  C U T E S T _ e v a l _ H O C P R O D S    S U B R O U T I N E  -*-*-

     SUBROUTINE CUTEst_eval_HOCPRODS( status, X, V, userdata,                  &
                                      PO_val, PC_val, got_h )

!  Compute the products P_o = H(X) * V and P_c = H_i(X) * V involving the
!  Hessians of the objective (H) and constraints (H_i). If got_h is PRESENT
!  and TRUE, the Hessians are as recorded at the last point at which they
!  were evaluated.

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: PO_val
     REAL ( KIND = rp_ ), DIMENSION( : ), INTENT( INOUT ) :: PC_val
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

!  local variables

     INTEGER ( KIND = ip_ ) :: m, n, nnzohp
     INTEGER ( KIND = ip_ ) :: ichpind, ichpptr, iohpind, lchp, lohp
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  extract scalar and array addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )
     lohp = userdata%integer( loc_lohp )
     lchp = userdata%integer( loc_lchp )
     iohpind = userdata%integer( loc_ohpind )
     ichpind = userdata%integer( loc_chpind )
     ichpptr = userdata%integer( loc_chpptr )

!  compute the values of P_o and P_c

     CALL CUTEST_cohprods_r( status, n, got_h_value, X, V,                     &
                             nnzohp, lohp, PO_val,                             &
                           userdata%integer( iohpind + 1 : iohpind + nnzohp ) )

     CALL CUTEST_cchprods_r( status, n, m, got_h_value, X, V, lchp, PC_val,    &
                             userdata%integer( ichpind + 1 : ichpind + lchp ), &
                             userdata%integer( ichpptr + 1 : ichpptr + m + 1 ) )
     RETURN

     END SUBROUTINE CUTEst_eval_HOCPRODS

!-*-*-*-*-*-   C U T E S T  _ t e r m i n a t e   S U B R O U T I N E   -*-*-*-

     SUBROUTINE CUTEst_terminate( nlp, inform, userdata )

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!      ..............................................
!      .                                            .
!      .  Deallocate internal arrays at the end     .
!      .  of the computation                        .
!      .                                            .
!      ..............................................

! =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

!  dummy arguments

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( CUTEST_inform_type ), INTENT( OUT ) :: inform

     CALL NLPT_cleanup( nlp )
     CALL SPACE_dealloc_array( userdata%integer, inform%status,                &
                               inform%alloc_status )
     CALL SPACE_dealloc_array( userdata%real, inform%status,                   &
                               inform%alloc_status )
     RETURN

!  End of subroutine CUTEst_terminate

     END SUBROUTINE CUTEst_terminate

!-*-*-*-   C U T E S T  _ s t a r t _ t i m i n g   S U B R O U T I N E   -*-*-

     SUBROUTINE CUTEst_start_timing( status )

!  Dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     REAL ( KIND = rp_ ) :: time

!  local variables

     REAL :: time_real

!  enable recording of timings for CUTEst calls

     CALL CUTEST_timings_r( status, 'start', time_real )
     time = REAL( time_real, rp_ )

     RETURN

!  End of subroutine CUTEst_start_timing

     END SUBROUTINE CUTEst_start_timing

!-*-*-*-*-*-   C U T E S T  _ t i m i n g   S U B R O U T I N E   -*-*-*-*-

     SUBROUTINE CUTEst_timing( status, userdata, name, time )

!  ---------------------------------------------------------------------------
!  return the total CPU time spent in the GALAHAD CUTEst interface tool
!  called 'name' while the CPU monitor was turned on (see cutest_start_timing)
!  ---------------------------------------------------------------------------

!  dummy arguments

     INTEGER ( KIND = ip_ ), INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: name
     REAL ( KIND = rp_ ), INTENT( out ) :: time
     TYPE ( GALAHAD_userdata_type ), INTENT( INOUT ) :: userdata

!  local variables

     INTEGER ( KIND = ip_ ) :: m
     REAL :: time_real
     CHARACTER ( LEN = LEN( name ) ) :: name_lower_case

!  extract scalar addresses

     m = userdata%integer( loc_m )

!  convert the name to lower case

     name_lower_case = name
     CALL STRING_lower_word( name_lower_case )

!  record required time

     status = 0 ; time = 0.0_rp_

     SELECT CASE ( TRIM( name_lower_case ) )
     CASE ( 'cutest_eval_f' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings_r( status, 'cutest_cfn', time_real )
       ELSE
         CALL CUTEST_timings_r( status, 'cutest_ufn', time_real )
       END IF
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_fc' )
       CALL CUTEST_timings_r( status, 'cutest_cfn', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_c' )
       CALL CUTEST_timings_r( status, 'cutest_cfn', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_g' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings_r( status, 'cutest_cofg', time_real )
       ELSE
         CALL CUTEST_timings_r( status, 'cutest_ugr', time_real )
       END IF
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_gj' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings_r( status, 'cutest_csgr', time_real )
       ELSE
         CALL CUTEST_timings_r( status, 'cutest_ugr', time_real )
       END IF
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_j' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings_r( status, 'cutest_csgr', time_real )
         time = REAL( time_real, rp_ )
       END IF
     CASE ( 'cutest_eval_h' )
       CALL CUTEST_timings_r( status, 'cutest_ush', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_hprod' )
       CALL CUTEST_timings_r( status, 'cutest_uhprod', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_shprod' )
       CALL CUTEST_timings_r( status, 'cutest_ushprod', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_jprod' )
       CALL CUTEST_timings_r( status, 'cutest_cjprod', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_sjprod' )
       CALL CUTEST_timings_r( status, 'cutest_csjprod', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_hl' )
       CALL CUTEST_timings_r( status, 'cutest_csh', time_real )
       time = REAL( time_real, rp_ )
       CALL CUTEST_timings_r( status, 'cutest_cshc', time_real )
       time = time + REAL( time_real, rp_ )
     CASE ( 'cutest_eval_hlc' )
       CALL CUTEST_timings_r( status, 'cutest_cshc', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_hlprod' )
       CALL CUTEST_timings_r( status, 'cutest_chprod', time_real )
       time = REAL( time_real, rp_ )
       CALL CUTEST_timings_r( status, 'cutest_chcprod', time_real )
       time = time + REAL( time_real, rp_ )
     CASE ( 'cutest_eval_shlprod' )
       CALL CUTEST_timings_r( status, 'cutest_cshprod', time_real )
       time = REAL( time_real, rp_ )
       CALL CUTEST_timings_r( status, 'cutest_cshcprod', time_real )
       time = time + REAL( time_real, rp_ )
     CASE ( 'cutest_eval_hlcprod' )
       CALL CUTEST_timings_r( status, 'cutest_chcprod', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_shlcprod' )
       CALL CUTEST_timings_r( status, 'cutest_cshcprod', time_real )
       time = REAL( time_real, rp_ )
     CASE ( 'cutest_eval_hcprods' )
       CALL CUTEST_timings_r( status, 'cutest_cchprods', time_real )
       time = REAL( time_real, rp_ )
     CASE DEFAULT
       status = GALAHAD_unavailable_option
     END SELECT

     RETURN

!  End of subroutine CUTEst_timing

     END SUBROUTINE CUTEst_timing

   END MODULE GALAHAD_CUTEST_precision
