! THIS VERSION: GALAHAD 2.7 - 26/02/2016 AT 11:30 GMT.

!-*-*-*-*  G A L A H A D _ C U T E S T _ F U N C T I O N S  M O D U L E  *-*-*-

!  Copyright reserved, Gould/Orban/Toint, for GALAHAD productions
!  Principal authors: Daniel Robinson and Nick Gould
!
!  History -
!   originally released pre GALAHAD Version 2.2. February 22nd 2008
!
   MODULE GALAHAD_CUTEST_FUNCTIONS_double

     USE GALAHAD_SMT_double
     USE GALAHAD_SPACE_double
     USE GALAHAD_STRING_double
     USE GALAHAD_SYMBOLS
     USE GALAHAD_NLPT_double, ONLY: NLPT_problem_type, NLPT_userdata_type,     &
                                    NLPT_cleanup
     USE CUTEst_interface_double

     IMPLICIT NONE

!---------------------
!   P r e c i s i o n
!---------------------

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

     PRIVATE
     PUBLIC :: CUTEst_initialize, CUTEst_eval_F, CUTEst_eval_FC,               &
               CUTEst_eval_C, CUTEst_eval_G, CUTEst_eval_GJ, CUTEst_eval_J,    &
               CUTEst_eval_H, CUTEst_eval_HPROD, CUTEst_eval_SHPROD,           &
               CUTEst_eval_JPROD, CUTEst_eval_SJPROD, CUTEst_eval_HL,          &
               CUTEst_eval_HLC, CUTEst_eval_HLPROD, CUTEst_eval_SHLPROD,       &
               CUTEst_eval_HLCPROD, CUTEst_eval_SHLCPROD, CUTEst_eval_HCPRODS, &
               CUTEst_start_timing, CUTEst_timing,                             &
               CUTEst_terminate, NLPT_problem_type, NLPT_userdata_type

!------------------------------------------------
!  D e r i v e d   t y p e   d e f i n i t i o n
!------------------------------------------------

     TYPE, PUBLIC :: CUTEst_FUNCTIONS_control_type
       INTEGER :: input = 5
       INTEGER :: error = 6
       INTEGER :: io_buffer = 11
       LOGICAL :: separate_linear_constraints = .FALSE.
     END TYPE

     TYPE, PUBLIC :: CUTEst_FUNCTIONS_inform_type
       INTEGER :: status = 0
       INTEGER :: alloc_status = 0
       CHARACTER ( LEN = 80 ) :: bad_alloc = REPEAT( ' ', 80 )
     END TYPE

!----------------------
!   P a r a m e t e r s
!----------------------

     REAL ( KIND = wp ), PARAMETER ::  zero  = 0.0_wp
     REAL ( KIND = wp ), PARAMETER ::  one   = 1.0_wp
     REAL ( KIND = wp ), PARAMETER ::  two   = 2.0_wp
     REAL ( KIND = wp ), PARAMETER ::  ten   = 10.0_wp
     REAL ( KIND = wp ), PARAMETER ::  small = ten ** ( -8 )
     REAL ( KIND = wp ), PARAMETER ::  huge  = ten ** ( 19 )

     INTEGER, PARAMETER :: loc_m = 1
     INTEGER, PARAMETER :: loc_n = 2
     INTEGER, PARAMETER :: loc_m_a = 3
     INTEGER, PARAMETER :: loc_nnzh = 4
     INTEGER, PARAMETER :: loc_irnh = 5
     INTEGER, PARAMETER :: loc_icnh = 6
     INTEGER, PARAMETER :: loc_h = 7
     INTEGER, PARAMETER :: loc_nnzj = 8
     INTEGER, PARAMETER :: loc_indfun = 9
     INTEGER, PARAMETER :: loc_indvar = 10
     INTEGER, PARAMETER :: loc_cjac = 11
     INTEGER, PARAMETER :: loc_nnzchp = 12
     INTEGER, PARAMETER :: loc_chpind = 13
     INTEGER, PARAMETER :: loc_chpptr = 14

!---------------------------------
!   I n t e r f a c e  b l o c k s
!---------------------------------

!    INTERFACE CUTEst_eval_H
!      MODULE PROCEDURE CUTEst_eval_H, CUTEst_eval_HL
!    END INTERFACE

!    INTERFACE CUTEst_eval_HPROD
!      MODULE PROCEDURE CUTEst_eval_HPROD, CUTEst_eval_HLPROD
!    END INTERFACE

   CONTAINS

!-*-*-  C U T E R _ i n i t i a l i z e   S U B R O U T I N E  -*-*-*-*

     SUBROUTINE CUTEst_initialize( nlp, control, inform, userdata,             &
                                   no_hessian, no_jacobian, hessian_products )

     TYPE ( NLPT_problem_type ), INTENT( OUT ) :: nlp
     TYPE ( NLPT_userdata_type ), INTENT( OUT ) :: userdata
     TYPE ( CUTEst_FUNCTIONS_control_type ), INTENT( IN ) :: control
     TYPE ( CUTEst_FUNCTIONS_inform_type ), INTENT( OUT ) :: inform
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_hessian, no_jacobian
     LOGICAL, OPTIONAL, INTENT( IN ) :: hessian_products

! local variables.

     INTEGER :: i, j, l, lcjac, lh, status, iend, rend, h, cjac
     INTEGER :: m, n, nnzj, nnzh, indfun, indvar, irnh, icnh, ihsind, ihsptr
     INTEGER :: nnzchp, l_order, cutest_status
     REAL( KIND = wp ) :: f, f2, alpha, alpha_min
     LOGICAL :: no_hess, no_jac, hess_prods
     REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: Y, C_l, C_u, C, X
     REAL( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: C2, lin_const
     CHARACTER( LEN = 10 ), ALLOCATABLE, DIMENSION( : ) :: full_CNAMES
     CHARACTER( LEN = 80 ) :: array_name

! get dimensions

     CALL CUTEST_cdimen( cutest_status, control%input, n, m )
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

!  Get the problem with linear constraints possibly first.

       i = n ; j = m
       IF ( control%separate_linear_constraints ) THEN
         l_order = 1
       ELSE
         l_order = 0
       END IF
       CALL CUTEST_csetup( cutest_status, control%input, control%error,        &
                           control%io_buffer, n, m, nlp%X, nlp%X_l, nlp%X_u,   &
                           Y, C_l, C_u, nlp%EQUATION, nlp%LINEAR, 0, l_order, 0)
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

!  Obtain the names of the problem, its variables and general constraints

       CALL CUTEST_cnames( cutest_status, n, m, nlp%pname, nlp%VNAMES,         &
                           full_CNAMES )
       IF ( cutest_status /= 0 ) GO TO 930

!  Define the "corrected" separated vectors.

       nlp%Y_a = Y  ( 1 : nlp%m_a ) ;  nlp%Y   = Y  ( nlp%m_a + 1 : m )
       nlp%A_l = C_l( 1 : nlp%m_a ) ;  nlp%C_l = C_l( nlp%m_a + 1 : m )
       nlp%A_u = C_u( 1 : nlp%m_a ) ;  nlp%C_u = C_u( nlp%m_a + 1 : m )

       nlp%ANAMES = full_CNAMES( 1 : nlp%m_a )
       nlp%CNAMES = full_CNAMES( nlp%m_a + 1 : m )

!  Deallocate arrays no longer needed.

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

! Set up sparsity structure for A, J, and H.  ( Assume co-ordinate storage )

! Determine number of non-zeros in the matrix of gradients of the
! objective function AND constraint functions.

       IF ( no_jac ) THEN
         nnzj = 0
       ELSE
         CALL CUTEST_cdimsj( cutest_status, nnzj )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  Determine how many nonzeros are required to store the Hessian matrix of the
!  Lagrangian, when the matrix is stored as a sparse matrix in "co-ordinate"
!  format (only the lower triangular part is stored).

       IF ( no_hess ) THEN
         nnzh = 0
       ELSE
         CALL CUTEST_cdimsh( cutest_status, nnzh )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  Determine how many nonzeros are required to store the matrix of products
!  of the constraint Hessians with a vector, when the matrix is stored in
!  sparse column-wise format.

       IF (  hess_prods ) THEN
         CALL CUTEST_cdimchp( cutest_status, nnzchp )
         IF ( cutest_status /= 0 ) GO TO 930
       ELSE
         nnzchp = 0
       END IF

!  Set starting addresses for workspace array partitions

       irnh = loc_chpptr
       icnh = irnh + nnzh
       indfun = icnh + nnzh
       indvar = indfun + nnzj
       ihsind = indvar + nnzj
       IF (  hess_prods ) THEN
         ihsptr = ihsind + nnzchp
         iend = ihsptr + m + 1
       ELSE
         ihsptr = ihsind
         iend = ihsptr
       END IF

       h    = 0
       cjac = h + nnzh
       rend = MAX( cjac + nnzj, m, n )

! Allocate space to hold scalars/arrays needed for subsequent calls

       CALL SPACE_resize_array( iend, userdata%integer, inform%status,         &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%integer' ; GO TO 910 ; END IF

       CALL SPACE_resize_array( rend, userdata%real, inform%status,            &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'userdata%real' ; GO TO 910 ; END IF

! Record workspace partitions in userdata%integer

       userdata%integer( loc_m ) = m
       userdata%integer( loc_n ) = n
       userdata%integer( loc_m_a ) = nlp%m_a
       userdata%integer( loc_nnzh ) = nnzh
       userdata%integer( loc_irnh ) = irnh
       userdata%integer( loc_icnh ) = icnh
       userdata%integer( loc_h ) = h
       userdata%integer( loc_nnzj ) = nnzj
       userdata%integer( loc_indfun ) = indfun
       userdata%integer( loc_indvar ) = indvar
       userdata%integer( loc_cjac ) = cjac
       userdata%integer( loc_nnzchp ) = nnzchp
       userdata%integer( loc_chpind ) = ihsind
       userdata%integer( loc_chpptr ) = ihsptr

! Determine if there is a constant in the linear constraints.
! Adjust the bounds if necessary

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

! Make X feasible with respect to bounds

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
         alpha_min = max( alpha_min, 1.001_wp )

         CALL CUTEST_cfn( cutest_status, n, m, X, f, C )
         IF ( cutest_status /= 0 ) GO TO 930
         CALL CUTEST_cfn( cutest_status, n, m, alpha_min * X, f2, C2 )
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

! Evaluate the Jacobian and Hessian and get sparsity pattern.

       IF ( no_hess .AND. no_jac ) THEN
       ELSE IF ( no_jac ) THEN
         lh = nnzh
!        CALL CUTEST_csh( cutest_status, nlp%n, m, nlp%X, - Y,                 &
!                     nnzh, lh, userdata%real( h + 1 : h + nnzh ),             &
!                     userdata%integer( irnh + 1 : irnh + nnzh ),              &
!                     userdata%integer( icnh + 1 : icnh + nnzh ) )
         CALL CUTEST_cshp( cutest_status, nlp%n, nnzh, lh,                     &
                           userdata%integer( irnh + 1 : irnh + nnzh ),         &
                           userdata%integer( icnh + 1 : icnh + nnzh ) )
       ELSE IF ( no_hess ) THEN
         lcjac = nnzj
!        CALL CUTEST_csgr( cutest_status, nlp%n, m, nlp%X, - Y, .FALSE.,       &
!                     nnzj, lcjac, userdata%real( cjac + 1 : cjac + nnzj ),    &
!                     userdata%integer( indvar + 1 : indvar + nnzj ),          &
!                     userdata%integer( indfun + 1 : indfun + nnzj ) )
         CALL CUTEST_csgrp( cutest_status, nlp%n, nnzj, lcjac,                 &
                            userdata%integer( indvar + 1 : indvar + nnzj ),    &
                            userdata%integer( indfun + 1 : indfun + nnzj ) )
       ELSE
         lcjac = nnzj
         lh = nnzh
!        CALL CUTEST_csgrsh( cutest_status, nlp%n, m, nlp%X, - Y, .FALSE.,     &
!                     nnzj, lcjac, userdata%real( cjac + 1 : cjac + nnzj ),    &
!                     userdata%integer( indvar + 1 : indvar + nnzj ),          &
!                     userdata%integer( indfun + 1 : indfun + nnzj ),          &
!                     nnzh, lh, userdata%real( h + 1 : h + nnzh ),             &
!                     userdata%integer( irnh + 1 : irnh + nnzh ),              &
!                     userdata%integer( icnh + 1 : icnh + nnzh ) )
         CALL CUTEST_csgrshp( cutest_status, nlp%n, nnzj, lcjac,               &
                              userdata%integer( indvar + 1 : indvar + nnzj ),  &
                              userdata%integer( indfun + 1 : indfun + nnzj ),  &
                              nnzh, lh,                                        &
                              userdata%integer( irnh + 1 : irnh + nnzh ),      &
                              userdata%integer( icnh + 1 : icnh + nnzh ) )
       END IF
       IF ( cutest_status /= 0 ) GO TO 930

!  Evaluate the matrix of constraint Hessian-vector products to get its
!  sparsity pattern

       IF (  hess_prods ) THEN
         CALL SPACE_resize_array( nnzchp, nlp%P%val, inform%status,            &
                                  inform%alloc_status )
         IF ( inform%status /= 0 ) then
           inform%bad_alloc = 'nlp%P%val' ; GO TO 910 ; END IF

!        CALL CUTEST_cchprods( cutest_status, nlp%n, m, .FALSE., nlp%X,        &
!                              nlp%X, nnzchp, nlp%P%val( : nnzchp ),           &
!                              userdata%integer( ihsind + 1 : ihsind + nnzchp),&
!                              userdata%integer( ihsptr + 1 : ihsptr + m + 1 ) )
         CALL CUTEST_cchprodsp( cutest_status, m, nnzchp,                      &
                                userdata%integer( ihsind + 1 :ihsind + nnzchp),&
                                userdata%integer( ihsptr + 1 :ihsptr + m + 1 ) )
       END IF

! get the number of nonzeros in the linear constraints and Jacobian constraints
! only

       nlp%J%ne = 0
       nlp%A%ne = 0
       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) == 0 ) THEN
           ! Relax....objective gradient component.
         ELSEIF ( userdata%integer( indfun + l ) <= nlp%m_a ) THEN
           nlp%A%ne = nlp%A%ne + 1
         ELSE
           nlp%J%ne = nlp%J%ne + 1
         END IF
       END DO

!  Deallocate arrays no longer needed.

       array_name = 'cutest_functions : Y'
       CALL SPACE_dealloc_array( Y,                                            &
          inform%status, inform%alloc_status, array_name = array_name,         &
          bad_alloc = inform%bad_alloc, out = control%error )
       IF ( inform%status /= 0 ) THEN
           inform%bad_alloc = 'Y' ; GO TO 920 ; END IF

!  Allocate arrays that are now of correct length.

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

!  Untangle J: separate the gradient terms from the linear constraints
!              and the general constraints in the Jacobian

       nlp%A%n = n ; nlp%A%m = nlp%m_a ; nlp%A%ne = 0
       nlp%J%n = n ; nlp%J%m = m - nlp%m_a ; nlp%J%ne = 0

       DO l = 1, nnzj
         IF ( userdata%integer( indfun + l ) == 0 ) THEN
            ! Relax....objective gradient component.
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

!  Define the storage type for J

       CALL SMT_put( nlp%A%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

       CALL SMT_put( nlp%J%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

!  ----------------------------------
!  -- The problem is unconstrained --
!  ----------------------------------

     ELSE

!  Set up the correct data structures for subsequent computations

       CALL CUTEST_usetup( cutest_status, control%input, control%error,        &
                           control%io_buffer, n, nlp%X, nlp%X_l, nlp%X_u )
       IF ( cutest_status /= 0 ) GO TO 930

!  Obtain the names of the problem and its variables

       CALL CUTEST_unames( cutest_status, n, nlp%pname, nlp%VNAMES )
       IF ( cutest_status /= 0 ) GO TO 930

! Set up sparsity structure for H. ( Assume co-ordinate storage )

!  Determine how many nonzeros are required to store the Hessian matrix
!  when the matrix is stored as a sparse matrix in "co-ordinate" format
!  (only the lower triangular part is stored).

       IF ( no_hess ) THEN
         nnzh = 0
       ELSE
         CALL CUTEST_udimsh( cutest_status, nnzh )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  Set starting addresses for workspace array partitions

       irnh = loc_h
       icnh = irnh + nnzh
       iend = icnh + nnzh
       h = 0
       rend = MAX( h + nnzh, n )

! Allocate space to hold scalars/arrays needed for subsequent calls

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

! Record workspace partitions in userdata%integer

       userdata%integer( loc_m ) = m
       userdata%integer( loc_n ) = n
       userdata%integer( loc_m_a ) = nlp%m_a
       userdata%integer( loc_nnzh ) = nnzh
       userdata%integer( loc_irnh ) = irnh
       userdata%integer( loc_icnh ) = icnh
       userdata%integer( loc_h ) = h

! Evaluate the Hessian and get sparsity pattern

       IF ( .NOT. no_hess ) THEN
         lh = nnzh
!        CALL CUTEST_ush( cutest_status, nlp%n, nlp%X, nnzh, lh,               &
!                  userdata%real( h + 1 : h + nnzh ),                          &
!                  userdata%integer( irnh + 1 : irnh + nnzh ),                 &
!                  userdata%integer( icnh + 1 : icnh + nnzh ) )
         CALL CUTEST_ushp( cutest_status, nlp%n, nnzh, lh,                     &
                           userdata%integer( irnh + 1 : irnh + nnzh ),         &
                           userdata%integer( icnh + 1 : icnh + nnzh ) )
         IF ( cutest_status /= 0 ) GO TO 930
       END IF

!  Define the storage type for the null J and A

       nlp%J%ne = 0
       nlp%A%ne = 0

       CALL SMT_put( nlp%A%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

       CALL SMT_put( nlp%J%type, 'COORDINATE', status )
       IF ( status /= 0 ) GO TO 990

! Allocate zero-length arrays to prevent errors

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

       CALL SPACE_resize_array( 0, nlp%Ax, inform%status,                      &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%Ax' ; GO TO 910
       END IF

       CALL SPACE_resize_array( 0, nlp%Y_a, inform%status,                     &
                                inform%alloc_status )
       IF ( inform%status /= 0 ) THEN
          inform%bad_alloc = 'nlp%Ax' ; GO TO 910
       END IF

     END IF

! Define reduced costs

     nlp%Z = zero

! Define the storage type for H

     CALL SMT_put( nlp%H%type, 'COORDINATE', status )
     IF ( status /= 0 ) GO TO 990

! Set the sparsity pattern for H

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

!  Ensure that the lower triangle of H is stored

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

! Set the space and sparsity pattern for P

     IF ( hess_prods ) THEN
       nnzchp = userdata%integer( ihsptr + m + 1 ) - 1

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

       nlp%P%row( : nnzchp ) = userdata%integer( ihsind + 1 : ihsind + nnzchp )
       nlp%P%ptr( : m + 1 ) = userdata%integer( ihsptr + 1 : ihsptr + m + 1 )
     END IF

     RETURN

! Error format statements

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

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), INTENT( OUT ) :: f
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! local variables

     INTEGER :: m, n

!  Extract scalar addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )

     IF ( m > 0 ) THEN
       CALL CUTEST_cfn( status, n, m, X, f, userdata%real( : m ) )
     ELSE
       CALL CUTEST_ufn( status, n, X, f )
     END IF

     RETURN

     END SUBROUTINE CUTEst_eval_F

!-*-*-*-*-*-*-*-   C U T E S T _ e v a l _ C   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_C( status, X, userdata, C )

!  Evaluate the constraint functions C(X)

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: C
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! local variables

     INTEGER :: m, m_a, n
     REAL ( KIND = wp ) :: f

!  Extract scalar addresses

     m   = userdata%integer( loc_m )
     n   = userdata%integer( loc_n )
     m_a = userdata%integer( loc_m_a )

!    CALL CUTEST_cfn( status, n, m, X, f, userdata%real( : m ) )
     CALL CUTEST_cfn( status, n, m, X, f, userdata%real )
     IF ( status == 0 ) C = userdata%real( m_a + 1 : m )

     RETURN

     END SUBROUTINE CUTEst_eval_C

!-*-*-*-*-*-*-*-   C U T E S T _ e v a l _ F C   S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_FC( status, X, userdata, f, C )

!  Evaluate the objective function f(X) and constraint functions C(X)

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), OPTIONAL, INTENT( OUT ) :: f
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: C
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! local variables

     INTEGER :: m, m_a, n
     REAL ( KIND = wp ) :: f_dummy

!  Extract scalar addresses

     m   = userdata%integer( loc_m )
     m_a = userdata%integer( loc_m_a )
     n   = userdata%integer( loc_n )

!    CALL CUTEST_cfn( status, n, m, X, f_dummy, userdata%real( : m ) )
     CALL CUTEST_cfn( status, n, m, X, f_dummy, userdata%real )
     IF ( status == 0 ) THEN
       IF ( PRESENT( f ) ) f = f_dummy
       IF ( PRESENT( C ) ) C = userdata%real( m_a + 1 : m )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_FC

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ G   S U B R O U T I N E  -*-*-*-*-*-*-

     SUBROUTINE CUTEst_eval_G( status, X, userdata, G )

!  Evaluate the gradient of the objective function G(X)

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: G
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER :: m, n
     REAL ( KIND = wp ) :: f_dummy
!    INTEGER :: m, n, nnzj, indfun, indvar, cjac, lcjac, l
!    REAL ( KIND = wp ), DIMENSION( 1 ) :: Y_dummy = zero

!  Extract scalar and array addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )

     IF ( m > 0 ) THEN
       CALL CUTEST_cofg( status, n, X, f_dummy, G, .TRUE. )
       IF ( status /= 0 ) RETURN

!      nnzj = userdata%integer( loc_nnzj )
!      indfun = userdata%integer( loc_indfun )
!      indvar = userdata%integer( loc_indvar )
!      cjac = userdata%integer( loc_cjac )
!      lcjac = nnzj
!      CALL CUTEST_csgr( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,       &
!                 userdata%real( cjac + 1 : cjac + nnzj ),                     &
!                 userdata%integer( indvar + 1 : indvar + nnzj ),              &
!                 userdata%integer( indfun + 1 : indfun + nnzj ) )
!      IF ( status /= 0 ) RETURN
! Untangle A: separate the gradient terms from the constraint Jacobian
!      G( : n ) = zero
!      DO l = 1, nnzj
!         IF ( userdata%integer( indfun + l ) == 0 ) THEN
!            G( userdata%integer( indvar + l ) ) = userdata%real( cjac + l )
!         END IF
!      END DO
     ELSE
       CALL CUTEST_ugr( status, n, X, G )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_G

!-*-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ J   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_J( status, X, userdata, J_val )

!  Evaluate the values of the constraint Jacobian Jval(X) for the nonzeros
!  corresponding to the sparse coordinate format set in CUTEst_initialize

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: J_val
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER :: m, m_a, n, nnzj, indfun, indvar, cjac, Jne, lcjac, l
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: Y_dummy

!  Extract scalar and array addresses

     m      = userdata%integer( loc_m )

     IF ( m > 0 ) THEN
       m_a    = userdata%integer( loc_m_a )
       n      = userdata%integer( loc_n )
       nnzj   = userdata%integer( loc_nnzj )
       indfun = userdata%integer( loc_indfun )
       indvar = userdata%integer( loc_indvar )
       cjac   = userdata%integer( loc_cjac )

       lcjac = nnzj ; Y_dummy = zero
       CALL CUTEST_csgr( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,       &
                         userdata%real( cjac + 1 : cjac + nnzj ),              &
                         userdata%integer( indvar + 1 : indvar + nnzj ),       &
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

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), OPTIONAL, INTENT( OUT ) :: G, J_val
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER :: m, m_a, n, nnzj, indfun, indvar, cjac, Jne, lcjac, l
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: Y_dummy

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
       CALL CUTEST_csgr( status, n, m, X, Y_dummy, .FALSE., nnzj, lcjac,       &
                         userdata%real( cjac + 1 : cjac + nnzj ),              &
                         userdata%integer( indvar + 1 : indvar + nnzj ),       &
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
       CALL CUTEST_ugr( status, n, X, G )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_GJ

!-*-*-*-*-*-*-*-  C U T E S T _ e v a l _ H    S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_H( status, X, userdata, H_val )

!  Evaluate the values of the Herssian of the objective function H_val(X)
!  for the nonzeros corresponding to the sparse coordinate format set in
!  CUTEst_initialize.

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER :: n, nnzh, irnh, icnh, lh

!  Extract scalar and array addresses

     n = userdata%integer( loc_n )
     nnzh = userdata%integer( loc_nnzh )
     irnh = userdata%integer( loc_irnh )
     icnh = userdata%integer( loc_icnh )

! Evaluate the Hessian

     lh = nnzh
     CALL CUTEST_ush( status, n, X, nnzh, lh, H_val,                           &
                      userdata%integer( irnh + 1 : irnh + nnzh ),              &
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

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_f

! Local variables

     INTEGER :: m, m_a, n, nnzh, irnh, icnh, lh
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: Y_full
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
       CALL CUTEST_cshc( status, n, m, X, Y_full, nnzh, lh, H_val,             &
                         userdata%integer( irnh + 1 : irnh + nnzh ),           &
                         userdata%integer( icnh + 1 : icnh + nnzh ) )
     ELSE
       CALL CUTEST_csh( status, n, m, X, Y_full, nnzh, lh, H_val,              &
                        userdata%integer( irnh + 1 : irnh + nnzh ),            &
                        userdata%integer( icnh + 1 : icnh + nnzh ) )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_HL

!-*-*-*-*-*-*-  C U T E S T _ e v a l _ H L C   S U B R O U T I N E  -*-*-*-*-*-

     SUBROUTINE CUTEst_eval_HLC( status, X, Y, userdata, H_val )

!  Evaluate the values of the Hessian of the constraint Lagrangian function,
!  sum c_i y_i for the nonzeros corresponding to the sparse coordinate format
!  set in CUTEst_initialize

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: H_val
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

! Local variables

     INTEGER :: m, m_a, n, nnzh, irnh, icnh, lh
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: Y_full

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
     CALL CUTEST_cshc( status,  n, m, X, Y_full, nnzh, lh, H_val,              &
                       userdata%integer( irnh + 1 : irnh + nnzh ),             &
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

     INTEGER, INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j

! Local variables

     INTEGER :: m, m_a, n
     REAL( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: full_V
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
         CALL CUTEST_cjprod( status, n, m, .FALSE., transpose, X, full_V, m,   &
                      userdata%real( : n ), n )
       ELSE
         CALL CUTEST_cjprod( status, n, m, .FALSE., transpose, X, V, n,        &
                      userdata%real( : m ), m )
       END IF
     ELSE
       IF ( transpose ) THEN
         CALL CUTEST_cjprod( status, n, m, .TRUE., transpose,                  &
                      userdata%real( : n ), full_V, m, userdata%real( : n ), n )
       ELSE
         CALL CUTEST_cjprod( status, n, m, .TRUE., transpose,                  &
                      userdata%real( : n ), V, n, userdata%real( : m ), m )
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

     INTEGER, INTENT( IN ) :: nnz_v
     INTEGER, INTENT( OUT ) :: nnz_u
     INTEGER, INTENT( OUT ) :: status
     LOGICAL, INTENT( IN ) :: transpose
     INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_j

! Local variables

     INTEGER :: m, n
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
         CALL CUTEST_csjprod( status, n, m, .FALSE., transpose, X,             &
                              nnz_v, INDEX_nz_v, V, m, nnz_u, INDEX_nz_u, U, n )
       ELSE
         CALL CUTEST_csjprod( status, n, m, .FALSE., transpose, X,             &
                              nnz_v, INDEX_nz_v, V, n, nnz_u, INDEX_nz_u, U, m )
       END IF
     ELSE
       IF ( transpose ) THEN
         CALL CUTEST_csjprod( status, n, m, .TRUE., transpose, X,              &
                              nnz_v, INDEX_nz_v, V, m, nnz_u, INDEX_nz_u, U, n )
       ELSE
         CALL CUTEST_csjprod( status, n, m, .TRUE., transpose, X,              &
                              nnz_v, INDEX_nz_v, V, n, nnz_u, INDEX_nz_u, U, m )
       END IF
     END IF

     RETURN

     END SUBROUTINE CUTEst_eval_SJPROD

!-*-*-*-*-  C U T E S T _ e v a l _ H P R O D    S U B R O U T I N E  -*-*-*-*-

     SUBROUTINE CUTEst_eval_HPROD( status, X, userdata, U, V, got_h )

!  Compute the product U = U + H(X) * V involving the Hessian of the objective
!  H(X). If got_h is present, the Hessian is as recorded at the last point at
!  which it was evaluated.

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: n
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     n = userdata%integer( loc_n )

     IF ( got_h_value ) THEN
       CALL CUTEST_uhprod( status, n, .TRUE., X, V, userdata%real( : n ) )
     ELSE
       CALL CUTEST_uhprod( status, n, .FALSE., X, V, userdata%real( : n ) )
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

     INTEGER, INTENT( IN ) :: nnz_v
     INTEGER, INTENT( OUT ) :: nnz_u
     INTEGER, INTENT( OUT ) :: status
     INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: n
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     n = userdata%integer( loc_n )

     IF ( got_h_value ) THEN
       CALL CUTEST_ushprod( status, n, .TRUE., X,                              &
                            nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     ELSE
       CALL CUTEST_ushprod( status, n, .FALSE., X,                             &
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

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: m, m_a, n
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
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
         CALL CUTEST_chcprod( status, n, m, .TRUE., X, - full_Y, V,            &
                              userdata%real( : n ) )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_chcprod( status, n, m, .FALSE., X, - full_Y, V,           &
                              userdata%real( : n ) )
       END IF
     ELSE
       IF ( got_h_value ) THEN
         CALL CUTEST_chprod( status, n, m, .TRUE., X, - full_Y, V,             &
                             userdata%real( : n ) )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_chprod( status, n, m, .FALSE., X, - full_Y, V,            &
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

     INTEGER, INTENT( IN ) :: nnz_v
     INTEGER, INTENT( OUT ) :: nnz_u
     INTEGER, INTENT( OUT ) :: status
     INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: no_f
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: m, m_a, n
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
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
         CALL CUTEST_cshcprod( status, n, m, .TRUE., X, - full_Y,              &
                               nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_cshcprod( status, n, m, .FALSE., X, - full_Y,             &
                               nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       END IF
     ELSE
       IF ( got_h_value ) THEN
         CALL CUTEST_cshprod( status, n, m, .TRUE., X, - full_Y,               &
                              nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
       ELSE
         full_Y( 1 : m_a ) = zero
         full_Y( m_a + 1 : m  ) = Y
         CALL CUTEST_cshprod( status, n, m, .FALSE., X, - full_Y,              &
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

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: m, m_a, n
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
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
       CALL CUTEST_chcprod( status, n, m, .TRUE., X, full_Y, V,                &
                            userdata%real( : n ) )
     ELSE
       full_Y( 1 : m_a ) = zero
       full_Y( m_a + 1 : m  ) = Y
       CALL CUTEST_chcprod( status, n, m, .FALSE., X, full_Y, V,               &
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

     INTEGER, INTENT( IN ) :: nnz_v
     INTEGER, INTENT( OUT ) :: nnz_u
     INTEGER, INTENT( OUT ) :: status
     INTEGER, DIMENSION( : ), INTENT( IN ) :: INDEX_nz_v
     INTEGER, DIMENSION( : ), INTENT( OUT ) :: INDEX_nz_u
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X, Y
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( OUT ) :: U
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: m, m_a, n
     REAL ( KIND = wp ), DIMENSION( userdata%integer( loc_m ) ) :: full_Y
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
       CALL CUTEST_cshcprod( status, n, m, .TRUE., X, - full_Y,                &
                             nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     ELSE
       full_Y( 1 : m_a ) = zero
       full_Y( m_a + 1 : m  ) = Y
       CALL CUTEST_cshcprod( status, n, m, .FALSE., X, - full_Y,               &
                             nnz_v, INDEX_nz_v, V, nnz_u, INDEX_nz_u, U )
     END IF
     RETURN

     END SUBROUTINE CUTEst_eval_SHLCPROD

!-*-*-*-  C U T E S T _ e v a l _ H C P R O D S    S U B R O U T I N E  -*-*-*-

     SUBROUTINE CUTEst_eval_HCPRODS( status, X, V, userdata, P_val, got_h )

!  Compute the products P = H_i(X) * V involving the Hessians of the
!  constraints. If got_h is PRESENT and TRUE, the Hessians are as
!  recorded at the last point at which they were evaluated.

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: X
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( IN ) :: V
     REAL ( KIND = wp ), DIMENSION( : ), INTENT( INOUT ) :: P_val
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     LOGICAL, OPTIONAL, INTENT( IN ) :: got_h

! Local variables

     INTEGER :: m, n, ihsind, ihsptr, nnzchp
     LOGICAL :: got_h_value

     IF ( PRESENT( got_h ) ) THEN
       got_h_value = got_h
     ELSE
       got_h_value = .FALSE.
     END IF

!  Extract scalar and array addresses

     m = userdata%integer( loc_m )
     n = userdata%integer( loc_n )
     nnzchp = userdata%integer( loc_nnzchp )
     ihsind = userdata%integer( loc_chpind )
     ihsptr = userdata%integer( loc_chpptr )

!write(6,*) ' nnzchp', nnzchp, userdata%integer( loc_nnzchp ), loc_nnzchp
!write(6,*) ' s ', V
     CALL CUTEST_cchprods( status, n, m, got_h_value, X, V, nnzchp, P_val,     &
                           userdata%integer( ihsind + 1 : ihsind + nnzchp ),   &
                           userdata%integer( ihsptr + 1 : ihsptr + m + 1 ) )
     RETURN

     END SUBROUTINE CUTEst_eval_HCPRODS

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

!  Dummy arguments

     TYPE ( NLPT_problem_type ), INTENT( INOUT ) :: nlp
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata
     TYPE ( CUTEst_FUNCTIONS_inform_type ), INTENT( OUT ) :: inform

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

     INTEGER, INTENT( OUT ) :: status
     REAL ( KIND = wp ) :: time

!  enable recording of timings for CUTEst calls

     CALL CUTEST_timings( status, 'start', time )
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

     INTEGER, INTENT( OUT ) :: status
     CHARACTER ( LEN = * ), INTENT( IN ) :: name
     REAL ( KIND = wp ), INTENT( out ) :: time
     TYPE ( NLPT_userdata_type ), INTENT( INOUT ) :: userdata

!  local variables

     INTEGER :: m
     REAL ( KIND = wp ) :: part_time
     CHARACTER ( LEN = LEN( name ) ) :: name_lower_case

!  extract scalar addresses

     m = userdata%integer( loc_m )

!  convert the name to lower case

     name_lower_case = name
     CALL STRING_lower_word( name_lower_case )

!  record required time

     status = 0 ; time = 0.0_wp

     SELECT CASE ( TRIM( name_lower_case ) )
     CASE ( 'cutest_eval_f' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings( status, 'cutest_cfn', time )
       ELSE
         CALL CUTEST_timings( status, 'cutest_ufn', time )
       END IF
     CASE ( 'cutest_eval_fc' )
       CALL CUTEST_timings( status, 'cutest_cfn', time )
     CASE ( 'cutest_eval_c' )
       CALL CUTEST_timings( status, 'cutest_cfn', time )
     CASE ( 'cutest_eval_g' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings( status, 'cutest_cofg', time )
       ELSE
         CALL CUTEST_timings( status, 'cutest_ugr', time )
       END IF
     CASE ( 'cutest_eval_gj' )
       IF ( m > 0 ) THEN
         CALL CUTEST_timings( status, 'cutest_csgr', time )
       ELSE
         CALL CUTEST_timings( status, 'cutest_ugr', time )
       END IF
     CASE ( 'cutest_eval_j' )
       IF ( m > 0 ) CALL CUTEST_timings( status, 'cutest_csgr', time )
     CASE ( 'cutest_eval_h' )
       CALL CUTEST_timings( status, 'cutest_ush', time )
     CASE ( 'cutest_eval_hprod' )
       CALL CUTEST_timings( status, 'cutest_uhprod', time )
     CASE ( 'cutest_eval_shprod' )
       CALL CUTEST_timings( status, 'cutest_ushprod', time )
     CASE ( 'cutest_eval_jprod' )
       CALL CUTEST_timings( status, 'cutest_cjprod', time )
     CASE ( 'cutest_eval_sjprod' )
       CALL CUTEST_timings( status, 'cutest_csjprod', time )
     CASE ( 'cutest_eval_hl' )
       CALL CUTEST_timings( status, 'cutest_csh', time )
       CALL CUTEST_timings( status, 'cutest_cshc', part_time )
       time = time + part_time
     CASE ( 'cutest_eval_hlc' )
       CALL CUTEST_timings( status, 'cutest_cshc', time )
     CASE ( 'cutest_eval_hlprod' )
       CALL CUTEST_timings( status, 'cutest_chprod', time )
       CALL CUTEST_timings( status, 'cutest_chcprod', part_time )
       time = time + part_time
     CASE ( 'cutest_eval_shlprod' )
       CALL CUTEST_timings( status, 'cutest_cshprod', time )
       CALL CUTEST_timings( status, 'cutest_cshcprod', part_time )
       time = time + part_time
     CASE ( 'cutest_eval_hlcprod' )
       CALL CUTEST_timings( status, 'cutest_chcprod', time )
     CASE ( 'cutest_eval_shlcprod' )
       CALL CUTEST_timings( status, 'cutest_cshcprod', time )
     CASE ( 'cutest_eval_hcprods' )
       CALL CUTEST_timings( status, 'cutest_cchprods', time )
     CASE DEFAULT
       status = GALAHAD_unavailable_option
     END SELECT

     RETURN

!  End of subroutine CUTEst_timing

     END SUBROUTINE CUTEst_timing

   END MODULE GALAHAD_CUTEST_FUNCTIONS_double

