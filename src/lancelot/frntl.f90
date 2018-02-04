! THIS VERSION: GALAHAD 2.6 - 23/06/2013 AT 13:00 GMT.

!-*-*-*-*-*-*-*-  L A N C E L O T  -B-  FRNTL  M O D U L E  *-*-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 2nd 1995

   MODULE LANCELOT_FRNTL_double

     USE GALAHAD_SMT_double
     USE GALAHAD_SILS_double
     USE GALAHAD_SCU_double, ONLY : SCU_matrix_type, SCU_data_type,            &
       SCU_info_type, SCU_restart_m_eq_0, SCU_solve, SCU_append
     USE LANCELOT_ASMBL_double
     USE LANCELOT_MDCHL_double
   
     IMPLICIT NONE
   
     PRIVATE
     PUBLIC :: FRNTL_get_search_direction

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     INTEGER, PARAMETER :: liwmin = 1, lwmin = 1
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: four = 4.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

   CONTAINS

!-*-*  L A N C E L O T  -B-   FRNTL_get_search_direction   M O D U L E  *-*-*

     SUBROUTINE FRNTL_get_search_direction(                                    &
        n , ng, nel   , ntotel, nnza  , maxsel, nvargp, buffer, INTVAR,        &
        IELVAR, nvrels, INTREP, IELING, ISTADG, ISTAEV, A     , ICNA  ,        &
        ISTADA, GUVALS, lnguvl, HUVALS, lnhuvl, ISTADH, GXEQX , GVALS2,        &
        GVALS3, IFREE , nfree , GMODEL, P , XT, BND   , fmodel, GSCALE,        &
        ESCALE, X0    , boundx, nobnds, dxsqr , radius, gstop , number,        &
        next  , modchl, RANGE , nsemib, ratio , iprint, error , out   ,        &
        status, alloc_status, bad_alloc , ITYPEE, DIAG  , OFFDIA, IVUSE ,      &
        RHS, RHS2, P2, ISTAGV, ISVGRP, lirnh, ljcnh, lh,                       &
        ROW_start, POS_in_H, USED, FILLED, lrowst, lpos, lused, lfilled,       &
        IW_asmbl, W_ws, W_el, W_in, H_el, H_in,                                &
        matrix, SILS_data, SILS_cntl, SILS_infoa, SILS_infof, SILS_infos,      &
        SCU_matrix, SCU_data, SCU_info, SA, skipg, KNDOFG )

!  Use a factorization of H, the assembled Hessian matrix of a group
!  partially separable function, restricted to the IFREE variables,
!  to obtain the following search directions p:

!  If H is positive definite,
!     solve H * p = GMODEL,

!  If H is indefinite,
!     find a direction p such that

!      p( transpose ) * H * p < 0  AND  p( transpose ) * GMODEL > 0

!  If H is singular but H * p = GMODEL is consistent
!     solve H * p = GMODEL,

!  If H is singular but H * p = GMODEL is inconsistent
!     find a descent direction P such that

!     H * p = 0 and p( transpose ) * GMODEL > 0

!    Fortran 77 version:  M. Lescrenier

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER :: n , ng, maxsel, nvrels, ntotel, lnguvl, nvargp, status,        &
                lnhuvl, nfree , number, iprint, error, out, nnza, nel, buffer
     INTEGER, INTENT( IN ) :: nsemib
     INTEGER, INTENT( OUT ) :: alloc_status
     REAL ( KIND = wp ) fmodel, gstop, dxsqr, radius
     REAL ( KIND = wp ), INTENT( OUT ) :: ratio
     LOGICAL next, modchl, boundx, nobnds, skipg
     CHARACTER ( LEN = 80 ), INTENT( OUT ) :: bad_alloc
     INTEGER, DIMENSION( n       ) :: IFREE
     INTEGER, DIMENSION( ng  + 1 ) :: ISTADA, ISTADG
     INTEGER, DIMENSION( nel + 1 ) :: INTVAR, ISTAEV, ISTADH
     INTEGER, DIMENSION( nvrels  ) :: IELVAR
     INTEGER, DIMENSION( ntotel  ) :: IELING
     INTEGER, DIMENSION( nnza    ) :: ICNA
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ITYPEE
     REAL ( KIND = wp ), DIMENSION( n ) :: GMODEL, P, X0, XT
     REAL ( KIND = wp ), DIMENSION( ng ) :: GVALS2, GVALS3, GSCALE
     REAL ( KIND = wp ), DIMENSION( : , : ) :: BND
     REAL ( KIND = wp ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), DIMENSION( nnza ) :: A
     REAL ( KIND = wp ), DIMENSION( lnguvl ) :: GUVALS
     REAL ( KIND = wp ), DIMENSION( lnhuvl ) :: HUVALS
     LOGICAL, DIMENSION( nel ) :: INTREP
     LOGICAL, DIMENSION( ng  ) :: GXEQX
     TYPE ( SCU_matrix_type ), INTENT( INOUT ) :: SCU_matrix
     TYPE ( SCU_data_type ), INTENT( INOUT ) :: SCU_data
     TYPE ( SCU_info_type ), INTENT( INOUT ) :: SCU_info
     TYPE ( SMT_type ), INTENT( INOUT ) :: matrix
     TYPE ( SILS_factors ), INTENT( INOUT ) :: SILS_data
     TYPE ( SILS_control ), INTENT( INOUT ) :: SILS_cntl
     TYPE ( SILS_ainfo ), INTENT( INOUT ) :: SILS_infoa
     TYPE ( SILS_finfo ), INTENT( INOUT ) :: SILS_infof
     TYPE ( SILS_sinfo ), INTENT( INOUT ) :: SILS_infos
     TYPE ( ASMBL_save_type ), INTENT( INOUT ) :: SA

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!---------------------------------------------------------------
!   D u m m y   A r g u m e n t s   f o r   W o r k s p a c e 
!--------------------------------------------------------------

     INTEGER, ALLOCATABLE, DIMENSION( : ) :: IVUSE
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : ) :: RHS, RHS2, P2, DIAG
     REAL ( KIND = wp ), ALLOCATABLE, DIMENSION( : , : ) :: OFFDIA
     
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV

     INTEGER, INTENT( INOUT ) :: lirnh, ljcnh, lh, lrowst, lpos, lused, lfilled
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: ROW_start
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: POS_in_H
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: USED
     INTEGER, ALLOCATABLE, DIMENSION( : ) :: FILLED
     INTEGER, INTENT( OUT ), DIMENSION( : ) :: IW_asmbl
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_ws
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_in

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, nnzh, max_sc
     INTEGER :: nfixed, scu_status, nsemiw
     REAL ( KIND = wp ) :: dotprd, pertur, alpha , sgrad , sths
     REAL ( KIND = wp ) :: pi, hp, ptp   , dxtp  , steptr, big
     REAL ( KIND = wp ) :: onepep, amodel, atemp , gnrmsq, epsmch
     LOGICAL :: pronel, prnter, consis, negate, reallocate

!-----------------------------------------------
!   A l l o c a t a b l e   A r r a y s
!-----------------------------------------------

     big = HUGE( one ) ;  epsmch = EPSILON( one )
     pronel = iprint == 2 ; prnter = iprint >= 5
     IF ( pronel ) WRITE( out, 2120 ) gstop
     onepep = one + ten * epsmch
     nfixed = 0
     max_sc = SCU_matrix%m_max

!  Assemble the Hessian restricted to the IFREE variables

     nsemiw = nsemib
     CALL ASMBL_assemble_hessian(                                              &
         n, ng, nel, ntotel, nvrels, nnza, maxsel, nvargp, nfree, IFREE,       &
         ISTADH, ICNA, ISTADA, INTVAR, IELVAR, IELING, ISTADG,                 &
         ISTAEV, ISTAGV, ISVGRP, A, GUVALS, lnguvl, HUVALS, lnhuvl,            &
         GVALS2, GVALS3, GSCALE, ESCALE, GXEQX, ITYPEE, INTREP, RANGE,         &
         iprint, error, out, .FALSE., .TRUE., .FALSE.,                         &
         nsemiw, status, alloc_status, bad_alloc,                              &
         lirnh, ljcnh, lh, matrix%row, matrix%col, matrix%val,                 &
         ROW_start, POS_in_H, USED, FILLED, lrowst, lpos, lused, lfilled,      &
         IW_asmbl, W_ws, W_el, W_in, H_el, H_in, skipg,                        &
         nnzh = nnzh, KNDOFG = KNDOFG )                                      
     IF ( status /= 0 ) RETURN

!  Choose initial values for the control parameters

     IF ( iprint >= 1000 ) THEN
       SILS_cntl%lp = 6 ; SILS_cntl%mp = 6
       SILS_cntl%wp = 6 ; SILS_cntl%sp = 6
       SILS_cntl%ldiag = 1
     ELSE
       SILS_cntl%lp = 0 ; SILS_cntl%mp = 0
       SILS_cntl%wp = 0 ; SILS_cntl%sp = 0
       SILS_cntl%ldiag = 0
     END IF

!  Choose pivots for Gaussian elimination

     matrix%n = nfree
     matrix%ne = nnzh
     CALL SILS_analyse( matrix, SILS_data, SILS_cntl, SILS_infoa )

!  Allocate arrays for MA27B or MCFA, MA27C, MISC_negcur or MISC_dscdir

     IF ( modchl ) THEN

!  Allocate further workspace

       reallocate = .TRUE.
       IF ( ALLOCATED( DIAG ) ) THEN
          IF ( SIZE( DIAG ) < nfree ) THEN ; DEALLOCATE( DIAG )
          ELSE ; reallocate = .FALSE.
          END IF
       END IF
       IF ( reallocate ) THEN 
          ALLOCATE( DIAG( nfree ), STAT = alloc_status )
          IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'DIAG' ; GO TO 980
          END IF
       END IF
     END IF

!  Factorize the restricted Hessian

     IF ( modchl ) SILS_cntl%pivoting = 4
     CALL SILS_factorize( matrix, SILS_data, SILS_cntl, SILS_infof )
     pertur = SILS_infof%maxchange

!  Test that the factorization succeeded

     IF ( SILS_infof%flag < 0 ) RETURN

!  Record the relative fill-in

     IF ( nnzh > 0 ) THEN
       ratio = DBLE( FLOAT( SILS_infof%nrlbdu ) ) / DBLE( FLOAT( nnzh ) )     
     ELSE
       ratio = one
     END IF

!  Allocate further workspace

     reallocate = .TRUE.
     IF ( ALLOCATED( RHS ) ) THEN
       IF ( SIZE( RHS ) < nfree ) THEN
         DEALLOCATE( RHS ) ; ELSE ; reallocate = .FALSE. ; END IF
     END IF
     IF ( reallocate ) THEN 
       ALLOCATE( RHS( nfree ), STAT = alloc_status )
       IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'RHS' ; GO TO 980 ; END IF
     END IF

     SCU_matrix%n = nfree
     SCU_matrix%m = nfixed
     CALL SCU_restart_m_eq_0( SCU_data, SCU_info )
     
     reallocate = .TRUE.
     IF ( ALLOCATED( RHS2 ) ) THEN
        IF ( SIZE( RHS2 ) < nfree + max_sc ) THEN ; DEALLOCATE( RHS2 )
        ELSE ; reallocate = .FALSE.
        END IF
     END IF
     IF ( reallocate ) THEN 
        ALLOCATE( RHS2( nfree + max_sc ), STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'RHS2' ; GO TO 980
        END IF
     END IF
     
     reallocate = .TRUE.
     IF ( ALLOCATED( P2 ) ) THEN
        IF ( SIZE( P2 ) <  nfree + max_sc ) THEN ; DEALLOCATE( P2 )
        ELSE ; reallocate = .FALSE.
        END IF
     END IF
     IF ( reallocate ) THEN 
        ALLOCATE( P2(  nfree + max_sc ), STAT = alloc_status )
        IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'P2' ; GO TO 980
        END IF
     END IF
     
     IF ( .NOT. modchl ) THEN
       reallocate = .TRUE.
       IF ( ALLOCATED( OFFDIA ) ) THEN
         IF ( SIZE( OFFDIA, 1 ) /= 2 .OR.                                &
              SIZE( OFFDIA, 2 ) < nfree ) THEN ; DEALLOCATE( OFFDIA )
         ELSE ; reallocate = .FALSE. ; END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( OFFDIA( 2, nfree ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'OFFDIA'; GO TO 980
         END IF
       END IF

       reallocate = .TRUE.
       IF ( ALLOCATED( IVUSE ) ) THEN
          IF ( SIZE( IVUSE ) < nfree ) THEN
            DEALLOCATE( IVUSE ) ; ELSE ; reallocate = .FALSE.
          END IF
       END IF
       IF ( reallocate ) THEN 
         ALLOCATE( IVUSE( nfree ), STAT = alloc_status )
         IF ( alloc_status /= 0 ) THEN ; bad_alloc = 'IVUSE' ; GO TO 980 ; END IF
       END IF

     END IF

     IF ( modchl ) THEN
       status = 1
     ELSE
       status = MDCHL_block_type( nfree, SILS_infof%rank, SILS_data,           &
                                  IVUSE( : nfree ), OFFDIA )
     END IF

!  Define the model of the gradient restricted to the free variables in RHS

     RHS( : nfree ) = - GMODEL( IFREE( : nfree ) )
     IF ( status == 1 ) THEN

!  The restricted Hessian is positive definite -
!  solve the linear system H * s = - GMODEL. s is output in GMODEL.

       CALL SILS_solve( matrix, SILS_data, RHS, SILS_cntl, SILS_infos )
       negate = .FALSE.
     END IF
     IF ( status == 2 ) THEN

!  The restricted Hessian is not positive definite. Compute a direction of
!  negative curvature

       CALL MDCHL_get_donc( nfree, SILS_infof%rank, SILS_data,                 &
                            IVUSE( : nfree ), OFFDIA, number, next, RHS, sths, &
                            SILS_cntl, alloc_status )
       IF ( alloc_status /= 0 ) THEN
         bad_alloc = 'SILS_data%w'; GO TO 980 ; END IF

!  Check if the dot product of this direction times GMODEL is > 0

!      dotprd = - DOT_PRODUCT( GMODEL( IFREE( : nfree ) ),                     &
!                              RHS( : nfree ) )
       dotprd = zero
       DO i = 1, nfree
          dotprd = dotprd - GMODEL( IFREE( i ) ) * RHS( i )
       END DO
       negate = dotprd <= zero
     END IF
     IF ( status == 3 ) THEN
       CALL MDCHL_get_singular_direction(                                      &
                 nfree, SILS_infof%rank, SILS_data, IVUSE( : nfree ), OFFDIA,  &
                 RHS( : nfree ), consis, SILS_cntl, alloc_status )
       IF ( alloc_status /= 0 ) THEN
         bad_alloc = 'SILS_data%w'; GO TO 980 ; END IF
       negate = .FALSE.
       IF ( .NOT. consis ) THEN
         status = 4
         sths = zero
       END IF
     END IF

!  Construct the solution P from RHS

     IF ( negate ) THEN
       DO j = 1, nfree
         P( IFREE( j ) ) = - RHS( j )
       END DO
     ELSE
       DO j = 1, nfree
         P( IFREE( j ) ) = RHS( j )
       END DO
     END IF

!  Calculate the slope of the quadratic model from the Cauchy point

!    sgrad = DOT_PRODUCT( P     ( IFREE( : nfree ) ),                      &
!                         GMODEL( IFREE( : nfree ) ) )
     sgrad = zero
     DO i = 1, nfree
       sgrad = sgrad + P( IFREE( i ) ) * GMODEL( IFREE( i ) )
     END DO

!  Obtain the curvature

     IF ( status == 1 .OR. status == 3 ) sths = - sgrad

!  Modify the curvature if the Hessian has been perturbed

     IF ( pertur > zero ) THEN
        status = 5
!       sths = sths - DOT_PRODUCT( DIAG( : nfree ),                            &
!                                  RHS ( : nfree ) ** 2 )
        DO i = 1, nfree
           sths = sths - DIAG( i ) * RHS ( i ) ** 2
        END DO
     END IF
 100 CONTINUE

!  Compute the steplength to the minimizer of the model along the
!  current search direction

     IF ( status == 1 .OR. status == 3 .OR. status == 5 ) THEN
       IF ( status == 5 ) THEN
         IF ( sths > zero ) THEN
           alpha = - sgrad / sths
         ELSE
           alpha = big
         END IF
       ELSE
         alpha = one
       END IF
     ELSE
       alpha = big
     END IF

!  If required, print details of the current step

     IF ( iprint >= 20 ) THEN
       WRITE( out, 2060 )
       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) WRITE( out, 2070 )                                      &
           i, XT( i ), GMODEL( i ), P( i ), BND( i, 1 ), BND( i, 2 )
       END DO
     END IF

!  Find the largest feasible step in the direction P from XT

     amodel = alpha
     IF ( .NOT. nobnds ) THEN
       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) THEN
           IF ( ABS( P( i ) ) > epsmch ) THEN
             IF ( P( i ) > zero ) THEN
               alpha = MIN( alpha, ( BND( i, 2 ) - XT( i ) ) / P( i ) )
             ELSE
               alpha = MIN( alpha, ( BND( i, 1 ) - XT( i ) ) / P( i ) )
             END IF
           END IF
         END IF
       END DO
     END IF
     
     IF ( boundx ) THEN
       dxtp = zero ; ptp = zero
!DIR$ IVDEP
       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) THEN
           pi = P( i )
           dxtp = dxtp + pi * ( XT( i ) - X0( i ) )
           ptp = ptp + pi * pi
         END IF
       END DO

!  Now compute the distance to the spherical boundary, steptr, and
!  find the smaller of this and the distance to the boundary of the
!  feasible box, STEPMX

       steptr = ( SQRT( dxtp * dxtp - ptp * ( dxsqr - radius ** 2 ) ) -  &
                  dxtp ) / ptp
       alpha = MIN( steptr, alpha )
     END IF

!  Update the model function value

     fmodel = fmodel + alpha * ( sgrad + alpha * half * sths )

!  If required, print the model value, slope and the step taken

     IF ( prnter ) WRITE( out, 2040 ) amodel, alpha, fmodel, sgrad, sths

!  Check to see if the boundary is encountered

     gnrmsq = zero
     IF ( alpha < amodel ) THEN

!  A free variable has encountered a bound. Make a second pass to determine
!  which variable and compute the size of the model gradient at the new point

       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) THEN
           IF ( ABS( P( i ) ) > epsmch ) THEN
             atemp = big
             IF ( .NOT. nobnds ) THEN
               IF ( P( i ) > zero ) THEN
                 atemp = ( BND( i, 2 ) - XT( i ) ) / P( i )
               ELSE
                 atemp = ( BND( i, 1 ) - XT( i ) ) / P( i )
               END IF
             END IF

!  Variable I encounters its bound

             IF ( atemp <= alpha * onepep ) THEN
               IF ( prnter ) WRITE( out, 2090 ) i
               IFREE( j ) = - i
             ELSE

!  Update the gradient

                IF ( status == 1 .OR. status >= 4 ) hp = - GMODEL( i )
                IF ( status == 4 ) hp = zero
                IF ( status == 5 ) hp = hp - DIAG( j ) * P( i )
                GMODEL( i ) = GMODEL( i ) + alpha * hp
                gnrmsq = gnrmsq + GMODEL( i ) ** 2
              END IF
             XT( i ) = XT( i ) + alpha * P( i )
           END IF
         END IF
       END DO
     ELSE

!  Step to the new point

       DO j = 1, nfree
         i = IFREE( j )
         IF ( i > 0 ) THEN
           IF ( ABS( P( i ) ) > epsmch ) XT( i ) = XT( i ) + alpha * P( i )
         END IF
       END DO
     END IF
     IF ( boundx ) THEN
       IF ( alpha == steptr ) THEN
         IF ( prnter ) WRITE( out, 2050 ) gnrmsq, gstop ; GO TO 500 ; END IF
     END IF

!  If the model gradient is sufficiently small, exit

     IF ( prnter ) WRITE( out, 2050 ) gnrmsq, gstop
     IF ( pronel ) WRITE( out, 2130 ) nfixed, fmodel, gnrmsq
     IF ( gnrmsq <= gstop ) GO TO 500

!  Continue the minimization in the restricted subspace

     IF ( .NOT. modchl ) GO TO 500

!  Exit if there are no bounds

     IF ( alpha < amodel .AND. nobnds ) GO TO 500

!  Set up further workspace addresses to partition the unused portions of W
!  and IW. Calculate the largest number of variables which can be fixed

!  Determine which variables are to be fixed

     DO j = 1, nfree
       i = IFREE( j )
       IF ( i < 0 ) THEN

!  If more than max_sc variables have been fixed, return

         IF ( nfixed >= max_sc ) GO TO 500

!  Update the factorization of the Schur complement to allow for the removal of
!  the J-th row and column of the original Hessian - this removal is effected
!  by appending the J-th row and column of the identity matrix to the Hessian

         SCU_matrix%BD_val( nfixed + 1 ) = one
         SCU_matrix%BD_row( nfixed + 1 ) = j
         SCU_matrix%BD_col_start( nfixed + 2 ) = nfixed + 2
         scu_status = 1
 210     CONTINUE

!  Call SCU_append to update the Schur-complement

         CALL SCU_append( SCU_matrix, SCU_data, RHS, scu_status, SCU_info )

!  SCU_append requires additional information. Compute the solution to
!  the equation H * S = RHS, returning the solution S in RHS

         IF ( scu_status > 0 ) THEN
            CALL SILS_solve( matrix, SILS_data, RHS, SILS_cntl, SILS_infos )
            GO TO 210
         END IF
         IF ( scu_status < 0 ) THEN
            IF ( prnter ) WRITE( out, 2100 ) scu_status
            GO TO 500
         END IF
         nfixed = nfixed + 1
       END IF
     END DO

!  Define the model of the gradient restricted to the free variables in RHS

     DO j = 1, nfree
       i = IFREE( j )
       IF ( i > 0 ) THEN
         RHS2( j ) = - GMODEL( i )
       ELSE
         RHS2( j ) = zero
       END IF
     END DO
     RHS2( nfree + 1 : nfree + nfixed ) = zero

!  Solve the new linear system H * P2 = RHS2

     scu_status = 1
 310 CONTINUE

!  Call SCU_solve to solve the system

     CALL SCU_solve( SCU_matrix, SCU_data, RHS2, P2, RHS, scu_status )
     IF ( scu_status > 0 ) THEN

!  SCU_solve requires additional information. Compute the solution to
!  the equation H * S = RHS, returning the solution S in RHS

       CALL SILS_solve( matrix, SILS_data, RHS, SILS_cntl, SILS_infos )
       GO TO 310
     END IF

!  Store the corrections to the free variables

     DO j = 1, nfree
       i = IFREE( j )
       IF ( i < 0 ) IFREE( j ) = 0
       IF ( i > 0 ) P( i ) = P2( j )
     END DO

!  Calculate the slope of the quadratic model from the Cauchy point

     sgrad = zero
     DO j = 1, nfree
       i = IFREE( j )
       IF ( i > 0 ) sgrad = sgrad + P( i ) * GMODEL( i )
     END DO

!  Obtain the curvature

     IF ( status == 1 .OR. status == 3 .OR. status == 5 ) sths = - sgrad

!  Modify the curvature if the Hessian has been perturbed

     IF ( pertur > zero ) THEN
       DO j = 1, nfree
         IF ( IFREE( j ) > 0 ) sths = sths - DIAG( j ) * P2( j ) ** 2
       END DO
     END IF
     GO TO 100

!  Successful return

 500 CONTINUE
     RETURN

!  Unsuccessful returns

 980 CONTINUE
     WRITE( error, 2990 ) alloc_status, TRIM( bad_alloc )
     RETURN

!  Non-executable statements

 2040  FORMAT( /,' Model step and actual step = ', 2ES12.4, /,                 &
                 ' fmodel, sgrad, sths = ', 3ES12.4 )
 2050  FORMAT( /,' Model gradient ** 2 ', ES12.4, ' gstop ', ES12.4 )
 2060  FORMAT( /,'    i      XT          G           P           BL',          &
                '          BU' )
 2070  FORMAT( I5, 5ES12.4 )
 2090  FORMAT( ' Variable ',I6,' has encountered a bound in FRNTL ' )
 2100  FORMAT( ' ** Message from -FRNTL-', /,                                  &
               '    value of status after SCU_append = ', I3 )
 2120  FORMAT( /,'    ** FRNTL entered ** nfixed     MODEL   ',                &
                 '   Gradient gstop = ', ES12.4 )
 2130  FORMAT( 24X, I7, 3ES12.4 )
 2990  FORMAT( ' ** Message from -FRNTL_get_search_direction-', /,             &
               ' Allocation error (status = ', I0, ') for ', A )

!  End of subroutine FRNTL_get_search_direction

     END SUBROUTINE FRNTL_get_search_direction

!  End of module LANCELOT_FRNTL

   END MODULE LANCELOT_FRNTL_double

