! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  OTHERS  M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 26th 1995

   MODULE LANCELOT_OTHERS_double

     IMPLICIT NONE
     
     INTERFACE OTHERS_gauss_elim
       MODULE PROCEDURE OTHERS_gauss_elim_1d, OTHERS_gauss_elim_2d
     END INTERFACE

!  Parameters

     PRIVATE
     PUBLIC :: OTHERS_secant, OTHERS_secant_flexible,                          &
               OTHERS_scaleh, OTHERS_scaleh_flexible,                          &
               OTHERS_which_variables_changed, OTHERS_fdgrad_save_type,        &
               OTHERS_fdgrad, OTHERS_fdgrad_flexible,                          &
               OTHERS_gauss_elim, OTHERS_gauss_solve, OTHERS_symmh,            &
               OTHERS_time, OTHERS_iter, OTHERS_time6, OTHERS_iter5

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp

!  ========================================
!  The OTHERS_fdgrad_save_type derived type
!  ========================================

     TYPE :: OTHERS_fdgrad_save_type
       LOGICAL :: backwd
     END TYPE OTHERS_fdgrad_save_type

!  =============================
!  The OTHERS_random derived type
!  =============================

!    TYPE :: OTHERS_random_save_type
!      REAL ( KIND = wp ) :: gl, gr
!    END TYPE OTHERS_random_save_type

   CONTAINS

!-*-*-  L A N C E L O T  -B-  OTHERS_secant   S U B R O U T I N E   -*-*

     SUBROUTINE OTHERS_secant( n, nel, lfuval, nvrels, ntotin, IELVAR, ISTAEV, &
                               INTVAR, ITYPEE, INTREP, ISTADH, FUVALS, ICALCF, &
                               ncalcf, S , Y , second_derivatives    , iskip , &
                               idebug, iout  , W_el, W_in, S_in, RANGE )

!  Computes the secant update to the second derivative matrix
!  for each element function

!  If second_derivatives =  1, the B.F.G.S. update is used
!  If second_derivatives =  2, the D.F.P. update is used
!  If second_derivatives =  3, the P.S.B. update is used
!  If second_derivatives >= 4, the S.R.1 update is used

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval, nvrels, ntotin
     INTEGER, INTENT( IN ) :: ncalcf, second_derivatives, idebug, iout
     INTEGER, INTENT( INOUT ) :: iskip
     INTEGER, INTENT( IN ), DIMENSION( nvrels ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ICALCF
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotin ) :: Y
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: S_in

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                          lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

     REAL ( KIND = wp ), PARAMETER :: skipr1 = ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: skipbd = ten ** ( - 8 )

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ii, ll, ipos, ipos1
     INTEGER :: j, jj, k, kk, nin, nvarel
     REAL ( KIND = wp ) :: sts, wts, wtw, yts, yty, yj, sj, wj
     LOGICAL :: intrnl, prnter
     
     prnter = idebug >= 4 .AND. iout > 0

!  Consider the IEL-th element

     DO i = 1, ncalcf
       iel = ICALCF( i )
       ll = ISTAEV( iel ) - 1
       kk = ISTADH( iel ) - 1
       jj = INTVAR( iel ) - 1 - nel
       nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
       nin = INTVAR( iel + 1 ) - INTVAR( iel )
       intrnl = INTREP( iel )

!  If the element has an internal representation, transform S.

       IF ( intrnl ) THEN
         W_el( : nvarel ) = S( IELVAR( ll + 1 : ll + nvarel ) )
         CALL RANGE ( iel, .FALSE., W_el, S_in, nvarel, nin,                   &
                      ITYPEE( iel ), nvarel, nin )
       END IF

!  Compute scalars for the Broyden-Fletcher-Goldfarb-Shanno
!  and Davidon-Fletcher-Powell updates.

       IF ( second_derivatives <= 2 ) THEN
         IF ( intrnl ) THEN
!          yts = DOT_PRODUCT( Y( jj + 1 : jj + nin ), S_in( : nin ) )
!          yty = DOT_PRODUCT( Y( jj + 1 : jj + nin ), Y( jj + 1 : jj + nin ) )
           yts = zero ; yty = zero
           DO ii = 1, nin
             yts = yts + Y( jj + ii ) * S_in( ii )
             yty = yty + Y( jj + ii ) ** 2
           END DO
         ELSE
!          yts = DOT_PRODUCT( Y( jj + 1 : jj + nvarel ),                       &
!                             S( IELVAR( ll + 1 : ll + nvarel ) ) )
!          yty = DOT_PRODUCT( Y( jj + 1 : jj + nvarel ),                       &
!                             Y( jj + 1 : jj + nvarel ) )
           yts = zero ; yty = zero
           DO ii = 1, nvarel
             yts = yts + Y( jj + ii ) * S( IELVAR( ll + ii ) )
             yty = yty + Y( jj + ii ) ** 2
           END DO
         END IF
         IF ( yts <= skipbd * yty ) THEN
           IF ( second_derivatives == 1 ) THEN
             IF ( prnter ) WRITE(iout,                                         &
               "( /, ' BFGS update skipped in element ', I5,                   &
            &        ' Y(trans) S is negative' )" ) iel
           ELSE
             IF ( prnter ) WRITE(iout,                                         &
               "( /, ' DFP update skipped in element ', I5,                    &
            &        ' Y(trans) S is negative' )" ) iel 
           END IF
           iskip = iskip + 1
           CYCLE
         END IF
       END IF

!  Calculate the product of the element Hessian with the vector S and for
!  the DFP, PSB and SR1 updates, add the answer to Y. First initialize W

       IF ( second_derivatives == 1 ) THEN
         W_in( : nin ) = zero
       ELSE
         W_in( : nin ) = - Y( jj + 1 : jj + nin )
       END IF
       DO ipos = 1, nin
         IF ( intrnl ) THEN
           sj = S_in( ipos )
         ELSE
           sj = S( IELVAR( ll + ipos ) )
         END IF
         k = kk + ipos * ( ipos - 1 ) / 2

!  Form the product of the IPOS-th component with the element Hessian

         IF ( ipos > 0 ) THEN
           W_in( : ipos ) =                                                    &
             W_in( : ipos ) + sj * FUVALS( k + 1 : k + ipos )
           k = k + ipos
         END IF
         ipos1 = ipos + 1
         DO ii = ipos1, nin
           k = k + ii - 1
           W_in( ii ) = W_in( ii ) + sj * FUVALS( k )
         END DO
       END DO

!  Compute the inner product of this vector with S

       IF ( intrnl ) THEN
!        wts = DOT_PRODUCT( W_in( : nin ), S_in( : nin ) )
         wts = zero
         DO ii = 1, nin
           wts = wts + W_in( ii ) * S_in( ii )
         END DO
       ELSE
!        wts = DOT_PRODUCT( W_in( : nin ), S( IELVAR( ll + 1 : ll + nin ) ) )
         wts = zero
         DO ii = 1, nin
           wts = wts + W_in( ii ) * S( IELVAR( ll + ii ) )
         END DO
       END IF

!  Compute S(trans) S for all updates, except the Symmetric Rank One

       IF ( second_derivatives < 4 ) THEN
         IF ( intrnl ) THEN
!          sts = SUM( S_in( : nin ) ** 2 )
           sts = zero
           DO ii = 1, nin ; sts = sts + S_in( ii ) ** 2 ; END DO
         ELSE
!          sts = SUM( S( IELVAR( ll + 1 : ll + nin ) ) ** 2 )
           sts = zero
           DO ii = ll + 1, ll + nin
             sts = sts + S( IELVAR( ii ) ) ** 2
           END DO
         END IF

!  Skip the positive definite updates if the element Hessian is not
!  positive definite enough, due to rounding errors

         IF ( wts <= skipbd * sts .AND. second_derivatives /= 3 ) THEN
           IF ( second_derivatives == 1 ) THEN
             IF ( prnter ) WRITE( iout,                                        &
               "( /, ' BFGS update skipped in element ', I5,                   &
            &        ' S(trans) B S is negative')" ) iel
           ELSE
             IF ( prnter ) WRITE( iout,                                        &
               "( /, ' DFP update skipped in element ', I5,                    &
            &        ' S(trans) B S is negative' )" ) iel
           END IF
           iskip = iskip + 1
           CYCLE
         END IF
       END IF

!  Broyden Fletcher Goldfarb Shanno update

       IF ( second_derivatives == 1 ) THEN
         DO j = 1, nin
           yj = Y( jj + j ) / yts
           wj = W_in( j ) / wts
           FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )               &
             - wj * W_in( : j ) + yj * Y( jj + 1 : jj + j )
           kk = kk + j
         END DO
       END IF

!  Davidon Fletcher Powell update

       IF ( second_derivatives == 2 ) THEN
         DO j = 1, nin
           wj = Y( jj + j ) / yts
           yj = ( W_in( j ) - wj * wts ) / yts
           FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )               &
             - wj * W_in( : j ) - yj * Y( jj + 1 : jj + j )
           kk = j + kk
         END DO
       END IF

!  Powell symmetric Broyden update

       IF ( second_derivatives == 3 ) THEN

!  Update for an internal elemental representation

         IF ( sts /= zero ) THEN
           IF ( intrnl ) THEN
             DO j = 1, nin
               wj = - S_in( j ) / sts
               sj = ( - W_in( j ) - wj * wts ) / sts
               FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )           &
                 + wj * W_in( : j ) + sj * S_in( : j )
               kk = kk + j
             END DO
           ELSE

!  Update for an elemental representation

             DO j = 1, nin
               wj = - S( IELVAR( ll + j ) ) / sts
               sj = ( - W_in( j ) - wj * wts ) / sts
               FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )           &
                 + wj * W_in( : j ) + sj * S( IELVAR( ll + 1 : ll + j ) )
               kk = kk + j
             END DO
         
           END IF
         ELSE   
           IF ( prnter ) WRITE( iout,                                          &
             "( /, ' PSB update skipped in element ', I5,                      &
          &        ' S(trans) S is zero ')" ) iel
           iskip = iskip + 1
           CYCLE
         END IF
       END IF

!  Symmetric Rank One update

       IF ( second_derivatives >= 4 ) THEN
!        wtw = DOT_PRODUCT( W_in( : nin ), W_in( : nin ) )
         wtw = zero ; DO ii = 1, nin ; wtw = wtw + W_in( ii ) ** 2 ; END DO
         IF ( ABS( wts ) <= skipr1 * wtw ) THEN
           IF ( prnter ) WRITE( iout,                                          &
             "( /, ' SR1 update skipped in element ', I5,                      &
          &        ' W(trans) S is too small' )" ) iel
           iskip = iskip + 1
           CYCLE
         ELSE
           DO j = 1, nin
             wj = W_in( j ) / wts
             FUVALS( kk + 1 : kk + j ) =                                       &
               FUVALS( kk + 1 : kk + j ) - wj * W_in( : j )
             kk = j + kk
           END DO
         END IF
       END IF
     END DO

     RETURN

!  End of subroutine OTHERS_secant

     END SUBROUTINE OTHERS_secant

!-*-*-  L A N C E L O T  -B-  OTHERS_secant_flexible   S U B R O U T I N E  -*-*

     SUBROUTINE OTHERS_secant_flexible(                                        &
                               n, nel, lfuval, nvrels, ntotin, IELVAR, ISTAEV, &
                               INTVAR, ITYPEE, INTREP, ISTADH, FUVALS, ICALCF, &
                               ncalcf, S , Y , iskip , idebug, iout, W_el,     &
                               W_in, S_in, EL2DER, RANGE )

!  Computes the secant update to the second derivative matrix
!  for each element function

!  If EL2DER( i ) <= 0, the update is skipped for element i
!  If EL2DER( i ) =  1, the B.F.G.S. update is used for element i
!  If EL2DER( i ) =  2, the D.F.P. update is used for element  
!  If EL2DER( i ) =  3, the P.S.B. update is used for element i
!  If EL2DER( i ) >= 4, the S.R.1 update is used for element i

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval, nvrels, ntotin
     INTEGER, INTENT( IN ) :: ncalcf, idebug, iout
     INTEGER, INTENT( INOUT ) :: iskip
     INTEGER, INTENT( IN ), DIMENSION( nvrels ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ICALCF
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: EL2DER
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotin ) :: Y
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: S_in

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE ( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,      &
                          lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   P a r a m e t e r s
!-----------------------------------------------

     REAL ( KIND = wp ), PARAMETER :: skipr1 = ten ** ( - 8 )
     REAL ( KIND = wp ), PARAMETER :: skipbd = ten ** ( - 8 )

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ii, ll, ipos, ipos1
     INTEGER :: j, jj, k, kk, nin, nvarel, second_derivatives
     REAL ( KIND = wp ) :: sts, wts, wtw, yts, yty, yj, sj, wj
     LOGICAL :: intrnl, prnter
     
     prnter = idebug >= 4 .AND. iout > 0

!  Consider the IEL-th element

     DO i = 1, ncalcf
       iel = ICALCF( i )
       second_derivatives = EL2DER( iel )
       IF ( second_derivatives <= 0 ) CYCLE
       ll = ISTAEV( iel ) - 1
       kk = ISTADH( iel ) - 1
       jj = INTVAR( iel ) - 1 - nel
       nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
       nin = INTVAR( iel + 1 ) - INTVAR( iel )
       intrnl = INTREP( iel )

!  If the element has an internal representation, transform S.

       IF ( intrnl ) THEN
         W_el( : nvarel ) = S( IELVAR( ll + 1 : ll + nvarel ) )
         CALL RANGE ( iel, .FALSE., W_el, S_in, nvarel, nin,                   &
                      ITYPEE( iel ), nvarel, nin )
       END IF

!  Compute scalars for the Broyden-Fletcher-Goldfarb-Shanno
!  and Davidon-Fletcher-Powell updates.

       IF ( second_derivatives <= 2 ) THEN
         IF ( intrnl ) THEN
!          yts = DOT_PRODUCT( Y( jj + 1 : jj + nin ), S_in( : nin ) )
!          yty = DOT_PRODUCT( Y( jj + 1 : jj + nin ), Y( jj + 1 : jj + nin ) )
           yts = zero ; yty = zero
           DO ii = 1, nin
             yts = yts + Y( jj + ii ) * S_in( ii )
             yty = yty + Y( jj + ii ) ** 2
           END DO
         ELSE
!          yts = DOT_PRODUCT( Y( jj + 1 : jj + nvarel ),                       &
!                             S( IELVAR( ll + 1 : ll + nvarel ) ) )
!          yty = DOT_PRODUCT( Y( jj + 1 : jj + nvarel ),                       &
!                             Y( jj + 1 : jj + nvarel ) )
           yts = zero ; yty = zero
           DO ii = 1, nvarel
             yts = yts + Y( jj + ii ) * S( IELVAR( ll + ii ) )
             yty = yty + Y( jj + ii ) ** 2
           END DO
         END IF
         IF ( yts <= skipbd * yty ) THEN
           IF ( second_derivatives == 1 ) THEN
             IF ( prnter ) WRITE(iout,                                         &
               "( /, ' BFGS update skipped in element ', I5,                   &
            &        ' Y(trans) S is negative' )" ) iel
           ELSE
             IF ( prnter ) WRITE(iout,                                         &
               "( /, ' DFP update skipped in element ', I5,                    &
            &        ' Y(trans) S is negative' )" ) iel 
           END IF
           iskip = iskip + 1
           CYCLE
         END IF
       END IF

!  Calculate the product of the element Hessian with the vector S and for
!  the DFP, PSB and SR1 updates, add the answer to Y. First initialize W

       IF ( second_derivatives == 1 ) THEN
         W_in( : nin ) = zero
       ELSE
         W_in( : nin ) = - Y( jj + 1 : jj + nin )
       END IF
       DO ipos = 1, nin
         IF ( intrnl ) THEN
           sj = S_in( ipos )
         ELSE
           sj = S( IELVAR( ll + ipos ) )
         END IF
         k = kk + ipos * ( ipos - 1 ) / 2

!  Form the product of the IPOS-th component with the element Hessian

         IF ( ipos > 0 ) THEN
           W_in( : ipos ) =                                                    &
             W_in( : ipos ) + sj * FUVALS( k + 1 : k + ipos )
           k = k + ipos
         END IF
         ipos1 = ipos + 1
         DO ii = ipos1, nin
           k = k + ii - 1
           W_in( ii ) = W_in( ii ) + sj * FUVALS( k )
         END DO
       END DO

!  Compute the inner product of this vector with S

       IF ( intrnl ) THEN
!        wts = DOT_PRODUCT( W_in( : nin ), S_in( : nin ) )
         wts = zero
         DO ii = 1, nin
           wts = wts + W_in( ii ) * S_in( ii )
         END DO
       ELSE
!        wts = DOT_PRODUCT( W_in( : nin ), S( IELVAR( ll + 1 : ll + nin ) ) )
         wts = zero
         DO ii = 1, nin
           wts = wts + W_in( ii ) * S( IELVAR( ll + ii ) )
         END DO
       END IF

!  Compute S(trans) S for all updates, except the Symmetric Rank One

       IF ( second_derivatives < 4 ) THEN
         IF ( intrnl ) THEN
!          sts = SUM( S_in( : nin ) ** 2 )
           sts = zero
           DO ii = 1, nin ; sts = sts + S_in( ii ) ** 2 ; END DO
         ELSE
!          sts = SUM( S( IELVAR( ll + 1 : ll + nin ) ) ** 2 )
           sts = zero
           DO ii = ll + 1, ll + nin
             sts = sts + S( IELVAR( ii ) ) ** 2
           END DO
         END IF

!  Skip the positive definite updates if the element Hessian is not
!  positive definite enough, due to rounding errors

         IF ( wts <= skipbd * sts .AND. second_derivatives /= 3 ) THEN
           IF ( second_derivatives == 1 ) THEN
             IF ( prnter ) WRITE( iout,                                        &
               "( /, ' BFGS update skipped in element ', I5,                   &
            &        ' S(trans) B S is negative')" ) iel
           ELSE
             IF ( prnter ) WRITE( iout,                                        &
               "( /, ' DFP update skipped in element ', I5,                    &
            &        ' S(trans) B S is negative' )" ) iel
           END IF
           iskip = iskip + 1
           CYCLE
         END IF
       END IF

!  Broyden Fletcher Goldfarb Shanno update

       IF ( second_derivatives == 1 ) THEN
         DO j = 1, nin
           yj = Y( jj + j ) / yts
           wj = W_in( j ) / wts
           FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )               &
             - wj * W_in( : j ) + yj * Y( jj + 1 : jj + j )
           kk = kk + j
         END DO
       END IF

!  Davidon Fletcher Powell update

       IF ( second_derivatives == 2 ) THEN
         DO j = 1, nin
           wj = Y( jj + j ) / yts
           yj = ( W_in( j ) - wj * wts ) / yts
           FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )               &
             - wj * W_in( : j ) - yj * Y( jj + 1 : jj + j )
           kk = j + kk
         END DO
       END IF

!  Powell symmetric Broyden update

       IF ( second_derivatives == 3 ) THEN

!  Update for an internal elemental representation

         IF ( sts /= zero ) THEN
           IF ( intrnl ) THEN
             DO j = 1, nin
               wj = - S_in( j ) / sts
               sj = ( - W_in( j ) - wj * wts ) / sts
               FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )           &
                 + wj * W_in( : j ) + sj * S_in( : j )
               kk = kk + j
             END DO
           ELSE

!  Update for an elemental representation

             DO j = 1, nin
               wj = - S( IELVAR( ll + j ) ) / sts
               sj = ( - W_in( j ) - wj * wts ) / sts
               FUVALS( kk + 1 : kk + j ) = FUVALS( kk + 1 : kk + j )           &
                 + wj * W_in( : j ) + sj * S( IELVAR( ll + 1 : ll + j ) )
               kk = kk + j
             END DO
         
           END IF
         ELSE   
           IF ( prnter ) WRITE( iout,                                          &
             "( /, ' PSB update skipped in element ', I5,                      &
          &        ' S(trans) S is zero ')" ) iel
           iskip = iskip + 1
           CYCLE
         END IF
       END IF

!  Symmetric Rank One update

       IF ( second_derivatives >= 4 ) THEN
!        wtw = DOT_PRODUCT( W_in( : nin ), W_in( : nin ) )
         wtw = zero ; DO ii = 1, nin ; wtw = wtw + W_in( ii ) ** 2 ; END DO
         IF ( ABS( wts ) <= skipr1 * wtw ) THEN
           IF ( prnter ) WRITE( iout,                                          &
             "( /, ' SR1 update skipped in element ', I5,                      &
          &        ' W(trans) S is too small' )" ) iel
           iskip = iskip + 1
           CYCLE
         ELSE
           DO j = 1, nin
             wj = W_in( j ) / wts
             FUVALS( kk + 1 : kk + j ) =                                       &
               FUVALS( kk + 1 : kk + j ) - wj * W_in( : j )
             kk = j + kk
           END DO
         END IF
       END IF
     END DO

     RETURN

!  End of subroutine OTHERS_secant_flexible

     END SUBROUTINE OTHERS_secant_flexible

!-*-*-*-  L A N C E L O T  -B-  OTHERS_scaleh   S U B R O U T I N E   -*-*-*

     SUBROUTINE OTHERS_scaleh( inith , n, nel, lfuval, nvrels, ntotin,         &
                               ncalcf, ISTAEV, ISTADH, ICALCF,                 &
                               INTVAR, IELVAR, ITYPEE, INTREP, FUVALS,         &
                               S , Y , ISYMMD, W_el, S_in, RANGE )

!  Initialize the approximate second derivative matrix for each nonlinear
!  element as a (scaled) identity matrix, if inith is .TRUE., and scale
!  these matrices to satisfy the weak secant equation, if inith is .FALSE.

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval, nvrels, ntotin, ncalcf
     LOGICAL, INTENT( IN ) :: inith
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: ICALCF
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: ITYPEE
     INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotin ) :: Y
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISYMMD
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: S_in

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

     INTEGER :: i, iel, j, nin, k, l, lhuval, nel1, nvarel, lhxi
     REAL ( KIND = wp ) :: yts, sths, si

!  If a secant method is to be used, initialize the second
!  derivatives of each element as a scaled identity matrix

     nel1   = nel + 1
     lhxi   = INTVAR( nel1 ) - 1
     lhuval = ISTADH( nel + 1 ) - ISTADH( 1 )
     IF ( inith ) THEN

!  Set all values to zero

       FUVALS( lhxi + 1 : lhxi + lhuval ) = zero

!  Reset the diagonals to the one

       DO iel = 1, nel
         nin = INTVAR( iel + 1 ) - INTVAR( iel )
         FUVALS( ISTADH( iel ) + ISYMMD( : nin ) ) = one
       END DO
     ELSE

!  At the end of the first successful iteration, scale the initial
!  second derivative matrix for each element so as to satisfy
!  the weak secant condition of Shanno and Phua

       DO i = 1, ncalcf
         iel = ICALCF( i )
         nin = INTVAR( iel + 1 ) - INTVAR( iel )
         nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )

!  If the element has an internal representation, transform S
!  into its internal variables, S_in.

         IF ( INTREP( iel ) ) THEN
           W_el( : nvarel ) =                                                  &
             S( IELVAR( ISTAEV( iel ) : ISTAEV( iel + 1 ) - 1 ) )
           CALL RANGE( iel, .FALSE., W_el, S_in, nvarel, nin,                  &
                       ITYPEE( iel ), nvarel, nin )

!  Compute the scalars YTS = Y(TRANS) S and STHS =
!  S(TRANS) H S, remembering that H = I here

!          yts = SUM( S_in( : nin ) *                                         &
!            Y( INTVAR( iel ) - nel1 + 1 : INTVAR( iel ) - nel1 + nin ) )
!          sths = SUM( S_in( : nin ) ** 2 )
           yts = zero ; sths = zero ; k = INTVAR( iel ) - nel1
           DO j = 1, nin
             yts  = yts  + S_in( j ) * Y( k + j )
             sths = sths + S_in( j ) ** 2
           END DO                 
         ELSE
           yts = zero ; sths = zero
           k = INTVAR( iel ) - nel1 ; l = ISTAEV( iel ) - 1
           DO j = 1, nvarel
             si = S( IELVAR( l + j ) )
             yts = yts + Y( k + j ) * si
             sths = sths + si ** 2
           END DO
         END IF

!  Scale the element Hessians by the quantity YTS / STHS as suggested
!  by Shanno and Phua

         FUVALS( ISTADH( iel ) + ISYMMD( : nin ) ) = yts / sths
       END DO
     END IF

     RETURN

!  END OF SUBROUTINE OTHERS_scaleh

     END SUBROUTINE OTHERS_scaleh

!-*-*-  L A N C E L O T  -B-  OTHERS_scaleh_flexible  S U B R O U T I N E  -*-*-

     SUBROUTINE OTHERS_scaleh_flexible(                                        &
                               inith , n, nel, lfuval, nvrels, ntotin,         &
                               ncalcf, ISTAEV, ISTADH, ICALCF,                 &
                               INTVAR, IELVAR, ITYPEE, INTREP, FUVALS,         &
                               S , Y , ISYMMD, W_el, S_in, EL2DER, RANGE )

!  Initialize the approximate second derivative matrix for every nonlinear
!  element i for which EL2DER(i) > 0 as a (scaled) identity matrix, 
!  if inith is .TRUE., and scale
!  these matrices to satisfy the weak secant equation, if inith is .FALSE.

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval, nvrels, ntotin, ncalcf
     LOGICAL, INTENT( IN ) :: inith
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: ICALCF
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: ITYPEE
     INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: EL2DER
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ntotin ) :: Y
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISYMMD
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: S_in

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

     INTEGER :: i, iel, j, nin, k, l, nel1, nvarel
     REAL ( KIND = wp ) :: yts, sths, si

!  If a secant method is to be used, initialize the second
!  derivatives of each element as a scaled identity matrix

     nel1   = nel + 1
     IF ( inith ) THEN

!  Set all values to zero

!  Reset the diagonals to the one

       DO iel = 1, nel
         IF ( EL2DER( iel ) > 0 ) THEN
           nin = INTVAR( iel + 1 ) - INTVAR( iel )
           FUVALS( ISTADH( iel ) : ISTADH( iel + 1 ) - 1 ) = zero
           FUVALS( ISTADH( iel ) + ISYMMD( : nin ) ) = one
         END IF
       END DO
     ELSE

!  At the end of the first successful iteration, scale the initial
!  second derivative matrix for each element so as to satisfy
!  the weak secant condition of Shanno and Phua

       DO i = 1, ncalcf
         iel = ICALCF( i )
         IF ( EL2DER( iel ) <= 0 ) CYCLE
         nin = INTVAR( iel + 1 ) - INTVAR( iel )
         nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )

!  If the element has an internal representation, transform S
!  into its internal variables, S_in.

         IF ( INTREP( iel ) ) THEN
           W_el( : nvarel ) =                                                  &
             S( IELVAR( ISTAEV( iel ) : ISTAEV( iel + 1 ) - 1 ) )
           CALL RANGE( iel, .FALSE., W_el, S_in, nvarel, nin,                  &
                       ITYPEE( iel ), nvarel, nin )

!  Compute the scalars YTS = Y(TRANS) S and STHS =
!  S(TRANS) H S, remembering that H = I here

!          yts = SUM( S_in( : nin ) *                                         &
!            Y( INTVAR( iel ) - nel1 + 1 : INTVAR( iel ) - nel1 + nin ) )
!          sths = SUM( S_in( : nin ) ** 2 )
           yts = zero ; sths = zero ; k = INTVAR( iel ) - nel1
           DO j = 1, nin
             yts  = yts  + S_in( j ) * Y( k + j )
             sths = sths + S_in( j ) ** 2
           END DO                 
         ELSE
           yts = zero ; sths = zero
           k = INTVAR( iel ) - nel1 ; l = ISTAEV( iel ) - 1
           DO j = 1, nvarel
             si = S( IELVAR( l + j ) )
             yts = yts + Y( k + j ) * si
             sths = sths + si ** 2
           END DO
         END IF

!  Scale the element Hessians by the quantity YTS / STHS as suggested
!  by Shanno and Phua

         FUVALS( ISTADH( iel ) + ISYMMD( : nin ) ) = yts / sths
       END DO
     END IF

     RETURN

!  END OF SUBROUTINE OTHERS_scaleh_flexible

     END SUBROUTINE OTHERS_scaleh_flexible

!-*-*-  L A N C E L O T  -B-  OTHERS_fdgrad   S U B R O U T I N E   -*-*-*

     SUBROUTINE OTHERS_fdgrad(                                                 &
                       n, nel, lfuval, ntotel, nvrels, nsets, IELVAR,          &
                       ISTAEV, IELING, ICALCF, ncalcf, INTVAR,                 &
                       ntype , X , XT, FUVALS, centrl, igetfd, S,              &
                       ISVSET, ISET, INVSET, ISSWTR, ISSITR, ITYPER,           &
                       LIST_elements, LINK_elem_uses_var, WTRANS, ITRANS )

!  Obtain finite-difference estimates of the first derivatives of the
!  nonlinear element functions

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval, nvrels, ntotel, nsets
     INTEGER, INTENT( IN ) :: ntype
     INTEGER, INTENT( INOUT ) :: igetfd, ncalcf
     LOGICAL, INTENT( IN ) :: centrl
     INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( ntotel  ) :: IELING
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( INOUT ), DIMENSION( nel     ) :: ICALCF
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: XT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     TYPE ( OTHERS_fdgrad_save_type ), INTENT( INOUT ) :: S

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISSWTR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISSITR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISET
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVSET
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: INVSET
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ITYPER
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LIST_elements
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ITRANS
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LINK_elem_uses_var
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: WTRANS

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, j, ipt, k , l , itype , ninvar, lwfree
     INTEGER :: liwfre, iell  , ivar  , intv
     REAL ( KIND = wp ) :: diff, twodif

     IF ( igetfd > nsets ) THEN
        igetfd = - 1
        RETURN
     END IF

!  Calculate the difference intervals.

     IF ( centrl ) THEN
       diff = EPSILON( one ) ** 0.33333
       twodif = two * diff
     ELSE
       diff = EPSILON( one ) ** 0.5
     END IF

!  -------------------------------------------------------------
!  Compute the finite differences corresponding to the variables
!  from the IGETFD-th set
!  -------------------------------------------------------------

     IF ( igetfd > 0 ) THEN
       DO i = ISVSET( igetfd ), ISVSET( igetfd + 1 ) - 1
         ivar = ISET( i )

!  Loop over the elements which use variable IVAR.
!  The elements are obtained from a linked-list

         ipt = LINK_elem_uses_var( ivar )
         IF ( ipt >= 0 ) THEN
           iell = LIST_elements( ivar )
  410      CONTINUE
           iel = IELING( iell )

!  If the element has an internal representation, check that the
!  variable IVAR belongs in the "independence" set

           itype = ITYPER( iel )
           IF ( itype > 0 ) THEN
             liwfre = ISSITR( itype )
             ninvar = ITRANS( liwfre )
             DO j = 1, ninvar
               k = j - 1
               l = ITRANS( liwfre + ninvar + 1 + j ) - 1
               IF ( ivar == IELVAR( ISTAEV( iel ) + l ) ) GO TO 440
             END DO
             GO TO 470
           ELSE

!  Find which internal variable is used

             k = 0
             DO j = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1
               IF ( IELVAR( j ) == ivar ) GO TO 440
               k = k + 1
             END DO
           END IF
  440      CONTINUE

!  Form a central difference

           intv = INTVAR( iel )
           l = intv + k
           IF ( centrl ) THEN
             IF ( S%backwd ) THEN
               IF ( INVSET( iel ) == 2 ) THEN
                 FUVALS( l ) = ( FUVALS( l ) - FUVALS( iel ) ) / twodif
!
!  Ensure that derivatives for repeated variables are set to zero
!
                 IF ( itype == 0 ) THEN
                   k = k + 1
                   l = k
                   IF ( ISTAEV( iel ) + l <= ISTAEV( iel + 1 ) - 1 ) THEN
                     WHERE ( IELVAR( ISTAEV( iel ) + l :                       &
                             ISTAEV( iel + 1 ) - 1 ) == ivar )                 &
                       FUVALS( intv + k : intv + k + ISTAEV( iel + 1 )         &
                               - l - ISTAEV( iel ) - 1 ) = zero
                     k = ISTAEV( iel + 1 ) - l - ISTAEV( iel ) + k
                   END IF
                 END IF
                 INVSET( iel ) = 0
               END IF
             ELSE
               IF ( INVSET( iel ) == 1 ) THEN
                 FUVALS( l ) = FUVALS( iel )
                 INVSET( iel ) = 2
               END IF
             END IF

!  Form a forward difference

           ELSE
             IF ( INVSET( iel ) == 1 ) THEN
               FUVALS( l ) = ( FUVALS( iel ) - FUVALS( l ) ) / diff

!  Ensure that derivatives for repeated variables are set to zero

               IF ( itype == 0 ) THEN
                 k = k + 1
                 l = k
                 IF ( ISTAEV( iel ) + l <= ISTAEV( iel + 1 ) - 1 ) THEN
                   WHERE ( IELVAR( ISTAEV( iel ) + l :                         &
                           ISTAEV( iel + 1 ) - 1 ) == ivar )                   &
                     FUVALS( intv + k : intv + k + ISTAEV( iel + 1 ) -         &
                             ISTAEV( iel ) - l - 1 ) = zero
                   k = k + ISTAEV( iel + 1 ) - ISTAEV( iel ) - l
                 END IF
               END IF
               INVSET( iel ) = 0
             END IF
           END IF
  470      CONTINUE

!  See if further variables use variable IVAR

           IF ( ipt > 0 ) THEN
              iell = LIST_elements( ipt )
              ipt = LINK_elem_uses_var( ipt )
              GO TO 410
           END IF
         END IF

!  Reset the variables from the IGETFD-th set to their initial values

         IF ( centrl ) THEN
           IF ( S%backwd ) THEN
             XT( ivar ) = X( ivar )
           ELSE
             XT( ivar ) = X( ivar ) - diff
           END IF
         ELSE
           XT( ivar ) = X( ivar )
         END IF
       END DO
     ELSE

!  ---------------------------------------------------------------------
!  Initialise the point, XT,  at which the elements are to be computed
!  as the current point X
!  ---------------------------------------------------------------------

       IF ( centrl ) S%backwd = .TRUE.
       XT = X

!  Empty the list, INVSET, of elements which need to be re-evaluated.
!  Those marked - 1 need not be re-evaluated

       INVSET = - 1
       DO k = 1, ncalcf
         j = ICALCF( k )
         FUVALS( INTVAR( j ) : INTVAR( j + 1 ) - 1 ) = FUVALS( j )
         INVSET( j ) = 0
       END DO
     END IF

!  Compute the next set of finite difference intervals

     IF ( centrl ) THEN
       IF ( S%backwd ) THEN
         S%backwd = .FALSE.
         igetfd = igetfd + 1
       ELSE
         S%backwd = .TRUE.
       END IF
     ELSE
       igetfd = igetfd + 1
     END IF

!  If all the difference have been computed, prepare to return

     IF ( igetfd > nsets ) THEN

!  Run through the list of elements with internal variables,
!  transforming the differences by the matrix W(-Transpose)

       IF ( ntype > 0 ) THEN
         ncalcf = 0
         DO iel = 1, nel
           itype = ITYPER( iel )
           IF ( itype > 0 ) THEN
             liwfre = ISSITR( itype )
             lwfree = ISSWTR( itype )
             ninvar = ITRANS( liwfre )

!  Transform the differences

             CALL OTHERS_gauss_solve(                                          &
                 ninvar, ITRANS( liwfre + 2 : ), WTRANS( lwfree : ),           &
                 FUVALS( INTVAR( iel ) ) )
           END IF

!  Reset ICALCF to its original value

           IF ( INVSET( iel ) /= - 1 ) THEN
             ncalcf = ncalcf + 1
             ICALCF( ncalcf ) = iel
           END IF
         END DO
       END IF
       igetfd = - 1
       RETURN
     END IF

!  Prepare to return to obtain additional element function values.
!  Compute the difference intervals for the IGETFD-th set

     IF ( .NOT. ( centrl .AND. S%backwd ) ) THEN
       ncalcf = 0
       DO i = ISVSET( igetfd ), ISVSET( igetfd + 1 ) - 1
         ivar = ISET( i )
         XT( ivar ) = X( ivar ) + diff

!  Loop over the elements which use variable IVAR.
!  The elements are obtained from a linked-list

         ipt = LINK_elem_uses_var( ivar )
         IF ( ipt >= 0 ) THEN
           iell = LIST_elements( ivar )
  630      CONTINUE
           iel = IELING( iell )
           itype = ITYPER( iel )

!  Check that the variable belongs to the "indendence" set of an
!  element with an internal representation

           IF ( itype > 0 ) THEN
             liwfre = ISSITR( itype )
             ninvar = ITRANS( liwfre )
             DO j = 1, ninvar
               k = ITRANS( liwfre + ninvar + 1 + j ) - 1
               IF ( ivar == IELVAR( ISTAEV( iel ) + k ) ) GO TO 660
             END DO
             GO TO 670
           END IF

!  Flag the nonlinear elements which will need to be recalculated

  660      CONTINUE
           IF ( INVSET( iel ) == 0 ) THEN
             INVSET( iel ) = 1
             ncalcf = ncalcf + 1
             ICALCF( ncalcf ) = iel
           END IF
  670      CONTINUE
           IF ( ipt > 0 ) THEN
             iell = LIST_elements( ipt )
             ipt = LINK_elem_uses_var( ipt )
             GO TO 630
           END IF
         END IF
       END DO
     END IF

     RETURN

!  End of subroutine OTHERS_fdgrad

     END SUBROUTINE OTHERS_fdgrad

!-*-  L A N C E L O T  -B-  OTHERS_fdgrad_flexible   S U B R O U T I N E   -*-

     SUBROUTINE OTHERS_fdgrad_flexible(                                        &
                       n, nel, lfuval, ntotel, nvrels, nsets, IELVAR,          &
                       ISTAEV, IELING, ICALCF, ncalcf, INTVAR, ntype, X , XT,  &
                       FUVALS, centrl, igetfd, S, ISVSET, ISET, INVSET,        &
                       ISSWTR, ISSITR, ITYPER, LIST_elements,                  &
                       LINK_elem_uses_var, WTRANS, ITRANS, EL1DER )

!  Obtain finite-difference estimates of the first derivatives of the
!  nonlinear element functions i for which EL1DER( i ) > 0

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, nel, lfuval, nvrels, ntotel, nsets
     INTEGER, INTENT( IN ) :: ntype
     INTEGER, INTENT( INOUT ) :: igetfd, ncalcf
     LOGICAL, INTENT( IN ) :: centrl
     INTEGER, INTENT( IN ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( ntotel  ) :: IELING
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( INOUT ), DIMENSION( nel     ) :: ICALCF
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( nel     ) :: EL1DER
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: XT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( lfuval ) :: FUVALS
     TYPE ( OTHERS_fdgrad_save_type ), INTENT( INOUT ) :: S

     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISSWTR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISSITR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISET
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVSET
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: INVSET
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ITYPER
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LIST_elements
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ITRANS
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LINK_elem_uses_var
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( : ) :: WTRANS

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, j, ipt, k , l , itype , ninvar, lwfree
     INTEGER :: liwfre, iell  , ivar  , intv
     REAL ( KIND = wp ) :: diff, twodif

     IF ( igetfd > nsets ) THEN
        igetfd = - 1
        RETURN
     END IF

!  Calculate the difference intervals.

     IF ( centrl ) THEN
       diff = EPSILON( one ) ** 0.33333
       twodif = two * diff
     ELSE
       diff = EPSILON( one ) ** 0.5
     END IF

!  -------------------------------------------------------------
!  Compute the finite differences corresponding to the variables
!  from the IGETFD-th set
!  -------------------------------------------------------------

     IF ( igetfd > 0 ) THEN
       DO i = ISVSET( igetfd ), ISVSET( igetfd + 1 ) - 1
         ivar = ISET( i )

!  Loop over the elements which use variable IVAR.
!  The elements are obtained from a linked-list

         ipt = LINK_elem_uses_var( ivar )
         IF ( ipt >= 0 ) THEN
           iell = LIST_elements( ivar )
  410      CONTINUE
           iel = IELING( iell )

!  If the element has an internal representation, check that the
!  variable IVAR belongs in the "independence" set

           itype = ITYPER( iel )
           IF ( itype > 0 ) THEN
             liwfre = ISSITR( itype )
             ninvar = ITRANS( liwfre )
             DO j = 1, ninvar
               k = j - 1
               l = ITRANS( liwfre + ninvar + 1 + j ) - 1
               IF ( ivar == IELVAR( ISTAEV( iel ) + l ) ) GO TO 440
             END DO
             GO TO 470
           ELSE

!  Find which internal variable is used

             k = 0
             DO j = ISTAEV( iel ), ISTAEV( iel + 1 ) - 1
               IF ( IELVAR( j ) == ivar ) GO TO 440
               k = k + 1
             END DO
           END IF
  440      CONTINUE

!  Form a central difference

           intv = INTVAR( iel )
           l = intv + k
           IF ( centrl ) THEN
             IF ( S%backwd ) THEN
               IF ( INVSET( iel ) == 2 ) THEN
                 FUVALS( l ) = ( FUVALS( l ) - FUVALS( iel ) ) / twodif
!
!  Ensure that derivatives for repeated variables are set to zero
!
                 IF ( itype == 0 ) THEN
                   k = k + 1
                   l = k
                   IF ( ISTAEV( iel ) + l <= ISTAEV( iel + 1 ) - 1 ) THEN
                     WHERE ( IELVAR( ISTAEV( iel ) + l :                       &
                             ISTAEV( iel + 1 ) - 1 ) == ivar )                 &
                       FUVALS( intv + k : intv + k + ISTAEV( iel + 1 )         &
                               - l - ISTAEV( iel ) - 1 ) = zero
                     k = ISTAEV( iel + 1 ) - l - ISTAEV( iel ) + k
                   END IF
                 END IF
                 INVSET( iel ) = 0
               END IF
             ELSE
               IF ( INVSET( iel ) == 1 ) THEN
                 FUVALS( l ) = FUVALS( iel )
                 INVSET( iel ) = 2
               END IF
             END IF

!  Form a forward difference

           ELSE
             IF ( INVSET( iel ) == 1 ) THEN
               FUVALS( l ) = ( FUVALS( iel ) - FUVALS( l ) ) / diff

!  Ensure that derivatives for repeated variables are set to zero

               IF ( itype == 0 ) THEN
                 k = k + 1
                 l = k
                 IF ( ISTAEV( iel ) + l <= ISTAEV( iel + 1 ) - 1 ) THEN
                   WHERE ( IELVAR( ISTAEV( iel ) + l :                         &
                           ISTAEV( iel + 1 ) - 1 ) == ivar )                   &
                     FUVALS( intv + k : intv + k + ISTAEV( iel + 1 ) -         &
                             ISTAEV( iel ) - l - 1 ) = zero
                   k = k + ISTAEV( iel + 1 ) - ISTAEV( iel ) - l
                 END IF
               END IF
               INVSET( iel ) = 0
             END IF
           END IF
  470      CONTINUE

!  See if further variables use variable IVAR

           IF ( ipt > 0 ) THEN
              iell = LIST_elements( ipt )
              ipt = LINK_elem_uses_var( ipt )
              GO TO 410
           END IF
         END IF

!  Reset the variables from the IGETFD-th set to their initial values

         IF ( centrl ) THEN
           IF ( S%backwd ) THEN
             XT( ivar ) = X( ivar )
           ELSE
             XT( ivar ) = X( ivar ) - diff
           END IF
         ELSE
           XT( ivar ) = X( ivar )
         END IF
       END DO
     ELSE

!  ---------------------------------------------------------------------
!  Initialise the point, XT,  at which the elements are to be computed
!  as the current point X
!  ---------------------------------------------------------------------

       IF ( centrl ) S%backwd = .TRUE.
       XT = X

!  Empty the list, INVSET, of elements which need to be re-evaluated.
!  Those marked - 1 need not be re-evaluated

       INVSET = - 1
       DO k = 1, ncalcf
         j = ICALCF( k )
         IF ( EL1DER( j ) > 0 ) THEN
           FUVALS( INTVAR( j ) : INTVAR( j + 1 ) - 1 ) = FUVALS( j )
           INVSET( j ) = 0
         END IF
       END DO
     END IF

!  Compute the next set of finite difference intervals

     IF ( centrl ) THEN
       IF ( S%backwd ) THEN
         S%backwd = .FALSE.
         igetfd = igetfd + 1
       ELSE
         S%backwd = .TRUE.
       END IF
     ELSE
       S%backwd = .FALSE.
       igetfd = igetfd + 1
     END IF

!  If all the difference have been computed, prepare to return

     IF ( igetfd > nsets ) THEN

!  Run through the list of elements with internal variables,
!  transforming the differences by the matrix W(-Transpose)

       IF ( ntype > 0 ) THEN
         ncalcf = 0
         DO iel = 1, nel
           itype = ITYPER( iel )
           IF ( itype > 0 ) THEN
             liwfre = ISSITR( itype )
             lwfree = ISSWTR( itype )
             ninvar = ITRANS( liwfre )

!  Transform the differences

             CALL OTHERS_gauss_solve(                                          &
                 ninvar, ITRANS( liwfre + 2 : ), WTRANS( lwfree : ),           &
                 FUVALS( INTVAR( iel ) ) )
           END IF

!  Reset ICALCF to its original value

           IF ( INVSET( iel ) /= - 1 ) THEN
             ncalcf = ncalcf + 1
             ICALCF( ncalcf ) = iel
           END IF
         END DO
       END IF
       igetfd = - 1
       RETURN
     END IF

!  Prepare to return to obtain additional element function values.
!  Compute the difference intervals for the IGETFD-th set

     IF ( .NOT. ( centrl .AND. S%backwd ) ) THEN
       ncalcf = 0
       DO i = ISVSET( igetfd ), ISVSET( igetfd + 1 ) - 1
         ivar = ISET( i )
         XT( ivar ) = X( ivar ) + diff

!  Loop over the elements which use variable IVAR.
!  The elements are obtained from a linked-list

         ipt = LINK_elem_uses_var( ivar )
         IF ( ipt >= 0 ) THEN
           iell = LIST_elements( ivar )
  630      CONTINUE
           iel = IELING( iell )
           itype = ITYPER( iel )

!  Check that the variable belongs to the "indendence" set of an
!  element with an internal representation

           IF ( itype > 0 ) THEN
             liwfre = ISSITR( itype )
             ninvar = ITRANS( liwfre )
             DO j = 1, ninvar
               k = ITRANS( liwfre + ninvar + 1 + j ) - 1
               IF ( ivar == IELVAR( ISTAEV( iel ) + k ) ) GO TO 660
             END DO
             GO TO 670
           END IF

!  Flag the nonlinear elements which will need to be recalculated

  660      CONTINUE
           IF ( INVSET( iel ) == 0 ) THEN
             INVSET( iel ) = 1
             ncalcf = ncalcf + 1
             ICALCF( ncalcf ) = iel
           END IF
  670      CONTINUE
           IF ( ipt > 0 ) THEN
             iell = LIST_elements( ipt )
             ipt = LINK_elem_uses_var( ipt )
             GO TO 630
           END IF
         END IF
       END DO
     END IF

     RETURN

!  End of subroutine OTHERS_fdgrad_flexible

     END SUBROUTINE OTHERS_fdgrad_flexible

!-*-  L A N C E L O T  -B-  OTHERS_gauss_elim_1d   S U B R O U T I N E   -*-

     SUBROUTINE OTHERS_gauss_elim_1d( m, n, IPVT, JCOL, A )

!  Perform the first M steps of Gaussian Elimination with
!  complete pivoting on the M by N ( M <= N) matrix A

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, n
     INTEGER, INTENT( OUT ), DIMENSION( m ) :: IPVT
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: JCOL
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m * n ) :: A

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k, l1, l2, ipivot, jpivot
     REAL ( KIND = wp ) :: apivot, atemp
     
     DO j = 1, n
       JCOL( j ) = j
     END DO

!  Main loop

     DO k = 1, m

!  Compute the K-th pivot

       apivot = - one
       l2 = m * ( k - 1 ) ; l1 = l2
       DO j = k, n
         DO i = k, m
           IF ( ABS( A( l1 + i ) ) > apivot ) THEN
             apivot = ABS( A( l1 + i ) )
             ipivot = i ; jpivot = j
           END IF
         END DO
         l1 = l1 + m
       END DO

!  Interchange rows I and IPIVOT

       IPVT( k ) = ipivot
       IF ( ipivot > k ) THEN
         DO j = k, n
           atemp = A( l1 + ipivot )
           A( l2 + ipivot ) = A( l2 + k )
           A( l2 + k ) = atemp
           l2 = l2 + m
         END DO
       END IF

!  Interchange columns J and JPIVOT

       IF ( jpivot > k ) THEN
         j = JCOL( jpivot )
         JCOL( jpivot ) = JCOL( k ) ; JCOL( k ) = j
         l1 = m * ( jpivot - 1 )
         DO i = 1, m
           atemp = A( l1 + i ) ; A( l1 + i ) = A( l2 + i )
           A( l2 + i ) = atemp
         END DO
       END IF

!  Perform the elimination

       apivot = A( l2 + k )
       DO i = k + 1, m
         atemp = A( l2 + i ) / apivot
         A( l2 + i ) = atemp ; l1 = l2
         DO j = k + 1, n
           l1 = l1 + m
           A( l1 + i ) = A( l1 + i ) - atemp * A( l1 + k )
         END DO
       END DO
     END DO

     RETURN

!  End of subroutine OTHERS_gauss_elim_1d

     END SUBROUTINE OTHERS_gauss_elim_1d

!-*-  L A N C E L O T  -B-  OTHERS_gauss_elim_2d   S U B R O U T I N E   -*-

     SUBROUTINE OTHERS_gauss_elim_2d( m, n, IPVT, JCOL, A )

!  Perform the first M steps of Gaussian Elimination with
!  complete pivoting on the M by N ( M <= N) matrix A

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m, n
     INTEGER, INTENT( OUT ), DIMENSION( m ) :: IPVT
     INTEGER, INTENT( OUT ), DIMENSION( n ) :: JCOL
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m, n ) :: A

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k, ipivot, jpivot
     REAL ( KIND = wp ) :: apivot, atemp
     
     DO j = 1, n
       JCOL( j ) = j
     END DO

!  Main loop

     DO k = 1, m

!  Compute the K-th pivot

       apivot = - one
       DO j = k, n
         DO i = k, m
           IF ( ABS( A( i, j ) ) > apivot ) THEN
             apivot = ABS( A( i, j ) )
             ipivot = i
             jpivot = j
           END IF
         END DO
       END DO

!  Interchange rows I and IPIVOT

       IPVT( k ) = ipivot
       IF ( ipivot > k ) THEN
         DO j = k, n
           atemp = A( ipivot, j )
           A( ipivot, j ) = A( k, j )
           A( k, j ) = atemp
         END DO
       END IF

!  Interchange columns J and JPIVOT

       IF ( jpivot > k ) THEN
         j = JCOL( jpivot )
         JCOL( jpivot ) = JCOL( k )
         JCOL( k ) = j
         DO i = 1, m
           atemp = A( i, jpivot )
           A( i, jpivot ) = A( i, k )
           A( i, k ) = atemp
         END DO
       END IF

!  Perform the elimination

       apivot = A( k, k )
       DO i = k + 1, m
         atemp = A( i, k ) / apivot
         A( i, k ) = atemp
         A( i, k + 1 : n ) = A( i, k + 1 : n ) - atemp * A( k, k + 1 : n )
       END DO
     END DO

     RETURN

!  End of subroutine OTHERS_gauss_elim_2d

     END SUBROUTINE OTHERS_gauss_elim_2d

!-*-*-  L A N C E L O T  -B-  OTHERS_gauss_solve   S U B R O U T I N E   -*-*

     SUBROUTINE OTHERS_gauss_solve( m, IPVT, A, X )

!  Solve the equations A(T)x = b. The vector b is input in X.
!  The LU factors of P A are input in A; The permutation P is stored
!  in IPVT. The solution x is output in X

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: m
     INTEGER, INTENT( IN ), DIMENSION( m ) :: IPVT
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( m, m ) :: A
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( m ) :: X

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, k
     REAL ( KIND = wp ) :: xtemp
    
     DO k = 1, m
!      X( k ) = ( X( k ) - SUM( A( : k - 1, k ) * X( : k - 1 ) ) ) / A( k, k )
       xtemp = X( k )
       DO i = 1, k - 1 ; xtemp = xtemp - A( i, k ) * X( i ) ; END DO
       X( k ) = xtemp / A( k, k )
     END DO

!  Solve L(T) x = y. The vector y is input in X; x is output in X

     DO k = m - 1, 1, - 1
!      X( k ) = X( k ) - SUM( A( k + 1 : m, k ) * X( k + 1 : m ) )
       xtemp = X( k )
       DO i = k + 1, m ; xtemp = xtemp - A( i, k ) * X( i ) ; END DO
       X( k ) = xtemp
       i = IPVT( k )
       IF ( i /= k ) THEN
         xtemp = X( i )
         X( i ) = X( k )
         X( k ) = xtemp
       END IF
     END DO

     RETURN

!  End of subroutine OTHERS_gauss_solve

     END SUBROUTINE OTHERS_gauss_solve

!-*-*-*-  L A N C E L O T  -B-  OTHERS_symmh   S U B R O U T I N E   -*-*-*

     SUBROUTINE OTHERS_symmh( maxszh, ISYMMH, ISYMMD )

!  Given a columnwise storage scheme of the upper triangle of a
!  symmetric matrix of order MAXSZH, compute the position of the
!  I,J-th entry of the symmetric matrix in this scheme

!  The value ISYMMH( I, J ) + 1 gives the position of the I,J-th
!  entry of the matrix in the upper triangular scheme

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: maxszh
     INTEGER, INTENT( OUT ), DIMENSION( maxszh, maxszh ) :: ISYMMH
     INTEGER, INTENT( OUT ), DIMENSION( maxszh ) :: ISYMMD

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, j, k
     
     k = 0
     DO j = 1, maxszh
       DO i = 1, j - 1
         ISYMMH( i, j ) = k ; ISYMMH( j, i ) = k ; k = k + 1
       END DO
       ISYMMD( j ) = k ; ISYMMH( j, j ) = k ; k = k + 1
     END DO
     RETURN

!  End of OTHERS_symmh

     END SUBROUTINE OTHERS_symmh

!-  L A N C E L O T -B- OTHERS_which_variables_changed S U B R O U T I N E -

     SUBROUTINE OTHERS_which_variables_changed(                                &
                       unsucc, n, ng, nel, ncalcf, ncalcg, ISTAEV, ISTADG,     &
                       IELING, ICALCF, ICALCG, X     , XNEW  ,                 &
                       ISTAJC, IGCOLJ, LIST_elements, LINK_elem_uses_var )

!  Determines the elements and groups whose value must be recomputed because
!  of the change in the variable vector from X to XNEW. It is assumed that the
!  values of the elements and groups are available at X.  These elements
!  (groups) are stored in the vector ICALCF (ICALCG) from position 1 to NCALCF
!  (NCALCG)


!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, ng, nel
     INTEGER, INTENT( INOUT ) :: ncalcf, ncalcg
     LOGICAL, INTENT( IN  ) :: unsucc
     INTEGER, INTENT( INOUT ), DIMENSION( ng + 1 ) :: ISTADG
     INTEGER, INTENT( IN    ),                                            &
              DIMENSION( ISTADG( ng + 1 ) - 1 ) :: IELING
     INTEGER, INTENT( INOUT ), DIMENSION( nel ) :: ICALCF
     INTEGER, INTENT( INOUT ), DIMENSION( ng ) :: ICALCG
     INTEGER, INTENT( INOUT ), DIMENSION( nel + 1 ) :: ISTAEV
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: X, XNEW
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAJC
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IGCOLJ
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LIST_elements
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LINK_elem_uses_var

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, k, iel, iell, ipt, ig
     REAL ( KIND = wp ) :: diff, xi, smallest, epsmch
     
     epsmch = EPSILON( one )
     smallest = TINY( one )
     
     IF ( .NOT. unsucc ) THEN
       ncalcf = 0
       ncalcg = 0
     ELSE

!  Reset the element pointers to their correct signs

       DO i = 1, ncalcf
         iel = ICALCF( i )
         ISTAEV( iel ) = - ISTAEV( iel )
       END DO

!  Reset the group pointers to their correct signs

       DO i = 1, ncalcg
         ig = ICALCG( i )
         ISTADG( ig ) = - ISTADG( ig )
       END DO
     END IF

!  Detect the variables that have changed significantly from X

     DO i = 1, n
       xi = XNEW( i )
       diff = ABS( xi - X( i ) )
       IF ( xi /= zero ) THEN
         IF ( diff < epsmch * ABS( xi ) ) CYCLE
       ELSE
         IF ( diff < smallest ) CYCLE
       END IF

!  The I-th variable has been modified: flag all elements where it appears

       ipt = LINK_elem_uses_var( i )
       IF ( ipt >= 0 ) THEN
         iell = LIST_elements( i )
  200    CONTINUE
         iel = IELING( iell )
         IF ( ISTAEV( iel ) > 0 ) THEN
           ncalcf = ncalcf + 1
           ICALCF( ncalcf ) = iel
           ISTAEV( iel ) = - ISTAEV( iel )
         END IF
         IF ( ipt > 0 ) THEN
           iell = LIST_elements( ipt )
           ipt = LINK_elem_uses_var( ipt )
           GO TO 200
         END IF
       END IF

!  Flag all groups that contain the I-th variable

       DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
         ig = IGCOLJ( k )
         IF ( ISTADG( ig ) > 0 ) THEN
           ncalcg = ncalcg + 1
           ICALCG( ncalcg ) = ig
           ISTADG( ig ) = - ISTADG( ig )
        END IF
       END DO

!  End of the loop on the variables

     END DO

!  Reset the element pointers to their correct signs

     DO i = 1, ncalcf
       iel = ICALCF( i )
       ISTAEV( iel ) = - ISTAEV( iel )
     END DO

!  Reset the group pointers to their correct signs

     DO i = 1, ncalcg
       ig = ICALCG( i )
       ISTADG( ig ) = - ISTADG( ig )
     END DO

     RETURN

!  End of subroutine OTHERS_which_variables_changed

     END SUBROUTINE OTHERS_which_variables_changed

!-*-*-*-*  L A N C E L O T  -B-   OTHERS_time   F U N C T I O N -*-*-*-*

     FUNCTION OTHERS_time( time )
     CHARACTER ( LEN = 7 ) :: OTHERS_time

!  Obtain a 7 character representation of the time TIME

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = KIND( 1.0E0 ) ), INTENT( IN ) :: time

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: itim
     REAL ( KIND = KIND( 1.0E0 ) ) :: tim, timm, timh, timd
     CHARACTER ( len = 7 ) :: ctim

     OTHERS_time( 1 : 7 ) = '       '
     tim = time
     timm = time / 60.0
     timh = timm / 60.0
     timd = timh / 24.0
     IF ( tim <= 9999.9 ) THEN
        tim = time
        WRITE( UNIT = ctim, FMT = 2000 ) tim
        OTHERS_time = ctim
     ELSE IF ( tim <= 99999.9 ) THEN
        tim = time
        WRITE( UNIT = ctim, FMT = 2000 ) tim
        OTHERS_time( 1 : 1 ) = ' '
        OTHERS_time( 2 : 7 ) = ctim( 1 : 6 )
     ELSE IF ( tim <= 999999.0 ) THEN
        itim = INT(time)
        WRITE( UNIT = ctim, FMT = 2010 ) itim
        OTHERS_time = ctim
     ELSE IF ( timm <= 99999.9 ) THEN
        itim = INT( timm )
        WRITE( UNIT = ctim( 1 : 6 ), FMT = 2020 ) itim
        OTHERS_time = ctim( 1 : 6 ) // 'm'
     ELSE IF ( timh <= 99999.9 ) THEN
        itim = INT( timh )
        WRITE( UNIT = ctim( 1 : 6 ), FMT = 2020 ) itim
        OTHERS_time = ctim( 1 : 6 ) // 'h'
     ELSE IF ( timd <= 99999.9 ) THEN
        itim = INT( timd )
        WRITE( UNIT = ctim( 1 : 6 ), FMT = 2020 ) itim
        OTHERS_time = ctim( 1 : 6 ) // 'd'
     ELSE
        OTHERS_time = ' ******'
     END IF

     RETURN

!  Non-executable statements

 2000    FORMAT( 0P, F7.1 )
 2010    FORMAT( I7 )
 2020    FORMAT( I6 )

!  End of OTHERS_time

     END FUNCTION OTHERS_time

!-*-*-*-*  L A N C E L O T  -B-   OTHERS_time6   F U N C T I O N -*-*-*-*

     FUNCTION OTHERS_time6( time )
     CHARACTER ( LEN = 6 ) :: OTHERS_time6

!  Obtain a 7 character representation of the time TIME

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     REAL ( KIND = KIND( 1.0E0 ) ), INTENT( IN ) :: time

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: itim
     REAL ( KIND = KIND( 1.0E0 ) ) :: tim, timm, timh, timd
     CHARACTER ( len = 6 ) :: ctim

     OTHERS_time6( 1 : 6 ) = '      '
     tim = time
     timm = time / 60.0
     timh = timm / 60.0
     timd = timh / 24.0
     IF ( tim <= 999.9 ) THEN
        tim = time
        WRITE( UNIT = ctim, FMT = 2000 ) tim
        OTHERS_time6 = ctim
     ELSE IF ( tim <= 9999.9 ) THEN
        tim = time
        WRITE( UNIT = ctim, FMT = 2000 ) tim
        OTHERS_time6( 1 : 1 ) = ' '
        OTHERS_time6( 2 : 6 ) = ctim( 1 : 5 )
     ELSE IF ( tim <= 99999.0 ) THEN
        itim = INT(time)
        WRITE( UNIT = ctim, FMT = 2010 ) itim
        OTHERS_time6 = ctim
     ELSE IF ( timm <= 9999.9 ) THEN
        itim = INT( timm )
        WRITE( UNIT = ctim( 1 : 5 ), FMT = 2020 ) itim
        OTHERS_time6 = ctim( 1 : 5 ) // 'm'
     ELSE IF ( timh <= 9999.9 ) THEN
        itim = INT( timh )
        WRITE( UNIT = ctim( 1 : 5 ), FMT = 2020 ) itim
        OTHERS_time6 = ctim( 1 : 5 ) // 'h'
     ELSE IF ( timd <= 9999.9 ) THEN
        itim = INT( timd )
        WRITE( UNIT = ctim( 1 : 5 ), FMT = 2020 ) itim
        OTHERS_time6 = ctim( 1 : 5 ) // 'd'
     ELSE
        OTHERS_time6 = ' *****'
     END IF

     RETURN

!  Non-executable statements

 2000    FORMAT( 0P, F6.1 )
 2010    FORMAT( I6 )
 2020    FORMAT( I5 )

!  End of OTHERS_time6

     END FUNCTION OTHERS_time6

!-*-*-*-*  L A N C E L O T  -B-   OTHERS_iter   F U N C T I O N -*-*-*-*

     FUNCTION OTHERS_iter( iter )
     CHARACTER ( LEN = 6 ) :: OTHERS_iter

!  Obtain a 6 character representation of the iteration count iter

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: iter

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: iterk, iterm, iterg
     CHARACTER ( len = 6 ) :: citer

     OTHERS_iter( 1 : 6 ) = '      '
     iterk = iter / 1000
     iterm = iterk / 1000
     iterg = iterm / 1000
     IF ( iter <= 99999 ) THEN
        WRITE( UNIT = citer, FMT = 2010 ) iter
        OTHERS_iter = citer
     ELSE IF ( iter <= 9999999 ) THEN
        WRITE( UNIT = citer( 1 : 5 ), FMT = 2020 ) iterk
        OTHERS_iter = citer( 1 : 5 ) // 'k'
     ELSE IF ( iter <= 999999999 ) THEN
        WRITE( UNIT = citer( 1 : 5 ), FMT = 2020 ) iterm
        OTHERS_iter = citer( 1 : 5 ) // 'm'
     ELSE
        WRITE( UNIT = citer( 1 : 5 ), FMT = 2020 ) iterg
        OTHERS_iter = citer( 1 : 5 ) // 'g'
     END IF

     RETURN

!  Non-executable statements

 2010    FORMAT( I6 )
 2020    FORMAT( I5 )

!  End of OTHERS_iter

     END FUNCTION OTHERS_iter

!-*-*-*-*  L A N C E L O T  -B-   OTHERS_iter5   F U N C T I O N -*-*-*-*

     FUNCTION OTHERS_iter5( iter )
     CHARACTER ( LEN = 5 ) :: OTHERS_iter5

!  Obtain a 6 character representation of the iteration count iter

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: iter

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: iterk, iterm, iterg
     CHARACTER ( len = 5 ) :: citer

     OTHERS_iter5( 1 : 5 ) = '     '
     iterk = iter / 1000
     iterm = iterk / 1000
     iterg = iterm / 1000
     IF ( iter <= 9999 ) THEN
        WRITE( UNIT = citer, FMT = 2010 ) iter
        OTHERS_iter5 = citer
     ELSE IF ( iter <= 999999 ) THEN
        WRITE( UNIT = citer( 1 : 4 ), FMT = 2020 ) iterk
        OTHERS_iter5 = citer( 1 : 4 ) // 'k'
     ELSE IF ( iter <= 99999999 ) THEN
        WRITE( UNIT = citer( 1 : 4 ), FMT = 2020 ) iterm
        OTHERS_iter5 = citer( 1 : 4 ) // 'm'
     ELSE
        WRITE( UNIT = citer( 1 : 4 ), FMT = 2020 ) iterg
        OTHERS_iter5 = citer( 1 : 4 ) // 'g'
     END IF

     RETURN

!  Non-executable statements

 2010    FORMAT( I5 )
 2020    FORMAT( I4 )

!  End of OTHERS_iter5

     END FUNCTION OTHERS_iter5

!  End of module LANCELOT_OTHERS

   END MODULE LANCELOT_OTHERS_double




