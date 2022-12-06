! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  STRUTR  M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  May 6th 2001

   MODULE LANCELOT_STRUTR_double

     PRIVATE
     PUBLIC :: STRUTR_radius_update

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp

   CONTAINS

!-*-*-*-*-  L A N C E L O T -B- STRUTR_radius_update  S U B R O U T I N E -*-*-*

     SUBROUTINE STRUTR_radius_update(                                          &
                              n, ng, nel, S,                                   &
                              IELING, ISTADG, IELVAR, ISTAEV, INTVAR,          &
                              ISTADH, ISTADA, ICNA  , A     , ESCALE,          &
                              GSCALE, FT    , GXEQX , ITYPEE, INTREP,          &
                              FUVALS, lfuval, GV_old, GVALS ,                  &
                              GVALS2_old    , GVALS3_old    , GRJAC ,          &
                              nvargp, ared  , prered, RADII , radmax,          &
                              eta_successful, eta_very_successful,             &
                              eta_extremely_successful, gamma_decrease,        &
                              gamma_increase, mu_meaningful_model,             &
                              mu_meaningful_group, ISTAGV, ISVGRP, IVALJR,     &
                              ISYMMH, W_el  , W_in  , H_in  , TEMP_el, RANGE )

!  Update the trust-region radii for structured trust-region models
!  using the recipe in the Conn-Gould-Toint Trust-Region book (Section 10.2)
!  having computed the changes in the element models and values


!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN ) :: n, ng, nel
     REAL ( KIND = wp ), INTENT( IN ) :: ared, prered, radmax
     REAL ( KIND = wp ), INTENT( IN ) :: eta_successful, eta_very_successful
     REAL ( KIND = wp ), INTENT( IN ) :: eta_extremely_successful
     REAL ( KIND = wp ), INTENT( IN ) :: gamma_increase, mu_meaningful_model
     REAL ( KIND = wp ), INTENT( IN ) :: mu_meaningful_group, gamma_decrease
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( ng ) :: RADII
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( n ) :: S
     INTEGER, INTENT( IN ) :: lfuval, nvargp
     INTEGER, INTENT( IN ), DIMENSION( ng + 1 ) :: ISTADA, ISTADG
     INTEGER, INTENT( IN ), DIMENSION( ISTADG( ng + 1 ) - 1 ) :: IELING
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTAEV
     INTEGER, INTENT( IN ), DIMENSION( ISTAEV( nel + 1 ) - 1 ) :: IELVAR
     INTEGER, INTENT( IN ), DIMENSION( nel + 1 ) :: ISTADH, INTVAR
     INTEGER, INTENT( IN ), DIMENSION( ISTADA( ng + 1 ) - 1 ) :: ICNA
     INTEGER, INTENT( IN ), DIMENSION( nel ) :: ITYPEE
     REAL ( KIND = wp ), INTENT( IN ),                                         &
              DIMENSION( ISTADA( ng + 1 ) - 1  ) :: A
     REAL ( KIND = wp ), INTENT( IN ),                                         &
              DIMENSION( ISTADG( ng + 1 ) - 1 ) :: ESCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: FT
     LOGICAL, INTENT( IN ), DIMENSION( ng  ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( lfuval ) :: FUVALS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GV_old
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS2_old
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( ng ) :: GVALS3_old
     REAL ( KIND = wp ), INTENT( IN ), DIMENSION( nvargp ) :: GRJAC
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IVALJR
     INTEGER, INTENT( IN ), DIMENSION( : , : ) :: ISYMMH
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: TEMP_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
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

     INTEGER :: i, ii, ig, ig1, istrgv, iendgv, iel, iell, j, k, l, ll, nin
     INTEGER :: nvarel, ielhst, jcol, ijhess, lthvar, nelow, nelup, irow
     REAL ( KIND = wp ) :: si, d_mod, d_mod2,  gdash, dm, df, md, hg2dash
     REAL ( KIND = wp ) :: scalee
     LOGICAL ::  nullwk
  
  
!    WRITE(6, "( ' s ', 5ES12.4 )" ) S
!    WRITE(6, "( '    ig     d_f         d_m     GXEQX  radius')" )
     md = prered / ng
     DO ig = 1, ng

!  First, compute the changes in the i-th model and element

!  ::::::::::::::::::::::::
!  Change in the i-th model
!  ::::::::::::::::::::::::

!  The change is - [ g_i' s^T ( grad g_i + 1/2 hess g_i * s ) 
!                    + 1/2 g_i'' (s^T grad g_i )^2 ]
!  where g_i is the ith group

       ig1 = ig + 1
       istrgv = ISTAGV( ig ) ; iendgv = ISTAGV( ig1 ) - 1
       nelow  = ISTADG( ig ) ; nelup  = ISTADG( ig1 ) - 1
       TEMP_el( ISVGRP( istrgv : iendgv ) ) = zero

!  Compute the term hess g_i * s from the nonlinear elements for the group
!  -----------------------------------------------------------------------

       DO iell = nelow, nelup
         iel = IELING( iell )
         nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
         scalee = ESCALE( iell )
         IF ( INTREP( iel ) ) THEN

!  The IEL-th element Hessian has an internal representation. Copy the
!  elemental variables into W

           nullwk = .TRUE.
           ll = ISTAEV( iel )
           DO ii = 1, nvarel
             si = S( IELVAR( ll ) )
             W_el( ii ) = si
             IF ( si /= zero ) nullwk = .FALSE.
             ll = ll + 1
           END DO
           IF ( nullwk ) CYCLE

!  Find the internal variables, W_in

           nin = INTVAR( iel + 1 ) - INTVAR( iel )
           CALL RANGE ( iel, .FALSE., W_el, W_in, nvarel, nin,                 &
                        ITYPEE( iel ), nvarel, nin )

!  Multiply the internal variables by the element Hessian and put the
!  product in H_in. Consider the first column of the element Hessian

           ielhst = ISTADH( iel )
           si = scalee * W_in( 1 )
           H_in( : nin ) = si * FUVALS( ISYMMH( 1, : nin ) + ielhst )

!  Now consider the remaining columns of the element Hessian

           DO jcol = 2, nin
             si = scalee * W_in( jcol )
             IF ( si /= zero ) THEN
               H_in( : nin ) =                                                 &
                 H_in( : nin ) + si * FUVALS( ISYMMH( jcol, : nin ) + ielhst )
              END IF
           END DO

!  Scatter the product back onto the elemental variables, W

           CALL RANGE ( iel, .TRUE., H_in, W_el, nvarel, nin,                  &
                        ITYPEE( iel ), nin, nvarel )

!  Add the scattered product to TEMP_el

           ll = ISTAEV( iel )
           DO ii = 1, nvarel
             l = IELVAR( ll )
             TEMP_el( l ) = TEMP_el( l ) + W_el( ii )
             ll = ll + 1
           END DO
         ELSE

!  The iel-th element Hessian has no internal representation

           lthvar = ISTAEV( iel ) - 1
           ielhst = ISTADH( iel )
           DO jcol = 1, nvarel
             si = scalee * S( IELVAR( lthvar + jcol ) )
             IF ( si /= zero ) THEN
               DO irow = 1, nvarel
                 ijhess = ISYMMH( jcol, irow ) + ielhst
                 l = IELVAR( lthvar + irow )
                 TEMP_el( l ) = TEMP_el( l ) + si * FUVALS( ijhess )
               END DO
             END IF
           END DO
         END IF
       END DO

!  Now form grad g_i + 1/2 hess g_i * s
!  ------------------------------------

       IF ( GXEQX( ig ) ) THEN
         TEMP_el( ISVGRP( istrgv : iendgv ) ) =                                &
           half * TEMP_el( ISVGRP( istrgv : iendgv ) )


!  Loop over the group's nonlinear elements

         DO ii = nelow, nelup
           iel = IELING( ii )
           k = INTVAR( iel )
           l = ISTAEV( iel )
           nvarel = ISTAEV( iel + 1 ) - l
           scalee = ESCALE( ii )
           IF ( INTREP( iel ) ) THEN

!  The IEL-th element has an internal representation

             nin = INTVAR( iel + 1 ) - k
             CALL RANGE ( iel, .TRUE., FUVALS( k : k + nin - 1 ),              &
                          W_el( : nvarel ), nvarel, nin,                       &
                          ITYPEE( iel ), nin, nvarel )
!DIR$ IVDEP
             DO i = 1, nvarel
               j = IELVAR( l )
               TEMP_el( j ) = TEMP_el( j ) + scalee * W_el( i )
               l = l + 1
             END DO
           ELSE

!  The IEL-th element has no internal representation

!DIR$ IVDEP
             DO i = 1, nvarel
               j = IELVAR( l )
               TEMP_el( j ) = TEMP_el( j ) + scalee * FUVALS( k )
               k = k + 1
               l = l + 1
             END DO
           END IF
         END DO

!  Include the contribution from the linear element

!DIR$ IVDEP
         DO k = ISTADA( ig ), ISTADA( ig1 ) - 1
           TEMP_el( ICNA( k ) ) = TEMP_el( ICNA( k ) ) + A( k )
         END DO
       ELSE
         TEMP_el( ISVGRP( istrgv : iendgv ) ) =                                &
           GRJAC( IVALJR( istrgv : iendgv ) ) + half *                         &
             TEMP_el( ISVGRP( istrgv : iendgv ) )
       END IF
!      WRITE(6,*) ' temp_el ', TEMP_el( ISVGRP( istrgv : iendgv ) ) 

!  Finally form - [ g_i' s^T ( grad g_i + 1/2 hess g_i * s ) 
!                   + 1/2 g_i'' (s^T grad g_i )^2 ]

       d_mod = zero ; d_mod2 = zero
       IF ( GXEQX( ig ) ) THEN
         gdash = GSCALE( ig )
         DO l = istrgv, iendgv
           i = ISVGRP( l )
           d_mod = d_mod + S( i ) * TEMP_el( i )
         END DO
         dm = - gdash * d_mod
         df = GSCALE( ig ) * ( GV_old( ig ) - FT( ig ) )
       ELSE
         gdash = GSCALE( ig ) * GVALS2_old( ig )
         hg2dash = half * GSCALE( ig ) * GVALS3_old( ig )
         DO l = istrgv, iendgv
           i = ISVGRP( l )
           d_mod = d_mod + S( i ) * TEMP_el( i )
           d_mod2 = d_mod2 + S( i ) * GRJAC( IVALJR( l ) )
!          WRITE(6,*) i, S( i ), GRJAC( IVALJR( l ) )
         END DO
!        WRITE(6,*) gdash,  d_mod, hg2dash, d_mod2
         dm = - gdash * d_mod - hg2dash * d_mod2 ** 2
         df = GSCALE( ig ) * ( GV_old( ig ) - GVALS( ig ) )
       END IF

!  ::::::::::::::::::::::::
!  i-th model radius update
!  ::::::::::::::::::::::::

!  Meaningful elements

       IF ( ABS( dm ) > mu_meaningful_model * md ) THEN
       
         IF ( df >= dm - ( one - eta_extremely_successful) * md ) THEN
           IF ( ared >= eta_successful * prered ) THEN
             RADII( ig ) = MIN( gamma_increase * RADII( ig ), radmax )
           ELSE
           END IF
         ELSE
           IF ( df >= dm - ( one - eta_very_successful ) * md ) THEN
           ELSE
             RADII( ig ) = gamma_decrease * RADII( ig )
           END IF
         END IF
!        WRITE(6,"( I6, 2ES12.4, L2, ES12.4, A12 )" )                          &
!          ig, df, dm, GXEQX( ig ), RADII( ig ), ' meaningful'

!  Negligible elements

       ELSE
         IF ( ABS( df ) <= mu_meaningful_group * md ) THEN
           IF ( ared >= eta_successful * prered ) THEN
             RADII( ig ) = MIN( gamma_increase * RADII( ig ), radmax )
           ELSE
           END IF
         ELSE
           RADII( ig ) = gamma_decrease * RADII( ig )
         END IF
!        WRITE(6,"( I6, 2ES12.4, L2, ES12.4, A12 )" )                          &
!          ig, df, dm, GXEQX( ig ), RADII( ig ), ' negligible'
       END IF
     END DO
     RETURN

!  End of subroutine STRUTR_radius_update

     END SUBROUTINE STRUTR_radius_update

!  End of module LANCELOT_STRUTR

   END MODULE LANCELOT_STRUTR_double



