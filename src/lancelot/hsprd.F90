! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-   HSPRD   M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  February 1st 1995

   MODULE LANCELOT_HSPRD_double

!  The elements of the array IUSED must be set to zero on entry; they will have
!  been reset to zero on exit. 

     IMPLICIT NONE

     PRIVATE
     PUBLIC :: HSPRD_hessian_times_vector

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp

   CONTAINS

!-*-  L A N C E L O T -B- HSPRD_hessian_times_vector  S U B R O U T I N E  -*

     SUBROUTINE HSPRD_hessian_times_vector(                                    &
                      n , ng, nel   , ntotel, nvrels, nvargp,  nfree , nvar1,  &
                      nvar2 , nnonnz, nbprod, alllin, IVAR  , ISTAEV, ISTADH,  &
                      INTVAR, IELING, IELVAR, ISWKSP, INONNZ, P , Q , GVALS2,  &
                      GVALS3, GRJAC , GSCALE, ESCALE, HUVALS, lhuval, GXEQX ,  &
                      INTREP, densep, IGCOLJ, ISLGRP, ISVGRP, ISTAGV, IVALJR,  &
                      ITYPEE, ISYMMH, ISTAJC, IUSED, LIST_elements,            &
                      LINK_elem_uses_var, NZ_comp_w, AP, W_el, W_in, H_in,     &
                      RANGE, skipg, KNDOFG )

!  Evaluate Q, the product of the hessian of a groups partially separable
!  function with the vector P

!  The nonzero components of P have indices IVAR( I ), I = NVAR1, ..., NVAR2.
!  The nonzero components of the product Q have indices INNONZ( I ),
!  I = 1, ..., NNONNZ

!  The components of ISWKSP must be less than
!  NBPROD on entry; on exit they will be no larger than NBPROD

!-----------------------------------------------
!   D u m m y   A r g u m e n t s
!-----------------------------------------------

     INTEGER, INTENT( IN    ) :: n , ng, nel   , ntotel, nvrels, nfree
     INTEGER, INTENT( IN    ) :: nvar1 , nvar2 , nbprod
     INTEGER, INTENT( IN    ) :: nvargp, lhuval
     INTEGER, INTENT( INOUT ) :: nnonnz
     LOGICAL, INTENT( IN    ) :: alllin, densep, skipg
     INTEGER, INTENT( IN    ), DIMENSION( n ) :: IVAR
     INTEGER, INTENT( IN    ), DIMENSION( nel + 1 ) :: ISTAEV, ISTADH
     INTEGER, INTENT( IN    ), DIMENSION( nel + 1 ) :: INTVAR
     INTEGER, INTENT( IN    ), DIMENSION( ntotel  ) :: IELING
     INTEGER, INTENT( IN    ), DIMENSION( nvrels  ) :: IELVAR
     INTEGER, INTENT( IN    ), DIMENSION( nel     ) :: ITYPEE
     INTEGER, INTENT( INOUT ), DIMENSION( ntotel ) :: ISWKSP
     INTEGER, INTENT( INOUT ), DIMENSION( n ) :: INONNZ
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( n ) :: P
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GVALS2
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GVALS3
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ng ) :: GSCALE
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( nvargp ) :: GRJAC
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( ntotel ) :: ESCALE
     REAL ( KIND = wp ), INTENT( IN  ), DIMENSION( lhuval ) :: HUVALS
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( n ) :: Q
     LOGICAL, INTENT( IN ), DIMENSION( ng ) :: GXEQX
     LOGICAL, INTENT( IN ), DIMENSION( nel ) :: INTREP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IGCOLJ
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISLGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISVGRP
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAGV
     INTEGER, INTENT( IN ), DIMENSION( : ) :: IVALJR
     INTEGER, INTENT( IN ), DIMENSION( : ) :: ISTAJC
     INTEGER, INTENT( INOUT ), DIMENSION( : ) :: IUSED 
     INTEGER, INTENT( IN ), DIMENSION( : ) :: LIST_elements
     INTEGER, INTENT( IN ), DIMENSION( : , : ) :: ISYMMH

     INTEGER, INTENT( IN ), DIMENSION( : ) :: LINK_elem_uses_var
     INTEGER, INTENT( OUT ), DIMENSION( : ) :: NZ_comp_w
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: AP
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_el
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: W_in
     REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: H_in

     INTEGER, INTENT( IN ), OPTIONAL, DIMENSION( ng ) :: KNDOFG

!-----------------------------------------------
!   I n t e r f a c e   B l o c k s
!-----------------------------------------------

     INTERFACE
       SUBROUTINE RANGE( ielemn, transp, W1, W2, nelvar, ninvar, ieltyp,       &
                         lw1, lw2 )
       INTEGER, INTENT( IN ) :: ielemn, nelvar, ninvar, ieltyp, lw1, lw2
       LOGICAL, INTENT( IN ) :: transp
       REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( IN  ), DIMENSION ( lw1 ) :: W1
!      REAL ( KIND = KIND( 1.0D+0 ) ), INTENT( OUT ), DIMENSION ( lw2 ) :: W2
       REAL ( KIND = KIND( 1.0D+0 ) ), DIMENSION ( lw2 ) :: W2
       END SUBROUTINE RANGE
     END INTERFACE

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i, iel, ig, ii, ipt, j, irow  , jcol  , ijhess , lthvar
     INTEGER :: iell  , nin, k, l , ll, nvarel, ielhst, nnz_comp_w
     REAL ( KIND = wp ) :: pi, gi, smallest
     LOGICAL :: nullwk

     smallest = TINY( one )

!  ======================= rank-one terms ==========================

!  If the IG-th group is non-trivial, form the product of P with the
!  sum of rank-one first order terms, A(trans) * GVALS3 * A. A is
!  stored by both rows and columns. For maximum efficiency, the
!  product is formed in different ways if P is sparse or dense

!  -----------------  Case 1. P is not sparse -----------------------

     IF ( densep ) THEN

!  Initialize AP and Q as zero

       AP( : ng ) = zero ; Q = zero

!  Form the matrix-vector product AP = A * P, using the column-wise
!  storage of A

       DO j = nvar1, nvar2
         i = IVAR( j )
         pi = P( i )
!DIR$ IVDEP
         DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
           AP( IGCOLJ( k ) ) = AP( IGCOLJ( k ) ) + pi * GRJAC( k )
         END DO
       END DO

!  Multiply W by the diagonal matrix GVALS3

       IF ( skipg ) THEN
         DO ig = 1, ng
           IF ( KNDOFG( ig ) == 0 ) THEN
             AP( ig ) = zero
           ELSE
             IF ( GXEQX( ig ) ) THEN
               AP( ig ) = AP( ig ) * GSCALE( ig )
             ELSE
               AP( ig ) = AP( ig ) * GSCALE( ig ) * GVALS3( ig )
             END IF
           END IF
         END DO
       ELSE
         WHERE ( GXEQX( : ng ) ) ; AP( : ng ) = AP( : ng ) * GSCALE( : ng )
         ELSEWHERE ; AP( : ng ) = AP( : ng ) * GSCALE( : ng ) * GVALS3( : ng )
         END WHERE
       END IF

!  Form the matrix-vector product Q = A(trans) * W, once again using the
!  column-wise storage of A

       nnonnz = 0
       DO j = 1, nfree
         i = IVAR( j )
!        Q( i ) =                                                              &
!          DOT_PRODUCT( AP( IGCOLJ( ISTAJC( i ) : ISTAJC( i + 1 ) - 1 ) ),     &
!                       GRJAC ( ISTAJC( i ) : ISTAJC( i + 1 ) - 1 ) )
         pi = zero
         DO ii = ISTAJC( i ), ISTAJC( i + 1 ) - 1
           pi = pi + AP( IGCOLJ( ii ) ) * GRJAC( ii )
         END DO
         Q( i ) = pi
       END DO
!write(6,"('q in hsprd ', /, (6ES12.4))" ) Q

!  ------------------- Case 2. P is sparse --------------------------

     ELSE
       nnz_comp_w = 0
       Q( IVAR( : nfree ) ) = zero

!  Form the matrix-vector product W = A * P, using the column-wise
!  storage of A. Keep track of the nonzero components of W in NZ_comp_w.
!  Only store components corresponding to non trivial groups

       DO j = nvar1, nvar2
         i = IVAR( j )
         pi = P( i )
!DIR$ IVDEP
         DO k = ISTAJC( i ), ISTAJC( i + 1 ) - 1
           ig = IGCOLJ( k )
           IF ( IUSED( ig ) == 0 ) THEN
             AP( ig ) = pi * GRJAC( k )
             IUSED( ig ) = 1
             nnz_comp_w = nnz_comp_w + 1
             NZ_comp_w( nnz_comp_w ) = ig
           ELSE
             AP( ig ) = AP( ig ) + pi * GRJAC( k )
           END IF
         END DO
       END DO

!  Reset IUSED to zero

       IUSED( NZ_comp_w( : nnz_comp_w ) ) = 0

!  Form the matrix-vector product Q = A( TRANS ) * W, using the row-wise
!  storage of A

       nnonnz = 0
       IF ( skipg ) THEN
         DO j = 1, nnz_comp_w
           ig = NZ_comp_w( j )
           IF ( KNDOFG( ig ) == 0 ) CYCLE
           IF ( .NOT. GXEQX( ig ) ) THEN

!  If group ig is non trivial, there are contributions from its rank-one term

             pi = GSCALE( ig ) * GVALS3( ig ) * AP( ig )
!DIR$ IVDEP
             DO k = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
               l = ISVGRP( k )

!  If Q has a nonzero in position L, store its index in INONNZ

               IF ( IUSED( l ) == 0 ) THEN
                 Q( l ) = pi * GRJAC( IVALJR( k ) )
                 IUSED( l ) = 1
                 nnonnz = nnonnz + 1
                 INONNZ( nnonnz ) = l
               ELSE
                 Q( l ) = Q( l ) + pi * GRJAC( IVALJR( k ) )
               END IF
             END DO
           END IF
         END DO
       ELSE
         DO j = 1, nnz_comp_w
           ig = NZ_comp_w( j )
           IF ( .NOT. GXEQX( ig ) ) THEN

!  If group ig is non trivial, there are contributions from its rank-one term

             pi = GSCALE( ig ) * GVALS3( ig ) * AP( ig )
!DIR$ IVDEP
             DO k = ISTAGV( ig ), ISTAGV( ig + 1 ) - 1
               l = ISVGRP( k )

!  If Q has a nonzero in position L, store its index in INONNZ

               IF ( IUSED( l ) == 0 ) THEN
                 Q( l ) = pi * GRJAC( IVALJR( k ) )
                 IUSED( l ) = 1
                 nnonnz = nnonnz + 1
                 INONNZ( nnonnz ) = l
               ELSE
                 Q( l ) = Q( l ) + pi * GRJAC( IVALJR( k ) )
               END IF
             END DO
           END IF
         END DO
       END IF
     END IF

     IF ( .NOT. alllin ) THEN

!  ======================= second-order terms =======================

!  Now consider the product of P with the second order terms (that is, the
!  2nd derivatives of the elements). Again, for maximum efficiency, the
!  product is formed in different ways if P is sparse or dense

!  --------------------- Case 1. P is not sparse ---------------------

       IF ( densep ) THEN
         DO iell = 1, ntotel
           ig = ISLGRP( iell )
           IF ( skipg ) THEN ; IF ( KNDOFG( ig ) == 0 ) CYCLE ; END IF
           ISWKSP( iell ) = nbprod
           iel = IELING( iell )
           nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
           IF ( GXEQX( ig ) ) THEN
             gi = GSCALE( ig ) * ESCALE( iell )
           ELSE
             gi = GSCALE( ig ) * ESCALE( iell ) * GVALS2( ig )
           END IF
           IF ( INTREP( iel ) ) THEN

!  The IEL-th element Hessian has an internal representation. Copy the
!  elemental variables into W

             nullwk = .TRUE.
             ll = ISTAEV( iel )
!DIR$ IVDEP
             DO ii = 1, nvarel
               pi = P( IELVAR( ll ) )
               W_el( ii ) = pi
               IF ( pi /= zero ) nullwk = .FALSE.
               ll = ll + 1
             END DO
             IF ( nullwk ) CYCLE

!  Find the internal variables, W_in

             nin = INTVAR( iel + 1 ) - INTVAR( iel )
             CALL RANGE ( iel, .FALSE., W_el, W_in, nvarel, nin,               &
                          ITYPEE( iel ), nvarel, nin )

!  Multiply the internal variables by the element Hessian and put the
!  product in H_in. Consider the first column of the element Hessian

             ielhst = ISTADH( iel )
             pi = gi * W_in( 1 )
             H_in( : nin ) = pi * HUVALS( ISYMMH( 1, : nin ) + ielhst )

!  Now consider the remaining columns of the element Hessian

             DO jcol = 2, nin
               pi = gi * W_in( jcol )
               IF ( pi /= zero ) THEN
                 H_in( : nin ) = H_in( : nin ) +                               &
                   pi * HUVALS( ISYMMH( jcol, : nin )+ ielhst )
               END IF
             END DO

!  Scatter the product back onto the elemental variables, W

             CALL RANGE ( iel, .TRUE., H_in, W_el, nvarel, nin,                &
                          ITYPEE( iel ), nin, nvarel )

!  Add the scattered product to Q

             ll = ISTAEV( iel )
!DIR$ IVDEP
             DO ii = 1, nvarel
                l = IELVAR( ll )
                Q( l ) = Q( l ) + W_el( ii )
                ll = ll + 1
             END DO
           ELSE

!  The IEL-th element Hessian has no internal representation

             lthvar = ISTAEV( iel ) - 1
             ielhst = ISTADH( iel )
             DO jcol = 1, nvarel
               pi = gi * P( IELVAR( lthvar + jcol ) )
               IF ( pi /= zero ) THEN
!DIR$ IVDEP  
                 DO irow = 1, nvarel
                   ijhess = ISYMMH( jcol, irow ) + ielhst
                   l = IELVAR( lthvar + irow )
                   Q( l ) = Q( l ) + pi * HUVALS( ijhess )
                 END DO
               END IF
             END DO
           END IF
         END DO
       ELSE

!  -------------------- Case 2. P is sparse ------------------------

         IF ( skipg ) THEN 
           DO j = nvar1, nvar2

!  Consider each nonzero component of P separately

             i = IVAR( j )
             ipt = LINK_elem_uses_var( i )
             IF ( ipt >= 0 ) THEN

!  The index of the I-th component lies in the IEL-th nonlinear element

               iell = LIST_elements( i )
  300          CONTINUE

!  Check to ensure that the IEL-th element has not already been used

               IF ( ISWKSP( iell ) < nbprod ) THEN
                 ISWKSP( iell ) = nbprod
                 ig = ISLGRP( iell )
                 IF ( KNDOFG( ig ) /= 0 ) THEN
                   iel = IELING( iell )
                   nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
                   IF ( GXEQX( ig ) ) THEN
                     gi = GSCALE( ig ) * ESCALE( iell )
                   ELSE
                     gi = GSCALE( ig ) * ESCALE( iell ) * GVALS2( ig )
                   END IF
                   IF ( INTREP( iel ) ) THEN

!  The IEL-th element Hessian has an internal representation. Copy the
!  elemental variables into W

                     ll = ISTAEV( iel )
                     W_el( : nvarel ) = P( IELVAR( ll : ll + nvarel - 1 ) )

!  Find the internal variables

                     nin = INTVAR( iel + 1 ) - INTVAR( iel )
                     CALL RANGE ( iel, .FALSE., W_el, W_in, nvarel, nin,       &
                                  ITYPEE( iel ), nvarel, nin )

!  Multiply the internal variables by the element Hessian and put the
!  product in W_in. Consider the first column of the element Hessian

                     ielhst = ISTADH( iel )
                     pi = gi * W_in( 1 )
                     H_in( : nin ) = pi * HUVALS( ISYMMH( 1, : nin ) + ielhst )

!  Now consider the remaining columns of the element Hessian

                     DO jcol = 2, nin
                       pi = gi * W_in( jcol )
                       IF ( pi /= zero ) THEN
                         H_in( : nin ) = H_in( : nin ) + pi *                  &
                           HUVALS( ISYMMH( jcol, : nin ) + ielhst )
                       END IF
                     END DO

!  Scatter the product back onto the elemental variables, W

                     CALL RANGE ( iel, .TRUE., H_in, W_el, nvarel, nin,        &
                                  ITYPEE( iel ), nin, nvarel )

!  Add the scattered product to Q

                     ll = ISTAEV( iel )
!DIR$ IVDEP
                     DO ii = 1, nvarel
                       l = IELVAR( ll )

!  If Q has a nonzero in position L, store its index in INONNZ

                       IF ( ABS( W_el( ii ) ) > smallest ) THEN
                         IF ( IUSED( l ) == 0 ) THEN
                           Q( l ) = W_el( ii )
                           IUSED( l ) = 1
                           nnonnz = nnonnz + 1
                           INONNZ( nnonnz ) = l
                         ELSE
                           Q( l ) = Q( l ) + W_el( ii )
                         END IF
                       END IF
                       ll = ll + 1
                     END DO
                   ELSE

!  The IEL-th element Hessian has no internal representation

                     lthvar = ISTAEV( iel ) - 1
                     ielhst = ISTADH( iel )
                     DO jcol = 1, nvarel
                       pi = gi * P( IELVAR( lthvar + jcol ) )
                       IF ( pi /= zero ) THEN
!DIR$ IVDEP        
                         DO irow = 1, nvarel
                           ijhess = ISYMMH( jcol, irow ) + ielhst

!  If Q has a nonzero in position L, store its index in INONNZ

                           IF ( ABS( HUVALS( ijhess ) ) > smallest ) THEN
                             l = IELVAR( lthvar + irow )
                             IF ( IUSED( l ) == 0 ) THEN
                               Q( l ) = pi * HUVALS( ijhess )
                               IUSED( l ) = 1
                               nnonnz = nnonnz + 1
                               INONNZ( nnonnz ) = l
                             ELSE
                                Q( l ) = Q( l ) + pi * HUVALS( ijhess )
                             END IF
                           END IF
                         END DO
                       END IF
                     END DO
                   END IF
                 END IF
               END IF

!  Check to see if there are any further elements whose variables
!  include the I-th variable

               IF ( ipt > 0 ) THEN
                 iell = LIST_elements( ipt )
                 ipt = LINK_elem_uses_var( ipt )
                 GO TO 300
               END IF
             END IF
           END DO
         ELSE
           DO j = nvar1, nvar2

!  Consider each nonzero component of P separately

             i = IVAR( j )
             ipt = LINK_elem_uses_var( i )
             IF ( ipt >= 0 ) THEN

!  The index of the I-th component lies in the IEL-th nonlinear element

               iell = LIST_elements( i )
  310          CONTINUE

!  Check to ensure that the IEL-th element has not already been used

               IF ( ISWKSP( iell ) < nbprod ) THEN
                 ISWKSP( iell ) = nbprod
                 iel = IELING( iell )
                 nvarel = ISTAEV( iel + 1 ) - ISTAEV( iel )
                 ig = ISLGRP( iell )
                 IF ( GXEQX( ig ) ) THEN
                   gi = GSCALE( ig ) * ESCALE( iell )
                 ELSE
                   gi = GSCALE( ig ) * ESCALE( iell ) * GVALS2( ig )
                 END IF
                 IF ( INTREP( iel ) ) THEN

!  The IEL-th element Hessian has an internal representation. Copy the
!  elemental variables into W

                   ll = ISTAEV( iel )
                   W_el( : nvarel ) = P( IELVAR( ll : ll + nvarel - 1 ) )

!  Find the internal variables

                   nin = INTVAR( iel + 1 ) - INTVAR( iel )
                   CALL RANGE ( iel, .FALSE., W_el, W_in, nvarel, nin,         &
                                ITYPEE( iel ), nvarel, nin )

!  Multiply the internal variables by the element Hessian and put the
!  product in W_in. Consider the first column of the element Hessian

                   ielhst = ISTADH( iel )
                   pi = gi * W_in( 1 )
                   H_in( : nin ) = pi * HUVALS( ISYMMH( 1, : nin ) + ielhst )

!  Now consider the remaining columns of the element Hessian

                   DO jcol = 2, nin
                     pi = gi * W_in( jcol )
                     IF ( pi /= zero ) THEN
                       H_in( : nin ) = H_in( : nin ) + pi *                    &
                         HUVALS( ISYMMH( jcol, : nin ) + ielhst )
                     END IF
                   END DO

!  Scatter the product back onto the elemental variables, W

                   CALL RANGE ( iel, .TRUE., H_in, W_el, nvarel, nin,          &
                                ITYPEE( iel ), nin, nvarel )

!  Add the scattered product to Q

                   ll = ISTAEV( iel )
!DIR$ IVDEP
                   DO ii = 1, nvarel
                     l = IELVAR( ll )

!  If Q has a nonzero in position L, store its index in INONNZ

                     IF ( ABS( W_el( ii ) ) > smallest ) THEN
                       IF ( IUSED( l ) == 0 ) THEN
                         Q( l ) = W_el( ii )
                         IUSED( l ) = 1
                         nnonnz = nnonnz + 1
                         INONNZ( nnonnz ) = l
                       ELSE
                         Q( l ) = Q( l ) + W_el( ii )
                       END IF
                     END IF
                     ll = ll + 1
                   END DO
                 ELSE

!  The IEL-th element Hessian has no internal representation

                   lthvar = ISTAEV( iel ) - 1
                   ielhst = ISTADH( iel )
                   DO jcol = 1, nvarel
                     pi = gi * P( IELVAR( lthvar + jcol ) )
                     IF ( pi /= zero ) THEN
!DIR$ IVDEP      
                       DO irow = 1, nvarel
                         ijhess = ISYMMH( jcol, irow ) + ielhst

!  If Q has a nonzero in position L, store its index in INONNZ

                         IF ( ABS( HUVALS( ijhess ) ) > smallest ) THEN
                           l = IELVAR( lthvar + irow )
                           IF ( IUSED( l ) == 0 ) THEN
                             Q( l ) = pi * HUVALS( ijhess )
                             IUSED( l ) = 1
                             nnonnz = nnonnz + 1
                             INONNZ( nnonnz ) = l
                           ELSE
                              Q( l ) = Q( l ) + pi * HUVALS( ijhess )
                           END IF
                         END IF
                       END DO
                     END IF
                   END DO
                 END IF
               END IF

!  Check to see if there are any further elements whose variables
!  include the I-th variable

               IF ( ipt > 0 ) THEN
                 iell = LIST_elements( ipt )
                 ipt = LINK_elem_uses_var( ipt )
                 GO TO 310
               END IF
             END IF
           END DO
         END IF
       END IF
     END IF

!  ==================== the product is complete =======================

!  Reset IUSED to zero

     IF ( .NOT. densep ) IUSED( INONNZ( : nnonnz ) ) = 0
     RETURN

!  End of subroutine HSPRD_hessian_times_vector

     END SUBROUTINE HSPRD_hessian_times_vector

!  End of module LANCELOT_HSPRD

   END MODULE LANCELOT_HSPRD_double
