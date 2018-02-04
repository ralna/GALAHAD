! THIS VERSION: GALAHAD 2.6 - 15/10/2014 AT 20:00 GMT.

!-*-*-*-*-*-*-  L A N C E L O T  -B-   CAUCHY    M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 30th 1995

   MODULE LANCELOT_CAUCHY_double

     USE GALAHAD_SORT_double, ONLY : SORT_heapsort_build, SORT_heapsort_smallest
     IMPLICIT NONE

     PRIVATE
     PUBLIC :: CAUCHY_save_type, CAUCHY_get_exact_gcp, CAUCHY_get_approx_gcp

!  Set precision

     INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

     REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
     REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
     REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
     REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
     REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
     REAL ( KIND = wp ), PARAMETER :: gzero = ten ** ( - 10 )
     REAL ( KIND = wp ), PARAMETER :: hzero = ten ** ( - 10 )
     REAL ( KIND = wp ), PARAMETER :: alpha = one
     REAL ( KIND = wp ), PARAMETER :: beta = half

!  ================================
!  The CAUCHY_save_type derived type
!  ================================

    TYPE :: CAUCHY_save_type
      INTEGER :: iterca, iter, itmax, nfreed, nbreak, nzero
      REAL ( KIND = wp ) :: tk, gxt, hxt, epstl2, tpttp, tcauch 
      REAL ( KIND = wp ) :: tbreak, deltat, epsqrt, gxtold, g0tp
      REAL ( KIND = wp ) :: t, tamax , ptp, gtp, flxt, tnew
      LOGICAL :: prnter, pronel, recomp
    END TYPE CAUCHY_save_type

  CONTAINS

!-*-*-  L A N C E L O T  -B-   CAUCHY_get_exact_gcp  S U B R O U T I N E  -*-*

     SUBROUTINE CAUCHY_get_exact_gcp( n, X0, XT, G, BND, X_status, f, tmax,    &
                                      epstol, boundx, dxsqr, r, qxt, P, Q,     &
                                      IVAR, nfree, nvar1, nvar2 , nnonnz,      &
                                      INONNZ, iout, jumpto, idebug, findmx,    &
                                      BREAKP, S )

!  Find the Generalized Cauchy Point (GCP) in the direction D from X0
!  for a given quadratic function within a box shaped region

!  If we define the 'Cauchy arc' X(t) = projection of X0 + t*P into the box
!  region BND(*,1) <= X(*) <= BND(*,2), the GCP is the first local minimizer
!  of the quadratic function

!     0.5 (X-X0) (transpose ) B (X-X0) + G (transpose) (X-X0) + F

!  for points lying on X(t), with 0 <= t <= tmax. Optionally, the search for the
!  GCP may be terminated at the first point encountered on the boundary of the
!  spherical region ||X - X0|| <= R, where ||.|| denotes the 2-norm.
!  Control is passed from the routine whenever a product of the vector P with
!  B is required, and the user is responsible for forming the product in the
!  vector Q

!  Progress through the routine is controlled by the parameter JUMPTO

!  If JUMPTO = 0, the GCP has been found
!  If JUMPTO = 1, an initial entry has been made
!  If JUMPTO = 2, 3, 4 the vector Q = B * P is required

!  The value of the array X_status gives the status of the variables

!  IF X_status( I ) = 0, the I-TH variable is free
!  IF X_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF X_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF X_status( I ) = 3, the I-th variable is permanently fixed
!  IF X_status( I ) = 4, the I-th variable is fixed at some other value

!  The addresses of the free variables are given in the first NVAR entries
!  of the array IVAR

!  If the product B * P is required (JUMPTO = 2,3,4), the nonzero components
!  of P occur in positions IVAR(I) for I lying between NVAR1 and NVAR2

!  At the initial point, variables within EPSTOL of their bounds and
!  for which the search direction P points out of the box will be fixed

!  ------------------------- dummy arguments --------------------------

!  N      (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  X0     (REAL array of length at least N) the point x0 from which the Cauchy
!          arc commences. ** this variable is not altered by the subroutine
!  XT     (REAL array of length at least N) the current estimate of the GCP
!  G      (REAL array of length at least N) the coefficients of
!          the linear term in the quadratic function
!          ** this variable is not altered by the subroutine
!  BND    (two dimensional REAL array with leading dimension N and second
!          dimension 2) the lower (BND(*,1)) and upper (BND(*,2)) bounds on
!          the variables. ** this variable is not altered by the subroutine
!  X_status (INTEGER array of length at least N) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          X_status should be set as follows:
!          If X_status( I ) = 0, the I-th variable is free
!          If X_status( I ) = 1, the I-th variable is on its lower bound
!          If X_status( I ) = 2, the I-th variable is on its upper bound
!          If X_status( I ) = 3, 4, the I-th variable is fixed at XT(I)
!  F      (REAL) the value of the quadratic at X0, see above.
!          ** this variable is not altered by the subroutine
!  EPSTOL (REAL) a tolerance on feasibility of X0, see above.
!          ** this variable is not altered by the subroutine.
!  BOUNDX (LOGICAL) the search for the generalized cauchy point will be
!          terminated on the boundary of the spherical region ||X-X0|| <= R
!          if and only if BOUNDX is set to .TRUE. on initial (JUMPTO=1) entry.
!          ** this variable is not altered by the subroutine
!  DXSQR  (REAL) the square of the two norm of the distance between the
!          current estimate of the GCP and X0. DXSQR will only be set if
!          BOUNDX is .TRUE.
!  r      (REAL) the radius, R, of the spherical region. r need not be
!          set if boundx is .FALSE. on initial entry.
!          ** This variable is not altered by the subroutine
!  QXT    (REAL) the value of the piecewise quadratic function at the current
!          estimate of the GCP
!  P      (REAL array of length at least N) contains the values of the
!          components of the vector P. On initial (JUMPTO=1) entry, P must
!          contain the initial direction of the 'Cauchy arc'. On a non optimal
!          exit, (JUMPTO=2,3,4), P is the vector for which the product B * P
!          is required before the next re-entry. On a terminal exit (JUMPTO=0),
!          P contains the step XT - X0. The components IVAR(I) = NVAR1, ... ,
!          NVAR2 of P contain the values of the nonzero components of P
!          (see, IVAR, NVAR1, NVAR2)
!  Q      (REAL array of length at least N) on a non initial entry
!         (JUMPTO=2,3,4), Q must contain the vector B * P. Only the
!          components IVAR(I), I=1,...,NFREE, of Q need be set (the other
!          components are not used)
!  IVAR   (INTEGER array of length at least N) on all normal exits
!         (JUMPTO=0,2,3,4), IVAR(I), I = NVAR1, ..., NVAR2, gives the indices
!          of the nonzero components of P
!  NFREE  (INTEGER) the number of free variables at the initial point
!  NVAR1  (INTEGER) see IVAR, above
!  NVAR2  (INTEGER) see IVAR, above
!  NNONNZ (INTEGER) the number of nonzero components of Q on a JUMPTO=3 entry.
!          NNONNZ need not be set on other entries
!  INONNZ (INTEGER array of length at least NNONNZ) on JUMPTO = 3 entries,
!          INONN(I), I = 1,....,NNONNZ, must give the indices of the nonzero
!          components of Q. On other entries, INONNZ need not be set
!  IOUT   (INTEGER) the fortran output channel number to be used
!  JUMPTO (INTEGER) controls flow through the subroutine.
!          If JUMPTO = 0, the GCP has been found
!          If JUMPTO = 1, an initial entry has been made
!          If JUMPTO = 2, 3, 4, the vector Q = B * P is required
!  IDEBUG (INTEGER) allows detailed printing. If IDEBUG is larger than 4,
!          detailed output from the routine will be given. Otherwise, no
!          output occurs
!  findmx (REAL) when printing the value of the objective function,
!          the value calculated in fmodel will be multiplied by the
!          scale factor findmx. This allows the user, for instance,
!          to find the maximum of a quadratic function F, by minimizing
!          the function - F, while monitoring the progress as
!          if a maximization were actually taking place, by setting
!          findmx to - 1.0. Normally findmx should be set to 1.0
!  BREAKP (REAL) workspace that must be preserved between calls
!  S      (CAUCHY_save_type) private data that must be preserved between calls

!  ------------------ end of dummy arguments --------------------------

     INTEGER, INTENT( IN    ):: n, iout, idebug
     INTEGER, INTENT( INOUT ):: nfree, nvar1, nvar2, nnonnz, jumpto
     REAL ( KIND = wp ), INTENT( IN ):: r, tmax, findmx, epstol, f
     REAL ( KIND = wp ), INTENT( INOUT ):: dxsqr, qxt
     LOGICAL, INTENT( IN ) :: boundx
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: X_status
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: IVAR, INONNZ
     REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n, 2 ) :: BND
     REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n ) :: X0, G
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: XT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: P, Q
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: BREAKP
     TYPE( CAUCHY_save_type ), INTENT( INOUT ) :: S

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      N, X0, G, TMAX, P, BND, F, EPSTOL, BOUNDX, IOUT, JUMPTO, IDEBUG

!  JUMPTO must have the value 1. In addition, if the I-th variable is required
!  to be fixed at its initial value, X0(I), X_status(I) must be set to
!  3 or 4. r must be specified if boundx is .TRUE. on initial entry

!  RE-ENTRY:

!  If the variable JUMPTO has the value 2, 3 or 4 on exit, the
!  subroutine must be re-entered with the vector Q containing
!  the product of the second derivative matrix B and the output
!  vector P. All other parameters MUST NOT BE ALTERED

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i , j , ibreak, insort, nfreed
     REAL ( KIND = wp ) :: epsmch, t, tstar, tbndsq, gp, pbp, feasep, qipi
     LOGICAL :: xlower, xupper

     epsmch = EPSILON( one )

     SELECT CASE ( jumpto )
     CASE ( 1 ) ; GO TO 100
     CASE ( 2 ) ; GO TO 200
     CASE ( 3 ) ; GO TO 300
     CASE ( 4 ) ; GO TO 400
     END SELECT

!  On initial entry, set constants

 100 CONTINUE
     S%prnter = idebug >= 4 .AND. iout > 0
     S%pronel = idebug == 2 .AND. iout > 0
!    S%prnter = .TRUE. ; S%pronel = .FALSE.
     IF ( S%prnter ) WRITE( iout, 2010 )
     IF ( S%pronel ) WRITE( iout, 2120 )
     S%nbreak = 0 ; nfreed = 0 ; S%nzero = n + 1
     S%epstl2 = ten * epsmch ; S%epsqrt = SQRT( epsmch ) ; S%tbreak = zero

!  If necessary, initialize the distances to the spherical boundary.
!  ptp is the sum of the squares of the free components of the Cauchy
!  direction. tpttp is the sum of the squares of the distances to the
!  boundary of the box for the variables which are fixed as the
!  computation proceeds

     IF ( boundx ) THEN
       dxsqr = zero ; S%ptp = zero ; S%tpttp = zero
     END IF
     IF ( idebug >= 100 ) THEN
       DO i = 1, n
         WRITE( iout, 2150 ) i, BND( i, 1 ), X0( i ), BND( i, 2 ), P( i )
       END DO
     END IF

!  Find the status of the variables

!DIR$ IVDEP
     DO i = 1, n

!  Check to see whether the variable is fixed

       IF ( X_status( i ) <= 2 ) THEN
         X_status( i ) = 0
         xupper = BND( i, 2 ) - X0( i ) <= epstol
         xlower = X0( i ) - BND( i, 1 ) <= epstol
         IF ( .NOT. ( xupper .OR. xlower ) ) THEN

!  The variable lies between its bounds. Check to see if the search
!  direction is zero

           IF ( ABS( P( i ) ) > epsmch ) GO TO 110
           S%nzero = S%nzero - 1
           IVAR( S%nzero ) = i

!  The variable lies close to its lower bound

         ELSE
           IF ( xlower ) THEN
             IF ( P( i ) > epsmch ) THEN
               nfreed = nfreed + 1
               GO TO 110
             END IF
             X_status( i ) = 1

!  The variable lies close to its upper bound

           ELSE
             IF ( P( i ) < - epsmch ) THEN
               nfreed = nfreed + 1
               GO TO 110
             END IF
             X_status( i ) = 2
           END IF
         END IF
       END IF

!  Set the search direction to zero

       XT( i ) = X0( i )
       P( i ) = zero
!      IF (  S%prnter  ) WRITE(  iout, 2020  ) i, zero
       CYCLE
 110   CONTINUE

!  If the variable is free, set up the pointers to the nonzeros in the vector
!  P ready for calculating Q = B * P

       S%nbreak = S%nbreak + 1
       IVAR( S%nbreak ) = i
       IF ( boundx ) S%ptp = S%ptp + P( i ) * P( i )
     END DO

!  Record the number of free variables at the starting point

     nfree = S%nbreak ; nvar2 = nfree ; qxt = f

!  If all of the variables are fixed, exit

     IF ( S%pronel ) WRITE( iout, 2070 ) nfreed, n - S%nbreak
     IF ( S%prnter ) WRITE( iout, 2110 ) nfreed, n - S%nbreak
     IF ( S%nbreak == 0 ) GO TO 600
     S%iter = 0

!  Find the breakpoints for the piecewise linear arc (the distances
!  to the boundary)

     DO j = 1, S%nbreak
       i = IVAR( j )
       IF ( P( i ) > epsmch ) THEN
         t = ( BND( i, 2 ) - X0( i ) ) / P( i )
       ELSE
         t = ( BND( i, 1 ) - X0( i ) ) / P( i )
       END IF
       BREAKP( j ) = t
     END DO

!  Order the breakpoints in increasing size using a heapsort. Build the heap

     CALL SORT_heapsort_build( S%nbreak, BREAKP, insort, INDA = IVAR )

!  Return to the main routine to evaluate Q = B * P

     jumpto = 2 ; nvar1 = 1 ; nvar2 = nfree
     RETURN

!  Calculate the function value ( QXT ), first derivative ( GXT ) and
!  second derivative ( HXT ) of the univariate piecewise quadratic
!  function at the start of the piecewise linear arc

 200 CONTINUE
!    S%gxt = DOT_PRODUCT( G( IVAR( : nfree ) ), P( IVAR( : nfree ) ) )
!    S%hxt = DOT_PRODUCT( Q( IVAR( : nfree ) ), P( IVAR( : nfree ) ) )
     S%gxt = zero ; S%hxt = zero
     DO i = 1, nfree
       S%gxt = S%gxt + G( IVAR( i ) ) * P( IVAR( i ) )
       S%hxt = S%hxt + Q( IVAR( i ) ) * P( IVAR( i ) )
     END DO

!  Start the main loop to find the first local minimizer of the piecewise
!  quadratic function. Consider the problem over successive pieces

 210 CONTINUE

!  Print details of the piecewise quadratic in the next interval

     S%iter = S%iter + 1
     IF ( S%prnter ) WRITE( iout, 2030 )                                       &
       S%iter, qxt * findmx, S%gxt * findmx, S%hxt * findmx, S%tbreak
     IF ( S%pronel ) WRITE( iout, 2080 )                                       &
       S%iter, qxt * findmx, S%gxt * findmx, S%hxt * findmx, S%tbreak

!  If the gradient of the univariate function increases, exit

     IF ( S%gxt > gzero ) GO TO 600

!  Record the value of the last breakpoint

     S%tk = S%tbreak

!  Find the next breakpoint ( end of the piece )

     S%tbreak = BREAKP(  1  )
     CALL SORT_heapsort_smallest( S%nbreak, BREAKP, insort, INDA = IVAR )

!  Compute the length of the current piece

     S%deltat = MIN( S%tbreak, tmax ) - S%tk

!  If necessary, compute the distance to the spherical boundary

     IF ( boundx ) THEN
       IF ( S%ptp > zero ) THEN
         tbndsq = SQRT( ( r ** 2 - S%tpttp ) / S%ptp )
         IF ( tbndsq < S%tbreak ) S%deltat = tbndsq - S%tk
       END IF
     END IF

!  Print details of the breakpoint

     IF ( S%prnter ) THEN
       IF ( boundx ) THEN
         WRITE( iout, 2140 ) S%tbreak, tmax, tbndsq
       ELSE
         WRITE( iout, 2040 ) S%tbreak, tmax
       END IF
     END IF

!  If the gradient of the univariate function is small and its curvature
!  is positive, exit

     IF ( ABS( S%gxt ) <= gzero ) THEN
       IF ( S%hxt > - hzero .OR. S%deltat >= HUGE( one ) ) THEN
         S%tcauch = S%tk
         GO TO 600
       ELSE
         S%tcauch = S%tbreak
       END IF
     ELSE

!  If the gradient of the univariate function is nonzero and its curvature is
!  positive, compute the line minimum

       IF ( S%hxt > zero ) THEN
         tstar = - S%gxt / S%hxt
         IF ( S%prnter ) WRITE( iout, 2050 ) tstar

!  If the line minimum occurs before the breakpoint, the line minimum gives
!  the generalized cauchy point. Exit

         S%tcauch = MIN( S%tk + tstar, S%tbreak )
         IF ( tstar < S%deltat ) THEN
           S%deltat = tstar
           GO TO 500
         END IF
       ELSE
         S%tcauch = S%tbreak
       END IF
     END IF

!  If the breakpoint occurs on the spherical boundary, exit.

     IF ( boundx ) THEN
       IF ( tbndsq <= S%tcauch ) THEN
         S%tcauch = tbndsq
         IF ( S%prnter ) WRITE( iout, 2130 )
         IF ( S%pronel ) WRITE( iout, 2130 )
         GO TO 500
       END IF
     END IF

!  If the Cauchy point occurs at tmax, exit.

     IF ( tmax < S%tcauch ) THEN
       S%tcauch = tmax
       S%deltat = tmax - S%tk
       GO TO 500
     END IF

!  Update the univariate function and gradient values

     qxt = qxt + S%deltat * ( S%gxt + half * S%deltat * S%hxt )
     S%gxtold = S%gxt ; S%gxt = S%gxt + S%deltat * S%hxt

!  Record the new breakpoint and the amount by which other breakpoints
!  are allowed to vary from this one and still be considered to be
!  within the same cluster

     feasep = S%tbreak + S%epstl2

!  Move the appropriate variable( s ) to their bound( s )

 220 CONTINUE
     ibreak = IVAR( S%nbreak )
     S%nbreak = S%nbreak - 1
     IF ( S%prnter ) WRITE( iout, 2020 ) ibreak, S%tbreak

!  Indicate the status of the newly fixed variable - the value is negated to
!  indicate that the variable has just been fixed

     IF ( P( ibreak ) < zero ) THEN
       X_status( ibreak ) = - 1
     ELSE
       X_status( ibreak ) = - 2
     END IF

!  If all of the remaining search direction is zero, return

     IF ( S%nbreak == 0 ) THEN
!DIR$ IVDEP
       DO j = 1, nvar2
         i = IVAR( j )

!  Restore X_status to its correct sign

         X_status( i ) = - X_status( i )

!  Move the variable onto its bound

         XT( i ) = BND( i, X_status( i ) )

!  Store the step from the initial point to the cauchy point in P

         P( i ) = XT( i ) - X0( i )
       END DO
       nvar2 = 0
       GO TO 600
     END IF

!  Determine if other variables hit their bounds at the breakpoint

     IF (  BREAKP(  1  ) < feasep  ) THEN
       CALL SORT_heapsort_smallest( S%nbreak, BREAKP, insort, INDA = IVAR )
       GO TO 220
     END IF

!  Return to the main routine to evaluate Q = B * P

     jumpto = 3 ; nvar1 = S%nbreak + 1
     RETURN

!  Update the first and second derivatives of the univariate function

 300 CONTINUE

!  Start with the second-order terms. Only process nonzero components of Q

!DIR$ IVDEP
     DO j = 1, nnonnz
       i = INONNZ( j )
       qipi = Q( i ) * P( i )
       IF ( X_status( i ) == 0 ) THEN

!  Include contributions from the free components of Q

         S%gxt = S%gxt - qipi * S%tbreak
         S%hxt = S%hxt - qipi * two
       ELSE
         IF ( X_status( i ) < 0 ) THEN

!  Include contributions from the components of Q which were just fixed

           S%gxt = S%gxt - qipi * S%tbreak
           S%hxt = S%hxt - qipi
         ELSE

!  Include contributions from the components of Q which were previously fixed

           S%gxt = S%gxt - qipi
         END IF
       END IF
     END DO

!  Now include the contributions from the variables which have just been fixed

!DIR$ IVDEP
     DO j = nvar1, nvar2
       i = IVAR( j )
       S%gxt = S%gxt - P( i ) * G( i )

!  Continue updating the distances to the spherical boundary

       IF ( boundx ) THEN
         S%tpttp = S%tpttp + ( S%tbreak * P( i ) ) ** 2
         S%ptp = S%ptp - P( i )  ** 2
       END IF

!  Restore X_status to its correct sign

       X_status( i ) = - X_status( i )

!  Move the variable onto its bound

       XT( i ) = BND( i, X_status( i ) )

!  Store the step from the initial point to the Cauchy point into P

       P( i ) = XT( i ) - X0( i )
     END DO

!  Compute the square of the distance to the current point

     IF ( boundx ) dxsqr = S%tpttp + S%ptp * S%tcauch ** 2

!  Reset the number of free variables

     nvar2 = S%nbreak

!  Check that the size of the line gradient has not shrunk significantly in
!  the current segment of the piecewise arc. If it has, there may be a loss
!  of accuracy, so the line derivatives will be recomputed

     S%recomp = ABS( S%gxt ) < - S%epsqrt * S%gxtold

!  If required, compute the true line gradient and curvature.
!  Firstly, compute the matrix-vector product B * P

     IF ( S%recomp .OR. S%prnter ) THEN
       jumpto = 4 ; nvar1 = 1
       RETURN
     END IF

!  Calculate the line gradient and curvature

 400 CONTINUE
     IF ( S%recomp .OR. S%prnter ) THEN
       pbp = zero ; gp = zero
       IF ( idebug > 100 .AND. iout > 0 ) WRITE( iout, 2100 )                  &
          ( IVAR( i ), P( IVAR( i ) ), i = 1, nvar2 )
       DO j = 1, nvar2
         i = IVAR( j )
         qipi = P( i ) * Q( i )
         pbp = pbp + qipi ; gp = gp + P( i ) * G( i ) + S%tbreak * qipi
       END DO
!      gp = gp + DOT_PRODUCT( P( IVAR( nvar2 + 1 : nfree ) ),                  &
!                             Q( IVAR( nvar2 + 1 : nfree ) ) )
       DO i = nvar2 + 1, nfree
         gp = gp + P( IVAR( i ) ) * Q( IVAR( i ) )
       END DO
       IF ( S%prnter ) WRITE( iout, 2090 ) gp  * findmx, pbp * findmx,         &
                                           S%gxt * findmx, S%hxt * findmx
       IF ( S%recomp ) THEN
         S%gxt = gp ; S%hxt = pbp
       END IF
     END IF

!  Jump back to calculate the next breakpoint

     GO TO 210

!  Step to the generalized Cauchy point

 500 CONTINUE

!  Calculate the function value for the piecewise quadratic

     qxt = qxt + S%deltat * ( S%gxt + half * S%deltat * S%hxt )
     IF ( S%pronel ) WRITE( iout, 2160 ) S%iter + 1, qxt * findmx, S%tcauch

!  If necessary, update the distances to the spherical boundary

     IF ( boundx ) dxsqr = S%tpttp + S%ptp * S%deltat ** 2
     IF ( S%prnter ) WRITE( iout, 2060 ) qxt * findmx

!  The generalized Cauchy point has been found. Set the array P to the step
!  from the initial point to the Cauchy point

 600 CONTINUE
     P( IVAR( : nvar2 ) ) = S%tcauch * P( IVAR( : nvar2 ) )
     XT( IVAR( : nvar2 ) ) = X0( IVAR( : nvar2 ) ) + P( IVAR( : nvar2 ) )

!  Record that variables whose gradients were zero at the initial
!  point are free variables

     DO j = S%nzero, n
       nfree = nfree + 1
       IVAR( nfree ) = IVAR( j )
     END DO

!  Set return conditions

     jumpto = 0 ; nvar1 = 1 ; nvar2 = nfree

     RETURN

!  Non-executable statements

 2010  FORMAT( / ' ------------ CAUCHY_get_exact_gcp entered -------------' )
 2020  FORMAT(   ' Variable ', I4, ' is fixed, step = ', ES12.4 )
 2030  FORMAT( /, ' Piece', I5, ': f, G & H at start point', 4ES11.3 )
 2040  FORMAT( /, ' Next break point = ', ES12.4,                              &
           /, ' Maximum step     = ', ES12.4 )
 2050  FORMAT(    ' Stationary point = ', ES12.4 )
 2060  FORMAT( /, ' Function value at the Cauchy point ', ES12.4 )
 2070  FORMAT( /, '  ',I7,' vars. freed ', I7, ' vars. remain fixed' )
 2080  FORMAT( 24X, I7, 4ES12.4 )
 2090  FORMAT( /, ' Calculated gxt and hxt = ', 2ES12.4, /,                    &
              ' Recurred   gxt and hxt = ', 2ES12.4 )
 2100  FORMAT(    ' Current search direction ', /, ( 4( I6, ES12.4 ) ) )
 2110  FORMAT( /, I8, ' variables freed from their bounds ', /, I8,            &
              ' variables remain fixed ',/ )
 2120  FORMAT( /, '    ** CAUCHY entered ** Segment    Model   ',              &
              '   Gradient   Curvature     Step ' )
 2130  FORMAT( /, '       Spherical trust region encountered ', / )
 2140  FORMAT( /, ' Next break point = ', ES12.4,                              &
          /, ' Maximum step     = ', ES12.4, /, ' Spherical bound  = ', ES12.4 )
 2150  FORMAT( ' Var low X up P ', I6, 4ES12.4 )
 2160  FORMAT( 24X, I7, ES12.4, 24X, ES12.4 )

!  End of subroutine CAUCHY_get_exact_gcp

     END SUBROUTINE CAUCHY_get_exact_gcp

!-*-*-  L A N C E L O T  -B-  CAUCHY_get_approx_gcp  S U B R O U T I N E  -*-*

     SUBROUTINE CAUCHY_get_approx_gcp( n, X0, XT, G, BND, X_status, f, epstol, &
                                       tmax, rmu, boundx, r, qxt, P, Q, IVAR,  &
                                       nfree, nvar1, nvar2, iout, jumpto,      &
                                       idebug, findmx, BREAKP, GRAD, S )

!  Find a suitable approximation to the Generalized Cauchy Point (GCP)
!  for a given quadratic function within a box shaped region

!  If we define the 'Cauchy arc' X(t) = projection of X0 + t*P into the box
!  region BND(*,1) <= X(*) <= BND(*,2), the GCP is the first local minimizer
!  of the quadratic function quad( X ) =

!     0.5 (X-X0) (transpose ) B (X-X0) + G (transpose) (X-X0) + F

!  for points lying on X(t), with t >= 0. A suitable Cauchy point is
!  determined as follows:

!  1) If the minimizer of quad( X ) along X0 + T * p lies on the Cauchy
!     arc, this is the required point. Otherwise,

!  2) Starting from some specified T0, construct a decreasing sequence
!     of values T1, T2, T3, .... . Given 0 < MU < 1, pick the first
!     TI (I = 0, 1, ...) for which the Armijo condition

!        quad( X(TI) ) <= linear( X(TI), MU ) =
!                         F + MU * G (transpose) ( X(TI) - X0 )

!     is satisfied. X0 + TI * P is then the required point

!  Progress through the routine is controlled by the parameter JUMPTO

!  If JUMPTO = 0, the GCP has been found
!  If JUMPTO = 1, an initial entry has been made
!  If JUMPTO = 2, 3, 4 the vector Q = B * P is required

!  The value of the array X_status gives the status of the variables

!  IF X_status( I ) = 0, the I-TH variable is free
!  IF X_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF X_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF X_status( I ) = 3, the I-th variable is permanently fixed
!  IF X_status( I ) = 4, the I-th variable is fixed at some other value

!  The addresses of the free variables are given in the first NVAR entries
!  of the array IVAR

!  If the product B * P is required (JUMPTO = 2,3,4), the nonzero components
!  of P occur in positions IVAR(I) for I lying between NVAR1 and NVAR2

!  At the initial point, variables within EPSTOL of their bounds and
!  for which the search direction P points out of the box will be fixed

!  ------------------------- dummy arguments --------------------------

!  N      (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  X0     (REAL array of length at least N) the point x0 from which the Cauchy
!          arc commences. ** this variable is not altered by the subroutine
!  XT     (REAL array of length at least N) the current estimate of the GCP
!  G      (REAL array of length at least N) the coefficients of
!          the linear term in the quadratic function
!          ** this variable is not altered by the subroutine
!  BND    (two dimensional REAL array with leading dimension N and second
!          dimension 2) the lower (BND(*,1)) and upper (BND(*,2)) bounds on
!          the variables. ** this variable is not altered by the subroutine
!  X_status (INTEGER array of length at least N) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          X_status should be set as follows:
!          If X_status( I ) = 0, the I-th variable is free
!          If X_status( I ) = 1, the I-th variable is on its lower bound
!          If X_status( I ) = 2, the I-th variable is on its upper bound
!          If X_status( I ) = 3, 4, the I-th variable is fixed at XT(I)
!  F      (REAL) the value of the quadratic at X0, see above.
!          ** this variable is not altered by the subroutine
!  EPSTOL (REAL) a tolerance on feasibility of X0, see above.
!          ** this variable is not altered by the subroutine.
!  TMAX   (REAL) the largest allowable value of T
!  RMU    (REAL) the slope of the majorizing linear model linear(X,MU)
!  BOUNDX (LOGICAL) the search for the generalized cauchy point will be
!          terminated on the boundary of the spherical region ||X-X0|| <= R
!          if and only if BOUNDX is set to .TRUE. on initial (JUMPTO=1) entry.
!          ** this variable is not altered by the subroutine
!  r       (REAL) the radius, R, of the spherical region. r need not be
!          set if boundx is .FALSE. on initial entry.
!          ** This variable is not altered by the subroutine
!  QXT    (REAL) the value of the piecewise quadratic function at the current
!          estimate of the GCP
!  P      (REAL array of length at least N) contains the values of the
!          components of the vector P. On initial (JUMPTO=1) entry, P must
!          contain the initial direction of the 'Cauchy arc'. On a non optimal
!          exit, (JUMPTO=2,3,4), P is the vector for which the product B * P
!          is required before the next re-entry. On a terminal exit (JUMPTO=0),
!          P contains the step XT - X0. The components IVAR(I) = NVAR1, ... ,
!          NVAR2 of P contain the values of the nonzero components of P
!          (see, IVAR, NVAR1, NVAR2)
!  Q      (REAL array of length at least N) on a non initial entry
!         (JUMPTO=2,3,4), Q must contain the vector B * P. Only the
!          components IVAR(I), I=1,...,NFREE, of Q need be set (the other
!          components are not used)
!  IVAR   (INTEGER array of length at least N) on all normal exits
!         (JUMPTO=0,2,3,4), IVAR(I), I = NVAR1, ..., NVAR2, gives the indices
!          of the nonzero components of P
!  NFREE  (INTEGER) the number of free variables at the initial point
!  NVAR1  (INTEGER) see IVAR, above
!  NVAR2  (INTEGER) see IVAR, above
!  IOUT   (INTEGER) the fortran output channel number to be used
!  JUMPTO (INTEGER) controls flow through the subroutine.
!          If JUMPTO = 0, the GCP has been found
!          If JUMPTO = 1, an initial entry has been made
!          If JUMPTO = 2, 3, 4, the vector Q = B * P is required
!  IDEBUG (INTEGER) allows detailed printing. If IDEBUG is larger than 4,
!          detailed output from the routine will be given. Otherwise, no
!          output occurs
!  findmx (REAL) when printing the value of the objective function,
!          the value calculated in fmodel will be multiplied by the
!          scale factor findmx. This allows the user, for instance,
!          to find the maximum of a quadratic function F, by minimizing
!          the function - F, while monitoring the progress as
!          if a maximization were actually taking place, by setting
!          findmx to - 1.0. Normally findmx should be set to 1.0
!  BREAKP, GRAD (REAL) workspace that must be preserved between calls
!  S      (CAUCHY_save_type) private data that must be preserved between calls

!  ------------------ end of dummy arguments --------------------------

     INTEGER, INTENT( IN    ):: n, iout, idebug
     INTEGER, INTENT( INOUT ):: nfree, nvar1, nvar2, jumpto
     REAL ( KIND = wp ), INTENT( IN    ):: tmax, r, rmu, findmx, epstol, f
     REAL ( KIND = wp ), INTENT( INOUT ):: qxt
     LOGICAL, INTENT( IN ) :: boundx
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: X_status
     INTEGER, DIMENSION( n ), INTENT( INOUT ) :: IVAR
     REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n, 2 ) :: BND
     REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n ) :: X0, G
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: XT
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: P, Q
     REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: BREAKP, GRAD
     TYPE( CAUCHY_save_type ), INTENT( INOUT ) :: S

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      N, X0, G, D, BND, F, EPSTOL, TMAX, RMU, BOUNDX, IOUT, JUMPTO, IDEBUG

!  JUMPTO must have the value 1. In addition, if the I-th variable is required
!  to be fixed at its initial value, X0(I), X_status(I) must be set to
!  3 or 4. r must be specified if BOUNDX is .TRUE. on initial entry

!  RE-ENTRY:

!  If the variable JUMPTO has the value 2, 3 or 4 on exit, the
!  subroutine must be re-entered with the vector Q containing
!  the product of the second derivative matrix B and thne output
!  vector P. All other parameters MUST NOT BE ALTERED

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

     INTEGER :: i , j, nbreak, n3
     REAL ( KIND = wp ) :: epsmch, ptbp, tbmax, tspher
     LOGICAL :: xlower, xupper

     epsmch = EPSILON( one )

     SELECT CASE ( jumpto )
     CASE ( 1 ) ; GO TO 100
     CASE ( 2 ) ; GO TO 200
     CASE ( 3 ) ; GO TO 300
     CASE ( 4 ) ; GO TO 400
     END SELECT

!  On initial entry, set constants

 100 CONTINUE
     S%prnter = idebug >= 4 .AND. iout > 0
     S%pronel = idebug == 2 .AND. iout > 0
     S%iterca = 0 ; S%itmax = 100
     S%nfreed = 0 ; nbreak = 0 ; S%nzero = n + 1
     S%t = zero
     IF ( boundx ) S%ptp = zero

!  Find the status of the variables

!DIR$ IVDEP
     DO i = 1, n

!  Check to see whether the variable is fixed

       IF ( X_status( i ) <= 2 ) THEN
         X_status( i ) = 0
         xupper = BND( i, 2 ) - X0( i ) <= epstol
         xlower = X0( i ) - BND( i, 1 ) <= epstol

!  The variable lies between its bounds. Check to see if the search direction
!  is zero

         IF ( .NOT. ( xupper .OR. xlower ) ) THEN
           IF ( ABS( P( i ) ) > epsmch ) GO TO 120
           S%nzero = S%nzero - 1
           IVAR( S%nzero ) = i
!
!  The variable lies close to its lower bound
!
         ELSE
           IF ( xlower ) THEN
             IF ( P( i ) > epsmch ) THEN
               S%nfreed = S%nfreed + 1
               GO TO 120
             END IF
             X_status( i ) = 1

!  The variable lies close to its upper bound

           ELSE
             IF ( P( i ) < - epsmch ) THEN
               S%nfreed = S%nfreed + 1
               GO TO 120
             END IF
             X_status( i ) = 2
           END IF
         END IF
       END IF

!  Set the search direction to zero

       P( i ) = zero
       XT( i ) = X0( i )
       CYCLE

!  If the variable is free, set up the pointers to the nonzeros in the vector
!  P ready for calculating Q = B * P. The vector XT is used temporarily to
!  store the original input direction P

 120   CONTINUE
       nbreak = nbreak + 1
       IVAR( nbreak ) = i
       XT( i ) = P( i )
       IF ( boundx ) S%ptp = S%ptp + P( i ) * P( i )
     END DO

!  If all of the variables are fixed, exit

     IF ( S%prnter ) WRITE( iout, 2020 ) S%nfreed, n - nbreak
     IF ( S%pronel .OR. S%prnter )                                             &
        WRITE( iout, 2000 ) S%iterca, zero, f * findmx 
     S%iterca = S%iterca + 1
     nfree = nbreak
     IF ( nbreak == 0 ) GO TO 600

!  Return to the calling program to calculate Q = B * P

     nvar1 = 1 ; nvar2 = nbreak ; jumpto = 2 ; RETURN

!  Compute the slope and curvature of quad( X ) in the direction P

 200 CONTINUE
     S%gtp = zero ; ptbp = zero
     S%tbreak = HUGE( one ) ; tbmax = zero
!DIR$ IVDEP
     DO j = 1, nfree
       i = IVAR( j )
       S%gtp = S%gtp + P( i ) * G( i )
       ptbp = ptbp + P( i ) * Q( i )

!  Find the breakpoints for the piecewise linear arc (the distances to the
!  boundary)

       IF ( P( i ) > zero ) THEN
         BREAKP( i ) = ( BND( i, 2 ) - X0( i ) ) / P( i )
       ELSE
         BREAKP( i ) = ( BND( i, 1 ) - X0( i ) ) / P( i )
       END IF

!  Compute the maximum feasible distance, TBREAK, allowable in the direction
!  P. Also compute TBMAX, the largest breakpoint

       S%tbreak = MIN( S%tbreak, BREAKP( i ) )
       tbmax = MAX( tbmax, BREAKP( i ) )
     END DO
     IF ( boundx ) tspher = r / SQRT( S%ptp )

!  Check that the curvature is positive

     IF ( ptbp > zero ) THEN

!  Compute the minimizer, T, of quad(  X( T )  ) in the direction P

       S%t = - S%gtp / ptbp

!  If the spherical trust region cuts off this point, prepare to exit

       IF ( boundx ) S%t = MIN( S%t, tspher )

!  Compare the values of T and TBREAK. If the calculated minimizer or the
!  boundary of the spherical trust region is the Cauchy point, exit

       IF ( S%t <= S%tbreak ) THEN
         GO TO 500
       ELSE
         IF ( S%pronel .OR. S%prnter ) WRITE( iout, 2040 ) S%iterca, S%t

!  Ensure that the initial value of T for backtracking is no larger than
!  ALPHA times the step to the first line minimum

         S%tamax = MIN( tmax, alpha * S%t )
       END IF
     ELSE

!  If the boundary of the spherical trust region is the Cauchy point, exit

       IF ( boundx ) THEN
         IF ( tspher <= S%tbreak ) THEN
           S%t = tspher
           GO TO 500
         END IF
       END IF
     END IF

!  -----------------------
!  The remaining intervals
!  -----------------------

!  The calculated minimizer is infeasible; prepare to backtrack from T until
!  a Cauchy point is found

     S%t = MIN( S%tamax, tbmax )

!  Calculate P, the difference between the projection of the point
!  X( T ) and X0, and PTP, the square of the norm of this distance

 250 CONTINUE
     IF ( boundx ) S%ptp = zero
!DIR$    IVDEP
     DO j = 1, nfree
       i = IVAR( j )
       P( i ) = MAX( MIN( X0( i ) + S%t * P( i ), BND( i, 2 ) ),               &
                                                  BND( i, 1 ) ) - X0( i )
       IF ( boundx ) S%ptp = S%ptp + P( i ) * P( i )
     END DO

!  If required, ensure that X( T ) lies within the spherical trust region.
!  If X( T ) lies outside the region, reduce T

     IF ( boundx ) THEN
       IF ( S%ptp > r * r ) THEN
         S%t = beta * S%t
         GO TO 250
       END IF
     END IF

!  Return to the calling program to calculate Q = B * P

     jumpto = 3 ; RETURN

!  Compute the slope and curvature of quad( X ) in the direction P

 300 CONTINUE
!    S%gtp  = DOT_PRODUCT( P( IVAR( : nfree ) ), G( IVAR( : nfree ) ) )
!    ptbp = DOT_PRODUCT( P( IVAR( : nfree ) ), Q( IVAR( : nfree ) ) )
     S%gtp = zero ; ptbp = zero
     DO i = 1, nfree
        S%gtp = S%gtp + G( IVAR( i ) ) * P( IVAR( i ) )
        ptbp = ptbp + Q( IVAR( i ) ) * P( IVAR( i ) )
     END DO


!  Form the gradient at the point X( T )

     GRAD( IVAR( : nfree ) ) = Q( IVAR( : nfree ) ) + G( IVAR( : nfree ) )

!  Evaluate quad( X( T ) ) and linear( X( T ), MU )

     qxt = f + S%gtp + half * ptbp ; S%flxt = f + rmu * S%gtp

!  --------------------------------
!  Start of the main iteration loop
!  --------------------------------

 350 CONTINUE
     S%iterca = S%iterca + 1

!  Print details of the current point

     IF ( S%pronel .OR. S%prnter ) WRITE( iout, 2010 )                         &
       S%iterca, S%t, qxt * findmx, S%flxt * findmx

!  Compare quad( X( T ) ) with linear( X( T ), MU ). If X( T ) satisfies the
!  Armijo condition and thus qualifies as a Cauchy point, exit

     IF ( S%iterca > S%itmax .OR. qxt <= S%flxt ) THEN
!DIR$ IVDEP
       DO j = 1, nfree
         i = IVAR( j )
         XT( i ) = X0( i ) + MIN( S%t, BREAKP( i ) ) * XT( i )
       END DO
       GO TO 600
     END IF

!  X( T ) is not acceptable. Reduce T

     S%tnew = beta * S%t

!  Compute P = X( tnew ) - X( t )

!DIR$ IVDEP
     DO j = 1, nfree
       i = IVAR( j )
       P( i ) = ( MIN( S%tnew,                                                 &
                  BREAKP( i ) ) - MIN( S%t, BREAKP( i ) ) ) * XT( i )
     END DO

!  Return to the calling program to calculate Q = B * P

     jumpto = 4 ; RETURN

!  Compute the slope and curvature of quad( X ) in the direction P

 400 CONTINUE
!    S%g0tp = DOT_PRODUCT( P( IVAR( : nfree ) ), G   ( IVAR( : nfree ) ) )
!    S%gtp  = DOT_PRODUCT( P( IVAR( : nfree ) ), GRAD( IVAR( : nfree ) ) )
!    ptbp = DOT_PRODUCT( P( IVAR( : nfree ) ), Q   ( IVAR( : nfree ) ) )
     S%g0tp = zero ; S%gtp = zero ; ptbp = zero
     DO i = 1, nfree
       S%g0tp = S%g0tp + G   ( IVAR( i ) ) * P( IVAR( i ) )
       S%gtp  = S%gtp  + GRAD( IVAR( i ) ) * P( IVAR( i ) )
       ptbp = ptbp + Q   ( IVAR( i ) ) * P( IVAR( i ) )
     END DO

!  Update the existing gradient to find that at the point X( tnew )

     GRAD( IVAR( : nfree ) ) = GRAD( IVAR( : nfree ) ) + Q( IVAR( : nfree ) )

!  Evaluate quad( X( T ) ) and linear( X( T ), MU )

     qxt = qxt + S%gtp + half * ptbp ; S%flxt = S%flxt + rmu * S%g0tp
     S%t = S%tnew
     GO TO 350

!  ------------------------------
!  End of the main iteration loop
!  ------------------------------

!  The Cauchy point occured in the first interval. Record the point and the
!  value of the quadratic at the point

 500 CONTINUE
     qxt = f + S%t * ( S%gtp + half * S%t * ptbp )
     XT( IVAR( : nfree ) ) = X0( IVAR( : nfree ) ) + S%t * P( IVAR( : nfree ) )

!  Print details of the Cauchy point

     IF ( S%pronel .OR. S%prnter )                                             &
       WRITE( iout, 2030 ) S%iterca, S%t, qxt * findmx

!  An approximation to the generalized Cauchy point has been found. Set the
!  array P to the step from the initial point to the Cauchy point

 600 CONTINUE
     n3 = 0
!DIR$ IVDEP
     DO j = 1, nfree
       i = IVAR( j )
       P( i ) = XT( i ) - X0( i )

!  Find which variables are free at X( T )

       IF ( S%t <= BREAKP( i ) ) THEN
         n3 = n3 + 1
       ELSE
         IF ( P( i ) < zero ) X_status( i ) = 1
         IF ( P( i ) > zero ) X_status( i ) = 2

!  Move the fixed variables to their bounds

         IF ( P( i ) /= zero ) XT( i ) = BND( i, X_status( i ) )
       END IF
     END DO
     IF ( S%pronel ) WRITE( iout, 2050 ) S%nzero - n3 - 1

!  Record that variables whose gradients were zero at the initial point are
!  free variables

     DO j = S%nzero, n
       nfree = nfree + 1
       IVAR( nfree ) = IVAR( j )
     END DO

!  Set return conditions

     nvar1 = 1 ; nvar2 = nfree ; jumpto = 0
     RETURN

!  Non-executable statements

 2000  FORMAT( /, 3X, ' ** CAUCHY  entered  iter     step     ',               &
           '  Q( step )   L( step,mu )', /, 21X, I6, 2ES12.4 )
 2010  FORMAT( 21X, I6, 3ES12.4 )
 2020  FORMAT( /, ' ----------- CAUCHY_get_approx_gcp -------------', //,      &
           I8, ' variables freed from their bounds ', /, I8,                   &
           ' variables remain fixed ', / )
 2030  FORMAT( 21X, I6, 2ES12.4 )
 2040  FORMAT( 21X, I6, '  1st line mimimizer infeasible. Step = ', ES10.2 )
 2050  FORMAT( 27X, I6, ' variables are fixed ' )

!  End OF subroutine CAUCHY_get_approx_gcp

     END SUBROUTINE CAUCHY_get_approx_gcp

!  End of module LANCELOT_CAUCHY

   END MODULE LANCELOT_CAUCHY_double
