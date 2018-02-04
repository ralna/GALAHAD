! THIS VERSION: GALAHAD 2.1 - 22/03/2007 AT 09:00 GMT.

!-*-*-*-*-*-  L A N C E L O T  -B-  OTHERS  M O D U L E  *-*-*-*-*-*-*-*

!  Nick Gould, for GALAHAD productions
!  Copyright reserved
!  January 29th 1995

   MODULE LANCELOT_CG_double

      IMPLICIT NONE

      PRIVATE
      PUBLIC :: CG_save_type, CG_solve

!  Set precision

      INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )

!  Set other parameters

      REAL ( KIND = wp ), PARAMETER :: zero = 0.0_wp
      REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp
      REAL ( KIND = wp ), PARAMETER :: two = 2.0_wp
      REAL ( KIND = wp ), PARAMETER :: half = 0.5_wp
      REAL ( KIND = wp ), PARAMETER :: ten = 10.0_wp
      REAL ( KIND = wp ), PARAMETER :: infinity = HUGE( one ) / ten

!  =============================
!  The CG_save_type derived type
!  =============================

     TYPE :: CG_save_type
       INTEGER :: iter, itsle
       REAL ( KIND = wp ) :: alpha, oldgns, onepep
       LOGICAL :: prnter, pronel
     END TYPE CG_save_type

   CONTAINS

!-*-*-*-*-  L A N C E L O T  -B-    CG_solve   S U B R O U T I N E   -*-*-*-*

      SUBROUTINE CG_solve( n , X0, XT, G , BND, nbnd, VARIABLE_STATUS,         &
                           epsgrd, fmodel, GRAD  , status, P , Q ,             &
                           IVAR  , nvar  , nresrt, twonrm, r,  nobnds,         &
                           gnrmsq, dxsqr , iout  , jumpto, idebug, findmx,     &
                           itercg, itcgmx, ifixed, DIST_bound, S, XSCALE )

!  Find an approximation to the minimizer of the quadratic function

!     0.5 (X-X0) (transpose ) B (X-X0) + G (transpose) (X-X0) + F

!  within the box region BND(*,1) <= X(*) <= BND(*,2) subject to the further
!  restriction that specified variables remain fixed at their initial values

!  ** Version to allow restarts when bounds are encountered

!  Optionally, the search may be terminated at the first point encountered on
!  the boundary of the spherical region ||X - X0|| <= R, where ||.|| denotes
!  the 2-norm. Furthermore, the minimization may continue along the edges of
!  the box if so desired

!  Control is passed from the routine whenever a product of the vector P with
!  B is required, and the user is responsible for forming the product in the
!  vector Q. Control is also passed to the user when the 'preconditioned'
!  gradient W ** (-1) GRAD is to be formed in Q. Here, W is any positive
!  definite and symmetric approximation to the Hessian B

!  Progress through the routine is controlled by the parameter JUMPTO

!  If JUMPTO = 0, no further entry is called for
!  If JUMPTO = 1, an initial entry has been made
!  If JUMPTO = 2, the vector Q = W ** (-1) * GRAD is required
!  If JUMPTO = 3, the vector Q = B * P is required
!  If JUMPTO = 4, the norm of the gradient of the current iterate is smaller
!                 than epsgrd. The user may wish to perform additional
!                 convergence tests in the calling program and re-enter cg
!                 with a smaller value of EPSGRD. If such a re-entry is
!                 required, JUMPTO should not be altered. Otherwise, JUMPTO
!                 should be reset to 0
!  If JUMPTO = 5, an edge of the box has been encountered. If the user wishes
!                 to continue the minimization along the edge, JUMPTO should
!                 be reset to 2 and the the vector Q calculated as above
!                 before re-entry. Otherwise, JUMPTO should be reset to 0

!  The value of the array VARIABLE_status gives the status of the variables

!  IF VARIABLE_status( I ) = 0, the I-TH variable is free
!  IF VARIABLE_status( I ) = 1, the I-th variable is fixed on its lower bound
!  IF VARIABLE_status( I ) = 2, the I-th variable is fixed on its upper bound
!  IF VARIABLE_status( I ) = 3, the I-th variable is permanently fixed
!  IF VARIABLE_status( I ) = 4, the I-th variable is fixed at some other value

!  The addresses of the free variables are given in the first NVAR entries
!  of the array IVAR

!  If the product W ** (-1) * GRAD is required (JUMPTO = 2), the nonzero
!  components of GRAD occur in positions IVAR(I) for I lying between one and
!  NVAR and have the values GRAD(I)

!  If the product B * P is required (JUMPTO = 3), the nonzero components of P
!  occur in positions IVAR(I) for I lying between 1 and NVAR

!  THE PRECONDITIONER:

!  When JUMPTO = 2, the user is asked to provide the product of the inverse
!  of a 'preconditioning' matrix W and the vector GRAD. The matrix W should
!  have the following properties:

!  a) W must be symmetric and positive definite.
!  b) W should be an approximation to the second derivative matrix
!     B in the sense that the eigenvalues of W ** (-1) B should be close to
!     unity whenever B is positive definite. Ideally, these eigenvalues should
!     be clustered about a small number of distinct values.
!  c) W should have the property that the product W ** (-1) * GRAD is cheap to
!     obtain. By this we mean that the amount of arithmetic work involved in
!     forming the product should be a small multiple of N.

!  Popular preconditioners include the diagonal preconditioner, in which W is
!  a diagonal matrix with diagonal values equal to those of B (modified to be
!  positive if necessary) and the incomplete factorization preconditioner, in
!  which elements of B are changed to zeros in W in order to reduce the number
!  of nonzeros in the cholesky factorization of W. Unfortunately, the choice
!  of a good preconditioner is problem dependent

!  ------------------------- dummy arguments --------------------------

!  n      (INTEGER) the number of independent variables.
!          ** this variable is not altered by the subroutine
!  X0     (REAL array of length at least n) the point x0.
!          ** this variable is not altered by the subroutine
!  XT     (REAL array of length at least n) the current estimate
!          of the minimizer for the problem. XT should be initialized
!          at the user's estimate of the minimizer and must satisfy
!          the restriction BND(1,i) <= XT(i) <= BND(2,i), i=1,...,n
!  G      (REAL array of length at least n) the coefficients of
!          the linear term in the quadratic function
!          ** this variable is not altered by the subroutine
!  BND    (two dimensional REAL array with leading dimension nbnd and second
!          dimension 2) the lower (BND(*,1)) and upper (BND(*,2)) bounds on
!          the variables. ** this variable is not altered by the subroutine
!  nbnd   (INTEGER) leading dimension of BND
!         ** this variable is not altered by the subroutine
!  VARIABLE_status (INTEGER array of length at least n) specifies which
!          of the variables are to be fixed at the start of the minimization.
!          VARIABLE_status should be set as follows:
!          If VARIABLE_status( i ) = 0, the i-th variable is free
!          If VARIABLE_status( i ) = 1, the i-th variable is on its lower bound
!          If VARIABLE_status( i ) = 2, the i-th variable is on its upper bound
!          If VARIABLE_status( i ) = 3, 4, the i-th variable is fixed at XT(i)
!  epsgrd (REAL) the minimization will continue until the norm of the scaled
!          gradient is less than sqrt(epsgrd).
!          ** this variable is not altered by the subroutine
!  fmodel (REAL) the value of the quadratic function at the current
!          estimate of the minimizer. fmodel should be initialized at
!          the value of the quadratic at the initial point
!  XSCALE (REAL array of length n) the scale factors used in computing the
!          norm of the scaled gradient, gnorm =
!          SQRT( SUM(from i=1)(to nvar) (GRAD(i) * XSCALE(IVAR(i)))**2 ).
!          ** this variable is not altered by the subroutine
!  GRAD   (REAL array of length n) the value of nonzero components of the
!          gradient of the quadratic function at the current estimate of the
!          minimizer. The component GRAD(i), i = 1,.., nvar, gives the value
!          of the IVAR(I)-th component of the gradient of the quadratic
!          function (see IVAR, nvar)
!  status (INTEGER) the value of status on terminal (jumpto=0) exit
!          describes the progress of the algorithm as follows:
!          If status = 10, the output value XT contains the required
!                          minimizer (see epsgrd)
!          If status = 11, too many iterations have been performed
!                          by the routine without solving the problem.
!                          The best estimate of the solution is given in XT
!          If status = 12, a free variable has encountered one of its bounds,
!                          or the trust-region boundary has been encountered.
!                          The best estimate of the solution found to date is
!                          given in XT
!          If status = 13, a direction of negative curvature has been
!                          determined and a step to one of the bounds has been
!                          taken. The best estimate of the solution obtained
!                          is given in XT
!          If status = 14, the last step taken is so small that further
!                          progress is unlikely. The best estimate of the
!                          solution is given in XT
!  P      (REAL array of length at least n) contains the values of the
!          components of the vector P. On an initial entry (jumpto=1),
!          the components of P must be set to zero. On a non optimal exit
!          (jumpto=3), P is the vector for which the product B*P is
!          required before the next re-entry. The components IVAR(i),
!          i = 1, ... , nvar of P contain the values of the nonzero components
!          of P (see, IVAR, nvar)
!  Q      (REAL array of length at least n) on the initial entry (jumpto=1),
!          Q must contain the product B * ( XT - X0 ). Q must contain the
!          vector W ** (-1) *  P (jumpto=2) or B * P (jumpto=3), on a non
!          initial entry. For non initial entries, only the components with
!          indices IVAR(i) i=1,..,nvar need be set (the other components are
!          not used)
!  IVAR   (INTEGER array of length at least n) on all normal exits
!         (jumpto=0,2,3), IVAR(i), i=1,...,nvar, gives the indices of the
!          nonzero components of P and GRAD
!  nvar   (INTEGER) see IVAR, above
!  nresrt (INTEGER) the search for the minimizer will be restarted in
!          the preconditioned steepest-descent direction every nresrt
!          iterations. ** this variable is not altered by the subroutine
!  twonrm (LOGICAL) the search for the minimizer will be terminated on the
!          boundary of the spherical (2-norm) region ||X-X0||_2 <= R if and 
!          only if twonrm is set to .TRUE. on initial (jumpto=1) entry.
!          ** this variable is not altered by the subroutine
!  r      (REAL) the radius, R, of the spherical region. r need not be
!          set if twonrm is .FALSE. on initial entry.
!          ** This variable is not altered by the subroutine
!  nobnds (LOGICAL) there are no bounds on the variables.
!          ** this variable is not altered by the subroutine
!  gnrmsq (REAL) the norm of the preconditioned gradient at the current
!          estimate of the minimizer
!  dxsqr  (REAL) the square of the two norm of the distance between the
!          current estimate of the minimizer and X0. dxsqr will only be set if
!          twonrm is .TRUE.
!  iout   (INTEGER) the standard fortran output unit to be used
!  jumpto (INTEGER) controls flow through the subroutine.
!          If jumpto = 0, no further entry is called for
!          If jumpto = 1, an initial entry has been made
!          If jumpto = 2, the vector Q = W ** (-1) GRAD is required
!          If jumpto = 3, the vector Q = B * P is required
!          If jumpto = 4, the norm of the gradient of the current iterate is
!                         smaller than epsgrd. The user may wish to perform
!                         additional convergence tests in the calling program
!                         and re-enter CG with a smaller value of epsgrd. If
!                         such a re-entry is required, jumpto should not be
!                         altered. Otherwise, jumpto should be reset to 0
!          If jumpto = 5, an edge of the box has been encountered. If the
!                         user wishes to continue the minimization along the
!                         edge, jumpto should be reset to 2 and the the vector
!                         Q calculated as above before re-entry. Otherwise,
!                         jumpto should be reset to 0
!  idebug (INTEGER) allows detailed printing. if idebug is larger than 4,
!          detailed output from the routine will be given. Otherwise, no
!          output occurs
!  findmx (REAL) when printing the value of the objective function,
!          the value calculated in fmodel will be multiplied by the
!          scale factor findmx. This allows the user, for instance,
!          to find the maximum of a quadratic function F, by minimizing
!          the function - F, while monitoring the progress as
!          if a maximization were actually taking place, by setting
!          findmx to - 1.0. Normally findmx should be set to 1.0
!  itercg (INTEGER) gives the number of conjugate gradient iterations taken
!  itcgmx (INTEGER) gives the maximum number of conjugate gradient 
!         iterations permitted
!  ifixed (INTEGER) gives the variable which most recently hits a bound
!  DIST_bound (REAL) workspace of dimension n
!  S      (CG_save_type) private data that must be preserved between calls

!  ------------------ end of dummy arguments --------------------------

      INTEGER, INTENT( IN    ) :: n, nbnd, nresrt, idebug, iout, itcgmx
      INTEGER, INTENT( INOUT ) :: jumpto, status, nvar, itercg, ifixed
      REAL ( KIND = wp ), INTENT( IN    ) :: r     , epsgrd, findmx
      REAL ( KIND = wp ), INTENT( INOUT ) :: gnrmsq, dxsqr , fmodel
      LOGICAL, INTENT( IN    ) :: twonrm, nobnds
      INTEGER, INTENT( INOUT ), DIMENSION( n ) :: IVAR
      INTEGER, INTENT( IN    ), DIMENSION( n ) :: VARIABLE_STATUS
      REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( n ) :: X0, G
      REAL ( KIND = wp ), INTENT( IN    ), DIMENSION( nbnd, 2 ) :: BND
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: XT
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: GRAD
      REAL ( KIND = wp ), INTENT( INOUT ), DIMENSION( n ) :: P, Q
      TYPE( CG_save_type ), INTENT( INOUT ) :: S
      REAL ( KIND = wp ), INTENT( OUT ), DIMENSION( : ) :: DIST_bound
      REAL ( KIND = wp ), INTENT( IN    ), OPTIONAL, DIMENSION( n ) :: XSCALE

!  INITIALIZATION:

!  On the initial call to the subroutine the following variables MUST BE SET
!  by the user:

!      N, X0, XT, G, FMODEL, EPSGRD, TWONRM, NOBNDS, P, Q, IOUT, JUMPTO,
!      XSCALE, NBND, and (if NOBNDS is .FALSE.) BND.

!  JUMPTO must have the value 1. In addition, if the I-th variable is required
!  to be fixed at its initial value, X0(I), VARIABLE_status(I) must be set to
!  3 or 4. r must be specified if twonrm is .TRUE. on initial entry

!  RE-ENTRY:

!  If the variable JUMPTO has the value 2 on exit, the
!  subroutine must be re-entered with the vector Q containing
!  the product of the inverse of the preconditioning matrix W and
!  the output vector GRAD. all other parameters MUST NOT BE ALTERED

!  If the variable JUMPTO has the value 3 on exit, the
!  subroutine must be re-entered with the vector Q containing
!  the product of the second derivative matrix B and the output
!  vector P. All other parameters MUST NOT BE ALTERED

!-----------------------------------------------
!   L o c a l   V a r i a b l e s
!-----------------------------------------------

      INTEGER :: i, j, newvar
      REAL ( KIND = wp ) :: stepmx, stepib, stepbd, steptr, realgr
      REAL ( KIND = wp ) :: beta  , dxtp  , pi    , ptp   , curvat, epsmch

      IF ( jumpto <= 0 ) RETURN

      epsmch = EPSILON( one )

      SELECT CASE ( jumpto )
      CASE ( 1 ) ; GO TO 100
      CASE ( 2 ) ; GO TO 200
      CASE ( 3 ) ; GO TO 300
      CASE ( 4 ) ; GO TO 210
      END SELECT

!  On initial entry, set constants

 100  CONTINUE
      S%iter = 0 ; S%itsle = 0 ; nvar = 0
      S%onepep = one + ten * epsmch
      S%alpha = zero ; gnrmsq = zero
      S%prnter = idebug >= 4 .AND. iout > 0
      S%pronel = idebug == 2 .AND. iout > 0

      IF ( S%prnter ) WRITE( iout, 2000 )

!  If necessary, check that XT lies within the spherical boundary

      IF ( twonrm ) THEN
        IF ( dxsqr > r * r ) THEN
          IF ( S%prnter ) WRITE( iout, 2080 )
          IF ( S%pronel ) WRITE( iout, 2190 )
          status = 12
          jumpto = 0
          GO TO 500
        END IF
      END IF
      DO i = 1, n

!  Place the addresses of the free variables in IVAR and the gradient
!  of the quadratic with respect to the free variables in GRAD

        IF ( VARIABLE_STATUS( i ) == 0 ) THEN
          nvar = nvar + 1
          IVAR( nvar ) = i
          GRAD( nvar ) = G( i ) + Q( i )
        END IF
      END DO

!  If there are no free variables, exit

      IF ( nvar == 0 ) THEN

!  Print details of the current iteration

        IF ( S%prnter ) THEN
          IF ( S%itsle == 0 ) WRITE( iout, 2000 )
          WRITE( iout, 2160 ) S%iter, fmodel * findmx, zero, zero, epsgrd
        END IF
        IF ( S%pronel ) THEN
          IF ( S%itsle == 0 ) WRITE( iout, 2110 ) epsgrd
          WRITE( iout, 2170 ) S%iter, fmodel * findmx, zero, zero, zero
        END IF
        IF ( S%prnter ) WRITE( iout, 2050 ) S%iter
        status = 10 ; jumpto = 4
        GO TO 500
      END IF

!  Start the main iteration

 110  CONTINUE

!  Return to the main routine to obtain the preconditioned
!  gradient Q = M ** (-1) GRAD

      jumpto = 2
      RETURN

!  Compute the norm of the preconditioned gradient

 200  CONTINUE
!     gnrmsq =  DOT_PRODUCT( Q( IVAR( : nvar ) ), GRAD( : nvar ) )
      gnrmsq =  zero
!write(6,*) ' nvar ', nvar
      DO i = 1, nvar
        gnrmsq = gnrmsq + Q( IVAR( i ) ) * GRAD( i )
!write(6,*)  IVAR( i  ), Q( IVAR( i ) ), GRAD( i )
      END DO
      IF ( idebug >= 100 ) THEN
        DO i = 1, nvar
          j = IVAR( i )
          IF ( nobnds ) THEN
            WRITE( iout, 2230 ) j, XT( j ), GRAD( i )
          ELSE
            WRITE( iout, 2210 ) j, BND( j, 1 ), XT( j ), BND( j, 2 ), GRAD( i )
          END IF
        END DO
      END IF

!  If the preconditioned gradient is sufficiently small, evaluate the norm of
!  the true gradient of the quadratic model

      IF ( gnrmsq <= epsgrd ) THEN
        IF ( PRESENT( XSCALE ) ) THEN
          realgr = MAX( zero, MAXVAL(                                          &
                     ( GRAD( : nvar ) * XSCALE( IVAR( : nvar ) ) ) ** 2 ) )
        ELSE
          realgr = MAX( zero, MAXVAL( ( GRAD( : nvar ) ) ** 2 ) )
        END IF

!  Print details of the current iteration

        IF ( S%prnter ) THEN
          WRITE( iout, 2160 ) S%iter, fmodel * findmx, gnrmsq, realgr, epsgrd
        END IF
        IF ( S%pronel ) THEN
          IF ( S%itsle == 0) WRITE( iout, 2110 ) epsgrd
          WRITE( iout, 2170 ) S%iter, fmodel * findmx, gnrmsq, realgr, S%alpha
        END IF

!  If the gradient of the model is sufficiently small, exit

        IF ( realgr <= epsgrd ) THEN
          IF ( S%pronel ) WRITE( iout, 2050 ) S%iter
          IF ( S%prnter ) WRITE( iout, 2050 ) S%iter
          status = 10 ; jumpto = 4
          GO TO 500
        END IF
      ELSE

!  Print details of the current iteration

        IF ( S%prnter ) THEN
          WRITE( iout, 2020 ) S%iter, fmodel * findmx, gnrmsq, epsgrd
        END IF
        IF ( S%pronel ) THEN
          IF ( S%itsle == 0 ) WRITE( iout, 2110 ) epsgrd
          WRITE( iout, 2120 ) S%iter, fmodel * findmx, gnrmsq, S%alpha
        END IF
      END IF

!  Start of the new iteration

 210  CONTINUE
      S%iter = S%iter + 1 ; S%itsle = S%itsle + 1

!  ---------------------------------------------------------------------
!                Compute the search direction
!  ---------------------------------------------------------------------

      IF ( MOD( S%iter, nresrt + 1 ) == 1 ) THEN
        P( IVAR( : nvar ) ) = - Q( IVAR( : nvar ) )
      ELSE

!  Calculate the scale factor, BETA, which makes the search direction
!  B-conjugate to its predecessors

        beta = gnrmsq / S%oldgns
        P( IVAR( : nvar ) ) = - Q( IVAR( : nvar ) ) + beta * P( IVAR( : nvar ) )
      END IF

!  Save the norm of the preconditioned gradient

      S%oldgns = gnrmsq

!  Return to the main routine to evaluate Q = B * P

      jumpto = 3
      RETURN

!  Compute the maximum possible step, STEPMX

 300  CONTINUE
      curvat = zero
      stepbd = infinity

!  ---------------------------------------------------------------------
!                Compute the curvature
!  ---------------------------------------------------------------------

      IF ( nobnds ) THEN
!DIR$ IVDEP
        DO j = 1, nvar

!  Compute the curvature along the current search direction

          i = IVAR( j )
          curvat = curvat + Q( i ) * P( i )
        END DO
      ELSE
!DIR$ IVDEP
        DO j = 1, nvar
          i = IVAR( j )
          pi = P( i )
          curvat = curvat + Q( i ) * pi

!  Find the distance to the I-th bound, STEPIB, and save it for later.
!  Take precautions if the current iterate is very close to a bound

          IF ( pi > epsmch ) THEN
            stepib = ( epsmch + ( BND( i, 2 ) - XT( i ) ) ) / pi
          ELSE
            IF ( pi < - epsmch ) THEN
              stepib = ( - epsmch + ( BND( i, 1 ) - XT( i ) ) ) / pi
            ELSE
              stepib = infinity
            END IF
          END IF
          IF ( stepib <= zero ) stepib = epsmch / ABS( pi )
          DIST_bound( i ) = stepib
!         IF ( S%prnter ) WRITE( 6, 2180 ) i, XT( i ), P( i ), DIST_bound( i )

!  Find the distance to nearest bound, stepbd

          stepbd = MIN( stepbd, stepib )
        END DO
      END IF

!  If necessary, compute the terms needed to find the distance to the
!  spherical boundary

      stepmx = stepbd
      IF ( twonrm ) THEN
        dxtp = zero ; ptp = zero
!DIR$ IVDEP
        DO j = 1, nvar
          i = IVAR( j )
          pi = P( i )
          dxtp = dxtp + pi * ( XT( i ) - X0( i ) )
          ptp = ptp + pi * pi
        END DO

!  Now compute the distance to the spherical boundary, STEPTR, and
!  find the smaller of this and the distance to the boundary of the
!  feasible box, STEPMX

        steptr = ( SQRT( dxtp * dxtp - ptp * ( dxsqr - r * r ) ) - dxtp ) / ptp
        stepmx = MIN( steptr, stepmx )
      END IF

!   If the curvature is positive, compute the step to the minimizer of
!   the quadratic along the search direction, ALPHA

      IF ( curvat > zero ) THEN
        S%alpha = S%oldgns / curvat
        IF ( S%prnter ) WRITE( iout, 2030 ) S%alpha, stepmx

!  ---------------------------------------------------------------------
!                Line minimizer encountered
!  ---------------------------------------------------------------------

!  If the minimizer lies before the boundary, update the solution and
!  prepare for another C.G. step

        IF ( S%alpha <= stepmx ) THEN

!  Update the function value

          fmodel = fmodel - half * S%alpha * S%oldgns

!  The step lies within the bounds. Update the solution

!DIR$ IVDEP
          DO j = 1, nvar
            i = IVAR( j )
            XT( i ) = XT( i ) + S%alpha * P( i )
            GRAD( j ) = GRAD( j ) + S%alpha * Q( i )
          END DO

!  If necessary, update the square of the distance from X0 to XT

          IF ( twonrm ) dxsqr = dxsqr + S%alpha * ( two * dxtp + S%alpha * ptp )

!  Perform another iteration

          IF ( S%iter < itcgmx ) THEN
            IF ( S%alpha >= epsmch ) THEN
              GO TO 110
            ELSE

!  The step taken is very small. Further progress is unlikely. Exit

              IF ( S%prnter ) WRITE( iout, 2010 ) S%alpha
              IF ( S%pronel ) WRITE( iout, 2220 ) S%iter, S%alpha
              status = 14
              jumpto = 0
              GO TO 500
            END IF

!  More than MAXIT iterations have been performed. Exit

          ELSE
            IF ( S%pronel ) WRITE( iout, 2040 )
            IF ( S%prnter ) WRITE( iout, 2040 )
            status = 11
            jumpto = 0
            GO TO 500
          END IF
        END IF

!  If negative curvature is encountered, prepare to exit

      ELSE
        status = 13
        IF ( S%prnter ) WRITE( iout, 2060 )
        IF ( S%pronel ) WRITE( iout, 2150 ) S%iter
      END IF

!  ---------------------------------------------------------------------
!                Boundary encountered
!  ---------------------------------------------------------------------

!  Update the function value

      fmodel = fmodel + stepmx * ( - S%oldgns + half * stepmx * curvat )

!  If a step is made outside the spherical boundary, exit

      IF ( twonrm ) THEN
        IF ( steptr <= stepbd ) THEN

!  Update the solution values

          IF ( .NOT. nobnds ) THEN
!DIR$ IVDEP
            DO j = 1, nvar
              i = IVAR( j )
              XT( i ) = MIN( BND( i, 2 ),                                      &
                        MAX( BND( i, 1 ), XT( i ) + stepmx * P( i ) ) )
            END DO
          ELSE
!DIR$ IVDEP
            DO j = 1, nvar
              i = IVAR( j )
              XT( i ) = XT( i ) + stepmx * P( i )
            END DO
          END IF

!  Update the square of the distance from X0 to XT

          dxsqr = dxsqr + stepmx * ( two * dxtp + stepmx * ptp )

!  Set the exit conditions

          status = 12 ; jumpto = 0
          IF ( S%prnter ) WRITE( iout, 2200 ) S%iter
          IF ( S%pronel ) WRITE( iout, 2200 ) S%iter
          GO TO 500
        END IF
      END IF

!  A free variable encounters a bound; prepare for a restart.
!  Compute which variables encounter their bound

      newvar = 0
      IF ( .NOT. nobnds ) THEN

!DIR$ IVDEP
        DO j = 1, nvar
          i = IVAR( j )

!  Variable I lies off its bound. Update the point

          IF ( DIST_bound( i ) >= stepbd * S%onepep ) THEN
            XT( i ) = XT( i ) + stepbd * P( i )

!  Shift the components of IVAR and GRAD and update the gradient
!  to account for the reduced set of free variables

            newvar = newvar + 1
            IVAR( newvar ) = i
            GRAD( newvar ) = GRAD( j ) + stepbd * Q( i )
          ELSE

!  Variable I encounters its bound. Flag the variable

            IVAR( j ) = - i

!  An upper bound is encountered

            IF ( P( i ) > epsmch ) THEN

!  Move to the point at which the boundary is encountered

              XT( i ) = BND( i, 2 )

!  The modulus of ifixed gives the status of one of the variables which
!  encounters a bound. IFIXED is set positive, to indicate the variable hits
!  its upper bound

              ifixed = i
              IF ( S%prnter ) WRITE( iout, 2090 ) BND( i, 2 ), i
              IF ( S%pronel ) WRITE( iout, 2130 ) S%iter, fmodel, i

!  A lower bound is encountered

            ELSE
              XT( i ) = BND( i, 1 )

!  The modulus of ifixed gives the status of one of the variables which
!  encounters a bound. ifixed is set negative, to indicate the variable hits
!  its lower bound

              ifixed = - i
              IF ( S%prnter ) WRITE( iout, 2100 ) BND( i, 1 ), i
              IF ( S%pronel ) WRITE( iout, 2140 ) S%iter, fmodel, i
            END IF
            P( i ) = zero
          END IF
        END DO
      END IF
      nvar = newvar

!  There are still some free variables

      IF ( nvar > 0 ) THEN
        IF ( S%prnter ) WRITE( iout, 2070 ) S%iter
        S%iter = 0
        IF ( status /= 13 ) status = 12
        jumpto = 5
      ELSE

!  If there are no free variables, exit

        IF ( S%prnter ) WRITE( iout, 2050 ) S%iter
        IF ( status /= 13 ) status = 12
        jumpto = 0
      END IF

!  Prepare to exit

 500  CONTINUE
      itercg = itercg + S%itsle ; S%itsle = 0
      RETURN
!
!  Non-executable statements.
!
 2000    FORMAT( / ' -----------------  CG_solve entered ----------------' )
 2010    FORMAT( / ' Step ', ES12.4, ' too small in C.G.' )
 2020    FORMAT( / ' CG iteration ', I3, / ' Function/gradient norm**2 are ', &
              2ES12.4, ' epsgrd = ', ES12.4 )
 2030    FORMAT( ' Minimizing step and step to bound = ', 2ES12.4 )
 2040    FORMAT( / ' Maximum number of iterations exceeded in C.G.' )
 2050    FORMAT( / ' Convergence criterion satisfied after ', I3,             &
                ' C.G. iterations')
 2060    FORMAT( / ' Direction of negative curvature', / )
 2070    FORMAT( / ' Bound encountered in C.G. after ', I4, ' iterations ' )
 2080    FORMAT( / ' Boundary of spherical region reached ', / )
 2090    FORMAT( / ' Upper bound of ', ES12.4,' on variable ', I3,            &
                ' encountered ')
 2100    FORMAT( ' Lower bound of ', ES12.4,' on variable ', I3,              &
                ' encountered ')
 2110    FORMAT( / '                                           ',             &
                '   Grad tol  =', ES8.1,                                      &
              / '    **  CG entered **    Iteration  Model  ',                &
                '   Proj.Grad.  True Grad.  Step ')
 2120    FORMAT( 24X, I7, 2ES12.4, 12X, ES12.4 )
 2130    FORMAT( 24X, I7, ES12.4, '  -- Upper bound ', I5, ' encountered' )
 2140    FORMAT( 24X, I7, ES12.4, '  -- Lower bound ', I5, ' encountered' )
 2150    FORMAT( 24X, I7, 13X, ' -- Negative curvature encountered ' )
 2160    FORMAT( / ' CG iteration ', I3, /,                                   &
                ' Function/pr. gradient norm**2 are ', 2ES12.4, /,            &
                ' Model gradient norm**2/epsgrd are ', 2ES12.4 )
 2170    FORMAT( 24X, I7, 4ES12.4 )
!2180    FORMAT( ' I, XT, P, DIST_bound = ', I5, 3ES12.4 )
 2190    FORMAT( / '    **  CG entered **    Iteration  Model  ', /           &
                '   Boundary of the spherical region reached ' )
 2200    FORMAT( / ' Spherical bound encountered in C.G. after ', I4,         &
                ' iterations ' )
 2210    FORMAT( ' VAR L X U G ', I6 , 4ES12.4 )
 2220    FORMAT( 24X, I7, 13X, ' -- Step ', ES12.4, ' too small ' )
 2230    FORMAT( ' VAR X G ', I6 , 2ES12.4 )

!  End of subroutine CG_solve

      END SUBROUTINE CG_solve

!  End of module LANCELOT_CG

   END MODULE LANCELOT_CG_double


