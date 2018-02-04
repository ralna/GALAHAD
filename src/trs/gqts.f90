   PROGRAM GALAHAD_GQT_EXAMPLE  !  GALAHAD 2.2 - 05/06/2008 AT 13:30 GMT.
   USE GALAHAD_TRS_DOUBLE                          ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 )       ! set precision
   REAL ( KIND = wp ), PARAMETER :: one = 1.0_wp, zero = 0.0_wp
   INTEGER, PARAMETER :: n = 3, h_ne = 4           ! problem dimensions
   INTEGER :: i, info, iter, facts
   INTEGER :: itmax = 100
   REAL ( KIND = wp ), DIMENSION( n ) :: C, X, Z, WA1, WA2
   REAL ( KIND = wp ), DIMENSION( n, n ) :: H
   REAL ( KIND = wp ) :: radius = one           ! trust-region radius
   REAL ( KIND = wp ) :: rtol = 10.0_wp ** ( - 12 )
   REAL ( KIND = wp ) :: atol = 0.0_wp
   REAL ( KIND = wp ) :: par, f
   H = zero
   H(1,1) = 1.0_wp
   H(2,2) = 2.0_wp
   H(3,3) = 3.0_wp
   H(3,1) = 4.0_wp
   H(1,3) = H(3,1)
   DO i = 1, 3
     IF ( i == 1 ) THEN             !  (normal case)
       WRITE( 6, "( ' Normal case:' )" )
       par = 4.468089744720383D+0
       C = (/ 5.0_wp, 0.0_wp, 4.0_wp /)
     ELSE IF ( i == 2 ) THEN        !  (hard case)
       WRITE( 6, "( ' Hard case:' )" )
       par = 3.258147959821393D+0
       C = (/ 0.0_wp, 2.0_wp, 0.0_wp /)
     ELSE IF ( i == 3 ) THEN        !  (almost hard case)
       WRITE( 6, "( ' Almost hard case:' )" )
       par = 3.258147960635930D+0
       C = (/ 0.0_wp, 2.0_wp, 0.0001_wp /)
     END IF
     CALL dgqt(n,H,n,C,radius,rtol,atol,itmax,6,1,par,f,x,info,iter,facts,   &
               z,wa1,wa2)
     IF ( info == 1 ) THEN !  Successful return
       WRITE( 6, "( 1X, I0,' factorizations. Solution and Lagrange multiplier =',&
      &    2ES12.4 )" ) iter, f, par
     ELSE  !  Error returns
       WRITE( 6, "( ' GQT_solve exit status = ', I0 ) " ) info
     END IF
   END DO
   END PROGRAM GALAHAD_GQT_EXAMPLE
