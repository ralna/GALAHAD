! THIS VERSION: GALAHAD 2.4 - 29/03/2010 AT 10:30 GMT.
   PROGRAM GALAHAD_FIT_EXAMPLE
   USE GALAHAD_FIT_double         ! double precision version
   IMPLICIT NONE
   INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
   TYPE ( FIT_data_type ) :: data
   TYPE ( FIT_control_type ) :: control        
   TYPE ( FIT_inform_type ) :: inform
   INTEGER :: i, m
!   INTEGER, PARAMETER :: n = 9
!  INTEGER, PARAMETER :: n = 16
   INTEGER, PARAMETER :: n = 7
   REAL ( KIND = wp ), DIMENSION( n ) :: ALPHA, F, COEF
!   ALPHA = (/ 0.5_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 1.0_wp, 2.0_wp /)
!  ALPHA = (/    0.000010000000000_wp,    0.000010000000000_wp, &  
!                0.000010000000000_wp,    0.000010000000000_wp, &  
!                0.000100000000000_wp,    0.000100000000000_wp, &
!                0.000100000000000_wp,    0.000100000000000_wp, &
!                0.001000000000000_wp,    0.001000000000000_wp, &   
!                0.001000000000000_wp,    0.001000000000000_wp, &   
!                0.010000000000000_wp,    0.010000000000000_wp, &
!                0.010000000000000_wp,    0.010000000000000_wp  /)
!  ALPHA = (/    0.000010000000000_wp,    0.000010000000000_wp, &  
!                0.000100000000000_wp,    0.000100000000000_wp, &
!                0.001000000000000_wp,    0.001000000000000_wp, &   
!                0.010000000000000_wp,    0.010000000000000_wp /)
   ALPHA = (/    0.000010000000000_wp,    0.000010000000000_wp, &  
                 0.000010000000000_wp,    0.000010000000000_wp, &
                 0.000010000000000_wp,    0.000010000000000_wp, &
                 0.000010000000000_wp /)
   DO i = 1, n
     IF ( i == 3 ) THEN
       F( i ) = - SIN( ALPHA( i ) ) * COS( ALPHA( i ) )
     ELSE
       F( i ) = SIN( ALPHA( i ) ) * COS( ALPHA( i ) )
     END IF
  END DO
!  F = (/        3.157271962785552d-03,    1.576121037670156d+02, &
!                -7.905783112223458d+06,  1.185858573948378d+12, &
!                9.949442142947820d-03,    4.949438086242822d+01,  &
!                -2.500281648054056d+05,   3.750141544236343d+09, &
!                3.112516941579632d-02,    1.529350491094530d+01,  &
!                -7.914583196215527d+03,   1.186308835085532d+07, &
!                9.416422302570707d-02,    4.446679708196878d+00,  &
!                -2.525767621668448d+02,   3.764383959472400d+04 /)
! F = (/        3.157271962785552d-03,    1.576121037670156d+02, &
!               9.949442142947820d-03,    4.949438086242822d+01,  &
!               3.112516941579632d-02,    1.529350491094530d+01,  &
!               9.416422302570707d-02,    4.446679708196878d+00 /)
!   F = (/        3.157271962785552d-03,    1.576121037670156d+02, &
!                 -7.905783112223458d+06,  1.185858573948378d+12, &
!                 -2.964641955283499d+17,   1.037624017259029d+23, &
!                 -4.669306450035532d+28 /)
   F = (/             3.157265801763533d-03,      1.576121042613548d+02, &
                     -7.905783084354697d+06,      1.185858569773871d+12, &
                     -2.964641976964682d+17,      1.037624024869216d+23, &
                     -4.669306444265897d+28 /)

   m = 6
! problem data complete
   CALL FIT_initialize( data, control, inform )  ! Initialize control parameters
!  CALL FIT_hermite_interpolation( m, ALPHA, F, COEF, data, control, inform )
   CALL FIT_puiseux_interpolation( m, ALPHA, F, COEF, data, control, inform )
   IF ( inform%status == 0 ) THEN               !  Successful return
     WRITE( 6, "( ' FIT: coefficients' /, ( 5ES12.4 ) )" ) COEF( : m )
   ELSE                                       !  Error returns
     WRITE( 6, "( ' FIT_solve exit status = ', I6 ) " ) inform%status
   END IF
   CALL FIT_terminate( data, control, inform )  !  delete internal workspace
   END PROGRAM GALAHAD_FIT_EXAMPLE

   
   
 
     
     
     


